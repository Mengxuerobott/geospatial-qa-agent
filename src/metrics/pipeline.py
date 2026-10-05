import os
import sys
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.enums import ColorInterp
from rasterio.windows import Window, bounds as window_bounds
from shapely.geometry import box
import duckdb
import xgboost as xgb
import shap
import warnings
from dotenv import load_dotenv
from langsmith import traceable

warnings.filterwarnings('ignore')

# --- Robust Dotenv Loading (must run before any traceable call) ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
load_dotenv(dotenv_path=os.path.join(ROOT_DIR, ".env"))

# Run as a script, sys.path[0] is src/metrics, so the project root has to be added
sys.path.insert(0, ROOT_DIR)
from src.metrics.shapefiles import shapefile_uri  # noqa: E402
from src.agent.verdict import FAIL_IOU_THRESHOLD  # noqa: E402

# --- Directory Setup ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
TIFF_DIR = os.path.join(BASE_DIR, 'tiffs')
GT_DIR = os.path.join(BASE_DIR, 'ground_truth')
PRED_DIR = os.path.join(BASE_DIR, 'predictions')
DB_PATH = os.path.join(BASE_DIR, 'metrics.duckdb')

# --- Analysis Settings ---
# Lines are buffered by this many metres either side so they have an area to overlap
BUFFER_M = 5.0
# Each tile is analysed on a grid of square cells this many metres a side
CELL_SIZE_M = 50.0
MAX_CELL_READ_PX = 1024
# Cells with less imagery than this are mostly padding and are left out
MIN_VALID_FRACTION = 0.05
# A cell needs at least this much buffered trail (m^2), ground truth and prediction
# together, to get an IoU. Below it the score is decided by a sliver at the cell's edge.
MIN_SCORED_AREA = 25.0
# A pixel whose brightest band is under this share of full scale counts as shadow
SHADOW_VALUE = 0.25
# What the meta-model predicts error from, one value per cell
FEATURES = ['brightness', 'contrast', 'shadow_fraction', 'greenness', 'sharpness']

def get_image_metrics(tiff_path):
    """
    Brightness (mean) and contrast (std) of the tile's valid colour pixels.

    Drone tiles are irregular shapes padded out to a rectangle, with an alpha band marking
    the padding. Averaging every band over every pixel measures how much padding a tile
    has, not how bright the imagery is: the alpha band is 255 wherever there is data, and
    the padding is black. So read the colour bands only and drop masked pixels.
    """
    try:
        with rasterio.open(tiff_path) as src:
            colour_bands = [i for i, interp in enumerate(src.colorinterp, start=1)
                            if interp != ColorInterp.alpha]
            img = src.read(colour_bands)
            # 0 where the alpha band, nodata value or internal mask marks a pixel invalid
            valid = src.dataset_mask() > 0

        if not valid.any():
            print(f"Error reading {tiff_path}: no valid pixels")
            return None, None

        pixels = img[:, valid]
        return float(pixels.mean()), float(pixels.std())
    except Exception as e:
        print(f"Error reading {tiff_path}: {e}")
        return None, None

def load_buffered_geometries(gt_path, pred_path, tiff_path):
    """
    Ground truth and prediction as one geometry each, in the raster's CRS, with lines
    buffered into polygons.

    Returns None when the comparison cannot be made at all -- missing or unreadable
    shapefiles, a CRS that will not project. That is deliberately distinct from an empty
    geometry, which means the file was read fine and holds nothing. Collapsing the two hid
    three unreadable archives behind a plausible-looking score.
    """
    try:
        gt_uri = shapefile_uri(gt_path)
        pred_uri = shapefile_uri(pred_path)

        if not gt_uri or not pred_uri:
            missing = [n for n, u in (("ground truth", gt_uri), ("prediction", pred_uri)) if not u]
            print(f"  -> Skipping: no readable shapefile for {' and '.join(missing)}")
            return None

        gt_gdf = gpd.read_file(gt_uri)
        pred_gdf = gpd.read_file(pred_uri)

        with rasterio.open(tiff_path) as src:
            tiff_crs = src.crs

        geoms = []
        for gdf in (gt_gdf, pred_gdf):
            if gdf.empty:
                geoms.append(None)
                continue
            if gdf.crs != tiff_crs:
                gdf = gdf.to_crs(tiff_crs)
            # Animal trails are lines, which have no area to overlap, so give them one.
            # This runs on the whole tile, before any cell is cut out of it: buffering
            # inside a cell would lose the part of a trail's buffer that belongs to the
            # cell next door.
            if gdf.geometry.geom_type.isin(['LineString', 'MultiLineString']).any():
                gdf = gdf.assign(geometry=gdf.geometry.buffer(BUFFER_M))
            geoms.append(gdf.geometry.unary_union)
        return tuple(geoms)
    except Exception as e:
        print(f"  -> Skipping: could not read geometries: {e}")
        return None


def overlap_iou(gt_geom, pred_geom):
    """IoU of two geometries. 0.0 when either is missing or they do not overlap."""
    if not gt_geom or not pred_geom:
        return 0.0
    union = gt_geom.union(pred_geom).area
    return gt_geom.intersection(pred_geom).area / union if union > 0 else 0.0


def cell_iou(gt_geom, pred_geom, cell_box):
    """
    IoU inside one grid cell, or None when the cell has too little trail to score.

    A cell with no trail in either layer is not a perfect match and not a miss; there is
    nothing in it to get right or wrong, so it stays out of the training data. A cell with
    trail in one layer only is a real miss or a real false positive, and scores 0.0.
    """
    gt_part = gt_geom.intersection(cell_box) if gt_geom else None
    pred_part = pred_geom.intersection(cell_box) if pred_geom else None
    gt_area = gt_part.area if gt_part else 0.0
    pred_area = pred_part.area if pred_part else 0.0

    if gt_area and pred_area:
        union = gt_part.union(pred_part).area
    else:
        union = gt_area + pred_area
    if union < MIN_SCORED_AREA:
        return None, gt_area, pred_area

    iou = gt_part.intersection(pred_part).area / union if gt_area and pred_area else 0.0
    return iou, gt_area, pred_area


def image_features(img, valid, scale):
    """
    The FEATURES of one block of imagery, measured on its valid pixels only.

    img is (bands, rows, cols) of colour bands, valid is a (rows, cols) boolean mask and
    scale is the value of a fully bright pixel. Returns None when nothing is valid.
    """
    if not valid.any():
        return None

    img = img.astype(np.float64)
    pixels = img[:, valid]
    # Brightest band per pixel, 0..1: a pixel is in shadow when no band is bright
    value = pixels.max(axis=0) / scale

    greenness = None
    if img.shape[0] >= 3:
        # Excess Green on chromatic coordinates, so a bright field and a dim one with the
        # same colour score the same. Assumes the first three bands are red, green, blue.
        r, g, b = pixels[0], pixels[1], pixels[2]
        total = r + g + b
        lit = total > 0
        if lit.any():
            greenness = float(((2 * g[lit] - r[lit] - b[lit]) / total[lit]).mean())

    # Variance of the Laplacian, on a 0..255 grey image: low when the imagery is blurred.
    # Only pixels whose four neighbours are also valid count, or the edge of the padding
    # would register as the sharpest thing in the tile.
    grey = img.mean(axis=0) / scale * 255.0
    laplacian = (grey[:-2, 1:-1] + grey[2:, 1:-1] + grey[1:-1, :-2] + grey[1:-1, 2:]
                 - 4 * grey[1:-1, 1:-1])
    inner = (valid[1:-1, 1:-1] & valid[:-2, 1:-1] & valid[2:, 1:-1]
             & valid[1:-1, :-2] & valid[1:-1, 2:])
    sharpness = float(laplacian[inner].var()) if inner.any() else None

    return {
        'brightness': float(pixels.mean()),
        'contrast': float(pixels.std()),
        'shadow_fraction': float((value < SHADOW_VALUE).mean()),
        'greenness': greenness,
        'sharpness': sharpness,
    }


def get_cell_metrics(tile_id, tiff_path, gt_geom, pred_geom, cell_size_m=CELL_SIZE_M):
    """
    One row per grid cell of the tile: where it is, what the imagery looks like there and
    how well the prediction matched the ground truth inside it.

    The cells are windows read out of the TIFF, which is never cut up; each row keeps its
    map bounds so a cell can be drawn back onto the whole tile. Cells that are almost all
    padding are left out.
    """
    rows = []
    with rasterio.open(tiff_path) as src:
        colour_bands = [i for i, interp in enumerate(src.colorinterp, start=1)
                        if interp != ColorInterp.alpha]
        scale = 255.0 if src.dtypes[0] == 'uint8' else float(np.iinfo(src.dtypes[0]).max)
        cell_px = max(1, round(cell_size_m / abs(src.res[0])))
        # A cell is read at no more than this many pixels a side, so memory stays flat
        # however fine the imagery is
        read_px = min(cell_px, MAX_CELL_READ_PX)

        for cell_row, top in enumerate(range(0, src.height, cell_px)):
            for cell_col, left in enumerate(range(0, src.width, cell_px)):
                window = Window(left, top, min(cell_px, src.width - left),
                                min(cell_px, src.height - top))
                out_shape = (max(1, round(window.height * read_px / cell_px)),
                             max(1, round(window.width * read_px / cell_px)))
                valid = src.dataset_mask(window=window, out_shape=out_shape) > 0
                if valid.mean() < MIN_VALID_FRACTION:
                    continue

                img = src.read(colour_bands, window=window,
                               out_shape=(len(colour_bands), *out_shape))
                minx, miny, maxx, maxy = window_bounds(window, src.transform)
                iou, gt_area, pred_area = cell_iou(gt_geom, pred_geom,
                                                   box(minx, miny, maxx, maxy))
                rows.append({
                    'tile_id': tile_id, 'cell_row': cell_row, 'cell_col': cell_col,
                    'minx': minx, 'miny': miny, 'maxx': maxx, 'maxy': maxy,
                    'valid_fraction': float(valid.mean()),
                    **image_features(img, valid, scale),
                    'gt_area': gt_area, 'pred_area': pred_area, 'iou': iou,
                })
    return rows


@traceable(run_type="chain", name="train_and_explain")
def train_and_explain(cells):
    """
    Trains the XGBoost meta-model on per-cell error and attaches a shap_<feature> column
    for each of FEATURES. Cells with no IoU were not scored; they are not trained on and
    their SHAP values are left empty.
    """
    print("🧠 Training XGBoost Meta-Model on spatial errors...")
    scored = cells['iou'].notna()
    X = cells.loc[scored, FEATURES].astype(float)
    y = 1.0 - cells.loc[scored, 'iou'].astype(float)

    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=50, max_depth=3)
    model.fit(X, y)

    print("🔍 Generating SHAP Values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

    for i, feature in enumerate(FEATURES):
        cells[f'shap_{feature}'] = np.nan
        cells.loc[scored, f'shap_{feature}'] = shap_values.values[:, i]
    return cells


def summarise_tiles(tiles, cells):
    """
    Adds to each tile how many of its cells were scored, how many of those failed, and
    the mean SHAP value of its scored cells for each feature.
    """
    scored = cells[cells['iou'].notna()]
    shap_columns = [f'shap_{feature}' for feature in FEATURES]
    per_tile = scored.groupby('tile_id').agg(
        cells_scored=('iou', 'size'),
        cells_failing=('iou', lambda iou: int((iou < FAIL_IOU_THRESHOLD).sum())),
        **{column: (column, 'mean') for column in shap_columns},
    ).reset_index()

    tiles = tiles.merge(per_tile, on='tile_id', how='left')
    for column in ('cells_scored', 'cells_failing'):
        tiles[column] = tiles[column].fillna(0).astype(int)
    return tiles


@traceable(run_type="chain", name="run_pipeline")
def run_pipeline():
    print("🚀 Starting the Geospatial QA Data Pipeline...")
    tile_rows = []
    cell_rows = []
    skipped = []
    tiff_files = glob.glob(os.path.join(TIFF_DIR, '*.tif'))
    
    if not tiff_files:
        print("❌ No TIFF files found. Exiting.")
        return

    for tiff_path in tiff_files:
        tile_id = os.path.basename(tiff_path).replace('.tif', '')
        print(f"Processing Tile: {tile_id}...")
        
        gt_path = os.path.join(GT_DIR, f"{tile_id}.zip")
        pred_path = os.path.join(PRED_DIR, f"{tile_id}.zip")
        
        brightness, contrast = get_image_metrics(tiff_path)
        geometries = load_buffered_geometries(gt_path, pred_path, tiff_path)
        
        if brightness is None or geometries is None:
            skipped.append(tile_id)
            continue

        gt_geom, pred_geom = geometries
        try:
            tile_cells = get_cell_metrics(tile_id, tiff_path, gt_geom, pred_geom)
        except Exception as e:
            print(f"  -> Skipping: could not compute cell metrics: {e}")
            skipped.append(tile_id)
            continue

        # The tile's own IoU is still measured on the whole tile. It is not an average of
        # its cells, which would let cells holding a sliver of trail outweigh the rest.
        tile_rows.append({
            'tile_id': tile_id, 'brightness': brightness,
            'contrast': contrast, 'iou': overlap_iou(gt_geom, pred_geom)
        })
        cell_rows.extend(tile_cells)
        print(f"  -> {len(tile_cells)} cells, "
              f"{sum(c['iou'] is not None for c in tile_cells)} with trail to score")

    if skipped:
        print(f"\n⚠️  Skipped {len(skipped)} tile(s) with unusable data: {', '.join(skipped)}")

    if not tile_rows:
        print("❌ No tile produced usable metrics. Nothing written to the database.")
        return

    cells = pd.DataFrame(cell_rows)
    if cells.empty or cells['iou'].notna().sum() == 0:
        print("❌ No cell holds any trail to score. Nothing written to the database.")
        return

    cells = train_and_explain(cells)
    tiles = summarise_tiles(pd.DataFrame(tile_rows), cells)

    print(f"💾 Saving complete dataset to DuckDB at {DB_PATH}...")
    conn = duckdb.connect(DB_PATH)
    conn.execute("CREATE OR REPLACE TABLE tile_metrics AS SELECT * FROM tiles")
    conn.execute("CREATE OR REPLACE TABLE cell_metrics AS SELECT * FROM cells")
    tile_count = conn.execute("SELECT COUNT(*) FROM tile_metrics").fetchone()[0]
    cell_count = conn.execute("SELECT COUNT(*) FROM cell_metrics").fetchone()[0]
    print(f"✅ Pipeline Complete! Successfully wrote {tile_count} tiles and {cell_count} cells to the database.")
    conn.close()

if __name__ == "__main__":
    run_pipeline()