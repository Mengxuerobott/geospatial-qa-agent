import os
import sys
import glob
import time
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.enums import ColorInterp
from rasterio.windows import Window, bounds as window_bounds
from shapely.geometry import GeometryCollection, box
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
from src.agent.verdict import FAIL_IOU_THRESHOLD, MATCH_TOLERANCE_M  # noqa: E402

# --- Directory Setup ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
TIFF_DIR = os.path.join(BASE_DIR, 'tiffs')
GT_DIR = os.path.join(BASE_DIR, 'ground_truth')
PRED_DIR = os.path.join(BASE_DIR, 'predictions')
DB_PATH = os.path.join(BASE_DIR, 'metrics.duckdb')

# --- Analysis Settings ---
# Each tile is analysed on a grid of square cells this many metres a side
CELL_SIZE_M = 50.0
MAX_CELL_READ_PX = 1024
# The whole tile is read in square blocks this many pixels a side
READ_BLOCK_PX = 2048
# Cells with less imagery than this are mostly padding and are left out
MIN_VALID_FRACTION = 0.05
# A cell needs at least this much trail, ground truth and prediction together, to get an
# IoU: metres of line, or square metres when the layers are polygons. Below it the score
# is decided by a stub at the cell's edge.
MIN_SCORED_LENGTH = 5.0
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
            # The tile is read a block at a time and only running totals are kept, so a
            # 300 MB TIFF does not have to fit in memory several times over
            count, total, total_of_squares = 0, 0.0, 0.0
            for top in range(0, src.height, READ_BLOCK_PX):
                for left in range(0, src.width, READ_BLOCK_PX):
                    window = Window(left, top, min(READ_BLOCK_PX, src.width - left),
                                    min(READ_BLOCK_PX, src.height - top))
                    # 0 where the alpha band, nodata value or internal mask marks a
                    # pixel invalid
                    valid = src.dataset_mask(window=window) > 0
                    if not valid.any():
                        continue
                    pixels = src.read(colour_bands, window=window)[:, valid].astype(np.float64)
                    count += pixels.size
                    total += pixels.sum()
                    total_of_squares += np.square(pixels).sum()

        if count == 0:
            print(f"Error reading {tiff_path}: no valid pixels")
            return None, None

        mean = total / count
        return float(mean), float(np.sqrt(max(total_of_squares / count - mean ** 2, 0.0)))
    except Exception as e:
        print(f"Error reading {tiff_path}: {e}")
        return None, None

def crs_problem(crs):
    """
    Why a raster's CRS cannot be measured in metres, or None when it can.

    The match tolerance and the cell size are distances in metres, applied in the
    raster's own coordinates. In degrees a tolerance of 5 spans the planet: every line
    matches every other and the tile scores a perfect 1.0. In feet the tolerance is a
    third of what was meant. Neither raises an error, so it is checked here.
    """
    if crs is None:
        return "it has no CRS"
    if not crs.is_projected:
        return f"its CRS ({crs.to_string()}) is in degrees, not metres"
    factor = crs.linear_units_factor[1]
    if abs(factor - 1.0) > 1e-6:
        return f"its CRS ({crs.to_string()}) is in {crs.linear_units}, not metres"
    return None


def full_scale(src, colour_bands):
    """
    The value of a fully bright pixel in this raster.

    8-bit imagery fills its range, so 255. Deeper imagery rarely does: a 12-bit camera
    writing a 16-bit file never passes 4095, and measured against 65535 every pixel of
    it would count as shadow. So the brightest the imagery actually gets is used, read
    from a reduced copy and taken just under the maximum so a few blown-out pixels do
    not set the scale.
    """
    if src.dtypes[0] == 'uint8':
        return 255.0

    shrink = min(1.0, MAX_CELL_READ_PX / max(src.height, src.width))
    shape = (max(1, int(src.height * shrink)), max(1, int(src.width * shrink)))
    valid = src.dataset_mask(out_shape=shape) > 0
    if not valid.any():
        return 1.0
    pixels = src.read(colour_bands, out_shape=(len(colour_bands), *shape))[:, valid]
    return float(max(np.percentile(pixels, 99.9), 1e-9))


def load_geometries(gt_path, pred_path, tiff_path):
    """
    Ground truth and prediction as one geometry each, in the raster's CRS. Either is None
    when its file was read fine and holds nothing.

    Returns None, not a pair, when the comparison cannot be made at all -- missing or
    unreadable shapefiles, a CRS that will not project. Collapsing that into an empty
    geometry hid three unreadable archives behind a plausible-looking score.
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
            geoms.append(gdf.geometry.unary_union)
        return tuple(geoms)
    except Exception as e:
        print(f"  -> Skipping: could not read geometries: {e}")
        return None


def split_agreement(gt_geom, pred_geom, tolerance=MATCH_TOLERANCE_M):
    """
    Sorts the trail into three geometries: matched, annotated only, predicted only.

    The annotation is a person's line near the trail, not on it, so the two are not
    compared by how much they overlap: a prediction running alongside the annotation a few
    metres away is the same trail. A stretch of annotated trail is matched when a
    prediction lies within the tolerance of it, and annotated only when none does. A
    stretch of prediction with no annotation within the tolerance is predicted only.

    This is done on the whole tile, before any cell is cut out of it, so a prediction just
    across a cell boundary still matches the annotation on this side.

    Polygons are compared by plain overlap, with no tolerance. A line set against polygons
    is widened by the tolerance first so that it has an area to overlap with.
    """
    gt = gt_geom if gt_geom else GeometryCollection()
    pred = pred_geom if pred_geom else GeometryCollection()

    if gt.area > 0 or pred.area > 0:
        gt = gt if gt.area > 0 else gt.buffer(tolerance)
        pred = pred if pred.area > 0 else pred.buffer(tolerance)
        return gt.intersection(pred), gt.difference(pred), pred.difference(gt)

    return (gt.intersection(pred.buffer(tolerance)),
            gt.difference(pred.buffer(tolerance)),
            pred.difference(gt.buffer(tolerance)))


def agreement(parts, region=None):
    """
    (iou, matched, annotated_only, predicted_only) for the output of split_agreement,
    over the whole tile or inside one region of it.

    The three sizes are metres of trail, or square metres when the layers are polygons.
    The IoU is the matched share of all three. It is None when the region holds too little
    trail to score: with no trail in either layer there is nothing to agree or disagree
    about, which is neither a perfect match nor a miss.
    """
    areal = any(part.area > 0 for part in parts)
    if region is not None:
        parts = [part.intersection(region) for part in parts]
    matched, annotated_only, predicted_only = (
        float(part.area if areal else part.length) for part in parts)

    total = matched + annotated_only + predicted_only
    if total < (MIN_SCORED_AREA if areal else MIN_SCORED_LENGTH):
        return None, matched, annotated_only, predicted_only
    return matched / total, matched, annotated_only, predicted_only


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
        # On a 0..255 scale whatever the bit depth, so tiles of different depths can be
        # trained on together
        'brightness': float(pixels.mean() / scale * 255.0),
        'contrast': float(pixels.std() / scale * 255.0),
        'shadow_fraction': float((value < SHADOW_VALUE).mean()),
        'greenness': greenness,
        'sharpness': sharpness,
    }


def get_cell_metrics(tile_id, tiff_path, parts, cell_size_m=CELL_SIZE_M):
    """
    One row per grid cell of the tile: where it is, what the imagery looks like there and
    how well the prediction matched the ground truth inside it. parts is the tile's trail
    as sorted by split_agreement.

    The cells are windows read out of the TIFF, which is never cut up; each row keeps its
    map bounds so a cell can be drawn back onto the whole tile. Cells that are almost all
    padding are left out.
    """
    rows = []
    with rasterio.open(tiff_path) as src:
        colour_bands = [i for i, interp in enumerate(src.colorinterp, start=1)
                        if interp != ColorInterp.alpha]
        scale = full_scale(src, colour_bands)
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
                iou, matched, annotated_only, predicted_only = agreement(
                    parts, box(minx, miny, maxx, maxy))
                rows.append({
                    'tile_id': tile_id, 'cell_row': cell_row, 'cell_col': cell_col,
                    'minx': minx, 'miny': miny, 'maxx': maxx, 'maxy': maxy,
                    'valid_fraction': float(valid.mean()),
                    **image_features(img, valid, scale),
                    'matched': matched, 'annotated_only': annotated_only,
                    'predicted_only': predicted_only, 'iou': iou,
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


def write_database(tiles, cells, db_path, attempts=5, wait_seconds=2.0):
    """
    Writes the two tables to a new database file and swaps it into place.

    The viewer and the API read the database while this runs. Writing into the live file
    would need a lock they hold, and would let them see one table rebuilt and the other
    not. So the new database is built beside the old one and renamed over it in one step:
    a reader gets the whole old database or the whole new one.

    On Windows the rename fails while a reader has the file open. Readers hold it for the
    length of one query, so it is retried a few times before giving up.
    """
    new_path = db_path + ".new"
    for leftover in (new_path, new_path + ".wal"):
        if os.path.exists(leftover):
            os.remove(leftover)

    conn = duckdb.connect(new_path)
    conn.execute("CREATE TABLE tile_metrics AS SELECT * FROM tiles")
    conn.execute("CREATE TABLE cell_metrics AS SELECT * FROM cells")
    conn.close()

    for attempt in range(attempts):
        try:
            os.replace(new_path, db_path)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(wait_seconds)


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

        with rasterio.open(tiff_path) as src:
            problem = crs_problem(src.crs)
        if problem:
            print(f"  -> Skipping: {problem}. Reproject the TIFF to a CRS in metres, "
                  "such as its UTM zone.")
            skipped.append(tile_id)
            continue
        
        gt_path = os.path.join(GT_DIR, f"{tile_id}.zip")
        pred_path = os.path.join(PRED_DIR, f"{tile_id}.zip")
        
        brightness, contrast = get_image_metrics(tiff_path)
        geometries = load_geometries(gt_path, pred_path, tiff_path)
        
        if brightness is None or geometries is None:
            skipped.append(tile_id)
            continue

        try:
            parts = split_agreement(*geometries)
            tile_cells = get_cell_metrics(tile_id, tiff_path, parts)
        except Exception as e:
            print(f"  -> Skipping: could not compute cell metrics: {e}")
            skipped.append(tile_id)
            continue

        # The tile's own IoU is measured on the whole tile. It is not an average of its
        # cells, which would let cells holding a stub of trail outweigh the rest. A tile
        # with no trail in either layer scores 0.0, as it always has.
        iou, matched, annotated_only, predicted_only = agreement(parts)
        with rasterio.open(tiff_path) as src:
            # Cell bounds are in this CRS; the review list needs it to place them on a map
            crs = src.crs.to_string() if src.crs else None
        tile_rows.append({
            'tile_id': tile_id, 'crs': crs, 'brightness': brightness, 'contrast': contrast,
            'iou': iou if iou is not None else 0.0, 'matched': matched,
            'annotated_only': annotated_only, 'predicted_only': predicted_only,
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
    write_database(tiles, cells, DB_PATH)
    print(f"✅ Pipeline Complete! Successfully wrote {len(tiles)} tiles and {len(cells)} cells to the database.")

if __name__ == "__main__":
    run_pipeline()