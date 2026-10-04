import os
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.enums import ColorInterp
import duckdb
import zipfile
import xgboost as xgb
import shap
import warnings
from dotenv import load_dotenv
from langsmith import traceable

warnings.filterwarnings('ignore')

# --- Robust Dotenv Loading (must run before any traceable call) ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
load_dotenv(dotenv_path=os.path.join(ROOT_DIR, ".env"))

# --- Directory Setup ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
TIFF_DIR = os.path.join(BASE_DIR, 'tiffs')
GT_DIR = os.path.join(BASE_DIR, 'ground_truth')
PRED_DIR = os.path.join(BASE_DIR, 'predictions')
DB_PATH = os.path.join(BASE_DIR, 'metrics.duckdb')

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

def shapefile_uri(zip_path):
    """
    Build a GeoPandas URI for the .shp inside a zipped shapefile.

    Some exports put the files at the archive root, others wrap them in a folder named
    after the tile. A bare zip:// URI only finds the former, so locate the .shp and
    address it explicitly. Returns None if the archive holds no shapefile.
    """
    if not os.path.exists(zip_path):
        return None

    with zipfile.ZipFile(zip_path) as archive:
        # __MACOSX holds resource-fork stubs that look like real entries but are not.
        shps = [n for n in archive.namelist()
                if n.lower().endswith('.shp') and not n.startswith('__MACOSX/')]

    if not shps:
        return None
    if len(shps) > 1:
        print(f"  -> Warning: {os.path.basename(zip_path)} holds {len(shps)} shapefiles, using {shps[0]}")

    inner = shps[0]
    return f"zip://{zip_path}" if '/' not in inner else f"zip://{zip_path}!{inner}"


def get_spatial_metrics(gt_path, pred_path, tiff_path):
    """
    Buffer-tolerant IoU between ground truth and prediction.

    Returns None when the comparison could not be made at all -- missing or unreadable
    shapefiles, a CRS that will not project. That is deliberately distinct from 0.0,
    which means the geometries were read fine and simply do not overlap. Collapsing the
    two hid three unreadable archives behind a plausible-looking score.
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
            
        if not gt_gdf.empty and gt_gdf.crs != tiff_crs:
            gt_gdf = gt_gdf.to_crs(tiff_crs)
        if not pred_gdf.empty and pred_gdf.crs != tiff_crs:
            pred_gdf = pred_gdf.to_crs(tiff_crs)
            
        # --- THE FIX: Handle LineStrings (Animal Trails) ---
        # If the geometries are lines, buffer them by 2 meters to create measurable area
        buffer_distance = 5.0 
        
        if not gt_gdf.empty and gt_gdf.geometry.geom_type.isin(['LineString', 'MultiLineString']).any():
            gt_gdf['geometry'] = gt_gdf.geometry.buffer(buffer_distance)
            
        if not pred_gdf.empty and pred_gdf.geometry.geom_type.isin(['LineString', 'MultiLineString']).any():
            pred_gdf['geometry'] = pred_gdf.geometry.buffer(buffer_distance)
        # ---------------------------------------------------

        gt_geom = gt_gdf.geometry.unary_union if not gt_gdf.empty else None
        pred_geom = pred_gdf.geometry.unary_union if not pred_gdf.empty else None
        
        if not gt_geom or not pred_geom:
            return 0.0 
            
        intersection = gt_geom.intersection(pred_geom).area
        union = gt_geom.union(pred_geom).area
        
        # Debug print to verify the fix
        print(f"  -> Buffered GT Area: {gt_geom.area:.2f}, Buffered Pred Area: {pred_geom.area:.2f}, Intersection: {intersection:.2f}")
        
        return intersection / union if union > 0 else 0.0
    except Exception as e:
        print(f"  -> Skipping: could not compare geometries: {e}")
        return None


@traceable(run_type="chain", name="train_and_explain")
def train_and_explain(df):
    """
    Trains the XGBoost meta-model on spatial error and attaches per-tile SHAP values.
    Returns the dataframe with shap_brightness / shap_contrast columns added.
    """
    print("🧠 Training XGBoost Meta-Model on spatial errors...")
    X = df[['brightness', 'contrast']]
    y = 1.0 - df['iou']

    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=50, max_depth=3)
    model.fit(X, y)

    print("🔍 Generating SHAP Values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

    df['shap_brightness'] = shap_values.values[:, 0]
    df['shap_contrast'] = shap_values.values[:, 1]
    return df

@traceable(run_type="chain", name="run_pipeline")
def run_pipeline():
    print("🚀 Starting the Geospatial QA Data Pipeline...")
    results = []
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
        iou = get_spatial_metrics(gt_path, pred_path, tiff_path)
        
        if brightness is None or iou is None:
            skipped.append(tile_id)
            continue

        results.append({
            'tile_id': tile_id, 'brightness': brightness,
            'contrast': contrast, 'iou': iou
        })

    if skipped:
        print(f"\n⚠️  Skipped {len(skipped)} tile(s) with unusable data: {', '.join(skipped)}")

    if not results:
        print("❌ No tile produced usable metrics. Nothing written to the database.")
        return

    df = pd.DataFrame(results)

    df = train_and_explain(df)

    print(f"💾 Saving complete dataset to DuckDB at {DB_PATH}...")
    conn = duckdb.connect(DB_PATH)
    conn.execute("CREATE OR REPLACE TABLE tile_metrics AS SELECT * FROM df")
    row_count = conn.execute("SELECT COUNT(*) FROM tile_metrics").fetchone()[0]
    print(f"✅ Pipeline Complete! Successfully wrote {row_count} records to the database.")
    conn.close()

if __name__ == "__main__":
    run_pipeline()