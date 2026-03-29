import os
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import duckdb
import xgboost as xgb
import shap
import warnings

warnings.filterwarnings('ignore')

# --- Directory Setup ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
TIFF_DIR = os.path.join(BASE_DIR, 'tiffs')
GT_DIR = os.path.join(BASE_DIR, 'ground_truth')
PRED_DIR = os.path.join(BASE_DIR, 'predictions')
DB_PATH = os.path.join(BASE_DIR, 'metrics.duckdb')

def get_image_metrics(tiff_path):
    try:
        with rasterio.open(tiff_path) as src:
            img = src.read()
            brightness = np.mean(img)
            contrast = np.std(img)
            return brightness, contrast
    except Exception as e:
        print(f"Error reading {tiff_path}: {e}")
        return None, None

def get_spatial_metrics(gt_path, pred_path, tiff_path):
    try:
        gt_uri = f"zip://{gt_path}" if os.path.exists(gt_path) else None
        pred_uri = f"zip://{pred_path}" if os.path.exists(pred_path) else None
        
        if not gt_uri or not pred_uri:
            return 0.0 

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
        print(f"Error calculating spatial metrics: {e}")
        return 0.0
           

def run_pipeline():
    print("🚀 Starting the Geospatial QA Data Pipeline...")
    results = []
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
        
        if brightness is not None:
            results.append({
                'tile_id': tile_id, 'brightness': brightness, 
                'contrast': contrast, 'iou': iou
            })

    df = pd.DataFrame(results)
    
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
    
    print(f"💾 Saving complete dataset to DuckDB at {DB_PATH}...")
    conn = duckdb.connect(DB_PATH)
    conn.execute("CREATE OR REPLACE TABLE tile_metrics AS SELECT * FROM df")
    row_count = conn.execute("SELECT COUNT(*) FROM tile_metrics").fetchone()[0]
    print(f"✅ Pipeline Complete! Successfully wrote {row_count} records to the database.")
    conn.close()

if __name__ == "__main__":
    run_pipeline()