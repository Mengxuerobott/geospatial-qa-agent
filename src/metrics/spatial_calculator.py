import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

def calculate_polygon_errors(gt_path: str, pred_path: str, tile_id: str) -> pd.DataFrame:
    """
    Compares Ground Truth polygons with Predicted polygons to calculate spatial errors.
    Useful for 'Cultivated Areas'.
    """
    # Load shapefiles
    gt_gdf = gpd.read_file(gt_path)
    pred_gdf = gpd.read_file(pred_path)
    
    # Ensure they are in the same Coordinate Reference System (CRS)
    if gt_gdf.crs != pred_gdf.crs:
        pred_gdf = pred_gdf.to_crs(gt_gdf.crs)
        
    # Create a single unified geometry for GT and Pred to handle overlapping polygons
    gt_union = gt_gdf.geometry.union_all
    pred_union = pred_gdf.geometry.union_all
    
    # Handle empty predictions or ground truths safely
    if gt_union.is_empty and pred_union.is_empty:
        return pd.DataFrame() # Nothing to evaluate
        
    # Calculate Intersections and Differences using Shapely
    intersection = gt_union.intersection(pred_union)
    union = gt_union.union(pred_union)
    
    # False Positives: Area predicted but not in Ground Truth
    false_positives = pred_union.difference(gt_union)
    
    # False Negatives: Area in Ground Truth but missed by Prediction
    false_negatives = gt_union.difference(pred_union)
    
    # Calculate metrics
    iou = intersection.area / union.area if union.area > 0 else 0
    fp_area = false_positives.area
    fn_area = false_negatives.area
    
    # Store results in a dictionary
    results = {
        "tile_id": tile_id,
        "iou_score": round(iou, 4),
        "false_positive_area_sqm": round(fp_area, 2),
        "false_negative_area_sqm": round(fn_area, 2),
        "total_gt_area_sqm": round(gt_union.area, 2),
        "total_pred_area_sqm": round(pred_union.area, 2)
    }
    
    return pd.DataFrame([results])

# --- Test the function ---
if __name__ == "__main__":
    # Replace these with actual paths to test on your local machine
    # gt_file = "../../data/gt_shapes/tile_001_gt.shp"
    # pred_file = "../../data/pred_shapes/tile_001_pred.shp"
    # df_metrics = calculate_polygon_errors(gt_file, pred_file, tile_id="tile_001")
    # print(df_metrics)
    print("Spatial calculator ready. Awaiting data paths to test.")