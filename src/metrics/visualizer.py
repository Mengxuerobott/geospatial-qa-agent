import os
import rasterio
from rasterio.plot import show
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.metrics.shapefiles import shapefile_uri

def _read_overlay(zip_path: str, tiff_crs):
    """
    Read a zipped shapefile and put it in the TIFF's CRS, the same way the pipeline does
    before computing IoU. Returns None when there is nothing to draw.
    """
    uri = shapefile_uri(zip_path)
    if not uri:
        return None

    gdf = gpd.read_file(uri)
    if gdf.empty:
        return None
    if tiff_crs is not None and gdf.crs is not None and gdf.crs != tiff_crs:
        gdf = gdf.to_crs(tiff_crs)
    return gdf


def plot_tile_results(tiff_path: str, gt_path: str, pred_path: str, tile_id: str):
    """
    Overlays Ground Truth (Green) and Predictions (Red) on top of the RGB TIFF.
    Returns a matplotlib figure that Streamlit can render.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    tiff_crs = None

    # 1. Plot the Base TIFF Image
    if os.path.exists(tiff_path):
        with rasterio.open(tiff_path) as src:
            # rasterio.plot.show automatically handles the RGB rendering
            show(src, ax=ax, title=f"Tile Analysis: {tile_id}")
            # FIX: Lock the camera to the TIFF extent ---
            left, bottom, right, top = src.bounds
            ax.set_xlim(left, right)
            ax.set_ylim(bottom, top)
            tiff_crs = src.crs
    else:
        ax.set_title(f"TIFF Image not found for {tile_id}")
        ax.text(0.5, 0.5, 'Image Missing', horizontalalignment='center', verticalalignment='center')

    # 2. Plot Ground Truth Shapefile (Solid Green Line)
    gt_gdf = _read_overlay(gt_path, tiff_crs)
    if gt_gdf is not None:
        gt_gdf.plot(ax=ax, facecolor="none", edgecolor="green", linewidth=2.5)

    # 3. Plot Predicted Shapefile (Dashed Red Line)
    pred_gdf = _read_overlay(pred_path, tiff_crs)
    if pred_gdf is not None:
        pred_gdf.plot(ax=ax, facecolor="none", edgecolor="red", linewidth=2.5, linestyle="--")

    # 4. Create a Custom Legend
    custom_lines = [
        Line2D([0], [0], color="green", lw=2.5),
        Line2D([0], [0], color="red", lw=2.5, linestyle="--")
    ]
    ax.legend(custom_lines, ['Ground Truth (Actual)', 'Model Prediction'], loc="upper right")
    
    # Hide axis ticks for a cleaner map look
    ax.set_xticks([])
    ax.set_yticks([])
    
    return fig