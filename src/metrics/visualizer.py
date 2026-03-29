import os
import rasterio
from rasterio.plot import show
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_tile_results(tiff_path: str, gt_path: str, pred_path: str, tile_id: str):
    """
    Overlays Ground Truth (Green) and Predictions (Red) on top of the RGB TIFF.
    Returns a matplotlib figure that Streamlit can render.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 1. Plot the Base TIFF Image
    if os.path.exists(tiff_path):
        with rasterio.open(tiff_path) as src:
            # rasterio.plot.show automatically handles the RGB rendering
            show(src, ax=ax, title=f"Tile Analysis: {tile_id}")
            # FIX: Lock the camera to the TIFF extent ---
            left, bottom, right, top = src.bounds
            ax.set_xlim(left, right)
            ax.set_ylim(bottom, top)
    else:
        ax.set_title(f"TIFF Image not found for {tile_id}")
        ax.text(0.5, 0.5, 'Image Missing', horizontalalignment='center', verticalalignment='center')

    # 2. Plot Ground Truth Shapefile (Solid Green Line)
    if os.path.exists(gt_path):
        gt_gdf = gpd.read_file(gt_path)
        if not gt_gdf.empty:
            # Ensure CRS matches the TIFF if necessary (assuming they are pre-aligned for now)
            gt_gdf.plot(ax=ax, facecolor="none", edgecolor="green", linewidth=2.5)

    # 3. Plot Predicted Shapefile (Dashed Red Line)
    if os.path.exists(pred_path):
        pred_gdf = gpd.read_file(pred_path)
        if not pred_gdf.empty:
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