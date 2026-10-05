import os
import rasterio
from rasterio.plot import show
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from src.agent.cells import cell_name
from src.agent.verdict import FAIL_IOU_THRESHOLD
from src.metrics.shapefiles import shapefile_uri

# One hue, light to dark, for how much of a cell's trail the prediction and the annotation
# disagree on. Blue, because
# green and red already mean ground truth and prediction on this map.
ERROR_CMAP = LinearSegmentedColormap.from_list(
    "cell_error", ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
# Past this many, naming every failing cell hides the map; the worst are named
MAX_CELL_LABELS = 20


def _draw_cells(ax, cells):
    """
    Shade each scored grid cell by how much the prediction and the annotation disagree in
    it, and outline and name the ones below the pass threshold. Returns False when there was nothing to draw.

    A cell that failed is marked by its outline and its name, not by its shade alone, and
    the name is the one to use in the chat ("look at r1c5"). Cells with no trail in them
    were never scored and are left clear.
    """
    scored = cells[cells["iou"].notna()] if cells is not None else None
    if scored is None or scored.empty:
        return False

    for _, cell in scored.iterrows():
        error = 1.0 - cell["iou"]
        failed = cell["iou"] < FAIL_IOU_THRESHOLD
        ax.add_patch(Rectangle(
            (cell["minx"], cell["miny"]), cell["maxx"] - cell["minx"], cell["maxy"] - cell["miny"],
            facecolor=ERROR_CMAP(error), alpha=0.15 + 0.5 * error,
            edgecolor="none", zorder=2))
        if failed:
            ax.add_patch(Rectangle(
                (cell["minx"], cell["miny"]), cell["maxx"] - cell["minx"],
                cell["maxy"] - cell["miny"],
                facecolor="none", edgecolor="white", linewidth=1.5, zorder=3))

    worst = scored[scored["iou"] < FAIL_IOU_THRESHOLD].nsmallest(MAX_CELL_LABELS, "iou")
    for _, cell in worst.iterrows():
        ax.text((cell["minx"] + cell["maxx"]) / 2, (cell["miny"] + cell["maxy"]) / 2,
                cell_name(cell["cell_row"], cell["cell_col"]),
                ha="center", va="center", fontsize=8, color="#1a1a1a", zorder=6,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85))
    return True

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


def plot_tile_results(tiff_path: str, gt_path: str, pred_path: str, tile_id: str, cells=None):
    """
    Overlays Ground Truth (Green) and Predictions (Red) on top of the RGB TIFF.
    Returns a matplotlib figure that Streamlit can render.

    cells is the tile's rows from cell_metrics. When given, the grid cells are shaded by
    how much the prediction and the annotation disagree in each, under the trail lines.
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
        gt_gdf.plot(ax=ax, facecolor="none", edgecolor="green", linewidth=2.5, zorder=4)

    # 3. Plot Predicted Shapefile (Dashed Red Line)
    pred_gdf = _read_overlay(pred_path, tiff_crs)
    if pred_gdf is not None:
        pred_gdf.plot(ax=ax, facecolor="none", edgecolor="red", linewidth=2.5, linestyle="--", zorder=5)

    # 4. Create a Custom Legend
    custom_lines = [
        Line2D([0], [0], color="green", lw=2.5),
        Line2D([0], [0], color="red", lw=2.5, linestyle="--")
    ]
    labels = ['Ground Truth (Actual)', 'Model Prediction']

    # 5. Shade the grid cells by error, with their own entry in the legend and a scale
    if _draw_cells(ax, cells):
        custom_lines.append(Patch(facecolor=ERROR_CMAP(0.8), edgecolor="white", linewidth=1.5))
        labels.append(f"Cell below the pass threshold ({FAIL_IOU_THRESHOLD:.0%} match)")
        scale = fig.colorbar(ScalarMappable(norm=Normalize(0, 100), cmap=ERROR_CMAP), ax=ax,
                             orientation="horizontal", fraction=0.035, pad=0.02)
        scale.set_label("Trail the prediction and the annotation disagree on, per cell (%)")
        scale.outline.set_visible(False)

    ax.legend(custom_lines, labels, loc="upper right", facecolor="white", framealpha=0.9)
    
    # Hide axis ticks for a cleaner map look
    ax.set_xticks([])
    ax.set_yticks([])
    
    return fig