"""
Tests for the grid cells drawn on the map viewer.

    pytest tests/ -q
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import rasterio  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.metrics.visualizer as visualizer  # noqa: E402
from src.metrics.visualizer import MAX_CELL_LABELS, _draw_cells, plot_tile_results  # noqa: E402


def _cells(ious):
    return pd.DataFrame([
        {"cell_row": 0, "cell_col": col, "minx": col * 50, "maxx": col * 50 + 50,
         "miny": 0, "maxy": 50, "iou": iou}
        for col, iou in enumerate(ious)
    ])


@pytest.fixture
def ax():
    fig, ax = plt.subplots()
    yield ax
    plt.close(fig)


def _outlined(ax):
    return [p for p in ax.patches if isinstance(p, Rectangle) and p.get_linewidth() > 0
            and p.get_facecolor()[3] == 0]


def test_nothing_is_drawn_without_cells(ax):
    assert _draw_cells(ax, None) is False
    assert _draw_cells(ax, _cells([None, None])) is False
    assert not ax.patches


def test_only_scored_cells_are_shaded(ax):
    assert _draw_cells(ax, _cells([0.9, None, 0.8])) is True
    assert len(ax.patches) == 2


def test_failing_cells_are_outlined_and_named_as_well_as_shaded(ax):
    """A failed cell is not marked by its shade alone."""
    _draw_cells(ax, _cells([0.9, 0.2, 0.75, 0.74]))
    assert len(_outlined(ax)) == 2
    assert sorted(t.get_text() for t in ax.texts) == ["r0c1", "r0c3"]


def test_a_worse_cell_is_shaded_more_heavily(ax):
    _draw_cells(ax, _cells([0.9, 0.1]))
    good, bad = ax.patches[0], ax.patches[1]
    assert bad.get_alpha() > good.get_alpha()
    assert sum(bad.get_facecolor()[:3]) < sum(good.get_facecolor()[:3])


def test_only_the_worst_cells_are_named_when_there_are_many(ax):
    count = MAX_CELL_LABELS + 5
    _draw_cells(ax, _cells([i / (count * 2) for i in range(count)]))
    assert len(_outlined(ax)) == count
    assert len(ax.texts) == MAX_CELL_LABELS
    assert "r0c0" in {t.get_text() for t in ax.texts}


def test_the_map_still_draws_without_cells_or_files(tmp_path):
    fig = plot_tile_results(str(tmp_path / "none.tif"), str(tmp_path / "gt.zip"),
                            str(tmp_path / "pred.zip"), "t")
    assert len(fig.axes) == 1
    plt.close(fig)


def test_cells_add_a_scale_and_a_legend_entry(tmp_path):
    fig = plot_tile_results(str(tmp_path / "none.tif"), str(tmp_path / "gt.zip"),
                            str(tmp_path / "pred.zip"), "t", cells=_cells([0.9, 0.2]))
    assert len(fig.axes) == 2
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert len(labels) == 3 and "below the pass threshold" in labels[2]
    plt.close(fig)


def _write_tile(path, size=200, dtype="uint8", value=200) -> str:
    """A square tile, 1 m a pixel, red on its left half and blue on its right."""
    rgb = np.zeros((3, size, size), dtype=dtype)
    rgb[0, :, : size // 2] = value
    rgb[2, :, size // 2:] = value
    with rasterio.open(path, "w", driver="GTiff", height=size, width=size, count=3,
                       dtype=dtype, crs="EPSG:32612",
                       transform=from_origin(1000, 5000, 1, 1)) as dst:
        dst.write(rgb)
    return str(path)


def _base_image(fig):
    return fig.axes[0].images[0]


def test_the_map_is_read_no_larger_than_it_is_drawn(tmp_path, monkeypatch):
    """The bug this exists for: every pixel of a 300 MB TIFF was read to draw a 1000 px map."""
    monkeypatch.setattr(visualizer, "MAX_MAP_PX", 50)
    fig = plot_tile_results(_write_tile(tmp_path / "t.tif"), "none.zip", "none.zip", "t")
    assert _base_image(fig).get_array().shape[:2] == (50, 50)
    plt.close(fig)


def test_a_small_tile_is_not_enlarged(tmp_path):
    fig = plot_tile_results(_write_tile(tmp_path / "t.tif"), "none.zip", "none.zip", "t")
    assert _base_image(fig).get_array().shape[:2] == (200, 200)
    plt.close(fig)


def test_the_shrunk_image_still_sits_on_the_tile_s_map_extent(tmp_path, monkeypatch):
    """Cells and trails are drawn in map coordinates, so the image must be too."""
    monkeypatch.setattr(visualizer, "MAX_MAP_PX", 50)
    fig = plot_tile_results(_write_tile(tmp_path / "t.tif"), "none.zip", "none.zip", "t")
    image = _base_image(fig)
    assert tuple(image.get_extent()) == (1000, 1200, 4800, 5000)
    left_half, right_half = image.get_array()[25, 10], image.get_array()[25, 40]
    assert left_half[0] > 0.7 and left_half[2] < 0.1      # red in the west
    assert right_half[2] > 0.7 and right_half[0] < 0.1    # blue in the east
    plt.close(fig)


def test_sixteen_bit_imagery_is_stretched_into_view(tmp_path):
    path = _write_tile(tmp_path / "t.tif", dtype="uint16", value=4000)
    fig = plot_tile_results(path, "none.zip", "none.zip", "t")
    assert _base_image(fig).get_array().max() == pytest.approx(1.0)
    plt.close(fig)
