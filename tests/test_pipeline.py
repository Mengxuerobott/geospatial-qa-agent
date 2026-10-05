"""
Tests for the image and per-cell metrics the pipeline feeds to the meta-model.

    pytest tests/ -q
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.enums import ColorInterp
from rasterio.transform import from_origin
from shapely.geometry import LineString, box

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.metrics.pipeline import (  # noqa: E402
    cell_iou,
    get_cell_metrics,
    get_image_metrics,
    image_features,
    overlap_iou,
    summarise_tiles,
    train_and_explain,
)

SIZE = 40


def _write_tile(path, rgb_value: int, padding_cols: int, spread: int = 0) -> str:
    """
    An RGBA tile whose left `padding_cols` columns are black, transparent padding and whose
    remaining pixels are `rgb_value`, +/- `spread` in alternating rows.
    """
    rgb = np.full((3, SIZE, SIZE), rgb_value, dtype=np.uint8)
    rgb[:, ::2, :] += spread
    rgb[:, 1::2, :] -= spread
    alpha = np.full((SIZE, SIZE), 255, dtype=np.uint8)
    rgb[:, :, :padding_cols] = 0
    alpha[:, :padding_cols] = 0

    with rasterio.open(path, "w", driver="GTiff", height=SIZE, width=SIZE, count=4,
                       dtype="uint8", crs="EPSG:32612",
                       transform=from_origin(0, SIZE, 1, 1)) as dst:
        dst.write(rgb, [1, 2, 3])
        dst.write(alpha, 4)
        dst.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue,
                           ColorInterp.alpha]
    return str(path)


def test_brightness_is_the_mean_of_the_imagery(tmp_path):
    brightness, _ = get_image_metrics(_write_tile(tmp_path / "t.tif", 100, padding_cols=0))
    assert brightness == pytest.approx(100)


def test_padding_does_not_change_brightness(tmp_path):
    """The bug this exists for: a tile that is 50% padding looked half as bright."""
    full, _ = get_image_metrics(_write_tile(tmp_path / "full.tif", 100, padding_cols=0))
    half, _ = get_image_metrics(_write_tile(tmp_path / "half.tif", 100, padding_cols=SIZE // 2))
    assert half == pytest.approx(full)


def test_padding_does_not_change_contrast(tmp_path):
    """Black padding next to imagery is a huge spread; it must not count as contrast."""
    _, full = get_image_metrics(_write_tile(tmp_path / "full.tif", 100, 0, spread=10))
    _, half = get_image_metrics(_write_tile(tmp_path / "half.tif", 100, SIZE // 2, spread=10))
    assert full == pytest.approx(10)
    assert half == pytest.approx(full)


def test_alpha_band_is_not_averaged_in(tmp_path):
    """Alpha is 255 over all valid pixels; including it pulled every tile towards 255."""
    brightness, contrast = get_image_metrics(_write_tile(tmp_path / "t.tif", 40, padding_cols=0))
    assert brightness == pytest.approx(40)
    assert contrast == pytest.approx(0)


def test_tile_with_no_valid_pixels_is_rejected(tmp_path):
    assert get_image_metrics(_write_tile(tmp_path / "t.tif", 100, padding_cols=SIZE)) == (None, None)


# --- Per-cell metrics ---

def _features(rgb, valid=None):
    img = np.array(rgb, dtype=np.uint8)
    if valid is None:
        valid = np.ones(img.shape[1:], dtype=bool)
    return image_features(img, valid, 255.0)


def _flat(r, g, b, size=6):
    return [np.full((size, size), v) for v in (r, g, b)]


def test_shadow_fraction_counts_dark_pixels():
    rgb = np.array(_flat(200, 200, 200))
    rgb[:, :3, :] = 20
    assert _features(rgb)["shadow_fraction"] == pytest.approx(0.5)


def test_a_dark_pixel_with_one_bright_band_is_not_shadow():
    """Saturated green vegetation is dark on average but it is lit."""
    assert _features(_flat(10, 220, 10))["shadow_fraction"] == 0


def test_greenness_separates_vegetation_from_bare_ground():
    assert _features(_flat(100, 100, 100))["greenness"] == pytest.approx(0)
    assert _features(_flat(40, 160, 40))["greenness"] > 0.4


def test_greenness_ignores_how_bright_the_tile_is():
    dim = _features(_flat(20, 60, 20))["greenness"]
    bright = _features(_flat(60, 180, 60))["greenness"]
    assert dim == pytest.approx(bright)


def test_sharpness_is_zero_for_flat_imagery_and_high_for_detail():
    checker = np.indices((8, 8)).sum(axis=0) % 2 * 200
    assert _features(_flat(100, 100, 100))["sharpness"] == pytest.approx(0)
    assert _features([checker, checker, checker])["sharpness"] > 1000


def test_the_edge_of_the_padding_is_not_sharpness():
    """Black padding next to imagery is the hardest edge in the tile and means nothing."""
    rgb = np.array(_flat(100, 100, 100, size=8))
    valid = np.ones((8, 8), dtype=bool)
    rgb[:, :, :4] = 0
    valid[:, :4] = False
    features = _features(rgb, valid)
    assert features["sharpness"] == pytest.approx(0)
    assert features["brightness"] == pytest.approx(100)
    assert features["shadow_fraction"] == 0


def test_features_of_all_padding_are_none():
    assert _features(_flat(0, 0, 0), np.zeros((6, 6), dtype=bool)) is None


CELL = box(0, 0, 50, 50)


def test_cell_with_no_trail_is_not_scored():
    far_away = box(200, 200, 260, 260)
    iou, gt_area, pred_area = cell_iou(far_away, far_away, CELL)
    assert iou is None
    assert (gt_area, pred_area) == (0, 0)


def test_cell_with_no_geometry_at_all_is_not_scored():
    assert cell_iou(None, None, CELL)[0] is None


def test_missed_trail_scores_zero():
    assert cell_iou(box(0, 0, 50, 10), None, CELL)[0] == 0


def test_false_positive_scores_zero():
    assert cell_iou(None, box(0, 0, 50, 10), CELL)[0] == 0


def test_cell_iou_only_counts_what_is_inside_the_cell():
    """They agree inside the cell and disagree outside it."""
    gt = box(0, 0, 50, 10)
    pred = box(0, 0, 500, 10)
    assert overlap_iou(gt, pred) == pytest.approx(0.1)
    assert cell_iou(gt, pred, CELL)[0] == pytest.approx(1)


def test_a_sliver_of_trail_is_not_scored():
    assert cell_iou(box(0, 0, 50, 0.1), None, CELL)[0] is None


def test_cells_cover_the_tile_and_keep_their_place(tmp_path):
    """40 m tile, 10 m cells: 16 cells whose bounds tile the raster with none missing."""
    path = _write_tile(tmp_path / "t.tif", 100, padding_cols=0)
    cells = get_cell_metrics("t", path, None, None, cell_size_m=10)

    assert len(cells) == 16
    assert {(c["cell_row"], c["cell_col"]) for c in cells} == {
        (r, c) for r in range(4) for c in range(4)}
    top_left = next(c for c in cells if (c["cell_row"], c["cell_col"]) == (0, 0))
    assert (top_left["minx"], top_left["miny"], top_left["maxx"], top_left["maxy"]) == (0, 30, 10, 40)
    assert sum((c["maxx"] - c["minx"]) * (c["maxy"] - c["miny"]) for c in cells) == SIZE * SIZE


def test_cells_that_are_all_padding_are_left_out(tmp_path):
    path = _write_tile(tmp_path / "t.tif", 100, padding_cols=SIZE // 2)
    cells = get_cell_metrics("t", path, None, None, cell_size_m=10)
    assert len(cells) == 8
    assert all(c["minx"] >= SIZE // 2 for c in cells)


def test_a_buffered_trail_is_shared_out_between_cells_without_loss(tmp_path):
    """
    The trail runs along a cell boundary. Its buffer is made on the whole tile, so the
    cells on both sides hold their share and the shares add up to the whole.
    """
    path = _write_tile(tmp_path / "t.tif", 100, padding_cols=0)
    trail = LineString([(5, 20), (35, 20)]).buffer(3)
    cells = get_cell_metrics("t", path, trail, trail, cell_size_m=10)

    assert sum(c["gt_area"] for c in cells) == pytest.approx(trail.area)
    above = [c for c in cells if c["cell_row"] == 1 and c["gt_area"] > 0]
    below = [c for c in cells if c["cell_row"] == 2 and c["gt_area"] > 0]
    assert len(above) == len(below) == 4


def test_only_the_cell_where_the_model_missed_fails(tmp_path):
    """The tile passes overall; one cell of it does not."""
    path = _write_tile(tmp_path / "t.tif", 100, padding_cols=0)
    gt = box(0, 2, 40, 8)
    pred = box(0, 2, 30, 8)
    cells = get_cell_metrics("t", path, gt, pred, cell_size_m=10)
    scored = {c["cell_col"]: c["iou"] for c in cells if c["iou"] is not None}

    assert overlap_iou(gt, pred) == pytest.approx(0.75)
    assert scored == {0: pytest.approx(1), 1: pytest.approx(1), 2: pytest.approx(1), 3: 0}


def _cells(rows):
    """rows of (tile_id, shadow_fraction, iou); the other features are held constant."""
    return pd.DataFrame([
        {"tile_id": tile_id, "brightness": 100.0, "contrast": 10.0,
         "shadow_fraction": shadow, "greenness": 0.1, "sharpness": 50.0, "iou": iou}
        for tile_id, shadow, iou in rows
    ])


def test_shap_blames_the_feature_that_drives_the_error():
    cells = _cells([("a", 0.9, 0.1)] * 20 + [("b", 0.0, 0.95)] * 20)
    cells = train_and_explain(cells)

    shadowed = cells[cells["tile_id"] == "a"]
    assert (shadowed["shap_shadow_fraction"] > 0).all()
    assert (cells[cells["tile_id"] == "b"]["shap_shadow_fraction"] < 0).all()
    assert shadowed["shap_brightness"].abs().max() < shadowed["shap_shadow_fraction"].min()


def test_unscored_cells_are_not_trained_on_and_get_no_shap():
    cells = _cells([("a", 0.9, 0.1)] * 10 + [("a", 0.0, 0.95)] * 10 + [("a", 0.5, None)] * 5)
    cells = train_and_explain(cells)
    assert cells[cells["iou"].isna()]["shap_shadow_fraction"].isna().all()
    assert cells[cells["iou"].notna()]["shap_shadow_fraction"].notna().all()


def test_tile_summary_counts_failing_cells_and_averages_shap():
    cells = _cells([("a", 0.9, 0.1)] * 3 + [("a", 0.0, 0.95)] * 7 + [("a", 0.5, None)] * 2
                   + [("b", 0.0, 0.9)] * 4)
    cells = train_and_explain(cells)
    tiles = pd.DataFrame([{"tile_id": t, "brightness": 100.0, "contrast": 10.0, "iou": 0.8}
                          for t in ("a", "b", "no-trail")])
    tiles = summarise_tiles(tiles, cells).set_index("tile_id")

    assert (tiles.loc["a", "cells_scored"], tiles.loc["a", "cells_failing"]) == (10, 3)
    assert (tiles.loc["b", "cells_scored"], tiles.loc["b", "cells_failing"]) == (4, 0)
    assert tiles.loc["no-trail", "cells_scored"] == 0
    assert tiles.loc["a", "shap_shadow_fraction"] == pytest.approx(
        cells[(cells["tile_id"] == "a") & cells["iou"].notna()]["shap_shadow_fraction"].mean())
    assert {"iou", "shap_brightness", "shap_contrast"} <= set(tiles.columns)
