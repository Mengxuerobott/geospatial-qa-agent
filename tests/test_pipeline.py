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
from rasterio.crs import CRS
from rasterio.enums import ColorInterp
from rasterio.transform import from_origin
from shapely.geometry import LineString, box
from shapely.ops import unary_union

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.metrics.pipeline as pipeline  # noqa: E402
from src.metrics.pipeline import (  # noqa: E402
    agreement,
    crs_problem,
    full_scale,
    get_cell_metrics,
    get_image_metrics,
    image_features,
    split_agreement,
    summarise_tiles,
    train_and_explain,
)
from src.agent.verdict import MATCH_TOLERANCE_M  # noqa: E402

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


def test_reading_in_blocks_gives_the_same_numbers(tmp_path, monkeypatch):
    """The tile is read a block at a time; the totals must not depend on the block size."""
    path = _write_tile(tmp_path / "t.tif", 100, padding_cols=7, spread=10)
    whole = get_image_metrics(path)
    monkeypatch.setattr(pipeline, "READ_BLOCK_PX", 16)
    in_blocks = get_image_metrics(path)
    assert in_blocks == pytest.approx(whole)
    assert in_blocks == pytest.approx((100, 10))


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


def _line(*points):
    return LineString(points)


def _score(gt, pred, region=None):
    return agreement(split_agreement(gt, pred), region)


def test_identical_lines_match_completely():
    trail = _line((0, 10), (100, 10))
    iou, matched, annotated_only, predicted_only = _score(trail, trail)
    assert iou == pytest.approx(1)
    assert (matched, annotated_only, predicted_only) == pytest.approx((100, 0, 0))


def test_a_prediction_alongside_the_annotation_is_the_same_trail():
    """
    The bug this exists for: overlapping two 5 m buffers scored a prediction 3 m to one
    side of the annotation 0.54, a fail, when by eye it is the same trail.
    """
    annotated = _line((0, 10), (100, 10))
    predicted = _line((0, 13), (100, 13))
    assert _score(annotated, predicted)[0] == pytest.approx(1)


def test_a_prediction_beyond_the_tolerance_is_a_different_trail():
    annotated = _line((0, 10), (100, 10))
    predicted = _line((0, 10 + MATCH_TOLERANCE_M + 1), (100, 10 + MATCH_TOLERANCE_M + 1))
    iou, matched, annotated_only, predicted_only = _score(annotated, predicted)
    assert iou == 0
    assert (matched, annotated_only, predicted_only) == pytest.approx((0, 100, 100))


def test_an_unpredicted_stretch_counts_as_annotated_only():
    """The prediction stops at 60 m; the annotation within 5 m of its end still matches."""
    iou, matched, annotated_only, predicted_only = _score(
        _line((0, 10), (100, 10)), _line((0, 10), (60, 10)))
    assert matched == pytest.approx(65)
    assert annotated_only == pytest.approx(35)
    assert predicted_only == pytest.approx(0)
    assert iou == pytest.approx(0.65)


def test_an_unannotated_prediction_counts_as_predicted_only():
    annotated = _line((0, 10), (100, 10))
    predicted = unary_union([annotated, _line((0, 80), (50, 80))])
    iou, matched, annotated_only, predicted_only = _score(annotated, predicted)
    assert (matched, annotated_only, predicted_only) == pytest.approx((100, 0, 50))
    assert iou == pytest.approx(100 / 150)


def test_one_layer_empty_is_total_disagreement_and_both_empty_is_unscored():
    trail = _line((0, 10), (100, 10))
    assert _score(trail, None)[0] == 0
    assert _score(None, trail)[0] == 0
    assert _score(None, None)[0] is None


def test_polygons_are_still_compared_by_overlap():
    iou, matched, annotated_only, predicted_only = _score(box(0, 0, 10, 10), box(0, 0, 10, 5))
    assert iou == pytest.approx(0.5)
    assert (matched, annotated_only, predicted_only) == pytest.approx((50, 50, 0))


def test_cell_with_no_trail_is_not_scored():
    far_away = _line((200, 200), (260, 260))
    assert _score(far_away, far_away, CELL) == (None, 0, 0, 0)


def test_cell_score_only_counts_what_is_inside_the_cell():
    """They agree inside the cell and disagree outside it."""
    annotated = _line((0, 10), (50, 10))
    predicted = _line((0, 10), (500, 10))
    assert _score(annotated, predicted)[0] == pytest.approx(50 / 495)
    assert _score(annotated, predicted, CELL)[0] == pytest.approx(1)


def test_a_match_reaches_across_the_cell_boundary():
    """
    The annotation runs just inside the cell and the prediction just outside it, 4 m
    apart. Matching is done on the whole tile, so the cell sees its trail as matched and
    not as a miss with the prediction lost to the cell next door.
    """
    annotated = _line((0, 48), (50, 48))
    predicted = _line((0, 52), (50, 52))
    iou, matched, annotated_only, predicted_only = _score(annotated, predicted, CELL)
    assert iou == pytest.approx(1)
    assert (annotated_only, predicted_only) == (0, 0)


def test_a_stub_of_trail_is_not_scored():
    assert _score(_line((0, 10), (3, 10)), None, CELL)[0] is None


def _tile_cells(path, gt, pred, cell_size_m=10):
    return get_cell_metrics("t", path, split_agreement(gt, pred), cell_size_m=cell_size_m)


def test_cells_cover_the_tile_and_keep_their_place(tmp_path):
    """40 m tile, 10 m cells: 16 cells whose bounds tile the raster with none missing."""
    path = _write_tile(tmp_path / "t.tif", 100, padding_cols=0)
    cells = _tile_cells(path, None, None)

    assert len(cells) == 16
    assert {(c["cell_row"], c["cell_col"]) for c in cells} == {
        (r, c) for r in range(4) for c in range(4)}
    top_left = next(c for c in cells if (c["cell_row"], c["cell_col"]) == (0, 0))
    assert (top_left["minx"], top_left["miny"], top_left["maxx"], top_left["maxy"]) == (0, 30, 10, 40)
    assert sum((c["maxx"] - c["minx"]) * (c["maxy"] - c["miny"]) for c in cells) == SIZE * SIZE


def test_cells_that_are_all_padding_are_left_out(tmp_path):
    path = _write_tile(tmp_path / "t.tif", 100, padding_cols=SIZE // 2)
    cells = _tile_cells(path, None, None)
    assert len(cells) == 8
    assert all(c["minx"] >= SIZE // 2 for c in cells)


def test_a_trail_is_shared_out_between_cells_without_loss(tmp_path):
    path = _write_tile(tmp_path / "t.tif", 100, padding_cols=0)
    trail = _line((2, 5), (38, 33))
    cells = _tile_cells(path, trail, trail)
    assert sum(c["matched"] for c in cells) == pytest.approx(trail.length)


def test_only_the_cell_where_they_disagree_fails(tmp_path):
    """The tile passes overall; one cell of it does not."""
    path = _write_tile(tmp_path / "t.tif", 100, padding_cols=0)
    annotated = _line((0, 5), (40, 5))
    predicted = _line((0, 5), (25, 5))
    cells = _tile_cells(path, annotated, predicted)
    scored = {c["cell_col"]: c["iou"] for c in cells if c["iou"] is not None}

    assert _score(annotated, predicted)[0] == pytest.approx(0.75)
    assert scored == {0: pytest.approx(1), 1: pytest.approx(1), 2: pytest.approx(1), 3: 0}
    last = next(c for c in cells if c["cell_col"] == 3 and c["iou"] is not None)
    assert (last["matched"], last["annotated_only"], last["predicted_only"]) == pytest.approx((0, 10, 0))


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


# --- Units and bit depth ---

def test_a_crs_in_metres_is_accepted():
    assert crs_problem(CRS.from_epsg(32612)) is None      # UTM zone 12N
    assert crs_problem(CRS.from_epsg(3400)) is None       # Alberta 10-TM


def test_a_crs_in_degrees_is_refused():
    """In degrees a 5 "metre" tolerance matches everything and the tile scores 1.0."""
    assert "degrees" in crs_problem(CRS.from_epsg(4326))


def test_a_crs_in_feet_is_refused():
    assert "not metres" in crs_problem(CRS.from_epsg(2227))   # California zone 3, US feet


def test_a_missing_crs_is_refused():
    assert crs_problem(None) == "it has no CRS"


def _write_deep_tile(path, brightest: int, dark_rows: int = 0) -> str:
    """A 16-bit tile whose pixels reach `brightest`, with its top rows nearly black."""
    rgb = np.full((3, SIZE, SIZE), brightest, dtype=np.uint16)
    rgb[:, :, ::2] = int(brightest * 0.8)
    rgb[:, :dark_rows, :] = int(brightest * 0.05)
    with rasterio.open(path, "w", driver="GTiff", height=SIZE, width=SIZE, count=3,
                       dtype="uint16", crs="EPSG:32612",
                       transform=from_origin(0, SIZE, 1, 1)) as dst:
        dst.write(rgb)
    return str(path)


def test_full_scale_of_8_bit_imagery_is_255(tmp_path):
    with rasterio.open(_write_tile(tmp_path / "t.tif", 40, padding_cols=0)) as src:
        assert full_scale(src, [1, 2, 3]) == 255


def test_full_scale_of_deep_imagery_is_what_it_actually_reaches(tmp_path):
    """A 12-bit camera in a 16-bit file: full scale is about 4000, not 65535."""
    with rasterio.open(_write_deep_tile(tmp_path / "t.tif", brightest=4000)) as src:
        assert full_scale(src, [1, 2, 3]) == pytest.approx(4000, rel=0.01)


def test_deep_imagery_is_not_all_shadow(tmp_path):
    """
    The bug this exists for: measured against 65535, every pixel of 12-bit imagery was
    under the shadow threshold, so shadow_fraction was 1.0 for every cell of every tile.
    """
    path = _write_deep_tile(tmp_path / "t.tif", brightest=4000, dark_rows=SIZE // 4)
    cells = get_cell_metrics("t", path, split_agreement(None, None), cell_size_m=SIZE)
    assert len(cells) == 1
    assert cells[0]["shadow_fraction"] == pytest.approx(0.25)


def test_brightness_is_on_one_scale_whatever_the_bit_depth(tmp_path):
    """So tiles of different depths can be trained on together."""
    deep = get_cell_metrics("t", _write_deep_tile(tmp_path / "deep.tif", brightest=4000),
                            split_agreement(None, None), cell_size_m=SIZE)[0]
    assert 200 < deep["brightness"] <= 255


def test_a_tile_in_degrees_is_skipped_by_the_pipeline(tmp_path, monkeypatch, capsys):
    for folder in ("tiffs", "ground_truth", "predictions"):
        (tmp_path / folder).mkdir()
    with rasterio.open(tmp_path / "tiffs" / "t.tif", "w", driver="GTiff", height=10, width=10,
                       count=3, dtype="uint8", crs="EPSG:4326",
                       transform=from_origin(-111, 45, 0.0001, 0.0001)) as dst:
        dst.write(np.full((3, 10, 10), 100, dtype=np.uint8))
    monkeypatch.setattr(pipeline, "TIFF_DIR", str(tmp_path / "tiffs"))
    monkeypatch.setattr(pipeline, "GT_DIR", str(tmp_path / "ground_truth"))
    monkeypatch.setattr(pipeline, "PRED_DIR", str(tmp_path / "predictions"))
    monkeypatch.setattr(pipeline, "DB_PATH", str(tmp_path / "m.duckdb"))

    pipeline.run_pipeline()

    output = capsys.readouterr().out
    assert "is in degrees, not metres" in output
    assert "Nothing written to the database" in output
    assert not (tmp_path / "m.duckdb").exists()
