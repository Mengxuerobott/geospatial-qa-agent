"""
Tests for the image metrics the pipeline feeds to the meta-model.

    pytest tests/ -q
"""

import os
import sys

import numpy as np
import pytest
import rasterio
from rasterio.enums import ColorInterp
from rasterio.transform import from_origin

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.metrics.pipeline import get_image_metrics  # noqa: E402

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
