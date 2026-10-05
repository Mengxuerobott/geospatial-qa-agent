"""
Tests for cutting one grid cell out of a tile for the vision model.

    pytest tests/ -q
"""

import base64
import os
import sys

import cv2
import numpy as np
import rasterio
from rasterio.transform import from_origin

# vision_tool refuses to import without a key. Nothing here calls the API.
os.environ.setdefault("OPENAI_API_KEY", "not-a-real-key")
os.environ["LANGCHAIN_TRACING_V2"] = "false"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agent.cells import parse_cell_name  # noqa: E402
from src.agent.vision_tool import (  # noqa: E402
    ONE_CELL,
    VISION_SYSTEM_PROMPT,
    WHOLE_TILE,
    encode_and_resize_tiff,
)

SIZE = 400


def _write_tile(path) -> str:
    """A 400 m tile at 1 m a pixel: red everywhere, with a blue 100 m square top right."""
    rgb = np.zeros((3, SIZE, SIZE), dtype=np.uint8)
    rgb[0] = 200
    rgb[:, :100, 300:] = 0
    rgb[2, :100, 300:] = 200
    with rasterio.open(path, "w", driver="GTiff", height=SIZE, width=SIZE, count=3,
                       dtype="uint8", crs="EPSG:32612",
                       transform=from_origin(0, SIZE, 1, 1)) as dst:
        dst.write(rgb)
    return str(path)


def _decode(b64: str) -> np.ndarray:
    """The JPEG as an RGB array."""
    bgr = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def test_bounds_cut_out_that_part_of_the_tile(tmp_path):
    """Map bounds of the blue square: x 300-400, y 300-400 (north is up)."""
    img = _decode(encode_and_resize_tiff(_write_tile(tmp_path / "t.tif"),
                                         bounds=(300, 300, 400, 400)))
    assert img.shape[:2] == (100, 100)
    red, _, blue = img.reshape(-1, 3).mean(axis=0)
    assert blue > 150 and red < 50


def test_a_cell_elsewhere_does_not_show_the_square(tmp_path):
    img = _decode(encode_and_resize_tiff(_write_tile(tmp_path / "t.tif"),
                                         bounds=(0, 0, 100, 100)))
    red, _, blue = img.reshape(-1, 3).mean(axis=0)
    assert red > 150 and blue < 50


def test_a_crop_keeps_its_detail_where_the_whole_tile_is_shrunk(tmp_path):
    path = _write_tile(tmp_path / "t.tif")
    assert _decode(encode_and_resize_tiff(path, max_size=200)).shape[:2] == (200, 200)
    assert _decode(encode_and_resize_tiff(path, max_size=200,
                                          bounds=(300, 300, 400, 400))).shape[:2] == (100, 100)


def test_bounds_past_the_edge_are_clipped_to_the_tile(tmp_path):
    img = _decode(encode_and_resize_tiff(_write_tile(tmp_path / "t.tif"),
                                         bounds=(350, 350, 450, 450)))
    assert img.shape[:2] == (50, 50)


def test_no_bounds_still_sends_the_whole_tile(tmp_path):
    img = _decode(encode_and_resize_tiff(_write_tile(tmp_path / "t.tif")))
    assert img.shape[:2] == (SIZE, SIZE)


def test_the_prompt_says_whether_the_image_is_a_tile_or_a_cell():
    whole = VISION_SYSTEM_PROMPT.format(image=WHOLE_TILE)
    cell = VISION_SYSTEM_PROMPT.format(image=ONE_CELL.format(location="north-east"))
    assert "downscaled" in whole and "north-east" not in whole
    assert "north-east" in cell and "downscaled" not in cell
    # Neither tells the model the area did badly; that would presuppose what it should check
    assert "do not assume it did badly" in whole and "do not assume it did badly" in cell


def test_cell_names_are_parsed_and_anything_else_refused():
    assert parse_cell_name("r3c5") == (3, 5)
    assert parse_cell_name(" R12C0 ") == (12, 0)
    for bad in ("3,5", "r3", "r3c5; DROP TABLE cell_metrics", "r-1c2", "", None, "../r1c1"):
        assert parse_cell_name(bad) is None
