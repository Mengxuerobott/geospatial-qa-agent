"""
Tests for cutting one grid cell out of a tile for the vision model.

    pytest tests/ -q
"""

import base64
import os
import sys
import zipfile

import cv2
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString

# vision_tool refuses to import without a key. Nothing here calls the API.
os.environ.setdefault("OPENAI_API_KEY", "not-a-real-key")
os.environ["LANGCHAIN_TRACING_V2"] = "false"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agent.cells import parse_cell_name  # noqa: E402
import src.agent.vision_tool as vision_tool  # noqa: E402
from src.agent.vision_tool import (  # noqa: E402
    ONE_CELL,
    VISION_SYSTEM_PROMPT,
    WHOLE_TILE,
    analyze_image_visually,
    encode_and_resize_tiff,
    encode_cell_with_trails,
)
from src.metrics.shapefiles import trails_within  # noqa: E402

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


# --- Trails drawn on the crop ---

CELL = (0, 0, 100, 100)   # bottom-left square of the tile: all red, rows 300-400 of the raster
ANNOTATED = LineString([(0, 80), (100, 80)])   # 20 px below the top of the crop
PREDICTED = LineString([(30, 0), (30, 100)])   # 30 px in from its left edge


def _is(pixel, colour) -> bool:
    """JPEG smears colours a little, so compare loosely. colour is RGB."""
    return all(abs(int(a) - b) < 70 for a, b in zip(pixel, colour))


CYAN, MAGENTA, RED = (0, 255, 255), (255, 0, 255), (200, 0, 0)


def test_the_first_image_has_no_lines_on_it(tmp_path):
    """A line drawn over a thin trail hides it, so the ground is also sent untouched."""
    plain, _ = encode_cell_with_trails(_write_tile(tmp_path / "t.tif"), CELL, ANNOTATED, PREDICTED)
    plain = _decode(plain)
    assert _is(plain[20, 60], RED) and _is(plain[60, 30], RED)


def test_each_line_is_drawn_where_it_runs_in_its_own_colour(tmp_path):
    _, marked = encode_cell_with_trails(_write_tile(tmp_path / "t.tif"), CELL, ANNOTATED, PREDICTED)
    marked = _decode(marked)
    assert marked.shape[:2] == (100, 100)
    assert _is(marked[20, 60], CYAN)        # y = 80 m is 20 px down from the top at 100 m
    assert _is(marked[60, 30], MAGENTA)     # x = 30 m is 30 px in from the left
    assert _is(marked[60, 60], RED)         # and nothing anywhere else


def test_lines_are_scaled_with_the_image(tmp_path):
    _, marked = encode_cell_with_trails(_write_tile(tmp_path / "t.tif"), CELL, ANNOTATED,
                                        PREDICTED, max_size=50)
    marked = _decode(marked)
    assert marked.shape[:2] == (50, 50)
    assert _is(marked[10, 30], CYAN)
    assert _is(marked[30, 15], MAGENTA)


def test_one_layer_may_be_absent(tmp_path):
    _, marked = encode_cell_with_trails(_write_tile(tmp_path / "t.tif"), CELL, None, PREDICTED)
    marked = _decode(marked)
    assert _is(marked[60, 30], MAGENTA)
    assert _is(marked[20, 60], RED)


def _zip_trails(tmp_path, name, lines) -> str:
    folder = tmp_path / name
    folder.mkdir()
    gpd.GeoDataFrame(geometry=lines, crs="EPSG:32612").to_file(folder / f"{name}.shp")
    archive = tmp_path / f"{name}.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for file in folder.iterdir():
            z.write(file, f"{name}/{file.name}")
    return str(archive)


def test_trails_are_clipped_to_the_crop(tmp_path):
    archive = _zip_trails(tmp_path, "gt", [LineString([(50, 50), (350, 50)]),
                                           LineString([(300, 300), (390, 390)])])
    inside = trails_within(archive, "EPSG:32612", CELL)
    assert inside.length == 50
    assert trails_within(archive, "EPSG:32612", (0, 200, 100, 300)) is None
    assert trails_within(str(tmp_path / "missing.zip"), "EPSG:32612", CELL) is None


class _FakeLLM:
    """Stands in for ChatOpenAI and keeps what it was sent."""
    sent = None

    def __init__(self, **_):
        pass

    def invoke(self, messages):
        _FakeLLM.sent = messages
        return type("Reply", (), {"content": "seen"})()


def _sent_to_model(monkeypatch, tmp_path, **kwargs):
    monkeypatch.setattr(vision_tool, "ChatOpenAI", _FakeLLM)
    assert analyze_image_visually(_write_tile(tmp_path / "t.tif"), "what is here?", **kwargs) == "seen"
    system, human = _FakeLLM.sent
    images = [part for part in human.content if part["type"] == "image_url"]
    return system.content, images


def test_a_cell_with_trails_is_sent_twice_with_the_colours_explained(monkeypatch, tmp_path):
    zips = (_zip_trails(tmp_path, "gt", [ANNOTATED]), _zip_trails(tmp_path, "pred", [PREDICTED]))
    prompt, images = _sent_to_model(monkeypatch, tmp_path, bounds=CELL, location="south-west",
                                    trail_zips=zips)
    assert len(images) == 2
    assert "Cyan lines are trails drawn by a human annotator" in prompt
    assert "Magenta lines are the trails the model predicted" in prompt
    assert "within 5 metres" in prompt and "100 metres across" in prompt
    # It must be free to say it cannot tell, and must not be told who is right
    assert "cannot tell" in prompt
    assert "Neither the annotator nor the model" in prompt


def test_a_cell_with_no_trails_in_it_is_sent_once_as_before(monkeypatch, tmp_path):
    elsewhere = LineString([(300, 300), (390, 390)])
    zips = (_zip_trails(tmp_path, "gt", [elsewhere]), _zip_trails(tmp_path, "pred", [elsewhere]))
    prompt, images = _sent_to_model(monkeypatch, tmp_path, bounds=CELL, trail_zips=zips)
    assert len(images) == 1
    assert "Cyan" not in prompt


def test_unreadable_trail_files_do_not_stop_the_crop_being_sent(monkeypatch, tmp_path):
    zips = (str(tmp_path / "missing.zip"), str(tmp_path / "also-missing.zip"))
    prompt, images = _sent_to_model(monkeypatch, tmp_path, bounds=CELL, trail_zips=zips)
    assert len(images) == 1


def test_the_whole_tile_is_never_drawn_on(monkeypatch, tmp_path):
    zips = (_zip_trails(tmp_path, "gt", [ANNOTATED]), _zip_trails(tmp_path, "pred", [PREDICTED]))
    prompt, images = _sent_to_model(monkeypatch, tmp_path, trail_zips=zips)
    assert len(images) == 1
    assert "Cyan" not in prompt and "downscaled" in prompt
