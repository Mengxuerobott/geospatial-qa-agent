"""
Tests for the review list of disagreeing cells exported as map features.

    pytest tests/ -q
"""

import json
import os
import sys

import duckdb
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.metrics.review import load_review_list, review_list  # noqa: E402

# UTM zone 12N; 500000 m east is its central meridian, 111 degrees west
X0, Y0 = 500000, 5000000


def _cell(tile_id, row, col, iou, matched=None, annotated_only=None, predicted_only=0.0):
    """A 50 m cell of a 3x3 tile; row 0 is the northern edge."""
    if matched is None:
        matched = 100.0 * (iou or 0)
    if annotated_only is None:
        annotated_only = 100.0 - matched - predicted_only
    return {
        "tile_id": tile_id, "cell_row": row, "cell_col": col,
        "minx": X0 + col * 50, "maxx": X0 + col * 50 + 50,
        "miny": Y0 + 100 - row * 50, "maxy": Y0 + 150 - row * 50,
        "shadow_fraction": 0.8, "matched": matched, "annotated_only": annotated_only,
        "predicted_only": predicted_only, "iou": iou, "shap_shadow_fraction": 0.3,
    }


def _tiles(*tile_ids, crs="EPSG:32612"):
    return pd.DataFrame({"tile_id": list(tile_ids), "crs": crs})


def test_the_list_is_ordered_by_trail_in_dispute():
    cells = pd.DataFrame([_cell("a", 0, 0, 0.0, matched=0.0, annotated_only=6.0),
                          _cell("a", 1, 1, 0.4, matched=40.0, annotated_only=60.0)])
    assert review_list(_tiles("a"), cells)["cell"].to_list() == ["r1c1", "r0c0"]


def test_only_cells_below_the_threshold_are_listed():
    cells = pd.DataFrame([_cell("a", 0, 0, 0.9), _cell("a", 1, 1, 0.5), _cell("a", 2, 2, 0.1),
                          _cell("a", 0, 2, None), _cell("a", 2, 0, 0.75)])
    review = review_list(_tiles("a"), cells)
    assert review["cell"].to_list() == ["r2c2", "r1c1"]
    assert review["location"].to_list() == ["south-east", "centre"]


def test_each_cell_says_how_the_two_disagree():
    cells = pd.DataFrame([
        _cell("a", 0, 0, 0.0, annotated_only=80.0),
        _cell("a", 2, 2, 0.4, matched=40.0, annotated_only=45.0, predicted_only=15.0)])
    review = review_list(_tiles("a"), cells).set_index("cell")
    assert review.loc["r0c0", "disagreement"] == "80 m of annotated trail with no prediction near it"
    assert review.loc["r2c2", "match"] == 0.4
    assert (review.loc["r2c2", "annotated_only_m"], review.loc["r2c2", "predicted_only_m"]) == (45, 15)
    assert review.loc["r2c2", "main_driver"] == "shadow_fraction"


def test_cells_are_placed_in_longitude_and_latitude():
    review = review_list(_tiles("a"), pd.DataFrame([_cell("a", 0, 0, 0.1), _cell("a", 2, 2, 0.9)]))
    assert review.crs.to_string() == "EPSG:4326"
    west, south, east, north = review.geometry.iloc[0].bounds
    assert -111.0 <= west < east < -110.99
    assert 45.1 < south < north < 45.2
    # 50 m across, give or take the projection
    assert review.to_crs("EPSG:32612").geometry.iloc[0].area == pytest.approx(2500, rel=0.01)


def test_tiles_in_different_projections_end_up_on_one_map():
    cells = pd.DataFrame([_cell("a", 0, 0, 0.1), _cell("b", 0, 0, 0.1)])
    tiles = pd.DataFrame({"tile_id": ["a", "b"], "crs": ["EPSG:32612", "EPSG:32613"]})
    review = review_list(tiles, cells)
    assert review["tile_id"].to_list() == ["a", "b"]
    a, b = review.geometry
    assert b.centroid.x - a.centroid.x == pytest.approx(6, abs=0.01)   # one UTM zone east


def test_a_tile_with_no_recorded_crs_is_left_out():
    cells = pd.DataFrame([_cell("a", 0, 0, 0.1), _cell("b", 0, 0, 0.1)])
    tiles = pd.DataFrame({"tile_id": ["a", "b"], "crs": ["EPSG:32612", None]})
    assert review_list(tiles, cells)["tile_id"].to_list() == ["a"]


def test_nothing_to_review_is_still_a_valid_file():
    review = review_list(_tiles("a"), pd.DataFrame([_cell("a", 0, 0, 0.9)]))
    assert review.empty
    assert json.loads(review.to_json())["features"] == []


def test_the_geojson_carries_the_cell_and_its_numbers():
    review = review_list(_tiles("a"), pd.DataFrame([_cell("a", 0, 0, 0.1), _cell("a", 2, 2, 0.9)]))
    feature = json.loads(review.to_json())["features"][0]
    assert feature["geometry"]["type"] == "Polygon"
    assert feature["properties"]["tile_id"] == "a"
    assert feature["properties"]["cell"] == "r0c0"
    assert feature["properties"]["match"] == 0.1


def _database(path, tiles, cells) -> str:
    with duckdb.connect(str(path)) as conn:
        conn.execute("CREATE TABLE tile_metrics AS SELECT * FROM tiles")
        conn.execute("CREATE TABLE cell_metrics AS SELECT * FROM cells")
    return str(path)


def test_the_list_is_read_for_one_tile_or_for_all(tmp_path):
    cells = pd.DataFrame([_cell("a", 0, 0, 0.1), _cell("a", 1, 1, 0.9), _cell("b", 2, 2, 0.2)])
    db = _database(tmp_path / "m.duckdb", _tiles("a", "b"), cells)
    assert load_review_list(db)["tile_id"].to_list() == ["a", "b"]
    assert load_review_list(db, "b")["cell"].to_list() == ["r2c2"]
    assert load_review_list(db, "nope").empty


def test_an_old_database_is_refused_with_a_reason(tmp_path):
    cells = pd.DataFrame([_cell("a", 0, 0, 0.1)])
    db = _database(tmp_path / "m.duckdb", pd.DataFrame({"tile_id": ["a"]}), cells)
    with pytest.raises(ValueError, match="Run the pipeline again"):
        load_review_list(db)
