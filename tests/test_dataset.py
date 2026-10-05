"""
Tests for choosing which tiles the eval examples are written against.

    pytest tests/ -q
"""

import pytest

from evals.dataset import MISSING_TILE, build_examples, pick_tiles

IOUS = {"a": 0.31, "b": 0.62, "c": 0.74, "d": 0.76, "e": 0.88, "f": 0.93, "g": 0.99}


def test_each_part_goes_to_the_tile_that_fits_it():
    tiles = pick_tiles(IOUS)
    assert tiles["worst"] == "a"
    assert tiles["good"] == ["g", "f", "e"]
    assert tiles["near_fail"] == "c"      # the failing tile closest to 0.75
    assert tiles["near_pass"] == "d"      # the passing tile closest to 0.75


def test_the_parts_follow_the_scores_when_they_change():
    """
    The bug this exists for: tile IDs were written into the dataset by hand, and stayed
    there after the IoU was redefined and their verdicts moved.
    """
    rescored = dict(IOUS, a=0.97, c=0.80, g=0.40)
    tiles = pick_tiles(rescored)
    assert tiles["worst"] == "g"
    assert "g" not in tiles["good"] and "a" in tiles["good"]
    assert tiles["near_fail"] == "b"
    assert tiles["near_pass"] == "d"


def test_a_database_where_nothing_fails_is_refused_with_a_reason():
    with pytest.raises(SystemExit, match="No tile in the database fails QA"):
        pick_tiles({"a": 0.9, "b": 0.95, "c": 0.99})


def test_too_few_passing_tiles_is_refused_with_a_reason():
    with pytest.raises(SystemExit, match="need three"):
        pick_tiles({"a": 0.2, "b": 0.9, "c": 0.95})


def test_the_examples_are_written_against_the_chosen_tiles():
    examples = build_examples(IOUS)
    questions = " ".join(e["question"] for e in examples)
    assert len(examples) == 19
    for tile_id in ("a", "c", "d", "e", "f", "g"):
        assert f" {tile_id}" in questions or f"{tile_id} " in questions
    assert MISSING_TILE in questions


def test_reference_values_come_from_the_scores_given():
    examples = build_examples(IOUS)
    first = next(e for e in examples if e["question"] == "What is the IoU for tile a?")
    assert first["expected_iou"] == 0.31
    verdicts = {e["question"]: e["expected_verdict"] for e in examples if "expected_verdict" in e}
    assert verdicts["Did c pass QA?"] == "failed"
    assert verdicts["d failed QA, right?"] == "passed"
