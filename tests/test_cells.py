"""
Tests for the description of a tile's weak areas given to the LLM.

    pytest tests/ -q
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evals.evaluators import iou_grounded  # noqa: E402
from src.agent.cells import cell_name, compass, describe_weak_cells  # noqa: E402

EXTENT = (0, 0, 300, 300)


def _cell(row, col, iou, matched=None, annotated_only=None, predicted_only=0.0,
          shadow=0.0, shap_shadow=0.0):
    """A 100 m cell of a 3x3 tile; row 0 is the northern edge."""
    # By default the disagreement is all annotated trail the model did not predict
    trail = 100.0
    if matched is None:
        matched = trail * (iou or 0)
    if annotated_only is None:
        annotated_only = trail - matched - predicted_only
    return {
        "tile_id": "t", "cell_row": row, "cell_col": col,
        "minx": col * 100, "maxx": col * 100 + 100,
        "miny": 200 - row * 100, "maxy": 300 - row * 100,
        "brightness": 120.0, "shadow_fraction": shadow,
        "matched": matched, "annotated_only": annotated_only,
        "predicted_only": predicted_only, "iou": iou,
        "shap_brightness": -0.01, "shap_shadow_fraction": shap_shadow,
    }


def _tile(*cells):
    return pd.DataFrame(cells)


def test_compass_names_the_ninth_of_the_tile():
    where = {(r, c): compass(pd.Series(_cell(r, c, 1.0)), EXTENT)
             for r in range(3) for c in range(3)}
    assert where[(0, 0)] == "north-west"
    assert where[(0, 2)] == "north-east"
    assert where[(1, 1)] == "centre"
    assert where[(1, 0)] == "west"
    assert where[(2, 1)] == "south"


def test_cell_name():
    assert cell_name(3, 5) == "r3c5"


def test_tile_with_no_weak_cells_says_so():
    text = describe_weak_cells(_tile(_cell(0, 0, 0.9), _cell(1, 1, 0.8), _cell(2, 2, None)))
    assert text.startswith("Weak areas: none")
    assert "all 2 grid cells" in text


def test_a_cell_exactly_on_the_threshold_is_not_weak():
    assert describe_weak_cells(_tile(_cell(0, 0, 0.75))).startswith("Weak areas: none")


def test_tile_with_no_scored_cells_is_unknown_not_clean():
    text = describe_weak_cells(_tile(_cell(0, 0, None), _cell(1, 1, None)))
    assert "unknown" in text
    assert "none" not in text


def test_weak_cells_are_counted_located_and_named():
    text = describe_weak_cells(_tile(
        _cell(0, 2, 0.2), _cell(1, 2, 0.5), _cell(1, 1, 0.9), _cell(2, 0, 0.95),
        _cell(2, 2, None)))
    assert "2 of 4 grid cells" in text
    assert "1 in the north-east" in text and "1 in the east" in text
    assert "cell r0c2 (north-east)" in text
    assert "r1c1" not in text


def test_worst_cell_comes_first():
    text = describe_weak_cells(_tile(_cell(0, 0, 0.6), _cell(2, 2, 0.1), _cell(1, 1, 0.3)))
    assert text.index("r2c2") < text.index("r1c1") < text.index("r0c0")


def test_the_two_directions_of_disagreement_are_told_apart():
    text = describe_weak_cells(_tile(
        _cell(0, 0, 0.0, annotated_only=80.0),
        _cell(2, 2, 0.0, annotated_only=0.0, predicted_only=60.0),
        _cell(1, 1, 0.4, matched=40.0, annotated_only=45.0, predicted_only=15.0)))
    assert "r0c0 (north-west): 80 m of annotated trail with no prediction near it" in text
    assert "r2c2 (south-east): 60 m of predicted trail with no annotation near it" in text
    assert ("r1c1 (centre): 40% of the trail matched, 45 m annotated but not predicted "
            "and 15 m predicted but not annotated") in text


def test_only_the_direction_that_happened_is_mentioned():
    text = describe_weak_cells(_tile(_cell(1, 1, 0.5, matched=50.0, annotated_only=50.0)))
    assert "50 m annotated but not predicted" in text
    assert "predicted but not annotated" not in text


def test_the_wording_does_not_say_who_is_wrong():
    """The annotation is a person's work; a disagreement is not proof the model erred."""
    text = describe_weak_cells(_tile(
        _cell(0, 0, 0.0, annotated_only=80.0),
        _cell(2, 2, 0.0, annotated_only=0.0, predicted_only=60.0))).lower()
    for blame in ("missed", "false", "wrong", "error", "where there is none"):
        assert blame not in text


def test_main_driver_is_the_largest_positive_shap_with_its_value():
    text = describe_weak_cells(_tile(_cell(0, 0, 0.1, shadow=0.8, shap_shadow=0.3)))
    assert "main driver shadow_fraction (0.80)" in text


def test_no_driver_is_named_when_nothing_pushed_the_error_up():
    text = describe_weak_cells(_tile(_cell(0, 0, 0.1, shap_shadow=-0.2)))
    assert "main driver" not in text


def test_only_the_worst_few_are_listed_and_the_rest_counted():
    cells = [_cell(r, c, 0.1 * (r * 3 + c) / 10) for r in range(3) for c in range(3)]
    text = describe_weak_cells(_tile(*cells))
    assert "9 of 9" in text
    assert text.count("cell r") == 5
    assert "and 4 more" in text


def test_cell_scores_cannot_be_mistaken_for_the_tile_iou():
    """
    The grounding scorer flags any decimal presented as an IoU that is not the tile's.
    An answer that repeats the weak-area text next to the true IoU must still pass it.
    """
    weak = describe_weak_cells(_tile(_cell(0, 0, 0.42), _cell(1, 1, 0.9)))
    assert "IoU" not in weak and "iou" not in weak
    answer = f"The tile passed with an IoU of 0.93. {weak}"
    assert iou_grounded({"answer": answer}, {"expected_iou": 0.93})["score"] == 1


def test_cells_are_ordered_by_trail_in_dispute_not_by_score():
    """
    A stub of unmatched trail scores 0 and used to be listed first. A cell with ten times
    as much in dispute is the one to check first, whatever its score.
    """
    text = describe_weak_cells(_tile(
        _cell(0, 0, 0.0, matched=0.0, annotated_only=6.0),
        _cell(1, 1, 0.4, matched=40.0, annotated_only=60.0),
        _cell(2, 2, 0.7, matched=70.0, annotated_only=20.0, predicted_only=10.0)))
    assert "Most trail in dispute first" in text
    assert text.index("r1c1") < text.index("r2c2") < text.index("r0c0")


def test_both_directions_count_towards_what_is_in_dispute():
    text = describe_weak_cells(_tile(
        _cell(0, 0, 0.5, matched=50.0, annotated_only=50.0),
        _cell(2, 2, 0.4, matched=40.0, annotated_only=30.0, predicted_only=30.0)))
    assert text.index("r2c2") < text.index("r0c0")


def test_equal_lengths_fall_back_to_the_lower_score():
    text = describe_weak_cells(_tile(
        _cell(0, 0, 0.5, matched=30.0, annotated_only=30.0),
        _cell(2, 2, 0.0, matched=0.0, annotated_only=30.0)))
    assert text.index("r2c2") < text.index("r0c0")
