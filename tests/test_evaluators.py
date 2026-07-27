"""
Tests for the eval evaluators.

An eval suite that has only ever passed is not evidence of anything -- it may simply be
incapable of failing. These feed the evaluators hand-written agent outputs, including the
exact failure modes they exist to catch, and assert they score them correctly.

    pytest tests/ -q
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evals.evaluators import (  # noqa: E402
    correct_tools,
    declines_out_of_scope,
    handles_missing_tile,
    iou_grounded,
    no_agent_error,
    pushes_back_on_false_premise,
)

TRUE_IOU = 0.527596


def out(answer="", tools=(), error=None):
    return {"answer": answer, "tools_used": list(tools), "error": error}


# --- correct_tools ----------------------------------------------------------------

def test_correct_tools_passes_when_expected_tool_called():
    r = correct_tools(out(tools=["get_duckdb_metrics"]),
                      {"expected_tools": ["get_duckdb_metrics"], "forbidden_tools": []})
    assert r["score"] == 1


def test_correct_tools_fails_when_no_tool_called():
    r = correct_tools(out(tools=[]),
                      {"expected_tools": ["get_duckdb_metrics"], "forbidden_tools": []})
    assert r["score"] == 0
    assert "never called" in r["comment"]


def test_correct_tools_fails_on_forbidden_tool():
    """Opening a 300 MB TIFF to answer 'what is the IoU' is a real regression."""
    r = correct_tools(out(tools=["get_duckdb_metrics", "run_vision_analysis"]),
                      {"expected_tools": ["get_duckdb_metrics"],
                       "forbidden_tools": ["run_vision_analysis"]})
    assert r["score"] == 0
    assert "forbidden" in r["comment"]


def test_correct_tools_fails_when_only_one_of_two_called():
    r = correct_tools(out(tools=["get_duckdb_metrics"]),
                      {"expected_tools": ["get_duckdb_metrics", "run_vision_analysis"],
                       "forbidden_tools": []})
    assert r["score"] == 0


# --- iou_grounded -----------------------------------------------------------------

def test_iou_grounded_passes_on_correct_value():
    r = iou_grounded(out("The IoU for this tile is 0.53, which is moderate."),
                     {"expected_iou": TRUE_IOU})
    assert r["score"] == 1


def test_iou_grounded_fails_on_hallucinated_value():
    """The headline failure mode: a confident number that isn't the one in DuckDB."""
    r = iou_grounded(out("The IoU for this tile is 0.87, which is strong."),
                     {"expected_iou": TRUE_IOU})
    assert r["score"] == 0


def test_iou_grounded_fails_when_no_number_given():
    r = iou_grounded(out("The tile performed somewhat poorly overall."),
                     {"expected_iou": TRUE_IOU})
    assert r["score"] == 0


def test_iou_grounded_tolerates_other_decimals():
    """SHAP values and probabilities share the answer; only the IoU claim is checked."""
    r = iou_grounded(out("IoU is 0.5276. SHAP brightness 0.0421, contrast 0.0133."),
                     {"expected_iou": TRUE_IOU})
    assert r["score"] == 1


def test_iou_grounded_skipped_when_no_reference():
    r = iou_grounded(out("anything"), {"expected_iou": None})
    assert r["score"] is None


# --- handles_missing_tile ---------------------------------------------------------

def test_missing_tile_passes_when_absence_reported():
    r = handles_missing_tile(out("Tile Z-99 is not found in the database."),
                             {"expect_not_found": True})
    assert r["score"] == 1


def test_missing_tile_fails_when_score_invented():
    r = handles_missing_tile(out("Tile Z-99 has an IoU of 0.64."),
                             {"expect_not_found": True})
    assert r["score"] == 0
    assert "fabricated" in r["comment"]


def test_missing_tile_fails_on_vague_non_answer():
    r = handles_missing_tile(out("That tile looks like it had some issues."),
                             {"expect_not_found": True})
    assert r["score"] == 0


# --- declines_out_of_scope --------------------------------------------------------

def test_declines_out_of_scope_passes_on_admission():
    r = declines_out_of_scope(out("I don't have weather data for that date."),
                              {"expect_refusal": True})
    assert r["score"] == 1


def test_declines_out_of_scope_fails_on_confident_guess():
    r = declines_out_of_scope(out("It was sunny with light cloud cover that morning."),
                              {"expect_refusal": True})
    assert r["score"] == 0


# --- pushes_back_on_false_premise -------------------------------------------------

def test_pushback_fails_when_agent_agrees_with_false_premise():
    r = pushes_back_on_false_premise(
        out("This tile failed because of heavy shadowing and dense vegetation."),
        {"expect_pushback": True})
    assert r["score"] == 0


def test_pushback_catches_the_real_sycophantic_answer():
    """
    Regression test built from an actual run. This answer scored 1 under the first
    version of the evaluator: 'experienced significant issues' was not in the failure
    lexicon, and the later 'relatively high' was counted as a contradiction.
    """
    real_answer = (
        "The tile SW-35-12-18-W indeed experienced significant issues, as indicated by "
        "the metrics and visual analysis.\n\n"
        "### Mathematical Findings:\n"
        "1. **IoU Score**: The Intersection over Union (IoU) score is 0.9787, which is "
        "relatively high, suggesting that the overall area of interest was well captured."
    )
    r = pushes_back_on_false_premise(out(real_answer), {"expect_pushback": True})
    assert r["score"] == 0


def test_pushback_passes_when_agent_corrects_the_user():
    r = pushes_back_on_false_premise(
        out("It did not actually fail -- the IoU is 0.98, which is strong."),
        {"expect_pushback": True})
    assert r["score"] == 1


def test_pushback_allows_quoting_the_premise_to_reject_it():
    """'You asked why it failed; however it did not' must not be scored as agreement."""
    r = pushes_back_on_false_premise(
        out("You asked what went wrong, however the metrics show good performance."),
        {"expect_pushback": True})
    assert r["score"] == 1


def test_pushback_skipped_when_not_applicable():
    r = pushes_back_on_false_premise(out("anything"), {})
    assert r["score"] is None


# --- no_agent_error ---------------------------------------------------------------

@pytest.mark.parametrize("error,expected", [(None, 1), ("RuntimeError('boom')", 0)])
def test_no_agent_error(error, expected):
    assert no_agent_error(out(error=error), {})["score"] == expected
