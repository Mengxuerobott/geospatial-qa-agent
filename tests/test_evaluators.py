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
    correct_tile,
    correct_tools,
    declines_out_of_scope,
    handles_missing_tile,
    iou_grounded,
    judge_pushback,
    no_agent_error,
    parse_judge_response,
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


# --- correct_tile -----------------------------------------------------------------

def tiles(*queried):
    return {**out(), "tool_tiles": list(queried)}


def test_correct_tile_passes_when_all_calls_name_the_expected_tile():
    r = correct_tile(tiles("A-1", "A-1"), {"expected_tile": "A-1"})
    assert r["score"] == 1


def test_correct_tile_fails_on_the_previously_discussed_tile():
    """The failure it exists for: the viewer moved to A-1, the agent answered about B-2."""
    r = correct_tile(tiles("B-2"), {"expected_tile": "A-1"})
    assert r["score"] == 0
    assert "B-2" in r["comment"]


def test_correct_tile_fails_when_one_of_several_calls_is_wrong():
    r = correct_tile(tiles("A-1", "B-2"), {"expected_tile": "A-1"})
    assert r["score"] == 0


def test_correct_tile_fails_when_no_tile_was_queried():
    r = correct_tile(tiles(), {"expected_tile": "A-1"})
    assert r["score"] == 0


def test_correct_tile_not_applicable_without_expected_tile():
    r = correct_tile(tiles("B-2"), {})
    assert r["score"] is None


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


def test_iou_grounded_allows_quoting_the_pass_threshold():
    """'IoU below 0.75 fails' states the cut-off; it is not a second, wrong IoU."""
    r = iou_grounded(out("The tile failed with an IoU of 0.53. Any tile with an IoU below "
                         "0.75 is considered a failure."),
                     {"expected_iou": TRUE_IOU})
    assert r["score"] == 1


def test_iou_grounded_still_fails_when_only_the_threshold_is_quoted():
    r = iou_grounded(out("The tile failed: its IoU is below 0.75."),
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

def test_judge_verdict_correct_scores_one():
    assert parse_judge_response('{"verdict": "correct", "reason": "said it performed well"}') == (
        1, "said it performed well")


def test_judge_verdict_incorrect_scores_zero():
    score, reason = parse_judge_response('{"verdict": "incorrect", "reason": "agreed it failed"}')
    assert score == 0
    assert reason == "agreed it failed"


def test_judge_tolerates_a_code_fence():
    """Models wrap JSON in ```json despite being told not to."""
    score, _ = parse_judge_response('```json\n{"verdict": "incorrect", "reason": "x"}\n```')
    assert score == 0


def test_judge_abstains_on_unparseable_reply():
    """An unreadable judge must not be counted as a pass."""
    score, reason = parse_judge_response("I think the answer was pretty good overall.")
    assert score is None
    assert "could not parse" in reason


def test_judge_abstains_on_unknown_verdict():
    score, _ = parse_judge_response('{"verdict": "maybe", "reason": "unsure"}')
    assert score is None


def test_judge_skipped_when_not_applicable():
    """Must not call the model for examples that are not false-premise cases."""
    r = judge_pushback({"question": "q"}, out("anything"), {})
    assert r["score"] is None
    assert r["comment"] == "n/a for this example"


# --- no_agent_error ---------------------------------------------------------------

@pytest.mark.parametrize("error,expected", [(None, 1), ("RuntimeError('boom')", 0)])
def test_no_agent_error(error, expected):
    assert no_agent_error(out(error=error), {})["score"] == expected
