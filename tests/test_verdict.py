"""
Tests for the pass/fail cut-off and the scorer that checks the agent respected it.

    pytest tests/ -q
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evals.evaluators import VERDICT_JUDGE_PROMPT, judge_verdict  # noqa: E402
from src.agent.verdict import FAIL_IOU_THRESHOLD, verdict  # noqa: E402


def test_low_iou_fails():
    assert verdict(0.5147) == "failed"


def test_high_iou_passes():
    assert verdict(0.9787) == "passed"


def test_just_below_the_threshold_fails():
    assert verdict(FAIL_IOU_THRESHOLD - 0.0001) == "failed"


def test_exactly_on_the_threshold_passes():
    assert verdict(FAIL_IOU_THRESHOLD) == "passed"


def test_judge_verdict_not_applicable_without_expected_verdict():
    """Must return before importing the model client, so this needs no API key."""
    r = judge_verdict({"question": "q"}, {"answer": "a"}, {"expected_iou": 0.5})
    assert r["score"] is None
    assert "n/a" in r["comment"]


def test_judge_prompt_states_the_verdict_and_threshold():
    """The judge grades against the cut-off it is given, not its own idea of a good IoU."""
    prompt = VERDICT_JUDGE_PROMPT.format(
        iou=0.5147, threshold=FAIL_IOU_THRESHOLD, verdict="failed", verdict_upper="FAILED",
        question="q", answer="a")
    assert "0.5147" in prompt
    assert str(FAIL_IOU_THRESHOLD) in prompt
    assert "this tile FAILED" in prompt
