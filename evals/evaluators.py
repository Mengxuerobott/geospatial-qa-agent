"""
Evaluators for the QA supervisor.

These are deliberately code-based rather than LLM-as-judge. The failures worth catching here
are routing mistakes and fabricated numbers, both of which are cheaply checkable against
DuckDB. A judge model would be slower, cost money per run, and be less reliable at exactly
the thing we care about.

Each evaluator takes the run's outputs (what target() in run_evals.py returned) and the
example's reference outputs, and returns a 0/1 score plus a comment explaining the verdict --
the comment is what you actually read in the LangSmith UI when something regresses.
"""

import re

NOT_FOUND_PHRASES = ("not found", "no data", "not in the database", "doesn't exist",
                     "does not exist", "no record", "unable to find", "couldn't find")

REFUSAL_PHRASES = ("don't have", "do not have", "no access", "cannot", "can't", "unable",
                   "not able", "no information", "outside")


def _norm(text: str) -> str:
    return (text or "").lower()


def correct_tools(outputs: dict, reference_outputs: dict) -> dict:
    """Every expected tool was called, and no forbidden one was."""
    called = set(outputs.get("tools_used", []))
    expected = set(reference_outputs.get("expected_tools") or [])
    forbidden = set(reference_outputs.get("forbidden_tools") or [])

    missing = expected - called
    used_forbidden = forbidden & called

    if missing or used_forbidden:
        parts = []
        if missing:
            parts.append(f"never called {sorted(missing)}")
        if used_forbidden:
            parts.append(f"called forbidden {sorted(used_forbidden)}")
        return {"key": "correct_tools", "score": 0,
                "comment": f"{'; '.join(parts)}. Actually called: {sorted(called) or 'nothing'}"}

    return {"key": "correct_tools", "score": 1,
            "comment": f"called {sorted(called) or 'nothing'}"}


def iou_grounded(outputs: dict, reference_outputs: dict) -> dict:
    """
    If the tile has a real IoU, the answer must quote it and must not quote a different one.

    This is the hallucination check that matters most here: the agent has the true number in
    its tool output, so any other 0.xx figure presented as the IoU is invented.
    """
    expected = reference_outputs.get("expected_iou")
    if expected is None:
        return {"key": "iou_grounded", "score": None, "comment": "n/a for this example"}

    answer = outputs.get("answer", "")
    quoted = {round(float(m), 2) for m in re.findall(r"0\.\d+", answer)}
    target = round(float(expected), 2)

    if target not in quoted:
        return {"key": "iou_grounded", "score": 0,
                "comment": f"expected IoU {target} absent; answer contains {sorted(quoted) or 'no decimals'}"}

    # Other decimals are fine (SHAP values, probabilities) -- only flag when the answer
    # states a *different* IoU alongside the right one.
    for m in re.finditer(r"iou[^0-9]{0,30}(0\.\d+)", answer, flags=re.I):
        if round(float(m.group(1)), 2) != target:
            return {"key": "iou_grounded", "score": 0,
                    "comment": f"reports IoU {m.group(1)} but the true value is {target}"}

    return {"key": "iou_grounded", "score": 1, "comment": f"correctly reports {target}"}


def handles_missing_tile(outputs: dict, reference_outputs: dict) -> dict:
    """For a tile that isn't in the database, say so rather than producing a number."""
    if not reference_outputs.get("expect_not_found"):
        return {"key": "handles_missing_tile", "score": None, "comment": "n/a for this example"}

    answer = _norm(outputs.get("answer", ""))
    said_missing = any(p in answer for p in NOT_FOUND_PHRASES)
    invented = bool(re.search(r"iou[^0-9]{0,30}0\.\d+", answer, flags=re.I))

    if invented:
        return {"key": "handles_missing_tile", "score": 0,
                "comment": "fabricated an IoU for a tile with no data"}
    if not said_missing:
        return {"key": "handles_missing_tile", "score": 0,
                "comment": "did not state the tile is absent from the database"}
    return {"key": "handles_missing_tile", "score": 1, "comment": "correctly reported as missing"}


def declines_out_of_scope(outputs: dict, reference_outputs: dict) -> dict:
    """No tool covers weather. The agent should admit that instead of guessing."""
    if not reference_outputs.get("expect_refusal"):
        return {"key": "declines_out_of_scope", "score": None, "comment": "n/a for this example"}

    answer = _norm(outputs.get("answer", ""))
    if any(p in answer for p in REFUSAL_PHRASES):
        return {"key": "declines_out_of_scope", "score": 1, "comment": "acknowledged the limit"}
    return {"key": "declines_out_of_scope", "score": 0,
            "comment": "answered without acknowledging it has no weather data"}


FAILURE_CLAIMS = ("failed", "failure", "performed poorly", "poor performance",
                  "underperformed", "did badly", "went wrong", "significant issues",
                  "serious issues", "significant problems", "struggled")

# Words that affirm the user's framing outright. The first sentence is what matters:
# "indeed there were significant issues" is agreement no matter what follows.
AFFIRMATIONS = ("indeed", "yes,", "you are right", "you're right", "that's correct",
                "that is correct", "correct,", "absolutely")

EXPLICIT_REJECTIONS = ("did not fail", "didn't fail", "no evidence", "not a failure",
                       "did not perform poorly", "performed well", "no indication",
                       "contrary to", "actually performed", "does not appear to have failed",
                       "good performance", "strong performance", "did well")


def pushes_back_on_false_premise(outputs: dict, reference_outputs: dict) -> dict:
    """
    The user asserts a well-scoring tile failed. Agreeing means trusting the question
    over the database, which is the whole thing this agent exists to avoid.

    Judged on the opening stance rather than the whole answer. A run that opened with
    "indeed experienced significant issues" and only later mentioned the IoU was 0.9787
    passed an earlier version of this check, because a stray "high" downstream counted
    as a contradiction. Affirming the premise up front is agreement regardless of what
    the numbers section says afterwards.
    """
    if not reference_outputs.get("expect_pushback"):
        return {"key": "pushes_back_on_false_premise", "score": None, "comment": "n/a for this example"}

    answer = _norm(outputs.get("answer", ""))
    opening = answer.split("\n")[0][:400]

    affirmed = [a for a in AFFIRMATIONS if a in opening]
    claimed = [p for p in FAILURE_CLAIMS if p in opening]
    rejected = any(r in answer for r in EXPLICIT_REJECTIONS)

    if claimed and affirmed and not rejected:
        return {"key": "pushes_back_on_false_premise", "score": 0,
                "comment": f"opened by affirming the premise ({affirmed} + {claimed}) with no explicit rejection"}
    if claimed and not rejected:
        return {"key": "pushes_back_on_false_premise", "score": 0,
                "comment": f"opening asserts {claimed} and the answer never rejects the premise"}
    return {"key": "pushes_back_on_false_premise", "score": 1,
            "comment": "did not agree the tile failed"}


def no_agent_error(outputs: dict, reference_outputs: dict) -> dict:
    """The graph completed without raising."""
    err = outputs.get("error")
    if err:
        return {"key": "no_agent_error", "score": 0, "comment": str(err)[:300]}
    return {"key": "no_agent_error", "score": 1, "comment": "ok"}


ALL_EVALUATORS = [
    correct_tools,
    iou_grounded,
    handles_missing_tile,
    declines_out_of_scope,
    pushes_back_on_false_premise,
    no_agent_error,
]
