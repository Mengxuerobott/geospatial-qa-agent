"""
What counts as a match, and what counts as a failed tile.

Kept free of agent imports so the evals and unit tests can use it without an API key.
"""

# An annotated trail and a predicted trail this many metres apart or less are the same trail.
# The annotation is drawn by a person and sits near the trail, not exactly on it, so a
# prediction a few metres to one side of it is not an error.
MATCH_TOLERANCE_M = 5.0

# Tiles below this IoU fail QA. The same cut-off xai_engine.py uses for "needs human review".
FAIL_IOU_THRESHOLD = 0.75


def verdict(iou: float) -> str:
    """'failed' or 'passed'. A tile exactly on the threshold passes."""
    return "failed" if iou < FAIL_IOU_THRESHOLD else "passed"
