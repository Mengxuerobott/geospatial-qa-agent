"""
What counts as a failed tile.

Kept free of agent imports so the evals and unit tests can use it without an API key.
"""

# Tiles below this IoU fail QA. The same cut-off xai_engine.py uses for "needs human review".
FAIL_IOU_THRESHOLD = 0.75


def verdict(iou: float) -> str:
    """'failed' or 'passed'. A tile exactly on the threshold passes."""
    return "failed" if iou < FAIL_IOU_THRESHOLD else "passed"
