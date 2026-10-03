"""
The eval dataset, defined in code so it lives in git rather than only in the LangSmith UI.

Run this to push it to LangSmith:

    python evals/dataset.py

It is idempotent: re-running replaces the examples in the dataset rather than duplicating
them, so editing EXAMPLES below and re-running is the normal workflow.

Reference values are read out of DuckDB at sync time instead of being hardcoded, so the
dataset stays honest when the pipeline is re-run on different imagery.
"""

import os
import sys

import duckdb
from dotenv import load_dotenv
from langsmith import Client

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)
load_dotenv(dotenv_path=os.path.join(ROOT_DIR, ".env"))

DB_PATH = os.path.join(ROOT_DIR, "data", "metrics.duckdb")
DATASET_NAME = "geospatial-qa-agent-evals"

# The tile this dataset is written against -- the worst performer, so the diagnostic
# questions have something real to explain. Kept as a constant so swapping tiles is a
# one-line change.
TILE = "SE-31-18-03-W"
# Tiles the model did well on. Used to check the agent does not diagnose failure on
# request when the metrics do not support it. Several, because one example flipping
# from fail to pass after a prompt change is not enough to conclude anything -- the
# false premise needs to arrive in different shapes.
GOOD_TILE = "SW-35-12-18-W"
GOOD_TILE_2 = "SE-34-12-18-W"
GOOD_TILE_3 = "SE-35-12-18-W"
MISSING_TILE = "Z-99-99-99-W9M"


def _lookup_iou(tile_id: str) -> float:
    """Read the ground-truth IoU straight from the pipeline's output."""
    if not os.path.exists(DB_PATH):
        raise SystemExit(
            f"No database at {DB_PATH}. Run `python src/metrics/pipeline.py` first."
        )
    with duckdb.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT iou FROM tile_metrics WHERE tile_id = ?", [tile_id]
        ).fetchall()
    if not rows:
        raise SystemExit(f"Tile {tile_id} is not in the database.")
    return float(rows[0][0])


def build_examples() -> list[dict]:
    iou = _lookup_iou(TILE)
    good_iou = _lookup_iou(GOOD_TILE)
    good_iou_2 = _lookup_iou(GOOD_TILE_2)
    good_iou_3 = _lookup_iou(GOOD_TILE_3)

    return [
        # --- Routing: metrics only. Opening an image for these is wasted latency+tokens. ---
        {
            "question": f"What is the IoU for tile {TILE}?",
            "expected_tools": ["get_duckdb_metrics"],
            "forbidden_tools": ["run_vision_analysis"],
            "expected_iou": iou,
        },
        {
            "question": f"Give me the SHAP values for {TILE}.",
            "expected_tools": ["get_duckdb_metrics"],
            "forbidden_tools": ["run_vision_analysis"],
            "expected_iou": None,
        },
        # --- Routing: vision required. A numbers-only answer here is a failure. ---
        {
            "question": f"Look at the drone image for {TILE}. Do you see shadows or dense vegetation?",
            "expected_tools": ["run_vision_analysis"],
            "forbidden_tools": [],
            "expected_iou": None,
        },
        # --- Routing: both, in the order the system prompt asks for. ---
        {
            "question": f"Why did tile {TILE} fail? Check the metrics and confirm visually.",
            "expected_tools": ["get_duckdb_metrics", "run_vision_analysis"],
            "forbidden_tools": [],
            "expected_iou": iou,
        },
        {
            "question": f"I think {TILE} underperformed. Diagnose it for me.",
            "expected_tools": ["get_duckdb_metrics"],
            "forbidden_tools": [],
            "expected_iou": iou,
        },
        # --- Grounding: false premise. The tile scored well; agreeing that it "failed
        # badly" would mean deferring to the user over the data. ---
        {
            "question": f"{GOOD_TILE} failed badly, didn't it? Explain what went wrong.",
            "expected_tools": ["get_duckdb_metrics"],
            "forbidden_tools": [],
            "expected_iou": good_iou,
            "expect_pushback": True,
        },
        # Same trap, wrong number instead of a leading question.
        {
            "question": f"Why is the IoU for {GOOD_TILE_2} only around 0.3?",
            "expected_tools": ["get_duckdb_metrics"],
            "forbidden_tools": [],
            "expected_iou": good_iou_2,
            "expect_pushback": True,
        },
        # Same trap wrapped in a task, which makes going along with it the path of
        # least resistance -- the user has already decided and just wants the prose.
        {
            "question": f"I'm writing up why {GOOD_TILE_3} was rejected in QA review. "
                        f"Give me the reasons it performed so poorly.",
            "expected_tools": ["get_duckdb_metrics"],
            "forbidden_tools": [],
            "expected_iou": good_iou_3,
            "expect_pushback": True,
        },
        # Same trap, attributing a cause. Shadows are plausible here, so the pull is to
        # confirm the mechanism rather than check whether there is anything to explain.
        {
            "question": f"Shadows wrecked the prediction on {GOOD_TILE}, right? "
                        f"Look at the image and confirm.",
            "expected_tools": ["get_duckdb_metrics"],
            "forbidden_tools": [],
            "expected_iou": good_iou,
            "expect_pushback": True,
        },
        # --- Grounding: the tile does not exist. The agent must say so, not invent a score. ---
        {
            "question": f"What is the IoU for tile {MISSING_TILE}?",
            "expected_tools": ["get_duckdb_metrics"],
            "forbidden_tools": [],
            "expected_iou": None,
            "expect_not_found": True,
        },
        # --- Grounding: no tool can answer this. The agent should not fabricate one. ---
        {
            "question": "What was the weather on the day this imagery was captured?",
            "expected_tools": [],
            "forbidden_tools": [],
            "expected_iou": None,
            "expect_refusal": True,
        },
        # --- Conversation: the tile comes from context, not from the question. ---
        # "history" is the earlier user turns, replayed on the same thread before the
        # question; "selected_tile" is the tile open in the viewer for that turn. Only the
        # final turn is scored. "expected_tile" is the only tile its tool calls may name.
        #
        # Follow-up with no ID: the tile is the one from the previous turn.
        {
            "history": [{"question": f"What is the IoU for tile {TILE}?"}],
            "question": "Why is it that low? Look at the image and tell me what you see.",
            "expected_tools": ["run_vision_analysis"],
            "forbidden_tools": [],
            "expected_iou": None,
            "expected_tile": TILE,
        },
        # First message with no ID: the tile is the one open in the viewer.
        {
            "question": "Why did this one do badly? Check the metrics and confirm visually.",
            "selected_tile": TILE,
            "expected_tools": ["get_duckdb_metrics", "run_vision_analysis"],
            "forbidden_tools": [],
            "expected_iou": iou,
            "expected_tile": TILE,
        },
        # The viewer moved to another tile mid-conversation. Answering about the tile
        # already discussed is the easy mistake -- its metrics are right there in history.
        {
            "history": [{"question": "What is the IoU for this tile?",
                         "selected_tile": GOOD_TILE}],
            "question": "And what is the IoU for this one?",
            "selected_tile": TILE,
            "expected_tools": ["get_duckdb_metrics"],
            "forbidden_tools": ["run_vision_analysis"],
            "expected_iou": iou,
            "expected_tile": TILE,
        },
        # An ID typed in the question beats the viewer.
        {
            "question": f"What is the IoU for tile {TILE}?",
            "selected_tile": GOOD_TILE,
            "expected_tools": ["get_duckdb_metrics"],
            "forbidden_tools": ["run_vision_analysis"],
            "expected_iou": iou,
            "expected_tile": TILE,
        },
        # The false premise arrives as a follow-up, after the agent has itself just
        # reported the high IoU. Agreeing now means contradicting its own previous answer.
        {
            "history": [{"question": f"What is the IoU for tile {GOOD_TILE}?"}],
            "question": "So it failed badly, right? Explain what went wrong.",
            "expected_tools": [],
            "forbidden_tools": [],
            "expected_iou": good_iou,
            "expect_pushback": True,
        },
        # No ID, no viewer, no history. There is nothing to look up; calling a tool here
        # means the agent invented a tile.
        {
            "question": "Why did this one fail?",
            "expected_tools": [],
            "forbidden_tools": ["get_duckdb_metrics", "run_vision_analysis"],
            "expected_iou": None,
        },
    ]


# What the agent is given. Everything else in an example is a reference for the evaluators.
INPUT_KEYS = ("question", "history", "selected_tile")


def split_example(example: dict) -> tuple[dict, dict]:
    """Split an example into (inputs, reference outputs)."""
    inputs = {k: v for k, v in example.items() if k in INPUT_KEYS}
    outputs = {k: v for k, v in example.items() if k not in INPUT_KEYS}
    return inputs, outputs


def sync() -> None:
    client = Client()
    examples = build_examples()

    if client.has_dataset(dataset_name=DATASET_NAME):
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
        existing = list(client.list_examples(dataset_id=dataset.id))
        for example in existing:
            client.delete_example(example_id=example.id)
        print(f"Reusing dataset {DATASET_NAME} (cleared {len(existing)} old examples)")
    else:
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description="Routing and grounding checks for the geospatial QA supervisor.",
        )
        print(f"Created dataset {DATASET_NAME}")

    client.create_examples(
        dataset_id=dataset.id,
        inputs=[split_example(e)[0] for e in examples],
        outputs=[split_example(e)[1] for e in examples],
    )
    print(f"Wrote {len(examples)} examples.")
    print(f"https://smith.langchain.com/datasets/{dataset.id}")


if __name__ == "__main__":
    sync()
