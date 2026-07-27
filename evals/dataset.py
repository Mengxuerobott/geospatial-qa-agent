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

# The tile this dataset is written against. Kept as a constant so that swapping in a
# different tile is a one-line change.
TILE = "E-16-70-26-W5M"
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
    ]


def sync() -> None:
    client = Client()
    examples = build_examples()

    if client.has_dataset(dataset_name=DATASET_NAME):
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
        existing = list(client.list_examples(dataset_id=dataset.id))
        if existing:
            client.delete_examples(example_ids=[e.id for e in existing])
        print(f"Reusing dataset {DATASET_NAME} (cleared {len(existing)} old examples)")
    else:
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description="Routing and grounding checks for the geospatial QA supervisor.",
        )
        print(f"Created dataset {DATASET_NAME}")

    client.create_examples(
        dataset_id=dataset.id,
        inputs=[{"question": e["question"]} for e in examples],
        outputs=[{k: v for k, v in e.items() if k != "question"} for e in examples],
    )
    print(f"Wrote {len(examples)} examples.")
    print(f"https://smith.langchain.com/datasets/{dataset.id}")


if __name__ == "__main__":
    sync()
