"""
Run the eval suite against the LangGraph supervisor.

    python evals/dataset.py      # once, and after any dataset edit
    python evals/run_evals.py

Every run creates a new experiment in LangSmith, so two runs can be compared side by side --
that is the point of the harness: change the system prompt or the model, re-run, and see
which rows moved.

This calls the real agent, so it costs tokens and hits the OpenAI API once or twice per
example.
"""

import argparse
import os
import sys
import time
import uuid

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langsmith import Client
from langsmith.evaluation import evaluate

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)
load_dotenv(dotenv_path=os.path.join(ROOT_DIR, ".env"))

from evals.dataset import DATASET_NAME  # noqa: E402
from evals.evaluators import ALL_EVALUATORS  # noqa: E402
from evals.retry import is_rate_limit, with_backoff  # noqa: E402
from src.agent.graph_agent import create_graph_agent, with_viewer_context  # noqa: E402
from src.agent.history import recent_turns  # noqa: E402


def _tools_called(messages) -> list[str]:
    """Pull the tool names the supervisor actually invoked out of the final graph state."""
    names = []
    for msg in messages:
        for call in getattr(msg, "tool_calls", None) or []:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
            if name:
                names.append(name)
    return names


def _tiles_queried(messages) -> list[str]:
    """The tile_id argument of every tool call, in order."""
    tiles = []
    for msg in messages:
        for call in getattr(msg, "tool_calls", None) or []:
            args = call.get("args") if isinstance(call, dict) else getattr(call, "args", None)
            if args and args.get("tile_id"):
                tiles.append(args["tile_id"])
    return tiles


def make_target(agent, sleep=time.sleep):
    """
    `agent` must have a checkpointer. Each example runs on its own thread: any "history"
    turns are replayed first, then the question. Only the final turn is reported, so a
    single-turn example is scored exactly as it was before conversations existed.

    An example that hits an OpenAI rate limit is retried after a wait; see evals/retry.py.
    """
    def target(inputs: dict) -> dict:
        turns = list(inputs.get("history") or []) + [inputs]

        def run_conversation():
            # A new thread on every attempt: a turn that died on a rate limit may have left
            # half of itself in the checkpoint, so a retry replays the conversation from
            # the start rather than resuming it.
            config = {"configurable": {"thread_id": str(uuid.uuid4())}}
            for turn in turns:
                content = with_viewer_context(turn["question"], turn.get("selected_tile"))
                state = agent.invoke({"messages": [HumanMessage(content=content)]}, config=config)
            return state

        try:
            state = with_backoff(run_conversation, sleep=sleep)
        except Exception as exc:  # surfaced by the no_agent_error evaluator
            return {"answer": "", "tools_used": [], "tool_tiles": [], "error": repr(exc),
                    "rate_limited": is_rate_limit(exc)}

        final_turn = recent_turns(state["messages"], max_turns=1)
        return {
            "answer": final_turn[-1].content,
            "tools_used": _tools_called(final_turn),
            "tool_tiles": _tiles_queried(final_turn),
            "error": None,
        }

    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="supervisor",
                        help="experiment name prefix shown in LangSmith")
    parser.add_argument("--concurrency", type=int, default=2,
                        help="parallel examples; keep low to avoid rate limits")
    args = parser.parse_args()

    client = Client()
    if not client.has_dataset(dataset_name=DATASET_NAME):
        raise SystemExit(f"Dataset {DATASET_NAME} not found. Run `python evals/dataset.py` first.")

    # Same configuration as the API: with memory, so multi-turn examples can be replayed.
    agent = create_graph_agent(checkpointer=MemorySaver())

    results = evaluate(
        make_target(agent),
        data=DATASET_NAME,
        evaluators=ALL_EVALUATORS,
        experiment_prefix=args.prefix,
        max_concurrency=args.concurrency,
        client=client,
        metadata={"model": "gpt-4o-mini"},
    )

    print()
    print(f"Experiment: {results.experiment_name}")


if __name__ == "__main__":
    main()
