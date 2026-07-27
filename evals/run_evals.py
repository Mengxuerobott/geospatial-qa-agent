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

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langsmith import Client
from langsmith.evaluation import evaluate

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)
load_dotenv(dotenv_path=os.path.join(ROOT_DIR, ".env"))

from evals.dataset import DATASET_NAME  # noqa: E402
from evals.evaluators import ALL_EVALUATORS  # noqa: E402
from src.agent.graph_agent import create_graph_agent  # noqa: E402


def _tools_called(messages) -> list[str]:
    """Pull the tool names the supervisor actually invoked out of the final graph state."""
    names = []
    for msg in messages:
        for call in getattr(msg, "tool_calls", None) or []:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
            if name:
                names.append(name)
    return names


def make_target(agent):
    def target(inputs: dict) -> dict:
        try:
            state = agent.invoke({"messages": [HumanMessage(content=inputs["question"])]})
        except Exception as exc:  # surfaced by the no_agent_error evaluator
            return {"answer": "", "tools_used": [], "error": repr(exc)}

        messages = state["messages"]
        return {
            "answer": messages[-1].content,
            "tools_used": _tools_called(messages),
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

    agent = create_graph_agent()

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
