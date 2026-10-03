"""
Tests for the history window sent to the supervisor.

    pytest tests/ -q
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402

from src.agent.history import recent_turns  # noqa: E402


def _turn(n: int, with_tool: bool = True) -> list:
    """One user turn: question, optional tool round-trip, final answer."""
    messages = [HumanMessage(content=f"question {n}")]
    if with_tool:
        call = {"name": "get_duckdb_metrics", "args": {"tile_id": f"T{n}"}, "id": f"call_{n}"}
        messages.append(AIMessage(content="", tool_calls=[call]))
        messages.append(ToolMessage(content=f"metrics {n}", tool_call_id=f"call_{n}"))
    messages.append(AIMessage(content=f"answer {n}"))
    return messages


def _conversation(turns: int) -> list:
    return [msg for n in range(turns) for msg in _turn(n)]


def test_short_history_is_returned_unchanged():
    messages = _conversation(3)
    assert recent_turns(messages, max_turns=6) == messages


def test_history_at_the_limit_is_returned_unchanged():
    messages = _conversation(6)
    assert recent_turns(messages, max_turns=6) == messages


def test_long_history_keeps_only_the_last_turns():
    kept = recent_turns(_conversation(10), max_turns=6)
    questions = [msg.content for msg in kept if isinstance(msg, HumanMessage)]
    assert questions == [f"question {n}" for n in range(4, 10)]


def test_window_starts_on_a_human_message():
    kept = recent_turns(_conversation(10), max_turns=6)
    assert isinstance(kept[0], HumanMessage)


def test_tool_results_are_never_orphaned():
    kept = recent_turns(_conversation(10), max_turns=2)
    requested = {call["id"] for msg in kept for call in getattr(msg, "tool_calls", None) or []}
    answered = {msg.tool_call_id for msg in kept if isinstance(msg, ToolMessage)}
    assert answered == requested


def test_in_progress_turn_is_kept_whole():
    # Mid-turn: the latest question has a tool result but no final answer yet.
    messages = _conversation(8) + _turn(8)[:-1]
    kept = recent_turns(messages, max_turns=1)
    assert kept == _turn(8)[:-1]


def test_empty_history():
    assert recent_turns([], max_turns=6) == []
