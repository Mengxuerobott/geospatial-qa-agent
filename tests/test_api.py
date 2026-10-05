"""
Tests for what the chat endpoint tells the person asking when something goes wrong.

    pytest tests/ -q
"""

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

import src.api.server as server


class _Agent:
    """Stands in for the LangGraph agent: answers, or raises what it was given."""

    def __init__(self, error=None):
        self.error = error
        self.calls = []

    def invoke(self, state, config):
        self.calls.append((state, config))
        if self.error:
            raise self.error
        return {"messages": [AIMessage(content="the answer")]}


class RateLimitError(Exception):
    status_code = 429


@pytest.fixture
def client():
    return TestClient(server.app)


def _ask(client, monkeypatch, agent, **body):
    monkeypatch.setattr(server, "graph_agent", agent)
    return client.post("/chat", json={"message": "why did it fail?", **body})


def test_an_answer_comes_back_with_its_thread(client, monkeypatch):
    agent = _Agent()
    reply = _ask(client, monkeypatch, agent, thread_id="abc", selected_tile="SE-31-18-03-W")
    assert reply.status_code == 200
    assert reply.json() == {"reply": "the answer", "thread_id": "abc"}
    state, config = agent.calls[0]
    assert config == {"configurable": {"thread_id": "abc"}}
    assert state["messages"][0].content.startswith("[Viewer: tile SE-31-18-03-W is open]")


def test_a_failure_does_not_show_the_person_its_details(client, monkeypatch):
    """The error text can hold file paths and upstream responses; it belongs in the log."""
    secret = "could not open /srv/data/tiffs/x.tif with key sk-abc"
    reply = _ask(client, monkeypatch, _Agent(RuntimeError(secret)))
    assert reply.status_code == 500
    assert "sk-abc" not in reply.text and "/srv/data" not in reply.text
    assert "API log" in reply.json()["detail"]


def test_a_rate_limit_is_reported_as_one_and_says_to_wait(client, monkeypatch):
    reply = _ask(client, monkeypatch, _Agent(RateLimitError("429 from upstream")))
    assert reply.status_code == 429
    assert "Wait a minute" in reply.json()["detail"]


def test_a_malformed_tile_is_refused_before_the_agent_is_asked(client, monkeypatch):
    agent = _Agent()
    reply = _ask(client, monkeypatch, agent, selected_tile="../../etc/passwd")
    assert reply.status_code == 422
    assert agent.calls == []
