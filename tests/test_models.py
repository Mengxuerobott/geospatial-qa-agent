"""
Tests for choosing the agent and vision models from the environment.

    pytest tests/ -q
"""

from evals.evaluators import JUDGE_MODEL
from src.agent.models import DEFAULT_MODEL, agent_model, vision_model


def test_both_default_to_the_same_model(monkeypatch):
    monkeypatch.delenv("AGENT_MODEL", raising=False)
    monkeypatch.delenv("VISION_MODEL", raising=False)
    assert agent_model() == vision_model() == DEFAULT_MODEL


def test_each_is_set_on_its_own(monkeypatch):
    monkeypatch.delenv("AGENT_MODEL", raising=False)
    monkeypatch.setenv("VISION_MODEL", "a-stronger-model")
    assert vision_model() == "a-stronger-model"
    assert agent_model() == DEFAULT_MODEL


def test_a_blank_setting_means_the_default(monkeypatch):
    """.env.example ships the lines; someone may clear a value instead of deleting the line."""
    monkeypatch.setenv("AGENT_MODEL", "")
    assert agent_model() == DEFAULT_MODEL


def test_the_judges_do_not_follow_the_model_under_test(monkeypatch):
    monkeypatch.setenv("AGENT_MODEL", "a-stronger-model")
    assert JUDGE_MODEL != agent_model()
