"""
Tests for waiting out rate limits in the eval run.

    pytest tests/ -q
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evals.evaluators import no_agent_error  # noqa: E402
from evals.retry import is_rate_limit, with_backoff  # noqa: E402


class RateLimitError(Exception):
    """Same name and status_code as openai.RateLimitError, without importing openai."""
    status_code = 429


class Flaky:
    """Raises `error` the first `failures` times it is called, then returns 'ok'."""

    def __init__(self, failures: int, error: Exception):
        self.failures, self.error, self.calls = failures, error, 0

    def __call__(self):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error
        return "ok"


def test_success_first_time_does_not_sleep():
    slept = []
    assert with_backoff(Flaky(0, RateLimitError()), sleep=slept.append) == "ok"
    assert slept == []


def test_rate_limit_is_retried_after_waiting():
    slept = []
    fn = Flaky(2, RateLimitError())
    assert with_backoff(fn, delays=(15, 30, 60), sleep=slept.append) == "ok"
    assert fn.calls == 3
    assert slept == [15, 30]


def test_gives_up_when_the_delays_run_out():
    slept = []
    fn = Flaky(99, RateLimitError())
    with pytest.raises(RateLimitError):
        with_backoff(fn, delays=(15, 30, 60), sleep=slept.append)
    assert fn.calls == 4
    assert slept == [15, 30, 60]


def test_other_errors_are_not_retried():
    """A real agent failure must surface, not be retried until it happens to pass."""
    slept = []
    fn = Flaky(1, ValueError("bad tool call"))
    with pytest.raises(ValueError):
        with_backoff(fn, sleep=slept.append)
    assert fn.calls == 1
    assert slept == []


def test_is_rate_limit_recognises_a_429_by_status_code():
    class APIStatusError(Exception):
        status_code = 429
    assert is_rate_limit(APIStatusError())


def test_is_rate_limit_rejects_other_errors():
    class APIStatusError(Exception):
        status_code = 500
    assert not is_rate_limit(APIStatusError())
    assert not is_rate_limit(ValueError("429"))


def test_no_agent_error_abstains_when_still_rate_limited():
    r = no_agent_error({"error": "RateLimitError('429')", "rate_limited": True}, {})
    assert r["score"] is None
    assert "rate limited" in r["comment"]


def test_no_agent_error_still_fails_on_a_real_error():
    r = no_agent_error({"error": "ValueError('boom')", "rate_limited": False}, {})
    assert r["score"] == 0
