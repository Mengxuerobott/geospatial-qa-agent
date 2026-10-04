"""
Waiting out OpenAI rate limits during an eval run.

The client already retries a 429 a couple of times, but within a second or two. The limit
this suite hits is tokens-per-minute -- the vision calls are token-heavy and examples run
in parallel -- and that only clears after tens of seconds.
"""

import time

# Seconds to wait before the 2nd, 3rd and 4th attempt.
BACKOFF_SECONDS = (15, 30, 60)


def is_rate_limit(exc: BaseException) -> bool:
    """True for an HTTP 429. Checked by shape so this module needs no openai import."""
    return getattr(exc, "status_code", None) == 429 or type(exc).__name__ == "RateLimitError"


def with_backoff(fn, *, delays=BACKOFF_SECONDS, sleep=time.sleep):
    """
    Call fn(), retrying after each delay while it raises a rate-limit error.

    Any other exception is raised at once: a real agent failure should not be retried into
    a pass. The last rate-limit error is raised once the delays run out.
    """
    for delay in (*delays, None):
        try:
            return fn()
        except Exception as exc:
            if delay is None or not is_rate_limit(exc):
                raise
            sleep(delay)
