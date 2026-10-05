"""
Which OpenAI model does which job.

Set in the environment (or .env), so a stronger model can be tried without a code change:

    AGENT_MODEL   the ReAct agent: reads the question, calls the tools, writes the answer
    VISION_MODEL  the vision tool: looks at the tile or the cell

They are separate because the jobs are. Judging whether a thin trail is visible in a crop
is the hardest thing asked of any model here, and may be worth a stronger one than the
agent needs. Kept free of agent imports so it can be unit-tested without an API key.
"""

import os

DEFAULT_MODEL = "gpt-4o-mini"


def agent_model() -> str:
    return os.getenv("AGENT_MODEL") or DEFAULT_MODEL


def vision_model() -> str:
    return os.getenv("VISION_MODEL") or DEFAULT_MODEL
