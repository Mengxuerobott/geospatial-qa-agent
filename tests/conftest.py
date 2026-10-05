"""
Settings every test needs before the code under test is imported.

The unit tests call no API and need no key, but the vision tool refuses to import without
one, and tracing would otherwise try to reach LangSmith if the developer's .env turns it on.
"""

import os

os.environ.setdefault("OPENAI_API_KEY", "not-a-real-key")
os.environ["LANGCHAIN_TRACING_V2"] = "false"
