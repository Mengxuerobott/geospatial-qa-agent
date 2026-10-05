"""
Bounding how much conversation history is sent to the agent's LLM.

Kept free of agent imports (OpenAI, DuckDB, rasterio) so it can be unit-tested without an
API key.
"""

from langchain_core.messages import HumanMessage

# A turn is one user message plus everything the agent did in response to it.
MAX_HISTORY_TURNS = 6


def recent_turns(messages: list, max_turns: int = MAX_HISTORY_TURNS) -> list:
    """
    Return the messages belonging to the last `max_turns` user turns.

    The cut always lands on a HumanMessage, so a tool result is never separated from the
    AI message that requested it -- OpenAI rejects a history that starts with an orphaned
    tool message.
    """
    human_positions = [i for i, msg in enumerate(messages) if isinstance(msg, HumanMessage)]
    if len(human_positions) <= max_turns:
        return messages
    return messages[human_positions[-max_turns]:]
