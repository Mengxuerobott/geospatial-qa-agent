"""
What a tile ID may look like.

Tile IDs arrive from the LLM and from API clients, and end up in a file path and in the
prompt. Kept free of agent imports so it can be unit-tested without an API key.
"""

import re

# Letters, digits, hyphens and underscores, e.g. "SE-31-18-03-W". No dots, slashes or
# spaces, so an ID can neither climb out of the TIFF directory nor carry a sentence.
TILE_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,64}")


def is_valid_tile_id(tile_id) -> bool:
    return isinstance(tile_id, str) and TILE_ID_PATTERN.fullmatch(tile_id) is not None
