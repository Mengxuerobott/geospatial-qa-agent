"""
Describing the weak areas of a tile from its grid cells.

A tile's IoU says whether it passed; its cells say where the model did badly, which a tile
that passed overall can still have. Kept free of agent imports so it can be unit-tested
without an API key.
"""

import re

import pandas as pd

from src.agent.verdict import FAIL_IOU_THRESHOLD

# How many weak cells are spelled out; the rest are only counted
MAX_CELLS_LISTED = 5

_ROWS = ("north", "", "south")
_COLS = ("west", "", "east")


def cell_name(cell_row: int, cell_col: int) -> str:
    """The name the tools use for a cell, e.g. 'r3c5': row 3 from the top, column 5."""
    return f"r{int(cell_row)}c{int(cell_col)}"


def parse_cell_name(name):
    """(cell_row, cell_col) for a name like 'r3c5', or None if it is not one."""
    match = re.fullmatch(r"r(\d{1,4})c(\d{1,4})", name.strip().lower()) if isinstance(name, str) else None
    return (int(match.group(1)), int(match.group(2))) if match else None


def compass(cell, extent) -> str:
    """
    Which ninth of the tile a cell's centre falls in: 'north-east', 'west', 'centre'...

    extent is (minx, miny, maxx, maxy) of the tile's cells. Assumes a north-up raster.
    """
    minx, miny, maxx, maxy = extent
    x = ((cell["minx"] + cell["maxx"]) / 2 - minx) / ((maxx - minx) or 1)
    y = (maxy - (cell["miny"] + cell["maxy"]) / 2) / ((maxy - miny) or 1)
    third = lambda share: min(2, max(0, int(share * 3)))
    name = "-".join(part for part in (_ROWS[third(y)], _COLS[third(x)]) if part)
    return name or "centre"


def _what_went_wrong(cell) -> str:
    """
    How the prediction and the annotation disagree in a cell, without saying which is
    right: the annotation is a person's work and can be the one that is wrong.
    """
    annotated, predicted = cell["annotated_only"], cell["predicted_only"]
    if cell["matched"] == 0 and predicted == 0:
        return f"{annotated:.0f} m of annotated trail with no prediction near it"
    if cell["matched"] == 0 and annotated == 0:
        return f"{predicted:.0f} m of predicted trail with no annotation near it"
    disagreements = " and ".join(
        f"{length:.0f} m {kind}" for length, kind in
        ((annotated, "annotated but not predicted"), (predicted, "predicted but not annotated"))
        if length >= 0.5)
    return f"{cell['iou']:.0%} of the trail matched, {disagreements}"


def _main_driver(cell) -> str:
    """The attribute SHAP holds most responsible for this cell's error, with its value."""
    shap = {column[len("shap_"):]: cell[column] for column in cell.index
            if column.startswith("shap_") and pd.notna(cell[column])}
    if not shap:
        return ""
    feature = max(shap, key=shap.get)
    if shap[feature] <= 0:
        return ""
    return f", main driver {feature} ({cell[feature]:.2f})"


def describe_weak_cells(cells: pd.DataFrame) -> str:
    """
    One paragraph on the cells of a tile that scored below the pass threshold: how many,
    where they cluster, and the worst few by name.

    cells is every cell_metrics row of one tile. Cell scores are given as percentages and
    never called an IoU, so they cannot be mistaken for the tile's own IoU.
    """
    scored = cells[cells["iou"].notna()]
    if scored.empty:
        return "Weak areas: unknown; no grid cell of this tile holds enough trail to score."

    weak = scored[scored["iou"] < FAIL_IOU_THRESHOLD].sort_values(
        ["iou", "cell_row", "cell_col"])
    if weak.empty:
        return (f"Weak areas: none; all {len(scored)} grid cells with trail in them "
                "scored above the threshold.")

    extent = (cells["minx"].min(), cells["miny"].min(), cells["maxx"].max(), cells["maxy"].max())
    weak = weak.assign(where=[compass(cell, extent) for _, cell in weak.iterrows()])

    regions = weak["where"].value_counts()
    spread = ", ".join(f"{count} in the {region}" for region, count in regions.items())

    listed = weak.head(MAX_CELLS_LISTED)
    details = "; ".join(
        f"cell {cell_name(cell['cell_row'], cell['cell_col'])} ({cell['where']}): "
        f"{_what_went_wrong(cell)}{_main_driver(cell)}"
        for _, cell in listed.iterrows()
    )
    more = f"; and {len(weak) - len(listed)} more" if len(weak) > len(listed) else ""

    return (f"Weak areas: {len(weak)} of {len(scored)} grid cells with trail in them scored "
            f"below the threshold ({spread}). Worst first: {details}{more}.")
