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


def how_they_disagree(cell) -> str:
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


def main_driver(cell):
    """
    The attribute SHAP holds most responsible for this cell's error, or None when no
    attribute pushed the error up.
    """
    shap = {column[len("shap_"):]: cell[column] for column in cell.index
            if column.startswith("shap_") and pd.notna(cell[column])}
    if not shap:
        return None
    feature = max(shap, key=shap.get)
    return feature if shap[feature] > 0 else None


def in_dispute(cells: pd.DataFrame) -> pd.Series:
    """Metres of trail in each cell that the prediction and the annotation disagree on."""
    return cells["annotated_only"] + cells["predicted_only"]


def weak_cells(cells: pd.DataFrame) -> pd.DataFrame:
    """
    The cells of one tile that scored below the pass threshold, with a 'where' column
    saying which part of the tile each is in.

    They are ordered by how much trail is in dispute, most first, not by score. A cell
    holding 6 m of trail, none of it matched, scores 0; a cell with 60 m unmatched out of
    100 scores 0.4. The second is ten times the work to check and ten times the trail
    that is wrong somewhere, so it comes first.
    """
    weak = cells[cells["iou"].notna() & (cells["iou"] < FAIL_IOU_THRESHOLD)]
    weak = weak.assign(in_dispute=in_dispute(weak)).sort_values(
        ["in_dispute", "iou", "cell_row", "cell_col"], ascending=[False, True, True, True])
    extent = (cells["minx"].min(), cells["miny"].min(), cells["maxx"].max(), cells["maxy"].max())
    return weak.assign(where=[compass(cell, extent) for _, cell in weak.iterrows()])


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

    weak = weak_cells(cells)
    if weak.empty:
        return (f"Weak areas: none; all {len(scored)} grid cells with trail in them "
                "scored above the threshold.")

    regions = weak["where"].value_counts()
    spread = ", ".join(f"{count} in the {region}" for region, count in regions.items())

    def driver(cell):
        feature = main_driver(cell)
        return f", main driver {feature} ({cell[feature]:.2f})" if feature else ""

    listed = weak.head(MAX_CELLS_LISTED)
    details = "; ".join(
        f"cell {cell_name(cell['cell_row'], cell['cell_col'])} ({cell['where']}): "
        f"{how_they_disagree(cell)}{driver(cell)}"
        for _, cell in listed.iterrows()
    )
    more = f"; and {len(weak) - len(listed)} more" if len(weak) > len(listed) else ""

    return (f"Weak areas: {len(weak)} of {len(scored)} grid cells with trail in them scored "
            f"below the threshold ({spread}). Most trail in dispute first: {details}{more}.")
