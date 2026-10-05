"""
The review list: every grid cell where the prediction and the annotation disagree enough
to fall below the pass threshold, as map features a person can open in QGIS or ArcGIS over
the original imagery.

    python src/metrics/review.py

writes data/review/disagreements.geojson for every tile in the database. The annotation is
a person's work and can be the side that is wrong, so the list goes to annotators as well
as to whoever looks after the model.
"""

import os
import sys

import duckdb
import geopandas as gpd
import pandas as pd
from shapely.geometry import box

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# Run as a script, sys.path[0] is src/metrics, so the project root has to be added
sys.path.insert(0, ROOT_DIR)
from src.agent.cells import cell_name, how_they_disagree, main_driver, weak_cells  # noqa: E402

DB_PATH = os.path.join(ROOT_DIR, 'data', 'metrics.duckdb')
REVIEW_PATH = os.path.join(ROOT_DIR, 'data', 'review', 'disagreements.geojson')

# GeoJSON is longitude and latitude by its specification, and tiles in one batch need not
# share a projection
GEOJSON_CRS = "EPSG:4326"

COLUMNS = ["tile_id", "cell", "location", "match", "matched_m", "annotated_only_m",
           "predicted_only_m", "disagreement", "main_driver", "geometry"]


def review_list(tiles: pd.DataFrame, cells: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    One square per weak cell, most trail in dispute first within each tile, in longitude
    and latitude.

    tiles needs tile_id and crs; cells is cell_metrics rows for those tiles. A tile whose
    CRS was not recorded cannot be placed on a map and is left out.
    """
    frames = []
    for tile_id, tile_cells in cells.groupby("tile_id", sort=True):
        crs = tiles.loc[tiles["tile_id"] == tile_id, "crs"]
        if crs.empty or pd.isna(crs.iloc[0]):
            continue
        weak = weak_cells(tile_cells)
        if weak.empty:
            continue
        frames.append(gpd.GeoDataFrame({
            "tile_id": tile_id,
            "cell": [cell_name(c["cell_row"], c["cell_col"]) for _, c in weak.iterrows()],
            "location": weak["where"].to_list(),
            "match": weak["iou"].round(3).to_list(),
            "matched_m": weak["matched"].round(1).to_list(),
            "annotated_only_m": weak["annotated_only"].round(1).to_list(),
            "predicted_only_m": weak["predicted_only"].round(1).to_list(),
            "disagreement": [how_they_disagree(c) for _, c in weak.iterrows()],
            "main_driver": [main_driver(c) for _, c in weak.iterrows()],
        }, geometry=[box(c["minx"], c["miny"], c["maxx"], c["maxy"]) for _, c in weak.iterrows()],
            crs=crs.iloc[0]).to_crs(GEOJSON_CRS))

    if not frames:
        return gpd.GeoDataFrame({column: [] for column in COLUMNS[:-1]}, geometry=[],
                                crs=GEOJSON_CRS)
    return pd.concat(frames, ignore_index=True)[COLUMNS]


def load_review_list(db_path: str = DB_PATH, tile_id=None) -> gpd.GeoDataFrame:
    """
    The review list for one tile, or for every tile in the database when tile_id is None.

    Raises ValueError when the database was built before the pipeline recorded what the
    list needs.
    """
    with duckdb.connect(db_path, read_only=True) as conn:
        try:
            where, values = ("WHERE tile_id = ?", [tile_id]) if tile_id else ("", [])
            tiles = conn.execute(f"SELECT tile_id, crs FROM tile_metrics {where}", values).df()
            cells = conn.execute(f"SELECT * FROM cell_metrics {where}", values).df()
        except (duckdb.CatalogException, duckdb.BinderException) as e:
            raise ValueError("This database has no grid cells or no CRS recorded. "
                             "Run the pipeline again to rebuild it.") from e
    return review_list(tiles, cells)


if __name__ == "__main__":
    if not os.path.exists(DB_PATH):
        sys.exit("❌ Database not found. Run pipeline.py first.")
    try:
        review = load_review_list()
    except ValueError as e:
        sys.exit(f"❌ {e}")

    os.makedirs(os.path.dirname(REVIEW_PATH), exist_ok=True)
    with open(REVIEW_PATH, "w") as f:
        f.write(review.to_json())
    print(f"✅ Wrote {len(review)} cells from {review['tile_id'].nunique()} tile(s) "
          f"to {REVIEW_PATH}")
