"""
Tests for several people reading the database while one person rebuilds it.

    pytest tests/ -q
"""

import os
import subprocess
import sys
import textwrap

import duckdb
import pandas as pd
import pytest

# graph_agent imports the vision tool, which refuses to import without a key.
# Nothing here calls the API.
os.environ.setdefault("OPENAI_API_KEY", "not-a-real-key")
os.environ["LANGCHAIN_TRACING_V2"] = "false"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.agent.graph_agent as graph_agent  # noqa: E402
from src.metrics.pipeline import write_database  # noqa: E402


def _tiles(iou):
    return pd.DataFrame([{"tile_id": "t", "crs": "EPSG:32612", "brightness": 100.0,
                          "contrast": 10.0, "iou": iou, "matched": 90.0, "annotated_only": 5.0,
                          "predicted_only": 5.0, "shap_brightness": 0.01}])


def _cells():
    return pd.DataFrame([{"tile_id": "t", "cell_row": 0, "cell_col": 0, "minx": 0.0,
                          "miny": 0.0, "maxx": 50.0, "maxy": 50.0, "brightness": 100.0,
                          "matched": 90.0, "annotated_only": 5.0, "predicted_only": 5.0,
                          "iou": 0.9, "shap_brightness": 0.01}])


def _iou(path):
    with duckdb.connect(str(path), read_only=True) as conn:
        return conn.execute("SELECT iou FROM tile_metrics").fetchone()[0]


@pytest.fixture
def another_reader(tmp_path):
    """Starts a second process that opens the database read-only and holds it open."""
    processes = []

    def start(db_path):
        code = textwrap.dedent(f"""
            import duckdb, sys
            conn = duckdb.connect({str(db_path)!r}, read_only=True)
            print(conn.execute("SELECT iou FROM tile_metrics").fetchone()[0], flush=True)
            sys.stdin.read()
        """)
        process = subprocess.Popen([sys.executable, "-c", code], stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, text=True)
        processes.append(process)
        assert process.stdout.readline().strip() != ""
        return process

    yield start
    for process in processes:
        process.stdin.close()
        process.wait(timeout=10)


def test_the_database_is_written_with_both_tables(tmp_path):
    db = tmp_path / "m.duckdb"
    write_database(_tiles(0.8), _cells(), str(db))
    with duckdb.connect(str(db), read_only=True) as conn:
        assert conn.execute("SELECT COUNT(*) FROM tile_metrics").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM cell_metrics").fetchone()[0] == 1
    assert not os.path.exists(str(db) + ".new")


def test_a_rebuild_replaces_the_old_database(tmp_path):
    db = tmp_path / "m.duckdb"
    write_database(_tiles(0.5), _cells(), str(db))
    write_database(_tiles(0.9), _cells(), str(db))
    assert _iou(db) == 0.9


def test_a_half_written_file_from_a_crashed_run_is_cleared(tmp_path):
    db = tmp_path / "m.duckdb"
    (tmp_path / "m.duckdb.new").write_bytes(b"not a database")
    write_database(_tiles(0.9), _cells(), str(db))
    assert _iou(db) == 0.9


def test_the_metrics_tool_works_while_someone_else_is_reading(tmp_path, monkeypatch, another_reader):
    """
    The bug this exists for: the tools opened the database read-write, which locks the
    file against every other process, so a second person asking at the same moment failed.
    """
    db = tmp_path / "m.duckdb"
    write_database(_tiles(0.9), _cells(), str(db))
    another_reader(db)

    monkeypatch.setattr(graph_agent, "DB_PATH", str(db))
    reply = graph_agent.get_duckdb_metrics.invoke({"tile_id": "t"})
    assert "IoU: 0.9000" in reply and "Weak areas: none" in reply


@pytest.mark.skipif(os.name == "nt", reason="Windows cannot rename over an open file; "
                                            "write_database retries until the reader is done")
def test_the_database_can_be_rebuilt_while_someone_is_reading(tmp_path, another_reader):
    db = tmp_path / "m.duckdb"
    write_database(_tiles(0.5), _cells(), str(db))
    another_reader(db)

    write_database(_tiles(0.9), _cells(), str(db))
    assert _iou(db) == 0.9
