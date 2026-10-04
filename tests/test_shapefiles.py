"""
Tests for locating the shapefile inside a zipped export.

    pytest tests/ -q
"""

import os
import sys
import zipfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.metrics.shapefiles import shapefile_uri  # noqa: E402


def _zip(path, names) -> str:
    with zipfile.ZipFile(path, "w") as archive:
        for name in names:
            archive.writestr(name, b"")
    return str(path)


def test_shapefile_at_the_archive_root(tmp_path):
    path = _zip(tmp_path / "T1.zip", ["T1.shp", "T1.dbf", "T1.shx"])
    assert shapefile_uri(path) == f"zip://{path}"


def test_shapefile_wrapped_in_a_folder(tmp_path):
    """Every archive in data/ looks like this; a bare zip path cannot be read."""
    path = _zip(tmp_path / "T1.zip", ["T1/T1.shp", "T1/T1.dbf", "T1/T1.shx"])
    assert shapefile_uri(path) == f"zip://{path}!T1/T1.shp"


def test_macosx_stubs_are_ignored(tmp_path):
    path = _zip(tmp_path / "T1.zip", ["__MACOSX/T1/._T1.shp", "T1/T1.shp"])
    assert shapefile_uri(path) == f"zip://{path}!T1/T1.shp"


def test_archive_without_a_shapefile(tmp_path):
    assert shapefile_uri(_zip(tmp_path / "T1.zip", ["readme.txt"])) is None


def test_missing_archive(tmp_path):
    assert shapefile_uri(str(tmp_path / "nope.zip")) is None
