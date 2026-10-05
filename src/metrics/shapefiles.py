"""
Locating and reading the shapefile inside a zipped export.

Its own module so the Streamlit viewer and the vision tool can use it without importing the
pipeline, which pulls in XGBoost and SHAP.
"""

import os
import zipfile

import geopandas as gpd
from shapely.geometry import box


def shapefile_uri(zip_path):
    """
    Build a GeoPandas URI for the .shp inside a zipped shapefile.

    Some exports put the files at the archive root, others wrap them in a folder named
    after the tile. A bare zip:// URI only finds the former, so locate the .shp and
    address it explicitly. Returns None if the archive holds no shapefile.
    """
    if not os.path.exists(zip_path):
        return None

    with zipfile.ZipFile(zip_path) as archive:
        # __MACOSX holds resource-fork stubs that look like real entries but are not.
        shps = [n for n in archive.namelist()
                if n.lower().endswith('.shp') and not n.startswith('__MACOSX/')]

    if not shps:
        return None
    if len(shps) > 1:
        print(f"  -> Warning: {os.path.basename(zip_path)} holds {len(shps)} shapefiles, using {shps[0]}")

    inner = shps[0]
    return f"zip://{zip_path}" if '/' not in inner else f"zip://{zip_path}!{inner}"


def read_trails(zip_path, crs):
    """
    A zipped shapefile as a GeoDataFrame in the given CRS, the same way the pipeline reads
    it before computing IoU. Returns None when there is nothing in it to draw.
    """
    uri = shapefile_uri(zip_path)
    if not uri:
        return None

    gdf = gpd.read_file(uri)
    if gdf.empty:
        return None
    if crs is not None and gdf.crs is not None and gdf.crs != crs:
        gdf = gdf.to_crs(crs)
    return gdf


def trails_within(zip_path, crs, bounds):
    """
    The part of a zipped shapefile inside bounds, (minx, miny, maxx, maxy) in the given
    CRS, as one geometry. Returns None when none of it falls there.
    """
    gdf = read_trails(zip_path, crs)
    if gdf is None:
        return None
    inside = gdf.geometry.unary_union.intersection(box(*bounds))
    return None if inside.is_empty else inside
