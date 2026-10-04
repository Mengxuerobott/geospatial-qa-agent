"""
Locating the shapefile inside a zipped export.

Its own module so the Streamlit viewer can use it without importing the pipeline, which
pulls in XGBoost and SHAP.
"""

import os
import zipfile


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
