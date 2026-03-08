"""Compute distance to nearest high-voltage transmission line (132kV+)."""

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree

from src.utils import load_config, resolve_path


def extract_grid_connection(sites: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """For each site, compute distance (km) to nearest 132kV+ transmission line."""
    cfg = config or load_config()

    if "transmission_lines" not in cfg["features"]["infrastructure"]:
        return pd.DataFrame(index=sites.index)

    shp_path = resolve_path(cfg["paths"]["osm_transmission_shp"])
    print(f"[Phase2] Loading transmission lines from {shp_path} …")
    lines = gpd.read_file(shp_path)

    sites_gdf = gpd.GeoDataFrame(
        sites,
        geometry=gpd.points_from_xy(sites["longitude"], sites["latitude"]),
        crs="EPSG:4326",
    ).to_crs(epsg=27700)
    site_coords = np.column_stack([sites_gdf.geometry.x, sites_gdf.geometry.y])

    # Each row in the shapefile is a point centroid of a line segment
    lines_proj = lines.to_crs(epsg=27700)
    line_coords = np.column_stack([lines_proj.geometry.x, lines_proj.geometry.y])

    tree = cKDTree(line_coords)
    dists_m, _ = tree.query(site_coords)

    result = pd.DataFrame({"dist_transmission_km": dists_m / 1000.0}, index=sites.index)
    print(f"[Phase2] Grid connection features: {list(result.columns)}")
    return result


if __name__ == "__main__":
    sample = pd.DataFrame({"latitude": [51.5, 53.4], "longitude": [-0.1, -2.2]})
    print(extract_grid_connection(sample))
