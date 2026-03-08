"""Compute distance-to-infrastructure features (substations, major roads)."""

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree

from src.utils import load_config, resolve_path


def _build_kdtree(gdf: gpd.GeoDataFrame):
    """Build a KD-tree from point geometries projected to EPSG:27700 (British National Grid)."""
    gdf_proj = gdf.to_crs(epsg=27700)
    coords = np.column_stack([gdf_proj.geometry.x, gdf_proj.geometry.y])
    return cKDTree(coords)


def extract_infrastructure_distances(sites: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """For each site, compute distance (km) to nearest substation and nearest major road."""
    cfg = config or load_config()
    infra_list = cfg["features"]["infrastructure"]

    # Convert sites to GeoDataFrame in BNG for metric distances
    sites_gdf = gpd.GeoDataFrame(
        sites,
        geometry=gpd.points_from_xy(sites["longitude"], sites["latitude"]),
        crs="EPSG:4326",
    ).to_crs(epsg=27700)
    site_coords = np.column_stack([sites_gdf.geometry.x, sites_gdf.geometry.y])

    results = {}

    if "substations" in infra_list:
        shp_path = resolve_path(cfg["paths"]["osm_substations_shp"])
        print(f"[Phase2] Loading substations from {shp_path} …")
        subs = gpd.read_file(shp_path)
        # Use centroids for polygon geometries
        subs["geometry"] = subs.geometry.centroid
        tree = _build_kdtree(subs)
        dists_m, _ = tree.query(site_coords)
        results["dist_substation_km"] = dists_m / 1000.0

    if "major_roads" in infra_list:
        shp_path = resolve_path(cfg["paths"]["osm_roads_shp"])
        print(f"[Phase2] Loading major roads from {shp_path} …")
        roads = gpd.read_file(shp_path)
        # Data is already point centroids from Overpass `out center`
        tree = _build_kdtree(roads)
        dists_m, _ = tree.query(site_coords)
        results["dist_major_road_km"] = dists_m / 1000.0

    df = pd.DataFrame(results, index=sites.index)
    print(f"[Phase2] Infrastructure features: {list(df.columns)}")
    return df


if __name__ == "__main__":
    sample = pd.DataFrame({"latitude": [51.5, 53.4], "longitude": [-0.1, -2.2]})
    print(extract_infrastructure_distances(sample))
