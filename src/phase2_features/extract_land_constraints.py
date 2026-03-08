"""Extract binary land-constraint flags (protected areas, agricultural land)."""

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from src.utils import load_config, resolve_path


def extract_land_constraints(sites: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """For each site, flag whether it falls within a protected or constrained area."""
    cfg = config or load_config()
    constraint_names = cfg["features"]["land_constraints"]

    sites_gdf = gpd.GeoDataFrame(
        sites,
        geometry=gpd.points_from_xy(sites["longitude"], sites["latitude"]),
        crs="EPSG:4326",
    )

    results = {}

    if "national_park" in constraint_names or "aonb" in constraint_names:
        shp_path = resolve_path(cfg["paths"]["protected_areas_shp"])
        print(f"[Phase2] Loading protected areas from {shp_path} …")
        protected = gpd.read_file(shp_path).to_crs("EPSG:4326")

        # Spatial join: check if each site is within any protected polygon
        joined = gpd.sjoin(sites_gdf, protected, how="left", predicate="within")

        # The 'DESIG' or 'designation' column typically tells us the type
        desig_col = _find_designation_col(protected)
        if desig_col:
            results["in_national_park"] = (
                joined[desig_col].str.contains("National Park", case=False, na=False)
                .groupby(joined.index).any().astype(int).reindex(sites.index, fill_value=0)
            )
            results["in_aonb"] = (
                joined[desig_col].str.contains("AONB|Outstanding Natural Beauty", case=False, na=False)
                .groupby(joined.index).any().astype(int).reindex(sites.index, fill_value=0)
            )
        else:
            # No designation column — just flag any overlap
            in_protected = joined.index.duplicated(keep=False) | joined["index_right"].notna()
            results["in_protected_area"] = in_protected.groupby(joined.index).any().astype(int).reindex(sites.index, fill_value=0)

    if "grade1_agricultural" in constraint_names:
        shp_path = resolve_path(cfg["paths"]["agricultural_land_shp"])
        print(f"[Phase2] Loading agricultural land from {shp_path} …")
        agri = gpd.read_file(shp_path).to_crs("EPSG:4326")
        joined_agri = gpd.sjoin(sites_gdf, agri, how="left", predicate="within")
        results["in_grade1_agri"] = (
            joined_agri["index_right"].notna()
            .groupby(joined_agri.index).any().astype(int).reindex(sites.index, fill_value=0)
        )

    if "sssi" in constraint_names:
        shp_path = resolve_path(cfg["paths"]["sssi_shp"])
        if shp_path.exists():
            print(f"[Phase2] Loading SSSI from {shp_path} …")
            sssi = gpd.read_file(shp_path).to_crs("EPSG:4326")
            joined_sssi = gpd.sjoin(sites_gdf, sssi, how="left", predicate="within")
            results["in_sssi"] = (
                joined_sssi["index_right"].notna()
                .groupby(joined_sssi.index).any().astype(int).reindex(sites.index, fill_value=0)
            )
        else:
            print(f"[Phase2] WARNING: SSSI shapefile not found at {shp_path}, skipping.")

    if "flood_zone" in constraint_names:
        shp_path = resolve_path(cfg["paths"]["flood_zones_shp"])
        if shp_path.exists():
            print(f"[Phase2] Loading flood zones from {shp_path} …")
            flood = gpd.read_file(shp_path).to_crs("EPSG:4326")
            joined_flood = gpd.sjoin(sites_gdf, flood, how="left", predicate="within")
            results["in_flood_zone"] = (
                joined_flood["index_right"].notna()
                .groupby(joined_flood.index).any().astype(int).reindex(sites.index, fill_value=0)
            )
        else:
            print(f"[Phase2] WARNING: Flood zones shapefile not found at {shp_path}, skipping.")

    df = pd.DataFrame(results, index=sites.index)
    print(f"[Phase2] Land constraint features: {list(df.columns)}")
    return df


def _find_designation_col(gdf: gpd.GeoDataFrame):
    for col in gdf.columns:
        if col.lower() in ("desig", "designation", "desig_abbr", "type"):
            return col
    return None


if __name__ == "__main__":
    sample = pd.DataFrame({"latitude": [51.5, 54.5], "longitude": [-0.1, -1.6]})
    print(extract_land_constraints(sample))
