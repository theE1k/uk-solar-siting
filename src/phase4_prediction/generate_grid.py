"""Generate a dense lat/lon grid clipped to the UK land boundary."""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.prepared import prep

from src.utils import load_config, resolve_path


def generate_grid(config: dict | None = None) -> pd.DataFrame:
    cfg = config or load_config()
    pred = cfg["prediction"]
    bbox = pred["bbox"]
    spacing_km = pred["grid_spacing_km"]

    # 1 degree latitude ~ 111 km
    lat_step = spacing_km / 111.0
    # 1 degree longitude ~ 111 * cos(mid_lat) km
    mid_lat = np.radians((bbox["min_lat"] + bbox["max_lat"]) / 2)
    lon_step = spacing_km / (111.0 * np.cos(mid_lat))

    lats = np.arange(bbox["min_lat"], bbox["max_lat"], lat_step)
    lons = np.arange(bbox["min_lon"], bbox["max_lon"], lon_step)

    grid_lat, grid_lon = np.meshgrid(lats, lons, indexing="ij")
    df = pd.DataFrame({
        "latitude": grid_lat.ravel(),
        "longitude": grid_lon.ravel(),
    })
    print(f"[Phase4] Raw grid: {len(df)} points")

    # Clip to UK land boundary
    boundary_path = resolve_path(cfg["paths"]["uk_boundary_shp"])
    if boundary_path.exists():
        print("[Phase4] Clipping grid to UK land boundary …")
        uk = gpd.read_file(boundary_path).to_crs("EPSG:4326").unary_union
        uk_prep = prep(uk)
        mask = [uk_prep.contains(Point(lon, lat))
                for lat, lon in zip(df["latitude"], df["longitude"])]
        df = df[mask].reset_index(drop=True)
        print(f"[Phase4] After clipping: {len(df)} land points")
    else:
        print("[Phase4] WARNING: UK boundary not found, using full bbox.")

    return df


if __name__ == "__main__":
    grid = generate_grid()
    print(grid.describe())
