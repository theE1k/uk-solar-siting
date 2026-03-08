"""Generate negative samples: random UK land points far from existing solar sites."""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from src.utils import load_config, resolve_path, haversine_km


def generate_negative(positives: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    cfg = config or load_config()
    lab = cfg["labeling"]
    seed = lab["random_seed"]
    ratio = lab["negative_ratio"]
    min_dist = lab["min_distance_km"]
    n_target = int(len(positives) * ratio)

    # Load UK boundary for land-only sampling
    boundary_path = resolve_path(cfg["paths"]["uk_boundary_shp"])
    if boundary_path.exists():
        uk = gpd.read_file(boundary_path).unary_union
    else:
        # Fallback: approximate UK bounding box with no land mask
        print("[Phase1] WARNING: UK boundary shapefile not found — using bounding box only.")
        uk = None

    pos_lat = positives["latitude"].values
    pos_lon = positives["longitude"].values

    rng = np.random.default_rng(seed)
    # UK approximate bounding box
    lat_min, lat_max = 49.9, 60.9
    lon_min, lon_max = -8.2, 1.8

    negatives = []
    attempts = 0
    max_attempts = n_target * 20

    print(f"[Phase1] Generating {n_target} negative samples (min {min_dist} km from positives) …")
    while len(negatives) < n_target and attempts < max_attempts:
        batch = 5000
        lats = rng.uniform(lat_min, lat_max, batch)
        lons = rng.uniform(lon_min, lon_max, batch)

        for lat, lon in zip(lats, lons):
            if len(negatives) >= n_target:
                break

            # Check within UK land boundary if available
            if uk is not None and not uk.contains(Point(lon, lat)):
                continue

            # Check minimum distance to all positive sites
            dists = haversine_km(lat, lon, pos_lat, pos_lon)
            if dists.min() >= min_dist:
                negatives.append({"latitude": lat, "longitude": lon, "label": 0})

        attempts += batch

    df_neg = pd.DataFrame(negatives)
    print(f"[Phase1] Negative samples: {len(df_neg)}")
    return df_neg


if __name__ == "__main__":
    from src.phase1_labeling.generate_positive import generate_positive
    pos = generate_positive()
    neg = generate_negative(pos)
    print(neg.head())
