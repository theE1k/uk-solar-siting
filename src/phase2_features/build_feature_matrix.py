"""Assemble the full feature matrix by running all extractors on the labelled sites."""

import pandas as pd

from src.utils import load_config, resolve_path
from src.phase2_features.extract_solar_irradiance import extract_era5_features
from src.phase2_features.extract_pvgis import extract_pvgis_features
from src.phase2_features.extract_terrain import extract_terrain_features
from src.phase2_features.extract_infrastructure import extract_infrastructure_distances
from src.phase2_features.extract_grid_connection import extract_grid_connection
from src.phase2_features.extract_land_constraints import extract_land_constraints


def build_feature_matrix(config: dict | None = None) -> pd.DataFrame:
    cfg = config or load_config()
    labeled_path = resolve_path(cfg["paths"]["processed_data"]) / "labeled_sites.csv"

    print("[Phase2] Loading labelled sites …")
    sites = pd.read_csv(labeled_path)

    # Run each extractor and concatenate
    era5 = extract_era5_features(sites, cfg)
    pvgis = extract_pvgis_features(sites, cfg)
    terrain = extract_terrain_features(sites, cfg)
    infra = extract_infrastructure_distances(sites, cfg)
    grid = extract_grid_connection(sites, cfg)
    land = extract_land_constraints(sites, cfg)

    matrix = pd.concat([sites, era5, pvgis, terrain, infra, grid, land], axis=1)

    # Drop rows with critical NaNs
    before = len(matrix)
    matrix = matrix.dropna(subset=["ghi_pvgis_kwh_m2_day", "elevation_m"]).reset_index(drop=True)
    after = len(matrix)
    if before != after:
        print(f"[Phase2] Dropped {before - after} rows with missing critical features.")

    out_path = resolve_path(cfg["paths"]["feature_matrix_csv"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(out_path, index=False)

    print(f"[Phase2] Feature matrix: {matrix.shape[0]} rows x {matrix.shape[1]} cols")
    print(f"[Phase2] Saved to {out_path}")
    return matrix


if __name__ == "__main__":
    df = build_feature_matrix()
    print(df.describe())
