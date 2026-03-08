"""Run the trained model on the prediction grid to produce suitability probabilities."""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from tqdm import tqdm

from src.utils import load_config, resolve_path
from src.phase4_prediction.generate_grid import generate_grid
from src.phase2_features.extract_solar_irradiance import extract_era5_features
from src.phase2_features.extract_pvgis import extract_pvgis_features
from src.phase2_features.extract_terrain import extract_terrain_features
from src.phase2_features.extract_infrastructure import extract_infrastructure_distances
from src.phase2_features.extract_grid_connection import extract_grid_connection
from src.phase2_features.extract_land_constraints import extract_land_constraints


def predict_grid(config: dict | None = None) -> pd.DataFrame:
    cfg = config or load_config()

    # Load model
    model_path = resolve_path(cfg["paths"]["model_file"])
    model = XGBClassifier()
    model.load_model(str(model_path))
    feature_names = model.get_booster().feature_names
    print(f"[Phase4] Model loaded. Expected features: {feature_names}")

    # Generate grid
    grid = generate_grid(cfg)

    # Extract features in batches to manage memory
    batch_size = 10_000
    all_results = []

    for start in tqdm(range(0, len(grid), batch_size), desc="Predicting"):
        batch = grid.iloc[start:start + batch_size].copy()

        era5 = extract_era5_features(batch, cfg)
        pvgis = extract_pvgis_features(batch, cfg)
        terrain = extract_terrain_features(batch, cfg)
        infra = extract_infrastructure_distances(batch, cfg)
        grid_conn = extract_grid_connection(batch, cfg)
        land = extract_land_constraints(batch, cfg)

        features = pd.concat([batch.reset_index(drop=True),
                               era5.reset_index(drop=True),
                               pvgis.reset_index(drop=True),
                               terrain.reset_index(drop=True),
                               infra.reset_index(drop=True),
                               grid_conn.reset_index(drop=True),
                               land.reset_index(drop=True)], axis=1)

        # Align columns with training features
        X = features.reindex(columns=feature_names, fill_value=0)
        probs = model.predict_proba(X)[:, 1]

        batch_result = batch.reset_index(drop=True).copy()
        batch_result["probability"] = probs
        all_results.append(batch_result)

    result = pd.concat(all_results, ignore_index=True)

    # Save
    out_path = resolve_path(cfg["paths"]["grid_predictions_csv"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f"[Phase4] Predictions saved to {out_path} ({len(result)} points)")

    return result


if __name__ == "__main__":
    predict_grid()
