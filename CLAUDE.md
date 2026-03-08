# UK Solar Siting — Claude Code Context

## What This Project Does

ML-based solar PV site selection for Northumbrian Water (NW). Predicts suitability across a 1 km UK grid using XGBoost, then scores NW's ~150 candidate sites with financial metrics and land constraints.

## Key Commands

```bash
python run_pipeline.py              # Run full pipeline (phases 1-4)
python run_pipeline.py --phase 3    # Run specific phase(s)
python -m src.score_financial       # Score NW sites (financial + composite)
streamlit run dashboard/app.py      # Launch interactive dashboard
```

## Architecture

- **Phase 1** (`src/phase1_labeling/`): Generate binary labels from REPD
- **Phase 2** (`src/phase2_features/`): Extract 15 features from 7 spatial layers
- **Phase 3** (`src/phase3_model/`): Train/evaluate XGBoost classifier
- **Phase 4** (`src/phase4_prediction/`): Predict on 400k UK grid cells, generate heatmap
- **Industrial model** (`src/build_industrial_dataset.py`, `src/train_industrial_model.py`): Separate model for self-consumption sites
- **Financial scoring** (`src/score_financial.py`): NPV, payback, composite ranking
- **Dashboard** (`dashboard/app.py`): Streamlit + Folium + Plotly

## Critical Files

- `configs/config.yaml` — all paths, hyperparameters, feature config
- `data/output/nw_sites_final_ranking.csv` — final ranked NW sites
- `data/output/grid_predictions.csv` — 400k suitability predictions
- `data/output/xgb_model.json` — trained XGBoost model

## Data Flow

```
data/raw/ → src/phase1 → data/processed/labeled_sites.csv
          → src/phase2 → data/processed/feature_matrix.csv
          → src/phase3 → data/output/xgb_model.json
          → src/phase4 → data/output/grid_predictions.csv
          → src/score_financial → data/output/nw_sites_final_ranking.csv
          → dashboard/app.py (reads output + raw)
```

## Coordinate Systems

- Raw terrain: EPSG:27700 (British National Grid)
- All other data: EPSG:4326 (WGS84)
- `pyproj.Transformer` used for conversions in feature extractors

## Dashboard Data Dependencies

The dashboard reads these files:
- `data/output/grid_predictions.csv` (11 MB) — suitability heatmap
- `data/output/nw_sites_final_ranking.csv` (31 KB) — site rankings
- `data/raw/pvgis_grid.csv` (72 KB) — PVGIS irradiance layer
- `data/raw/protected_areas.shp` (optional) — protected areas layer
- `data/raw/flood_zones.shp` (optional) — flood zones layer

## Conventions

- All user-facing text in English
- Config-driven paths (avoid hardcoding)
- Dashboard financial params adjustable via sidebar sliders
- Composite score weights sum to 1.0 (enforced in UI)
