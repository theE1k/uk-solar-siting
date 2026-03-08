# UK Solar Siting — ML-Driven Site Selection for Northumbrian Water

An end-to-end machine learning pipeline that identifies optimal solar PV installation sites across Northumbrian Water's asset portfolio. Combines geospatial feature engineering, XGBoost classification, financial modelling, and an interactive Streamlit dashboard.

## Architecture

```
Raw Data (9 sources)          ML Pipeline                    Dashboard
───────────────────    ─────────────────────────    ─────────────────────────
UK REPD (solar farms)  ┌─ Phase 1: Labeling ──┐    ┌─ Interactive Streamlit ─┐
OS Terrain 50 (DEM)    │  pos/neg samples      │    │  Tab 1: Folium Map      │
ERA5 (climate)    ───► │                       │    │    5 toggleable layers  │
PVGIS (irradiance)     ├─ Phase 2: Features ──┤    │  Tab 2: Site Detail     │
OSM (infrastructure)   │  15 spatial features  │───►│    load/price/finance   │
Natural England (PA)   ├─ Phase 3: XGBoost ───┤    │  Tab 3: Rankings Table  │
EA Flood Zones         │  binary classifier    │    │    sortable + CSV       │
DEFRA (agri land)      ├─ Phase 4: Predict ───┤    │  Sidebar: live params   │
Ofgem FiT (industrial) │  400k grid cells      │    └─────────────────────────┘
                       └───────────────────────┘
```

## Features

- **15 geospatial features** extracted from terrain, climate, solar resource, infrastructure distance, and land constraint layers
- **Two XGBoost models**: general suitability (REPD-trained) + industrial self-consumption (FiT-trained)
- **Financial scoring**: NPV, payback period, battery storage value per site type
- **Composite ranking**: 40% geographic + 40% financial + 20% constraint score (adjustable)
- **Interactive dashboard** with real-time parameter adjustment, 5-layer Folium map, Plotly charts, and CSV export

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the ML pipeline

```bash
# Full pipeline (all 4 phases)
python run_pipeline.py

# Or run specific phases
python run_pipeline.py --phase 1 2    # labeling + features only
python run_pipeline.py --phase 3      # training only
```

### 3. Score Northumbrian Water sites

```bash
python -m src.score_financial
```

### 4. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

Open http://localhost:8501 in your browser.

## Pipeline Phases

### Phase 1 — Data Labeling

| Step | Description | Output |
|------|-------------|--------|
| Download REPD | UK Renewable Energy Planning Database | `data/raw/repd.csv` |
| Generate positives | Operational solar PV sites (BNG → WGS84) | ~1000 sites |
| Generate negatives | Random land points (min 2 km from positives) | ~700 sites |
| Combine | Merge, shuffle, save | `data/processed/labeled_sites.csv` |

### Phase 2 — Feature Engineering

15 features extracted per site from 7 spatial data layers:

| Category | Features | Source |
|----------|----------|--------|
| Terrain | elevation, slope, aspect, south-facing score | OS Terrain 50 DEM |
| Climate | mean temperature, cloud cover fraction | Copernicus ERA5 |
| Solar | GHI (kWh/m2/day), annual yield (kWh/kWp) | EU JRC PVGIS |
| Infrastructure | distance to substations, major roads (km) | OpenStreetMap |
| Grid | distance to 132 kV+ transmission lines (km) | OpenStreetMap |
| Constraints | national park, AONB, SSSI, flood zone, Grade 1 agri | Natural England, DEFRA, EA |

### Phase 3 — Model Training

- **Algorithm**: XGBoost (`binary:logistic`, AUC metric)
- **Split**: 80/20 stratified train/test
- **Hyperparameters**: max_depth=6, lr=0.1, 300 estimators, subsample=0.8
- **Outputs**: `xgb_model.json`, ROC curve, feature importance chart, evaluation report

### Phase 4 — Grid Prediction

- Generate 1 km grid over Great Britain (49.9–60.9 N, -8.2–1.8 E)
- Clip to UK land boundary
- Batch prediction (10 k points/batch) with full feature extraction
- Output: `grid_predictions.csv` (~400 k probability scores) + interactive heatmap

### Industrial Model (Bonus)

Separate model trained on Ofgem FiT non-domestic PV installations (>= 50 kW, self-consumption) with OSM industrial sites as negatives. Predictions merged into the composite geo-score.

### Financial Scoring

For each Northumbrian Water candidate site:

| Metric | Description |
|--------|-------------|
| `capacity_kwp` | Estimated PV capacity by site type (100–500 kWp) |
| `annual_gen_kwh` | PVGIS yield x capacity |
| `annual_saving_gbp` | Generation x self-consumption ratio x tariff |
| `payback_yr` | CapEx / annual saving |
| `npv_gbp` | 25-year NPV at 6% discount, 0.5% degradation |
| `battery_value` | Battery storage suitability score (0–1) |
| `fin_score` | Normalised financial score (payback 5 yr → 1.0, 25 yr → 0.0) |

**Composite score** = w_geo x geo_score + w_fin x fin_score + w_const x constraint_score

## Dashboard

### Tab 1 — Map

Five toggleable Folium layers:

1. **Solar Suitability Heatmap** — ML model probability (blue → red)
2. **NW Sites** — Circle markers colour-coded by final score (green/orange/red)
3. **PVGIS Irradiance** — Satellite solar resource heatmap (normalised 0–1)
4. **Protected Areas** — National parks + AONB polygons (red)
5. **Flood Zones** — EA Flood Zone 2+3 polygons (blue)

### Tab 2 — Site Detail

Three Plotly sub-charts for the selected site:

1. **24 h Load & Solar** — site-type load profile vs solar generation curve
2. **Electricity Price** — UK commercial tariff simulation with solar hour overlay
3. **Financial Projection** — 25-year cumulative savings vs CapEx with payback marker

### Tab 3 — Rankings

Styled dataframe with colour gradients on final score and NPV. Battery value displayed as star ratings. One-click CSV download.

### Sidebar Controls

- Electricity tariff (GBP/kWh)
- CapEx per kWp
- Project lifetime (years)
- Score weight sliders (geographic / financial / constraint)
- Site type filter
- Map layer toggles

## Data Sources

| Dataset | Provider | Licence |
|---------|----------|---------|
| Renewable Energy Planning Database | UK Government | OGL v3.0 |
| OS Terrain 50 | Ordnance Survey | OS OpenData |
| ERA5 Monthly Means | Copernicus / ECMWF | Copernicus Licence |
| PVGIS 5.2 | EU Joint Research Centre | Open |
| Substations, Roads, Transmission | OpenStreetMap | ODbL |
| National Parks, AONB | Natural England | OGL v3.0 |
| SSSI | Natural England | OGL v3.0 |
| Flood Zones 2+3 | Environment Agency | OGL v3.0 |
| Agricultural Land Classification | DEFRA | OGL v3.0 |
| FiT Installation Report | Ofgem | OGL v3.0 |

## Project Structure

```
uk-solar-siting/
├── configs/config.yaml          # Central pipeline configuration
├── dashboard/app.py             # Streamlit interactive dashboard
├── data/
│   ├── raw/                     # Downloaded spatial data (~1.1 GB)
│   ├── processed/               # Feature matrices
│   └── output/                  # Models, predictions, maps
├── src/
│   ├── phase1_labeling/         # REPD download, pos/neg sampling
│   ├── phase2_features/         # 7 feature extractors
│   ├── phase3_model/            # XGBoost train + evaluate
│   ├── phase4_prediction/       # Grid generation, prediction, heatmap
│   ├── build_industrial_dataset.py
│   ├── train_industrial_model.py
│   ├── score_financial.py       # Financial metrics + composite ranking
│   ├── download_all.py          # Orchestrate all data downloads
│   └── utils.py                 # Config loader, haversine
├── run_pipeline.py              # CLI entry point
├── requirements.txt
└── pyproject.toml
```

## Requirements

- Python >= 3.10
- ~2 GB disk for raw spatial data
- No GPU required (CPU XGBoost)
- Copernicus CDS API key for ERA5 download (optional — pre-cached data included)

## Licence

This project was developed for Northumbrian Water as part of an academic research project.
