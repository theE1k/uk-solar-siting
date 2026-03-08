# UK Solar Siting — Presentation Notes for PPT Generation

> Feed this document to Gemini with the prompt:
> "Generate a professional PowerPoint presentation based on these structured slide notes. Use a clean, modern design with blue/yellow solar-themed colour palette. Include suggested diagram layouts where indicated."

---

## Slide 1: Title Slide

**Title:** UK Solar Siting: ML-Driven Site Selection for Northumbrian Water

**Subtitle:** Combining XGBoost, Geospatial Analysis & Financial Modelling

**Visual:** Solar panel array with UK map overlay

**Speaker Notes:**
- This project builds an end-to-end machine learning pipeline to identify optimal solar PV installation sites across Northumbrian Water's asset portfolio
- The system combines geospatial feature engineering, gradient-boosted classification, financial modelling, and an interactive web dashboard
- All data sourced from UK public open data under OGL v3.0

---

## Slide 2: Executive Summary

**Key Points (3 bullets):**
- **Problem:** Northumbrian Water operates ~2,000 sites across North East England and needs to identify which are best suited for solar PV self-consumption
- **Solution:** End-to-end ML pipeline that processes 9 spatial data sources, trains XGBoost classifiers, and ranks ~150 candidate sites with financial projections
- **Result:** Interactive Streamlit dashboard with adjustable parameters, 5-layer map, per-site financial analysis, and downloadable ranked list

**Visual:** Simple 3-box flow: Problem → Solution → Result

**Speaker Notes:**
- The system goes from raw open data to actionable site rankings in a single pipeline run
- Stakeholders can adjust tariff, CapEx, and scoring weights in real time through the dashboard
- All code is reproducible and config-driven

---

## Slide 3: Project Motivation

**Key Points:**
- Northumbrian Water's **net-zero carbon** commitment
- Solar PV for **self-consumption** (behind-the-meter), not grid export
- UK North East has viable solar resource (~750–850 kWh/kWp/yr from PVGIS)
- Significant **cost savings** from avoided electricity purchases at commercial rates (£0.15+/kWh)
- NW has diverse site types: wastewater plants, pumping stations, water works, reservoirs — each with different load profiles and solar potential

**Visual:** Map of NW region with solar irradiance overlay

**Speaker Notes:**
- NW's sites have large available land area and high electricity demand — ideal for self-consumption solar
- Behind-the-meter solar avoids grid charges and distribution losses
- The diversity of site types requires a flexible assessment framework, not a one-size-fits-all approach

---

## Slide 4: System Architecture

**Visual — Flow diagram:**
```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   9 Data Sources │────►│   ML Pipeline    │────►│   Dashboard     │
│                  │     │                  │     │                  │
│  REPD            │     │  Phase 1: Label  │     │  Tab 1: Map     │
│  OS Terrain 50   │     │  Phase 2: Feats  │     │  Tab 2: Detail  │
│  ERA5 Climate    │     │  Phase 3: Train  │     │  Tab 3: Ranking │
│  PVGIS Solar     │     │  Phase 4: Pred   │     │                  │
│  OSM Infra       │     │                  │     │  Sidebar Params  │
│  Constraints x4  │     │  + Industrial    │     │                  │
│                  │     │  + Financial     │     │                  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

**Speaker Notes:**
- Left: 9 open data sources covering terrain, climate, solar resource, infrastructure, and land constraints
- Centre: 4-phase ML pipeline plus supplementary industrial model and financial scoring
- Right: Interactive Streamlit dashboard for stakeholder exploration
- The entire pipeline is config-driven via a single YAML file

---

## Slide 5: Data Sources

**Table:**

| Dataset | Provider | Format | Purpose |
|---------|----------|--------|---------|
| Renewable Energy Planning Database | UK Government | CSV | Training labels (solar farm locations) |
| OS Terrain 50 | Ordnance Survey | GeoTIFF | Elevation, slope, aspect |
| ERA5 Monthly Means | Copernicus / ECMWF | NetCDF | Temperature, cloud cover |
| PVGIS 5.2 | EU Joint Research Centre | CSV/API | Solar irradiance, annual yield |
| Substations & Roads | OpenStreetMap | Shapefile | Infrastructure distances |
| 132kV+ Transmission | OpenStreetMap | Shapefile | Grid connection distance |
| National Parks & AONB | Natural England | Shapefile | Protected area constraints |
| SSSI | Natural England | Shapefile | Ecological constraints |
| Flood Zones 2+3 | Environment Agency | Shapefile | Flood risk constraints |
| Agricultural Land | DEFRA | Shapefile | Grade 1 farmland constraint |

**Speaker Notes:**
- All data is freely available under UK Open Government Licence v3.0 or equivalent
- Total raw data: ~1.1 GB (mostly terrain tiles)
- Data download is orchestrated by a single script (`download_all.py`) with fallback caching

---

## Slide 6: Phase 1 — Data Labeling

**Key Points:**
- **Positive samples**: ~1,000 operational solar PV sites from UK REPD (>10 kW, operational or under construction)
- **Negative samples**: ~700 random points on UK land, minimum 2 km from any positive site
- Coordinate conversion: British National Grid (EPSG:27700) → WGS84 (EPSG:4326)
- Final output: `labeled_sites.csv` with ~1,700 binary-labeled sites

**Visual:** Map showing positive (green dots) and negative (red dots) samples across UK

**Speaker Notes:**
- REPD is the authoritative UK database of renewable energy installations
- The 2 km exclusion zone ensures negatives are genuinely different from positives
- Balanced sampling (1:1 ratio) prevents class imbalance issues
- BNG to WGS84 conversion is essential because REPD uses Ordnance Survey grid references

---

## Slide 7: Phase 2 — Feature Engineering

**Table: 15 Features from 7 Spatial Layers**

| Category | Feature | Source | Description |
|----------|---------|--------|-------------|
| Terrain | `elevation_m` | OS Terrain 50 | Metres above sea level |
| Terrain | `slope_deg` | OS Terrain 50 | Gradient in degrees |
| Terrain | `aspect_deg` | OS Terrain 50 | Compass direction of slope |
| Terrain | `south_facing_score` | OS Terrain 50 | cos(aspect - 180°), 1.0 = due south |
| Climate | `mean_temp_c` | ERA5 | 5-year average temperature (°C) |
| Climate | `cloud_cover_frac` | ERA5 | Mean cloud cover fraction (0–1) |
| Solar | `ghi_pvgis_kwh_m2_day` | PVGIS | Global horizontal irradiance |
| Solar | `pvout_kwh_kwp` | PVGIS | Annual PV yield per kWp installed |
| Infra | `dist_substation_km` | OSM | Distance to nearest substation |
| Infra | `dist_major_road_km` | OSM | Distance to nearest A/B road |
| Grid | `dist_transmission_km` | OSM | Distance to nearest 132kV+ line |
| Constraint | `in_national_park` | Natural England | Binary flag |
| Constraint | `in_aonb` | Natural England | Binary flag |
| Constraint | `in_sssi` | Natural England | Binary flag |
| Constraint | `in_flood_zone` | EA | Binary flag |
| Constraint | `in_grade1_agri` | DEFRA | Binary flag |

**Speaker Notes:**
- Features capture the key factors that determine solar suitability: solar resource, terrain, grid access, and planning constraints
- Infrastructure distances computed using KD-trees on projected coordinates for metric accuracy
- Constraint flags from spatial joins with vector polygons
- PVGIS data interpolated from a 0.25° grid using bilinear interpolation with nearest-neighbour fallback

---

## Slide 8: Feature Engineering — Technical Methods

**Key Points:**
- **Terrain**: Rasterio DEM sampling with on-the-fly BNG projection; slope/aspect from finite differences
- **Infrastructure**: SciPy KD-tree on BNG-projected OSM centroids (O(log n) lookup per site)
- **Constraints**: GeoPandas spatial join (`sjoin`, predicate="within") against multi-polygon shapefiles
- **Solar (PVGIS)**: SciPy `griddata` with bilinear mode + nearest fallback for coastal cells
- **Batch processing**: 10,000 points per batch for Phase 4 grid prediction (~400k total)

**Visual:** Diagram showing data flow from raw spatial layers through extractors to feature matrix

**Speaker Notes:**
- KD-tree provides sub-millisecond nearest-neighbour queries even for large point sets
- Spatial joins handle complex multi-polygon features (e.g., SSSI with thousands of polygons)
- Batch processing keeps memory usage manageable when predicting across the full UK grid
- All extractors use module-level caching to avoid reloading large rasters/shapefiles

---

## Slide 9: Phase 3 — XGBoost Model Training

**Key Points:**
- **Algorithm**: XGBoost gradient-boosted decision tree
- **Objective**: Binary logistic (probability output 0–1)
- **Hyperparameters**: max_depth=6, learning_rate=0.1, 300 estimators, subsample=0.8, colsample=0.8
- **Regularisation**: L1 (alpha=0.1) + L2 (lambda=1.0)
- **Split**: 80% train / 20% test (stratified by label)
- **Evaluation metric**: ROC-AUC

**Visual:** XGBoost decision tree ensemble diagram

**Speaker Notes:**
- XGBoost is well-suited for tabular geospatial data with mixed feature types (continuous + binary)
- Regularisation prevents overfitting on the relatively small training set (~1,700 samples)
- Stratified split preserves the positive/negative ratio in both train and test sets
- The model outputs a probability score per location, enabling threshold-free ranking

---

## Slide 10: Model Evaluation Results

**Key Metrics (from evaluation_report.json):**
- ROC-AUC: ~0.92
- Accuracy: ~0.87
- Precision / Recall / F1 scores

**Visuals:**
1. ROC curve (screenshot from `data/output/roc_curve.png`)
2. Feature importance bar chart (screenshot from `data/output/feature_importance.png`)

**Top Predictive Features (typical):**
1. `pvout_kwh_kwp` — Annual solar yield (strongest predictor)
2. `dist_substation_km` — Grid connection proximity
3. `slope_deg` — Terrain gradient
4. `south_facing_score` — Southern exposure
5. `elevation_m` — Altitude

**Speaker Notes:**
- AUC of 0.92 indicates strong discriminative power between suitable and unsuitable sites
- Solar yield (PVGIS) is the strongest feature — sites with higher irradiance are more likely to be developed
- Infrastructure proximity is the second most important group — grid connection and road access are practical requirements
- Constraint features have lower importance because they act as binary filters rather than continuous gradients

---

## Slide 11: Industrial Self-Consumption Model

**Key Points:**
- Separate XGBoost model trained specifically for **behind-the-meter** scenarios
- **Positives**: Ofgem Feed-in Tariff register — non-domestic PV installations ≥50 kW with "No Export" or self-consumption tariff (~360 sites)
- **Negatives**: OSM industrial/commercial landuse centroids (≥5 km from positives)
- **Geocoding**: UK postcode district centroids (GeoNames)
- **Same 15 features** as the main model
- **Output merged** with REPD model: `geo_score = mean(repd_prob, industrial_prob)`

**Visual:** Venn diagram showing REPD model + Industrial model → Composite geo-score

**Speaker Notes:**
- The FiT register captures real-world industrial self-consumption installations — a closer analogue to NW's use case than utility-scale REPD farms
- Smaller training set (~360 vs ~1,700) but more targeted to the actual deployment scenario
- Averaging the two model probabilities balances general suitability with industrial relevance

---

## Slide 12: Phase 4 — UK-Wide Prediction

**Key Points:**
- Generate 1 km resolution grid over Great Britain (49.9–60.9°N, -8.2–1.8°E)
- Clip to UK land boundary (Natural Earth shapefile)
- **~400,000 land cells** after clipping
- Full feature extraction per cell (same 15 features)
- XGBoost inference: probability [0, 1] per cell
- Output: `grid_predictions.csv` (11 MB)

**Visual:** Before/after — empty grid → coloured probability heatmap

**Speaker Notes:**
- 1 km resolution is a good balance between detail and computational cost
- Feature extraction is the bottleneck — 10k-point batches keep memory manageable
- The probability surface can be overlaid with NW site locations to assess local suitability context
- Total processing time: approximately 2-3 hours on CPU

---

## Slide 13: Suitability Heatmap

**Visual:** Screenshot of the interactive Folium heatmap (from `data/output/suitability_heatmap.html`)

**Colour Gradient:**
- Blue (0.0–0.2): Low suitability
- Green (0.2–0.4): Below average
- Yellow (0.4–0.6): Moderate
- Orange (0.6–0.8): Good suitability
- Red (0.8–1.0): Excellent suitability

**Key Observations:**
- Southern and eastern England show highest suitability (higher irradiance, flatter terrain)
- NW's operating region (North East) shows moderate-to-good suitability
- Mountainous areas (Lake District, Scottish Highlands) show low suitability
- Coastal areas show mixed results (good irradiance but constraint overlaps)

**Speaker Notes:**
- The heatmap covers all of Great Britain at 1 km resolution
- Hot spots correlate well with known solar farm clusters (e.g., East Anglia, Somerset)
- NW sites can be assessed against their local suitability context
- Interactive version allows zooming, panning, and layer toggling

---

## Slide 14: Financial Scoring Framework

**Formula:**
```
Composite Score = w_geo × geo_score + w_fin × fin_score + w_const × constraint_score
```
Default weights: **40% geographic + 40% financial + 20% constraints** (adjustable in dashboard)

**Financial Metrics per Site:**

| Metric | Formula |
|--------|---------|
| Capacity (kWp) | Site-type default (100–500 kWp) |
| Annual Generation | capacity × PVGIS yield |
| Annual Savings | generation × self-consumption ratio × tariff |
| CapEx | capacity × £/kWp |
| Payback | CapEx / annual savings |
| NPV | 25-year discounted cash flow (6% rate, 0.5%/yr degradation) |
| Financial Score | normalised payback: 5yr → 1.0, 25yr → 0.0 |

**Speaker Notes:**
- The composite score balances technical suitability, financial viability, and planning risk
- NPV uses a 6% discount rate (typical for UK infrastructure) and 0.5%/yr panel degradation
- Constraint score penalises sites in protected areas, SSSIs, and flood zones
- All parameters can be adjusted interactively in the dashboard sidebar

---

## Slide 15: Site Type Analysis

**Table:**

| Site Type | Capacity (kWp) | Self-Consumption | Battery Value | Load Profile |
|-----------|----------------|------------------|---------------|--------------|
| Wastewater Plant | 500 | 85% | Medium | 24h stable, peaks 6am/7pm |
| Water Works | 300 | 80% | Medium | Dual peaks (morning/evening) |
| Pumping Station | 100 | 60% | High (★★★) | Binary: high daytime, low night |
| Industrial | 200 | 70% | Medium | Weekday daytime high |
| Reservoir (covered) | 150 | 65% | Medium | Moderate daytime, low night |

**Visual:** 5 small 24h load profile charts side-by-side

**Speaker Notes:**
- Wastewater plants are the strongest candidates: large capacity, very high self-consumption, and stable baseload
- Pumping stations have the highest battery storage value due to intermittent high-power demand
- Self-consumption ratio determines how much generated solar directly offsets grid purchases
- Load profiles are estimated templates — real data would improve accuracy

---

## Slide 16: Interactive Dashboard — Overview

**Visual:** Full dashboard screenshot showing all 3 tabs and sidebar

**Key Features:**
- **Real-time parameter adjustment**: tariff, CapEx, project life, score weights
- **3 analysis tabs**: Map, Site Detail, Rankings
- **KPI strip**: total sites, average payback, average NPV, high-priority count
- **Site type filtering**: toggle any combination of 5 site types
- **Built with**: Streamlit + Folium + Plotly

**Speaker Notes:**
- The dashboard is designed for stakeholder engagement — non-technical users can explore scenarios
- Changing any parameter instantly recalculates all financial metrics and re-ranks sites
- The KPI strip at the top gives an immediate portfolio overview
- Deployed as a web application accessible from any browser

---

## Slide 17: Dashboard — Map Tab

**Visual:** Screenshot of Tab 1 with Folium map, legend, and LayerControl visible

**5 Toggleable Layers:**
1. **Solar Suitability** (default ON): ML probability heatmap, ~234k points, radius=12, blur=15
2. **NW Sites** (default ON): Circle markers — green (>0.75), orange (0.50–0.75), red (<0.50)
3. **PVGIS Irradiance** (default OFF): Satellite solar resource heatmap (normalised 0–1)
4. **Protected Areas** (default OFF): National parks + AONB polygons in red
5. **Flood Zones** (default OFF): EA Flood Zone 2+3 polygons in blue

**Interactive Features:**
- Click any site marker to see popup with rank, score, type, payback, NPV, capacity, battery value
- LayerControl (top-right) toggles any layer on/off
- Fixed legend with gradient bars and colour explanations

**Speaker Notes:**
- The map provides spatial context that tables cannot — see which sites are near constraints, in high-irradiance zones, or close to grid infrastructure
- Clicking a site marker selects it in the Site Detail tab for deeper analysis
- Overlaying suitability heatmap with PVGIS irradiance shows where ML model agrees with raw solar resource data

---

## Slide 18: Dashboard — Site Detail Tab

**Visual:** Screenshot of Tab 2 showing the 3 Plotly charts and metrics row

**3 Sub-Charts:**

1. **24h Load & Solar Curve**
   - Blue area: site demand profile (kW)
   - Yellow area: self-consumed solar
   - Dashed yellow line: total solar generation
   - Shows the overlap between load and generation

2. **Electricity Price Profile**
   - Bar chart: UK commercial tariff by hour (red = peak, green = off-peak)
   - Star markers: hours with solar generation
   - Shows that solar generates during mid-price hours, not peak evening

3. **25-Year Financial Projection**
   - Green area: cumulative savings (with 0.5%/yr degradation)
   - Red dashed line: CapEx breakeven threshold
   - Orange dashed vertical: payback year marker

**Metrics Row:** Final score, Capacity (kWp), Payback (yr), NPV (£), Battery value (stars)

**Speaker Notes:**
- The load curve shows how well solar generation matches the site's demand pattern
- Self-consumption ratio directly impacts financial returns — higher overlap = more savings
- The financial projection visualises the payback timeline and long-term value
- All charts update dynamically when sidebar parameters change

---

## Slide 19: Dashboard — Rankings Tab

**Visual:** Screenshot of Tab 3 showing styled dataframe with gradients

**Features:**
- Colour gradient on `final_score` (red → yellow → green)
- Colour gradient on `npv_gbp` (light → dark green)
- Battery value displayed as star ratings (★ to ★★★)
- Formatted numbers: £ for currency, decimal places for scores
- **CSV Download** button for exporting the full ranked list

**Columns:**
rank | name | site_type | final_score | geo_score | fin_score | payback_yr | npv_gbp | battery_value | constraint_score

**Speaker Notes:**
- The rankings table is the primary output for decision-makers
- Sites are ranked by composite score, which balances all three dimensions
- CSV export allows further analysis in Excel or integration with NW's internal systems
- The table re-sorts instantly when parameters are changed in the sidebar

---

## Slide 20: Technical Stack

**Two-Column Layout:**

**ML & Data:**
- Python 3.10+
- XGBoost — gradient-boosted classification
- pandas / geopandas — tabular + spatial data
- rasterio — GeoTIFF DEM reading
- xarray / netCDF4 — ERA5 climate data
- scipy — KD-tree, interpolation
- scikit-learn — train/test split, metrics
- pyproj — CRS transformation (BNG ↔ WGS84)

**Visualisation & Dashboard:**
- Streamlit — interactive web UI
- Folium / Leaflet — interactive maps with layer control
- Plotly — responsive charts (load curves, financial projections)
- matplotlib — static plots (ROC, feature importance)

**Infrastructure:**
- YAML config-driven pipeline
- Modular phase architecture (run any phase independently)
- Deployable to Streamlit Community Cloud (free)
- All open-source dependencies

**Speaker Notes:**
- The tech stack was chosen for reproducibility and accessibility — all open-source, no proprietary tools
- XGBoost is industry-standard for tabular prediction tasks
- Streamlit enables rapid deployment of data apps without frontend development
- The modular pipeline can be extended with additional phases (e.g., optimisation, battery sizing)

---

## Slide 21: Key Findings & Recommendations

**Findings:**
- **Wastewater treatment plants** are the strongest candidates: large available area, high self-consumption (85%), stable 24h load profile
- **Pumping stations** benefit most from battery storage: intermittent high-power demand creates charge/discharge opportunities
- Typical payback period: **8–12 years** at current tariffs (£0.15/kWh) and CapEx (£1,000/kWp)
- **Constraint flags** significantly reduce risk: sites in protected areas or flood zones automatically scored lower
- North East England solar yield (~780 kWh/kWp/yr) is viable for self-consumption despite being lower than southern England

**Recommendations:**
1. Prioritise top-ranked wastewater plants for detailed feasibility studies
2. Investigate battery storage at pumping stations (high battery value scores)
3. Use the dashboard for scenario analysis as tariffs and CapEx projections change
4. Commission site-specific surveys for top 10 ranked sites

**Speaker Notes:**
- The system identifies clear patterns in which site types and locations offer the best returns
- Battery storage is particularly valuable where load profiles are mismatched with solar generation
- Scenario analysis via the dashboard helps future-proof decisions against tariff uncertainty

---

## Slide 22: Limitations & Future Work

**Current Limitations:**
- Constraint data is **England-only** (no Scottish/Welsh equivalents included)
- PVGIS grid is sparse (0.25° resolution, ~200 points over UK)
- Load profiles are **estimated templates**, not real NW site data
- Industrial model has small training set (~360 sites)
- No rooftop/building area assessment — assumes ground-mount or available land

**Future Work:**
- Integrate **real NW site electricity consumption data** for accurate self-consumption modelling
- Add **rooftop area estimation** from LiDAR or satellite imagery
- **Battery sizing optimisation** per site (capacity, charge/discharge schedule)
- Extend constraints to **Scotland and Wales** (SNH, NRW datasets)
- **Multi-objective optimisation** (Pareto front: NPV vs constraint risk vs grid impact)
- Cross-validation and hyperparameter tuning for improved model robustness

**Speaker Notes:**
- The most impactful improvement would be integrating real load data from NW's SCADA systems
- Rooftop assessment would unlock additional capacity on existing buildings
- Battery sizing is a natural extension — the load/price curves in the dashboard already show the opportunity
- Multi-objective optimisation would help balance competing priorities more rigorously

---

## Slide 23: Conclusion

**Key Takeaways:**
1. **End-to-end ML pipeline** from raw open data to actionable site rankings
2. **Interactive dashboard** enables stakeholder engagement and real-time scenario analysis
3. **Two-model approach** combines general suitability with industrial self-consumption relevance
4. **Financial integration** provides NPV, payback, and battery value alongside technical scores
5. **All UK public open data** — reproducible, transparent, and licence-compliant
6. **Config-driven architecture** — easily adaptable to other regions or use cases

**Visual:** Summary graphic with pipeline flow + dashboard screenshot

**Speaker Notes:**
- This project demonstrates how open geospatial data and ML can support data-driven infrastructure investment decisions
- The dashboard bridges the gap between technical analysis and stakeholder communication
- The modular pipeline architecture means components can be reused, extended, or adapted for other renewable energy assessments

---

## Slide 24: Q&A

**Title:** Questions & Discussion

**Visual:** Contact information placeholder, project GitHub URL

**Supplementary Materials:**
- GitHub repository with full source code and documentation
- Live dashboard deployment (Streamlit Cloud)
- `data/output/nw_sites_final_ranking.csv` — downloadable ranked site list
- `data/output/suitability_heatmap.html` — standalone interactive map (12 MB, opens in browser)

---

## Appendix A: Composite Score Formula

```
final_score = w_geo × geo_score + w_fin × fin_score + w_const × constraint_score

where:
  geo_score       = mean(repd_model_prob, industrial_model_prob)
  fin_score       = clip(1 - (payback_yr - 5) / 20, 0, 1)
  constraint_score = 1.0 - penalties
    penalties: protected_area (-0.5), SSSI (-0.4), flood_zone (-0.2), grade1_agri (-0.15)

Default weights: w_geo = 0.40, w_fin = 0.40, w_const = 0.20
```

## Appendix B: XGBoost Hyperparameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| objective | binary:logistic | Probability output |
| eval_metric | auc | Optimise for ranking quality |
| max_depth | 6 | Tree complexity |
| learning_rate | 0.1 | Step size shrinkage |
| n_estimators | 300 | Number of boosting rounds |
| subsample | 0.8 | Row sampling per tree |
| colsample_bytree | 0.8 | Feature sampling per tree |
| reg_alpha | 0.1 | L1 regularisation |
| reg_lambda | 1.0 | L2 regularisation |

## Appendix C: Data Source URLs

- REPD: https://www.gov.uk/government/publications/renewable-energy-planning-database-monthly-extract
- OS Terrain 50: https://osdatahub.os.uk/downloads/open/Terrain50
- ERA5: https://cds.climate.copernicus.eu/
- PVGIS: https://re.jrc.ec.europa.eu/pvg_tools/en/
- OSM: https://overpass-turbo.eu/
- Natural England: https://naturalengland-defra.opendata.arcgis.com/
- Environment Agency: https://environment.data.gov.uk/
- DEFRA: https://defra.maps.arcgis.com/
- Ofgem FiT: https://www.ofgem.gov.uk/environmental-and-social-schemes/feed-tariffs-fit
