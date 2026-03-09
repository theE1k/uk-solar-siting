#!/usr/bin/env python3
"""
UK Solar Siting — Interactive Dashboard
Northumbrian Water Team
Run: streamlit run dashboard/app.py
"""

import base64
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from streamlit_folium import st_folium

# ── Background image ──────────────────────────────────────────────────────────
_bg_path = Path(__file__).parent / "assets" / "background.png"
_bg_b64 = base64.b64encode(_bg_path.read_bytes()).decode() if _bg_path.exists() else ""

_logo_path = Path(__file__).parent / "assets" / "nw_logo.png"
_logo_b64 = base64.b64encode(_logo_path.read_bytes()).decode() if _logo_path.exists() else ""

_sidebar_logo_path = Path(__file__).parent / "assets" / "nw_sidebar.png"
_sidebar_logo_b64 = base64.b64encode(_sidebar_logo_path.read_bytes()).decode() if _sidebar_logo_path.exists() else ""

_welcome_path = Path(__file__).parent / "assets" / "image_welcome.png"
_welcome_b64 = base64.b64encode(_welcome_path.read_bytes()).decode() if _welcome_path.exists() else ""

# ── Custom CSS ────────────────────────────────────────────────────────────────
_CSS = """
<style>
/* ── Font ── */
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@100;200;300;400;500;600;700&display=swap');
html, body, [class*="css"],
*, *::before, *::after,
p, h1, h2, h3, h4, h5, h6,
div, span, label, button, input, select, textarea,
.stMarkdown, .stText, .stButton, .stSelectbox, .stSlider,
.stDataFrame, .stTable, .stMetric,
[data-testid], [data-baseweb] {
    font-family: 'Montserrat', sans-serif !important;
}

/* ── Layout ── */
.block-container { padding-top: 3.5rem !important; }

/* ── Sidebar subheaders ── */
[data-testid="stSidebar"] h3 {
    font-weight: 400 !important;
    font-family: 'Montserrat', sans-serif !important;
    margin-top: 1.8rem !important;
    padding-top: 1.2rem !important;
    border-top: 1px solid rgba(255,255,255,0.1);
    color: rgba(255,255,255,0.9) !important;
}

/* ── Sidebar text visibility ── */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stSlider [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] .stMultiSelect [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] .stCheckbox label span,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {
    color: rgba(255,255,255,0.85) !important;
}
[data-testid="stSidebar"] .stSlider [data-testid="stThumbValue"],
[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMin"],
[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMax"] {
    color: rgba(255,255,255,0.7) !important;
}
[data-testid="stSidebar"] [data-testid="stMetricValue"] {
    color: #d0b75c !important;
}
[data-testid="stSidebar"] [data-testid="stMetricLabel"] p {
    color: rgba(255,255,255,0.6) !important;
}
[data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] .stCaption p {
    color: rgba(255,255,255,0.5) !important;
}

/* ── Streamlit top navbar transparent ── */
[data-testid="stHeader"] {
    background: transparent !important;
    backdrop-filter: none !important;
}


/* ── Header banner ── */
.nw-header {
    background: transparent;
    padding: 1.2rem 1.8rem;
    border-radius: 12px;
    margin-bottom: 1.4rem;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.8rem;
}
.nw-header img {
    width: 500px;
    height: auto;
    filter: drop-shadow(0 2px 8px rgba(0,0,0,0.5));
}
.nw-header-title {
    font-size: 2.9rem;
    font-weight: 100;
    font-family: 'Montserrat', sans-serif;
    color: #ffffff;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    text-shadow: 0 2px 12px rgba(0,0,0,0.6);
}
.nw-header-subtitle {
    font-size: 1.1rem;
    color: #7aaecf;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ── KPI cards ── */
.kpi-row { display: flex; gap: 1rem; margin-bottom: 1.4rem; }
.kpi-card {
    flex: 1;
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(3px);
    -webkit-backdrop-filter: blur(4px);
    border-radius: 35px;
    padding: 2rem 1.5rem;
    border: 1.5px solid rgba(255, 255, 255, 0.2);
    box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    text-align: center;
}
.kpi-value {
    font-size: 2.2rem;
    font-weight: 200;
    color: #ffffff;
    line-height: 1.15;
    font-family: 'Montserrat', sans-serif;
}
.kpi-label {
    font-size: 0.72rem;
    color: rgba(255,255,255,0.6);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 300;
    margin-top: 0.2rem;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: rgba(10, 26, 46, 0.6);
    backdrop-filter: blur(6px);
    -webkit-backdrop-filter: blur(6px);
    border-radius: 16px;
    padding: 6px;
    border: 1px solid rgba(255,255,255,0.1);
    justify-content: center;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 12px;
    padding: 0.7rem 2rem;
    font-weight: 200;
    font-family: 'Montserrat', sans-serif;
    font-size: 0.95rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.6);
}
.stTabs [aria-selected="true"] {
    background: rgba(208, 183, 92, 0.15) !important;
    color: #d0b75c !important;
    font-weight: 300 !important;
    border: 1px solid rgba(246,201,14,0.4) !important;
}

/* ── Alert/info box ── */
.stAlert { border-radius: 8px !important; }
</style>
"""

# ── Path setup ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_OUT = ROOT / "data" / "output"
DATA_RAW = ROOT / "data" / "raw"
# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NW Solar Siting",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load-profile templates ───────────────────────────────────────────────────
HOURS = np.arange(24)

def _gaussian(peak_h: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((HOURS - peak_h) / sigma) ** 2)

LOAD_PROFILES = {
    "wastewater_plant": (
        0.80 + 0.20 * (_gaussian(6, 2) + _gaussian(19, 2)) / 1.2
    ),
    "pumping_station": np.where(
        (HOURS >= 7) & (HOURS <= 19), 1.0, 0.30
    ).astype(float),
    "water_works": (
        0.40 + 0.60 * (_gaussian(7.5, 1.5) + _gaussian(18, 1.5)) / 1.4
    ),
    "industrial": np.where(
        (HOURS >= 7) & (HOURS <= 18), 1.0, 0.20
    ).astype(float),
    "reservoir_covered": np.where(
        (HOURS >= 6) & (HOURS <= 20), 0.70, 0.35
    ).astype(float),
    "default": np.full(24, 0.60),
}
# Normalise all profiles to [0,1]
for k in LOAD_PROFILES:
    p = LOAD_PROFILES[k]
    LOAD_PROFILES[k] = p / p.max()

# UK commercial half-hour electricity price simulation (£/kWh), summed to hourly
UK_PRICE_HOURLY = np.array([
    0.10, 0.10, 0.09, 0.09, 0.09, 0.10,   # 0–5  off-peak
    0.12, 0.16, 0.22, 0.20, 0.18, 0.17,   # 6–11 morning peak
    0.15, 0.15, 0.14, 0.14, 0.15, 0.20,   # 12–17
    0.24, 0.22, 0.18, 0.15, 0.12, 0.11,   # 18–23 evening peak
])

# ── Financial helpers ────────────────────────────────────────────────────────
SITE_DEFAULTS = {
    "wastewater_plant":  (500, 0.85, 0.45),
    "water_works":       (300, 0.80, 0.50),
    "pumping_station":   (100, 0.60, 0.75),
    "industrial":        (200, 0.70, 0.60),
    "reservoir_covered": (150, 0.65, 0.55),
    "default":           (150, 0.70, 0.55),
}


def _npv(annual_saving: float, capex: float,
         discount_rate: float, project_years: int,
         degradation: float = 0.005) -> float:
    total = 0.0
    for t in range(1, project_years + 1):
        gen_factor = (1 - degradation) ** t
        total += annual_saving * gen_factor / (1 + discount_rate) ** t
    return total - capex


def compute_financials(row: pd.Series, tariff: float, capex_per_kwp: float,
                       project_years: int) -> pd.Series:
    cap_kwp, self_ratio, batt_score = SITE_DEFAULTS.get(
        row["site_type"], SITE_DEFAULTS["default"]
    )
    pvout = row.get("pvout_kwh_kwp", 750)
    if pd.isna(pvout):
        pvout = 750

    annual_gen = cap_kwp * pvout
    annual_saving = annual_gen * self_ratio * tariff
    capex = cap_kwp * capex_per_kwp

    payback = capex / annual_saving if annual_saving > 0 else 99
    npv = _npv(annual_saving, capex, 0.06, project_years)
    fin_score = float(np.clip(1 - (payback - 5) / 20, 0, 1))

    return pd.Series({
        "capacity_kwp":      cap_kwp,
        "annual_gen_kwh":    round(annual_gen),
        "annual_saving_gbp": round(annual_saving),
        "capex_gbp":         capex,
        "payback_yr":        round(payback, 1),
        "npv_gbp":           round(npv),
        "battery_value":     batt_score,
        "fin_score":         round(fin_score, 3),
    })


# ── Data loaders (cached) ────────────────────────────────────────────────────
@st.cache_data
def load_grid_predictions() -> pd.DataFrame:
    df = pd.read_csv(DATA_OUT / "grid_predictions.csv")
    return df[["latitude", "longitude", "probability"]].dropna()


@st.cache_data
def load_pvgis() -> pd.DataFrame:
    df = pd.read_csv(DATA_RAW / "pvgis_grid.csv")
    return df[["lat", "lon", "pvout_kwh_kwp"]].dropna()


HAS_PROTECTED = (DATA_RAW / "protected_areas.shp").exists()
HAS_FLOOD = (DATA_RAW / "flood_zones.shp").exists()


@st.cache_data
def load_protected_areas():
    import geopandas as gpd
    gdf = gpd.read_file(DATA_RAW / "protected_areas.shp")
    gdf = gdf.to_crs("EPSG:4326")
    gdf["geometry"] = gdf["geometry"].simplify(0.001)
    return gdf.__geo_interface__


@st.cache_data
def load_flood_zones():
    import geopandas as gpd
    gdf = gpd.read_file(DATA_RAW / "flood_zones.shp")
    gdf = gdf.to_crs("EPSG:4326")
    gdf["geometry"] = gdf["geometry"].simplify(0.001)
    return gdf.__geo_interface__


@st.cache_data
def load_base_sites() -> pd.DataFrame:
    """Load sites without financial recomputation (cached)."""
    return pd.read_csv(DATA_OUT / "nw_sites_final_ranking.csv")


# ── Re-score sites with updated parameters ──────────────────────────────────
def recompute_sites(tariff: float, capex_per_kwp: float, project_years: int,
                    w_geo: float, w_fin: float, w_const: float) -> pd.DataFrame:
    df = load_base_sites().copy()
    fin = df.apply(
        lambda r: compute_financials(r, tariff, capex_per_kwp, project_years),
        axis=1,
    )
    # Drop old financial cols and replace
    drop_cols = [c for c in ["capacity_kwp", "annual_gen_kwh", "annual_saving_gbp",
                              "capex_gbp", "payback_yr", "npv_gbp", "battery_value",
                              "fin_score"] if c in df.columns]
    df = df.drop(columns=drop_cols)
    df = pd.concat([df, fin], axis=1)

    df["final_score"] = (
        w_geo   * df["geo_score"] +
        w_fin   * df["fin_score"] +
        w_const * df["constraint_score"]
    ).round(3)
    df = df.sort_values("final_score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


# ── Build Folium map ─────────────────────────────────────────────────────────
def build_map(sites_df: pd.DataFrame,
              show_pvgis: bool, show_protected: bool, show_flood: bool) -> folium.Map:
    centre = [54.5, -1.5]
    m = folium.Map(location=centre, zoom_start=8, tiles="CartoDB positron")

    # ── Layer 1: Solar Suitability Heatmap ──
    fg1 = folium.FeatureGroup(name="Solar Suitability", show=True)
    grid = load_grid_predictions()
    heat_data = grid[["latitude", "longitude", "probability"]].values.tolist()
    HeatMap(
        heat_data,
        min_opacity=0.3,
        max_val=1.0,
        radius=12,
        blur=15,
        gradient={
            "0.2": "#2196F3",
            "0.4": "#4CAF50",
            "0.6": "#FFEB3B",
            "0.8": "#FF9800",
            "1.0": "#F44336",
        },
    ).add_to(fg1)
    fg1.add_to(m)

    # ── Layer 2: NW Sites markers ──
    fg2 = folium.FeatureGroup(name="NW Sites", show=True)
    for _, row in sites_df.iterrows():
        score = row["final_score"]
        color = "green" if score > 0.75 else ("orange" if score > 0.50 else "red")
        stars = "★★★" if row["battery_value"] >= 0.7 else ("★★" if row["battery_value"] >= 0.5 else "★")
        popup_html = (
            f"<b>#{int(row['rank'])} {row.get('name') or row['site_type']}</b><br>"
            f"<b>Score:</b> {score:.3f} &nbsp; <b>Type:</b> {row['site_type']}<br>"
            f"<b>Payback:</b> {row['payback_yr']} yrs &nbsp; <b>NPV:</b> £{row['npv_gbp']:,.0f}<br>"
            f"<b>Capacity:</b> {int(row['capacity_kwp'])} kWp<br>"
            f"<b>Annual saving:</b> £{int(row['annual_saving_gbp']):,}<br>"
            f"<b>Battery value:</b> {stars}"
        )
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=8,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=f"#{int(row['rank'])} {row.get('name') or row['site_type']} | {score:.2f}",
        ).add_to(fg2)
    fg2.add_to(m)

    # ── Layer 3: PVGIS Irradiance ──
    fg3 = folium.FeatureGroup(name="PVGIS Irradiance", show=show_pvgis)
    pvgis = load_pvgis()
    # Normalise pvout to 0–1 so HeatMap weights are consistent
    pv_min, pv_max = pvgis["pvout_kwh_kwp"].min(), pvgis["pvout_kwh_kwp"].max()
    pvgis_norm = (pvgis["pvout_kwh_kwp"] - pv_min) / (pv_max - pv_min)
    pvgis_heat = list(zip(pvgis["lat"], pvgis["lon"], pvgis_norm))
    HeatMap(
        pvgis_heat,
        min_opacity=0.3,
        max_val=1.0,
        radius=20,
        blur=25,
        gradient={
            "0.2": "#2196F3",
            "0.4": "#4CAF50",
            "0.6": "#FFEB3B",
            "0.8": "#FF9800",
            "1.0": "#F44336",
        },
    ).add_to(fg3)
    fg3.add_to(m)

    # ── Layer 4: Protected Areas ──
    if HAS_PROTECTED:
        fg4 = folium.FeatureGroup(name="Protected Areas", show=show_protected)
        geojson_pa = load_protected_areas()
        folium.GeoJson(
            geojson_pa,
            style_function=lambda _: {
                "fillColor": "#d9534f", "color": "#d9534f",
                "weight": 1, "fillOpacity": 0.35,
            },
        ).add_to(fg4)
        fg4.add_to(m)

    # ── Layer 5: Flood Zones ──
    if HAS_FLOOD:
        fg5 = folium.FeatureGroup(name="Flood Zones", show=show_flood)
        geojson_fz = load_flood_zones()
        folium.GeoJson(
            geojson_fz,
            style_function=lambda _: {
                "fillColor": "#5bc0de", "color": "#31708f",
                "weight": 1, "fillOpacity": 0.35,
            },
        ).add_to(fg5)
        fg5.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # ── Legend ──
    legend_html = """
    <div style="
        position: fixed;
        bottom: 36px; left: 12px;
        z-index: 9999;
        background: rgba(255,255,255,0.95);
        padding: 12px 15px;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.25);
        font-family: Arial, sans-serif;
        font-size: 12px;
        line-height: 1.7;
        min-width: 210px;
        color: #222;
    ">
      <b style="font-size:13px">Legend</b>

      <div style="margin-top:8px; border-top:1px solid #ddd; padding-top:6px">
        <b>● NW Sites — Final Score</b><br>
        <span style="color:#2ca02c">●</span> &gt; 0.75 &nbsp; High priority<br>
        <span style="color:#ff7f0e">●</span> 0.50–0.75 &nbsp; Medium priority<br>
        <span style="color:#d62728">●</span> &lt; 0.50 &nbsp; Low priority<br>
        <span style="font-size:10px;color:#444">Score = 40% geo + 40% financial + 20% constraints</span>
      </div>

      <div style="margin-top:8px; border-top:1px solid #ddd; padding-top:6px">
        <b>Solar Suitability Heatmap</b><br>
        <span style="display:inline-block;width:130px;height:10px;
          background:linear-gradient(to right,#2196F3,#4CAF50,#FFEB3B,#FF9800,#F44336);
          border-radius:3px;vertical-align:middle"></span><br>
        <span style="color:#222">Low → High &nbsp; ML suitability probability (0–1)</span><br>
        <span style="font-size:10px;color:#444">Combines terrain, grid proximity, land use, constraints</span>
      </div>

      <div style="margin-top:8px; border-top:1px solid #ddd; padding-top:6px">
        <b>PVGIS Irradiance Heatmap</b><br>
        <span style="display:inline-block;width:130px;height:10px;
          background:linear-gradient(to right,#2196F3,#4CAF50,#FFEB3B,#FF9800,#F44336);
          border-radius:3px;vertical-align:middle"></span><br>
        <span style="color:#222">Low → High &nbsp; Annual yield (kWh/kWp, normalised)</span><br>
        <span style="font-size:10px;color:#444">Satellite solar resource data only</span>
      </div>

      <div style="margin-top:8px; border-top:1px solid #ddd; padding-top:6px">
        <span style="display:inline-block;width:14px;height:14px;
          background:rgba(217,83,79,0.35);border:1px solid #d9534f;
          vertical-align:middle;border-radius:2px"></span>
        Protected Areas<br>
        <span style="display:inline-block;width:14px;height:14px;
          background:rgba(91,192,222,0.35);border:1px solid #31708f;
          vertical-align:middle;border-radius:2px"></span>
        Flood Zones
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Inject Montserrat font into the map iframe
    m.get_root().html.add_child(folium.Element("""
    <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@200;300;400;500&display=swap" rel="stylesheet">
    <style>
    * { font-family: 'Montserrat', sans-serif !important; }
    </style>
    """))

    return m


# ── Tab 2: Load & price charts ───────────────────────────────────────────────
def plot_site_detail(row: pd.Series, tariff: float) -> go.Figure:
    site_type = row.get("site_type", "default")
    cap_kwp = row.get("capacity_kwp", 150)
    pvout = row.get("pvout_kwh_kwp", 750)
    if pd.isna(pvout):
        pvout = 750

    load_norm = LOAD_PROFILES.get(site_type, LOAD_PROFILES["default"])
    # Peak demand normalised to capacity (kW)
    peak_kw = cap_kwp * 1.5
    load_kw = load_norm * peak_kw

    # Solar generation: gaussian 9-17, peak = pvout * cap / (8 * 3600s normalised to hrs)
    solar_peak_kw = pvout * cap_kwp / (8 * 365)  # average daily peak kW
    solar_kw = solar_peak_kw * _gaussian(13, 2)

    self_ratio = SITE_DEFAULTS.get(site_type, SITE_DEFAULTS["default"])[1]
    solar_consumed = np.minimum(solar_kw * self_ratio, load_kw)
    solar_exported = solar_kw - solar_consumed
    net_load = np.maximum(load_kw - solar_consumed, 0)

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["24h Load & Solar", "Electricity Price", "25yr Financial Projection"],
        horizontal_spacing=0.12,
    )

    # -- Chart 1: Load curve --
    fig.add_trace(go.Scatter(
        x=HOURS, y=load_kw, name="Demand", line=dict(color="#2c7be5", width=2),
        fill="tozeroy", fillcolor="rgba(44,123,229,0.1)",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=HOURS, y=solar_consumed + solar_exported, name="Solar Total",
        line=dict(color="#d0b75c", width=2, dash="dot"),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=HOURS, y=solar_consumed, name="Self-consumed",
        fill="tozeroy", fillcolor="rgba(246,201,14,0.35)",
        line=dict(color="rgba(0,0,0,0)"),
    ), row=1, col=1)

    # -- Chart 2: Price curve --
    fig.add_trace(go.Bar(
        x=HOURS, y=UK_PRICE_HOURLY, name="Price £/kWh",
        marker_color=np.where(UK_PRICE_HOURLY > 0.19, "#d9534f",
                    np.where(UK_PRICE_HOURLY > 0.14, "#f0ad4e", "#5cb85c")),
        showlegend=False,
    ), row=1, col=2)
    # Overlay solar generation hours
    solar_mask = solar_kw > solar_peak_kw * 0.05
    fig.add_trace(go.Scatter(
        x=HOURS[solar_mask], y=UK_PRICE_HOURLY[solar_mask],
        mode="markers", name="Solar hours",
        marker=dict(color="#d0b75c", size=10, symbol="star"),
    ), row=1, col=2)

    # -- Chart 3: Financial projection --
    capex = row.get("capex_gbp", cap_kwp * 1000)
    annual_saving = row.get("annual_saving_gbp", 0)
    years = np.arange(0, 26)
    cumulative = np.array([
        sum(annual_saving * (1 - 0.005) ** t for t in range(1, y + 1))
        for y in years
    ])
    fig.add_trace(go.Scatter(
        x=years, y=cumulative, name="Cumulative savings",
        line=dict(color="#5cb85c", width=2), fill="tozeroy",
        fillcolor="rgba(92,184,92,0.15)",
    ), row=1, col=3)
    fig.add_hline(y=capex, line_dash="dash", line_color="#d9534f",
                  annotation_text=f"CapEx £{capex:,.0f}", row=1, col=3)
    # Payback point
    payback_yr = row.get("payback_yr", 99)
    if payback_yr < 25:
        idx = int(np.searchsorted(years, payback_yr))
        fig.add_vline(x=payback_yr, line_dash="dot", line_color="#f0ad4e",
                      annotation_text=f"Payback {payback_yr}yr", row=1, col=3)

    fig.update_layout(
        height=460,
        margin=dict(l=40, r=40, t=110, b=80),
        legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center", font=dict(color="#ffffff")),
        paper_bgcolor="rgba(10, 26, 46, 0.55)",
        plot_bgcolor="rgba(10, 26, 46, 0.35)",
        font=dict(color="#ffffff", family="Montserrat, sans-serif"),
    )
    fig.update_xaxes(title_text="Hour", row=1, col=1, title_font=dict(color="#ffffff"), tickfont=dict(color="#cccccc"))
    fig.update_xaxes(title_text="Hour", row=1, col=2, title_font=dict(color="#ffffff"), tickfont=dict(color="#cccccc"))
    fig.update_xaxes(title_text="Year", row=1, col=3, title_font=dict(color="#ffffff"), tickfont=dict(color="#cccccc"))
    fig.update_yaxes(title_text="kW", row=1, col=1, title_font=dict(color="#ffffff"), tickfont=dict(color="#cccccc"))
    fig.update_yaxes(title_text="£/kWh", row=1, col=2, title_font=dict(color="#ffffff"), tickfont=dict(color="#cccccc"))
    fig.update_yaxes(title_text="£", row=1, col=3, title_font=dict(color="#ffffff"), tickfont=dict(color="#cccccc"))

    # Push only subplot titles (those with yref="paper" and y close to 1.0) higher
    subplot_title_texts = ["24h Load & Solar", "Electricity Price", "25yr Financial Projection"]
    for annotation in fig.layout.annotations:
        if annotation.text in subplot_title_texts:
            annotation.y += 0.12

    return fig


# ── Splash screen ────────────────────────────────────────────────────────────
def splash():
    st.markdown(_CSS, unsafe_allow_html=True)

    st.markdown(f"""
    <style>
    .stApp {{ background: #000 !important; }}
    [data-testid="stHeader"] {{ display: none !important; }}
    [data-testid="stSidebar"] {{ display: none !important; }}
    .block-container {{ padding: 0 !important; margin: 0 !important; max-width: 100% !important; }}
    </style>
    <div style="position:fixed; inset:0; z-index:0;">
        <img src="data:image/png;base64,{_welcome_b64}" style="width:100%; height:100%; object-fit:cover; filter:brightness(0.8);" />
    </div>
    <div style="position:fixed; bottom:8vh; left:0; right:0; z-index:1; display:flex; justify-content:center;">
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
    div[data-testid="stButton"] > button {
        background: rgba(255, 255, 255, 0.75) !important;
        border: none !important;
        color: #0a1a2e !important;
        font-family: 'Montserrat', sans-serif !important;
        font-weight: 500 !important;
        font-size: 1.3rem !important;
        letter-spacing: 0.15em !important;
        text-transform: uppercase !important;
        border-radius: 50px !important;
        padding: 1.5rem 3.5rem !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
    div[data-testid="stButton"] > button:hover {
        background: rgba(255, 255, 255, 0.15) !important;
        color: #ffffff !important;
        transition: all 0.8s ease !important;
    }
    .splash-btn-wrapper {
        position: fixed;
        bottom: 8vh;
        left: 0; right: 0;
        z-index: 10;
        display: flex;
        justify-content: center;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div style="margin-top: calc(86vh - 60px);"></div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1.59, 1, 1])
    with col2:
        if st.button("Start"):
            st.session_state["started"] = True
            st.rerun()


# ── Main app ─────────────────────────────────────────────────────────────────
def main():
    st.markdown(_CSS, unsafe_allow_html=True)

    if _bg_b64:
        st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{_bg_b64}") !important;
            background-size: cover !important;
            background-position: center !important;
            background-attachment: fixed !important;
        }}
        [data-testid="stSidebar"] {{
            background-image: none !important;
            background-color: rgba(10, 26, 46, 0.15) !important;
            backdrop-filter: blur(10px) !important;
            -webkit-backdrop-filter: blur(10px) !important;
        }}
        .stApp::before {{
            content: "";
            position: fixed;
            inset: 0;
            background: rgba(5, 15, 30, 0.7);
            pointer-events: none;
            z-index: 0;
        }}
        .block-container {{
            position: relative;
            z-index: 1;
        }}
        </style>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="nw-header">
      <img src="data:image/png;base64,{_logo_b64}" alt="Northumbrian Water logo" />
      <div class="nw-header-title">Solar Site Finder</div>
      <div class="nw-header-subtitle">ML · Geospatial · Financial Modelling</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="
            text-align:center;
            padding:0.75rem 1rem;
            margin-bottom:2rem;
            border-radius:30px;
            border:1px solid #1e4a72;
            background:rgba(10, 26, 46, 0.01);
            backdrop-filter:blur(6px);
        ">
          <div style="font-size:1rem; font-weight:300; color:#7aaecf; letter-spacing:0.02em;">
            Adjust parameters to explore scenarios
          </div>
        </div>
        <hr style="border:none; border-top:2px solid #d0b75c; margin:0 0 0.8rem;">
        """, unsafe_allow_html=True)

        st.subheader("Financial")
        tariff = st.slider("Electricity tariff (£/kWh)", 0.08, 0.25, 0.15, 0.01,
                           format="£%.2f")
        capex_per_kwp = st.slider("CapEx (£/kWp)", 600, 1500, 1000, 50,
                                  format="£%d")
        project_years = st.slider("Project life (years)", 15, 30, 25, 1)

        st.subheader("Score weights")
        st.caption("Three weights must sum to 1.0")
        w_geo = st.slider("Geographic weight", 0.0, 1.0, 0.40, 0.05)
        w_fin = st.slider("Financial weight", 0.0, 1.0 - w_geo,
                          min(0.40, 1.0 - w_geo), 0.05)
        w_const = round(1.0 - w_geo - w_fin, 2)
        st.metric("Constraint weight (auto)", f"{w_const:.2f}")

        st.subheader("Filter")
        all_types = ["wastewater_plant", "water_works", "pumping_station",
                     "industrial", "reservoir_covered"]
        site_types = st.multiselect("Site types", all_types, default=all_types)

        st.subheader("Map layers")
        show_pvgis = st.checkbox("PVGIS Irradiance", value=False)
        show_protected = st.checkbox("Protected Areas", value=False) if HAS_PROTECTED else False
        show_flood = st.checkbox("Flood Zones", value=False) if HAS_FLOOD else False

    # ── Recompute with sidebar parameters ────────────────────────────────────
    sites = recompute_sites(tariff, capex_per_kwp, project_years,
                            w_geo, w_fin, max(0.0, w_const))
    if site_types:
        sites = sites[sites["site_type"].isin(site_types)].reset_index(drop=True)
        sites["rank"] = sites.index + 1

    # ── KPI strip ────────────────────────────────────────────────────────────
    high_priority = int((sites["final_score"] > 0.75).sum())
    st.markdown(f"""
    <div class="kpi-row">
      <div class="kpi-card">
        <div class="kpi-value">{len(sites)}</div>
        <div class="kpi-label">Total Sites</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-value">{sites['payback_yr'].mean():.1f} yr</div>
        <div class="kpi-label">Avg Payback</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-value">£{sites['npv_gbp'].mean():,.0f}</div>
        <div class="kpi-label">Avg NPV (25 yr)</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-value">{high_priority}</div>
        <div class="kpi-label">High Priority (&gt;0.75)</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab_map, tab_detail, tab_table = st.tabs(["Map", "Site Detail", "Rankings"])

    # ── TAB 1: Map ───────────────────────────────────────────────────────────
    with tab_map:
        st.markdown("""
        <style>
        [data-testid="stNotificationContentInfo"] { color: rgba(255,255,255,0.45) !important; }
        [data-testid="stNotificationContentInfo"] p { color: rgba(255,255,255,0.45) !important; }
        </style>
        """, unsafe_allow_html=True)
        st.info(
            "Click a site marker to select it in **Site Detail** tab. "
            "Use the layer control (top-right) to toggle map layers."
        )
        m = build_map(sites, show_pvgis, show_protected, show_flood)
        st.markdown('<div style="border-radius:16px; border:2px solid #1e4a72; overflow:hidden; box-shadow:0 4px 20px rgba(0,0,0,0.4);">', unsafe_allow_html=True)
        map_data = st_folium(m, width="100%", height=580, returned_objects=["last_object_clicked"])
        st.markdown('</div>', unsafe_allow_html=True)

        # Store clicked coords in session state for Tab 2
        if map_data and map_data.get("last_object_clicked"):
            click = map_data["last_object_clicked"]
            lat, lon = click.get("lat"), click.get("lng")
            if lat and lon:
                dists = ((sites["latitude"] - lat) ** 2 +
                         (sites["longitude"] - lon) ** 2)
                nearest_idx = dists.idxmin()
                st.session_state["selected_site"] = sites.loc[nearest_idx, "name"] or \
                    sites.loc[nearest_idx, "site_type"]

    # ── TAB 2: Site Detail ───────────────────────────────────────────────────
    with tab_detail:
        # Build display labels
        site_labels = [
            f"#{int(r['rank'])}  {r.get('name') or r['site_type']}  ({r['site_type']})"
            for _, r in sites.iterrows()
        ]
        default_label = site_labels[0] if site_labels else None

        # Try to match clicked site from map
        if "selected_site" in st.session_state:
            sel_name = st.session_state["selected_site"]
            match = [l for l in site_labels if sel_name in l]
            if match:
                default_label = match[0]

        st.markdown("""
        <style>
        [data-testid="stSelectbox"] {
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.15);
            border-radius: 16px;
            padding: 0.8rem 1rem 0.6rem 1rem;
        }
        [data-testid="stSelectbox"] [data-baseweb="select"] > div {
            background: rgba(10, 26, 46, 0.6) !important;
            border: 1px solid rgba(255,255,255,0.2) !important;
            border-radius: 10px !important;
        }
        [data-testid="stSelectbox"] [data-testid="stWidgetLabel"] p {
            font-size: 1.4rem !important;
            font-weight: 300 !important;
            font-family: 'Montserrat', sans-serif !important;
        }
        </style>
        """, unsafe_allow_html=True)
        chosen_label = st.selectbox("Select site", site_labels,
                                    index=site_labels.index(default_label)
                                    if default_label in site_labels else 0)
        chosen_rank = int(chosen_label.split()[0].lstrip("#"))
        row = sites[sites["rank"] == chosen_rank].iloc[0]

        # Info card
        st.markdown("""
        <style>
        [data-testid="stMetricValue"] {
            font-weight: 200 !important;
            font-family: 'Montserrat', sans-serif !important;
            font-size: 2.2rem !important;
        }
        [data-testid="stMetricLabel"] {
            font-weight: 300 !important;
            font-family: 'Montserrat', sans-serif !important;
        }
        </style>
        """, unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Final score", f"{row['final_score']:.3f}")
        c2.metric("Capacity", f"{int(row['capacity_kwp'])} kWp")
        c3.metric("Payback", f"{row['payback_yr']} yr")
        c4.metric("NPV", f"£{row['npv_gbp']:,.0f}")
        stars = "★★★" if row["battery_value"] >= 0.7 else ("★★" if row["battery_value"] >= 0.5 else "★")
        c5.metric("Battery value", stars)

        st.markdown("""
        <style>
        [data-testid="stPlotlyChart"] {
            border-radius: 30px !important;
            overflow: hidden !important;
            border: 1px solid rgba(255,255,255,0.12);
        }
        </style>
        """, unsafe_allow_html=True)
        st.plotly_chart(plot_site_detail(row, tariff), use_container_width=True)

        # Constraint flags
        flags = []
        if row.get("in_protected_area", 0):
            flags.append("🔴 Protected area")
        if row.get("in_sssi", 0):
            flags.append("🔴 SSSI")
        if row.get("in_flood_zone", 0):
            flags.append("🟡 Flood zone")
        if row.get("in_grade1_agri", 0):
            flags.append("🟡 Grade 1 agricultural land")
        if not flags:
            flags.append("✅ No constraints")
        st.write("**Constraints:**", "  |  ".join(flags))

    # ── TAB 3: Rankings table ─────────────────────────────────────────────────
    with tab_table:
        display_cols = ["rank", "name", "site_type", "final_score", "geo_score",
                        "fin_score", "payback_yr", "npv_gbp", "battery_value",
                        "constraint_score"]
        display_cols = [c for c in display_cols if c in sites.columns]
        tbl = sites[display_cols].copy()

        # Battery value as stars
        tbl["battery_value"] = tbl["battery_value"].apply(
            lambda v: "★★★" if v >= 0.7 else ("★★" if v >= 0.5 else "★")
        )

        st.markdown("""
        <style>
        [data-testid="stDataFrame"] > div {
            border-radius: 16px !important;
            overflow: hidden !important;
        }
        </style>
        """, unsafe_allow_html=True)
        st.dataframe(
            tbl.style.background_gradient(subset=["final_score"], cmap="RdYlGn")
                     .background_gradient(subset=["npv_gbp"], cmap="Greens")
                     .format({
                         "final_score": "{:.3f}",
                         "geo_score": "{:.3f}",
                         "fin_score": "{:.3f}",
                         "payback_yr": "{:.1f}",
                         "npv_gbp": "£{:,.0f}",
                         "constraint_score": "{:.2f}",
                     }),
            use_container_width=True,
            height=550,
        )

        csv = tbl.to_csv(index=False).encode()
        st.download_button("Download CSV", csv, "nw_solar_ranking.csv", "text/csv")


if __name__ == "__main__":
    if not st.session_state.get("started", False):
        splash()
    else:
        main()
