#!/usr/bin/env python3
"""
Layer 2: Financial & business scoring for NW candidate sites.

All parameters use industry defaults; replace with NW actuals when available.
Outputs data/output/nw_sites_final_ranking.csv and an interactive HTML map.

Default assumptions
───────────────────
electricity_tariff_gbp_kwh : £0.24   (2026 UK commercial avg; NW likely £0.20–0.27)
capex_per_kwp              : £850    (2026 large commercial 50-100kWp+, incl. connection)
discount_rate              : 0.06    (6% WACC)
project_life_years         : 25
panel_degradation_pct_yr   : 0.005   (0.5%/yr)

System capacity is estimated from site_type because we have no area data.
Self-consumption ratio reflects typical 24-hr load profiles by site type.
Battery value score is a 0-1 heuristic based on load variability.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import folium
from folium.plugins import MarkerCluster

OUT  = Path(__file__).resolve().parent.parent / "data" / "output"
OUT.mkdir(parents=True, exist_ok=True)

# ── Default parameters (replace with NW actuals) ──────────────────────────
TARIFF          = 0.24     # £/kWh avoided (2026 UK commercial average)
CAPEX_PER_KWP   = 850      # £/kWp (2026 large commercial 50-100kWp+)
DISCOUNT_RATE   = 0.06
PROJECT_YEARS   = 25
DEGRADATION     = 0.005    # per year

# Site-type defaults  (capacity_kwp, self_consump_ratio, battery_value_score)
SITE_DEFAULTS = {
    #                          kWp   self-consump   battery
    "wastewater_plant":       (500,   0.85,          0.45),
    "water_works":            (300,   0.80,          0.50),
    "pumping_station":        (100,   0.60,          0.75),
    "industrial":             (200,   0.70,          0.60),
    "reservoir_covered":      (150,   0.65,          0.55),
    "default":                (150,   0.70,          0.55),
}

# Geo score weight, financial score weight, constraint penalty weight
W_GEO   = 0.40
W_FIN   = 0.40
W_CONST = 0.20


def _npv(annual_saving: float, capex: float) -> float:
    """20-yr NPV with degradation and discounting."""
    total = 0.0
    for t in range(1, PROJECT_YEARS + 1):
        gen_factor = (1 - DEGRADATION) ** t
        total += annual_saving * gen_factor / (1 + DISCOUNT_RATE) ** t
    return total - capex


def compute_financials(row: pd.Series) -> pd.Series:
    cap_kwp, self_ratio, batt_score = SITE_DEFAULTS.get(
        row["site_type"], SITE_DEFAULTS["default"]
    )

    pvout = row.get("pvout_kwh_kwp", 750)  # kWh/kWp/yr from PVGIS
    if pd.isna(pvout):
        pvout = 750

    annual_gen    = cap_kwp * pvout                     # kWh/yr total
    annual_saving = annual_gen * self_ratio * TARIFF    # £/yr
    capex         = cap_kwp * CAPEX_PER_KWP             # £

    payback = capex / annual_saving if annual_saving > 0 else 99
    npv     = _npv(annual_saving, capex)

    # Normalise payback to 0-1 score: 5yr→1.0, 25yr→0.0
    fin_score = float(np.clip(1 - (payback - 5) / 20, 0, 1))

    return pd.Series({
        "capacity_kwp":    cap_kwp,
        "annual_gen_kwh":  round(annual_gen),
        "annual_saving_gbp": round(annual_saving),
        "capex_gbp":       capex,
        "payback_yr":      round(payback, 1),
        "npv_gbp":         round(npv),
        "battery_value":   batt_score,
        "fin_score":       round(fin_score, 3),
    })


def constraint_penalty(row: pd.Series) -> float:
    """0 = hard constraint, 1 = no constraint."""
    score = 1.0
    if row.get("in_protected_area", 0):
        score -= 0.5
    if row.get("in_sssi", 0):
        score -= 0.4
    if row.get("in_flood_zone", 0):
        score -= 0.2
    if row.get("in_grade1_agri", 0):
        score -= 0.2
    return max(0.0, score)


def build_final_ranking(scored_path: str = "data/output/nw_sites_scored.csv",
                        features_path: str = "data/processed/nw_site_features.csv") -> pd.DataFrame:

    df = pd.read_csv(scored_path)

    # Merge PVGIS pvout if available
    feat_path = Path(features_path)
    if feat_path.exists():
        feats = pd.read_csv(feat_path)
        df = df.merge(feats[["latitude", "longitude", "pvout_kwh_kwp",
                               "in_protected_area", "in_sssi",
                               "in_flood_zone", "in_grade1_agri"]],
                      on=["latitude", "longitude"], how="left")

    # Financial metrics
    fin = df.apply(compute_financials, axis=1)
    df  = pd.concat([df, fin], axis=1)

    # Constraint penalty
    df["constraint_score"] = df.apply(constraint_penalty, axis=1)

    # Geo score = average of both ML models
    df["geo_score"] = (df["score_repd"] + df["score_industrial"]) / 2

    # Final composite score
    df["final_score"] = (
        W_GEO   * df["geo_score"] +
        W_FIN   * df["fin_score"] +
        W_CONST * df["constraint_score"]
    ).round(3)

    df = df.sort_values("final_score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)

    out_csv = OUT / "nw_sites_final_ranking.csv"
    df.to_csv(out_csv, index=False)
    print(f"[Financial] Saved ranking → {out_csv}")
    return df


def generate_map(df: pd.DataFrame) -> str:
    centre = [df["latitude"].mean(), df["longitude"].mean()]
    m = folium.Map(location=centre, zoom_start=9, tiles="CartoDB positron")

    cluster = MarkerCluster().add_to(m)

    for _, row in df.iterrows():
        score = row["final_score"]
        # Colour: green >0.75, orange 0.5-0.75, red <0.5
        color = "green" if score > 0.75 else ("orange" if score > 0.50 else "red")

        popup_html = f"""
        <b>#{int(row['rank'])} {row.get('name','') or row['site_type']}</b><br>
        <b>Final score:</b> {score:.3f}<br>
        <hr style='margin:4px 0'>
        <b>Geo score:</b> {row['geo_score']:.3f}
        &nbsp;(REPD {row['score_repd']:.2f} / Ind {row['score_industrial']:.2f})<br>
        <b>Financial score:</b> {row['fin_score']:.3f}<br>
        <b>Payback:</b> {row['payback_yr']} yrs &nbsp;
        <b>NPV:</b> £{row['npv_gbp']:,.0f}<br>
        <b>Capacity:</b> {int(row['capacity_kwp'])} kWp &nbsp;
        <b>Annual gen:</b> {int(row['annual_gen_kwh']):,} kWh<br>
        <b>Annual saving:</b> £{int(row['annual_saving_gbp']):,}<br>
        <b>Battery value:</b> {'⭐⭐⭐' if row['battery_value']>=0.7 else ('⭐⭐' if row['battery_value']>=0.5 else '⭐')}<br>
        <b>Constraints:</b>
        {'🔴 Protected area ' if row.get('in_protected_area',0) else ''}
        {'🔴 SSSI ' if row.get('in_sssi',0) else ''}
        {'🟡 Flood zone ' if row.get('in_flood_zone',0) else ''}
        {'✅ Clear' if not any([row.get('in_protected_area',0), row.get('in_sssi',0),
                                row.get('in_flood_zone',0), row.get('in_grade1_agri',0)]) else ''}
        """

        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"#{int(row['rank'])} {row.get('name','') or row['site_type']} | {score:.2f}",
            icon=folium.Icon(color=color, icon="bolt", prefix="fa"),
        ).add_to(cluster)

    # Legend
    legend_html = """
    <div style="position:fixed;bottom:40px;left:40px;z-index:1000;background:white;
                padding:12px 16px;border-radius:8px;box-shadow:2px 2px 6px rgba(0,0,0,.3);
                font-size:13px;line-height:1.8">
      <b>NW Solar Siting — Final Score</b><br>
      <span style="color:green">●</span> &gt;0.75 High priority<br>
      <span style="color:orange">●</span> 0.50–0.75 Medium<br>
      <span style="color:red">●</span> &lt;0.50 Low priority<br>
      <hr style="margin:6px 0">
      <small>Score = 40% geo + 40% financial + 20% constraints<br>
      Financial defaults: £0.24/kWh, £850/kWp CapEx</small>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    out_html = OUT / "nw_siting_final.html"
    m.save(str(out_html))
    print(f"[Financial] Interactive map saved → {out_html}")
    return str(out_html)


def main():
    # Save NW site features (pvout + constraints) from the full feature extraction
    # if not already cached
    feat_path = Path("data/processed/nw_site_features.csv")
    if not feat_path.exists():
        from src.utils import load_config
        from src.phase2_features.extract_pvgis import extract_pvgis_features
        from src.phase2_features.extract_land_constraints import extract_land_constraints

        cfg   = load_config()
        sites = pd.read_csv("data/raw/nw_sites_osm.csv")
        keep  = ["wastewater_plant","water_works","pumping_station",
                 "reservoir_covered","industrial"]
        sites = sites[sites["site_type"].isin(keep)].reset_index(drop=True)

        pvgis = extract_pvgis_features(sites, cfg)
        land  = extract_land_constraints(sites, cfg)
        feats = pd.concat([sites[["latitude","longitude"]], pvgis, land], axis=1)
        feats.to_csv(feat_path, index=False)
        print(f"[Financial] Cached NW features → {feat_path}")

    df = build_final_ranking()
    generate_map(df)

    print("\n=== Top 20 NW Sites ===")
    cols = ["rank","name","site_type","final_score","geo_score",
            "fin_score","payback_yr","npv_gbp","battery_value","constraint_score"]
    print(df[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
