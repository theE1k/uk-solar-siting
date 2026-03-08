"""Generate an interactive suitability heatmap using Folium."""

import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap

from src.utils import load_config, resolve_path


def generate_heatmap(config: dict | None = None):
    cfg = config or load_config()
    pred_path = resolve_path(cfg["paths"]["grid_predictions_csv"])

    print(f"[Phase4] Loading predictions from {pred_path} …")
    df = pd.read_csv(pred_path)

    # Centre the map
    center_lat = df["latitude"].mean()
    center_lon = df["longitude"].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles="cartodbpositron")

    # Prepare heatmap data: [lat, lon, weight]
    heat_data = df[["latitude", "longitude", "probability"]].values.tolist()

    HeatMap(
        heat_data,
        min_opacity=0.3,
        max_val=1.0,
        radius=12,
        blur=15,
        gradient={
            "0.2": "#2196F3",    # blue — low suitability
            "0.4": "#4CAF50",    # green
            "0.6": "#FFEB3B",    # yellow
            "0.8": "#FF9800",    # orange
            "1.0": "#F44336",    # red — high suitability
        },
    ).add_to(m)

    # Add a simple legend
    legend_html = """
    <div style="position:fixed; bottom:30px; left:30px; z-index:1000;
                background:white; padding:10px 14px; border-radius:8px;
                box-shadow:0 2px 6px rgba(0,0,0,0.3); font-size:13px;">
        <b>Solar Suitability</b><br>
        <span style="color:#F44336;">&#9632;</span> High (>0.8)<br>
        <span style="color:#FF9800;">&#9632;</span> Good (0.6–0.8)<br>
        <span style="color:#FFEB3B;">&#9632;</span> Moderate (0.4–0.6)<br>
        <span style="color:#4CAF50;">&#9632;</span> Low (0.2–0.4)<br>
        <span style="color:#2196F3;">&#9632;</span> Poor (<0.2)
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    out_path = resolve_path(cfg["paths"]["heatmap_html"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_path))
    print(f"[Phase4] Heatmap saved to {out_path}")

    # Also generate a static matplotlib version
    _save_static_heatmap(df, cfg)

    return m


def _save_static_heatmap(df: pd.DataFrame, cfg: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 12))
    sc = ax.scatter(
        df["longitude"], df["latitude"],
        c=df["probability"], cmap="RdYlGn", s=1, alpha=0.8,
        vmin=0, vmax=1,
    )
    plt.colorbar(sc, ax=ax, label="Suitability Probability", shrink=0.6)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("UK Solar Site Suitability — XGBoost Prediction")
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)

    static_path = resolve_path(cfg["paths"]["output"]) / "suitability_heatmap.png"
    fig.savefig(static_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Phase4] Static heatmap saved to {static_path}")


if __name__ == "__main__":
    generate_heatmap()
