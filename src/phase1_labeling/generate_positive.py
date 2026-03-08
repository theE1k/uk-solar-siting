"""Extract positive samples (operational solar sites) from REPD."""

import pandas as pd
import numpy as np
from pyproj import Transformer

from src.utils import load_config, resolve_path

# BNG (EPSG:27700) → WGS84 (EPSG:4326)
_bng_to_wgs84 = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)


def generate_positive(config: dict | None = None) -> pd.DataFrame:
    cfg = config or load_config()
    lab = cfg["labeling"]
    repd_path = resolve_path(cfg["paths"]["repd_csv"])

    print("[Phase1] Loading REPD …")
    df = pd.read_csv(repd_path, low_memory=False, encoding="latin-1")

    # Normalise column names: strip whitespace
    df.columns = df.columns.str.strip()

    tech_col = _find_col(df, ["Technology Type", "technology_type"])
    status_col = _find_col(df, ["Development Status", "development_status"])
    cap_col = _find_col(df, ["Installed Capacity (MWelec)", "installed_capacity_mwelec"])
    x_col = _find_col(df, ["X-coordinate", "x-coordinate", "Easting"])
    y_col = _find_col(df, ["Y-coordinate", "y-coordinate", "Northing"])

    # Filter: solar PV, operational / under construction, above minimum capacity
    mask_tech = df[tech_col].str.contains(lab["technology"], case=False, na=False)
    mask_status = df[status_col].isin(lab["status_include"])
    mask_cap = pd.to_numeric(df[cap_col], errors="coerce") >= lab["min_capacity_mw"]

    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    mask_coords = df[x_col].notna() & df[y_col].notna()

    positives = df[mask_tech & mask_status & mask_cap & mask_coords].copy()

    # Convert BNG → WGS84
    lon, lat = _bng_to_wgs84.transform(positives[x_col].values, positives[y_col].values)
    positives["latitude"] = lat
    positives["longitude"] = lon
    positives["label"] = 1

    # Drop any with invalid coords after transform
    positives = positives[positives["latitude"].between(49, 61) & positives["longitude"].between(-9, 2)]

    out = positives[["latitude", "longitude", "label"]].reset_index(drop=True)
    print(f"[Phase1] Positive samples: {len(out)}")
    return out


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    lower_map = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name in df.columns:
            return name
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    raise KeyError(f"Could not find column. Tried: {candidates}. Available: {list(df.columns[:20])}")


if __name__ == "__main__":
    df = generate_positive()
    print(df.head())
