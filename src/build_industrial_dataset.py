#!/usr/bin/env python3
"""
Build training dataset for industrial self-consumption solar model.

Positive samples: Ofgem FiT non-domestic PV installations >= 50kW,
                  No Export or Negotiated Tariff (self-consumption oriented).
Negative samples: OSM industrial/commercial site centroids with min
                  distance from any positive.

Geocoding: UK postcode districts → lat/lon via GeoNames UK dataset.
Output: data/processed/industrial_labeled_sites.csv
"""

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import requests
from scipy.spatial import cKDTree
from tqdm import tqdm

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
PROC = Path(__file__).resolve().parent.parent / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
UK_BBOX = "49.9,-8.2,60.9,1.8"


# ---------------------------------------------------------------------------
# Step 1: Build district postcode → (lat, lon) lookup from GeoNames
# ---------------------------------------------------------------------------

def build_district_centroids() -> dict:
    dest = RAW / "geonames_gb_postcodes.csv"
    if not dest.exists():
        print("[Geocode] Downloading GeoNames UK postcode data …")
        r = requests.get("https://download.geonames.org/export/zip/GB.zip", timeout=60)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            with zf.open("GB.txt") as f:
                df = pd.read_csv(f, sep="\t", header=None,
                                 names=["country", "postcode", "place", "a1_name", "a1",
                                        "a2_name", "a2", "a3_name", "a3", "lat", "lon", "acc"])
        df.to_csv(dest, index=False)
        print(f"[Geocode] Saved {len(df)} postcodes → {dest}")
    else:
        df = pd.read_csv(dest)

    # Extract district (chars before the space in full postcode, e.g. "NE28 6EY" → "NE28")
    df["district"] = df["postcode"].str.split(" ").str[0].str.strip()
    centroids = df.groupby("district")[["lat", "lon"]].mean()
    print(f"[Geocode] Built {len(centroids)} district centroids")
    return centroids.to_dict("index")   # {"NE28": {"lat": ..., "lon": ...}, ...}


# ---------------------------------------------------------------------------
# Step 2: Download & filter FiT data
# ---------------------------------------------------------------------------

def load_fit_positives(district_centroids: dict) -> pd.DataFrame:
    dest = RAW / "fit_industrial_positives.csv"
    if dest.exists():
        print(f"[FiT] Loading cached positives from {dest}")
        return pd.read_csv(dest)

    print("[FiT] Downloading Ofgem FiT installation reports (3 parts) …")
    parts = []
    for i in range(1, 4):
        url = (f"https://www.ofgem.gov.uk/sites/default/files/2025-10/"
               f"Feed-in%20Tariff%20Installation%20Report%20Part%20{i}.xlsx")
        r = requests.get(url, timeout=180)
        r.raise_for_status()
        import openpyxl  # noqa: F401
        xl = pd.ExcelFile(io.BytesIO(r.content))
        df = xl.parse(xl.sheet_names[0], header=4)
        parts.append(df)
        print(f"  Part {i}: {len(df)} rows")

    fit = pd.concat(parts, ignore_index=True)

    # Filter: PV, Non-Domestic, >=50kW, self-consumption export status
    mask = (
        (fit["Technology"] == "Photovoltaic") &
        (fit["Installation Type"].isin(["Non Domestic (Commercial)",
                                         "Non Domestic (Industrial)"])) &
        (fit["Installed capacity"] >= 50) &
        (fit["Export status"].isin(["No Export",
                                     "Export (Negotiated Tariff)",
                                     "No Export (Off-Grid)"]))
    )
    filtered = fit[mask].copy()
    filtered["district"] = filtered["PostCode "].astype(str).str.strip()
    print(f"[FiT] Filtered to {len(filtered)} non-domestic self-consumption PV ≥50kW")

    # Geocode via district centroid
    lats, lons = [], []
    for district in filtered["district"]:
        centroid = district_centroids.get(district)
        if centroid:
            lats.append(centroid["lat"])
            lons.append(centroid["lon"])
        else:
            lats.append(np.nan)
            lons.append(np.nan)

    filtered["latitude"] = lats
    filtered["longitude"] = lons
    filtered = filtered.dropna(subset=["latitude", "longitude"])

    result = filtered[["latitude", "longitude", "Installed capacity",
                        "Export status", "Installation Type",
                        "Installation Country"]].copy()
    result.columns = ["latitude", "longitude", "capacity_kw",
                      "export_status", "install_type", "country"]
    result["label"] = 1
    result = result.drop_duplicates(subset=["latitude", "longitude"])
    result.to_csv(dest, index=False)
    print(f"[FiT] {len(result)} positive samples after geocoding → {dest}")
    return result


# ---------------------------------------------------------------------------
# Step 3: Generate negative samples from OSM industrial sites
# ---------------------------------------------------------------------------

def load_osm_negatives(positives: pd.DataFrame, ratio: float = 1.0,
                       min_dist_km: float = 5.0, seed: int = 42) -> pd.DataFrame:
    dest = RAW / "osm_industrial_sites.shp"

    if not dest.exists():
        print("[OSM] Querying industrial/commercial sites …")
        query = f"""
        [out:json][timeout:300];
        (
          way["landuse"~"industrial|commercial"]({UK_BBOX});
          relation["landuse"~"industrial|commercial"]({UK_BBOX});
        );
        out center;
        """
        resp = requests.post(OVERPASS_URL, data={"data": query}, timeout=600)
        resp.raise_for_status()
        data = resp.json()

        points = []
        for el in data.get("elements", []):
            center = el.get("center", {})
            lat, lon = center.get("lat"), center.get("lon")
            if lat and lon:
                points.append({"latitude": lat, "longitude": lon,
                                "landuse": el.get("tags", {}).get("landuse", "")})
        gdf = gpd.GeoDataFrame(
            points,
            geometry=gpd.points_from_xy([p["longitude"] for p in points],
                                         [p["latitude"] for p in points]),
            crs="EPSG:4326",
        )
        gdf.to_file(dest)
        print(f"[OSM] Saved {len(gdf)} industrial/commercial sites → {dest}")
    else:
        gdf = gpd.read_file(dest)
        print(f"[OSM] Loaded {len(gdf)} industrial/commercial sites")

    candidates = gdf[["latitude", "longitude"]].dropna().copy()

    # Remove candidates too close to any positive
    pos_coords = positives[["latitude", "longitude"]].values
    cand_coords = candidates.values
    tree = cKDTree(pos_coords * np.array([111.0, 111.0]))  # rough km conversion
    dists, _ = tree.query(cand_coords * np.array([111.0, 111.0]))
    candidates = candidates[dists >= min_dist_km].reset_index(drop=True)

    # Sample negatives
    n_neg = int(len(positives) * ratio)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(candidates), size=min(n_neg, len(candidates)), replace=False)
    negatives = candidates.iloc[idx].copy()
    negatives["label"] = 0
    print(f"[OSM] {len(negatives)} negative samples (≥{min_dist_km}km from positives)")
    return negatives


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_industrial_dataset():
    out_path = PROC / "industrial_labeled_sites.csv"
    if out_path.exists():
        print(f"[Dataset] Already exists: {out_path}")
        df = pd.read_csv(out_path)
        print(f"  {len(df)} rows, label distribution: {df['label'].value_counts().to_dict()}")
        return df

    centroids = build_district_centroids()
    positives = load_fit_positives(centroids)
    negatives = load_osm_negatives(positives)

    dataset = pd.concat(
        [positives[["latitude", "longitude", "label"]],
         negatives[["latitude", "longitude", "label"]]],
        ignore_index=True
    ).sample(frac=1, random_state=42).reset_index(drop=True)

    dataset.to_csv(out_path, index=False)
    print(f"\n[Dataset] Saved {len(dataset)} rows → {out_path}")
    print(f"  Positives: {(dataset['label']==1).sum()}, Negatives: {(dataset['label']==0).sum()}")
    return dataset


if __name__ == "__main__":
    build_industrial_dataset()
