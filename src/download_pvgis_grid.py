#!/usr/bin/env python3
"""
Download PVGIS irradiance data on a regular 0.25° grid covering the UK.

PVGIS (Photovoltaic Geographical Information System) by EU JRC provides
~1km resolution solar resource data. We pre-cache a sparse grid here and
use bilinear interpolation at runtime to serve any site coordinates.

Grid: 0.25° spacing → ~200 points → ~3 minutes with 8 threads.
Saved to: data/raw/pvgis_grid.csv
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
DEST = RAW / "pvgis_grid.csv"

PVGIS_URL = "https://re.jrc.ec.europa.eu/api/v5_2/PVcalc"

# UK bounding box (matches config.yaml prediction bbox)
LAT_MIN, LAT_MAX = 49.9, 60.9
LON_MIN, LON_MAX = -8.2, 1.8
GRID_STEP = 0.25   # degrees; ~25 km; gives ~200 land + sea points


def _query_pvgis(lat: float, lon: float, retries: int = 3) -> dict:
    params = {
        "lat": round(lat, 4),
        "lon": round(lon, 4),
        "peakpower": 1,
        "loss": 14,
        "outputformat": "json",
        "browser": 0,
    }
    for attempt in range(retries):
        try:
            r = requests.get(PVGIS_URL, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            totals = data["outputs"]["totals"]["fixed"]
            return {
                "lat": lat,
                "lon": lon,
                "ghi_pvgis_kwh_m2_day": totals["H(i)_y"] / 365.0,
                "pvout_kwh_kwp": totals["E_y"],
                "optimal_tilt_deg": data["inputs"]["mounting_system"]["fixed"]["slope"]["value"],
            }
        except Exception as exc:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  WARN: PVGIS failed for ({lat:.2f}, {lon:.2f}): {exc}")
    return {"lat": lat, "lon": lon,
            "ghi_pvgis_kwh_m2_day": float("nan"),
            "pvout_kwh_kwp": float("nan"),
            "optimal_tilt_deg": float("nan")}


def download_pvgis_grid():
    if DEST.exists():
        print(f"[PVGIS] Grid already exists: {DEST} — skipping download.")
        return

    lats = np.arange(LAT_MIN, LAT_MAX + GRID_STEP, GRID_STEP)
    lons = np.arange(LON_MIN, LON_MAX + GRID_STEP, GRID_STEP)
    grid_lat, grid_lon = np.meshgrid(lats, lons, indexing="ij")
    points = list(zip(grid_lat.ravel(), grid_lon.ravel()))

    print(f"[PVGIS] Querying {len(points)} grid points (this takes ~3 minutes) …")

    results = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_query_pvgis, lat, lon): (lat, lon)
                   for lat, lon in points}
        done = 0
        for fut in as_completed(futures):
            results.append(fut.result())
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(points)} …")

    df = pd.DataFrame(results).sort_values(["lat", "lon"]).reset_index(drop=True)
    df.to_csv(DEST, index=False)
    valid = df["ghi_pvgis_kwh_m2_day"].notna().sum()
    print(f"[PVGIS] Saved {len(df)} grid points ({valid} valid) → {DEST}")


if __name__ == "__main__":
    download_pvgis_grid()
