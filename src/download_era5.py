#!/usr/bin/env python3
"""
Download ERA5 solar radiation data for the UK via the Copernicus CDS API.

Prerequisites:
  1. Register at https://cds.climate.copernicus.eu/ (free)
  2. Get your API key: https://cds.climate.copernicus.eu/how-to-api
  3. Copy .env.example → .env and fill in your credentials
  4. pip install cdsapi
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW = PROJECT_ROOT / "data" / "raw"


def _load_env():
    """Read .env file from project root into os.environ."""
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        print(f"ERROR: {env_file} not found. Copy .env.example → .env and fill in your credentials.")
        return False
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())
    return True


def download_era5():
    try:
        import cdsapi
    except ImportError:
        print("ERROR: cdsapi not installed. Run: pip install cdsapi")
        return

    if not _load_env():
        return

    cds_url = os.environ.get("CDS_URL", "https://cds.climate.copernicus.eu/api")
    cds_key = os.environ.get("CDS_KEY", "")
    if not cds_key or "<your" in cds_key:
        print("ERROR: CDS_KEY not configured. Edit .env in project root.")
        return

    dest = RAW / "era5_uk_solar.nc"
    if dest.exists():
        print(f"[ERA5] Already exists: {dest}")
        return

    RAW.mkdir(parents=True, exist_ok=True)

    c = cdsapi.Client(url=cds_url, key=cds_key)

    # Request 5 years of monthly means for the UK region
    # Variables: surface solar radiation downwards, 2m temperature, total cloud cover
    print("[ERA5] Submitting request to CDS (this may take several minutes) …")
    c.retrieve(
        "reanalysis-era5-single-levels-monthly-means",
        {
            "product_type": "monthly_averaged_reanalysis",
            "variable": [
                "surface_solar_radiation_downwards",
                "2m_temperature",
                "total_cloud_cover",
            ],
            "year": ["2019", "2020", "2021", "2022", "2023"],
            "month": [f"{m:02d}" for m in range(1, 13)],
            "time": "00:00",
            "area": [61, -9, 49, 2],  # North, West, South, East — UK bounding box
            "format": "netcdf",
        },
        str(dest),
    )
    print(f"[ERA5] Saved: {dest}")


if __name__ == "__main__":
    download_era5()
