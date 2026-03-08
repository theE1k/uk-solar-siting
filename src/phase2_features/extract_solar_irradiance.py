"""Extract solar irradiance and climate features from ERA5 NetCDF."""

import numpy as np
import pandas as pd
import xarray as xr

from src.utils import load_config, resolve_path


def extract_era5_features(sites: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """For each site (lat, lon), extract GHI, temperature, and cloud cover from ERA5.

    ERA5 'ssrd' is surface solar radiation downwards in J/m^2 per time step.
    We convert to annual mean daily GHI in kWh/m^2/day.
    """
    cfg = config or load_config()
    era5_path = resolve_path(cfg["paths"]["era5_nc"])
    variables = cfg["features"]["era5_variables"]

    print(f"[Phase2] Loading ERA5 data from {era5_path} …")
    ds = xr.open_dataset(era5_path)

    results = {}
    for var in variables:
        if var not in ds:
            print(f"  WARNING: variable '{var}' not found in ERA5 dataset, skipping.")
            continue

        # Compute annual mean across the time dimension
        time_dim = "valid_time" if "valid_time" in ds[var].dims else "time"
        annual_mean = ds[var].mean(dim=time_dim)

        # Interpolate to site coordinates using nearest-neighbour
        values = annual_mean.sel(
            latitude=xr.DataArray(sites["latitude"].values, dims="points"),
            longitude=xr.DataArray(sites["longitude"].values, dims="points"),
            method="nearest",
        ).values

        if var == "ssrd":
            # Convert J/m^2/hour → kWh/m^2/day (ERA5 hourly accumulations)
            values = values / 3_600_000 * 24
            results["ghi_kwh_m2_day"] = values
        elif var == "t2m":
            # Kelvin → Celsius
            results["mean_temp_c"] = values - 273.15
        elif var == "tcc":
            # Total cloud cover is already 0–1 fraction
            results["cloud_cover_frac"] = values
        else:
            results[var] = values

    df_features = pd.DataFrame(results, index=sites.index)
    print(f"[Phase2] ERA5 features extracted: {list(df_features.columns)}")
    return df_features


if __name__ == "__main__":
    sample = pd.DataFrame({"latitude": [51.5, 53.4], "longitude": [-0.1, -2.2]})
    print(extract_era5_features(sample))
