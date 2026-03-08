"""Extract high-resolution solar irradiance from pre-cached PVGIS grid.

Uses scipy.interpolate.griddata on valid (non-NaN) land points so that
coastal sites are handled correctly — no NaN propagation from sea cells.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import griddata

from src.utils import load_config, resolve_path

_PVGIS_COLS = ("ghi_pvgis_kwh_m2_day", "pvout_kwh_kwp")

# Module-level cache: csv_path → (valid_lat, valid_lon, {col: values})
_pvgis_cache: dict = {}


def _load_pvgis_grid(csv_path: str):
    if csv_path in _pvgis_cache:
        return _pvgis_cache[csv_path]

    df = pd.read_csv(csv_path).dropna(subset=list(_PVGIS_COLS))
    lats = df["lat"].values
    lons = df["lon"].values
    data = {col: df[col].values for col in _PVGIS_COLS}
    _pvgis_cache[csv_path] = (lats, lons, data)
    return lats, lons, data


def extract_pvgis_features(sites: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Interpolate PVGIS irradiance/yield for each site from the pre-cached grid."""
    cfg = config or load_config()
    csv_path = str(resolve_path(cfg["paths"]["pvgis_grid_csv"]))

    print(f"[Phase2] Loading PVGIS grid from {csv_path} …")
    grid_lats, grid_lons, grid_data = _load_pvgis_grid(csv_path)

    query_pts = np.column_stack([sites["latitude"].values, sites["longitude"].values])
    grid_pts = np.column_stack([grid_lats, grid_lons])

    result = pd.DataFrame(index=sites.index)
    for col in _PVGIS_COLS:
        result[col] = griddata(grid_pts, grid_data[col], query_pts,
                               method="linear", fill_value=np.nan)
        # Fall back to nearest for any remaining NaN (e.g. outside convex hull)
        nan_mask = result[col].isna()
        if nan_mask.any():
            result.loc[nan_mask, col] = griddata(
                grid_pts, grid_data[col], query_pts[nan_mask],
                method="nearest"
            )

    print(f"[Phase2] PVGIS features extracted: {list(result.columns)}")
    return result


if __name__ == "__main__":
    sample = pd.DataFrame({"latitude": [51.5, 53.4], "longitude": [-0.1, -2.2]})
    print(extract_pvgis_features(sample))
