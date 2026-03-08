"""Extract terrain features (elevation, slope, aspect) from DEM GeoTIFF."""

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from scipy.ndimage import sobel

from src.utils import load_config, resolve_path

# WGS84 → BNG (OS Terrain 50 is in EPSG:27700)
_wgs84_to_bng = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)


# Module-level cache so we only read the DEM once per run
_dem_cache: dict = {}


def _load_dem(tif_path):
    if tif_path not in _dem_cache:
        src = rasterio.open(tif_path)
        data = src.read(1).astype(np.float32)
        data[data == src.nodata] = np.nan
        _dem_cache[tif_path] = (src, data)
    return _dem_cache[tif_path]


def extract_terrain_features(sites: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Extract elevation, slope (degrees), and aspect (degrees) for each site."""
    cfg = config or load_config()
    tif_path = str(resolve_path(cfg["paths"]["terrain_tif"]))
    terrain_cfg = cfg["features"]["terrain"]

    print(f"[Phase2] Loading DEM from {tif_path} …")
    src, elevation = _load_dem(tif_path)
    transform = src.transform
    res_m = abs(transform.a)  # pixel resolution in metres

    # Pre-compute slope and aspect grids if needed
    slope_grid, aspect_grid = None, None
    if terrain_cfg.get("compute_slope"):
        dy = sobel(elevation, axis=0) / (8 * res_m)
        dx = sobel(elevation, axis=1) / (8 * res_m)
        slope_grid = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
    south_facing_grid = None
    if terrain_cfg.get("compute_aspect"):
        if slope_grid is None:
            dy = sobel(elevation, axis=0) / (8 * res_m)
            dx = sobel(elevation, axis=1) / (8 * res_m)
        aspect_grid = np.degrees(np.arctan2(-dx, dy))
        aspect_grid[aspect_grid < 0] += 360
    if terrain_cfg.get("compute_south_facing") and aspect_grid is not None:
        # 1 = due south (optimal), 0 = east/west, -1 = due north
        south_facing_grid = np.cos(np.radians(aspect_grid - 180))

    # Convert all coordinates WGS84 → BNG before sampling
    easting, northing = _wgs84_to_bng.transform(
        sites["longitude"].values, sites["latitude"].values
    )

    elevations, slopes, aspects, south_facings = [], [], [], []
    for e, n in zip(easting, northing):
        try:
            py, px = src.index(e, n)
            py = np.clip(py, 0, elevation.shape[0] - 1)
            px = np.clip(px, 0, elevation.shape[1] - 1)
        except Exception:
            elevations.append(np.nan)
            slopes.append(np.nan)
            aspects.append(np.nan)
            south_facings.append(np.nan)
            continue

        elevations.append(elevation[py, px])
        slopes.append(slope_grid[py, px] if slope_grid is not None else np.nan)
        aspects.append(aspect_grid[py, px] if aspect_grid is not None else np.nan)
        south_facings.append(south_facing_grid[py, px] if south_facing_grid is not None else np.nan)

    result = pd.DataFrame({"elevation_m": elevations}, index=sites.index)
    if terrain_cfg.get("compute_slope"):
        result["slope_deg"] = slopes
    if terrain_cfg.get("compute_aspect"):
        result["aspect_deg"] = aspects
    if terrain_cfg.get("compute_south_facing"):
        result["south_facing_score"] = south_facings

    print(f"[Phase2] Terrain features extracted: {list(result.columns)}")
    return result


if __name__ == "__main__":
    sample = pd.DataFrame({"latitude": [51.5, 53.4], "longitude": [-0.1, -2.2]})
    print(extract_terrain_features(sample))
