#!/usr/bin/env python3
"""
Download Copernicus DEM 90m tiles for the UK from AWS S3 (no auth required)
and create data/raw/os_terrain50.tif in EPSG:27700.
Uses rasterio only — no GDAL CLI required.
"""
from pathlib import Path
from typing import Optional
import numpy as np
import requests
import rasterio
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
DEST = RAW / "os_terrain50.tif"
TILES_DIR = RAW / "copernicus_tiles"

# Copernicus DEM 90m on AWS S3 (public, no auth)
BASE_URL = "https://copernicus-dem-90m.s3.amazonaws.com"

# UK bounding box: lat 49..61N, lon -9..2E
LAT_RANGE = range(49, 61)   # N49 … N60
LON_RANGE = range(-9, 3)    # W09 … E02


def tile_url(lat: int, lon: int) -> str:
    ns = f"N{lat:02d}"
    ew = f"W{abs(lon):03d}" if lon < 0 else f"E{lon:03d}"
    name = f"Copernicus_DSM_COG_30_{ns}_00_{ew}_00_DEM"
    return f"{BASE_URL}/{name}/{name}.tif"


def download_tile(lat: int, lon: int) -> Optional[Path]:
    url = tile_url(lat, lon)
    ns = f"N{lat:02d}"
    ew = f"W{abs(lon):03d}" if lon < 0 else f"E{lon:03d}"
    dest = TILES_DIR / f"cop_{ns}_{ew}.tif"
    if dest.exists():
        return dest
    try:
        resp = requests.get(url, timeout=60)
        if resp.status_code == 404:
            return None  # ocean-only tile
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        return dest
    except Exception as e:
        print(f"  [warn] {ns}{ew}: {e}")
        return None


def build_dem():
    if DEST.exists():
        print(f"[DEM] Already exists: {DEST}")
        return

    TILES_DIR.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)

    print(f"[DEM] Downloading Copernicus DEM 90m tiles for UK …")
    total = len(LAT_RANGE) * len(LON_RANGE)
    tile_paths = []
    for i, lat in enumerate(LAT_RANGE):
        for lon in LON_RANGE:
            p = download_tile(lat, lon)
            if p:
                tile_paths.append(p)
                print(f"  [{len(tile_paths):3d}] N{lat:02d} {'W' if lon<0 else 'E'}{abs(lon):02d} → {p.name}")
        print(f"  Row N{lat:02d} done ({i+1}/{len(LAT_RANGE)})")

    if not tile_paths:
        print("[DEM] ERROR: No tiles downloaded.")
        return

    print(f"\n[DEM] Merging {len(tile_paths)} tiles …")
    datasets = [rasterio.open(p) for p in tile_paths]
    try:
        merged, merged_transform = merge(datasets)
        src_crs = datasets[0].crs
        nodata_val = datasets[0].nodata
    finally:
        for ds in datasets:
            ds.close()

    # Replace nodata with NaN
    if nodata_val is not None:
        merged = merged.astype(np.float32)
        merged[merged == nodata_val] = np.nan

    # Reproject merged raster to EPSG:27700 (British National Grid)
    print("[DEM] Reprojecting to EPSG:27700 …")
    dst_crs = "EPSG:27700"
    h, w = merged.shape[1], merged.shape[2]
    left = merged_transform.c
    top = merged_transform.f
    right = left + merged_transform.a * w
    bottom = top + merged_transform.e * h

    transform, out_w, out_h = calculate_default_transform(
        src_crs, dst_crs, w, h, left=left, top=top, right=right, bottom=bottom
    )

    reprojected = np.full((1, out_h, out_w), np.nan, dtype=np.float32)
    reproject(
        source=merged.astype(np.float32),
        destination=reprojected,
        src_transform=merged_transform,
        src_crs=src_crs,
        dst_transform=transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
        src_nodata=np.nan,
        dst_nodata=-9999.0,
    )
    reprojected[np.isnan(reprojected)] = -9999.0

    with rasterio.open(
        DEST, "w",
        driver="GTiff",
        height=out_h,
        width=out_w,
        count=1,
        dtype=np.float32,
        crs=dst_crs,
        transform=transform,
        nodata=-9999.0,
        compress="lzw",
    ) as out:
        out.write(reprojected)

    size_mb = DEST.stat().st_size / 1e6
    print(f"[DEM] Saved: {DEST} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    build_dem()
