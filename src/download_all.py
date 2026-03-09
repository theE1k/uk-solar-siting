#!/usr/bin/env python3
"""
Download all required datasets for the UK Solar Siting project.

Datasets auto-downloaded (no credentials needed):
  1. REPD (UK Renewable Energy Planning Database) — GOV.UK
  2. UK boundary — Natural Earth 10m countries
  3. National Parks + AONB (England) — Natural England ArcGIS Hub
  4. OSM substations (UK) — Overpass API
  5. OSM major roads (UK) — Overpass API
  6. OSM transmission lines 132kV+ — Overpass API
  7. SSSI (England) — Natural England ArcGIS Hub
  8. Flood Zones (England) — Natural England ArcGIS Hub
  9. Agricultural Land Classification — DEFRA / Natural England ArcGIS Hub

Datasets requiring free credentials:
  10. ERA5 solar radiation — Copernicus CDS (see .env.example)

Datasets requiring manual download:
  11. OS Terrain 50 DEM — OS Data Hub (free account)
      https://osdatahub.os.uk/downloads/open/Terrain50
"""

import io
import json
import os
import time
import zipfile
from pathlib import Path

import requests
import geopandas as gpd
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW = PROJECT_ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _download_file(url: str, dest: Path, desc: str = ""):
    if dest.exists():
        print(f"  [skip] Already exists: {dest.name}")
        return
    print(f"  Downloading {desc or dest.name} …")
    resp = requests.get(url, stream=True, timeout=180)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f:
        with tqdm(total=total, unit="B", unit_scale=True, desc=dest.name, leave=False) as bar:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
                bar.update(len(chunk))
    print(f"  Saved: {dest}")


def _load_env():
    """Read .env file from project root into os.environ."""
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        return False
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())
    return True


# ---------------------------------------------------------------------------
# 1. REPD — dynamic URL resolution
# ---------------------------------------------------------------------------

REPD_PAGE = "https://www.gov.uk/government/publications/renewable-energy-planning-database-monthly-extract"
REPD_FALLBACK = "https://assets.publishing.service.gov.uk/media/6985c316d3f57710b50a9b1f/REPD_Publication_Q4_2025.csv"

def _resolve_repd_url() -> str:
    import re
    try:
        resp = requests.get(REPD_PAGE, timeout=30)
        resp.raise_for_status()
        matches = re.findall(
            r'href="(https://assets\.publishing\.service\.gov\.uk[^"]+\.csv)"',
            resp.text,
        )
        if matches:
            url = matches[0]
            print(f"  Auto-detected latest REPD URL: {url}")
            return url
    except Exception as e:
        print(f"  Could not auto-detect REPD URL ({e}), using fallback")
    return REPD_FALLBACK


def download_repd():
    print("\n[1] REPD — Renewable Energy Planning Database")
    url = _resolve_repd_url()
    _download_file(url, RAW / "repd.csv", "REPD")


# ---------------------------------------------------------------------------
# 2. UK boundary (Natural Earth 10m countries → filter UK)
# ---------------------------------------------------------------------------

def download_uk_boundary():
    print("\n[2] UK Boundary — Natural Earth 10m")
    dest = RAW / "uk_boundary.shp"
    if dest.exists():
        print(f"  [skip] Already exists: {dest.name}")
        return

    zip_url = "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip"
    print("  Downloading Natural Earth 10m countries …")
    resp = requests.get(zip_url, timeout=120)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        tmp_dir = RAW / "_ne_tmp"
        tmp_dir.mkdir(exist_ok=True)
        zf.extractall(tmp_dir)

    shp_file = list(tmp_dir.glob("*.shp"))[0]
    world = gpd.read_file(shp_file)
    uk = world[world["ISO_A3"].isin(["GBR"])].copy()
    if uk.empty:
        uk = world[world["ADMIN"].str.contains("United Kingdom", case=False, na=False)].copy()
    uk.to_file(dest)

    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"  Saved: {dest}")


# ---------------------------------------------------------------------------
# 3. Protected Areas — National Parks + AONB (Natural England)
# ---------------------------------------------------------------------------

def download_protected_areas():
    print("\n[3] Protected Areas — National Parks + AONB (England)")
    dest = RAW / "protected_areas.shp"
    if dest.exists():
        print(f"  [skip] Already exists: {dest.name}")
        return

    np_url = (
        "https://services.arcgis.com/JJzESW51TqeY9uat/arcgis/rest/services/"
        "National_Parks_England/FeatureServer/0/query"
        "?where=1%3D1&outFields=*&f=geojson"
    )
    print("  Downloading National Parks (England) …")
    resp_np = requests.get(np_url, timeout=120)
    resp_np.raise_for_status()
    gdf_np = gpd.GeoDataFrame.from_features(resp_np.json()["features"], crs="EPSG:4326")
    gdf_np["designation"] = "National Park"

    aonb_url = (
        "https://services.arcgis.com/JJzESW51TqeY9uat/arcgis/rest/services/"
        "Areas_of_Outstanding_Natural_Beauty_England/FeatureServer/0/query"
        "?where=1%3D1&outFields=*&f=geojson"
    )
    print("  Downloading AONB (England) …")
    resp_aonb = requests.get(aonb_url, timeout=120)
    resp_aonb.raise_for_status()
    gdf_aonb = gpd.GeoDataFrame.from_features(resp_aonb.json()["features"], crs="EPSG:4326")
    gdf_aonb["designation"] = "AONB"

    combined = pd.concat(
        [gdf_np[["geometry", "designation"]], gdf_aonb[["geometry", "designation"]]],
        ignore_index=True,
    )
    combined = gpd.GeoDataFrame(combined, crs="EPSG:4326")
    combined.to_file(dest)
    print(f"  Saved: {dest} ({len(combined)} features)")


# ---------------------------------------------------------------------------
# 4–6. OSM data via Overpass API
# ---------------------------------------------------------------------------

OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]
UK_BBOX = "49.9,-8.2,60.9,1.8"
OVERPASS_RETRIES = 4
OVERPASS_RETRY_DELAY = 30  # seconds between retries


def _overpass_query(query: str, timeout: int = 600) -> dict:
    """POST an Overpass query with automatic retry on 504/connection errors."""
    for attempt in range(1, OVERPASS_RETRIES + 1):
        try:
            resp = requests.post(OVERPASS_URL, data={"data": query}, timeout=timeout)
            if resp.status_code == 504:
                print(f"  Overpass 504 (attempt {attempt}/{OVERPASS_RETRIES}), retrying in {OVERPASS_RETRY_DELAY}s …")
                time.sleep(OVERPASS_RETRY_DELAY)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.ConnectionError as e:
            if attempt < OVERPASS_RETRIES:
                print(f"  Connection error (attempt {attempt}/{OVERPASS_RETRIES}), retrying in {OVERPASS_RETRY_DELAY}s …")
                time.sleep(OVERPASS_RETRY_DELAY)
            else:
                raise
    raise RuntimeError(f"Overpass API failed after {OVERPASS_RETRIES} attempts (504 timeout)")


def _overpass_post(query, timeout=300):
    """Try each Overpass mirror in turn; raise on all failures."""
    import time
    for url in OVERPASS_MIRRORS:
        try:
            resp = requests.post(url, data={"data": query}, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception as e:
            print(f"  [warn] {url} failed: {e} — trying next mirror …")
            time.sleep(3)
    raise RuntimeError("All Overpass mirrors failed. Try again later.")


def download_osm_substations():
    print("\n[4] OSM Substations (UK)")
    dest = RAW / "osm_substations.shp"
    if dest.exists():
        print(f"  [skip] Already exists: {dest.name}")
        return

    query = f"""
    [out:json][timeout:180];
    (
      node["power"="substation"]({UK_BBOX});
      way["power"="substation"]({UK_BBOX});
      relation["power"="substation"]({UK_BBOX});
    );
    out center;
    """
    print("  Querying Overpass API for substations …")
    resp = _overpass_post(query, timeout=300)
    data = resp.json()

    points = []
    for el in data.get("elements", []):
        lat = el.get("lat") or el.get("center", {}).get("lat")
        lon = el.get("lon") or el.get("center", {}).get("lon")
        if lat and lon:
            points.append({"latitude": lat, "longitude": lon,
                           "name": el.get("tags", {}).get("name", "")})

    gdf = gpd.GeoDataFrame(
        points,
        geometry=gpd.points_from_xy([p["longitude"] for p in points],
                                     [p["latitude"] for p in points]),
        crs="EPSG:4326",
    )
    gdf.to_file(dest)
    print(f"  Saved: {dest} ({len(gdf)} substations)")


def download_osm_roads():
    print("\n[5] OSM Major Roads (UK)")
    dest = RAW / "osm_roads.shp"
    if dest.exists():
        print(f"  [skip] Already exists: {dest.name}")
        return

    query = f"""
    [out:json][timeout:300];
    (
      way["highway"~"motorway|trunk|primary"]({UK_BBOX});
    );
    out center;
    """
    print("  Querying Overpass API for major roads …")
    resp = _overpass_post(query, timeout=600)
    data = resp.json()

    points = []
    for el in data.get("elements", []):
        center = el.get("center", {})
        lat, lon = center.get("lat"), center.get("lon")
        if lat and lon:
            points.append({"latitude": lat, "longitude": lon,
                           "highway": el.get("tags", {}).get("highway", "")})

    gdf = gpd.GeoDataFrame(
        points,
        geometry=gpd.points_from_xy([p["longitude"] for p in points],
                                     [p["latitude"] for p in points]),
        crs="EPSG:4326",
    )
    gdf.to_file(dest)
    print(f"  Saved: {dest} ({len(gdf)} road segments)")


def download_osm_transmission():
    print("\n[6] OSM Transmission Lines 132kV+ (UK)")
    dest = RAW / "osm_transmission.shp"
    if dest.exists():
        print(f"  [skip] Already exists: {dest.name}")
        return

    query = f"""
    [out:json][timeout:300];
    (
      way["power"="line"]["voltage"~"132000|275000|400000"]({UK_BBOX});
    );
    out center;
    """
    print("  Querying Overpass API for 132kV+ transmission lines …")

    resp = _overpass_post(query, timeout=600)
    data = resp.json()

    points = []
    for el in data.get("elements", []):
        center = el.get("center", {})
        lat, lon = center.get("lat"), center.get("lon")
        if lat and lon:
            points.append({"latitude": lat, "longitude": lon,
                           "voltage": el.get("tags", {}).get("voltage", "")})

    gdf = gpd.GeoDataFrame(
        points,
        geometry=gpd.points_from_xy([p["longitude"] for p in points],
                                     [p["latitude"] for p in points]),
        crs="EPSG:4326",
    )
    gdf.to_file(dest)
    print(f"  Saved: {dest} ({len(gdf)} transmission line segments)")


# ---------------------------------------------------------------------------
# 7. SSSI (Natural England)
# ---------------------------------------------------------------------------

def download_sssi():
    print("\n[7] SSSI — Sites of Special Scientific Interest (England)")
    dest = RAW / "sssi.shp"
    if dest.exists():
        print(f"  [skip] Already exists: {dest.name}")
        return

    base_url = (
        "https://services.arcgis.com/JJzESW51TqeY9uat/arcgis/rest/services/"
        "SSSI_England/FeatureServer/0/query"
    )
    all_features = []
    offset, page_size = 0, 1000

    while True:
        url = (f"{base_url}?where=1+%3D+1&outFields=NAME"
               f"&f=json&resultOffset={offset}&resultRecordCount={page_size}")
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            print(f"  SSSI error: {data['error']}")
            break
        features = data.get("features", [])
        all_features.extend(features)
        print(f"  SSSI: fetched {len(all_features)} features …", end="\r")
        if len(features) < page_size:
            break
        offset += page_size

    print()
    from shapely.geometry import Polygon, MultiPolygon
    geoms, names = [], []
    for f in all_features:
        rings = f["geometry"]["rings"]
        polys = [Polygon(r) for r in rings]
        geoms.append(MultiPolygon(polys) if len(polys) > 1 else polys[0])
        names.append(f["attributes"].get("NAME", ""))
    gdf = gpd.GeoDataFrame({"SSSI_NAME": names, "geometry": geoms}, crs="EPSG:27700")
    gdf = gdf.to_crs("EPSG:4326")
    gdf.to_file(dest)
    print(f"  Saved: {dest} ({len(gdf)} SSSI polygons)")


# ---------------------------------------------------------------------------
# 8. Flood Zones (Natural England / EA)
# ---------------------------------------------------------------------------

def download_flood_zones():
    print("\n[8] Flood Extents (England) — EA")
    dest = RAW / "flood_zones.shp"
    if dest.exists():
        print(f"  [skip] Already exists: {dest.name}")
        return

    base_url = (
        "https://services.arcgis.com/JJzESW51TqeY9uat/arcgis/rest/services/"
        "FloodExtents_16_07_24_shapefile/FeatureServer/0/query"
    )
    all_features = []
    offset, page_size = 0, 2000

    while True:
        url = (f"{base_url}?where=1+%3D+1&outFields=Q__predica"
               f"&f=json&resultOffset={offset}&resultRecordCount={page_size}")
        resp = requests.get(url, timeout=180)
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            print(f"  Flood zones error: {data['error']}")
            break
        features = data.get("features", [])
        all_features.extend(features)
        print(f"  Flood extents: fetched {len(all_features)} features …", end="\r")
        if len(features) < page_size:
            break
        offset += page_size

    print()
    from shapely.geometry import Polygon, MultiPolygon
    geoms, attrs = [], []
    for f in all_features:
        rings = f["geometry"]["rings"]
        polys = [Polygon(r) for r in rings]
        geoms.append(MultiPolygon(polys) if len(polys) > 1 else polys[0])
        attrs.append(f["attributes"].get("Q__predica", ""))
    gdf = gpd.GeoDataFrame({"flood_type": attrs, "geometry": geoms}, crs="EPSG:27700")
    gdf = gdf.to_crs("EPSG:4326")
    gdf.to_file(dest)
    print(f"  Saved: {dest} ({len(gdf)} flood polygons)")


# ---------------------------------------------------------------------------
# 9. Agricultural Land Classification (DEFRA / Natural England)
# ---------------------------------------------------------------------------

def download_agricultural_land():
    print("\n[9] Agricultural Land Classification (England) — DEFRA")
    dest = RAW / "agricultural_land.shp"
    if dest.exists():
        print(f"  [skip] Already exists: {dest.name}")
        return

    base_url = (
        "https://services.arcgis.com/JJzESW51TqeY9uat/arcgis/rest/services/"
        "Agricultural_Land_Classification_Statutory_Land_Use_England/FeatureServer/0/query"
    )
    all_features = []
    offset, page_size = 0, 1000

    while True:
        url = (f"{base_url}?where=1+%3D+1&outFields=ALC_GRADE"
               f"&f=json&resultOffset={offset}&resultRecordCount={page_size}")
        try:
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  Agricultural Land download failed: {e}")
            print("  Manual download: https://naturalengland-defra.opendata.arcgis.com/")
            print("  Search 'Agricultural Land Classification' → download as Shapefile")
            print(f"  Save as: {dest}")
            return

        if "error" in data:
            print(f"  Agricultural Land API error: {data['error']}")
            print("  Manual download: https://naturalengland-defra.opendata.arcgis.com/")
            return

        features = data.get("features", [])
        all_features.extend(features)
        print(f"  Agricultural Land: fetched {len(all_features)} features …", end="\r")
        if len(features) < page_size:
            break
        offset += page_size

    print()
    from shapely.geometry import Polygon, MultiPolygon
    geoms, grades = [], []
    for f in all_features:
        rings = f["geometry"]["rings"]
        polys = [Polygon(r) for r in rings]
        geoms.append(MultiPolygon(polys) if len(polys) > 1 else polys[0])
        grades.append(f["attributes"].get("ALC_GRADE", ""))
    gdf = gpd.GeoDataFrame({"ALC_GRADE": grades, "geometry": geoms}, crs="EPSG:27700")
    gdf = gdf.to_crs("EPSG:4326")
    gdf.to_file(dest)
    print(f"  Saved: {dest} ({len(gdf)} polygons)")


# ---------------------------------------------------------------------------
# 10. ERA5 — auto-download if .env is configured
# ---------------------------------------------------------------------------

def download_era5_if_configured():
    print("\n[10] ERA5 Solar Radiation — Copernicus CDS")
    dest = RAW / "era5_uk_solar.nc"
    if dest.exists():
        print(f"  [skip] Already exists: {dest.name}")
        return

    _load_env()
    cds_key = os.environ.get("CDS_KEY", "")

    if not cds_key or "<your" in cds_key:
        print("  CDS credentials not configured. To download ERA5:")
        print("  1. Register (free): https://cds.climate.copernicus.eu/")
        print("  2. Get API key:     https://cds.climate.copernicus.eu/how-to-api")
        print("  3. Edit .env:       CDS_KEY=<uid>:<key>")
        print("  4. Re-run:         python src/download_all.py")
        print("     or:             python -m src.download_era5")
        return

    try:
        import cdsapi
    except ImportError:
        print("  cdsapi not installed. Run: pip install cdsapi")
        return

    cds_url = os.environ.get("CDS_URL", "https://cds.climate.copernicus.eu/api")
    print("  Submitting ERA5 request to Copernicus CDS (may take several minutes) …")
    c = cdsapi.Client(url=cds_url, key=cds_key)
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
            "area": [61, -9, 49, 2],
            "format": "netcdf",
        },
        str(dest),
    )
    print(f"  Saved: {dest}")


# ---------------------------------------------------------------------------
# 11. OS Terrain 50 — merge tiles if they exist, otherwise guide
# ---------------------------------------------------------------------------

def setup_terrain():
    print("\n[11] OS Terrain 50 DEM")
    dest = RAW / "os_terrain50.tif"
    if dest.exists():
        print(f"  [skip] Already exists: {dest.name}")
        return

    # Try to merge from existing tiles
    tile_dirs = [RAW / "terrain50_tiles", RAW / "terrain50_asc"]
    asc_files = []
    for d in tile_dirs:
        if d.exists():
            asc_files.extend(list(d.rglob("*.asc")))

    zip_file = RAW / "terr50_gagg_gb.zip"
    if not asc_files and zip_file.exists():
        print("  Extracting terrain tiles from zip …")
        import zipfile as zf
        extract_dir = RAW / "terrain50_tiles"
        extract_dir.mkdir(exist_ok=True)
        with zf.ZipFile(zip_file) as z:
            z.extractall(extract_dir)
        asc_files = list(extract_dir.rglob("*.asc"))

    if asc_files:
        print(f"  Found {len(asc_files)} terrain tiles — merging into {dest.name} …")
        try:
            import rasterio
            from rasterio.merge import merge as rio_merge

            datasets = [rasterio.open(f) for f in asc_files]
            mosaic, transform = rio_merge(datasets)
            profile = datasets[0].profile.copy()
            profile.update(
                driver="GTiff",
                height=mosaic.shape[1],
                width=mosaic.shape[2],
                transform=transform,
                compress="lzw",
            )
            with rasterio.open(dest, "w", **profile) as out:
                out.write(mosaic)
            for ds in datasets:
                ds.close()
            print(f"  Saved merged DEM: {dest}")
        except Exception as e:
            print(f"  Merge failed: {e}")
            print("  Try manually: gdal_merge.py -o data/raw/os_terrain50.tif data/raw/terrain50_tiles/*.asc")
        return

    print("  OS Terrain 50 not found. Manual download required (free):")
    print("  1. Go to: https://osdatahub.os.uk/downloads/open/Terrain50")
    print("  2. Download the full GB dataset (GeoTIFF or ASCII Grid)")
    print(f"  3. Save merged file as: {dest}")
    print("     If you download ASCII tiles, merge them with:")
    print("     gdal_merge.py -o data/raw/os_terrain50.tif data/raw/terrain50_tiles/*.asc")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("UK Solar Siting — Data Downloader")
    print("=" * 70)

    download_repd()
    download_uk_boundary()
    download_protected_areas()
    download_osm_substations()
    download_osm_roads()
    download_osm_transmission()
    download_sssi()
    download_flood_zones()
    download_agricultural_land()
    download_era5_if_configured()
    setup_terrain()

    print("\n" + "=" * 70)
    print("Download complete. Check messages above for any skipped/failed items.")
    print(f"Data directory: {RAW}")
    print("=" * 70)


if __name__ == "__main__":
    main()
