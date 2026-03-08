#!/usr/bin/env python3
"""
Download all required datasets for the UK Solar Siting project.

Datasets that can be auto-downloaded:
  1. REPD (UK Renewable Energy Planning Database) — GOV.UK
  2. UK boundary — Natural Earth 10m countries
  3. National Parks (England) — Natural England ArcGIS Hub
  4. AONB (England) — Natural England ArcGIS Hub
  5. OSM substations (UK) — Overpass API
  6. OSM major roads (UK) — Overpass API

Datasets that need manual download / API key:
  7. ERA5 solar radiation — Copernicus CDS (requires free registration + API key)
  8. OS Terrain 50 DEM — OS Data Hub (requires free registration)
  9. Agricultural Land Classification — DEFRA (manual download)
"""

import io
import json
import zipfile
from pathlib import Path

import requests
import geopandas as gpd
import pandas as pd
from tqdm import tqdm

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
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


def _download_geojson(url: str, dest_shp: Path, desc: str):
    """Download GeoJSON from ArcGIS Hub and save as shapefile."""
    if dest_shp.exists():
        print(f"  [skip] Already exists: {dest_shp.name}")
        return
    print(f"  Downloading {desc} (GeoJSON → Shapefile) …")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    gdf = gpd.GeoDataFrame.from_features(resp.json()["features"], crs="EPSG:4326")
    gdf.to_file(dest_shp)
    print(f"  Saved: {dest_shp}")


# ---------------------------------------------------------------------------
# 1. REPD
# ---------------------------------------------------------------------------

def download_repd():
    print("\n[1/6] REPD — Renewable Energy Planning Database")
    url = "https://assets.publishing.service.gov.uk/media/6985c316d3f57710b50a9b1f/REPD_Publication_Q4_2025.csv"
    _download_file(url, RAW / "repd.csv", "REPD Q4 2025")


# ---------------------------------------------------------------------------
# 2. UK boundary (Natural Earth 10m countries → filter UK)
# ---------------------------------------------------------------------------

def download_uk_boundary():
    print("\n[2/6] UK Boundary — Natural Earth 10m")
    dest = RAW / "uk_boundary.shp"
    if dest.exists():
        print(f"  [skip] Already exists: {dest.name}")
        return

    zip_url = "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip"
    print("  Downloading Natural Earth 10m countries …")
    resp = requests.get(zip_url, timeout=120)
    resp.raise_for_status()

    # Extract in memory, filter to UK, save
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        tmp_dir = RAW / "_ne_tmp"
        tmp_dir.mkdir(exist_ok=True)
        zf.extractall(tmp_dir)

    shp_file = list(tmp_dir.glob("*.shp"))[0]
    world = gpd.read_file(shp_file)
    uk = world[world["ISO_A3"].isin(["GBR"])].copy()

    if uk.empty:
        # fallback: try NAME or ADMIN field
        uk = world[world["ADMIN"].str.contains("United Kingdom", case=False, na=False)].copy()

    uk.to_file(dest)

    # Clean up temp
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"  Saved: {dest}")


# ---------------------------------------------------------------------------
# 3 & 4. Protected Areas — National Parks + AONB (Natural England)
# ---------------------------------------------------------------------------

def download_protected_areas():
    print("\n[3/6] Protected Areas — National Parks + AONB (England)")
    dest = RAW / "protected_areas.shp"
    if dest.exists():
        print(f"  [skip] Already exists: {dest.name}")
        return

    # National Parks (England) — ArcGIS Feature Server
    np_url = ("https://services.arcgis.com/JJzESW51TqeY9uat/arcgis/rest/services/"
              "National_Parks_England/FeatureServer/0/query"
              "?where=1%3D1&outFields=*&f=geojson")
    print("  Downloading National Parks (England) …")
    resp_np = requests.get(np_url, timeout=120)
    resp_np.raise_for_status()
    gdf_np = gpd.GeoDataFrame.from_features(resp_np.json()["features"], crs="EPSG:4326")
    gdf_np["designation"] = "National Park"
    print(f"    National Parks: {len(gdf_np)} features")

    # AONB (England) — ArcGIS Feature Server
    aonb_url = ("https://services.arcgis.com/JJzESW51TqeY9uat/arcgis/rest/services/"
                "Areas_of_Outstanding_Natural_Beauty_England/FeatureServer/0/query"
                "?where=1%3D1&outFields=*&f=geojson")
    print("  Downloading AONB (England) …")
    resp_aonb = requests.get(aonb_url, timeout=120)
    resp_aonb.raise_for_status()
    gdf_aonb = gpd.GeoDataFrame.from_features(resp_aonb.json()["features"], crs="EPSG:4326")
    gdf_aonb["designation"] = "AONB"
    print(f"    AONBs: {len(gdf_aonb)} features")

    # Merge and save
    combined = pd.concat([gdf_np[["geometry", "designation"]], gdf_aonb[["geometry", "designation"]]],
                         ignore_index=True)
    combined = gpd.GeoDataFrame(combined, crs="EPSG:4326")
    combined.to_file(dest)
    print(f"  Saved: {dest}")


# ---------------------------------------------------------------------------
# 5 & 6. OSM data — substations + major roads via Overpass API
# ---------------------------------------------------------------------------

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# UK bounding box for Overpass
UK_BBOX = "49.9,-8.2,60.9,1.8"


def download_osm_substations():
    print("\n[4/6] OSM Substations (UK)")
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
    resp = requests.post(OVERPASS_URL, data={"data": query}, timeout=300)
    resp.raise_for_status()
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
    print("\n[5/6] OSM Major Roads (UK)")
    dest = RAW / "osm_roads.shp"
    if dest.exists():
        print(f"  [skip] Already exists: {dest.name}")
        return

    # motorway + trunk + primary roads only to keep manageable
    query = f"""
    [out:json][timeout:300];
    (
      way["highway"~"motorway|trunk|primary"]({UK_BBOX});
    );
    out center;
    """
    print("  Querying Overpass API for major roads (motorway/trunk/primary) …")
    resp = requests.post(OVERPASS_URL, data={"data": query}, timeout=600)
    resp.raise_for_status()
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
    print("\n[+] OSM Transmission Lines 132kV+ (UK)")
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
    resp = requests.post(OVERPASS_URL, data={"data": query}, timeout=600)
    resp.raise_for_status()
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


def download_sssi():
    print("\n[+] SSSI — Sites of Special Scientific Interest (England)")
    dest = RAW / "sssi.shp"
    if dest.exists():
        print(f"  [skip] Already exists: {dest.name}")
        return

    # Natural England ArcGIS Feature Server — paginate to get all features
    # Note: 'where=1=1' must be passed as a pre-encoded URL parameter
    base_url = ("https://services.arcgis.com/JJzESW51TqeY9uat/arcgis/rest/services/"
                "SSSI_England/FeatureServer/0/query")
    all_features = []
    offset = 0
    page_size = 1000  # server maxRecordCount is 1000

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
        print(f"  SSSI: fetched {len(all_features)} features …")
        if len(features) < page_size:
            break
        offset += page_size

    # Convert Esri JSON (BNG EPSG:27700 rings) to GeoDataFrame then reproject
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


def download_flood_zones():
    print("\n[+] Flood Extents (England) — Natural England / EA")
    dest = RAW / "flood_zones.shp"
    if dest.exists():
        print(f"  [skip] Already exists: {dest.name}")
        return

    # FloodExtents dataset — Natural England ArcGIS (76k features, BNG)
    # Paginate in batches of 2000 (server maxRecordCount)
    base_url = ("https://services.arcgis.com/JJzESW51TqeY9uat/arcgis/rest/services/"
                "FloodExtents_16_07_24_shapefile/FeatureServer/0/query")

    all_features = []
    offset = 0
    page_size = 2000

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
        print(f"  Flood extents: fetched {len(all_features)} features …")
        if len(features) < page_size:
            break
        offset += page_size

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
# Manual download instructions
# ---------------------------------------------------------------------------

def print_manual_instructions():
    print("\n" + "=" * 70)
    print("[6/6] MANUAL DOWNLOADS REQUIRED")
    print("=" * 70)

    print("""
The following datasets require free registration or manual download.
Place the files in: {raw}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

A) ERA5 Solar Radiation (NetCDF)
   → Save as: era5_uk_solar.nc

   1. Register (free) at: https://cds.climate.copernicus.eu/
   2. Get your API key from: https://cds.climate.copernicus.eu/how-to-api
   3. Install: pip install cdsapi
   4. Run the script below or use:
      python -m src.download_era5

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

B) OS Terrain 50 DEM (GeoTIFF / ASCII Grid)
   → Save as: os_terrain50.tif

   1. Go to: https://osdatahub.os.uk/downloads/open/Terrain50
   2. Register (free) for an OS Data Hub account
   3. Download the full GB dataset (ASCII Grid or GeoTIFF)
   4. Merge tiles into a single .tif if needed:
      gdal_merge.py -o os_terrain50.tif data/raw/terr50_tiles/*.asc

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

C) Agricultural Land Classification (Shapefile) [optional]
   → Save as: agricultural_land.shp

   1. Go to: https://naturalengland-defra.opendata.arcgis.com/
   2. Search for "Agricultural Land Classification"
   3. Download Grade 1 & 2 boundaries as Shapefile

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""".format(raw=RAW))


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
    print_manual_instructions()

    print("\nAuto-downloads complete. Check messages above for any errors.")
    print(f"Data directory: {RAW}")


if __name__ == "__main__":
    main()
