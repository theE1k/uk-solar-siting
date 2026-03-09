"""Download the UK Renewable Energy Planning Database (REPD) CSV."""

import re

import requests
from tqdm import tqdm

from src.utils import load_config, resolve_path

REPD_PAGE = "https://www.gov.uk/government/publications/renewable-energy-planning-database-monthly-extract"


def _resolve_repd_url(fallback_url: str) -> str:
    """Scrape the REPD publication page to find the latest CSV download URL."""
    try:
        resp = requests.get(REPD_PAGE, timeout=30)
        resp.raise_for_status()
        # Find all CSV links on the page
        matches = re.findall(r'href="(https://assets\.publishing\.service\.gov\.uk[^"]+\.csv)"', resp.text)
        if matches:
            url = matches[0]
            print(f"[REPD] Found latest URL: {url}")
            return url
    except Exception as e:
        print(f"[REPD] Could not auto-detect URL ({e}), falling back to config URL")
    return fallback_url


def download_repd(config: dict | None = None):
    cfg = config or load_config()
    url = _resolve_repd_url(cfg["labeling"]["repd_url"])
    dest = resolve_path(cfg["paths"]["repd_csv"])
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        print(f"[REPD] Already exists: {dest}")
        return dest

    print(f"[REPD] Downloading from {url} ...")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))

    with open(dest, "wb") as f:
        with tqdm(total=total, unit="B", unit_scale=True, desc="REPD") as pbar:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

    print(f"[REPD] Saved to {dest}")
    return dest


if __name__ == "__main__":
    download_repd()
