"""Shared utilities — config loading and coordinate helpers."""

from pathlib import Path

import yaml
import numpy as np


ROOT_DIR = Path(__file__).resolve().parent.parent


def load_config(config_path: str | None = None) -> dict:
    if config_path is None:
        config_path = ROOT_DIR / "configs" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def resolve_path(relative: str) -> Path:
    """Resolve a config-relative path against the project root."""
    return ROOT_DIR / relative


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorised haversine distance in kilometres."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))
