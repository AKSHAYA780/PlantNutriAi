"""Shared configuration, paths, seeding, and image loading."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import yaml

CLASS_ORDER = [
    "Healthy",
    "Nitrogen",
    "Phosphorus",
    "Potassium",
    "Iron",
    "Magnesium",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

ROOT = Path(__file__).resolve().parents[1]

DUMMY_BANNER = (
    "****************************************************************\n"
    "* DUMMY DATA - DO NOT REPORT                                   *\n"
    "* data/raw was empty. Synthetic leaves were generated so the   *\n"
    "* code path can be tested. Metrics are NOT from real images.   *\n"
    "****************************************************************"
)


def load_config(path: Path | None = None) -> dict:
    """Load config.yaml and attach the repository root."""
    cfg_path = path or (ROOT / "config.yaml")
    with cfg_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["_root"] = ROOT
    return cfg


def abs_path(cfg: dict, relative: str | Path) -> Path:
    """Resolve a config-relative path against the repository root."""
    return cfg["_root"] / Path(relative)


def path_from_key(cfg: dict, key: str) -> Path:
    return abs_path(cfg, cfg["paths"][key])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def imread_bgr(path: Path) -> np.ndarray | None:
    """Read an image as BGR. Returns None when the file is missing or corrupt.

    np.fromfile + imdecode keeps Windows paths with non-ASCII characters working,
    which cv2.imread does not always do.
    """
    import cv2

    file_path = Path(path)
    if not file_path.is_file():
        return None
    try:
        data = np.fromfile(str(file_path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        return None
    return image


def write_data_source(cfg: dict, source: str, detail: str) -> None:
    flag = path_from_key(cfg, "data_source_flag")
    ensure_dir(flag.parent)
    flag.write_text(f"{source}\n{detail}\n", encoding="utf-8")


def read_data_source(cfg: dict) -> str:
    flag = path_from_key(cfg, "data_source_flag")
    if not flag.is_file():
        return "unknown"
    return flag.read_text(encoding="utf-8").splitlines()[0].strip() or "unknown"
