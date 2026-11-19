"""Handcrafted colour, texture, and shape features from the masked leaf.

The vector is flat and named. Names come only from config so tests can check
that the layout stays stable. Texture uses a quantised grey level so GLCM stays
small, and non-finite values are replaced with zero.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from tqdm import tqdm

from src.common import ensure_dir, imread_bgr, load_config, path_from_key, read_data_source
from src.preprocessing import preprocess, save_preprocessing_samples


def _channel_names() -> dict[str, tuple[str, ...]]:
    return {
        "rgb": ("r", "g", "b"),
        "hsv": ("h", "s", "v"),
        "lab": ("l", "a", "b"),
    }


def feature_names(cfg: dict) -> list[str]:
    """Return feature names in the same order as extract_feature_vector."""
    names: list[str] = []
    for space, channels in _channel_names().items():
        for channel in channels:
            for stat in ("mean", "std", "skew"):
                names.append(f"{space}_{channel}_{stat}")
    for index in range(int(cfg["features"]["hue_bins"])):
        names.append(f"hue_hist_bin_{index}")
    for distance in cfg["features"]["glcm_distances"]:
        for prop in cfg["features"]["glcm_properties"]:
            names.append(f"glcm_d{distance}_{prop}")
    lbp_bins = int(cfg["features"]["lbp_points"]) + 2
    for index in range(lbp_bins):
        names.append(f"lbp_uniform_bin_{index}")
    for index in range(1, 8):
        names.append(f"hu_log_{index}")
    names.extend(
        ["area_ratio", "perimeter", "aspect_ratio", "solidity", "extent", "circularity"]
    )
    return names


def _skewness(values: np.ndarray) -> float:
    array = values.astype(np.float64)
    if array.size == 0:
        return 0.0
    standard = array.std()
    if standard < 1e-8:
        return 0.0
    centered = (array - array.mean()) / standard
    return float(np.mean(centered ** 3))


def _colour_features(image: np.ndarray, mask: np.ndarray, cfg: dict) -> np.ndarray:
    leaf = mask > 0
    if not np.any(leaf):
        leaf = np.ones(mask.shape[:2], dtype=bool)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    values: list[float] = []
    for colour_image in (rgb, hsv, lab):
        for channel_index in range(3):
            pixels = colour_image[:, :, channel_index][leaf].astype(np.float64)
            values.append(float(pixels.mean()) if pixels.size else 0.0)
            values.append(float(pixels.std()) if pixels.size else 0.0)
            values.append(_skewness(pixels))
    hue = hsv[:, :, 0][leaf].astype(np.float64)
    bins = int(cfg["features"]["hue_bins"])
    # OpenCV hue stops at 179. The right edge is exclusive in histogram, so use 180.
    histogram, _edges = np.histogram(hue, bins=bins, range=(0, 180))
    histogram = histogram.astype(np.float64)
    total = histogram.sum()
    if total > 0:
        histogram /= total
    values.extend(histogram.tolist())
    return np.asarray(values, dtype=np.float64)


def _texture_features(image: np.ndarray, mask: np.ndarray, cfg: dict) -> np.ndarray:
    feature_cfg = cfg["features"]
    distances = [int(distance) for distance in feature_cfg["glcm_distances"]]
    angles = [np.deg2rad(float(angle)) for angle in feature_cfg["glcm_angles_deg"]]
    properties = list(feature_cfg["glcm_properties"])
    levels = int(feature_cfg["glcm_levels"])
    points = int(feature_cfg["lbp_points"])
    radius = int(feature_cfg["lbp_radius"])
    lbp_bins = points + 2
    zeros = np.zeros(len(distances) * len(properties) + lbp_bins, dtype=np.float64)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    leaf = mask > 0
    if int(leaf.sum()) < 16:
        return zeros
    rows, cols = np.where(leaf)
    row0, row1 = int(rows.min()), int(rows.max())
    col0, col1 = int(cols.min()), int(cols.max())
    crop = gray[row0 : row1 + 1, col0 : col1 + 1]
    crop_mask = mask[row0 : row1 + 1, col0 : col1 + 1]
    if crop.shape[0] < 4 or crop.shape[1] < 4:
        return zeros

    quantised = np.floor(crop.astype(np.float32) / 256.0 * levels).astype(np.uint8)
    quantised = np.clip(quantised, 0, levels - 1)
    # Fill background inside the bounding box with the leaf median so the
    # rectangle edge does not dominate the co-occurrence matrix.
    median = int(np.median(quantised[crop_mask > 0]))
    quantised = quantised.copy()
    quantised[crop_mask == 0] = median

    matrix = graycomatrix(
        quantised,
        distances=distances,
        angles=angles,
        levels=levels,
        symmetric=True,
        normed=True,
    )
    texture: list[float] = []
    for distance_index in range(len(distances)):
        for prop in properties:
            # Average over angles, keep each distance as its own feature.
            property_values = graycoprops(matrix, prop)[distance_index]
            texture.append(float(np.mean(property_values)))

    lbp = local_binary_pattern(gray, P=points, R=radius, method="uniform")
    lbp_values = lbp[leaf]
    histogram, _edges = np.histogram(lbp_values, bins=np.arange(lbp_bins + 1))
    histogram = histogram.astype(np.float64)
    total = histogram.sum()
    if total > 0:
        histogram /= total
    texture.extend(histogram.tolist())
    return np.asarray(texture, dtype=np.float64)


def _log_hu(moments: np.ndarray) -> np.ndarray:
    """Signed log10 of Hu moments. Tiny values become 0 so the log stays finite."""
    logged = np.zeros(7, dtype=np.float64)
    for index, value in enumerate(np.asarray(moments, dtype=np.float64).reshape(-1)[:7]):
        if abs(value) < 1e-30:
            logged[index] = 0.0
        else:
            logged[index] = float(-np.sign(value) * np.log10(abs(value)))
    return logged


def _shape_features(mask: np.ndarray) -> np.ndarray:
    zeros = np.zeros(13, dtype=np.float64)
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return zeros
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 3:
        return zeros
    height, width = mask.shape[:2]
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    _x, _y, box_w, box_h = cv2.boundingRect(contour)
    aspect = float(box_w) / float(box_h) if box_h else 0.0
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / hull_area if hull_area > 1e-6 else 0.0
    extent = area / float(box_w * box_h) if box_w and box_h else 0.0
    circularity = (4.0 * np.pi * area / (perimeter ** 2)) if perimeter > 1e-6 else 0.0
    area_ratio = area / float(height * width) if height and width else 0.0
    hu = _log_hu(cv2.HuMoments(cv2.moments(contour)))
    shape = np.concatenate(
        [hu, np.array([area_ratio, perimeter, aspect, solidity, extent, circularity], dtype=np.float64)]
    )
    return shape


def extract_feature_vector(image: np.ndarray, mask: np.ndarray, cfg: dict) -> np.ndarray:
    """One finite feature vector for a preprocessed image and its leaf mask."""
    vector = np.concatenate(
        [
            _colour_features(image, mask, cfg),
            _texture_features(image, mask, cfg),
            _shape_features(mask),
        ]
    )
    vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
    expected = len(feature_names(cfg))
    if vector.shape != (expected,):
        raise RuntimeError(f"Feature length {vector.shape[0]} does not match {expected} names.")
    return vector


def _cache_is_current(cache: pd.DataFrame, split: pd.DataFrame, names: list[str]) -> bool:
    required = ["path", "crop", "label", "file_mtime_ns", "file_size", *names]
    if any(column not in cache.columns for column in required):
        return False
    if set(cache["path"]) != set(split["path"]):
        return False
    merged = split.merge(cache[["path", "file_mtime_ns", "file_size"]], on="path", how="left")
    if merged[["file_mtime_ns", "file_size"]].isna().any().any():
        return False
    return bool((merged["file_mtime_ns"] == merged["mtime_ns"]).all() and (merged["file_size"] == merged["size"]).all())


def _file_stamp(root: Path, relative: str) -> tuple[int, int]:
    file_path = root / relative
    stat = file_path.stat()
    return int(stat.st_mtime_ns), int(stat.st_size)


def extract_all(cfg: dict | None = None) -> pd.DataFrame:
    """Extract features for every image in the split, reusing a fresh cache."""
    cfg = cfg or load_config()
    split_path = path_from_key(cfg, "split_csv")
    if not split_path.is_file():
        from src.data_loader import prepare_dataset

        prepare_dataset(cfg)
    split = pd.read_csv(split_path)
    stamps = [_file_stamp(cfg["_root"], path) for path in split["path"]]
    split = split.copy()
    split["mtime_ns"] = [stamp[0] for stamp in stamps]
    split["size"] = [stamp[1] for stamp in stamps]

    names = feature_names(cfg)
    cache_path = path_from_key(cfg, "features_csv")
    if cache_path.is_file():
        cache = pd.read_csv(cache_path)
        if _cache_is_current(cache, split, names):
            print(f"Feature cache is up to date ({len(cache)} rows): {cache_path.as_posix()}")
            figure = path_from_key(cfg, "preprocessing_figure")
            if not figure.is_file():
                save_preprocessing_samples(split_path, cfg)
            return cache
        print("Feature cache is stale. Re-extracting.")

    save_preprocessing_samples(split_path, cfg)
    source = read_data_source(cfg)
    rows: list[dict] = []
    for record in tqdm(split.to_dict(orient="records"), desc="Extracting features"):
        image = imread_bgr(cfg["_root"] / record["path"])
        if image is None:
            print(f"WARNING: unreadable during feature extraction, skipped: {record['path']}")
            continue
        cleaned, mask = preprocess(image, cfg)
        vector = extract_feature_vector(cleaned, mask, cfg)
        row = {
            "path": record["path"],
            "crop": record["crop"],
            "label": record["label"],
            "data_source": source,
            "file_mtime_ns": int(record["mtime_ns"]),
            "file_size": int(record["size"]),
        }
        row.update({name: float(value) for name, value in zip(names, vector)})
        rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("Feature extraction produced no rows.")
    ensure_dir(cache_path.parent)
    frame.to_csv(cache_path, index=False)
    print(f"Wrote {cache_path.as_posix()} ({len(frame)} rows, {len(names)} features).")
    if source == "dummy":
        print("DUMMY DATA - DO NOT REPORT")
    return frame


def main() -> None:
    extract_all(load_config())


if __name__ == "__main__":
    main()
