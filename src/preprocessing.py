"""Resize, denoise, and segment the leaf region.

Segmentation combines an HSV tissue mask with an Otsu mask, cleans the result
with an opening and a closing, and keeps the largest contour. If that region
covers less than the configured fraction of the image, the full frame is used
so a difficult leaf is not dropped.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.common import CLASS_ORDER, ensure_dir, imread_bgr, load_config, path_from_key


def resize_image(image: np.ndarray, size: int) -> np.ndarray:
    height, width = image.shape[:2]
    interpolation = cv2.INTER_AREA if max(height, width) > size else cv2.INTER_LINEAR
    return cv2.resize(image, (size, size), interpolation=interpolation)


def denoise(image: np.ndarray, cfg: dict) -> np.ndarray:
    gaussian_ksize = int(cfg["preprocessing"]["gaussian_ksize"])
    median_ksize = int(cfg["preprocessing"]["median_ksize"])
    blurred = cv2.GaussianBlur(image, (gaussian_ksize, gaussian_ksize), 0)
    return cv2.medianBlur(blurred, median_ksize)


def _hsv_tissue_mask(image: np.ndarray, cfg: dict) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array(cfg["preprocessing"]["hsv_lower"], dtype=np.uint8)
    upper = np.array(cfg["preprocessing"]["hsv_upper"], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    for extra_lower, extra_upper in cfg["preprocessing"].get("extra_hsv", []):
        extra = cv2.inRange(
            hsv,
            np.array(extra_lower, dtype=np.uint8),
            np.array(extra_upper, dtype=np.uint8),
        )
        mask = cv2.bitwise_or(mask, extra)
    return mask


def _otsu_aligned(image: np.ndarray, hsv_mask: np.ndarray) -> np.ndarray:
    """Otsu on grayscale, flipped so it agrees with the HSV tissue mask.

    On a white background the leaf is the darker region; on a dark background
    it is the brighter one. The polarity with more overlap with the HSV mask
    is kept. If HSV found nothing, the smaller region is treated as the leaf.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _threshold, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted = cv2.bitwise_not(binary)
    if int(hsv_mask.sum()) == 0:
        return binary if binary.mean() <= inverted.mean() else inverted
    score_binary = np.count_nonzero((binary > 0) & (hsv_mask > 0))
    score_inverted = np.count_nonzero((inverted > 0) & (hsv_mask > 0))
    return binary if score_binary >= score_inverted else inverted


def _largest_contour_mask(binary: np.ndarray, min_area_ratio: float) -> np.ndarray:
    full = np.full(binary.shape[:2], 255, dtype=np.uint8)
    contours, _hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return full
    largest = max(contours, key=cv2.contourArea)
    mask = np.zeros(binary.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [largest], -1, 255, thickness=-1)
    area_ratio = float(np.count_nonzero(mask)) / float(mask.size)
    if area_ratio < min_area_ratio:
        return full
    return mask


def segment_leaf(image: np.ndarray, cfg: dict) -> np.ndarray:
    """Return a uint8 mask (0 or 255) for the leaf."""
    hsv_mask = _hsv_tissue_mask(image, cfg)
    otsu_mask = _otsu_aligned(image, hsv_mask)
    combined = cv2.bitwise_or(hsv_mask, otsu_mask)
    kernel_size = int(cfg["preprocessing"]["morph_kernel"])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    min_ratio = float(cfg["preprocessing"]["min_mask_area_ratio"])
    return _largest_contour_mask(closed, min_ratio)


def preprocess(image: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Resize, denoise, and segment. Returns the cleaned BGR image and mask."""
    size = int(cfg["preprocessing"]["image_size"])
    resized = resize_image(image, size)
    cleaned = denoise(resized, cfg)
    mask = segment_leaf(cleaned, cfg)
    return cleaned, mask


def apply_mask(image: np.ndarray, mask: np.ndarray, background: int = 0) -> np.ndarray:
    """Keep leaf pixels and paint the background a flat value for display."""
    output = image.copy()
    output[mask == 0] = background
    return output


def _draw_pair(original: np.ndarray, cleaned: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    before = original.copy()
    after = apply_mask(cleaned, mask, background=30)
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(after, contours, -1, (0, 220, 255), thickness=2)
    return before, after


def save_preprocessing_samples(split_csv: Path, cfg: dict, per_class: int = 2) -> Path | None:
    """Save a before/after grid with up to two examples of each class."""
    if not split_csv.is_file():
        print(f"No split file at {split_csv}; skipping preprocessing figure.")
        return None
    frame = pd.read_csv(split_csv)
    chosen_rows = []
    for label in CLASS_ORDER:
        subset = frame[frame["label"] == label]
        if subset.empty:
            continue
        chosen_rows.extend(subset.head(per_class).to_dict(orient="records"))
    if not chosen_rows:
        print("No images available for the preprocessing figure.")
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels_present = [label for label in CLASS_ORDER if label in set(frame["label"])]
    figure, axes = plt.subplots(
        len(labels_present),
        per_class * 2,
        figsize=(2.4 * per_class * 2, 2.3 * len(labels_present)),
        squeeze=False,
    )
    for axis in axes.ravel():
        axis.axis("off")

    buckets: dict[str, list[dict]] = {label: [] for label in labels_present}
    for row in chosen_rows:
        buckets[row["label"]].append(row)

    for row_index, label in enumerate(labels_present):
        samples = buckets[label]
        for sample_index in range(per_class):
            column = sample_index * 2
            if sample_index >= len(samples):
                continue
            record = samples[sample_index]
            image = imread_bgr(cfg["_root"] / record["path"])
            if image is None:
                continue
            resized = resize_image(image, int(cfg["preprocessing"]["image_size"]))
            cleaned, mask = preprocess(image, cfg)
            before, after = _draw_pair(resized, cleaned, mask)
            axes[row_index, column].imshow(cv2.cvtColor(before, cv2.COLOR_BGR2RGB))
            axes[row_index, column].set_title(f"{label} before", fontsize=8)
            axes[row_index, column + 1].imshow(cv2.cvtColor(after, cv2.COLOR_BGR2RGB))
            axes[row_index, column + 1].set_title(f"{label} after", fontsize=8)

    figure.suptitle("Preprocessing: resized original and denoised leaf mask", fontsize=12)
    figure.tight_layout()
    destination = path_from_key(cfg, "preprocessing_figure")
    ensure_dir(destination.parent)
    figure.savefig(destination, dpi=140)
    plt.close(figure)
    print(f"Wrote {destination.as_posix()}")
    return destination


def main() -> None:
    cfg = load_config()
    save_preprocessing_samples(path_from_key(cfg, "split_csv"), cfg)


if __name__ == "__main__":
    main()
