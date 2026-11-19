"""Rule-based severity from the share of non-green pixels inside the leaf.

This is not a trained severity model. A pixel inside the leaf mask is marked
affected when it falls outside the healthy-green HSV range in config.yaml.
severity_ratio = affected pixels / leaf pixels. Healthy leaves are always
labelled None, regardless of the ratio.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.common import CLASS_ORDER, ensure_dir, imread_bgr, load_config, path_from_key
from src.preprocessing import preprocess


def severity_from_ratio(ratio: float, predicted_class: str, cfg: dict) -> str:
    """Map a severity ratio to None / Mild / Moderate / Severe using config cuts.

    Cuts are half-open on the left: ratio < mild_max is Mild, ratio <= moderate_max
    is Moderate, and anything larger is Severe. The Healthy class is always None.
    """
    if predicted_class == "Healthy":
        return "None"
    mild_max = float(cfg["severity"]["mild_max"])
    moderate_max = float(cfg["severity"]["moderate_max"])
    if ratio < mild_max:
        return "Mild"
    if ratio <= moderate_max:
        return "Moderate"
    return "Severe"


def affected_mask(image: np.ndarray, mask: np.ndarray, cfg: dict) -> np.ndarray:
    """Boolean mask of leaf pixels outside the healthy-green HSV range."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array(cfg["severity"]["healthy_hsv_lower"], dtype=np.uint8)
    upper = np.array(cfg["severity"]["healthy_hsv_upper"], dtype=np.uint8)
    healthy = cv2.inRange(hsv, lower, upper) > 0
    leaf = mask > 0
    return leaf & ~healthy


def severity_overlay(image: np.ndarray, affected: np.ndarray) -> np.ndarray:
    """Tint affected pixels red so the estimate can be shown beside the photo."""
    overlay = image.copy()
    if not np.any(affected):
        return overlay
    red = np.zeros_like(image)
    red[:] = (40, 40, 220)
    blended = cv2.addWeighted(image, 0.45, red, 0.55, 0)
    overlay[affected] = blended[affected]
    return overlay


def estimate_severity(
    image: np.ndarray,
    mask: np.ndarray,
    predicted_class: str,
    cfg: dict,
) -> tuple[str, float, np.ndarray]:
    """Return (severity label, ratio, overlay). Rule-based; not a learned model."""
    leaf_pixels = int(np.count_nonzero(mask))
    affected = affected_mask(image, mask, cfg)
    if leaf_pixels == 0:
        ratio = 0.0
    else:
        ratio = float(np.count_nonzero(affected)) / float(leaf_pixels)
    label = severity_from_ratio(ratio, predicted_class, cfg)
    return label, ratio, severity_overlay(image, affected)


def save_severity_samples(cfg: dict | None = None, limit: int = 6) -> Path | None:
    """Write a contact sheet of severity overlays, one sample per class when possible."""
    cfg = cfg or load_config()
    split_path = path_from_key(cfg, "split_csv")
    if not split_path.is_file():
        print("No split file; skipping severity samples.")
        return None
    frame = pd.read_csv(split_path)
    chosen: list[pd.Series] = []
    for label in CLASS_ORDER:
        subset = frame[frame["label"] == label]
        if not subset.empty:
            chosen.append(subset.iloc[0])
        if len(chosen) >= limit:
            break
    if len(chosen) < limit:
        for _index, row in frame.iterrows():
            if len(chosen) >= limit:
                break
            if all(row["path"] != existing["path"] for existing in chosen):
                chosen.append(row)
    if not chosen:
        print("No images for severity samples.")
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(10, 6.5))
    for axis in axes.ravel():
        axis.axis("off")
    for axis, row in zip(axes.ravel(), chosen):
        image = imread_bgr(cfg["_root"] / row["path"])
        if image is None:
            continue
        cleaned, mask = preprocess(image, cfg)
        # Samples are titled with the ground-truth class so the sheet shows the
        # rule itself. Inference uses the predicted class instead.
        label, ratio, overlay = estimate_severity(cleaned, mask, str(row["label"]), cfg)
        axis.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axis.set_title(f"{row['label']}: {label} ({ratio:.2f})", fontsize=9)
        axis.axis("off")
    source = "DUMMY DATA - DO NOT REPORT" if _is_dummy(cfg) else "Rule-based severity (not a trained model)"
    figure.suptitle(source, fontsize=12)
    figure.tight_layout()
    destination = path_from_key(cfg, "severity_figure")
    ensure_dir(destination.parent)
    figure.savefig(destination, dpi=140)
    plt.close(figure)
    print(f"Wrote {destination.as_posix()}")
    return destination


def _is_dummy(cfg: dict) -> bool:
    from src.common import read_data_source

    return read_data_source(cfg) == "dummy"


def main() -> None:
    save_severity_samples(load_config())


if __name__ == "__main__":
    main()
