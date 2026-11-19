"""Synthetic banana and coffee leaves for pipeline tests.

Patterns are deliberately simple (yellowing, purpling, edge scorch, interveinal
stripes) so feature extraction and training can be exercised. Every caller must
label results as dummy data. This is not a substitute for field images.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.common import ensure_dir, load_config

# Inconsistent folder names on purpose, so the alias map is exercised end to end.
FOLDER_NAMES = {
    "banana": {
        "Healthy": "healthy",
        "Nitrogen": "N_deficiency",
        "Phosphorus": "phosphorus",
        "Potassium": "K",
        "Iron": "Fe",
        "Magnesium": "magnesium_def",
    },
    "coffee": {
        "Healthy": "Healthy",
        "Nitrogen": "nitrogen",
        "Phosphorus": "P_deficiency",
        "Potassium": "potassium",
        "Iron": "iron_def",
        "Magnesium": "Mg",
    },
}


def _ellipse_mask(rng: np.random.Generator, size: int) -> tuple[np.ndarray, tuple[int, int]]:
    mask = np.zeros((size, size), dtype=np.uint8)
    center = (
        int(size // 2 + rng.integers(-10, 11)),
        int(size // 2 + rng.integers(-12, 13)),
    )
    axes = (int(rng.integers(78, 108)), int(rng.integers(42, 72)))
    angle = float(rng.integers(0, 180))
    cv2.ellipse(mask, center, axes, angle, 0, 360, 255, thickness=-1)
    return mask, center


def _paint_leaf(class_name: str, mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw one leaf. Colour and pattern depend on the deficiency; shape varies."""
    size = mask.shape[0]
    image = np.zeros((size, size, 3), dtype=np.int16)
    backgrounds = [
        (235, 235, 232),
        (42, 48, 36),
        (190, 196, 188),
    ]
    image[:] = backgrounds[int(rng.integers(0, len(backgrounds)))]

    leaf = mask > 0
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    yy, xx = np.mgrid[0:size, 0:size]

    # BGR base colours. Channels are jittered per image so copies are not identical.
    jitter = int(rng.integers(-12, 13))
    if class_name == "Healthy":
        base = np.array([30 + jitter, 145 + jitter, 40], dtype=np.int16)
    elif class_name == "Nitrogen":
        # Uniform yellowing of the lamina.
        base = np.array([35, 185 + jitter, 205], dtype=np.int16)
    elif class_name == "Phosphorus":
        base = np.array([145, 45 + abs(jitter) // 3, 125], dtype=np.int16)
    elif class_name == "Potassium":
        base = np.array([28, 130 + jitter, 42], dtype=np.int16)
    elif class_name == "Iron":
        base = np.array([32, 150 + jitter, 46], dtype=np.int16)
    else:
        # Magnesium starts green; interveinal tissue is recoloured below.
        base = np.array([30, 140 + jitter, 44], dtype=np.int16)

    image[leaf] = base

    if class_name == "Potassium":
        scorch = leaf & (distance < float(rng.integers(10, 18)))
        image[scorch] = np.array([22, 55, 105], dtype=np.int16)
    elif class_name == "Iron":
        stripes = leaf & (distance > 7) & (np.sin(xx / 5.5 + float(rng.random())) > 0.15)
        image[stripes] = np.array([40, 200, 215], dtype=np.int16)
    elif class_name == "Magnesium":
        side = xx > (size // 2 + int(rng.integers(-15, 16)))
        patches = leaf & (distance > 9) & side
        image[patches] = np.array([40, 195, 210], dtype=np.int16)
    elif class_name == "Nitrogen":
        # Older-leaf yellowing is slightly stronger toward one end of the lamina.
        older = leaf & (yy > size // 2)
        image[older, 1] = np.clip(image[older, 1] + 15, 0, 255)
        image[older, 2] = np.clip(image[older, 2] + 15, 0, 255)

    # Darker veins so texture features are not flat.
    veins = leaf & (np.abs(np.sin(yy / 9.0 + xx / 21.0)) < 0.08) & (distance > 4)
    image[veins] = np.clip(image[veins] - np.array([15, 35, 10]), 0, 255)

    noise = rng.normal(0, 7.0, size=image.shape)
    image = image.astype(np.float32) + noise.astype(np.float32)
    image = np.clip(image, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(image, (3, 3), 0)


def generate_dummy_dataset(cfg: dict) -> Path:
    """Write synthetic images under data/raw. Returns the raw directory.

    Also writes one unmapped folder so the loader's skip path is observable.
    """
    raw_dir = cfg["_root"] / cfg["paths"]["raw_dir"]
    count = int(cfg["dummy"]["images_per_class_per_crop"])
    size = int(cfg["dummy"]["image_size"])
    seed = int(cfg["seed"])
    class_names = list(cfg["classes"])

    for crop_index, crop in enumerate(FOLDER_NAMES):
        for class_index, class_name in enumerate(class_names):
            folder = raw_dir / crop / FOLDER_NAMES[crop][class_name]
            ensure_dir(folder)
            for index in range(count):
                rng = np.random.default_rng(seed + crop_index * 100_000 + class_index * 1_000 + index)
                mask, _center = _ellipse_mask(rng, size)
                image = _paint_leaf(class_name, mask, rng)
                filename = folder / f"{crop}_{class_name.lower()}_{index:03d}.jpg"
                ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                if not ok:
                    raise RuntimeError(f"Could not encode dummy image {filename}")
                encoded.tofile(str(filename))

    # One deliberately unmapped folder. The loader must print it and continue.
    stray = raw_dir / "banana" / "unknown_spot"
    ensure_dir(stray)
    stray_img = np.full((64, 64, 3), 127, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", stray_img)
    if not ok:
        raise RuntimeError("Could not encode unmapped dummy probe image.")
    encoded.tofile(str(stray / "ignore_me.jpg"))

    marker = raw_dir / "DUMMY_MARKER.txt"
    marker.write_text(
        "These images were generated by scripts/make_dummy_data.py.\n"
        "DUMMY DATA - DO NOT REPORT\n"
        "Delete this file and data/raw before adding real leaf photos.\n",
        encoding="utf-8",
    )
    print(
        f"Generated {count} synthetic images x {len(class_names)} classes x "
        f"{len(FOLDER_NAMES)} crops under {raw_dir.as_posix()}."
    )
    print("DUMMY DATA - DO NOT REPORT")
    return raw_dir


def main() -> None:
    generate_dummy_dataset(load_config())


if __name__ == "__main__":
    main()
