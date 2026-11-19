"""Run the saved classical pipeline on one leaf image.

predict_image accepts a filesystem path or a BGR uint8 array (OpenCV order).
Severity is the rule in src/severity.py, not a second trained model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.common import DUMMY_BANNER, ensure_dir, imread_bgr, load_config, path_from_key, read_data_source
from src.features import extract_feature_vector
from src.preprocessing import preprocess
from src.recommend import recommend
from src.severity import estimate_severity


def load_bundle(cfg: dict | None = None) -> dict:
    """Load outputs/models/best_model.joblib or raise a clear missing-file error."""
    cfg = cfg or load_config()
    model_path = path_from_key(cfg, "best_model")
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. "
            "Run `python -m src.train` from the repository root, then try again."
        )
    import joblib

    bundle = joblib.load(model_path)
    if "pipeline" not in bundle or "label_encoder" not in bundle:
        raise RuntimeError(f"Model file {model_path} is missing the pipeline or label encoder.")
    return bundle


def _as_bgr(image: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(image, (str, Path)):
        loaded = imread_bgr(Path(image))
        if loaded is None:
            raise FileNotFoundError(f"Could not read an image from {image}.")
        return loaded
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("predict_image expects a path or a BGR image with shape (H, W, 3).")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def _confidence(pipeline, features: np.ndarray) -> float | None:
    if not hasattr(pipeline, "predict_proba"):
        return None
    probabilities = pipeline.predict_proba(features)
    if probabilities is None or len(probabilities) == 0:
        return None
    return float(np.max(probabilities[0]))


def predict_image(image: str | Path | np.ndarray, cfg: dict | None = None) -> dict:
    """Classify one leaf and attach a rule-based severity and a recommendation.

    Returns predicted_class, confidence, severity, ratio, recommendation, overlay.
    `overlay` is a BGR image with affected pixels tinted.
    """
    cfg = cfg or load_config()
    bundle = load_bundle(cfg)
    bgr = _as_bgr(image)
    cleaned, mask = preprocess(bgr, cfg)
    vector = extract_feature_vector(cleaned, mask, cfg)
    expected = list(bundle.get("feature_names") or [])
    if expected and vector.shape[0] != len(expected):
        raise RuntimeError(
            f"This image produced {vector.shape[0]} features but the model expects {len(expected)}."
        )
    features = vector.reshape(1, -1)
    pipeline = bundle["pipeline"]
    encoded = pipeline.predict(features)
    predicted = str(bundle["label_encoder"].inverse_transform(encoded)[0])
    confidence = _confidence(pipeline, features)
    severity, ratio, overlay = estimate_severity(cleaned, mask, predicted, cfg)
    advice = recommend(predicted, severity, cfg)
    return {
        "predicted_class": predicted,
        "confidence": confidence,
        "severity": severity,
        "ratio": ratio,
        "recommendation": advice,
        "overlay": overlay,
        "mask": mask,
        "cleaned": cleaned,
    }


def _public_result(result: dict) -> dict:
    """Drop arrays so a CLI print stays readable."""
    advice = result["recommendation"]
    return {
        "predicted_class": result["predicted_class"],
        "confidence": result["confidence"],
        "severity": result["severity"],
        "ratio": result["ratio"],
        "recommendation": advice.get("text") if isinstance(advice, dict) else advice,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict nutrient deficiency for one leaf image.")
    parser.add_argument("--image", required=True, help="Path to a leaf image (jpg or png).")
    args = parser.parse_args()
    cfg = load_config()
    source = read_data_source(cfg)
    if source == "dummy":
        print(DUMMY_BANNER)
    result = predict_image(args.image, cfg)
    overlay_path = path_from_key(cfg, "figures_dir") / "last_prediction_overlay.png"
    ensure_dir(overlay_path.parent)
    import cv2

    written = cv2.imencode(".png", result["overlay"])[1]
    written.tofile(str(overlay_path))
    payload = _public_result(result)
    payload["overlay_path"] = overlay_path.as_posix()
    payload["data_source"] = source
    print(json.dumps(payload, indent=2))
    if source == "dummy":
        print(DUMMY_BANNER)


if __name__ == "__main__":
    main()
