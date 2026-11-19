"""Upload a leaf photo and show the classical deficiency pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import load_config, path_from_key, read_data_source
from src.predict import predict_image
from src.preprocessing import preprocess

SEVERITY_COLOUR = {
    "None": "#2e7d32",
    "Mild": "#f9a825",
    "Moderate": "#ef6c00",
    "Severe": "#c62828",
}


def _bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _decode_upload(payload: bytes) -> np.ndarray | None:
    buffer = np.frombuffer(payload, dtype=np.uint8)
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def main() -> None:
    st.set_page_config(page_title="PlantNutriAI", layout="wide")
    cfg = load_config()
    source = read_data_source(cfg)

    st.sidebar.title("PlantNutriAI")
    st.sidebar.markdown(
        "Machine learning-based plant nutrient deficiency detection "
        "with severity estimation and a recommendation."
    )
    st.sidebar.info(
        "Severity is a rule-based estimate from the share of non-green pixels "
        "inside the leaf. It is not a trained severity model."
    )
    st.sidebar.caption("Crops: banana and coffee. Classical features and classifiers only.")
    if source == "dummy":
        st.sidebar.error("DUMMY DATA - DO NOT REPORT. The saved model was fit on synthetic leaves.")

    st.title("Plant nutrient deficiency check")
    st.write(
        "Upload one banana or coffee leaf. The app predicts a deficiency class, "
        "estimates severity from leaf colour, and shows a corrective note."
    )

    model_path = path_from_key(cfg, "best_model")
    if not model_path.is_file():
        st.warning(
            "The trained model file is not here yet "
            f"(`{model_path.as_posix()}`). From the project folder run "
            "`python -m src.train`, then reload this page."
        )

    uploaded = st.file_uploader("Leaf image", type=["jpg", "jpeg", "png", "bmp"])
    if uploaded is None:
        st.caption("Choose a jpg or png of a single leaf.")
        return

    image = _decode_upload(uploaded.getvalue())
    if image is None:
        st.error("That file could not be read as an image. Try a different jpg or png.")
        return

    cleaned, mask = preprocess(image, cfg)
    columns = st.columns(3)
    columns[0].image(_bgr_to_rgb(image), caption="Original", use_container_width=True)
    masked = cleaned.copy()
    masked[mask == 0] = 30
    columns[1].image(_bgr_to_rgb(masked), caption="Leaf mask", use_container_width=True)

    if not model_path.is_file():
        columns[2].info("Severity overlay appears after the model file is saved.")
        return

    try:
        result = predict_image(image, cfg)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return

    columns[2].image(_bgr_to_rgb(result["overlay"]), caption="Severity overlay", use_container_width=True)

    confidence = result["confidence"]
    confidence_text = "n/a" if confidence is None else f"{confidence:.1%}"
    severity = str(result["severity"])
    colour = SEVERITY_COLOUR.get(severity, "#455a64")
    st.subheader(result["predicted_class"])
    st.write(f"Confidence: **{confidence_text}**")
    st.markdown(
        f"Severity: <span style='background:{colour};color:white;padding:2px 8px;"
        f"border-radius:8px'>{severity}</span> "
        f"(affected ratio {float(result['ratio']):.2f})",
        unsafe_allow_html=True,
    )
    advice = result["recommendation"]
    st.markdown("**Recommendation**")
    if isinstance(advice, dict):
        st.write(advice.get("symptoms", ""))
        sources = advice.get("sources") or []
        if sources:
            st.write("Common sources: " + "; ".join(str(item) for item in sources))
        st.write(advice.get("advice", ""))
        st.info(advice.get("disclaimer", ""))
    else:
        st.write(str(advice))


main()
