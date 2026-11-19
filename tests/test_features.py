"""Feature layout and folder-name alias tests on small synthetic inputs."""

from __future__ import annotations

import numpy as np

from src.common import load_config
from src.data_loader import canonical_class
from src.features import extract_feature_vector, feature_names


def test_feature_names_are_stable_and_unique():
    cfg = load_config()
    first = feature_names(cfg)
    second = feature_names(cfg)
    assert first == second
    assert len(first) == len(set(first))
    assert first[0] == "rgb_r_mean"
    assert "hue_hist_bin_0" in first
    assert "hu_log_1" in first
    assert first[-1] == "circularity"


def test_feature_vector_matches_names_and_has_no_nans():
    cfg = load_config()
    rng = np.random.default_rng(42)
    image = rng.integers(0, 255, size=(48, 48, 3), dtype=np.uint8)
    mask = np.zeros((48, 48), dtype=np.uint8)
    mask[8:40, 10:38] = 255
    first = extract_feature_vector(image, mask, cfg)
    second = extract_feature_vector(image, mask, cfg)
    assert first.shape == (len(feature_names(cfg)),)
    assert np.isfinite(first).all()
    assert np.allclose(first, second)


def test_empty_mask_still_returns_a_finite_vector():
    cfg = load_config()
    image = np.full((32, 32, 3), 80, dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)
    vector = extract_feature_vector(image, mask, cfg)
    assert vector.shape == (len(feature_names(cfg)),)
    assert np.isfinite(vector).all()


def test_folder_aliases_map_to_canonical_classes():
    cfg = load_config()
    expected = {
        "healthy": "Healthy",
        "Healthy": "Healthy",
        "normal": "Healthy",
        "N_deficiency": "Nitrogen",
        "nitrogen": "Nitrogen",
        "phosphorus": "Phosphorus",
        "P_deficiency": "Phosphorus",
        "K": "Potassium",
        "potassium": "Potassium",
        "Fe": "Iron",
        "iron_def": "Iron",
        "magnesium_def": "Magnesium",
        "Mg": "Magnesium",
    }
    for folder, label in expected.items():
        assert canonical_class(folder, cfg) == label
    assert canonical_class("unknown_spot", cfg) is None
    assert canonical_class("random_blight", cfg) is None
