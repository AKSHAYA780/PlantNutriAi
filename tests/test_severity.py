"""Boundary checks for the rule-based severity estimate."""

from __future__ import annotations

import numpy as np

from src.common import load_config
from src.severity import estimate_severity, severity_from_ratio


def test_severity_thresholds_follow_config_cuts():
    cfg = load_config()
    assert severity_from_ratio(0.0, "Nitrogen", cfg) == "Mild"
    assert severity_from_ratio(0.149, "Potassium", cfg) == "Mild"
    assert severity_from_ratio(0.15, "Iron", cfg) == "Moderate"
    assert severity_from_ratio(0.40, "Magnesium", cfg) == "Moderate"
    assert severity_from_ratio(0.401, "Phosphorus", cfg) == "Severe"
    assert severity_from_ratio(1.0, "Nitrogen", cfg) == "Severe"


def test_healthy_class_is_always_none():
    cfg = load_config()
    assert severity_from_ratio(0.0, "Healthy", cfg) == "None"
    assert severity_from_ratio(0.15, "Healthy", cfg) == "None"
    assert severity_from_ratio(0.95, "Healthy", cfg) == "None"


def test_yellow_leaf_is_more_affected_than_green():
    cfg = load_config()
    mask = np.full((40, 40), 255, dtype=np.uint8)
    green = np.zeros((40, 40, 3), dtype=np.uint8)
    green[:, :] = (30, 160, 40)
    yellow = np.zeros((40, 40, 3), dtype=np.uint8)
    yellow[:, :] = (30, 210, 220)
    green_label, green_ratio, green_overlay = estimate_severity(green, mask, "Nitrogen", cfg)
    yellow_label, yellow_ratio, yellow_overlay = estimate_severity(yellow, mask, "Nitrogen", cfg)
    assert yellow_ratio > green_ratio
    assert green_label == "Mild"
    assert yellow_label in {"Moderate", "Severe"}
    assert green_overlay.shape == green.shape
    assert yellow_overlay.shape == yellow.shape


def test_healthy_prediction_keeps_none_even_if_pixels_are_yellow():
    cfg = load_config()
    mask = np.full((24, 24), 255, dtype=np.uint8)
    yellow = np.zeros((24, 24, 3), dtype=np.uint8)
    yellow[:, :] = (30, 210, 220)
    label, ratio, _overlay = estimate_severity(yellow, mask, "Healthy", cfg)
    assert ratio > 0.4
    assert label == "None"
