"""Recommendation lookup: known keys, disclaimer, and unknown input."""

from __future__ import annotations

import pytest

from src.common import CLASS_ORDER
from src.recommend import DISCLAIMER, recommend


def test_every_configured_pair_returns_a_disclaimer():
    for deficiency in CLASS_ORDER:
        severities = ["None"] if deficiency == "Healthy" else ["Mild", "Moderate", "Severe"]
        for severity in severities:
            result = recommend(deficiency, severity)
            assert result["deficiency"] == deficiency
            assert result["severity"] == severity
            assert result["symptoms"]
            assert result["advice"]
            assert isinstance(result["sources"], list)
            assert result["text"].endswith(DISCLAIMER)
            assert result["text"].endswith(result["disclaimer"])


def test_nitrogen_mild_lists_expected_sources():
    result = recommend("Nitrogen", "Mild")
    joined = " ".join(result["sources"]).lower()
    assert "urea" in joined
    assert "compost" in joined
    assert "manure" in joined


def test_lookup_is_case_insensitive():
    result = recommend("  magnesium ", "SEVERE")
    assert result["deficiency"] == "Magnesium"
    assert result["severity"] == "Severe"
    assert "magnesium sulfate" in " ".join(result["sources"]).lower()


def test_unknown_deficiency_and_severity_raise():
    with pytest.raises(ValueError, match="Unknown deficiency"):
        recommend("Zinc", "Mild")
    with pytest.raises(ValueError, match="Unknown severity"):
        recommend("Iron", "Critical")
    with pytest.raises(ValueError, match="Unknown severity"):
        recommend("Healthy", "Mild")
