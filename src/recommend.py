"""Look up a corrective recommendation for a deficiency and severity.

Entries live in knowledge/recommendations.yaml. There are no application rates.
Unknown deficiency or severity values raise ValueError.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from src.common import load_config, path_from_key

DISCLAIMER = (
    "Consult your local agriculture extension officer or do a soil test before applying fertilizer."
)


@lru_cache(maxsize=1)
def _load_table(path_str: str) -> dict:
    path = Path(path_str)
    with path.open("r", encoding="utf-8") as handle:
        table = yaml.safe_load(handle)
    if not isinstance(table, dict):
        raise ValueError(f"Recommendation file is not a mapping: {path}")
    return table


def load_recommendations(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    return _load_table(str(path_from_key(cfg, "recommendations")))


def _match_key(name: str, mapping: dict) -> str:
    wanted = name.strip().lower()
    for key in mapping:
        if str(key).strip().lower() == wanted:
            return str(key)
    raise KeyError(name)


def recommend(deficiency: str, severity: str, cfg: dict | None = None) -> dict:
    """Return symptoms, sources, advice, and a full text that ends with the disclaimer.

    Raises ValueError if the deficiency or the severity is not in the knowledge file.
    """
    table = load_recommendations(cfg)
    deficiencies = {key: value for key, value in table.items() if key != "disclaimer"}
    try:
        deficiency_key = _match_key(deficiency, deficiencies)
    except KeyError as exc:
        known = ", ".join(str(key) for key in deficiencies)
        raise ValueError(
            f"Unknown deficiency {deficiency!r}. Known deficiencies: {known}."
        ) from exc
    severities = deficiencies[deficiency_key]
    if not isinstance(severities, dict):
        raise ValueError(f"Recommendation entry for {deficiency_key} is malformed.")
    try:
        severity_key = _match_key(severity, severities)
    except KeyError as exc:
        known = ", ".join(str(key) for key in severities)
        raise ValueError(
            f"Unknown severity {severity!r} for {deficiency_key}. Known severities: {known}."
        ) from exc

    entry = severities[severity_key]
    sources = [str(source) for source in entry.get("sources", [])]
    symptoms = str(entry.get("symptoms", "")).strip()
    advice = str(entry.get("advice", "")).strip()
    file_disclaimer = str(table.get("disclaimer", DISCLAIMER)).strip()
    disclaimer = file_disclaimer or DISCLAIMER
    source_text = "; ".join(sources) if sources else "None listed."
    text = (
        f"{deficiency_key} ({severity_key}). {symptoms} "
        f"Common sources: {source_text} "
        f"{advice} {disclaimer}"
    )
    if not text.endswith(DISCLAIMER):
        text = f"{text.rstrip()} {DISCLAIMER}"
    return {
        "deficiency": deficiency_key,
        "severity": severity_key,
        "symptoms": symptoms,
        "sources": sources,
        "advice": advice,
        "disclaimer": disclaimer,
        "text": text,
    }
