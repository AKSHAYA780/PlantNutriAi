"""Build outputs/review_content.md from the tables this run actually wrote.

Metric cells are read from the CSVs. They are not typed into this file.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common import ensure_dir, load_config, path_from_key, read_data_source


def _markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _index, row in frame.iterrows():
        cells = [_format_cell(row[column]) for column in frame.columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _format_cell(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("|", "/")


def _require(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run training and evaluation before exporting review content."
        )


def build_review(cfg: dict | None = None) -> str:
    cfg = cfg or load_config()
    per_class_path = path_from_key(cfg, "per_class_metrics")
    results_path = path_from_key(cfg, "results_all")
    summary_path = path_from_key(cfg, "best_summary")
    selected_path = path_from_key(cfg, "selected_features")
    for path in (per_class_path, results_path, summary_path, selected_path):
        _require(path)

    source = read_data_source(cfg)
    per_class = pd.read_csv(per_class_path)
    results = pd.read_csv(results_path)
    summary = pd.read_csv(summary_path)
    selected = pd.read_csv(selected_path)

    top = results.sort_values(
        ["test_macro_f1", "cv_accuracy_mean"],
        ascending=[False, False],
        na_position="last",
    ).head(5)
    top_columns = [
        "selector",
        "classifier",
        "test_accuracy",
        "test_macro_precision",
        "test_macro_recall",
        "test_macro_f1",
        "cv_accuracy",
    ]
    top = top[[column for column in top_columns if column in top.columns]]

    figures_dir = path_from_key(cfg, "figures_dir")
    figures = sorted(path.relative_to(cfg["_root"]).as_posix() for path in figures_dir.glob("*.png"))
    figure_lines = "\n".join(f"- `{path}`" for path in figures) or "- (no figure files yet)"
    feature_lines = "\n".join(f"- {row.feature}" for row in selected.itertuples(index=False))

    banner = ""
    if source == "dummy":
        banner = (
            "> **DUMMY DATA - DO NOT REPORT.** "
            "`data/raw` was empty, so these numbers are from synthetic leaves.\n\n"
        )

    text = f"""# PlantNutriAI — review content

Data source read from `outputs/data_source.txt`: **{source}**

{banner}## (a) What was implemented

### Dataset handling — `src/data_loader.py`, `config.yaml`
- Folder names are normalised with the alias map in `config.yaml` to Healthy, Nitrogen, Phosphorus, Potassium, Iron, and Magnesium. Unmapped folders are printed and skipped.
- Unreadable files are dropped. Crop is stored on every row. The stratified 80/20 split (seed 42) is written to `outputs/split.csv`.
- An empty `data/raw` triggers `scripts/make_dummy_data.py` and labels the run as dummy data.

### Module 1 — Preprocessing — `src/preprocessing.py`
- Each image is resized to 256×256 and denoised with a Gaussian blur and a median blur.
- The leaf is segmented with an HSV threshold, Otsu, and morphological open/close. The largest contour is kept, and the full frame is used if the mask covers less than 5% of the image.
- A before/after grid is saved to `outputs/figures/preprocessing_samples.png`.

### Module 2 — Feature extraction — `src/features.py`
- Each masked leaf becomes one named vector: RGB/HSV/Lab mean, standard deviation and skewness, an 8-bin hue histogram, GLCM texture, a uniform LBP histogram, log Hu moments, and shape ratios.
- Vectors are cached in `data/processed/features.csv` with path, crop, and label. A current cache is reused.

### Module 3 — Deficiency detection — `src/feature_selection.py`, `src/train.py`
- Selectors (ANOVA F-test, mutual information, random-forest importance, RFE, and none) are fit on the training split only inside a `StandardScaler` → selector → classifier pipeline.
- Classifiers: KNN, RBF SVM, GaussianNB, logistic regression, MLP, random forest, decision tree, and XGBoost. `class_weight='balanced'` is used where the estimator supports it.
- The winner by test macro-F1 (tie-break: CV accuracy) is refit and saved to `outputs/models/best_model.joblib` with the label encoder and the selected feature names.

### Module 4 — Evaluation — `src/evaluate.py`
- Writes `outputs/tables/results_all_combinations.csv`, `outputs/tables/results_best_per_classifier.csv`, the accuracy heatmap, the confusion matrix, and `classification_report_best.txt`.
- Writes `outputs/tables/per_class_metrics.csv` and per-crop scores for the banana and coffee test subsets.
- Saves a bar chart of the best Random Forest accuracy against the base paper figure, labelled as a different dataset and class set.

### Module 5 — Severity — `src/severity.py`
- Rule-based estimate, not a trained model. Pixels inside the leaf and outside the healthy-green HSV range count as affected.
- `severity_ratio = affected / leaf pixels`, mapped with the cuts in `config.yaml`. The Healthy class is always `None`.
- Six sample overlays are saved to `outputs/figures/severity_samples.png`.

### Module 6 — Recommendation — `src/recommend.py`, `knowledge/recommendations.yaml`
- The knowledge file is keyed by deficiency and severity and lists symptoms, common sources, and advice that escalates with severity. There are no application rates.
- `recommend(deficiency, severity)` returns a dict. Unknown keys raise `ValueError`. Every text ends with the extension-officer disclaimer.

### Inference — `src/predict.py`, `app/streamlit_app.py`
- `predict_image` returns the predicted class, confidence, severity, ratio, recommendation, and overlay.
- The Streamlit app uploads a leaf and shows the original, the mask, the overlay, and the recommendation. A missing model file shows a warning instead of a stack trace.

## (b) Slide-10 results table

Read from `{per_class_path.relative_to(cfg["_root"]).as_posix()}`. The Overall accuracy row stores accuracy in the F1 column.

{_markdown_table(per_class)}

## (c) Top-5 combinations

Read from `{results_path.relative_to(cfg["_root"]).as_posix()}`, ranked by test macro-F1 then CV accuracy.

{_markdown_table(top)}

## (d) Best model and selected features

Read from `{summary_path.relative_to(cfg["_root"]).as_posix()}` and `{selected_path.relative_to(cfg["_root"]).as_posix()}`.

{_markdown_table(summary)}

Selected features ({len(selected)}):

{feature_lines}

## (e) Figure files to screenshot

{figure_lines}
"""
    destination = path_from_key(cfg, "review_content")
    ensure_dir(destination.parent)
    destination.write_text(text, encoding="utf-8")
    print(f"Wrote {destination.as_posix()}")
    return text


def main() -> None:
    build_review(load_config())


if __name__ == "__main__":
    main()
