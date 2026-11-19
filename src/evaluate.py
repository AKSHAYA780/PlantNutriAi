"""Score the saved search and write the review tables and figures.

Per-class and per-crop numbers are computed from the refit best pipeline on
the held-out split. Combination rows are the ones written by training.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.common import CLASS_ORDER, DUMMY_BANNER, ensure_dir, load_config, path_from_key, read_data_source, set_seed
from src.features import feature_names
from src.train import load_joined_table


def _use_agg() -> None:
    import matplotlib

    matplotlib.use("Agg")


def _best_per_classifier(results: pd.DataFrame, classifiers: list[str]) -> pd.DataFrame:
    ordered = results.sort_values(
        ["test_macro_f1", "cv_accuracy_mean"],
        ascending=[False, False],
        na_position="last",
    )
    picked = ordered.groupby("classifier", as_index=False).head(1)
    rank = {name: index for index, name in enumerate(classifiers)}
    picked = picked.copy()
    picked["_order"] = picked["classifier"].map(lambda name: rank.get(name, len(rank)))
    return picked.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def _save_heatmap(results: pd.DataFrame, cfg: dict, source: str) -> Path:
    _use_agg()
    import matplotlib.pyplot as plt
    import seaborn as sns

    selectors = list(cfg["feature_selection"]["selectors"])
    classifiers = list(cfg["classifiers"])
    pivot = results.pivot(index="selector", columns="classifier", values="test_accuracy")
    pivot = pivot.reindex(index=selectors, columns=classifiers)
    figure, axis = plt.subplots(figsize=(11, 4.8))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGn", vmin=0, vmax=1, ax=axis)
    axis.set_title("Test accuracy by selector and classifier")
    axis.set_xlabel("Classifier")
    axis.set_ylabel("Selector")
    if source == "dummy":
        figure.suptitle("DUMMY DATA - DO NOT REPORT", fontsize=12, color="darkred")
    figure.tight_layout()
    destination = path_from_key(cfg, "heatmap_figure")
    ensure_dir(destination.parent)
    figure.savefig(destination, dpi=140)
    plt.close(figure)
    print(f"Wrote {destination.as_posix()}")
    return destination


def _save_confusion(
    matrix: np.ndarray,
    class_names: list[str],
    cfg: dict,
    source: str,
) -> Path:
    _use_agg()
    import matplotlib.pyplot as plt
    import seaborn as sns

    figure, axis = plt.subplots(figsize=(7.2, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axis,
    )
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("Confusion matrix — best pipeline (test split)")
    if source == "dummy":
        figure.suptitle("DUMMY DATA - DO NOT REPORT", fontsize=12, color="darkred")
    figure.tight_layout()
    destination = path_from_key(cfg, "confusion_figure")
    ensure_dir(destination.parent)
    figure.savefig(destination, dpi=140)
    plt.close(figure)
    print(f"Wrote {destination.as_posix()}")
    return destination


def _save_rf_comparison(results: pd.DataFrame, cfg: dict, source: str) -> Path | None:
    """Bar chart of our best Random Forest test accuracy against the base paper."""
    _use_agg()
    import matplotlib.pyplot as plt

    forest = results[results["classifier"] == "random_forest"]
    if forest.empty:
        print("No Random Forest rows; skipping the base-paper comparison chart.")
        return None
    best = forest.sort_values(
        ["test_accuracy", "test_macro_f1"],
        ascending=[False, False],
    ).iloc[0]
    ours = float(best["test_accuracy"]) * 100.0
    paper = float(cfg["base_paper_accuracy_percent"])
    labels = [
        f"PlantNutriAI RF\n({best['selector']})",
        str(cfg["base_paper_label"]),
    ]
    values = [ours, paper]
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    bars = axis.bar(labels, values, color=["#2e7d32", "#90a4ae"])
    axis.set_ylim(0, 100)
    axis.set_ylabel("Accuracy (%)")
    axis.set_title("Best Random Forest accuracy vs base paper")
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 1.2, f"{value:.2f}%", ha="center", fontsize=9)
    note = str(cfg["base_paper_note"])
    axis.set_xlabel(note)
    if source == "dummy":
        figure.suptitle("DUMMY DATA - DO NOT REPORT", fontsize=12, color="darkred")
    figure.tight_layout()
    destination = path_from_key(cfg, "rf_comparison_figure")
    ensure_dir(destination.parent)
    figure.savefig(destination, dpi=140)
    plt.close(figure)
    print(f"Wrote {destination.as_posix()}")
    return destination


def _per_class_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> pd.DataFrame:
    """Slide-style table. The overall-accuracy row stores accuracy in the F1 column."""
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    rows: list[dict] = []
    for name in CLASS_ORDER:
        if name in report:
            entry = report[name]
            rows.append(
                {
                    "Class": name,
                    "Precision": float(entry["precision"]),
                    "Recall": float(entry["recall"]),
                    "F1": float(entry["f1-score"]),
                }
            )
        else:
            rows.append({"Class": name, "Precision": np.nan, "Recall": np.nan, "F1": np.nan})
    accuracy = float(accuracy_score(y_true, y_pred))
    rows.append({"Class": "Overall accuracy", "Precision": np.nan, "Recall": np.nan, "F1": accuracy})
    return pd.DataFrame(rows)


def _per_crop_table(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    source: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    crops = sorted(frame["crop"].astype(str).unique())
    for crop in crops:
        mask = frame["crop"].astype(str).to_numpy() == crop
        if not np.any(mask):
            continue
        true = y_true[mask]
        pred = y_pred[mask]
        rows.append(
            {
                "crop": crop,
                "n_test": int(mask.sum()),
                "accuracy": float(accuracy_score(true, pred)),
                "macro_precision": float(precision_score(true, pred, average="macro", zero_division=0)),
                "macro_recall": float(recall_score(true, pred, average="macro", zero_division=0)),
                "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
                "data_source": source,
            }
        )
    return pd.DataFrame(rows)


def evaluate(cfg: dict | None = None) -> dict:
    """Write combination tables, the best-model report, and the review figures."""
    cfg = cfg or load_config()
    set_seed(int(cfg["seed"]))
    source = read_data_source(cfg)
    if source == "dummy":
        print(DUMMY_BANNER)

    model_path = path_from_key(cfg, "best_model")
    results_path = path_from_key(cfg, "results_all")
    if not model_path.is_file() or not results_path.is_file():
        from src.train import run_training

        run_training(cfg)
        source = read_data_source(cfg)

    import joblib

    results = pd.read_csv(results_path)
    ensure_dir(results_path.parent)
    results.to_csv(results_path, index=False)

    best_rows = _best_per_classifier(results, list(cfg["classifiers"]))
    best_path = path_from_key(cfg, "results_best_clf")
    best_rows.to_csv(best_path, index=False)
    print(f"Wrote {best_path.as_posix()}")

    _save_heatmap(results, cfg, source)
    _save_rf_comparison(results, cfg, source)

    bundle = joblib.load(model_path)
    pipeline = bundle["pipeline"]
    encoder = bundle["label_encoder"]
    names = list(bundle.get("feature_names") or feature_names(cfg))
    table = load_joined_table(cfg)
    test = table[table["split"] == "test"].copy()
    known = set(str(name) for name in encoder.classes_)
    unseen = ~test["label"].astype(str).isin(known)
    if bool(unseen.any()):
        print(
            f"WARNING: dropping {int(unseen.sum())} test row(s) with a class the model never saw."
        )
        test = test.loc[~unseen].copy()
    x_test = np.nan_to_num(test[names].to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    y_true = encoder.transform(test["label"].astype(str))
    y_pred = pipeline.predict(x_test)
    class_names = [str(name) for name in encoder.classes_]

    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    _save_confusion(matrix, class_names, cfg, source)

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    report_path = path_from_key(cfg, "classification_report")
    ensure_dir(report_path.parent)
    header = (
        f"data_source: {source}\n"
        f"selector: {bundle.get('selector')}\n"
        f"classifier: {bundle.get('classifier')}\n"
    )
    if source == "dummy":
        header = "DUMMY DATA - DO NOT REPORT\n" + header
    report_path.write_text(header + "\n" + report + "\n", encoding="utf-8")
    print(f"Wrote {report_path.as_posix()}")

    per_class = _per_class_table(y_true, y_pred, class_names)
    per_class_path = path_from_key(cfg, "per_class_metrics")
    per_class.to_csv(per_class_path, index=False)
    print(f"Wrote {per_class_path.as_posix()}")
    print(per_class.to_string(index=False))

    per_crop = _per_crop_table(test.reset_index(drop=True), y_true, y_pred, source)
    per_crop_path = path_from_key(cfg, "per_crop_metrics")
    per_crop.to_csv(per_crop_path, index=False)
    print(f"Wrote {per_crop_path.as_posix()}")
    if not per_crop.empty:
        print(per_crop.to_string(index=False))

    if source == "dummy":
        print(DUMMY_BANNER)
    return {"results": results, "per_class": per_class, "per_crop": per_crop}


def main() -> None:
    evaluate(load_config())


if __name__ == "__main__":
    main()
