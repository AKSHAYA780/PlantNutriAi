"""Train every selector/classifier combination and save the best pipeline.

The search reads the cached feature table and the stratified split. Each
candidate is a StandardScaler -> selector -> classifier pipeline, so the scaler
and the selector see training rows only. The winner is the highest test
macro-F1, with mean cross-validated accuracy as the tie break.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

from src.common import (
    DUMMY_BANNER,
    ensure_dir,
    load_config,
    path_from_key,
    read_data_source,
    set_seed,
)
from src.feature_selection import build_pipeline, selected_feature_names
from src.features import feature_names


def load_joined_table(cfg: dict) -> pd.DataFrame:
    """Join the split file to the feature cache on image path."""
    split_path = path_from_key(cfg, "split_csv")
    feature_path = path_from_key(cfg, "features_csv")
    if not split_path.is_file() or not feature_path.is_file():
        from src.features import extract_all

        extract_all(cfg)
    split = pd.read_csv(split_path)
    features = pd.read_csv(feature_path)
    names = feature_names(cfg)
    missing = [name for name in names if name not in features.columns]
    if missing:
        raise RuntimeError(
            f"Feature cache is missing {len(missing)} columns, for example {missing[:3]}. "
            "Delete data/processed/features.csv and re-run feature extraction."
        )
    merged = split.merge(features[["path", *names]], on="path", how="inner")
    if len(merged) != len(split):
        absent = sorted(set(split["path"]) - set(merged["path"]))
        raise RuntimeError(
            f"{len(absent)} split image(s) have no feature row. First missing path: {absent[:1]}."
        )
    return merged


def _matrix(frame: pd.DataFrame, names: list[str]) -> np.ndarray:
    values = frame[names].to_numpy(dtype=np.float64)
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def _fold_count(y_train: np.ndarray, requested: int) -> int:
    """Use fewer folds when a class has too few training rows for the request."""
    _counts = np.bincount(y_train)
    smallest = int(_counts[_counts > 0].min()) if np.any(_counts > 0) else 0
    return max(0, min(int(requested), smallest))


def score_combination(
    selector_name: str,
    classifier_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    k: int,
    seed: int,
    folds: int,
) -> dict:
    """Cross-validate on train, then fit once and score the held-out test split."""
    pipeline = build_pipeline(
        selector_name,
        classifier_name,
        k,
        seed,
        svm_probability=False,
    )
    usable_folds = _fold_count(y_train, folds)
    if usable_folds < 2:
        cv_mean = float("nan")
        cv_std = float("nan")
        print(
            f"WARNING: {selector_name}/{classifier_name} skipped CV "
            f"(smallest class has {usable_folds} training row(s))."
        )
    else:
        splitter = StratifiedKFold(n_splits=usable_folds, shuffle=True, random_state=seed)
        scores = cross_val_score(
            pipeline,
            x_train,
            y_train,
            cv=splitter,
            scoring="accuracy",
            n_jobs=1,
        )
        cv_mean = float(np.mean(scores))
        cv_std = float(np.std(scores))

    pipeline.fit(x_train, y_train)
    predicted = pipeline.predict(x_test)
    return {
        "selector": selector_name,
        "classifier": classifier_name,
        "test_accuracy": float(accuracy_score(y_test, predicted)),
        "test_macro_precision": float(precision_score(y_test, predicted, average="macro", zero_division=0)),
        "test_macro_recall": float(recall_score(y_test, predicted, average="macro", zero_division=0)),
        "test_macro_f1": float(f1_score(y_test, predicted, average="macro", zero_division=0)),
        "cv_accuracy_mean": cv_mean,
        "cv_accuracy_std": cv_std,
        "cv_folds": int(usable_folds if usable_folds >= 2 else 0),
    }


def choose_best(results: pd.DataFrame) -> pd.Series:
    """Highest test macro-F1, then highest mean CV accuracy."""
    ordered = results.sort_values(
        ["test_macro_f1", "cv_accuracy_mean"],
        ascending=[False, False],
        na_position="last",
    )
    return ordered.iloc[0]


def run_training(cfg: dict | None = None) -> pd.DataFrame:
    """Evaluate every configured combination, refit the winner, and save it."""
    cfg = cfg or load_config()
    seed = int(cfg["seed"])
    set_seed(seed)
    source = read_data_source(cfg)
    if source == "dummy":
        print(DUMMY_BANNER)

    table = load_joined_table(cfg)
    names = feature_names(cfg)
    train = table[table["split"] == "train"].copy()
    test = table[table["split"] == "test"].copy()
    if train.empty or test.empty:
        raise RuntimeError("The split file needs both a train and a test partition.")

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train["label"].astype(str)).astype(np.int64)
    known = set(encoder.classes_)
    unseen = ~test["label"].astype(str).isin(known)
    if bool(unseen.any()):
        print(
            f"WARNING: dropping {int(unseen.sum())} test row(s) whose class was absent from train: "
            f"{sorted(test.loc[unseen, 'label'].astype(str).unique())}."
        )
        test = test.loc[~unseen].copy()
    if test.empty:
        raise RuntimeError("No test rows remain after dropping labels that were absent from train.")
    y_test = encoder.transform(test["label"].astype(str)).astype(np.int64)
    x_train = _matrix(train, names)
    x_test = _matrix(test, names)

    k = int(cfg["feature_selection"]["k"])
    k = max(1, min(k, x_train.shape[1]))
    folds = int(cfg["cv_folds"])
    selectors = list(cfg["feature_selection"]["selectors"])
    classifiers = list(cfg["classifiers"])
    print(
        f"Training {len(selectors) * len(classifiers)} pipelines on "
        f"{len(train)} train / {len(test)} test rows, {x_train.shape[1]} features, k={k}."
    )

    rows: list[dict] = []
    for selector_name in selectors:
        for classifier_name in classifiers:
            print(f"Fitting {selector_name} + {classifier_name} ...")
            row = score_combination(
                selector_name,
                classifier_name,
                x_train,
                y_train,
                x_test,
                y_test,
                k,
                seed,
                folds,
            )
            row["data_source"] = source
            row["k"] = k
            row["cv_accuracy"] = _format_cv(row["cv_accuracy_mean"], row["cv_accuracy_std"])
            rows.append(row)
            print(
                f"  test macro-F1 {row['test_macro_f1']:.4f}  "
                f"accuracy {row['test_accuracy']:.4f}  CV {row['cv_accuracy']}"
            )

    results = pd.DataFrame(rows)
    results = results.sort_values(
        ["test_macro_f1", "cv_accuracy_mean"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)

    results_path = path_from_key(cfg, "results_all")
    ensure_dir(results_path.parent)
    results.to_csv(results_path, index=False)
    print(f"Wrote {results_path.as_posix()}")

    best = choose_best(results)
    # Refit with probability enabled so inference can report a confidence.
    pipeline = build_pipeline(
        str(best["selector"]),
        str(best["classifier"]),
        k,
        seed,
        svm_probability=True,
    )
    pipeline.fit(x_train, y_train)
    kept = selected_feature_names(pipeline, names)
    metrics = {
        "selector": str(best["selector"]),
        "classifier": str(best["classifier"]),
        "test_accuracy": float(best["test_accuracy"]),
        "test_macro_precision": float(best["test_macro_precision"]),
        "test_macro_recall": float(best["test_macro_recall"]),
        "test_macro_f1": float(best["test_macro_f1"]),
        "cv_accuracy_mean": float(best["cv_accuracy_mean"]) if pd.notna(best["cv_accuracy_mean"]) else None,
        "cv_accuracy_std": float(best["cv_accuracy_std"]) if pd.notna(best["cv_accuracy_std"]) else None,
        "cv_accuracy": str(best["cv_accuracy"]),
        "k": k,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_features_in": int(len(names)),
        "n_features_selected": int(len(kept)),
        "data_source": source,
    }
    bundle = {
        "pipeline": pipeline,
        "label_encoder": encoder,
        "feature_names": names,
        "selected_feature_names": kept,
        "selector": str(best["selector"]),
        "classifier": str(best["classifier"]),
        "metrics": metrics,
        "data_source": source,
        "seed": seed,
    }
    model_path = path_from_key(cfg, "best_model")
    ensure_dir(model_path.parent)
    import joblib

    joblib.dump(bundle, model_path)
    print(
        f"Best pipeline: {best['selector']} + {best['classifier']} "
        f"(test macro-F1 {float(best['test_macro_f1']):.4f})."
    )
    print(f"Wrote {model_path.as_posix()}")

    _write_selection_tables(cfg, kept, metrics)
    if source == "dummy":
        print(DUMMY_BANNER)
    return results


def _format_cv(mean: float, std: float) -> str:
    if pd.isna(mean) or pd.isna(std):
        return "n/a"
    return f"{mean:.4f} +/- {std:.4f}"


def _write_selection_tables(cfg: dict, kept: list[str], metrics: dict) -> None:
    features_path = path_from_key(cfg, "selected_features")
    summary_path = path_from_key(cfg, "best_summary")
    ensure_dir(features_path.parent)
    pd.DataFrame({"rank": np.arange(1, len(kept) + 1), "feature": kept}).to_csv(features_path, index=False)
    pd.DataFrame({"key": list(metrics.keys()), "value": [str(value) for value in metrics.values()]}).to_csv(
        summary_path, index=False
    )
    print(f"Wrote {features_path.as_posix()}")
    print(f"Wrote {summary_path.as_posix()}")


def main() -> None:
    run_training(load_config())


if __name__ == "__main__":
    main()
