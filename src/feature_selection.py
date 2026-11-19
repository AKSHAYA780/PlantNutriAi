"""Feature selectors and classifiers used inside leakage-safe sklearn Pipelines.

Selectors are fit only on the rows the Pipeline sees, which is the training
fold during cross-validation and the training split during the final fit.
class_weight='balanced' is set on estimators that implement it. XGBoost has no
class_weight argument, so a thin subclass supplies balanced sample weights.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


def safe_f_classif(features: np.ndarray, target: np.ndarray):
    """ANOVA F-scores with non-finite scores replaced by zero.

    A constant column makes f_classif emit NaN, which SelectKBest cannot rank.
    """
    scores, pvalues = f_classif(features, target)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    pvalues = np.nan_to_num(pvalues, nan=1.0, posinf=1.0, neginf=1.0)
    return scores, pvalues


def mutual_info_scores(features: np.ndarray, target: np.ndarray):
    """Mutual information with a fixed seed so selector ties do not drift."""
    return mutual_info_classif(features, target, random_state=42)


class BalancedXGBClassifier(XGBClassifier):
    """XGBoost that applies sklearn-style balanced sample weights during fit."""

    def fit(self, X, y, **kwargs):
        sample_weight = kwargs.get("sample_weight")
        if sample_weight is None:
            kwargs["sample_weight"] = compute_sample_weight(class_weight="balanced", y=y)
        return super().fit(X, y, **kwargs)


def build_selector(name: str, k: int, seed: int):
    """Return a transformer that reduces features to k, or 'passthrough'."""
    if name == "none":
        return "passthrough"
    if name == "anova":
        return SelectKBest(score_func=safe_f_classif, k=k)
    if name == "mutual_info":
        return SelectKBest(score_func=mutual_info_scores, k=k)
    if name == "rf_importance":
        forest = RandomForestClassifier(
            n_estimators=100,
            random_state=seed,
            class_weight="balanced",
            n_jobs=1,
        )
        # threshold=-inf keeps the ranking; max_features keeps the top k.
        return SelectFromModel(forest, max_features=k, threshold=-np.inf)
    if name == "rfe":
        logistic = LogisticRegression(
            max_iter=500,
            class_weight="balanced",
            random_state=seed,
        )
        return RFE(estimator=logistic, n_features_to_select=k, step=5)
    raise ValueError(f"Unknown selector {name!r}.")


def build_classifier(name: str, seed: int, *, svm_probability: bool = True):
    """Build one classifier. class_weight is set only where the estimator has it.

    SVC(probability=True) does not change hard predictions; it only adds Platt
    scaling so inference can report predict_proba. Cross-validation can turn
    that off to save time.
    """
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=5, weights="distance")
    if name == "svm":
        return SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            probability=svm_probability,
            random_state=seed,
        )
    if name == "gaussian_nb":
        return GaussianNB()
    if name == "logreg":
        return LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=seed,
        )
    if name == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(32,),
            max_iter=200,
            random_state=seed,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            random_state=seed,
            class_weight="balanced",
            n_jobs=1,
        )
    if name == "decision_tree":
        return DecisionTreeClassifier(class_weight="balanced", random_state=seed)
    if name == "xgboost":
        return BalancedXGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            random_state=seed,
            n_jobs=1,
            tree_method="hist",
            eval_metric="mlogloss",
        )
    raise ValueError(f"Unknown classifier {name!r}.")


def build_pipeline(
    selector_name: str,
    classifier_name: str,
    k: int,
    seed: int,
    *,
    svm_probability: bool = True,
) -> Pipeline:
    """StandardScaler -> selector -> classifier. The scaler and selector fit on train only."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("selector", build_selector(selector_name, k, seed)),
            ("clf", build_classifier(classifier_name, seed, svm_probability=svm_probability)),
        ]
    )


def selected_feature_names(pipeline: Pipeline, feature_names: list[str]) -> list[str]:
    """Names of features that survived the fitted selector."""
    selector = pipeline.named_steps["selector"]
    if selector == "passthrough" or not hasattr(selector, "get_support"):
        return list(feature_names)
    support = selector.get_support()
    return [name for name, keep in zip(feature_names, support) if keep]
