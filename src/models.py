"""
src/models.py
─────────────
Model factory and evaluation utilities for the SBER intraday alpha pipeline.

Design principles:
- No sklearn Pipeline wrapper for tree-based models (scale-invariant).
- scale_pos_weight computed from training labels to handle class imbalance.
- early_stopping_rounds on validation set used only for stopping, not for
  weight updates — no data leakage.
- CalibratedClassifierCV with cv='prefit' trains the sigmoid on a held-out
  calibration segment, fully separate from training.
"""

from __future__ import annotations

import sklearn
from packaging.version import Version
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from xgboost import XGBClassifier


# ── sklearn version compatibility ─────────────────────────────────────────────
_SKLEARN_VERSION = Version(sklearn.__version__)
_CALIB_ESTIMATOR_KWARG = "estimator" if _SKLEARN_VERSION >= Version("1.2") else "base_estimator"


# ── Default XGBoost hyperparameters ──────────────────────────────────────────
DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 500,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "eval_metric": "logloss",
    "early_stopping_rounds": 30,
    "random_state": 42,
    "n_jobs": -1,
}


def build_xgb_classifier(
    y_train: pd.Series,
    xgb_params: dict[str, Any] | None = None,
) -> XGBClassifier:
    """
    Instantiate an XGBClassifier with scale_pos_weight derived from y_train.

    Parameters
    ----------
    y_train : pd.Series
        Training labels (0/1). Used only to compute class balance.
    xgb_params : dict | None
        Override any default hyperparameter. Merged on top of DEFAULT_XGB_PARAMS.

    Returns
    -------
    XGBClassifier (not yet fitted)
    """
    params = {**DEFAULT_XGB_PARAMS, **(xgb_params or {})}
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    params["scale_pos_weight"] = neg / pos if pos > 0 else 1.0
    return XGBClassifier(**params)


def fit_xgb(
    model: XGBClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    verbose: bool = False,
) -> XGBClassifier:
    """
    Fit XGBClassifier with early stopping on the validation set.

    The validation set is used ONLY for early stopping — its labels never
    influence the gradient updates. This is methodologically safe.

    Parameters
    ----------
    model : XGBClassifier
        Unfitted classifier from build_xgb_classifier().
    X_train, y_train : training data
    X_valid, y_valid : validation data (for early stopping only)
    verbose : bool
        If True, print XGBoost training log.

    Returns
    -------
    XGBClassifier (fitted)
    """
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=verbose,
    )
    print(f"Best iteration: {model.best_iteration} | Best score: {model.best_score:.6f}")
    return model


def calibrate_model(
    fitted_model: Any,
    X_calib: pd.DataFrame,
    y_calib: pd.Series,
    method: str = "sigmoid",
) -> CalibratedClassifierCV:
    """
    Wrap a pre-fitted model with Platt scaling (sigmoid) or isotonic regression.

    Uses cv='prefit' — the model is already trained, calibration only
    learns a monotone transform of the output probabilities on X_calib.

    Parameters
    ----------
    fitted_model : Any sklearn-compatible classifier with predict_proba
    X_calib, y_calib : held-out calibration segment (never seen during training)
    method : 'sigmoid' (Platt) or 'isotonic'

    Returns
    -------
    CalibratedClassifierCV (fitted)
    """
    cal = CalibratedClassifierCV(
        **{_CALIB_ESTIMATOR_KWARG: fitted_model},
        method=method,
        cv="prefit",
    )
    cal.fit(X_calib, y_calib)
    return cal


def evaluate_model(
    model: Any,
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame,
    y_va: pd.Series,
    X_ca: pd.DataFrame,
    y_ca: pd.Series,
    model_name: str = "Model",
) -> dict[str, dict[str, float]]:
    """
    Compute Accuracy, AUC-ROC, Log Loss, and Brier Score on three splits.

    Parameters
    ----------
    model : fitted classifier with predict / predict_proba
    X_tr/y_tr : training split
    X_va/y_va : validation split
    X_ca/y_ca : calibration split
    model_name : label for print output

    Returns
    -------
    dict with keys 'Train', 'Valid', 'Calib', each mapping to metric dict.
    """
    results: dict[str, dict[str, float]] = {}
    for split_name, Xs, ys in [
        ("Train", X_tr, y_tr),
        ("Valid", X_va, y_va),
        ("Calib", X_ca, y_ca),
    ]:
        preds = model.predict(Xs)
        probas = model.predict_proba(Xs)[:, 1]
        metrics = {
            "accuracy": accuracy_score(ys, preds),
            "roc_auc": roc_auc_score(ys, probas),
            "log_loss": log_loss(ys, probas),
            "brier": brier_score_loss(ys, probas),
        }
        results[split_name] = metrics
        print(
            f"[{model_name}] {split_name:6s} | "
            f"Acc={metrics['accuracy']:.4f}  "
            f"AUC={metrics['roc_auc']:.4f}  "
            f"LogLoss={metrics['log_loss']:.4f}  "
            f"Brier={metrics['brier']:.4f}"
        )
    return results


def metrics_summary_df(
    baseline_results: dict[str, dict[str, float]],
    calib_results: dict[str, dict[str, float]],
    baseline_label: str = "XGB",
    calib_label: str = "XGB+Calib",
) -> pd.DataFrame:
    """
    Build a comparison DataFrame of baseline vs calibrated metrics.

    Returns
    -------
    pd.DataFrame indexed by split name.
    """
    rows = []
    for split in ["Train", "Valid", "Calib"]:
        rb = baseline_results[split]
        rc = calib_results[split]
        rows.append({
            "Split":                          split,
            f"{baseline_label} Accuracy":     round(rb["accuracy"], 4),
            f"{baseline_label} AUC":          round(rb["roc_auc"],  4),
            f"{baseline_label} LogLoss":      round(rb["log_loss"], 4),
            f"{baseline_label} Brier":        round(rb["brier"],    4),
            f"{calib_label} Accuracy":        round(rc["accuracy"], 4),
            f"{calib_label} AUC":             round(rc["roc_auc"],  4),
            f"{calib_label} LogLoss":         round(rc["log_loss"], 4),
            f"{calib_label} Brier":           round(rc["brier"],    4),
        })
    return pd.DataFrame(rows).set_index("Split")
