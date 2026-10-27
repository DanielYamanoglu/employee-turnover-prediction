"""Model construction, evaluation and persistence."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate
from xgboost import XGBClassifier

from .config import CV_FOLDS, SEED


def build_model(scale_pos_weight: float, n_estimators: int = 300) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=2,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=1,
    )


def cross_validate_model(model, X, y, folds: int = CV_FOLDS) -> dict:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    res = cross_validate(model, X, y, cv=cv,
                         scoring=["accuracy", "roc_auc", "f1"])
    return {m: {"mean": float(res[f"test_{m}"].mean()),
                "std": float(res[f"test_{m}"].std())}
            for m in ("accuracy", "roc_auc", "f1")}


def evaluate(model, X, y, threshold: float = 0.5) -> dict:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, proba)),
        "f1": float(f1_score(y, pred)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "threshold": threshold,
        "n": int(len(y)),
    }


def save_bundle(path, **bundle) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path) -> dict:
    return joblib.load(path)
