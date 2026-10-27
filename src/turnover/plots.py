"""Figures written to reports/figures (all produced by train.py)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.metrics import RocCurveDisplay

from .explain import global_importance


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def roc_curve(y, proba, path):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    RocCurveDisplay.from_predictions(y, proba, ax=ax, name="XGBoost (held-out)")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Chance")
    ax.legend(loc="lower right")
    ax.set_title("ROC curve")
    _save(fig, path)


def confusion(cm: dict, path):
    m = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(m, cmap="Blues")
    for (i, j), v in np.ndenumerate(m):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=16,
                color="white" if v > m.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Stays", "Leaves"])
    ax.set_yticks([0, 1], ["Stays", "Leaves"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (held-out)")
    _save(fig, path)


def shap_summary(values, X, path):
    plt.figure()
    shap.summary_plot(values, X, max_display=15, show=False)
    fig = plt.gcf()
    fig.suptitle("SHAP summary (held-out)", y=1.0)
    _save(fig, path)


def feature_importance(values, columns, path, top: int = 15):
    imp = global_importance(values, columns).head(top)[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.barh(imp.index, imp.values, color="#2b6cb0")
    ax.set_xlabel("Mean |SHAP value| (log-odds)")
    ax.set_title("Top feature importance")
    _save(fig, path)
