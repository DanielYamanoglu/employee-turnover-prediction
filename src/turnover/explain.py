"""SHAP explanations (TreeExplainer on the XGBoost model)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap


def shap_values(model, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(shap.TreeExplainer(model).shap_values(X))


def global_importance(values: np.ndarray, columns) -> pd.Series:
    """Mean |SHAP| per feature, sorted descending."""
    return (pd.Series(np.abs(values).mean(axis=0), index=columns)
            .sort_values(ascending=False))


def local_explanation(values_row: np.ndarray, X_row: pd.Series, top: int = 8) -> pd.DataFrame:
    """Top features pushing one prediction up (+) or down (-), in log-odds."""
    df = pd.DataFrame({"feature": X_row.index, "value": X_row.values, "shap": values_row})
    return df.reindex(df["shap"].abs().sort_values(ascending=False).index).head(top).reset_index(drop=True)
