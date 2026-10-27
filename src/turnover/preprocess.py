"""Loading and feature engineering. One-hot encoding against a fixed column schema."""
from __future__ import annotations

import pandas as pd

from .config import DATA_PATH, DROP_COLS, ID_COL, TARGET


def load_data(path=DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return raw features (indexed by employee id) and a 0/1 target."""
    y = (df[TARGET] == "Yes").astype(int)
    X = df.drop(columns=[TARGET] + [c for c in DROP_COLS if c in df.columns])
    X = X.set_index(ID_COL)
    y.index = X.index
    return X, y


def encode(X: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """One-hot encode categoricals.

    With `columns` given (inference), the output is aligned to the training
    schema: unseen categories are dropped and missing dummies are filled with 0.
    """
    cat_cols = X.select_dtypes(include="object").columns
    out = pd.get_dummies(X, columns=list(cat_cols), dtype=int)
    if columns is not None:
        out = out.reindex(columns=columns, fill_value=0)
    return out


def class_weight(y: pd.Series) -> float:
    """XGBoost scale_pos_weight = negatives / positives."""
    pos = int(y.sum())
    return (len(y) - pos) / max(pos, 1)


def raw_defaults(X: pd.DataFrame) -> dict:
    """Median (numeric) / mode (categorical) per raw column, used to fill missing inputs at predict time."""
    return {
        c: (X[c].mode().iloc[0] if X[c].dtype == "object" else float(X[c].median()))
        for c in X.columns
    }
