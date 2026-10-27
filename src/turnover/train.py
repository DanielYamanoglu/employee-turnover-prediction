"""Train, evaluate and explain. Usage: python -m turnover.train"""
from __future__ import annotations

import argparse
import json

from sklearn.model_selection import train_test_split

from . import plots
from .config import (DATA_PATH, FIGURES_DIR, METRICS_PATH, MODEL_PATH, SEED,
                     TEST_SIZE)
from .explain import global_importance, shap_values
from .model import (build_model, cross_validate_model, evaluate, save_bundle)
from .preprocess import (class_weight, encode, load_data, raw_defaults,
                         split_xy)


def run(data_path=DATA_PATH, model_path=MODEL_PATH, metrics_path=METRICS_PATH,
        figures_dir=FIGURES_DIR, make_figures: bool = True,
        n_estimators: int = 300, cv_folds: int = 5) -> dict:
    X_raw, y = split_xy(load_data(data_path))
    X = encode(X_raw)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)

    model = build_model(class_weight(y_tr), n_estimators)
    cv = cross_validate_model(model, X_tr, y_tr, cv_folds)
    model.fit(X_tr, y_tr)
    holdout = evaluate(model, X_te, y_te)

    metrics = {
        "dataset": {"rows": int(len(y)), "features": int(X.shape[1]),
                    "attrition_rate": float(y.mean())},
        "split": {"train": int(len(y_tr)), "test": int(len(y_te)), "seed": SEED},
        "cross_validation_train": {"folds": cv_folds, **cv},
        "holdout": holdout,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

    save_bundle(model_path, model=model, columns=list(X.columns),
                defaults=raw_defaults(X_raw), raw_columns=list(X_raw.columns),
                test_ids=[int(i) for i in X_te.index], metrics=metrics)

    if make_figures:
        values = shap_values(model, X_te)
        proba = model.predict_proba(X_te)[:, 1]
        plots.roc_curve(y_te, proba, figures_dir / "roc_curve.png")
        plots.confusion(holdout["confusion_matrix"], figures_dir / "confusion_matrix.png")
        plots.shap_summary(values, X_te, figures_dir / "shap_summary.png")
        plots.feature_importance(values, X_te.columns, figures_dir / "feature_importance.png")
    return metrics


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    m = run()
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
