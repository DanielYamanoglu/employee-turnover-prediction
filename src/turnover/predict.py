"""Score one employee. Usage:
python -m turnover.predict --employee-json '{"Age": 29, "OverTime": "Yes", ...}'
python -m turnover.predict --employee-json employee.json
Missing fields are filled with training-set medians/modes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import DROP_COLS, HIGH_RISK, ID_COL, MEDIUM_RISK, MODEL_PATH, TARGET
from .explain import local_explanation, shap_values
from .model import load_bundle
from .preprocess import encode


def risk_band(p: float) -> str:
    return "high" if p >= HIGH_RISK else "medium" if p >= MEDIUM_RISK else "low"


def predict_employee(bundle: dict, employee: dict, top: int = 5) -> dict:
    ignored = {ID_COL, TARGET, *DROP_COLS}  # ids, label and constant columns are not model inputs
    employee = {k: v for k, v in employee.items() if k not in ignored}
    unknown = set(employee) - set(bundle["raw_columns"])
    if unknown:
        raise ValueError(f"Unknown fields: {sorted(unknown)}")
    row = {**bundle["defaults"], **employee}
    X = encode(pd.DataFrame([row], columns=bundle["raw_columns"]), bundle["columns"])
    p = float(bundle["model"].predict_proba(X)[0, 1])
    drivers = local_explanation(shap_values(bundle["model"], X)[0], X.iloc[0], top)
    return {
        "attrition_probability": round(p, 4),
        "risk": risk_band(p),
        "top_drivers": [
            {"feature": r.feature, "value": float(r.value), "shap": round(float(r.shap), 4),
             "direction": "raises risk" if r.shap > 0 else "lowers risk"}
            for r in drivers.itertuples()
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--employee-json", required=True, help="JSON string or path to a JSON file")
    ap.add_argument("--model", default=str(MODEL_PATH))
    ap.add_argument("--top", type=int, default=5)
    a = ap.parse_args()
    src = a.employee_json
    employee = json.loads(Path(src).read_text() if Path(src).is_file() else src)
    print(json.dumps(predict_employee(load_bundle(Path(a.model)), employee, a.top), indent=2))


if __name__ == "__main__":
    main()
