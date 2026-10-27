import pandas as pd
import pytest

from turnover import train
from turnover.model import load_bundle
from turnover.predict import predict_employee
from turnover.preprocess import class_weight, encode, split_xy


def test_dataset_shape(raw):
    assert raw.shape == (1470, 35)


def test_preprocessing_shape_and_types(raw):
    X_raw, y = split_xy(raw)
    X = encode(X_raw)
    assert len(X) == len(y) == 1470
    assert "Attrition" not in X.columns and "EmployeeCount" not in X.columns
    assert all(pd.api.types.is_numeric_dtype(t) for t in X.dtypes)
    assert set(y.unique()) == {0, 1}
    assert class_weight(y) > 1  # attrition is the minority class


def test_encode_aligns_to_training_schema(raw):
    X_raw, _ = split_xy(raw)
    cols = list(encode(X_raw).columns)
    one = X_raw.iloc[:1].copy()
    one["Department"] = "Unseen"
    out = encode(one, cols)
    assert list(out.columns) == cols


@pytest.fixture(scope="module")
def trained(small, tmp_path_factory):
    d = tmp_path_factory.mktemp("out")
    csv = d / "small.csv"
    small.to_csv(csv, index=False)
    metrics = train.run(csv, d / "m.joblib", d / "metrics.json", d / "fig",
                        make_figures=True, n_estimators=20, cv_folds=3)
    return metrics, load_bundle(d / "m.joblib"), d


def test_training_runs_on_small_sample(trained):
    metrics, _, d = trained
    assert 0.0 <= metrics["holdout"]["roc_auc"] <= 1.0
    assert (d / "metrics.json").exists()
    assert (d / "fig" / "roc_curve.png").exists()


def test_predict_returns_probability(trained, small):
    _, bundle, _ = trained
    emp = small.iloc[0].drop(["Attrition", "EmployeeNumber"]).to_dict()
    out = predict_employee(bundle, emp)
    assert 0.0 <= out["attrition_probability"] <= 1.0
    assert out["risk"] in {"low", "medium", "high"}
    assert out["top_drivers"]


def test_predict_fills_missing_and_rejects_unknown(trained):
    _, bundle, _ = trained
    assert 0.0 <= predict_employee(bundle, {"OverTime": "Yes"})["attrition_probability"] <= 1.0
    with pytest.raises(ValueError):
        predict_employee(bundle, {"NotAField": 1})
