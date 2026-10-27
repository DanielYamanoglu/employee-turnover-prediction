"""People-analytics dashboard. Run: make dashboard"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import streamlit as st

from turnover.config import DATA_PATH, MODEL_PATH
from turnover.explain import global_importance, local_explanation, shap_values
from turnover.model import load_bundle
from turnover.predict import risk_band
from turnover.preprocess import encode, load_data, split_xy

st.set_page_config(page_title="Attrition risk", layout="wide")
st.title("Employee attrition risk")
st.caption("Reference implementation on the public IBM HR Analytics dataset.")

if not MODEL_PATH.exists():
    st.error("No trained model found. Run `make train` first.")
    st.stop()


@st.cache_data
def load():
    b = load_bundle(MODEL_PATH)
    X_raw, y = split_xy(load_data(DATA_PATH))
    X = encode(X_raw, b["columns"])
    return b, X_raw, X, y


bundle, X_raw, X, y = load()
model = bundle["model"]
scope = st.sidebar.radio("Employees", ["Held-out test set", "All employees (includes training rows)"])
if scope.startswith("Held"):
    X = X.loc[bundle["test_ids"]]
    X_raw = X_raw.loc[X.index]
    y = y.loc[X.index]

proba = pd.Series(model.predict_proba(X)[:, 1], index=X.index, name="risk")
band = proba.map(risk_band)
dept = st.sidebar.multiselect("Department", sorted(X_raw["Department"].unique()))
mask = X_raw["Department"].isin(dept) if dept else pd.Series(True, index=X.index)
proba, band, X, X_raw, y = proba[mask], band[mask], X[mask], X_raw[mask], y[mask]

values = shap_values(model, X)
c1, c2, c3 = st.columns(3)
c1.metric("Employees", len(X))
c2.metric("High risk", int((band == "high").sum()))
c3.metric("Mean predicted risk", f"{proba.mean():.1%}")

left, right = st.columns(2)
with left:
    st.subheader("Risk distribution")
    st.bar_chart(pd.cut(proba, [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0], include_lowest=True)
                 .value_counts().sort_index().rename(lambda i: f"{i.left:.1f}-{i.right:.1f}"))
    st.write(band.value_counts().reindex(["high", "medium", "low"], fill_value=0).rename("employees"))
with right:
    st.subheader("Top drivers (mean |SHAP|)")
    st.bar_chart(global_importance(values, X.columns).head(12)[::-1])

st.subheader("Per-employee explanation")
ranked = proba.sort_values(ascending=False)
emp = st.selectbox("Employee ID (sorted by risk)", ranked.index.tolist())
pos = X.index.get_loc(emp)
st.metric("Predicted attrition probability", f"{proba[emp]:.1%}", help=f"Risk band: {band[emp]}")
expl = local_explanation(values[pos], X.loc[emp], top=10)
expl["effect"] = expl["shap"].map(lambda s: "raises risk" if s > 0 else "lowers risk")
st.dataframe(expl, width="stretch", hide_index=True)
st.bar_chart(expl.set_index("feature")["shap"][::-1])
