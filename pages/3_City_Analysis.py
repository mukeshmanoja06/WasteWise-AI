# -*- coding: utf-8 -*-
"""
pages/3_City_Analysis.py  —  WasteWise AI City Analysis page
"""
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.analytics  import load_cleaned_data, city_trend, predicted_vs_historical
from src.prediction import predict_waste
from src.priority   import classify, THRESHOLDS, PRIORITY_META, label_dataframe
from src.ui_utils   import (
    apply_theme_css, render_sidebar_branding,
    section_header, threshold_explainer, priority_badge,
)

st.set_page_config(page_title="City Analysis — WasteWise AI", page_icon="🏙️", layout="wide")
apply_theme_css()
render_sidebar_branding()

@st.cache_data
def _df():       return load_cleaned_data()
@st.cache_data
def _cities():
    df = load_cleaned_data()
    return sorted(df["city"].unique().tolist())
@st.cache_data
def _pop_map():
    df = load_cleaned_data()
    return df.drop_duplicates("city").set_index("city")["estimated_population_in_2011"].to_dict()

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown("# 🏙️ City Analysis")
st.caption("Explore historical waste data, model predictions, and priority for any city.")
st.markdown("---")

# ── City selector ────────────────────────────────────────────────────────────
cities   = _cities()
pop_map  = _pop_map()

selected = st.selectbox(
    "Select a city",
    options=cities,
    index=cities.index("Indore") if "Indore" in cities else 0,
)
df       = _df()
pop_2011 = int(pop_map[selected])
city_df  = city_trend(selected, df)

st.markdown("---")

# ── City KPIs ────────────────────────────────────────────────────────────────
obs_rows   = city_df[city_df["waste_generated_is_observed"]]
latest_obs = obs_rows.sort_values("year_start").iloc[-1] if len(obs_rows) > 0 else None
all_years  = sorted(city_df["year_start"].tolist())

try:
    pred_2024 = predict_waste(selected, pop_2011, 2024)
    pred_val  = pred_2024.prediction_tpd
    pred_prio = classify(pred_val)
except Exception:
    pred_val  = None
    pred_prio = "—"

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("2011 Population",        f"{pop_2011:,}")
k2.metric("Observed Years",         f"{len(obs_rows)} / {len(city_df)}")
k3.metric("Latest Observed Waste",  f"{latest_obs['waste_generated']:,.1f} t/day" if latest_obs is not None else "—",
           delta=None)
k4.metric("Predicted (2024)",       f"{pred_val:,.1f} t/day" if pred_val else "—")
k5.metric("Priority (2024 pred.)",  pred_prio)

st.markdown("---")

# ── Trend chart ──────────────────────────────────────────────────────────────
section_header(f"Waste Trend — {selected}", "Blue bars = observed values  |  Red dashed = model prediction for each year")

pvh = predicted_vs_historical(selected, pop_2011, all_years, df)

fig, ax = plt.subplots(figsize=(10, 4))
obs_mask  = pvh["observed_tpd"].notna()
pred_mask = pvh["predicted_tpd"].notna()

ax.bar(pvh.loc[obs_mask, "fiscal_year"],
       pvh.loc[obs_mask, "observed_tpd"],
       color="#3498db", alpha=0.75, label="Observed", edgecolor="white", zorder=2)

ax.plot(pvh.loc[pred_mask, "fiscal_year"],
        pvh.loc[pred_mask, "predicted_tpd"],
        color="#e74c3c", linestyle="--", linewidth=2, marker="o", markersize=5,
        label="Model prediction", zorder=3)

# Mark missing years
missing = pvh[~obs_mask]
if len(missing) > 0:
    ax.scatter(missing["fiscal_year"], [0] * len(missing),
               marker="x", color="#888", s=40, zorder=4, label="Missing observation")

# Threshold lines
ax.axhline(THRESHOLDS.p33, color="#2ecc71", linestyle=":", linewidth=1.2, alpha=0.8,
           label=f"LOW/MED ({THRESHOLDS.p33:.0f} t/d)")
ax.axhline(THRESHOLDS.p67, color="#f39c12", linestyle=":", linewidth=1.2, alpha=0.8,
           label=f"MED/HIGH ({THRESHOLDS.p67:.0f} t/d)")

ax.set_xlabel("Fiscal Year", fontsize=9)
ax.set_ylabel("Waste (tonnes/day)", fontsize=9)
ax.tick_params(axis="x", rotation=35, labelsize=8)
ax.tick_params(axis="y", labelsize=8)
ax.legend(fontsize=8, loc="upper left")
ax.set_title(f"{selected} — Observed vs Model Prediction", fontsize=10, fontweight="bold")
fig.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)
threshold_explainer()

st.markdown("---")

# ── Observed vs Predicted table ───────────────────────────────────────────────
section_header("Observed vs Predicted Values")
pvh_display = pvh.copy()
pvh_display["residual_tpd"]   = pvh_display["residual"].map(lambda v: f"{v:+.1f}" if pd.notna(v) else "—")
pvh_display["observed_tpd"]   = pvh_display["observed_tpd"].map(lambda v: f"{v:,.1f}" if pd.notna(v) else "⚠️ Missing")
pvh_display["predicted_tpd"]  = pvh_display["predicted_tpd"].map(lambda v: f"{v:,.1f}" if pd.notna(v) else "—")
pvh_display["priority"]       = pvh["predicted_tpd"].apply(lambda v: classify(v) if pd.notna(v) else "—")

st.dataframe(
    pvh_display[["fiscal_year","year_start","observed_tpd","predicted_tpd","residual_tpd","priority"]].rename(
        columns={"fiscal_year":"Fiscal Year","year_start":"Year",
                 "observed_tpd":"Observed (t/day)","predicted_tpd":"Predicted (t/day)",
                 "residual_tpd":"Residual (t/day)","priority":"Priority (predicted)"}
    ),
    use_container_width=True, hide_index=True,
)
st.caption("Residual = Observed − Predicted. Missing observations are excluded from model training/evaluation.")

st.markdown("---")

# ── Per-capita trend ─────────────────────────────────────────────────────────
if obs_rows["waste_per_capita"].notna().any():
    section_header("Per-Capita Waste Trend", "kg / person / day  (2011 population anchor)")
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    wpc_obs = city_df[city_df["waste_generated_is_observed"] & city_df["waste_per_capita"].notna()]
    ax2.plot(wpc_obs["fiscal_year"], wpc_obs["waste_per_capita"],
             marker="o", color="#9b59b6", linewidth=2, markersize=5)
    ax2.fill_between(wpc_obs["fiscal_year"], wpc_obs["waste_per_capita"],
                     alpha=0.15, color="#9b59b6")
    ax2.set_xlabel("Fiscal Year", fontsize=9)
    ax2.set_ylabel("kg/person/day", fontsize=9)
    ax2.tick_params(axis="x", rotation=35, labelsize=8)
    ax2.tick_params(axis="y", labelsize=8)
    ax2.set_title(f"{selected} — Per-capita waste (2011 pop. anchor; approx.)", fontsize=10, fontweight="bold")
    fig2.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)
    st.caption("⚠️ Per-capita values use the static 2011 Census population. "
               "Actual per-capita figures differ as populations have grown since 2011.")
