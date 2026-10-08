# -*- coding: utf-8 -*-
"""
pages/1_Dashboard.py  —  WasteWise AI Dashboard
"""
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import numpy as np

from src.analytics import (
    load_cleaned_data, waste_by_year, top_cities,
    bottom_cities, model_performance_summary,
)
from src.priority import classify, THRESHOLDS, PRIORITY_META, label_dataframe
from src.ui_utils import apply_theme_css, render_sidebar_branding, section_header, threshold_explainer

st.set_page_config(page_title="Dashboard — WasteWise AI", page_icon="📊", layout="wide")
apply_theme_css()
render_sidebar_branding()

@st.cache_data
def _load():
    df = load_cleaned_data()
    obs = df[df["waste_generated"].notna()].copy()
    return df, obs

@st.cache_data
def _by_year():   return waste_by_year()
@st.cache_data
def _top(n, fy):  return top_cities(n, fy)
@st.cache_data
def _bottom(n, fy): return bottom_cities(n, fy)
@st.cache_data
def _perf():      return model_performance_summary()

df, obs = _load()
by_year  = _by_year()
perf     = _perf()

# Priority labels on observed data
obs_labelled = label_dataframe(obs, "waste_generated", out_col="priority")
latest_fy = by_year["fiscal_year"].iloc[-1]
latest    = obs_labelled[obs_labelled["fiscal_year"] == latest_fy]

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown("# 📊 Dashboard")
st.caption(f"Overview based on {len(obs)} observed records across {obs['city'].nunique()} cities (1999–2023)")
st.markdown("---")

# ── KPI row ─────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
total_latest = latest["waste_generated"].sum()
high_count   = (obs_labelled["priority"] == "HIGH").sum()
n_cities     = obs["city"].nunique()
latest_year_obs = latest["waste_generated"].sum()
max_waste_city   = latest.sort_values("waste_generated", ascending=False).iloc[0]["city"] if len(latest) > 0 else "—"

k1.metric("Cities Tracked",          f"{n_cities}")
k2.metric("Observed Records",         f"{len(obs):,}")
k3.metric(f"Total Waste ({latest_fy})", f"{latest_year_obs:,.0f} t/day")
k4.metric("HIGH Priority (all years)", f"{high_count}")
k5.metric("Best Model R²",            f"{perf['best_R2_t']:.3f}")

st.markdown("---")

# ── Yearly trend ─────────────────────────────────────────────────────────────
col_left, col_right = st.columns([3, 2])

with col_left:
    section_header("Waste Generation Trend by Year", "Total tonnes/day across all observed cities")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(by_year["fiscal_year"], by_year["total_waste_tpd"],
           color="#2ecc71", alpha=0.8, edgecolor="white", linewidth=0.5)
    ax.plot(by_year["fiscal_year"], by_year["mean_waste_tpd"] * by_year["n_cities_observed"],
            color="#e74c3c", linewidth=1.5, marker="o", markersize=4, label="Total (line)")
    ax.set_xlabel("Fiscal Year", fontsize=9)
    ax.set_ylabel("Total waste (t/day)", fontsize=9)
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax2 = ax.twinx()
    ax2.plot(by_year["fiscal_year"], by_year["n_cities_observed"],
             color="#3498db", linestyle="--", linewidth=1, marker="s", markersize=3,
             label="Cities observed")
    ax2.set_ylabel("Cities with data", fontsize=8, color="#3498db")
    ax2.tick_params(axis="y", labelsize=8, colors="#3498db")
    ax.set_title("Annual waste generation + city coverage", fontsize=10, fontweight="bold")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption(f"Note: 2018-19 and 2019-20 show fewer cities due to systematic data gaps.")

with col_right:
    section_header("Priority Distribution", f"All {len(obs_labelled)} observed city-year records")
    prio_counts = obs_labelled["priority"].value_counts().reindex(["LOW","MEDIUM","HIGH"], fill_value=0)
    colors_pie  = [PRIORITY_META["LOW"]["color"], PRIORITY_META["MEDIUM"]["color"], PRIORITY_META["HIGH"]["color"]]
    fig2, ax2 = plt.subplots(figsize=(4, 3.5))
    wedges, texts, autotexts = ax2.pie(
        prio_counts.values, labels=prio_counts.index,
        colors=colors_pie, autopct="%1.1f%%",
        startangle=90, pctdistance=0.8,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5}
    )
    for t in autotexts: t.set_fontsize(9)
    ax2.set_title("Priority distribution", fontsize=10, fontweight="bold")
    fig2.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)
    threshold_explainer()

st.markdown("---")

# ── Top / Bottom cities ──────────────────────────────────────────────────────
section_header(f"City Snapshot — {latest_fy}")
ctab1, ctab2 = st.tabs(["🔴 Top 10 Highest Waste", "🟢 Bottom 10 Lowest Waste"])

with ctab1:
    top_df = _top(10, latest_fy).copy()
    top_df["priority"] = top_df["waste_generated"].apply(classify)
    top_df["waste_generated"] = top_df["waste_generated"].map("{:,.1f}".format)
    top_df["waste_per_capita"] = top_df["waste_per_capita"].map("{:.4f}".format)
    st.dataframe(
        top_df[["city","pop_tier","waste_generated","waste_per_capita","priority"]].rename(
            columns={"waste_generated":"Waste (t/day)","waste_per_capita":"kg/person/day",
                     "pop_tier":"Tier","city":"City","priority":"Priority"}
        ),
        use_container_width=True, hide_index=True,
    )

with ctab2:
    bot_df = _bottom(10, latest_fy).copy()
    bot_df["priority"] = bot_df["waste_generated"].apply(classify)
    bot_df["waste_generated"] = bot_df["waste_generated"].map("{:,.1f}".format)
    bot_df["waste_per_capita"] = bot_df["waste_per_capita"].map("{:.4f}".format)
    st.dataframe(
        bot_df[["city","pop_tier","waste_generated","waste_per_capita","priority"]].rename(
            columns={"waste_generated":"Waste (t/day)","waste_per_capita":"kg/person/day",
                     "pop_tier":"Tier","city":"City","priority":"Priority"}
        ),
        use_container_width=True, hide_index=True,
    )

st.markdown("---")

# ── Mean waste per fiscal year bar ───────────────────────────────────────────
section_header("Mean Waste per City by Year", "Tracks the average across cities with data in that year")
fig3, ax3 = plt.subplots(figsize=(10, 3))
ax3.bar(by_year["fiscal_year"], by_year["mean_waste_tpd"],
        color="#3498db", alpha=0.8, edgecolor="white")
ax3.set_xlabel("Fiscal Year", fontsize=9)
ax3.set_ylabel("Mean waste (t/day)", fontsize=9)
ax3.tick_params(axis="x", rotation=35, labelsize=8)
ax3.tick_params(axis="y", labelsize=8)
for i, (fy, v, n) in enumerate(zip(by_year["fiscal_year"], by_year["mean_waste_tpd"],
                                    by_year["n_cities_observed"])):
    ax3.text(i, v + 20, f"{v:.0f}", ha="center", va="bottom", fontsize=7.5)
ax3.set_title("Mean waste per city per year (observed cities only)", fontsize=10, fontweight="bold")
fig3.tight_layout()
st.pyplot(fig3, use_container_width=True)
plt.close(fig3)
