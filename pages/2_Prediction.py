# -*- coding: utf-8 -*-
"""
pages/2_Prediction.py  —  WasteWise AI Prediction page
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

from src.prediction import predict_waste, TRAINING_CITIES
from src.priority   import classify, THRESHOLDS, PRIORITY_META
from src.analytics  import load_cleaned_data
from src.ui_utils   import (
    apply_theme_css, render_sidebar_branding,
    priority_badge, threshold_explainer, section_header, extrapolation_warning,
    LAST_OBSERVED_YEAR,
)

st.set_page_config(page_title="Prediction — WasteWise AI", page_icon="🔮", layout="wide")
apply_theme_css()
render_sidebar_branding()

# ── Data loaders ─────────────────────────────────────────────────────────────
@st.cache_data
def _city_list():
    df = load_cleaned_data()
    return sorted(df["city"].unique().tolist())

@st.cache_data
def _city_pop_map():
    df = load_cleaned_data()
    return df.drop_duplicates("city").set_index("city")["estimated_population_in_2011"].to_dict()

CITIES       = _city_list()
CITY_POP_MAP = _city_pop_map()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🔮 Waste Generation Prediction")
st.caption(
    "Estimate municipal solid waste generation for a selected city and year "
    "using the trained Gradient Boosting model."
)
st.markdown("---")

# ── Input form ────────────────────────────────────────────────────────────────
with st.form("prediction_form"):
    st.markdown("#### Input Parameters")
    col1, col2, col3 = st.columns(3)

    with col1:
        city_option = st.selectbox(
            "City",
            options=["— Custom city —"] + CITIES,
            index=CITIES.index("Mumbai") + 1 if "Mumbai" in CITIES else 0,
            help="Select a known city or choose 'Custom city' to enter manually."
        )
        if city_option == "— Custom city —":
            city_input = st.text_input(
                "Custom city name", value="NewCity",
                help="Enter any city name. A warning is shown for cities not seen during training."
            )
        else:
            city_input = city_option

    with col2:
        default_pop = (
            CITY_POP_MAP.get(city_input, 2_000_000)
            if city_option != "— Custom city —" else 2_000_000
        )
        population = st.number_input(
            "2011 Census Population",
            min_value=100_000,
            max_value=30_000_000,
            value=int(default_pop),
            step=100_000,
            help=(
                "Use the 2011 Census population as the model's anchor. "
                "The model was trained with this fixed value per city."
            ),
        )

    with col3:
        year = st.number_input(
            "Year (fiscal year start)",
            min_value=1999,
            max_value=2035,
            value=2024,
            step=1,
            help=(
                "The starting calendar year of the fiscal year "
                "(e.g. 2024 for FY 2024-25). "
                f"Values beyond {LAST_OBSERVED_YEAR} are extrapolations."
            ),
        )

    submitted = st.form_submit_button("🔮 Estimate", use_container_width=True, type="primary")

# ── Result ────────────────────────────────────────────────────────────────────
if submitted:
    try:
        result   = predict_waste(city_input, int(population), int(year))
        priority = classify(result.prediction_tpd)
        meta     = PRIORITY_META[priority]

        # ── Extrapolation warning (prominent, shown first) ────────────────
        extrapolation_warning(int(year))

        st.markdown("---")
        st.markdown("#### Estimation Result")

        # ── Four key metrics ──────────────────────────────────────────────
        res1, res2, res3, res4 = st.columns(4)
        res1.metric(
            "Estimated Waste",
            f"{result.prediction_tpd:,.1f} t/day",
            help="Predicted municipal solid waste generation in tonnes per day.",
        )
        res2.metric(
            "Collection Priority",
            priority,
            help=(
                f"Assigned using frozen training-period thresholds: "
                f"LOW ≤ {THRESHOLDS.p33:.0f}  |  MEDIUM ≤ {THRESHOLDS.p67:.0f}  |  HIGH > {THRESHOLDS.p67:.0f} t/day."
            ),
        )
        res3.metric(
            "Model Used",
            "Gradient Boosting",
            help="GradientBoostingRegressor (sklearn). Test RMSE: 581.7 t/day, R²: 0.920.",
        )
        res4.metric(
            "Fiscal Year",
            f"{result.year}-{str(result.year + 1)[-2:]}",
        )

        # ── Priority badge ────────────────────────────────────────────────
        st.markdown(
            f"<div style='margin:1rem 0;'>"
            f"<span style='font-size:1rem;font-weight:600;'>Collection Priority:&nbsp;</span>"
            f"{priority_badge(priority)}"
            f"<span style='font-size:0.82rem;opacity:0.65;margin-left:10px;'>"
            f"{meta['description']}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        threshold_explainer()

        st.markdown("---")

        # ── Main model drivers ────────────────────────────────────────────
        section_header("Main Model Drivers")
        tier_label = {1: "Mega (>4 M)", 2: "Large (2–4 M)", 3: "Medium (<2 M)"}

        drv1, drv2, drv3 = st.columns(3)
        with drv1:
            st.markdown(
                f"**1. Population scale** (~69% of model importance)  \n"
                f"log(population) = {result.log_population:.3f}  \n"
                f"City tier: {result.pop_tier} — {tier_label.get(result.pop_tier, '')}"
            )
        with drv2:
            st.markdown(
                f"**2. Time trend** (~16% of model importance)  \n"
                f"Year: {result.year}  \n"
                "Waste generation has trended upward across Indian cities over time."
            )
        with drv3:
            if result.is_known_city:
                st.markdown(
                    f"**3. City-specific adjustment** (~15% combined for all cities)  \n"
                    f"✅ **{city_input}** was seen during training.  \n"
                    "A city-level offset from the OneHot encoding is applied."
                )
            else:
                st.markdown(
                    f"**3. City-specific adjustment**  \n"
                    f"⚠️ **{city_input}** was **not seen during training**.  \n"
                    "No city-level offset is applied. Estimate relies on "
                    "population scale and time trend only."
                )

        st.caption(
            "⚠️ **Non-causal disclaimer:** Feature importance scores reflect the model's "
            "reliance on each input for splitting decisions. They are correlational, "
            "not causal. A high importance score does not mean a feature *causes* "
            "waste generation."
        )

        st.markdown("---")

        # ── City-specific warnings ────────────────────────────────────────
        if not result.is_known_city:
            st.warning(
                f"**Unseen city:** '{city_input}' was not present in the training data. "
                "The estimate is based solely on the provided population size and year. "
                "No city-specific historical pattern has been learned."
            )
        if result.warnings:
            for w in result.warnings:
                if "not seen" not in w.lower():   # avoid duplicating the city warning above
                    st.warning(w)

        # ── Historical context chart ─────────────────────────────────────
        df = load_cleaned_data()
        city_hist = (
            df[(df["city"] == city_input) & df["waste_generated"].notna()]
            .sort_values("year_start")
        )
        if len(city_hist) > 0:
            section_header(f"Historical Context — {city_input}")
            dark = st.session_state.get("theme", "light") == "dark"
            fig_bg  = "#0e1117" if dark else "#ffffff"
            ax_text = "#e8ecf0" if dark else "#1a1a2e"
            bar_col = "#4a9eff" if dark else "#3498db"

            fig, ax = plt.subplots(figsize=(9, 3.2), facecolor=fig_bg)
            ax.set_facecolor(fig_bg)
            ax.bar(
                city_hist["fiscal_year"], city_hist["waste_generated"],
                color=bar_col, alpha=0.75, label="Observed", edgecolor="none",
            )
            ax.axhline(
                result.prediction_tpd, color="#e74c3c", linestyle="--", linewidth=1.8,
                label=f"Estimate ({result.year}): {result.prediction_tpd:,.0f} t/day",
            )
            ax.axhline(
                THRESHOLDS.p33, color="#2ecc71", linestyle=":", linewidth=1.2, alpha=0.85,
                label=f"LOW/MED ({THRESHOLDS.p33:.0f} t/d)",
            )
            ax.axhline(
                THRESHOLDS.p67, color="#f39c12", linestyle=":", linewidth=1.2, alpha=0.85,
                label=f"MED/HIGH ({THRESHOLDS.p67:.0f} t/d)",
            )
            for spine in ax.spines.values():
                spine.set_edgecolor(ax_text)
                spine.set_alpha(0.2)
            ax.set_xlabel("Fiscal Year", fontsize=9, color=ax_text)
            ax.set_ylabel("Waste (t/day)", fontsize=9, color=ax_text)
            ax.tick_params(axis="x", rotation=35, labelsize=8, colors=ax_text)
            ax.tick_params(axis="y", labelsize=8, colors=ax_text)
            legend = ax.legend(fontsize=7.5, facecolor=fig_bg, labelcolor=ax_text,
                               edgecolor="none")
            ax.set_title(
                f"{city_input} — Historical observed waste + current estimate",
                fontsize=10, fontweight="bold", color=ax_text,
            )
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    except ValueError as e:
        st.error(f"Input error: {e}")
    except Exception as e:
        st.error(f"Estimation failed: {e}")

# ── Priority Threshold Reference (always visible) ────────────────────────────
st.markdown("---")
st.markdown("#### Priority Threshold Reference")
thresh_data = {
    "Priority":  ["🟢 LOW", "🟡 MEDIUM", "🔴 HIGH"],
    "Range (t/day)": [
        f"≤ {THRESHOLDS.p33:,.2f}",
        f"{THRESHOLDS.p33:,.2f} – {THRESHOLDS.p67:,.1f}",
        f"> {THRESHOLDS.p67:,.1f}",
    ],
    "Threshold source": [
        "p33 of training-period waste",
        "p33–p67 of training-period waste",
        "p67 of training-period waste",
    ],
}
st.dataframe(pd.DataFrame(thresh_data), hide_index=True, use_container_width=True)
st.caption(
    f"Thresholds frozen from {THRESHOLDS.n_train} training-period observations "
    "(fiscal years 1999-00 to 2019-20, year_start < 2020). "
    "Test-set data was NOT used to establish these thresholds."
)
