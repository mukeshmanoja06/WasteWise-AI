# -*- coding: utf-8 -*-
"""
pages/4_Analytics.py  —  WasteWise AI Analytics page
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

from src.analytics  import (
    load_cleaned_data, waste_by_year, top_cities, city_comparison,
    model_performance_summary, feature_importance_table,
    predicted_vs_historical,
)
from src.priority   import classify, label_dataframe, THRESHOLDS
from src.ui_utils   import apply_theme_css, render_sidebar_branding, section_header, threshold_explainer

st.set_page_config(page_title="Analytics — WasteWise AI", page_icon="📈", layout="wide")
apply_theme_css()
render_sidebar_branding()

@st.cache_data
def _df():      return load_cleaned_data()
@st.cache_data
def _by_year(): return waste_by_year()
@st.cache_data
def _perf():    return model_performance_summary()
@st.cache_data
def _fi():      return feature_importance_table(top_n=15)

df    = _df()
perf  = _perf()
by_yr = _by_year()

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown("# 📈 Analytics")
st.caption("Year-over-year trends, city comparisons, model metrics, and feature importance.")
st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📅 Yearly Trend", "🏙️ City Comparison", "📊 Model Comparison",
    "🎯 Actual vs Predicted", "🔑 Feature Importance"
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Yearly Trend
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    section_header("Waste Generation by Year")
    st.dataframe(
        by_yr.rename(columns={
            "fiscal_year":"Fiscal Year","year_start":"Year",
            "n_cities_observed":"Cities","total_waste_tpd":"Total (t/day)",
            "mean_waste_tpd":"Mean (t/day)","median_waste_tpd":"Median (t/day)",
            "min_waste_tpd":"Min (t/day)","max_waste_tpd":"Max (t/day)",
        }),
        use_container_width=True, hide_index=True,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
    axes[0].plot(by_yr["fiscal_year"], by_yr["total_waste_tpd"],
                 marker="o", color="#2ecc71", linewidth=2)
    axes[0].fill_between(by_yr["fiscal_year"], by_yr["total_waste_tpd"], alpha=0.15, color="#2ecc71")
    axes[0].set_title("Total waste (t/day)", fontsize=10, fontweight="bold")
    axes[0].tick_params(axis="x", rotation=35, labelsize=8)

    axes[1].plot(by_yr["fiscal_year"], by_yr["median_waste_tpd"],
                 marker="s", color="#3498db", linewidth=2, label="Median")
    axes[1].plot(by_yr["fiscal_year"], by_yr["mean_waste_tpd"],
                 marker="o", color="#e74c3c", linewidth=2, linestyle="--", label="Mean")
    axes[1].legend(fontsize=8)
    axes[1].set_title("Mean vs Median per city", fontsize=10, fontweight="bold")
    axes[1].tick_params(axis="x", rotation=35, labelsize=8)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption("Mean is consistently higher than median — confirming right-skew from mega-cities.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — City Comparison
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    section_header("City Comparison")
    all_fy    = sorted(df["fiscal_year"].unique().tolist())
    sel_fy    = st.selectbox("Fiscal Year", options=all_fy[::-1], index=0, key="comp_fy")
    all_cities = sorted(df["city"].unique().tolist())
    sel_cities = st.multiselect(
        "Cities to compare (leave empty for all)",
        options=all_cities, default=[],
    )
    comp_df = city_comparison(
        cities=sel_cities if sel_cities else None,
        fiscal_year=sel_fy, df=df,
    )
    comp_df = label_dataframe(comp_df, "waste_generated", out_col="priority")

    if len(comp_df) > 0:
        st.dataframe(
            comp_df[["city","estimated_population_in_2011","pop_tier",
                     "waste_generated","waste_per_capita","priority"]].rename(
                columns={"city":"City","estimated_population_in_2011":"Population (2011)",
                         "pop_tier":"Tier","waste_generated":"Waste (t/day)",
                         "waste_per_capita":"kg/person/day","priority":"Priority"}
            ),
            use_container_width=True, hide_index=True,
        )
        threshold_explainer()

        # Horizontal bar
        plot_df = comp_df.dropna(subset=["waste_generated"]).head(20)
        fig2, ax2 = plt.subplots(figsize=(9, max(3, len(plot_df)*0.28)))
        colors_bar = [THRESHOLDS.p67 < v and "#e74c3c" or
                      (THRESHOLDS.p33 < v <= THRESHOLDS.p67 and "#f39c12" or "#2ecc71")
                      for v in plot_df["waste_generated"]]
        ax2.barh(plot_df["city"], plot_df["waste_generated"],
                 color=colors_bar, edgecolor="white")
        ax2.set_xlabel("Waste (t/day)", fontsize=9)
        ax2.invert_yaxis()
        ax2.axvline(THRESHOLDS.p33, color="#2ecc71", linestyle=":", linewidth=1)
        ax2.axvline(THRESHOLDS.p67, color="#f39c12", linestyle=":", linewidth=1)
        ax2.set_title(f"City comparison — {sel_fy}", fontsize=10, fontweight="bold")
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)
    else:
        st.info("No data for selected fiscal year.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Model Comparison
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    section_header("Model Comparison (Test Set)")
    st.caption("All metrics computed on chronological holdout: 2020-21, 2021-22, 2022-23.")

    model_table = pd.DataFrame({
        "Model":     perf["model_names"],
        "MAE (t/d)": perf["MAE_t"],
        "RMSE (t/d)":perf["RMSE_t"],
        "R²":        perf["R2_t"],
        "MAPE %":    perf["MAPE_pct"],
    })
    best_idx = model_table[model_table["Model"] != "Dummy (mean)"]["RMSE (t/d)"].idxmin()

    def highlight_best(row):
        return ["background-color: #d4edda" if row.name == best_idx else "" for _ in row]

    st.dataframe(
        model_table.style.apply(highlight_best, axis=1).format(
            {"MAE (t/d)": "{:.1f}", "RMSE (t/d)": "{:.1f}", "R²": "{:.4f}", "MAPE %": "{:.2f}"}
        ),
        use_container_width=True,
    )
    st.caption("✅ Green row = best model (lowest RMSE). R² is a measure of variance explained, not 'accuracy'.")

    fig3, axes3 = plt.subplots(1, 2, figsize=(10, 3.5))
    colors3 = ["#aec7e8", "#1f77b4", "#ff7f0e", "#2ca02c"]
    names3  = model_table["Model"].tolist()
    axes3[0].bar(names3, model_table["RMSE (t/d)"], color=colors3, edgecolor="white")
    axes3[0].set_title("RMSE (t/day)", fontsize=10, fontweight="bold")
    axes3[0].tick_params(axis="x", labelsize=8, rotation=10)
    for i, v in enumerate(model_table["RMSE (t/d)"]):
        axes3[0].text(i, v+10, f"{v:.0f}", ha="center", fontsize=8)

    axes3[1].bar(names3, model_table["MAE (t/d)"], color=colors3, edgecolor="white")
    axes3[1].set_title("MAE (t/day)", fontsize=10, fontweight="bold")
    axes3[1].tick_params(axis="x", labelsize=8, rotation=10)
    for i, v in enumerate(model_table["MAE (t/d)"]):
        axes3[1].text(i, v+5, f"{v:.0f}", ha="center", fontsize=8)
    fig3.tight_layout()
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Actual vs Predicted
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    section_header("Actual vs Predicted (Test Set, 2020-21 to 2022-23)")

    test_df = df[(df["year_start"] >= 2020) & df["waste_generated"].notna()].copy()
    pop_map  = df.drop_duplicates("city").set_index("city")["estimated_population_in_2011"].to_dict()

    @st.cache_data
    def _test_preds():
        from src.prediction import predict_waste as pw
        rows = []
        for _, row in test_df.iterrows():
            try:
                r = pw(row["city"], int(row["estimated_population_in_2011"]), int(row["year_start"]))
                rows.append({"city": row["city"], "fiscal_year": row["fiscal_year"],
                             "observed": row["waste_generated"], "predicted": r.prediction_tpd,
                             "pop_tier": row["pop_tier"]})
            except Exception:
                pass
        return pd.DataFrame(rows)

    test_preds = _test_preds()
    if len(test_preds) > 0:
        test_preds["residual"] = test_preds["observed"] - test_preds["predicted"]

        tier_colors_scatter = {1: "#e74c3c", 2: "#f39c12", 3: "#3498db"}
        tier_labels_scatter = {1: "Tier 1 (>4M)", 2: "Tier 2 (2-4M)", 3: "Tier 3 (<2M)"}
        fig4, ax4 = plt.subplots(figsize=(7, 5))
        for tier in [1, 2, 3]:
            sub = test_preds[test_preds["pop_tier"] == tier]
            ax4.scatter(sub["observed"], sub["predicted"],
                        c=tier_colors_scatter[tier], label=tier_labels_scatter[tier],
                        alpha=0.75, edgecolors="white", linewidths=0.5, s=60)
        max_v = max(test_preds["observed"].max(), test_preds["predicted"].max()) * 1.05
        ax4.plot([0, max_v], [0, max_v], "k--", lw=1, label="Perfect prediction")
        ax4.set_xlabel("Actual waste (t/day)", fontsize=9)
        ax4.set_ylabel("Predicted waste (t/day)", fontsize=9)
        ax4.legend(fontsize=8)
        ax4.set_title("Actual vs Predicted — test set", fontsize=10, fontweight="bold")
        fig4.tight_layout()
        st.pyplot(fig4, use_container_width=True)
        plt.close(fig4)

        st.dataframe(
            test_preds.sort_values("observed", ascending=False).rename(
                columns={"city":"City","fiscal_year":"Fiscal Year",
                         "observed":"Observed (t/d)","predicted":"Predicted (t/d)",
                         "residual":"Residual (t/d)","pop_tier":"Tier"}
            ),
            use_container_width=True, hide_index=True,
        )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — Feature Importance
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    section_header("Feature Importance — Gradient Boosting")
    st.info(
        "⚠️ Feature importance reflects the model's reliance on each feature for splitting decisions. "
        "**It is correlational, not causal.** A high importance score does not mean that feature "
        "*causes* waste generation — it means the model found it useful for prediction."
    )
    fi_df = _fi()
    fig5, ax5 = plt.subplots(figsize=(8, max(4, len(fi_df)*0.32)))
    ax5.barh(fi_df["feature"][::-1], fi_df["importance"][::-1],
             color="#1f77b4", edgecolor="white")
    ax5.set_xlabel("Importance", fontsize=9)
    ax5.set_title("Top feature importances (GradientBoosting)", fontsize=10, fontweight="bold")
    fig5.tight_layout()
    st.pyplot(fig5, use_container_width=True)
    plt.close(fig5)

    st.dataframe(
        fi_df.rename(columns={"feature": "Feature", "importance": "Importance"})
             .assign(Importance=fi_df["importance"].map("{:.5f}".format)),
        use_container_width=True, hide_index=True,
    )
    st.caption(
        "log_population (~69%) and year_start (~16%) dominate. "
        "City-level OHE dummies together account for the remaining ~15%."
    )
