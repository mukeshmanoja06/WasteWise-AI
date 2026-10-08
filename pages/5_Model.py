# -*- coding: utf-8 -*-
"""
pages/5_Model.py  —  WasteWise AI Model page
"""
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
from src.analytics  import model_performance_summary, feature_importance_table
from src.priority   import THRESHOLDS
from src.ui_utils   import apply_theme_css, render_sidebar_branding

st.set_page_config(page_title="Model — WasteWise AI", page_icon="🤖", layout="wide")
apply_theme_css()
render_sidebar_branding()

perf = model_performance_summary()

st.markdown("# 🤖 Model Information")
st.caption("Technical details about the machine learning model used in WasteWise AI.")
st.markdown("---")

# ── Selected model ────────────────────────────────────────────────────────────
st.markdown("### Selected Model: Gradient Boosting Regressor")
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("""
    **Algorithm**: `sklearn.ensemble.GradientBoostingRegressor`

    **Why selected**: Lowest RMSE (581.7 t/day) and highest R² (0.920) on the chronological
    holdout test set, combined with strong time-aware cross-validation performance (log-RMSE 0.463).

    Gradient Boosting captures the non-linear relationship between city population and waste
    generation better than Ridge regression, without the high variance of an unconstrained
    Random Forest on this small dataset.
    """)
with col2:
    st.markdown("""
    **Hyperparameters**
    ```
    n_estimators   = 300
    learning_rate  = 0.05
    max_depth      = 4
    subsample      = 0.8
    min_samples_leaf = 2
    random_state   = 42
    ```
    """)

st.markdown("---")

# ── Evaluation results ────────────────────────────────────────────────────────
st.markdown("### Evaluation Results (Test Set)")
st.caption("Chronological holdout: test = fiscal years 2020-21, 2021-22, 2022-23 (135 rows).")

m1, m2, m3, m4 = st.columns(4)
m1.metric("MAE",  "405.9 t/day",  help="Mean Absolute Error — average prediction error in original units.")
m2.metric("RMSE", "581.7 t/day",  help="Root Mean Squared Error — penalises large errors more than MAE.")
m3.metric("R²",   "0.920",        help="Coefficient of determination — proportion of variance explained by the model.")
m4.metric("MAPE", "34.8 %",       help="Mean Absolute Percentage Error — averaged over test observations.")

st.warning(
    "**R² = 0.920 does NOT mean '92% accuracy'.** R² measures the proportion of variance in "
    "waste generation explained by the model on the test set. It does not imply the model is "
    "correct 92% of the time."
)

st.markdown("---")

# ── All models comparison ─────────────────────────────────────────────────────
st.markdown("### All Models Compared")
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

st.markdown("---")

# ── Train/test strategy ───────────────────────────────────────────────────────
st.markdown("### Train / Test Strategy")
c1, c2 = st.columns(2)
with c1:
    st.markdown("""
    **Method**: Chronological holdout — no random split.

    | Set | Period | Rows |
    |-----|--------|------|
    | Training | 1999-00 → 2019-20 | **172** |
    | Test | 2020-21 → 2022-23 | **135** |

    Time-aware cross-validation (expanding window, 5 folds) was used
    within the training period before evaluating on the test set.

    Future observations were **never** used to train or validate the model.
    """)
with c2:
    st.markdown("""
    **Features used**

    | Feature | Type | Note |
    |---------|------|------|
    | `year_start` | Numeric | Fiscal year start year |
    | `log_population` | Numeric | log(2011 Census pop.) |
    | `pop_tier` | Numeric | 1/2/3 size tier |
    | `city` | Categorical | OneHotEncoded in Pipeline |

    **Excluded** (leakage prevention):
    `waste_generated`, `log_waste_generated`,
    `waste_per_capita`, `city_mean_waste_all`,
    `dq_flag_low_early`
    """)

st.markdown("---")

# ── Priority thresholds ───────────────────────────────────────────────────────
st.markdown("### Priority Thresholds")
st.info(
    f"Thresholds are **frozen** from the training-period distribution "
    f"({THRESHOLDS.n_train} observations, year_start < 2020). "
    "Test-set data was NOT used to establish thresholds."
)
thresh_df = pd.DataFrame({
    "Priority": ["🟢 LOW", "🟡 MEDIUM", "🔴 HIGH"],
    "Range (t/day)": [
        f"≤ {THRESHOLDS.p33:,.2f}",
        f"{THRESHOLDS.p33:,.2f} – {THRESHOLDS.p67:,.1f}",
        f"> {THRESHOLDS.p67:,.1f}",
    ],
    "Method":  ["p33 of training waste", "p33–p67 of training waste", "p67 of training waste"],
    "Training obs.": ["~33%", "~35%", "~32%"],
})
st.dataframe(thresh_df, hide_index=True, use_container_width=True)

st.markdown("---")

# ── Known limitations ─────────────────────────────────────────────────────────
st.markdown("### Known Limitations")
st.markdown("""
1. **Small dataset**: 307 observed target rows across 45 cities × 9 fiscal years.
   City OHE dummies risk overfitting for cities with few training observations.

2. **No lagged features**: The model is not a true time-series model.
   The 2018-19 / 2019-20 missing-data gap prevents safe use of lagged waste values.

3. **Static 2011 population**: The population anchor does not reflect actual growth
   over the 1999–2023 window. Cities like Bengaluru and Hyderabad have grown
   substantially since 2011.

4. **Tier 2 / Tier 3 performance**: Negative R² for smaller cities on the test set
   (R²=-0.47 for Tier 3, -0.76 for Tier 2). The model is most reliable for Tier 1
   mega-cities (R²=0.82). Small cities have very few training observations.

5. **Three test cities unseen in training**: Gaziabad, Kota, and Jodhpur first appear
   in 2020-21. Predictions for these cities use only population and year trend.

6. **Extrapolation**: Predictions beyond 2022-23 extrapolate the time trend learned
   from 1999–2020. Uncertainty increases with distance from the training period.

7. **No external covariates**: No GDP, urbanisation rate, waste-treatment efficiency,
   or climate data are included (none available in the raw dataset).

8. **Sensitivity to DQ rows**: Excluding the suspect 1999-00 Delhi (400 t/d) and
   Bengaluru (200 t/d) observations increases RMSE by +61 t/day on the test set.
   These rows are retained with a flag.
""")

st.markdown("---")

# ── Feature importance ────────────────────────────────────────────────────────
st.markdown("### Feature Importance")
st.warning(
    "Feature importance is **correlational, not causal**. "
    "High importance means the model found the feature useful for prediction — "
    "it does not imply the feature causes waste generation."
)
fi_df = feature_importance_table(top_n=10)
st.dataframe(
    fi_df.rename(columns={"feature": "Feature", "importance": "Importance"})
         .assign(Importance=fi_df["importance"].map("{:.5f}".format)),
    use_container_width=True, hide_index=True,
)
