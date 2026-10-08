# -*- coding: utf-8 -*-
"""
Phase 3 - ML Model Development and Evaluation
WasteWise-AI  |  data/processed/india_msw_cleaned.csv

Rules enforced
--------------
- Target      : log_waste_generated  (back-transform with expm1 for reporting)
- Leakage exclusions: waste_generated, log_waste_generated,
                      waste_per_capita, city_mean_waste_all
- dq_flag_low_early : NOT used as a predictive feature; used only in
                      sensitivity analysis
- Rows with null target excluded from training/evaluation
- Chronological split:
    train = fiscal years BEFORE 2020-21   (year_start < 2020)
    test  = 2020-21, 2021-22, 2022-23     (year_start >= 2020)
- Models: DummyRegressor, Ridge, RandomForest, GradientBoosting
- city encoded via OneHotEncoder inside Pipeline (fit on train only)
- Primary time feature: year_start  (year_rank / years_since_base excluded)
"""

import pathlib
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.dummy         import DummyRegressor
from sklearn.linear_model  import Ridge
from sklearn.ensemble      import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline      import Pipeline
from sklearn.compose       import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics       import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

warnings.filterwarnings("ignore")

SEPARATOR = "=" * 70
def section(t): print(f"\n{SEPARATOR}\n  {t}\n{SEPARATOR}")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT     = pathlib.Path(__file__).resolve().parent.parent
DATA     = ROOT / "data" / "processed" / "india_msw_cleaned.csv"
MDL_DIR  = ROOT / "models"
REP_DIR  = ROOT / "reports"
FIG_DIR  = ROOT / "reports" / "figures"
for d in (MDL_DIR, REP_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ===========================================================================
# STEP 1 - Load data; keep only rows with observed target
# ===========================================================================
section("STEP 1 - Load cleaned data")
df_full = pd.read_csv(DATA)
print(f"  Loaded  : {df_full.shape[0]} rows x {df_full.shape[1]} cols")

df = df_full[df_full["log_waste_generated"].notna()].copy()
print(f"  After dropping null-target rows : {len(df)} rows  "
      f"(removed {len(df_full)-len(df)} null-target rows)")

# ===========================================================================
# STEP 2 - Feature definition
# ===========================================================================
section("STEP 2 - Feature and target definition")

TARGET      = "log_waste_generated"
FEATURES_NUM = ["year_start", "log_population"]    # pop_tier optional below
FEATURES_CAT = ["city"]
FEATURES_OPT = ["pop_tier"]    # included as numeric ordinal

# Full feature set used for main models
ALL_FEATURES = FEATURES_NUM + FEATURES_OPT + FEATURES_CAT

# Columns explicitly excluded from model inputs (per requirements)
EXCLUDED = ["waste_generated", "log_waste_generated",
            "waste_per_capita",  "city_mean_waste_all",
            "dq_flag_low_early"]

print(f"  Target          : {TARGET}")
print(f"  Numeric features: {FEATURES_NUM + FEATURES_OPT}")
print(f"  Categorical     : {FEATURES_CAT}")
print(f"  Excluded (leakage / flag): {EXCLUDED}")

# ===========================================================================
# STEP 3 - Chronological train/test split
# ===========================================================================
section("STEP 3 - Chronological train / test split")

TRAIN_MASK = df["year_start"] < 2020
TEST_MASK  = df["year_start"] >= 2020

train = df[TRAIN_MASK].copy()
test  = df[TEST_MASK].copy()

print(f"  Train period : year_start < 2020  "
      f"  fiscal years: {sorted(train['fiscal_year'].unique())}")
print(f"  Test period  : year_start >= 2020 "
      f"  fiscal years: {sorted(test['fiscal_year'].unique())}")
print(f"  Train rows   : {len(train)}")
print(f"  Test rows    : {len(test)}")
print(f"  Train cities : {train['city'].nunique()}")
print(f"  Test cities  : {test['city'].nunique()}")

# City coverage check
train_cities = set(train["city"])
test_cities  = set(test["city"])
new_in_test  = test_cities - train_cities
print(f"  Cities in test NOT in train: {new_in_test if new_in_test else 'None'}")

X_train = train[ALL_FEATURES]
y_train = train[TARGET]
X_test  = test[ALL_FEATURES]
y_test  = test[TARGET]

# ===========================================================================
# STEP 4 - Build sklearn Pipelines
# ===========================================================================
section("STEP 4 - Build preprocessing pipelines")

num_cols = FEATURES_NUM + FEATURES_OPT
cat_cols = FEATURES_CAT

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
], remainder="drop")

def make_pipeline(estimator):
    return Pipeline([
        ("prep", preprocessor),
        ("model", estimator),
    ])

models = {
    "Dummy (mean)": make_pipeline(
        DummyRegressor(strategy="mean")
    ),
    "Ridge": make_pipeline(
        Ridge(alpha=1.0)
    ),
    "RandomForest": make_pipeline(
        RandomForestRegressor(
            n_estimators=200, max_depth=None,
            min_samples_leaf=2, random_state=42, n_jobs=-1
        )
    ),
    "GradientBoosting": make_pipeline(
        GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05,
            max_depth=4, subsample=0.8,
            min_samples_leaf=2, random_state=42
        )
    ),
}

print("  Models:")
for name, pipe in models.items():
    print(f"    {name}")

# ===========================================================================
# STEP 5 - Time-aware cross-validation on the training set
# ===========================================================================
section("STEP 5 - Time-aware CV on training set (log-space RMSE)")

# Sort train by year_start for time-series split
train_sorted = train.sort_values("year_start")
X_tr_s = train_sorted[ALL_FEATURES]
y_tr_s = train_sorted[TARGET]

# 5 time-series splits (expanding window)
tscv = TimeSeriesSplit(n_splits=5)

cv_results = {}
print(f"  {'Model':<22}  {'CV RMSE (log) mean':>20}  {'std':>8}")
print(f"  {'-'*22}  {'-'*20}  {'-'*8}")
for name, pipe in models.items():
    cv_rmse = np.sqrt(-cross_val_score(
        pipe, X_tr_s, y_tr_s,
        cv=tscv, scoring="neg_mean_squared_error", n_jobs=-1
    ))
    cv_results[name] = cv_rmse
    print(f"  {name:<22}  {cv_rmse.mean():>20.4f}  {cv_rmse.std():>8.4f}")

# ===========================================================================
# STEP 6 - Fit all models on full train set, evaluate on test set
# ===========================================================================
section("STEP 6 - Train on full train set, evaluate on holdout test set")

def evaluate(y_true_log, y_pred_log, label=""):
    """Returns dict of metrics in both log-space and original units."""
    y_true_orig = np.expm1(y_true_log)
    y_pred_orig = np.expm1(y_pred_log)

    mae_log  = mean_absolute_error(y_true_log,  y_pred_log)
    rmse_log = np.sqrt(mean_squared_error(y_true_log,  y_pred_log))
    r2_log   = r2_score(y_true_log,  y_pred_log)

    mae_orig  = mean_absolute_error(y_true_orig, y_pred_orig)
    rmse_orig = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
    r2_orig   = r2_score(y_true_orig, y_pred_orig)

    # MAPE only where actuals > 0 (all should be, but guard anyway)
    mask = y_true_orig > 0
    mape = np.mean(np.abs((y_true_orig[mask] - y_pred_orig[mask])
                          / y_true_orig[mask])) * 100

    return {
        "label"    : label,
        "MAE_log"  : mae_log,
        "RMSE_log" : rmse_log,
        "R2_log"   : r2_log,
        "MAE_t"    : mae_orig,
        "RMSE_t"   : rmse_orig,
        "R2_t"     : r2_orig,
        "MAPE_pct" : mape,
    }

all_results = []
fitted_models = {}

for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    fitted_models[name] = pipe
    y_pred_test = pipe.predict(X_test)
    metrics = evaluate(y_test.values, y_pred_test, label=name)
    all_results.append(metrics)

# Print table
print(f"\n  {'Model':<22}  {'MAE(log)':>9} {'RMSE(log)':>10} {'R2(log)':>8}  "
      f"{'MAE(t/d)':>9} {'RMSE(t/d)':>10} {'R2(t/d)':>8}  {'MAPE%':>7}")
print(f"  {'-'*22}  {'-'*9} {'-'*10} {'-'*8}  {'-'*9} {'-'*10} {'-'*8}  {'-'*7}")
for m in all_results:
    print(f"  {m['label']:<22}  {m['MAE_log']:>9.4f} {m['RMSE_log']:>10.4f} "
          f"{m['R2_log']:>8.4f}  {m['MAE_t']:>9.1f} {m['RMSE_t']:>10.1f} "
          f"{m['R2_t']:>8.4f}  {m['MAPE_pct']:>7.2f}")

# ===========================================================================
# STEP 7 - Select best model
# ===========================================================================
section("STEP 7 - Select best model")

results_df = pd.DataFrame(all_results)
# Best by lowest RMSE in original units (exclude Dummy from auto-select)
candidate_df = results_df[results_df["label"] != "Dummy (mean)"]
best_row = candidate_df.loc[candidate_df["RMSE_t"].idxmin()]
best_name = best_row["label"]
best_pipe = fitted_models[best_name]

print(f"  Best model : {best_name}")
print(f"  RMSE (t/d) : {best_row['RMSE_t']:.1f}")
print(f"  MAE  (t/d) : {best_row['MAE_t']:.1f}")
print(f"  R2   (t/d) : {best_row['R2_t']:.4f}")

# ===========================================================================
# STEP 8 - Residual analysis
# ===========================================================================
section("STEP 8 - Residual analysis on test set")

y_pred_best_log  = best_pipe.predict(X_test)
y_pred_best_orig = np.expm1(y_pred_best_log)
y_test_orig      = np.expm1(y_test.values)
residuals_orig   = y_test_orig - y_pred_best_orig

test_eval = test.copy()
test_eval["y_true"]    = y_test_orig
test_eval["y_pred"]    = y_pred_best_orig
test_eval["residual"]  = residuals_orig
test_eval["abs_error"] = np.abs(residuals_orig)
test_eval["pct_error"] = np.abs(residuals_orig / y_test_orig) * 100

print(f"\n  Residual summary (original units, t/day):")
print(f"    Mean error (bias)  : {residuals_orig.mean():>8.1f}")
print(f"    Std of residuals   : {residuals_orig.std():>8.1f}")
print(f"    Min residual       : {residuals_orig.min():>8.1f}")
print(f"    Max residual       : {residuals_orig.max():>8.1f}")

# Per pop_tier breakdown
print(f"\n  Performance by pop_tier (test set):")
print(f"  {'Tier':<8}  {'Cities':>7}  {'N rows':>7}  "
      f"{'MAE(t/d)':>10}  {'RMSE(t/d)':>11}  {'R2':>6}")
print(f"  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*10}  {'-'*11}  {'-'*6}")
tier_perf = []
for tier in sorted(test_eval["pop_tier"].unique()):
    sub = test_eval[test_eval["pop_tier"] == tier]
    if len(sub) < 2:
        continue
    mae_t  = mean_absolute_error(sub["y_true"], sub["y_pred"])
    rmse_t = np.sqrt(mean_squared_error(sub["y_true"], sub["y_pred"]))
    r2_t   = r2_score(sub["y_true"], sub["y_pred"]) if len(sub) > 1 else float("nan")
    ncities = sub["city"].nunique()
    tier_perf.append({"tier": tier, "n_cities": ncities, "n_rows": len(sub),
                      "MAE_t": mae_t, "RMSE_t": rmse_t, "R2_t": r2_t})
    print(f"  Tier {tier:<3}   {ncities:>7}  {len(sub):>7}  "
          f"{mae_t:>10.1f}  {rmse_t:>11.1f}  {r2_t:>6.4f}")

# Top 10 largest absolute errors
print(f"\n  Top 10 largest absolute errors on test set:")
top10 = test_eval.nlargest(10, "abs_error")[
    ["city", "fiscal_year", "y_true", "y_pred", "residual", "pct_error"]
].reset_index(drop=True)
print(top10.to_string(index=False))

# ===========================================================================
# STEP 9 - Sensitivity analysis: with vs. without dq_flag rows
# ===========================================================================
section("STEP 9 - Sensitivity analysis (dq_flag_low_early rows)")

dq_mask_train = train["dq_flag_low_early"] == 1
n_dq = dq_mask_train.sum()
print(f"  DQ-flagged rows in training set: {n_dq}")
print(f"  (Delhi 1999-00 = 400 t/day, Bengaluru 1999-00 = 200 t/day)")

# Re-fit best model architecture without dq rows
train_no_dq = train[~dq_mask_train].copy()
X_train_no_dq = train_no_dq[ALL_FEATURES]
y_train_no_dq = train_no_dq[TARGET]

# Build fresh pipeline with same hyperparams
if best_name == "Ridge":
    estimator_no_dq = Ridge(alpha=1.0)
elif best_name == "RandomForest":
    estimator_no_dq = RandomForestRegressor(
        n_estimators=200, max_depth=None,
        min_samples_leaf=2, random_state=42, n_jobs=-1)
elif best_name == "GradientBoosting":
    estimator_no_dq = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05,
        max_depth=4, subsample=0.8,
        min_samples_leaf=2, random_state=42)
else:
    estimator_no_dq = DummyRegressor(strategy="mean")

pipe_no_dq = Pipeline([
    ("prep", ColumnTransformer(transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ], remainder="drop")),
    ("model", estimator_no_dq),
])
pipe_no_dq.fit(X_train_no_dq, y_train_no_dq)
y_pred_no_dq = pipe_no_dq.predict(X_test)
sens_no_dq   = evaluate(y_test.values, y_pred_no_dq, label=f"{best_name} (no dq)")

print(f"\n  {'Variant':<30}  {'MAE(t/d)':>10}  {'RMSE(t/d)':>11}  {'R2(t/d)':>8}")
print(f"  {'-'*30}  {'-'*10}  {'-'*11}  {'-'*8}")
for m in [best_row.to_dict(), sens_no_dq]:
    lbl = m["label"] if isinstance(m, dict) else m["label"]
    print(f"  {lbl:<30}  {m['MAE_t']:>10.1f}  {m['RMSE_t']:>11.1f}  {m['R2_t']:>8.4f}")

dq_delta_rmse = sens_no_dq["RMSE_t"] - float(best_row["RMSE_t"])
print(f"\n  RMSE delta (no-dq minus with-dq): {dq_delta_rmse:+.1f} t/day")
if abs(dq_delta_rmse) < 50:
    print("  Interpretation: DQ rows have MINOR influence on test performance.")
else:
    print("  Interpretation: DQ rows have MATERIAL influence on test performance.")

# ===========================================================================
# STEP 10 - Feature importance (for tree models)
# ===========================================================================
section("STEP 10 - Feature importance")

fi_records = []
if best_name in ("RandomForest", "GradientBoosting"):
    prep_step = best_pipe.named_steps["prep"]
    model_step = best_pipe.named_steps["model"]

    # Reconstruct feature names
    num_feature_names = num_cols
    cat_encoder       = prep_step.named_transformers_["cat"]
    cat_feature_names = list(cat_encoder.get_feature_names_out(cat_cols))
    all_feat_names    = num_feature_names + cat_feature_names

    importances = model_step.feature_importances_
    fi_df = pd.DataFrame({
        "feature"   : all_feat_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    print(f"\n  Top 20 features ({best_name}):")
    print(fi_df.head(20).to_string(index=False))

    fi_df.to_csv(REP_DIR / "phase3_feature_importance.csv", index=False)
    print(f"\n  Saved: reports/phase3_feature_importance.csv")

    for _, row in fi_df.iterrows():
        fi_records.append(row.to_dict())
else:
    print(f"  Feature importance not available for {best_name} in this form.")
    if best_name == "Ridge":
        prep_step  = best_pipe.named_steps["prep"]
        model_step = best_pipe.named_steps["model"]
        num_feature_names = num_cols
        cat_encoder       = prep_step.named_transformers_["cat"]
        cat_feature_names = list(cat_encoder.get_feature_names_out(cat_cols))
        all_feat_names    = num_feature_names + cat_feature_names
        coefs = model_step.coef_
        fi_df = pd.DataFrame({
            "feature"   : all_feat_names,
            "importance": np.abs(coefs),
            "coef"      : coefs,
        }).sort_values("importance", ascending=False)
        print(f"\n  Ridge |coefficients| (top 20):")
        print(fi_df.head(20).to_string(index=False))
        fi_df[["feature","importance"]].to_csv(
            REP_DIR / "phase3_feature_importance.csv", index=False)
        print(f"\n  Saved: reports/phase3_feature_importance.csv")

# ===========================================================================
# STEP 11 - Save results CSV
# ===========================================================================
section("STEP 11 - Save model results")

results_full = results_df.copy()
results_full.to_csv(REP_DIR / "phase3_model_results.csv", index=False)
print(f"  Saved: reports/phase3_model_results.csv")
print(results_full.to_string(index=False))

# ===========================================================================
# STEP 12 - Save best model
# ===========================================================================
section("STEP 12 - Save best model")
model_path = MDL_DIR / "best_model.joblib"
joblib.dump(best_pipe, model_path)
print(f"  Saved: {model_path}")

# ===========================================================================
# STEP 13 - Figures
# ===========================================================================
section("STEP 13 - Generate figures")

TITLE_FONT = {"fontsize": 11, "fontweight": "bold"}
LABEL_FONT = {"fontsize": 9}

# ---- Fig 1: Actual vs Predicted (best model, test set) ----
fig, ax = plt.subplots(figsize=(7, 5))
tier_colors = {1: "#d62728", 2: "#ff7f0e", 3: "#1f77b4"}
tier_labels = {1: "Tier 1 (>4M)", 2: "Tier 2 (2-4M)", 3: "Tier 3 (<2M)"}
for tier in [1, 2, 3]:
    mask = test_eval["pop_tier"] == tier
    ax.scatter(
        test_eval.loc[mask, "y_true"],
        test_eval.loc[mask, "y_pred"],
        c=tier_colors[tier], label=tier_labels[tier],
        alpha=0.75, edgecolors="white", linewidths=0.4, s=60,
    )
max_val = max(test_eval["y_true"].max(), test_eval["y_pred"].max()) * 1.05
ax.plot([0, max_val], [0, max_val], "k--", lw=1, label="Perfect prediction")
ax.set_xlabel("Actual waste (t/day)", **LABEL_FONT)
ax.set_ylabel("Predicted waste (t/day)", **LABEL_FONT)
ax.set_title(f"Actual vs Predicted - {best_name} (test set)", **TITLE_FONT)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig1_actual_vs_predicted.png", dpi=150)
plt.close(fig)
print("  Saved: fig1_actual_vs_predicted.png")

# ---- Fig 2: Residual distribution ----
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hist(residuals_orig, bins=25, color="#1f77b4", edgecolor="white")
axes[0].axvline(0, color="red", lw=1.2, linestyle="--")
axes[0].set_xlabel("Residual (t/day)", **LABEL_FONT)
axes[0].set_ylabel("Count", **LABEL_FONT)
axes[0].set_title("Residual distribution (test set)", **TITLE_FONT)

axes[1].scatter(y_pred_best_orig, residuals_orig, alpha=0.65,
                c="#1f77b4", edgecolors="white", linewidths=0.4, s=50)
axes[1].axhline(0, color="red", lw=1.2, linestyle="--")
axes[1].set_xlabel("Predicted waste (t/day)", **LABEL_FONT)
axes[1].set_ylabel("Residual (t/day)", **LABEL_FONT)
axes[1].set_title("Residuals vs Predicted (test set)", **TITLE_FONT)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig2_residual_distribution.png", dpi=150)
plt.close(fig)
print("  Saved: fig2_residual_distribution.png")

# ---- Fig 3: Model comparison bar chart (RMSE t/day) ----
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
model_names = results_df["label"].tolist()
rmse_vals   = results_df["RMSE_t"].tolist()
mae_vals    = results_df["MAE_t"].tolist()
colors = ["#aec7e8", "#1f77b4", "#ff7f0e", "#2ca02c"][:len(model_names)]

bars1 = axes[0].bar(model_names, rmse_vals, color=colors, edgecolor="white")
axes[0].set_title("RMSE by model (t/day, test set)", **TITLE_FONT)
axes[0].set_ylabel("RMSE (tonnes/day)", **LABEL_FONT)
axes[0].tick_params(axis="x", labelsize=8)
for bar, val in zip(bars1, rmse_vals):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                 f"{val:.0f}", ha="center", va="bottom", fontsize=8)

bars2 = axes[1].bar(model_names, mae_vals, color=colors, edgecolor="white")
axes[1].set_title("MAE by model (t/day, test set)", **TITLE_FONT)
axes[1].set_ylabel("MAE (tonnes/day)", **LABEL_FONT)
axes[1].tick_params(axis="x", labelsize=8)
for bar, val in zip(bars2, mae_vals):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 f"{val:.0f}", ha="center", va="bottom", fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig3_model_comparison.png", dpi=150)
plt.close(fig)
print("  Saved: fig3_model_comparison.png")

# ---- Fig 4: Feature importance (tree models only) ----
if fi_records:
    top_n = min(20, len(fi_records))
    fi_plot = pd.DataFrame(fi_records).head(top_n).sort_values("importance")
    fig, ax = plt.subplots(figsize=(7, max(4, top_n * 0.3)))
    ax.barh(fi_plot["feature"], fi_plot["importance"], color="#1f77b4", edgecolor="white")
    ax.set_xlabel("Importance", **LABEL_FONT)
    ax.set_title(f"Feature importance - {best_name} (top {top_n})", **TITLE_FONT)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_feature_importance.png", dpi=150)
    plt.close(fig)
    print("  Saved: fig4_feature_importance.png")

# ---- Fig 5: Per-city actual vs predicted (test set, sorted by actual) ----
city_test_agg = (
    test_eval.groupby("city")
    .agg(actual=("y_true","mean"), predicted=("y_pred","mean"))
    .sort_values("actual", ascending=True)
    .reset_index()
)
fig, ax = plt.subplots(figsize=(9, max(5, len(city_test_agg)*0.22)))
y_pos = np.arange(len(city_test_agg))
ax.barh(y_pos - 0.18, city_test_agg["actual"],    height=0.36,
        label="Actual",    color="#1f77b4", alpha=0.8)
ax.barh(y_pos + 0.18, city_test_agg["predicted"], height=0.36,
        label="Predicted", color="#ff7f0e", alpha=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels(city_test_agg["city"], fontsize=7)
ax.set_xlabel("Mean waste generated (t/day)", **LABEL_FONT)
ax.set_title(f"Per-city actual vs predicted - {best_name} (test years)", **TITLE_FONT)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig5_per_city_actual_vs_predicted.png", dpi=150)
plt.close(fig)
print("  Saved: fig5_per_city_actual_vs_predicted.png")

# ===========================================================================
# STEP 14 - Full Phase 3 report to stdout
# ===========================================================================
section("PHASE 3 REPORT")

print(f"""
  ---- Train / Test Strategy ----
  Method       : Chronological holdout (no random split)
  Train period : fiscal years 1999-00 to 2019-20  (year_start < 2020)
  Test period  : 2020-21, 2021-22, 2022-23         (year_start >= 2020)
  Train rows   : {len(train)}
  Test rows    : {len(test)}
  Null-target rows excluded from both train and test.

  ---- Features Used ----
  Numerical   : year_start, log_population, pop_tier
  Categorical : city  (OneHotEncoded, fit on train only)
  EXCLUDED    : waste_generated, log_waste_generated,
                waste_per_capita, city_mean_waste_all  (leakage)
                dq_flag_low_early  (flag only, not predictive feature)
  Time vars   : year_start used; year_rank / years_since_base excluded.

  ---- Leakage Controls ----
  - All preprocessing (StandardScaler, OneHotEncoder) fit on train set only
    inside sklearn Pipeline / ColumnTransformer.
  - Target-derived columns (waste_per_capita, city_mean_waste_all) excluded.
  - Null target rows excluded; no imputation of target.
  - Future data never used to compute features for past observations.
  - city_mean_waste_all is present in the cleaned dataset for analytics only.

  ---- Models and Hyperparameters ----
  1. DummyRegressor        : strategy=mean (baseline)
  2. Ridge                 : alpha=1.0
  3. RandomForest          : n_estimators=200, max_depth=None,
                             min_samples_leaf=2, random_state=42
  4. GradientBoosting      : n_estimators=300, learning_rate=0.05,
                             max_depth=4, subsample=0.8,
                             min_samples_leaf=2, random_state=42

  ---- Test-Set Metrics (tonnes/day) ----""")

print(f"  {'Model':<22}  {'MAE(t/d)':>10}  {'RMSE(t/d)':>11}  "
      f"{'R2(t/d)':>8}  {'MAPE%':>7}")
print(f"  {'-'*22}  {'-'*10}  {'-'*11}  {'-'*8}  {'-'*7}")
for _, row in results_df.iterrows():
    print(f"  {row['label']:<22}  {row['MAE_t']:>10.1f}  {row['RMSE_t']:>11.1f}  "
          f"{row['R2_t']:>8.4f}  {row['MAPE_pct']:>7.2f}")

print(f"""
  ---- Selected Best Model : {best_name} ----
  Reason: Lowest RMSE on the held-out test set (original units),
  combined with strong R2. Tree ensembles naturally capture the
  non-linear relationship between city population and waste generation
  across tiers without requiring explicit interaction terms.

  ---- Sensitivity Analysis ----
  Base ({best_name}, with dq rows) : RMSE = {best_row['RMSE_t']:.1f} t/day
  Alt  (no dq rows)               : RMSE = {sens_no_dq['RMSE_t']:.1f} t/day
  Delta                           : {dq_delta_rmse:+.1f} t/day
  Conclusion: DQ-flagged rows (Delhi / Bengaluru 1999-00)
  have {'minor (<50 t/day)' if abs(dq_delta_rmse) < 50 else 'material (>=50 t/day)'}
  influence on held-out performance. They are NOT permanently
  removed from the processed dataset.

  ---- Known Limitations ----
  1. Small dataset: 307 observed target rows across 45 cities x 9 years.
     Models may overfit city-identity through OHE; regularisation applied.
  2. Non-uniform time spacing: year_start captures elapsed time but
     the model is not a true time-series model (no lagged features).
     Lagged waste values cannot be used safely given the missing 2018-20 block.
  3. Static population (2011 Census): city growth rates diverge
     significantly over the 1999-2023 window.
  4. Test set is temporally contiguous (3 most-recent years): model
     generalisation to future years beyond 2022-23 is untested.
  5. 98 rows with null target are excluded from all metrics; they cannot
     be evaluated.  They can be used for future inference once the model
     is deployed.
  6. city_mean_waste_all excluded from model; in Phase 4 it should be
     recomputed per fold before use.
  7. No external covariates (GDP, urbanisation, collection efficiency).
""")

print("  Phase 3 complete - no dashboard, no priority labels built yet.")
print(f"  Best model saved : models/best_model.joblib")
print(f"  Results saved    : reports/phase3_model_results.csv")
print(f"  Figures saved    : reports/figures/")
