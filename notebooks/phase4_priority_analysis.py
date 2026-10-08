# -*- coding: utf-8 -*-
"""
notebooks/phase4_priority_analysis.py
--------------------------------------
Runs the priority threshold analysis and saves reports/phase4_priority_analysis.csv.
Also runs smoke-tests on the prediction and analytics modules.
"""

import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from src.priority   import (StrategyA, StrategyB, build_priority_report,
                             label_dataframe, _load_observed_waste,
                             STRATEGY_COMPARISON)
from src.prediction import predict_waste, explain_prediction
from src.analytics  import (waste_by_year, top_cities, bottom_cities,
                             city_trend, model_performance_summary,
                             feature_importance_table, predicted_vs_historical)

SEPARATOR = "=" * 70
def section(t): print(f"\n{SEPARATOR}\n  {t}\n{SEPARATOR}")

REP_DIR = ROOT / "reports"
REP_DIR.mkdir(exist_ok=True)

# ===========================================================================
# 1. Load observed waste
# ===========================================================================
section("1. Observed waste distribution")
data = _load_observed_waste()
print(f"  n = {len(data)}")
print(f"  min={data.min():.1f}  p25={np.percentile(data,25):.1f}  "
      f"median={np.median(data):.1f}  mean={data.mean():.1f}  "
      f"p75={np.percentile(data,75):.1f}  p95={np.percentile(data,95):.1f}  "
      f"max={data.max():.1f}")

# ===========================================================================
# 2. Strategy thresholds
# ===========================================================================
section("2. Strategy thresholds")
sa = StrategyA.from_data(data)
sb = StrategyB.from_data(data)

print(f"  Strategy A (quantile p33/p67):")
print(f"    LOW  : waste_generated <= {sa.p33:.1f} t/day")
print(f"    MEDIUM: {sa.p33:.1f} < waste_generated <= {sa.p67:.1f} t/day")
print(f"    HIGH  : waste_generated > {sa.p67:.1f} t/day")

print(f"\n  Strategy B (Jenks natural breaks):")
print(f"    LOW   : waste_generated <= {sb.break1:.1f} t/day")
print(f"    MEDIUM: {sb.break1:.1f} < waste_generated <= {sb.break2:.1f} t/day")
print(f"    HIGH  : waste_generated > {sb.break2:.1f} t/day")

# ===========================================================================
# 3. Distribution comparison
# ===========================================================================
section("3. Class distribution comparison")
report = build_priority_report(data)
print(report.to_string(index=False))

# ===========================================================================
# 4. Label the full cleaned dataset (both strategies)
# ===========================================================================
section("4. Full dataset labelled with both strategies")
df = pd.read_csv(ROOT / "data" / "processed" / "india_msw_cleaned.csv")
obs = df[df["waste_generated"].notna()].copy()

obs_a = label_dataframe(obs, "waste_generated", sa, out_col="priority_A")
obs_ab = label_dataframe(obs_a, "waste_generated", sb, out_col="priority_B")

print("  Strategy A distribution (observed rows):")
print(obs_ab["priority_A"].value_counts().to_string())
print("\n  Strategy B distribution (observed rows):")
print(obs_ab["priority_B"].value_counts().to_string())

# Agreement between strategies
agree = (obs_ab["priority_A"] == obs_ab["priority_B"]).sum()
total = len(obs_ab)
print(f"\n  Agreement between A and B: {agree}/{total} ({agree/total*100:.1f}%)")

# ===========================================================================
# 5. Per-tier breakdown
# ===========================================================================
section("5. Priority distribution by pop_tier")
for strat_name, col in [("A", "priority_A"), ("B", "priority_B")]:
    print(f"\n  Strategy {strat_name}:")
    ct = pd.crosstab(obs_ab["pop_tier"], obs_ab[col])
    ct["total"] = ct.sum(axis=1)
    print(ct.to_string())

# ===========================================================================
# 6. Save phase4_priority_analysis.csv
# ===========================================================================
section("6. Save priority analysis CSV")
output_rows = []
for _, row in obs_ab.iterrows():
    output_rows.append({
        "city":            row["city"],
        "fiscal_year":     row["fiscal_year"],
        "year_start":      row["year_start"],
        "pop_tier":        row["pop_tier"],
        "waste_generated": row["waste_generated"],
        "priority_A":      row["priority_A"],
        "priority_B":      row["priority_B"],
    })
out_df = pd.DataFrame(output_rows)
out_path = REP_DIR / "phase4_priority_analysis.csv"
out_df.to_csv(out_path, index=False)
print(f"  Saved: {out_path}  ({len(out_df)} rows)")

# ===========================================================================
# 7. Strategy comparison narrative
# ===========================================================================
section("7. Strategy comparison")
print(STRATEGY_COMPARISON)

# ===========================================================================
# 8. Smoke-test prediction module
# ===========================================================================
section("8. Prediction module smoke-test")
test_cases = [
    {"city": "Mumbai",   "population_2011": 12_442_373, "year": 2024},
    {"city": "Delhi",    "population_2011": 16_787_941, "year": 2024},
    {"city": "Indore",   "population_2011":  1_960_631, "year": 2024},
    {"city": "NewCity",  "population_2011":  2_500_000, "year": 2024},
]
for tc in test_cases:
    r = predict_waste(**tc)
    prio_a = sa.classify(r.prediction_tpd)
    prio_b = sb.classify(r.prediction_tpd)
    flag = " [UNSEEN]" if not r.is_known_city else ""
    print(f"  {r.city:<20}{flag}  {r.year}  "
          f"pred={r.prediction_tpd:>8,.1f} t/day  "
          f"Tier={r.pop_tier}  PrioA={prio_a}  PrioB={prio_b}")

# ===========================================================================
# 9. Analytics module smoke-test
# ===========================================================================
section("9. Analytics module smoke-test")

print("\n  -- waste_by_year --")
print(waste_by_year().to_string(index=False))

print("\n  -- top_cities (2022-23) --")
print(top_cities(n=5, fiscal_year="2022-23").to_string(index=False))

print("\n  -- bottom_cities (2022-23) --")
print(bottom_cities(n=5, fiscal_year="2022-23").to_string(index=False))

print("\n  -- city_trend (Indore) --")
print(city_trend("Indore").to_string(index=False))

print("\n  -- predicted_vs_historical (Indore) --")
print(predicted_vs_historical("Indore").to_string(index=False))

print("\n  -- model_performance_summary --")
import json
print(json.dumps(model_performance_summary(), indent=4))

print("\n  -- feature_importance_table (top 5) --")
print(feature_importance_table(top_n=5).to_string(index=False))

# ===========================================================================
# 10. Final summary
# ===========================================================================
section("PHASE 4 SUMMARY")
print(f"""
  PART A - Prediction module
    src/prediction.py        -- predict_waste(), predict_batch(),
                                explain_prediction(), load_model()
    Handles unseen cities    : yes (warnings issued; OHE handle_unknown=ignore)
    Back-transform           : expm1(log_pred)
    Input validation         : population, year plausibility checks

  PART B - Priority strategies
    src/priority.py
    Strategy A thresholds    : LOW <= {sa.p33:.1f}  |  MEDIUM <= {sa.p67:.1f}  |  HIGH > {sa.p67:.1f}  t/day
    Strategy B thresholds    : LOW <= {sb.break1:.1f}  |  MEDIUM <= {sb.break2:.1f}  |  HIGH > {sb.break2:.1f}  t/day
    Strategy A distribution  : {obs_ab['priority_A'].value_counts().to_dict()}
    Strategy B distribution  : {obs_ab['priority_B'].value_counts().to_dict()}
    Final strategy           : NOT auto-selected -- awaiting user approval

  PART C - Analytics module
    src/analytics.py         -- 8 analytics functions, all return DataFrames/dicts

  PART D - Explainability
    explain_prediction()     -- plain-language explanation per prediction
                                Drivers: population scale, time trend, city adj.
                                Disclaimer: correlational not causal.

  PART E - Outputs
    src/prediction.py        DONE
    src/priority.py          DONE
    src/analytics.py         DONE
    reports/phase4_priority_analysis.csv  DONE
    tests/test_prediction.py DONE
    tests/test_priority.py   DONE
""")
print("  Phase 4 complete - no dashboard built, no priority strategy auto-selected.")
