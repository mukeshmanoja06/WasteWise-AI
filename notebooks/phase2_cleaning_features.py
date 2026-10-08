# -*- coding: utf-8 -*-
"""
Phase 2 - Data Cleaning and Feature Engineering
WasteWise-AI  |  data/raw/india_msw_city_year.csv

Requirements
------------
- Raw CSV is never modified.
- Aurangabad removed (0 observed target values).
- note (100% NaN) and unit (constant) dropped.
- No fabrication / random imputation of waste_generated.
- fiscal_year converted to numeric year fields; original label preserved.
- City-level features derived only from the available data.
- Log-transform of waste_generated investigated and applied.
- 1999-00 Delhi and Bengaluru flagged as data-quality rows; NOT silently deleted.
- Cleaned dataset saved to data/processed/india_msw_cleaned.csv.
- No ML model is trained.
"""

import pathlib
import numpy as np
import pandas as pd

SEPARATOR = "=" * 70

def section(title: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT     = pathlib.Path(__file__).resolve().parent.parent
RAW_CSV  = ROOT / "data" / "raw"  / "india_msw_city_year.csv"
OUT_DIR  = ROOT / "data" / "processed"
OUT_CSV  = OUT_DIR / "india_msw_cleaned.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ===========================================================================
# STEP 0 - Load raw data (read-only reference)
# ===========================================================================
section("STEP 0 - Load raw CSV")
raw = pd.read_csv(RAW_CSV)
print(f"  Raw shape  : {raw.shape[0]} rows x {raw.shape[1]} columns")
print(f"  Columns    : {list(raw.columns)}")

# ===========================================================================
# STEP 1 - Drop constant / fully-null columns
# ===========================================================================
section("STEP 1 - Drop 'unit' (constant) and 'note' (100% NaN)")
df = raw.drop(columns=["unit", "note"])
print(f"  Shape after drop : {df.shape[0]} rows x {df.shape[1]} columns")
print(f"  Remaining cols   : {list(df.columns)}")

# ===========================================================================
# STEP 2 - Remove Aurangabad (zero observed waste_generated values)
# ===========================================================================
section("STEP 2 - Remove Aurangabad")
abad_rows = df[df["city"] == "Aurangabad"]
print(f"  Aurangabad rows removed : {len(abad_rows)}")
df = df[df["city"] != "Aurangabad"].copy()
print(f"  Shape after removal : {df.shape[0]} rows x {df.shape[1]} columns")

# ===========================================================================
# STEP 3 - Fiscal-year to numeric year representation
# ===========================================================================
section("STEP 3 - Fiscal-year -> numeric year fields")

# fiscal_year format: 'YYYY-YY' (e.g. '2022-23')
# Convention: use the START calendar year as the anchor.
# This gives a clean integer usable directly in regression.
df["year_start"] = df["fiscal_year"].str.split("-").str[0].astype(int)

# Ordinal rank (0-indexed) over the sorted set of fiscal years.
# Useful as a discrete time-step feature (equal spacing assumption is
# explicitly NOT made; the raw year_start encodes actual elapsed time).
fy_order   = sorted(df["fiscal_year"].unique())
fy_to_rank = {fy: i for i, fy in enumerate(fy_order)}
df["year_rank"] = df["fiscal_year"].map(fy_to_rank)

# Years elapsed since first observation (1999-00 -> 0)
base_year    = df["year_start"].min()
df["years_since_base"] = df["year_start"] - base_year

print("  Fiscal-year -> numeric mapping:")
print(f"  {'fiscal_year':<12}  {'year_start':>10}  {'year_rank':>9}  {'years_since_base':>17}")
print(f"  {'-'*12}  {'-'*10}  {'-'*9}  {'-'*17}")
for fy in fy_order:
    ys = int(fy.split("-")[0])
    print(f"  {fy:<12}  {ys:>10}  {fy_to_rank[fy]:>9}  {ys - base_year:>17}")

# ===========================================================================
# STEP 4 - City-level features from available data
# ===========================================================================
section("STEP 4 - City-level features")

# ---- 4a. Population (log) ----
# estimated_population_in_2011 is a 2011 Census anchor; static per city.
# Log-population is standard practice for city-size modelling (Zipf law).
df["log_population"] = np.log(df["estimated_population_in_2011"])
print("  4a. log_population  (log of 2011 Census population)")

# ---- 4b. Population tier ----
# Segmenting cities by scale is well-established (Census / MoHUA categories).
#   Tier 1 : pop > 4 million   (mega-cities)
#   Tier 2 : pop 2 - 4 million
#   Tier 3 : pop < 2 million
def pop_tier(pop: int) -> int:
    if pop > 4_000_000:
        return 1
    elif pop > 2_000_000:
        return 2
    else:
        return 3

df["pop_tier"] = df["estimated_population_in_2011"].apply(pop_tier)
tier_counts = df.drop_duplicates("city")["pop_tier"].value_counts().sort_index()
print(f"  4b. pop_tier  (1=mega >4M, 2=large 2-4M, 3=medium <2M)")
for t, n in tier_counts.items():
    cities_in_tier = sorted(
        df[df["pop_tier"] == t]["city"].unique().tolist()
    )
    print(f"       Tier {t}: {n} cities  {cities_in_tier}")

# ---- 4c. Per-capita waste proxy ----
# Only computable where waste_generated is not null.
# Expressed as kg/person/day (tonnes * 1000 / population).
# This captures structural efficiency differences across cities.
# NOTE: population is 2011 Census; waste years span 1999-2023.
#       The ratio is an approximation, not a precise per-capita figure.
df["waste_per_capita"] = np.where(
    df["waste_generated"].notna(),
    (df["waste_generated"] * 1000) / df["estimated_population_in_2011"],
    np.nan
)
wpc_stats = df["waste_per_capita"].dropna().describe()
print(f"\n  4c. waste_per_capita  (kg/person/day, from observed rows only)")
print(f"       min={wpc_stats['min']:.4f}  median={wpc_stats['50%']:.4f}"
      f"  mean={wpc_stats['mean']:.4f}  max={wpc_stats['max']:.4f}")

# ---- 4d. City-level historical mean waste (cross-year) ----
# Average waste generated by a city across all available years.
# Encodes each city's "baseline scale" without using future information
# during training (this will need to be recalculated split-by-split in Phase 3).
# We compute it here for EDA / reporting purposes.
city_mean_waste = (
    df.groupby("city")["waste_generated"]
    .mean()
    .rename("city_mean_waste_all")
)
df = df.merge(city_mean_waste.reset_index(), on="city", how="left")
print(f"\n  4d. city_mean_waste_all  (city avg across all available years)")
print(f"       (NOTE: data-leakage risk in ML; will be recomputed per fold in Phase 3)")

# ---- 4e. Data-quality flag for suspicious early-period values ----
# Phase 1 found Delhi (400) and Bengaluru (200) in 1999-00 are extreme low
# outliers (14.8x and 8.3x jumps to next observation).  We preserve the rows
# but mark them so models can optionally exclude or down-weight them.
dq_mask = (
    (df["fiscal_year"] == "1999-00") &
    (df["city"].isin(["Delhi", "Bengaluru"]))
)
df["dq_flag_low_early"] = dq_mask.astype(int)
flagged = df[df["dq_flag_low_early"] == 1][["fiscal_year","city","waste_generated"]]
print(f"\n  4e. dq_flag_low_early  (data-quality flag, value=1 -> suspect)")
print(flagged.to_string(index=False))

# ===========================================================================
# STEP 5 - Log-transform investigation on waste_generated
# ===========================================================================
section("STEP 5 - Log-transform of waste_generated")

wg = df["waste_generated"].dropna()

# Skewness statistics
raw_skew  = wg.skew()
log_skew  = np.log1p(wg).skew()

# Kurtosis
raw_kurt  = wg.kurtosis()
log_kurt  = np.log1p(wg).kurtosis()

# Coefficient of variation
raw_cv    = wg.std() / wg.mean()
log_cv    = np.log1p(wg).std() / np.log1p(wg).mean()

print(f"\n  {'Metric':<30}  {'Raw':>10}  {'log1p(x)':>10}")
print(f"  {'-'*30}  {'-'*10}  {'-'*10}")
print(f"  {'Skewness':<30}  {raw_skew:>10.4f}  {log_skew:>10.4f}")
print(f"  {'Kurtosis (excess)':<30}  {raw_kurt:>10.4f}  {log_kurt:>10.4f}")
print(f"  {'Coeff of Variation':<30}  {raw_cv:>10.4f}  {log_cv:>10.4f}")
print(f"  {'Min':<30}  {wg.min():>10.2f}  {np.log1p(wg.min()):>10.4f}")
print(f"  {'Max':<30}  {wg.max():>10.2f}  {np.log1p(wg.max()):>10.4f}")
print(f"  {'Range (max/min ratio)':<30}  {wg.max()/wg.min():>10.1f}x  {'n/a':>10}")

print("""
  Interpretation:
    - Raw skewness {:.2f} is very high (>1 = right-skewed).
    - After log1p, skewness drops to {:.2f}, closer to symmetry.
    - Kurtosis falls from {:.1f} to {:.1f}, indicating fewer heavy-tail effects.
    - log1p(x) is safe here because min(waste_generated) = 77 > 0.
    - log(waste_generated) is the recommended target for regression models.
    - Predictions will be back-transformed via expm1() for evaluation.
    - The dq-flagged rows (Delhi 400, Bengaluru 200 in 1999-00) become
      log(401)=5.99 and log(201)=5.30 -- still within the log-transformed
      distribution rather than extreme pull-down values.
  Decision: APPLY log1p transform. Store as 'log_waste_generated'.
""".format(raw_skew, log_skew, raw_kurt, log_kurt))

df["log_waste_generated"] = np.where(
    df["waste_generated"].notna(),
    np.log1p(df["waste_generated"]),
    np.nan
)

log_wg = df["log_waste_generated"].dropna()
print(f"  log_waste_generated stats:")
print(f"    min={log_wg.min():.4f}  p25={log_wg.quantile(0.25):.4f}"
      f"  median={log_wg.median():.4f}  mean={log_wg.mean():.4f}"
      f"  p75={log_wg.quantile(0.75):.4f}  max={log_wg.max():.4f}")

# ===========================================================================
# STEP 6 - Final column selection and ordering
# ===========================================================================
section("STEP 6 - Final column ordering")

FINAL_COLS = [
    # Identifiers
    "fiscal_year",
    "city",
    # Population anchor
    "estimated_population_in_2011",
    # Time features
    "year_start",
    "year_rank",
    "years_since_base",
    # City-level size features
    "log_population",
    "pop_tier",
    # Target (raw and log-transformed)
    "waste_generated",
    "log_waste_generated",
    # Derived from target (only for rows with observed waste)
    "waste_per_capita",
    # City historical baseline (leakage-aware; recompute in Phase 3)
    "city_mean_waste_all",
    # Quality flag
    "dq_flag_low_early",
]
df = df[FINAL_COLS]
print(f"  Final columns ({len(FINAL_COLS)}):")
for i, c in enumerate(FINAL_COLS, 1):
    nullable = "nullable" if df[c].isnull().any() else "complete"
    null_n   = df[c].isnull().sum()
    print(f"    {i:2d}. {c:<35s}  {str(df[c].dtype):<10s}  nulls={null_n}  ({nullable})")

# ===========================================================================
# STEP 7 - Save cleaned dataset
# ===========================================================================
section("STEP 7 - Save cleaned dataset")
df.to_csv(OUT_CSV, index=False)
print(f"  Written : {OUT_CSV}")
print(f"  Shape   : {df.shape[0]} rows x {df.shape[1]} columns")

# Quick verification round-trip
verify = pd.read_csv(OUT_CSV)
assert verify.shape == df.shape, "Round-trip shape mismatch!"
print("  Round-trip verification : PASSED")

# ===========================================================================
# STEP 8 - Missing-values report on cleaned dataset
# ===========================================================================
section("STEP 8 - Missing values in cleaned dataset")
mv = df.isnull().sum()
mv_pct = (mv / len(df) * 100).round(2)
print(f"\n  {'Column':<35}  {'NaN count':>10}  {'%':>8}")
print(f"  {'-'*35}  {'-'*10}  {'-'*8}")
for col in df.columns:
    print(f"  {col:<35}  {mv[col]:>10d}  {mv_pct[col]:>7.2f}%")

print(f"""
  Notes:
    - waste_generated / log_waste_generated / waste_per_capita have the
      same 107 NaN rows -- these are the original missing target observations.
    - city_mean_waste_all is populated for every row (city always has >=1 obs
      since Aurangabad -- the only all-null city -- was removed).
    - All identifier, time, and city-size features are complete (0 NaN).
""")

# ===========================================================================
# STEP 9 - Feature list and target definition summary
# ===========================================================================
section("STEP 9 - Feature / target summary")
print("""
  TARGET
  ------
  Primary   : log_waste_generated   (log1p of waste_generated tonnes/day)
  Secondary : waste_generated        (raw, for back-transform evaluation)

  FEATURES (modeling candidates)
  --------------------------------
  Time features:
    year_start          -- calendar start year of fiscal year (e.g. 2022)
    year_rank           -- ordinal index 0..8 over 9 fiscal years
    years_since_base    -- years elapsed since 1999 (0..23)

  City size / population features:
    log_population      -- log(2011 Census population); static per city
    pop_tier            -- 1/2/3 tier based on population thresholds

  Derived target features (use with care / only train-fold):
    waste_per_capita    -- kg/person/day; null where target is null
    city_mean_waste_all -- full-dataset city mean; RECOMPUTE per fold in Phase 3

  Quality / auxiliary:
    dq_flag_low_early   -- 1 for Delhi/Bengaluru 1999-00 suspect rows

  IDENTIFIERS (not features):
    fiscal_year, city, estimated_population_in_2011

  DROPPED COLUMNS:
    unit  -- constant string (zero variance)
    note  -- 100% null

  REMOVED CITY:
    Aurangabad -- 9 rows, 0 observed waste values; cannot train or evaluate

  REMAINING LIMITATIONS
  ----------------------
  1. Missing target (25.8% of rows, 107/405):
       - Concentrated in 2018-19 and 2019-20 for ~25 cities.
       - These rows are RETAINED with null target; they can be used for
         prediction (inference) but not training.
  2. Static population anchor:
       - estimated_population_in_2011 is fixed; actual populations diverge
         over the 1999-2023 span. Cannot be corrected without external data.
  3. Non-uniform time spacing:
       - Gaps between snapshots: 5yr, 6yr, 5yr, 1yr, 1yr, 1yr, 1yr, 1yr.
       - year_start encodes actual elapsed time; year_rank treats all steps
         as equal. Both are provided; Phase 3 should test both.
  4. data-quality flags (1999-00 Delhi & Bengaluru):
       - Rows are preserved with dq_flag_low_early=1.
       - Should be optionally excluded in ablation experiments.
  5. city_mean_waste_all leakage:
       - Computed over the full dataset. Must be recomputed inside
         cross-validation folds in Phase 3.
  6. No external features:
       - No GDP, urbanisation rate, collection efficiency, or waste-treatment
         data have been added (none available in the raw dataset).
""")

# ===========================================================================
# PHASE 2 REPORT SUMMARY
# ===========================================================================
section("PHASE 2 SUMMARY REPORT")
print(f"""
  ---- Rows ----
  Raw CSV rows (excluding header) : {len(raw)}
  After drop (unit, note)         : {len(raw)}     (column drop only)
  After removing Aurangabad       : {len(df)}     (-9 rows)
  Rows with observed target       : {df['waste_generated'].notna().sum()}
  Rows with null target           : {df['waste_generated'].isnull().sum()}

  ---- Columns ----
  Raw columns                     : {len(raw.columns)}   {list(raw.columns)}
  Cleaned columns                 : {len(df.columns)}  {list(df.columns)}

  ---- Transformations ----
  1. Dropped 'unit', 'note'
  2. Removed Aurangabad (9 rows, 0 target observations)
  3. fiscal_year -> year_start (int), year_rank (0-8), years_since_base (0-23)
  4. log_population = log(estimated_population_in_2011)
  5. pop_tier = 1/2/3 city-size bucket
  6. waste_per_capita = waste_generated * 1000 / population  (kg/person/day)
  7. city_mean_waste_all = per-city mean waste (all available years)
  8. dq_flag_low_early = 1 for Delhi & Bengaluru 1999-00
  9. log_waste_generated = log1p(waste_generated)  [primary model target]
     skewness: {raw.rename(columns={})['waste_generated'].dropna().skew():.3f} -> {df['log_waste_generated'].dropna().skew():.3f}

  ---- Output ----
  {OUT_CSV}

  Phase 2 complete - raw CSV untouched, no model trained.
""")
