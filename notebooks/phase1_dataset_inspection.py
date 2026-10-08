# -*- coding: utf-8 -*-
"""
Phase 1 - Dataset Inspection
WasteWise-AI  |  data/raw/india_msw_city_year.csv

Purpose : Purely inspect the raw CSV - no cleaning, imputation, or modelling.
Output  : Printed findings to stdout / log.
"""

import pathlib
import pandas as pd
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parent.parent
CSV  = ROOT / "data" / "raw" / "india_msw_city_year.csv"

SEPARATOR = "=" * 70

def section(title: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)

# ── load ───────────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV)

# ══════════════════════════════════════════════════════════════════════════════
# 1. SHAPE & COLUMNS
# ══════════════════════════════════════════════════════════════════════════════
section("1. SHAPE & COLUMNS")
print(f"  Rows    : {len(df)}")
print(f"  Columns : {df.shape[1]}")
print()
print("  Column names and dtypes:")
for col in df.columns:
    print(f"    {col:<40s}  {str(df[col].dtype):<12s}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. UNIQUE CITIES
# ══════════════════════════════════════════════════════════════════════════════
section("2. UNIQUE CITIES")
cities = sorted(df["city"].dropna().unique().tolist())
print(f"  Total unique cities : {len(cities)}")
print()
print("  City list:")
for c in cities:
    print(f"    {c}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. FISCAL-YEAR VALUES
# ══════════════════════════════════════════════════════════════════════════════
section("3. FISCAL-YEAR VALUES")
fy_vals = sorted(df["fiscal_year"].dropna().unique().tolist())
print(f"  Total unique fiscal years : {len(fy_vals)}")
print(f"  Values : {fy_vals}")

# rows-per-year quick check
print()
print("  Rows per fiscal year:")
for fy, cnt in df["fiscal_year"].value_counts().sort_index().items():
    print(f"    {fy}  ->  {cnt} rows")

# ══════════════════════════════════════════════════════════════════════════════
# 4. MISSING VALUES
# ══════════════════════════════════════════════════════════════════════════════
section("4. MISSING VALUES")
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
print(f"  {'Column':<40s}  {'NaN count':>10s}  {'%':>8s}")
print(f"  {'-'*40}  {'-'*10}  {'-'*8}")
for col in df.columns:
    print(f"  {col:<40s}  {missing[col]:>10d}  {missing_pct[col]:>7.2f}%")

# rows that are entirely null (waste_generated + note both null)
full_null_waste = df["waste_generated"].isnull().sum()
print(f"\n  Rows with missing waste_generated : {full_null_waste}"
      f"  ({full_null_waste/len(df)*100:.1f}%)")

# ══════════════════════════════════════════════════════════════════════════════
# 5. DUPLICATE ROWS
# ══════════════════════════════════════════════════════════════════════════════
section("5. DUPLICATE ROWS")
dup_all   = df.duplicated().sum()
dup_key   = df.duplicated(subset=["fiscal_year", "city"]).sum()
print(f"  Exact duplicate rows          : {dup_all}")
print(f"  Duplicate (fiscal_year, city) : {dup_key}")
if dup_key > 0:
    print("\n  Duplicate key pairs:")
    print(df[df.duplicated(subset=["fiscal_year", "city"], keep=False)]
          [["fiscal_year", "city", "waste_generated"]].to_string(index=True))

# ══════════════════════════════════════════════════════════════════════════════
# 6. WASTE-GENERATION STATISTICS
# ══════════════════════════════════════════════════════════════════════════════
section("6. WASTE-GENERATION STATISTICS  (tonnes/day, non-null values only)")
wg = df["waste_generated"].dropna()
print(f"  Non-null observations : {len(wg)}")
print()
print(f"  {'Statistic':<20s}  {'Value':>12s}")
print(f"  {'-'*20}  {'-'*12}")
stats = {
    "min"   : wg.min(),
    "p5"    : wg.quantile(0.05),
    "p25"   : wg.quantile(0.25),
    "median": wg.median(),
    "mean"  : wg.mean(),
    "p75"   : wg.quantile(0.75),
    "p95"   : wg.quantile(0.95),
    "max"   : wg.max(),
    "std"   : wg.std(),
}
for k, v in stats.items():
    print(f"  {k:<20s}  {v:>12.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# 7. SUSPICIOUS / INVALID VALUES
# ══════════════════════════════════════════════════════════════════════════════
section("7. SUSPICIOUS / INVALID VALUES")

# 7a. Zero or negative waste
nonpos = df[df["waste_generated"] <= 0]
print(f"  7a. Zero / negative waste_generated rows: {len(nonpos)}")
if len(nonpos) > 0:
    print(nonpos[["fiscal_year", "city", "waste_generated"]].to_string(index=False))

# 7b. Statistical outliers (IQR method, 3x IQR fence)
q1, q3 = wg.quantile(0.25), wg.quantile(0.75)
iqr    = q3 - q1
lower  = q1 - 3 * iqr
upper  = q3 + 3 * iqr
outliers = df[(df["waste_generated"] < lower) | (df["waste_generated"] > upper)]
print(f"\n  7b. IQR-fence outliers  (3x IQR: [{lower:.1f}, {upper:.1f}]):")
print(f"      Count : {len(outliers)}")
if len(outliers) > 0:
    print(outliers[["fiscal_year", "city", "waste_generated"]].to_string(index=False))

# 7c. Extreme year-over-year swings per city (>=5x increase or <=0.2x ratio)
section_label = "  7c. Extreme YoY swings (ratio >=5x or <=0.2x) per city:"
print(section_label)
fy_order = sorted(df["fiscal_year"].dropna().unique().tolist())
flag_rows = []
for city_name, grp in df.groupby("city"):
    grp_sorted = grp.sort_values("fiscal_year")
    prev_val = None
    prev_fy  = None
    for _, row in grp_sorted.iterrows():
        cur_val = row["waste_generated"]
        if pd.notna(cur_val) and prev_val is not None and prev_val > 0:
            ratio = cur_val / prev_val
            if ratio >= 5 or ratio <= 0.2:
                flag_rows.append({
                    "city"       : city_name,
                    "from_fy"    : prev_fy,
                    "to_fy"      : row["fiscal_year"],
                    "from_val"   : prev_val,
                    "to_val"     : cur_val,
                    "ratio"      : round(ratio, 3),
                })
        if pd.notna(cur_val):
            prev_val = cur_val
            prev_fy  = row["fiscal_year"]

if flag_rows:
    flag_df = pd.DataFrame(flag_rows)
    print(flag_df.to_string(index=False))
else:
    print("      None found.")

# 7d. Population anomalies
section("  7d. POPULATION ANOMALIES")
pop = df["estimated_population_in_2011"]
print(f"  Non-null population rows : {pop.notna().sum()}")
print(f"  Null population rows     : {pop.isnull().sum()}")
print(f"  Min  : {pop.min():,}")
print(f"  Max  : {pop.max():,}")
pop_multi = df.groupby("city")["estimated_population_in_2011"].nunique()
inconsistent_pop = pop_multi[pop_multi > 1]
print(f"\n  Cities with >1 distinct population value : {len(inconsistent_pop)}")
if len(inconsistent_pop) > 0:
    print(inconsistent_pop)

# 7e. Unexpected values in 'unit' and 'note' columns
section("  7e. UNIT & NOTE COLUMN INSPECTION")
print(f"  Unique 'unit' values  : {df['unit'].nunique()}")
print(df["unit"].value_counts(dropna=False).to_string())
print()
note_nulls = df["note"].isnull().sum()
note_vals  = df["note"].dropna().nunique()
print(f"  Null 'note' values    : {note_nulls}")
print(f"  Unique 'note' values  : {note_vals}")

# ══════════════════════════════════════════════════════════════════════════════
# 8. SAMPLE RECORDS
# ══════════════════════════════════════════════════════════════════════════════
section("8. SAMPLE RECORDS")
print("\n  First 5 rows:")
print(df.head(5).to_string(index=False))
print("\n  Last 5 rows:")
print(df.tail(5).to_string(index=False))
print("\n  5 rows with missing waste_generated:")
missing_sample = df[df["waste_generated"].isnull()].head(5)
print(missing_sample.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# 9. COVERAGE MATRIX  (city x fiscal_year)
# ══════════════════════════════════════════════════════════════════════════════
section("9. DATA-COVERAGE MATRIX  (H = has value, . = NaN)")
pivot = df.pivot_table(
    index="city", columns="fiscal_year",
    values="waste_generated", aggfunc="first",
    dropna=False,
)
# Re-add any city missing from pivot (all-NaN cities get dropped by pivot_table)
all_cities = df["city"].unique()
for c in all_cities:
    if c not in pivot.index:
        pivot.loc[c] = np.nan
# Sort cities by population descending
pop_map = df.groupby("city")["estimated_population_in_2011"].first()
pivot = pivot.loc[pop_map.sort_values(ascending=False).index]

header_fys = list(pivot.columns)
col_w = 10
print(f"\n  {'City':<25s}" + "".join(f"{fy:>{col_w}}" for fy in header_fys))
print(f"  {'-'*25}" + "-" * (col_w * len(header_fys)))
for city_name, row in pivot.iterrows():
    cells = "".join(
        f"{'H':>{col_w}}" if pd.notna(v) else f"{'.'*1:>{col_w}}"
        for v in row
    )
    print(f"  {city_name:<25s}{cells}")

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
section("PHASE 1 SUMMARY")
print(f"""
  Dataset       : {CSV}
  Total rows    : {len(df)}   (414 data rows + 1 header)
  Total columns : {df.shape[1]}
  Unique cities : {df['city'].nunique()}
  Fiscal years  : {len(fy_vals)}  ->  {fy_vals}
  Missing waste : {full_null_waste} rows  ({full_null_waste/len(df)*100:.1f}%)
  Duplicates    : {dup_all} exact  |  {dup_key} key-duplicate (city x year)
  Outliers(IQR) : {len(outliers)} rows flagged
  Zero/negative : {len(nonpos)} rows
  YoY flags     : {len(flag_rows)} city-year transitions
""")
print("  Phase 1 complete - no data was modified.")
