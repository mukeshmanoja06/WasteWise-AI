# -*- coding: utf-8 -*-
"""
src/analytics.py
----------------
Historical analytics and model-performance summary functions for WasteWise-AI.

All functions return plain Python dicts or pandas DataFrames — no UI
dependencies so they can be consumed by any frontend (dashboard, CLI, API).

Public API
----------
city_trend(city)                    -> pd.DataFrame  rows per fiscal year
city_comparison(cities, year)       -> pd.DataFrame  multi-city snapshot
waste_by_year()                     -> pd.DataFrame  total/mean per fiscal year
top_cities(n, year)                 -> pd.DataFrame  highest-waste cities
bottom_cities(n, year)              -> pd.DataFrame  lowest-waste cities
predicted_vs_historical(city)       -> pd.DataFrame  historical + model predictions
model_performance_summary()         -> dict           MAE, RMSE, R2 per model
feature_importance_table()          -> pd.DataFrame  feature importance
load_cleaned_data()                 -> pd.DataFrame  full cleaned dataset
load_model_results()                -> pd.DataFrame  phase3_model_results.csv
"""

from __future__ import annotations

import pathlib
from typing import Optional

import numpy as np
import pandas as pd

ROOT       = pathlib.Path(__file__).resolve().parent.parent
DATA_PATH  = ROOT / "data" / "processed" / "india_msw_cleaned.csv"
RES_PATH   = ROOT / "reports" / "phase3_model_results.csv"
FI_PATH    = ROOT / "reports" / "phase3_feature_importance.csv"


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_cleaned_data(path: pathlib.Path = DATA_PATH) -> pd.DataFrame:
    """Load the Phase-2 cleaned dataset."""
    if not path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found at {path}. Run Phase 2 script first."
        )
    return pd.read_csv(path)


def load_model_results(path: pathlib.Path = RES_PATH) -> pd.DataFrame:
    """Load the Phase-3 model comparison results."""
    if not path.exists():
        raise FileNotFoundError(
            f"Model results not found at {path}. Run Phase 3 script first."
        )
    return pd.read_csv(path)


def load_feature_importance(path: pathlib.Path = FI_PATH) -> pd.DataFrame:
    """Load Phase-3 feature importance CSV."""
    if not path.exists():
        raise FileNotFoundError(
            f"Feature importance file not found at {path}. "
            "Run Phase 3 script first."
        )
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# 1. City historical waste trend
# ---------------------------------------------------------------------------

def city_trend(
    city: str,
    df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Return the observed waste generation trend for a single city.

    Returns a DataFrame with columns:
        fiscal_year, year_start, waste_generated, log_waste_generated,
        waste_per_capita, waste_generated_is_observed (bool)

    Rows with null target are included with a flag so callers can distinguish
    observed from missing years.
    """
    if df is None:
        df = load_cleaned_data()

    city_df = df[df["city"] == city].sort_values("year_start").copy()
    if city_df.empty:
        raise ValueError(f"City '{city}' not found in the dataset.")

    city_df["waste_generated_is_observed"] = city_df["waste_generated"].notna()
    return city_df[[
        "fiscal_year", "year_start", "waste_generated",
        "log_waste_generated", "waste_per_capita",
        "pop_tier", "estimated_population_in_2011",
        "waste_generated_is_observed",
    ]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. City comparison
# ---------------------------------------------------------------------------

def city_comparison(
    cities: Optional[list[str]] = None,
    fiscal_year: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Compare multiple cities for a given fiscal year (or latest available).

    Parameters
    ----------
    cities      : List of city names. If None, all cities are included.
    fiscal_year : e.g. '2022-23'. If None, uses the most recent available year.

    Returns a DataFrame sorted by waste_generated descending.
    """
    if df is None:
        df = load_cleaned_data()

    if fiscal_year is None:
        fiscal_year = df.sort_values("year_start")["fiscal_year"].iloc[-1]

    sub = df[df["fiscal_year"] == fiscal_year].copy()
    if cities is not None:
        sub = sub[sub["city"].isin(cities)]

    return sub[[
        "city", "fiscal_year", "estimated_population_in_2011",
        "pop_tier", "waste_generated", "waste_per_capita",
    ]].sort_values("waste_generated", ascending=False, na_position="last").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Waste generation by year
# ---------------------------------------------------------------------------

def waste_by_year(df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Aggregate waste generation statistics by fiscal year.

    Returns a DataFrame with:
        fiscal_year, year_start, n_cities_observed, total_waste_tpd,
        mean_waste_tpd, median_waste_tpd, min_waste_tpd, max_waste_tpd
    """
    if df is None:
        df = load_cleaned_data()

    observed = df[df["waste_generated"].notna()]
    agg = (
        observed.groupby(["fiscal_year", "year_start"])["waste_generated"]
        .agg(
            n_cities_observed="count",
            total_waste_tpd="sum",
            mean_waste_tpd="mean",
            median_waste_tpd="median",
            min_waste_tpd="min",
            max_waste_tpd="max",
        )
        .reset_index()
        .sort_values("year_start")
    )
    agg["total_waste_tpd"]  = agg["total_waste_tpd"].round(1)
    agg["mean_waste_tpd"]   = agg["mean_waste_tpd"].round(1)
    agg["median_waste_tpd"] = agg["median_waste_tpd"].round(1)
    return agg


# ---------------------------------------------------------------------------
# 4. Top N highest-waste cities
# ---------------------------------------------------------------------------

def top_cities(
    n: int = 10,
    fiscal_year: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Return the top-N cities by observed waste generation for a given year.
    Uses the most recent fiscal year if fiscal_year is None.
    """
    if df is None:
        df = load_cleaned_data()

    if fiscal_year is None:
        # Use latest year that has at least some data
        fy_counts = (
            df[df["waste_generated"].notna()]
            .groupby("fiscal_year")["city"].count()
            .sort_index()
        )
        fiscal_year = fy_counts.index[-1]

    sub = (
        df[(df["fiscal_year"] == fiscal_year) & df["waste_generated"].notna()]
        .sort_values("waste_generated", ascending=False)
        .head(n)
        [["city", "fiscal_year", "estimated_population_in_2011",
          "pop_tier", "waste_generated", "waste_per_capita"]]
        .reset_index(drop=True)
    )
    return sub


# ---------------------------------------------------------------------------
# 5. Bottom N lowest-waste cities (where meaningful)
# ---------------------------------------------------------------------------

def bottom_cities(
    n: int = 10,
    fiscal_year: Optional[str] = None,
    min_population: int = 0,
    df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Return the bottom-N cities by observed waste generation for a given year.
    Only cities with observed (non-null) waste are included.

    Parameters
    ----------
    min_population : Optionally filter to cities above this population size
                     (avoids comparing data-sparse cities).
    """
    if df is None:
        df = load_cleaned_data()

    if fiscal_year is None:
        fy_counts = (
            df[df["waste_generated"].notna()]
            .groupby("fiscal_year")["city"].count()
            .sort_index()
        )
        fiscal_year = fy_counts.index[-1]

    mask = (
        (df["fiscal_year"] == fiscal_year)
        & df["waste_generated"].notna()
        & (df["estimated_population_in_2011"] >= min_population)
    )
    sub = (
        df[mask]
        .sort_values("waste_generated", ascending=True)
        .head(n)
        [["city", "fiscal_year", "estimated_population_in_2011",
          "pop_tier", "waste_generated", "waste_per_capita"]]
        .reset_index(drop=True)
    )
    return sub


# ---------------------------------------------------------------------------
# 6. Predicted vs historical values for a city
# ---------------------------------------------------------------------------

def predicted_vs_historical(
    city: str,
    population_2011: Optional[int] = None,
    years: Optional[list[int]] = None,
    df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Return a DataFrame combining historical observed values with model
    predictions for every year in the dataset.

    If population_2011 is None, the value is inferred from the cleaned dataset.
    If years is None, all years in the dataset plus a 2023 projection are used.

    Note: model is loaded lazily via prediction.predict_waste.
    """
    from src.prediction import predict_waste  # lazy import to avoid circular

    if df is None:
        df = load_cleaned_data()

    city_df = df[df["city"] == city].sort_values("year_start").copy()
    if city_df.empty:
        raise ValueError(f"City '{city}' not found in the dataset.")

    if population_2011 is None:
        population_2011 = int(city_df["estimated_population_in_2011"].iloc[0])

    if years is None:
        years = sorted(city_df["year_start"].tolist())

    # Build predictions for every year in scope
    preds = {}
    for yr in years:
        try:
            r = predict_waste(city, population_2011, yr)
            preds[yr] = r.prediction_tpd
        except Exception:
            preds[yr] = np.nan

    # Merge
    hist = city_df.set_index("year_start")[["fiscal_year", "waste_generated"]].copy()
    result_rows = []
    for yr in years:
        fy  = hist.loc[yr, "fiscal_year"] if yr in hist.index else f"{yr}-{str(yr+1)[-2:]}"
        obs = float(hist.loc[yr, "waste_generated"]) if yr in hist.index and pd.notna(hist.loc[yr, "waste_generated"]) else np.nan
        result_rows.append({
            "fiscal_year":    fy,
            "year_start":     yr,
            "observed_tpd":   obs,
            "predicted_tpd":  preds.get(yr, np.nan),
            "residual":       (obs - preds.get(yr, np.nan)) if pd.notna(obs) and pd.notna(preds.get(yr)) else np.nan,
        })

    return pd.DataFrame(result_rows)


# ---------------------------------------------------------------------------
# 7. Model performance summary
# ---------------------------------------------------------------------------

def model_performance_summary(path: pathlib.Path = RES_PATH) -> dict:
    """
    Return a dict summarising Phase-3 model comparison metrics.

    Keys: model_names, MAE_t, RMSE_t, R2_t, best_model
    """
    results = load_model_results(path)
    best_idx = results[results["label"] != "Dummy (mean)"]["RMSE_t"].idxmin()
    best_row = results.loc[best_idx]
    return {
        "model_names": results["label"].tolist(),
        "MAE_t":       results["MAE_t"].round(1).tolist(),
        "RMSE_t":      results["RMSE_t"].round(1).tolist(),
        "R2_t":        results["R2_t"].round(4).tolist(),
        "MAPE_pct":    results["MAPE_pct"].round(2).tolist(),
        "best_model":  best_row["label"],
        "best_RMSE_t": round(float(best_row["RMSE_t"]), 1),
        "best_MAE_t":  round(float(best_row["MAE_t"]), 1),
        "best_R2_t":   round(float(best_row["R2_t"]), 4),
    }


# ---------------------------------------------------------------------------
# 8. Feature importance
# ---------------------------------------------------------------------------

def feature_importance_table(
    top_n: int = 20,
    path: pathlib.Path = FI_PATH,
) -> pd.DataFrame:
    """Return top-N features by importance from Phase-3 model."""
    fi = load_feature_importance(path)
    return (
        fi.sort_values("importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== waste_by_year ===")
    print(waste_by_year().to_string(index=False))

    print("\n=== top_cities (2022-23) ===")
    print(top_cities(n=10, fiscal_year="2022-23").to_string(index=False))

    print("\n=== bottom_cities (2022-23) ===")
    print(bottom_cities(n=10, fiscal_year="2022-23").to_string(index=False))

    print("\n=== city_trend (Mumbai) ===")
    print(city_trend("Mumbai").to_string(index=False))

    print("\n=== model_performance_summary ===")
    import json
    print(json.dumps(model_performance_summary(), indent=2))

    print("\n=== feature_importance_table ===")
    print(feature_importance_table(top_n=10).to_string(index=False))
