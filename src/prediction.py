# -*- coding: utf-8 -*-
"""
src/prediction.py
-----------------
Reusable prediction module for WasteWise-AI.

Accepts city, population, and year; generates the same features used in
Phase 3; returns predicted waste generation in tonnes/day.

Public API
----------
predict_waste(city, population_2011, year)  -> PredictionResult
explain_prediction(result)                  -> str
load_model(path)                            -> sklearn Pipeline
"""

from __future__ import annotations

import math
import pathlib
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = pathlib.Path(__file__).resolve().parent.parent
_MODEL_PATH = ROOT / "models" / "best_model.joblib"
_DATA_PATH  = ROOT / "data" / "processed" / "india_msw_cleaned.csv"

# ---------------------------------------------------------------------------
# Population-tier logic (must match Phase 2 exactly)
# ---------------------------------------------------------------------------
def _pop_tier(population: int) -> int:
    """Return city size tier: 1=mega(>4M), 2=large(2-4M), 3=medium(<2M)."""
    if population > 4_000_000:
        return 1
    elif population > 2_000_000:
        return 2
    return 3

_TIER_LABELS = {1: "Mega (>4M)", 2: "Large (2-4M)", 3: "Medium (<2M)"}

# ---------------------------------------------------------------------------
# Cities seen during training  (42 cities with observed data before 2020-21)
# ---------------------------------------------------------------------------
TRAINING_CITIES = frozenset([
    "Agra", "Ahmedabad", "Allahabad", "Amritsar", "Bengaluru", "Bhopal",
    "Chennai", "Coimbatore", "Delhi", "Dhanbad", "Faridabad", "Gwalior",
    "Howrah", "Hyderabad", "Indore", "Jabalpur", "Jaipur",
    "Kalyan-dombivali", "Kanpur", "Kolkata", "Lucknow", "Ludhiana",
    "Madurai", "Meerut", "Mumbai", "Nagpur", "Nashik", "Navi Mumbai",
    "Patna", "Pimpri-chinchwad", "Pune", "Raipur", "Rajkot", "Ranchi",
    "Srinagar", "Surat", "Thane", "Vadodara", "Varanasi", "Vasai-virar",
    "Vijayawada", "Vishakhapatnam",
])

# Cities present in the cleaned dataset but NOT in training
#   (appeared only from 2020-21 onward)
DATASET_CITIES = frozenset([
    "Gaziabad", "Jodhpur", "Kota",
]) | TRAINING_CITIES


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class PredictionResult:
    """Holds a single waste-generation prediction with metadata."""
    city:            str
    population_2011: int
    year:            int
    pop_tier:        int
    log_population:  float
    log_prediction:  float                   # log1p space (model output)
    prediction_tpd:  float                   # back-transformed tonnes/day
    is_known_city:   bool                    # True if city was seen in training
    warnings:        list[str] = field(default_factory=list)

    # Set after priority classification (optional, filled by priority.py)
    priority_label:  Optional[str] = None
    priority_score:  Optional[float] = None


# ---------------------------------------------------------------------------
# Model loader (cached after first call)
# ---------------------------------------------------------------------------
_CACHED_MODEL = None

def load_model(path: pathlib.Path | str = _MODEL_PATH):
    """Load and cache the best_model.joblib pipeline."""
    global _CACHED_MODEL
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run Phase 3 training script first."
        )
    _CACHED_MODEL = joblib.load(path)
    return _CACHED_MODEL


def _get_model():
    global _CACHED_MODEL
    if _CACHED_MODEL is None:
        _CACHED_MODEL = load_model()
    return _CACHED_MODEL


# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------
def _build_feature_row(city: str, population_2011: int, year: int) -> pd.DataFrame:
    """
    Build a single-row DataFrame with the exact features used in Phase 3.

    Features:
        year_start      -- calendar year (e.g. 2024)
        log_population  -- log(population_2011)
        pop_tier        -- 1/2/3
        city            -- raw string (OHE handled inside Pipeline)
    """
    row = {
        "year_start":     year,
        "log_population": math.log(population_2011),
        "pop_tier":       _pop_tier(population_2011),
        "city":           city,
    }
    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# Main prediction function
# ---------------------------------------------------------------------------
def predict_waste(
    city:            str,
    population_2011: int,
    year:            int,
    model_path:      pathlib.Path | str = _MODEL_PATH,
) -> PredictionResult:
    """
    Predict municipal solid waste generation for a given city and year.

    Parameters
    ----------
    city            : City name string.
    population_2011 : 2011 Census population (or closest available estimate).
    year            : Calendar year of the START of the fiscal year (e.g. 2024
                      for fiscal year 2024-25).
    model_path      : Path to the saved model joblib file.

    Returns
    -------
    PredictionResult with predicted tonnes/day and metadata.
    """
    # --- Input validation ---
    if population_2011 <= 0:
        raise ValueError(f"population_2011 must be positive, got {population_2011}")
    if year < 1990 or year > 2100:
        raise ValueError(f"year {year} seems implausible (expected 1990-2100)")

    model = load_model(model_path) if model_path != _MODEL_PATH else _get_model()

    row = _build_feature_row(city, population_2011, year)
    log_pred = float(model.predict(row)[0])
    pred_tpd = float(np.expm1(log_pred))

    # Build warnings
    warns = []
    is_known = city in TRAINING_CITIES
    if not is_known:
        warns.append(
            f"City '{city}' was not seen during model training. "
            "Prediction relies on population and year trend only; "
            "no city-specific adjustment applied."
        )
    if year < 1999 or year > 2030:
        warns.append(
            f"Year {year} is outside the observed training range "
            "(1999-2022). Extrapolation uncertainty is higher."
        )
    if population_2011 < 500_000:
        warns.append(
            "Population below 500,000 is outside the training distribution. "
            "Prediction confidence is lower."
        )

    return PredictionResult(
        city=city,
        population_2011=population_2011,
        year=year,
        pop_tier=_pop_tier(population_2011),
        log_population=math.log(population_2011),
        log_prediction=log_pred,
        prediction_tpd=round(pred_tpd, 2),
        is_known_city=is_known,
        warnings=warns,
    )


# ---------------------------------------------------------------------------
# Batch prediction
# ---------------------------------------------------------------------------
def predict_batch(
    records: list[dict],
    model_path: pathlib.Path | str = _MODEL_PATH,
) -> list[PredictionResult]:
    """
    Predict for a list of dicts, each with keys: city, population_2011, year.
    Returns a list of PredictionResult objects.
    """
    return [
        predict_waste(
            city=r["city"],
            population_2011=r["population_2011"],
            year=r["year"],
            model_path=model_path,
        )
        for r in records
    ]


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------
def explain_prediction(result: PredictionResult) -> str:
    """
    Return a plain-language explanation of a prediction.

    Describes the main drivers without claiming causality from feature
    importance alone.
    """
    tier_label   = _TIER_LABELS.get(result.pop_tier, "Unknown")
    city_note    = (
        "City-specific historical adjustment applied (city seen in training)."
        if result.is_known_city
        else "No city-specific adjustment (city not seen during training). "
             "Prediction based on population scale and year trend only."
    )

    lines = [
        "=" * 56,
        f"  Waste generation prediction",
        "=" * 56,
        f"  City              : {result.city}",
        f"  Year              : {result.year}",
        f"  Population (2011) : {result.population_2011:,}",
        f"  City size tier    : {result.pop_tier} - {tier_label}",
        "",
        f"  Predicted waste   : {result.prediction_tpd:,.1f} tonnes/day",
        f"  (log-space value  : {result.log_prediction:.4f})",
        "",
        "  Main drivers:",
        f"    1. Population scale   -- log(pop) = {result.log_population:.3f}. "
             f"Larger cities generate proportionally more waste "
             f"(accounts for ~70% of model importance).",
        f"    2. Year / time trend  -- year_start = {result.year}. "
             f"Waste generation has generally trended upward "
             f"over time across Indian cities (~16% of importance).",
        f"    3. {city_note}",
    ]

    if result.priority_label:
        lines += [
            "",
            f"  Collection priority : {result.priority_label}",
        ]

    if result.warnings:
        lines += ["", "  Warnings:"]
        for w in result.warnings:
            lines.append(f"    - {w}")

    lines.append("=" * 56)
    lines.append(
        "  NOTE: Drivers described by feature importance are "
        "correlational, not causal."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    examples = [
        {"city": "Mumbai",    "population_2011": 12_442_373, "year": 2024},
        {"city": "Delhi",     "population_2011": 16_787_941, "year": 2024},
        {"city": "Indore",    "population_2011":  1_960_631, "year": 2024},
        {"city": "NewCity",   "population_2011":  2_500_000, "year": 2024},
    ]
    for ex in examples:
        r = predict_waste(**ex)
        print(explain_prediction(r))
        print()
