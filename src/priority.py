# -*- coding: utf-8 -*-
"""
src/priority.py
---------------
Collection-priority classification for WasteWise-AI.

FROZEN THRESHOLDS  (Phase 5 correction)
----------------------------------------
Thresholds are derived exclusively from the TRAINING-PERIOD observed waste
values (fiscal years 1999-00 through 2019-20; year_start < 2020; n=172).

Test-set observations are NOT used to establish thresholds.

Method: Quantile-based p33/p67 (Strategy A).
    LOW    : waste_generated <= p33  (33rd percentile of training distribution)
    MEDIUM : p33 < waste_generated <= p67
    HIGH   : waste_generated > p67

Computed values (do NOT change without retraining):
    p33 = 510.43  t/day
    p67 = 1010.0  t/day

Derivation: from 172 training-period observed rows via np.percentile(data, 33/67).

Public API
----------
THRESHOLDS                          -- frozen NamedTuple with p33, p67
classify(waste_tpd)                 -> PriorityLabel
classify_with_strategy(waste_tpd, strategy) -> PriorityLabel
label_dataframe(df, col, strategy)  -> df with priority column
build_priority_report(data)         -> pd.DataFrame  comparison (training vs full)
StrategyA                           -- quantile strategy class (kept for tests)
StrategyB                           -- Jenks class (kept for reference; not default)
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Literal, NamedTuple

import numpy as np
import pandas as pd

PriorityLabel = Literal["LOW", "MEDIUM", "HIGH"]

ROOT      = pathlib.Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "india_msw_cleaned.csv"

# ---------------------------------------------------------------------------
# FROZEN THRESHOLDS  — derived from training set (year_start < 2020, n=172)
# ---------------------------------------------------------------------------

class _Thresholds(NamedTuple):
    p33: float   # 33rd percentile of training-period waste_generated
    p67: float   # 67th percentile of training-period waste_generated
    n_train: int # number of training observations used


THRESHOLDS = _Thresholds(
    p33=510.43,
    p67=1010.0,
    n_train=172,
)
"""
Frozen priority thresholds derived from Phase 3 training period only.

    LOW    : waste_generated <= 510.43 t/day
    MEDIUM : 510.43 < waste_generated <= 1010.0 t/day
    HIGH   : waste_generated > 1010.0 t/day

Source  : data/processed/india_msw_cleaned.csv
Filter  : year_start < 2020 AND waste_generated IS NOT NULL
n       : 172 observations
Method  : np.percentile(training_waste, [33, 67])
"""

# ---------------------------------------------------------------------------
# Primary classify API (uses frozen thresholds)
# ---------------------------------------------------------------------------

def classify(waste_tpd: float) -> PriorityLabel:
    """
    Classify a waste-generation value into LOW / MEDIUM / HIGH using the
    FROZEN training-period thresholds.

    Parameters
    ----------
    waste_tpd : Predicted or observed waste in tonnes/day.

    Returns
    -------
    "LOW", "MEDIUM", or "HIGH"
    """
    if waste_tpd <= THRESHOLDS.p33:
        return "LOW"
    elif waste_tpd <= THRESHOLDS.p67:
        return "MEDIUM"
    return "HIGH"


# Priority metadata for UI display
PRIORITY_META = {
    "LOW":    {"color": "#2ecc71", "emoji": "🟢", "description": "Below 510 t/day"},
    "MEDIUM": {"color": "#f39c12", "emoji": "🟡", "description": "510 – 1,010 t/day"},
    "HIGH":   {"color": "#e74c3c", "emoji": "🔴", "description": "Above 1,010 t/day"},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_observed_waste(data_path: pathlib.Path = DATA_PATH) -> np.ndarray:
    """Return 1-D array of all non-null observed waste_generated values (t/day)."""
    df = pd.read_csv(data_path)
    return df["waste_generated"].dropna().values


def _load_training_waste(data_path: pathlib.Path = DATA_PATH) -> np.ndarray:
    """Return observed waste from training period only (year_start < 2020)."""
    df = pd.read_csv(data_path)
    return df[(df["year_start"] < 2020) & df["waste_generated"].notna()][
        "waste_generated"
    ].values


# ---------------------------------------------------------------------------
# Strategy A — quantile class (kept for backward compat and tests)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StrategyA:
    """
    Quantile-based thresholds derived from a supplied data array.

    The default production instance is THRESHOLDS (frozen training values).
    This class is retained for test parameterisation and what-if analysis.
    """
    p33: float
    p67: float

    @classmethod
    def from_data(cls, data: np.ndarray | None = None) -> "StrategyA":
        if data is None:
            data = _load_observed_waste()
        return cls(
            p33=float(np.percentile(data, 33)),
            p67=float(np.percentile(data, 67)),
        )

    @classmethod
    def training_period(cls, data_path: pathlib.Path = DATA_PATH) -> "StrategyA":
        """Construct from training-period data (year_start < 2020) — matches THRESHOLDS."""
        data = _load_training_waste(data_path)
        return cls.from_data(data)

    def classify(self, waste_tpd: float) -> PriorityLabel:
        if waste_tpd <= self.p33:
            return "LOW"
        elif waste_tpd <= self.p67:
            return "MEDIUM"
        return "HIGH"

    def thresholds(self) -> dict:
        return {
            "strategy":   "A - Quantile (p33/p67)",
            "LOW_max":     round(self.p33, 2),
            "MEDIUM_max":  round(self.p67, 2),
            "HIGH_min":    round(self.p67, 2),
        }


# ---------------------------------------------------------------------------
# Strategy B — Jenks (kept for tests; NOT the production strategy)
# ---------------------------------------------------------------------------

def _jenks_breaks(data: np.ndarray, k: int = 3) -> list[float]:
    """Fisher-Jenks natural-breaks algorithm. Returns k-1 break points."""
    data_sorted = np.sort(data)
    n = len(data_sorted)
    mat1 = np.zeros((n + 1, k + 1), dtype=int)
    mat2 = np.full((n + 1, k + 1), np.inf)
    mat2[0][0] = 0.0
    for i in range(1, n + 1):
        s1 = s2 = 0.0
        for j in range(1, i + 1):
            elt = float(data_sorted[j - 1])
            s2 += elt * elt
            s1 += elt
            count = i - j + 1
            variance = s2 - (s1 * s1) / count if count > 0 else 0.0
            for kk in range(1, k + 1):
                new_val = (mat2[j - 1][kk - 1] + variance) if j > 1 else variance
                if new_val < mat2[i][kk]:
                    mat1[i][kk] = j
                    mat2[i][kk] = new_val
    breaks = []
    kk = k
    idx = n
    while kk >= 2:
        start = mat1[idx][kk]
        if start > 1:
            breaks.append(float(data_sorted[start - 2]))
        idx = start - 1
        kk -= 1
    breaks.sort()
    return breaks


@dataclass(frozen=True)
class StrategyB:
    """
    Jenks natural-breaks thresholds.
    NOT used in production (collapses on this right-skewed dataset).
    Retained for tests and future reference.
    """
    break1: float
    break2: float

    @classmethod
    def from_data(cls, data: np.ndarray | None = None) -> "StrategyB":
        if data is None:
            data = _load_observed_waste()
        breaks = _jenks_breaks(data, k=3)
        if len(breaks) < 2:
            m, s = data.mean(), data.std()
            breaks = [max(data.min(), m - 0.5 * s), min(data.max(), m + 0.5 * s)]
        return cls(break1=float(breaks[0]), break2=float(breaks[1]))

    def classify(self, waste_tpd: float) -> PriorityLabel:
        if waste_tpd <= self.break1:
            return "LOW"
        elif waste_tpd <= self.break2:
            return "MEDIUM"
        return "HIGH"

    def thresholds(self) -> dict:
        return {
            "strategy":   "B - Jenks natural breaks",
            "LOW_max":     round(self.break1, 1),
            "MEDIUM_max":  round(self.break2, 1),
            "HIGH_min":    round(self.break2, 1),
        }


# ---------------------------------------------------------------------------
# Strategy-agnostic classify API (used by tests)
# ---------------------------------------------------------------------------

def classify_with_strategy(
    waste_tpd: float,
    strategy: StrategyA | StrategyB,
) -> PriorityLabel:
    """Classify using the supplied strategy object."""
    return strategy.classify(waste_tpd)


def label_dataframe(
    df: pd.DataFrame,
    waste_col: str,
    strategy: StrategyA | StrategyB | None = None,
    out_col: str = "priority",
) -> pd.DataFrame:
    """
    Add a priority column to df.
    If strategy is None, uses the frozen THRESHOLDS (production default).
    Rows where waste_col is NaN are labelled 'UNKNOWN'.
    """
    df = df.copy()
    if strategy is None:
        fn = classify
    else:
        fn = strategy.classify
    df[out_col] = df[waste_col].apply(
        lambda v: fn(v) if pd.notna(v) else "UNKNOWN"
    )
    return df


# ---------------------------------------------------------------------------
# Distribution report builder
# ---------------------------------------------------------------------------

def build_priority_report(
    data: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Build a DataFrame comparing the frozen training-period thresholds applied
    to (a) training data and (b) full observed data.
    """
    train_data = _load_training_waste()
    full_data   = _load_observed_waste() if data is None else data

    rows = []
    for label_str, arr in [("Training period (n=172)", train_data),
                            ("Full observed (n=307)",   full_data)]:
        n = len(arr)
        counts = {
            "LOW":    int((arr <= THRESHOLDS.p33).sum()),
            "MEDIUM": int(((arr > THRESHOLDS.p33) & (arr <= THRESHOLDS.p67)).sum()),
            "HIGH":   int((arr > THRESHOLDS.p67).sum()),
        }
        rows.append({
            "dataset":      label_str,
            "LOW_threshold":  THRESHOLDS.p33,
            "HIGH_threshold": THRESHOLDS.p67,
            "LOW_count":    counts["LOW"],
            "MEDIUM_count": counts["MEDIUM"],
            "HIGH_count":   counts["HIGH"],
            "LOW_pct":      round(counts["LOW"]    / n * 100, 1),
            "MEDIUM_pct":   round(counts["MEDIUM"] / n * 100, 1),
            "HIGH_pct":     round(counts["HIGH"]   / n * 100, 1),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Frozen thresholds : LOW <= {THRESHOLDS.p33}  |  MEDIUM <= {THRESHOLDS.p67}  |  HIGH > {THRESHOLDS.p67}")
    print(f"  from n={THRESHOLDS.n_train} training-period observations")
    print()
    print(build_priority_report().to_string(index=False))
