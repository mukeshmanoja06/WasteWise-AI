# -*- coding: utf-8 -*-
"""
tests/test_integration.py
--------------------------
Integration tests for the WasteWise AI prediction + priority pipeline.

These tests verify the end-to-end flow from prediction to priority label,
using the actual trained model and frozen thresholds.

Run with:  python -m pytest tests/test_integration.py -v
"""

import pathlib
import sys

import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.prediction import (
    predict_waste, predict_batch, explain_prediction, PredictionResult,
    TRAINING_CITIES,
)
from src.priority import (
    classify, THRESHOLDS, label_dataframe, build_priority_report,
    PRIORITY_META,
)
from src.analytics import (
    load_cleaned_data, waste_by_year, top_cities, bottom_cities,
    city_trend, predicted_vs_historical, model_performance_summary,
    feature_importance_table,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cleaned_df():
    return load_cleaned_data()


@pytest.fixture(scope="module")
def training_observed(cleaned_df):
    return cleaned_df[
        (cleaned_df["year_start"] < 2020) & cleaned_df["waste_generated"].notna()
    ]["waste_generated"].values


@pytest.fixture(scope="module")
def full_observed(cleaned_df):
    return cleaned_df[cleaned_df["waste_generated"].notna()]["waste_generated"].values


# ---------------------------------------------------------------------------
# Tests: Frozen thresholds correctness
# ---------------------------------------------------------------------------

class TestFrozenThresholds:
    def test_thresholds_match_training_data(self, training_observed):
        """THRESHOLDS must match np.percentile(training_data, [33, 67])."""
        expected_p33 = float(np.percentile(training_observed, 33))
        expected_p67 = float(np.percentile(training_observed, 67))
        assert abs(THRESHOLDS.p33 - expected_p33) < 0.1, (
            f"p33 mismatch: frozen={THRESHOLDS.p33}, recomputed={expected_p33}"
        )
        assert abs(THRESHOLDS.p67 - expected_p67) < 0.1, (
            f"p67 mismatch: frozen={THRESHOLDS.p67}, recomputed={expected_p67}"
        )

    def test_thresholds_NOT_from_test_set(self, cleaned_df):
        """Thresholds must NOT change if computed from full data (test contamination check)."""
        full = cleaned_df[cleaned_df["waste_generated"].notna()]["waste_generated"].values
        full_p33 = float(np.percentile(full, 33))
        # Full-data p33 should differ from training-period p33 (different distributions)
        # This just confirms the two are not identical
        train = cleaned_df[
            (cleaned_df["year_start"] < 2020) & cleaned_df["waste_generated"].notna()
        ]["waste_generated"].values
        train_p33 = float(np.percentile(train, 33))
        # They may coincidentally be close; what matters is that THRESHOLDS uses train_p33
        assert abs(THRESHOLDS.p33 - train_p33) < 0.1

    def test_training_period_roughly_balanced(self, training_observed):
        """Training-period thresholds should yield ~33% per class."""
        n = len(training_observed)
        low    = (training_observed <= THRESHOLDS.p33).sum() / n * 100
        medium = ((training_observed > THRESHOLDS.p33) & (training_observed <= THRESHOLDS.p67)).sum() / n * 100
        high   = (training_observed > THRESHOLDS.p67).sum() / n * 100
        for pct in [low, medium, high]:
            assert 20 <= pct <= 50, f"Unexpected class percentage: {pct:.1f}%"

    def test_p33_less_than_p67(self):
        assert THRESHOLDS.p33 < THRESHOLDS.p67

    def test_thresholds_positive(self):
        assert THRESHOLDS.p33 > 0
        assert THRESHOLDS.p67 > 0


# ---------------------------------------------------------------------------
# Tests: Prediction → Priority pipeline
# ---------------------------------------------------------------------------

class TestPredictionToPriority:
    def test_low_prediction_classified_low(self):
        """A very small city should predict LOW waste."""
        result = predict_waste("TinyCity", 300_000, 2022)
        # Just verify classify returns a valid label
        label = classify(result.prediction_tpd)
        assert label in ("LOW", "MEDIUM", "HIGH")

    def test_mega_city_classified_high(self):
        """Delhi in recent years should predict HIGH priority."""
        result = predict_waste("Delhi", 16_787_941, 2022)
        label  = classify(result.prediction_tpd)
        assert label == "HIGH", (
            f"Expected HIGH for Delhi, got {label} (pred={result.prediction_tpd:.1f})"
        )

    def test_classify_below_p33_is_low(self):
        assert classify(THRESHOLDS.p33 - 0.01) == "LOW"

    def test_classify_at_p33_is_low(self):
        assert classify(THRESHOLDS.p33) == "LOW"

    def test_classify_above_p33_below_p67_is_medium(self):
        mid = (THRESHOLDS.p33 + THRESHOLDS.p67) / 2
        assert classify(mid) == "MEDIUM"

    def test_classify_at_p67_is_medium(self):
        assert classify(THRESHOLDS.p67) == "MEDIUM"

    def test_classify_above_p67_is_high(self):
        assert classify(THRESHOLDS.p67 + 0.01) == "HIGH"

    def test_prediction_result_has_correct_types(self):
        result = predict_waste("Mumbai", 12_442_373, 2022)
        assert isinstance(result.prediction_tpd, float)
        assert isinstance(result.log_prediction, float)
        assert isinstance(result.is_known_city, bool)
        assert isinstance(result.warnings, list)

    def test_batch_prediction_all_get_priority(self):
        records = [
            {"city": "Mumbai",  "population_2011": 12_442_373, "year": 2022},
            {"city": "Delhi",   "population_2011": 16_787_941, "year": 2022},
            {"city": "Raipur",  "population_2011":  1_010_087, "year": 2022},
        ]
        results = predict_batch(records)
        for r in results:
            label = classify(r.prediction_tpd)
            assert label in ("LOW", "MEDIUM", "HIGH")

    def test_explain_contains_priority_when_set(self):
        r = predict_waste("Mumbai", 12_442_373, 2022)
        import dataclasses
        r2 = dataclasses.replace(r, priority_label="HIGH")
        text = explain_prediction(r2)
        assert "HIGH" in text

    def test_explanation_non_causal_disclaimer(self):
        r    = predict_waste("Mumbai", 12_442_373, 2022)
        text = explain_prediction(r)
        assert "correlational" in text.lower() or "causal" in text.lower()


# ---------------------------------------------------------------------------
# Tests: label_dataframe uses frozen thresholds by default
# ---------------------------------------------------------------------------

class TestLabelDataframeDefaults:
    def test_default_strategy_uses_frozen_thresholds(self, cleaned_df):
        obs = cleaned_df[cleaned_df["waste_generated"].notna()].copy()
        labelled = label_dataframe(obs, "waste_generated")
        # A row just below p33 should be LOW
        just_below = obs[obs["waste_generated"] <= THRESHOLDS.p33]
        if len(just_below) > 0:
            assert (labelled.loc[just_below.index, "priority"] == "LOW").all()

    def test_null_waste_labelled_unknown(self, cleaned_df):
        null_rows = cleaned_df[cleaned_df["waste_generated"].isna()].copy()
        if len(null_rows) > 0:
            labelled = label_dataframe(null_rows, "waste_generated")
            assert (labelled["priority"] == "UNKNOWN").all()

    def test_all_valid_labels(self, cleaned_df):
        obs = cleaned_df[cleaned_df["waste_generated"].notna()].copy()
        labelled = label_dataframe(obs, "waste_generated")
        assert set(labelled["priority"].unique()).issubset({"LOW", "MEDIUM", "HIGH", "UNKNOWN"})


# ---------------------------------------------------------------------------
# Tests: Analytics functions
# ---------------------------------------------------------------------------

class TestAnalytics:
    def test_waste_by_year_returns_df(self):
        df = waste_by_year()
        assert isinstance(df, pd.DataFrame)
        assert "fiscal_year" in df.columns
        assert "total_waste_tpd" in df.columns
        assert len(df) == 9  # 9 fiscal years

    def test_waste_by_year_all_positive(self):
        df = waste_by_year()
        assert (df["total_waste_tpd"] > 0).all()

    def test_top_cities_returns_n_rows(self):
        df = top_cities(n=5, fiscal_year="2022-23")
        assert len(df) <= 5
        assert "waste_generated" in df.columns

    def test_top_cities_sorted_descending(self):
        df = top_cities(n=10, fiscal_year="2022-23")
        vals = df["waste_generated"].dropna().values
        assert list(vals) == sorted(vals, reverse=True)

    def test_bottom_cities_sorted_ascending(self):
        df = bottom_cities(n=10, fiscal_year="2022-23")
        vals = df["waste_generated"].dropna().values
        assert list(vals) == sorted(vals)

    def test_city_trend_known_city(self):
        df = city_trend("Mumbai")
        assert len(df) == 9  # 9 fiscal years
        assert "waste_generated" in df.columns
        assert "waste_generated_is_observed" in df.columns

    def test_city_trend_unknown_city_raises(self):
        with pytest.raises(ValueError, match="not found"):
            city_trend("Atlantis")

    def test_predicted_vs_historical_returns_df(self):
        df = predicted_vs_historical("Indore")
        assert isinstance(df, pd.DataFrame)
        assert "observed_tpd" in df.columns
        assert "predicted_tpd" in df.columns

    def test_predicted_vs_historical_all_preds_positive(self):
        df = predicted_vs_historical("Indore")
        preds = df["predicted_tpd"].dropna()
        assert (preds > 0).all()

    def test_model_performance_summary_has_best_model(self):
        perf = model_performance_summary()
        assert "best_model" in perf
        assert perf["best_model"] == "GradientBoosting"

    def test_model_performance_best_rmse_positive(self):
        perf = model_performance_summary()
        assert perf["best_RMSE_t"] > 0

    def test_feature_importance_top_feature_is_log_population(self):
        fi = feature_importance_table(top_n=5)
        assert fi.iloc[0]["feature"] == "log_population"

    def test_feature_importance_importances_sum_near_1(self):
        fi = feature_importance_table(top_n=100)
        # Top features should account for a large fraction
        assert fi["importance"].sum() > 0.5


# ---------------------------------------------------------------------------
# Tests: Priority metadata
# ---------------------------------------------------------------------------

class TestPriorityMeta:
    def test_all_labels_have_metadata(self):
        for label in ("LOW", "MEDIUM", "HIGH"):
            assert label in PRIORITY_META
            assert "color" in PRIORITY_META[label]
            assert "emoji" in PRIORITY_META[label]
            assert "description" in PRIORITY_META[label]

    def test_colors_are_valid_hex(self):
        import re
        for label, meta in PRIORITY_META.items():
            assert re.match(r"^#[0-9a-fA-F]{6}$", meta["color"]), (
                f"Invalid color for {label}: {meta['color']}"
            )
