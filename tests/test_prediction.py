# -*- coding: utf-8 -*-
"""
tests/test_prediction.py
------------------------
Unit tests for src/prediction.py.

Run with:  python -m pytest tests/test_prediction.py -v
"""

import math
import pathlib
import sys

import pytest
import numpy as np
import pandas as pd

# Ensure src/ is importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.prediction import (
    predict_waste,
    predict_batch,
    explain_prediction,
    load_model,
    _pop_tier,
    _build_feature_row,
    PredictionResult,
    TRAINING_CITIES,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

KNOWN_CITY   = "Mumbai"
KNOWN_POP    = 12_442_373
UNKNOWN_CITY = "Atlantis"
UNKNOWN_POP  = 2_000_000
TEST_YEAR    = 2023


@pytest.fixture(scope="module")
def known_result():
    return predict_waste(KNOWN_CITY, KNOWN_POP, TEST_YEAR)


@pytest.fixture(scope="module")
def unknown_result():
    return predict_waste(UNKNOWN_CITY, UNKNOWN_POP, TEST_YEAR)


# ---------------------------------------------------------------------------
# Tests: _pop_tier helper
# ---------------------------------------------------------------------------

class TestPopTier:
    def test_mega_city(self):
        assert _pop_tier(12_000_000) == 1

    def test_large_city(self):
        assert _pop_tier(3_000_000) == 2

    def test_medium_city(self):
        assert _pop_tier(1_500_000) == 3

    def test_boundary_tier1(self):
        assert _pop_tier(4_000_001) == 1

    def test_boundary_tier2_lower(self):
        assert _pop_tier(2_000_001) == 2

    def test_boundary_tier3(self):
        assert _pop_tier(2_000_000) == 3


# ---------------------------------------------------------------------------
# Tests: _build_feature_row
# ---------------------------------------------------------------------------

class TestBuildFeatureRow:
    def test_returns_dataframe(self):
        row = _build_feature_row("Delhi", 16_787_941, 2023)
        assert isinstance(row, pd.DataFrame)
        assert len(row) == 1

    def test_correct_columns(self):
        row = _build_feature_row("Delhi", 16_787_941, 2023)
        assert set(row.columns) == {"year_start", "log_population", "pop_tier", "city"}

    def test_log_population_value(self):
        pop = 12_442_373
        row = _build_feature_row("Mumbai", pop, 2023)
        assert abs(float(row["log_population"].iloc[0]) - math.log(pop)) < 1e-9

    def test_year_passed_through(self):
        row = _build_feature_row("Mumbai", 12_442_373, 2022)
        assert int(row["year_start"].iloc[0]) == 2022


# ---------------------------------------------------------------------------
# Tests: predict_waste (known city)
# ---------------------------------------------------------------------------

class TestPredictWasteKnownCity:
    def test_returns_prediction_result(self, known_result):
        assert isinstance(known_result, PredictionResult)

    def test_prediction_positive(self, known_result):
        assert known_result.prediction_tpd > 0

    def test_prediction_plausible_range(self, known_result):
        # Mumbai historical range: 5300-11000 t/day; some extrapolation OK
        assert 1_000 < known_result.prediction_tpd < 30_000

    def test_log_pred_consistent_with_tpd(self, known_result):
        back = math.expm1(known_result.log_prediction)
        assert abs(back - known_result.prediction_tpd) < 0.01

    def test_is_known_city_true(self, known_result):
        assert known_result.is_known_city is True

    def test_no_warnings_for_known_city_normal_year(self, known_result):
        # Known city, normal year, large population -> no warnings expected
        assert len(known_result.warnings) == 0

    def test_city_stored(self, known_result):
        assert known_result.city == KNOWN_CITY

    def test_year_stored(self, known_result):
        assert known_result.year == TEST_YEAR

    def test_pop_tier_correct(self, known_result):
        assert known_result.pop_tier == 1  # Mumbai is Tier 1


# ---------------------------------------------------------------------------
# Tests: predict_waste (unseen city)
# ---------------------------------------------------------------------------

class TestPredictWasteUnknownCity:
    def test_returns_result_not_error(self, unknown_result):
        assert isinstance(unknown_result, PredictionResult)

    def test_is_known_city_false(self, unknown_result):
        assert unknown_result.is_known_city is False

    def test_warning_issued(self, unknown_result):
        assert len(unknown_result.warnings) >= 1
        assert any("not seen" in w.lower() or "not" in w.lower()
                   for w in unknown_result.warnings)

    def test_prediction_positive(self, unknown_result):
        assert unknown_result.prediction_tpd > 0


# ---------------------------------------------------------------------------
# Tests: predict_waste input validation
# ---------------------------------------------------------------------------

class TestPredictWasteValidation:
    def test_negative_population_raises(self):
        with pytest.raises(ValueError, match="population_2011"):
            predict_waste("Delhi", -1, 2023)

    def test_zero_population_raises(self):
        with pytest.raises(ValueError):
            predict_waste("Delhi", 0, 2023)

    def test_implausible_year_raises(self):
        with pytest.raises(ValueError, match="year"):
            predict_waste("Delhi", 1_000_000, 1800)

    def test_far_future_year_warns(self):
        result = predict_waste("Delhi", 16_787_941, 2050)
        assert any("outside" in w.lower() or "extrapolat" in w.lower()
                   for w in result.warnings)

    def test_small_population_warns(self):
        result = predict_waste("Delhi", 100_000, 2023)
        assert any("population" in w.lower() for w in result.warnings)


# ---------------------------------------------------------------------------
# Tests: predict_batch
# ---------------------------------------------------------------------------

class TestPredictBatch:
    def test_returns_list(self):
        records = [
            {"city": "Delhi",  "population_2011": 16_787_941, "year": 2023},
            {"city": "Mumbai", "population_2011": 12_442_373, "year": 2023},
        ]
        results = predict_batch(records)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_all_items_are_results(self):
        records = [
            {"city": "Pune",   "population_2011": 3_124_458, "year": 2022},
            {"city": "Indore", "population_2011": 1_960_631, "year": 2022},
        ]
        for r in predict_batch(records):
            assert isinstance(r, PredictionResult)

    def test_empty_batch(self):
        assert predict_batch([]) == []


# ---------------------------------------------------------------------------
# Tests: explain_prediction
# ---------------------------------------------------------------------------

class TestExplainPrediction:
    def test_returns_string(self, known_result):
        text = explain_prediction(known_result)
        assert isinstance(text, str)

    def test_contains_city_name(self, known_result):
        text = explain_prediction(known_result)
        assert KNOWN_CITY in text

    def test_contains_prediction_value(self, known_result):
        text = explain_prediction(known_result)
        assert str(int(known_result.prediction_tpd)) in text or \
               f"{known_result.prediction_tpd:,.1f}" in text

    def test_contains_disclaimer(self, known_result):
        text = explain_prediction(known_result)
        assert "correlational" in text.lower() or "causal" in text.lower()

    def test_unknown_city_note_in_explanation(self, unknown_result):
        text = explain_prediction(unknown_result)
        assert "not seen" in text.lower() or "not" in text.lower()

    def test_priority_label_shown_when_set(self, known_result):
        r = predict_waste(KNOWN_CITY, KNOWN_POP, TEST_YEAR)
        r_with_priority = PredictionResult(
            **{**r.__dict__, "priority_label": "HIGH"}
        )
        text = explain_prediction(r_with_priority)
        assert "HIGH" in text

    def test_priority_label_absent_when_not_set(self, known_result):
        # Default result has no priority_label
        text = explain_prediction(known_result)
        assert "Collection priority" not in text


# ---------------------------------------------------------------------------
# Tests: determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_input_same_output(self):
        r1 = predict_waste("Delhi", 16_787_941, 2022)
        r2 = predict_waste("Delhi", 16_787_941, 2022)
        assert r1.prediction_tpd == r2.prediction_tpd

    def test_different_year_different_output(self):
        r1 = predict_waste("Delhi", 16_787_941, 2010)
        r2 = predict_waste("Delhi", 16_787_941, 2022)
        assert r1.prediction_tpd != r2.prediction_tpd

    def test_larger_city_generally_more_waste(self):
        """Larger population should yield higher predicted waste, all else equal."""
        r_small = predict_waste("CityA", 1_000_000, 2022)
        r_large = predict_waste("CityB", 8_000_000, 2022)
        assert r_large.prediction_tpd > r_small.prediction_tpd
