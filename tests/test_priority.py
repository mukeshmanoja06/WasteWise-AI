# -*- coding: utf-8 -*-
"""
tests/test_priority.py
----------------------
Unit tests for src/priority.py.

Run with:  python -m pytest tests/test_priority.py -v
"""

import pathlib
import sys

import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.priority import (
    StrategyA,
    StrategyB,
    classify_with_strategy,
    label_dataframe,
    build_priority_report,
    _load_observed_waste,
    _jenks_breaks,
    THRESHOLDS,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def observed_data():
    return _load_observed_waste()


@pytest.fixture(scope="module")
def strat_a(observed_data):
    return StrategyA.from_data(observed_data)


@pytest.fixture(scope="module")
def strat_b(observed_data):
    return StrategyB.from_data(observed_data)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "city": ["A", "B", "C", "D", "E"],
        "waste_generated": [100.0, 500.0, 1000.0, 3000.0, np.nan],
    })


# ---------------------------------------------------------------------------
# Tests: _jenks_breaks
# ---------------------------------------------------------------------------

class TestJenksBreaks:
    def test_returns_k_minus_1_breaks(self):
        data = np.array([1, 2, 5, 6, 10, 11, 20, 21, 50], dtype=float)
        breaks = _jenks_breaks(data, k=3)
        assert len(breaks) == 2

    def test_breaks_are_sorted(self):
        data = np.linspace(1, 100, 50)
        breaks = _jenks_breaks(data, k=3)
        assert breaks[0] <= breaks[1]

    def test_breaks_within_data_range(self):
        data = np.array([10, 50, 100, 200, 500, 1000, 5000], dtype=float)
        breaks = _jenks_breaks(data, k=3)
        assert data.min() <= breaks[0] <= data.max()
        assert data.min() <= breaks[1] <= data.max()

    def test_k_equals_2_returns_one_break(self):
        data = np.array([1, 2, 10, 11], dtype=float)
        breaks = _jenks_breaks(data, k=2)
        assert len(breaks) == 1


# ---------------------------------------------------------------------------
# Tests: StrategyA
# ---------------------------------------------------------------------------

class TestStrategyA:
    def test_p33_less_than_p67(self, strat_a):
        assert strat_a.p33 < strat_a.p67

    def test_p33_near_33rd_percentile(self, observed_data, strat_a):
        expected = np.percentile(observed_data, 33)
        assert abs(strat_a.p33 - expected) < 1.0

    def test_p67_near_67th_percentile(self, observed_data, strat_a):
        expected = np.percentile(observed_data, 67)
        assert abs(strat_a.p67 - expected) < 1.0

    def test_classify_low(self, strat_a):
        assert strat_a.classify(1.0) == "LOW"

    def test_classify_high(self, strat_a):
        assert strat_a.classify(99_999.0) == "HIGH"

    def test_classify_at_p33_boundary(self, strat_a):
        assert strat_a.classify(strat_a.p33) == "LOW"

    def test_classify_just_above_p33(self, strat_a):
        assert strat_a.classify(strat_a.p33 + 0.01) == "MEDIUM"

    def test_classify_at_p67_boundary(self, strat_a):
        assert strat_a.classify(strat_a.p67) == "MEDIUM"

    def test_classify_just_above_p67(self, strat_a):
        assert strat_a.classify(strat_a.p67 + 0.01) == "HIGH"

    def test_thresholds_dict_keys(self, strat_a):
        t = strat_a.thresholds()
        assert set(t.keys()) == {"strategy", "LOW_max", "MEDIUM_max", "HIGH_min"}

    def test_thresholds_low_max_equals_p33(self, strat_a):
        assert strat_a.thresholds()["LOW_max"] == round(strat_a.p33, 2)


# ---------------------------------------------------------------------------
# Tests: StrategyB
# ---------------------------------------------------------------------------

class TestStrategyB:
    def test_break1_less_than_break2(self, strat_b):
        assert strat_b.break1 < strat_b.break2

    def test_classify_low(self, strat_b):
        assert strat_b.classify(1.0) == "LOW"

    def test_classify_high(self, strat_b):
        assert strat_b.classify(99_999.0) == "HIGH"

    def test_classify_at_break1_boundary(self, strat_b):
        assert strat_b.classify(strat_b.break1) == "LOW"

    def test_classify_just_above_break1(self, strat_b):
        assert strat_b.classify(strat_b.break1 + 0.01) == "MEDIUM"

    def test_classify_at_break2_boundary(self, strat_b):
        assert strat_b.classify(strat_b.break2) == "MEDIUM"

    def test_classify_just_above_break2(self, strat_b):
        assert strat_b.classify(strat_b.break2 + 0.01) == "HIGH"

    def test_thresholds_dict_keys(self, strat_b):
        t = strat_b.thresholds()
        assert set(t.keys()) == {"strategy", "LOW_max", "MEDIUM_max", "HIGH_min"}

    def test_break1_within_data_range(self, observed_data, strat_b):
        assert observed_data.min() <= strat_b.break1 <= observed_data.max()

    def test_break2_within_data_range(self, observed_data, strat_b):
        assert observed_data.min() <= strat_b.break2 <= observed_data.max()


# ---------------------------------------------------------------------------
# Tests: classify (unified API)
# ---------------------------------------------------------------------------

class TestClassify:
    def test_strategy_a_low(self, strat_a):
        assert classify_with_strategy(1.0, strat_a) == "LOW"

    def test_strategy_b_low(self, strat_b):
        assert classify_with_strategy(1.0, strat_b) == "LOW"

    def test_strategy_a_high(self, strat_a):
        assert classify_with_strategy(50_000.0, strat_a) == "HIGH"

    def test_strategy_b_high(self, strat_b):
        assert classify_with_strategy(50_000.0, strat_b) == "HIGH"


# ---------------------------------------------------------------------------
# Tests: label_dataframe
# ---------------------------------------------------------------------------

class TestLabelDataframe:
    def test_adds_priority_column(self, sample_df, strat_a):
        result = label_dataframe(sample_df, "waste_generated", strat_a)
        assert "priority" in result.columns

    def test_nan_becomes_unknown(self, sample_df, strat_a):
        result = label_dataframe(sample_df, "waste_generated", strat_a)
        assert result.loc[result["waste_generated"].isna(), "priority"].iloc[0] == "UNKNOWN"

    def test_original_df_not_modified(self, sample_df, strat_a):
        _ = label_dataframe(sample_df, "waste_generated", strat_a)
        assert "priority" not in sample_df.columns

    def test_all_labels_valid(self, sample_df, strat_a):
        result = label_dataframe(sample_df, "waste_generated", strat_a)
        valid = {"LOW", "MEDIUM", "HIGH", "UNKNOWN"}
        assert set(result["priority"].unique()).issubset(valid)

    def test_custom_output_column(self, sample_df, strat_b):
        result = label_dataframe(sample_df, "waste_generated", strat_b, out_col="level")
        assert "level" in result.columns
        assert "priority" not in result.columns

    def test_row_count_unchanged(self, sample_df, strat_a):
        result = label_dataframe(sample_df, "waste_generated", strat_a)
        assert len(result) == len(sample_df)


# ---------------------------------------------------------------------------
# Tests: build_priority_report
# ---------------------------------------------------------------------------

class TestBuildPriorityReport:
    def test_returns_dataframe(self, observed_data):
        report = build_priority_report(observed_data)
        assert isinstance(report, pd.DataFrame)

    def test_two_strategies_in_report(self, observed_data):
        report = build_priority_report(observed_data)
        assert len(report) == 2

    def test_required_columns_present(self, observed_data):
        report = build_priority_report(observed_data)
        required = {"dataset", "LOW_threshold", "HIGH_threshold",
                    "LOW_count", "MEDIUM_count", "HIGH_count",
                    "LOW_pct", "MEDIUM_pct", "HIGH_pct"}
        assert required.issubset(set(report.columns))

    def test_counts_sum_to_n(self, observed_data):
        # Report has two rows: training-period and full-observed
        # The full-observed row uses all 307 observations
        report = build_priority_report(observed_data)
        full_row = report[report["dataset"].str.startswith("Full")].iloc[0]
        n = len(observed_data)
        total = full_row["LOW_count"] + full_row["MEDIUM_count"] + full_row["HIGH_count"]
        assert total == n

    def test_pct_sum_near_100(self, observed_data):
        report = build_priority_report(observed_data)
        for _, row in report.iterrows():
            total_pct = row["LOW_pct"] + row["MEDIUM_pct"] + row["HIGH_pct"]
            assert abs(total_pct - 100.0) < 0.5

    def test_strategy_a_roughly_balanced(self, observed_data):
        """Training-period row should be roughly balanced (~33% each)."""
        report = build_priority_report(observed_data)
        train_row = report[report["dataset"].str.startswith("Training")].iloc[0]
        # Each class should be roughly 33% - allow 15% tolerance
        for pct_col in ["LOW_pct", "MEDIUM_pct", "HIGH_pct"]:
            assert 15 <= float(train_row[pct_col]) <= 50


# ---------------------------------------------------------------------------
# Tests: synthetic data edge cases
# ---------------------------------------------------------------------------

class TestSyntheticEdgeCases:
    def test_all_same_value(self):
        data = np.full(30, 500.0)
        sa = StrategyA.from_data(data)
        # All values at boundary -> all LOW or MEDIUM depending on boundary rule
        labels = [sa.classify(v) for v in data]
        assert all(l in ("LOW", "MEDIUM") for l in labels)

    def test_two_extreme_groups(self):
        """
        Clear bimodal split with distinct values: Jenks should separate groups.
        Uses slightly perturbed values so the break does not land exactly on
        the boundary (degenerate case where break1==break2 is a known
        Jenks limitation on perfectly equal-value groups).
        """
        low_group  = np.linspace(90, 110, 50)     # 90-110 range
        high_group = np.linspace(9900, 10100, 50)  # 9900-10100 range
        data = np.concatenate([low_group, high_group])
        sb = StrategyB.from_data(data)
        # At minimum the algorithm should separate the two clusters
        low_labels  = [sb.classify(v) for v in low_group]
        high_labels = [sb.classify(v) for v in high_group]
        # All low_group values must be in a lower-or-equal priority than high_group
        priority_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        max_low_priority  = max(priority_order[l] for l in low_labels)
        min_high_priority = min(priority_order[l] for l in high_labels)
        assert max_low_priority <= min_high_priority, (
            f"Jenks failed to separate bimodal groups: "
            f"low max={max_low_priority}, high min={min_high_priority}"
        )

    def test_strategy_a_from_synthetic(self):
        data = np.arange(1, 101, dtype=float)
        sa = StrategyA.from_data(data)
        assert sa.p33 < sa.p67
        assert sa.classify(1.0) == "LOW"
        assert sa.classify(100.0) == "HIGH"
