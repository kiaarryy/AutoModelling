from __future__ import annotations

import math

import pytest

import numpy as np

from autofmu.evaluation import chronological_three_way_split, coverage_row, metric_pairs


def test_chronological_three_way_split_is_disjoint():
    split = chronological_three_way_split(
        100,
        train_fraction=0.6,
        select_fraction=0.2,
    )

    assert len(split.train) == 60
    assert len(split.select) == 20
    assert len(split.test) == 20
    assert set(split.train).isdisjoint(split.select)
    assert set(split.train).isdisjoint(split.test)
    assert set(split.select).isdisjoint(split.test)
    assert split.train.max() < split.select.min() < split.test.min()


def test_chronological_three_way_split_keeps_each_role_nonempty():
    split = chronological_three_way_split(5)

    assert len(split.train) >= 1
    assert len(split.select) >= 1
    assert len(split.test) >= 1
    assert len(split.train) + len(split.select) + len(split.test) == 5


@pytest.mark.parametrize(
    "rows,train_fraction,select_fraction",
    [
        (4, 0.6, 0.2),
        (100, 0.0, 0.2),
        (100, 0.8, 0.2),
        (100, 0.6, -0.1),
    ],
)
def test_chronological_three_way_split_rejects_invalid_requests(
    rows,
    train_fraction,
    select_fraction,
):
    with pytest.raises(ValueError):
        chronological_three_way_split(rows, train_fraction, select_fraction)


def test_coverage_row_reports_explicit_denominators():
    row = coverage_row(
        rows_total=1000,
        rows_on=500,
        rows_valid=300,
        rows_evaluated=150,
    )

    assert row == {
        "rows_total": 1000,
        "rows_on": 500,
        "rows_valid": 300,
        "rows_evaluated": 150,
        "coverage_of_on_pct": 30.0,
        "coverage_of_total_pct": 15.0,
    }


def test_coverage_row_uses_nan_for_zero_denominator():
    row = coverage_row(100, 0, 0, 0)

    assert math.isnan(row["coverage_of_on_pct"])
    assert row["coverage_of_total_pct"] == 0.0


def test_coverage_row_rejects_impossible_counts():
    with pytest.raises(ValueError, match="rows_evaluated"):
        coverage_row(100, 50, 20, 21)


def test_metric_gate_rejects_low_finite_output_coverage():
    measured = np.arange(100, dtype=float)
    simulated = measured.copy()
    simulated[:95] = np.nan
    with pytest.raises(ValueError, match="finite output coverage"):
        metric_pairs(measured, simulated, min_coverage=0.95)
