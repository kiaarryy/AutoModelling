"""Tests for the three-way temporal split.

The properties that matter for the manuscript claim are: the three segments are
disjoint, a buffer really separates them in time, and a device with too little
data is degraded rather than given a meaningless test score.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autofmu.temporal_split import (
    SEGMENTS, SplitSpec, build_split, coverage_report,
)


@pytest.fixture
def year():
    return pd.date_range("2024-05-01", "2025-04-30 23:55", freq="5min")


def test_segments_are_disjoint_and_within_the_record(year):
    result = build_split(year)
    stacked = np.stack([result.masks[s] for s in SEGMENTS] + [result.buffer_mask])
    assert stacked.sum(axis=0).max() <= 1
    assert stacked.shape[1] == len(year)


def test_buffer_separates_identification_from_test(year):
    spec = SplitSpec(buffer_hours=72.0)
    result = build_split(year, spec)
    ident = result.timestamps[result.masks["identification"]]
    test = result.timestamps[result.masks["test"]]
    assert not ident.empty and not test.empty
    # every test sample is at least the buffer away from every identification block
    for edge in (ident.min(), ident.max()):
        gap = (test - edge).abs().min()
        assert gap >= pd.Timedelta(hours=72) - pd.Timedelta(minutes=5)


def test_explicit_blocks_are_honoured(year):
    spec = SplitSpec(
        buffer_hours=24.0,
        blocks={"identification": [("2024-06-01", "2024-06-08")],
                "selection": [("2024-08-01", "2024-08-08")],
                "test": [("2024-10-01", "2024-11-01")]})
    result = build_split(year, spec)
    ident = result.timestamps[result.masks["identification"]]
    assert ident.min() >= pd.Timestamp("2024-06-01")
    assert ident.max() < pd.Timestamp("2024-06-08")
    test = result.timestamps[result.masks["test"]]
    assert test.min() >= pd.Timestamp("2024-10-01")
    assert test.max() < pd.Timestamp("2024-11-01")


def test_overlapping_declared_blocks_are_resolved_and_warned(year):
    spec = SplitSpec(blocks={
        "identification": [("2024-06-01", "2024-06-10")],
        "selection": [("2024-06-05", "2024-06-15")]})
    result = build_split(year, spec)
    assert not (result.masks["identification"] & result.masks["selection"]).any()
    assert any("both identification" in w for w in result.warnings)


def test_short_record_is_degraded_not_scored():
    """CT-02 has 1,442 valid rows; it must not receive a test score."""
    short = pd.date_range("2024-06-01", periods=1442, freq="5min")
    result = build_split(short, SplitSpec(buffer_hours=72.0))
    assert not result.usable
    assert result.status in {"identification_only",
                             "blocked_insufficient_identification"}
    assert result.warnings


def test_full_year_supports_a_scored_test_segment(year):
    result = build_split(year)
    counts = result.counts()
    assert result.usable, counts
    assert result.status == "full_split"
    assert counts["test"] > counts["identification"]
    assert counts["buffer"] > 0


def test_counts_sum_to_the_record_length(year):
    result = build_split(year)
    counts = result.counts()
    assert sum(counts[k] for k in ("identification", "selection", "test",
                                   "buffer", "unused")) == len(year)


def test_comparability_reports_ranges_per_segment(year):
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "DateTime": year,
        "twb_C": 18 + 8 * np.sin(np.arange(len(year)) * 2 * np.pi / len(year))
                 + rng.normal(0, 0.4, len(year)),
    })
    result = build_split(frame["DateTime"])
    table = coverage_report(frame, result, ["twb_C"])
    assert set(table.segment) == set(SEGMENTS)
    assert (table.n > 0).all()
    # identification deliberately spans several months, so its wet-bulb range
    # should not be a narrow slice
    ident = table[table.segment == "identification"].iloc[0]
    assert ident.maximum - ident.minimum > 3.0


def test_spec_round_trips_through_a_mapping():
    spec = SplitSpec.from_mapping({
        "buffer_hours": 48,
        "min_test_rows": 1000,
        "seasonal_identification_months": [1, 7],
        "blocks": {"identification": [["2024-06-01", "2024-06-08"]]},
    })
    assert spec.buffer_hours == 48
    assert spec.min_test_rows == 1000
    assert spec.seasonal_identification_months == (1, 7)
    assert spec.blocks["identification"] == [("2024-06-01", "2024-06-08")]


def test_unparseable_timestamps_raise():
    with pytest.raises(ValueError):
        build_split(pd.Series(["not a date", "also not"]))


# --- integration with the device engines -----------------------------------

def test_buffered_split_returns_disjoint_index_arrays(year):
    from autofmu.evaluation import buffered_three_way_split
    elapsed = (year - year[0]).total_seconds()
    split = buffered_three_way_split(elapsed, {"buffer_hours": 72})
    parts = [split.train, split.select, split.test, split.buffer]
    joined = np.concatenate(parts)
    assert len(set(joined.tolist())) == len(joined)      # no index used twice
    assert joined.max() < len(year)
    assert split.usable and split.status == "full_split"


def test_buffered_split_degrades_a_short_record():
    from autofmu.evaluation import buffered_three_way_split
    short = pd.date_range("2024-06-01", periods=1442, freq="5min")
    split = buffered_three_way_split((short - short[0]).total_seconds(), {})
    assert not split.usable
    assert split.status != "full_split"


def test_buffered_split_beats_the_contiguous_one_on_coverage(year):
    """The old split puts the whole test segment in one season; the new one does not."""
    from autofmu.evaluation import (buffered_three_way_split,
                                    chronological_three_way_split)
    elapsed = (year - year[0]).total_seconds()
    old = chronological_three_way_split(len(year))
    new = buffered_three_way_split(elapsed, {"buffer_hours": 72})
    months_old = year[old.test].month.nunique()
    months_new = year[new.test].month.nunique()
    assert months_new > months_old


def test_row_minimums_scale_with_subsampling():
    """A requirement of "a day of data" must survive a downsample.

    The chiller engine fits on 600 rows drawn from tens of thousands. Applying
    a 288-row minimum -- written to mean 24 h of five-minute samples -- to that
    subsample blocked every Site A chiller for no physical reason.
    """
    from autofmu.evaluation import resolve_split

    native = 40000
    full = pd.DataFrame({"time_s": np.linspace(0, 365 * 86400, native)})
    sub = full.iloc[:: native // 600].reset_index(drop=True)
    thresholds = {"split": {"buffer_hours": 72, "min_identification_rows": 288,
                            "min_selection_rows": 288, "min_test_rows": 2016}}

    unscaled = resolve_split(sub, thresholds)
    scaled = resolve_split(sub, thresholds, native_rows=native)
    assert not unscaled.usable          # the artefact
    assert scaled.usable                # the same record, judged in real time

    # the full record is unaffected: nothing to scale
    assert resolve_split(full, thresholds, native_rows=native).usable


def test_scaling_keeps_an_absolute_floor():
    """Scaling must not let a handful of rows count as a fitted segment."""
    from autofmu.evaluation import resolve_split

    tiny = pd.DataFrame({"time_s": np.linspace(0, 365 * 86400, 80)})
    thresholds = {"split": {"buffer_hours": 72, "min_identification_rows": 288,
                            "min_selection_rows": 288, "min_test_rows": 2016}}
    assert not resolve_split(tiny, thresholds, native_rows=200000).usable
