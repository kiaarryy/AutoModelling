"""Shared evaluation protocol for calibrated equipment models.

Calibration, model selection, and final testing need disjoint chronological
rows. This module owns those role boundaries and the row-coverage fields that
make filtered operating-point metrics auditable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EvaluationSplit:
    train: np.ndarray
    select: np.ndarray
    test: np.ndarray
    buffer: np.ndarray = None  # type: ignore[assignment]
    status: str = "full_split"

    def __post_init__(self):
        if self.buffer is None:
            object.__setattr__(self, "buffer", np.empty(0, dtype=int))

    @property
    def usable(self) -> bool:
        return self.status == "full_split"


# Every equipment family routes its measured/simulated arrays through
# metric_pairs on the way to a metric, which makes it the one place a
# validation trace can be captured without touching four device engines. The
# recorder is off unless a caller opens it, so the normal path is unchanged.
_RECORDER: list | None = None


def start_recording() -> None:
    """Begin collecting every measured/simulated pair passed to metric_pairs."""
    global _RECORDER
    _RECORDER = []


def recorded() -> list:
    """The pairs collected since :func:`start_recording`, in call order."""
    return list(_RECORDER or ())


def stop_recording() -> None:
    global _RECORDER
    _RECORDER = None


def metric_pairs(measured, simulated, min_coverage: float = 0.95):
    """Return finite metric pairs only when their coverage is acceptable."""
    measured_array = np.asarray(measured, dtype=float)
    simulated_array = np.asarray(simulated, dtype=float)
    if measured_array.shape != simulated_array.shape:
        raise ValueError("measured and simulated arrays must have the same shape")
    if measured_array.ndim != 1 or measured_array.size == 0:
        raise ValueError("metric pairs require a non-empty one-dimensional series")
    threshold = float(min_coverage)
    if not 0.0 < threshold <= 1.0:
        raise ValueError("min_coverage must be in (0, 1]")
    finite = np.isfinite(measured_array) & np.isfinite(simulated_array)
    coverage = float(finite.mean())
    if coverage < threshold:
        raise ValueError(
            f"finite output coverage {coverage * 100.0:.2f}% is below "
            f"the required {threshold * 100.0:.2f}%"
        )
    pair = (measured_array[finite], simulated_array[finite])
    if _RECORDER is not None:
        _RECORDER.append(pair)
    return pair


def chronological_three_way_split(
    n: int,
    train_fraction: float = 0.6,
    select_fraction: float = 0.2,
) -> EvaluationSplit:
    """Split ordered rows into non-overlapping train, selection, and test roles."""
    rows = int(n)
    train_fraction = float(train_fraction)
    select_fraction = float(select_fraction)
    if rows < 5:
        raise ValueError("at least 5 rows are required for train/select/test")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0.0 < select_fraction < 1.0:
        raise ValueError("select_fraction must be between 0 and 1")
    if train_fraction + select_fraction >= 1.0:
        raise ValueError("train_fraction + select_fraction must be less than 1")

    train_end = max(1, int(rows * train_fraction))
    select_end = max(train_end + 1, int(rows * (train_fraction + select_fraction)))
    select_end = min(select_end, rows - 1)
    indices = np.arange(rows, dtype=int)
    return EvaluationSplit(
        train=indices[:train_end],
        select=indices[train_end:select_end],
        test=indices[select_end:],
    )


def buffered_three_way_split(elapsed_seconds, spec_mapping=None) -> EvaluationSplit:
    """Buffered, block-interleaved replacement for the contiguous split.

    :func:`chronological_three_way_split` cuts the record into three contiguous
    pieces, so the test rows are always the tail of the record.  On a year of
    plant data that makes the test segment a different season from the training
    segment, and it leaves no gap between them, so the first test rows are the
    thermal continuation of the last training rows.  Both problems are what the
    reviewers objected to; see docs/REVISION_ENERGY_04_MANUSCRIPT_FIXES.md M-09.

    This variant deals equal-duration blocks out to the three roles in a
    repeating pattern and discards a buffer at every boundary, so each role
    samples the whole record and none of them touches its neighbour.  Devices
    whose record cannot support all three roles come back with a ``status``
    other than ``full_split`` and must be reported as degraded rather than
    scored.
    """
    from .temporal_split import SplitSpec, build_split_from_elapsed

    spec = SplitSpec.from_mapping(spec_mapping)
    result = build_split_from_elapsed(elapsed_seconds, spec)
    index = np.arange(len(result.timestamps), dtype=int)
    return EvaluationSplit(
        train=index[result.masks["identification"]],
        select=index[result.masks["selection"]],
        test=index[result.masks["test"]],
        buffer=index[result.buffer_mask],
        status=result.status,
    )


def resolve_split(table, thresholds=None, time_column: str = "time_s",
                  native_rows: int | None = None) -> EvaluationSplit:
    """Pick the split a device engine should use, from configuration.

    One resolver rather than the same three-line branch in four engines: when
    ``thresholds['split']`` is present and the table carries a usable time
    column, the buffered block-interleaved split applies; otherwise the legacy
    contiguous fractions do, so a project that has not opted in behaves exactly
    as before.

    ``table`` may be a DataFrame or a plain row count; a row count can only ever
    get the legacy split, since without timestamps there is nothing to buffer
    against.

    ``native_rows`` is the size of the record *before* any subsampling. The
    chiller engine fits on 600 rows drawn from tens of thousands, the pump on
    2000, and the split's minimums are written in native five-minute samples --
    288 rows meaning a day, 2016 meaning a week. Applied to a subsample those
    numbers stop meaning any amount of time at all, and every chiller at Site A
    came back blocked and every pump identification-only purely from that.
    Passing the pre-subsample count scales the minimums by the same ratio, so a
    requirement of "a day of data" survives the downsample.
    """
    thresholds = thresholds or {}
    spec = thresholds.get("split")
    legacy = lambda n: chronological_three_way_split(  # noqa: E731
        int(n),
        train_fraction=float(thresholds.get("train_fraction", 0.6)),
        select_fraction=float(thresholds.get("select_fraction", 0.2)))

    if hasattr(table, "columns"):
        n_rows = len(table)
        times = table[time_column] if time_column in table.columns else None
    else:
        n_rows, times = int(table), None

    if spec is None or times is None:
        return legacy(n_rows)

    if native_rows and n_rows and native_rows > n_rows:
        ratio = n_rows / float(native_rows)
        spec = dict(spec)
        for key, floor in (("min_identification_rows", 288),
                           ("min_selection_rows", 288),
                           ("min_test_rows", 2016)):
            scaled = int(round(float(spec.get(key, floor)) * ratio))
            # never fall below what a fit actually needs to be meaningful
            spec[key] = max(scaled, 30)
    return buffered_three_way_split(times, spec)


def coverage_row(
    rows_total: int,
    rows_on: int,
    rows_valid: int,
    rows_evaluated: int,
) -> dict:
    """Return explicit evaluation counts and coverage percentages."""
    total = int(rows_total)
    on = int(rows_on)
    valid = int(rows_valid)
    evaluated = int(rows_evaluated)
    if min(total, on, valid, evaluated) < 0:
        raise ValueError("row counts must be non-negative")
    if on > total:
        raise ValueError("rows_on cannot exceed rows_total")
    if valid > on:
        raise ValueError("rows_valid cannot exceed rows_on")
    if evaluated > valid:
        raise ValueError("rows_evaluated cannot exceed rows_valid")
    return {
        "rows_total": total,
        "rows_on": on,
        "rows_valid": valid,
        "rows_evaluated": evaluated,
        "coverage_of_on_pct": evaluated / on * 100.0 if on else float("nan"),
        "coverage_of_total_pct": evaluated / total * 100.0 if total else float("nan"),
    }
