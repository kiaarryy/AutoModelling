"""Single-device-operating window detection (Layer 2 attribution enabler).

Rationale (AlphaDataCenterCooling SI 2.4/2.5): when only ONE device of a type is
running, shared / main-pipe meter readings can be attributed to that single
device. These "solo" windows are where un-metered flow can be assigned, so the
count of solo rows is a key modelability signal for CT / pump / chiller.
"""
from __future__ import annotations

from typing import Dict, Mapping

import numpy as np
import pandas as pd


def _on(series: pd.Series, run_on: float) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0) > run_on


def solo_run_windows(
    run_signals: Mapping[str, pd.Series],
    run_on: float = 0.5,
) -> Dict[str, int]:
    """Given {device_id -> run_signal series} aligned by position, return the
    number of timestamps at which each device is the ONLY one running.

    All series must share the same index/length (align on timestamp upstream).
    """
    if not run_signals:
        return {}
    on_frame = pd.DataFrame({dev: _on(sig, run_on) for dev, sig in run_signals.items()})
    running_count = on_frame.sum(axis=1)
    solo_mask = running_count == 1
    result: Dict[str, int] = {}
    for dev in on_frame.columns:
        result[dev] = int((on_frame[dev] & solo_mask).sum())
    return result


def solo_run_mask(run_signals: Mapping[str, pd.Series], run_on: float = 0.5) -> Dict[str, pd.Series]:
    """Like solo_run_windows but returns the per-device boolean mask (indexed as
    the input series) marking timestamps where the device runs alone."""
    if not run_signals:
        return {}
    on_frame = pd.DataFrame({dev: _on(sig, run_on) for dev, sig in run_signals.items()})
    solo = on_frame.sum(axis=1) == 1
    return {dev: (on_frame[dev] & solo) for dev in on_frame.columns}


def align_run_signals(frames: Mapping[str, pd.DataFrame], run_signal: str = "run_signal") -> Dict[str, pd.Series]:
    """Align per-device canonical frames on timestamp and extract run signals.

    Devices missing the run signal are skipped. Returns {device_id -> series}
    reindexed onto the union of timestamps (missing -> 0/off).
    """
    series_by_device: Dict[str, pd.Series] = {}
    for dev, frame in frames.items():
        if run_signal not in frame or "timestamp" not in frame:
            continue
        s = pd.Series(
            pd.to_numeric(frame[run_signal], errors="coerce").to_numpy(),
            index=pd.to_datetime(frame["timestamp"], errors="coerce"),
        )
        s = s[~s.index.isna()]
        series_by_device[dev] = s
    if not series_by_device:
        return {}
    union = None
    for s in series_by_device.values():
        union = s.index if union is None else union.union(s.index)
    return {dev: s.reindex(union).fillna(0.0) for dev, s in series_by_device.items()}
