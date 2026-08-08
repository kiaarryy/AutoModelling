"""Is the reconstructed quantity physically attainable?

The synthetic-channel detector asks whether a published point is a function of
another point. This module asks a different question: taking the published
points at face value, does the resulting machine obey thermodynamics?

Two findings across the four archives, neither of which any earlier check
could see:

**A frozen channel.** HKUST CH-08 reports chilled-water flow as 70.1 m3/h on
96.4% of its operating record and condenser flow as 84.4 m3/h on the same rows,
while compressor power swings from 210 to 619 kW. A load reconstructed from a
constant flow is an affine function of the temperature difference alone. The
synthetic-channel detector cannot see this -- it deliberately skips degenerate
targets to avoid false positives, and a constant is exactly what it skips.

**An unattainable efficiency.** The same chiller returns COP <= 1 on 33.1% of
its usable rows; no vapour-compression machine does that, and the rows are
quarantined. At the other extreme Site A's chillers imply a median COP of 10.0
to 21.6 -- 124 kW of compressor power against 2,400 kW of reconstructed cooling
on CH_01. The condenser balance closes there (Q_cond / (Q_evap + P) = 1.04), so
the thermal channels agree with each other and it is the power channel that is
not measuring the whole machine. That does not invalidate the power model,
which is fitted and scored against measured power; it invalidates the
*derived* quantities and the calibrated FMU's physical parameters. The device
is flagged, not blocked, and the distinction is recorded so a reader can tell
which reported numbers rest on a measurement and which rest on a
reconstruction.

Rules of engagement, same as the rest of the observability layer: convict only
on evidence, quarantine only what is impossible, flag what is merely
implausible, and never silently drop a device.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

# A channel stuck on one non-zero value for this fraction of the operating
# record is a declared constant, not a meter reading. The zero level is
# excluded first: "off" is a legitimate repeated value.
FROZEN_MODAL_FRACTION = 0.95
FROZEN_MIN_ROWS = 200

# Water-cooled vapour compression. Below 1 is impossible -- the compressor
# would be producing more cooling than the total energy entering the
# evaporator. The upper bound is generous: a centrifugal machine at low load
# with cold condenser water reaches 9-10, so a median past that says the load
# and the power are not describing the same equipment.
COP_IMPOSSIBLE_AT_OR_BELOW = 1.0
COP_PLAUSIBLE_MAX = 10.0
# Rows are quarantined however few they are; the device is only described as
# having a problem once the share is past sampling noise.
REPORT_SHARE_PCT = 1.0


@dataclass(frozen=True)
class EnvelopeFinding:
    check: str
    subject: str
    verdict: str      # "quarantine" (impossible) | "flag" (implausible)
    value: float
    detail: str

    def as_flag(self) -> str:
        return f"{self.check}:{self.subject}={self.value:.4g}"


def frozen_channels(frame: pd.DataFrame, columns, operating: np.ndarray,
                    modal_fraction: float = FROZEN_MODAL_FRACTION,
                    min_rows: int = FROZEN_MIN_ROWS) -> List[EnvelopeFinding]:
    """Continuous channels stuck on a single non-zero value while operating."""
    findings: List[EnvelopeFinding] = []
    for column in columns:
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        known = operating & np.isfinite(values)
        sample = values[known]
        sample = sample[sample != 0.0]
        if sample.size < min_rows:
            continue
        counts = pd.Series(np.round(sample, 6)).value_counts()
        modal = float(counts.iloc[0]) / sample.size
        if modal >= modal_fraction:
            findings.append(EnvelopeFinding(
                "frozen_channel", column, "flag", round(100.0 * modal, 1),
                f"{column} holds {counts.index[0]:.6g} on {100.0 * modal:.1f}% of "
                f"{sample.size} operating rows: a design value published as a "
                f"measurement, so anything reconstructed from it is declared, "
                f"not measured"))
    return findings


def cop_envelope(load_W: np.ndarray, power_W: np.ndarray,
                 operating: np.ndarray) -> Tuple[np.ndarray, List[EnvelopeFinding]]:
    """(impossible-row mask, findings) for a reconstructed coefficient of
    performance. The mask marks rows to quarantine; the findings describe the
    device."""
    findings: List[EnvelopeFinding] = []
    usable = (operating & np.isfinite(load_W) & np.isfinite(power_W)
              & (load_W > 0) & (power_W > 0))
    impossible = np.zeros(len(load_W), dtype=bool)
    if not usable.any():
        return impossible, findings

    cop = np.full(len(load_W), np.nan)
    cop[usable] = load_W[usable] / power_W[usable]
    impossible = usable & (cop <= COP_IMPOSSIBLE_AT_OR_BELOW)
    share = 100.0 * float(impossible.sum()) / int(usable.sum())
    # Quarantine every impossible row, but only describe the device when the
    # share is past sampling noise: Site A CH_01 has 2 such rows in 29,084 and
    # reporting that as a finding would bury CH-08's 33%.
    if share >= REPORT_SHARE_PCT:
        findings.append(EnvelopeFinding(
            "cop_below_unity", "reconstructed_COP", "quarantine", round(share, 1),
            f"{impossible.sum()} of {int(usable.sum())} rows ({share:.1f}%) "
            f"reconstruct more cooling than the total energy entering the "
            f"evaporator; quarantined as physically impossible"))

    median = float(np.nanmedian(cop[usable & ~impossible])) if (usable & ~impossible).any() else np.nan
    if np.isfinite(median) and median > COP_PLAUSIBLE_MAX:
        findings.append(EnvelopeFinding(
            "cop_above_envelope", "reconstructed_COP", "flag", round(median, 2),
            f"median reconstructed COP {median:.1f} exceeds what water-cooled "
            f"vapour compression attains, so the load and the power are not "
            f"describing the same equipment; metrics against measured power "
            f"remain valid, metrics against reconstructed load do not"))
    return impossible, findings
