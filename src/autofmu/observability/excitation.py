"""Did the driving variable ever move?

A model is identifiable from a record only if the record excites it. This is
independent of how many rows there are, and it is invisible to every check that
counts rows, closes energy balances, or holds out a test window.

Site A CDWP_01 runs at a speed ratio of 0.700 at the 5th percentile and 0.701
at the 95th, across 28,619 rows -- 18 distinct values in the whole archive. Its
affinity-law power model reports 5.93% test CVRMSE. That number cannot
distinguish a cubic law from a constant, because over a span of 0.001 no smooth
function is distinguishable from its own first-order expansion.

Measured across the four archives, the pattern inverts the naive reading of the
results:

    site      pump speed-ratio span (p5-p95)   test CVRMSE
    Site A    0.001 - 0.218                    5.93 - 15.33%
    Tencent   0.060 - 0.160                    1.26 - 17.40%
    LBNL      0.348 - 0.655                    3.23 - 22.58%

The public archive is the only one whose pumps genuinely change speed, and its
models score worst. The sites with the best-looking numbers are the sites whose
pumps never moved -- those are not better models, they are narrower tests.

So excitation is reported alongside every score rather than used to block one.
A score on an unexcited record is not wrong, it is weak evidence, and the
distinction has to reach the reader instead of being averaged away.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Relative span (p95-p5 over the median) below which the driver has not moved
# enough to identify a nonlinear law. At +-5% around an operating point a cubic
# and a straight line differ by less than the measurement noise of a BMS power
# channel, so the fitted exponent is not evidenced by the data.
MIN_RELATIVE_SPAN = 0.10
MIN_ROWS = 100


@dataclass(frozen=True)
class ExcitationFinding:
    channel: str
    rows: int
    p5: float
    p95: float
    relative_span: float
    distinct: int
    excited: bool

    def as_flag(self) -> str:
        return f"driver_not_excited:{self.channel}={self.relative_span:.3f}"

    @property
    def detail(self) -> str:
        return (f"{self.channel} spans {self.p5:.4g} to {self.p95:.4g} "
                f"({100 * self.relative_span:.1f}% of its median) over {self.rows} "
                f"operating rows with {self.distinct} distinct values; a model "
                f"whose driver did not move is not identified by this record, "
                f"however well it scores")


def excitation(frame: pd.DataFrame, channel: str,
               operating: np.ndarray,
               min_relative_span: float = MIN_RELATIVE_SPAN):
    """How far the driving variable moved over the operating record."""
    if channel not in frame:
        return None
    values = pd.to_numeric(frame[channel], errors="coerce").to_numpy(dtype=float)
    sample = values[operating & np.isfinite(values) & (values > 0)]
    if sample.size < MIN_ROWS:
        return None
    p5, p95 = float(np.percentile(sample, 5)), float(np.percentile(sample, 95))
    centre = float(np.median(sample))
    if not np.isfinite(centre) or centre == 0.0:
        return None
    relative = (p95 - p5) / abs(centre)
    return ExcitationFinding(
        channel, int(sample.size), p5, p95, relative,
        int(pd.Series(np.round(sample, 6)).nunique()),
        bool(relative >= min_relative_span))
