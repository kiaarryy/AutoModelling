from __future__ import annotations

import math
from typing import Iterable

import numpy as np


#: Quantities measured on an interval scale, where the zero point is a
#: convention rather than an absence of the thing being measured.
INTERVAL_QUANTITIES = frozenset({"temperature"})


def regression_metrics(measured: Iterable[float], simulated: Iterable[float],
                       quantity: str = "generic") -> dict:
    """Raw-interval regression metrics with an explicitly custom criterion.

    ASHRAE Guideline 14 is not attached here: that label requires a timestamped,
    hourly aggregated power/energy series. Device FMU tables are independent raw
    operating points, so their threshold result is deliberately named custom.

    ``quantity="temperature"`` suppresses every normalised metric (fix M-12,
    reviewer 1 comment 6). CVRMSE, NMBE and MAPE all divide by the mean of the
    measured signal, which presumes a ratio scale -- a meaningful zero. Celsius
    has no such zero: the identical physical error on a 26 degC outlet reads
    about 2% in Celsius and about 0.15% in Kelvin. Reporting either number
    invites a comparison that the units alone decide, so for temperatures only
    RMSE, MAE, MBE and R2 are returned, and no pass/fail criterion is offered.
    """
    measured_array = np.asarray(list(measured), dtype=float)
    simulated_array = np.asarray(list(simulated), dtype=float)
    mask = np.isfinite(measured_array) & np.isfinite(simulated_array)
    if not mask.any():
        raise ValueError("no finite metric pairs")
    measured_array = measured_array[mask]
    simulated_array = simulated_array[mask]
    residual = simulated_array - measured_array
    rmse = float(math.sqrt(float(np.mean(residual ** 2))))
    mean = float(np.mean(measured_array))
    nonzero = measured_array != 0.0
    mape = (
        float(np.mean(np.abs(residual[nonzero] / measured_array[nonzero])) * 100.0)
        if nonzero.any()
        else float("nan")
    )
    cvrmse = float(rmse / mean * 100.0) if mean else float("nan")
    nmbe = float(np.mean(residual) / mean * 100.0) if mean else float("nan")
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((measured_array - mean) ** 2))
    spread = float(math.sqrt(ss_tot / len(measured_array))) if ss_tot else 0.0
    base = {
        "N": int(len(measured_array)),
        "RMSE": rmse,
        "MAE": float(np.mean(np.abs(residual))),
        "MBE": float(np.mean(residual)),
        "R2": float(1.0 - ss_res / ss_tot) if ss_tot else float("nan"),
        # Model RMSE over the RMSE of simply predicting the measured mean. Below
        # 1 the model has skill; at or above 1 the trivial predictor would have
        # done better. Unlike CVRMSE this is a ratio of two errors in the same
        # unit, so it is meaningful on an interval scale too -- it is the only
        # normalised number a Celsius outlet temperature can honestly carry.
        #
        # It is here because CVRMSE cannot tell a good model from an easy
        # signal. Site A CDWP_01 reports 5.93% test CVRMSE on a power signal
        # that varies by 3.85% in the same window: skill 1.54, worse than a
        # constant. CHWP_01 reports 15.33% against 6.28%: skill 2.44. Both were
        # counted as validated devices.
        "skill_vs_mean": float(rmse / spread) if spread > 0 else float("nan"),
        "quantity": quantity,
    }
    if quantity in INTERVAL_QUANTITIES:
        base["criterion"] = "absolute_only_interval_scale"
        base["criterion_note"] = (
            "normalised metrics omitted: the zero of this scale is a convention, "
            "so CVRMSE/NMBE/MAPE depend on the unit chosen; skill_vs_mean is a "
            "ratio of two errors in the same unit and remains valid")
        return base
    base.update({
        "MAPE_pct": mape,
        "CVRMSE_pct": cvrmse,
        "NMBE_pct": nmbe,
        "criterion": "raw_interval_custom",
        "criterion_thresholds": {"CVRMSE_max_pct": 30.0, "abs_NMBE_max_pct": 5.0},
        "criterion_pass": bool(cvrmse <= 30.0 and abs(nmbe) <= 5.0),
    })
    return base


def gl14_pass(cvrmse_pct: float, nmbe_pct: float) -> bool:
    """ASHRAE Guideline 14 calibration criterion (hourly): CVRMSE<=30%, |NMBE|<=5%."""
    return bool(cvrmse_pct <= 30.0 and abs(nmbe_pct) <= 5.0)
