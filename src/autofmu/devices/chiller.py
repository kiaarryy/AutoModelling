"""Shared chiller data helpers (canonical-field accessors + steady-operating mask).

The chiller is modelled by driving the Buildings ElectricEIR / ElectricReformulatedEIR
FMUs (see ``chiller_fmu.py``). The former Python log-linear power surrogate
(``predict_chiller`` / ``calibrate_chiller``) was removed in FMU-5 -- device
thermal physics must come from the FMU, never a Python re-implementation. These
helpers (field access, cooling-load reconstruction, steady-operating filter) are
retained because the FMU table builder reuses them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RHO_WATER_KG_M3 = 1000.0
CP_WATER_J_KG_K = 4186.0


def _num(frame: pd.DataFrame, col: str) -> np.ndarray:
    if col not in frame.columns:
        return np.full(len(frame), np.nan)
    return pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)


def _cooling_load(frame: pd.DataFrame) -> np.ndarray:
    q = _num(frame, "cooling_load_W")
    if np.isfinite(q).sum() > 0:
        return q
    flow = _num(frame, "chw_flow_m3_h")
    dT = _num(frame, "tchwr_C") - _num(frame, "tchws_C")
    return RHO_WATER_KG_M3 * CP_WATER_J_KG_K * (flow / 3600.0) * dT


def _steady_mask(frame: pd.DataFrame, run_on: float, P: np.ndarray, Q: np.ndarray,
                 min_load_frac: float = 0.2) -> np.ndarray:
    """Operating + roughly steady: on, positive load/power, ABOVE a load floor
    (near-off points where P is tiny make COP=Q/P explode and are outside the
    chiller's valid modelling envelope), and small relative load change vs the
    previous sample (drops start-up / fast transients).

    The load floor is relative to the median power of the operating rows, which
    makes it self-defeating if the operating mask is wrong: LBNL CHI1's status
    bit admitted 12,392 standby samples, the median fell to the 1.94 kW standby
    draw, and the floor dropped to 0.39 kW -- it excluded nothing. So the mask
    comes from ``operating_mask``, which corroborates the bit against the
    compressor command before any of this runs.
    """
    from autofmu.contracts.profiles import get_profile
    from autofmu.modelability.operating import operating_mask

    on = operating_mask(frame, get_profile("chiller"), run_on).mask
    base = on & np.isfinite(P) & (P > 0) & np.isfinite(Q) & (Q > 0)
    if base.any():
        base &= P > (min_load_frac * float(np.nanmedian(P[base])))
    q = np.where(base, Q, np.nan)
    dq = np.abs(np.diff(q, prepend=q[0])) / np.where(np.abs(q) > 0, np.abs(q), np.nan)
    return base & (np.nan_to_num(dq, nan=1.0) < 0.10)
