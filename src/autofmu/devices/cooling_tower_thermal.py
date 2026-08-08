"""Shared cooling-tower data helpers (feature extraction + heat-rejection).

The cooling tower is modelled by driving the Buildings Merkel / YorkCalc FMUs
(see ``cooling_tower_fmu.py``). The former Python York/Merkel surrogate
(``predict_ct_tout`` / ``calibrate_ct_thermal``) was removed in FMU-5 -- device
thermal physics must come from the FMU. ``_features`` (canonical -> Tin/Tout/Twb/
flow/fan-count, with validity) and ``_q`` are retained because the FMU table
builder reuses them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from autofmu.observability.attrition import AttritionLedger

CP_WATER = 4180.0
RHO = 1000.0
FAN_ON_HZ = 5.0   # matches the active_count reconstruction threshold


def _num(frame, col):
    if col not in frame.columns:
        return np.full(len(frame), np.nan)
    return pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)


def _features(frame, thresholds):
    fan_nom = float(thresholds.get("fan_nominal_hz", 50.0))
    Tin = _num(frame, "tcwr_C")          # entering tower (hot return)
    Tout = _num(frame, "tcws_1_C")       # leaving tower (cold supply)
    if not np.isfinite(Tout).any():
        Tout = _num(frame, "tcws_C")
    Twb = _num(frame, "twb_C")
    total_mdot = _num(frame, "attributed_flow_m3_h") * RHO / 3600.0   # kg/s total
    nfan = _num(frame, "fans_on_count")
    nfan = np.where(np.isfinite(nfan) & (nfan > 0), nfan, np.nan)
    # Mean speed of the drives that are actually running, normalised by nominal.
    #
    # Summing fan1_Hz and fan2_Hz breaks on towers that expose a single drive:
    # the absent channel reads NaN, the sum goes NaN, and the device is gated
    # out entirely -- which is what silenced Site A CT_06 and CT_07. Averaging
    # over the running channels is also the right physics: a tower with one of
    # two cells running at 40 Hz is running that cell at 0.8, not at 0.4. How
    # many cells that represents is the job of fans_on_count, not of this ratio.
    fan_columns = [c for c in ("fan1_Hz", "fan2_Hz") if c in frame.columns]
    if fan_columns:
        speeds = np.stack([_num(frame, c) for c in fan_columns])
        running = np.isfinite(speeds) & (speeds > FAN_ON_HZ)
        n_running = running.sum(axis=0)
        total_hz = np.where(running, speeds, 0.0).sum(axis=0)
        fan_hz = np.divide(total_hz, n_running, out=np.full(len(nfan), np.nan),
                           where=n_running > 0)
    else:
        fan_hz = np.full(len(nfan), np.nan)
    fr_air = fan_hz / fan_nom
    n_cells = 2.0
    mdot_cell = total_mdot / n_cells
    tran = Tin - Tout
    tapp = Tout - Twb
    mdot_nom = np.nanmedian(mdot_cell[np.isfinite(mdot_cell) & (mdot_cell > 0)]) if np.isfinite(mdot_cell).any() else 1.0
    fr_wat = mdot_cell / max(float(mdot_nom), 1e-9)
    rlg = np.clip(fr_wat / np.clip(fr_air, 0.05, 1.0), 0.05, 8.0)

    # Exclusions are applied one at a time and attributed (M-11), and the
    # target-conditioned ones are separated from the operating-state ones
    # (M-10).  Availability of the validation reference is a gate -- you cannot
    # score a row with no measured outlet -- but *conditioning* on its value is
    # not, so a tower running backwards or below the wet bulb is quarantined and
    # counted rather than quietly dropped.
    ledger = AttritionLedger(n_total=len(Tin), family="cooling_tower")
    ledger.gate("entering water temperature present", np.isfinite(Tin),
                category="missing_channel")
    ledger.gate("wet-bulb temperature present", np.isfinite(Twb),
                category="missing_channel")
    ledger.gate("attributed flow present", np.isfinite(total_mdot),
                category="missing_channel")
    ledger.gate("active fan count present", np.isfinite(nfan),
                category="missing_channel")
    ledger.gate("leaving water temperature present", np.isfinite(Tout),
                category="missing_reference")
    # Absolute floor, not merely non-zero. Below about a kilogram per second
    # there is no meaningful water passing through the tower, and the heat
    # rejection reconstructed as m*cp*dT becomes a large relative error on a
    # tiny base. FMU_Modelica used the same 1 kg/s threshold; applying it here
    # brings the retained record to within one row of that pipeline on CT_01,
    # CT_03 and CT_05.
    min_flow = float(thresholds.get("ct_min_total_flow_kg_s", 1.0))
    ledger.gate(f"total water flow above {min_flow:g} kg/s", total_mdot > min_flow,
                category="operating_state")
    ledger.gate("fans running", fr_air > 0.05, category="operating_state")
    ledger.quarantine("leaving water hotter than entering (range <= 0)",
                      np.isfinite(tran) & (tran <= 0.0))
    ledger.quarantine("leaving water at or below wet bulb (approach <= 0)",
                      np.isfinite(tapp) & (tapp <= 0.0))

    return dict(Tin=Tin, Tout=Tout, Twb=Twb, tran=tran, tapp=tapp, rlg=rlg,
                total_mdot=total_mdot, nfan=nfan, fan_hz=fan_hz, fr_air=fr_air,
                valid=ledger.mask, ledger=ledger)


def _q(total_mdot, Tin, Tout):
    return total_mdot * CP_WATER * (Tin - Tout)   # total tower heat rejection (W)
