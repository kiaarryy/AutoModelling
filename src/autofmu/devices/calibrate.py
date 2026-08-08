"""L3 device-level empirical calibration from BMS data.

Scope: the data-driven equipment power models that the AlphaDataCenterCooling
paper itself fits empirically -- cooling-tower fan power (affinity law,
Eq. 15: P_fan = P_nom * w^3) and variable-speed pump power. These are real
least-squares calibrations on measured power (NOT passthrough), validated on a
held-out fold and scored with MAPE so results are comparable to the paper.

Candidates that include ``attributed_flow_m3_h`` test whether L2 loop-flow
attribution closes the speed->flow->power gap the paper resolved with an MLP
head model. They are only fit on the rows where attribution is valid.

Physical-FMU candidates (YorkCalc water temperature, ElectricReformulatedEIR)
need Dymola/FMPy and are a later milestone.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd

from autofmu.contracts.profiles import EquipmentProfile
from autofmu.metrics import regression_metrics

# builder: function(frame, opts) -> (design_matrix, feature_names)
FeatureBuilder = Callable[[pd.DataFrame, Mapping], Tuple[np.ndarray, List[str]]]


def _col(frame: pd.DataFrame, name: str) -> np.ndarray:
    if name not in frame.columns:
        return np.full(len(frame), np.nan)
    return pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)


def _pump_affinity(frame, opts):
    w = _col(frame, "speed_Hz") / float(opts.get("pump_nominal_hz", 50.0))
    return np.column_stack([w ** 3]), ["w3"]


def _pump_speed_poly(frame, opts):
    w = _col(frame, "speed_Hz") / float(opts.get("pump_nominal_hz", 50.0))
    return np.column_stack([np.ones_like(w), w, w ** 2, w ** 3]), ["1", "w", "w2", "w3"]


def _pump_speed_flow(frame, opts):
    w = _col(frame, "speed_Hz") / float(opts.get("pump_nominal_hz", 50.0))
    q = _col(frame, "attributed_flow_m3_h") / 100.0
    return np.column_stack([np.ones_like(w), w ** 3, q, q ** 2]), ["1", "w3", "q", "q2"]


def _ct_fan_affinity(frame, opts):
    fn = float(opts.get("fan_nominal_hz", 50.0))
    s = (_col(frame, "fan1_Hz") / fn) ** 3 + (_col(frame, "fan2_Hz") / fn) ** 3
    return np.column_stack([s]), ["sum_w3"]


def _ct_fan_affinity_intercept(frame, opts):
    fn = float(opts.get("fan_nominal_hz", 50.0))
    s = (_col(frame, "fan1_Hz") / fn) ** 3 + (_col(frame, "fan2_Hz") / fn) ** 3
    return np.column_stack([np.ones_like(s), s]), ["1", "sum_w3"]


def _ct_fan_flow(frame, opts):
    fn = float(opts.get("fan_nominal_hz", 50.0))
    s = (_col(frame, "fan1_Hz") / fn) ** 3 + (_col(frame, "fan2_Hz") / fn) ** 3
    q = _col(frame, "attributed_flow_m3_h") / 100.0
    return np.column_stack([np.ones_like(s), s, q, q ** 2]), ["1", "sum_w3", "q", "q2"]


# name -> (builder, required feature columns)
CANDIDATES: Dict[str, Dict[str, Tuple[FeatureBuilder, List[str]]]] = {
    "pump": {
        "affinity_power": (_pump_affinity, ["speed_Hz"]),
        "speed_poly_power": (_pump_speed_poly, ["speed_Hz"]),
        "speed_flow_power": (_pump_speed_flow, ["speed_Hz", "attributed_flow_m3_h"]),
    },
    "cooling_tower": {
        "fan_affinity_power": (_ct_fan_affinity, ["fan1_Hz", "fan2_Hz"]),
        "fan_affinity_intercept": (_ct_fan_affinity_intercept, ["fan1_Hz", "fan2_Hz"]),
        "fan_flow_power": (_ct_fan_flow, ["fan1_Hz", "fan2_Hz", "attributed_flow_m3_h"]),
    },
}


def feature_columns(equipment_type: str, candidate: str) -> List[str]:
    return list(CANDIDATES.get(equipment_type, {}).get(candidate, (None, []))[1])


def predict(frame: pd.DataFrame, equipment_type: str, candidate: str, params: Mapping,
            thresholds: Mapping = None) -> np.ndarray:
    """Apply a fitted candidate model to a frame (for full-period validation)."""
    builder = CANDIDATES[equipment_type][candidate][0]
    x, names = builder(frame, thresholds or {})
    beta = np.array([float(params[n]) for n in names], dtype=float)
    return np.maximum(x @ beta, 0.0)


def on_mask(frame: pd.DataFrame, profile: EquipmentProfile, run_on: float) -> np.ndarray:
    """Operating rows for this device: the status bit corroborated by the
    profile's liveness channel (see ``modelability.operating``). Fitting and
    full-period validation must agree on which rows count, so both go through
    here."""
    from autofmu.modelability.operating import operating_mask

    return operating_mask(frame, profile, run_on).mask


def _split(n: int, mode: str, fold: int) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.arange(n)
    if mode == "tail":
        cut = max(1, int(n * 0.7))
        return idx[:cut], idx[cut:]
    return idx[idx % fold != 0], idx[idx % fold == 0]  # interleaved k-fold holdout


def calibrate_power_model(device_id, frame, profile: EquipmentProfile, thresholds=None):
    """Fit candidate power models (each on its own valid rows), validate on a
    held-out fold, select by MAPE. Returns (all_candidate_rows, best, best_ts)."""
    thresholds = thresholds or {}
    run_on = float(thresholds.get("run_on", 0.5))
    # This module is the explicitly enabled legacy empirical *power* engine.
    # Physical cooling-tower FMUs use heat_rejection_W as their primary target;
    # the legacy fan-affinity candidate remains a power model by definition.
    target = "power_W"
    candidates = CANDIDATES.get(profile.equipment_type, {})
    if not candidates:
        return [{"device_id": device_id, "status": "no_empirical_candidates"}], {}, pd.DataFrame()

    split_mode = "tail" if str(thresholds.get("validation_split", "interleaved")) == "tail" else "interleaved"
    fold = max(2, int(thresholds.get("validation_fold", 3)))
    min_rows = int(thresholds.get("min_calibration_rows", 200))
    base_on = on_mask(frame, profile, run_on)

    rows: List[dict] = []
    best: dict = {}
    best_ts = pd.DataFrame()
    for name, (builder, req_cols) in candidates.items():
        finite = base_on & np.isfinite(_col(frame, target))
        for c in req_cols:
            finite &= np.isfinite(_col(frame, c))
        valid = frame.loc[finite].copy()
        if "timestamp" in valid:
            valid = valid.sort_values("timestamp")
        n = len(valid)
        if n < min_rows:
            rows.append({"device_id": device_id, "equipment_type": profile.equipment_type,
                         "candidate": name, "target": target, "status": "insufficient_rows",
                         "n_valid": int(n), "reason": f"{n} valid rows < {min_rows} (needs {req_cols})"})
            continue
        tr, va = _split(n, split_mode, fold)
        train, validate = valid.iloc[tr], valid.iloc[va]
        y_tr, y_va = _col(train, target), _col(validate, target)
        x_tr, names = builder(train, thresholds)
        beta, *_ = np.linalg.lstsq(x_tr, y_tr, rcond=None)
        x_va, _ = builder(validate, thresholds)
        pred = np.maximum(x_va @ beta, 0.0)
        metric = regression_metrics(y_va, pred)
        row = {
            "device_id": device_id, "equipment_type": profile.equipment_type, "candidate": name,
            "target": target, "status": "ok", "n_train": int(len(train)), "n_validate": int(len(validate)),
            "validation_split": split_mode, "params": {nm: float(b) for nm, b in zip(names, beta)},
            **{k: metric[k] for k in ("N", "RMSE", "MAE", "MAPE_pct", "CVRMSE_pct",
                                      "NMBE_pct", "skill_vs_mean")},
        }
        rows.append(row)
        if not best or row["MAPE_pct"] < best["MAPE_pct"]:
            best = row
            ts = validate[["timestamp"]].copy() if "timestamp" in validate else pd.DataFrame()
            ts["measured"], ts["simulated"], ts["candidate"] = y_va, pred, name
            best_ts = ts
    return rows, best, best_ts
