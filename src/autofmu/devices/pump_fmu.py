"""L3 pump modelling via the exported PumpEmpiricalPower FMU.

The empirical power model (models/pump/PumpEmpiricalPower.mo) is:

    P_s = P_nominal * (c0 + c1*phi + c2*y + c3*y^3 + c4*phi*y)
    phi = m_flow / m_flow_nominal ,  y = clamp(speed_ratio)

autofmu fits the c0..c4 coefficients from the pump's own BMS data by ordinary
least squares (P is linear in them) for three candidate forms, drives the FMU
  (input-driven: measured m_flow + speed ratio) for each, selects by power CVRMSE
  on a disjoint selection period, and reports an untouched test period.

Candidates (feature subset -> coefficients):
  - affinity     : c3*y^3                          (cube-law, no flow)
  - speed_poly   : c0 + c2*y + c3*y^3              (speed only)
  - speed_flow   : c0 + c1*phi + c2*y + c3*y^3 + c4*phi*y   (full 5-term)
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from autofmu.devices import model_types as mt
from autofmu.fmu.runner import extracted_fmu, run_device_fmu
from autofmu.evaluation import metric_pairs, resolve_split
from autofmu.metrics import regression_metrics

DT_S = 300.0
FAN_NOM_HZ = 50.0
# Share of the running record that must carry attributed flow before a flow term
# is allowed to compete. Below this the term is dropped and every running row is
# kept, rather than keeping the term and throwing the record away.
FLOW_COVERAGE_MIN = 0.5
# Default empirical-power "forms": feature subset -> design-matrix columns (each
# maps to a c0..c4 slot). The config may override these per candidate under
# `forms:`; kept here as the fallback + for tests.
CANDIDATES = {
    "affinity": ["y3"],
    "speed_poly": ["1", "y", "y3"],
    "speed_flow": ["1", "phi", "y", "y3", "phi_y"],
}


def _num(frame, col):
    if col not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce")


def build_pump_table(frame: pd.DataFrame, thresholds: Optional[Mapping] = None) -> pd.DataFrame:
    """Input table (time, measured flow, speed ratio, measured power) for valid
    operating rows (running + positive power)."""
    thresholds = thresholds or {}
    run_on = float(thresholds.get("run_on", 0.5))
    fan_nom = float(thresholds.get("pump_speed_nominal_hz", FAN_NOM_HZ))
    P = _num(frame, "power_W")
    speed = _num(frame, "speed_Hz")
    flow = _num(frame, "attributed_flow_m3_h")
    if not flow.notna().any():
        flow = _num(frame, "flow_m3_h")
    m_flow = flow * 1000.0 / 3600.0
    # No flow meter (e.g. Tencent pumps): keep the device modellable speed-only by
    # filling a placeholder flow. The affinity / speed-poly candidates ignore the
    # flow term (c1 = c4 = 0); the speed-flow candidate degenerates to speed-poly
    # when the flow is constant, so this never fabricates a flow dependence.
    has_flow = np.isfinite(m_flow.to_numpy(dtype=float)) & (m_flow.to_numpy(dtype=float) > 0)
    flow_observed = has_flow.copy()
    y = (speed / fan_nom).clip(lower=0.0)
    running = P.gt(0.0) & speed.gt(run_on)
    # Partial attribution is not the same as no attribution. Solo-window flow
    # exists only while this pump is the only one of its group running, so it
    # thins out as the group grows: adding Site A's other eight pumps took
    # CDWP_01 from 28,619 attributed rows to 2,431 of the same 29,549 running
    # samples. Requiring finite flow then discarded 92% of the record for a
    # channel the selected candidate (affinity, speed-poly) never reads -- the
    # same mistake as demanding fan2_Hz from a single-drive tower.
    #
    # So the flow term is kept only where it is supported on a real share of the
    # record; below that it is dropped outright rather than half-filled, because
    # mixing measured flow with a constant on the same fit would invent a
    # relationship on the constant rows. flow_observed already gates which
    # candidate forms compete, so dropping it costs the device nothing it could
    # have earned.
    covered = float((has_flow & running.to_numpy()).sum()) / max(1, int(running.sum()))
    if covered < FLOW_COVERAGE_MIN:
        m_flow = pd.Series(1.0, index=frame.index)
        flow_observed = np.zeros(len(frame), dtype=bool)
    valid = running & np.isfinite(m_flow)
    out = pd.DataFrame({
        "m_flow_in": m_flow,
        "y_in": y,
        "P_meas_W": P,
        "flow_observed": flow_observed,
    }).loc[valid].reset_index(drop=True)
    if out.empty:
        out["time_s"] = []
        return out
    if "timestamp" in frame.columns:
        ts = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True).iloc[np.where(valid.to_numpy())[0]]
        out["time_s"] = (ts - ts.iloc[0]).dt.total_seconds().to_numpy(dtype=float)
    else:
        out["time_s"] = np.arange(len(out), dtype=float) * DT_S
    return out


def _compress_time(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    out["time_s"] = np.arange(len(out), dtype=float) * DT_S
    return out


def _features(table: pd.DataFrame, m_flow_nominal: float, names: Sequence[str]) -> np.ndarray:
    phi = (table["m_flow_in"].to_numpy(float) / max(m_flow_nominal, 1e-6)).clip(min=0.0)
    y = table["y_in"].to_numpy(float)
    cols = {"1": np.ones(len(table)), "phi": phi, "y": y, "y3": y ** 3, "phi_y": phi * y}
    return np.column_stack([cols[n] for n in names])


def _coeffs_from_fit(beta: np.ndarray, names: Sequence[str]) -> dict:
    """Map fitted feature coefficients to the FMU's c0..c4 slots."""
    slot = {"1": "c0", "phi": "c1", "y": "c2", "y3": "c3", "phi_y": "c4"}
    out = {f"c{i}": 0.0 for i in range(5)}
    for name, b in zip(names, beta):
        out[slot[name]] = float(b)
    return out


def _nominals(table: pd.DataFrame) -> dict:
    return {
        "P_nominal": float(table["P_meas_W"].quantile(0.95)),
        "m_flow_nominal": float(table["m_flow_in"].quantile(0.95)),
        "y_min": 0.05, "y_max": 1.20,
    }


def fit_candidate(table: pd.DataFrame, name: str, nominals: Mapping, train: np.ndarray,
                  feats: Optional[Sequence[str]] = None) -> dict:
    feats = list(feats) if feats is not None else CANDIDATES[name]
    X = _features(table, nominals["m_flow_nominal"], feats)
    p_norm = table["P_meas_W"].to_numpy(float) / max(nominals["P_nominal"], 1e-6)
    beta, *_ = np.linalg.lstsq(X[train], p_norm[train], rcond=None)
    return _coeffs_from_fit(beta, feats)


def drive_pump_fmu(fmu, table: pd.DataFrame, start_values: Mapping, stop: float,
                   input_columns: Sequence[str] = ("m_flow_in", "y_in")) -> pd.DataFrame:
    return run_device_fmu(
        fmu, start_values=dict(start_values), inputs=table, input_columns=list(input_columns),
        input_time_column="time_s", output=["P_s", "m_flow_s", "y_s"],
        stop_time=stop, output_interval=DT_S, require_finite=True,
    )


def _mover_start_values(nominals: Mapping, spec: Mapping) -> dict:
    """Start values for the Buildings-mover candidate: a real nominal flow shapes
    the preconfigured pump curve; dp's come from the contract; power is scaled."""
    return {"m_flow_nominal": float(nominals["m_flow_nominal"]),
            "y_min": float(nominals["y_min"]), "y_max": float(nominals["y_max"]),
            "dp_nominal": float(spec.get("dp_nominal", 300000.0)),
            "dp_system_nominal": float(spec.get("dp_system_nominal", 250000.0)),
            "P_scale": 1.0}


def _fit_pscale(frame: pd.DataFrame, table: pd.DataFrame, rows: np.ndarray) -> float:
    """Least-squares power scale: P_s is linear in P_scale, so driving at
    P_scale=1 gives the unit power; the optimal scale is the LS fit of measured
    power onto it over the training rows."""
    Ps = frame["P_s"].to_numpy(float)
    Pm = table["P_meas_W"].to_numpy(float)
    n = min(len(Ps), len(Pm))
    sel = rows[rows < n]
    ps, pm = Ps[sel], Pm[sel]
    ok = np.isfinite(ps) & np.isfinite(pm) & (ps > 0)
    if ok.sum() < 2 or float(np.sum(ps[ok] ** 2)) < 1e-9:
        return 1.0
    return float(np.sum(pm[ok] * ps[ok]) / np.sum(ps[ok] ** 2))


def _score(frame: pd.DataFrame, table: pd.DataFrame, rows: Optional[np.ndarray] = None) -> dict:
    n = min(len(frame), len(table))
    sel = np.arange(n) if rows is None else rows[rows < n]
    meas = table["P_meas_W"].to_numpy(float)[sel]
    sim = frame["P_s"].to_numpy(float)[sel]
    metric_pairs(meas, sim)
    ok = np.isfinite(meas) & np.isfinite(sim)
    m = regression_metrics(pd.Series(meas[ok]), pd.Series(sim[ok]))
    return {"N": int(ok.sum()), "P_CVRMSE_pct": m["CVRMSE_pct"], "P_NMBE_pct": m["NMBE_pct"],
            "P_skill": m["skill_vs_mean"],
            "criterion": "raw_interval_custom", "score": float(m["CVRMSE_pct"])}


def _resolve_fmu(cand: Mapping, fmu_root: Path) -> Path:
    raw = Path(str(cand["fmu"]))
    return raw if raw.is_absolute() else (fmu_root / raw)


def _subset(table: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if len(table) <= max_rows:
        sub = table.copy()
    else:
        sub = table.iloc[:: max(1, len(table) // max_rows)].reset_index(drop=True).copy()
    # Keep the real elapsed time before the FMU axis overwrites it. The FMU is
    # driven on a gap-free sequential axis, but the train/test split has to be
    # made in real time -- buffering against the compressed axis turns 261 days
    # into 7 and a 72 h buffer then swallows entire blocks (Site A pumps came
    # back with zero test rows).
    sub["source_time_s"] = sub["time_s"].to_numpy(dtype=float)
    sub["time_s"] = np.arange(len(sub), dtype=float) * DT_S
    return sub


def fit_pump_fmu(device_id: str, frame: pd.DataFrame, pump_cfg: Mapping, fmu_root: Path,
                 workdir: Path, thresholds: Optional[Mapping] = None) -> dict:
    thresholds = thresholds or {}
    min_rows = int(thresholds.get("min_calibration_rows", thresholds.get("min_full_physical_rows", 200)))
    subset_rows = int(thresholds.get("pump_subset_rows", 2000))
    workdir.mkdir(parents=True, exist_ok=True)

    table = build_pump_table(frame, thresholds)
    if len(table) < min_rows:
        return {"device_id": device_id, "status": "data_limited",
                "reason": f"{len(table)} valid pump rows < {min_rows}", "rows": [], "candidates": []}
    sub = _subset(table, subset_rows)
    n = len(sub)
    split = resolve_split(sub, thresholds, time_column="source_time_s",
                          native_rows=len(table))
    if not split.usable:
        return {"device_id": device_id, "status": split.status,
                "reason": (f"record cannot support an independent test segment: "
                           f"train={len(split.train)} select={len(split.select)} "
                           f"test={len(split.test)}"),
                "rows": [], "candidates": []}
    nominals = _nominals(sub.iloc[split.train])
    stop = float(sub["time_s"].iloc[-1])

    # build_pump_table has already zeroed this when coverage was too thin, so a
    # single stray attributed row can no longer let a flow term into the contest
    flow_observed = bool(sub["flow_observed"].mean() >= FLOW_COVERAGE_MIN)
    rows, candidates = [], []
    # Each candidate is a model *type*: the empirical-power FMU (ols feature-subset
    # forms) and, opt-in, the Buildings mover (scale_fit). Only enabled types compete.
    for spec in mt.select_candidates(pump_cfg["candidates"], pump_cfg.get("enabled_candidates")):
        strategy = spec.get("fit_strategy", "ols")
        fmu_path = _resolve_fmu(spec, fmu_root)
        fmu_rel = spec["fmu"]
        if not Path(fmu_path).exists():
            rows.append({"device_id": device_id, "candidate": spec["name"], "status": "fmu_missing"})
            continue

        if strategy == "scale_fit":
            # Buildings mover (SpeedControlled_y + system curve): speed-only drive
            # predicts flow + power from the pump curve; calibrate one power scale.
            # Needs a real measured nominal flow to shape the curve.
            name = spec["name"]
            if not flow_observed:
                rows.append({"device_id": device_id, "candidate": name, "status": "no_flow_for_mover"})
                continue
            inputs = list(spec.get("inputs", ["y_in"]))
            sv0 = _mover_start_values(nominals, spec)
            try:
                with extracted_fmu(fmu_path) as fmu_dir:
                    fr0 = drive_pump_fmu(fmu_dir, sub, sv0, stop, inputs)
                    pscale = _fit_pscale(fr0, sub, split.train)
                    sv = {**sv0, "P_scale": pscale}
                    fr = drive_pump_fmu(fmu_dir, sub, sv, stop, inputs)
            except Exception as exc:
                rows.append({"device_id": device_id, "candidate": name, "status": "fmu_failed", "reason": str(exc)})
                continue
            selection = _score(fr, sub, split.select)
            test = _score(fr, sub, split.test)
            rows.append({"device_id": device_id, "candidate": name, "status": "ok", "stage": "train"})
            rows.append({"device_id": device_id, "candidate": name, "status": "ok", "stage": "selection", **selection})
            rows.append({"device_id": device_id, "candidate": name, "status": "ok", "stage": "test", **test})
            candidates.append({"candidate": name, "coeffs": {"P_scale": pscale}, "nominals": sv0,
                               "inputs": inputs, "fmu": fmu_rel, "selection": selection, "test": test,
                               "score": selection["score"]})
            continue

        # default ols: empirical-power feature-subset forms (affinity/speed-poly/speed-flow)
        forms = spec.get("forms") or CANDIDATES
        req_flow = spec.get("requires_flow", ["speed_flow"])
        inputs = list(spec.get("inputs", ["m_flow_in", "y_in"]))
        names = [n for n in forms if (flow_observed or n not in req_flow)]
        with extracted_fmu(fmu_path) as fmu_dir:
            for name in names:
                coeffs = fit_candidate(sub, name, nominals, split.train, forms[name])
                try:
                    fr = drive_pump_fmu(fmu_dir, sub, {**nominals, **coeffs}, stop, inputs)
                except Exception as exc:
                    rows.append({"device_id": device_id, "candidate": name, "status": "fmu_failed", "reason": str(exc)})
                    continue
                selection = _score(fr, sub, split.select)
                test = _score(fr, sub, split.test)
                rows.append({"device_id": device_id, "candidate": name, "status": "ok", "stage": "train"})
                rows.append({"device_id": device_id, "candidate": name, "status": "ok", "stage": "selection", **selection})
                rows.append({"device_id": device_id, "candidate": name, "status": "ok", "stage": "test", **test})
                candidates.append({"candidate": name, "coeffs": coeffs, "nominals": nominals,
                                   "inputs": inputs, "fmu": fmu_rel, "selection": selection, "test": test,
                                   "score": selection["score"]})

    if not candidates:
        return {"device_id": device_id, "status": "no_candidate", "rows": rows, "candidates": []}
    best = min(candidates, key=lambda c: c["score"])
    return {"device_id": device_id, "status": "ok", "rows": rows, "candidates": candidates,
            "selected_candidate": best["candidate"], "best": best}


def validate_pump_fmu(device_id: str, frame: pd.DataFrame, candidate_params: Mapping, fmu_root: Path,
                      workdir: Path, thresholds: Optional[Mapping] = None) -> dict:
    table = _compress_time(build_pump_table(frame, thresholds))
    if table.empty:
        return {"status": "no_valid_rows"}
    workdir.mkdir(parents=True, exist_ok=True)
    fmu_path = _resolve_fmu(candidate_params, fmu_root)
    sv = {**dict(candidate_params["nominals"]), **dict(candidate_params["coeffs"])}
    inputs = list(candidate_params.get("inputs", ["m_flow_in", "y_in"]))
    with extracted_fmu(fmu_path) as fmu_dir:
        fr = drive_pump_fmu(fmu_dir, table, sv, float(table["time_s"].iloc[-1]), inputs)
    full = _score(fr, table)
    ts = pd.DataFrame({"measured": table["P_meas_W"].to_numpy(float)[: len(fr)],
                       "simulated": fr["P_s"].to_numpy(float)[: len(table)]})
    return {"status": "ok", "candidate": candidate_params["candidate"], "full_period": full,
            "rows_valid": len(table), "ts": ts}
