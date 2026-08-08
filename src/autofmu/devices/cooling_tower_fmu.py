"""L3 cooling-tower modelling via the Buildings Merkel / YorkCalc FMUs.

Replaces the Python TOut surrogate with the real CTM_0FMU (Merkel) / CTY_0FMU
(YorkCalc) FMUs as the simulator for both the parameter search and the
validation. The thermal parameters and fan-power curve are grid-searched
FMU-in-the-loop (ported, de-hard-coded, from
FMU_Modelica/scripts/calibrate_site_a_ct_fmus.py); the FMU computes T / Q / P.

Two autofmu/L2 improvements over the original script:

- the measured table is built from L2 canonical data (attributed flow + wet-bulb
  join already done by the attribute stage), and
- the per-cell FMU heat rejection is scaled to whole-tower totals with the L2
  data-driven per-timestep effective fan count (``fans_on_count``) instead of the
  hard-coded ``EXTERNAL_TOTAL_SCALE = 2``.

Selection between the thermal candidates is on leaving-water RMSE alone, which
is the rule the manuscript states; heat rejection is reported but deliberately
kept out of the ranking because flow weighting can reverse the two.

Fan power cannot discriminate between the candidates -- they share a fan-power
representation, and Site A confirms it (15.9% against 15.9% on CT-01) -- but it
is scored all the same, because the fan curve's own parameters have to be fitted
against the only quantity they reach. Ranking them by the thermal score instead
leaves them chosen by a tie-break.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from autofmu.devices import model_types as mt
from autofmu.devices.cooling_tower_thermal import _features, _num
from autofmu.evaluation import metric_pairs, resolve_split
from autofmu.fmu.runner import extracted_fmu, run_device_fmu
from autofmu.metrics import regression_metrics

TABLE_COLUMNS = ["time_s", "Tin_C", "Tout_meas_C", "Twb_C", "TRan_C", "TAppAct_C",
                 "mdot_cell_kgps", "y_used", "fanHz", "fans_on_count", "Q_flow_W", "PFan_meas_W"]
DT_S = 300.0
FAN_NOM_HZ = 50.0
CT_OUTPUTS = ["TOut_m", "TOut_s", "Q_m", "Q_s", "P_m", "P_s"]


# --------------------------------------------------------------------------- #
# Table build (from L2 canonical) + nominal estimation
# --------------------------------------------------------------------------- #
def build_ct_table(frame: pd.DataFrame, thresholds: Optional[Mapping] = None) -> pd.DataFrame:
    """CT FMU input table from L2 canonical data, per-cell flow scaled by the
    L2 per-timestep effective fan count."""
    thresholds = thresholds or {}
    f = _features(frame, thresholds)
    v = f["valid"]
    idx = np.where(v)[0]
    if len(idx) == 0:
        return pd.DataFrame(columns=TABLE_COLUMNS)
    # Drop degenerate near-zero-flow rows. When a tower's paired chiller is
    # blocked, loop-flow attribution collapses (e.g. Site A CT_05: ~96% of rows
    # carry almost no flow), and the FMU would otherwise be scored on rows where
    # it sees no water. Keep rows whose total flow is a real fraction of the
    # tower's own operating (p90) flow; if too few remain the fit gates honestly.
    tot = f["total_mdot"][idx]
    floor = max(1.0, float(thresholds.get("ct_min_flow_frac", 0.25)) * float(np.nanpercentile(tot, 90)))
    idx = idx[tot >= floor]
    if len(idx) == 0:
        return pd.DataFrame(columns=TABLE_COLUMNS)
    nfan = f["nfan"][idx]
    # mean running-drive speed, computed once in _features so the single-VSD
    # towers (CT_06 / CT_07) are handled in one place
    fanHz = f["fan_hz"][idx]
    # REAL elapsed time (with gaps) so select_windows can detect contiguous
    # physical runs -- compressing to arange*300 here would let a calibration
    # window straddle real-time discontinuities (matches the old prep behaviour).
    if "timestamp" in frame.columns:
        ts = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True).iloc[idx]
        time_s = (ts - ts.iloc[0]).dt.total_seconds().to_numpy(dtype=float)
    else:
        time_s = np.arange(len(idx), dtype=float) * DT_S
    table = pd.DataFrame({
        "time_s": np.asarray(time_s, dtype=float),
        "Tin_C": f["Tin"][idx], "Tout_meas_C": f["Tout"][idx], "Twb_C": f["Twb"][idx],
        "TRan_C": f["tran"][idx], "TAppAct_C": f["tapp"][idx],
        "mdot_cell_kgps": f["total_mdot"][idx] / np.where(nfan > 0, nfan, 1.0),
        "y_used": np.clip(fanHz / FAN_NOM_HZ, 0.0, 1.0),
        "fanHz": fanHz, "fans_on_count": nfan,
        "Q_flow_W": f["total_mdot"][idx] * 4180.0 * f["tran"][idx],
        "PFan_meas_W": _num(frame, "power_W")[idx],
    })
    finite = np.all(np.isfinite(table.drop(columns=["time_s"]).to_numpy(dtype=float)), axis=1)
    return table.loc[finite].reset_index(drop=True)


def _compress_time(table: pd.DataFrame) -> pd.DataFrame:
    """Re-index rows to sequential DT_S steps (for an independent-point full-period
    drive, where real-time gaps would create a huge empty simulation horizon)."""
    out = table.copy()
    out["time_s"] = np.arange(len(out), dtype=float) * DT_S
    return out


def write_dymola_table(path: Path, table: pd.DataFrame, table_name: str = "CT_data") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("#1\n")
        fh.write(f"double {table_name}({len(table)},{len(table.columns)})\n")
        for _, row in table.iterrows():
            fh.write(",".join(f"{float(row[c]):.10g}" for c in table.columns) + "\n")


def base_values(table: pd.DataFrame) -> Dict[str, float]:
    return {
        "m_flow_nominal": float(table["mdot_cell_kgps"].median()),
        "TAirInWB_nominal": float(table["Twb_C"].median() + 273.15),
        "TApp_nominal": max(0.2, float(table["TAppAct_C"].median())),
        "TRan_nominal": max(0.3, float(table["TRan_C"].median())),
        "TWatIn_nominal": float(table["Tin_C"].median() + 273.15),
        "TWatOut_nominal": float(table["Tout_meas_C"].median() + 273.15),
        "yMin": 0.05, "nFan": 1.0, "fraFreCon": 0.1,
    }


# --------------------------------------------------------------------------- #
# Candidate grids + FMU start-value builders (ported, de-hard-coded)
# --------------------------------------------------------------------------- #
def _fan_curve(alpha: float, prefix: str) -> Dict[str, float]:
    rv = [0.0, 0.1, 0.3, 0.6, 1.0]
    return {f"{prefix}[{i + 1}]": float(v ** alpha) for i, v in enumerate(rv)}


def estimate_pfan_nominal(table: pd.DataFrame, alpha: float) -> float:
    y = table["y_used"].clip(lower=0.05, upper=1.0)
    nfan = table["fans_on_count"].clip(lower=1.0)
    est = table["PFan_meas_W"] / (nfan * y ** alpha)
    est = est.replace([np.inf, -np.inf], np.nan).dropna()
    est = est[est > 0]
    return float(est.median()) if len(est) else float(table["PFan_meas_W"].median())


# --------------------------------------------------------------------------- #
# Drive + score (per-row fan-count scaling of Q replaces the hard-coded x2)
# --------------------------------------------------------------------------- #
def _drive(fmu, table_path: Path, spec: Mapping, base: Mapping, params: Mapping, stop: float) -> pd.DataFrame:
    """Drive a cooling-tower FMU. ``spec`` is the candidate's declarative type
    spec (config candidate at fit time, stored payload at validate time): it
    supplies ``table_param`` and the ``static_start_values`` map that replace the
    former per-model ``cty_values`` / ``ctm_values`` builders."""
    table_param = spec["table_param"]
    sv = mt.assemble_start_values(spec.get("static_start_values"), base, params)
    return run_device_fmu(fmu, start_values=sv, table_overrides={table_param: table_path},
                          output=CT_OUTPUTS,
                          stop_time=stop, output_interval=DT_S, fmi_type="CoSimulation",
                          require_finite=True)


def _fit_york27_closed_loop(table: pd.DataFrame, base: Mapping) -> Dict[str, float]:
    """Identify the 27 York coefficients by closed-loop simulation-error fitting.

    The declarative grid cannot do this job. It scales each coefficient by
    0.9/1.0/1.1 one axis at a time, which is a coordinate search over a
    27-dimensional space whose axes are strongly coupled; and it optimises the
    open-loop residual. ``york27_fit.fit_closed_loop`` minimises the error of
    the *closed-loop* outlet temperature and constrains the result to be well
    posed, which is what stops a fitted vector from being multi-valued in
    operation (Site A: 98.2% of CT-06's operating points before the constraint,
    0.0% after).

    Returned in FMU parameter-name space so the caller can treat it exactly like
    a grid result.
    """
    from autofmu.devices.york27_fit import fit_closed_loop
    from autofmu.devices.york27_reference import (
        MBL_DEFAULT_COEFFICIENTS, York27Params, solve_frwat0)

    m_nominal = float(base.get("m_flow_nominal", 1.0)) or 1.0
    frwat0 = solve_frwat0(
        twb_nominal_c=float(base["TAirInWB_nominal"]) - 273.15,
        tran_nominal=float(base["TRan_nominal"]),
        tapp_nominal=float(base["TApp_nominal"]),
        coefficients=MBL_DEFAULT_COEFFICIENTS)
    if not np.isfinite(frwat0):
        frwat0 = 1.0

    result = fit_closed_loop(
        tin_c=table["Tin_C"].to_numpy(float),
        twb_c=table["Twb_C"].to_numpy(float),
        tout_meas_c=table["Tout_meas_C"].to_numpy(float),
        mdot_cell_kgps=table["mdot_cell_kgps"].to_numpy(float),
        y_used=table["y_used"].to_numpy(float),
        params=York27Params(m_flow_nominal=m_nominal), frwat0=frwat0)
    params = {f"f[{i + 1}]": float(v) for i, v in enumerate(result.coefficients)}
    params["FRWat0"] = float(frwat0)
    return params


def _score(frame: pd.DataFrame, fans: np.ndarray, rows: Optional[np.ndarray] = None) -> dict:
    """T from per-cell FMU output (no scaling: outlet temp is identical per cell).
    Q_m is the table's TOTAL measured heat rejection; the FMU's per-cell Q_s is
    scaled up to a total by the per-row effective fan/cell count."""
    n = min(len(frame), len(fans))
    sel = np.arange(1, n) if rows is None else rows[(rows >= 1) & (rows < n)]
    Tm = frame["TOut_m"].to_numpy(float)[sel] - 273.15
    Ts = frame["TOut_s"].to_numpy(float)[sel] - 273.15
    Qm = frame["Q_m"].to_numpy(float)[sel]              # already total in the table
    Qs = frame["Q_s"].to_numpy(float)[sel] * fans[sel]  # per-cell -> total
    metric_pairs(Tm, Ts)
    metric_pairs(Qm, Qs)
    ok = np.isfinite(Tm) & np.isfinite(Ts) & np.isfinite(Qm) & np.isfinite(Qs)
    mT = regression_metrics(pd.Series(Tm[ok]), pd.Series(Ts[ok]))
    mT_abs = regression_metrics(pd.Series(Tm[ok]), pd.Series(Ts[ok]),
                                quantity="temperature")
    mQ = regression_metrics(pd.Series(Qm[ok]), pd.Series(Qs[ok]))

    # Fan power, scored separately. It cannot discriminate between the thermal
    # candidates -- both use the same fan-power representation, which the Site A
    # comparison confirms (15.9% vs 15.9% on CT-01) -- but the fan curve's own
    # parameters have to be fitted against something, and this is the only
    # quantity they reach. Leaving it unscored is what let the fan exponent be
    # chosen by a tie-break; see _search_model.
    Pm = frame["P_m"].to_numpy(float)[sel]
    Ps = frame["P_s"].to_numpy(float)[sel] * fans[sel]   # per-fan -> total
    ok_p = np.isfinite(Pm) & np.isfinite(Ps)
    power = {}
    if ok_p.sum() >= 10:
        mP = regression_metrics(pd.Series(Pm[ok_p]), pd.Series(Ps[ok_p]))
        power = {"P_N": int(ok_p.sum()), "P_CVRMSE_pct": mP["CVRMSE_pct"],
                 "P_NMBE_pct": mP["NMBE_pct"], "P_skill": mP["skill_vs_mean"]}

    # Rank on leaving-water RMSE alone (fix M-12, and M-01's tie rule). Section
    # 3.3 states the rule as "the candidate with the lower out-of-sample
    # leaving-water-temperature RMSE is retained", and section 2.6 explains why
    # reconstructed heat is deliberately kept out of it: flow weighting can
    # reverse the two, as it does on CT-01 where Merkel wins on temperature and
    # loses on heat. Heat and fan power are reported, not voted on. A CVRMSE on
    # Celsius would not be a physical quantity in any case.
    return {"N": int(ok.sum()),
            "T_RMSE_K": mT_abs["RMSE"], "T_MBE_K": mT_abs["MBE"],
            "T_CVRMSE_pct": mT["CVRMSE_pct"], "T_NMBE_pct": mT["NMBE_pct"],
            "Q_CVRMSE_pct": mQ["CVRMSE_pct"], "Q_NMBE_pct": mQ["NMBE_pct"],
            # The leaving-water temperature is the ranking quantity and it is on
            # an interval scale, so M-12 leaves it with no normalised number at
            # all. Skill is a ratio of two errors in the same unit and is the
            # one it can honestly carry.
            "T_skill": mT_abs["skill_vs_mean"], "Q_skill": mQ["skill_vs_mean"],
            **power,
            "criterion": "raw_interval_custom",
            "score": float(mT_abs["RMSE"])}


# --- Representative steady-window selection (reused verbatim from the validated
# FMU_Modelica/scripts/calibrate_site_a_ct_fmus.py, so calibration data quality
# matches the baseline instead of a naive full-period downsample). ---
WINDOW_FEATURES = ["TRan_C", "TAppAct_C", "mdot_cell_kgps", "y_used", "PFan_meas_W", "Q_flow_W"]
WINDOW_ROWS = 288
WINDOW_STRIDE = 12
CALIBRATION_WINDOWS = 3


def _contiguous_runs(df: pd.DataFrame, step: float = DT_S):
    runs, start, prev = [], 0, None
    for idx, t in enumerate(df["time_s"]):
        if idx == 0:
            prev = t
            continue
        if abs(t - prev - step) > 1e-6:
            runs.append((start, idx - 1))
            start = idx
        prev = t
    runs.append((start, len(df) - 1))
    return runs


def _representative_candidates(table: pd.DataFrame, max_rows=WINDOW_ROWS, stride=WINDOW_STRIDE):
    feats = [c for c in WINDOW_FEATURES if c in table.columns]
    if len(table) < max_rows or not feats:
        return [(0.0, 0, min(len(table), max_rows) - 1)]
    full = table[feats]
    fmean, fq25, fq75 = full.mean(), full.quantile(0.25), full.quantile(0.75)
    scale = (fq75 - fq25).replace(0.0, np.nan).fillna(full.std()).replace(0.0, 1.0).fillna(1.0)
    out = []
    for rs, re in _contiguous_runs(table):
        if re - rs + 1 < max_rows:
            continue
        for s in range(rs, re - max_rows + 2, stride):
            win = table.iloc[s:s + max_rows]
            score = float(((win[feats].mean() - fmean).abs() / scale).mean()
                          + 0.5 * ((win[feats].quantile(0.25) - fq25).abs() / scale).mean()
                          + 0.5 * ((win[feats].quantile(0.75) - fq75).abs() / scale).mean())
            out.append((score, s, s + max_rows - 1))
    return sorted(out, key=lambda r: r[0]) or [(0.0, 0, min(len(table), max_rows) - 1)]


def select_windows(table: pd.DataFrame, n_windows=CALIBRATION_WINDOWS, max_rows=WINDOW_ROWS) -> pd.DataFrame:
    """Pick up to n non-overlapping representative steady windows for calibration."""
    selected = []
    for score, s, e in _representative_candidates(table, max_rows):
        if any(not (e < ss - max_rows or s > ee + max_rows) for _, ss, ee in selected):
            continue
        selected.append((score, s, e))
        if len(selected) >= n_windows:
            break
    if not selected:
        selected = [(0.0, 0, min(len(table), max_rows) - 1)]
    out = pd.concat([table.iloc[s:e + 1] for _, s, e in selected], ignore_index=True)
    out["time_s"] = np.arange(len(out), dtype=float) * DT_S
    return out


def _search_model(spec: Mapping, fmu_path: Path, table: pd.DataFrame, base: Mapping,
                  workdir: Path, fan_alphas: Sequence[float]) -> dict:
    """Grid-search a candidate type's thermal parameters, then its fan-power
    curve, FMU-in-the-loop. The thermal grid and the fan-power parameter names
    are declared on the candidate ``spec`` (config), not hard-coded per model."""
    model = spec["name"]
    sub_path = workdir / f"_ct_subset_{model}.txt"
    write_dymola_table(sub_path, table)
    fans = table["fans_on_count"].to_numpy(float)
    stop = float(table["time_s"].iloc[-1])
    fan_power = spec["fan_power"]
    prefix, pkey = fan_power["curve_prefix"], fan_power["nominal_key"]
    strategy = spec.get("fit_strategy", "grid_search")
    with extracted_fmu(fmu_path) as fmu_dir:
        best_thermal, best_score = None, float("inf")
        grid = spec.get("grid") or {}
        if strategy == "closed_loop_simulation_error":
            best_thermal = _fit_york27_closed_loop(table, base)
            best_score = float("nan")
        elif grid.get("mode") == "coordinate":
            best_thermal = {}
            for axis in grid.get("axes", []):
                axis_best, axis_score = None, float("inf")
                for axis_params in mt.expand_grid({"mode": "coordinate", "axes": [axis]}, base):
                    cand = {**best_thermal, **axis_params}
                    try:
                        frame = _drive(fmu_dir, sub_path, spec, base, cand, stop)
                    except Exception:
                        continue
                    s = _score(frame, fans)["score"]
                    if s < axis_score:
                        axis_score, axis_best = s, dict(cand)
                if axis_best is not None:
                    best_thermal = axis_best
                    best_score = axis_score
            if not best_thermal:
                best_thermal = None
        else:
            for cand in mt.expand_grid(grid, base):
                try:
                    frame = _drive(fmu_dir, sub_path, spec, base, cand, stop)
                except Exception:
                    continue
                s = _score(frame, fans)["score"]
                if s < best_score:
                    best_score, best_thermal = s, dict(cand)
        if best_thermal is None:
            return {"status": "no_thermal_candidate"}
        # Fan stage: rank on fan-power error, not on the thermal score. The fan
        # curve and its nominal power reach only P_s -- they have no path to the
        # outlet temperature or the heat rejection -- so ranking them by the
        # thermal score picks the exponent by a tie-break and leaves the fan
        # parameters effectively unfitted. Measured on CT-07: TOut RMSE is
        # identical to four decimals across the whole alpha grid while fan-power
        # CVRMSE moves from 7.8% to 14.8%. Fixing the equivalent defect in the
        # standalone refit script took Site A fan power from 30-57% to 12-27%.
        best, best_full = None, float("inf")
        for alpha in fan_alphas:
            pfan = estimate_pfan_nominal(table, alpha)
            cand = {**best_thermal, pkey: pfan, **_fan_curve(alpha, prefix)}
            try:
                frame = _drive(fmu_dir, sub_path, spec, base, cand, stop)
            except Exception:
                continue
            metrics = _score(frame, fans)
            s = metrics.get("P_CVRMSE_pct")
            if s is None or not np.isfinite(s):
                continue
            if s < best_full:
                best_full, best = float(s), dict(cand)
        if best is None:
            best = best_thermal
    return {"status": "ok", "params": best, "model": model}


# --------------------------------------------------------------------------- #
# Orchestration: fit (subset) + select, and full-period validate
# --------------------------------------------------------------------------- #
def _resolve_fmu(cand: Mapping, fmu_root: Path) -> Path:
    raw = Path(str(cand["fmu"]))
    return raw if raw.is_absolute() else (fmu_root / raw)


def fit_ct_fmu(device_id: str, frame: pd.DataFrame, ct_cfg: Mapping, fmu_root: Path,
               workdir: Path, thresholds: Optional[Mapping] = None) -> dict:
    thresholds = thresholds or {}
    min_rows = int(thresholds.get("min_calibration_rows", thresholds.get("min_full_physical_rows", 200)))
    fan_alphas = (1.2, 1.6, 2.0, 2.5, 3.0)
    workdir.mkdir(parents=True, exist_ok=True)

    table = build_ct_table(frame, thresholds)
    if len(table) < min_rows:
        return {"device_id": device_id, "status": "data_limited",
                "reason": f"{len(table)} valid CT rows < {min_rows}", "rows": [], "candidates": []}
    split = resolve_split(table, thresholds)
    if not split.usable:
        return {"device_id": device_id, "status": split.status,
                "reason": (f"record cannot support an independent test segment: "
                           f"train={len(split.train)} select={len(split.select)} "
                           f"test={len(split.test)}"),
                "rows": [], "candidates": []}
    # Representative windows and all fitted nominals come only from training.
    train_table = table.iloc[split.train].reset_index(drop=True)
    subset = select_windows(train_table)
    base = base_values(subset)
    evaluation = _compress_time(table)
    eval_path = workdir / f"{device_id}_ct_evaluation.txt"
    write_dymola_table(eval_path, evaluation)
    eval_stop = float(evaluation["time_s"].iloc[-1])
    eval_fans = evaluation["fans_on_count"].to_numpy(float)

    rows, candidates = [], []
    for cand in ct_cfg["candidates"]:
        model = cand["name"]
        fmu_path = _resolve_fmu(cand, fmu_root)
        if not Path(fmu_path).exists():
            rows.append({"device_id": device_id, "candidate": model, "status": "fmu_missing"})
            continue
        res = _search_model(cand, fmu_path, subset, base, workdir, fan_alphas)
        if res.get("status") != "ok":
            rows.append({"device_id": device_id, "candidate": model, "status": res.get("status")})
            continue
        with extracted_fmu(fmu_path) as fmu_dir:
            try:
                fr = _drive(fmu_dir, eval_path, cand, base, res["params"], eval_stop)
            except Exception as exc:
                rows.append({
                    "device_id": device_id,
                    "candidate": model,
                    "status": "evaluation_failed",
                    "reason": str(exc),
                })
                continue
        selection = _score(fr, eval_fans, split.select)
        test = _score(fr, eval_fans, split.test)
        rows.append({"device_id": device_id, "candidate": model, "status": "ok", "stage": "train"})
        rows.append({"device_id": device_id, "candidate": model, "status": "ok", "stage": "selection", **selection})
        rows.append({"device_id": device_id, "candidate": model, "status": "ok", "stage": "test", **test})
        # table_param + static_start_values travel with the result so validate can
        # re-drive the same FMU from the stored payload (no config lookup needed).
        candidates.append({"candidate": model, "params": res["params"], "fmu": str(cand["fmu"]),
                           "base": base, "selection": selection, "test": test,
                           "score": selection["score"], "table_param": cand["table_param"],
                           "static_start_values": cand.get("static_start_values")})

    if not candidates:
        return {"device_id": device_id, "status": "no_candidate", "rows": rows, "candidates": []}
    best = min(candidates, key=lambda c: c["score"])
    return {"device_id": device_id, "status": "ok", "rows": rows, "candidates": candidates,
            "selected_candidate": best["candidate"], "best": best}


def validate_ct_fmu(device_id: str, frame: pd.DataFrame, candidate_params: Mapping, fmu_root: Path,
                    workdir: Path, thresholds: Optional[Mapping] = None) -> dict:
    thresholds = thresholds or {}
    table = _compress_time(build_ct_table(frame, thresholds))   # independent points
    if table.empty:
        return {"status": "no_valid_rows"}
    workdir.mkdir(parents=True, exist_ok=True)
    full_path = workdir / f"{device_id}_ct_full.txt"
    write_dymola_table(full_path, table)
    model = candidate_params["candidate"]
    fmu_path = _resolve_fmu(candidate_params, fmu_root)
    with extracted_fmu(fmu_path) as fmu_dir:
        fr = _drive(fmu_dir, full_path, candidate_params, candidate_params["base"], candidate_params["params"],
                    float(table["time_s"].iloc[-1]))
    full = _score(fr, table["fans_on_count"].to_numpy(float))
    ts = pd.DataFrame({"measured": fr["TOut_m"].to_numpy(float)[1:] - 273.15,
                       "simulated": fr["TOut_s"].to_numpy(float)[1:] - 273.15})
    return {"status": "ok", "candidate": model, "full_period": full,
            "rows_valid": len(table), "ts": ts}
