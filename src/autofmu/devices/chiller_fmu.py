"""L3 chiller modelling via the Buildings FMUs (ElectricEIR / ElectricReformulatedEIR).

This replaces the Python log-linear ``predict_chiller`` surrogate with the real
FMU as the simulator for BOTH steps:

1. fit the curve coefficients (capFunT / EIRFunT / EIRFunPLR) from the chiller's
   own BMS data, and
2. drive the FMU with those coefficients and score P / Q / COP + ASHRAE GL14.

Fitting strategy (faithful to Hydeman & Gillespie 2002 + CHILLER_METHODOLOGY):

- **screen** the reference-curve library for the best-matching row (technique 2,
  also the old project's baseline) -> warm start, and
- **refine** the coefficients FMU-in-the-loop with least squares (technique 1),
  accepting the refinement only if it improves held-out power. The result is
  therefore never worse than the screening baseline.

The FMU reads measured operating points from an ``AllData2`` Dymola table built
from L2 canonical data; it outputs measured (``P_m``/``Q_m``) and simulated
(``P_s``/``Q_s``) signals. Q is essentially fixed by the imposed water-side
boundary conditions, so fitting targets power.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from autofmu.devices import model_types as mt
from autofmu.devices.chiller import _cooling_load, _num, _steady_mask
from autofmu.evaluation import metric_pairs, resolve_split
from autofmu.fmu.runner import run_device_fmu
from autofmu.metrics import regression_metrics

TABLE_COLUMNS = ["Time", "CHWS", "CHWR", "CDWS", "CDWR", "CHW", "CDW",
                 "P/kw", "VSD", "deltaT_chw", "deltaT_cdw", "Q_evap_kW"]
TABLE_DT_S = 1800.0          # table time step (proven EIR/AllData2 convention)
CP_WATER_KJ = 4.186          # kJ/kg/K, matches the existing prepare script


# --------------------------------------------------------------------------- #
# 1. Build the AllData2 measured table from L2 canonical data
# --------------------------------------------------------------------------- #
def build_alldata2(frame: pd.DataFrame, run_on: float = 0.5) -> pd.DataFrame:
    """Steady-filter canonical chiller data into the FMU's AllData2 table shape.

    Flows are converted m3/h -> L/s (== kg/s for water), which is what the FMU's
    CombiTimeTable feeds to the evaporator/condenser mass-flow sources.
    """
    P = _num(frame, "power_W")
    Q = _cooling_load(frame)
    mask = _steady_mask(frame, run_on, P, Q)
    valid = frame.loc[mask].copy()
    if "timestamp" in valid:
        valid = valid.sort_values("timestamp")
    valid = valid.reset_index(drop=True)
    if valid.empty:
        return pd.DataFrame(columns=TABLE_COLUMNS)

    chws, chwr = _num(valid, "tchws_C"), _num(valid, "tchwr_C")
    cdws, cdwr = _num(valid, "tcws_C"), _num(valid, "tcwr_C")
    chw_lps = _num(valid, "chw_flow_m3_h") / 3.6
    cdw_lps = _num(valid, "cw_flow_m3_h") / 3.6
    dchw = chwr - chws
    dcdw = cdwr - cdws
    q_kw = chw_lps * CP_WATER_KJ * dchw
    table = pd.DataFrame({
        "Time": np.arange(len(valid), dtype=float) * TABLE_DT_S,
        "CHWS": chws, "CHWR": chwr, "CDWS": cdws, "CDWR": cdwr,
        "CHW": chw_lps, "CDW": cdw_lps,
        "P/kw": _num(valid, "power_W") / 1000.0, "VSD": np.ones(len(valid)),
        "deltaT_chw": dchw, "deltaT_cdw": dcdw, "Q_evap_kW": q_kw,
    })
    finite = np.all(np.isfinite(table[TABLE_COLUMNS].to_numpy(dtype=float)), axis=1)
    physically_valid = (
        (table["P/kw"] > 0)
        & (table["Q_evap_kW"] > 0)
        & (table["CHW"] > 0)
        & (table["CDW"] > 0)
        & (table["deltaT_chw"] > 0)
        & (table["deltaT_cdw"] > 0)
        # COP > 1: a vapour-compression machine cannot reconstruct more cooling
        # than the total energy entering the evaporator. HKUST CH-08 does so on
        # 116 of its 350 usable rows, because its chilled-water flow is frozen
        # at 70.1 m3/h while power swings 210-619 kW, and it scored 46.9% test
        # CVRMSE against 2.6-10.5% for its eight siblings. No other chiller at
        # any of the four sites has a single such row.
        & (table["Q_evap_kW"] > table["P/kw"])
    )
    table = table.loc[finite & physically_valid].reset_index(drop=True)
    table["Time"] = np.arange(len(table), dtype=float) * TABLE_DT_S
    return table


def write_alldata2(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("#1\n")
        fh.write(f"double AllData2({len(table)},12)\n")
        fh.write("#" + "\t".join(TABLE_COLUMNS) + "\n")
        for _, row in table.iterrows():
            fh.write("\t".join(f"{float(row[c]):.10g}" for c in TABLE_COLUMNS) + "\n")


def estimate_nominals(table: pd.DataFrame, condenser: str) -> dict:
    """Measured nominal/design values from the prepared table (max-load row).

    ``condenser`` selects the condenser nominal temperature basis: ``entering``
    (CDWS, ElectricEIR) or ``leaving`` (CDWR, ElectricReformulatedEIR).
    """
    i = int(table["Q_evap_kW"].idxmax())
    row = table.loc[i]
    q_nom_kw = float(row["Q_evap_kW"])
    p_nom_kw = float(row["P/kw"])
    plr = table["Q_evap_kW"] / q_nom_kw
    plr = plr[(plr > 0) & np.isfinite(plr)]
    plr_min = max(0.05, float(plr.min())) if not plr.empty else 0.05
    tcon_col = "CDWS" if condenser == "entering" else "CDWR"
    return {
        "QEva_flow_nominal": -q_nom_kw * 1000.0,
        "COP_nominal": q_nom_kw / p_nom_kw if p_nom_kw > 0 else 5.0,
        "mEva_flow_nominal": float(row["CHW"]),
        "mCon_flow_nominal": float(row["CDW"]),
        "TEvaLvg_nominal": float(row["CHWS"]) + 273.15,
        "TEvaLvgMin": float(table["CHWS"].min()) + 273.15,
        "TEvaLvgMax": float(table["CHWS"].max()) + 273.15,
        "TCon_nominal": float(row[tcon_col]) + 273.15,
        "TConMin": float(table[tcon_col].min()) + 273.15,
        "TConMax": float(table[tcon_col].max()) + 273.15,
        "PLRMax": 1.15,
        "PLRMin": plr_min,
        "PLRMinUnl": plr_min,
        "etaMotor": 1.0,
    }


def nominal_start_values(nominals: Mapping[str, float], candidate: Mapping) -> dict:
    mapping = candidate.get("nominal_params", {}) or {}
    out = {}
    for canon, fmu_name in mapping.items():
        if canon in nominals and np.isfinite(nominals[canon]):
            out[fmu_name] = float(nominals[canon])
    return out


def coeff_names(candidate: Mapping) -> List[str]:
    """Flattened tunable-coefficient names, in declared group order.

    EIR/EEIR declare capFT/EIRFT/EIRFPLR (15/22 names); a new type such as Carnot
    declares its own group(s) (e.g. etaCarnot + a1..a6). Iterating every group
    keeps the existing EIR/EEIR order while supporting added types without code.
    """
    groups = candidate.get("tunable_parameters", {}) or {}
    names: List[str] = []
    for vals in groups.values():
        names.extend(vals)
    return names


# --------------------------------------------------------------------------- #
# 2. FMU driving + scoring
# --------------------------------------------------------------------------- #
def _interleave_split(n: int, fold: int) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.arange(n)
    return idx[idx % fold != 0], idx[idx % fold == 0]


def drive_fmu(fmu_path: Path, table_path: Path, table_param: str, start_values: Mapping,
              output_interval: float, stop_time: float) -> pd.DataFrame:
    return run_device_fmu(
        fmu_path,
        start_values=dict(start_values),
        table_overrides={table_param: table_path},
        output=["P_m", "P_s", "Q_m", "Q_s"],
        stop_time=stop_time,
        output_interval=output_interval,
        require_finite=True,
    )


def _cvrmse(meas: np.ndarray, sim: np.ndarray) -> float:
    meas = np.asarray(meas, float)
    sim = np.asarray(sim, float)
    ok = np.isfinite(meas) & np.isfinite(sim)
    if ok.sum() < 2 or abs(np.mean(meas[ok])) < 1e-9:
        return float("inf")
    return float(np.sqrt(np.mean((sim[ok] - meas[ok]) ** 2)) / np.mean(meas[ok]) * 100.0)


def score_frame(frame: pd.DataFrame, rows: Optional[np.ndarray] = None) -> dict:
    """P / Q / COP regression metrics (+ GL14) over the given output rows."""
    f = frame.iloc[rows] if rows is not None else frame
    Pm, Ps = f["P_m"].to_numpy(float), f["P_s"].to_numpy(float)
    Qm, Qs = f["Q_m"].to_numpy(float), f["Q_s"].to_numpy(float)
    metric_pairs(Pm, Ps)
    metric_pairs(Qm, Qs)
    ok = np.isfinite(Pm) & np.isfinite(Ps) & (Pm > 0) & (Ps > 0)
    mP = regression_metrics(pd.Series(Pm[ok]), pd.Series(Ps[ok]))
    mQ = regression_metrics(pd.Series(Qm[ok]), pd.Series(Qs[ok]))
    copm, cops = Qm[ok] / Pm[ok], Qs[ok] / Ps[ok]
    mC = regression_metrics(pd.Series(copm), pd.Series(cops))
    return {
        "N": int(ok.sum()),
        "P_CVRMSE_pct": mP["CVRMSE_pct"], "P_NMBE_pct": mP["NMBE_pct"],
        "Q_CVRMSE_pct": mQ["CVRMSE_pct"],
        "COP_CVRMSE_pct": mC["CVRMSE_pct"], "COP_NMBE_pct": mC["NMBE_pct"],
        # Skill against predicting the measured mean on these same rows. A small
        # CVRMSE on a signal that barely moves is not evidence of a good model.
        "P_skill": mP["skill_vs_mean"], "Q_skill": mQ["skill_vs_mean"],
        "criterion": "raw_interval_custom",
    }


# --------------------------------------------------------------------------- #
# 3. Fit: screen (warm start / baseline) -> refine (FMU-in-the-loop)
# --------------------------------------------------------------------------- #
def screen_library(fmu_path: Path, table_path: Path, table_param: str, nom_sv: Mapping,
                   library: pd.DataFrame, names: Sequence[str], output_interval: float,
                   stop_time: float, max_rows: Optional[int] = None,
                   search_rows: Optional[np.ndarray] = None) -> Tuple[dict, dict]:
    """Pick the library coefficient row that best matches measured P (+ Q)."""
    if max_rows is not None:
        library = library.head(int(max_rows))
    best_coeffs: dict = {}
    best_metrics: dict = {"P_CVRMSE_pct": float("inf")}
    best_score = float("inf")
    for _, lib_row in library.iterrows():
        coeffs = {n: float(lib_row[n]) for n in names if n in lib_row and np.isfinite(lib_row[n])}
        if len(coeffs) != len(names):
            continue
        try:
            frame = drive_fmu(fmu_path, table_path, table_param, {**nom_sv, **coeffs}, output_interval, stop_time)
        except Exception:
            continue
        evaluated = frame.iloc[search_rows] if search_rows is not None else frame
        sP = _cvrmse(evaluated["P_m"], evaluated["P_s"])
        sQ = _cvrmse(evaluated["Q_m"], evaluated["Q_s"])
        score = sP + sQ
        if np.isfinite(score) and score < best_score:
            best_score = score
            best_coeffs = coeffs
            best_metrics = {"P_CVRMSE_pct": sP, "Q_CVRMSE_pct": sQ}
    return best_coeffs, best_metrics


def refine_coeffs(fmu_path: Path, table_path: Path, table_param: str, nom_sv: Mapping,
                  warm: Mapping[str, float], names: Sequence[str], output_interval: float,
                  stop_time: float, train: np.ndarray, select: np.ndarray, max_nfev: int) -> Tuple[dict, str]:
    """Refine coefficients FMU-in-the-loop; fall back to warm start if it does
    not improve selection power (guarantees non-regression vs screening)."""
    try:
        from scipy.optimize import least_squares
    except ImportError:
        return dict(warm), "no_scipy"

    x0 = np.array([warm[n] for n in names], dtype=float)
    scale = np.where(np.abs(x0) > 1e-6, np.abs(x0), 1.0)
    lam = 0.02  # light Tikhonov pull toward the (physical) warm start

    def residual(x: np.ndarray) -> np.ndarray:
        coeffs = {n: float(v) for n, v in zip(names, x)}
        try:
            frame = drive_fmu(fmu_path, table_path, table_param, {**nom_sv, **coeffs}, output_interval, stop_time)
        except Exception:
            return np.full(len(train) + len(names), 1e3)
        Pm = frame["P_m"].to_numpy(float)
        Ps = frame["P_s"].to_numpy(float)
        n = min(len(Pm), int(train.max()) + 1) if len(train) else 0
        tr = train[train < len(Pm)]
        ref = np.full(len(tr) + len(names), 0.0)
        denom = np.where(np.abs(Pm[tr]) > 1e-6, Pm[tr], 1.0)
        res = (Ps[tr] - Pm[tr]) / denom
        ref[: len(tr)] = np.nan_to_num(res, nan=1e3, posinf=1e3, neginf=1e3)
        ref[len(tr):] = lam * (x - x0) / scale
        return ref

    def selection_P(coeffs: Mapping) -> float:
        try:
            frame = drive_fmu(fmu_path, table_path, table_param, {**nom_sv, **coeffs}, output_interval, stop_time)
        except Exception:
            return float("inf")
        te = select[select < len(frame)]
        return _cvrmse(frame["P_m"].to_numpy(float)[te], frame["P_s"].to_numpy(float)[te])

    warm_selection = selection_P(warm)
    try:
        sol = least_squares(residual, x0, method="trf", x_scale=scale, max_nfev=max_nfev, diff_step=1e-3)
        refined = {n: float(v) for n, v in zip(names, sol.x)}
    except Exception:
        return dict(warm), "refine_failed"
    refined_selection = selection_P(refined)
    if np.isfinite(refined_selection) and refined_selection < warm_selection:
        return refined, "refined"
    return dict(warm), "fallback_to_screen"


# --------------------------------------------------------------------------- #
# 4. Orchestration: per-chiller, both candidates, select
# --------------------------------------------------------------------------- #
def _subset(table: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if len(table) <= max_rows:
        sub = table.copy()
    else:
        step = max(1, len(table) // max_rows)
        sub = table.iloc[::step].reset_index(drop=True).copy()
    # Keep the real elapsed time before the FMU axis overwrites it. The FMU is
    # driven on a gap-free sequential axis, but the train/test split has to be
    # made in real time -- buffering against the compressed axis turns 261 days
    # into 7 and a 72 h buffer then swallows entire blocks (Site A pumps came
    # back with zero test rows).
    sub["source_time_s"] = sub["Time"].to_numpy(dtype=float)
    sub["Time"] = np.arange(len(sub), dtype=float) * TABLE_DT_S
    return sub


def _resolve_fmu(cand: Mapping, fmu_root: Path) -> Path:
    raw = Path(str(cand["fmu"]))
    return raw if raw.is_absolute() else (fmu_root / raw)


def fit_chiller_fmu(device_id: str, frame: pd.DataFrame, chiller_cfg: Mapping,
                    fmu_root: Path, workdir: Path, thresholds: Optional[Mapping] = None) -> dict:
    """L3 calibrate (fast): screen + FMU-in-the-loop refine on a downsampled
    steady subset, select on a disjoint chronological partition, and report an
    untouched test partition. Full-valid scoring is deferred to validation.

    Returns the selected candidate, every candidate's coefficients+nominals, and
    per-candidate selection and test metrics.
    """
    thresholds = thresholds or {}
    run_on = float(thresholds.get("run_on", 0.5))
    min_rows = int(thresholds.get("min_calibration_rows", thresholds.get("min_full_physical_rows", 200)))
    cal = chiller_cfg.get("calibration", {})
    subset_rows = int(thresholds.get("subset_rows", cal.get("subset_rows", 600)))
    max_nfev = int(thresholds.get("max_refine_nfev", cal.get("max_refine_nfev", 80)))
    screen_max = thresholds.get("screening_max_rows", cal.get("screening_max_rows"))
    screen_max = int(screen_max) if screen_max is not None else None

    table = build_alldata2(frame, run_on)
    workdir.mkdir(parents=True, exist_ok=True)
    if len(table) < min_rows:
        return {"device_id": device_id, "status": "insufficient_rows",
                "reason": f"{len(table)} steady table rows < {min_rows}", "rows": [], "candidates": []}

    subset = _subset(table, subset_rows)
    sub_path = workdir / f"{device_id}_AllData2_subset.txt"
    write_alldata2(subset, sub_path)
    sub_stop = float(subset["Time"].iloc[-1])
    split = resolve_split(subset, thresholds, time_column="source_time_s",
                          native_rows=len(table))
    if not split.usable:
        return {"device_id": device_id, "status": split.status,
                "reason": (f"record cannot support an independent test segment: "
                           f"train={len(split.train)} select={len(split.select)} "
                           f"test={len(split.test)}"),
                "rows": [], "candidates": []}

    rows: List[dict] = []
    candidates_out: List[dict] = []
    # Only the enabled model types compete (default = all contract candidates).
    # The pipeline neutralises this after applying any project override, so the
    # contract default is honoured for direct callers without double-filtering.
    for cand in mt.select_candidates(chiller_cfg["candidates"], chiller_cfg.get("enabled_candidates")):
        name = cand["name"]
        try:
            fmu_path = _resolve_fmu(cand, fmu_root)
            tparam = cand["table_param"]
            oi = float(cand.get("output_interval", TABLE_DT_S))
            names = coeff_names(cand)
            train_table = subset.iloc[split.train].reset_index(drop=True)
            nom_sv = nominal_start_values(estimate_nominals(train_table, cand.get("condenser", "entering")), cand)
            strategy = cand.get("fit_strategy", "screen_refine")

            # warm start for the FMU-in-the-loop refinement comes either from screening
            # the reference-curve library (screen_refine: EIR/EEIR) or from declared
            # coefficients (refine_only: a type without a curve library, e.g. Carnot).
            if strategy == "refine_only":
                if not Path(fmu_path).exists():
                    rows.append({"device_id": device_id, "candidate": name, "status": "fmu_missing"})
                    continue
                warm = {n: float((cand.get("warm_start") or {}).get(n, 0.0)) for n in names}
                refined, refine_status = refine_coeffs(fmu_path, sub_path, tparam, nom_sv, warm, names,
                                                       oi, sub_stop, split.train, split.select, max_nfev)
                warm_sub = drive_fmu(fmu_path, sub_path, tparam, {**nom_sv, **warm}, oi, sub_stop)
                ref_sub = drive_fmu(fmu_path, sub_path, tparam, {**nom_sv, **refined}, oi, sub_stop)
            else:
                lib_path = (fmu_root / cand["screening_library"]) if cand.get("screening_library") else None
                if not lib_path or not Path(lib_path).exists() or not Path(fmu_path).exists():
                    rows.append({"device_id": device_id, "candidate": name, "status": "fmu_or_library_missing"})
                    continue
                library = pd.read_excel(lib_path, sheet_name=cand.get("screening_sheet", "Chiller Data"))
                # Some Dymola chiller FMUs do not support multiple instances
                # from one extracted directory. Use the original .fmu per run so
                # FMPy extracts a fresh instance for each screening/refine call.
                warm, _ = screen_library(
                    fmu_path, sub_path, tparam, nom_sv, library, names, oi, sub_stop,
                    screen_max, search_rows=split.train,
                )
                if not warm:
                    rows.append({"device_id": device_id, "candidate": name, "status": "screen_failed"})
                    continue
                refined, refine_status = refine_coeffs(fmu_path, sub_path, tparam, nom_sv, warm, names,
                                                       oi, sub_stop, split.train, split.select, max_nfev)
                warm_sub = drive_fmu(fmu_path, sub_path, tparam, {**nom_sv, **warm}, oi, sub_stop)
                ref_sub = drive_fmu(fmu_path, sub_path, tparam, {**nom_sv, **refined}, oi, sub_stop)
            base_selection = score_frame(warm_sub, split.select[split.select < len(warm_sub)])
            selection = score_frame(ref_sub, split.select[split.select < len(ref_sub)])
            test = score_frame(ref_sub, split.test[split.test < len(ref_sub)])
            rows.append({"device_id": device_id, "candidate": name, "status": "ok", "stage": "train",
                         "refine_status": refine_status})
            rows.append({"device_id": device_id, "candidate": name, "status": "ok", "stage": "selection", **selection})
            rows.append({"device_id": device_id, "candidate": name, "status": "ok", "stage": "test", **test})
            candidates_out.append({
                "candidate": name, "coeffs": refined, "nominals": nom_sv,
                "fmu": str(cand["fmu"]), "table_param": tparam, "output_interval": oi,
                "condenser": cand.get("condenser", "entering"), "refine_status": refine_status,
                "selection": selection, "test": test, "baseline_selection": base_selection,
                "score": selection["P_CVRMSE_pct"],
            })
        except Exception as exc:
            # A misbehaving candidate type (e.g. an FMU that fails to initialise for
            # some parameters) is gated, not fatal -- the other types still fit.
            rows.append({"device_id": device_id, "candidate": name,
                         "status": "fit_error", "reason": str(exc)[:200]})

    if not candidates_out:
        return {"device_id": device_id, "status": "no_candidate", "rows": rows, "candidates": []}
    best = min(candidates_out, key=lambda c: c["score"])
    return {"device_id": device_id, "status": "ok", "rows": rows, "candidates": candidates_out,
            "selected_candidate": best["candidate"], "best": best}


def validate_chiller_fmu(device_id: str, frame: pd.DataFrame, candidate_params: Mapping,
                         fmu_root: Path, workdir: Path, thresholds: Optional[Mapping] = None) -> dict:
    """Full-period validation of one selected chiller candidate via its FMU.

    Rebuilds the AllData2 table from canonical data, drives the FMU once over the
    full steady period with the calibrated coefficients+nominals, and reports
    raw-interval P / Q / COP over all valid points plus a timeseries frame.
    """
    thresholds = thresholds or {}
    run_on = float(thresholds.get("run_on", 0.5))
    table = build_alldata2(frame, run_on)
    if table.empty:
        return {"status": "no_valid_rows"}
    workdir.mkdir(parents=True, exist_ok=True)
    full_path = workdir / f"{device_id}_AllData2.txt"
    write_alldata2(table, full_path)
    fmu_path = _resolve_fmu(candidate_params, fmu_root)
    sv = {**dict(candidate_params.get("nominals", {})), **dict(candidate_params.get("coeffs", {}))}
    oi = float(candidate_params.get("output_interval", TABLE_DT_S))
    frame_out = drive_fmu(fmu_path, full_path, candidate_params["table_param"], sv, oi, float(table["Time"].iloc[-1]))
    full = score_frame(frame_out)
    ts = frame_out[["P_m", "P_s", "Q_m", "Q_s"]].copy()
    return {"status": "ok", "candidate": candidate_params.get("candidate"),
            "full_period": full, "rows_valid": len(table), "ts": ts}
