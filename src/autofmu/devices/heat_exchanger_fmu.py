"""L3 heat-exchanger modelling via the Buildings ConstantEffectiveness /
PlateHeatExchangerEffectivenessNTU FMUs.

Replaces the Python eff-NTU surrogate with the real SiteAHXConstantEffectiveness
/ SiteAHXPlateEffectivenessNTU FMUs as the simulator for both the parameter grid
search and the validation. Table build (valid operating-window filter + energy
balance + nominals) is ported from
FMU_Modelica/scripts/prepare_site_a_hx_fmu_tables.py; the candidate grid is ported
from calibrate_site_a_hx_fmus.py. The FMU computes T1Out / T2Out / Q.

Validation targets the UNCONTROLLED condenser-side outlet T2Out (=tcwr) and Q;
the controlled chilled-water outlet T1Out (=tchws) is reported, not scored.
Side 1 = chilled water (T1In=tchwr, T1Out=tchws, m1=chw_flow); side 2 = condenser
water (T2In=tcws, T2Out=tcwr, m2=cw_flow).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import pandas as pd

from autofmu.devices import model_types as mt
from autofmu.fmu.runner import extracted_fmu, run_device_fmu
from autofmu.evaluation import metric_pairs, resolve_split
from autofmu.metrics import regression_metrics

#: Outlet-temperature error above which a heat-exchanger candidate is not
#: considered admissible, regardless of how well it reproduces heat flow.
T_OUT_ADMISSIBLE_K = 1.0

CP = 4186.0
DT_S = 300.0
MIN_FLOW_KG_S = 1.0
MIN_DT_C = 0.05
MIN_Q_W = 5000.0
MAX_BALANCE_REL = 0.50
# FMU AllData table column order (must match the exported HX FMU table reader).
TABLE_COLUMNS = ["time_s", "T1In_m_C", "T1Out_m_C", "T2In_m_C", "T2Out_m_C",
                 "m1_flow_kg_s", "m2_flow_kg_s", "Q_m_W", "Q1_m_W", "Q2_m_W",
                 "eps_m", "dT_lm_m_C", "P_aux_W", "active_proxy"]


def _num(frame, col):
    if col not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce")


def _lmtd(da, db):
    da = pd.to_numeric(da, errors="coerce")
    db = pd.to_numeric(db, errors="coerce")
    close = (da - db).abs() < 1e-9
    valid = (da > 0.0) & (db > 0.0)
    ratio = (da / db).where(valid, np.nan)
    out = (da - db) / np.log(ratio)
    return out.where(~close, (da + db) / 2.0).where(valid & np.isfinite(out), np.nan)


def _heat_balance(q1, q2):
    denom = pd.concat([q1.abs(), q2.abs()], axis=1).max(axis=1).replace(0.0, np.nan)
    return (q1 - q2).abs() / denom


def _usable(m1, m2, t1i, t1o, t2i, t2o):
    q1 = m1 * CP * (t1i - t1o)
    q2 = m2 * CP * (t2o - t2i)
    base = m1.gt(MIN_FLOW_KG_S) & m2.gt(MIN_FLOW_KG_S) & t1i.notna() & t1o.notna() & t2i.notna() & t2o.notna()
    return int((base & q1.gt(MIN_Q_W) & q2.gt(MIN_Q_W)).sum())


def _select_direction_columns(frame: pd.DataFrame):
    """Pick chilled/condenser inlet-outlet direction by positive-heat usable rows
    (ported from prepare_site_a_hx_fmu_tables.select_temperature_mapping). Site A
    HX condenser supply/return labels are swapped on running rows, so the
    direction must be chosen from the data rather than the column name."""
    temperatures = {
        "tchws_C": _num(frame, "tchws_C"), "tchwr_C": _num(frame, "tchwr_C"),
        "tcws_C": _num(frame, "tcws_C"), "tcwr_C": _num(frame, "tcwr_C"),
    }
    m1 = _num(frame, "chw_flow_m3_h") * 1000.0 / 3600.0
    m2 = _num(frame, "cw_flow_m3_h") * 1000.0 / 3600.0
    best, best_n = None, -1
    for side1 in (("tchwr_C", "tchws_C"), ("tchws_C", "tchwr_C")):
        for side2 in (("tcws_C", "tcwr_C"), ("tcwr_C", "tcws_C")):
            t1i, t1o = temperatures[side1[0]], temperatures[side1[1]]
            t2i, t2o = temperatures[side2[0]], temperatures[side2[1]]
            n = _usable(m1, m2, t1i, t1o, t2i, t2o)
            if n > best_n:
                best_n, best = n, (*side1, *side2)
    return best


def _select_directions(frame: pd.DataFrame, direction_source: Optional[pd.DataFrame] = None):
    columns = _select_direction_columns(direction_source if direction_source is not None else frame)
    t1i, t1o, t2i, t2o = (_num(frame, col) for col in columns)
    m1 = _num(frame, "chw_flow_m3_h") * 1000.0 / 3600.0
    m2 = _num(frame, "cw_flow_m3_h") * 1000.0 / 3600.0
    return t1i, t1o, t2i, t2o, m1, m2


def build_hx_table(frame: pd.DataFrame, direction_source: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """HX FMU input table from L2 canonical data (operating-window filtered).

    Side 1 = chilled water (cooled), side 2 = condenser water (warmed); the
    inlet/outlet direction of each side is selected from the data.
    """
    t1i, t1o, t2i, t2o, m1, m2 = _select_directions(frame, direction_source)
    q1 = m1 * CP * (t1i - t1o)
    q2 = m2 * CP * (t2o - t2i)
    qm = (q1 + q2) / 2.0
    cmin = pd.concat([m1 * CP, m2 * CP], axis=1).min(axis=1)
    eps = (qm.abs() / (cmin * (t1i - t2i).abs().replace(0.0, np.nan))).clip(0.0, 1.5)
    dtlm = _lmtd((t1i - t2o).abs(), (t1o - t2i).abs())
    p_aux = _num(frame, "power_W").clip(lower=0.0).fillna(0.0)
    out = pd.DataFrame({
        "T1In_m_C": t1i, "T1Out_m_C": t1o, "T2In_m_C": t2i, "T2Out_m_C": t2o,
        "m1_flow_kg_s": m1, "m2_flow_kg_s": m2, "Q_m_W": qm, "Q1_m_W": q1, "Q2_m_W": q2,
        "eps_m": eps, "dT_lm_m_C": dtlm, "P_aux_W": p_aux,
    })
    valid = (out[["m1_flow_kg_s", "m2_flow_kg_s", "T1In_m_C", "T1Out_m_C", "T2In_m_C", "T2Out_m_C", "Q1_m_W", "Q2_m_W"]]
             .replace([np.inf, -np.inf], np.nan).notna().all(axis=1))
    valid &= m1.gt(MIN_FLOW_KG_S) & m2.gt(MIN_FLOW_KG_S)
    valid &= (t1i - t1o).gt(MIN_DT_C) & (t2o - t2i).gt(MIN_DT_C)
    valid &= q1.gt(MIN_Q_W) & q2.gt(MIN_Q_W)
    valid &= _heat_balance(q1, q2).le(MAX_BALANCE_REL)
    out["active_proxy"] = 1.0
    # eps_m / dT_lm_m_C are measured diagnostics the FMU passes through, not inputs
    # to its physics; null them to 0 so the table is finite (LMTD is undefined when
    # the two approach temperatures are equal/non-positive).
    out["eps_m"] = out["eps_m"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["dT_lm_m_C"] = out["dT_lm_m_C"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out = out.loc[valid].reset_index(drop=True)
    if out.empty:
        out["time_s"] = []
        return out[TABLE_COLUMNS]
    if "timestamp" in frame.columns:
        ts = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True).iloc[np.where(valid.to_numpy())[0]]
        out["time_s"] = (ts - ts.iloc[0]).dt.total_seconds().to_numpy(dtype=float)
    else:
        out["time_s"] = np.arange(len(out), dtype=float) * DT_S
    return out[TABLE_COLUMNS]


def _compress_time(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    out["time_s"] = np.arange(len(out), dtype=float) * DT_S
    return out


def write_dymola_table(path: Path, table: pd.DataFrame, table_name: str = "HX_data") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    numeric = table[TABLE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("#1\n")
        f.write(f"double {table_name}({len(numeric)},{len(numeric.columns)})\n")
        for _, row in numeric.iterrows():
            f.write(",".join(f"{float(row[c]):.10g}" for c in TABLE_COLUMNS) + "\n")


def estimate_nominals(table: pd.DataFrame) -> dict:
    c1 = table["m1_flow_kg_s"] * CP
    c2 = table["m2_flow_kg_s"] * CP
    cmin = pd.concat([c1, c2], axis=1).min(axis=1)
    dtin = (table["T1In_m_C"] - table["T2In_m_C"]).abs().replace(0.0, np.nan)
    eps = (table["Q_m_W"].abs() / (cmin * dtin)).replace([np.inf, -np.inf], np.nan).clip(0.05, 0.98)
    return {
        "m1_flow_nominal": float(table["m1_flow_kg_s"].quantile(0.95)),
        "m2_flow_nominal": float(table["m2_flow_kg_s"].quantile(0.95)),
        "Q_flow_nominal": float(table["Q_m_W"].abs().quantile(0.95)),
        "eps_nominal": float(eps.median()) if eps.notna().any() else 0.8,
        "T_a1_nominal": float(table["T1In_m_C"].median() + 273.15),
        "T_b1_nominal": float(table["T1Out_m_C"].median() + 273.15),
        "T_a2_nominal": float(table["T2In_m_C"].median() + 273.15),
        "T_b2_nominal": float(table["T2Out_m_C"].median() + 273.15),
        "dp1_nominal": 0.0, "dp2_nominal": 0.0,
    }


def _drive(fmu, table_path: Path, start_values: Mapping, stop: float) -> pd.DataFrame:
    """Drive an HX FMU with already-assembled start values (static nominals + the
    grid candidate's tuned params). The per-type grid + static-value spec live on
    the candidate (config), expanded by model_types -- no hard-coded grid here."""
    return run_device_fmu(fmu, start_values=dict(start_values), table_overrides={"table_path": table_path},
                          output=["T1Out_m", "T1Out_s", "T2Out_m", "T2Out_s", "Q_m", "Q_s"],
                          stop_time=stop, output_interval=DT_S, require_finite=True)


def _score(frame: pd.DataFrame, rows: Optional[np.ndarray] = None) -> dict:
    """Score the uncontrolled T2Out + Q (selection); report T1Out separately."""
    n = len(frame)
    sel = np.arange(1, n) if rows is None else rows[(rows >= 1) & (rows < n)]
    def cv(mc, sc):
        m = frame[mc].to_numpy(float)[sel]
        s = frame[sc].to_numpy(float)[sel]
        metric_pairs(m, s)
        ok = np.isfinite(m) & np.isfinite(s)
        return regression_metrics(pd.Series(m[ok]), pd.Series(s[ok]))
    def cv_abs(mc, sc):
        m = frame[mc].to_numpy(float)[sel]
        s = frame[sc].to_numpy(float)[sel]
        ok = np.isfinite(m) & np.isfinite(s)
        return regression_metrics(pd.Series(m[ok]), pd.Series(s[ok]),
                                  quantity="temperature")

    mT2 = cv("T2Out_m", "T2Out_s")
    mQ = cv("Q_m", "Q_s")
    mT1 = cv("T1Out_m", "T1Out_s")
    mT2_abs = cv_abs("T2Out_m", "T2Out_s")
    mT1_abs = cv_abs("T1Out_m", "T1Out_s")
    # The manuscript's stated rule: outlet temperatures are an admissibility
    # check, and reconstructed heat decides between candidates. Scoring on the
    # sum of a temperature CVRMSE and a heat CVRMSE, as before, let a
    # unit-dependent number cast half the vote (fix M-12).
    admissible = float(mT2_abs["RMSE"]) <= T_OUT_ADMISSIBLE_K
    score = float(mQ["CVRMSE_pct"]) + (0.0 if admissible else 1e3)
    return {"N": int(mT2["N"]),
            "T2_RMSE_K": mT2_abs["RMSE"], "T1_RMSE_K": mT1_abs["RMSE"],
            "T2_admissible": admissible,
            "T2_CVRMSE_pct": mT2["CVRMSE_pct"], "T2_NMBE_pct": mT2["NMBE_pct"],
            "Q_CVRMSE_pct": mQ["CVRMSE_pct"], "Q_NMBE_pct": mQ["NMBE_pct"],
            "T1_CVRMSE_pct": mT1["CVRMSE_pct"],
            "Q_skill": mQ["skill_vs_mean"], "T2_skill": mT2_abs["skill_vs_mean"],
            "score": score}


def _resolve_fmu(cand: Mapping, fmu_root: Path) -> Path:
    raw = Path(str(cand["fmu"]))
    return raw if raw.is_absolute() else (fmu_root / raw)


def _subset(table: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if len(table) <= max_rows:
        sub = table.copy()
    else:
        sub = table.iloc[:: max(1, len(table) // max_rows)].reset_index(drop=True).copy()
    sub["time_s"] = np.arange(len(sub), dtype=float) * DT_S
    return sub


def fit_hx_fmu(device_id: str, frame: pd.DataFrame, hx_cfg: Mapping, fmu_root: Path,
               workdir: Path, thresholds: Optional[Mapping] = None) -> dict:
    thresholds = thresholds or {}
    min_rows = int(thresholds.get("min_calibration_rows", thresholds.get("min_full_physical_rows", 200)))
    subset_rows = int(thresholds.get("hx_subset_rows", 1500))
    workdir.mkdir(parents=True, exist_ok=True)

    train_fraction = float(thresholds.get("train_fraction", 0.6))
    raw_train_end = max(1, int(len(frame) * train_fraction))
    table = build_hx_table(frame, direction_source=frame.iloc[:raw_train_end])
    if len(table) < min_rows:
        return {"device_id": device_id, "status": "data_limited",
                "reason": f"{len(table)} valid HX operating rows < {min_rows}", "rows": [], "candidates": []}
    split = resolve_split(table, thresholds)
    if not split.usable:
        return {"device_id": device_id, "status": split.status,
                "reason": (f"record cannot support an independent test segment: "
                           f"train={len(split.train)} select={len(split.select)} "
                           f"test={len(split.test)}"),
                "rows": [], "candidates": []}
    subset = _compress_time(_subset(table.iloc[split.train].reset_index(drop=True), subset_rows))
    sub_path = workdir / f"{device_id}_hx_subset.txt"
    write_dymola_table(sub_path, subset)
    stop = float(subset["time_s"].iloc[-1])
    nominals = estimate_nominals(subset)
    evaluation = _compress_time(table)
    eval_path = workdir / f"{device_id}_hx_evaluation.txt"
    write_dymola_table(eval_path, evaluation)
    eval_stop = float(evaluation["time_s"].iloc[-1])

    rows, candidates = [], []
    for spec in mt.select_candidates(hx_cfg["candidates"], hx_cfg.get("enabled_candidates")):
        model = spec["name"]
        fmu_path = _resolve_fmu(spec, fmu_root)
        if not Path(fmu_path).exists():
            rows.append({"device_id": device_id, "candidate": model, "status": "fmu_missing"})
            continue
        # Grid + fixed nominals are declared on the candidate (config); expand them
        # generically and pick the best by score -- no per-model Python grid.
        best_sv, best_score = None, float("inf")
        with extracted_fmu(fmu_path) as fmu_dir:
            for grid_params in mt.expand_grid(spec.get("grid"), nominals):
                sv = mt.assemble_start_values(spec.get("static_start_values"), nominals, grid_params)
                try:
                    fr = _drive(fmu_dir, sub_path, sv, stop)
                except Exception:
                    continue
                sc = _score(fr)["score"]
                if sc < best_score:
                    best_score, best_sv = sc, sv
        if best_sv is None:
            rows.append({"device_id": device_id, "candidate": model, "status": "no_candidate_converged"})
            continue
        with extracted_fmu(fmu_path) as fmu_dir:
            evaluation_frame = _drive(fmu_dir, eval_path, best_sv, eval_stop)
        selection = _score(evaluation_frame, split.select)
        test = _score(evaluation_frame, split.test)
        rows.append({"device_id": device_id, "candidate": model, "status": "ok", "stage": "train"})
        rows.append({"device_id": device_id, "candidate": model, "status": "ok", "stage": "selection", **selection})
        rows.append({"device_id": device_id, "candidate": model, "status": "ok", "stage": "test", **test})
        candidates.append({"candidate": model, "params": best_sv, "fmu": spec["fmu"],
                           "selection": selection, "test": test, "score": selection["score"]})

    if not candidates:
        return {"device_id": device_id, "status": "no_candidate", "rows": rows, "candidates": []}
    best = min(candidates, key=lambda c: c["score"])
    return {"device_id": device_id, "status": "ok", "rows": rows, "candidates": candidates,
            "selected_candidate": best["candidate"], "best": best}


def validate_hx_fmu(device_id: str, frame: pd.DataFrame, candidate_params: Mapping, fmu_root: Path,
                    workdir: Path, thresholds: Optional[Mapping] = None) -> dict:
    table = _compress_time(build_hx_table(frame))
    if table.empty:
        return {"status": "no_valid_rows"}
    workdir.mkdir(parents=True, exist_ok=True)
    full_path = workdir / f"{device_id}_hx_full.txt"
    write_dymola_table(full_path, table)
    fmu_path = _resolve_fmu(candidate_params, fmu_root)
    with extracted_fmu(fmu_path) as fmu_dir:
        fr = _drive(fmu_dir, full_path, candidate_params["params"], float(table["time_s"].iloc[-1]))
    full = _score(fr)
    ts = pd.DataFrame({"measured": fr["T2Out_m"].to_numpy(float)[1:],
                       "simulated": fr["T2Out_s"].to_numpy(float)[1:]})
    return {"status": "ok", "candidate": candidate_params["candidate"], "full_period": full,
            "rows_valid": len(table), "ts": ts}
