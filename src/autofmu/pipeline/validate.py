"""Stage 4 (validate): apply selected calibrated models over the FULL operating
period (not just the held-out fold) and report full-period metrics + time series.

nominal_only / blocked devices carry their status forward -- no fabricated fit.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from autofmu.config import run_dir
from autofmu.contracts.profiles import get_profile
from autofmu.devices.calibrate import feature_columns, on_mask, predict
from autofmu.evaluation import coverage_row
from autofmu.manifest import RunManifest
from autofmu.metrics import regression_metrics
from autofmu.reporting import table_to_markdown


def _frame_for(base: Path, device: dict) -> pd.DataFrame:
    dev_dir = base / device["type"] / device["id"]
    attributed = dev_dir / "canonical_attributed.csv"
    return pd.read_csv(attributed if attributed.exists() else dev_dir / "canonical.csv")


def validate(config: dict, run_id: str) -> Path:
    base = run_dir(config, run_id)
    manifest = RunManifest(base / "manifest.json", run_id)
    manifest.bind_run(config)
    thresholds = config.get("thresholds", {})
    run_on = float(thresholds.get("run_on", 0.5))

    sel_path = base / "calibrate" / "selected_models.csv"
    par_path = base / "calibrate" / "parameters.csv"
    if not sel_path.exists():
        raise FileNotFoundError("run 'calibrate' before 'validate'")
    selected = pd.read_csv(sel_path).set_index("device_id")
    # A run in which nothing calibrated leaves an empty parameters.csv, and
    # pandas raises EmptyDataError on a file with no header rather than
    # returning an empty frame. Validation of a fleet where every device was
    # refused is a legitimate outcome -- it should report the refusals, not
    # crash on the way to them.
    try:
        params = pd.read_csv(par_path)
    except pd.errors.EmptyDataError:
        params = pd.DataFrame(columns=["device_id", "candidate", "status", "parameters"])

    rows = []
    control_rows = []
    ts_dir = base / "validate" / "timeseries"
    ts_dir.mkdir(parents=True, exist_ok=True)
    for device in config["devices"]:
        dev_id = device["id"]
        profile = get_profile(device["type"])
        frame = _frame_for(base, device)

        # Controlled variables (held at a set point) are reported separately and
        # NEVER used as a model-error metric (advisor Q2). We record their spread
        # as evidence of regulation, not as model accuracy.
        on = on_mask(frame, profile, run_on)
        for var in profile.controlled:
            if var not in frame:
                continue
            series = pd.to_numeric(frame[var], errors="coerce").to_numpy()
            series = series[on & np.isfinite(series)]
            if series.size == 0:
                continue
            control_rows.append({
                "device_id": dev_id, "equipment_type": device["type"], "variable": var,
                "role": "controlled_setpoint", "N": int(series.size),
                "mean": float(np.mean(series)), "std": float(np.std(series)),
                "note": "regulated to set point; excluded from model error",
            })

        if dev_id not in selected.index:
            rows.append({"device_id": dev_id, "equipment_type": device["type"],
                         "status": "no_candidate"})
            continue
        status = str(selected.loc[dev_id, "status"])
        selected_row = selected.loc[dev_id]
        provenance = {
            "execution_engine": str(selected_row.get("execution_engine", "none")),
            "fmu_sha256": str(selected_row.get("fmu_sha256", "")) if pd.notna(selected_row.get("fmu_sha256", "")) else "",
            "fmu_model_name": str(selected_row.get("fmu_model_name", "")) if pd.notna(selected_row.get("fmu_model_name", "")) else "",
        }
        if status != "ok":
            rows.append({"device_id": dev_id, "equipment_type": device["type"],
                         "status": status, "candidate": str(selected.loc[dev_id, "selected_candidate"]),
                         **provenance})
            continue
        candidate = str(selected.loc[dev_id, "selected_candidate"])
        # Chiller via FMU (FMU-1): drive the selected EIR/EEIR FMU over the full
        # steady period and report raw-interval P / Q / COP diagnostics.
        # predict path below when the params carry no FMU coefficients.
        if device["type"] == "chiller":
            prow = params[(params.device_id == dev_id) & (params.candidate == candidate)]
            p = json.loads(prow.iloc[0]["parameters"]) if not prow.empty else {}
            if "coeffs" in p:
                from autofmu.devices.chiller_fmu import validate_chiller_fmu
                workdir = base / "chiller" / dev_id / "fmu"
                res = validate_chiller_fmu(dev_id, frame, p, Path(p.get("fmu_root", ".")), workdir, thresholds)
                if res.get("status") != "ok":
                    rows.append({"device_id": dev_id, "equipment_type": "chiller",
                                 "status": res.get("status", "no_valid_rows"), "candidate": candidate})
                    continue
                fp = res["full_period"]
                coverage = coverage_row(len(frame), int(on.sum()), int(res["rows_valid"]), int(fp["N"]))
                rows.append({"device_id": dev_id, "equipment_type": "chiller", "status": "ok",
                             "candidate": candidate, "period": "full_valid_points", "target": "power_W",
                             "N": fp["N"], "CVRMSE_pct": fp["P_CVRMSE_pct"], "NMBE_pct": fp["P_NMBE_pct"],
                             "criterion": fp["criterion"], "Q_CVRMSE_pct": fp["Q_CVRMSE_pct"],
                             "COP_CVRMSE_pct": fp["COP_CVRMSE_pct"], "COP_NMBE_pct": fp["COP_NMBE_pct"],
                             **coverage, **provenance})
                res["ts"].to_csv(ts_dir / f"{dev_id}.csv", index=False)
                manifest.add_artifact(ts_dir / f"{dev_id}.csv", base)
                continue
        # Pump via FMU (FMU-4): drive the PumpEmpiricalPower FMU full-period.
        if device["type"] == "pump":
            prow = params[(params.device_id == dev_id) & (params.candidate == candidate)]
            p = json.loads(prow.iloc[0]["parameters"]) if not prow.empty else {}
            if p.get("kind") == "pump_fmu":
                from autofmu.devices.pump_fmu import validate_pump_fmu
                res = validate_pump_fmu(dev_id, frame, p, Path(p.get("fmu_root", ".")),
                                        base / "pump" / dev_id / "fmu", thresholds)
                if res.get("status") != "ok":
                    rows.append({"device_id": dev_id, "equipment_type": "pump",
                                 "status": res.get("status", "no_valid_rows"), "candidate": candidate})
                    continue
                fp = res["full_period"]
                coverage = coverage_row(len(frame), int(on.sum()), int(res["rows_valid"]), int(fp["N"]))
                rows.append({"device_id": dev_id, "equipment_type": "pump", "status": "ok",
                             "candidate": candidate, "period": "full_valid_points", "target": "power_W",
                             "N": fp["N"], "CVRMSE_pct": fp["P_CVRMSE_pct"], "NMBE_pct": fp["P_NMBE_pct"],
                             "criterion": fp["criterion"], **coverage, **provenance})
                res["ts"].to_csv(ts_dir / f"{dev_id}.csv", index=False)
                manifest.add_artifact(ts_dir / f"{dev_id}.csv", base)
                continue
        # cooling-tower thermal model: validate TOut and Q (the discriminators)
        if device["type"] == "cooling_tower" and candidate in ("YorkCalc", "Merkel"):
            prow = params[(params.device_id == dev_id) & (params.candidate == candidate)]
            p = json.loads(prow.iloc[0]["parameters"]) if not prow.empty else {}
            # FMU-2 path: drive the selected Merkel/YorkCalc FMU full-period.
            if p.get("kind") == "ct_fmu":
                from autofmu.devices.cooling_tower_fmu import validate_ct_fmu
                res = validate_ct_fmu(dev_id, frame, p, Path(p.get("fmu_root", ".")),
                                      base / "cooling_tower" / dev_id / "fmu", thresholds)
                if res.get("status") != "ok":
                    rows.append({"device_id": dev_id, "equipment_type": "cooling_tower",
                                 "status": res.get("status", "no_valid_rows"), "candidate": candidate})
                    continue
                fp = res["full_period"]
                coverage = coverage_row(len(frame), int(on.sum()), int(res["rows_valid"]), int(fp["N"]))
                rows.append({"device_id": dev_id, "equipment_type": "cooling_tower", "status": "ok",
                             "candidate": candidate, "period": "full_valid_points", "target": "heat_rejection_W",
                             "N": fp["N"], "CVRMSE_pct": fp["Q_CVRMSE_pct"], "NMBE_pct": fp["Q_NMBE_pct"],
                             "criterion": fp["criterion"],
                             "T_RMSE_K": fp.get("T_RMSE_K"),
                             "T_CVRMSE_pct_diagnostic": fp["T_CVRMSE_pct"],
                             "T_NMBE_pct_diagnostic": fp["T_NMBE_pct"],
                             "P_CVRMSE_pct": fp.get("P_CVRMSE_pct"),
                             "Q_CVRMSE_pct": fp["Q_CVRMSE_pct"], **coverage, **provenance})
                res["ts"].to_csv(ts_dir / f"{dev_id}.csv", index=False)
                manifest.add_artifact(ts_dir / f"{dev_id}.csv", base)
                continue
            # YorkCalc/Merkel are only ever selected by the FMU path; without an
            # FMU result there is no Python thermal fallback (removed in FMU-5).
            rows.append({"device_id": dev_id, "equipment_type": device["type"],
                         "status": "fmu_unavailable", "candidate": candidate})
            continue
        # heat exchanger: validate uncontrolled T2Out (=tcwr) and Q over operating rows
        if device["type"] == "heat_exchanger" and candidate in ("ConstantEffectiveness", "PlateEffectivenessNTU"):
            prow = params[(params.device_id == dev_id) & (params.candidate == candidate)]
            p = json.loads(prow.iloc[0]["parameters"]) if not prow.empty else {}
            # FMU-3 path: drive the selected HX FMU full-period (uncontrolled T2Out + Q).
            if p.get("kind") == "hx_fmu":
                from autofmu.devices.heat_exchanger_fmu import validate_hx_fmu
                res = validate_hx_fmu(dev_id, frame, p, Path(p.get("fmu_root", ".")),
                                      base / "heat_exchanger" / dev_id / "fmu", thresholds)
                if res.get("status") != "ok":
                    rows.append({"device_id": dev_id, "equipment_type": "heat_exchanger",
                                 "status": res.get("status", "no_valid_rows"), "candidate": candidate})
                    continue
                fp = res["full_period"]
                coverage = coverage_row(len(frame), int(on.sum()), int(res["rows_valid"]), int(fp["N"]))
                rows.append({"device_id": dev_id, "equipment_type": "heat_exchanger", "status": "ok",
                             "candidate": candidate, "period": "full_valid_points",
                             "target": "heat_transfer_W",
                             "N": fp["N"], "CVRMSE_pct": fp["Q_CVRMSE_pct"],
                             "NMBE_pct": fp["Q_NMBE_pct"],
                             "criterion": "raw_interval_custom",
                             "Q_CVRMSE_pct": fp["Q_CVRMSE_pct"],
                             # outlet temperatures are an admissibility check,
                             # reported in kelvin because a CVRMSE on Celsius is
                             # unit-dependent (M-12)
                             "T2_RMSE_K_diagnostic": fp.get("T2_RMSE_K"),
                             "T1_RMSE_K_diagnostic": fp.get("T1_RMSE_K"),
                             "T2_admissible": fp.get("T2_admissible"),
                             **coverage, **provenance})
                res["ts"].to_csv(ts_dir / f"{dev_id}.csv", index=False)
                manifest.add_artifact(ts_dir / f"{dev_id}.csv", base)
                continue
            # Const/PlateNTU are only ever selected by the FMU path; no Python
            # effectiveness-NTU fallback (removed in FMU-5).
            rows.append({"device_id": dev_id, "equipment_type": device["type"],
                         "status": "fmu_unavailable", "candidate": candidate})
            continue
        if profile.is_controlled(profile.target):  # defensive contract guard
            rows.append({"device_id": dev_id, "equipment_type": device["type"],
                         "status": "controlled_target_rejected", "candidate": candidate})
            continue
        # Empirical power models only (pump / CT fan) reach here -- thermal devices
        # are FMU-driven above. No Python thermal physics fallback (FMU-5).
        prow = params[(params.device_id == dev_id) & (params.candidate == candidate)]
        p = json.loads(prow.iloc[0]["parameters"]) if not prow.empty else {}
        empirical_target = "power_W"
        mask = on & np.isfinite(pd.to_numeric(frame[empirical_target], errors="coerce").to_numpy())
        for c in feature_columns(device["type"], candidate):
            mask &= np.isfinite(pd.to_numeric(frame.get(c), errors="coerce").to_numpy()) if c in frame else False
        valid = frame.loc[mask]
        if valid.empty:
            rows.append({"device_id": dev_id, "equipment_type": device["type"],
                         "status": "no_valid_rows", "candidate": candidate})
            continue
        measured = pd.to_numeric(valid[empirical_target], errors="coerce").to_numpy()
        sim = predict(valid, device["type"], candidate, p, thresholds)
        metric = regression_metrics(measured, sim)
        coverage = coverage_row(len(frame), int(on.sum()), len(valid), int(metric["N"]))
        row = {"device_id": dev_id, "equipment_type": device["type"], "status": "ok",
               "candidate": candidate, "period": "full_valid_points", "target": empirical_target,
               **{k: metric[k] for k in ("N", "RMSE", "MAE", "MAPE_pct", "CVRMSE_pct", "NMBE_pct", "R2", "criterion", "criterion_pass")},
               **coverage, **provenance}
        rows.append(row)
        out = valid[["timestamp"]].copy() if "timestamp" in valid else pd.DataFrame()
        out["measured"], out["simulated"] = measured, sim
        out.to_csv(ts_dir / f"{dev_id}.csv", index=False)
        manifest.add_artifact(ts_dir / f"{dev_id}.csv", base)

    report = pd.DataFrame(rows)
    stage = base / "validate"
    metrics_csv = stage / "full_period_metrics.csv"
    report.to_csv(metrics_csv, index=False)
    control = pd.DataFrame(control_rows)
    control_csv = stage / "control_variables.csv"
    control.to_csv(control_csv, index=False)
    (stage / "summary.md").write_text(
        "# Full-Valid-Points Validation\n\nModel error uses UNCONTROLLED outputs only "
        "(power / Q). Selected models applied over the full operating period "
        "(selection/test metrics are in `calibrate/`). Raw-interval threshold "
        "checks are custom and are not labelled ASHRAE Guideline 14.\n\n"
        "## Model error (uncontrolled targets)\n\n"
        + (table_to_markdown(report) if not report.empty else "_no devices_")
        + "\n\n## Controlled variables (excluded from model error)\n\n"
        + "These track a set point, not the model, so they are reported for "
        "context only -- never as accuracy.\n\n"
        + (table_to_markdown(control) if not control.empty else "_none_") + "\n",
        encoding="utf-8",
    )
    manifest.add_artifact(metrics_csv, base)
    manifest.add_artifact(control_csv, base)
    manifest.add_artifact(stage / "summary.md", base)
    manifest.record_stage("validate")
    manifest.write()
    return base
