"""Stage 3 (L3): calibrate device models, gated by the L2 modelability report.

- full_physical -> fit empirical power candidates, validate, select by MAPE.
- nominal_only  -> emit nominal/design parameters; record that target-power
                   calibration is blocked (no fabricated fit).
- blocked       -> skipped, recorded.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import yaml

from autofmu.config import fmu_root_from_env, resolve_path, run_dir
from autofmu.contracts.profiles import get_profile
from autofmu.devices import model_types as mt
from autofmu.devices.calibrate import calibrate_power_model
from autofmu.devices.chiller_fmu import fit_chiller_fmu
from autofmu.manifest import RunManifest
from autofmu.reporting import table_to_markdown


def _load_device_fmu_cfg(config: dict, device_type: str):
    """Load configs/fmu/<type>.yaml if the project declares fmu_config_dir and
    the contract + at least one candidate FMU are present. Returns (cfg, fmu_root)
    or (None, None) so L3 falls back to the Python path in non-FMU environments."""
    fmu_dir = config.get("fmu_config_dir")
    if not fmu_dir:
        return None, None
    path = resolve_path(config, str(Path(fmu_dir) / f"{device_type}.yaml"))
    if not path.exists():
        return None, None
    try:
        import fmpy  # noqa: F401
    except ImportError:
        return None, None
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    fmu_root = Path(fmu_root_from_env(cfg.get("fmu_root")))   # $AUTOFMU_FMU_ROOT override
    # Restrict which model *types* compete: project `fmu_candidates: {<type>: [...]}`
    # overrides the contract's `enabled_candidates`; default = all (model_types).
    # Neutralise enabled_candidates after applying it here so a device engine that
    # re-checks it (e.g. chiller, for direct callers) does not re-filter and undo a
    # project override.
    enabled = mt.resolve_enabled(cfg, config, device_type)
    cfg["candidates"] = mt.select_candidates(cfg.get("candidates", []), enabled)
    cfg["enabled_candidates"] = None
    for cand in cfg.get("candidates", []):
        raw = Path(str(cand["fmu"]))
        fmu_path = raw if raw.is_absolute() else (fmu_root / raw)
        if fmu_path.exists():
            return cfg, fmu_root
    return None, None


# AlphaDataCenterCooling reported power MAPE (same Tencent plant), for context.
# SI Table S4 / main-text Table 2. method: manufacturer characteristic curves +
# single-device windows (+ MLP head model for pumps), not a pure speed-only fit.
PAPER_MAPE = {
    "pump": "5.27 (CHWP) / 0.8 (CWP) [manufacturer curve + single-pump windows]",
    "cooling_tower": "2.29 (fan) [affinity Eq.15 + single-tower windows]",
    "chiller": "4.31-8.25 [ElectricReformulatedEIR FMU; needs measured power]",
    "heat_exchanger": "Tchws 2.68 / Tcwr 1.5 [eff-NTU, FC mode; needs flow]",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fmu_provenance(best: dict, fmu_root: Path) -> dict:
    raw = Path(str(best["fmu"]))
    path = raw if raw.is_absolute() else (fmu_root / raw)
    return {
        "execution_engine": "fmpy_fmu",
        "fmu_sha256": _sha256_file(path),
        "fmu_model_name": str(best["candidate"]),
    }


def _frame_for(base: Path, device: dict) -> pd.DataFrame:
    dev_dir = base / device["type"] / device["id"]
    attributed = dev_dir / "canonical_attributed.csv"
    path = attributed if attributed.exists() else dev_dir / "canonical.csv"
    return pd.read_csv(path)


def _nominal_params(frame: pd.DataFrame, profile) -> dict:
    out: dict = {}
    for col in set(profile.nominal_fields) | {"cooling_load_W", "power_W_recon"}:
        if col in frame:
            series = pd.to_numeric(frame[col], errors="coerce")
            if np.isfinite(series.to_numpy(dtype=float)).any():
                out[col + "_p95"] = float(series.quantile(0.95))
                out[col + "_median"] = float(series.median())
    return out


def calibrate(config: dict, run_id: str) -> Path:
    base = run_dir(config, run_id)
    manifest = RunManifest(base / "manifest.json", run_id)
    manifest.bind_run(config)
    thresholds = config.get("thresholds", {})

    report_path = base / "attribute" / "modelability_report.csv"
    if not report_path.exists():
        raise FileNotFoundError("run 'attribute' before 'calibrate': %s" % report_path)
    gate = pd.read_csv(report_path).set_index("device_id")
    chiller_fmu_cfg, chiller_fmu_root = _load_device_fmu_cfg(config, "chiller")
    ct_fmu_cfg, ct_fmu_root = _load_device_fmu_cfg(config, "cooling_tower")
    hx_fmu_cfg, hx_fmu_root = _load_device_fmu_cfg(config, "heat_exchanger")
    pump_fmu_cfg, pump_fmu_root = _load_device_fmu_cfg(config, "pump")

    metric_rows = []
    selected_rows = []
    param_rows = []
    ts_dir = base / "calibrate" / "best_timeseries"
    ts_dir.mkdir(parents=True, exist_ok=True)

    for device in config["devices"]:
        dev_id = device["id"]
        if dev_id not in gate.index:
            continue
        level = str(gate.loc[dev_id, "level"])
        profile = get_profile(device["type"])
        frame = _frame_for(base, device)

        # Chiller via the Buildings EIR/EEIR FMU (FMU-1): fit curves, drive FMU,
        # select. Falls back to the Python path below when no FMU is available.
        if level == "full_physical" and device["type"] == "chiller" and chiller_fmu_cfg:
            workdir = base / "chiller" / dev_id / "fmu"
            fit = fit_chiller_fmu(dev_id, frame, chiller_fmu_cfg, chiller_fmu_root, workdir, thresholds)
            for r in fit.get("rows", []):
                metric_rows.append({"equipment_type": "chiller", **r,
                                    "params": json.dumps(r.get("params", {}), sort_keys=True)})
            if fit.get("status") == "ok":
                best = fit["best"]
                selection, test = best["selection"], best["test"]
                provenance = _fmu_provenance(best, chiller_fmu_root)
                selected_rows.append({
                    "device_id": dev_id, "level": level, "selected_candidate": best["candidate"],
                    "MAPE_pct": "", "CVRMSE_pct": test["P_CVRMSE_pct"],
                    "selection_CVRMSE_pct": selection["P_CVRMSE_pct"],
                    "test_CVRMSE_pct": test["P_CVRMSE_pct"], "period": "test", "status": "ok",
                    "test_skill": test.get("P_skill"),
                    "selection_skill": selection.get("P_skill"),
                    **provenance,
                })
                payload = {"candidate": best["candidate"], "coeffs": best["coeffs"],
                           "nominals": best["nominals"], "fmu": best["fmu"],
                           "table_param": best["table_param"], "output_interval": best["output_interval"],
                           "condenser": best["condenser"], "fmu_root": str(chiller_fmu_root), **provenance}
                param_rows.append({"device_id": dev_id, "candidate": best["candidate"],
                                   "status": "ok", "parameters": json.dumps(payload, sort_keys=True)})
            else:
                selected_rows.append({"device_id": dev_id, "level": level, "selected_candidate": "",
                                      "MAPE_pct": "", "CVRMSE_pct": "", "execution_engine": "none",
                                      "status": fit.get("status", "fit_failed")})
            continue

        # Cooling tower via the Buildings Merkel / YorkCalc FMU (FMU-2): calibrate
        # on representative steady windows, drive FMU, select by TOut + Q (Q scaled
        # by L2 per-timestep fan count). Falls back to the Python path below.
        if level == "full_physical" and device["type"] == "cooling_tower" and ct_fmu_cfg:
            from autofmu.devices.cooling_tower_fmu import fit_ct_fmu
            workdir = base / "cooling_tower" / dev_id / "fmu"
            fit = fit_ct_fmu(dev_id, frame, ct_fmu_cfg, ct_fmu_root, workdir, thresholds)
            for r in fit.get("rows", []):
                metric_rows.append({"equipment_type": "cooling_tower", **r,
                                    "params": json.dumps(r.get("params", {}), sort_keys=True)})
            if fit.get("status") == "ok":
                best = fit["best"]
                selection, test = best["selection"], best["test"]
                provenance = _fmu_provenance(best, ct_fmu_root)
                selected_rows.append({
                    "device_id": dev_id, "level": level, "selected_candidate": best["candidate"],
                    "MAPE_pct": "", "CVRMSE_pct": test["Q_CVRMSE_pct"],
                    "selection_CVRMSE_pct": selection["Q_CVRMSE_pct"],
                    "test_CVRMSE_pct": test["Q_CVRMSE_pct"], "target": "heat_rejection_W",
                    "T_RMSE_K": test.get("T_RMSE_K"),
                    "T_CVRMSE_pct_diagnostic": test["T_CVRMSE_pct"],
                    "P_CVRMSE_pct": test.get("P_CVRMSE_pct"),
                    "period": "test", "status": "ok",
                    "test_skill": test.get("T_skill"),
                    "selection_skill": selection.get("T_skill"),
                    **provenance,
                })
                payload = {"kind": "ct_fmu", "candidate": best["candidate"], "params": best["params"],
                           "base": best["base"], "fmu": best["fmu"], "fmu_root": str(ct_fmu_root),
                           "table_param": best["table_param"],
                           "static_start_values": best.get("static_start_values"), **provenance}
                param_rows.append({"device_id": dev_id, "candidate": best["candidate"],
                                   "status": "ok", "parameters": json.dumps(payload, sort_keys=True)})
            else:
                selected_rows.append({"device_id": dev_id, "level": level, "selected_candidate": "",
                                      "MAPE_pct": "", "CVRMSE_pct": "", "execution_engine": "none",
                                      "status": fit.get("status", "fit_failed")})
            continue

        # Heat exchanger via the Buildings Constant/PlateNTU FMU (FMU-3): build the
        # operating-window table (data-selected stream directions), grid-search,
        # drive FMU, select by uncontrolled T2Out + Q. Falls back to Python below.
        if level == "full_physical" and device["type"] == "heat_exchanger" and hx_fmu_cfg:
            from autofmu.devices.heat_exchanger_fmu import fit_hx_fmu
            workdir = base / "heat_exchanger" / dev_id / "fmu"
            fit = fit_hx_fmu(dev_id, frame, hx_fmu_cfg, hx_fmu_root, workdir, thresholds)
            for r in fit.get("rows", []):
                metric_rows.append({"equipment_type": "heat_exchanger", **r,
                                    "params": json.dumps(r.get("params", {}), sort_keys=True)})
            if fit.get("status") == "ok":
                best = fit["best"]
                selection, test = best["selection"], best["test"]
                provenance = _fmu_provenance(best, hx_fmu_root)
                selected_rows.append({
                    "device_id": dev_id, "level": level, "selected_candidate": best["candidate"],
                    # Heat, not outlet temperature: section 3.3 decides between
                    # heat-exchanger candidates on reconstructed-heat CVRMSE,
                    # and a CVRMSE on Celsius is unit-dependent anyway (M-12).
                    # HX-01 previously headlined 0.23%, which was a temperature
                    # CVRMSE and not an error on any energy quantity.
                    "MAPE_pct": "", "CVRMSE_pct": test["Q_CVRMSE_pct"],
                    "selection_CVRMSE_pct": selection["Q_CVRMSE_pct"],
                    "test_CVRMSE_pct": test["Q_CVRMSE_pct"],
                    "test_skill": test.get("Q_skill"),
                    "selection_skill": selection.get("Q_skill"),
                    "target": "heat_transfer_W",
                    "T2_RMSE_K_diagnostic": test.get("T2_RMSE_K"),
                    "T1_RMSE_K_diagnostic": test.get("T1_RMSE_K"),
                    "T2_admissible": test.get("T2_admissible"),
                    "period": "test", "status": "ok",
                    **provenance,
                })
                payload = {"kind": "hx_fmu", "candidate": best["candidate"], "params": best["params"],
                           "fmu": best["fmu"], "fmu_root": str(hx_fmu_root), **provenance}
                param_rows.append({"device_id": dev_id, "candidate": best["candidate"],
                                   "status": "ok", "parameters": json.dumps(payload, sort_keys=True)})
            else:
                selected_rows.append({"device_id": dev_id, "level": level, "selected_candidate": "",
                                      "MAPE_pct": "", "CVRMSE_pct": "", "execution_engine": "none",
                                      "status": fit.get("status", "fit_failed")})
            continue

        # Pump via the exported PumpEmpiricalPower FMU (FMU-4): OLS-fit c0..c4 for
        # affinity / speed-poly / speed-flow forms, drive FMU, select by power.
        if level == "full_physical" and device["type"] == "pump" and pump_fmu_cfg:
            from autofmu.devices.pump_fmu import fit_pump_fmu
            workdir = base / "pump" / dev_id / "fmu"
            fit = fit_pump_fmu(dev_id, frame, pump_fmu_cfg, pump_fmu_root, workdir, thresholds)
            for r in fit.get("rows", []):
                metric_rows.append({"equipment_type": "pump", **r,
                                    "params": json.dumps(r.get("params", {}), sort_keys=True)})
            if fit.get("status") == "ok":
                best = fit["best"]
                selection, test = best["selection"], best["test"]
                provenance = _fmu_provenance(best, pump_fmu_root)
                selected_rows.append({
                    "device_id": dev_id, "level": level, "selected_candidate": best["candidate"],
                    "MAPE_pct": "", "CVRMSE_pct": test["P_CVRMSE_pct"],
                    "selection_CVRMSE_pct": selection["P_CVRMSE_pct"],
                    "test_CVRMSE_pct": test["P_CVRMSE_pct"], "period": "test", "status": "ok",
                    "test_skill": test.get("P_skill"),
                    "selection_skill": selection.get("P_skill"),
                    **provenance,
                })
                payload = {"kind": "pump_fmu", "candidate": best["candidate"], "coeffs": best["coeffs"],
                           "nominals": best["nominals"], "fmu": best["fmu"], "fmu_root": str(pump_fmu_root),
                           "inputs": best.get("inputs", ["m_flow_in", "y_in"]), **provenance}
                param_rows.append({"device_id": dev_id, "candidate": best["candidate"],
                                   "status": "ok", "parameters": json.dumps(payload, sort_keys=True)})
            else:
                selected_rows.append({"device_id": dev_id, "level": level, "selected_candidate": "",
                                      "MAPE_pct": "", "CVRMSE_pct": "", "execution_engine": "none",
                                      "status": fit.get("status", "fit_failed")})
            continue

        if level == "full_physical":
            # Thermal devices are FMU-only (handled by the FMU branches above when
            # an FMU contract is available). Reaching here means no FMU was
            # available; FMU-5 removed the Python-physics fallback, so record an
            # explicit gate rather than fabricating a Python thermal result.
            strict_fmu = bool(config.get("fmu_config_dir")) and not bool(config.get("allow_python_empirical", False))
            if device["type"] in ("chiller", "heat_exchanger", "cooling_tower") or strict_fmu:
                selected_rows.append({"device_id": dev_id, "level": level, "selected_candidate": "",
                                      "MAPE_pct": "", "CVRMSE_pct": "", "execution_engine": "none",
                                      "fmu_sha256": "", "fmu_model_name": "", "status": "fmu_unavailable"})
            else:
                # Pump empirical power (affinity / speed-poly / speed-flow) -- the
                # same form as the pump FMU, used only when no pump FMU is set.
                rows, best, best_ts = calibrate_power_model(dev_id, frame, profile, thresholds)
                for row in rows:
                    metric_rows.append({**row, "params": json.dumps(row.get("params", {}), sort_keys=True)})
                if best:
                    selected_rows.append({
                        "device_id": dev_id, "level": level, "selected_candidate": best["candidate"],
                        "MAPE_pct": best["MAPE_pct"], "CVRMSE_pct": best["CVRMSE_pct"], "status": "ok",
                        "test_skill": best.get("skill_vs_mean"),
                        "execution_engine": "python_empirical", "fmu_sha256": "", "fmu_model_name": "",
                    })
                    param_rows.append({"device_id": dev_id, "candidate": best["candidate"],
                                       "status": "ok", "parameters": json.dumps(best.get("params", {}), sort_keys=True)})
                    if not best_ts.empty:
                        best_ts.to_csv(ts_dir / f"{dev_id}_best_timeseries.csv", index=False)
        elif level == "nominal_only":
            params = _nominal_params(frame, profile)
            reason = str(gate.loc[dev_id, "reason"])
            selected_rows.append({"device_id": dev_id, "level": level,
                                  "selected_candidate": "nominal_only", "MAPE_pct": "", "CVRMSE_pct": "",
                                  "execution_engine": "none", "status": "power_calibration_blocked"})
            param_rows.append({"device_id": dev_id, "candidate": "nominal_only",
                               "status": "power_calibration_blocked", "parameters": json.dumps(params, sort_keys=True)})
            metric_rows.append({"device_id": dev_id, "equipment_type": device["type"],
                                "candidate": "nominal_only", "target": profile.target,
                                "status": "power_calibration_blocked", "reason": reason, "params": json.dumps(params, sort_keys=True)})
        else:
            selected_rows.append({"device_id": dev_id, "level": level,
                                  "selected_candidate": "", "MAPE_pct": "", "CVRMSE_pct": "",
                                  "execution_engine": "none", "status": "blocked"})

    stage = base / "calibrate"
    paths = {
        "all_candidate_metrics.csv": pd.DataFrame(metric_rows),
        "selected_models.csv": pd.DataFrame(selected_rows),
        "parameters.csv": pd.DataFrame(param_rows),
    }
    for name, table in paths.items():
        table.to_csv(stage / name, index=False)
        manifest.add_artifact(stage / name, base)

    selected = paths["selected_models.csv"]
    # paper-comparison table for full_physical devices
    cmp_rows = []
    for _, row in selected.iterrows():
        if str(row.get("status")) != "ok":
            continue
        etype = next((d["type"] for d in config["devices"] if d["id"] == row["device_id"]), "")
        mape = pd.to_numeric(pd.Series([row.get("MAPE_pct")]), errors="coerce").iloc[0]
        if not np.isfinite(mape):
            continue
        cmp_rows.append({
            "device_id": row["device_id"],
            "equipment_type": etype,
            "selected_candidate": row["selected_candidate"],
            "our_MAPE_pct": round(float(mape), 2),
            "paper_MAPE_pct": PAPER_MAPE.get(etype, ""),
        })
    summary = stage / "summary.md"
    summary.write_text(
        "# Calibration Report\n\n"
        "Empirical power calibration gated by L2 modelability. full_physical "
        "devices are fit on training data, selected on a disjoint selection period, "
        "and reported on an untouched test period. Metric names are preserved "
        "without relabelling CVRMSE as MAPE. nominal_only devices emit "
        "design/observed nominals only -- no fabricated power fit.\n\n"
        "## Selected models\n\n"
        + (table_to_markdown(selected) if not selected.empty else "_no devices_")
        + "\n\n## Comparison vs AlphaDataCenterCooling (same Tencent plant)\n\n"
        + (table_to_markdown(pd.DataFrame(cmp_rows)) if cmp_rows else "_no calibrated devices_")
        + "\n\n> Our speed/affinity fit uses BMS data only. The paper's lower MAPE "
        "uses manufacturer characteristic curves, restriction to single-device "
        "operating windows, and (for pumps) an MLP head model to resolve the "
        "speed->flow->power ambiguity. The residual gap is the irreducible scatter "
        "of power at fixed speed caused by system head/flow variation -- which "
        "defines the next milestone: attribute flow on solo-run windows (CT01 has "
        "thousands) and add it as a feature, or export the Buildings FMU.\n",
        encoding="utf-8",
    )
    manifest.add_artifact(summary, base)
    manifest.record_stage("calibrate")
    manifest.write()
    return base
