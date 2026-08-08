"""Stage 5 (report): consolidate the run into a single markdown report."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from autofmu.config import run_dir
from autofmu.manifest import RunManifest
from autofmu.reporting import table_to_markdown


def _read(path: Path):
    return pd.read_csv(path) if path.exists() else None


def report(config: dict, run_id: str) -> Path:
    base = run_dir(config, run_id)
    manifest = RunManifest(base / "manifest.json", run_id)
    manifest.bind_run(config)
    parts = [f"# autofmu run report: {run_id}\n", f"Dataset: {config.get('dataset', '?')}\n"]

    gate = _read(base / "attribute" / "modelability_report.csv")
    if gate is not None:
        counts = ", ".join(f"{k}={v}" for k, v in gate["level"].value_counts().items())
        parts += ["## L2 modelability gating", "", f"Levels: {counts}", "",
                  table_to_markdown(gate[["device_id", "equipment_type", "level", "solo_run_rows"]]), ""]

    sel = _read(base / "calibrate" / "selected_models.csv")
    if sel is not None:
        parts += ["## L3 calibration (selection and untouched test periods)", "", table_to_markdown(sel), ""]

    # chiller: per-device EIR vs EEIR on untouched test P and COP.
    cand = _read(base / "calibrate" / "all_candidate_metrics.csv")
    if cand is not None and "equipment_type" in cand:
        ch = cand[(cand["equipment_type"] == "chiller") & (cand.get("status") == "ok")]
        if "stage" in ch:
            ch = ch[ch["stage"] == "test"]
        if ch is not None and not ch.empty:
            short = {"ElectricReformulatedEIR": "EEIR", "ElectricEIR": "EIR"}
            rows = []
            for dev, g in ch.groupby("device_id"):
                row = {"device_id": dev}
                for _, r in g.iterrows():
                    m = short.get(r["candidate"], r["candidate"])
                    row[f"{m}_P_CVRMSE"] = round(float(r["P_CVRMSE_pct"]), 1)
                    row[f"{m}_COP_CVRMSE"] = round(float(r["COP_CVRMSE_pct"]), 1)
                selected_row = sel[sel["device_id"] == dev] if sel is not None else pd.DataFrame()
                best = selected_row.iloc[0]["selected_candidate"] if not selected_row.empty else ""
                row["selected_by_selection_period"] = short.get(best, best)
                rows.append(row)
            order = ["device_id", "EIR_P_CVRMSE", "EEIR_P_CVRMSE", "EIR_COP_CVRMSE",
                     "EEIR_COP_CVRMSE", "selected_by_selection_period"]
            tbl = pd.DataFrame(rows)
            tbl = tbl[[c for c in order if c in tbl.columns]]
            parts += ["## Chiller EIR vs EEIR (test P / COP CVRMSE %, per device)", "",
                      "Both Buildings models are fitted from training data and selected on a separate selection period.",
                      "", table_to_markdown(tbl), ""]

    val = _read(base / "validate" / "full_period_metrics.csv")
    if val is not None:
        cols = [c for c in ["device_id", "equipment_type", "status", "candidate", "N", "MAPE_pct", "CVRMSE_pct"] if c in val]
        parts += ["## Full-valid-points diagnostics (uncontrolled targets)", "", table_to_markdown(val[cols]), ""]

    ctrl = _read(base / "validate" / "control_variables.csv")
    if ctrl is not None and not ctrl.empty:
        parts += ["## Controlled variables (excluded from model error)", "",
                  "Regulated to a set point; reported for context only, never as accuracy.", "",
                  table_to_markdown(ctrl), ""]

    fmu = _read(base / "fmu_run" / "fmu_metrics.csv")
    if fmu is not None:
        parts += ["## Physical FMU runs", "", table_to_markdown(fmu), ""]

    out = base / "run_report.md"
    out.write_text("\n".join(parts) + "\n", encoding="utf-8")
    manifest.add_artifact(out, base)
    manifest.record_stage("report")
    manifest.write()
    return base
