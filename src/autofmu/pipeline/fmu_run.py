"""Stage: real FMU execution + measured-vs-simulated metrics (L3 physical).

Config-driven. Each entry under ``fmu_runs`` points at an exported FMU and lists
the measured/simulated output pairs to score. The measured operating points are
embedded in the FMU's own data table; a machine-specific table path can be
overridden via ``table_param`` -> ``table_file``.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from autofmu.config import resolve_path, run_dir
from autofmu.fmu.runner import run_fmu
from autofmu.manifest import RunManifest
from autofmu.metrics import regression_metrics
from autofmu.reporting import table_to_markdown


def load_fmu_config(path: Path) -> dict:
    config_path = Path(path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    config["_config_path"] = config_path
    config["_root"] = config_path.parent
    config.setdefault("outputs_dir", "outputs")
    config.setdefault("fmu_runs", [])
    return config


def fmu_run(config: dict, run_id: str) -> Path:
    base = run_dir(config, run_id)
    manifest = RunManifest(base / "manifest.json", run_id)
    manifest.bind_run(config)
    stage = base / "fmu_run"
    ts_dir = stage / "timeseries"
    ts_dir.mkdir(parents=True, exist_ok=True)

    metric_rows = []
    for spec in config["fmu_runs"]:
        run_name = spec["id"]
        fmu_path = resolve_path(config, spec["fmu"])
        start_values = dict(spec.get("start_values", {}))
        if spec.get("table_param") and spec.get("table_file"):
            table = resolve_path(config, spec["table_file"])
            start_values[spec["table_param"]] = str(table).replace("\\", "/")
        outputs = spec.get("outputs")
        try:
            frame = run_fmu(
                fmu_path,
                start_values=start_values or None,
                output=outputs,
                stop_time=spec.get("stop_time"),
                output_interval=spec.get("output_interval"),
            )
        except Exception as exc:  # surface, never fake
            manifest.add_warning(f"{run_name}: FMU run failed: {exc}")
            metric_rows.append({"run": run_name, "status": "fmu_failed", "reason": str(exc)})
            continue
        frame.to_csv(ts_dir / f"{run_name}.csv", index=False)
        manifest.add_artifact(ts_dir / f"{run_name}.csv", base)
        for pair in spec.get("compare", []):
            measured, simulated = frame[pair["measured"]], frame[pair["simulated"]]
            metric = regression_metrics(measured, simulated)
            metric_rows.append({"run": run_name, "variable": pair["name"], "status": "ok",
                                "measured_col": pair["measured"], "simulated_col": pair["simulated"], **metric})

    report = pd.DataFrame(metric_rows)
    metrics_csv = stage / "fmu_metrics.csv"
    report.to_csv(metrics_csv, index=False)
    (stage / "summary.md").write_text(
        "# FMU Run Report\n\n"
        "Real FMPy execution of exported equipment FMUs; measured (`*_m`) vs "
        "simulated (`*_s`) outputs scored by CVRMSE/MAPE.\n\n"
        + (table_to_markdown(report) if not report.empty else "_no fmu runs_")
        + "\n",
        encoding="utf-8",
    )
    manifest.add_artifact(metrics_csv, base)
    manifest.add_artifact(stage / "summary.md", base)
    manifest.record_stage("fmu_run")
    manifest.write()
    return base
