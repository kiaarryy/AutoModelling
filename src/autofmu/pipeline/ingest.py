"""Stage 1 (L1): raw BMS -> canonical, per device, with QA + manifest."""
from __future__ import annotations

from pathlib import Path

from autofmu.adapters import adapt_csv
from autofmu.config import adapter_config_path, data_root, run_dir
from autofmu.manifest import RunManifest


def _device_dir(config: dict, run_id: str, device: dict) -> Path:
    return run_dir(config, run_id) / device["type"] / device["id"]


def ingest(config: dict, run_id: str) -> Path:
    base = run_dir(config, run_id)
    manifest = RunManifest(base / "manifest.json", run_id)
    manifest.bind_run(config)
    root = data_root(config)
    for device in config["devices"]:
        out = _device_dir(config, run_id, device) / "canonical.csv"
        result = adapt_csv(adapter_config_path(config, device), root, out)
        manifest.add_artifact(result.output_csv, base)
        manifest.add_artifact(result.qa_csv, base)
        if result.rows == 0:
            manifest.add_warning(f"{device['id']}: 0 rows ingested")
    manifest.record_stage("ingest")
    manifest.write()
    return base
