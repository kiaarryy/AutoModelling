from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from autofmu.pipeline.calibrate import calibrate


def test_declared_fmu_project_does_not_silently_fallback_for_pump(tmp_path):
    base = tmp_path / "runs" / "strict"
    device_dir = base / "pump" / "P01"
    device_dir.mkdir(parents=True)
    n = 30
    speed = np.linspace(20.0, 50.0, n)
    pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="5min").astype(str),
        "power_W": 1000.0 + 20.0 * speed ** 2,
        "speed_Hz": speed,
        "run_signal": np.ones(n),
    }).to_csv(device_dir / "canonical.csv", index=False)
    (base / "attribute").mkdir(parents=True)
    pd.DataFrame([{"device_id": "P01", "level": "full_physical", "reason": ""}]).to_csv(
        base / "attribute" / "modelability_report.csv", index=False
    )
    fmu_dir = tmp_path / "fmu"
    fmu_dir.mkdir()
    (fmu_dir / "pump.yaml").write_text(yaml.safe_dump({
        "fmu_root": str(tmp_path),
        "candidates": [{"name": "PumpEmpiricalPower", "fmu": "missing.fmu"}],
    }), encoding="utf-8")
    config = {
        "_root": tmp_path,
        "outputs_dir": str(tmp_path),
        "fmu_config_dir": str(fmu_dir),
        "devices": [{"id": "P01", "type": "pump"}],
        "thresholds": {"min_calibration_rows": 5},
    }

    calibrate(config, "strict")

    selected = pd.read_csv(base / "calibrate" / "selected_models.csv").iloc[0]
    assert selected["status"] == "fmu_unavailable"
    assert selected["execution_engine"] == "none"
