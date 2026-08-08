"""FMU-4: pump L3 engine -- fit c0..c4 and drive PumpEmpiricalPower FMU.

Pure table-build + coefficient-mapping tests run everywhere; the end-to-end
fit/validate needs fmpy + the exported pump FMU + ingested pump canonical.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from autofmu.devices import pump_fmu as pf

REPO = Path(__file__).resolve().parents[1]
PUMP_CFG = yaml.safe_load((REPO / "configs" / "fmu" / "pump.yaml").read_text(encoding="utf-8"))
FMU_ROOT = Path(PUMP_CFG["fmu_root"])
PUMP_FMU = FMU_ROOT / "outputs" / "pump" / "stage4_modelica_interface" / "fmu" / "PumpEmpiricalPower.fmu"
RUN_CANON = REPO / "outputs" / "runs" / "fmu1dev" / "pump" / "CDWP_01" / "canonical_attributed.csv"


def _synthetic_pump(n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, n)
    speed = 35.0 + 8.0 * np.sin(t) + rng.normal(0, 0.3, n)
    flow = 700.0 + 60.0 * np.sin(t / 2)
    y = speed / 50.0
    power = 40000.0 * (0.1 + 0.85 * y ** 3)         # cube-law dominated
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-06-01", periods=n, freq="5min").astype(str),
        "power_W": power, "speed_Hz": speed, "run_signal": speed,
        "attributed_flow_m3_h": flow,
    })


def test_build_pump_table_columns():
    table = pf.build_pump_table(_synthetic_pump())
    assert {"time_s", "m_flow_in", "y_in", "P_meas_W"} <= set(table.columns)
    assert len(table) > 200
    assert (table["m_flow_in"] > 0).all()
    assert np.isfinite(table[["time_s", "m_flow_in", "y_in", "P_meas_W"]].to_numpy(float)).all()


def test_build_pump_table_handles_missing_flow():
    # Datasets without a pump flow meter (e.g. Tencent) must stay modellable
    # speed-only: a placeholder flow keeps rows valid; speed candidates ignore it.
    frame = _synthetic_pump().drop(columns=["attributed_flow_m3_h"])
    table = pf.build_pump_table(frame)
    assert len(table) > 100
    assert np.isfinite(table["m_flow_in"].to_numpy(dtype=float)).all()
    assert (table["m_flow_in"] > 0).all()


def test_coeff_mapping_slots():
    c = pf._coeffs_from_fit(np.array([0.1, 0.9]), ["1", "y3"])
    assert c == {"c0": 0.1, "c1": 0.0, "c2": 0.0, "c3": 0.9, "c4": 0.0}
    c2 = pf._coeffs_from_fit(np.array([2.0, 3.0, 4.0, 5.0, 6.0]), ["1", "phi", "y", "y3", "phi_y"])
    assert c2 == {"c0": 2.0, "c1": 3.0, "c2": 4.0, "c3": 5.0, "c4": 6.0}


HAS_FMPY = importlib.util.find_spec("fmpy") is not None


@pytest.mark.skipif(not HAS_FMPY or not (PUMP_FMU.exists() and RUN_CANON.exists()),
                    reason="pump FMU / ingested pump canonical not available")
def test_fit_and_validate_pump_fmu_runs(tmp_path):
    frame = pd.read_csv(RUN_CANON)
    th = {"run_on": 0.5, "min_calibration_rows": 200, "validation_fold": 3, "pump_subset_rows": 2000}
    fit = pf.fit_pump_fmu("CDWP_01", frame, PUMP_CFG, FMU_ROOT, tmp_path, th)
    assert fit["status"] == "ok"
    assert fit["selected_candidate"] in pf.CANDIDATES
    for cand in fit["candidates"]:
        assert "heldout" not in cand
        assert np.isfinite(cand["selection"]["P_CVRMSE_pct"])
        assert np.isfinite(cand["test"]["P_CVRMSE_pct"])
    val = pf.validate_pump_fmu("CDWP_01", frame, fit["best"], FMU_ROOT, tmp_path, th)
    assert val["status"] == "ok"
    assert val["full_period"]["P_CVRMSE_pct"] < 8.0   # CDWP_01 baseline ~3.5-3.75%
    assert val["full_period"]["N"] > 100
