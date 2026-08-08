"""FMU-3: heat-exchanger L3 engine -- drive Constant/PlateNTU FMUs from L2 data.

Pure table-build test runs everywhere; the end-to-end fit/validate needs fmpy +
the external HX FMUs + ingested HX canonical and skips otherwise.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from autofmu.devices import heat_exchanger_fmu as hf

REPO = Path(__file__).resolve().parents[1]
HX_CFG = yaml.safe_load((REPO / "configs" / "fmu" / "heat_exchanger.yaml").read_text(encoding="utf-8"))
FMU_ROOT = Path(HX_CFG["fmu_root"])
HX_CONST = FMU_ROOT / "outputs" / "heat_exchanger" / "modelica_interface" / "fmu" / "SiteAHXConstantEffectiveness.fmu"
RUN_CANON = REPO / "outputs" / "runs" / "fmu1dev" / "heat_exchanger" / "HX_01" / "canonical_attributed.csv"


def _synthetic_hx(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, n)
    t2i = 20.0 + 1.5 * np.sin(t)            # condenser inlet (cold)
    t1i = t2i + 4.0 + 0.5 * np.sin(t / 2)   # chilled inlet (hot side)
    approach = 1.0 + 0.2 * np.cos(t)
    t1o = t2i + approach                    # chilled outlet (cooled toward t2i)
    t2o = t1i - approach                    # condenser outlet (warmed toward t1i)
    flow = 400.0 + 30.0 * np.sin(t / 3)     # m3/h, both sides similar
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-06-01", periods=n, freq="5min").astype(str),
        "tchwr_C": t1i, "tchws_C": t1o, "tcws_C": t2i, "tcwr_C": t2o,
        "chw_flow_m3_h": flow, "cw_flow_m3_h": flow * 1.02, "power_W": 5000.0,
    })


def test_build_hx_table_columns_and_direction():
    table = hf.build_hx_table(_synthetic_hx())
    assert list(table.columns) == hf.TABLE_COLUMNS
    assert len(table) > 200
    # side 1 cooled (T1In > T1Out), side 2 warmed (T2Out > T2In)
    assert (table["T1In_m_C"] > table["T1Out_m_C"]).mean() > 0.9
    assert (table["T2Out_m_C"] > table["T2In_m_C"]).mean() > 0.9
    assert (table["Q_m_W"] > 0).all()
    assert np.isfinite(table.to_numpy(dtype=float)).all()


HAS_FMPY = importlib.util.find_spec("fmpy") is not None


@pytest.mark.skipif(not HAS_FMPY or not (HX_CONST.exists() and RUN_CANON.exists()),
                    reason="HX FMU / ingested HX canonical not available")
def test_fit_and_validate_hx_fmu_runs(tmp_path):
    frame = pd.read_csv(RUN_CANON)
    th = {"min_calibration_rows": 200, "validation_fold": 3, "hx_subset_rows": 1500}
    fit = hf.fit_hx_fmu("HX_01", frame, HX_CFG, FMU_ROOT, tmp_path, th)
    assert fit["status"] == "ok"
    assert fit["selected_candidate"] in ("ConstantEffectiveness", "PlateEffectivenessNTU")
    for cand in fit["candidates"]:
        assert "heldout" not in cand
        assert np.isfinite(cand["selection"]["T2_CVRMSE_pct"])
        assert np.isfinite(cand["test"]["T2_CVRMSE_pct"])
    val = hf.validate_hx_fmu("HX_01", frame, fit["best"], FMU_ROOT, tmp_path, th)
    assert val["status"] == "ok"
    # uncontrolled condenser-outlet temp tracks tightly on Site A HX_01
    assert val["full_period"]["T2_CVRMSE_pct"] < 3.0
    assert val["full_period"]["N"] > 100
