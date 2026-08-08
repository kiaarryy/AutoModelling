"""FMU-1: chiller modelling driven by the Buildings EIR/EEIR FMUs.

Pure-function tests (table build / nominals / coeff mapping / scoring) run
everywhere. The end-to-end fit+validate test needs fmpy + the external EIR FMU +
the screening library + real Site A data, and skips otherwise.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from autofmu.devices import chiller_fmu as cf

REPO = Path(__file__).resolve().parents[1]
CHILLER_CFG = yaml.safe_load((REPO / "configs" / "fmu" / "chiller.yaml").read_text(encoding="utf-8"))
FMU_ROOT = Path(CHILLER_CFG["fmu_root"])
EIR_FMU = FMU_ROOT / "Cali_EIR_BSU_CH1" / "Modelica" / "ChillerCalibration_FMU_chiller_0EIR3.fmu"
EIR_LIB = FMU_ROOT / "Cali_EIR_BSU_CH1" / "ChillerData" / "ChillerData_v3.xlsx"
RAW_CH01 = FMU_ROOT / "CT_Model" / "DATA" / "Site_A" / "chiller" / "CH_01.csv"


def _smooth_canonical(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """A slowly varying, steady operating series the steady mask will keep."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 6 * np.pi, n)
    q = 2.0e6 + 4.0e5 * np.sin(t) + rng.normal(0, 5e3, n)   # ~2 MW, smooth
    te = 16.0 + 1.0 * np.sin(t / 2)                          # warm DC chilled water
    tcws = 25.0 + 1.5 * np.sin(t / 3 + 1.0)
    cop = 6.0 + 0.5 * np.sin(t / 4)
    power = q / cop
    chw_flow = q / (cf.CP_WATER_KJ * 1000 * 4.0) * 3.6       # m3/h for ~4 K dT
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="30min").astype(str),
        "power_W": power, "chw_flow_m3_h": chw_flow, "cw_flow_m3_h": chw_flow * 1.4,
        "tchws_C": te, "tchwr_C": te + 4.0, "tcws_C": tcws, "tcwr_C": tcws + 1.2,
        "run_signal": np.ones(n),
    })


# --------------------------------------------------------------------------- #
# Pure-function unit tests (no FMU)
# --------------------------------------------------------------------------- #
def test_build_alldata2_shape_and_units():
    frame = _smooth_canonical()
    table = cf.build_alldata2(frame)
    assert list(table.columns) == cf.TABLE_COLUMNS
    assert len(table) > 100
    # CHW column is L/s (== kg/s for water) == m3/h / 3.6
    assert np.allclose(table["CHW"], frame["chw_flow_m3_h"].iloc[: len(table)].to_numpy() / 3.6, rtol=0.5)
    assert (table["P/kw"] > 0).all() and (table["Q_evap_kW"] > 0).all()
    assert np.isfinite(table.to_numpy(dtype=float)).all()


def test_build_alldata2_rejects_zero_condenser_flow():
    frame = _smooth_canonical(200)
    baseline = cf.build_alldata2(frame)
    frame.loc[:9, "cw_flow_m3_h"] = 0.0
    table = cf.build_alldata2(frame)
    assert len(table) < len(baseline)
    assert (table["CDW"] > 0).all()


def test_estimate_nominals_keys_and_condenser_basis():
    table = cf.build_alldata2(_smooth_canonical())
    nom_in = cf.estimate_nominals(table, "entering")
    nom_lv = cf.estimate_nominals(table, "leaving")
    for key in ("QEva_flow_nominal", "COP_nominal", "TEvaLvg_nominal", "TCon_nominal", "PLRMax"):
        assert key in nom_in
    assert nom_in["QEva_flow_nominal"] < 0           # Buildings sign convention
    assert nom_in["COP_nominal"] > 1
    # entering (CDWS) is colder than leaving (CDWR) -> different nominal Tcon
    assert nom_lv["TCon_nominal"] >= nom_in["TCon_nominal"]


def test_nominal_start_values_maps_to_fmu_names():
    table = cf.build_alldata2(_smooth_canonical())
    nominals = cf.estimate_nominals(table, "entering")
    eir = next(c for c in CHILLER_CFG["candidates"] if c["name"] == "EIR")
    sv = cf.nominal_start_values(nominals, eir)
    assert "datChi.COP_nominal" in sv and "datChi.QEva_flow_nominal" in sv
    assert all(np.isfinite(v) for v in sv.values())


def test_pipeline_binds_chiller_fmu_fit():
    # Guard against dropping the module-level import the chiller FMU branch needs
    # (the branch references fit_chiller_fmu at runtime, so a missing import only
    # surfaces when the full pipeline runs with an FMU contract).
    import importlib
    cal_mod = importlib.import_module("autofmu.pipeline.calibrate")
    assert callable(cal_mod.fit_chiller_fmu)
    assert callable(cal_mod._load_device_fmu_cfg)


def test_coeff_names_counts():
    eir = next(c for c in CHILLER_CFG["candidates"] if c["name"] == "EIR")
    eeir = next(c for c in CHILLER_CFG["candidates"] if c["name"] == "EEIR")
    assert len(cf.coeff_names(eir)) == 15      # 6 + 6 + 3
    assert len(cf.coeff_names(eeir)) == 22     # 6 + 6 + 10 (FMU exposes EIRFunPLR_1..10)


def test_score_frame_perfect_and_offset():
    base = pd.DataFrame({"P_m": [100.0, 200, 300], "Q_m": [600.0, 1200, 1800]})
    perfect = base.assign(P_s=base["P_m"], Q_s=base["Q_m"])
    m = cf.score_frame(perfect)
    assert m["P_CVRMSE_pct"] < 1e-6 and m["COP_CVRMSE_pct"] < 1e-6
    assert m["criterion"] == "raw_interval_custom" and "P_GL14" not in m
    offset = base.assign(P_s=base["P_m"] * 1.1, Q_s=base["Q_m"])
    assert cf.score_frame(offset)["P_CVRMSE_pct"] > 5.0


def test_write_alldata2_roundtrip(tmp_path):
    table = cf.build_alldata2(_smooth_canonical())
    path = tmp_path / "AllData2.txt"
    cf.write_alldata2(table, path)
    lines = path.read_text().splitlines()
    assert lines[0] == "#1"
    assert lines[1] == f"double AllData2({len(table)},12)"
    assert lines[2].startswith("#Time")
    assert len(lines) == len(table) + 3


# --------------------------------------------------------------------------- #
# End-to-end fit + validate via the real FMU (guarded)
# --------------------------------------------------------------------------- #
HAS_FMPY = importlib.util.find_spec("fmpy") is not None


def _real_ch01_canonical(start: int = 20000, stop: int = 50000) -> pd.DataFrame:
    """Map raw Site A CH_01 columns to canonical names (adapter contract).

    Uses a mid-period slice; the early rows of the file have the chiller off.
    """
    raw = pd.read_csv(RAW_CH01).iloc[start:stop].reset_index(drop=True)
    g = lambda rx: pd.to_numeric(raw.filter(regex=rx).iloc[:, 0], errors="coerce")
    return pd.DataFrame({
        "timestamp": raw["DateTime"],
        "power_W": pd.to_numeric(raw["P/kw"], errors="coerce") * 1000.0,
        "chw_flow_m3_h": g(r"CHW-WFM") * 3.6, "cw_flow_m3_h": g(r"CDW-WFM") * 3.6,
        "tchws_C": g(r"CHWS-WTS"), "tchwr_C": g(r"CHWR-WTS"),
        "tcws_C": g(r"CDWS-WTS"), "tcwr_C": g(r"CDWR-WTS"),
        "run_signal": g(r"VSD-STS"),
    })


@pytest.mark.skipif(not HAS_FMPY or not (EIR_FMU.exists() and EIR_LIB.exists() and RAW_CH01.exists()),
                    reason="EIR FMU / library / Site A data not available")
def test_fit_and_validate_chiller_fmu_non_regression(tmp_path):
    frame = _real_ch01_canonical()
    # Fast settings: tiny library screen + tiny refine budget.
    th = {"run_on": 0.5, "validation_fold": 3, "min_full_physical_rows": 150,
          "subset_rows": 200, "max_refine_nfev": 6, "screening_max_rows": 8}
    fit = cf.fit_chiller_fmu("CH_01", frame, CHILLER_CFG, FMU_ROOT, tmp_path, th)
    assert fit["status"] == "ok"
    assert fit["selected_candidate"] in ("EIR", "EEIR")
    for cand in fit["candidates"]:
        assert "heldout" not in cand
        assert np.isfinite(cand["selection"]["P_CVRMSE_pct"])
        assert np.isfinite(cand["test"]["P_CVRMSE_pct"])
        assert cand["selection"]["P_CVRMSE_pct"] <= cand["baseline_selection"]["P_CVRMSE_pct"] + 1e-6
    val = cf.validate_chiller_fmu("CH_01", frame, fit["best"], FMU_ROOT, tmp_path, th)
    assert val["status"] == "ok"
    assert np.isfinite(val["full_period"]["P_CVRMSE_pct"])
    assert val["full_period"]["N"] > 100
    assert "P_GL14" not in val["full_period"]
