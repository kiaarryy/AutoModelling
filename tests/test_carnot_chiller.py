"""Phase-4 proof: Carnot is a *new* chiller type added entirely declaratively.

It shows the extensibility end to end: a third Buildings model type (Carnot_TEva,
its FMU already exported) competes via the selection knob with a different fit
strategy (refine_only, no curve library) -- and the framework still auto-selects
the best, so the EIR/EEIR defaults are untouched (Carnot is opt-in).
"""
from __future__ import annotations

import copy
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
CARNOT_FMU = FMU_ROOT / "Cali_EIR_BSU_CH1" / "Modelica" / "ChillerCalibration_FMU_chiller_Carnot.fmu"
RAW_CH01 = FMU_ROOT / "CT_Model" / "DATA" / "Site_A" / "chiller" / "CH_01.csv"
HAS_FMPY = importlib.util.find_spec("fmpy") is not None


def _carnot():
    return next(c for c in CHILLER_CFG["candidates"] if c["name"] == "Carnot")


# --------------------------------------------------------------------------- #
# Pure-config / declaration tests (no FMU)
# --------------------------------------------------------------------------- #
def test_carnot_is_declared_but_opt_in():
    names = [c["name"] for c in CHILLER_CFG["candidates"]]
    assert "Carnot" in names                              # in the catalog
    assert CHILLER_CFG["enabled_candidates"] == ["EIR", "EEIR"]   # but not a default


def test_carnot_refine_only_tunable_all_warm_started():
    carnot = _carnot()
    assert carnot["fit_strategy"] == "refine_only"
    # Only the nominal Carnot effectiveness is fitted; the part-load polynomial keeps
    # the model's valid default (Carnot_TEva asserts etaPL(y=1)=1, so a is not fitted).
    names = cf.coeff_names(carnot)
    assert names == ["Chi.etaCarnot_nominal"]
    assert set(carnot["warm_start"]) == set(names)        # every tunable has a warm start


# --------------------------------------------------------------------------- #
# End-to-end: Carnot competes via the real FMU (guarded)
# --------------------------------------------------------------------------- #
def _real_ch01() -> pd.DataFrame:
    raw = pd.read_csv(RAW_CH01).iloc[20000:50000].reset_index(drop=True)
    g = lambda rx: pd.to_numeric(raw.filter(regex=rx).iloc[:, 0], errors="coerce")
    return pd.DataFrame({
        "timestamp": raw["DateTime"],
        "power_W": pd.to_numeric(raw["P/kw"], errors="coerce") * 1000.0,
        "chw_flow_m3_h": g(r"CHW-WFM") * 3.6, "cw_flow_m3_h": g(r"CDW-WFM") * 3.6,
        "tchws_C": g(r"CHWS-WTS"), "tchwr_C": g(r"CHWR-WTS"),
        "tcws_C": g(r"CDWS-WTS"), "tcwr_C": g(r"CDWR-WTS"), "run_signal": g(r"VSD-STS"),
    })


@pytest.mark.skipif(
    not HAS_FMPY or not (EIR_FMU.exists() and EIR_LIB.exists() and CARNOT_FMU.exists() and RAW_CH01.exists()),
    reason="chiller FMUs / library / Site A data not available")
def test_carnot_competes_when_enabled_and_best_is_selected(tmp_path):
    frame = _real_ch01()
    cfg = copy.deepcopy(CHILLER_CFG)
    cfg["enabled_candidates"] = ["EIR", "EEIR", "Carnot"]   # opt the new type in
    th = {"run_on": 0.5, "min_full_physical_rows": 150,
          "subset_rows": 200, "max_refine_nfev": 8, "screening_max_rows": 6}
    fit = cf.fit_chiller_fmu("CH_01", frame, cfg, FMU_ROOT, tmp_path, th)

    assert fit["status"] == "ok"
    competed = {c["candidate"] for c in fit["candidates"]}
    assert competed == {"EIR", "EEIR", "Carnot"}            # the new type really ran
    carnot = next(c for c in fit["candidates"] if c["candidate"] == "Carnot")
    assert np.isfinite(carnot["selection"]["P_CVRMSE_pct"])  # drove its FMU + scored
    assert carnot["selection"]["P_CVRMSE_pct"] <= carnot["baseline_selection"]["P_CVRMSE_pct"] + 1e-6
    # Honest selection: the framework auto-selects the best of the three by held-out
    # power -- whichever type that is. The point is that the new type competes.
    assert fit["selected_candidate"] in competed
    assert fit["best"]["score"] == min(c["score"] for c in fit["candidates"])
