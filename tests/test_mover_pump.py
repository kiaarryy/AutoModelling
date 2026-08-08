"""#5 proof: the Buildings mover (SpeedControlled_y + system curve) is a 2nd pump
*type*, added declaratively and opt-in, with its own fit strategy (scale_fit:
speed-only drive, one power scale) -- so the empirical-power default is untouched.
"""
from __future__ import annotations

import copy
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
EMP_FMU = FMU_ROOT / "outputs" / "pump" / "stage4_modelica_interface" / "fmu" / "PumpEmpiricalPower.fmu"
MOVER_FMU = FMU_ROOT / "outputs" / "pump" / "mbl_speed_system" / "fmu" / "SiteAPumpMblSpeedSystemCurve.fmu"
RUN_CANON = REPO / "outputs" / "runs" / "fmu1dev" / "pump" / "CDWP_01" / "canonical_attributed.csv"
HAS_FMPY = importlib.util.find_spec("fmpy") is not None


def test_mover_declared_opt_in():
    names = [c["name"] for c in PUMP_CFG["candidates"]]
    assert "mover" in names                                   # in the catalog
    assert PUMP_CFG["enabled_candidates"] == ["empirical_power"]   # but not a default
    mover = next(c for c in PUMP_CFG["candidates"] if c["name"] == "mover")
    assert mover["fit_strategy"] == "scale_fit"
    assert mover["inputs"] == ["y_in"]                        # speed-only drive


@pytest.mark.skipif(not HAS_FMPY or not (EMP_FMU.exists() and MOVER_FMU.exists() and RUN_CANON.exists()),
                    reason="pump FMUs / ingested pump canonical not available")
def test_mover_competes_when_enabled_and_best_is_selected(tmp_path):
    frame = pd.read_csv(RUN_CANON)
    cfg = copy.deepcopy(PUMP_CFG)
    cfg["enabled_candidates"] = ["empirical_power", "mover"]   # opt the new type in
    th = {"run_on": 0.5, "min_calibration_rows": 200, "pump_subset_rows": 800}
    fit = pf.fit_pump_fmu("CDWP_01", frame, cfg, FMU_ROOT, tmp_path, th)

    assert fit["status"] == "ok"
    competed = {c["candidate"] for c in fit["candidates"]}
    assert "mover" in competed                                # the new type ran
    assert competed >= set(pf.CANDIDATES)                     # alongside the empirical forms
    mover = next(c for c in fit["candidates"] if c["candidate"] == "mover")
    assert np.isfinite(mover["selection"]["P_CVRMSE_pct"])    # drove its FMU + scored
    assert mover["inputs"] == ["y_in"]
    # framework auto-selects the best of all variants by held-out power
    assert fit["best"]["score"] == min(c["score"] for c in fit["candidates"])
    # the selected model validates full-period regardless of which type won
    val = pf.validate_pump_fmu("CDWP_01", frame, fit["best"], FMU_ROOT, tmp_path, th)
    assert val["status"] == "ok"
    assert np.isfinite(val["full_period"]["P_CVRMSE_pct"]) and val["full_period"]["N"] > 100
