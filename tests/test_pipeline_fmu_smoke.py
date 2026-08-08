"""Fast guard for the FULL pipeline on the FMU path (calibrate + validate with an
`fmu_config_dir`). The Site A golden also covers this but is slow + data-gated;
this runs only the chiller FMU branch on a small slice with a tiny fit budget, so
a regression like a dropped FMU import (FMU-5) is caught quickly. Skips without
fmpy / the external chiller FMU + library + Site A data."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from autofmu.pipeline import calibrate, validate

REPO = Path(__file__).resolve().parents[1]
CHILLER_CFG = yaml.safe_load((REPO / "configs" / "fmu" / "chiller.yaml").read_text(encoding="utf-8"))
FMU_ROOT = Path(CHILLER_CFG["fmu_root"])
EIR_FMU = FMU_ROOT / "Cali_EIR_BSU_CH1" / "Modelica" / "ChillerCalibration_FMU_chiller_0EIR3.fmu"
EIR_LIB = FMU_ROOT / "Cali_EIR_BSU_CH1" / "ChillerData" / "ChillerData_v3.xlsx"
RAW_CH01 = FMU_ROOT / "CT_Model" / "DATA" / "Site_A" / "chiller" / "CH_01.csv"

HAS_FMPY = importlib.util.find_spec("fmpy") is not None


def _canonical(start=20000, stop=32000) -> pd.DataFrame:
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
                    reason="chiller FMU / library / Site A data not available")
def test_full_pipeline_drives_chiller_fmu_branch(tmp_path):
    # Lay down a minimal run (canonical + gate), bypassing the slow ingest/attribute.
    base = tmp_path / "runs" / "smoke"
    (base / "chiller" / "CH_01").mkdir(parents=True)
    _canonical().to_csv(base / "chiller" / "CH_01" / "canonical.csv", index=False)
    (base / "attribute").mkdir(parents=True)
    pd.DataFrame([{"device_id": "CH_01", "level": "full_physical", "reason": ""}]).to_csv(
        base / "attribute" / "modelability_report.csv", index=False)

    config = {
        "_root": REPO, "outputs_dir": str(tmp_path),
        "fmu_config_dir": str(REPO / "configs" / "fmu"),
        "devices": [{"id": "CH_01", "type": "chiller"}],
        "thresholds": {"run_on": 0.5, "validation_fold": 3, "min_full_physical_rows": 150,
                       "subset_rows": 200, "max_refine_nfev": 6, "screening_max_rows": 8},
    }
    calibrate(config, "smoke")
    validate(config, "smoke")

    sel = pd.read_csv(base / "calibrate" / "selected_models.csv").set_index("device_id")
    assert sel.loc["CH_01", "status"] == "ok"                       # FMU branch ran
    assert sel.loc["CH_01", "selected_candidate"] in ("EIR", "EEIR")
    assert sel.loc["CH_01", "execution_engine"] == "fmpy_fmu"
    assert len(str(sel.loc["CH_01", "fmu_sha256"])) == 64

    val = pd.read_csv(base / "validate" / "full_period_metrics.csv").set_index("device_id")
    assert str(val.loc["CH_01", "status"]) == "ok"
    assert np.isfinite(float(val.loc["CH_01", "CVRMSE_pct"]))
    assert val.loc["CH_01", "execution_engine"] == "fmpy_fmu"
    assert val.loc["CH_01", "fmu_sha256"] == sel.loc["CH_01", "fmu_sha256"]
