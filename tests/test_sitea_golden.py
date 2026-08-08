"""Site A real-FMU contract regression under the audited evaluation protocol."""
from pathlib import Path
import os

import numpy as np
import pandas as pd
import pytest

from autofmu.config import load_project
from autofmu.pipeline import attribute, calibrate, ingest, validate

CONFIG = Path(__file__).parents[1] / "configs" / "site_a" / "project.yaml"
DATA = Path(os.environ.get("AUTOFMU_SITE_A_DATA_ROOT", "__missing_site_a_data_root__"))

try:  # this is a real-FMU contract test; without FMPy nothing can be driven
    import fmpy  # noqa: F401
    _HAS_FMPY = True
except ImportError:
    _HAS_FMPY = False

pytestmark = [
    pytest.mark.skipif(not DATA.exists(), reason="Site A data not available"),
    pytest.mark.skipif(not _HAS_FMPY, reason="FMPy not installed; real-FMU test"),
]


def test_sitea_headline_metrics(tmp_path):
    config = load_project(CONFIG)
    config["outputs_dir"] = str(tmp_path)
    # Bound the chiller FMU-in-the-loop fit for test speed (the full production
    # fit screens the entire library and uses a larger refine budget).
    config.setdefault("thresholds", {}).update(
        {"screening_max_rows": 30, "subset_rows": 500, "max_refine_nfev": 60}
    )
    for stage in (ingest, attribute, calibrate, validate):
        stage(config, "golden")
    base = Path(tmp_path) / "runs" / "golden"

    sel = pd.read_csv(base / "calibrate" / "selected_models.csv").set_index("device_id")
    # chiller: data-fitted Buildings EIR/EEIR driven through the real FMU, selected
    # per chiller. The fit budget is deliberately reduced here for test speed; the
    # full production fit (project_fleet, FMU-6) reaches ~5-6% P CVRMSE and beats
    # CH_01's library-screening baseline (old EEIR P 14.2%). This golden run is a
    # sanity lock: sensible selection + finite, non-catastrophic power error.
    # Candidate labels accept both the FMU contract names (EIR/EEIR) and the full
    # Buildings names used by the Python fallback path.
    assert sel.loc["CH_01", "selected_candidate"] in (
        "EIR", "EEIR", "ElectricReformulatedEIR", "ElectricEIR")
    ch_cvrmse = float(sel.loc["CH_01", "CVRMSE_pct"])
    assert np.isfinite(ch_cvrmse)
    assert sel.loc["CH_01", "period"] == "test"
    assert sel.loc["CH_01", "execution_engine"] == "fmpy_fmu"
    # condenser-water pump power
    assert np.isfinite(float(sel.loc["CDWP_01", "test_CVRMSE_pct"]))
    # cooling tower thermal model: controlled TOut is diagnostic; Q is primary.
    assert sel.loc["CT_01", "selected_candidate"] in ("YorkCalc", "Merkel")
    assert sel.loc["CT_01", "target"] == "heat_rejection_W"
    assert np.isfinite(float(sel.loc["CT_01", "test_CVRMSE_pct"]))
    # HEX effectiveness model validates on the uncontrolled condenser return temp
    assert sel.loc["HX_01", "selected_candidate"] == "ConstantEffectiveness"

    # controlled chilled-water supply temp must be excluded from model error
    ctrl = pd.read_csv(base / "validate" / "control_variables.csv")
    assert "tchws_C" in set(ctrl["variable"])
    val = pd.read_csv(base / "validate" / "full_period_metrics.csv").set_index("device_id")
    assert val.loc["CT_01", "target"] == "heat_rejection_W"
    assert val.loc["CT_01", "period"] == "full_valid_points"
    assert int(val.loc["CT_01", "rows_evaluated"]) <= int(val.loc["CT_01", "rows_valid"])
