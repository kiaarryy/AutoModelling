"""Guarded regression for the real FMU path. Skips when the external GS FMU is
not present (it lives read-only under FMU_Modelica, outside the repo)."""
from pathlib import Path
import importlib.util
import os

import pytest

FMU_ROOT = Path(os.environ.get("AUTOFMU_FMU_ROOT", "__missing_external_fmu_root__"))
FMU = FMU_ROOT / "Cali_EIR_BSU_CH1" / "Modelica" / "ChillerCalibration_FMU_chiller_0EIR3.fmu"
TABLE = FMU_ROOT / "Cali_EIR_BSU_CH1" / "Modelica" / "ChillerData.txt"

HAS_FMPY = importlib.util.find_spec("fmpy") is not None
pytestmark = pytest.mark.skipif(not HAS_FMPY or not (FMU.exists() and TABLE.exists()), reason="FMPy or GS FMU/table not available")


def test_chiller_eir_fmu_power_within_band():
    from autofmu.fmu.runner import run_fmu
    from autofmu.metrics import regression_metrics

    frame = run_fmu(
        FMU,
        start_values={"VSD2.fileName": str(TABLE).replace("\\", "/")},
        output=["P_s", "P_m"],
        output_interval=1800,
    )
    m = regression_metrics(frame["P_m"], frame["P_s"])
    assert m["N"] > 500
    assert m["CVRMSE_pct"] < 8.0  # validated GS chiller stays within paper-range
