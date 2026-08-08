"""FMU-0 acceptance: drive each device-type FMU through autofmu's unified
`run_device_fmu`, one Site A device per type, and assert finite outputs.

External FMUs + prepared Dymola tables live read-only under FMU_Modelica; tests
skip when they are absent. fmpy is required for the simulation tests.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autofmu.fmu.runner import run_device_fmu, _apply_scale, _check_finite, _build_input_array
from autofmu.fmu.inspect import validate_config_file
from autofmu.metrics import regression_metrics

HAS_FMPY = importlib.util.find_spec("fmpy") is not None

REPO = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO / "configs" / "fmu"
FMU_ROOT = Path(os.environ.get("AUTOFMU_FMU_ROOT", "__missing_external_fmu_root__"))

# --- external FMUs (read-only) ---
CHILLER_EIR_FMU = FMU_ROOT / "Cali_EIR_BSU_CH1" / "Modelica" / "ChillerCalibration_FMU_chiller_0EIR3.fmu"
CHILLER_TABLE = FMU_ROOT / "Cali_EIR_BSU_CH1" / "Modelica" / "ChillerData.txt"
CTM_FMU = FMU_ROOT / "Cali_EIR_BSU_CH1" / "Modelica" / "CTM_0FMU.fmu"
CTY_FMU = FMU_ROOT / "Cali_EIR_BSU_CH1" / "Modelica" / "CTY_0FMU.fmu"
CT_TABLE = FMU_ROOT / "outputs" / "cooling_tower" / "auto_model_site_a" / "tables" / "CT_01_calibration_window.txt"
HX_CONST_FMU = FMU_ROOT / "outputs" / "heat_exchanger" / "modelica_interface" / "fmu" / "SiteAHXConstantEffectiveness.fmu"
HX_PLATE_FMU = FMU_ROOT / "outputs" / "heat_exchanger" / "modelica_interface" / "fmu" / "SiteAHXPlateEffectivenessNTU.fmu"
HX_TABLE = FMU_ROOT / "outputs" / "heat_exchanger" / "calibration" / "HX_01_calibration_window.txt"
PUMP_FMU = FMU_ROOT / "outputs" / "pump" / "stage4_modelica_interface" / "fmu" / "PumpEmpiricalPower.fmu"
PUMP_PARAMS = FMU_ROOT / "outputs" / "pump" / "auto_model_site_a" / "best_pump_parameters.csv"
PUMP_TABLE = FMU_ROOT / "outputs" / "pump" / "site_a_tables" / "CDWP_03_window_table.csv"


def _dymola_rows(txt: Path) -> int:
    with txt.open() as f:
        f.readline()
        header = f.readline()
    inside = header[header.find("(") + 1 : header.find(")")]
    return int(inside.split(",")[0])


def _stop_from_table(txt: Path, step: float = 300.0) -> float:
    return (_dymola_rows(txt) - 1) * step


def _have(*paths: Path) -> bool:
    return all(p.exists() for p in paths)


# ----------------------------------------------------------------------------
# 1. Config contract matches the real FMU interface
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["chiller", "cooling_tower", "heat_exchanger", "pump"])
@pytest.mark.skipif(not HAS_FMPY, reason="FMPy not installed")
def test_fmu_config_matches_interface(name):
    rows = validate_config_file(CONFIG_DIR / f"{name}.yaml")
    assert rows, f"no candidates in {name}.yaml"
    present = [r for r in rows if r["status"] != "missing_fmu"]
    if not present:
        pytest.skip(f"no {name} FMUs available to validate")
    mismatched = [r for r in present if r["status"] != "ok"]
    assert not mismatched, f"{name} config/FMU mismatch: {mismatched}"


# ----------------------------------------------------------------------------
# 2. Smoke: drive each candidate FMU through run_device_fmu -> finite outputs
# ----------------------------------------------------------------------------
@pytest.mark.skipif(not HAS_FMPY or not _have(CHILLER_EIR_FMU, CHILLER_TABLE), reason="FMPy or chiller EIR FMU/table absent")
def test_chiller_eir_smoke():
    frame = run_device_fmu(
        CHILLER_EIR_FMU,
        table_overrides={"VSD2.fileName": CHILLER_TABLE},
        output=["P_m", "P_s", "Q_m", "Q_s"],
        output_interval=1800,
        finite_columns=["P_s", "Q_s"],
    )
    assert len(frame) > 500
    m = regression_metrics(frame["P_m"], frame["P_s"])
    assert m["CVRMSE_pct"] < 8.0  # validated GS chiller stays within paper range


@pytest.mark.skipif(not HAS_FMPY or not _have(CTM_FMU, CT_TABLE), reason="FMPy or CT Merkel FMU/table absent")
def test_ct_merkel_smoke():
    frame = run_device_fmu(
        CTM_FMU,
        table_overrides={"Tout1.fileName": CT_TABLE},
        output=["TOut_m", "TOut_s", "Q_m", "Q_s"],
        stop_time=_stop_from_table(CT_TABLE),
        output_interval=300,
        fmi_type="CoSimulation",
        finite_columns=["TOut_s", "Q_s"],
    )
    assert len(frame) > 100
    assert np.isfinite(frame["TOut_s"]).all()


@pytest.mark.skipif(not HAS_FMPY or not _have(CTY_FMU, CT_TABLE), reason="FMPy or CT YorkCalc FMU/table absent")
def test_ct_york_smoke():
    frame = run_device_fmu(
        CTY_FMU,
        table_overrides={"tableFileName": CT_TABLE},
        output=["TOut_m", "TOut_s", "Q_m", "Q_s"],
        stop_time=_stop_from_table(CT_TABLE),
        output_interval=300,
        fmi_type="CoSimulation",
        finite_columns=["TOut_s", "Q_s"],
    )
    assert len(frame) > 100
    assert np.isfinite(frame["TOut_s"]).all()


@pytest.mark.parametrize("fmu", [HX_CONST_FMU, HX_PLATE_FMU])
def test_hx_smoke(fmu):
    if not HAS_FMPY:
        pytest.skip("FMPy not installed")
    if not _have(fmu, HX_TABLE):
        pytest.skip("HX FMU/table absent")
    frame = run_device_fmu(
        fmu,
        table_overrides={"table_path": HX_TABLE},
        output=["T2Out_m", "T2Out_s", "Q_m", "Q_s"],
        stop_time=_stop_from_table(HX_TABLE),
        output_interval=300,
        finite_columns=["T2Out_s", "Q_s"],
    )
    assert len(frame) > 100
    assert np.isfinite(frame["T2Out_s"]).all()


def _pump_inputs(rows: int | None = None) -> pd.DataFrame:
    # Map prepared source columns -> FMU input port names (FMU-4 does this from
    # the config `input_map`; the smoke renames inline).
    table = pd.read_csv(PUMP_TABLE).rename(columns={"m_flow_kg_s": "m_flow_in", "y_used": "y_in"})
    return table.head(rows) if rows else table


def _pump_start_values() -> dict:
    params = pd.read_csv(PUMP_PARAMS)
    row = params[params["pump"] == "CDWP_03"].iloc[0]
    return {k: float(row[k]) for k in ["P_nominal", "m_flow_nominal", "y_min", "y_max", "c0", "c1", "c2", "c3", "c4"]}


@pytest.mark.skipif(not HAS_FMPY or not _have(PUMP_FMU, PUMP_PARAMS, PUMP_TABLE), reason="FMPy or pump FMU/params/table absent")
def test_pump_input_smoke():
    table = _pump_inputs(48)
    frame = run_device_fmu(
        PUMP_FMU,
        start_values=_pump_start_values(),
        inputs=table,
        input_columns=["m_flow_in", "y_in"],
        input_time_column="time_s",
        output=["P_s", "m_flow_s"],
        stop_time=float(table["time_s"].iloc[-1]),
        output_interval=300,
        finite_columns=["P_s"],
    )
    assert np.isfinite(frame["P_s"]).all()
    measured = table["P_meas_W"].to_numpy(float)[: len(frame)]
    m = regression_metrics(pd.Series(measured), frame["P_s"].iloc[: len(measured)])
    assert m["CVRMSE_pct"] < 15.0


@pytest.mark.skipif(not HAS_FMPY or not _have(PUMP_FMU, PUMP_PARAMS, PUMP_TABLE), reason="FMPy or pump FMU/params/table absent")
def test_pump_chunked_matches_whole():
    # The empirical pump-power model is a static map of (m_flow, y), so chunked
    # simulation must reproduce the whole-run outputs.
    table = _pump_inputs(48)
    kwargs = dict(
        start_values=_pump_start_values(),
        inputs=table,
        input_columns=["m_flow_in", "y_in"],
        input_time_column="time_s",
        output=["P_s"],
        output_interval=300,
    )
    whole = run_device_fmu(PUMP_FMU, stop_time=float(table["time_s"].iloc[-1]), **kwargs)
    chunked = run_device_fmu(PUMP_FMU, max_input_rows_per_chunk=12, **kwargs)
    n = min(len(whole), len(chunked))
    assert n > 0
    np.testing.assert_allclose(
        whole["P_s"].to_numpy()[:n], chunked["P_s"].to_numpy()[:n], rtol=1e-3, atol=1.0
    )


# ----------------------------------------------------------------------------
# 3. Honest failure: never silently degrade
# ----------------------------------------------------------------------------
@pytest.mark.skipif(not HAS_FMPY or not CHILLER_EIR_FMU.exists(), reason="FMPy or chiller FMU absent")
def test_missing_table_raises():
    with pytest.raises(FileNotFoundError):
        run_device_fmu(
            CHILLER_EIR_FMU,
            table_overrides={"VSD2.fileName": CHILLER_TABLE.parent / "does_not_exist.txt"},
            output=["P_s"],
        )


@pytest.mark.skipif(not HAS_FMPY, reason="FMPy not installed")
def test_missing_fmu_raises():
    with pytest.raises(FileNotFoundError):
        run_device_fmu(REPO / "no_such.fmu", output=["P_s"])


# ----------------------------------------------------------------------------
# 4. Pure-python unit tests for the runner guards (no FMU needed)
# ----------------------------------------------------------------------------
def test_check_finite_raises_on_nonfinite():
    frame = pd.DataFrame({"P_s": [1.0, np.nan, 3.0]})
    with pytest.raises(ValueError, match="non-finite"):
        _check_finite(frame, require_finite=True, finite_columns=["P_s"], output=None)


def test_check_finite_passes_when_clean():
    frame = pd.DataFrame({"P_s": [1.0, 2.0, 3.0]})
    _check_finite(frame, require_finite=True, finite_columns=["P_s"], output=None)


def test_apply_scale_multiplies_named_columns():
    frame = pd.DataFrame({"Q_s": [1.0, 2.0], "TOut_s": [10.0, 20.0]})
    out = _apply_scale(frame, {"Q_s": 2.0})
    assert list(out["Q_s"]) == [2.0, 4.0]
    assert list(out["TOut_s"]) == [10.0, 20.0]  # unscaled column untouched


def test_build_input_array_shapes_fields():
    df = pd.DataFrame({"time_s": [0.0, 300.0], "m_flow_in": [1.0, 2.0], "y_in": [0.5, 0.6]})
    arr = _build_input_array(df, ["m_flow_in", "y_in"], "time_s")
    assert arr.dtype.names == ("time", "m_flow_in", "y_in")
    assert list(arr["time"]) == [0.0, 300.0]
    assert list(arr["m_flow_in"]) == [1.0, 2.0]
