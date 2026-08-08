import numpy as np
import pandas as pd

from autofmu.contracts.profiles import get_profile
from autofmu.modelability.gating import gate_device
from autofmu.modelability.windows import solo_run_windows

THRESHOLDS = {"min_full_physical_rows": 5, "min_nominal_rows": 3, "run_on": 0.5}


def _chiller_frame(n=20, with_power=True):
    data = {
        "timestamp": pd.date_range("2022-03-01", periods=n, freq="5min").astype(str),
        "tchws_C": np.full(n, 8.0),
        "tchwr_C": np.full(n, 13.0),
        "tcwr_C": np.full(n, 30.0),
        "tcws_C": np.full(n, 25.0),
        "chw_flow_m3_h": np.full(n, 700.0),
        "cw_flow_m3_h": np.full(n, 900.0),
        "run_signal": np.ones(n),
    }
    if with_power:
        data["power_W"] = np.full(n, 500000.0)
    return pd.DataFrame(data)


def test_full_physical_when_all_present():
    r = gate_device("CHI01", _chiller_frame(), get_profile("chiller"), THRESHOLDS)
    assert r.level == "full_physical"


def test_nominal_only_when_target_missing():
    r = gate_device("CHI01", _chiller_frame(with_power=False), get_profile("chiller"), THRESHOLDS)
    assert r.level == "nominal_only"
    assert "power_W" in r.missing_full_fields


def test_heat_exchanger_no_flow_is_nominal_only():
    n = 20
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2022-03-01", periods=n, freq="5min").astype(str),
        "tchws_C": np.full(n, 13.0),
        "tchwr_C": np.full(n, 16.0),
        "tcws_C": np.full(n, 19.0),
        "tcwr_C": np.full(n, 14.0),
    })
    r = gate_device("HEX01", frame, get_profile("heat_exchanger"), THRESHOLDS)
    assert r.level == "nominal_only"
    assert "chw_flow_m3_h" in r.missing_full_fields


def test_chiller_missing_condenser_inputs_is_not_full_physical():
    frame = _chiller_frame().drop(columns=["cw_flow_m3_h", "tcws_C"])
    r = gate_device("CHI01", frame, get_profile("chiller"), THRESHOLDS)
    assert r.level != "full_physical"
    assert {"cw_flow_m3_h", "tcws_C"} <= set(r.missing_full_fields)


def test_cooling_tower_missing_fmu_drivers_is_not_full_physical():
    n = 20
    frame = pd.DataFrame({
        "power_W": np.full(n, 40000.0), "heat_rejection_W": np.full(n, 2e6),
        "fan1_Hz": np.full(n, 40.0), "run_signal": np.ones(n),
    })
    r = gate_device("CT01", frame, get_profile("cooling_tower"), THRESHOLDS)
    assert r.level != "full_physical"
    assert {"twb_C", "attributed_flow_m3_h", "fans_on_count"} <= set(r.missing_full_fields)


def test_heat_exchanger_missing_condenser_flow_is_not_full_physical():
    frame = _chiller_frame().drop(columns=["cw_flow_m3_h"]).rename(columns={"power_W": "unused_power"})
    r = gate_device("HX01", frame, get_profile("heat_exchanger"), THRESHOLDS)
    assert r.level != "full_physical"
    assert "cw_flow_m3_h" not in frame
    assert "cw_flow_m3_h" in r.missing_full_fields


def test_dead_target_and_run_signal_is_nominal_only():
    # Mirrors Tencent CHI01: flow/temps alive but power and run signal all-zero.
    n = 20
    frame = _chiller_frame(n)
    frame["power_W"] = np.zeros(n)
    frame["run_signal"] = np.zeros(n)
    r = gate_device("CHI01", frame, get_profile("chiller"), THRESHOLDS)
    assert r.level == "nominal_only"
    assert "target_sensor_dead" in r.flags
    assert "run_signal_dead" in r.flags


def test_solo_windows():
    sig = {
        "A": pd.Series([1, 1, 0, 1]),
        "B": pd.Series([0, 1, 0, 0]),
    }
    solo = solo_run_windows(sig)
    assert solo["A"] == 2  # rows 0 and 3
    assert solo["B"] == 0
