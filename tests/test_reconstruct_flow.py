import numpy as np
import pandas as pd

from autofmu.modelability.reconstruct import (
    apply_reconstructions,
    power_energy_balance,
    total_source_flow,
)
from autofmu.pipeline.attribute import _run_signal_usable


def test_energy_balance_power_and_cop_verify_flag():
    n = 10
    frame = pd.DataFrame({
        "cooling_load_W": np.full(n, 600000.0),   # Q_evap
        "heat_rejection_W": np.full(n, 700000.0),  # Q_cond -> P = 100kW, COP = 6
    })
    spec = {"power_W": {"method": "energy_balance_power", "condenser": "heat_rejection_W",
                        "evaporator": "cooling_load_W", "verify_cop": [3.0, 9.0]}}
    out, flags = apply_reconstructions(frame, spec)
    assert "power_W_recon" in out
    assert abs(out["power_W_recon"].iloc[0] - 100000.0) < 1e-6
    assert "verify=pass" in flags[0]  # implied COP 6 is in band


def test_energy_balance_power_flags_fail_when_cop_implausible():
    n = 10
    frame = pd.DataFrame({
        "cooling_load_W": np.full(n, 600000.0),
        "heat_rejection_W": np.full(n, 1200000.0),  # P = 600kW -> COP = 1 (implausible)
    })
    spec = {"power_W": {"method": "energy_balance_power", "condenser": "heat_rejection_W",
                        "evaporator": "cooling_load_W", "verify_cop": [3.0, 9.0]}}
    _, flags = apply_reconstructions(frame, spec)
    assert "verify=FAIL" in flags[0]


def test_active_count_replaces_hardcoded_scale():
    frame = pd.DataFrame({"fan1_Hz": [40, 0, 35, 0], "fan2_Hz": [42, 38, 0, 0]})
    spec = {"fans_on_count": {"method": "active_count", "signals": ["fan1_Hz", "fan2_Hz"], "threshold": 5.0}}
    out, flags = apply_reconstructions(frame, spec)
    assert list(out["fans_on_count"]) == [2, 1, 1, 0]   # logical per-row count, not x2
    assert "active_count" in flags[0]


def test_total_source_flow_sums_aligned():
    ts = pd.date_range("2022-03-01", periods=4, freq="5min").astype(str)
    a = pd.DataFrame({"timestamp": ts, "cw_flow_m3_h": [100, 0, 200, 0]})
    b = pd.DataFrame({"timestamp": ts, "cw_flow_m3_h": [50, 50, 0, 0]})
    total = total_source_flow({"A": a, "B": b}, "cw_flow_m3_h")
    assert list(total.to_numpy()) == [150, 50, 200, 0]


def test_flow_attribution_requires_usable_target_run_signal():
    assert not _run_signal_usable(pd.DataFrame({"timestamp": ["2024-01-01"]}))
    assert not _run_signal_usable(pd.DataFrame({"run_signal": [np.nan, np.nan]}))
    assert _run_signal_usable(pd.DataFrame({"run_signal": [0.0, 1.0]}))
