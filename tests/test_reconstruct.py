import numpy as np
import pandas as pd

from autofmu.modelability.reconstruct import (
    apply_reconstructions,
    apportion_total,
    energy_balance_load,
    power_from_cop,
)


def test_energy_balance_load():
    frame = pd.DataFrame({"flow": [360.0], "th": [13.0], "tc": [8.0]})  # 0.1 m3/s, dT=5
    q = energy_balance_load(frame, "flow", "th", "tc")
    # rho*cp*0.1*5 = 1000*4186*0.1*5 = 2,093,000 W
    assert round(float(q.iloc[0]), 0) == 2093000.0


def test_power_from_cop():
    p = power_from_cop(pd.Series([7610.0]), 7.61)
    assert round(float(p.iloc[0]), 1) == 1000.0


def test_apportion_total_by_share():
    total = pd.Series([300.0, 100.0])
    shares = {"A": pd.Series([2.0, 0.0]), "B": pd.Series([1.0, 1.0])}
    out = apportion_total(total, shares)
    assert round(float(out["A"].iloc[0]), 1) == 200.0  # 300*2/3
    assert round(float(out["B"].iloc[0]), 1) == 100.0
    assert round(float(out["B"].iloc[1]), 1) == 100.0  # only B running


def test_apply_reconstructions_writes_recon_column():
    frame = pd.DataFrame({
        "chw_flow_m3_h": [360.0],
        "tchwr_C": [13.0],
        "tchws_C": [8.0],
        "power_W": [0.0],  # dead sensor
    })
    specs = {
        "cooling_load_W": {"method": "chw_energy_balance", "flow": "chw_flow_m3_h", "t_hot": "tchwr_C", "t_cold": "tchws_C"},
        "power_W": {"method": "cop_estimate", "from": "cooling_load_W", "cop": 7.61},
    }
    out, flags = apply_reconstructions(frame, specs)
    assert "cooling_load_W" in out
    assert "power_W_recon" in out          # measured power_W is NOT overwritten
    assert float(out["power_W"].iloc[0]) == 0.0
    assert any("cop_estimate" in f for f in flags)
