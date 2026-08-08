import numpy as np
import pandas as pd

from autofmu.contracts.profiles import get_profile
from autofmu.devices.calibrate import calibrate_power_model

THRESHOLDS = {"run_on": 0.5, "min_calibration_rows": 50, "pump_nominal_hz": 50.0, "fan_nominal_hz": 50.0}


def test_pump_affinity_recovers_cubic():
    n = 400
    rng = np.random.default_rng(0)
    f = rng.uniform(25, 50, n)
    w = f / 50.0
    power = 130000.0 * w ** 3  # true cubic, Pnom=130kW
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2022-03-01", periods=n, freq="5min").astype(str),
        "power_W": power,
        "speed_Hz": f,
        "run_signal": np.ones(n),
    })
    rows, best, ts = calibrate_power_model("CHWP01", frame, get_profile("pump"), THRESHOLDS)
    assert best["candidate"] in ("affinity_power", "speed_poly_power")
    assert best["MAPE_pct"] < 1.0  # near-perfect on noise-free cubic
    assert not ts.empty


def test_cooling_tower_two_fan_affinity():
    n = 400
    rng = np.random.default_rng(1)
    f1 = rng.uniform(0, 50, n)
    f2 = rng.uniform(0, 50, n)
    w1, w2 = f1 / 50.0, f2 / 50.0
    power = 55000.0 * (w1 ** 3 + w2 ** 3)
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2022-03-01", periods=n, freq="5min").astype(str),
        "power_W": power,
        "fan1_Hz": f1,
        "fan2_Hz": f2,
        "run_signal": np.ones(n),
    })
    rows, best, ts = calibrate_power_model("CT01", frame, get_profile("cooling_tower"), THRESHOLDS)
    assert best["MAPE_pct"] < 1.0
