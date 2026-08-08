from types import SimpleNamespace

import pandas as pd

from autofmu.modelability.operating import operating_mask


def test_dead_run_signal_defaults_to_legacy_fallback():
    frame = pd.DataFrame({"run_signal": [0.0, 0.0], "load_pct": [0.0, 0.0]})
    profile = SimpleNamespace(run_signal="run_signal", liveness=("load_pct",))
    result = operating_mask(frame, profile)
    assert result.rows == 2
    assert "run_signal_dead" in result.flags


def test_site_d_can_treat_dead_run_signal_as_off():
    frame = pd.DataFrame({"run_signal": [0.0, 0.0], "load_pct": [0.0, 0.0]})
    profile = SimpleNamespace(run_signal="run_signal", liveness=("load_pct",))
    result = operating_mask(frame, profile, dead_run_policy="off")
    assert result.rows == 0
    assert "run_signal_dead" in result.flags
