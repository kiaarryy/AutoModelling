"""Interval-scale quantities must not be given normalised error metrics.

Reviewer 1 comment 6, fix M-12: the manuscript reports temperature CVRMSE and
applies an ASHRAE Guideline 14 threshold to it. Dividing by the mean of a
Celsius series presumes a meaningful zero, which Celsius does not have.
"""
from __future__ import annotations

import numpy as np
import pytest

from autofmu.metrics import gl14_pass, regression_metrics


@pytest.fixture
def outlet_series():
    rng = np.random.default_rng(0)
    measured_c = 26.0 + rng.normal(0, 1.5, 4000)
    simulated_c = measured_c + rng.normal(0, 0.4, 4000)
    return measured_c, simulated_c


def test_temperature_metrics_are_absolute_only(outlet_series):
    measured, simulated = outlet_series
    out = regression_metrics(measured, simulated, quantity="temperature")
    for suppressed in ("CVRMSE_pct", "NMBE_pct", "MAPE_pct", "criterion_pass"):
        assert suppressed not in out
    for kept in ("RMSE", "MAE", "MBE", "R2"):
        assert kept in out
    assert out["criterion"] == "absolute_only_interval_scale"


def test_temperature_rmse_is_unit_shift_invariant_but_cvrmse_is_not(outlet_series):
    """The reason normalised metrics are dropped, stated as an assertion."""
    measured_c, simulated_c = outlet_series
    celsius = regression_metrics(measured_c, simulated_c, quantity="temperature")
    kelvin = regression_metrics(measured_c + 273.15, simulated_c + 273.15,
                                quantity="temperature")
    assert kelvin["RMSE"] == pytest.approx(celsius["RMSE"])

    # the same physical error, expressed as CVRMSE, changes by an order of
    # magnitude purely from the choice of unit
    as_generic_c = regression_metrics(measured_c, simulated_c)
    as_generic_k = regression_metrics(measured_c + 273.15, simulated_c + 273.15)
    assert as_generic_c["CVRMSE_pct"] > 10 * as_generic_k["CVRMSE_pct"]


def test_energy_quantities_keep_the_normalised_metrics_and_criterion():
    rng = np.random.default_rng(1)
    measured = 500_000 + rng.normal(0, 40_000, 3000)
    simulated = measured + rng.normal(0, 20_000, 3000)
    out = regression_metrics(measured, simulated, quantity="heat_flow")
    assert "CVRMSE_pct" in out and "NMBE_pct" in out
    assert out["criterion"] == "raw_interval_custom"
    assert out["criterion_pass"] == gl14_pass(out["CVRMSE_pct"], out["NMBE_pct"])


def test_default_quantity_preserves_existing_behaviour():
    rng = np.random.default_rng(2)
    measured = 100 + rng.normal(0, 5, 500)
    out = regression_metrics(measured, measured + 1.0)
    assert out["quantity"] == "generic"
    assert "CVRMSE_pct" in out and "criterion_pass" in out


# --- selection scores follow the manuscript's stated rules ------------------

def test_cooling_tower_score_ranks_on_absolute_outlet_error():
    """Section 3.3 ranks towers on leaving-water RMSE, not on its CVRMSE."""
    import pandas as pd
    from autofmu.devices.cooling_tower_fmu import _score

    n = 600
    rng = np.random.default_rng(3)
    base = pd.DataFrame({
        "TOut_m": 299.15 + rng.normal(0, 1.0, n),
        "Q_m": 500_000 + rng.normal(0, 20_000, n),
        "P_m": 9_000 + rng.normal(0, 300, n),
    })
    fans = np.ones(n)

    def build(t_err, q_rel):
        frame = base.copy()
        frame["TOut_s"] = frame["TOut_m"] + t_err
        frame["Q_s"] = frame["Q_m"] * (1 + q_rel)
        frame["P_s"] = frame["P_m"]
        return _score(frame, fans)

    accurate_t = build(0.10, 0.05)
    accurate_q = build(0.60, 0.01)
    assert accurate_t["T_RMSE_K"] < accurate_q["T_RMSE_K"]
    # the tower with the better outlet temperature must win despite worse heat
    assert accurate_t["score"] < accurate_q["score"]
    assert "T_RMSE_K" in accurate_t and "T_MBE_K" in accurate_t


def test_heat_exchanger_score_is_decided_by_heat_with_a_temperature_gate():
    import pandas as pd
    from autofmu.devices.heat_exchanger_fmu import T_OUT_ADMISSIBLE_K, _score

    n = 400
    rng = np.random.default_rng(4)
    base = pd.DataFrame({
        "T1Out_m": 8.0 + rng.normal(0, 0.2, n),
        "T2Out_m": 30.0 + rng.normal(0, 0.2, n),
        "Q_m": 800_000 + rng.normal(0, 20_000, n),
    })

    def build(t2_err, q_rel):
        frame = base.copy()
        frame["T1Out_s"] = frame["T1Out_m"]
        frame["T2Out_s"] = frame["T2Out_m"] + t2_err
        frame["Q_s"] = frame["Q_m"] * (1 + q_rel)
        return _score(frame)

    better_heat = build(0.2, 0.01)
    worse_heat = build(0.2, 0.10)
    assert better_heat["score"] < worse_heat["score"]

    # a candidate outside the admissibility band is pushed to the bottom even
    # when its heat flow is excellent
    inadmissible = build(T_OUT_ADMISSIBLE_K + 1.0, 0.001)
    assert not inadmissible["T2_admissible"]
    assert inadmissible["score"] > worse_heat["score"]


def test_skill_against_the_trivial_predictor():
    """CVRMSE cannot tell a good model from an easy signal.

    Site A CDWP_01 reports 5.93% test CVRMSE on a power signal whose own
    variability in that window is 3.85% -- predicting the mean would have been
    better. The skill ratio says so directly, and unlike CVRMSE it stays
    meaningful on an interval scale.
    """
    import numpy as np

    from autofmu.metrics import regression_metrics

    rng = np.random.default_rng(7)
    measured = 100.0 + rng.normal(0.0, 4.0, 5_000)

    # a model that tracks the signal
    good = regression_metrics(measured, measured + rng.normal(0.0, 1.0, 5_000))
    assert good["skill_vs_mean"] < 0.5

    # a constant predictor: exactly the baseline, by construction
    flat = regression_metrics(measured, np.full(5_000, float(np.mean(measured))))
    assert abs(flat["skill_vs_mean"] - 1.0) < 1e-9

    # worse than the baseline, which CVRMSE alone would still report as small
    poor = regression_metrics(measured, measured.mean() + rng.normal(0.0, 6.0, 5_000))
    assert poor["skill_vs_mean"] > 1.0
    assert poor["CVRMSE_pct"] < 10.0          # looks fine, is not

    # interval-scale quantities keep the skill ratio and lose the rest
    temps = regression_metrics(measured, measured + 0.5, quantity="temperature")
    assert "CVRMSE_pct" not in temps
    assert np.isfinite(temps["skill_vs_mean"])
