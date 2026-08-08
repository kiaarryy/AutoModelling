"""Tests for the synthetic-channel detector.

The detector's whole value is that it separates two populations -- genuinely
correlated instrumentation from channels that are stored multiples of one
another -- so the tests pin down both sides of that boundary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autofmu.observability import (
    EvidenceTier, analyse_device, detect_dependencies, profile_channel,
)
from autofmu.observability.synthetic_channels import SYNTHETIC_RESIDUAL_PCT


@pytest.fixture
def speed():
    rng = np.random.default_rng(0)
    return np.clip(rng.uniform(0.2, 1.0, 5000), 0, 1) * 100.0


def test_profile_channel_flags_dead_and_all_zero():
    n = 500
    dead = profile_channel("x", "src", np.full(n, 7.0), n)
    zero = profile_channel("x", "src", np.zeros(n), n)
    live = profile_channel("x", "src", np.linspace(0, 1, n), n)
    assert dead.dead and not dead.all_zero
    assert zero.dead and zero.all_zero
    assert not live.dead and not live.all_zero
    assert live.coverage_pct == pytest.approx(100.0)


def test_detects_a_nameplate_scaled_channel(speed):
    """The HKUST pattern: power and flow are the speed signal times a constant."""
    series = {"speed_Hz": speed,
              "power_W": 64.66 * speed,
              "flow_m3_h": 15 * 64.66 * speed}
    findings = detect_dependencies(series)
    synthetic = {f.target for f in findings if f.synthetic}
    assert {"power_W", "flow_m3_h"} <= synthetic
    prop = [f for f in findings if f.target == "power_W" and f.synthetic][0]
    assert prop.form == "proportional"
    assert prop.max_residual_pct_of_range < SYNTHETIC_RESIDUAL_PCT


def test_does_not_flag_a_noisy_affinity_law(speed):
    """Real instrumentation: power follows y^3 with measurement noise."""
    rng = np.random.default_rng(1)
    y = speed / 100.0
    power = 50_000 * y ** 3 + rng.normal(0, 300, speed.size)
    findings = detect_dependencies({"speed_Hz": speed, "power_W": power})
    assert not any(f.synthetic for f in findings)


def test_flags_an_exact_cubic_but_names_the_right_form(speed):
    y = speed / 100.0
    series = {"speed_Hz": speed, "power_W": 1000 + 5.0 * y + 50_000 * y ** 3}
    findings = [f for f in detect_dependencies(series) if f.target == "power_W"]
    assert findings and findings[0].synthetic
    assert findings[0].form == "cubic"


def test_duplicate_mapping_is_reported_separately(speed):
    findings = detect_dependencies({"speed_Hz": speed, "run_signal": speed.copy()})
    assert any(f.duplicate for f in findings)


def test_a_few_inconsistent_rows_do_not_hide_an_identity(speed):
    """Seven rows of status/value disagreement must not mask the relation."""
    power = 64.66 * speed
    power[:7] = 0.0                      # flagged running, zero power
    findings = [f for f in detect_dependencies(
        {"speed_Hz": speed, "power_W": power}) if f.target == "power_W"]
    assert findings[0].synthetic
    assert findings[0].n_outlier_rows == 7
    assert findings[0].max_residual_pct_of_range > SYNTHETIC_RESIDUAL_PCT


def test_independent_signal_count_collapses_a_dependent_group(speed):
    frame = pd.DataFrame({
        "spd": speed,
        "pwr": 64.66 * speed,
        "flw": 15 * 64.66 * speed,
        "run": (speed > 0).astype(float),
        "tin": 25 + 0.01 * np.arange(speed.size) % 3,
    })
    result = analyse_device(
        frame,
        {"speed_Hz": {"source": "spd"}, "power_W": {"source": "pwr"},
         "flow_m3_h": {"source": "flw"}, "run_signal": {"source": "run"},
         "tcwr_C": {"source": "tin"}},
        site="test", device="pump_x", equipment_type="pump")
    # speed/power/flow collapse to one; run_signal is constant-on -> dead
    assert len(result.independent_roles) < len(result.channels)
    assert {"power_W", "flow_m3_h"} & result.synthetic_roles


def test_synthetic_tier_ranks_below_nameplate():
    order = list(EvidenceTier)
    assert order.index(EvidenceTier.TX_SYNTHETIC) > order.index(EvidenceTier.T4_NAMEPLATE)
