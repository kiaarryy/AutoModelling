"""Tests for the cleaning-attrition ledger.

The ledger exists so two claims can be made honestly: that the waterfall adds
up, and that exclusions conditioned on the prediction target are separated from
operating-state gates rather than mixed into one mask.
"""
from __future__ import annotations

import numpy as np
import pytest

from autofmu.observability import AttritionLedger


@pytest.fixture
def ledger():
    return AttritionLedger(n_total=1000, device="CT_01", family="cooling_tower")


def test_waterfall_adds_up(ledger):
    ledger.gate("a", np.arange(1000) >= 100)      # drops 100
    ledger.gate("b", np.arange(1000) < 900)       # drops 100 more
    frame = ledger.to_frame()
    assert frame.iloc[0].remaining == 1000
    assert list(frame.removed)[1:] == [100, 100]
    assert frame.iloc[-1].remaining == 800
    # every step's remaining equals the previous remaining minus its removals
    remaining = list(frame.remaining)
    removed = list(frame.removed)
    for i in range(1, len(frame)):
        assert remaining[i] == remaining[i - 1] - removed[i]


def test_overlapping_criteria_are_not_double_counted(ledger):
    same = np.arange(1000) >= 500
    ledger.gate("first", same)
    ledger.gate("second", same)       # already excluded; must cost nothing
    assert [s.removed for s in ledger.steps] == [500, 0]
    assert ledger.remaining == 500


def test_quarantine_is_counted_separately_from_gates(ledger):
    ledger.gate("fans running", np.arange(1000) >= 200)
    impossible = np.zeros(1000, bool)
    impossible[300:340] = True                     # 40 rows, all still alive
    ledger.quarantine("approach <= 0", impossible)
    summary = ledger.summary()
    assert summary["removed_by_gate"] == 200
    assert summary["removed_by_quarantine"] == 40
    assert summary["quarantine_approach <= 0"] == 40
    assert summary["n_retained"] == 760


def test_quarantine_only_counts_rows_still_alive(ledger):
    ledger.gate("fans running", np.arange(1000) >= 500)
    impossible = np.zeros(1000, bool)
    impossible[:600] = True            # 500 of these were already gated out
    ledger.quarantine("impossible", impossible)
    assert ledger.quarantine_summary()["impossible"] == 100
    assert ledger.remaining == 400


def test_mask_shape_is_checked(ledger):
    with pytest.raises(ValueError):
        ledger.gate("wrong length", np.ones(10, bool))


def test_retained_percentage_is_reported(ledger):
    ledger.gate("half", np.arange(1000) < 500)
    assert ledger.summary()["retained_pct"] == pytest.approx(50.0)
    assert ledger.to_frame().iloc[-1].retained_pct == pytest.approx(50.0)


def test_empty_ledger_retains_everything():
    empty = AttritionLedger(n_total=42)
    assert empty.remaining == 42
    assert empty.summary()["removed_by_gate"] == 0
    assert len(empty.to_frame()) == 1


def test_cooling_tower_features_expose_the_ledger():
    """The cooling-tower feature builder must return an auditable ledger."""
    import pandas as pd
    from autofmu.devices.cooling_tower_thermal import _features

    n = 500
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "tcwr_C": 30 + rng.normal(0, 0.5, n),
        "tcws_1_C": 26 + rng.normal(0, 0.5, n),
        "twb_C": 22 + rng.normal(0, 0.5, n),
        "attributed_flow_m3_h": np.full(n, 300.0),
        "fans_on_count": np.full(n, 2.0),
        "fan1_Hz": np.full(n, 40.0),
        "fan2_Hz": np.full(n, 40.0),
    })
    # inject rows that are physically impossible rather than merely off
    frame.loc[:9, "tcws_1_C"] = 35.0    # leaving hotter than entering
    frame.loc[10:19, "tcws_1_C"] = 20.0  # leaving below wet bulb

    out = _features(frame, {})
    ledger = out["ledger"]
    quarantined = ledger.quarantine_summary()
    assert sum(quarantined.values()) == 20
    assert ledger.remaining == n - 20
    assert out["valid"].sum() == ledger.remaining


# --- flow attribution grades ------------------------------------------------

def test_paired_attribution_keeps_rows_solo_masking_would_discard():
    """Explicit pairing must not be gated on the target running alone."""
    import pandas as pd
    from autofmu.modelability.reconstruct import total_source_flow

    ts = pd.date_range("2024-05-01", periods=100, freq="5min")
    stamps = ts.strftime("%Y-%m-%dT%H:%M:%SZ")
    chiller = pd.DataFrame({"timestamp": stamps, "cw_flow_m3_h": np.full(100, 120.0)})
    hx = pd.DataFrame({"timestamp": stamps, "cw_flow_m3_h": np.full(100, 80.0)})
    total = total_source_flow({"CH_01": chiller, "HX_01": hx}, "cw_flow_m3_h")
    assert len(total) == 100
    assert np.allclose(total.to_numpy(), 200.0)


def test_single_vsd_tower_is_not_silently_gated_out():
    """A tower exposing one drive instead of two must still produce features.

    Site A CT_06 and CT_07 carry a single VSD. Summing fan1_Hz and fan2_Hz made
    the absent channel poison the sum with NaN, and both towers were gated out
    of the entire record before any model was tried.
    """
    import pandas as pd
    from autofmu.devices.cooling_tower_thermal import _features

    n = 400
    common = dict(
        tcwr_C=np.full(n, 30.0), tcws_1_C=np.full(n, 26.0),
        twb_C=np.full(n, 22.0), attributed_flow_m3_h=np.full(n, 300.0))

    one_drive = _features(pd.DataFrame(
        {**common, "fan1_Hz": np.full(n, 40.0), "fans_on_count": np.full(n, 2.0)}), {})
    two_drives = _features(pd.DataFrame(
        {**common, "fan1_Hz": np.full(n, 40.0), "fan2_Hz": np.full(n, 40.0),
         "fans_on_count": np.full(n, 2.0)}), {})

    assert one_drive["valid"].sum() == n
    # both cells at 40 Hz is 0.8 of nominal whether that is one drive or two
    assert np.allclose(one_drive["fr_air"], 0.8)
    assert np.allclose(two_drives["fr_air"], 0.8)


def test_one_of_two_fans_running_reports_that_fan_speed():
    """Speed is the mean over *running* drives, not over all channels."""
    import pandas as pd
    from autofmu.devices.cooling_tower_thermal import _features

    n = 100
    out = _features(pd.DataFrame({
        "tcwr_C": np.full(n, 30.0), "tcws_1_C": np.full(n, 26.0),
        "twb_C": np.full(n, 22.0), "attributed_flow_m3_h": np.full(n, 300.0),
        "fan1_Hz": np.full(n, 40.0), "fan2_Hz": np.zeros(n),
        "fans_on_count": np.ones(n)}), {})
    assert np.allclose(out["fr_air"], 0.8)


# --- synthetic target blocks calibration ------------------------------------

def test_gate_blocks_a_device_whose_target_is_synthetic():
    """The HKUST pattern must never reach calibration."""
    import pandas as pd
    from autofmu.contracts.profiles import get_profile
    from autofmu.modelability.gating import gate_device

    n = 3000
    rng = np.random.default_rng(0)
    speed = rng.uniform(20, 50, n)
    frame = pd.DataFrame({
        "power_W": 1293.2 * speed,          # exact multiple of the drive signal
        "speed_Hz": speed,
        "flow_m3_h": 15 * 1293.2 * speed / 1000.0,
        "run_signal": np.ones(n),
    })
    columns = {"power_W": {"source": "power_W"}, "speed_Hz": {"source": "speed_Hz"},
               "flow_m3_h": {"source": "flow_m3_h"},
               "run_signal": {"source": "run_signal"}}
    result = gate_device("PUMP_X", frame, get_profile("pump"),
                         {"min_full_physical_rows": 100},
                         adapter_columns=columns)
    assert result.level == "blocked"
    assert "synthetic point published as a measurement" in result.reason


def test_gate_check_can_be_disabled_for_noise_free_datasets():
    import pandas as pd
    from autofmu.contracts.profiles import get_profile
    from autofmu.modelability.gating import gate_device

    n = 3000
    speed = np.random.default_rng(1).uniform(20, 50, n)
    frame = pd.DataFrame({"power_W": 1293.2 * speed, "speed_Hz": speed,
                          "run_signal": np.ones(n)})
    columns = {"power_W": {"source": "power_W"}, "speed_Hz": {"source": "speed_Hz"},
               "run_signal": {"source": "run_signal"}}
    result = gate_device("PUMP_X", frame, get_profile("pump"),
                         {"min_full_physical_rows": 100,
                          "block_synthetic_targets": False},
                         adapter_columns=columns)
    assert result.level != "blocked"


def test_single_drive_tower_is_not_downgraded_by_the_contract():
    """A tower with one VSD must still reach full_physical.

    Site A CT_06 and CT_07 were gated to nominal_only because the capability
    contract listed fan2_Hz as required, though the models need only a speed and
    a running-cell count.
    """
    import pandas as pd
    from autofmu.contracts.profiles import get_profile
    from autofmu.modelability.gating import gate_device

    n = 2000
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "heat_rejection_W": 400_000 + rng.normal(0, 20_000, n),
        "power_W": 9_000 + rng.normal(0, 400, n),
        "fan1_Hz": np.full(n, 40.0),                      # single drive
        "tcwr_C": 30 + rng.normal(0, 0.5, n),
        "tcws_1_C": 26 + rng.normal(0, 0.5, n),
        "twb_C": 22 + rng.normal(0, 0.5, n),
        "attributed_flow_m3_h": np.full(n, 300.0),
        "fans_on_count": np.full(n, 2.0),
        "run_signal": np.ones(n),
    })
    result = gate_device("CT_SINGLE", frame, get_profile("cooling_tower"),
                         {"min_full_physical_rows": 500})
    assert result.level == "full_physical", result.reason
    assert "fan2_Hz" not in result.missing_full_fields


def test_gate_detects_synthetic_targets_on_a_canonical_frame():
    """The gate sees canonical frames, not raw historian columns.

    It used to be handed the adapter's column map, whose sources are raw names
    like ``PP2.power_consumption``, against a frame whose columns are already
    ``power_W``. Nothing matched, the detector reported zero channels, and every
    HKUST pump passed the gate and was scored at 0.35-1.1% -- the BMS formula
    being rediscovered. A silent no-op, so it needs a test that counts.
    """
    import pandas as pd
    from autofmu.contracts.profiles import get_profile
    from autofmu.modelability.gating import _synthetic_targets, gate_device

    n = 4000
    rng = np.random.default_rng(0)
    speed = rng.uniform(20, 50, n)
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="5min"),
        "power_W": 1293.2 * speed,        # stored multiple of the drive signal
        "speed_Hz": speed,
        "run_signal": np.ones(n),
    })
    assert _synthetic_targets(frame, get_profile("pump"), None) == {"power_W"}
    assert gate_device("P", frame, get_profile("pump"),
                       {"min_full_physical_rows": 500}).level == "blocked"


def test_status_bit_alone_does_not_define_running():
    """Rows claiming to run at zero drive speed must not mask an identity.

    HKUST PP2 sets operation_status on 6,148 samples where speed reads zero and
    power does not. Scored on the status bit alone the inlier fraction drops to
    0.45 and the exact relation on the genuinely running samples is hidden.
    """
    import pandas as pd
    from autofmu.contracts.profiles import get_profile
    from autofmu.modelability.gating import _synthetic_targets

    n = 4000
    rng = np.random.default_rng(1)
    speed = rng.uniform(20, 50, n)
    power = 1293.2 * speed
    speed[: n // 2] = 0.0             # half the record: status on, drive stopped
    frame = pd.DataFrame({
        "power_W": power, "speed_Hz": speed, "run_signal": np.ones(n)})
    assert _synthetic_targets(frame, get_profile("pump"), None) == {"power_W"}


def test_stuck_status_bit_is_caught_by_the_compressor_command():
    """A status bit that reads "on" all year must not define operating rows.

    LBNL CHL_STA_1 reads 1 for all 17,518 samples while the chiller draws a
    constant 1.94 kW standby load on 12,392 of them, compressor command at 0.9%.
    The gate counted every row as modellable and CHI1 scored 105% test CVRMSE
    against 11.7% for its honest-bit twin CHI2.
    """
    import pandas as pd
    from autofmu.contracts.profiles import get_profile
    from autofmu.modelability.operating import operating_mask

    n = 10_000
    idle = np.zeros(n, dtype=bool)
    idle[: 7_000] = True                       # 70% of the year: on paper only
    frame = pd.DataFrame({
        "power_W": np.where(idle, 1_940.0, 120_000.0),
        "load_pct": np.where(idle, 0.9, 65.0),  # standby command is not zero
        "run_signal": np.ones(n),               # stuck
    })
    result = operating_mask(frame, get_profile("chiller"))
    assert result.bit.sum() == n                # the bit claims everything
    assert result.rows == n - 7_000             # the compressor command does not
    assert result.channel == "load_pct"
    assert any(f.startswith("run_signal_uncorroborated:70pct") for f in result.flags), result.flags


def test_liveness_convicts_only_on_evidence():
    """A missing liveness value is not evidence of standby.

    Reconstructed channels are absent wherever attribution failed; dropping
    those rows here would confuse "could not be attributed" with "switched off",
    and they are already handled by the finite-row requirements downstream.
    """
    import pandas as pd
    from autofmu.contracts.profiles import get_profile
    from autofmu.modelability.operating import operating_mask

    n = 1_000
    load = np.full(n, 60.0)
    load[:200] = np.nan                        # unknown, not off
    load[200:400] = 0.0                        # known off
    frame = pd.DataFrame({"power_W": np.full(n, 100_000.0), "load_pct": load,
                          "run_signal": np.ones(n)})
    result = operating_mask(frame, get_profile("chiller"))
    assert result.rows == n - 200               # only the known-off rows go


def test_cooling_tower_keeps_its_low_load_rows():
    """Towers declare no liveness channel, and that is deliberate.

    A cell in service with its fan off is a state the models represent, and
    attributed flow is near zero on genuine low-load rows -- used as a liveness
    test it discarded 79% of Site A CT_05, whose fan turns throughout. This
    pins the empty declaration so it cannot be "helpfully" filled in.
    """
    import pandas as pd
    from autofmu.contracts.profiles import get_profile
    from autofmu.modelability.operating import operating_mask

    n = 1_000
    frame = pd.DataFrame({
        "heat_rejection_W": np.concatenate([np.full(800, 1_000.0), np.full(200, 400_000.0)]),
        "attributed_flow_m3_h": np.concatenate([np.full(800, 0.5), np.full(200, 300.0)]),
        "fan1_Hz": np.concatenate([np.zeros(800), np.full(200, 40.0)]),
        "run_signal": np.ones(n),
    })
    result = operating_mask(frame, get_profile("cooling_tower"))
    assert get_profile("cooling_tower").liveness == ()
    assert result.rows == n
    assert result.channel == ""


def test_a_chiller_cannot_out_cool_its_own_energy_input():
    """COP <= 1 is quarantined, not fitted.

    HKUST CH-08 reconstructs more cooling than the total energy entering the
    evaporator on 116 of its 350 usable rows, because its chilled-water flow is
    frozen at 70.1 m3/h while power swings 210-619 kW. No other chiller at any
    of the four sites produces a single such row.
    """
    import pandas as pd
    from autofmu.devices.chiller_fmu import build_alldata2

    n = 600
    rng = np.random.default_rng(3)
    flow = np.full(n, 70.1)                      # frozen meter
    d_chw = rng.uniform(4.0, 7.0, n)
    power = rng.uniform(210_000, 619_000, n)     # unrelated to the load
    frame = pd.DataFrame({
        "power_W": power,
        "tchws_C": 5.6 + np.zeros(n), "tchwr_C": 5.6 + d_chw,
        "tcws_C": np.full(n, 31.0), "tcwr_C": np.full(n, 34.6),
        "chw_flow_m3_h": flow, "cw_flow_m3_h": np.full(n, 84.4),
        "load_pct": np.full(n, 70.0), "run_signal": np.ones(n),
    })
    table = build_alldata2(frame)
    assert len(table) > 0
    assert bool((table["Q_evap_kW"] > table["P/kw"]).all())


def test_frozen_flow_meter_is_reported_not_hidden():
    """A channel stuck on one value is a design constant, not a measurement.

    The synthetic-channel detector cannot see this: it skips degenerate targets
    to avoid false positives, and a constant is exactly what it skips.
    """
    import pandas as pd
    from autofmu.observability import frozen_channels

    n = 1_000
    rng = np.random.default_rng(4)
    frame = pd.DataFrame({
        "chw_flow_m3_h": np.full(n, 70.1),                 # frozen
        "cw_flow_m3_h": rng.uniform(150.0, 220.0, n),      # a real meter
    })
    found = frozen_channels(frame, ["chw_flow_m3_h", "cw_flow_m3_h"],
                            np.ones(n, dtype=bool))
    assert [f.subject for f in found] == ["chw_flow_m3_h"]
    assert found[0].verdict == "flag"


def test_unattainable_cop_is_flagged_but_never_blocks():
    """Site A CH_01 implies COP 20.2 -- 124 kW of power against 2,400 kW of
    reconstructed cooling, with the condenser balance closing at 1.04. The
    power model fitted to measured power stays valid; the derived quantities do
    not. Flag, do not block."""
    import numpy as _np
    from autofmu.observability import cop_envelope

    n = 1_000
    load = _np.full(n, 2_399_700.0)
    power = _np.full(n, 124_400.0)
    impossible, findings = cop_envelope(load, power, _np.ones(n, dtype=bool))
    assert not impossible.any()
    assert [f.check for f in findings] == ["cop_above_envelope"]
    assert all(f.verdict == "flag" for f in findings)


def test_a_pump_that_never_changed_speed_is_reported_as_unidentified():
    """Site A CDWP_01 holds a speed ratio of 0.700-0.701 across 28,619 rows and
    reports 5.93% test CVRMSE. The score is real; the identification is not --
    over a span of 0.001 no smooth function is distinguishable from its own
    first-order expansion, so the cubic exponent is not evidenced.

    Reported, never blocking: an unexcited score is weak evidence, not a wrong
    number, and the distinction has to reach the reader.
    """
    import pandas as pd
    from autofmu.contracts.profiles import get_profile
    from autofmu.modelability.gating import gate_device
    from autofmu.observability import excitation

    n = 4_000
    rng = np.random.default_rng(5)
    fixed = np.full(n, 35.0) + rng.uniform(0.0, 0.05, n)      # 0.700-0.701 of 50 Hz
    swept = rng.uniform(17.0, 50.0, n)                         # a drive that moves

    stuck = pd.DataFrame({"speed_Hz": fixed})
    moving = pd.DataFrame({"speed_Hz": swept})
    on = np.ones(n, dtype=bool)
    assert not excitation(stuck, "speed_Hz", on).excited
    assert excitation(moving, "speed_Hz", on).excited

    frame = pd.DataFrame({
        "power_W": 1_000.0 + 4_000.0 * (fixed / 50.0) ** 3 + rng.normal(0, 60, n),
        "speed_Hz": fixed, "run_signal": np.ones(n),
    })
    result = gate_device("CDWP_01", frame, get_profile("pump"),
                         {"min_full_physical_rows": 500})
    assert result.level == "full_physical"          # still modelled, not blocked
    assert result.driver == "speed_Hz"
    assert result.driver_excitation < 0.10
    assert any(f.startswith("driver_not_excited") for f in result.flags), result.flags


def test_thin_attribution_does_not_discard_a_pump_record():
    """A pump is not unmodellable because a channel its model never reads is
    missing on most rows.

    Solo-window flow exists only while the pump is the only one of its group
    running, so it thins as the group grows: adding Site A's other eight pumps
    took CDWP_01 from 28,619 attributed rows to 2,431 of the same 29,549
    running samples, and requiring finite flow discarded 92% of the record.
    The selected candidates (affinity, speed-poly) carry no flow term at all.
    """
    import pandas as pd
    from autofmu.devices.pump_fmu import FLOW_COVERAGE_MIN, build_pump_table

    n = 10_000
    rng = np.random.default_rng(11)
    speed = rng.uniform(20.0, 45.0, n)
    flow = np.full(n, np.nan)
    flow[: 800] = rng.uniform(200.0, 400.0, 800)      # 8% attributed
    frame = pd.DataFrame({
        "power_W": 5_000.0 + 40_000.0 * (speed / 50.0) ** 3,
        "speed_Hz": speed,
        "attributed_flow_m3_h": flow,
        "run_signal": np.ones(n),
    })
    table = build_pump_table(frame, {})
    assert len(table) == n                        # the record survives
    assert not bool(table["flow_observed"].any())  # and the flow term is gone

    # with real coverage the flow term stays and only attributed rows are used
    flow[:] = rng.uniform(200.0, 400.0, n)
    frame["attributed_flow_m3_h"] = flow
    covered = build_pump_table(frame, {})
    assert len(covered) == n
    assert float(covered["flow_observed"].mean()) >= FLOW_COVERAGE_MIN
