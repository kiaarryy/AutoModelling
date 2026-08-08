"""Phase A: controlled variables must never be model-error targets (advisor Q2)."""
from autofmu.contracts.profiles import PROFILES, get_profile


def test_no_profile_target_is_controlled():
    # the calibration/validation target must be an uncontrolled output
    for name, profile in PROFILES.items():
        assert not profile.is_controlled(profile.target), \
            f"{name}: target {profile.target} is a controlled variable"


def test_validation_targets_are_uncontrolled():
    for name, profile in PROFILES.items():
        for v in profile.validation_targets:
            assert v not in profile.controlled, f"{name}: validation target {v} is controlled"


def test_chiller_and_hex_mark_tchws_controlled():
    assert "tchws_C" in PROFILES["chiller"].controlled
    assert "tchws_C" in PROFILES["heat_exchanger"].controlled
    # HEX target must be the uncontrolled condenser-side temperature
    assert PROFILES["heat_exchanger"].target == "tcwr_C"


def test_cooling_tower_supply_temp_controlled():
    assert "tcws_C" in PROFILES["cooling_tower"].controlled


def test_cooling_tower_controlled_temperature_is_not_primary_metric():
    profile = get_profile("cooling_tower")
    assert "tcws_1_C" in profile.controlled
    assert "tcws_1_C" not in profile.validation_targets
    assert profile.target == "heat_rejection_W"
    assert profile.validation_targets == ("heat_rejection_W", "power_W")
