from __future__ import annotations

from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[1]


def test_yorkcalc_candidate_uses_27_parameter_fmu_contract():
    cfg = yaml.safe_load((REPO / "configs" / "fmu" / "cooling_tower.yaml").read_text(encoding="utf-8"))
    york = next(c for c in cfg["candidates"] if c["name"] == "YorkCalc")

    # The closed-loop wrapper, not the original SiteACTYork27.fmu: that one read
    # the cooling range from the measured table, so its predicted outlet was a
    # function of the measured outlet (docs/REVISION_ENERGY_01_AUDIT.md B).
    assert york["fmu"].endswith("SiteACTYork27ClosedLoop.fmu")
    assert "FRWat0" in york["static_start_values"], (
        "MBL water-flow-ratio normalisation must be settable")
    assert york["table_param"] == "table_path"
    # the declarative grid cannot fit 27 coupled coefficients under a
    # closed-loop objective; york27_fit does
    assert york["fit_strategy"] == "closed_loop_simulation_error"

    thermal = york["tunable_parameters"]["thermal"]
    assert thermal == [f"f[{i}]" for i in range(1, 28)]
    assert york["fan_power"]["curve_prefix"] == "fanRelPow_r_P"
    assert york["fan_power"]["nominal_key"] == "PFan_nominal"

    # the grid is retained in the config as documentation of the old approach,
    # but the strategy above is what runs
    f_axes = [axis for axis in york["grid"]["axes"] if axis["name"].startswith("f")]
    assert len(f_axes) == 27
    assert {next(iter(axis["apply"].keys())) for axis in f_axes} == set(thermal)


def test_yorkcalc27_system_component_is_not_table_driven():
    component = REPO / "modelica" / "components" / "cooling_tower" / "YorkCalc27Component.mo"
    text = component.read_text(encoding="utf-8")

    assert "model YorkCalc27Component" in text
    assert "parameter Real f[27]" in text
    assert "CombiTimeTable" not in text
    assert "RealInput Tin_C" in text
    assert "RealInput Twb_C" in text
    assert "RealInput m_flow_kg_s" in text
    assert "RealInput y" in text
    assert "RealOutput TOut_C" in text
    assert "RealOutput Q_flow_W" in text
    assert "RealOutput PFan_W" in text


def test_closed_loop_wrapper_does_not_consume_the_measured_range():
    """The prediction path must not touch any measured downstream quantity."""
    wrapper = REPO / "modelica" / "wrappers" / "SiteACTYork27ClosedLoop.mo"
    text = wrapper.read_text(encoding="utf-8")

    # the model's own range closes the loop
    assert "TRan_s_C = Tin_C - (Twb_C + TApp_s)" in text
    # the measured range and approach are still read, but only for *_m outputs
    assert "TRan_meas_C    = tab.y[4]" in text
    assert "TApp_m = TAppAct_C" in text
    # and never inside the correlation
    correlation = text.split("TApp_forced_C = ")[1].split(";")[0]
    assert "TRan_s_C" in correlation
    assert "TRan_meas_C" not in correlation
