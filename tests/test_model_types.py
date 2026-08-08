"""Unit tests for the declarative model-type layer (autofmu.devices.model_types).

These pin the grid mini-DSL, the static start-value assembly and the
candidate-selection knob -- the pieces that let a new device *type* be added or
selected from config without touching the engine.
"""
from __future__ import annotations

from autofmu.devices import model_types as mt


def test_expand_grid_empty_is_one_empty_candidate():
    assert mt.expand_grid(None, {}) == [{}]
    assert mt.expand_grid({}, {}) == [{}]
    assert mt.expand_grid({"axes": []}, {}) == [{}]


def test_expand_grid_value_axis_and_constant():
    grid = {"axes": [{"name": "rat", "values": [0.8, 1.2],
                      "apply": {"merkel.ratWatAir_nominal": {"value": "axis"}}}]}
    out = mt.expand_grid(grid, {})
    assert out == [{"merkel.ratWatAir_nominal": 0.8}, {"merkel.ratWatAir_nominal": 1.2}]


def test_expand_grid_scale_of_base_with_floor():
    # YorkCalc-style: axis values are scale factors of an estimated nominal.
    grid = {"axes": [{"name": "tapp", "values": [0.25, 1.0],
                      "apply": {"TApp_nominal": {"base": "TApp_nominal", "scale": "axis", "floor": 0.2}}}]}
    out = mt.expand_grid(grid, {"TApp_nominal": 0.5})
    assert out[0]["TApp_nominal"] == 0.2          # 0.5*0.25=0.125 -> floored to 0.2
    assert out[1]["TApp_nominal"] == 0.5          # 0.5*1.0


def test_expand_grid_floor_and_ceil_clip():
    # ConstantEffectiveness ε = clip(eps_nominal*scale, 0.05, 0.98).
    grid = {"axes": [{"name": "eps", "values": [0.5, 1.25],
                      "apply": {"eps": {"base": "eps_nominal", "scale": "axis", "floor": 0.05, "ceil": 0.98}}}]}
    out = mt.expand_grid(grid, {"eps_nominal": 0.8})
    assert out[0]["eps"] == 0.4                    # 0.8*0.5
    assert out[1]["eps"] == 0.98                   # 0.8*1.25=1.0 -> ceil 0.98


def test_expand_grid_one_axis_drives_multiple_params():
    # Merkel-style: a single "scale" axis sets three cWatFra coefficients.
    grid = {"axes": [{"name": "cwat", "values": [1.0, 2.0],
                      "apply": {"a": {"default": 0.1, "scale": "axis"},
                                "b": {"default": -0.5, "scale": "axis"}}}]}
    out = mt.expand_grid(grid, {})
    assert out[0] == {"a": 0.1, "b": -0.5}
    assert out[1] == {"a": 0.2, "b": -1.0}


def test_expand_grid_is_cartesian_product():
    grid = {"axes": [
        {"name": "x", "values": [1, 2], "apply": {"px": {"value": "axis"}}},
        {"name": "y", "values": [10, 20], "apply": {"py": {"value": "axis"}}},
    ]}
    out = mt.expand_grid(grid, {})
    assert len(out) == 4
    assert {"px": 1.0, "py": 10.0} in out
    assert {"px": 2.0, "py": 20.0} in out


def test_expand_grid_coordinate_mode_scans_one_axis_at_a_time():
    grid = {"mode": "coordinate", "axes": [
        {"name": "f1", "values": [0.9, 1.0, 1.1], "apply": {"f[1]": {"default": 2.0, "scale": "axis"}}},
        {"name": "f2", "values": [0.9, 1.0, 1.1], "apply": {"f[2]": {"default": -4.0, "scale": "axis"}}},
    ]}

    out = mt.expand_grid(grid, {})

    assert out == [
        {"f[1]": 1.8},
        {"f[1]": 2.0},
        {"f[1]": 2.2},
        {"f[2]": -3.6},
        {"f[2]": -4.0},
        {"f[2]": -4.4},
    ]


def test_assemble_start_values_base_const_and_params():
    static = {"m_flow_nominal": {"base": "m_flow_nominal"}, "nFan": {"const": 1.0}}
    base = {"m_flow_nominal": 42.0}
    out = mt.assemble_start_values(static, base, {"TApp_nominal": 3.0, "_note": "ignored"})
    assert out == {"m_flow_nominal": 42.0, "nFan": 1.0, "TApp_nominal": 3.0}  # underscore key dropped


def test_resolve_enabled_precedence():
    fmu_cfg = {"enabled_candidates": ["Merkel"]}
    # project override wins
    assert mt.resolve_enabled(fmu_cfg, {"fmu_candidates": {"cooling_tower": ["YorkCalc"]}},
                              "cooling_tower") == ["YorkCalc"]
    # contract default when no project override
    assert mt.resolve_enabled(fmu_cfg, {}, "cooling_tower") == ["Merkel"]
    # neither -> None (all compete)
    assert mt.resolve_enabled({}, {}, "cooling_tower") is None


def test_select_candidates_filters_and_orders():
    cands = [{"name": "Merkel"}, {"name": "YorkCalc"}, {"name": "FixedApproach"}]
    assert mt.select_candidates(cands, None) == cands
    # requested order is honoured; unknown names ignored (not fatal)
    sel = mt.select_candidates(cands, ["YorkCalc", "Nope", "Merkel"])
    assert [c["name"] for c in sel] == ["YorkCalc", "Merkel"]
