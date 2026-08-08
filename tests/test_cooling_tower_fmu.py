"""FMU-2: cooling-tower L3 engine -- drive Merkel / YorkCalc FMUs from L2 data.

Covers the pure table build and a guarded end-to-end fit/validate. This locks the
FMU-driving + L2 per-timestep fan-count scaling. NOTE: matching the prepared-data
baseline (CT_01 York TOut ~2%) additionally requires the L2 CT data-prep
upgrade (representative steady window + flow-attribution quality); that is a
separate, still-open task, so this test asserts structure + finiteness, not the
baseline accuracy.
"""
from __future__ import annotations

import importlib.util
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from autofmu.devices import cooling_tower_fmu as ctf

REPO = Path(__file__).resolve().parents[1]
CT_CFG = yaml.safe_load((REPO / "configs" / "fmu" / "cooling_tower.yaml").read_text(encoding="utf-8"))
FMU_ROOT = Path(CT_CFG["fmu_root"])
CTM = FMU_ROOT / "Cali_EIR_BSU_CH1" / "Modelica" / "CTM_0FMU.fmu"
RUN_CANON = REPO / "outputs" / "runs" / "fmu1dev" / "cooling_tower" / "CT_01" / "canonical_attributed.csv"


def _synthetic_ct(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, n)
    twb = 22.0 + 2.0 * np.sin(t)
    tout = twb + 2.0 + 0.3 * np.sin(t / 2)
    tin = tout + 4.0 + 0.2 * np.cos(t)
    fan = 35.0 + 8.0 * np.sin(t / 3) + rng.normal(0, 0.5, n)
    flow = 700.0 + 40.0 * np.sin(t / 4)        # m3/h attributed total
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-06-01", periods=n, freq="5min").astype(str),
        "power_W": 40000 + 5000 * np.sin(t / 3), "fan1_Hz": fan, "fan2_Hz": fan,
        "tcwr_C": tin, "tcws_1_C": tout, "twb_C": twb, "run_signal": np.ones(n),
        "attributed_flow_m3_h": flow, "fans_on_count": np.full(n, 2.0),
        "heat_rejection_W": flow * 1000 / 3600 * 4180 * (tin - tout),
    })


def test_build_ct_table_columns_and_percell_flow():
    table = ctf.build_ct_table(_synthetic_ct())
    assert list(table.columns) == ctf.TABLE_COLUMNS
    assert len(table) > 100
    # per-cell flow = total / fan count; with 2 fans, mdot_cell ~= total_kg_s / 2
    assert (table["mdot_cell_kgps"] > 0).all()
    assert (table["fans_on_count"] > 0).all()
    assert np.isfinite(table.to_numpy(dtype=float)).all()


def test_build_ct_table_drops_degenerate_low_flow_rows():
    # When a tower's paired chiller is blocked, loop-flow attribution collapses
    # to near-zero on most rows (Site A CT_05). The flow floor must drop those so
    # the FMU is scored only on rows that carry real water.
    frame = _synthetic_ct(n=500)
    af = frame["attributed_flow_m3_h"].to_numpy(dtype=float).copy()
    af[:300] = 1.0   # 60% of rows: degenerate near-zero flow
    frame["attributed_flow_m3_h"] = af
    table = ctf.build_ct_table(frame)
    assert 0 < len(table) < 300                                  # near-zero rows excluded
    assert (table["mdot_cell_kgps"] * table["fans_on_count"] > 5.0).all()   # real flow only


def test_build_ct_table_uses_active_fan_speed_when_one_fan_is_off():
    frame = _synthetic_ct()
    frame["fan2_Hz"] = 0.0
    frame["fans_on_count"] = 1.0

    table = ctf.build_ct_table(frame)

    expected = frame["fan1_Hz"].to_numpy(dtype=float) / ctf.FAN_NOM_HZ
    assert len(table) == len(frame)
    assert np.allclose(table["y_used"].to_numpy(dtype=float), expected)


def test_fit_ct_fmu_records_evaluation_failure_without_aborting_fleet(tmp_path, monkeypatch):
    fake_fmu = tmp_path / "fake.fmu"
    fake_fmu.write_text("not a real fmu", encoding="utf-8")
    cfg = {
        "candidates": [{
            "name": "SyntheticCT",
            "fmu": str(fake_fmu),
            "table_param": "table_path",
            "fan_power": {"curve_prefix": "fanRelPow_r_P", "nominal_key": "PFan_nominal"},
        }]
    }

    @contextmanager
    def fake_extracted_fmu(_path):
        yield tmp_path

    monkeypatch.setattr(ctf, "extracted_fmu", fake_extracted_fmu)
    monkeypatch.setattr(
        ctf,
        "_search_model",
        lambda *args, **kwargs: {"status": "ok", "params": {"PFan_nominal": 1.0}},
    )
    monkeypatch.setattr(
        ctf,
        "_drive",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("FMU evaluation failed")),
    )

    fit = ctf.fit_ct_fmu(
        "CT_SYN",
        _synthetic_ct(n=500),
        cfg,
        tmp_path,
        tmp_path,
        {"min_calibration_rows": 200},
    )

    assert fit["status"] == "no_candidate"
    assert fit["rows"][0]["candidate"] == "SyntheticCT"
    assert fit["rows"][0]["status"] == "evaluation_failed"
    assert "FMU evaluation failed" in fit["rows"][0]["reason"]


HAS_FMPY = importlib.util.find_spec("fmpy") is not None


@pytest.mark.skipif(not HAS_FMPY or not (CTM.exists() and RUN_CANON.exists()),
                    reason="CT Merkel FMU / ingested CT canonical not available")
def test_fit_and_validate_ct_fmu_runs(tmp_path):
    frame = pd.read_csv(RUN_CANON)
    th = {"run_on": 0.5, "min_calibration_rows": 200, "ct_subset_rows": 400}
    fit = ctf.fit_ct_fmu("CT_01", frame, CT_CFG, FMU_ROOT, tmp_path, th)
    assert fit["status"] == "ok"
    assert fit["selected_candidate"] in ("Merkel", "YorkCalc")
    for cand in fit["candidates"]:
        assert "heldout" not in cand
        assert np.isfinite(cand["selection"]["T_CVRMSE_pct"])
        assert np.isfinite(cand["test"]["T_CVRMSE_pct"])
    val = ctf.validate_ct_fmu("CT_01", frame, fit["best"], FMU_ROOT, tmp_path, th)
    assert val["status"] == "ok"
    assert np.isfinite(val["full_period"]["T_CVRMSE_pct"])
    assert val["full_period"]["N"] > 100


def test_score_reports_fan_power_but_does_not_rank_on_it():
    """Fan power must be measurable (the fan curve is fitted against it) and
    must not enter the candidate ranking (both candidates share the curve)."""
    import numpy as np
    import pandas as pd
    from autofmu.devices.cooling_tower_fmu import _score

    n = 300
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "TOut_m": 299.15 + rng.normal(0, 0.5, n),
        "Q_m": 500_000 + rng.normal(0, 10_000, n),
        "P_m": 9_000 + rng.normal(0, 200, n),
    })
    frame["TOut_s"] = frame["TOut_m"] + 0.2
    frame["Q_s"] = frame["Q_m"] * 1.02
    fans = np.ones(n)

    good_fan = frame.assign(P_s=frame["P_m"] * 1.01)
    bad_fan = frame.assign(P_s=frame["P_m"] * 1.40)
    a, b = _score(good_fan, fans), _score(bad_fan, fans)

    assert "P_CVRMSE_pct" in a and "P_NMBE_pct" in a
    assert a["P_CVRMSE_pct"] < b["P_CVRMSE_pct"]
    # the ranking score is temperature only, so fan power cannot move it
    assert a["score"] == pytest.approx(b["score"])


def test_closed_loop_strategy_returns_fmu_named_coefficients():
    """The York fit must hand back parameters the FMU can accept directly."""
    import numpy as np
    import pandas as pd
    from autofmu.devices.cooling_tower_fmu import _fit_york27_closed_loop

    n = 900
    rng = np.random.default_rng(0)
    twb = 20 + 4 * rng.random(n)
    tin = twb + 6 + rng.normal(0, 0.4, n)
    tout = twb + 3 + rng.normal(0, 0.3, n)
    table = pd.DataFrame({
        "Tin_C": tin, "Twb_C": twb, "Tout_meas_C": tout,
        "mdot_cell_kgps": np.full(n, 100.0), "y_used": np.full(n, 0.8),
    })
    base = {"m_flow_nominal": 100.0, "TAirInWB_nominal": 295.15,
            "TRan_nominal": 3.0, "TApp_nominal": 3.0}

    params = _fit_york27_closed_loop(table, base)
    assert [f"f[{i}]" for i in range(1, 28)] == [k for k in params if k.startswith("f[")]
    assert np.isfinite(list(params.values())).all()
    assert "FRWat0" in params and params["FRWat0"] > 0
