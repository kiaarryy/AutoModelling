from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "generate_unified_automodelling_reports.py"


def load_module():
    spec = importlib.util.spec_from_file_location("generate_unified_automodelling_reports", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_normalize_chiller_metrics_reports_power_q_and_cop():
    module = load_module()

    row = {
        "device_id": "CH-01",
        "equipment_type": "chiller",
        "status": "ok",
        "candidate": "EEIR",
        "N": 100,
        "CVRMSE_pct": 5.2,
        "NMBE_pct": -0.5,
        "Q_CVRMSE_pct": 2.1,
        "COP_CVRMSE_pct": 6.3,
        "COP_NMBE_pct": 1.2,
        "coverage_of_on_pct": 98.0,
    }

    out = module.normalize_device_metrics(row)

    assert out["metric_basis"] == "power_W + cooling_load_W + COP"
    assert out["P_CVRMSE_pct"] == "5.20"
    assert out["P_NMBE_pct"] == "-0.50"
    assert out["Q_CVRMSE_pct"] == "2.10"
    assert out["COP_CVRMSE_pct"] == "6.30"
    assert out["COP_NMBE_pct"] == "1.20"
    assert out["missing_metrics"] == ""


def test_normalize_cooling_tower_metrics_uses_power_sidecar_when_available():
    module = load_module()

    row = {
        "device_id": "CT1",
        "equipment_type": "cooling_tower",
        "status": "ok",
        "candidate": "Merkel",
        "N": 200,
        "CVRMSE_pct": 41.0,
        "NMBE_pct": -4.0,
        "T_CVRMSE_pct_diagnostic": 8.4,
        "T_NMBE_pct_diagnostic": 1.1,
        "coverage_of_on_pct": 87.5,
    }
    power_row = {"CVRMSE_pct": 12.3, "NMBE_pct": -2.5, "MAPE_pct": 9.1, "R2": 0.88}

    out = module.normalize_device_metrics(row, power_row=power_row)

    assert out["metric_basis"] == "heat_rejection_W + outlet_temperature + power_W"
    assert out["Q_CVRMSE_pct"] == "41.00"
    assert out["T_out_CVRMSE_pct"] == "8.40"
    assert out["P_CVRMSE_pct"] == "12.30"
    assert out["P_MAPE_pct"] == "9.10"
    assert out["P_R2"] == "0.88"
    assert out["missing_metrics"] == ""


def test_normalize_cooling_tower_metrics_marks_missing_power():
    module = load_module()

    row = {
        "device_id": "CT01",
        "equipment_type": "cooling_tower",
        "status": "ok",
        "candidate": "YorkCalc",
        "N": 50,
        "CVRMSE_pct": 33.0,
        "NMBE_pct": 3.0,
        "T_CVRMSE_pct_diagnostic": 11.0,
    }

    out = module.normalize_device_metrics(row)

    assert out["P_CVRMSE_pct"] == ""
    assert out["missing_metrics"] == "P validation artifact missing"


def test_normalize_heat_exchanger_metrics_reports_temperature_and_q():
    module = load_module()

    row = {
        "device_id": "HX_01",
        "equipment_type": "heat_exchanger",
        "status": "ok",
        "candidate": "HXEffectiveness",
        "N": 80,
        "CVRMSE_pct": 4.5,
        "NMBE_pct": 0.3,
        "Q_CVRMSE_pct": 12.0,
        "T2_CVRMSE_pct": 4.5,
    }

    out = module.normalize_device_metrics(row)

    assert out["metric_basis"] == "leaving_temperature + heat_transfer_W"
    assert out["T_out_CVRMSE_pct"] == "4.50"
    assert out["Q_CVRMSE_pct"] == "12.00"
