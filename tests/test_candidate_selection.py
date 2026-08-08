"""The candidate-selection knob: choose which model *types* compete in a run.

Covers the CLI parser and that ``_load_device_fmu_cfg`` actually filters the
loaded contract candidates by the project ``fmu_candidates`` override (the path
all device branches inherit).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

from autofmu.cli import _parse_candidates
from autofmu.pipeline.calibrate import _load_device_fmu_cfg

REPO = Path(__file__).resolve().parents[1]
CT_CFG = yaml.safe_load((REPO / "configs" / "fmu" / "cooling_tower.yaml").read_text(encoding="utf-8"))
CTM = Path(CT_CFG["fmu_root"]) / "Cali_EIR_BSU_CH1" / "Modelica" / "CTM_0FMU.fmu"
HAS_FMPY = importlib.util.find_spec("fmpy") is not None


def test_parse_candidates_cli():
    assert _parse_candidates(["chiller=EIR,EEIR,Carnot"]) == {"chiller": ["EIR", "EEIR", "Carnot"]}
    assert _parse_candidates(["cooling_tower=Merkel", "pump=affinity"]) == {
        "cooling_tower": ["Merkel"], "pump": ["affinity"]}
    assert _parse_candidates(None) == {}


def test_parse_candidates_rejects_malformed():
    with pytest.raises(SystemExit):
        _parse_candidates(["no_equals_sign"])


def _project_cfg(fmu_candidates=None):
    cfg = {"_root": REPO, "fmu_config_dir": "configs/fmu"}
    if fmu_candidates is not None:
        cfg["fmu_candidates"] = fmu_candidates
    return cfg


@pytest.mark.skipif(not (HAS_FMPY and CTM.exists()), reason="CT FMU / fmpy unavailable")
def test_load_device_fmu_cfg_default_is_all_candidates():
    cfg, _ = _load_device_fmu_cfg(_project_cfg(), "cooling_tower")
    assert [c["name"] for c in cfg["candidates"]] == ["Merkel", "YorkCalc"]


@pytest.mark.skipif(not (HAS_FMPY and CTM.exists()), reason="CT FMU / fmpy unavailable")
def test_load_device_fmu_cfg_project_override_filters_and_orders():
    cfg, _ = _load_device_fmu_cfg(_project_cfg({"cooling_tower": ["YorkCalc"]}), "cooling_tower")
    assert [c["name"] for c in cfg["candidates"]] == ["YorkCalc"]
