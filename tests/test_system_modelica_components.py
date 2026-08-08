from __future__ import annotations

from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
COMPONENT_ROOT = REPO / "modelica" / "components"


COMPONENTS = [
    {
        "path": "chiller/EIRComponent.mo",
        "model": "EIRComponent",
        "inputs": ["TEvaEnt_C", "TEvaLvgSet_C", "TConEnt_C", "mEva_flow_kg_s", "mCon_flow_kg_s", "y"],
        "outputs": ["TEvaLvg_C", "TConLvg_C", "QEva_flow_W", "P_W", "COP", "PLR"],
        "parameters": ["capFunT[6]", "EIRFunT[6]", "EIRFunPLR[3]", "QEva_flow_nominal", "COP_nominal"],
    },
    {
        "path": "chiller/EEIRComponent.mo",
        "model": "EEIRComponent",
        "inputs": ["TEvaEnt_C", "TEvaLvgSet_C", "TConEnt_C", "mEva_flow_kg_s", "mCon_flow_kg_s", "y"],
        "outputs": ["TEvaLvg_C", "TConLvg_C", "QEva_flow_W", "P_W", "COP", "PLR"],
        "parameters": ["capFunT[6]", "EIRFunT[6]", "EIRFunPLR[10]", "QEva_flow_nominal", "COP_nominal"],
    },
    {
        "path": "chiller/CarnotComponent.mo",
        "model": "CarnotComponent",
        "inputs": ["TEvaEnt_C", "TEvaLvgSet_C", "TConEnt_C", "mEva_flow_kg_s", "mCon_flow_kg_s", "y"],
        "outputs": ["TEvaLvg_C", "TConLvg_C", "QEva_flow_W", "P_W", "COP", "PLR"],
        "parameters": ["etaCarnot_nominal", "a[6]", "QEva_flow_nominal"],
    },
    {
        "path": "cooling_tower/MerkelComponent.mo",
        "model": "MerkelComponent",
        "inputs": ["Tin_C", "Twb_C", "m_flow_kg_s", "y", "TRan_C"],
        "outputs": ["TOut_C", "Q_flow_W", "PFan_W", "TApp_C"],
        "parameters": ["ratWatAir_nominal", "cWatFra[3]", "PFan_nominal", "fanRelPow_r_P[5]"],
    },
    {
        "path": "cooling_tower/YorkCalc27Component.mo",
        "model": "YorkCalc27Component",
        "inputs": ["Tin_C", "Twb_C", "m_flow_kg_s", "y", "TRan_C"],
        "outputs": ["TOut_C", "Q_flow_W", "PFan_W", "TApp_C"],
        "parameters": ["f[27]", "PFan_nominal", "fanRelPow_r_P[5]"],
    },
    {
        "path": "heat_exchanger/ConstantEffectivenessComponent.mo",
        "model": "ConstantEffectivenessComponent",
        "inputs": ["T1In_C", "T2In_C", "m1_flow_kg_s", "m2_flow_kg_s"],
        "outputs": ["T1Out_C", "T2Out_C", "Q_flow_W", "eps_s"],
        "parameters": ["eps", "m1_flow_nominal", "m2_flow_nominal"],
    },
    {
        "path": "heat_exchanger/PlateEffectivenessNTUComponent.mo",
        "model": "PlateEffectivenessNTUComponent",
        "inputs": ["T1In_C", "T2In_C", "m1_flow_kg_s", "m2_flow_kg_s"],
        "outputs": ["T1Out_C", "T2Out_C", "Q_flow_W", "eps_s"],
        "parameters": ["Q_flow_nominal", "n1", "n2", "r_nominal", "m1_flow_nominal", "m2_flow_nominal"],
    },
    {
        "path": "pump/PumpEmpiricalPowerComponent.mo",
        "model": "PumpEmpiricalPowerComponent",
        "inputs": ["m_flow_kg_s", "y"],
        "outputs": ["P_W", "m_flow_s", "y_s"],
        "parameters": ["P_nominal", "m_flow_nominal", "c0", "c1", "c2", "c3", "c4"],
    },
    {
        "path": "pump/PumpMoverComponent.mo",
        "model": "PumpMoverComponent",
        "inputs": ["y"],
        "outputs": ["P_W", "m_flow_s", "dp_s", "head_s", "y_s"],
        "parameters": ["P_scale", "P_nominal", "m_flow_nominal", "dp_nominal", "dp_system_nominal"],
    },
]


def test_each_fmu_candidate_has_a_table_free_system_component():
    missing = [spec["path"] for spec in COMPONENTS if not (COMPONENT_ROOT / spec["path"]).exists()]
    assert missing == []


def test_system_components_expose_standard_signal_contracts():
    forbidden = ["CombiTimeTable", "tableOnFile", "table_path", "fileName="]

    for spec in COMPONENTS:
        text = (COMPONENT_ROOT / spec["path"]).read_text(encoding="utf-8")

        assert f"model {spec['model']}" in text
        assert f"end {spec['model']};" in text
        for token in forbidden:
            assert token not in text, f"{spec['path']} still contains {token}"
        for name in spec["inputs"]:
            assert f"RealInput {name}" in text, f"{spec['path']} missing input {name}"
        for name in spec["outputs"]:
            assert f"RealOutput {name}" in text, f"{spec['path']} missing output {name}"
        for name in spec["parameters"]:
            assert name in text, f"{spec['path']} missing parameter {name}"
