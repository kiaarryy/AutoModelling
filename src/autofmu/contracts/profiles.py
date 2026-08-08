"""Canonical data contracts per equipment type.

A profile declares, in canonical field names, what each equipment model NEEDS.
The modelability gate (L2) compares available canonical fields against these to
decide the achievable modelling level: full_physical / nominal_only / blocked.

Canonical field naming (SI units):
  power_W, cooling_load_W, chw_flow_m3_h, cw_flow_m3_h,
  tchws_C, tchwr_C, tcws_C, tcwr_C, run_signal, speed_Hz, fan*_Hz, *_valve_pct
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class EquipmentProfile:
    equipment_type: str
    target: str  # primary calibration target -- MUST be an uncontrolled variable
    # fields required to attempt full physical / energy-curve calibration
    full_physical_required: Tuple[str, ...]
    # actual fields required by the configured physical FMU candidate(s). When
    # set, L2 uses this stricter contract for full_physical gating.
    fmu_required: Tuple[str, ...] = field(default_factory=tuple)
    # fields enabling a nominal-only (design/thermal) model when target is missing
    nominal_fields: Tuple[str, ...] = field(default_factory=tuple)
    optional: Tuple[str, ...] = field(default_factory=tuple)
    run_signal: str = "run_signal"
    # candidate models migrated from the validated Site A / paper workflows
    candidates: Tuple[str, ...] = field(default_factory=tuple)
    # variables regulated to a set point by a controller. They track Tset, not
    # the model, so they MUST NOT be used as a model-error metric (advisor Q2).
    controlled: Tuple[str, ...] = field(default_factory=tuple)
    # ordered preference of UNCONTROLLED outputs to validate the model against.
    validation_targets: Tuple[str, ...] = field(default_factory=tuple)
    # Channels whose near-zero value means the machine is not operating,
    # whatever the status bit claims. Used to corroborate run_signal -- see
    # modelability/operating.py for why the calibration target must NOT be used
    # here. First channel present in the frame wins; empty means the status bit
    # is the only available evidence.
    liveness: Tuple[str, ...] = field(default_factory=tuple)
    # The variable the selected model's nonlinearity rides on. Reported as an
    # excitation measure next to every score: a record in which the driver
    # never moved cannot identify the law, however many rows it has and however
    # well it scores. See observability/excitation.py.
    driver: str = ""

    def is_controlled(self, variable: str) -> bool:
        return variable in self.controlled


PROFILES = {
    "chiller": EquipmentProfile(
        equipment_type="chiller",
        target="power_W",
        full_physical_required=("power_W", "tchws_C", "tcwr_C", "chw_flow_m3_h"),
        fmu_required=("power_W", "tchws_C", "tchwr_C", "tcws_C", "tcwr_C",
                      "chw_flow_m3_h", "cw_flow_m3_h"),
        nominal_fields=("tchws_C", "tchwr_C", "chw_flow_m3_h"),
        optional=("cw_flow_m3_h", "tcws_C", "load_pct", "cooling_load_W"),
        candidates=("ElectricReformulatedEIR", "ElectricEIR", "Carnot_TEva"),
        controlled=("tchws_C",),  # chilled water supply temp is held at Tchws_set
        validation_targets=("power_W", "cooling_load_W"),
        # compressor command: a chiller commanded to 0% is off, and on LBNL CHI1
        # the plant status bit says otherwise for a full year.
        liveness=("load_pct",),
        driver="cooling_load_W",
    ),
    "cooling_tower": EquipmentProfile(
        equipment_type="cooling_tower",
        target="heat_rejection_W",
        full_physical_required=("power_W", "fan1_Hz"),
        # fan2_Hz is deliberately NOT required. Site A CT_06 and CT_07 carry a
        # single variable-speed drive commanding both cells, and demanding a
        # second channel downgraded both towers to nominal_only even though the
        # models need only a speed and a running-cell count -- fans_on_count
        # supplies the latter, and the feature builder averages over whichever
        # drive channels exist. Requiring a channel the physics does not need
        # discards a device for a wiring convention.
        fmu_required=("heat_rejection_W", "power_W", "fan1_Hz",
                      "tcwr_C", "tcws_1_C", "twb_C", "attributed_flow_m3_h",
                      "fans_on_count"),
        nominal_fields=("tcwr_C", "tcws_1_C"),
        optional=("fan2_Hz", "tcws_2_C", "tcws_C"),
        candidates=("YorkCalc", "Merkel", "fan_affinity_power"),
        # condenser water supply temp is held at Tcws_set by the fan controller
        controlled=("tcws_C", "tcws_1_C", "tcws_2_C"),
        validation_targets=("heat_rejection_W", "power_W"),
        # Deliberately empty. A cell in service with its fan off is a state the
        # models represent (the free-convection branch of the York wrapper), so
        # fan speed is not a liveness test; and attributed flow is near zero on
        # legitimate low-load rows -- as a liveness channel it discarded 79% of
        # Site A CT_05, whose fan is demonstrably turning throughout.
        liveness=(),
        driver="fan1_Hz",
    ),
    "pump": EquipmentProfile(
        equipment_type="pump",
        target="power_W",
        full_physical_required=("power_W", "speed_Hz"),
        fmu_required=("power_W", "speed_Hz"),
        nominal_fields=("speed_Hz",),
        optional=("head_m", "flow_m3_h", "attributed_flow_m3_h"),
        candidates=("SpeedControlled_Nrpm", "speed_poly_power", "affinity_power"),
        controlled=(),
        validation_targets=("power_W",),
        liveness=("speed_Hz",),  # a drive at zero is a pump that is not pumping
        driver="speed_Hz",
    ),
    "heat_exchanger": EquipmentProfile(
        equipment_type="heat_exchanger",
        # target is the UNCONTROLLED condenser-side return temp, NOT the
        # set-point-controlled chilled water supply temp (tchws_C).
        target="tcwr_C",
        # heat-transfer model needs flow to compute Q; Tencent lacks it
        full_physical_required=("tchws_C", "tchwr_C", "tcws_C", "tcwr_C", "chw_flow_m3_h"),
        fmu_required=("tchws_C", "tchwr_C", "tcws_C", "tcwr_C",
                      "chw_flow_m3_h", "cw_flow_m3_h"),
        nominal_fields=("tchws_C", "tchwr_C", "tcws_C", "tcwr_C"),
        optional=("chw_valve_pct", "cw_valve_pct", "cw_flow_m3_h", "Q_W", "attributed_flow_m3_h"),
        candidates=("ConstantEffectiveness", "PlateEffectivenessNTU"),
        controlled=("tchws_C",),  # chilled water supply temp is held at Tchws_set
        validation_targets=("Q_W", "tcwr_C"),
    ),
}


def get_profile(equipment_type: str) -> EquipmentProfile:
    if equipment_type not in PROFILES:
        raise KeyError(f"unknown equipment type: {equipment_type}")
    return PROFILES[equipment_type]
