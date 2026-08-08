"""Capability gating (Layer 2): decide the achievable modelling level per device.

Three states, never a silent passthrough:
  - full_physical : target + driving fields present with enough valid rows ->
                    physical / energy-curve calibration can be attempted.
  - nominal_only  : thermal/design fields present but the calibration target
                    (e.g. measured power, or flow for HX heat balance) is
                    missing -> only nominal/design parameters can be produced.
  - blocked       : not enough valid data to model at all.

Every result records the reason and the missing canonical fields, so a
data-poor dataset (e.g. Tencent heat exchanger) degrades explicitly rather than
failing or faking a fit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Mapping, Tuple

import numpy as np
import pandas as pd

from autofmu.contracts.profiles import EquipmentProfile
from autofmu.modelability.operating import operating_mask


@dataclass
class GateResult:
    device_id: str
    equipment_type: str
    level: str  # full_physical | nominal_only | blocked
    reason: str
    rows: int
    on_rows: int
    valid_full_rows: int
    valid_nominal_rows: int
    missing_full_fields: Tuple[str, ...] = field(default_factory=tuple)
    target: str = ""
    flags: Tuple[str, ...] = field(default_factory=tuple)
    driver: str = ""
    driver_excitation: float = float("nan")

    def as_row(self) -> dict:
        return {
            "device_id": self.device_id,
            "equipment_type": self.equipment_type,
            "level": self.level,
            "target": self.target,
            "rows": self.rows,
            "on_rows": self.on_rows,
            "valid_full_rows": self.valid_full_rows,
            "valid_nominal_rows": self.valid_nominal_rows,
            "driver": self.driver,
            "driver_excitation": self.driver_excitation,
            "missing_full_fields": ",".join(self.missing_full_fields),
            "flags": ",".join(self.flags),
            "reason": self.reason,
        }


def _present(frame: pd.DataFrame, columns) -> List[str]:
    """Columns present in the frame with at least one finite value."""
    present = []
    for column in columns:
        if column in frame:
            series = pd.to_numeric(frame[column], errors="coerce")
            if np.isfinite(series.to_numpy(dtype=float)).any():
                present.append(column)
    return present


def _finite_rows(frame: pd.DataFrame, columns, on_mask: pd.Series) -> int:
    if not columns:
        return 0
    available = [c for c in columns if c in frame]
    if len(available) < len(list(columns)):
        return 0
    block = frame[available].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(block.to_numpy(dtype=float)).all(axis=1)
    return int((finite & on_mask.to_numpy()).sum())


def _target_usable(frame: pd.DataFrame, target: str) -> bool:
    """A target is usable for calibration only if it is present, finite, and not
    a dead sensor (all-zero / no variation)."""
    if target not in frame:
        return False
    values = pd.to_numeric(frame[target], errors="coerce").to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return False
    # dead sensor := all-zero (or all-NaN, handled above)
    return float(np.max(np.abs(finite))) > 0.0


def _synthetic_targets(frame: pd.DataFrame, profile: EquipmentProfile,
                       columns: Mapping[str, Mapping] | None):
    """Roles that are exact functions of another channel, among those that matter.

    A calibration target which the BMS computed from another point cannot
    validate anything: fitting to it recovers the BMS's own formula. On the
    HKUST archive 21 of 25 pumps publish ``power = P_rated * speed``, and the
    pipeline happily scored one of them at 0.00% CVRMSE. Detecting that here,
    before calibration, is the point of the observability layer.

    The map is built from the canonical frame's own role columns rather than
    from the adapter, because by this stage the frame is canonical: its columns
    are ``power_W``, ``speed_Hz`` and so on, while the adapter's sources are
    still raw historian names like ``PP2.power_consumption``. Passing the
    adapter map here matched nothing and the detector silently reported no
    channels at all -- HKUST pumps sailed through the gate and were scored at
    0.35% to 1.1%, which is the BMS formula being rediscovered.
    """
    from autofmu.observability import analyse_device

    roles = [c for c in frame.columns if c != "timestamp"]
    if not roles:
        return frozenset()
    result = analyse_device(frame, {r: {"source": r} for r in roles},
                            site="", device="",
                            equipment_type=profile.equipment_type)
    interesting = {profile.target, *profile.validation_targets}
    return frozenset(result.synthetic_roles & interesting)


def _envelope_flags(frame: pd.DataFrame, profile: EquipmentProfile,
                    operating: np.ndarray) -> List[str]:
    """Physical-plausibility findings, recorded but never used to block.

    A frozen flow meter or an unattainable reconstructed COP says the derived
    quantities cannot be trusted; it does not say the measured target cannot.
    Blocking on these would discard Site A's chillers, whose power models are
    fitted and scored against measured power and are valid, while their implied
    COP of 10-22 is not. The distinction is the point, so it is recorded rather
    than acted on.
    """
    from autofmu.observability import cop_envelope, frozen_channels

    flags: List[str] = []
    continuous = [c for c in profile.fmu_required
                  if c.endswith(("_m3_h", "_W")) and c != profile.target]
    for finding in frozen_channels(frame, continuous, operating):
        flags.append(finding.as_flag())

    if profile.equipment_type == "chiller":
        from autofmu.devices.chiller import _cooling_load, _num

        _, findings = cop_envelope(_cooling_load(frame), _num(frame, "power_W"),
                                   operating)
        flags.extend(f.as_flag() for f in findings)
    return flags


def _driver_excitation(frame: pd.DataFrame, profile: EquipmentProfile,
                       operating: np.ndarray):
    """How far the model's driving variable moved over the operating record.

    Reported next to the score, never used to block. Site A CDWP_01 holds a
    speed ratio of 0.700-0.701 across 28,619 rows and reports 5.93% test
    CVRMSE; the score is real and the identification is not.
    """
    from autofmu.observability import excitation

    if not profile.driver:
        return None
    source = frame
    if profile.driver == "cooling_load_W" and "cooling_load_W" not in frame:
        # reconstructed from flow and temperature difference like everywhere else
        from autofmu.devices.chiller import _cooling_load

        source = pd.DataFrame({"cooling_load_W": _cooling_load(frame)})
    return excitation(source, profile.driver, operating)


def gate_device(
    device_id: str,
    frame: pd.DataFrame,
    profile: EquipmentProfile,
    thresholds: Mapping[str, float] = None,
    adapter_columns: Mapping[str, Mapping] | None = None,
) -> GateResult:
    thresholds = thresholds or {}
    run_on = float(thresholds.get("run_on", 0.5))
    min_full = int(thresholds.get("min_full_physical_rows", 500))
    min_nominal = int(thresholds.get("min_nominal_rows", 200))

    rows = int(len(frame))
    flags: List[str] = []

    # The operating mask is the status bit corroborated by the profile's
    # liveness channel, never the bit alone -- see modelability/operating.py for
    # the LBNL chiller whose bit read "on" for a year while it idled.
    operating = operating_mask(
        frame,
        profile,
        run_on,
        dead_run_policy=str(thresholds.get("dead_run_signal_policy", "fallback")),
    )
    on_mask = pd.Series(operating.mask, index=frame.index)
    flags.extend(operating.flags)
    on_rows = int(on_mask.sum())

    full_requirements = profile.fmu_required or profile.full_physical_required
    present_full = set(_present(frame, full_requirements))
    missing_full = tuple(c for c in full_requirements if c not in present_full)
    valid_full = _finite_rows(frame, full_requirements, on_mask)
    valid_nominal = _finite_rows(frame, profile.nominal_fields, on_mask)
    target_ok = _target_usable(frame, profile.target)
    if not target_ok:
        flags.append("target_sensor_dead")
    flags.extend(_envelope_flags(frame, profile, operating.mask))
    drive = _driver_excitation(frame, profile, operating.mask)
    driver_name = drive.channel if drive else profile.driver
    driver_span = drive.relative_span if drive else float("nan")
    if drive is not None and not drive.excited:
        flags.append(drive.as_flag())

    # Noise-free datasets -- simulation benchmarks, synthetic fixtures -- have
    # exact functional relations by construction, and there the relation is the
    # point rather than a defect. Real BMS archives do not, so the check is on
    # by default and turned off per project.
    if bool(thresholds.get("block_synthetic_targets", True)):
        synthetic = _synthetic_targets(frame, profile, adapter_columns)
    else:
        synthetic = frozenset()
    if synthetic:
        flags.extend(f"synthetic_channel:{role}" for role in sorted(synthetic))
    if profile.target in synthetic:
        # Evidence tier TX: the target is published by the BMS but derived from
        # another channel, so it is below even a declared nameplate value.
        # Nothing can be validated against it.
        return GateResult(
            device_id, profile.equipment_type, "blocked",
            f"calibration target '{profile.target}' is an exact function of "
            f"another channel (synthetic point published as a measurement); "
            f"fitting to it would recover the BMS formula, not the device",
            rows, on_rows, valid_full, valid_nominal, missing_full,
            profile.target, tuple(flags), driver_name, driver_span,
        )

    if not missing_full and target_ok and valid_full >= min_full:
        return GateResult(
            device_id, profile.equipment_type, "full_physical",
            "target and driving fields available with sufficient valid rows",
            rows, on_rows, valid_full, valid_nominal, missing_full, profile.target,
            tuple(flags), driver_name, driver_span,
        )
    if valid_nominal >= min_nominal:
        if not target_ok and not missing_full:
            cause = "target '%s' is a dead/all-zero sensor" % profile.target
        else:
            cause = "full-physical fields missing/insufficient (%s)" % (", ".join(missing_full) or "few valid rows")
        reason = "%s; thermal/design fields support nominal-only modelling" % cause
        return GateResult(
            device_id, profile.equipment_type, "nominal_only", reason,
            rows, on_rows, valid_full, valid_nominal, missing_full, profile.target,
            tuple(flags), driver_name, driver_span,
        )
    return GateResult(
        device_id, profile.equipment_type, "blocked",
        "insufficient valid rows for any model (valid_full=%d, valid_nominal=%d)" % (valid_full, valid_nominal),
        rows, on_rows, valid_full, valid_nominal, missing_full, profile.target,
        tuple(flags), driver_name, driver_span,
    )
