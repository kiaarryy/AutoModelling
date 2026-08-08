"""Sensor-observability layer.

Decides, before any modelling is attempted, whether a device's BMS channels can
support the candidate models at all -- and at what evidence grade.  See
docs/REVISION_ENERGY_02_PLAN.md phase 3.
"""
from .attrition import AttritionLedger, AttritionStep
from .excitation import ExcitationFinding, excitation
from .physical_envelope import EnvelopeFinding, cop_envelope, frozen_channels
from .synthetic_channels import (
    ChannelReport,
    DependencyFinding,
    EvidenceTier,
    analyse_device,
    detect_dependencies,
    profile_channel,
)

__all__ = [
    "AttritionLedger",
    "AttritionStep",
    "ChannelReport",
    "DependencyFinding",
    "EnvelopeFinding",
    "ExcitationFinding",
    "EvidenceTier",
    "analyse_device",
    "cop_envelope",
    "detect_dependencies",
    "excitation",
    "frozen_channels",
    "profile_channel",
]
