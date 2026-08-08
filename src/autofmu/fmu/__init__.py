"""FMU layer (L3): inspect modelDescription.xml and run via FMPy.

Never degrades to passthrough: the runner raises a clear error if FMPy is
missing, the FMU/data table is absent, or the simulation returns non-finite
values.
"""
from autofmu.fmu.runner import run_fmu, run_device_fmu
from autofmu.fmu.inspect import (
    inspect_fmu,
    load_device_fmu_config,
    validate_against_config,
    validate_config_file,
    resolve_candidate_fmu,
)

__all__ = [
    "run_fmu",
    "run_device_fmu",
    "inspect_fmu",
    "load_device_fmu_config",
    "validate_against_config",
    "validate_config_file",
    "resolve_candidate_fmu",
]
