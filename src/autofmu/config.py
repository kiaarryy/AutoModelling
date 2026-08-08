from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from autofmu.contracts.profiles import PROFILES

# Environment variables that override the (machine-specific) roots in the shipped
# configs, so a clone runs without editing YAML. When unset, the config value is
# used as-is (keeps the original host + the test suite working). Config path
# strings also expand ``${VAR}`` so they can reference any environment variable.
ENV_DATA_ROOT = "AUTOFMU_DATA_ROOT"   # external BMS data directory
ENV_FMU_ROOT = "AUTOFMU_FMU_ROOT"     # external exported-FMU directory
SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def expand(value: str) -> str:
    """Expand ``${VAR}`` / ``$VAR`` against the environment."""
    return os.path.expandvars(str(value))


def fmu_root_from_env(config_value: Any) -> str:
    """Resolved external FMU root: ``$AUTOFMU_FMU_ROOT`` wins, else the config
    value (both ``${VAR}``-expanded)."""
    return expand(os.environ.get(ENV_FMU_ROOT) or str(config_value or "."))


def load_project(path: Path) -> dict:
    """Load a project config and attach resolved roots.

    Required keys: ``devices`` (list of {id, type, adapter}).
    Optional: ``outputs_dir`` (default ``outputs``), ``data_root``,
    ``adapters_dir`` (default ``adapters``), ``thresholds``.
    """
    config_path = Path(path).resolve()
    config: dict = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    config["_config_path"] = config_path
    config["_root"] = config_path.parent
    config.setdefault("outputs_dir", "outputs")
    config.setdefault("adapters_dir", "adapters")
    config.setdefault("thresholds", {})
    config.setdefault("devices", [])
    return config


def resolve_path(config: dict, value: str) -> Path:
    path = Path(expand(value))
    return path.resolve() if path.is_absolute() else (Path(config["_root"]) / path).resolve()


def data_root(config: dict) -> Path:
    # $AUTOFMU_DATA_ROOT overrides the config value (external read-only BMS dir).
    raw = os.environ.get(ENV_DATA_ROOT) or config.get("data_root")
    if not raw:
        raise ValueError(
            "project config requires 'data_root' (external read-only BMS dir) "
            "or the AUTOFMU_DATA_ROOT environment variable"
        )
    return resolve_path(config, str(raw))


def adapter_config_path(config: dict, device: dict) -> Path:
    adapter = device.get("adapter")
    if not adapter:
        raise ValueError(f"device {device.get('id', '?')}: 'adapter' config is required")
    return resolve_path(config, str(Path(config["adapters_dir"]) / adapter))


def validate_run_id(run_id: str) -> str:
    """Return a filesystem-safe run identifier or reject it.

    A run ID becomes part of an output path controlled by the CLI. Restricting
    it to one path segment prevents traversal and accidental absolute paths.
    """
    value = str(run_id)
    if not SAFE_RUN_ID.fullmatch(value):
        raise ValueError(
            "run-id must be a safe run identifier: letters, digits, '.', '_', '-' only"
        )
    return value


def run_dir(config: dict, run_id: str) -> Path:
    safe_run_id = validate_run_id(run_id)
    return (
        resolve_path(config, str(config["outputs_dir"])) / "runs" / safe_run_id
    ).resolve()


def validate_project(config: dict) -> list:
    errors: list = []
    if not config.get("devices"):
        errors.append("no devices defined")
    if not config.get("data_root"):
        errors.append("data_root is required")
    elif not data_root(config).exists():
        errors.append(f"data_root does not exist: {data_root(config)}")
    for device in config.get("devices", []):
        dev_id = device.get("id", "?")
        if not device.get("id"):
            errors.append("device id is required")
        equipment_type = device.get("type")
        if equipment_type not in PROFILES:
            errors.append(f"{dev_id}: unsupported equipment type: {equipment_type}")
        try:
            adapter = adapter_config_path(config, device)
            if not adapter.exists():
                errors.append(f"{dev_id}: adapter config not found: {adapter}")
        except ValueError as exc:
            errors.append(str(exc))
    return errors
