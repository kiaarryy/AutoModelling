"""Inspect an FMU's interface and validate it against a device FMU config.

`inspect_fmu` reads modelDescription.xml. `load_device_fmu_config` /
`validate_against_config` check that a `configs/fmu/<type>.yaml` contract matches
the real FMU interface (declared table param / inputs / outputs / tunables
actually exist), so a stale config surfaces as an explicit mismatch rather than a
confusing simulation failure later.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import yaml


def inspect_fmu(fmu_path: Path) -> dict:
    from fmpy import read_model_description

    md = read_model_description(str(Path(fmu_path)), validate=False)
    inputs, outputs, params = [], [], []
    for v in md.modelVariables:
        if v.causality == "input":
            inputs.append(v.name)
        elif v.causality == "output":
            outputs.append(v.name)
        elif v.causality == "parameter":
            params.append({"name": v.name, "start": v.start, "variability": v.variability})
    stop = md.defaultExperiment.stopTime if md.defaultExperiment else None
    return {
        "model_name": md.modelName,
        "fmi_version": md.fmiVersion,
        "inputs": inputs,
        "outputs": outputs,
        "n_parameters": len(params),
        "parameters": params,
        "default_stop_time": stop,
    }


def resolve_candidate_fmu(config: dict, candidate: dict) -> Path:
    """Resolve a candidate's FMU path against the external fmu_root
    (``$AUTOFMU_FMU_ROOT`` overrides the config value; both ``${VAR}``-expanded)."""
    from autofmu.config import fmu_root_from_env, expand

    raw = Path(expand(str(candidate["fmu"])))
    if raw.is_absolute():
        return raw.resolve()
    return (Path(fmu_root_from_env(config.get("fmu_root"))) / raw).resolve()


def load_device_fmu_config(path: Union[str, Path]) -> dict:
    """Load a configs/fmu/<type>.yaml contract and resolve candidate FMU paths."""
    config_path = Path(path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    config["_config_path"] = config_path
    for candidate in config.get("candidates", []):
        candidate["_fmu_path"] = resolve_candidate_fmu(config, candidate)
    return config


def _flatten_tunables(candidate: dict) -> list[str]:
    out: list[str] = []
    groups = candidate.get("tunable_parameters", {}) or {}
    for names in groups.values():
        out.extend(names)
    return out


def validate_against_config(config: dict) -> list[dict]:
    """Check each candidate FMU's real interface against the config contract.

    Returns one report row per candidate. Status is ``ok`` when the FMU's real
    interface contains every declared output / table param / input / tunable;
    ``missing_fmu`` when the (external) FMU is absent; ``mismatch`` otherwise.
    Never raises on a missing external FMU -- callers decide to skip or fail.
    """
    drive = config.get("drive", "data_table")
    rows: list[dict] = []
    for candidate in config.get("candidates", []):
        fmu_path = candidate.get("_fmu_path") or resolve_candidate_fmu(config, candidate)
        name = candidate.get("name", "?")
        if not Path(fmu_path).exists():
            rows.append({"candidate": name, "status": "missing_fmu", "fmu": str(fmu_path), "missing": []})
            continue
        info = inspect_fmu(Path(fmu_path))
        fmu_outputs = set(info["outputs"])
        fmu_inputs = set(info["inputs"])
        fmu_params = {p["name"] for p in info["parameters"]}

        missing: list[str] = []
        for out in candidate.get("outputs", []):
            if out not in fmu_outputs:
                missing.append(f"output:{out}")
        if drive == "data_table":
            tp = candidate.get("table_param")
            if tp and tp not in fmu_params:
                missing.append(f"table_param:{tp}")
        elif drive == "input":
            for port in candidate.get("inputs", []):
                if port not in fmu_inputs:
                    missing.append(f"input:{port}")
        for tun in _flatten_tunables(candidate):
            if tun not in fmu_params:
                missing.append(f"tunable:{tun}")

        rows.append({
            "candidate": name,
            "status": "ok" if not missing else "mismatch",
            "fmu": str(fmu_path),
            "model_name": info["model_name"],
            "missing": missing,
        })
    return rows


def validate_config_file(path: Union[str, Path]) -> list[dict]:
    """Convenience: load a config file and validate every candidate."""
    return validate_against_config(load_device_fmu_config(path))
