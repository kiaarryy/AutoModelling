"""Config-driven raw BMS CSV -> canonical CSV (Layer 1 ingest).

Ported from AUTO_FMU/adapters/site_a/common.py and generalized. A per-device
adapter YAML maps raw columns to canonical SI fields:

    source_csv: CHI01_202203_202303.csv
    timestamp: DateTime
    columns:
      power_W: {source: "CHI01.P_CHI01 (kW)", scale: 1000.0, unit: W}
      ...
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml

from autofmu.reporting import table_to_markdown


@dataclass(frozen=True)
class AdapterResult:
    rows: int
    output_csv: Path
    qa_csv: Path
    provenance_json: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _status(value: float) -> str:
    return "pass" if float(value) == 0.0 else "review"


def adapt_csv(config_path: Path, input_dir: Path, output_csv: Path) -> AdapterResult:
    config_path = Path(config_path).resolve()
    input_dir = Path(input_dir).resolve()
    output_csv = Path(output_csv).resolve()
    config: dict = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source_csv = _resolve(input_dir, config["source_csv"])
    # utf-8-sig: BMS exports often carry a BOM on the first header (e.g. DateTime)
    source = pd.read_csv(source_csv, encoding="utf-8-sig")
    timestamp_source = config["timestamp"]
    if timestamp_source not in source:
        raise ValueError(f"missing timestamp column: {timestamp_source}")
    canonical = pd.DataFrame({"timestamp": source[timestamp_source].astype(str)})
    mapping: dict = {}
    for canonical_name, specification in config["columns"].items():
        source_name = specification["source"]
        if source_name not in source:
            raise ValueError(f"missing source column: {source_name}")
        scale = float(specification.get("scale", 1.0))
        offset = float(specification.get("offset", 0.0))
        canonical[canonical_name] = (
            pd.to_numeric(source[source_name], errors="coerce") * scale + offset
        )
        mapping[canonical_name] = {
            "source": source_name,
            "scale": scale,
            "offset": offset,
            "unit": specification.get("unit", ""),
        }
    policy = str(config.get("timestamp_policy", "reject"))
    if policy not in {"reject", "sort", "sort_deduplicate"}:
        raise ValueError(
            "timestamp_policy must be one of: reject, sort, sort_deduplicate"
        )
    timestamps = pd.to_datetime(canonical["timestamp"], errors="coerce", utc=True)
    invalid_timestamps = int(timestamps.isna().sum())
    if invalid_timestamps:
        raise ValueError(f"invalid timestamp values: {invalid_timestamps}")
    duplicate_timestamps = int(timestamps.duplicated().sum())
    if duplicate_timestamps and policy != "sort_deduplicate":
        raise ValueError(f"duplicate timestamp values: {duplicate_timestamps}")
    if not bool(timestamps.is_monotonic_increasing) and policy == "reject":
        raise ValueError("timestamps are not sorted chronologically")

    canonical = canonical.assign(_timestamp=timestamps)
    if policy == "sort_deduplicate":
        canonical = canonical.drop_duplicates(subset=["_timestamp"], keep="first")
    if policy in {"sort", "sort_deduplicate"}:
        canonical = canonical.sort_values("_timestamp", kind="stable")
    canonical = canonical.reset_index(drop=True)
    timestamps = canonical.pop("_timestamp")
    canonical["timestamp"] = timestamps.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    intervals = timestamps.diff().dropna().dt.total_seconds()
    numeric = canonical.drop(columns=["timestamp"])
    non_finite_cells = int((~np.isfinite(numeric.to_numpy(dtype=float))).sum())
    qa_rows = [
        {"check": "rows", "value": int(len(canonical)), "status": "info"},
        {"check": "duplicate_timestamps", "value": duplicate_timestamps, "status": _status(duplicate_timestamps)},
        {"check": "invalid_timestamps", "value": invalid_timestamps, "status": _status(invalid_timestamps)},
        {"check": "non_finite_cells", "value": non_finite_cells, "status": "info"},
        {"check": "median_interval_seconds", "value": float(intervals.median()) if len(intervals) else "", "status": "info"},
    ]
    for column in numeric:
        missing_rate = float(numeric[column].isna().mean())
        qa_rows.append({"check": f"missing_rate:{column}", "value": round(missing_rate, 6), "status": "info"})
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    canonical.to_csv(output_csv, index=False)
    qa_csv = output_csv.parent / "adapter_qa.csv"
    qa_md = output_csv.parent / "adapter_qa.md"
    provenance_json = output_csv.parent / "source_mapping.json"
    qa = pd.DataFrame(qa_rows)
    qa.to_csv(qa_csv, index=False)
    qa_md.write_text("# Adapter QA\n\n" + table_to_markdown(qa) + "\n", encoding="utf-8")
    provenance_json.write_text(
        json.dumps(
            {
                "source_csv": str(source_csv),
                "source_sha256": _sha256(source_csv),
                "config": str(config_path),
                "timestamp": timestamp_source,
                "columns": mapping,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return AdapterResult(rows=len(canonical), output_csv=output_csv, qa_csv=qa_csv, provenance_json=provenance_json)
