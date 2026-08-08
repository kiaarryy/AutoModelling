"""Generate Site D adapters and fleet config from downloaded canonical CSVs."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--point-map", type=Path, required=True)
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    point_map = pd.read_csv(args.point_map, encoding="utf-8-sig")
    types = dict(
        point_map[["device_id", "equipment_type"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )

    config_dir = REPO / "configs" / "site_d"
    adapters_dir = config_dir / "adapters"
    adapters_dir.mkdir(parents=True, exist_ok=True)
    devices = []
    for csv_path in sorted((data_dir / "devices").glob("*.csv")):
        device_id = csv_path.stem
        equipment_type = types[device_id]
        columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
        mapped = {
            column: {"source": column}
            for column in columns
            if column != "timestamp"
        }
        adapter_name = "%s_%s.yaml" % (equipment_type, device_id.lower())
        adapter = {
            "source_csv": "devices/%s.csv" % device_id,
            "timestamp": "timestamp",
            "timestamp_policy": "reject",
            "columns": mapped,
        }
        (adapters_dir / adapter_name).write_text(
            yaml.safe_dump(adapter, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        device = {
            "id": device_id,
            "type": equipment_type,
            "adapter": adapter_name,
        }
        if equipment_type == "pump":
            device["group"] = "chilled_water_pump" if device_id.startswith("CHP") else "sea_water_pump"
        devices.append(device)

    project = {
        "dataset": "site_d",
        "data_root": str(data_dir),
        "adapters_dir": "adapters",
        "outputs_dir": "../../outputs",
        "fmu_config_dir": "../fmu",
        "source": {
            "brick_model": str(data_dir.parent / "new_model_0609.ttl"),
            "download_manifest": str(data_dir / "download_manifest.csv"),
            "point_map": str(args.point_map.resolve()),
        },
        "thresholds": {
            "run_on": 0.5,
            "dead_run_signal_policy": "off",
            "min_full_physical_rows": 48,
            "min_nominal_rows": 24,
            "min_calibration_rows": 48,
            "validation_fold": 3,
            "validation_split": "interleaved",
            "block_synthetic_targets": True,
        },
        "fmu_candidates": {
            "chiller": ["EIR", "EEIR"],
            "pump": ["empirical_power"],
        },
        "devices": devices,
    }
    project_path = config_dir / "project_fleet.yaml"
    project_path.write_text(
        yaml.safe_dump(project, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    counts = pd.Series([device["type"] for device in devices]).value_counts().to_dict()
    print("wrote %d devices to %s: %s" % (len(devices), project_path, counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
