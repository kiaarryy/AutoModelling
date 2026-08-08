"""Generate adapters + a fleet project.yaml for the full Tencent device set.

Introspects each raw CSV header so source column names are exact (robust to
per-device naming). Emits:
  configs/tencent/adapters/<type>_<id>.yaml   (one per device)
  configs/tencent/project_fleet.yaml          (6 per type, with reconstruction +
                                               flow attribution)

Usage (from repo root):
  python scripts/generate_tencent_fleet.py
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ["AUTOFMU_DATA_ROOT"])
ADAPTERS = REPO / "configs" / "tencent" / "adapters"

# canonical field -> (list of substrings that must all appear in the source col, scale)
SPECS = {
    "chiller": {
        "ids": [f"CHI{i:02d}" for i in range(1, 7)],
        "cols": {
            "power_W": ([".P_CHI"], 1000.0), "chw_flow_m3_h": (["mChw_flow"], 1.0),
            "cw_flow_m3_h": (["mCw_flow"], 1.0), "tchws_C": (["Tchws"], 1.0),
            "tchwr_C": (["Tchwr"], 1.0), "tcws_C": ([".Tcws_"], 1.0),
            "tcwr_C": ([".Tcwr_"], 1.0), "run_signal": ([".y_CHI"], 1.0),
            "load_pct": (["rOp"], 1.0),
        },
    },
    "cooling_tower": {
        "ids": [f"CT{i:02d}" for i in range(1, 7)],
        "cols": {
            "power_W": ([".P_CT"], 1000.0), "tcws_1_C": (["Tcws", "_1 "], 1.0),
            "tcws_2_C": (["Tcws", "_2 "], 1.0), "tcwr_C": (["Tcwr_CT"], 1.0),
            "run_signal": ([".y_CT"], 1.0), "fan1_Hz": (["fFan1"], 1.0),
            "fan2_Hz": (["fFan2"], 1.0),
        },
    },
    "pump": {
        "ids": [f"CHWP{i:02d}" for i in range(1, 7)] + [f"CWP{i:02d}" for i in range(1, 7)],
        "cols": {
            "power_W": ([".P_"], 1000.0), "run_signal": ([".y_"], 1.0), "speed_Hz": ([".f_"], 1.0),
        },
    },
    "heat_exchanger": {
        "ids": [f"HEX{i:02d}" for i in range(1, 7)],
        "cols": {
            "tchws_C": (["Tchws"], 1.0), "tchwr_C": (["Tchwr"], 1.0),
            "tcwr_C": (["Tcwr"], 1.0), "tcws_C": (["Tcws"], 1.0),
            "chw_valve_pct": (["yValChwIn"], 1.0), "cw_valve_pct": (["yValCwIn"], 1.0),
        },
    },
}


def _header(csv_path: Path):
    with open(csv_path, encoding="utf-8-sig") as fh:
        return next(csv.reader(fh))


def _find(header, parts):
    hits = [c for c in header if all(p in c for p in parts)]
    if len(hits) != 1:
        return None
    return hits[0]


def main():
    ADAPTERS.mkdir(parents=True, exist_ok=True)
    devices = []
    missing = []
    for etype, spec in SPECS.items():
        for dev_id in spec["ids"]:
            csv_name = f"{dev_id}_202203_202303.csv"
            csv_path = DATA / csv_name
            if not csv_path.exists():
                missing.append(dev_id)
                continue
            header = _header(csv_path)
            ts = _find(header, ["DateTime"]) or "DateTime"
            columns = {}
            for canon, (parts, scale) in spec["cols"].items():
                src = _find(header, parts)
                if src is None:
                    continue
                entry = {"source": src}
                if scale != 1.0:
                    entry["scale"] = scale
                columns[canon] = entry
            adapter = {"source_csv": csv_name, "timestamp": ts, "columns": columns}
            adapter_name = f"{etype}_{dev_id.lower()}.yaml"
            (ADAPTERS / adapter_name).write_text(
                yaml.safe_dump(adapter, sort_keys=False, allow_unicode=True), encoding="utf-8"
            )
            device = {"id": dev_id, "type": etype, "adapter": adapter_name}
            if etype == "chiller":
                device["reconstruct"] = {
                    "cooling_load_W": {"method": "chw_energy_balance", "flow": "chw_flow_m3_h",
                                       "t_hot": "tchwr_C", "t_cold": "tchws_C"},
                    "heat_rejection_W": {"method": "cw_energy_balance", "flow": "cw_flow_m3_h",
                                         "t_hot": "tcwr_C", "t_cold": "tcws_C"},
                    "power_W": {"method": "energy_balance_power", "condenser": "heat_rejection_W",
                                "evaporator": "cooling_load_W", "verify_cop": [3.0, 9.0]},
                }
            elif etype == "cooling_tower":
                device["flow_attribution"] = {"source_type": "chiller", "source_column": "cw_flow_m3_h"}
                device["reconstruct"] = {
                    "heat_rejection_W": {"method": "heat_rate", "flow": "attributed_flow_m3_h",
                                         "t_hot": "tcwr_C", "t_cold": "tcws_1_C"},
                }
            elif dev_id.startswith("CWP"):
                device["group"] = "cwp"
                device["flow_attribution"] = {"source_type": "chiller", "source_column": "cw_flow_m3_h"}
            elif dev_id.startswith("CHWP"):
                device["group"] = "chwp"
                device["flow_attribution"] = {"source_type": "chiller", "source_column": "chw_flow_m3_h"}
            devices.append(device)

    project = {
        "dataset": "tencent",
        "data_root": "${AUTOFMU_DATA_ROOT}",
        "adapters_dir": "adapters",
        "outputs_dir": "../../outputs",
        "fmu_config_dir": "../fmu",
        "thresholds": {"run_on": 0.5, "min_full_physical_rows": 500, "min_nominal_rows": 200,
                       "validation_split": "interleaved", "validation_fold": 3},
        "devices": devices,
    }
    out = REPO / "configs" / "tencent" / "project_fleet.yaml"
    out.write_text(yaml.safe_dump(project, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"wrote {len(devices)} devices, {len(list(ADAPTERS.glob('*.yaml')))} adapters")
    if missing:
        print("missing CSVs:", missing)
    print("project:", out)


if __name__ == "__main__":
    main()
