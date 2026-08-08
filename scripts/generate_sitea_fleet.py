"""Generate adapters + a full Site A project config (all devices).

7 chillers + 7 cooling towers + 9 pumps (CDWP/CHWP/HXCWP) + 3 heat exchangers.
Introspects each raw CSV header so source column names are exact. Run from repo root:
    python scripts/generate_sitea_fleet.py
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ["AUTOFMU_DATA_ROOT"])
ADAP = REPO / "configs" / "site_a" / "adapters"
TWB = "WET_BALL/OUTDOOR/Local_IO.RF1-OUTDOOR_202405_202504_with_Twb.csv"

# canonical -> (substrings all present, scale)
SPECS = {
    "chiller": {"ids": [f"CH_{i:02d}" for i in range(1, 8)], "sub": "chiller", "cols": {
        "power_W": (["P/kw"], 1000.0), "chw_flow_m3_h": (["CHW-WFM"], 3.6), "cw_flow_m3_h": (["CDW-WFM"], 3.6),
        "tchws_C": (["CHWS-WTS"], 1.0), "tchwr_C": (["CHWR-WTS"], 1.0), "tcws_C": (["CDWS-WTS"], 1.0),
        "tcwr_C": (["CDWR-WTS"], 1.0), "run_signal": (["VSD-STS"], 1.0)}},
    "cooling_tower": {"ids": [f"CT_{i:02d}" for i in range(1, 8)], "sub": "cooling_tower", "cols": {
        "power_W": (["P/kw"], 1000.0), "fan1_Hz": (["A-VSD-STS"], 1.0), "fan2_Hz": (["B-VSD-STS"], 1.0),
        "tcwr_C": (["CDWR-WTS"], 1.0), "tcws_1_C": (["CDWS-WTS"], 1.0), "run_signal": (["A-VSD-STS"], 1.0)}},
    "pump": {"ids": [f"CDWP_{i:02d}" for i in range(1, 4)] + [f"CHWP_{i:02d}" for i in range(1, 4)]
             + [f"HXCWP_{i:02d}" for i in range(1, 4)], "sub": None, "cols": {
        "power_W": (["P/kw"], 1000.0), "speed_Hz": (["VSD-STS"], 1.0), "run_signal": (["VSD-STS"], 1.0)}},
    "heat_exchanger": {"ids": [f"HX_{i:02d}" for i in range(1, 4)], "sub": "heat_exchanger", "cols": {
        "tchws_C": (["CHWS-SWTS"], 1.0), "tchwr_C": (["CHWS-RWTS"], 1.0), "tcws_C": (["CDWS-SWTS"], 1.0),
        "tcwr_C": (["CDWS-RWTS"], 1.0), "chw_flow_m3_h": (["CHWS-WFM"], 3.6), "cw_flow_m3_h": (["CDWS-WFM"], 3.6)}},
}
PUMP_DIR = {"CDWP": "pump", "CHWP": "pump", "HXCWP": "heat_exchanger"}


def _find(header, parts):
    hits = [c for c in header if all(p in c for p in parts)]
    return hits[0] if len(hits) == 1 else None


def main():
    ADAP.mkdir(parents=True, exist_ok=True)
    devices = []
    for etype, spec in SPECS.items():
        for dev_id in spec["ids"]:
            sub = spec["sub"] or PUMP_DIR[dev_id.split("_")[0]]
            csv_path = DATA / sub / f"{dev_id}.csv"
            if not csv_path.exists():
                continue
            with open(csv_path, encoding="utf-8-sig", errors="replace") as fh:
                header = next(csv.reader(fh))
            columns = {}
            for canon, (parts, scale) in spec["cols"].items():
                src = _find(header, parts)
                if src is None:
                    continue
                e = {"source": src}
                if scale != 1.0:
                    e["scale"] = scale
                columns[canon] = e
            name = f"{etype}_{dev_id.lower()}.yaml"
            (ADAP / name).write_text(yaml.safe_dump(
                {"source_csv": f"{sub}/{dev_id}.csv", "timestamp": "DateTime", "columns": columns},
                sort_keys=False, allow_unicode=True), encoding="utf-8")
            dev = {"id": dev_id, "type": etype, "adapter": name}
            if etype == "cooling_tower":
                dev["flow_attribution"] = {"source_type": "chiller", "source_column": "cw_flow_m3_h"}
                dev["join"] = [{"source_csv": TWB, "timestamp": "DateTime", "column": "Twb (°C)", "as": "twb_C"}]
                dev["reconstruct"] = {
                    "fans_on_count": {"method": "active_count", "signals": ["fan1_Hz", "fan2_Hz"], "threshold": 5.0},
                    "heat_rejection_W": {"method": "heat_rate", "flow": "attributed_flow_m3_h", "t_hot": "tcwr_C", "t_cold": "tcws_1_C"}}
            elif dev_id.startswith("CDWP") or dev_id.startswith("HXCWP"):
                dev["group"] = "cdwp" if dev_id.startswith("CDWP") else "hxcwp"
                dev["flow_attribution"] = {"source_type": "chiller", "source_column": "cw_flow_m3_h"}
            elif dev_id.startswith("CHWP"):
                dev["group"] = "chwp"
                dev["flow_attribution"] = {"source_type": "chiller", "source_column": "chw_flow_m3_h"}
            devices.append(dev)

    project = {"dataset": "site_a", "data_root": "${AUTOFMU_DATA_ROOT}", "adapters_dir": "adapters",
               "outputs_dir": "../../outputs", "fmu_config_dir": "../fmu",
               "thresholds": {"run_on": 0.5, "min_full_physical_rows": 500, "min_nominal_rows": 200,
                              "min_calibration_rows": 200, "validation_fold": 3, "validation_split": "interleaved"},
               "devices": devices}
    out = REPO / "configs" / "site_a" / "project_fleet.yaml"
    out.write_text(yaml.safe_dump(project, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"wrote {len(devices)} devices -> {out}")
    from collections import Counter
    print(Counter(d["type"] for d in devices))


if __name__ == "__main__":
    main()
