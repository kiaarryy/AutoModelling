"""Apply the three-way temporal split to the Site A cooling towers.

Produces the segment definitions and the comparability evidence that replace the
manuscript's 2%-holdout scheme (fix M-09).  Two artefacts:

``site_a_ct_split_counts.csv``
    Samples per segment per tower, plus the resulting status.  Towers whose
    record cannot support an independent test segment are marked
    ``identification_only`` rather than being given a test score.

``site_a_ct_split_coverage.csv``
    Wet-bulb, range and flow statistics per segment.  This is the evidence that
    the test segment exercises conditions the identification segment also saw --
    without it, a seasonal split is just a different way of extrapolating.

Timestamps are reconstructed from each tower's ``*_time_mapping.csv``
(``source_time_s``, elapsed seconds) anchored on the first timestamp of the raw
adapter source.

Usage:
    python scripts/build_site_a_temporal_split.py [--buffer-hours 72]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from autofmu.temporal_split import SplitSpec, build_split  # noqa: E402

FMU_MODELICA = Path(os.environ.get(
    "AUTOFMU_FMU_MODELICA_ROOT", "__missing_external_fmu_root__"))
TABLES = FMU_MODELICA / "outputs" / "cooling_tower" / "auto_model_site_a_full_period" / "tables"
SITE_A_DATA = Path(os.environ.get(
    "AUTOFMU_SITE_A_DATA_ROOT", "__missing_site_a_data_root__"))
ADAPTERS = ROOT / "configs" / "site_a" / "adapters"

TOWERS = [f"CT_0{i}" for i in range(1, 8)]
COVERAGE_VARIABLES = ["Twb_C", "TRan_C", "mdot_cell_kgps", "y_used", "Tin_C",
                      "flow_m3_h"]


def record_start(tower: str) -> pd.Timestamp | None:
    adapter = ADAPTERS / f"cooling_tower_{tower.lower()}.yaml"
    if not adapter.exists():
        return None
    spec = yaml.safe_load(adapter.read_text(encoding="utf-8")) or {}
    csv = SITE_A_DATA / str(spec.get("source_csv", ""))
    if not csv.exists():
        return None
    col = spec.get("timestamp", "DateTime")
    stamps = pd.to_datetime(pd.read_csv(csv, usecols=[col])[col], errors="coerce")
    return stamps.min()


def autofmu_table(run: str, tower: str):
    """Retained rows of one tower, from an autofmu run's attributed frame.

    Uses the same feature builder the calibration path uses, so the split is
    computed over exactly the rows that will be modelled -- not over the raw
    record, where gaps would put buffers in the wrong places.
    """
    from autofmu.devices.cooling_tower_thermal import _features

    source = (ROOT / "outputs" / "runs" / run / "cooling_tower" / tower
              / "canonical_attributed.csv")
    if not source.exists():
        return None
    frame = pd.read_csv(source, low_memory=False)
    keep = _features(frame, {})["valid"]
    if not keep.any():
        return None
    table = frame.loc[keep].reset_index(drop=True)
    table["DateTime"] = pd.to_datetime(table["timestamp"], errors="coerce", utc=True)
    table["DateTime"] = table["DateTime"].dt.tz_localize(None)
    return table.rename(columns={"twb_C": "Twb_C", "tcwr_C": "Tin_C",
                                 "attributed_flow_m3_h": "flow_m3_h"})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--buffer-hours", type=float, default=72.0)
    ap.add_argument("--run", default=None,
                    help="use an autofmu run's attributed canonical frames "
                         "instead of the FMU_Modelica prepared tables")
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/analysis/site_a_temporal_split"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    spec = SplitSpec(buffer_hours=args.buffer_hours)
    count_rows, coverage_frames = [], []

    for tower in TOWERS:
        if args.run:
            table = autofmu_table(args.run, tower)
            if table is None:
                continue
        else:
            table_path = TABLES / f"{tower}_full_period_table.csv"
            mapping_path = TABLES / f"{tower}_full_period_time_mapping.csv"
            if not (table_path.exists() and mapping_path.exists()):
                continue
            table = pd.read_csv(table_path)
            elapsed = pd.read_csv(mapping_path)["source_time_s"].to_numpy(float)
            start = record_start(tower)
            if start is None:
                continue
            n = min(len(table), len(elapsed))
            table = table.iloc[:n].reset_index(drop=True)
            table["DateTime"] = start + pd.to_timedelta(elapsed[:n], unit="s")
        n = len(table)

        result = build_split(table["DateTime"], spec)
        counts = result.counts()
        count_rows.append(dict(
            tower=tower, rows=n,
            span_days=round((table["DateTime"].max()
                             - table["DateTime"].min()).total_seconds() / 86400, 1),
            **counts, status=result.status, usable=result.usable,
            warnings="; ".join(result.warnings)))

        cov = result.comparability(table, COVERAGE_VARIABLES)
        cov.insert(0, "tower", tower)
        coverage_frames.append(cov)

    counts_df = pd.DataFrame(count_rows)
    coverage_df = pd.concat(coverage_frames, ignore_index=True) if coverage_frames \
        else pd.DataFrame()
    counts_df.to_csv(args.out / "site_a_ct_split_counts.csv", index=False)
    coverage_df.to_csv(args.out / "site_a_ct_split_coverage.csv", index=False)

    pd.set_option("display.width", 220)
    print(f"buffer = {args.buffer_hours} h\n")
    print("[1] Samples per segment")
    print(counts_df[["tower", "rows", "span_days", "identification", "selection",
                     "test", "buffer", "unused", "status"]].to_string(index=False))

    scored = counts_df[counts_df.usable]
    print(f"\n    towers with an independent test segment: "
          f"{len(scored)}/{len(counts_df)}")
    if len(scored) < len(counts_df):
        for _, r in counts_df[~counts_df.usable].iterrows():
            print(f"    {r.tower}: {r.status} ({r.rows} valid rows)")

    print("\n[2] Wet-bulb coverage per segment (comparability evidence)")
    twb = coverage_df[coverage_df.variable == "Twb_C"]
    if not twb.empty:
        pivot = twb.pivot_table(index="tower", columns="segment",
                                values=["minimum", "maximum"])
        print(pivot.round(2).to_string())

    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
