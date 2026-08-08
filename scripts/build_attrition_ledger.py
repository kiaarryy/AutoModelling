"""Produce the cleaning-attrition waterfall for the Site A cooling towers.

Answers reviewer 1 comment 5 (fix M-11): the manuscript reports that 17.4% of
raw cooling-tower records survive cleaning but never says what removed the other
82.6%.  This walks the exclusions in order and attributes each one.

It also separates the two kinds of exclusion (fix M-10).  Operating-state gates
-- no flow, fans off, a channel missing -- define the domain the model claims to
cover.  Quarantined rows are physically impossible ones: water leaving the tower
hotter than it entered, or at or below the ambient wet bulb.  The published code
folded both into a single mask, which both hid the losses and conditioned the
sample on the prediction target.

Reads the attributed canonical frames written by a previous pipeline run.

Usage:
    python scripts/build_attrition_ledger.py [--run sitea_20260625_full_recalc]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from autofmu.contracts.profiles import get_profile  # noqa: E402
from autofmu.devices.cooling_tower_thermal import _features  # noqa: E402
from autofmu.observability import AttritionLedger  # noqa: E402


def generic_ledger(frame: pd.DataFrame, family: str) -> AttritionLedger:
    """Channel-availability waterfall for a family with no engine-level ledger.

    Walks the profile's required fields one at a time, then the run signal.  It
    cannot express family-specific physical impossibilities the way the cooling
    tower does -- those live in the engine -- but it answers the question the
    reviewer actually asked for every family: which requirement removed how
    much.
    """
    profile = get_profile(family)
    ledger = AttritionLedger(n_total=len(frame), family=family)
    required = profile.fmu_required or profile.full_physical_required
    for column in required:
        if column in frame.columns:
            present = np.isfinite(
                pd.to_numeric(frame[column], errors="coerce").to_numpy(float))
            ledger.gate(f"{column} present", present, category="missing_channel")
        else:
            ledger.gate(f"{column} present", np.zeros(len(frame), bool),
                        category="missing_channel")
    if profile.run_signal in frame.columns:
        running = pd.to_numeric(frame[profile.run_signal],
                                errors="coerce").fillna(0.0).to_numpy(float) > 0.5
        ledger.gate("device running", running, category="operating_state")
    return ledger


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="sitea_20260625_full_recalc")
    ap.add_argument("--family", default="cooling_tower")
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/analysis/attrition_ledger"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    run_dir = ROOT / "outputs" / "runs" / args.run / args.family
    if not run_dir.exists():
        print(f"no such run stage: {run_dir}")
        return 1

    frames, summaries = [], []
    for device_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        source = device_dir / "canonical_attributed.csv"
        if not source.exists():
            source = device_dir / "canonical.csv"
        if not source.exists():
            continue
        frame = pd.read_csv(source, low_memory=False)
        if args.family == "cooling_tower":
            # the tower engine builds its own ledger, including the physical
            # quarantine rules that only make sense for a tower
            ledger = _features(frame, {})["ledger"]
        else:
            ledger = generic_ledger(frame, args.family)
        ledger.device = device_dir.name
        frames.append(ledger.to_frame())
        summaries.append(ledger.summary())

    if not frames:
        print("no device frames found")
        return 1

    waterfall = pd.concat(frames, ignore_index=True)
    summary = pd.DataFrame(summaries)
    waterfall.to_csv(args.out / f"{args.family}_attrition_waterfall.csv", index=False)
    summary.to_csv(args.out / f"{args.family}_attrition_summary.csv", index=False)

    pd.set_option("display.width", 220)
    print("[1] Per-device summary")
    cols = ["device", "n_total", "n_retained", "retained_pct",
            "removed_by_gate", "removed_by_quarantine"]
    print(summary[cols].round(2).to_string(index=False))

    print("\n[2] Family waterfall (summed across devices)")
    agg = (waterfall[waterfall.order > 0]
           .groupby(["order", "name", "category", "kind"], as_index=False)
           .agg(removed=("removed", "sum")).sort_values("order"))
    total = int(summary.n_total.sum())
    agg["pct_of_raw"] = (100 * agg.removed / total).round(2)
    running = total
    remaining_col = []
    for r in agg.removed:
        running -= int(r)
        remaining_col.append(running)
    agg["remaining"] = remaining_col
    agg["retained_pct"] = (100 * agg.remaining / total).round(2)
    print(f"    raw records: {total}")
    print(agg.to_string(index=False))

    print("\n[3] Quarantined (physically impossible) rows by device")
    qcols = [c for c in summary.columns if c.startswith("quarantine_")]
    if qcols:
        q = summary[["device", "n_total"] + qcols].copy()
        q["quarantined_total"] = q[qcols].sum(axis=1)
        q["quarantined_pct_of_raw"] = (100 * q.quarantined_total / q.n_total).round(3)
        print(q.to_string(index=False))

    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
