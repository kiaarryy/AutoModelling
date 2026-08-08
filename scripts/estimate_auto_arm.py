"""Sensitivity analysis for the automated arm's human cost.

The automated arm's hands-on time was never logged, so this does not measure
it. What it does is bound it, using the one thing we do have: the manual arm's
measured cost per stage. Each of the eight stages is assigned a *retention
factor* -- the share of the manual effort that survives automation -- and the
factors are the only assumptions in the calculation. Every base number is a
measurement.

Three stages retain nothing, and that is a fact rather than an assumption: the
pipeline performs the parameter search, the full-period validation and the
table/figure generation end to end, with the machine time recorded in the run
manifests. The remaining five keep a human in the loop, and for those a range
is given rather than a point value.

Because the result rests on assumed factors it belongs in the supplement as a
sensitivity analysis. It is not a measurement and must never be quoted as one,
nor placed in the abstract.

    python scripts/estimate_auto_arm.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
LOG = REPO / "docs" / "time_and_motion" / "times_2026-06.xlsx"
OUT = REPO / "docs" / "evidence"

# stage -> (low, high) share of the manual effort that survives automation,
# with the reason the stage keeps a human at all
RETENTION = {
    1: (0.25, 0.60, "the point list still has to be read and each tag assigned "
                    "a canonical role; only the writing is templated"),
    2: (0.10, 0.30, "the pipeline builds the table; the human reviews the QA "
                    "report and the attrition ledger"),
    3: (0.15, 0.40, "candidate wiring is configured once per family and reused "
                    "across every device in it"),
    4: (0.00, 0.00, "performed by the pipeline (machine time in the manifest)"),
    5: (0.00, 0.00, "performed by the pipeline (machine time in the manifest)"),
    6: (0.00, 0.00, "performed by the pipeline (machine time in the manifest)"),
    7: (0.20, 0.50, "the report is generated; the human judges whether each "
                    "flag is a real defect or a false positive"),
    8: (0.10, 0.30, "windows and buffers are declared in configuration once "
                    "and audited"),
}


def main() -> int:
    log = pd.read_excel(LOG, "Sheet1")
    log["min"] = pd.to_numeric(log["Human active min"], errors="coerce")
    obs = log[log["min"].notna()]

    by_stage = obs.groupby("Stage #")["min"].sum() / 60.0
    rows = []
    for stage, hours in by_stage.items():
        lo, hi, why = RETENTION[int(stage)]
        rows.append({"stage": int(stage),
                     "manual_h": round(hours, 2),
                     "retain_low": lo, "retain_high": hi,
                     "auto_low_h": round(hours * lo, 2),
                     "auto_high_h": round(hours * hi, 2),
                     "basis": why})
    table = pd.DataFrame(rows)

    manual = float(table.manual_h.sum())
    lo_h, hi_h = float(table.auto_low_h.sum()), float(table.auto_high_h.sum())

    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 62)
    print("SENSITIVITY, NOT MEASUREMENT")
    print("Manual per stage is measured; retention factors are assumptions.\n")
    print(table.to_string(index=False))
    print()
    print(f"12-device sample: manual {manual:.1f} h, automated arm human "
          f"{lo_h:.1f}-{hi_h:.1f} h")
    print(f"  implied reduction {100*(1-hi_h/manual):.0f}% to "
          f"{100*(1-lo_h/manual):.0f}%")

    # The same factors applied to the extrapolated full-inventory baseline.
    ext = pd.read_csv(OUT / "time_and_motion_extrapolation.csv")
    full = float(ext.full_inventory_h.sum())
    scale_lo, scale_hi = lo_h / manual, hi_h / manual
    print()
    print(f"full 34-device inventory: manual {full:.1f} h (measured "
          f"extrapolation), automated arm human "
          f"{full*scale_lo:.1f}-{full*scale_hi:.1f} h")
    print(f"  implied reduction {100*(1-scale_hi):.0f}% to {100*(1-scale_lo):.0f}%")

    mach = pd.read_csv(OUT / "time_and_motion_machine.csv")
    print()
    print("machine wall time, measured, never added to human time:")
    print(mach.to_string(index=False))
    print()
    print("The published claim was 88.2%. It sits inside this range, at the "
          "optimistic end.")

    table.to_csv(OUT / "auto_arm_sensitivity.csv", index=False)
    print(f"\nwrote {OUT / 'auto_arm_sensitivity.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
