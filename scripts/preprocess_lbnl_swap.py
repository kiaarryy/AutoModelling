"""Correct the LBNL chiller-plant archive's swapped outdoor temperature columns.

The published file reports `OA_TEMP_WB > OA_TEMP` on 99.31% of rows, which no
wet-bulb temperature does. Left alone, every cooling-tower approach temperature
computed from it is meaningless.

This script does one thing and says so: it exchanges those two columns row-wise
and leaves the other 76 untouched, so a reader reproducing our results can see
exactly what we changed rather than receiving a silently repaired copy.

    python scripts/preprocess_lbnl_swap.py IN.csv OUT.csv
    python scripts/preprocess_lbnl_swap.py --resample 30 IN.csv OUT.csv

With --resample N it keeps every Nth row instead, which is how
ChillerPlant_30min.csv is produced from the 1-minute preprocessed file.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

DRY = "OA_TEMP"
WET = "OA_TEMP_WB"


def swap(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in (DRY, WET) if c not in frame.columns]
    if missing:
        raise SystemExit(f"columns not found: {missing}")
    out = frame.copy()
    out[DRY], out[WET] = frame[WET].to_numpy(), frame[DRY].to_numpy()
    return out


def report(frame: pd.DataFrame, label: str) -> int:
    dry = pd.to_numeric(frame[DRY], errors="coerce")
    wet = pd.to_numeric(frame[WET], errors="coerce")
    both = dry.notna() & wet.notna()
    bad = int((wet[both] > dry[both]).sum())
    print(f"  {label}: wet bulb above dry bulb on {bad} of {int(both.sum())} rows"
          f" ({100.0 * bad / max(1, int(both.sum())):.2f}%)")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source")
    ap.add_argument("destination")
    ap.add_argument("--resample", type=int, default=0, metavar="N",
                    help="keep every Nth row instead of swapping columns")
    args = ap.parse_args(argv)

    src, dst = Path(args.source), Path(args.destination)
    if not src.exists():
        raise SystemExit(f"no such file: {src}")
    frame = pd.read_csv(src, low_memory=False)
    print(f"read {len(frame):,} rows x {len(frame.columns)} columns from {src.name}")

    if args.resample:
        out = frame.iloc[:: args.resample].reset_index(drop=True)
        print(f"kept every {args.resample}th row -> {len(out):,} rows")
    else:
        before = report(frame, "before")
        out = swap(frame)
        after = report(out, "after ")
        if after and after > before * 0.01:
            print("WARNING: violations remain after the swap; this file may not "
                  "be the archive this script expects", file=sys.stderr)

    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False)
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
