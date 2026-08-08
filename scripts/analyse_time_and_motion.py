"""Turn the observed June log into the effort result, and say what is missing.

The log records the manual arm only: one operator, 13 working days, 12 sampled
devices across four families plus one family setup each. There are no Auto rows,
so **no reduction figure can be computed from this file** and none is printed.
What can be computed is the manual baseline, decomposed the way the protocol
requires -- one-time family setup, marginal cost per accepted device, marginal
cost per refused device -- and extrapolated to the full Site A inventory.

Machine wall time comes from the pipeline's own run manifests and is reported
separately. It is never added to human time.

    python scripts/analyse_time_and_motion.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
LOG = REPO / "docs" / "time_and_motion" / "times_2026-06.xlsx"
OUT = REPO / "docs" / "evidence"

# Site A outcome from run sitea_e2e_20260804c
INVENTORY = {"Chiller": (7, 3, 4), "Cooling tower": (7, 4, 3),
             "Pump": (17, 10, 7), "Heat exchanger": (3, 2, 1)}


def load() -> pd.DataFrame:
    frame = pd.read_excel(LOG, "Sheet1")
    frame["min"] = pd.to_numeric(frame["Human active min"], errors="coerce")
    return frame[frame["min"].notna()].copy()


def per_family(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, group in frame.groupby("Family"):
        setup = float(group[group.Device == "FAMILY_SETUP"]["min"].sum())
        devices = (group[group.Device != "FAMILY_SETUP"]
                   .groupby(["Device", "Order"])["min"].sum()
                   .reset_index().sort_values("Order"))
        # A refused device stops at the blocker manifest, so it logs fewer
        # stages; the protocol prices it separately rather than averaging it in.
        blocked = devices[devices.Device.isin(
            group[group["Deliverable status"] == "Blocked"].Device.unique())]
        accepted = devices[~devices.Device.isin(blocked.Device)]
        d1, d2 = accepted.iloc[0], accepted.iloc[1]
        rows.append({
            "family": family,
            "setup_h": setup / 60.0,
            "device_1": d1.Device, "device_1_h": d1["min"] / 60.0,
            "device_2": d2.Device, "device_2_h": d2["min"] / 60.0,
            "accepted_marginal_h": float(accepted["min"].mean()) / 60.0,
            "blocked_device": blocked.iloc[0].Device if len(blocked) else "",
            "blocked_marginal_h": float(blocked["min"].mean()) / 60.0 if len(blocked) else 0.0,
            "learning_effect": 1.0 - d2["min"] / d1["min"],
        })
    return pd.DataFrame(rows)


def extrapolate(fam: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in fam.iterrows():
        n, acc, blk = INVENTORY[r.family]
        total = r.setup_h + acc * r.accepted_marginal_h + blk * r.blocked_marginal_h
        rows.append({"family": r.family, "inventory": n, "accepted": acc,
                     "refused": blk, "setup_h": round(r.setup_h, 2),
                     "accepted_marginal_h": round(r.accepted_marginal_h, 2),
                     "refused_marginal_h": round(r.blocked_marginal_h, 2),
                     "full_inventory_h": round(total, 1)})
    return pd.DataFrame(rows)


def machine_time() -> pd.DataFrame:
    rows = []
    for run in sorted((REPO / "outputs" / "runs").glob("*_e2e_20260804c")):
        manifest = run / "manifest.json"
        if not manifest.exists():
            continue
        m = json.loads(manifest.read_text(encoding="utf-8"))
        stages = m.get("stages", {})
        if not stages:
            continue
        times = [pd.Timestamp(v["updated_at"]) for v in stages.values()
                 if isinstance(v, dict) and v.get("updated_at")]
        start = pd.Timestamp(m["created_at"])
        rows.append({"run": run.name, "stages": len(stages),
                     "wall_min": round((max(times) - start).total_seconds() / 60.0, 1)})
    return pd.DataFrame(rows)


def main() -> int:
    frame = load()
    fam = per_family(frame)
    ext = extrapolate(fam)
    mach = machine_time()

    pd.set_option("display.width", 200)
    total_h = frame["min"].sum() / 60.0
    days = frame["Date"].nunique() if "Date" in frame else np.nan

    print("OBSERVED MANUAL ARM")
    print(f"  {len(frame)} logged stage records, {total_h:.2f} h over {days} working days")
    print(f"  operator {frame['Operator ID'].dropna().unique()[0]}, "
          f"recorder {frame['Observer ID'].dropna().unique()[0]}")
    print()
    print(fam.round(2).to_string(index=False))
    print()
    print("EXTRAPOLATION TO THE FULL SITE A INVENTORY")
    print(ext.to_string(index=False))
    print(f"  total manual baseline: {ext.full_inventory_h.sum():.1f} h "
          f"for {ext.inventory.sum()} devices")
    print()
    print("MACHINE WALL TIME (from the run manifests; never added to human time)")
    print(mach.to_string(index=False))
    print()
    print("NOT MEASURED: the automated arm's human-active minutes. No reduction")
    print("figure can be derived from this log alone, and none is printed here.")

    OUT.mkdir(parents=True, exist_ok=True)
    fam.round(3).to_csv(OUT / "time_and_motion_per_family.csv", index=False)
    ext.to_csv(OUT / "time_and_motion_extrapolation.csv", index=False)
    mach.to_csv(OUT / "time_and_motion_machine.csv", index=False)
    by_stage = (frame.groupby("Stage #")["min"].sum() / 60.0).round(2)
    by_stage.rename("manual_h").to_csv(OUT / "time_and_motion_by_stage.csv")
    print(f"\nwrote 4 CSVs to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
