"""Consolidate the four site runs into the cross-site evidence for Results §3.5.

Reads each run's ``attribute/modelability_report.csv`` and
``calibrate/selected_models.csv`` and writes, under ``docs/evidence/``:

  cross_site_summary.csv        per site x family: devices, scored, median error
  cross_site_attrition.csv      per site: why devices were not scored
  cross_site_observability.csv  per device: the observability flags it carries
  cross_site_devices.csv        per device: score next to its driver excitation
  cross_site_summary.svg        the figure

Usage:
    python scripts/build_cross_site_summary.py \
        --run site_a=sitea_e2e_20260803e --run tencent=tencent_e2e_20260803b \
        --run hkust=hkust_e2e_20260803c --run lbnl=lbnl_e2e_20260803c
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
EVIDENCE = REPO / "docs" / "evidence"
FAMILIES = ("chiller", "cooling_tower", "pump", "heat_exchanger")
SITE_ORDER = ("site_a", "tencent", "hkust", "lbnl")


def _load(run_id: str):
    base = REPO / "outputs" / "runs" / run_id
    gate = pd.read_csv(base / "attribute" / "modelability_report.csv")
    sel = pd.read_csv(base / "calibrate" / "selected_models.csv")
    return gate, sel


def summarise(runs: dict) -> tuple:
    rows, attrition, observability, devices = [], [], [], []
    for site, run_id in runs.items():
        gate, sel = _load(run_id)
        # both carry `level`; the gate is the authority on it, so drop the copy
        # rather than let the merge rename it out from under the callers below
        sel = sel.drop(columns=[c for c in ("level", "equipment_type") if c in sel])
        merged = gate.merge(sel, on="device_id", how="left")
        for _, r in merged.iterrows():
            # A score and the excitation behind it belong on the same line:
            # Site A CDWP_01 reports 5.93% on a drive that spans 0.1% of its
            # own median, and the two numbers only mean something together.
            devices.append({
                "site": site, "device_id": r["device_id"],
                "equipment_type": r["equipment_type"], "level": r["level"],
                "status": r.get("status"), "candidate": r.get("selected_candidate"),
                "test_CVRMSE_pct": r.get("test_CVRMSE_pct"),
                "test_skill": r.get("test_skill"),
                "selection_skill": r.get("selection_skill"),
                "driver": r.get("driver"),
                "driver_excitation": r.get("driver_excitation"),
                "on_rows": r.get("on_rows"),
                "flags": r.get("flags"),
            })
            flags = str(r.get("flags") or "")
            if flags and flags.lower() != "nan":
                for flag in flags.split(","):
                    observability.append({"site": site, "device_id": r["device_id"],
                                          "equipment_type": r["equipment_type"],
                                          "flag": flag.strip()})
        for family in FAMILIES:
            block = merged[merged["equipment_type"] == family]
            if block.empty:
                continue
            scored = block[block["status"] == "ok"]
            errors = pd.to_numeric(scored.get("test_CVRMSE_pct"), errors="coerce").dropna()
            synthetic = int(block["flags"].fillna("").str.contains("synthetic_channel").sum())
            rows.append({
                "site": site, "family": family,
                "devices": int(len(block)), "scored": int(len(scored)),
                "median_test_CVRMSE": round(float(errors.median()), 2) if len(errors) else np.nan,
                "blocked_synthetic": synthetic,
            })
            for status, n in block["status"].fillna("(not reached)").value_counts().items():
                if status == "ok":
                    continue
                attrition.append({"site": site, "family": family,
                                  "status": status, "devices": int(n)})
    order = {s: i for i, s in enumerate(SITE_ORDER)}
    summary = pd.DataFrame(rows).sort_values(
        ["site", "family"], key=lambda c: c.map(order) if c.name == "site" else c)
    return (summary, pd.DataFrame(attrition), pd.DataFrame(observability),
            pd.DataFrame(devices))


def figure(summary: pd.DataFrame, path: Path) -> None:
    """Two panels: what fraction of each fleet reached an independent test
    score, and the median error of those that did."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sites = [s for s in SITE_ORDER if s in set(summary["site"])]
    fams = [f for f in FAMILIES if f in set(summary["family"])]
    colours = {"chiller": "#2c6fbb", "cooling_tower": "#e07b39",
               "pump": "#5aa469", "heat_exchanger": "#a05195"}
    labels = {"chiller": "Chiller", "cooling_tower": "Cooling tower",
              "pump": "Pump", "heat_exchanger": "Heat exchanger"}
    site_labels = {"site_a": "Site A", "tencent": "Tencent",
                   "hkust": "HKUST", "lbnl": "LBNL (public)"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    width = 0.8 / max(1, len(fams))
    x = np.arange(len(sites), dtype=float)

    for i, fam in enumerate(fams):
        share, err = [], []
        for site in sites:
            row = summary[(summary.site == site) & (summary.family == fam)]
            if row.empty:
                share.append(np.nan); err.append(np.nan); continue
            devices = float(row["devices"].iloc[0])
            share.append(100.0 * float(row["scored"].iloc[0]) / devices if devices else np.nan)
            err.append(float(row["median_test_CVRMSE"].iloc[0]))
        pos = x - 0.4 + width * (i + 0.5)
        ax1.bar(pos, share, width * 0.92, label=labels[fam], color=colours[fam])
        ax2.bar(pos, err, width * 0.92, color=colours[fam])

    for ax, title, ylabel in ((ax1, "Devices reaching an independent test score",
                               "% of fleet"),
                              (ax2, "Median test error of those devices",
                               "CVRMSE (%)")):
        ax.set_xticks(x)
        ax.set_xticklabels([site_labels.get(s, s) for s in sites])
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
    ax1.set_ylim(0, 105)
    ax1.legend(fontsize=8, frameon=False, ncol=2)
    # A missing bar in the right panel means nothing was scored, not zero error.
    ax2.text(0.99, 0.96, "no bar = nothing scored", transform=ax2.transAxes,
             ha="right", va="top", fontsize=7.5, color="#666666")
    fig.tight_layout()
    fig.savefig(path, format="svg")
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="SITE=RUN_ID",
                        help="repeatable, e.g. --run site_a=sitea_e2e_20260803e")
    args = parser.parse_args()
    runs = {}
    for item in args.run:
        site, _, run_id = item.partition("=")
        if not run_id:
            raise SystemExit(f"--run must be SITE=RUN_ID (got {item!r})")
        runs[site.strip()] = run_id.strip()

    summary, attrition, observability, devices = summarise(runs)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    summary.to_csv(EVIDENCE / "cross_site_summary.csv", index=False)
    attrition.to_csv(EVIDENCE / "cross_site_attrition.csv", index=False)
    observability.to_csv(EVIDENCE / "cross_site_observability.csv", index=False)
    devices.to_csv(EVIDENCE / "cross_site_devices.csv", index=False)
    figure(summary, EVIDENCE / "cross_site_summary.svg")

    pd.set_option("display.width", 200)
    print(summary.to_string(index=False))
    print()
    totals = summary.groupby("site", sort=False)[["devices", "scored"]].sum()
    totals["pct"] = (100.0 * totals["scored"] / totals["devices"]).round(1)
    print(totals.to_string())
    span = pd.to_numeric(devices.driver_excitation, errors="coerce")
    unexcited = devices[(devices.status == "ok") & (span < 0.10)]
    skill = pd.to_numeric(devices.test_skill, errors="coerce")
    no_skill = devices[(devices.status == "ok") & (skill >= 1.0)]
    if not no_skill.empty:
        print()
        print("beaten by predicting the measured mean (skill >= 1):")
        print(no_skill[["site", "device_id", "equipment_type", "candidate",
                        "test_CVRMSE_pct", "test_skill",
                        "driver_excitation"]].to_string(index=False))
    if not unexcited.empty:
        print()
        print("scored on a driver that barely moved:")
        print(unexcited[["site", "device_id", "candidate", "test_CVRMSE_pct",
                         "driver", "driver_excitation"]].to_string(index=False))
    if not observability.empty:
        print()
        print(observability.groupby(["site", "flag"]).size()
              .rename("devices").reset_index().to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
