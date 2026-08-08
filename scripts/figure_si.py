# -*- coding: utf-8 -*-
"""Supplementary figures S1 and S2.

The previous Supplementary Information carried nine per-device validation
plates, one per Site A cooling tower and heat exchanger. Repeating that pattern
across four sites would mean roughly thirty-six plates to say what two dense
figures can say, and the reader who wants a single device is better served by
the per-device tables than by hunting through plates.

S1 replaces them. Every scored device in the study appears once, on three
aligned scales: the error actually reported, the skill of the model against a
mean predictor, and the excitation of its driving variable. Read across a row
and the reason a low error may not mean a good model becomes visible without
any prose.

S2 is the counterpart for the devices that produced no model at all. It is the
attrition ledger drawn rather than tabulated, and it makes the point Section 4.4
argues: the causes travel between sites, the sites do not.

Usage:
    python scripts/figure_si.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs" / "evidence"

SITES = ["A", "B", "C", "D"]
SITE_COLOUR = {"A": "#0F4D92", "B": "#3775BA", "C": "#93B3D7", "D": "#42949E"}
FAMILIES = ["chiller", "cooling_tower", "pump", "heat_exchanger"]
FAMILY_LABEL = {"chiller": "Chiller", "cooling_tower": "Cooling tower",
                "pump": "Pump", "heat_exchanger": "Heat exchanger"}
FAMILY_SHORT = {"chiller": "CH", "cooling_tower": "CT", "pump": "PU",
                "heat_exchanger": "HX"}

REFUSAL_ORDER = [
    ("blocked", "Channel absent"),
    ("power_calibration_blocked", "No energy balance"),
    ("blocked_insufficient_identification", "Cannot fill identification"),
    ("identification_only", "No untouched test segment"),
    ("data_limited", "Too few valid rows"),
]
REFUSAL_COLOUR = {
    "blocked": "#B64342",
    "power_calibration_blocked": "#D98C6A",
    "blocked_insufficient_identification": "#3775BA",
    "identification_only": "#93B3D7",
    "data_limited": "#767676",
}
SITE_LETTER = {"site_a": "A", "tencent": "B", "hkust": "C", "lbnl": "D"}

C_TEXT, C_MUTED, C_GRID = "#272727", "#767676", "#CFCECE"
C_WARN = "#B64342"


def style() -> None:
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams.update({
        "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5,
        "ytick.labelsize": 6, "legend.fontsize": 6.5,
        "axes.spines.right": False, "axes.spines.top": False,
        "axes.linewidth": 0.7, "axes.edgecolor": C_TEXT,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "legend.frameon": False, "text.color": C_TEXT,
        "axes.labelcolor": C_TEXT, "xtick.color": C_TEXT, "ytick.color": C_TEXT,
    })


def scored_devices() -> pd.DataFrame:
    d = pd.read_csv(EVIDENCE / "cross_site_devices_named.csv")
    d = d[d["status"] == "ok"].copy()
    d["fam_rank"] = d["equipment_type"].map({f: i for i, f in enumerate(FAMILIES)})
    d = d.sort_values(["site_letter", "fam_rank", "device_id"],
                      ascending=[True, True, True])
    return d.reset_index(drop=True)


def figure_s1() -> None:
    d = scored_devices()
    n = len(d)
    y = np.arange(n)[::-1]

    fig, axes = plt.subplots(
        1, 3, figsize=(7.2, 0.135 * n + 1.15), sharey=True,
        gridspec_kw=dict(width_ratios=[1.25, 1.0, 1.0], wspace=0.10,
                         left=0.215, right=0.985, top=0.90, bottom=0.075))
    ax_err, ax_skill, ax_exc = axes

    colours = [SITE_COLOUR[s] for s in d["site_letter"]]

    ax_err.barh(y, d["test_CVRMSE_pct"], height=0.62, color=colours,
                edgecolor="none")
    ax_err.set_xlabel("Test error (%)")
    ax_err.set_xlim(0, max(60, float(d["test_CVRMSE_pct"].max()) * 1.08))

    # skill: above unity the model is beaten by predicting the mean
    skill = d["test_skill"].to_numpy()
    ax_skill.barh(y, skill, height=0.62,
                  color=[C_WARN if v >= 1 else c for v, c in zip(skill, colours)],
                  edgecolor="none")
    ax_skill.axvline(1.0, color=C_TEXT, linewidth=0.8, zorder=3)
    ax_skill.set_xlabel("Skill vs mean predictor")
    ax_skill.set_xlim(0, max(2.6, float(np.nanmax(skill)) * 1.08))
    beaten = int((skill >= 1).sum())
    # inside the panel, low down where the bars are short, rather than above it
    # where the figure legend already sits
    ax_skill.text(0.97, 0.012,
                  f"{beaten} of {n} at or above unity:\nthe mean predictor wins",
                  transform=ax_skill.transAxes, ha="right", va="bottom",
                  fontsize=6, color=C_WARN, linespacing=1.3)

    exc = d["driver_excitation"].to_numpy()
    ax_exc.barh(y, np.nan_to_num(exc), height=0.62, color=colours,
                edgecolor="none")
    for yi, v in zip(y, exc):
        if np.isnan(v):
            ax_exc.text(0.01, yi, "no driver", va="center", ha="left",
                        fontsize=5.5, color=C_MUTED, style="italic")
    ax_exc.set_xlabel("Driver excitation")
    ax_exc.set_xlim(0, 0.92)

    labels = [f"{r.site_letter}  {r.device_id.replace('_', '-')}"
              for r in d.itertuples()]
    ax_err.set_yticks(y)
    ax_err.set_yticklabels(labels, fontsize=5.6)
    ax_err.set_ylim(-0.8, n - 0.2)
    ax_err.tick_params(axis="y", length=0, pad=2)

    # a hairline between sites, so the grouping reads without extra labels
    boundaries = d["site_letter"].ne(d["site_letter"].shift()).to_numpy()
    for ax in axes:
        for idx in np.where(boundaries)[0][1:]:
            ax.axhline(y[idx] + 0.5, color=C_GRID, linewidth=0.5)
        ax.grid(axis="x", color=C_GRID, linewidth=0.4, alpha=0.55)
        ax.set_axisbelow(True)

    handles = [Patch(facecolor=SITE_COLOUR[s], edgecolor="none", label=f"Site {s}")
               for s in SITES]
    handles.append(Patch(facecolor=C_WARN, edgecolor="none",
                         label="beaten by the mean"))
    ax_err.legend(handles=handles, loc="lower left",
                  bbox_to_anchor=(0.0, 1.005), ncol=5, handlelength=1.1,
                  columnspacing=1.0, borderpad=0.0)

    out = EVIDENCE / "figureS1_device_performance"
    for ext in ("svg", "pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=600 if ext == "png" else None,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"S1: {n} scored devices, {beaten} beaten by the mean predictor")


def figure_s2() -> None:
    a = pd.read_csv(EVIDENCE / "cross_site_attrition.csv")
    a["letter"] = a["site"].map(SITE_LETTER)
    pivot = (a.pivot_table(index="letter", columns="status", values="devices",
                           aggfunc="sum", fill_value=0)
             .reindex(SITES, fill_value=0))

    fig, ax = plt.subplots(figsize=(7.2, 2.35))
    fig.subplots_adjust(left=0.115, right=0.985, top=0.80, bottom=0.20)
    y = np.arange(len(SITES))[::-1]
    left = np.zeros(len(SITES))
    for status, label in REFUSAL_ORDER:
        if status not in pivot:
            continue
        vals = pivot[status].to_numpy(float)
        ax.barh(y, vals, left=left, height=0.55,
                color=REFUSAL_COLOUR[status], edgecolor="white", linewidth=0.6,
                label=label)
        for yi, v, l in zip(y, vals, left):
            if v >= 2:
                ax.text(l + v / 2, yi, f"{int(v)}", ha="center", va="center",
                        fontsize=6, color="white")
        left += vals

    for yi, total in zip(y, left):
        ax.text(total + 0.5, yi, f"{int(total)} refused", va="center",
                ha="left", fontsize=6.5, color=C_TEXT)

    ax.set_yticks(y)
    ax.set_yticklabels([f"Site {s}" for s in SITES])
    ax.set_ylim(-0.65, len(SITES) - 0.35)
    ax.set_xlim(0, float(left.max()) * 1.20)
    ax.set_xlabel("Devices refused")
    ax.tick_params(axis="y", length=0, pad=2)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=C_GRID, linewidth=0.4, alpha=0.55)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=3,
              handlelength=1.1, columnspacing=1.2, borderpad=0.0)

    out = EVIDENCE / "figureS2_attrition"
    for ext in ("svg", "pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=600 if ext == "png" else None,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"S2: {int(left.sum())} refused devices across {len(SITES)} sites")


def main() -> int:
    style()
    figure_s1()
    figure_s2()
    return 0


if __name__ == "__main__":
    sys.exit(main())
