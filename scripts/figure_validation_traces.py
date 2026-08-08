# -*- coding: utf-8 -*-
"""Measured against simulated, on the test segment, one device per family.

Redrawing Figure 7 around the temporal partition answered the reviewers but
removed the paper's only measured-versus-simulated evidence. A coefficient of
variation is a scalar: it cannot show bias, lag, or a model that fails in one
operating regime and compensates in another. This figure puts that evidence
back at a cost of one plate rather than the thirty-six a per-device treatment
across four sites would need.

Two honesty constraints shape it.

The series are the test segment only -- the samples withheld from both
parameter identification and candidate selection -- so what is plotted is the
same data the reported error is computed from, not a training fit.

The test segment is block-interleaved, so consecutive scored samples are not
consecutive in wall-clock time. The x axis is therefore the scored-sample index
and says so. The window rule is fixed and stated: a segment of at most 1,000 scored samples is
shown in full, and a longer one is truncated to its first 500. The quoted metric
is always for the whole segment, never for the window, so the window cannot
flatter the result.

Usage:
    python scripts/figure_validation_traces.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SERIES = ROOT / "docs" / "evidence" / "validation_series"
EVIDENCE = ROOT / "docs" / "evidence"

WINDOW = 500
SHOW_ALL_BELOW = 1000

C_MEAS = "#272727"
C_SIM = "#3775BA"
C_RESID = "#B64342"
C_GRID = "#CFCECE"
C_MUTED = "#767676"

# device -> panel title, y label, scale applied to the stored series
PANELS = [
    ("CH_03", "Site A CH-03  ·  ElectricReformulatedEIR",
     "Electrical power (kW)", 1e-3,
     "test CVRMSE 4.49%   skill 0.22   n = 282"),
    ("CT_01", "Site A CT-01  ·  YorkCalc (27-coefficient)",
     "Leaving water temperature (°C)", 1.0,
     "test RMSE 0.52 K   skill 0.098   n = 17,894"),
    ("CDWP_03", "Site A CDWP-03  ·  Cubic affinity",
     "Electrical power (kW)", 1e-3,
     "test CVRMSE 9.98%   skill 0.60   n = 790"),
    ("HX_02", "Site A HX-02  ·  PlateHeatExchangerEffectivenessNTU",
     "Heat transfer rate (kW)", 1e-3,
     "test CVRMSE 7.97%   skill 0.16   n = 9,909"),
]


def style() -> None:
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams.update({
        "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "axes.spines.right": False, "axes.spines.top": False,
        "axes.linewidth": 0.7, "axes.edgecolor": C_MEAS,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "legend.frameon": False, "text.color": C_MEAS,
        "axes.labelcolor": C_MEAS, "xtick.color": C_MEAS, "ytick.color": C_MEAS,
    })


def panel(ax, device: str, title: str, ylabel: str, scale: float,
          note: str) -> int:
    path = SERIES / f"{device}.csv"
    if not path.exists():
        raise SystemExit(f"missing series: {path}; run capture_validation_series.py")
    d = pd.read_csv(path)
    total = len(d)
    if total > SHOW_ALL_BELOW:
        d = d.iloc[:WINDOW]
    x = np.arange(len(d))
    meas = d["measured"].to_numpy(float) * scale
    sim = d["simulated"].to_numpy(float) * scale

    ax.fill_between(x, meas, sim, color=C_RESID, alpha=0.20, linewidth=0,
                    zorder=1, label="residual")
    ax.plot(x, meas, color=C_MEAS, linewidth=0.85, zorder=3, label="measured")
    ax.plot(x, sim, color=C_SIM, linewidth=0.85, zorder=2, label="simulated")

    ax.set_ylabel(ylabel)
    ax.set_xlim(0, len(d) - 1)
    ax.grid(axis="y", color=C_GRID, linewidth=0.4, alpha=0.55)
    ax.set_axisbelow(True)
    ax.text(0.0, 1.14, title, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=7.2)
    ax.text(0.0, 1.015, note, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=6, color=C_MUTED)
    # inside the axes, bottom right: above the panel it collided with the
    # metric line on the wider devices
    span = (f"first {len(d)} of {total:,} scored samples" if total > len(d)
            else f"all {total:,} scored samples")
    ax.text(0.99, 0.035, span, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=5.8, color=C_MUTED)
    return total


def main() -> int:
    style()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.885, bottom=0.10,
                        hspace=0.62, wspace=0.26)

    for ax, (dev, title, ylabel, scale, note) in zip(axes.ravel(), PANELS):
        panel(ax, dev, title, ylabel, scale, note)
    for ax in axes[1]:
        ax.set_xlabel("Scored test-segment sample")

    for ax, letter in zip(axes.ravel(), "abcd"):
        ax.text(-0.115, 1.14, letter, transform=ax.transAxes, fontsize=9,
                fontweight="bold", va="bottom", ha="right")

    handles, labels = axes[0][0].get_legend_handles_labels()
    order = [labels.index(k) for k in ("measured", "simulated", "residual")]
    fig.legend([handles[i] for i in order], ["Measured", "Simulated", "Residual"],
               loc="upper right", bbox_to_anchor=(0.985, 0.995), ncol=3,
               handlelength=1.4, columnspacing=1.3)

    out = EVIDENCE / "figure_validation_traces"
    for ext in ("svg", "pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=600 if ext == "png" else None,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}.svg / .pdf / .png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
