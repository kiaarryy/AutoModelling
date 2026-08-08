# -*- coding: utf-8 -*-
"""Figure 6 -- what four sites do to the framework.

The published Figure 6 compared candidate models on Site A alone, which was the
whole study at the time. The study now spans four archives, and the earlier
stand-in for that breadth was a pair of default bar charts that printed the
operators' names on the axis. Both problems are fixed here.

(a) is the hero: the outcome matrix. Every device the framework was pointed at,
    by site and family, and how many of them reached an independent test score.
    A reader should be able to find the 58 of 112 and see immediately that the
    losses are concentrated, not spread evenly.

(b) is the candidate-selection evidence, drawn as a diverging bar so that the
    within-site split is visible rather than inferred. If a family were always
    won by one formulation its bar would sit entirely on one side; none does,
    and each side draws on more than one site.

(c) is the accuracy, on the one metric that is comparable across families --
    a ratio-scale error. Families that were never scored at a site are marked
    as such rather than left as an empty slot, because "not attempted",
    "attempted and refused" and "scored badly" are different facts.

Usage:
    python scripts/figure_cross_site.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs" / "evidence"

SITES = ["A", "B", "C", "D"]
SITE_COLOUR = {"A": "#0F4D92", "B": "#3775BA", "C": "#93B3D7", "D": "#42949E"}
SITE_NOTE = {"A": "Hyperscale\ndata centre", "B": "Commercial\ndata centre",
             "C": "University\ncampus", "D": "Public\ndataset"}

FAMILIES = ["chiller", "cooling_tower", "pump", "heat_exchanger"]
FAMILY_LABEL = {"chiller": "Chiller", "cooling_tower": "Cooling tower",
                "pump": "Pump", "heat_exchanger": "Heat exchanger"}

# the two candidates each family chooses between, in a fixed left/right order
PAIR = {
    "chiller": ("EIR", "EEIR"),
    "cooling_tower": ("Merkel", "YorkCalc"),
    "pump": ("affinity", "speed_poly"),
    "heat_exchanger": ("ConstantEffectiveness", "PlateEffectivenessNTU"),
}
PAIR_LABEL = {
    "EIR": "ElectricEIR", "EEIR": "ElectricReformulatedEIR",
    "Merkel": "Merkel", "YorkCalc": "YorkCalc (27-coeff.)",
    "affinity": "Cubic affinity", "speed_poly": "Speed polynomial",
    "ConstantEffectiveness": "ConstantEff.",
    "PlateEffectivenessNTU": "PlateEff.-NTU",
}

C_TEXT, C_MUTED, C_GRID = "#272727", "#767676", "#CFCECE"
C_EMPTY = "#F2F2F2"


def style() -> None:
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams.update({
        "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "axes.spines.right": False, "axes.spines.top": False,
        "axes.linewidth": 0.7, "axes.edgecolor": C_TEXT,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "legend.frameon": False, "text.color": C_TEXT,
        "axes.labelcolor": C_TEXT, "xtick.color": C_TEXT, "ytick.color": C_TEXT,
    })


def panel_a(ax, read: pd.DataFrame) -> None:
    """Scored over attempted, by site and family."""
    grid = np.full((len(SITES), len(FAMILIES)), np.nan)
    text = {}
    for i, s in enumerate(SITES):
        for j, f in enumerate(FAMILIES):
            row = read[(read["site"] == s) & (read["family"] == f)]
            if row.empty:
                text[(i, j)] = "—"
                continue
            n, k = int(row["devices"].iloc[0]), int(row["scored"].iloc[0])
            grid[i, j] = k / n if n else np.nan
            text[(i, j)] = f"{k}/{n}"

    # 0/6 means attempted and refused; a blank cell means the family is not
    # installed. They must not look alike, so zero gets a visible tint and
    # absence gets white.
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "scored", ["#DCE8F4", "#A9C6E4", "#5B90C8", "#0F4D92"])
    masked = np.ma.masked_invalid(grid)
    cmap.set_bad("#FFFFFF")
    ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    for (i, j), label in text.items():
        val = grid[i, j]
        colour = "white" if (not np.isnan(val) and val > 0.62) else C_TEXT
        ax.text(j, i, label, ha="center", va="center", fontsize=7.5,
                color=colour if label != "—" else C_MUTED)

    ax.set_xticks(range(len(FAMILIES)))
    ax.set_xticklabels([FAMILY_LABEL[f] for f in FAMILIES])
    ax.set_yticks(range(len(SITES)))
    ax.set_yticklabels([f"Site {s}" for s in SITES])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(FAMILIES), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(SITES), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.6)
    ax.tick_params(which="minor", length=0)

    # site totals on the right, so the 58 of 112 is readable off the figure
    for i, s in enumerate(SITES):
        g = read[read["site"] == s]
        n, k = int(g["devices"].sum()), int(g["scored"].sum())
        ax.text(len(FAMILIES) - 0.32, i, f"{k}/{n}", ha="left", va="center",
                fontsize=7, color=C_TEXT, fontweight="bold",
                transform=ax.transData)
        ax.text(len(FAMILIES) + 0.30, i, SITE_NOTE[s], ha="left", va="center",
                fontsize=5.8, color=C_MUTED, linespacing=1.15)
    ax.text(len(FAMILIES) - 0.32, -0.72, "Site total", ha="left", va="center",
            fontsize=6, color=C_MUTED)
    ax.set_xlim(-0.5, len(FAMILIES) + 1.35)

    total_n = int(read["devices"].sum())
    total_k = int(read["scored"].sum())
    ax.text(1.0, 1.045, f"{total_k} of {total_n} devices scored",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5,
            color=C_MUTED)


def panel_b(ax, devices: pd.DataFrame) -> None:
    """Which candidate won, per family, split by site."""
    ok = devices[devices["status"] == "ok"]
    y = np.arange(len(FAMILIES))[::-1]
    for yi, fam in zip(y, FAMILIES):
        left_c, right_c = PAIR[fam]
        for sign, cand in ((-1, left_c), (1, right_c)):
            cursor = 0.0
            for s in SITES:
                n = int(((ok["equipment_type"] == fam)
                         & (ok["candidate"] == cand)
                         & (ok["site_letter"] == s)).sum())
                if not n:
                    continue
                ax.barh(yi, sign * n, left=sign * cursor, height=0.40,
                        color=SITE_COLOUR[s], edgecolor="white", linewidth=0.6)
                if n >= 2:
                    # Site C's tint is light; white would vanish on it
                    ink = "white" if s in ("A", "B", "D") else C_TEXT
                    ax.text(sign * (cursor + n / 2), yi, s, ha="center",
                            va="center", fontsize=5.8, color=ink)
                cursor += n
            if cursor:
                ax.text(sign * (cursor + 0.35), yi, f"{int(cursor)}",
                        ha="left" if sign > 0 else "right", va="center",
                        fontsize=6.5, color=C_TEXT)
        ax.text(-0.6, yi + 0.40, PAIR_LABEL[left_c], ha="right", va="bottom",
                fontsize=5.8, color=C_MUTED)
        ax.text(0.6, yi + 0.40, PAIR_LABEL[right_c], ha="left", va="bottom",
                fontsize=5.8, color=C_MUTED)

    ax.axvline(0, color=C_TEXT, linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([FAMILY_LABEL[f] for f in FAMILIES])
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_xlim(-17, 15)
    ax.set_xticks([-15, -10, -5, 0, 5, 10])
    ax.set_xticklabels(["15", "10", "5", "0", "5", "10"])
    ax.set_xlabel("Devices selecting each candidate")
    ax.set_ylim(-0.75, len(FAMILIES) - 0.15)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=C_GRID, linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)


def panel_c(ax, read: pd.DataFrame) -> None:
    """Median test error, by family, one marker per site."""
    y = np.arange(len(FAMILIES))[::-1]
    offsets = {"A": 0.24, "B": 0.08, "C": -0.08, "D": -0.24}
    for yi, fam in zip(y, FAMILIES):
        for s in SITES:
            row = read[(read["site"] == s) & (read["family"] == fam)]
            if row.empty:
                continue
            err = row["median_test_error_pct"].iloc[0]
            n = int(row["scored"].iloc[0])
            attempted = int(row["devices"].iloc[0])
            if pd.isna(err):
                ax.text(1.2, yi + offsets[s], f"Site {s}: none of {attempted} scored",
                        ha="left", va="center", fontsize=5.8, color=C_MUTED,
                        style="italic")
                continue
            ax.plot([0, err], [yi + offsets[s]] * 2, color=SITE_COLOUR[s],
                    linewidth=0.8, alpha=0.5, zorder=1)
            ax.plot([err], [yi + offsets[s]], "o", markersize=4.2,
                    color=SITE_COLOUR[s], markeredgewidth=0, zorder=3)
            ax.text(err + 0.7, yi + offsets[s], f"{err:.1f}  (n={n})",
                    ha="left", va="center", fontsize=5.8, color=C_TEXT)

    ax.set_yticks(y)
    ax.set_yticklabels([FAMILY_LABEL[f] for f in FAMILIES])
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_xlim(0, 40)
    ax.set_xlabel("Median test-segment error (%)")
    ax.set_ylim(-0.75, len(FAMILIES) - 0.15)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=C_GRID, linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)


def main() -> int:
    style()
    read = pd.read_csv(EVIDENCE / "site_family_readiness.csv")
    devices = pd.read_csv(EVIDENCE / "cross_site_devices_named.csv")

    fig = plt.figure(figsize=(7.2, 6.4))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05],
                            width_ratios=[1.0, 1.0], hspace=0.46, wspace=0.38,
                            left=0.105, right=0.975, top=0.90, bottom=0.075)
    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[1, 1])

    panel_a(ax_a, read)
    panel_b(ax_b, devices)
    panel_c(ax_c, read)

    handles = [Patch(facecolor=SITE_COLOUR[s], edgecolor="none", label=f"Site {s}")
               for s in SITES]
    ax_b.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.20),
                ncol=4, handlelength=1.1, handleheight=0.8, columnspacing=1.0,
                borderpad=0.0)

    for ax, letter, dx, dy in ((ax_a, "a", -0.055, 1.10),
                               (ax_b, "b", -0.135, 1.20),
                               (ax_c, "c", -0.135, 1.20)):
        ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=9,
                fontweight="bold", va="bottom", ha="right")
    ax_a.text(0.0, 1.10, "Devices reaching an independent test score",
              transform=ax_a.transAxes, ha="left", va="bottom", fontsize=7.5)
    ax_b.text(0.0, 1.20, "Candidate selected, by family and site",
              transform=ax_b.transAxes, ha="left", va="bottom", fontsize=7.5)
    ax_c.text(0.0, 1.20, "Median test error of the scored devices",
              transform=ax_c.transAxes, ha="left", va="bottom", fontsize=7.5)

    out = EVIDENCE / "figure6_cross_site"
    fig.savefig(f"{out}.svg", bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    fig.savefig(f"{out}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}.svg / .pdf / .png")
    print(f"  {int(read['scored'].sum())} of {int(read['devices'].sum())} scored")
    return 0


if __name__ == "__main__":
    sys.exit(main())
