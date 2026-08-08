"""Figure 8 — workflow effort.

The old figure paired a manual bar against an automated bar per family. That
encoding is no longer available: the manual arm is measured and the automated
arm is not, so half of every pair would have been invented. Worse, three of the
eight stages have an automated cost of exactly zero, which in a paired bar chart
reads as missing data rather than as the point being made.

So the panels are reorganised around what the evidence actually supports.

(a) is the hero. Each bar is one workflow stage, measured. It is split into the
    part automation removes and the part that still needs a person, with a
    bracket showing how uncertain that split is. Stages 4-6 are entirely
    removed and carry no bracket, because the pipeline demonstrably performs
    them -- their machine time is in the run manifests. The awkward zero
    becomes the message.

(b) shows where the 139 h goes across the fleet, and that refused devices are
    not free -- a third of the cost of a modelled one, times fifteen of them.

(c) states the comparison the paper can defend: a measured manual total against
    a bounded automated one, with no point estimate anywhere.

    python scripts/figure_workflow_effort.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[1]
EV = REPO / "docs" / "evidence"
OUT = EV / "figure8_workflow_effort"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.7,
    "legend.frameon": False,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
})

RETAINED = "#0F4D92"     # certainly still needs a person
UNCERTAIN = "#8FA9CC"    # inside the assumed range
REMOVED = "#AADCA9"      # the pipeline does it
GREY = "#CFCECE"
DARK = "#4D4D4D"
ACCENT = "#B64342"

STAGE_LABEL = {
    1: "1  Data loading and\n    semantic mapping",
    2: "2  Input table\n    preparation",
    3: "3  Modelica/FMU\n    candidate setup",
    4: "4  FMU parameter\n    search",
    5: "5  Full-period\n    validation",
    6: "6  Tables and\n    figures",
    7: "7  Observability\n    review",
    8: "8  Temporal split\n    definition",
}


def panel_a(ax, sens: pd.DataFrame) -> None:
    """Measured manual hours per stage, split into removed and retained."""
    s = sens.sort_values("stage", ascending=False).reset_index(drop=True)
    y = np.arange(len(s))
    lo = s.auto_low_h.to_numpy()
    hi = s.auto_high_h.to_numpy()
    manual = s.manual_h.to_numpy()

    # Three segments rather than a bar edge that could be misread as a value:
    # solid is retained under every assumption, the band is the assumed range,
    # green is removed under every assumption.
    ax.barh(y, lo, height=0.62, color=RETAINED, edgecolor="white", linewidth=0.5)
    ax.barh(y, hi - lo, left=lo, height=0.62, color=UNCERTAIN,
            edgecolor="white", linewidth=0.5)
    ax.barh(y, manual - hi, left=hi, height=0.62, color=REMOVED,
            edgecolor="white", linewidth=0.5)

    for i in range(len(s)):
        if hi[i] == 0:
            ax.text(0.25, y[i], "fully automated", ha="left", va="center",
                    fontsize=6.0, color="#2F5E33", style="italic", zorder=6)
        ax.text(manual[i] + 0.22, y[i], f"{manual[i]:.1f} h", ha="left",
                va="center", fontsize=6.2, color=DARK)

    ax.set_yticks(y)
    ax.set_yticklabels([STAGE_LABEL[int(v)] for v in s.stage], fontsize=6.4,
                       linespacing=1.25)
    ax.set_xlabel("Measured manual effort (h), 12 sampled devices")
    ax.set_xlim(0, max(manual) * 1.30)
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)

    removed = float(s.loc[s.auto_high_h == 0, "manual_h"].sum())
    rows = np.where(s.auto_high_h.to_numpy() == 0)[0]
    top, bot = rows.max() + 0.42, rows.min() - 0.42
    xb = max(manual) * 1.075
    ax.plot([xb, xb + 0.16, xb + 0.16, xb], [bot, bot, top, top],
            color="#2F5E33", lw=0.8, clip_on=False)
    ax.text(xb + 0.32, (top + bot) / 2,
            "removed entirely\n%.1f of %.1f h" % (removed, manual.sum()),
            fontsize=6.2, color="#2F5E33", va="center", ha="left",
            linespacing=1.3, clip_on=False)

    ax.legend(handles=[
        Patch(facecolor=RETAINED, label="retained under every assumption"),
        Patch(facecolor=UNCERTAIN, label="inside the assumed range"),
        Patch(facecolor=REMOVED, label="removed: performed by the pipeline"),
    ], loc="lower right", fontsize=6.1, handlelength=1.1, borderpad=0.3,
        labelspacing=0.35)


def panel_b(ax, ext: pd.DataFrame) -> None:
    """Where the extrapolated 139 h sits across the fleet."""
    ext = ext.sort_values("full_inventory_h")
    y = np.arange(len(ext))
    setup = ext.setup_h.to_numpy()
    acc = (ext.accepted * ext.accepted_marginal_h).to_numpy()
    ref = (ext.refused * ext.refused_marginal_h).to_numpy()

    ax.barh(y, setup, height=0.6, color=GREY, edgecolor="white", linewidth=0.5,
            label="one-time family setup")
    ax.barh(y, acc, left=setup, height=0.6, color=RETAINED, edgecolor="white",
            linewidth=0.5, label="accepted devices")
    ax.barh(y, ref, left=setup + acc, height=0.6, color=ACCENT,
            edgecolor="white", linewidth=0.5, label="refused devices")

    for i, (_, r) in enumerate(ext.iterrows()):
        ax.text(r.full_inventory_h + 1.2, i, f"{r.full_inventory_h:.1f} h",
                va="center", fontsize=6.2, color=DARK)

    ax.set_yticks(y)
    ax.set_yticklabels([f"{f}\n({int(n)} devices)" for f, n in
                        zip(ext.family, ext.inventory)], fontsize=6.4,
                       linespacing=1.25)
    ax.set_xlabel("Extrapolated manual effort (h), full inventory")
    ax.set_xlim(0, ext.full_inventory_h.max() * 1.26)
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=6.2, handlelength=1.1, borderpad=0.3)
    ax.set_title(f"total {ext.full_inventory_h.sum():.1f} h for "
                 f"{int(ext.inventory.sum())} devices", fontsize=6.6,
                 color=DARK, pad=3, loc="left")


def panel_c(ax, manual_total: float, lo: float, hi: float,
            machine_h: float) -> None:
    """The comparison the evidence supports: one measured, one bounded."""
    ax.barh([1], [manual_total], height=0.42, color=RETAINED)
    ax.text(manual_total + 2.5, 1, f"{manual_total:.1f} h\nmeasured", va="center",
            fontsize=6.4, color=DARK, linespacing=1.3)

    ax.barh([0], [hi - lo], left=[lo], height=0.42, color=REMOVED,
            edgecolor=DARK, linewidth=0.6)
    for x in (lo, hi):
        ax.plot([x, x], [-0.26, 0.26], color=DARK, lw=0.8, zorder=4)
    ax.text(hi + 2.5, 0, f"{lo:.1f}–{hi:.1f} h\nbounded, not measured",
            va="center", fontsize=6.4, color=DARK, linespacing=1.3)

    ax.plot([machine_h], [0], marker="D", ms=3.4, color=ACCENT, zorder=6,
            clip_on=False)
    ax.annotate(f"scripted compute {machine_h*60:.0f} min",
                xy=(machine_h, -0.30), xytext=(manual_total * 0.16, -0.66),
                fontsize=6.0, color=ACCENT, va="center",
                arrowprops=dict(arrowstyle="-", lw=0.5, color=ACCENT))

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Automated arm,\nhuman effort",
                        "Manual arm,\nhuman effort"], fontsize=6.4,
                       linespacing=1.25)
    ax.set_xlabel("Effort for the full 34-device inventory (h)")
    ax.set_xlim(0, manual_total * 1.34)
    ax.set_ylim(-0.85, 1.6)
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)
    ax.set_title(f"implied reduction {100*(1-hi/manual_total):.0f}–"
                 f"{100*(1-lo/manual_total):.0f}%  (sensitivity, not a "
                 f"measurement)", fontsize=6.6, color=DARK, pad=3, loc="left")


def main() -> int:
    sens = pd.read_csv(EV / "auto_arm_sensitivity.csv")
    ext = pd.read_csv(EV / "time_and_motion_extrapolation.csv")
    mach = pd.read_csv(EV / "time_and_motion_machine.csv")

    manual_sample = float(sens.manual_h.sum())
    manual_full = float(ext.full_inventory_h.sum())
    lo = manual_full * float(sens.auto_low_h.sum()) / manual_sample
    hi = manual_full * float(sens.auto_high_h.sum()) / manual_sample
    machine_h = float(mach.loc[mach.run.str.startswith("sitea"),
                               "wall_min"].iloc[0]) / 60.0

    fig = plt.figure(figsize=(7.09, 5.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0],
                          width_ratios=[1.0, 1.0], hspace=0.52, wspace=0.42)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    panel_a(ax_a, sens)
    panel_b(ax_b, ext)
    panel_c(ax_c, manual_full, lo, hi, machine_h)

    for ax, letter in ((ax_a, "a"), (ax_b, "b"), (ax_c, "c")):
        ax.text(-0.035, 1.06, letter, transform=ax.transAxes, fontsize=9,
                fontweight="bold", va="bottom", ha="right")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    for ext_name in ("svg", "pdf"):
        fig.savefig(f"{OUT}.{ext_name}", bbox_inches="tight")
    fig.savefig(f"{OUT}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}.svg / .pdf / .png")
    print(f"  manual sample {manual_sample:.1f} h, full inventory {manual_full:.1f} h")
    print(f"  automated arm bounded {lo:.1f}-{hi:.1f} h "
          f"({100*(1-hi/manual_full):.0f}-{100*(1-lo/manual_full):.0f}% reduction)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
