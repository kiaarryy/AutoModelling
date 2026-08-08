# -*- coding: utf-8 -*-
"""Figure 7 -- the three-way temporal split, and what it costs.

The published Figure 7 showed measured-versus-simulated traces on an
"out-of-sample" period that was the retained record minus the parameter
identification windows. Two reviewers pointed out that the same remainder was
then used both to choose the candidate and to report the chosen candidate's
accuracy, so the reported number was a selection score. Redrawing the old
figure against the new split would answer none of that: a scatter plot cannot
show where its own samples came from.

So the figure now carries the split itself as its subject.

(a) is the hero and is drawn from one device's real record. Equal-duration
    blocks are dealt to the three roles in a repeating pattern and a 72 h
    buffer is discarded at every boundary, so each role samples the whole
    year rather than one season, and no role touches its neighbour. The
    reader can see the held-out segment rather than being told about it.

(b) answers the obvious objection to any temporal split -- that the test
    segment might simply be an easier operating region. It is not: on the
    device of panel (a) the test segment reaches 4.1 K below the coldest
    5% of identification wet-bulb, so it is if anything the harder period.

(c) answers "does the distinction matter?", which is the question the
    reviewers were really asking. Reporting the selection segment, as the
    published scheme effectively did, would have understated pump error by
    up to a factor of three.

Usage:
    python scripts/figure_temporal_split.py
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
sys.path.insert(0, str(ROOT / "src"))

EVIDENCE = ROOT / "docs" / "evidence"
RUN = "sitea_e2e_20260804c"
HERO = "CT_01"

# One neutral family for the segment that fits, one signal family for the
# segment that scores, and a muted third for the segment that only chooses.
# Buffers are deliberately drawn as absence rather than as a fourth colour.
C_IDENT = "#3775BA"
C_SELECT = "#B4C0E4"
C_TEST = "#B64342"
C_BUFFER = "#E8E8E8"
C_GRID = "#CFCECE"
C_TEXT = "#272727"
C_MUTED = "#767676"

SEGMENT_COLOUR = {"identification": C_IDENT, "selection": C_SELECT, "test": C_TEST}
SEGMENT_LABEL = {"identification": "Identification", "selection": "Selection",
                 "test": "Test"}

FAMILY_LABEL = {"chiller": "Chiller", "cooling_tower": "Cooling tower",
                "pump": "Pump", "heat_exchanger": "Heat exchanger"}
FAMILY_MARKER = {"chiller": "o", "cooling_tower": "s", "pump": "^",
                 "heat_exchanger": "D"}

VARIABLE_LABEL = {"Twb_C": "Wet-bulb\ntemperature (°C)",
                  "Tin_C": "Entering water\ntemperature (°C)",
                  "flow_m3_h": "Water flow\n(m³/h)"}


def style() -> None:
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams.update({
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "axes.edgecolor": C_TEXT,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "legend.frameon": False,
        "text.color": C_TEXT,
        "axes.labelcolor": C_TEXT,
        "xtick.color": C_TEXT,
        "ytick.color": C_TEXT,
    })


def hero_split():
    """Rebuild the split the pipeline used, on the hero device's own record."""
    import importlib.util

    from autofmu.temporal_split import SplitSpec, build_split, proportional_blocks

    spec_path = ROOT / "scripts" / "build_site_a_temporal_split.py"
    module_spec = importlib.util.spec_from_file_location("_sitea_split", spec_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    table = module.autofmu_table(RUN, HERO)
    if table is None:
        raise SystemExit(f"{HERO}: no attributed frame in run {RUN}")
    spec = SplitSpec()
    result = build_split(table["DateTime"], spec)
    blocks = proportional_blocks(table["DateTime"].dropna(), spec)
    return table, result, blocks, spec


def panel_a(ax, table, result, blocks, spec) -> None:
    """The split, on the record it was computed from."""
    stamps = table["DateTime"]
    t0, t1 = stamps.min(), stamps.max()

    # the dealt blocks, drawn as the schematic band
    for segment, spans in blocks.items():
        for start, end in spans:
            ax.axvspan(start, end, ymin=0.62, ymax=0.96,
                       facecolor=SEGMENT_COLOUR[segment], edgecolor="none")
    ax.axhspan(0, 0, color="none")

    # sample density beneath, so a reader can see where the record is actually
    # populated -- a block over a data gap contributes nothing and the counts
    # in the legend, not the block widths, are what the split is scored on
    daily = stamps.dt.floor("D").value_counts().sort_index()
    density = daily / daily.max() * 0.36
    ax.bar(daily.index, density.to_numpy(), width=1.0, bottom=0.06,
           color=C_MUTED, linewidth=0, alpha=0.55)

    counts = result.counts()
    ax.set_xlim(t0, t1)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="x", length=2.5, pad=1.5)

    ax.text(1.0, 0.475, f"Site A, {HERO.replace('_', '-')} retained operating record, "
                        f"{len(table):,} five-minute samples over "
                        f"{(t1 - t0).days} days",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5,
            color=C_MUTED)
    ax.text(0.0, 0.30, "Retained samples per day", transform=ax.transAxes,
            ha="left", va="center", fontsize=6, color=C_MUTED)

    # a buffer callout on the first boundary, since the gaps are the whole
    # point; the leader stops below the band so it never crosses a block
    first = sorted((s, e) for spans in blocks.values() for s, e in spans)
    if len(first) >= 2:
        gap_mid = first[0][1] + (first[1][0] - first[0][1]) / 2
        ax.annotate(f"{spec.buffer_hours:.0f} h buffer discarded\nat every boundary",
                    xy=(gap_mid, 0.605), xytext=(0.155, 0.005),
                    textcoords=ax.transAxes, fontsize=6, color=C_TEXT,
                    ha="center", va="bottom",
                    arrowprops=dict(arrowstyle="-", linewidth=0.6,
                                    color=C_TEXT, shrinkA=1, shrinkB=0))

    handles = [Patch(facecolor=SEGMENT_COLOUR[s], edgecolor="none",
                     label=f"{SEGMENT_LABEL[s]}  ({counts[s]:,})")
               for s in ("identification", "selection", "test")]
    handles.append(Patch(facecolor=C_BUFFER, edgecolor=C_MUTED, linewidth=0.5,
                         label=f"Buffer  ({counts['buffer']:,}, discarded)"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.20),
              ncol=4, handlelength=1.4, handleheight=0.8, columnspacing=1.3,
              borderpad=0.0)


def panel_b(ax, comparability: pd.DataFrame) -> None:
    """Per-segment driver ranges: is the test segment an easier region?"""
    sub = comparability[comparability["device"] == HERO.replace("_", "-")]
    variables = ["Twb_C", "Tin_C", "flow_m3_h"]
    segments = ["identification", "selection", "test"]

    # each variable normalised onto its own row, so three different units
    # share one axis without a broken scale
    ytick_pos, ytick_lab = [], []
    for vi, var in enumerate(variables):
        rows = sub[sub["variable"] == var].set_index("segment")
        lo = min(rows.loc[s, "p05"] for s in segments)
        hi = max(rows.loc[s, "p95"] for s in segments)
        span = hi - lo or 1.0
        for si, seg in enumerate(segments):
            y = vi * 4 + (2 - si)
            p05 = (rows.loc[seg, "p05"] - lo) / span
            p95 = (rows.loc[seg, "p95"] - lo) / span
            med = (rows.loc[seg, "median"] - lo) / span
            ax.plot([p05, p95], [y, y], color=SEGMENT_COLOUR[seg],
                    linewidth=3.0, solid_capstyle="butt", zorder=2)
            ax.plot([med], [y], marker="|", color="white", markersize=4,
                    markeredgewidth=0.9, zorder=3)
            if vi == 0:
                ax.text(p95 + 0.02, y, SEGMENT_LABEL[seg], va="center",
                        ha="left", fontsize=6, color=C_MUTED)
            ytick_pos.append(y)
            ytick_lab.append("")
        # the real units, printed at the ends rather than on a shared axis
        ax.text(-0.02, vi * 4 + 1, VARIABLE_LABEL[var], transform=
                ax.get_yaxis_transform(), ha="right", va="center", fontsize=6.5)
        ax.text(0.0, vi * 4 - 0.55, f"{lo:.0f}", ha="center", va="top",
                fontsize=5.8, color=C_MUTED)
        ax.text(1.0, vi * 4 - 0.55, f"{hi:.0f}", ha="center", va="top",
                fontsize=5.8, color=C_MUTED)

    ax.set_xlim(-0.02, 1.32)
    ax.set_ylim(-1.4, 11.2)
    ax.set_yticks([])
    ax.set_xticks([])
    for spine in ("left", "bottom"):
        ax.spines[spine].set_visible(False)
    ax.text(0.5, -0.10, "p5–p95 of each segment, scaled to the union of the three",
            transform=ax.transAxes, ha="center", va="top", fontsize=6,
            color=C_MUTED)


def panel_c(ax, paired: pd.DataFrame) -> None:
    """Test error relative to selection error, every scored device, by site.

    An earlier version listed the 19 Site A devices down the axis. With all four
    sites there are 58, too many to name, and naming them was never the point:
    what matters is that the penalty is systematic and that it is worst where
    observability is worst. So each site becomes a row and each device a dot.
    """
    rng = np.random.default_rng(0)          # jitter must be reproducible
    sites = ["A", "B", "C", "D"]
    y_of = {s: len(sites) - 1 - i for i, s in enumerate(sites)}

    ax.axvline(1.0, color=C_TEXT, linewidth=0.9, zorder=2)
    summary: dict[str, str] = {}
    for site in sites:
        sub = paired[paired["site"] == site]
        if sub.empty:
            continue
        base = y_of[site]
        jitter = rng.uniform(-0.17, 0.17, len(sub))
        worse = sub["ratio"].to_numpy() > 1.0
        ax.scatter(sub["ratio"][worse], base + jitter[worse], s=13,
                   facecolor=C_TEST, edgecolor="white", linewidth=0.35,
                   zorder=4)
        ax.scatter(sub["ratio"][~worse], base + jitter[~worse], s=13,
                   facecolor=C_MUTED, edgecolor="white", linewidth=0.35,
                   alpha=0.75, zorder=3)

        median = float(sub["ratio"].median())
        ax.plot([median, median], [base - 0.30, base + 0.30], color=C_IDENT,
                linewidth=1.6, solid_capstyle="butt", zorder=5)
        # chr(10) rather than an escape: the label is two lines, and a
        # newline inside the f-string keeps the count under the site name
        summary[site] = ("Site " + site + chr(10)
                         + f"{int(worse.sum())}/{len(sub)}  "
                         + f"med {median:.2f}" + "×")

    # the per-site counts live in the axis labels rather than floating in the
    # plot area, where they landed on top of the points they describe
    ax.set_yticks([y_of[s] for s in sites])
    ax.set_yticklabels([summary.get(s, f"Site {s}") for s in sites],
                       fontsize=6.5, linespacing=1.3)
    ax.set_ylim(-0.55, len(sites) - 0.45)
    ax.set_xscale("log")
    ax.set_xlim(0.4, 4.8)
    ax.set_xticks([0.5, 1, 2, 3, 4])
    ax.set_xticklabels(["0.5×", "1×", "2×", "3×", "4×"])
    # a log axis volunteers minor labels such as "6 x 10^-1", which on a ratio
    # axis read as a second, contradictory scale
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.set_xlabel("Test error ÷ selection error")
    ax.tick_params(axis="y", length=0, pad=2)
    ax.grid(axis="x", color=C_GRID, linewidth=0.4, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)

    total_worse = int((paired["ratio"] > 1.0).sum())
    ax.text(0.0, -0.30, f"Test error exceeds selection error on "
                        f"{total_worse} of {len(paired)} scored devices",
            transform=ax.transAxes, ha="left", va="top", fontsize=6,
            color=C_MUTED)

    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=3.6,
               color=C_TEST, label="worse on test"),
        Line2D([], [], marker="o", linestyle="none", markersize=3.6,
               color=C_MUTED, label="better on test"),
        Line2D([], [], linestyle="-", linewidth=1.6, color=C_IDENT,
               label="site median"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.02, 1.13),
              ncol=3, handletextpad=0.35, columnspacing=1.0, borderpad=0.0,
              handlelength=1.1)


def main() -> int:
    style()
    table, result, blocks, spec = hero_split()
    comparability = pd.read_csv(EVIDENCE / "table_si_comparability.csv")

    # all four sites, not Site A alone: the generalisation penalty is the
    # paper's answer to the reviewers and it is stronger across the fleet
    paired = pd.read_csv(EVIDENCE / "cross_site_selection_vs_test.csv")
    paired = paired[paired["selected"]].dropna(subset=["ratio"]).copy()

    fig = plt.figure(figsize=(7.2, 5.5))
    grid = fig.add_gridspec(
        2, 2, height_ratios=[1.0, 1.55], width_ratios=[1.0, 1.12],
        hspace=0.52, wspace=0.30,
        left=0.10, right=0.985, top=0.855, bottom=0.075)

    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[1, 1])

    panel_a(ax_a, table, result, blocks, spec)
    panel_b(ax_b, comparability)
    panel_c(ax_c, paired)

    # Panel titles stay descriptive. What the panels mean is the caption's job
    # and the text's; a title that states the conclusion is an editorial, not a
    # label, and this is a research article.
    for ax, letter, dx, dy in ((ax_a, "a", -0.045, 1.36), (ax_b, "b", -0.105, 1.10),
                               (ax_c, "c", -0.135, 1.22)):
        ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=9,
                fontweight="bold", va="bottom", ha="right")

    ax_a.text(0.0, 1.36, "Three-way temporal split of one device record",
              transform=ax_a.transAxes, ha="left", va="bottom", fontsize=7.5)
    ax_b.text(0.0, 1.10, "Operating-condition coverage per segment",
              transform=ax_b.transAxes, ha="left", va="bottom", fontsize=7.5)
    ax_c.text(0.0, 1.22, "Generalisation gap across scored devices",
              transform=ax_c.transAxes, ha="left", va="bottom", fontsize=7.5)

    out = EVIDENCE / "figure7_temporal_split"
    fig.savefig(f"{out}.svg", bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    fig.savefig(f"{out}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    counts = result.counts()
    print(f"panel a: {HERO} {counts}")
    print(f"panel c: {int((paired['ratio'] > 1).sum())}/{len(paired)} worse on test, "
          f"max {paired['ratio'].max():.2f}x")
    print(f"wrote {out}.svg / .pdf / .png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
