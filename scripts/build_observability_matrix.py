"""Phase 3 deliverables: the minimum sensor set, and the cross-site
observability matrix.

Two artefacts the manuscript's new section 2.3 needs.

**Minimum sensor set (3.1).** Not a checklist assembled from experience: each
row names the model equation that consumes the channel, so the requirement can
be checked against the Modelica source rather than taken on trust. This is the
technical argument for an equation-based modelling medium over a black box --
you cannot derive a black box's minimum sensor set, because it has no equations
to read.

**Observability matrix (3.5).** Which checks fire where. The point is not that
some archives are worse; it is that the failures are specific and nameable, and
that the same code produces different verdicts because the data differ.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
EVIDENCE = REPO / "docs" / "evidence"

# (family, model type, canonical channel, role in the equations, source)
#
# "Role" cites the equation that consumes the channel. Where a channel feeds a
# boundary condition rather than a constitutive equation that is said too,
# because a boundary condition can sometimes be supplied from a design value
# while a constitutive input cannot.
SENSOR_SET = [
    # ---- cooling tower -------------------------------------------------
    ("cooling_tower", "YorkCalc", "tcwr_C", "required",
     "entering water temperature: TRan = T_a - T_b, the correlation's range",
     "Buildings.Fluid.HeatExchangers.CoolingTowers.YorkCalc"),
    ("cooling_tower", "YorkCalc", "tcws_1_C", "required",
     "leaving water temperature: the model output being validated, and the "
     "other end of TRan", "same"),
    ("cooling_tower", "YorkCalc", "twb_C", "required",
     "TApp = TOut - TWetBul; the correlation is a polynomial in "
     "(TWetBul, TRan, liquid-to-gas ratio)", "same"),
    ("cooling_tower", "YorkCalc", "attributed_flow_m3_h", "required",
     "FRWat = m_flow / (FRWat0 * m_flow_nominal), the liquid side of the "
     "liquid-to-gas ratio", "same"),
    ("cooling_tower", "YorkCalc", "fan1_Hz", "required",
     "FRAir = y, the gas side of the liquid-to-gas ratio, and the switch "
     "between forced and free convection", "same"),
    ("cooling_tower", "YorkCalc", "fans_on_count", "required",
     "cell count: converts plant flow to per-cell flow before FRWat",
     "wrapper SiteACTYork27ClosedLoop.mo"),
    ("cooling_tower", "YorkCalc", "power_W", "optional",
     "fan power is reported, never used to rank candidates (section 2.6)", "-"),
    ("cooling_tower", "Merkel", "tcwr_C", "required",
     "entering water temperature: UA-corrected epsilon-NTU boundary", "Buildings...CoolingTowers.Merkel"),
    ("cooling_tower", "Merkel", "tcws_1_C", "required",
     "leaving water temperature: model output being validated", "same"),
    ("cooling_tower", "Merkel", "twb_C", "required",
     "air-side inlet enthalpy is taken at the wet bulb", "same"),
    ("cooling_tower", "Merkel", "attributed_flow_m3_h", "required",
     "water-side capacity rate m*cp in the NTU correction", "same"),
    ("cooling_tower", "Merkel", "fan1_Hz", "required",
     "air-side capacity rate through the fan-speed ratio", "same"),

    # ---- chiller -------------------------------------------------------
    ("chiller", "ElectricEIR", "power_W", "required",
     "the calibration target: P = P_nominal * capFT * EIRFT * EIRFPLR",
     "Buildings.Fluid.Chillers.ElectricEIR"),
    ("chiller", "ElectricEIR", "tchws_C", "required",
     "leaving evaporator temperature: the first argument of capFT and EIRFT", "same"),
    ("chiller", "ElectricEIR", "tcws_C", "required",
     "entering condenser temperature: the second argument of capFT and EIRFT "
     "for ElectricEIR", "same"),
    ("chiller", "ElectricEIR", "chw_flow_m3_h", "required",
     "evaporator mass flow: sets the load Q = m*cp*dT that drives PLR", "same"),
    ("chiller", "ElectricEIR", "tchwr_C", "required",
     "entering evaporator temperature: the other end of that dT", "same"),
    ("chiller", "ElectricEIR", "cw_flow_m3_h", "required",
     "condenser mass flow: boundary condition on the condenser volume", "same"),
    ("chiller", "ElectricReformulatedEIR", "tcwr_C", "required",
     "LEAVING condenser temperature replaces the entering one in EIRFT and "
     "EIRFPLR -- the reformulation is exactly this substitution",
     "Buildings.Fluid.Chillers.ElectricReformulatedEIR"),

    # ---- pump ----------------------------------------------------------
    ("pump", "empirical_power", "power_W", "required",
     "the calibration target: P = P_nominal * (c0 + c1*phi + c2*y + c3*y^3 + "
     "c4*phi*y)", "models/pump/PumpEmpiricalPower.mo"),
    ("pump", "empirical_power", "speed_Hz", "required",
     "y = speed / speed_nominal, the only argument of the affinity and "
     "speed-poly forms", "same"),
    ("pump", "empirical_power", "attributed_flow_m3_h", "conditional",
     "phi = m_flow / m_flow_nominal, used only by the speed-flow form and only "
     "where attribution covers at least half the running record", "same"),
    ("pump", "mover", "attributed_flow_m3_h", "required",
     "the pump curve is shaped by a measured nominal flow; without one the "
     "mover cannot be instantiated", "Buildings.Fluid.Movers.SpeedControlled_y"),

    # ---- heat exchanger ------------------------------------------------
    ("heat_exchanger", "ConstantEffectiveness", "tchws_C", "required",
     "side-1 inlet temperature in Q = eps * Cmin * (T_a1 - T_a2)",
     "Buildings.Fluid.HeatExchangers.ConstantEffectiveness"),
    ("heat_exchanger", "ConstantEffectiveness", "tcws_C", "required",
     "side-2 inlet temperature, the other end of that difference", "same"),
    ("heat_exchanger", "ConstantEffectiveness", "chw_flow_m3_h", "required",
     "side-1 capacity rate C1 = m1*cp; Cmin needs both", "same"),
    ("heat_exchanger", "ConstantEffectiveness", "cw_flow_m3_h", "required",
     "side-2 capacity rate C2 = m2*cp; Cmin needs both", "same"),
    ("heat_exchanger", "ConstantEffectiveness", "tcwr_C", "required",
     "side-2 outlet: the uncontrolled output validated against", "same"),
    ("heat_exchanger", "PlateEffectivenessNTU", "tchwr_C", "required",
     "side-1 outlet closes the duty used to identify UA in eps = f(NTU, Cr)",
     "Buildings.Fluid.HeatExchangers.PlateHeatExchangerEffectivenessNTU"),
]


def sensor_set() -> pd.DataFrame:
    return pd.DataFrame(SENSOR_SET, columns=[
        "family", "model_type", "canonical_channel", "necessity",
        "role_in_the_equations", "source"])


def matrix(observability: pd.DataFrame, devices: pd.DataFrame) -> pd.DataFrame:
    """site x check -> devices carrying it, plus the fleet size for context."""
    obs = observability.copy()
    obs["check"] = obs["flag"].str.split(":").str[0]
    counts = (obs.drop_duplicates(["site", "device_id", "check"])
                 .groupby(["site", "check"]).size().unstack(fill_value=0))
    fleet = devices.groupby("site").size().rename("devices")
    scored = (devices[devices.status == "ok"].groupby("site").size()
              .reindex(fleet.index, fill_value=0).rename("scored"))
    return counts.join(fleet).join(scored)


def figure(mat: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    site_labels = {"site_a": "Site A", "tencent": "Tencent",
                   "hkust": "HKUST", "lbnl": "LBNL (public)"}
    order = [s for s in ("site_a", "tencent", "hkust", "lbnl") if s in mat.index]
    checks = [c for c in mat.columns if c not in ("devices", "scored")]
    # order checks by how often they fire, so the busy ones read first
    checks = sorted(checks, key=lambda c: -mat[c].sum())

    share = np.zeros((len(order), len(checks)))
    counts = np.zeros_like(share, dtype=int)
    for i, s in enumerate(order):
        n = float(mat.loc[s, "devices"])
        for j, c in enumerate(checks):
            counts[i, j] = int(mat.loc[s, c])
            share[i, j] = 100.0 * counts[i, j] / n if n else 0.0

    fig, ax = plt.subplots(figsize=(1.25 * len(checks) + 3.6, 0.72 * len(order) + 2.6))
    im = ax.imshow(share, cmap="YlOrRd", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(checks)))
    ax.set_xticklabels([c.replace("_", "\n") for c in checks], fontsize=8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{site_labels.get(s, s)}\n{int(mat.loc[s,'scored'])}/"
                        f"{int(mat.loc[s,'devices'])} scored" for s in order],
                       fontsize=8.5)
    for i in range(len(order)):
        for j in range(len(checks)):
            if counts[i, j]:
                ax.text(j, i, str(counts[i, j]), ha="center", va="center",
                        fontsize=8.5,
                        color="white" if share[i, j] > 55 else "#222222")
    ax.set_title("Observability findings per site (devices carrying each flag)",
                 fontsize=10, pad=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("% of the site's devices", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.set_xticks(np.arange(-0.5, len(checks), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(order), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    fig.tight_layout()
    fig.savefig(path, format="svg")
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evidence", default=str(EVIDENCE))
    args = ap.parse_args()
    ev = Path(args.evidence)

    sset = sensor_set()
    sset.to_csv(ev / "minimum_sensor_set.csv", index=False)

    obs = pd.read_csv(ev / "cross_site_observability.csv")
    dev = pd.read_csv(ev / "cross_site_devices.csv")
    mat = matrix(obs, dev)
    mat.to_csv(ev / "observability_matrix.csv")
    figure(mat, ev / "observability_matrix.svg")

    pd.set_option("display.width", 220)
    print("minimum sensor set: %d rows, %d model types"
          % (len(sset), sset.model_type.nunique()))
    print(sset.groupby(["family", "model_type"]).size().rename("channels").to_string())
    print()
    print(mat.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
