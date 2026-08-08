# -*- coding: utf-8 -*-
"""Assemble the evidence for the three-way temporal split.

The published manuscript defined "out-of-sample" as the retained record minus
the parameter-identification windows, and then used that same remainder both to
choose between candidates and to report the chosen candidate's accuracy.  Two
reviewers objected independently.  :mod:`autofmu.temporal_split` replaced the
scheme with identification / selection / test segments separated by buffers, but
the manuscript's Section 3.3 and Table 4 still carry the superseded numbers.

This script produces the tables that replace them, all of them read back out of
a completed run rather than retyped:

``table4_site_a.csv``
    One row per scored Site A device: selected candidate, the family's primary
    acceptance metric on the *test* segment, bias, skill against a mean
    predictor, and the excitation of the model's driving variable.

``table_si_segments.csv``
    Scored sample counts per segment for every Site A device, with the split
    status that decides whether the device can be scored at all.

``table_si_selection_vs_test.csv``
    Every candidate's selection-segment and test-segment error, which is the
    evidence that selection was not done on the reported numbers.

``table_si_comparability.csv``
    Per-segment driver ranges, so a reader can check that the test segment is
    not an easier operating region than the identification segment.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs" / "runs" / "sitea_e2e_20260804c"
EVIDENCE = ROOT / "docs" / "evidence"

# The variable each family is ranked on, and the label the manuscript uses for
# it.  Kept here rather than in the prose so that the table and the text cannot
# drift apart again.
PRIMARY = {
    "chiller":        ("P_CVRMSE_pct", "P_NMBE_pct", "P_skill",
                       "Electrical power CVRMSE (%)"),
    "cooling_tower":  ("T_RMSE_K", "T_NMBE_pct", "T_skill",
                       "Leaving-water-temperature RMSE (K)"),
    "pump":           ("P_CVRMSE_pct", "P_NMBE_pct", "P_skill",
                       "Electrical power CVRMSE (%)"),
    "heat_exchanger": ("Q_CVRMSE_pct", "Q_NMBE_pct", "Q_skill",
                       "Reconstructed-heat CVRMSE (%)"),
}

DISPLAY = {"EIR": "ElectricEIR", "EEIR": "ElectricReformulatedEIR",
           "Merkel": "Merkel", "YorkCalc": "YorkCalc (27-coefficient)",
           "YorkCalc27": "YorkCalc (27-coefficient)",
           "affinity": "Cubic affinity", "speed_poly": "Speed polynomial",
           "ConstantEffectiveness": "ConstantEffectiveness",
           "PlateEffectivenessNTU": "PlateHeatExchangerEffectivenessNTU"}

FAMILY_ORDER = ["chiller", "cooling_tower", "pump", "heat_exchanger"]


def _dev_label(device_id: str) -> str:
    """CH_01 -> CH-01, matching the manuscript's device naming."""
    return device_id.replace("_", "-")


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(RUN / "calibrate" / "all_candidate_metrics.csv")
    devices = pd.read_csv(EVIDENCE / "cross_site_devices.csv")
    return metrics, devices[devices["site"] == "site_a"].copy()


def table4(metrics: pd.DataFrame, site_a: pd.DataFrame) -> pd.DataFrame:
    """The per-device test-segment result, for the devices that were scored."""
    test = metrics[metrics["stage"] == "test"]
    rows = []
    scored = site_a[site_a["status"] == "ok"]
    for _, dev in scored.iterrows():
        fam = dev["equipment_type"]
        err_col, bias_col, skill_col, label = PRIMARY[fam]
        hit = test[(test["device_id"] == dev["device_id"])
                   & (test["candidate"] == dev["candidate"])]
        if hit.empty:
            # the selected candidate must have a test-stage row; a missing one
            # means the run and the cross-site table disagree, which is a bug
            # rather than a device that merely lacks evidence
            raise SystemExit(
                f"{dev['device_id']}: selected candidate {dev['candidate']} "
                "has no test-stage metrics in the run")
        hit = hit.iloc[0]
        rows.append({
            "family": fam,
            "device": _dev_label(dev["device_id"]),
            "selected_model": DISPLAY.get(dev["candidate"], dev["candidate"]),
            "metric_label": label,
            "test_n": int(hit["N"]) if pd.notna(hit["N"]) else None,
            "test_error": float(hit[err_col]) if pd.notna(hit[err_col]) else None,
            "test_nmbe_pct": float(hit[bias_col]) if pd.notna(hit[bias_col]) else None,
            "test_skill": float(hit[skill_col]) if pd.notna(hit[skill_col]) else None,
            "selection_skill": (float(dev["selection_skill"])
                                if pd.notna(dev["selection_skill"]) else None),
            "driver": dev["driver"] if pd.notna(dev["driver"]) else "",
            "driver_excitation": (float(dev["driver_excitation"])
                                  if pd.notna(dev["driver_excitation"]) else None),
        })
    out = pd.DataFrame(rows)
    out["family"] = pd.Categorical(out["family"], FAMILY_ORDER, ordered=True)
    return out.sort_values(["family", "device"]).reset_index(drop=True)


def si_segments(metrics: pd.DataFrame, site_a: pd.DataFrame) -> pd.DataFrame:
    """Scored samples per segment, for every device the site offered.

    Devices that were refused appear here too, with the segment that failed:
    the refusal is part of the result, not an omission from it.
    """
    rows = []
    for _, dev in site_a.iterrows():
        dev_metrics = metrics[metrics["device_id"] == dev["device_id"]]
        counts = {}
        for stage in ("selection", "test"):
            sub = dev_metrics[dev_metrics["stage"] == stage]
            n = sub["N"].dropna()
            counts[stage] = int(n.max()) if not n.empty else 0
        rows.append({
            "family": dev["equipment_type"],
            "device": _dev_label(dev["device_id"]),
            "retained_operating_rows": int(dev["on_rows"]) if pd.notna(dev["on_rows"]) else 0,
            "selection_n": counts["selection"],
            "test_n": counts["test"],
            "split_status": dev["status"],
            "scored": dev["status"] == "ok",
        })
    out = pd.DataFrame(rows)
    out["family"] = pd.Categorical(out["family"], FAMILY_ORDER, ordered=True)
    return out.sort_values(["family", "device"]).reset_index(drop=True)


def si_selection_vs_test(metrics: pd.DataFrame, site_a: pd.DataFrame) -> pd.DataFrame:
    """Both scored segments, per candidate.

    This is the table that shows selection was decided on a segment that is not
    the one reported, and it is also where the generalisation gap becomes
    visible per candidate rather than only per device.
    """
    chosen = dict(zip(site_a["device_id"], site_a["candidate"]))
    rows = []
    for (dev_id, cand), grp in metrics.groupby(["device_id", "candidate"]):
        if dev_id not in chosen:
            continue
        fam = grp["equipment_type"].iloc[0]
        if fam not in PRIMARY:
            continue
        err_col = PRIMARY[fam][0]
        rec = {"family": fam, "device": _dev_label(dev_id),
               "candidate": DISPLAY.get(cand, cand),
               "selected": chosen.get(dev_id) == cand}
        for stage in ("selection", "test"):
            sub = grp[grp["stage"] == stage]
            value = sub[err_col].dropna() if err_col in sub else pd.Series(dtype=float)
            rec[f"{stage}_error"] = float(value.iloc[0]) if not value.empty else None
        if rec["selection_error"] is not None and rec["test_error"] is not None:
            rec["gap"] = rec["test_error"] - rec["selection_error"]
        else:
            rec["gap"] = None
        rows.append(rec)
    out = pd.DataFrame(rows)
    out = out[out[["selection_error", "test_error"]].notna().any(axis=1)]
    out["family"] = pd.Categorical(out["family"], FAMILY_ORDER, ordered=True)
    return out.sort_values(["family", "device", "candidate"]).reset_index(drop=True)


def si_comparability() -> pd.DataFrame:
    """Per-segment driver ranges, taken from the split's own coverage report."""
    path = ROOT / "outputs" / "analysis" / "site_a_temporal_split" / "site_a_ct_split_coverage.csv"
    cov = pd.read_csv(path)
    cov = cov[cov["n"] > 0].copy()
    cov["device"] = cov["tower"].map(_dev_label)
    return cov[["device", "segment", "variable", "n",
                "p05", "median", "p95"]].reset_index(drop=True)


def main() -> int:
    metrics, site_a = load()
    outputs = {
        "table4_site_a.csv": table4(metrics, site_a),
        "table_si_segments.csv": si_segments(metrics, site_a),
        "table_si_selection_vs_test.csv": si_selection_vs_test(metrics, site_a),
        "table_si_comparability.csv": si_comparability(),
    }
    for name, frame in outputs.items():
        frame.to_csv(EVIDENCE / name, index=False)
        print(f"{name}: {len(frame)} rows")

    t4 = outputs["table4_site_a.csv"]
    print(f"\nscored devices: {len(t4)}")
    print(t4.groupby("family", observed=True).size().to_string())
    worse = t4[t4["test_skill"] >= 1.0]
    print(f"\ndevices whose model is no better than predicting the mean: {len(worse)}")
    if not worse.empty:
        print(worse[["device", "test_skill", "driver_excitation"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
