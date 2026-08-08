# -*- coding: utf-8 -*-
"""Cross-site evidence, under the Site A-D naming the paper now uses.

Two things forced this script into existence. First, the manuscript's Table 2
and Figures 6 and 7 were built when the study covered one site; the study now
covers four, and a Site-A-only readiness table cannot introduce four case
studies. Second, the earlier cross-site figure printed the operators' real names
on its axis, which the confidentiality boundary does not allow.

Site letters follow the companion chiller study: its Site A and Site C are the
same physical plants as ours. Its Site B is a commercial complex that does not
appear here, so our Site B -- a second data centre -- is a different plant, and
the manuscript says so rather than leaving a reader to assume otherwise.

Outputs, all in docs/evidence/:

``site_profile.csv``            one row per site, for the case-study text
``site_family_readiness.csv``   site x family, the replacement for Table 2
``cross_site_devices_named.csv``  every device, with the site letter
``cross_site_selection_vs_test.csv``  every candidate at every site
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs" / "evidence"
RUNS = ROOT / "outputs" / "runs"

# internal key -> (letter, run directory, description)
# The sampling interval and record span are measured from each site's own
# canonical frames rather than declared here: an interval stated in prose is
# exactly the kind of number that goes stale unnoticed, and the first draft of
# this file had Site D at 60 min when the data say 30.
SITES = {
    "site_a":  ("A", "sitea_e2e_20260804c", "Hyperscale data centre"),
    "tencent": ("B", "tencent_e2e_20260804c", "Commercial data centre"),
    "hkust":   ("C", "hkust_e2e_20260804c", "University campus central plant"),
    "lbnl":    ("D", "lbnl_e2e_20260804c", "Public research dataset"),
}


def measured_cadence(run: str) -> tuple[str, int, str]:
    """(interval, span in days, calendar window) read off a canonical frame."""
    frames = sorted((RUNS / run).rglob("canonical.csv"))
    if not frames:
        raise SystemExit(f"{run}: no canonical frame to measure cadence from")
    stamps = pd.to_datetime(pd.read_csv(frames[0], usecols=[0]).iloc[:, 0],
                            errors="coerce", utc=True).dropna()
    step = stamps.diff().dt.total_seconds().dropna()
    minutes = int(round(step.mode().iloc[0] / 60)) if len(step) else 0
    span = int((stamps.max() - stamps.min()).days)
    window = f"{stamps.min():%Y-%m} to {stamps.max():%Y-%m}"
    return f"{minutes} min", span, window

# the variable each family is ranked on inside its own run
PRIMARY = {"chiller": "P_CVRMSE_pct", "pump": "P_CVRMSE_pct",
           "cooling_tower": "T_RMSE_K", "heat_exchanger": "Q_CVRMSE_pct"}

FAMILY_ORDER = ["chiller", "cooling_tower", "pump", "heat_exchanger"]
FAMILY_LABEL = {"chiller": "Chiller", "cooling_tower": "Cooling tower",
                "pump": "Pump", "heat_exchanger": "Heat exchanger"}

DISPLAY = {"EIR": "ElectricEIR", "EEIR": "ElectricReformulatedEIR",
           "Merkel": "Merkel", "YorkCalc": "YorkCalc27",
           "YorkCalc27": "YorkCalc27", "affinity": "Affinity",
           "speed_poly": "SpeedPoly",
           "ConstantEffectiveness": "ConstantEffectiveness",
           "PlateEffectivenessNTU": "PlateEffNTU"}


def letter(site_key: str) -> str:
    return SITES[site_key][0]


def load_devices() -> pd.DataFrame:
    """Every device at every site, with its site letter attached."""
    d = pd.read_csv(EVIDENCE / "cross_site_devices.csv")
    d["site_letter"] = d["site"].map(letter)
    if d["site_letter"].isna().any():
        unknown = sorted(d.loc[d["site_letter"].isna(), "site"].unique())
        raise SystemExit(f"unmapped site keys: {unknown}")
    d["family"] = pd.Categorical(d["equipment_type"], FAMILY_ORDER, ordered=True)
    return d.sort_values(["site_letter", "family", "device_id"])


def readiness() -> pd.DataFrame:
    """Raw and operating record counts, per site and family.

    The published Table 2 counted only Site A and omitted chillers entirely.
    Counts come from each run's own modelability report rather than from the
    prose, so the table cannot drift from the pipeline again.
    """
    rows = []
    for key, (lett, run, _desc) in SITES.items():
        path = RUNS / run / "attribute" / "modelability_report.csv"
        if not path.exists():
            raise SystemExit(f"{key}: no modelability report at {path}")
        m = pd.read_csv(path)
        rows.append(m.assign(site=key, site_letter=lett))
    m = pd.concat(rows, ignore_index=True)

    dev = load_devices()[["site", "device_id", "status", "test_CVRMSE_pct"]]
    m = m.merge(dev, on=["site", "device_id"], how="left")

    out = []
    for (lett, fam), g in m.groupby(["site_letter", "equipment_type"],
                                    observed=True):
        scored = g[g["status"] == "ok"]
        raw = int(pd.to_numeric(g["rows"], errors="coerce").fillna(0).sum())
        oper = int(pd.to_numeric(g["on_rows"], errors="coerce").fillna(0).sum())
        errs = pd.to_numeric(scored["test_CVRMSE_pct"], errors="coerce").dropna()
        out.append({
            "site": lett,
            "family": fam,
            "devices": len(g),
            "scored": len(scored),
            "raw_records": raw,
            "operating_records": oper,
            "operating_pct": round(100 * oper / raw, 1) if raw else 0.0,
            "median_test_error_pct": round(float(errs.median()), 2)
            if len(errs) else None,
        })
    r = pd.DataFrame(out)
    r["family"] = pd.Categorical(r["family"], FAMILY_ORDER, ordered=True)
    return r.sort_values(["site", "family"]).reset_index(drop=True)


def site_profile(read: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, (lett, run, desc) in SITES.items():
        step, span, window = measured_cadence(run)
        g = read[read["site"] == lett]
        fams = [FAMILY_LABEL[f] for f in FAMILY_ORDER
                if f in set(g["family"]) and int(g.loc[g["family"] == f, "devices"].sum())]
        rows.append({
            "site": lett,
            "description": desc,
            "sampling_interval": step,
            "span_days": span,
            "window": window,
            "families": len(fams),
            "family_list": ", ".join(fams),
            "devices": int(g["devices"].sum()),
            "scored": int(g["scored"].sum()),
            "scored_pct": round(100 * g["scored"].sum() / g["devices"].sum(), 1),
            "raw_records": int(g["raw_records"].sum()),
            "operating_records": int(g["operating_records"].sum()),
            "public": key == "lbnl",
        })
    return pd.DataFrame(rows).sort_values("site").reset_index(drop=True)


def selection_vs_test() -> pd.DataFrame:
    """Selection- and test-segment error for every candidate, at every site."""
    chosen = load_devices().set_index(["site", "device_id"])["candidate"].to_dict()
    rows = []
    for key, (lett, run, _d) in SITES.items():
        path = RUNS / run / "calibrate" / "all_candidate_metrics.csv"
        m = pd.read_csv(path)
        for (dev, cand), g in m.groupby(["device_id", "candidate"]):
            fam = g["equipment_type"].iloc[0]
            col = PRIMARY.get(fam)
            if col is None or col not in g:
                continue
            rec = {"site": lett, "family": fam,
                   "device": dev.replace("_", "-"),
                   "candidate": DISPLAY.get(cand, cand),
                   "selected": chosen.get((key, dev)) == cand}
            for stage in ("selection", "test"):
                v = pd.to_numeric(g.loc[g["stage"] == stage, col],
                                  errors="coerce").dropna()
                rec[f"{stage}_error"] = float(v.iloc[0]) if len(v) else None
            both = rec["selection_error"] is not None and rec["test_error"] is not None
            rec["ratio"] = (rec["test_error"] / rec["selection_error"]
                            if both and rec["selection_error"] else None)
            rows.append(rec)
    out = pd.DataFrame(rows)
    out = out[out[["selection_error", "test_error"]].notna().any(axis=1)]
    out["family"] = pd.Categorical(out["family"], FAMILY_ORDER, ordered=True)
    return out.sort_values(["site", "family", "device", "candidate"]).reset_index(drop=True)


def main() -> int:
    devices = load_devices()
    read = readiness()
    profile = site_profile(read)
    svt = selection_vs_test()

    devices.to_csv(EVIDENCE / "cross_site_devices_named.csv", index=False)
    read.to_csv(EVIDENCE / "site_family_readiness.csv", index=False)
    profile.to_csv(EVIDENCE / "site_profile.csv", index=False)
    svt.to_csv(EVIDENCE / "cross_site_selection_vs_test.csv", index=False)

    pd.set_option("display.width", 200)
    print(profile[["site", "description", "sampling_interval", "span_days",
                   "window", "families", "devices", "scored",
                   "scored_pct"]].to_string(index=False))
    print()
    sel = svt[svt["selected"]].dropna(subset=["ratio"])
    worse = int((sel["ratio"] > 1).sum())
    print(f"selected candidates with both segments scored: {len(sel)}")
    print(f"  test error worse than selection error: {worse} "
          f"({100*worse/len(sel):.0f}%), max {sel['ratio'].max():.2f}x")
    print(f"\ntotals: {int(profile['devices'].sum())} devices, "
          f"{int(profile['scored'].sum())} scored")
    return 0


if __name__ == "__main__":
    sys.exit(main())
