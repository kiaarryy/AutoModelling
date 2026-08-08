# -*- coding: utf-8 -*-
"""Capture measured-versus-simulated traces for four representative devices.

The manuscript lost its only measured-versus-simulated evidence when Figure 7
was redrawn around the temporal partition. Restoring it needs the per-sample
model output, which the pipeline scores but does not persist.

Rather than teach four device engines to write time series, this exploits the
one function they all pass through: ``evaluation.metric_pairs`` receives the
finite measured/simulated pair on the way to every metric. Recording there
costs one hook and no change to any engine.

The recorder has no notion of which segment a call belongs to, so the calls are
matched afterwards by sample count against the test-segment N already published
in docs/evidence. That is exact: N is the count of finite pairs, which is what
metric_pairs returns.

One device per equipment family, chosen to be defensible rather than flattering:

CH-03   the best-evidenced chiller, driver excitation 0.81
CT-01   the tower whose partition Figure 7a already shows, so the two figures
        describe the same device
CDWP-03 the only Site A pump whose drive genuinely moves (excitation 0.31);
        picking one of the near-static pumps would show a flat line agreeing
        with a flat line
HX-02   the exchanger that selects the effectiveness-NTU candidate

Usage:
    python scripts/capture_validation_series.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autofmu import evaluation                       # noqa: E402
from autofmu.config import load_project              # noqa: E402
from autofmu.pipeline.calibrate import calibrate     # noqa: E402

CONFIG = ROOT / "configs" / "site_a" / "project_fleet.yaml"
SOURCE_RUN = ROOT / "outputs" / "runs" / "sitea_e2e_20260804c"
RUN_ID = "sitea_validation_series"
OUT = ROOT / "docs" / "evidence" / "validation_series"

# device -> family, label, unit, published test N, and the published value of
# the family's primary metric.
#
# The sample count alone does not identify the right call: both candidates are
# scored on the same rows, and cooling towers and heat exchangers score two
# variables per candidate, so four calls can share one N. Matching the metric
# as well picks the selected candidate's primary variable, and doubles as a
# check that the captured series is the one the paper reports.
DEVICES = {
    "CH_03":   ("chiller", "Electrical power", "kW", 282, "cvrmse", 4.4869),
    "CT_01":   ("cooling_tower", "Leaving water temperature", "°C",
                17894, "rmse", 0.51785),
    "CDWP_03": ("pump", "Electrical power", "kW", 790, "cvrmse", 9.9774),
    "HX_02":   ("heat_exchanger", "Heat transfer rate", "kW",
                9909, "cvrmse", 7.9715),
}


def prepare_run(config: dict) -> dict:
    """A four-device copy of the Site A project, reusing the attribute stage."""
    config = dict(config)
    config["devices"] = [d for d in config["devices"] if d["id"] in DEVICES]
    missing = set(DEVICES) - {d["id"] for d in config["devices"]}
    if missing:
        raise SystemExit(f"config does not define: {sorted(missing)}")

    from autofmu.config import run_dir
    base = run_dir(config, RUN_ID)
    (base / "attribute").mkdir(parents=True, exist_ok=True)
    for name in ("modelability_report.csv", "summary.md"):
        src = SOURCE_RUN / "attribute" / name
        if src.exists():
            shutil.copy(src, base / "attribute" / name)
    # the device engines read the attributed frames the earlier run produced
    for dev in config["devices"]:
        fam = dev["type"]
        src = SOURCE_RUN / fam / dev["id"]
        if src.exists():
            dst = base / fam / dev["id"]
            dst.mkdir(parents=True, exist_ok=True)
            for f in src.glob("*.csv"):
                shutil.copy(f, dst / f.name)
    return config


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    config = prepare_run(load_project(CONFIG))

    evaluation.start_recording()
    try:
        calibrate(config, RUN_ID)
    finally:
        pairs = evaluation.recorded()
        evaluation.stop_recording()
    print(f"captured {len(pairs)} measured/simulated pairs")

    by_n: dict[int, list] = {}
    for measured, simulated in pairs:
        by_n.setdefault(len(measured), []).append((measured, simulated))

    def score(measured, simulated, kind):
        rmse = float(np.sqrt(np.mean((measured - simulated) ** 2)))
        if kind == "rmse":
            return rmse
        mean = float(np.mean(measured))
        return 100.0 * rmse / mean if mean else float("inf")

    written = 0
    for dev, (fam, label, unit, want, kind, expect) in DEVICES.items():
        hits = by_n.get(want)
        if not hits:
            near = sorted(by_n, key=lambda n: abs(n - want))[:4]
            print(f"  {dev}: no capture with N={want}; nearest {near}")
            continue
        scored = [(abs(score(m, s, kind) - expect), m, s) for m, s in hits]
        gap, measured, simulated = min(scored, key=lambda t: t[0])
        tol = max(0.02 * abs(expect), 1e-3)
        if gap > tol:
            print(f"  {dev}: no call at N={want} matches the published "
                  f"{kind} {expect} (closest differs by {gap:.4g}) -- skipped")
            continue
        frame = pd.DataFrame({"measured": measured, "simulated": simulated})
        frame.insert(0, "sample", np.arange(len(frame)))
        frame.to_csv(OUT / f"{dev}.csv", index=False)
        got = score(measured, simulated, kind)
        print(f"  {dev:9s} N={len(frame):>6,d}  {kind}={got:.4g} "
              f"(published {expect})  -> {dev}.csv")
        written += 1

    print(f"wrote {written} of {len(DEVICES)} series to {OUT}")
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
