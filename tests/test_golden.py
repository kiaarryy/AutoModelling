"""Golden regression: a committed synthetic dataset must keep producing the
same gating decisions and (near-exact) calibration on a known cubic pump.

Locks the end-to-end pipeline against silent regressions during refactors.
"""
import json
from pathlib import Path

import pandas as pd

from autofmu.config import load_project
from autofmu.pipeline import attribute, calibrate, ingest, report, validate

GOLDEN = Path(__file__).parent / "data" / "golden" / "project.yaml"


def _run(tmp_path):
    config = load_project(GOLDEN)
    config["outputs_dir"] = str(tmp_path)  # absolute -> resolve_path returns as-is
    for stage in (ingest, attribute, calibrate, validate, report):
        stage(config, "golden")
    base = Path(tmp_path) / "runs" / "golden"
    return base


def test_golden_gating_and_calibration(tmp_path):
    base = _run(tmp_path)

    gate = pd.read_csv(base / "attribute" / "modelability_report.csv").set_index("device_id")
    assert gate.loc["CH1", "level"] == "full_physical"   # full instrumentation
    assert gate.loc["P1", "level"] == "full_physical"
    assert gate.loc["HX1", "level"] == "nominal_only"     # no flow -> can't close Q

    sel = pd.read_csv(base / "calibrate" / "selected_models.csv").set_index("device_id")
    # pump power is an exact cubic in speed -> a speed candidate recovers it
    assert sel.loc["P1", "status"] == "ok"
    assert float(sel.loc["P1", "MAPE_pct"]) < 0.5

    val = pd.read_csv(base / "validate" / "full_period_metrics.csv").set_index("device_id")
    assert float(val.loc["P1", "MAPE_pct"]) < 0.5

    assert (base / "run_report.md").exists()
    manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["identity"]["dataset"] == "golden"
    assert manifest["identity"]["config_sha256"]
    assert manifest["identity"]["git_commit"]


def test_skill_reaches_the_output_csv(tmp_path):
    """A metric that is computed but never written is the same as absent.

    skill_vs_mean shipped inside regression_metrics and stopped there: every
    device module copies a hand-picked set of keys into its result dict, and
    none of them picked this one up, so no run ever reported it. The decision
    to report skill alongside CVRMSE rather than change the device count rests
    on this column existing, so the test counts it.
    """
    base = _run(tmp_path)
    sel = pd.read_csv(base / "calibrate" / "selected_models.csv").set_index("device_id")
    assert "test_skill" in sel.columns
    skill = pd.to_numeric(sel.loc["P1", "test_skill"], errors="coerce")
    # the golden pump is an exact cubic in speed, so the model must beat the mean
    assert 0.0 <= float(skill) < 0.1, skill
