from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from autofmu import __version__
from autofmu.config import load_project, validate_project
from autofmu.pipeline import attribute, calibrate, ingest, report, validate
from autofmu.pipeline import fmu_run as _fmu_run
from autofmu.pipeline import load_fmu_config


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="project YAML config")
    parser.add_argument("--run-id", required=True, help="run identifier")


def _parse_candidates(items) -> dict:
    """``["chiller=EIR,EEIR", "cooling_tower=Merkel"]`` -> ``{type: [names]}``.

    Feeds the project ``fmu_candidates`` override consumed by model_types so a run
    can pick which model types compete without editing YAML.
    """
    out: dict = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--candidates must be TYPE=name,name (got {item!r})")
        dtype, names = item.split("=", 1)
        out[dtype.strip()] = [n.strip() for n in names.split(",") if n.strip()]
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autofmu", description="HVAC FMU auto-modelling")
    parser.add_argument("--version", action="version", version=f"autofmu {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_validate = sub.add_parser("validate-config", help="validate a project config")
    p_validate.add_argument("--config", required=True)

    p_ingest = sub.add_parser("ingest", help="L1: raw BMS -> canonical")
    _add_run_args(p_ingest)

    p_attr = sub.add_parser("attribute", help="L2: modelability gating + windows")
    _add_run_args(p_attr)

    p_cal = sub.add_parser("calibrate", help="L3: gated device calibration")
    _add_run_args(p_cal)
    p_cal.add_argument("--candidates", action="append", default=None, metavar="TYPE=name,name",
                       help="restrict which model types compete (repeatable), e.g. "
                            "--candidates chiller=EIR,EEIR,Carnot --candidates cooling_tower=Merkel")

    p_val = sub.add_parser("validate", help="apply selected models over the full period")
    _add_run_args(p_val)

    p_rep = sub.add_parser("report", help="consolidate the run into one report")
    _add_run_args(p_rep)

    p_fmu = sub.add_parser("fmu-run", help="real FMPy execution of exported FMUs")
    _add_run_args(p_fmu)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate-config":
        config = load_project(Path(args.config))
        errors = validate_project(config)
        if errors:
            print("INVALID config:")
            for error in errors:
                print(f"  - {error}")
            return 1
        print(f"OK: {len(config['devices'])} device(s)")
        return 0

    # fmu-run uses its own lightweight config (no canonical devices required)
    if args.command == "fmu-run":
        config = load_fmu_config(Path(args.config))
        base = _fmu_run(config, args.run_id)
        print(f"fmu-run complete: {base / 'fmu_run' / 'fmu_metrics.csv'}")
        return 0

    config = load_project(Path(args.config))
    errors = validate_project(config)
    if errors:
        print("INVALID config; run validate-config for details")
        return 1

    if args.command == "ingest":
        base = ingest(config, args.run_id)
        print(f"ingest complete: {base}")
        return 0
    if args.command == "attribute":
        base = attribute(config, args.run_id)
        print(f"attribute complete: {base / 'attribute' / 'modelability_report.csv'}")
        return 0
    if args.command == "calibrate":
        overrides = _parse_candidates(getattr(args, "candidates", None))
        if overrides:
            config["fmu_candidates"] = {**config.get("fmu_candidates", {}), **overrides}
        base = calibrate(config, args.run_id)
        print(f"calibrate complete: {base / 'calibrate' / 'selected_models.csv'}")
        return 0
    if args.command == "validate":
        base = validate(config, args.run_id)
        print(f"validate complete: {base / 'validate' / 'full_period_metrics.csv'}")
        return 0
    if args.command == "report":
        base = report(config, args.run_id)
        print(f"report complete: {base / 'run_report.md'}")
        return 0
    return 1


def entrypoint() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    entrypoint()
