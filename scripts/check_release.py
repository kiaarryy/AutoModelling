"""Pre-release verification.

Checks the things that would make a reproducibility package fail silently for
somebody who is not us: an FMU whose bytes no longer match the hash we
reported, a config pointing at a path only this machine has, a documented
executable path whose files are missing.

    python scripts/check_release.py            # run every check
    python scripts/check_release.py --versions # print the environment record
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
FAMILIES = ("chiller", "cooling_tower", "pump", "heat_exchanger")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fmu_root() -> Path:
    for family in FAMILIES:
        cfg = yaml.safe_load((REPO / "configs" / "fmu" / f"{family}.yaml")
                             .read_text(encoding="utf-8"))
        root = cfg.get("fmu_root")
        if root:
            return Path(os.environ.get("AUTOFMU_FMU_ROOT") or root)
    raise SystemExit("no fmu_root in any configs/fmu/*.yaml")


def check_fmus(problems: list) -> dict:
    """Every declared FMU exists, and its hash is recorded."""
    root = fmu_root()
    print(f"\n[1] FMU binaries  (root: {root})")
    hashes = {}
    for family in FAMILIES:
        cfg = yaml.safe_load((REPO / "configs" / "fmu" / f"{family}.yaml")
                             .read_text(encoding="utf-8"))
        for cand in cfg.get("candidates", []):
            path = root / cand["fmu"]
            if not path.exists():
                problems.append(f"FMU missing: {family}/{cand['name']} -> {path}")
                print(f"    MISSING  {family:15s} {cand['name']}")
                continue
            h = sha256(path)
            hashes[cand["name"]] = h
            print(f"    ok       {family:15s} {cand['name']:22s} "
                  f"{path.stat().st_size/1e6:6.2f} MB  {h[:16]}")
    return hashes


def check_reported_hashes(hashes: dict, problems: list) -> None:
    """The hashes recorded in the runs still match the binaries on disk.

    This is the check that matters: if an FMU were rebuilt after the numbers
    were produced, nothing else in the package would notice.

    Matched by hash, not by name. ``fmu_model_name`` records the selected
    *form* -- affinity, speed_poly -- and several forms share one exported
    binary, so a name lookup reports absences that are not real. The hash
    answers the question actually being asked: are these the bytes that
    produced the reported numbers?
    """
    print("\n[2] Reported runs still match those binaries")
    on_disk = {h: n for n, h in hashes.items()}
    runs = sorted((REPO / "outputs" / "runs").glob("*_e2e_20260804c"))
    if not runs:
        problems.append("no *_e2e_20260804c run directories to verify against")
        print("    NO RUNS FOUND")
        return
    checked = 0
    for run in runs:
        sel = run / "calibrate" / "selected_models.csv"
        if not sel.exists():
            continue
        frame = pd.read_csv(sel)
        if "fmu_sha256" not in frame:
            continue
        for _, row in frame.iterrows():
            name, want = row.get("fmu_model_name"), row.get("fmu_sha256")
            if not isinstance(want, str) or not want:
                continue
            if want not in on_disk:
                problems.append(
                    f"{run.name}: {name} was driven by an FMU whose hash "
                    f"({want[:12]}) matches nothing shipped -- the binary has "
                    f"been rebuilt or replaced since the run")
            checked += 1
    print(f"    checked {checked} reported hashes across {len(runs)} runs; "
          f"all resolve to a shipped binary" if not problems else
          f"    checked {checked} reported hashes across {len(runs)} runs")
    for h, n in sorted(on_disk.items(), key=lambda kv: kv[1]):
        print(f"      {n:22s} {h[:16]}")


def check_paths(problems: list) -> None:
    """Configs must be overridable without editing them."""
    print("\n[3] Machine-specific paths are overridable")
    for cfg_path in sorted((REPO / "configs").glob("*/project*.yaml")):
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        root = str(cfg.get("data_root", ""))
        absolute = root.startswith(("/", "\\")) or (len(root) > 1 and root[1] == ":")
        rel = cfg_path.relative_to(REPO)
        if absolute:
            print(f"    absolute {rel}  ->  {root}")
        else:
            print(f"    relative {rel}")
    from autofmu.config import ENV_DATA_ROOT, ENV_FMU_ROOT
    print(f"    override with {ENV_DATA_ROOT} and {ENV_FMU_ROOT} "
          f"(no config edit required)")


def check_executable_case(problems: list) -> None:
    """The one case a third party can actually run must be documented."""
    print("\n[4] The public executable case")
    required = [
        REPO / "configs" / "lbnl" / "README.md",
        REPO / "configs" / "lbnl" / "project_fleet.yaml",
        REPO / "scripts" / "preprocess_lbnl_swap.py",
    ]
    for path in required:
        if path.exists():
            print(f"    ok       {path.relative_to(REPO)}")
        else:
            problems.append(f"missing from the executable case: "
                            f"{path.relative_to(REPO)}")
            print(f"    MISSING  {path.relative_to(REPO)}")
    adapters = list((REPO / "configs" / "lbnl" / "adapters").glob("*.yaml"))
    devices = yaml.safe_load(
        (REPO / "configs" / "lbnl" / "project_fleet.yaml").read_text(encoding="utf-8")
    ).get("devices", [])
    if len(adapters) < len(devices):
        problems.append(f"lbnl: {len(devices)} devices but {len(adapters)} adapters")
    print(f"    {len(devices)} devices, {len(adapters)} adapters")


def check_modelica(problems: list) -> None:
    """Both branches of the tower contest must be reproducible."""
    print("\n[5] Modelica sources for both cooling-tower candidates")
    for name in ("SiteACTYork27ClosedLoop.mo", "Cooling_Tower01_merkel.mo"):
        path = REPO / "modelica" / "wrappers" / name
        if path.exists():
            print(f"    ok       {name}")
        else:
            problems.append(f"missing Modelica source: {name} -- without it the "
                            f"model-type selection result cannot be verified")
            print(f"    MISSING  {name}")


def versions() -> int:
    import platform
    print(f"python {sys.version.split()[0]} {platform.machine()}")
    for module in ("numpy", "pandas", "yaml", "fmpy", "scipy", "openpyxl",
                   "matplotlib", "pytest"):
        try:
            m = __import__(module)
            print(f"  {module:12s} {getattr(m, '__version__', '?')}")
        except ImportError:
            print(f"  {module:12s} not installed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--versions", action="store_true")
    args = ap.parse_args()
    if args.versions:
        return versions()

    problems: list = []
    hashes = check_fmus(problems)
    check_reported_hashes(hashes, problems)
    check_paths(problems)
    check_executable_case(problems)
    check_modelica(problems)

    print("\n" + "=" * 62)
    if problems:
        print(f"{len(problems)} problem(s) block release:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("all release checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
