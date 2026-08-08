"""Stage 2 (L2): modelability gating + single-device window attribution.

This is the stage the previous AUTO_FMU pipeline skipped. It produces an
explicit, recorded capability decision per device instead of silently running a
passthrough candidate, plus:
  - reconstruction of un-metered quantities (energy balance, COP, etc.);
  - loop-flow attribution onto a device's solo-run windows.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

import yaml

from autofmu.config import adapter_config_path, run_dir
from autofmu.contracts.profiles import get_profile
from autofmu.manifest import RunManifest
from autofmu.modelability.gating import gate_device
from autofmu.modelability.reconstruct import apply_reconstructions, total_source_flow
from autofmu.modelability.windows import align_run_signals, solo_run_mask, solo_run_windows
from autofmu.reporting import table_to_markdown


def _canonical_path(base: Path, device: dict) -> Path:
    return base / device["type"] / device["id"] / "canonical.csv"


def _run_signal_usable(frame: pd.DataFrame, column: str = "run_signal") -> bool:
    if column not in frame:
        return False
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    return bool(np.isfinite(values).any())


def _normalized_timestamps(values: pd.Series, context: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    invalid = int(parsed.isna().sum())
    if invalid:
        raise ValueError(f"{context}: invalid timestamp values: {invalid}")
    duplicates = int(parsed.duplicated().sum())
    if duplicates:
        raise ValueError(f"{context}: duplicate timestamp values: {duplicates}")
    return parsed


def attribute(config: dict, run_id: str) -> Path:
    base = run_dir(config, run_id)
    manifest = RunManifest(base / "manifest.json", run_id)
    manifest.bind_run(config)
    thresholds = config.get("thresholds", {})
    run_on = float(thresholds.get("run_on", 0.5))

    # --- load canonical frames (raw; reconstruction happens AFTER attribution
    # so heat-rate reconstructions can use attributed flow) ---
    frames: Dict[str, pd.DataFrame] = {}
    device_by_id: Dict[str, dict] = {}
    recon_flags: Dict[str, str] = {}
    augmented = set()
    for device in config["devices"]:
        path = _canonical_path(base, device)
        if not path.exists():
            manifest.add_warning(f"{device['id']}: canonical.csv missing; run ingest first")
            continue
        frames[device["id"]] = pd.read_csv(path)
        device_by_id[device["id"]] = device

    # --- join shared boundary signals (e.g. outdoor wet-bulb) by timestamp ---
    from autofmu.config import data_root as _data_root
    root = _data_root(config) if config.get("data_root") else None
    for dev_id, device in device_by_id.items():
        for spec in device.get("join", []):
            src = (root / spec["source_csv"]) if root else Path(spec["source_csv"])
            try:
                ext = pd.read_csv(src, encoding="utf-8-sig")
            except Exception as exc:
                manifest.add_warning(f"{dev_id}: join source unreadable ({src}): {exc}")
                continue
            ts_col, col, alias = spec["timestamp"], spec["column"], spec["as"]
            if ts_col not in ext or col not in ext:
                manifest.add_warning(f"{dev_id}: join columns missing in {src}")
                continue
            sub = ext[[ts_col, col]].rename(columns={ts_col: "timestamp", col: alias})
            sub["timestamp"] = _normalized_timestamps(sub["timestamp"], f"{dev_id} join source {src}")
            f = frames[dev_id].copy()
            f["timestamp"] = _normalized_timestamps(f["timestamp"], f"{dev_id} canonical")
            joined = f.merge(sub, on="timestamp", how="left", validate="one_to_one")
            joined["timestamp"] = joined["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            frames[dev_id] = joined
            augmented.add(dev_id)

    # --- per-GROUP run-signal alignment: solo windows (counts + masks) ---
    # Group defaults to equipment type but can be set per device (e.g. chwp vs
    # cwp) so solo-window attribution respects the actual hydraulic loop.
    by_group: Dict[str, Dict[str, pd.DataFrame]] = {}
    for dev_id, device in device_by_id.items():
        group = device.get("group", device["type"])
        by_group.setdefault(group, {})[dev_id] = frames[dev_id]
    solo_counts: Dict[str, int] = {}
    solo_masks: Dict[str, pd.Series] = {}
    for group_frames in by_group.values():
        aligned = align_run_signals(group_frames)
        solo_counts.update(solo_run_windows(aligned, run_on=run_on))
        solo_masks.update(solo_run_mask(aligned, run_on=run_on))

    # --- loop-flow attribution onto each device's solo windows ---
    flow_flags: Dict[str, str] = {}
    for dev_id, device in device_by_id.items():
        spec = device.get("flow_attribution")
        if not spec:
            continue
        column = spec["source_column"]
        target = frames[dev_id]

        # Two evidence grades for the same reconstruction.
        #
        # `source_devices` names the specific upstream devices hydraulically
        # paired with this one. Where the plant topology is known -- Site A
        # tower CT_01 is fed by chiller CH_01 and heat exchanger HX_01, and by
        # nothing else -- the pairing itself says whose flow this is, so no
        # exclusivity argument is needed and the attribution holds whenever the
        # meters read.
        #
        # `source_type` is the fallback for a shared header with no known
        # pairing: sum every device of that type and keep only the target's
        # solo-run windows, because that is the only time the loop total can be
        # ascribed to one device. Correct, but it discards most of the record --
        # on Site A cooling towers it left 13.3% of rows on CT-01 and none at
        # all on CT-06 and CT-07.
        explicit = spec.get("source_devices")
        if explicit:
            missing = [i for i in explicit if i not in frames]
            if missing:
                raise KeyError(
                    f"{dev_id} flow_attribution.source_devices references "
                    f"device(s) not in this project: {missing}. Paired "
                    f"attribution reads their canonical frames, so they must be "
                    f"listed under `devices:` even if they are not themselves "
                    f"calibration targets. A project scoped to one equipment "
                    f"family cannot use paired attribution across families.")
            src_frames = {i: frames[i] for i in explicit}
            source_label = "+".join(explicit)
            use_solo = bool(spec.get("require_solo", False))
        else:
            src_type = spec["source_type"]
            src_frames = {i: frames[i] for i, d in device_by_id.items()
                          if d["type"] == src_type}
            source_label = f"all {src_type}"
            use_solo = bool(spec.get("require_solo", True))

        total = total_source_flow(src_frames, column)
        if use_solo and not _run_signal_usable(target):
            target["attributed_flow_m3_h"] = np.nan
            augmented.add(dev_id)
            flow_flags[dev_id] = "attribution_blocked:run_signal_absent"
            continue
        ts = _normalized_timestamps(target["timestamp"], f"{dev_id} flow attribution")
        attr = total.reindex(ts).to_numpy(dtype=float) if len(total) else np.full(len(target), np.nan)
        grade = "paired_devices"
        if use_solo:
            solo = solo_masks.get(dev_id)
            if solo is not None:
                solo_aligned = solo.reindex(ts).fillna(False).to_numpy()
                attr = np.where(solo_aligned, attr, np.nan)
                grade = "solo_window"
        target["attributed_flow_m3_h"] = attr
        augmented.add(dev_id)
        n_attr = int(np.isfinite(attr).sum())
        flow_flags[dev_id] = (
            f"attributed_flow_m3_h<-sum({source_label}.{column})"
            f"|grade={grade}|rows={n_attr}")

    # --- reconstruction (after attribution; heat_rate can use attributed flow) ---
    qa_rows = []
    for dev_id, device in device_by_id.items():
        spec = device.get("reconstruct")
        if not spec:
            continue
        frames[dev_id], flags = apply_reconstructions(frames[dev_id], spec)
        recon_flags[dev_id] = ";".join(flags)
        augmented.add(dev_id)
        # energy-conservation QA: if both chilled- and condenser-side heat were
        # reconstructed they must agree (HX) -> report the residual.
        f = frames[dev_id]
        if "Q_W" in f and "Q_cond_W" in f:
            q1 = pd.to_numeric(f["Q_W"], errors="coerce")
            q2 = pd.to_numeric(f["Q_cond_W"], errors="coerce")
            both = q1.notna() & q2.notna() & (q1 > 0)
            if bool(both.any()):
                resid = float((((q2 - q1).abs()) / q1)[both].median() * 100.0)
                qa_rows.append({"device_id": dev_id, "check": "HX_two_side_Q_residual_pct",
                                "value": round(resid, 2), "N": int(both.sum())})

    # --- write augmented canonical frames ---
    for dev_id in augmented:
        out = _canonical_path(base, device_by_id[dev_id]).with_name("canonical_attributed.csv")
        frames[dev_id].to_csv(out, index=False)
        manifest.add_artifact(out, base)

    # --- capability gate per device ---
    rows = []
    for dev_id, device in device_by_id.items():
        profile = get_profile(device["type"])
        # Adapter column map lets the gate run the synthetic-channel detector:
        # a calibration target that the BMS derived from another point cannot
        # validate anything, so it blocks the device instead of scoring it.
        try:
            adapter = yaml.safe_load(
                adapter_config_path(config, device).read_text(encoding="utf-8")) or {}
            adapter_columns = adapter.get("columns")
        except (OSError, yaml.YAMLError):
            adapter_columns = None
        result = gate_device(dev_id, frames[dev_id], profile, thresholds,
                             adapter_columns=adapter_columns)
        row = result.as_row()
        row["solo_run_rows"] = solo_counts.get(dev_id, "")
        row["reconstructed"] = recon_flags.get(dev_id, "")
        row["flow_attribution"] = flow_flags.get(dev_id, "")
        row["candidates"] = ",".join(profile.candidates)
        rows.append(row)

    report = pd.DataFrame(rows)
    stage = base / "attribute"
    stage.mkdir(parents=True, exist_ok=True)
    report_csv = stage / "modelability_report.csv"
    report.to_csv(report_csv, index=False)
    summary = stage / "summary.md"
    counts = report["level"].value_counts().to_dict() if not report.empty else {}
    summary.write_text(
        "# Modelability Report\n\n"
        "L2 capability gating per device (full_physical / nominal_only / blocked). "
        "`solo_run_rows` = timestamps where the device is the only one of its type "
        "running, i.e. where shared/main-pipe flow can be attributed to it.\n\n"
        + ("Level counts: " + ", ".join(f"{k}={v}" for k, v in counts.items()) + "\n\n" if counts else "")
        + (table_to_markdown(report) if not report.empty else "_no devices_")
        + "\n",
        encoding="utf-8",
    )
    if qa_rows:
        qa_csv = stage / "reconstruction_qa.csv"
        pd.DataFrame(qa_rows).to_csv(qa_csv, index=False)
        manifest.add_artifact(qa_csv, base)
    manifest.add_artifact(report_csv, base)
    manifest.add_artifact(summary, base)
    manifest.record_stage("attribute")
    manifest.write()
    return base
