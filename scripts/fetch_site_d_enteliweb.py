"""Download mapped Site D enteliWEB Trend Logs into per-device CSV files."""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Dict, List, Optional, Tuple
import urllib.request

import numpy as np
import pandas as pd

from autofmu.adapters.enteliweb import parse_trend_log


TRUE_VALUES = {"1", "active", "on", "running", "true", "yes"}
FALSE_VALUES = {"0", "inactive", "off", "stopped", "false", "no"}


def _load_env(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, value = text.split("=", 1)
        values[key.strip()] = value.strip()
    for key in ("SITE_D_API_USERNAME", "SITE_D_API_PASSWORD"):
        if not values.get(key):
            raise ValueError("credential file is missing %s" % key)
    return values


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _fetch(url: str, authorization: str, timeout: float, retries: int) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": authorization,
            "Accept": "application/xml",
            "User-Agent": "AutoFMU-SiteD/0.1",
        },
    )
    last_error = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except Exception as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(0.5 * (attempt + 1))
    raise RuntimeError("GET failed after %d attempt(s): %s" % (retries, last_error))


def _numeric(raw: str, field: str) -> float:
    text = str(raw).strip().lower()
    if field == "run_signal":
        if text in TRUE_VALUES:
            return 1.0
        if text in FALSE_VALUES:
            return 0.0
    try:
        return float(text)
    except ValueError:
        return np.nan


def _download_one(
    row: dict,
    authorization: str,
    raw_dir: Path,
    cache_dir: Optional[Path],
    cache_path: Optional[Path],
    timeout: float,
    retries: int,
) -> Tuple[dict, pd.DataFrame]:
    filename = "%s__%s.xml" % (
        _safe_name(row["device_id"]), _safe_name(row["canonical_field"])
    )
    cached = cache_path or (cache_dir / filename if cache_dir else None)
    trend = None
    if cached and cached.exists():
        payload = cached.read_bytes()
        trend = parse_trend_log(payload)
        if trend.object_name == row.get("trendlog_label"):
            transfer_source = "cache"
        else:
            payload = _fetch(row["trendlog_url"], authorization, timeout, retries)
            trend = None
            transfer_source = "api_cache_mismatch"
    else:
        payload = _fetch(row["trendlog_url"], authorization, timeout, retries)
        transfer_source = "api"
    if trend is None:
        trend = parse_trend_log(payload)
    raw_path = raw_dir / filename
    raw_path.write_bytes(payload)
    scale = float(row.get("scale") or 1.0)
    valid_samples = [
        sample for sample in trend.samples if "error-entry" not in sample.flags.split(";")
    ]
    frame = pd.DataFrame(
        {
            "timestamp": [sample.timestamp for sample in valid_samples],
            row["canonical_field"]: [
                _numeric(sample.value, row["canonical_field"]) * scale
                for sample in valid_samples
            ],
        }
    )
    manifest = {
        "device_id": row["device_id"],
        "equipment_type": row["equipment_type"],
        "canonical_field": row["canonical_field"],
        "trendlog_url": row["trendlog_url"],
        "object_name": trend.object_name,
        "interval_seconds": trend.interval_seconds,
        "record_count": trend.record_count,
        "valid_record_count": len(valid_samples),
        "error_entry_count": trend.record_count - len(valid_samples),
        "total_record_count": trend.total_record_count,
        "first_timestamp": trend.samples[0].timestamp.isoformat() if trend.samples else "",
        "last_timestamp": trend.samples[-1].timestamp.isoformat() if trend.samples else "",
        "raw_xml": str(raw_path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "transfer_source": transfer_source,
        "status": "ok",
        "error": "",
    }
    return manifest, frame


def _align_device_parts(parts: List[Tuple[pd.DataFrame, float]]) -> Tuple[pd.DataFrame, float]:
    """Align periodic channels and carry event channels onto their common grid."""
    periodic = [(frame, interval) for frame, interval in parts if interval > 0]
    events = [(frame, interval) for frame, interval in parts if interval <= 0]
    if not periodic:
        raise ValueError("device has no periodic Trend Log channel")
    interval_seconds = float(np.median([interval for _, interval in periodic]))
    # Lower-case ``s`` is accepted by both older pandas releases and pandas 3;
    # the upper-case alias was removed in pandas 3.
    frequency = "%ds" % int(round(interval_seconds))
    prepared = []
    starts = []
    ends = []
    for frame, _ in periodic:
        current = frame.copy()
        current["timestamp"] = pd.to_datetime(current["timestamp"], utc=True).dt.round(frequency)
        current = current.drop_duplicates("timestamp", keep="last").sort_values("timestamp")
        prepared.append(current)
        starts.append(current["timestamp"].min())
        ends.append(current["timestamp"].max())
    start = max(starts).ceil(frequency)
    end = min(ends).floor(frequency)
    if start > end:
        raise ValueError("periodic Trend Log channels have no common time window")
    merged = pd.DataFrame({"timestamp": pd.date_range(start, end, freq=frequency)})
    for current in prepared:
        merged = merged.merge(current, on="timestamp", how="left", validate="one_to_one")

    for frame, _ in events:
        current = frame.copy()
        current["timestamp"] = pd.to_datetime(current["timestamp"], utc=True)
        current = current.drop_duplicates("timestamp", keep="last").sort_values("timestamp")
        merged = pd.merge_asof(
            merged.sort_values("timestamp"),
            current,
            on="timestamp",
            direction="backward",
            allow_exact_matches=True,
        )
    return merged.sort_values("timestamp").reset_index(drop=True), interval_seconds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-csv", type=Path, required=True)
    parser.add_argument("--credentials", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--raw-cache-dir", type=Path)
    parser.add_argument("--raw-cache-manifest", type=Path)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 8:
        raise ValueError("workers must be between 1 and 8")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("output directory is not empty: %s" % args.output_dir)

    credentials = _load_env(args.credentials)
    token = base64.b64encode(
        (credentials["SITE_D_API_USERNAME"] + ":" + credentials["SITE_D_API_PASSWORD"]).encode()
    ).decode()
    authorization = "Basic " + token
    rows = list(csv.DictReader(args.map_csv.open(encoding="utf-8-sig")))
    if any(not row.get("trendlog_url") for row in rows):
        raise ValueError("point map contains an empty trendlog_url")

    raw_dir = args.output_dir / "raw_xml"
    device_dir = args.output_dir / "devices"
    raw_dir.mkdir(parents=True, exist_ok=True)
    device_dir.mkdir(parents=True, exist_ok=True)
    manifests = []
    frames: Dict[str, List[Tuple[pd.DataFrame, float]]] = {}
    cache_by_url: Dict[str, Path] = {}
    if args.raw_cache_manifest:
        cached_rows = csv.DictReader(args.raw_cache_manifest.open(encoding="utf-8-sig"))
        cache_by_url = {
            row["trendlog_url"]: Path(row["raw_xml"])
            for row in cached_rows
            if row.get("status") == "ok" and row.get("raw_xml")
        }
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _download_one,
                row,
                authorization,
                raw_dir,
                args.raw_cache_dir,
                cache_by_url.get(row["trendlog_url"]),
                args.timeout,
                args.retries,
            ): row
            for row in rows
        }
        for future in as_completed(futures):
            row = futures[future]
            try:
                manifest, frame = future.result()
                frames.setdefault(row["device_id"], []).append(
                    (frame, float(manifest["interval_seconds"]))
                )
            except Exception as exc:
                manifest = {
                    "device_id": row["device_id"],
                    "equipment_type": row["equipment_type"],
                    "canonical_field": row["canonical_field"],
                    "trendlog_url": row["trendlog_url"],
                    "status": "error",
                    "error": "%s: %s" % (type(exc).__name__, exc),
                }
            manifests.append(manifest)

    manifest_frame = pd.DataFrame(manifests).sort_values(
        ["equipment_type", "device_id", "canonical_field"]
    )
    manifest_frame.to_csv(args.output_dir / "download_manifest.csv", index=False)
    errors = manifest_frame[manifest_frame["status"] != "ok"]
    if not errors.empty:
        (args.output_dir / "download_summary.json").write_text(
            json.dumps(
                {"status": "failed", "errors": errors.to_dict(orient="records")},
                ensure_ascii=False,
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        return 1

    qa_rows = []
    for device_id, parts in sorted(frames.items()):
        merged, interval_seconds = _align_device_parts(parts)
        merged["timestamp"] = merged["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        merged.to_csv(device_dir / (device_id + ".csv"), index=False)
        numeric = merged.drop(columns=["timestamp"])
        for field in numeric:
            values = pd.to_numeric(numeric[field], errors="coerce")
            qa_rows.append(
                {
                    "device_id": device_id,
                    "field": field,
                    "rows": len(values),
                    "valid_rows": int(values.notna().sum()),
                    "missing_rate": float(values.isna().mean()),
                    "min": float(values.min()) if values.notna().any() else "",
                    "max": float(values.max()) if values.notna().any() else "",
                    "mean": float(values.mean()) if values.notna().any() else "",
                }
            )
    qa = pd.DataFrame(qa_rows).sort_values(["device_id", "field"])
    qa.to_csv(args.output_dir / "data_quality.csv", index=False)
    summary = {
        "status": "success",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "point_count": len(manifest_frame),
        "device_count": len(frames),
        "output_dir": str(args.output_dir),
        "credentials_file": str(args.credentials),
        "credentials_in_manifest": False,
    }
    (args.output_dir / "download_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
