from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "outputs" / "runs"
REPORT_DIR = ROOT / "outputs" / "reports"
REPORT_DATE = date(2026, 7, 1)


DETAIL_COLUMNS = [
    "device_id",
    "equipment_type",
    "level",
    "status",
    "model",
    "execution_engine",
    "N",
    "coverage_on_pct",
    "metric_basis",
    "P_CVRMSE_pct",
    "P_NMBE_pct",
    "P_MAPE_pct",
    "P_R2",
    "Q_CVRMSE_pct",
    "Q_NMBE_pct",
    "COP_CVRMSE_pct",
    "COP_NMBE_pct",
    "T_out_CVRMSE_pct",
    "T_out_NMBE_pct",
    "missing_metrics",
    "reason",
]


@dataclass(frozen=True)
class RunSet:
    project: str
    runs: dict[str, Path]
    power_sidecar_run: Path | None = None


PROJECTS = [
    RunSet(
        project="HKUST",
        runs={
            "chiller": RUN_ROOT / "hkust-chiller-fmu-retry-20260630",
            "cooling_tower": RUN_ROOT / "hkust-cooling-tower-fmu-wetbulb-20260630-v2",
            "pump": RUN_ROOT / "hkust-20260630-empirical",
        },
        power_sidecar_run=RUN_ROOT / "hkust-20260630-empirical",
    ),
    RunSet(
        project="Tencent",
        runs={
            "all": RUN_ROOT / "tencent_latest_20260701",
        },
    ),
]


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def get(row: dict[str, Any], key: str, default: Any = np.nan) -> Any:
    value = row.get(key, default)
    return default if is_missing(value) else value


def fmt(value: Any, digits: int = 2) -> str:
    if is_missing(value):
        return ""
    if isinstance(value, str):
        return value
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def fmt_int(value: Any) -> str:
    if is_missing(value):
        return ""
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value)


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"

    def cell(value: Any) -> str:
        if is_missing(value):
            return ""
        return str(value).replace("|", "\\|")

    columns = list(frame.columns)
    header = "| " + " | ".join(cell(col) for col in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(cell(row[col]) for col in columns) + " |"
        for _, row in frame.iterrows()
    ]
    return "\n".join([header, sep, *rows])


def normalize_device_metrics(
    row: dict[str, Any] | pd.Series,
    power_row: dict[str, Any] | pd.Series | None = None,
) -> dict[str, str]:
    src = row.to_dict() if isinstance(row, pd.Series) else dict(row)
    psrc = power_row.to_dict() if isinstance(power_row, pd.Series) else power_row
    equipment = str(get(src, "equipment_type", "")).strip()
    missing: list[str] = []
    out = {
        "device_id": str(get(src, "device_id", "")),
        "equipment_type": equipment,
        "level": str(get(src, "level", "")),
        "status": str(get(src, "status", "")),
        "model": str(get(src, "candidate", get(src, "selected_candidate", ""))),
        "execution_engine": str(get(src, "execution_engine", "")),
        "N": fmt_int(get(src, "N")),
        "coverage_on_pct": fmt(get(src, "coverage_of_on_pct")),
        "metric_basis": "",
        "P_CVRMSE_pct": "",
        "P_NMBE_pct": "",
        "P_MAPE_pct": "",
        "P_R2": "",
        "Q_CVRMSE_pct": "",
        "Q_NMBE_pct": "",
        "COP_CVRMSE_pct": "",
        "COP_NMBE_pct": "",
        "T_out_CVRMSE_pct": "",
        "T_out_NMBE_pct": "",
        "missing_metrics": "",
        "reason": str(get(src, "reason", "")),
    }

    if equipment == "chiller":
        out["metric_basis"] = "power_W + cooling_load_W + COP"
        out["P_CVRMSE_pct"] = fmt(get(src, "CVRMSE_pct"))
        out["P_NMBE_pct"] = fmt(get(src, "NMBE_pct"))
        out["Q_CVRMSE_pct"] = fmt(get(src, "Q_CVRMSE_pct"))
        out["Q_NMBE_pct"] = fmt(get(src, "Q_NMBE_pct"))
        out["COP_CVRMSE_pct"] = fmt(get(src, "COP_CVRMSE_pct"))
        out["COP_NMBE_pct"] = fmt(get(src, "COP_NMBE_pct"))
        for col, label in [
            ("P_CVRMSE_pct", "P"),
            ("Q_CVRMSE_pct", "Q"),
            ("COP_CVRMSE_pct", "COP"),
        ]:
            if not out[col] and out["status"] == "ok":
                missing.append(f"{label} validation artifact missing")
    elif equipment == "cooling_tower":
        out["metric_basis"] = "heat_rejection_W + outlet_temperature + power_W"
        out["Q_CVRMSE_pct"] = fmt(get(src, "Q_CVRMSE_pct", get(src, "CVRMSE_pct")))
        out["Q_NMBE_pct"] = fmt(get(src, "Q_NMBE_pct", get(src, "NMBE_pct")))
        out["T_out_CVRMSE_pct"] = fmt(get(src, "T_CVRMSE_pct_diagnostic"))
        out["T_out_NMBE_pct"] = fmt(get(src, "T_NMBE_pct_diagnostic"))
        if psrc is not None:
            out["P_CVRMSE_pct"] = fmt(get(psrc, "CVRMSE_pct"))
            out["P_NMBE_pct"] = fmt(get(psrc, "NMBE_pct"))
            out["P_MAPE_pct"] = fmt(get(psrc, "MAPE_pct"))
            out["P_R2"] = fmt(get(psrc, "R2"))
        elif out["status"] == "ok":
            missing.append("P validation artifact missing")
        if not out["Q_CVRMSE_pct"] and out["status"] == "ok":
            missing.append("Q validation artifact missing")
        if not out["T_out_CVRMSE_pct"] and out["status"] == "ok":
            missing.append("T_out validation artifact missing")
    elif equipment == "pump":
        out["metric_basis"] = "power_W"
        out["P_CVRMSE_pct"] = fmt(get(src, "CVRMSE_pct"))
        out["P_NMBE_pct"] = fmt(get(src, "NMBE_pct"))
        out["P_MAPE_pct"] = fmt(get(src, "MAPE_pct"))
        out["P_R2"] = fmt(get(src, "R2"))
        if not out["P_CVRMSE_pct"] and out["status"] == "ok":
            missing.append("P validation artifact missing")
    elif equipment == "heat_exchanger":
        out["metric_basis"] = "leaving_temperature + heat_transfer_W"
        out["T_out_CVRMSE_pct"] = fmt(get(src, "T2_CVRMSE_pct", get(src, "CVRMSE_pct")))
        out["T_out_NMBE_pct"] = fmt(get(src, "NMBE_pct"))
        out["Q_CVRMSE_pct"] = fmt(get(src, "Q_CVRMSE_pct"))
        out["Q_NMBE_pct"] = fmt(get(src, "Q_NMBE_pct"))
        if out["status"] == "ok":
            if not out["T_out_CVRMSE_pct"]:
                missing.append("T_out validation artifact missing")
            if not out["Q_CVRMSE_pct"]:
                missing.append("Q validation artifact missing")
    else:
        out["metric_basis"] = str(get(src, "target", ""))

    out["missing_metrics"] = "; ".join(missing)
    return out


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_run(run: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gate = read_csv_if_exists(run / "attribute" / "modelability_report.csv")
    selected = read_csv_if_exists(run / "calibrate" / "selected_models.csv")
    metrics = read_csv_if_exists(run / "validate" / "full_period_metrics.csv")
    return gate, selected, metrics


def add_gate_and_selection(metrics: pd.DataFrame, gate: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    frame = metrics.copy()
    if frame.empty:
        return frame
    if not gate.empty:
        gate_cols = [
            col
            for col in ["device_id", "equipment_type", "level", "reason", "flags", "valid_full_rows"]
            if col in gate.columns
        ]
        frame = frame.merge(gate[gate_cols].drop_duplicates("device_id"), on="device_id", how="left", suffixes=("", "_gate"))
        if "equipment_type_gate" in frame.columns:
            frame["equipment_type"] = frame["equipment_type"].fillna(frame["equipment_type_gate"])
            frame = frame.drop(columns=["equipment_type_gate"])
    if not selected.empty:
        sel_cols = [
            col
            for col in ["device_id", "selected_candidate", "selection_CVRMSE_pct", "test_CVRMSE_pct"]
            if col in selected.columns
        ]
        frame = frame.merge(selected[sel_cols].drop_duplicates("device_id"), on="device_id", how="left")
    return frame


def gate_only_rows(gate: pd.DataFrame, metrics: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    if gate.empty:
        return pd.DataFrame()
    metric_ids = set(metrics.get("device_id", pd.Series(dtype=str)).dropna().astype(str))
    missing = gate[~gate["device_id"].astype(str).isin(metric_ids)].copy()
    if missing.empty:
        return pd.DataFrame()
    if not selected.empty and "device_id" in selected.columns:
        sel_cols = [
            col
            for col in ["device_id", "selected_candidate", "status", "execution_engine"]
            if col in selected.columns
        ]
        missing = missing.merge(selected[sel_cols].drop_duplicates("device_id"), on="device_id", how="left")
    missing["status"] = missing.get("status", pd.Series(index=missing.index, dtype=object)).fillna(missing["level"])
    missing["candidate"] = missing.get("selected_candidate", "")
    return missing


def load_power_sidecar(run: Path | None) -> dict[str, pd.Series]:
    if run is None:
        return {}
    _, _, metrics = load_run(run)
    if metrics.empty:
        return {}
    frame = metrics[(metrics["equipment_type"].eq("pump")) & (metrics["target"].eq("power_W"))].copy()
    return {str(row["device_id"]): row for _, row in frame.iterrows()}


def build_project_detail(spec: RunSet) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    power_sidecar = load_power_sidecar(spec.power_sidecar_run)

    for equipment, run in spec.runs.items():
        gate, selected, metrics = load_run(run)
        metrics = add_gate_and_selection(metrics, gate, selected)
        extra = gate_only_rows(gate, metrics, selected)
        if not extra.empty:
            metrics = pd.concat([metrics, extra], ignore_index=True, sort=False)

        if equipment != "all" and not metrics.empty and "equipment_type" in metrics.columns:
            metrics = metrics[metrics["equipment_type"].eq(equipment)].copy()

        if spec.project == "HKUST" and equipment == "pump":
            metrics = metrics[~metrics["device_id"].astype(str).str.startswith("CT")].copy()

        for _, row in metrics.iterrows():
            device_id = str(row.get("device_id", ""))
            sidecar = power_sidecar.get(device_id) if row.get("equipment_type") == "cooling_tower" else None
            rows.append(normalize_device_metrics(row, power_row=sidecar))

    detail = pd.DataFrame(rows, columns=DETAIL_COLUMNS)
    if detail.empty:
        return detail
    order = {"chiller": 0, "cooling_tower": 1, "pump": 2, "heat_exchanger": 3}
    detail["_order"] = detail["equipment_type"].map(order).fillna(99)
    detail = detail.sort_values(["_order", "device_id"]).drop(columns=["_order"]).reset_index(drop=True)
    return detail


def numeric_series(frame: pd.DataFrame, col: str) -> pd.Series:
    if frame.empty or col not in frame.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[col].replace("", np.nan), errors="coerce").dropna()


def metric_for_equipment(equipment: str) -> str:
    if equipment == "chiller":
        return "P_CVRMSE_pct"
    if equipment == "cooling_tower":
        return "Q_CVRMSE_pct"
    if equipment == "pump":
        return "P_CVRMSE_pct"
    if equipment == "heat_exchanger":
        return "T_out_CVRMSE_pct"
    return "P_CVRMSE_pct"


def build_overview(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for equipment, frame in detail.groupby("equipment_type", sort=False):
        metric_col = metric_for_equipment(equipment)
        values = numeric_series(frame[frame["status"].eq("ok")], metric_col)
        rows.append(
            {
                "equipment_type": equipment,
                "devices": len(frame),
                "ok": int(frame["status"].eq("ok").sum()),
                "limited_or_blocked": int((~frame["status"].eq("ok")).sum()),
                "primary_metric": metric_col,
                "median_pct": fmt(values.median() if not values.empty else np.nan),
                "mean_pct": fmt(values.mean() if not values.empty else np.nan),
                "missing_metric_rows": int(frame["missing_metrics"].ne("").sum()),
            }
        )
    return pd.DataFrame(rows)


def compact_detail(detail: pd.DataFrame, equipment: str) -> pd.DataFrame:
    frame = detail[detail["equipment_type"].eq(equipment)].copy()
    if frame.empty:
        return frame
    common = ["device_id", "level", "status", "model", "N", "coverage_on_pct"]
    if equipment == "chiller":
        cols = common + [
            "P_CVRMSE_pct",
            "P_NMBE_pct",
            "Q_CVRMSE_pct",
            "COP_CVRMSE_pct",
            "COP_NMBE_pct",
            "missing_metrics",
        ]
    elif equipment == "cooling_tower":
        cols = common + [
            "Q_CVRMSE_pct",
            "Q_NMBE_pct",
            "T_out_CVRMSE_pct",
            "T_out_NMBE_pct",
            "P_CVRMSE_pct",
            "P_MAPE_pct",
            "P_R2",
            "missing_metrics",
        ]
    elif equipment == "pump":
        cols = common + ["P_CVRMSE_pct", "P_NMBE_pct", "P_MAPE_pct", "P_R2", "missing_metrics", "reason"]
    elif equipment == "heat_exchanger":
        cols = common + ["T_out_CVRMSE_pct", "T_out_NMBE_pct", "Q_CVRMSE_pct", "Q_NMBE_pct", "missing_metrics", "reason"]
    else:
        cols = DETAIL_COLUMNS
    return frame[[col for col in cols if col in frame.columns]]


def run_sources(spec: RunSet) -> pd.DataFrame:
    rows = []
    for label, path in spec.runs.items():
        rows.append({"source": label, "run_id": path.name, "path": str(path)})
    if spec.power_sidecar_run is not None:
        rows.append(
            {
                "source": "cooling_tower_power_sidecar",
                "run_id": spec.power_sidecar_run.name,
                "path": str(spec.power_sidecar_run),
            }
        )
    return pd.DataFrame(rows)


def write_rules_report() -> Path:
    lines = [
        "# Auto Modelling \u7edf\u4e00\u7ed3\u679c\u62a5\u544a\u89c4\u5219",
        "",
        f"\u751f\u6210\u65e5\u671f: {REPORT_DATE.isoformat()}",
        "",
        "## \u9002\u7528\u8303\u56f4",
        "",
        "These rules standardize Site A, HKUST, and Tencent device-level auto modelling result tables. Reports use validation artifacts only. Missing metrics are shown explicitly and are never treated as passing results.",
        "",
        "## \u6307\u6807\u89c4\u5219",
        "",
        markdown_table(
            pd.DataFrame(
                [
                    {
                        "equipment_type": "chiller",
                        "primary_accuracy": "P_CVRMSE_pct / P_NMBE_pct",
                        "required_secondary": "Q_CVRMSE_pct, COP_CVRMSE_pct, COP_NMBE_pct",
                        "basis": "power_W + cooling_load_W + COP",
                    },
                    {
                        "equipment_type": "cooling_tower",
                        "primary_accuracy": "Q_CVRMSE_pct / Q_NMBE_pct",
                        "required_secondary": "T_out_CVRMSE_pct, T_out_NMBE_pct, P_CVRMSE_pct when power validation exists",
                        "basis": "heat_rejection_W + outlet_temperature + power_W",
                    },
                    {
                        "equipment_type": "pump",
                        "primary_accuracy": "P_CVRMSE_pct / P_NMBE_pct",
                        "required_secondary": "P_MAPE_pct and P_R2 when empirical validation emits them",
                        "basis": "power_W",
                    },
                    {
                        "equipment_type": "heat_exchanger",
                        "primary_accuracy": "T_out_CVRMSE_pct / T_out_NMBE_pct",
                        "required_secondary": "Q_CVRMSE_pct / Q_NMBE_pct when heat-transfer validation exists",
                        "basis": "leaving_temperature + heat_transfer_W",
                    },
                ]
            )
        ),
        "",
        "## \u5c55\u793a\u89c4\u5219",
        "",
        "- Every device row includes level, status, selected model, validation N, and on-period coverage.",
        "- Chiller rows report power P error, heat-transfer/cooling-load Q error, and COP error.",
        "- Cooling tower rows report outlet-temperature T_out error and heat-rejection Q error. Power P is reported when a separate validation artifact exists; otherwise the row records `P validation artifact missing`.",
        "- Pump rows use power P error as the primary device-level accuracy metric.",
        "- Heat exchanger rows use leaving-temperature T_out error as the primary metric and Q error as secondary heat-transfer evidence.",
        "- `nominal_only`, `blocked`, `fmu_unavailable`, and other non-`ok` states stay visible with their reason.",
        "- Controlled variables and reconstructed boundary variables are not promoted into accuracy metrics unless validation artifacts contain measured and simulated target pairs.",
        "",
    ]
    path = REPORT_DIR / f"UNIFIED_AUTOMODELLING_REPORTING_RULES_{REPORT_DATE:%Y%m%d}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_project_report(spec: RunSet, detail: pd.DataFrame) -> Path:
    overview = build_overview(detail)
    lines = [
        f"# {spec.project} Auto Modelling \u7edf\u4e00\u7ed3\u679c\u62a5\u544a",
        "",
        f"\u751f\u6210\u65e5\u671f: {REPORT_DATE.isoformat()}",
        "",
        "## \u7edf\u8ba1\u53e3\u5f84",
        "",
        "This report follows the Site A-compatible device-level presentation rule: chiller = P/Q/COP, cooling tower = Q/T_out/P, pump = P, heat exchanger = T_out/Q. Missing validation artifacts are written to `missing_metrics` instead of being treated as passing results.",
        "",
        "## \u4f7f\u7528\u7684 run",
        "",
        markdown_table(run_sources(spec)),
        "",
        "## \u603b\u89c8",
        "",
        markdown_table(overview),
        "",
    ]
    if spec.project == "HKUST":
        lines.extend(
            [
                "## \u7ed3\u679c\u8bf4\u660e",
                "",
                "- HKUST chiller metrics come from the chiller FMU run. Cooling tower Q/T_out metrics come from the thermal FMU run.",
                "- HKUST cooling tower P metrics come from CT1-CT4 in `hkust-20260630-empirical` as a power sidecar validation, so each CT row combines thermal and power errors.",
                "- CT1-CT4 are excluded from the pump detail table to avoid double-counting the cooling tower power sidecar in the pump fleet.",
                "",
            ]
        )
    if spec.project == "Tencent":
        lines.extend(
            [
                "## \u7ed3\u679c\u8bf4\u660e",
                "",
                "- Tencent chiller, cooling tower, and pump rows all come from `tencent_latest_20260701`.",
                "- Tencent cooling tower validation timeseries currently contains only one measured/simulated thermal sequence and no independent P_s/P_m pair, so CT P error is explicitly marked missing.",
                "- Tencent heat exchangers are currently `nominal_only / power_calibration_blocked`; these devices stay visible without fake full-physical errors.",
                "",
            ]
        )
    for equipment in ["chiller", "cooling_tower", "pump", "heat_exchanger"]:
        table = compact_detail(detail, equipment)
        if table.empty:
            continue
        lines.extend(
            [
                f"## {equipment} \u660e\u7ec6",
                "",
                markdown_table(table),
                "",
            ]
        )
    missing = detail[detail["missing_metrics"].ne("")]
    if not missing.empty:
        lines.extend(
            [
                "## \u663e\u5f0f\u7f3a\u5931\u6307\u6807",
                "",
                markdown_table(missing[["device_id", "equipment_type", "status", "missing_metrics", "reason"]]),
                "",
            ]
        )
    path = REPORT_DIR / f"{spec.project.upper()}_UNIFIED_AUTOMODELLING_RESULTS_{REPORT_DATE:%Y%m%d}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    written = [write_rules_report()]
    for spec in PROJECTS:
        detail = build_project_detail(spec)
        written.append(write_project_report(spec, detail))
    for path in written:
        print(path)
        print(f"bytes={path.stat().st_size}")


if __name__ == "__main__":
    main()
