"""Real FMU execution via FMPy (no passthrough).

If FMPy is not installed, or the FMU/data table is missing, or the simulation
returns non-finite values, a clear error is raised -- the framework never
fabricates or silently degrades simulated outputs.

Two driving modes are supported through a single entry point ``run_device_fmu``:

- **data-table driven** (chiller EIR/EEIR, cooling-tower Merkel/YorkCalc, heat
  exchanger Constant/Plate): the measured operating points live in an external
  Dymola table; the machine-specific table path is injected by overriding one or
  more table parameters (e.g. ``VSD2.fileName``, ``Tout1.fileName``,
  ``tableFileName``, ``table_path``). Both measured (``*_m``) and simulated
  (``*_s``) signals come out of the FMU.
- **input driven** (pump): measured inputs (e.g. ``m_flow_in``, ``y_in``) are
  injected as an FMPy input time series.

The thin ``run_fmu`` wrapper is kept for backward compatibility.
"""
from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd


@contextmanager
def extracted_fmu(fmu_path: Union[str, Path]):
    """Extract an FMU once and yield the directory for reuse across many runs.

    ``simulate_fmu`` accepts an already-extracted directory as its filename, so
    passing this directory to repeated ``run_device_fmu`` calls (e.g. library
    screening / FMU-in-the-loop fitting / grid search) skips the per-call unzip,
    which dominates the wall time of those loops. The directory is removed on
    exit.
    """
    import shutil

    from fmpy import extract

    fmu_path = Path(fmu_path)
    if not fmu_path.exists():
        raise FileNotFoundError(f"FMU not found: {fmu_path}")
    unzipdir = extract(str(fmu_path.resolve()))
    try:
        yield Path(unzipdir)
    finally:
        shutil.rmtree(unzipdir, ignore_errors=True)


@contextmanager
def _isolated_cwd():
    """Run inside a throwaway directory.

    Dymola-exported FMUs write solver artifacts (``dsfinal.txt``, ``dsin.txt``,
    ``dymosim`` ...) into the current working directory. Without isolation a
    simulation launched from the repo would overwrite tracked files / pollute the
    user's cwd. All FMU/table paths are resolved to absolute beforehand, so the
    chdir is safe.
    """
    previous = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="autofmu_fmu_") as tmp:
        try:
            os.chdir(tmp)
            yield
        finally:
            os.chdir(previous)


def _quiet_logger(component, instanceName, status, category, message):
    # Only surface genuine errors/fatals (status >= 3); drop the benign
    # scalar-fluid-port sensor warnings Buildings FMUs emit at init.
    if status is not None and int(status) >= 3:
        print(f"[FMU {instanceName} status={status}] {message}")


def _require_fmpy():
    try:
        from fmpy import simulate_fmu  # noqa: F401

        return simulate_fmu
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "fmpy is required for FMU execution. Install with: pip install 'autofmu[fmu]'"
        ) from exc


def _posix(path: Union[str, Path]) -> str:
    return str(Path(path).resolve()).replace("\\", "/")


def _build_input_array(
    inputs: pd.DataFrame,
    input_columns: Sequence[str],
    time_column: str,
) -> np.ndarray:
    """Build an FMPy structured input array (time + one field per input port)."""
    if time_column not in inputs.columns:
        raise KeyError(
            f"input frame is missing time column '{time_column}'; columns={list(inputs.columns)}"
        )
    missing = [c for c in input_columns if c not in inputs.columns]
    if missing:
        raise KeyError(f"input frame is missing input column(s): {missing}")
    dtype = [("time", np.float64)] + [(c, np.float64) for c in input_columns]
    time = pd.to_numeric(inputs[time_column], errors="coerce").to_numpy(dtype=float)
    cols = [pd.to_numeric(inputs[c], errors="coerce").to_numpy(dtype=float) for c in input_columns]
    return np.array(list(zip(time, *cols)), dtype=dtype)


def run_device_fmu(
    fmu_path: Union[str, Path],
    *,
    start_values: Optional[Mapping[str, object]] = None,
    table_overrides: Optional[Mapping[str, Union[str, Path]]] = None,
    inputs: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    input_columns: Optional[Sequence[str]] = None,
    input_time_column: str = "time_s",
    output: Optional[Sequence[str]] = None,
    start_time: float = 0.0,
    stop_time: Optional[float] = None,
    output_interval: Optional[float] = None,
    fmi_type: Optional[str] = None,
    output_scale: Optional[Mapping[str, float]] = None,
    require_finite: bool = True,
    finite_columns: Optional[Sequence[str]] = None,
    max_input_rows_per_chunk: Optional[int] = None,
) -> pd.DataFrame:
    """Simulate one device FMU and return its output time series as a DataFrame.

    Parameters
    ----------
    fmu_path:
        Path to the exported FMU (external, read-only).
    start_values:
        FMU parameter overrides (fitted curve coefficients / UA / nominals).
    table_overrides:
        Mapping ``{parameter_name: table_path}`` for FMUs whose measured points
        live in an external Dymola table. Each path must exist or a
        ``FileNotFoundError`` is raised. The path is normalised to posix form
        and merged into ``start_values``.
    inputs / input_columns / input_time_column:
        For input-driven FMUs. ``inputs`` is a DataFrame (with a time column and
        one column per input port named in ``input_columns``) or a pre-built
        FMPy structured array.
    output:
        Output variable names to return.
    output_scale:
        Optional ``{column: factor}`` post-scaling applied to returned columns
        (e.g. one-cell FMU output * effective component count). A hook only --
        no physics is recomputed here.
    require_finite / finite_columns:
        When ``require_finite`` (default), the returned ``finite_columns`` (or
        all requested ``output`` columns) must be finite; otherwise a
        ``ValueError`` is raised instead of returning NaN/inf.
    max_input_rows_per_chunk:
        Optional chunked simulation for input-driven, quasi-static FMUs: the
        input series is split into consecutive row-blocks simulated independently
        and concatenated. Valid only when output is a static map of the inputs
        (e.g. empirical pump power). Not supported for data-table FMUs.
    """
    simulate_fmu = _require_fmpy()

    fmu_path = Path(fmu_path)
    if not fmu_path.exists():
        raise FileNotFoundError(f"FMU not found: {fmu_path}")

    merged: dict = dict(start_values or {})
    for param, raw in (table_overrides or {}).items():
        table_path = Path(raw)
        if not table_path.exists():
            raise FileNotFoundError(
                f"FMU data table not found for parameter '{param}': {table_path}"
            )
        merged[param] = _posix(table_path)

    input_array: Optional[np.ndarray] = None
    if inputs is not None:
        if isinstance(inputs, pd.DataFrame):
            if not input_columns:
                raise ValueError("input_columns is required when inputs is a DataFrame")
            input_array = _build_input_array(inputs, input_columns, input_time_column)
        else:
            input_array = inputs

    if max_input_rows_per_chunk is not None:
        if input_array is None:
            raise NotImplementedError(
                "chunked simulation is only supported for input-driven FMUs"
            )
        return _run_chunked(
            simulate_fmu,
            fmu_path,
            merged,
            input_array,
            output,
            output_interval,
            fmi_type,
            output_scale,
            require_finite,
            finite_columns,
            max_input_rows_per_chunk,
        )

    frame = _simulate_once(
        simulate_fmu,
        fmu_path,
        merged,
        input_array,
        output,
        start_time,
        stop_time,
        output_interval,
        fmi_type,
    )
    frame = _apply_scale(frame, output_scale)
    _check_finite(frame, require_finite, finite_columns, output)
    return frame


def _simulate_once(
    simulate_fmu,
    fmu_path: Path,
    start_values: Mapping[str, object],
    input_array: Optional[np.ndarray],
    output: Optional[Sequence[str]],
    start_time: float,
    stop_time: Optional[float],
    output_interval: Optional[float],
    fmi_type: Optional[str],
) -> pd.DataFrame:
    kwargs: dict = {
        "filename": str(Path(fmu_path).resolve()),
        "start_time": start_time,
        "validate": False,
        "logger": _quiet_logger,
    }
    if start_values:
        kwargs["start_values"] = dict(start_values)
    if input_array is not None:
        kwargs["input"] = input_array
    if output:
        kwargs["output"] = list(output)
    if stop_time is not None:
        kwargs["stop_time"] = stop_time
    if output_interval is not None:
        kwargs["output_interval"] = output_interval
    if fmi_type is not None:
        kwargs["fmi_type"] = fmi_type

    with _isolated_cwd():
        result = simulate_fmu(**kwargs)
    return pd.DataFrame({name: np.asarray(result[name]) for name in result.dtype.names})


def _run_chunked(
    simulate_fmu,
    fmu_path: Path,
    start_values: Mapping[str, object],
    input_array: np.ndarray,
    output: Optional[Sequence[str]],
    output_interval: Optional[float],
    fmi_type: Optional[str],
    output_scale: Optional[Mapping[str, float]],
    require_finite: bool,
    finite_columns: Optional[Sequence[str]],
    chunk_rows: int,
) -> pd.DataFrame:
    if chunk_rows < 1:
        raise ValueError("max_input_rows_per_chunk must be >= 1")
    frames = []
    n = len(input_array)
    for begin in range(0, n, chunk_rows):
        block = input_array[begin : begin + chunk_rows]
        if len(block) < 2:
            # FMPy needs >= 2 input samples to define the time grid; merge the
            # trailing singleton into the previous block instead of dropping it.
            if frames:
                block = input_array[begin - 1 : begin + chunk_rows]
            else:
                continue
        frame = _simulate_once(
            simulate_fmu,
            fmu_path,
            start_values,
            block,
            output,
            float(block["time"][0]),
            float(block["time"][-1]),
            output_interval,
            fmi_type,
        )
        frames.append(frame)
    if not frames:
        raise ValueError("chunked simulation produced no output rows")
    combined = pd.concat(frames, ignore_index=True)
    combined = _apply_scale(combined, output_scale)
    _check_finite(combined, require_finite, finite_columns, output)
    return combined


def _apply_scale(frame: pd.DataFrame, output_scale: Optional[Mapping[str, float]]) -> pd.DataFrame:
    for col, factor in (output_scale or {}).items():
        if col in frame.columns:
            frame[col] = frame[col] * float(factor)
    return frame


def _check_finite(
    frame: pd.DataFrame,
    require_finite: bool,
    finite_columns: Optional[Sequence[str]],
    output: Optional[Sequence[str]],
) -> None:
    if not require_finite:
        return
    cols = list(finite_columns) if finite_columns else (list(output) if output else list(frame.columns))
    for col in cols:
        if col not in frame.columns:
            continue
        series = pd.to_numeric(frame[col], errors="coerce")
        bad = int((~np.isfinite(series.to_numpy(dtype=float))).sum())
        if bad:
            raise ValueError(
                f"FMU output column '{col}' has {bad}/{len(series)} non-finite value(s); "
                "refusing to return a degraded result"
            )


def run_fmu(
    fmu_path: Path,
    start_values: Optional[Mapping[str, object]] = None,
    output: Optional[Sequence[str]] = None,
    start_time: float = 0.0,
    stop_time: Optional[float] = None,
    output_interval: Optional[float] = None,
) -> pd.DataFrame:
    """Backward-compatible thin wrapper around :func:`run_device_fmu`.

    Keeps the original positional signature used by ``pipeline.fmu_run`` and the
    GS regression test. Finite-checking is disabled here to preserve the exact
    historical return behaviour; new callers should use ``run_device_fmu``.
    """
    return run_device_fmu(
        fmu_path,
        start_values=start_values,
        output=output,
        start_time=start_time,
        stop_time=stop_time,
        output_interval=output_interval,
        require_finite=False,
    )
