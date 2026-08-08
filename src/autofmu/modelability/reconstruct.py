"""Layer-2 reconstruction of un-measured quantities.

Two complementary strategies for a device whose target sensor is dead/missing:

1. ``chw_energy_balance`` / ``cw_energy_balance`` -- reconstruct a heat-transfer
   rate (cooling load / heat rejection) from a device's OWN flow + temperature
   meters:  Q = rho * cp * (flow_m3_h / 3600) * dT.  This is measurement-based.

2. ``apportion_total`` -- split a system/main-pipe TOTAL series across the
   devices running at each timestamp, by a share key (e.g. each device's
   reconstructed load).  This is the literal "main-meter apportionment" and is
   only usable when such a total meter exists.

3. ``cop_estimate`` -- derive an electrical power ESTIMATE from a reconstructed
   load and an assumed COP.  This is an explicit physics assumption, NOT a
   measurement, and is always written to a separate ``*_recon`` column and
   flagged, so it can never be mistaken for measured power during calibration.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd

RHO_WATER_KG_M3 = 1000.0
CP_WATER_J_KG_K = 4186.0


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce")


def energy_balance_load(
    frame: pd.DataFrame,
    flow_col: str,
    t_hot_col: str,
    t_cold_col: str,
) -> pd.Series:
    """Q [W] = rho*cp*(flow/3600)*(T_hot - T_cold). Negative/zero -> NaN."""
    flow = _num(frame, flow_col)
    dT = _num(frame, t_hot_col) - _num(frame, t_cold_col)
    q = RHO_WATER_KG_M3 * CP_WATER_J_KG_K * (flow / 3600.0) * dT
    q = q.where((dT > 0) & (flow > 0))
    return q.replace([np.inf, -np.inf], np.nan)


def power_from_cop(load_w: pd.Series, cop: float) -> pd.Series:
    if cop <= 0:
        raise ValueError("cop must be positive")
    return (load_w / float(cop)).replace([np.inf, -np.inf], np.nan)


def power_energy_balance(condenser_w: pd.Series, evaporator_w: pd.Series) -> pd.Series:
    """Chiller electrical power by first-law balance: P = Q_cond - Q_evap.

    Measurement-based (uses two independently metered heat flows), so it is a
    real power proxy rather than an assumption -- BUT it is the small difference
    of two large heat rates, so it is only trustworthy when both dT are well
    above sensor noise. Caller must verify (e.g. implied COP band)."""
    p = (condenser_w - evaporator_w)
    return p.where((condenser_w > 0) & (evaporator_w > 0)).replace([np.inf, -np.inf], np.nan)


def apportion_total(
    total: pd.Series,
    shares: Mapping[str, pd.Series],
) -> Dict[str, pd.Series]:
    """Split a total series across devices proportional to per-device shares.

    ``shares`` is {device_id -> share series} (e.g. reconstructed load). At each
    timestamp each device gets total * share_i / sum(shares). Devices with zero
    share at a timestamp get 0. Requires aligned indices.
    """
    share_frame = pd.DataFrame(shares)
    share_frame = share_frame.where(share_frame > 0, 0.0)
    denom = share_frame.sum(axis=1).replace(0.0, np.nan)
    out: Dict[str, pd.Series] = {}
    for dev in share_frame.columns:
        out[dev] = total * share_frame[dev] / denom
    return out


def total_source_flow(source_frames: Mapping[str, pd.DataFrame], column: str) -> pd.Series:
    """Sum a flow column across device frames, aligned on timestamp.

    Used for loop-flow attribution: during a target device's solo-run window the
    whole loop flow (= sum of the chillers' metered evaporator/condenser flow)
    passes through that single pump / tower."""
    total = None
    for frame in source_frames.values():
        if column not in frame or "timestamp" not in frame:
            continue
        s = pd.Series(
            pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float),
            index=pd.to_datetime(frame["timestamp"], errors="coerce"),
        )
        s = s[~s.index.isna()].fillna(0.0)
        total = s if total is None else total.add(s, fill_value=0.0)
    return total if total is not None else pd.Series(dtype=float)


def apply_reconstructions(
    frame: pd.DataFrame,
    specs: Mapping[str, dict],
) -> Tuple[pd.DataFrame, List[str]]:
    """Apply per-device reconstruction specs to a single canonical frame.

    Returns an augmented copy and the list of reconstructed/flagged columns.
    Reconstructed power is written to ``<target>_recon`` (never overwriting a
    measured column), so calibration can still require a real measured target.
    """
    out = frame.copy()
    flags: List[str] = []
    for output_col, spec in (specs or {}).items():
        method = spec.get("method")
        if method in ("chw_energy_balance", "cw_energy_balance", "energy_balance", "heat_rate"):
            q = energy_balance_load(out, spec["flow"], spec["t_hot"], spec["t_cold"])
            out[output_col] = q
            n = int(q.notna().sum())
            flags.append("%s=%s(flow=%s,n=%d)" % (output_col, method, spec["flow"], n))
        elif method == "active_count":
            # number of parallel sub-components running (fans/cells/pumps): per
            # timestamp count of signal columns above a threshold. Replaces the
            # hard-coded x2 two-fan scale with a logical, dataset-agnostic count.
            signals = spec["signals"]
            thr = float(spec.get("threshold", 5.0))
            # One drive does not always mean one fan. Site A towers CT_06 and
            # CT_07 expose a single VSD that commands both cells, where CT_01 to
            # CT_05 expose an A and a B channel; counting drives would halve the
            # effective fan count on exactly those two towers. `per_signal` says
            # how many parallel sub-components each signal stands for.
            per_signal = float(spec.get("per_signal", 1.0))
            cnt = np.zeros(len(out)); present = 0
            for s in signals:
                if s in out:
                    cnt += per_signal * (
                        pd.to_numeric(out[s], errors="coerce").fillna(0.0).to_numpy() > thr
                    ).astype(float)
                    present += 1
            if present == 0:
                # no mapped signal: the count is unknown, not zero
                out[output_col] = np.nan
                flags.append("%s=active_count(UNAVAILABLE: none of %s mapped)"
                             % (output_col, ",".join(signals)))
                continue
            out[output_col] = cnt
            flags.append("%s=active_count(of %d/%d signals > %s, per_signal=%g)"
                         % (output_col, present, len(signals), thr, per_signal))
        elif method == "cop_estimate":
            source = spec["from"]
            if source not in out:
                continue
            recon_col = output_col if output_col.endswith("_recon") else output_col + "_recon"
            out[recon_col] = power_from_cop(_num(out, source), float(spec["cop"]))
            flags.append("%s=cop_estimate(COP=%s,ASSUMPTION)" % (recon_col, spec["cop"]))
        elif method == "energy_balance_power":
            cond = _num(out, spec["condenser"])
            evap = _num(out, spec["evaporator"])
            p = power_energy_balance(cond, evap)
            recon_col = output_col if output_col.endswith("_recon") else output_col + "_recon"
            out[recon_col] = p
            # verify via implied COP = Q_evap / P over rows with positive power
            band = spec.get("verify_cop", [2.0, 10.0])
            pos = p > 0
            implied = (evap / p).where(pos)
            med = float(implied.median()) if bool(pos.any()) else float("nan")
            n_ok = int(pos.sum())
            verdict = "pass" if (n_ok > 0 and band[0] <= med <= band[1]) else "FAIL"
            flags.append(
                "%s=energy_balance_power(impliedCOP_med=%.2f,n=%d,verify=%s)"
                % (recon_col, med, n_ok, verdict)
            )
        else:
            raise ValueError("unknown reconstruction method: %s" % method)
    return out, flags
