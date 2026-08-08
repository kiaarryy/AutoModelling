"""When is a device actually operating?

Every stage that counts rows -- the capability gate, the chiller table builder,
the empirical calibrator -- has to answer this, and until now they all answered
it the same wrong way: by trusting ``run_signal``.

A status bit is a declaration, not a measurement, and it can be wrong for
months. On the public LBNL chiller-plant archive ``CHL_STA_1`` reads 1 for all
17,518 samples of the year while the machine draws a constant 1.94 kW standby
load on 12,392 of them, with the compressor command at 0.9% and an evaporator
temperature difference of 0.06 K. The gate reported all 17,518 rows as
modellable; worse, the chiller table's load floor is a fraction of the *median*
power over those rows, so the contamination pushed the median down to standby
level and disabled the very filter meant to catch it. CHI1 came out at 105%
test CVRMSE. Its twin CHI2 -- same plant, same year, same code, honest status
bit -- came out at 11.7%.

Two rules keep this honest:

**Corroborate, do not assume.** The bit is intersected with a *liveness*
channel declared by the equipment profile: a channel whose near-zero value
means the machine is not operating at all. The disagreement is always recorded
as a flag, even when it changes nothing, so a stuck sensor is visible in the
modelability report rather than buried in a row count.

**Never use the calibration target as the liveness channel.** It is tempting --
zero power means an idle pump -- but a target floor cannot distinguish "off"
from "lightly loaded", and low-load rows are exactly the ones a part-load model
needs. Tried against the four archives, a 2%-of-p99 floor on the target
discarded 57% of Site A CT_05, whose fan turns throughout. The liveness channel
is a *command* (compressor percentage, drive frequency), which sits at zero
only when the machine is commanded off.

**Convict only on evidence.** A liveness channel that is NaN on a row leaves
that row alone. Absence of evidence is not evidence of standby -- reconstructed
channels are missing wherever attribution failed, and those rows are dropped
later by the finite-row requirements for honest reasons.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

# A liveness channel is a command signal, so "commanded off" is a small
# fraction of its own operating range rather than an absolute value: LBNL
# reports a chiller's idle compressor command as 0.9%, not 0. Referenced to the
# 99th percentile rather than the maximum so one spike cannot raise the floor.
LIVENESS_FLOOR_FRAC = 0.02
# Report the bit/liveness disagreement once it is past sampling noise.
STUCK_REPORT_PCT = 5.0


@dataclass(frozen=True)
class OperatingMask:
    mask: np.ndarray            # rows where the device is operating
    bit: np.ndarray             # rows the status bit claims
    channel: str                # liveness channel used ("" if none available)
    floor: float                # threshold applied to that channel
    stuck_pct: float            # % of bit-on rows the liveness channel denies
    flags: Tuple[str, ...]

    @property
    def rows(self) -> int:
        return int(self.mask.sum())


def _numeric(frame: pd.DataFrame, column: str):
    if column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    return values if np.isfinite(values).any() else None


def operating_mask(frame: pd.DataFrame, profile, run_on: float = 0.5,
                   floor_frac: float = LIVENESS_FLOOR_FRAC,
                   dead_run_policy: str = "fallback") -> OperatingMask:
    """Rows where the device is operating, per the status bit corroborated by
    the profile's liveness channel."""
    n = len(frame)
    flags: List[str] = []

    if dead_run_policy not in {"fallback", "off"}:
        raise ValueError("dead_run_policy must be 'fallback' or 'off'")
    run_signal = getattr(profile, "run_signal", "run_signal")
    values = _numeric(frame, run_signal)
    if values is None:
        flags.append("run_signal_absent")
        bit = np.ones(n, dtype=bool)
    else:
        bit = values > run_on
        if not bit.any():
            # A dead status sensor must not collapse an otherwise modellable
            # device; fall back to every row and say so.
            flags.append("run_signal_dead")
            if dead_run_policy == "off":
                return OperatingMask(
                    bit, bit, "", float("nan"), 0.0, tuple(flags)
                )
            bit = np.ones(n, dtype=bool)

    for channel in getattr(profile, "liveness", ()) or ():
        live = _numeric(frame, channel)
        if live is None:
            continue
        reference = bit & np.isfinite(live) & (live > 0.0)
        if not reference.any():
            continue
        floor = floor_frac * float(np.nanpercentile(live[reference], 99))
        # convict only on evidence: NaN leaves the row alone
        commanded_off = np.isfinite(live) & (live <= floor)
        mask = bit & ~commanded_off
        denied = int((bit & commanded_off).sum())
        stuck = 100.0 * denied / max(1, int(bit.sum()))
        if stuck >= STUCK_REPORT_PCT:
            flags.append(f"run_signal_uncorroborated:{stuck:.0f}pct_by_{channel}")
        if not mask.any():
            # Every row denied. Keep the bit, flag loudly: this is a broken
            # liveness channel, not a device that never ran.
            flags.append(f"liveness_channel_empty:{channel}")
            return OperatingMask(bit, bit, channel, floor, stuck, tuple(flags))
        return OperatingMask(mask, bit, channel, floor, stuck, tuple(flags))

    if getattr(profile, "liveness", ()):
        flags.append("liveness_channel_absent")
    return OperatingMask(bit, bit, "", float("nan"), 0.0, tuple(flags))
