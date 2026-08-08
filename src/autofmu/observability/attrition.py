"""Row-attrition accounting for the data-cleaning stage.

Two problems with the published cleaning code, both raised in review.

*The losses were never quantified.*  Cleaning was a single conjunctive mask, so
the only reportable number was the survivor count -- 17.4% of raw records for
Site A cooling towers -- with no way to say which criterion removed what
(fix M-11, reviewer 1 comment 5).  :class:`AttritionLedger` applies criteria one
at a time and keeps the running count, producing a waterfall that can go
straight into the supplement.

*Some criteria were conditioned on the prediction target.*  ``TRan_C > 0.05``
and ``TAppAct_C >= 0`` are both functions of the measured leaving-water
temperature, so rows were being dropped according to the very quantity the
model is scored against -- survivorship bias, and it also silently discarded the
negative-approach samples that are evidence of a sensor or mapping fault
(fix M-10, reviewer 1 comment 1).

The ledger therefore distinguishes two kinds of exclusion:

``gate``
    An operating-state condition on the *inputs*: the fan is off, there is no
    flow, the device is in a start-up transient.  These define the operating
    domain the model is claimed to cover, and are legitimate.

``quarantine``
    A sample that is physically impossible and cannot be fitted -- water leaving
    a cooling tower below the ambient wet bulb, or hotter than it entered.  These
    are still excluded from the fit, but they are counted, attributed and
    reported rather than disappearing.  A device with many of them has an
    instrumentation problem, which is a finding, not a nuisance.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = ["AttritionStep", "AttritionLedger"]


@dataclass(frozen=True)
class AttritionStep:
    order: int
    name: str
    category: str
    kind: str              # "gate" or "quarantine"
    removed: int
    remaining: int


@dataclass
class AttritionLedger:
    """Applies exclusion criteria in sequence and records what each one cost."""

    n_total: int
    device: str = ""
    family: str = ""
    steps: list[AttritionStep] = field(default_factory=list)
    _mask: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]
    quarantined: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if self._mask is None:
            self._mask = np.ones(int(self.n_total), dtype=bool)

    @property
    def mask(self) -> np.ndarray:
        """Rows surviving every criterion applied so far."""
        return self._mask

    @property
    def remaining(self) -> int:
        return int(self._mask.sum())

    def _record(self, name, category, kind, keep) -> np.ndarray:
        keep = np.asarray(keep, dtype=bool)
        if keep.shape != self._mask.shape:
            raise ValueError(
                f"criterion '{name}' has shape {keep.shape}, expected {self._mask.shape}")
        before = self.remaining
        self._mask = self._mask & keep
        self.steps.append(AttritionStep(
            order=len(self.steps) + 1, name=name, category=category, kind=kind,
            removed=before - self.remaining, remaining=self.remaining))
        return self._mask

    def gate(self, name: str, keep, category: str = "operating_state") -> np.ndarray:
        """Exclude on an input-side condition.  ``keep`` is True for rows to retain."""
        return self._record(name, category, "gate", keep)

    def quarantine(self, name: str, impossible,
                   category: str = "physically_impossible") -> np.ndarray:
        """Exclude physically impossible rows, retaining their mask for reporting.

        ``impossible`` is True for the offending rows -- the opposite sense of
        :meth:`gate`, because here the interesting object is what was removed.
        """
        impossible = np.asarray(impossible, dtype=bool)
        # only count rows still alive at this point, so the waterfall adds up
        offending = impossible & self._mask
        self.quarantined[name] = offending
        return self._record(name, category, "quarantine", ~impossible)

    def quarantine_summary(self) -> dict[str, int]:
        return {k: int(v.sum()) for k, v in self.quarantined.items()}

    def to_frame(self) -> pd.DataFrame:
        rows = [dict(device=self.device, family=self.family, order=0,
                     name="raw records", category="source", kind="",
                     removed=0, remaining=int(self.n_total),
                     retained_pct=100.0)]
        for s in self.steps:
            rows.append(dict(
                device=self.device, family=self.family, order=s.order,
                name=s.name, category=s.category, kind=s.kind,
                removed=s.removed, remaining=s.remaining,
                retained_pct=100.0 * s.remaining / self.n_total if self.n_total else 0.0))
        return pd.DataFrame(rows)

    def summary(self) -> dict:
        by_kind: dict[str, int] = {}
        for s in self.steps:
            by_kind[s.kind] = by_kind.get(s.kind, 0) + s.removed
        return {
            "device": self.device, "family": self.family,
            "n_total": int(self.n_total), "n_retained": self.remaining,
            "retained_pct": (100.0 * self.remaining / self.n_total
                             if self.n_total else 0.0),
            "removed_by_gate": by_kind.get("gate", 0),
            "removed_by_quarantine": by_kind.get("quarantine", 0),
            **{f"quarantine_{k}": v for k, v in self.quarantine_summary().items()},
        }
