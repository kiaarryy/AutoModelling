"""Detect BMS channels that are not independent observations.

Three failure modes, in increasing order of how badly they mislead:

``dead``
    Constant or all-zero over the whole record.  Easy to spot, and the pipeline
    already gates on it.

``duplicate``
    Two roles resolve to numerically identical series -- usually one physical
    point mapped twice by the adapter.

``synthetic``
    The channel is an exact algebraic function of another channel.  This is the
    dangerous one: the point looks like a healthy, varying measurement, passes
    every completeness and range check, and silently turns model validation
    into a circular exercise.  It was found on the HKUST archive, where 21 of
    25 devices publish ``power = P_rated * speed`` and ``flow = 15 * P_rated *
    speed`` -- one live signal scaled by two nameplate constants.

The test is deliberately blunt.  Fit the candidate closed forms, then ask what
*fraction* of samples the form reproduces to floating-point precision.  Real
instrumentation, even a well-behaved VFD, reproduces another channel that
closely on essentially no samples; a stored multiple does so on all of them.
The two populations sit at opposite ends of the interval, so neither the
tolerance (1e-6 % of range) nor the fraction (0.99) is a tuned parameter.

Three details are load-bearing, each learned from a wrong answer:

* the fit is trimmed once, because a handful of rows where the run status
  disagrees with the values it accompanies will pull an exact slope off by
  enough to make every residual non-zero;
* judgement is on the inlier fraction rather than the worst residual, for the
  same reason, with the disagreeing rows reported as ``n_outlier_rows``;
* targets that sit at one value for almost every active sample are skipped.
  Status flags are the trap: ``run_signal`` varies over the record but is
  constant on the active mask built from it, so every other channel appears to
  "predict" it with a zero slope.

What this is not: a correlation screen.  Highly correlated channels are normal
and expected in a chiller plant.  Only *exact* functional reproduction is
flagged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "EvidenceTier",
    "ChannelReport",
    "DependencyFinding",
    "profile_channel",
    "detect_dependencies",
    "analyse_device",
    "SYNTHETIC_RESIDUAL_PCT",
    "SYNTHETIC_INLIER_FRACTION",
]

# residual as a percentage of signal range, below which a sample is an exact match
SYNTHETIC_RESIDUAL_PCT = 1e-6
# fraction of samples that must match exactly for the pair to be an identity
SYNTHETIC_INLIER_FRACTION = 0.99

# candidate closed forms, in order of increasing flexibility.  The first that
# fits to machine precision is reported, so a proportional channel is not
# described as a cubic.
_FORMS: dict[str, int] = {
    "proportional": 0,   # y = b1*x
    "affine": 1,         # y = b0 + b1*x
    "quadratic": 2,
    "cubic": 3,
}


class EvidenceTier(str, Enum):
    """Grade of evidence behind a variable, best first.

    ``T0``-``T5`` follow docs/REVISION_ENERGY_02_PLAN.md.  ``TX_SYNTHETIC`` sits
    *below* ``T4``: a nameplate-derived value published as a measurement is
    worse than an openly declared design value, because nothing downstream can
    tell that it is not observed.
    """

    T0_DIRECT = "T0_direct_measurement"
    T1_CONSERVATION = "T1_conservation_reconstruction"
    T2_TOPOLOGICAL = "T2_topological_attribution"
    T3_SYSTEM_BOUNDARY = "T3_system_boundary_approximation"
    T4_NAMEPLATE = "T4_nameplate_substitution"
    TX_SYNTHETIC = "TX_synthetic_point_published_as_measurement"
    T5_UNAVAILABLE = "T5_unavailable"


@dataclass
class ChannelReport:
    role: str
    source: str
    n_finite: int
    n_total: int
    n_unique: int
    minimum: float
    maximum: float
    std: float
    dead: bool
    all_zero: bool

    @property
    def coverage_pct(self) -> float:
        return 100.0 * self.n_finite / self.n_total if self.n_total else 0.0


@dataclass
class DependencyFinding:
    target: str
    driver: str
    form: str
    coefficients: tuple[float, ...]
    max_residual_pct_of_range: float
    n: int
    duplicate: bool = False
    inlier_fraction: float = float("nan")
    n_outlier_rows: int = 0

    @property
    def synthetic(self) -> bool:
        """The relation holds to machine precision on essentially every sample.

        Judged on the *fraction* of samples within tolerance rather than on the
        worst one.  The residual distribution of a stored-multiple channel is
        bimodal: a mass at floating-point noise plus a handful of rows where the
        run status disagrees with the values it accompanies (HKUST PP1 has seven
        samples flagged running with zero power while speed is non-zero).  A
        genuine physical relation has an inlier fraction of essentially zero at
        this tolerance, so the two populations are separated by the whole
        interval and the exact threshold is immaterial.  The disagreeing rows are
        reported as ``n_outlier_rows`` -- an observability defect in its own
        right.
        """
        return self.inlier_fraction >= SYNTHETIC_INLIER_FRACTION

    def describe(self) -> str:
        c = self.coefficients
        if self.duplicate:
            return f"{self.target} is a copy of {self.driver}"
        if self.form == "proportional":
            body = f"{c[0]:.6g}*{self.driver}"
        else:
            terms = [f"{c[0]:.6g}"]
            terms += [f"{v:.6g}*{self.driver}^{i}" if i > 1 else f"{v:.6g}*{self.driver}"
                      for i, v in enumerate(c[1:], start=1)]
            body = " + ".join(terms)
        return f"{self.target} = {body}"


@dataclass
class DeviceObservability:
    site: str
    device: str
    equipment_type: str
    channels: list[ChannelReport] = field(default_factory=list)
    dependencies: list[DependencyFinding] = field(default_factory=list)

    @property
    def synthetic_roles(self) -> set[str]:
        return {d.target for d in self.dependencies if d.synthetic}

    @property
    def independent_roles(self) -> list[str]:
        """Roles carrying information no other role already carries.

        Built by union-find over the exact-dependency edges: every group of
        mutually reproducible channels collapses to one representative, and
        dead channels are dropped.  ``len(independent_roles)`` is the honest
        answer to "how many things does this device actually measure".
        """
        dead = {c.role for c in self.channels if c.dead}
        alive = [c.role for c in self.channels if c.role not in dead]
        parent = {r: r for r in alive}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for d in self.dependencies:
            if not (d.synthetic or d.duplicate):
                continue
            if d.target in parent and d.driver in parent:
                ra, rb = find(d.target), find(d.driver)
                if ra != rb:
                    parent[ra] = rb

        seen, out = set(), []
        for role in alive:
            root = find(role)
            if root not in seen:
                seen.add(root)
                out.append(role)
        return out


def profile_channel(role: str, source: str, values: np.ndarray,
                    n_total: int) -> ChannelReport:
    finite = np.isfinite(values)
    vals = values[finite]
    n_unique = int(pd.Series(vals).nunique()) if vals.size else 0
    std = float(np.std(vals)) if vals.size else 0.0
    return ChannelReport(
        role=role, source=source, n_finite=int(finite.sum()), n_total=int(n_total),
        n_unique=n_unique,
        minimum=float(vals.min()) if vals.size else float("nan"),
        maximum=float(vals.max()) if vals.size else float("nan"),
        std=std,
        dead=(n_unique <= 1),
        all_zero=bool(vals.size and np.allclose(vals, 0.0)),
    )


def _design(x: np.ndarray, degree: int, proportional: bool) -> np.ndarray:
    if proportional:
        return x[:, None]
    return np.column_stack([x ** k for k in range(degree + 1)])


def _fit_form(x: np.ndarray, y: np.ndarray, degree: int, proportional: bool,
              trim: float = 0.05):
    """Least squares with one trimmed re-fit.

    A plain fit is not usable here.  When a channel is a stored multiple of
    another, a handful of rows where the run status disagrees with the values
    is enough to pull the slope off by ~0.1% -- which turns *every* residual
    non-zero and hides an otherwise perfect identity.  So: fit, drop the worst
    ``trim`` of samples, fit again.  On a genuinely noisy relation the trimmed
    fit is essentially the same as the plain one, so nothing is gained by
    cheating; on a contaminated identity it recovers the exact coefficient.
    """
    design = _design(x, degree, proportional)
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = np.abs(y - design @ beta)
    if resid.size > 20:
        keep = resid <= np.quantile(resid, 1.0 - trim)
        if keep.sum() > design.shape[1] + 5:
            beta, *_ = np.linalg.lstsq(design[keep], y[keep], rcond=None)
    return beta, design @ beta


def detect_dependencies(series: Mapping[str, np.ndarray], *,
                        active: np.ndarray | None = None,
                        min_samples: int = 200,
                        max_fit_rows: int = 20000,
                        modal_fraction_max: float = 0.95,
                        rng_seed: int = 0) -> list[DependencyFinding]:
    """Find channels that are exact functions of another channel.

    Fitting is done on at most ``max_fit_rows`` samples for speed, but the
    reported residual is always evaluated on every sample -- a form that only
    matches the subsample is not an identity.
    """
    roles = [r for r, v in series.items() if np.isfinite(v).sum() >= min_samples]
    findings: list[DependencyFinding] = []
    rng = np.random.default_rng(rng_seed)

    def degenerate_target(values: np.ndarray) -> bool:
        """A channel that barely varies cannot be a meaningful target.

        Status flags are the trap.  ``run_signal`` is 0/1 over the record, so it
        survives the dead-channel check, but the dependence test runs on the
        *active* mask -- which is built from ``run_signal > 0`` -- where it is
        constant at 1.  Anything then 'predicts' it with a zero slope, and the
        detector reports every other channel as its driver.  Excluding targets
        that sit at one value for almost every active sample removes that whole
        class of spurious findings.
        """
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return True
        counts = pd.Series(finite).value_counts()
        return float(counts.iloc[0]) / finite.size > modal_fraction_max

    for target in roles:
        y_all = series[target]
        base_mask = np.isfinite(y_all) if active is None else (np.isfinite(y_all) & active)
        if degenerate_target(y_all[base_mask]):
            continue
        best: DependencyFinding | None = None
        exact_for_target: list[DependencyFinding] = []
        for driver in roles:
            if driver == target:
                continue
            x_all = series[driver]
            mask = np.isfinite(x_all) & np.isfinite(y_all)
            if active is not None:
                mask &= active
            n = int(mask.sum())
            if n < min_samples:
                continue
            x, y = x_all[mask], y_all[mask]
            span = float(np.ptp(y))
            if span <= 0:
                continue                      # constant target: handled as dead

            if np.array_equal(x, y):
                findings.append(DependencyFinding(
                    target=target, driver=driver, form="identity",
                    coefficients=(1.0,), max_residual_pct_of_range=0.0, n=n,
                    duplicate=True, inlier_fraction=1.0))
                best = None
                break

            if n > max_fit_rows:
                idx = rng.choice(n, max_fit_rows, replace=False)
            else:
                idx = slice(None)

            for form, degree in _FORMS.items():
                try:
                    beta, _ = _fit_form(x[idx], y[idx], degree,
                                        proportional=(form == "proportional"))
                except np.linalg.LinAlgError:
                    continue
                pred = _design(x, degree, form == "proportional") @ beta
                abs_err = np.abs(y - pred)
                resid = 100.0 * float(np.max(abs_err)) / span
                tol = SYNTHETIC_RESIDUAL_PCT * span / 100.0
                outliers = int((abs_err > tol).sum())
                cand = DependencyFinding(
                    target=target, driver=driver, form=form,
                    coefficients=tuple(float(b) for b in beta),
                    max_residual_pct_of_range=resid, n=n,
                    inlier_fraction=1.0 - outliers / n,
                    n_outlier_rows=outliers)
                if best is None or cand.inlier_fraction > best.inlier_fraction:
                    best = cand
                if cand.synthetic:
                    exact_for_target.append(cand)
                    break                     # simplest exact form wins
        # every exact driver is recorded, not just the first: when power, speed
        # and flow are mutually proportional the device has one independent
        # signal, and reporting a single edge would understate that.
        if exact_for_target:
            findings.extend(exact_for_target)
        elif best is not None:
            findings.append(best)
    return findings


def analyse_device(frame: pd.DataFrame, columns: Mapping[str, Mapping],
                   *, site: str, device: str, equipment_type: str,
                   active_roles: Sequence[str] = ("run_signal",),
                   ) -> DeviceObservability:
    """Profile every mapped channel of one device and test for dependence."""
    series: dict[str, np.ndarray] = {}
    reports: list[ChannelReport] = []
    n_total = len(frame)

    for role, spec in (columns or {}).items():
        src = (spec or {}).get("source")
        if not src or src not in frame.columns:
            continue
        values = pd.to_numeric(frame[src], errors="coerce").to_numpy(float)
        scale = float((spec or {}).get("scale", 1.0) or 1.0)
        values = values * scale
        series[role] = values
        reports.append(profile_channel(role, str(src), values, n_total))

    # Restrict the dependence test to running samples: an off pump trivially has
    # power == flow == 0, which would look like a perfect proportionality.
    #
    # "Running" has to mean the drive is actually turning, not merely that a
    # status bit is set. HKUST PP2 carries operation_status = 1 on 6,148 samples
    # where speed_Hz reads zero and power does not -- a status/value
    # contradiction. Judged on the status bit alone those rows drag the inlier
    # fraction to 0.45 and hide the fact that on every genuinely running sample
    # power is an exact multiple of speed. They are a finding in their own
    # right, surfaced as n_outlier_rows, but they must not mask the identity.
    active = None
    for role in active_roles:
        if role in series:
            run = series[role]
            active = np.isfinite(run) & (run > 0)
            break
    if active is None:
        candidates = [v for r, v in series.items() if "power" in r or "flow" in r]
        if candidates:
            active = np.isfinite(candidates[0]) & (candidates[0] > 0)

    drive = next((series[r] for r in ("speed_Hz", "fan1_Hz", "y_used")
                  if r in series), None)
    if drive is not None:
        turning = np.isfinite(drive) & (drive > 0)
        active = turning if active is None else (active & turning)

    deps = detect_dependencies(
        {r: v for r, v in series.items() if not np.allclose(
            v[np.isfinite(v)], 0.0) and pd.Series(v).nunique() > 1},
        active=active)

    return DeviceObservability(site=site, device=device,
                               equipment_type=equipment_type,
                               channels=reports, dependencies=deps)
