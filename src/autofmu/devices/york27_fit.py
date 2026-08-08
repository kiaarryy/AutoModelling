"""Closed-loop identification of the YorkCalc-27 coefficients.

What changed relative to the published fit
------------------------------------------
The published fit (``FMU_Modelica/scripts/fit_site_a_ct_york27.py``) regressed
the measured approach on a design matrix built from the *measured* range.  That
is equation-error identification -- a legitimate method, and at the optimum it
coincides with the closed-loop model.  Two things were wrong with how it was
used:

1. the exported FMU also *ran* open-loop, so the reported errors were one-step
   errors presented as simulation errors; and
2. nothing constrained the identified model to be well posed.  Audited on Site
   A, the published vectors are multi-valued at 98.2% (CT-06), 26.5% (CT-04)
   and 9.8% (CT-01) of operating points -- the closed-loop equation does not
   determine a unique outlet temperature.

This module minimises the **simulation error** of the closed-loop model subject
to a **well-posedness constraint**:

    0 < d TApp / d TRan < 1

Because the York correlation is linear in the coefficients, the constraint is a
linear inequality in ``f`` and can be imposed as a smooth barrier inside an
ordinary least-squares problem.  It is also physically readable: the approach
must grow with the range (more heat needs more driving force) but by less than
the range itself (otherwise the outlet runs away).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from .york27_reference import (
    MBL_DEFAULT_COEFFICIENTS, York27Params, approach_sensitivity, closed_loop,
    liquid_gas_ratio, york_design_matrix_dtran,
)

__all__ = ["ClosedLoopFitConfig", "ClosedLoopFitResult", "fit_closed_loop"]


@dataclass(frozen=True)
class ClosedLoopFitConfig:
    prior_weight: float = 1e-2
    """Tikhonov pull towards the prior, in units of the scaled coefficients."""

    barrier_weight: float = 10.0
    """Weight on the well-posedness violation residuals."""

    sensitivity_min: float = 0.02
    sensitivity_max: float = 0.90
    """Admissible band for d TApp / d TRan.  The upper bound is kept below 1 so
    the identified model has margin rather than sitting on the stability edge."""

    inadmissible_penalty: float = 5.0
    """Residual (K) charged for a sample with no physically admissible root."""

    max_nfev: int = 400


@dataclass
class ClosedLoopFitResult:
    coefficients: np.ndarray
    prior: np.ndarray
    frwat0: float
    success: bool
    cost: float
    n_identification: int
    sensitivity_violation_pct: float
    inadmissible_pct: float
    diagnostics: dict = field(default_factory=dict)


def _scaled(prior: np.ndarray) -> np.ndarray:
    """Per-coefficient scale so the optimiser sees a well-conditioned space.

    The York coefficients span seven orders of magnitude; optimising them raw
    makes the trust region meaningless.
    """
    return np.maximum(np.abs(prior), 1e-6)


def fit_closed_loop(tin_c, twb_c, tout_meas_c, mdot_cell_kgps, y_used,
                    params: York27Params, *, prior=None, frwat0: float = 1.0,
                    config: ClosedLoopFitConfig | None = None) -> ClosedLoopFitResult:
    """Identify ``f[1:27]`` by closed-loop simulation-error minimisation.

    ``frwat0`` rescales the liquid-to-gas ratio onto MBL's axis (see
    :func:`~.york27_reference.solve_frwat0`).  Pass 1.0 to reproduce the
    wrapper's original normalisation.
    """
    cfg = config or ClosedLoopFitConfig()
    prior = MBL_DEFAULT_COEFFICIENTS if prior is None else np.asarray(prior, float)
    scale = _scaled(prior)

    tin = np.asarray(tin_c, float)
    twb = np.asarray(twb_c, float)
    meas = np.asarray(tout_meas_c, float)
    rlg = liquid_gas_ratio(mdot_cell_kgps, y_used, params) * float(frwat0)

    ok = np.isfinite(tin) & np.isfinite(twb) & np.isfinite(meas) & np.isfinite(rlg)
    tin, twb, meas, rlg = tin[ok], twb[ok], meas[ok], rlg[ok]
    n = tin.size
    if n < 60:
        raise ValueError(f"too few identification samples: {n}")

    # sensitivity is evaluated at the measured operating point; d/dTRan of the
    # design matrix is independent of f, so this matrix is constant
    dmat = york_design_matrix_dtran(twb, tin - meas, rlg)
    weight = 1.0 / np.sqrt(n)

    def residuals(theta: np.ndarray) -> np.ndarray:
        f = prior + theta * scale
        tout, _, _, admissible, _ = closed_loop(
            tin_c=tin, twb_c=twb, mdot_cell_kgps=np.ones(n), y_used=np.ones(n),
            coefficients=f, params=params, precomputed_rlg=rlg)
        err = np.where(admissible, tout - meas, cfg.inadmissible_penalty)
        err = np.nan_to_num(err, nan=cfg.inadmissible_penalty,
                            posinf=cfg.inadmissible_penalty,
                            neginf=-cfg.inadmissible_penalty)

        sens = dmat @ f
        hi = np.maximum(0.0, sens - cfg.sensitivity_max)
        lo = np.maximum(0.0, cfg.sensitivity_min - sens)

        return np.concatenate([
            weight * err,
            cfg.barrier_weight * weight * hi,
            cfg.barrier_weight * weight * lo,
            np.sqrt(cfg.prior_weight) * theta,
        ])

    sol = least_squares(residuals, np.zeros(27), method="trf",
                        max_nfev=cfg.max_nfev, x_scale="jac")
    f_hat = prior + sol.x * scale

    sens = approach_sensitivity(twb, tin - meas, rlg, f_hat)
    _, _, _, admissible, n_roots = closed_loop(
        tin_c=tin, twb_c=twb, mdot_cell_kgps=np.ones(n), y_used=np.ones(n),
        coefficients=f_hat, params=params, precomputed_rlg=rlg)

    return ClosedLoopFitResult(
        coefficients=f_hat, prior=prior, frwat0=float(frwat0),
        success=bool(sol.success), cost=float(sol.cost), n_identification=n,
        sensitivity_violation_pct=100 * float(
            ((sens <= 0.0) | (sens >= 1.0)).mean()),
        inadmissible_pct=100 * float((~admissible).mean()),
        diagnostics={
            "sensitivity_min": float(np.nanmin(sens)),
            "sensitivity_max": float(np.nanmax(sens)),
            "multivalued_pct": 100 * float((n_roots > 1).mean()),
            "nfev": int(sol.nfev),
            "status": int(sol.status),
        },
    )
