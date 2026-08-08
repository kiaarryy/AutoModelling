"""Pure-Python reference implementation of the YorkCalc-27 cooling-tower wrapper.

Why this exists
---------------
``autofmu`` drives compiled FMUs; device physics must come from the FMU (FMU-5).
This module is **not** a production surrogate and must not be wired into the
pipeline as one.  It is a *verification oracle*:

* ``open_loop`` reproduces the shipped ``SiteACTYork27.fmu`` bit-for-bit
  (verified to 7e-9 K on all seven Site A towers), so the FMU can be regression
  tested without a Modelica toolchain.
* ``closed_loop`` implements the MBL-consistent formulation, in which the range
  temperature is the model's *own* range rather than a measured input.

The distinction matters.  Buildings 12.1.0
``Fluid/HeatExchangers/CoolingTowers/YorkCalc.mo`` line 20 defines::

    TRan = T_a - T_b

with ``T_b`` the component's own outlet port temperature, and drives the outlet
enthalpy to ``TAir + TAppAct``.  MBL therefore solves an implicit algebraic loop.
The Site A wrapper opened that loop by reading a measured ``TRan_C`` column that
is computed as ``Tin - Tout_measured`` -- i.e. it fed the prediction target back
in as an input.  ``closed_loop`` restores the loop.

See docs/REVISION_ENERGY_01_AUDIT.md section B.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "York27Params",
    "MBL_DEFAULT_COEFFICIENTS",
    "MBL_VALIDITY",
    "york_design_matrix",
    "york_design_matrix_dtran",
    "york_approach",
    "approach_sensitivity",
    "solve_frwat0",
    "liquid_gas_ratio",
    "open_loop",
    "closed_loop",
]

# Buildings 12.1.0 Fluid/HeatExchangers/CoolingTowers/Correlations/BoundsYorkCalc.mo
MBL_VALIDITY = {
    "TAirInWB_min_C": -34.4, "TAirInWB_max_C": 26.7,
    "TRan_min_K": 1.1, "TRan_max_K": 22.2,
    "TApp_min_K": 1.1, "TApp_max_K": 40.0,
    "FRWat_min": 0.75, "FRWat_max": 1.25,
    "liqGasRat_max": 8.0,
}

# Buildings 12.1.0 Fluid/HeatExchangers/CoolingTowers/Correlations/yorkCalc.mo, c[1..27]
MBL_DEFAULT_COEFFICIENTS = np.array([
    -0.359741205, -0.055053608, 0.0023850432,
    0.173926877, -0.0248473764, 0.00048430224,
    -0.005589849456, 0.0005770079712, -0.00001342427256,
    2.84765801111111, -0.121765149, 0.0014599242,
    1.680428651, -0.0166920786, -0.0007190532,
    -0.025485194448, 0.0000487491696, 0.00002719234152,
    -0.0653766255555556, -0.002278167, 0.0002500254,
    -0.0910565458, 0.00318176316, 0.000038621772,
    -0.0034285382352, 0.00000856589904, -0.000001516821552,
])


@dataclass(frozen=True)
class York27Params:
    """Start values of the SiteACTYork27 wrapper."""

    m_flow_nominal: float
    y_min: float = 0.05
    rlg_min: float = 0.05
    rlg_max: float = 8.0
    tapp_min_c: float = 0.0
    tapp_max_c: float = 40.0
    cp_wat: float = 4180.0


def york_design_matrix(twb_c, tran_c, rlg) -> np.ndarray:
    """27 monomials in MBL's ``c[1..27]`` order.  Shape ``(n, 27)``."""
    t, r, l = np.asarray(twb_c, float), np.asarray(tran_c, float), np.asarray(rlg, float)
    return np.stack([
        np.ones_like(t), t, t * t,
        r, t * r, t * t * r,
        r * r, t * r * r, t * t * r * r,
        l, t * l, t * t * l,
        r * l, t * r * l, t * t * r * l,
        r * r * l, t * r * r * l, t * t * r * r * l,
        l * l, t * l * l, t * t * l * l,
        r * l * l, t * r * l * l, t * t * r * l * l,
        r * r * l * l, t * r * r * l * l, t * t * r * r * l * l,
    ], axis=-1)


def york_design_matrix_dtran(twb_c, tran_c, rlg) -> np.ndarray:
    """d/dTRan of :func:`york_design_matrix`.  Shape ``(n, 27)``.

    The 27 monomials are ``T^r * R^q * L^p`` with ``index = 9p + 3q + r``
    (verified against MBL's written-out sum), so the derivative is
    ``q * T^r * R^(q-1) * L^p``.

    Used for the well-posedness test: the closed-loop fixed point
    ``a = york(Twb, Tin - Twb - a, Rlg)`` is unique when
    ``|d TApp / d TRan| < 1``.  Because ``york`` is linear in the coefficients,
    that condition is a *linear* inequality in ``f`` -- which is what makes it
    usable as an identification constraint rather than only a diagnostic.
    """
    t, r, l = np.asarray(twb_c, float), np.asarray(tran_c, float), np.asarray(rlg, float)
    cols = []
    for p in range(3):
        for q in range(3):
            for k in range(3):
                if q == 0:
                    cols.append(np.zeros_like(t))
                else:
                    cols.append(q * t ** k * r ** (q - 1) * l ** p)
    return np.stack(cols, axis=-1)


def york_approach(twb_c, tran_c, rlg, coefficients) -> np.ndarray:
    return york_design_matrix(twb_c, tran_c, rlg) @ np.asarray(coefficients, float)


def approach_sensitivity(twb_c, tran_c, rlg, coefficients) -> np.ndarray:
    """``d TApp / d TRan``.  Physically this must lie in ``(0, 1)``.

    Positive: a larger range rejects more heat and needs a larger driving
    force, so the approach rises.  Below one: if a 1 K increase in range raised
    the approach by more than 1 K the outlet temperature would run away, and
    the steady state stops being unique.
    """
    return york_design_matrix_dtran(twb_c, tran_c, rlg) @ np.asarray(coefficients, float)


def solve_frwat0(twb_nominal_c, tran_nominal, tapp_nominal, coefficients,
                 lo: float = 1e-3, hi: float = 8.0, steps: int = 200) -> float:
    """Back-solve MBL's ``FRWat0`` from the design point.

    Buildings 12.1.0 ``YorkCalc.mo`` initial equation::

        TApp_nominal = yorkCalc(TRan_nominal, TAirInWB_nominal, FRWat0, FRAir=1)
        mWat_flow_nominal = m_flow_nominal / FRWat0

    so MBL's water-flow ratio is ``(m_flow/m_flow_nominal) * FRWat0``.  The Site
    A wrapper omits ``FRWat0`` entirely, which puts its ``Rlg`` on a different
    axis from the one MBL's default coefficients were derived on.  Returns
    ``nan`` when no root exists in ``[lo, hi]``.
    """
    grid = np.linspace(lo, hi, steps)
    vals = york_approach(np.full_like(grid, float(twb_nominal_c)),
                         np.full_like(grid, float(tran_nominal)),
                         grid, coefficients) - float(tapp_nominal)
    sign_change = np.where(vals[:-1] * vals[1:] < 0)[0]
    if sign_change.size == 0:
        return float("nan")
    i = int(sign_change[0])
    a, b = grid[i], grid[i + 1]
    fa = vals[i]
    for _ in range(80):
        m = 0.5 * (a + b)
        fm = york_approach(np.array([twb_nominal_c]), np.array([tran_nominal]),
                           np.array([m]), coefficients)[0] - tapp_nominal
        if np.sign(fm) == np.sign(fa):
            a, fa = m, fm
        else:
            b = m
    return float(0.5 * (a + b))


def liquid_gas_ratio(mdot_cell_kgps, y_used, params: York27Params) -> np.ndarray:
    """Wrapper's Rlg.

    NOTE: this is *not* MBL's ``liqGasRat``.  MBL normalises the water flow by
    ``mWat_flow_nominal = m_flow_nominal / FRWat0``, where ``FRWat0`` is
    back-solved in the initial equation from the design approach.  The wrapper
    omits ``FRWat0`` (Site A CT-01 back-solve = 1.199, so the wrapper's Rlg is
    ~20% low).  Kept as-is here so ``open_loop`` reproduces the shipped FMU;
    the corrected form is applied by the refitting path.
    """
    fr_wat = np.maximum(0.0, np.asarray(mdot_cell_kgps, float)) / max(1e-6, params.m_flow_nominal)
    fr_air = np.clip(np.asarray(y_used, float), params.y_min, 1.0)
    return np.clip(fr_wat / np.maximum(1e-4, fr_air), params.rlg_min, params.rlg_max)


def open_loop(tin_c, twb_c, tran_c, mdot_cell_kgps, y_used, coefficients,
              params: York27Params):
    """Shipped-FMU behaviour: ``tran_c`` is read from the measured input table.

    Returns ``(tout_s_c, tapp_s_c, rlg)``.
    """
    rlg = liquid_gas_ratio(mdot_cell_kgps, y_used, params)
    twb = np.asarray(twb_c, float)
    tapp = np.clip(york_approach(twb, tran_c, rlg, coefficients),
                   params.tapp_min_c, params.tapp_max_c)
    return twb + tapp, tapp, rlg


def closed_loop(tin_c, twb_c, mdot_cell_kgps, y_used, coefficients,
                params: York27Params, *, bisection_steps: int = 60,
                scan_points: int = 33, precomputed_rlg=None):
    """MBL-consistent behaviour: ``TRan = Tin - TOut_s``, solved per sample.

    Solves the scalar residual ``h(a) = clip(york(Twb, Tin - Twb - a, Rlg)) - a``
    for the approach temperature ``a``.

    Deterministic by construction.  A damped fixed-point iteration was tried
    first and rejected: for coefficient vectors identified open-loop the map is
    not a contraction on several Site A towers, so the answer depended on the
    initial guess (CT-06 landed on 5 K or 37 K depending on warm start).  A
    warm-start-dependent number is not reportable.

    The search interval is the *physically admissible* one::

        0 <= a <= Tin - Twb          (equivalently TRan >= 0)

    which is tighter than the wrapper's ``[TAppMin_C, TAppMax_C]`` clip and
    encodes that the tower cannot cool below the wet bulb nor heat the water.
    Within it the smallest sign change is bracketed and bisected.

    Returns ``(tout_s_c, tapp_s_c, rlg, admissible, n_roots)``:

    * ``admissible`` -- a root was found inside the physical interval.  Samples
      where it is False have no physically valid closed-loop solution for this
      coefficient vector; that is a genuine identifiability failure and must be
      reported, not silently clipped away.
    * ``n_roots`` -- sign changes detected on the scan grid.  Values > 1 mean
      the closed-loop model is multi-valued at that operating point.
    """
    rlg = (liquid_gas_ratio(mdot_cell_kgps, y_used, params)
           if precomputed_rlg is None else np.asarray(precomputed_rlg, float))
    twb = np.asarray(twb_c, float)
    tin = np.asarray(tin_c, float)
    upper = np.maximum(0.0, tin - twb)

    def residual(a):
        tran = tin - twb - a
        return np.clip(york_approach(twb, tran, rlg, coefficients),
                       params.tapp_min_c, params.tapp_max_c) - a

    # scan the admissible interval for sign changes
    grid = np.linspace(0.0, 1.0, scan_points)
    values = np.stack([residual(upper * g) for g in grid])      # (scan, n)
    signs = np.sign(values)
    changes = signs[:-1] * signs[1:] < 0
    n_roots = changes.sum(axis=0)

    first = np.argmax(changes, axis=0)                          # 0 when none
    has_root = changes.any(axis=0)
    lo = upper * grid[first]
    hi = upper * grid[np.minimum(first + 1, scan_points - 1)]

    # exact zeros on the grid
    exact = np.isclose(values, 0.0, atol=1e-12)
    lo = np.where(has_root, lo, 0.0)
    hi = np.where(has_root, hi, upper)

    f_lo = residual(lo)
    for _ in range(bisection_steps):
        mid = 0.5 * (lo + hi)
        f_mid = residual(mid)
        go_left = (np.sign(f_mid) == np.sign(f_lo))
        lo = np.where(go_left, mid, lo)
        f_lo = np.where(go_left, f_mid, f_lo)
        hi = np.where(go_left, hi, mid)
    tapp = 0.5 * (lo + hi)

    admissible = has_root | exact.any(axis=0)
    tapp = np.where(admissible, tapp, np.nan)
    return twb + tapp, tapp, rlg, admissible, n_roots
