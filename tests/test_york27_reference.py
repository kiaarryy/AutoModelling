"""Regression tests for the YorkCalc-27 reference implementation.

The oracle test against the shipped Site A FMU lives in
``scripts/audit_york27_closed_loop.py`` because it needs the external Site A
tables.  These tests only need the library itself.
"""
from __future__ import annotations

import numpy as np
import pytest

from autofmu.devices.york27_reference import (
    MBL_DEFAULT_COEFFICIENTS,
    York27Params,
    closed_loop,
    liquid_gas_ratio,
    open_loop,
    york_approach,
    york_design_matrix,
)


def test_default_coefficients_match_mbl_length_and_head():
    assert MBL_DEFAULT_COEFFICIENTS.shape == (27,)
    # first and last entries of Buildings 12.1.0 Correlations/yorkCalc.mo c[:]
    assert MBL_DEFAULT_COEFFICIENTS[0] == pytest.approx(-0.359741205)
    assert MBL_DEFAULT_COEFFICIENTS[-1] == pytest.approx(-0.000001516821552)


def test_design_matrix_term_order_matches_mbl():
    """Spot-check the monomial order against yorkCalc.mo's written-out sum."""
    t, r, l = 20.0, 5.0, 1.3
    x = york_design_matrix(np.array([t]), np.array([r]), np.array([l]))[0]
    expected = [1, t, t * t, r, t * r, t * t * r, r * r, t * r * r, t * t * r * r,
                l, t * l, t * t * l, r * l, t * r * l, t * t * r * l,
                r * r * l, t * r * r * l, t * t * r * r * l,
                l * l, t * l * l, t * t * l * l, r * l * l, t * r * l * l,
                t * t * r * l * l, r * r * l * l, t * r * r * l * l,
                t * t * r * r * l * l]
    assert x == pytest.approx(np.array(expected))


def test_liquid_gas_ratio_clipping():
    p = York27Params(m_flow_nominal=100.0, rlg_min=0.05, rlg_max=8.0)
    rlg = liquid_gas_ratio(np.array([0.0, 100.0, 1e6]), np.array([1.0, 1.0, 0.05]), p)
    assert rlg[0] == pytest.approx(p.rlg_min)
    assert rlg[1] == pytest.approx(1.0)
    assert rlg[2] == pytest.approx(p.rlg_max)


def test_open_loop_consumes_measured_range():
    """Open loop is the leaking form: changing the measured range moves the output."""
    p = York27Params(m_flow_nominal=100.0)
    kw = dict(tin_c=np.array([30.0]), twb_c=np.array([22.0]),
              mdot_cell_kgps=np.array([100.0]), y_used=np.array([1.0]),
              coefficients=MBL_DEFAULT_COEFFICIENTS, params=p)
    a, _, _ = open_loop(tran_c=np.array([4.0]), **kw)
    b, _, _ = open_loop(tran_c=np.array([6.0]), **kw)
    assert abs(a[0] - b[0]) > 1e-3


def test_closed_loop_is_self_consistent():
    """The solution satisfies TRan = Tin - TOut_s by construction."""
    p = York27Params(m_flow_nominal=100.0)
    tin, twb = np.array([30.0]), np.array([22.0])
    tout, tapp, rlg, admissible, _ = closed_loop(
        tin_c=tin, twb_c=twb, mdot_cell_kgps=np.array([100.0]),
        y_used=np.array([1.0]), coefficients=MBL_DEFAULT_COEFFICIENTS, params=p)
    assert admissible.all()
    recomputed = york_approach(twb, tin - tout, rlg, MBL_DEFAULT_COEFFICIENTS)
    assert recomputed[0] == pytest.approx(tapp[0], abs=1e-5)
    assert tout[0] == pytest.approx(twb[0] + tapp[0])


def test_closed_loop_solution_stays_physically_admissible():
    """The approach can never exceed Tin - Twb (range must stay non-negative)."""
    p = York27Params(m_flow_nominal=100.0)
    tin = np.array([30.0, 26.0, 24.0])
    twb = np.array([22.0, 22.0, 22.0])
    tout, tapp, _, admissible, _ = closed_loop(
        tin_c=tin, twb_c=twb, mdot_cell_kgps=np.full(3, 100.0),
        y_used=np.full(3, 1.0), coefficients=MBL_DEFAULT_COEFFICIENTS, params=p)
    ok = admissible & np.isfinite(tapp)
    assert (tapp[ok] >= -1e-9).all()
    assert (tapp[ok] <= (tin - twb)[ok] + 1e-9).all()
    assert (tout[ok] <= tin[ok] + 1e-9).all()


def test_closed_loop_is_deterministic():
    """No warm start, no hidden state: repeated calls agree exactly."""
    p = York27Params(m_flow_nominal=100.0)
    sig = dict(tin_c=np.array([30.0]), twb_c=np.array([22.0]),
               mdot_cell_kgps=np.array([100.0]), y_used=np.array([1.0]),
               coefficients=MBL_DEFAULT_COEFFICIENTS, params=p)
    assert closed_loop(**sig)[0][0] == pytest.approx(closed_loop(**sig)[0][0])
