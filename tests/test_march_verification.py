# -*- coding: utf-8 -*-
"""
Wellbore March Verification
===========================

Code verification of wellbore_simulate(): does the forward-Euler
march solve Eqs. 4-5 of Scott (2026) correctly? The other test
modules check the march's bookkeeping and its behaviour in specific
regimes; these tests check the solution itself against results that
do not come from the march:

1. Adiabatic, frictionless limit. With no heat loss (U = 0) and a
   flow rate small enough that friction and kinetic energy vanish,
   Eqs. 4-5 reduce to dP = -rho g dz and dh = -g dz, so dh = dP/rho:
   the column is isentropic. Its wellhead state then follows from
   the equation of state alone: h_wh = h_fz - g L, and P_wh is the
   pressure at which the isentrope through the feedzone state has
   that enthalpy.
2. Energy-balance closure. With heat loss, the integrated energy
   equation is m (h_fz - h_wh) = m g L + m (v_wh^2 - v_fz^2)/2 + Q,
   with Q the heat lost to the rock, integrated here by the
   trapezoidal rule over the returned profiles.
3. Step-size convergence. Forward Euler is first order: halving the
   step should halve the change in the wellhead pressure, and the
   Richardson estimate of the error at the default 10 m step should
   be small.

Each case covers a regime of the paper: superheated vapor (an IDDP-1
-like well), a vapor that condenses into the dome, a liquid that
flashes, and a supercritical feed that crosses P_crit through the
IAPWS97 routing band. Measured values, with the march integrating
exactly the well depth (fix-k3-march-dz), CoolProp 8.0.0 and iapws
1.5.5:

- Isentropic limit, WHP error at 10 m / 2 m steps: superheated
  -0.12 / -0.03 kPa, flashing -10.1 / -2.0 kPa, supercritical
  -0.66 / -0.3 kPa (of 1.0, 6.1 and 3.7 MPa column pressure drops).
  The flashing error is pure discretization: it halves with the step
  (-5.1 kPa at 5 m, -1.0 kPa at 1 m). The supercritical floor of
  about 0.2 kPa is the IAPWS97 (march) versus IAPWS-95 (reference)
  difference. h_wh = h_fz - g L to 0.01 J/kg in every case.
- Energy closure: -0.16% to -0.40% at 10 m, -0.08% to -0.20% at 5 m,
  -0.03% to -0.08% at 2 m: the residual is the first-order
  difference between the march's start-of-step heat loss and the
  trapezoidal rule, and it vanishes with the step.
- Convergence, observed order from 20/10/5 m steps: 0.95-1.03.
  Richardson error at 10 m: +0.02 kPa (superheated, 20 kg/s),
  +3.2 kPa (IDDP-1-like, 48 kg/s), +0.2 kPa (supercritical,
  20 kg/s), -11 kPa (flashing, 90 kg/s; 0.5% of its 2.2 MPa WHP).

Run against the march before that fix (geophires-client at 3be9a9d),
which integrated one step past the wellhead, the three isentropic
tests fail (h_wh falls g * delta_z below h_fz - g L) and so do the
flashing and IDDP-1-like convergence tests.

References
----------
Scott, S.W. (2026). Thermo-hydraulic drivers of superhot geothermal
    well performance. Geothermics 141, 103784.
    https://doi.org/10.1016/j.geothermics.2026.103784
"""

import warnings

import CoolProp.CoolProp as CP
import numpy as np
import pytest
from scipy.optimize import brentq

from superhot_wellbore.reservoir import rock_temperature_linear
from superhot_wellbore.wellbore_physics import (DEFAULT_WELL_PARAMS, GRAVITY,
                                                wellbore_simulate)


# ====================================================================
# HELPERS
# ====================================================================

def _h_TP(T_C, P_MPa):
    """Specific enthalpy [J/kg] at (T, P)."""
    return CP.PropsSI('H', 'T', T_C + 273.15, 'P', P_MPa * 1e6, 'Water')


def _march(P_bh_MPa, h_fz_Jkg, mdot, depth_m, T_bottom_C, delta_z_m,
           heat_loss_factor=DEFAULT_WELL_PARAMS['heat_loss_factor']):
    """
    March a well with a linear rock profile (10 C at the surface,
    T_bottom_C at depth) and return the profiles as arrays.
    """
    wp = dict(DEFAULT_WELL_PARAMS)
    wp.update({'depth_m': depth_m, 'delta_z_m': delta_z_m,
               'heat_loss_factor': heat_loss_factor})
    rock = rock_temperature_linear(wp, T_bottom_C)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        profiles = wellbore_simulate(P_bh_MPa, h_fz_Jkg, mdot, rock, wp)
    names = ('T_C', 'P_MPa', 'h_MJkg', 'v_ms', 'x', 'T_rock_C')
    out = {name: np.array([value for _, value in profile])
           for name, profile in zip(names, profiles[:6])}
    out['depth_m'] = np.array([d for d, _ in profiles[1]], dtype=float)
    assert out['depth_m'][-1] == 0, 'the march reached the wellhead'
    return out


def _isentropic_wellhead_pressure(P_bh_MPa, h_fz_Jkg, depth_m):
    """
    Wellhead pressure [MPa] of an adiabatic, frictionless column: the
    pressure at which the feedzone isentrope (IAPWS-95, CoolProp)
    reaches h_fz - g L.
    """
    s_fz = CP.PropsSI('S', 'P', P_bh_MPa * 1e6, 'H', h_fz_Jkg, 'Water')
    h_wh = h_fz_Jkg - GRAVITY * depth_m

    def residual(P_MPa):
        return CP.PropsSI('H', 'P', P_MPa * 1e6, 'S', s_fz, 'Water') - h_wh

    return brentq(residual, 0.1, P_bh_MPa, xtol=1e-10)


def _trapezoid(values, depths):
    """Trapezoidal integral over depth, positive upward."""
    return float(np.sum(0.5 * (values[1:] + values[:-1])
                        * (depths[:-1] - depths[1:])))


def _energy_closure(march, mdot):
    """
    Relative residual of m (h_fz - h_wh) = m g L + m d(v^2/2) + Q.
    """
    U = DEFAULT_WELL_PARAMS['heat_loss_factor']
    h = march['h_MJkg'] * 1e6
    v = march['v_ms']
    L = march['depth_m'][0] - march['depth_m'][-1]
    Q = _trapezoid(U * (march['T_C'] - march['T_rock_C']),
                   march['depth_m'])
    lhs = mdot * (h[0] - h[-1])
    rhs = mdot * GRAVITY * L + mdot * (v[-1]**2 - v[0]**2) / 2.0 + Q
    return (lhs - rhs) / lhs


# ====================================================================
# CASES
# ====================================================================

# (P_bh [MPa], h_fz [J/kg], depth [m], T_bottom [C])
_ADIABATIC_CASES = {
    'superheated vapor, 16.4 MPa, 530 C, 2100 m':
        (16.4, _h_TP(530, 16.4), 2100, 530),
    'flashing liquid, 10 MPa, 300 C, 2000 m':
        (10.0, _h_TP(300, 10.0), 2000, 300),
    'supercritical, 25 MPa, 500 C, 4500 m':
        (25.0, _h_TP(500, 25.0), 4500, 500),
}

# (P_bh [MPa], h_fz [J/kg], mdot [kg/s], depth [m], T_bottom [C])
_HEATED_CASES = {
    'superheated vapor, 16.4 MPa, 530 C, 20 kg/s':
        (16.4, _h_TP(530, 16.4), 20.0, 2100, 530),
    'vapor into the dome, 8 MPa, 2.78 MJ/kg, 10 kg/s':
        (8.0, 2.78e6, 10.0, 2000, 300),
    'flashing liquid, 10 MPa, 300 C, 90 kg/s':
        (10.0, _h_TP(300, 10.0), 90.0, 2000, 300),
    'supercritical, 25 MPa, 500 C, 20 kg/s':
        (25.0, _h_TP(500, 25.0), 20.0, 4500, 500),
}

_CONVERGENCE_CASES = dict(_HEATED_CASES)
del _CONVERGENCE_CASES['vapor into the dome, 8 MPa, 2.78 MJ/kg, 10 kg/s']
_CONVERGENCE_CASES['IDDP-1-like, 12 MPa sandface, 48 kg/s'] = (
    12.0, _h_TP(500, 16.11), 48.0, 2100, 500)

# Small enough that friction (about 0.3 kPa over the superheated
# column at 0.5 kg/s) and kinetic energy are negligible
_ADIABATIC_MDOT_KGS = 0.05


# ====================================================================
# 1. ADIABATIC, FRICTIONLESS LIMIT: AN ISENTROPIC COLUMN
# ====================================================================

@pytest.mark.slow
@pytest.mark.parametrize('case', list(_ADIABATIC_CASES))
def test_adiabatic_column_follows_the_isentrope(case):
    """
    With U = 0 and negligible friction, the wellhead enthalpy is
    h_fz - g L and the wellhead pressure that of the feedzone
    isentrope at that enthalpy, within 0.25% of the column pressure
    drop at the default 10 m step and 0.05% at 2 m.
    """
    P_bh, h_fz, depth, T_bottom = _ADIABATIC_CASES[case]
    P_ref = _isentropic_wellhead_pressure(P_bh, h_fz, depth)
    column = P_bh - P_ref
    for delta_z, tolerance in ((10, 0.0025), (2, 0.0005)):
        march = _march(P_bh, h_fz, _ADIABATIC_MDOT_KGS, depth, T_bottom,
                       delta_z, heat_loss_factor=0.0)
        h_wh = march['h_MJkg'][-1] * 1e6
        assert h_wh == pytest.approx(h_fz - GRAVITY * depth, abs=1.0), \
            f'h_wh at {delta_z} m step'
        error = march['P_MPa'][-1] - P_ref
        assert abs(error) < tolerance * column, \
            (f'{delta_z} m step: WHP {march["P_MPa"][-1]:.5f} MPa, '
             f'isentropic {P_ref:.5f} MPa, error {error * 1e3:+.2f} kPa')


# ====================================================================
# 2. ENERGY-BALANCE CLOSURE
# ====================================================================

@pytest.mark.slow
@pytest.mark.parametrize('case', list(_HEATED_CASES))
def test_energy_balance_closes_at_first_order(case):
    """
    The enthalpy drop balances gravity, kinetic energy and the
    integrated heat loss to within 0.5% at the default 10 m step, and
    the residual halves with the step (first-order closure).
    """
    P_bh, h_fz, mdot, depth, T_bottom = _HEATED_CASES[case]
    closure_10 = _energy_closure(
        _march(P_bh, h_fz, mdot, depth, T_bottom, 10), mdot)
    closure_5 = _energy_closure(
        _march(P_bh, h_fz, mdot, depth, T_bottom, 5), mdot)
    assert abs(closure_10) < 0.005, f'closure {closure_10:+.4%} at 10 m'
    assert 0.4 < closure_5 / closure_10 < 0.6, \
        f'closure {closure_10:+.4%} at 10 m, {closure_5:+.4%} at 5 m'


# ====================================================================
# 3. STEP-SIZE CONVERGENCE
# ====================================================================

@pytest.mark.slow
@pytest.mark.parametrize('case', list(_CONVERGENCE_CASES))
def test_wellhead_pressure_converges_at_first_order(case):
    """
    Observed order of the WHP from 20, 10 and 5 m steps is about one,
    and the Richardson estimate of the error at the default 10 m step
    is under 15 kPa and 1% of the WHP.
    """
    P_bh, h_fz, mdot, depth, T_bottom = _CONVERGENCE_CASES[case]
    whp = {delta_z: _march(P_bh, h_fz, mdot, depth, T_bottom,
                           delta_z)['P_MPa'][-1]
           for delta_z in (20, 10, 5)}
    order = np.log2((whp[20] - whp[10]) / (whp[10] - whp[5]))
    assert 0.7 < order < 1.3, f'observed order {order:.2f}'
    # First-order Richardson: whp(0) ~ 2 whp(5) - whp(10)
    error_10 = whp[10] - (2.0 * whp[5] - whp[10])
    assert abs(error_10) < 0.015, f'error at 10 m {error_10 * 1e3:+.2f} kPa'
    assert abs(error_10) < 0.01 * whp[10]
