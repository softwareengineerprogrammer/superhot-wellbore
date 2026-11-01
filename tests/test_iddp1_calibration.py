# -*- coding: utf-8 -*-
"""
IDDP-1 Calibration
==================

Regression test of the IDDP-1 calibration behind Figure 5 of Scott
(2026), as run by examples/figure5_calibrate_iddp1.py. That script
fits the reservoir pressure and transmissivity of the coupled
reservoir-wellbore model to the six measured (mass flow rate,
wellhead pressure) points of IDDP-1 (Ingason et al., 2014) and
arrives at P_res = 16.11 MPa and k*b = 5318 md*m. The setup is the
script's: T_res = 500 C, a 2100 m well of 0.217 m internal diameter,
a 500 m drainage radius and the boiling-point-with-depth rock
temperature profile.

Nothing else pins this calibration, so a change anywhere in the
reservoir or wellbore physics could move it silently. The fast
tests evaluate the model only at the published parameters and at
+/-5% perturbations of each; the full Nelder-Mead refit is marked
slow.

Baseline, measured on the geophires-client branch at 3be9a9d with
CoolProp 8.0.0 and iapws 1.5.5: the WHP RMS error at (16.11 MPa,
5318 md*m) is 0.7548 MPa, and a 5% perturbation raises it to

    P_res  +5%: 1.4956 MPa      P_res  -5%: 1.9119 MPa
    k*b    +5%: 0.8065 MPa      k*b    -5%: 0.8236 MPa

The refit started from the published values stays there (33
objective evaluations, about a minute). Started instead from (15.5
MPa, 6000 md*m) or (16.8 MPa, 4500 md*m), the same refit converges
to (16.110 MPa, 5318 md*m) and (16.112 MPa, 5316 md*m), so the
published values are the minimum, not an artefact of the start.

References
----------
Scott, S.W. (2026). Thermo-hydraulic drivers of superhot geothermal
    well performance. Geothermics 141, 103784.
    https://doi.org/10.1016/j.geothermics.2026.103784
Ingason, K., Kristjansson, V., Einarsson, K. (2014). Design and
    development of the discharge system of IDDP-1. Geothermics 49,
    58-65.
"""

import warnings

import numpy as np
import pytest
from scipy.optimize import minimize

from superhot_wellbore.reservoir import (DEFAULT_WELL_PARAMS,
                                         evaluate_at_flow_rates,
                                         rock_temperature_boiling)


# ====================================================================
# IDDP-1 DATA AND MODEL SETUP (examples/figure5_calibrate_iddp1.py)
# ====================================================================

OBSERVED_WHP_MPA = np.array([4.1, 6.2, 7.7, 10.1, 13.5, 14.1])
OBSERVED_FLOW_KGS = np.array([48, 41, 44, 30, 11, 6.4])

T_RESERVOIR_C = 500.0
DRAINAGE_RADIUS_M = 500.0

WELL_PARAMS = dict(DEFAULT_WELL_PARAMS)
WELL_PARAMS['depth_m'] = 2100
WELL_PARAMS['diameter_m'] = 0.217

#: Published calibration (Scott, 2026, Figure 5)
P_RES_MPA = 16.11
TRANSMISSIVITY_MD_M = 5318.0

#: WHP RMS error [MPa] at the published calibration (see docstring)
BASELINE_RMS_MPA = 0.7548
BASELINE_RMS_TOLERANCE_MPA = 1e-3


@pytest.fixture(scope='module')
def rock_temperatures():
    """Boiling-point-with-depth rock profile of the 2100 m well."""
    return rock_temperature_boiling(WELL_PARAMS)


def _evaluate(P_res_MPa, transmissivity_md_m, rock_temperatures):
    """Model WHP at the six observed flow rates."""
    reservoir_params = {'drainage_radius_m': DRAINAGE_RADIUS_M,
                        'transmissivity_md_m': transmissivity_md_m}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return evaluate_at_flow_rates(
            OBSERVED_FLOW_KGS, P_res_MPa, T_RESERVOIR_C,
            rock_temperatures, reservoir_params, WELL_PARAMS)


def _whp_rms(P_res_MPa, transmissivity_md_m, rock_temperatures):
    """WHP RMS error [MPa] against the IDDP-1 data."""
    r = _evaluate(P_res_MPa, transmissivity_md_m, rock_temperatures)
    assert np.all(r['success']), \
        (f'model failed at P_res={P_res_MPa:.3f} MPa, '
         f'k*b={transmissivity_md_m:.0f} md*m: '
         f'flows {OBSERVED_FLOW_KGS[~r["success"]]}')
    return float(np.sqrt(np.mean((r['whp_MPa'] - OBSERVED_WHP_MPA)**2)))


@pytest.fixture(scope='module')
def published_rms(rock_temperatures):
    """WHP RMS error at the published calibration."""
    return _whp_rms(P_RES_MPA, TRANSMISSIVITY_MD_M, rock_temperatures)


# ====================================================================
# 1. RMS ERROR AT THE PUBLISHED CALIBRATION
# ====================================================================

def test_rms_at_published_calibration_matches_baseline(published_rms):
    """The fit quality has not drifted from the recorded baseline."""
    assert published_rms == pytest.approx(
        BASELINE_RMS_MPA, abs=BASELINE_RMS_TOLERANCE_MPA), \
        f'RMS = {published_rms:.4f} MPa'


# ====================================================================
# 2. THE PUBLISHED CALIBRATION IS A LOCAL MINIMUM
# ====================================================================

@pytest.mark.parametrize('P_factor, kb_factor',
                         [(1.05, 1.0), (0.95, 1.0),
                          (1.0, 1.05), (1.0, 0.95)],
                         ids=['P_res +5%', 'P_res -5%',
                              'k*b +5%', 'k*b -5%'])
def test_perturbing_a_parameter_increases_rms(P_factor, kb_factor,
                                              published_rms,
                                              rock_temperatures):
    """Moving either parameter 5% away makes the fit worse."""
    perturbed = _whp_rms(P_RES_MPA * P_factor,
                         TRANSMISSIVITY_MD_M * kb_factor,
                         rock_temperatures)
    assert perturbed > published_rms, \
        (f'perturbed RMS {perturbed:.4f} MPa <= published '
         f'{published_rms:.4f} MPa')


# ====================================================================
# 3. FULL NELDER-MEAD REFIT
# ====================================================================

@pytest.mark.slow
def test_nelder_mead_refit_stays_at_published_calibration(
        rock_temperatures):
    """
    The calibration of examples/figure5_calibrate_iddp1.py, with its
    starting point, bounds and solver options, returns the published
    values.
    """
    def objective(params):
        P_res, kxb = params
        if not (10 < P_res < 25 and 500 < kxb < 15000):
            return 1e6
        r = _evaluate(P_res, kxb, rock_temperatures)
        if not np.all(r['success']):
            return 1e6
        return np.sqrt(np.mean((r['whp_MPa'] - OBSERVED_WHP_MPA)**2))

    result = minimize(
        objective, x0=[P_RES_MPA, TRANSMISSIVITY_MD_M],
        method='Nelder-Mead',
        options={'maxiter': 100, 'xatol': 5.0, 'fatol': 0.005,
                 'adaptive': True})
    P_fit, kb_fit = result.x
    assert P_fit == pytest.approx(P_RES_MPA, abs=0.01), \
        f'P_res = {P_fit:.3f} MPa'
    assert kb_fit == pytest.approx(TRANSMISSIVITY_MD_M, rel=0.01), \
        f'k*b = {kb_fit:.0f} md*m'
    assert result.fun == pytest.approx(
        BASELINE_RMS_MPA, abs=BASELINE_RMS_TOLERANCE_MPA), \
        f'RMS = {result.fun:.4f} MPa'
