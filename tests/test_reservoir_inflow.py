# -*- coding: utf-8 -*-
"""
Reservoir Inflow Models
=======================

Covers the options of the radial Darcy inflow in
:mod:`superhot_wellbore.reservoir` that answer reviewer questions R1
and R3 of the GEOPHIRES documentation of the coupled wellbore model:

- the skin factor S, added to ln(r_e / r_w), which also represents a
  vertical fracture (Prats, 1961);
- the mass pseudo-pressure models, which integrate rho/mu along the
  isenthalpic or isothermal path from the far field instead of
  evaluating it at one state (Al-Hussainy et al., 1966).

The reference results do not come from the code under test: the
closed form of an ideal gas, the small-drawdown limit, the ordering of
the two paths and Prats' effective wellbore radius.

Measured, IDDP-1 conditions (16.11 MPa, 500 C, 5318 md*m, 500 m
drainage radius, 0.217 m well): at 48 kg/s the upstream model draws
the reservoir down by 6.18 MPa, the isenthalpic pseudo-pressure by
9.43 MPa and the isothermal one by 10.74 MPa, so the upstream model
underestimates the drawdown by 34 % there, in line with the 30-40 %
the module documents at about 8 MPa.
"""

import warnings

import CoolProp.CoolProp as CP
import numpy as np
import pytest

from superhot_wellbore import reservoir
from superhot_wellbore.client.config import (INFLOW_PROPERTY_MODES,
                                             CoupledWellboreRequest,
                                             ReservoirConfig)
from superhot_wellbore.reservoir import (INFLOW_PROPERTY_MODELS, MD_TO_M2,
                                         bottomhole_pressure,
                                         coupled_model, darcy_pressure_drop,
                                         pseudo_pressure_bottomhole,
                                         rock_temperature_boiling)

WELL = {'depth_m': 2100, 'diameter_m': 0.217}
IDDP_1 = {'drainage_radius_m': 500.0, 'transmissivity_md_m': 5318.0}
R_W = 0.217 / 2.0


def _drawdown(mdot, model, P_MPa=16.11, T_C=500.0, **extra):
    """Reservoir drawdown [MPa] of a model at IDDP-1 conditions."""
    params = dict(IDDP_1, inflow_properties=model, **extra)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return bottomhole_pressure(mdot, P_MPa, T_C, params,
                                   WELL)['dP_reservoir_MPa']


# ====================================================================
# 1. SKIN FACTOR
# ====================================================================

def test_skin_adds_to_the_log_term():
    """dP scales with ln(r_e/r_w) + S."""
    plain = darcy_pressure_drop(30.0, 16.11, 500.0, IDDP_1, WELL)
    skinned = darcy_pressure_drop(30.0, 16.11, 500.0,
                                  dict(IDDP_1, skin=3.0), WELL)
    ln_ratio = np.log(500.0 / R_W)
    assert skinned / plain == pytest.approx((ln_ratio + 3.0) / ln_ratio,
                                            rel=1e-12)


def test_fracture_skin_equals_an_enlarged_wellbore():
    """
    Prats (1961): an infinite-conductivity vertical fracture of
    half-length x_f acts as a wellbore of radius x_f / 2, i.e. as the
    skin S = ln(2 r_w / x_f).
    """
    x_f = 50.0
    with_skin = darcy_pressure_drop(
        30.0, 16.11, 500.0, dict(IDDP_1, skin=np.log(2.0 * R_W / x_f)),
        WELL)
    enlarged = darcy_pressure_drop(30.0, 16.11, 500.0, IDDP_1,
                                   dict(WELL, diameter_m=x_f))
    assert with_skin == pytest.approx(enlarged, rel=1e-12)


def test_skin_that_removes_the_inflow_resistance_is_rejected():
    """ln(r_e/r_w) + S must stay positive."""
    with pytest.raises(ValueError):
        darcy_pressure_drop(30.0, 16.11, 500.0, dict(IDDP_1, skin=-10.0),
                            WELL)


def test_skin_applies_to_the_pseudo_pressure_models():
    """
    The pseudo-pressure drop is mdot (ln(r_e/r_w) + S) / (2 pi k b):
    a skin S at mdot is the same well as no skin at
    mdot (ln(r_e/r_w) + S) / ln(r_e/r_w).
    """
    ln_ratio = np.log(500.0 / R_W)
    params = dict(IDDP_1, inflow_properties='pseudo_pressure')
    skinned = pseudo_pressure_bottomhole(20.0, 16.11, 500.0,
                                         dict(params, skin=2.0), WELL)
    scaled = pseudo_pressure_bottomhole(20.0 * (ln_ratio + 2.0) / ln_ratio,
                                        16.11, 500.0, params, WELL)
    assert skinned == pytest.approx(scaled, rel=1e-12)


# ====================================================================
# 2. MASS PSEUDO-PRESSURE
# ====================================================================

def test_pseudo_pressure_reproduces_the_ideal_gas(monkeypatch):
    """
    With rho = c P and a constant mu, m(P) = c P^2 / (2 mu), so the
    bottomhole pressure is sqrt(P_res^2 - 2 mu drop / c), the
    deliverability of a gas well (Al-Hussainy et al., 1966).
    """
    c, mu = 5e-6, 3e-5   # kg/m3 per Pa (80 kg/m3 at 16 MPa), Pa.s
    monkeypatch.setattr(
        reservoir, '_rho_over_mu_along_path',
        lambda P_MPa, P_res, T_res, path: c * np.asarray(P_MPa) * 1e6 / mu)
    reservoir._mass_pseudo_pressure_table.cache_clear()
    try:
        mdot = 20.0
        P_bh = pseudo_pressure_bottomhole(mdot, 16.0, 500.0, IDDP_1, WELL)
        drop = (mdot * np.log(500.0 / R_W)
                / (2.0 * np.pi * 5318.0 * MD_TO_M2))
        expected = np.sqrt(16e6**2 - 2.0 * mu * drop / c) / 1e6
        assert P_bh == pytest.approx(expected, rel=1e-9)
    finally:
        reservoir._mass_pseudo_pressure_table.cache_clear()


@pytest.mark.parametrize('model', ['pseudo_pressure',
                                   'pseudo_pressure_isothermal'])
def test_pseudo_pressure_agrees_with_upstream_at_small_drawdown(model):
    """
    The models differ only through the change of rho/mu over the
    drawdown, which is first order in dP / P_res for steam (0.5 % at
    1 kg/s, a 0.15 MPa drawdown): the relative difference stays below
    dP / P_res and shrinks in proportion with the flow rate.
    """
    differences = []
    for mdot in (1.0, 0.1):
        upstream = _drawdown(mdot, 'upstream')
        difference = abs(_drawdown(mdot, model) / upstream - 1.0)
        assert difference < upstream / 16.11, f'{mdot} kg/s'
        differences.append(difference)
    assert differences[1] == pytest.approx(differences[0] / 10.0,
                                           rel=0.2)


@pytest.mark.parametrize('mdot', [30.0, 48.0])
def test_paths_bound_the_upstream_model_at_large_drawdown(mdot):
    """
    Expanding superheated steam loses rho/mu toward the well, so
    integrating it raises the drawdown above the far-field-state
    value; holding the reservoir temperature (isothermal) lowers
    rho/mu further than the isenthalpic expansion, which cools.
    """
    upstream = _drawdown(mdot, 'upstream')
    isenthalpic = _drawdown(mdot, 'pseudo_pressure')
    isothermal = _drawdown(mdot, 'pseudo_pressure_isothermal')
    assert upstream < isenthalpic < isothermal


def test_iddp1_drawdown_at_48_kgs():
    """The drawdowns recorded in the module docstring."""
    assert _drawdown(48.0, 'upstream') == pytest.approx(6.18, abs=0.02)
    assert _drawdown(48.0, 'pseudo_pressure') == pytest.approx(9.43,
                                                               abs=0.02)
    assert _drawdown(48.0, 'pseudo_pressure_isothermal') == \
        pytest.approx(10.74, abs=0.02)


def test_flow_beyond_the_deliverability_has_no_bottomhole_state():
    """72 kg/s cannot be drawn from IDDP-1 above 0.1 MPa."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        bh = bottomhole_pressure(
            72.0, 16.11, 500.0,
            dict(IDDP_1, inflow_properties='pseudo_pressure'), WELL)
    assert bh['P_bh_MPa'] == 0.0
    assert np.isnan(bh['T_feedzone_C'])


@pytest.mark.parametrize('model', list(INFLOW_PROPERTY_MODELS))
def test_fluid_enters_the_well_isenthalpically(model):
    """Every model hands the wellbore the reservoir enthalpy."""
    h_res = CP.PropsSI('H', 'T', 773.15, 'P', 16.11e6, 'Water')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        bh = bottomhole_pressure(30.0, 16.11, 500.0,
                                 dict(IDDP_1, inflow_properties=model),
                                 WELL)
    assert bh['h_feedzone_Jkg'] == pytest.approx(h_res, rel=1e-12)


def test_unknown_inflow_model_is_rejected():
    with pytest.raises(ValueError):
        bottomhole_pressure(30.0, 16.11, 500.0,
                            dict(IDDP_1, inflow_properties='mean'), WELL)


@pytest.mark.slow
def test_coupled_model_runs_on_the_pseudo_pressure_inflow():
    """
    The IDDP-1 well at 30 kg/s reaches the surface on every model,
    with a lower wellhead pressure where the drawdown is larger.
    """
    wp = dict(WELL, depth_m=2100)
    rock = rock_temperature_boiling(wp)
    whp = {}
    for model in INFLOW_PROPERTY_MODELS:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            r = coupled_model(30.0, 16.11, 500.0, rock,
                              dict(IDDP_1, inflow_properties=model), wp)
        assert r['success'], model
        whp[model] = r['whp_MPa']
    assert (whp['upstream'] > whp['pseudo_pressure']
            > whp['pseudo_pressure_isothermal'])


# ====================================================================
# 3. CLIENT CONFIGURATION
# ====================================================================

def test_client_models_match_the_core():
    assert INFLOW_PROPERTY_MODES == INFLOW_PROPERTY_MODELS


def test_client_passes_the_inflow_options_to_the_core():
    params = ReservoirConfig(inflow_properties='pseudo_pressure',
                             skin=2.0).to_reservoir_params()
    assert params['inflow_properties'] == 'pseudo_pressure'
    assert params['skin'] == 2.0


def test_client_defaults_keep_the_published_inflow():
    params = ReservoirConfig().to_reservoir_params()
    assert params['inflow_properties'] == 'upstream'
    assert params['skin'] == 0.0


def test_client_rejects_an_unknown_inflow_model():
    request = CoupledWellboreRequest.from_dict(
        {'reservoir': {'inflow_properties': 'mean'}})
    with pytest.raises(ValueError):
        request.validate()
