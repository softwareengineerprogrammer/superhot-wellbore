# -*- coding: utf-8 -*-
"""
Client Behaviour
================

Two things the client does before and during a run. It resolves the
well depth, either deriving it from the reservoir pressure or
snapping a requested depth onto the integration step, and it says so
in its notes rather than changing the well behind the user's back.
And it solves the coupled reservoir-wellbore model, whose solution
has to come out physically ordered - that solve is real
thermodynamics and is therefore marked slow.

Author: superhot-wellbore GEOPHIRES client
"""

import numpy as np
import pytest

from superhot_wellbore.client.client import SuperhotWellboreClient
from superhot_wellbore.client.config import SuperhotRequest


# ====================================================================
# WELL DEPTH RESOLUTION
# ====================================================================

def test_depth_derived_from_reservoir_pressure():
    """Without a depth, 30 MPa hydrostatic implies 3500 m."""
    client = SuperhotWellboreClient(SuperhotRequest.from_dict(
        {'reservoir': {'P_reservoir_MPa': 30.0}}))
    assert client.depth_m == pytest.approx(3500.0, abs=1e-9), \
        'derived from 30 MPa'
    assert any('derived' in note for note in client.notes), \
        'derivation is reported'


def test_depth_snapped_to_the_integration_step():
    """A depth off the grid is snapped up and reported."""
    snapped = SuperhotWellboreClient(SuperhotRequest.from_dict(
        {'well': {'depth_m': 2106, 'delta_z_m': 10}}))
    assert snapped.depth_m == pytest.approx(2110.0, abs=1e-9), \
        'snapped to the integration step'
    assert any('snapped' in note for note in snapped.notes), \
        'snapping is reported'


# ====================================================================
# COUPLED MODEL SOLVE
# ====================================================================

@pytest.mark.slow
def test_coupled_solve_is_physically_ordered():
    """A real solve returns a usable, physically ordered profile."""
    request = SuperhotRequest.from_dict({
        'name': 'selftest',
        'reservoir': {'P_reservoir_MPa': 30.0,
                      'T_reservoir_C': 450.0,
                      'transmissivity_md_m': 1000.0},
        'operating': {'control': 'whp', 'target_whp_MPa': 10.0},
        'time': {'plant_lifetime_yr': 2, 'timesteps_per_year': 2},
        'solver': {'max_solve_points': 2},
    })
    profile = SuperhotWellboreClient(request).solve_profile()
    first = profile.initial

    assert first is not None, 'a solution was found'
    assert first.mass_flow_kgs > 0, \
        'flow rate is positive, got ' + str(first.mass_flow_kgs)
    assert abs(first.whp_MPa - 10.0) < 0.5, \
        'wellhead pressure meets the target, got ' + str(first.whp_MPa)
    assert (first.T_wellhead_C < first.T_feedzone_C
            < request.reservoir.T_reservoir_C), \
        ('temperatures are ordered from reservoir to surface: '
         f'{first.T_wellhead_C} < {first.T_feedzone_C} < '
         f'{request.reservoir.T_reservoir_C}')
    assert first.dP_reservoir_MPa > 0, \
        'drawdown is positive, got ' + str(first.dP_reservoir_MPa)
    assert first.P_bh_MPa < request.reservoir.P_reservoir_MPa, \
        'bottomhole pressure sits below the reservoir'
    assert profile.n_failed == 0, 'every timestep is usable'
    assert np.isfinite(first.power_MWe), \
        'the power diagnostic is available, got ' + str(first.power_MWe)


# ====================================================================
# HELD FLOW RATE
# ====================================================================

def test_hold_flow_solves_once_for_pressure_then_prescribes_flow(monkeypatch):
    """With hold='flow', the flow found at t = 0 is prescribed afterwards."""
    from superhot_wellbore.client import client as client_module

    calls = []

    def fake_solve_flow_for_whp(**kwargs):
        calls.append(('whp', kwargs['P_reservoir_MPa']))
        return {'success': True, 'converged': True, 'mass_flow_kgs': 50.0,
                'whp_MPa': kwargs['target_whp_MPa'], 'T_surface_C': 300.0,
                'h_surface_MJkg': 2.7, 'T_feedzone_C': 390.0,
                'h_feedzone_MJkg': 2.8, 'P_bh_MPa': 20.0,
                'dP_reservoir_MPa': 10.0, 'choked': False}

    def fake_coupled_model(**kwargs):
        calls.append(('flow', kwargs['mass_flow_rate']))
        return {'success': True, 'mass_flow_kgs': kwargs['mass_flow_rate'],
                'whp_MPa': 8.0, 'T_surface_C': 290.0, 'h_surface_MJkg': 2.6,
                'T_feedzone_C': 380.0, 'h_feedzone_MJkg': 2.7,
                'P_bh_MPa': 18.0, 'dP_reservoir_MPa': 9.0, 'choked': False}

    monkeypatch.setattr(client_module.core, 'solve_flow_for_whp',
                        fake_solve_flow_for_whp)
    monkeypatch.setattr(client_module.core, 'coupled_model',
                        fake_coupled_model)
    monkeypatch.setattr(client_module, 'power_cycle',
                        type('PC', (), {'power_cycle_analysis':
                                        staticmethod(lambda raw: None)}))

    request = SuperhotRequest.from_dict({
        'well': {'depth_m': 3500},
        'operating': {'control': 'whp', 'target_whp_MPa': 10.0,
                      'hold': 'flow'},
        'decline': {'pressure_mode': 'linear_percent',
                    'pressure_rate_per_year': 1.0},
        'time': {'plant_lifetime_yr': 3, 'timesteps_per_year': 1},
        'solver': {'max_solve_points': 3},
    })
    profile = SuperhotWellboreClient(request).solve_profile()

    assert [kind for kind, _ in calls] == ['whp', 'flow', 'flow'], \
        'one pressure solve, then prescribed-flow solves'
    assert all(flow == 50.0 for kind, flow in calls if kind == 'flow'), \
        'the initial flow rate is held'
    assert profile.timesteps[-1].whp_MPa == pytest.approx(8.0), \
        'wellhead pressure responds to the decline'
    assert any('held constant' in note for note in profile.notes), \
        'holding the flow is reported'


def test_unknown_hold_mode_rejected():
    """An unknown hold mode is caught by validate()."""
    request = SuperhotRequest.from_dict({'operating': {'hold': 'temperature'}})
    with pytest.raises(ValueError):
        request.validate()
