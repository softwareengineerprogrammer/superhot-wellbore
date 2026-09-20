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
                        type('PC', (), {
                            'power_cycle_analysis':
                                staticmethod(lambda raw, params: None),
                            # The client derives the dry-steam work of every
                            # timestep from its wellhead pressure.
                            'dry_steam_specific_work':
                                staticmethod(lambda whp, params=None: 0.5)}))

    request = SuperhotRequest.from_dict({
        'well': {'depth_m': 3500},
        'operating': {'control': 'whp', 'target_whp_MPa': 10.0,
                      'hold': 'flow'},
        'decline': {'pressure_mode': 'linear_percent',
                    'pressure_rate_per_year': 1.0},
        'time': {'plant_lifetime_yr': 3, 'timesteps_per_year': 1},
        'solver': {'max_solve_points': 3},
        # With the pump switched off the held-flow solves go straight
        # to reservoir.coupled_model(); see test_pump.py for the
        # pumped route.
        'pump': {'mode': 'never'},
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


# ====================================================================
# PRESCRIBED INFLOW AND FLOW SERIES
# ====================================================================

def _pumped_dict(mdot, P_fz, h_fz_Jkg):
    """A canned pump-stage result echoing its inputs."""
    return {'success': True, 'mass_flow_kgs': mdot, 'whp_MPa': 2.0,
            'T_surface_C': 200.0, 'h_surface_MJkg': 0.85,
            'T_feedzone_C': 210.0, 'h_feedzone_MJkg': h_fz_Jkg * 1e-6,
            'P_bh_MPa': P_fz, 'dP_reservoir_MPa': 30.0 - P_fz,
            'choked': False, 'pumped': False, 'self_flowing': True,
            'wellhead_phase': 'single_phase_liquid', 'pump_flags': []}


def test_prescribed_inflow_bypasses_the_darcy_model(monkeypatch):
    """With inflow 'prescribed' the feedzone state comes from the table."""
    from superhot_wellbore.client import client as client_module

    seen = []

    def fake_solve_pumped_state(**kwargs):
        seen.append(kwargs)
        return _pumped_dict(kwargs['mdot'], kwargs['P_fz_MPa'],
                            kwargs['h_fz_Jkg'])

    def boom(*args, **kwargs):
        raise AssertionError('the Darcy model must not be called')

    monkeypatch.setattr(client_module.core, 'bottomhole_pressure', boom)
    monkeypatch.setattr(client_module.core, 'coupled_model', boom)
    monkeypatch.setattr(client_module.core, 'solve_flow_for_whp', boom)
    monkeypatch.setattr(client_module.pump_module, 'solve_pumped_state',
                        fake_solve_pumped_state)

    request = SuperhotRequest.from_dict({
        'reservoir': {'P_reservoir_MPa': 30.0, 'T_reservoir_C': 220.0,
                      'inflow': 'prescribed', 'transmissivity_md_m': None},
        'well': {'depth_m': 3000},
        'operating': {'control': 'flow', 'mass_flow_kgs': 60.0},
        'decline': {'feedzone_profile': [[0.0, 22.0, 0.95],
                                         [2.0, 20.0, 0.90]]},
        'time': {'plant_lifetime_yr': 2, 'timesteps_per_year': 1},
        'solver': {'max_solve_points': 0},
    })
    profile = SuperhotWellboreClient(request).solve_profile()

    assert len(seen) == 2, 'one pump-stage solve per timestep'
    assert [call['P_fz_MPa'] for call in seen] == [22.0, 20.0], \
        'feedzone pressure from the table'
    assert [call['h_fz_Jkg'] for call in seen] == \
        pytest.approx([0.95e6, 0.90e6]), 'feedzone enthalpy in J/kg'
    assert seen[0]['P_farfield_MPa'] == 30.0, 'far-field pressure'
    assert 200.0 < seen[0]['T_feedzone_C'] < 230.0, \
        'feedzone temperature derived from (P, h), got ' + \
        str(seen[0]['T_feedzone_C'])
    assert seen[0]['mdot'] == 60.0, 'the operating flow rate'
    assert profile.timesteps[1].P_bh_MPa == pytest.approx(20.0), \
        'translated feedzone pressure'
    assert profile.n_failed == 0, 'every timestep usable'


def test_mass_flow_profile_is_honoured_per_timestep(monkeypatch):
    """A tabulated flow rate is passed to every solve, interpolated."""
    from superhot_wellbore.client import client as client_module

    flows = []

    def fake_solve_pumped_state(**kwargs):
        flows.append(kwargs['mdot'])
        return _pumped_dict(kwargs['mdot'], kwargs['P_fz_MPa'],
                            kwargs['h_fz_Jkg'])

    monkeypatch.setattr(
        client_module.core, 'bottomhole_pressure',
        lambda mdot, P, T, rp=None, wp=None, max_iterations=3:
        {'P_bh_MPa': P - 0.1 * mdot, 'h_feedzone_Jkg': 0.9e6,
         'T_feedzone_C': T - 1.0, 'dP_reservoir_MPa': 0.1 * mdot})
    monkeypatch.setattr(client_module.pump_module, 'solve_pumped_state',
                        fake_solve_pumped_state)

    request = SuperhotRequest.from_dict({
        'reservoir': {'P_reservoir_MPa': 30.0, 'T_reservoir_C': 220.0},
        'well': {'depth_m': 3000},
        'operating': {'control': 'flow', 'mass_flow_kgs': 60.0},
        'decline': {'mass_flow_profile': [[0.0, 60.0], [4.0, 40.0]]},
        'solver': {'max_solve_points': 3},
    })
    profile = SuperhotWellboreClient(request).solve_profile(
        time_yr=[0.0, 1.0, 2.0, 3.0, 4.0])

    assert flows == pytest.approx([60.0, 50.0, 40.0]), \
        'interpolated flow rate at each solved index (0, 2, 4)'
    assert profile.mass_flow_kgs == pytest.approx(
        [60.0, 55.0, 50.0, 45.0, 40.0]), 'flow rate history'
    assert profile.n_solved == 3, 'three solves'
