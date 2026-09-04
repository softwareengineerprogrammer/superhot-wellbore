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

from superhot_wellbore.geophires_client.client import SuperhotWellboreClient
from superhot_wellbore.geophires_client.config import SuperhotRequest


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
