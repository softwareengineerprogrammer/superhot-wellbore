# -*- coding: utf-8 -*-
"""
Shared Test Fixtures
====================

Holds the synthetic production profile used by the results and the
export tests. Building it here keeps the physics out of those tests:
the timestep values are written by hand, so a failure there points at
the bookkeeping (interpolation, GEOPHIRES parameter mapping, file
writing) rather than at the coupled reservoir-wellbore solve.

Author: superhot-wellbore GEOPHIRES client
"""

import pytest

from superhot_wellbore.client.config import SuperhotRequest
from superhot_wellbore.client.results import (ProductionProfile,
                                                        TimestepResult,
                                                        interpolate_timesteps)


@pytest.fixture
def synthetic_profile():
    """Build a profile without running the physics."""
    request = SuperhotRequest.from_dict({
        'name': 'synthetic',
        'reservoir': {'P_reservoir_MPa': 30.0,
                      'T_reservoir_C': 450.0},
        'well': {'depth_m': 3500},
        'time': {'plant_lifetime_yr': 2, 'timesteps_per_year': 2},
    })
    times = request.time.time_vector_yr()

    solved = {}
    for index in (0, len(times) - 1):
        fraction = index / (len(times) - 1)
        solved[index] = TimestepResult(
            time_yr=float(times[index]),
            P_reservoir_MPa=30.0 - fraction,
            T_reservoir_C=450.0 - 10.0 * fraction,
            mass_flow_kgs=72.0 - 2.0 * fraction,
            whp_MPa=10.0,
            T_wellhead_C=317.0 - 6.0 * fraction,
            h_wellhead_MJkg=2.77,
            T_feedzone_C=391.0 - 8.0 * fraction,
            h_feedzone_MJkg=2.82,
            P_bh_MPa=18.5,
            dP_reservoir_MPa=11.5,
            power_MWe=32.0,
            cycle='flash',
            eta_utilization=0.4,
            exergy_rate_MW=80.0,
            converged=True, success=True, solved=True)

    timesteps = interpolate_timesteps(times, solved)
    return ProductionProfile(request=request, depth_m=3500.0,
                             timesteps=timesteps)
