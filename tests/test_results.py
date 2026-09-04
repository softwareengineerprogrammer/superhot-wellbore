# -*- coding: utf-8 -*-
"""
Production Profile Results
==========================

Two things happen between the coupled solves and GEOPHIRES, and both
are easy to get wrong without anything looking broken. First the
solved timesteps are interpolated onto the full time vector, so the
filled-in steps must be marked as such and must stay physically
ordered. Second the profile is translated into GEOPHIRES parameters,
where every value changes unit and where a temperature drop that
GEOPHIRES also applies internally would be counted twice.

Author: superhot-wellbore GEOPHIRES client
"""

import pytest

from superhot_wellbore.geophires_client.results import interpolate_timesteps


# ====================================================================
# INTERPOLATION BETWEEN SOLVES
# ====================================================================

def test_interpolation_fills_the_time_vector(synthetic_profile):
    """Two solves are expanded to every timestep."""
    assert synthetic_profile.n_timesteps == 4, 'all timesteps present'
    assert synthetic_profile.n_solved == 2, 'two solves'
    assert synthetic_profile.n_failed == 0, 'no failures'


def test_interpolated_step_is_flagged(synthetic_profile):
    """A filled-in step is usable but not marked as solved."""
    interpolated = synthetic_profile.timesteps[1]
    assert not interpolated.solved and interpolated.success, \
        'interpolated step is flagged'


def test_interpolated_values(synthetic_profile):
    """The filled-in step lies on the line between the solves."""
    interpolated = synthetic_profile.timesteps[1]
    assert interpolated.mass_flow_kgs == \
        pytest.approx(72.0 - 2.0 / 3.0, abs=1e-6), \
        'flow rate interpolated'
    assert interpolated.T_wellhead_C < interpolated.T_feedzone_C, \
        'wellhead below feedzone'


def test_no_solves_means_no_success():
    """Without a single solve nothing is reported as usable."""
    empty = interpolate_timesteps([0.0, 1.0], {})
    assert not any(ts.success for ts in empty), \
        'no solves means no success'


# ====================================================================
# GEOPHIRES PARAMETER MAPPING
# ====================================================================

def test_geophires_parameters_wellhead(synthetic_profile):
    """The wellhead mapping hands GEOPHIRES the right units."""
    wellhead = synthetic_profile.to_geophires_parameters('wellhead')
    assert wellhead['Reservoir Model'] == 5, 'reservoir model is 5'
    assert wellhead['Reservoir Depth'] == pytest.approx(3.5, abs=1e-9), \
        'depth in km'
    assert wellhead['Production Wellhead Pressure'] == \
        pytest.approx(10000.0, abs=1e-9), 'wellhead pressure in kPa'
    assert wellhead['Reservoir Hydrostatic Pressure'] == \
        pytest.approx(30000.0, abs=1e-9), 'hydrostatic pressure in kPa'
    assert wellhead['Production Well Diameter'] == \
        pytest.approx(8.5433, abs=1e-4), 'diameter in inch'
    assert wellhead['Production Flow Rate per Well'] == \
        pytest.approx(72.0, abs=1e-9), 'flow rate in kg/s'


def test_geophires_parameters_avoid_double_counting(synthetic_profile):
    """Our wellbore model replaces the one inside GEOPHIRES."""
    wellhead = synthetic_profile.to_geophires_parameters('wellhead')
    assert wellhead['Ramey Production Wellbore Model'] is False, \
        "Ramey's model disabled"
    assert wellhead['Production Wellbore Temperature Drop'] == \
        pytest.approx(0.0, abs=1e-9), 'no double-counted temperature drop'
    assert wellhead['Maximum Temperature'] >= \
        synthetic_profile.request.reservoir.T_reservoir_C, \
        'maximum temperature admits the reservoir'


def test_feedzone_temperature_drop_is_clamped(synthetic_profile):
    """GEOPHIRES caps the drop at 50 C, and we say so."""
    feedzone = synthetic_profile.to_geophires_parameters('feedzone')
    assert feedzone['Production Wellbore Temperature Drop'] == \
        pytest.approx(50.0, abs=1e-9), 'feedzone drop is clamped to 50 C'
    assert any('clamped' in note for note in synthetic_profile.notes), \
        'clamping is reported'


def test_profile_rows_use_the_requested_temperature(synthetic_profile):
    """The exported rows follow the chosen reference point."""
    assert synthetic_profile.temperature_profile_rows('wellhead')[0][1] < \
        synthetic_profile.temperature_profile_rows('feedzone')[0][1], \
        'profile rows use the wellhead temperature'


def test_unknown_profile_temperature_rejected(synthetic_profile):
    """Only the two defined reference points are accepted."""
    with pytest.raises(ValueError):
        synthetic_profile.temperature_profile_rows('bottomhole')
