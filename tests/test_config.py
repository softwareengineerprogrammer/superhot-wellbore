# -*- coding: utf-8 -*-
"""
Request Configuration
=====================

Covers the request objects that describe a run: the JSON round trip
that lets a study be re-run from its own output, the validation that
turns a mistyped key into a loud error instead of a silently ignored
setting, the integer depth grid the wellbore march requires, the
decline laws that stand in for a reservoir simulation and the time
vector that has to line up with the GEOPHIRES discretisation.

Author: superhot-wellbore GEOPHIRES client
"""

import json

import numpy as np
import pytest

from superhot_wellbore.geophires_client.config import (DeclineConfig,
                                                       SuperhotRequest,
                                                       TimeConfig, WellConfig)


# ====================================================================
# CONFIGURATION
# ====================================================================

def test_json_round_trip():
    """A request survives serialisation and reconstruction."""
    request = SuperhotRequest.from_dict({
        'name': 'roundtrip',
        'reservoir': {'T_reservoir_C': 500.0},
        'operating': {'control': 'flow', 'mass_flow_kgs': 40.0},
    })
    restored = SuperhotRequest.from_dict(
        json.loads(json.dumps(request.to_dict())))
    assert restored == request, 'JSON round trip'
    assert restored.reservoir.T_reservoir_C == pytest.approx(500.0,
                                                             abs=1e-9), \
        'nested value survives'


def test_unknown_section_rejected():
    """A misspelled section is an error, not a no-op."""
    with pytest.raises(ValueError):
        SuperhotRequest.from_dict({'reservior': {}})


def test_unknown_key_rejected():
    """A misspelled key is an error, not a silently ignored value."""
    with pytest.raises(ValueError):
        SuperhotRequest.from_dict({'reservoir': {'T_res': 500}})


def test_control_flow_needs_a_flow_rate():
    """Flow control without a flow rate is caught by validate()."""
    request = SuperhotRequest.from_dict({'operating': {'control': 'flow'}})
    with pytest.raises(ValueError):
        request.validate()


def test_unknown_rock_temperature_mode_rejected():
    """An unknown rock temperature mode is caught by validate()."""
    request = SuperhotRequest.from_dict({'rock_temperature': {'mode': 'bpd'}})
    with pytest.raises(ValueError):
        request.validate()


# ====================================================================
# INTEGER DEPTH GRID
# ====================================================================

def test_well_params_use_an_integer_grid():
    """Depth and step reach the wellbore model as ints."""
    # wellbore_physics.wellbore_simulate() marches the well with
    # range(0, depth + delta_z, delta_z), so both must be ints.
    parameters = WellConfig(depth_m=2100.0,
                            delta_z_m=10.0).to_well_params()
    assert isinstance(parameters['depth_m'], int), \
        'depth is an int, got ' + str(type(parameters['depth_m']))
    assert isinstance(parameters['delta_z_m'], int), \
        'step is an int, got ' + str(type(parameters['delta_z_m']))
    assert isinstance(parameters['diameter_m'], float), \
        'diameter stays a float'


def test_non_integral_step_rejected():
    """A fractional integration step cannot index the march."""
    config = WellConfig(delta_z_m=2.5)
    with pytest.raises(ValueError):
        config.validate()


def test_non_integral_depth_rejected():
    """A fractional depth cannot index the march."""
    config = WellConfig(depth_m=2100.5)
    with pytest.raises(ValueError):
        config.validate()


# ====================================================================
# DECLINE LAWS
# ====================================================================

TIMES = np.array([0.0, 10.0, 20.0])


def test_steady_state_is_flat():
    """The default decline leaves the reservoir untouched."""
    steady = DeclineConfig()
    assert np.allclose(steady.temperatures_C(TIMES, 450.0), 450.0), \
        'steady state is flat'
    assert steady.is_steady(), 'is_steady'


def test_linear_percent_decline():
    """1 %/yr for 10 yr takes 450 C down to 405 C."""
    percent = DeclineConfig(temperature_mode='linear_percent',
                            temperature_rate_per_year=1.0)
    assert percent.temperatures_C(TIMES, 450.0)[1] == \
        pytest.approx(405.0, abs=1e-9), '1 %/yr after 10 yr'


def test_linear_absolute_decline():
    """2 C/yr for 20 yr takes 450 C down to 410 C."""
    absolute = DeclineConfig(temperature_mode='linear_absolute',
                             temperature_rate_per_year=2.0)
    assert absolute.temperatures_C(TIMES, 450.0)[2] == \
        pytest.approx(410.0, abs=1e-9), '2 C/yr after 20 yr'


def test_exponential_decline():
    """The pressure decline is a true exponential, not a linear one."""
    exponential = DeclineConfig(pressure_mode='exponential',
                                pressure_rate_per_year=0.01)
    assert exponential.pressures_MPa(TIMES, 30.0)[1] == \
        pytest.approx(30.0 * np.exp(-0.1), abs=1e-6), \
        'exponential after 10 yr'


def test_explicit_table_interpolates():
    """A tabulated profile is interpolated between its rows."""
    explicit = DeclineConfig(
        temperature_mode='explicit',
        temperature_profile=[[0.0, 450.0], [20.0, 350.0]])
    assert explicit.temperatures_C(TIMES, 450.0)[1] == \
        pytest.approx(400.0, abs=1e-9), 'explicit table interpolates'


def test_minimum_temperature_floor_applied():
    """A steep decline is clamped at the minimum temperature."""
    floored = DeclineConfig(temperature_mode='linear_absolute',
                            temperature_rate_per_year=100.0,
                            min_temperature_C=300.0)
    assert floored.temperatures_C(TIMES, 450.0)[2] == \
        pytest.approx(300.0, abs=1e-9), 'floor applied'


def test_unknown_decline_mode_rejected():
    """An unknown decline mode is caught by validate()."""
    config = DeclineConfig(temperature_mode='guess')
    with pytest.raises(ValueError):
        config.validate()


# ====================================================================
# TIME VECTOR
# ====================================================================

def test_time_vector_matches_geophires():
    """The time vector reproduces the GEOPHIRES discretisation."""
    # Must match Reservoir.timevector of GEOPHIRES, which is
    # linspace(0, lifetime, timesteps_per_year * lifetime).
    config = TimeConfig(plant_lifetime_yr=30, timesteps_per_year=4)
    times = config.time_vector_yr()
    reference = np.linspace(0, 30, 4 * 30)
    assert np.allclose(times, reference), \
        'matches the GEOPHIRES discretisation'
    assert len(times) == 120, 'timestep count'
    assert times[0] == pytest.approx(0.0, abs=1e-9), 'first time'
    assert times[-1] == pytest.approx(30.0, abs=1e-9), 'last time'


def test_zero_lifetime_rejected():
    """A plant that never runs is caught by validate()."""
    config = TimeConfig(plant_lifetime_yr=0)
    with pytest.raises(ValueError):
        config.validate()
