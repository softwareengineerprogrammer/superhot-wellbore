# -*- coding: utf-8 -*-
"""
Self-Test of the GEOPHIRES Client
==================================

Checks the parts of the client that can go silently wrong: unit
conversions across the GEOPHIRES boundary, the decline laws, the time
vector, the integer depth grid the wellbore model requires, the
interpolation between coupled-model solves and the exported file
formats.

Run it with::

    python -m geophires_client selftest          # fast checks only
    python -m geophires_client selftest --full   # plus a real solve

The fast checks do no thermodynamics and finish instantly. The full
run additionally solves the coupled reservoir-wellbore model once and
verifies that the solution is physically ordered.

Author: superhot-wellbore GEOPHIRES client
"""

import json
import os
import tempfile

import numpy as np

from geophires_client import export, units
from geophires_client.client import SuperhotWellboreClient
from geophires_client.config import (DeclineConfig, SuperhotRequest, TimeConfig,
                     WellConfig)
from geophires_client.results import (ProductionProfile, TimestepResult,
                      interpolate_timesteps)


# ====================================================================
# HARNESS
# ====================================================================

class _Checks:
    """Minimal check collector that prints as it goes."""

    def __init__(self):
        self.passed = 0
        self.failed = []

    def check(self, label, condition, detail=''):
        """Record one boolean check."""
        if condition:
            self.passed += 1
            print(f'  ok    {label}')
        else:
            self.failed.append(label)
            print(f'  FAIL  {label}'
                  + (f'  ({detail})' if detail else ''))

    def close(self, label, expected, actual, tolerance=1e-9):
        """Record a numeric closeness check."""
        ok = abs(float(expected) - float(actual)) <= tolerance
        self.check(label, ok, f'expected {expected}, got {actual}')

    def raises(self, label, exception, function, *args, **kwargs):
        """Record that a call raises the expected exception."""
        try:
            function(*args, **kwargs)
        except exception:
            self.check(label, True)
            return
        except Exception as exc:
            self.check(label, False,
                       f'raised {type(exc).__name__} instead')
            return
        self.check(label, False, 'no exception raised')


# ====================================================================
# FAST CHECKS
# ====================================================================

def _check_units(checks):
    print('Unit conversions')
    checks.close('MPa to kPa', 10000.0, units.mpa_to_kpa(10.0))
    checks.close('kPa to MPa', 10.0, units.kpa_to_mpa(10000.0))
    checks.close('m to km', 3.5, units.m_to_km(3500.0))
    checks.close('MJ/kg to kJ/kg', 2770.0,
                 units.mj_per_kg_to_kj_per_kg(2.77))
    checks.close('m to inch', 8.5433, units.m_to_inch(0.217), 1e-4)
    checks.close('gradient', 125.714,
                 units.gradient_C_per_km(450.0, 10.0, 3500.0), 1e-3)
    checks.check('None passes through',
                 units.mpa_to_kpa(None) is None)


def _check_config(checks):
    print('Configuration')
    request = SuperhotRequest.from_dict({
        'name': 'roundtrip',
        'reservoir': {'T_reservoir_C': 500.0},
        'operating': {'control': 'flow', 'mass_flow_kgs': 40.0},
    })
    restored = SuperhotRequest.from_dict(
        json.loads(json.dumps(request.to_dict())))
    checks.check('JSON round trip', restored == request)
    checks.close('nested value survives', 500.0,
                 restored.reservoir.T_reservoir_C)

    checks.raises('unknown section rejected', ValueError,
                  SuperhotRequest.from_dict, {'reservior': {}})
    checks.raises('unknown key rejected', ValueError,
                  SuperhotRequest.from_dict,
                  {'reservoir': {'T_res': 500}})
    checks.raises("control 'flow' needs a flow rate", ValueError,
                  SuperhotRequest.from_dict(
                      {'operating': {'control': 'flow'}}).validate)
    checks.raises('unknown rock temperature mode rejected', ValueError,
                  SuperhotRequest.from_dict(
                      {'rock_temperature': {'mode': 'bpd'}}).validate)


def _check_integer_grid(checks):
    print('Integer depth grid')
    # wellbore_physics.wellbore_simulate() marches the well with
    # range(0, depth + delta_z, delta_z), so both must be ints.
    parameters = WellConfig(depth_m=2100.0,
                            delta_z_m=10.0).to_well_params()
    checks.check('depth is an int',
                 isinstance(parameters['depth_m'], int),
                 str(type(parameters['depth_m'])))
    checks.check('step is an int',
                 isinstance(parameters['delta_z_m'], int),
                 str(type(parameters['delta_z_m'])))
    checks.check('diameter stays a float',
                 isinstance(parameters['diameter_m'], float))
    checks.raises('non-integral step rejected', ValueError,
                  WellConfig(delta_z_m=2.5).validate)
    checks.raises('non-integral depth rejected', ValueError,
                  WellConfig(depth_m=2100.5).validate)


def _check_decline(checks):
    print('Decline laws')
    times = np.array([0.0, 10.0, 20.0])

    steady = DeclineConfig()
    checks.check('steady state is flat',
                 np.allclose(steady.temperatures_C(times, 450.0),
                             450.0))
    checks.check('is_steady', steady.is_steady())

    percent = DeclineConfig(temperature_mode='linear_percent',
                            temperature_rate_per_year=1.0)
    checks.close('1 %/yr after 10 yr', 405.0,
                 percent.temperatures_C(times, 450.0)[1])

    absolute = DeclineConfig(temperature_mode='linear_absolute',
                             temperature_rate_per_year=2.0)
    checks.close('2 C/yr after 20 yr', 410.0,
                 absolute.temperatures_C(times, 450.0)[2])

    exponential = DeclineConfig(pressure_mode='exponential',
                                pressure_rate_per_year=0.01)
    checks.close('exponential after 10 yr', 30.0 * np.exp(-0.1),
                 exponential.pressures_MPa(times, 30.0)[1], 1e-6)

    explicit = DeclineConfig(
        temperature_mode='explicit',
        temperature_profile=[[0.0, 450.0], [20.0, 350.0]])
    checks.close('explicit table interpolates', 400.0,
                 explicit.temperatures_C(times, 450.0)[1])

    floored = DeclineConfig(temperature_mode='linear_absolute',
                            temperature_rate_per_year=100.0,
                            min_temperature_C=300.0)
    checks.close('floor applied', 300.0,
                 floored.temperatures_C(times, 450.0)[2])

    checks.raises('unknown decline mode rejected', ValueError,
                  DeclineConfig(temperature_mode='guess').validate)


def _check_time_vector(checks):
    print('Time vector')
    # Must match Reservoir.timevector of GEOPHIRES, which is
    # linspace(0, lifetime, timesteps_per_year * lifetime).
    config = TimeConfig(plant_lifetime_yr=30, timesteps_per_year=4)
    times = config.time_vector_yr()
    reference = np.linspace(0, 30, 4 * 30)
    checks.check('matches the GEOPHIRES discretisation',
                 np.allclose(times, reference))
    checks.close('timestep count', 120, len(times))
    checks.close('first time', 0.0, times[0])
    checks.close('last time', 30.0, times[-1])
    checks.raises('zero lifetime rejected', ValueError,
                  TimeConfig(plant_lifetime_yr=0).validate)


def _synthetic_profile():
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
            converged=True, success=True, solved=True)

    timesteps = interpolate_timesteps(times, solved)
    return ProductionProfile(request=request, depth_m=3500.0,
                             timesteps=timesteps)


def _check_interpolation(checks):
    print('Interpolation between solves')
    profile = _synthetic_profile()
    checks.close('all timesteps present', 4, profile.n_timesteps)
    checks.close('two solves', 2, profile.n_solved)
    checks.close('no failures', 0, profile.n_failed)

    interpolated = profile.timesteps[1]
    checks.check('interpolated step is flagged',
                 not interpolated.solved and interpolated.success)
    checks.close('flow rate interpolated',
                 72.0 - 2.0 / 3.0, interpolated.mass_flow_kgs, 1e-6)
    checks.check('wellhead below feedzone',
                 interpolated.T_wellhead_C
                 < interpolated.T_feedzone_C)

    empty = interpolate_timesteps([0.0, 1.0], {})
    checks.check('no solves means no success',
                 not any(ts.success for ts in empty))


def _check_geophires_mapping(checks):
    print('GEOPHIRES parameter mapping')
    profile = _synthetic_profile()

    wellhead = profile.to_geophires_parameters('wellhead')
    checks.close('reservoir model is 5', 5,
                 wellhead['Reservoir Model'])
    checks.close('depth in km', 3.5, wellhead['Reservoir Depth'])
    checks.close('wellhead pressure in kPa', 10000.0,
                 wellhead['Production Wellhead Pressure'])
    checks.close('hydrostatic pressure in kPa', 30000.0,
                 wellhead['Reservoir Hydrostatic Pressure'])
    checks.close('diameter in inch', 8.5433,
                 wellhead['Production Well Diameter'], 1e-4)
    checks.close('flow rate in kg/s', 72.0,
                 wellhead['Production Flow Rate per Well'])
    checks.check("Ramey's model disabled",
                 wellhead['Ramey Production Wellbore Model'] is False)
    checks.close('no double-counted temperature drop', 0.0,
                 wellhead['Production Wellbore Temperature Drop'])
    checks.check('maximum temperature admits the reservoir',
                 wellhead['Maximum Temperature']
                 >= profile.request.reservoir.T_reservoir_C)

    feedzone = profile.to_geophires_parameters('feedzone')
    checks.close('feedzone drop is clamped to 50 C', 50.0,
                 feedzone['Production Wellbore Temperature Drop'])
    checks.check('clamping is reported',
                 any('clamped' in note for note in profile.notes))

    checks.check('profile rows use the wellhead temperature',
                 profile.temperature_profile_rows('wellhead')[0][1]
                 < profile.temperature_profile_rows('feedzone')[0][1])
    checks.raises('unknown profile temperature rejected', ValueError,
                  profile.temperature_profile_rows, 'bottomhole')


def _check_export(checks):
    print('Exported files')
    profile = _synthetic_profile()

    with tempfile.TemporaryDirectory() as directory:
        paths = export.export_all(profile, directory)

        with open(paths['temperature_profile'], encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        data_lines = [line for line in lines
                      if not line.startswith('#')]
        checks.close('one profile row per timestep',
                     profile.n_timesteps, len(data_lines))
        first = data_lines[0].split(',')
        checks.check('profile row is time and temperature',
                     len(first) == 2
                     and float(first[0]) == 0.0
                     and abs(float(first[1]) - 317.0) < 1e-6)

        with open(paths['geophires_input'], encoding='utf-8') as f:
            deck = [line.strip() for line in f if line.strip()]
        parameters = {}
        for line in deck:
            if line.startswith(('#', '--', '*')):
                continue
            fields = line.split(',')
            if len(fields) >= 2:
                parameters[fields[0].strip()] = fields[1].strip()
        checks.check('deck selects the profile reservoir model',
                     parameters.get('Reservoir Model') == '5')
        checks.check('deck points at the profile file',
                     parameters.get('Reservoir Output File Name')
                     == os.path.basename(
                         paths['temperature_profile']))
        checks.check('deck writes booleans GEOPHIRES understands',
                     parameters.get('Ramey Production Wellbore Model')
                     == 'False')

        with open(paths['profile_json'], encoding='utf-8') as f:
            document = json.load(f)
        checks.check('JSON result carries the request',
                     document['request']['name'] == 'synthetic')
        checks.check('JSON result carries every timestep',
                     len(document['timesteps'])
                     == profile.n_timesteps)
        checks.check('JSON result carries the series',
                     len(document['series']['production_temperature_C'])
                     == profile.n_timesteps)


def _check_depth_resolution(checks):
    print('Well depth resolution')
    client = SuperhotWellboreClient(SuperhotRequest.from_dict(
        {'reservoir': {'P_reservoir_MPa': 30.0}}))
    checks.close('derived from 30 MPa', 3500.0, client.depth_m)
    checks.check('derivation is reported',
                 any('derived' in note for note in client.notes))

    snapped = SuperhotWellboreClient(SuperhotRequest.from_dict(
        {'well': {'depth_m': 2106, 'delta_z_m': 10}}))
    checks.close('snapped to the integration step', 2110.0,
                 snapped.depth_m)
    checks.check('snapping is reported',
                 any('snapped' in note for note in snapped.notes))


# ====================================================================
# SLOW CHECKS
# ====================================================================

def _check_coupled_solve(checks):
    print('Coupled model solve (slow)')
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

    checks.check('a solution was found', first is not None)
    if first is None:
        return

    checks.check('flow rate is positive', first.mass_flow_kgs > 0,
                 str(first.mass_flow_kgs))
    checks.check('wellhead pressure meets the target',
                 abs(first.whp_MPa - 10.0) < 0.5,
                 str(first.whp_MPa))
    checks.check('temperatures are ordered from reservoir to surface',
                 (first.T_wellhead_C < first.T_feedzone_C
                  < request.reservoir.T_reservoir_C),
                 f'{first.T_wellhead_C} < {first.T_feedzone_C} < '
                 f'{request.reservoir.T_reservoir_C}')
    checks.check('drawdown is positive',
                 first.dP_reservoir_MPa > 0,
                 str(first.dP_reservoir_MPa))
    checks.check('bottomhole pressure sits below the reservoir',
                 first.P_bh_MPa
                 < request.reservoir.P_reservoir_MPa)
    checks.check('every timestep is usable', profile.n_failed == 0)
    checks.check('the power diagnostic is available',
                 np.isfinite(first.power_MWe), str(first.power_MWe))


# ====================================================================
# ENTRY POINT
# ====================================================================

def run_selftest(quick=True):
    """
    Run the self-test.

    Parameters
    ----------
    quick : bool
        Skip the slow coupled-model solve.

    Returns
    -------
    int
        0 if every check passed, 1 otherwise.
    """
    checks = _Checks()

    _check_units(checks)
    _check_config(checks)
    _check_integer_grid(checks)
    _check_decline(checks)
    _check_time_vector(checks)
    _check_interpolation(checks)
    _check_geophires_mapping(checks)
    _check_export(checks)
    _check_depth_resolution(checks)

    if not quick:
        _check_coupled_solve(checks)

    print('')
    print(f'{checks.passed} checks passed, {len(checks.failed)} '
          f'failed')
    for label in checks.failed:
        print(f'  failed: {label}')
    if quick:
        print('Run with --full to include the coupled-model solve.')
    return 1 if checks.failed else 0


if __name__ == '__main__':
    run_selftest()