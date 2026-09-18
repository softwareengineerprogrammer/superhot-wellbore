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

import numpy as np
import pytest

from superhot_wellbore.client.results import interpolate_timesteps


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
    assert interpolated.eta_utilization == pytest.approx(0.4, abs=1e-9), \
        'power cycle metrics interpolated'
    assert interpolated.cycle == 'flash', 'cycle taken from nearest solve'


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


def test_failed_solve_is_interpolated_and_flagged():
    """A failed solve between good ones is filled in, but still visible."""
    from superhot_wellbore.client.results import TimestepResult

    times = [0.0, 1.0, 2.0]
    good = dict(mass_flow_kgs=70.0, whp_MPa=10.0, T_wellhead_C=300.0,
                h_wellhead_MJkg=2.7, T_feedzone_C=390.0,
                h_feedzone_MJkg=2.8, P_bh_MPa=18.0, dP_reservoir_MPa=12.0,
                power_MWe=30.0, converged=True, success=True, solved=True)
    solved = {
        0: TimestepResult(time_yr=0.0, **good),
        1: TimestepResult(time_yr=1.0, solved=True, success=False,
                          message='did not reach the surface'),
        2: TimestepResult(time_yr=2.0, **dict(good, mass_flow_kgs=60.0)),
    }
    filled = interpolate_timesteps(times, solved)

    failed = filled[1]
    assert failed.success and failed.solved, 'failed solve becomes usable'
    assert failed.mass_flow_kgs == pytest.approx(65.0, abs=1e-9), \
        'failed solve interpolated between neighbours'
    assert 'did not reach the surface' in failed.message, \
        'original failure stays visible'
    assert all(ts.success for ts in filled), 'every timestep is usable'


def test_failed_solve_without_any_success_stays_failed():
    """With nothing to interpolate from, a failure remains a failure."""
    from superhot_wellbore.client.results import TimestepResult

    solved = {0: TimestepResult(time_yr=0.0, solved=True, success=False,
                                message='boom')}
    filled = interpolate_timesteps([0.0, 1.0], solved)
    assert not filled[0].success and filled[0].message == 'boom', \
        'failure preserved'
    assert not filled[1].success, 'unsolved step has nothing to use'


# ====================================================================
# PUMP FIELDS
# ====================================================================

def _pump_step(t, **overrides):
    """A successful solved timestep with pump-stage fields."""
    from superhot_wellbore.client.results import TimestepResult

    values = dict(mass_flow_kgs=60.0, whp_MPa=1.6, T_wellhead_C=191.0,
                  h_wellhead_MJkg=0.81, T_feedzone_C=199.0,
                  h_feedzone_MJkg=0.85, P_bh_MPa=41.0,
                  dP_reservoir_MPa=8.0, power_MWe=3.0,
                  pumped=True, pump_depth_m=600.0, P_pump_intake_MPa=1.7,
                  T_pump_intake_C=191.8, dP_pump_MPa=5.0,
                  pump_power_MWe=0.4, self_flow_whp_MPa=float('nan'),
                  self_flowing=False, wellhead_phase='single_phase_liquid',
                  wellhead_quality=float('nan'), dry_steam_work_MJkg=0.50,
                  pump_flags=['depth_limit'],
                  converged=True, success=True, solved=True)
    values.update(overrides)
    return TimestepResult(time_yr=t, **values)


def test_pump_fields_are_interpolated():
    """The numeric pump fields lie on the line between the solves."""
    solved = {0: _pump_step(0.0),
              2: _pump_step(2.0, pump_depth_m=800.0, dP_pump_MPa=7.0,
                            pump_power_MWe=0.6, T_pump_intake_C=193.8,
                            P_pump_intake_MPa=1.9, dry_steam_work_MJkg=0.52)}
    filled = interpolate_timesteps([0.0, 1.0, 2.0], solved)
    middle = filled[1]
    assert not middle.solved and middle.success, 'interpolated step'
    assert middle.pump_depth_m == pytest.approx(700.0), 'pump depth'
    assert middle.dP_pump_MPa == pytest.approx(6.0), 'pump rise'
    assert middle.pump_power_MWe == pytest.approx(0.5), 'pump power'
    assert middle.T_pump_intake_C == pytest.approx(192.8), 'intake T'
    assert middle.P_pump_intake_MPa == pytest.approx(1.8), 'intake P'
    assert middle.dry_steam_work_MJkg == pytest.approx(0.51), 'dry steam'
    assert np.isnan(middle.self_flow_whp_MPa), 'NaN stays NaN'


def test_pump_flags_are_copied_from_the_nearest_solve():
    """Booleans, the phase label and the flag list follow the nearest solve."""
    solved = {0: _pump_step(0.0),
              3: _pump_step(3.0, pumped=False, self_flowing=True,
                            wellhead_phase='two_phase',
                            wellhead_quality=0.06, pump_flags=[],
                            pump_depth_m=0.0, dP_pump_MPa=0.0,
                            pump_power_MWe=0.0, self_flow_whp_MPa=1.8)}
    filled = interpolate_timesteps([0.0, 1.0, 2.0, 3.0], solved)
    near_first, near_last = filled[1], filled[2]
    assert near_first.pumped and not near_first.self_flowing, \
        'flags of the first solve'
    assert near_first.wellhead_phase == 'single_phase_liquid', 'phase'
    assert near_first.pump_flags == ['depth_limit'], 'flags copied'
    assert near_first.pump_flags is not solved[0].pump_flags, \
        'the list is copied, not shared'
    assert not near_last.pumped and near_last.self_flowing, \
        'flags of the last solve'
    assert near_last.wellhead_phase == 'two_phase', 'phase'
    assert near_last.pump_flags == [], 'no flags'


def test_wellhead_quality_interpolates_towards_the_liquid_limit():
    """Quality runs towards x = 0 near a two-phase solve, NaN otherwise."""
    solved = {0: _pump_step(0.0, pumped=False, self_flowing=True,
                            wellhead_phase='two_phase',
                            wellhead_quality=0.06),
              3: _pump_step(4.0, pumped=False, self_flowing=True,
                            wellhead_phase='single_phase_liquid',
                            wellhead_quality=float('nan'))}
    filled = interpolate_timesteps([0.0, 1.0, 3.0, 4.0], solved)
    assert filled[1].wellhead_phase == 'two_phase', 'nearest is two-phase'
    assert filled[1].wellhead_quality == pytest.approx(0.045), \
        'quality on the line towards x = 0'
    assert filled[2].wellhead_phase == 'single_phase_liquid', \
        'nearest is liquid'
    assert np.isnan(filled[2].wellhead_quality), \
        'no quality for a single-phase wellhead'


def test_profile_pump_accessors_and_summary():
    """ProductionProfile exposes the pump series and headline numbers."""
    from superhot_wellbore.client.results import ProductionProfile

    steps = [_pump_step(0.0),
             _pump_step(1.0, pumped=False, self_flowing=True,
                        wellhead_phase='two_phase', wellhead_quality=0.05,
                        pump_flags=[], pump_depth_m=0.0,
                        pump_power_MWe=0.0, self_flow_whp_MPa=1.8),
             _pump_step(2.0, success=False, pumped=False,
                        pump_power_MWe=0.0, pump_depth_m=900.0)]
    profile = ProductionProfile(depth_m=5000.0, timesteps=steps)
    assert profile.pump_power_MWe == [0.4, 0.0, 0.0], 'pump power series'
    assert profile.pump_depth_m == [600.0, 0.0, 900.0], 'pump depth series'
    assert profile.wellhead_phase[:2] == ['single_phase_liquid',
                                          'two_phase'], 'phase series'
    assert profile.wellhead_quality[1] == pytest.approx(0.05), 'quality'
    assert profile.dry_steam_work_MJkg[0] == pytest.approx(0.50), 'dry'
    assert profile.self_flow_whp_MPa[1] == pytest.approx(1.8), 'self flow'
    assert profile.any_pumped, 'pumped somewhere'
    assert profile.self_flowing_fraction == pytest.approx(0.5), \
        'one of two usable timesteps self-flows'
    assert profile.dominant_wellhead_phase in ('single_phase_liquid',
                                               'two_phase'), 'a tie'
    summary = profile.summary()
    assert summary['any_pumped'] is True, 'summary flag'
    assert summary['max_pump_depth_m'] == pytest.approx(600.0), \
        'max depth over usable timesteps only'
    assert summary['mean_pump_power_MWe'] == pytest.approx(0.2), 'mean'
    assert summary['pump_flags'] == ['depth_limit'], 'flags collected'
    assert summary['initial_self_flow_whp_MPa'] is None, 'NaN -> None'
    series = profile.to_dict()['series']
    assert series['pump_power_MWe'] == [0.4, 0.0, 0.0], 'serialised'
    assert series['wellhead_phase'][1] == 'two_phase', 'serialised phase'
    assert profile.timesteps[0].to_dict()['pump_flags'] == ['depth_limit'], \
        'timestep flags serialised'
