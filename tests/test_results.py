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


def test_dry_steam_work_is_not_interpolated():
    """A supercritical solve must not leave its neighbours without the work.

    dry_steam_specific_work is undefined above the critical pressure, so
    interpolating it linearly would spread that NaN into sub-critical
    timesteps, where the flash plant needs the number. It is derived from
    each timestep's own wellhead pressure instead (SuperhotClient), so the
    interpolator must not carry a value at all.
    """
    from superhot_wellbore.client.results import TimestepResult

    assert 'dry_steam_work_MJkg' not in TimestepResult.INTERPOLATED_FIELDS, \
        'dry-steam work is derived, never interpolated'

    supercritical = TimestepResult(
        time_yr=0.0, solved=True, success=True, whp_MPa=30.0,
        wellhead_phase='supercritical', dry_steam_work_MJkg=float('nan'))
    vapour = TimestepResult(
        time_yr=2.0, solved=True, success=True, whp_MPa=10.0,
        wellhead_phase='single_phase_vapor', dry_steam_work_MJkg=0.59)
    profile = interpolate_timesteps([0.0, 1.0, 2.0],
                                    {0: supercritical, 2: vapour})
    middle = profile[1]
    assert middle.success and not middle.solved, 'the middle step is filled in'
    assert np.isfinite(middle.whp_MPa), 'pressure still interpolates'


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


def _good_step(t, **overrides):
    """A successful self-flowing solved timestep."""
    from superhot_wellbore.client.results import TimestepResult

    values = dict(mass_flow_kgs=70.0, whp_MPa=10.0, T_wellhead_C=300.0,
                  h_wellhead_MJkg=2.7, T_feedzone_C=390.0,
                  h_feedzone_MJkg=2.8, P_bh_MPa=18.0, dP_reservoir_MPa=12.0,
                  power_MWe=30.0, pumped=False, self_flowing=True,
                  converged=True, success=True, solved=True)
    values.update(overrides)
    return TimestepResult(time_yr=t, **values)


def test_failed_solve_keeps_its_failure():
    """A failed solve late in the history stays failed and is never rebuilt.

    A well that stops delivering, or a pump outside its envelope, at the
    end of the profile must reach the caller as such: GEOPHIRES keys on
    success and pump_flags, so masking the failure with the neighbouring
    good state would report a well that kept flowing.
    """
    from superhot_wellbore.client.results import (ProductionProfile,
                                                  TimestepResult)

    times = [float(t) for t in range(31)]
    last_good = dict(mass_flow_kgs=60.0, whp_MPa=8.0, T_wellhead_C=290.0,
                     P_bh_MPa=16.0, dP_reservoir_MPa=14.0, power_MWe=25.0)
    solved = {
        0: _good_step(0.0),
        15: _good_step(15.0, **last_good),
        30: TimestepResult(time_yr=30.0, solved=True, success=False,
                           pumped=True, self_flowing=False,
                           pump_depth_m=1800.0,
                           message='coupled model failed at this state',
                           pump_flags=['pump_outside_envelope']),
    }
    filled = interpolate_timesteps(times, solved)

    failed = filled[30]
    assert failed is solved[30], 'the failed solve is returned as solved'
    assert failed.solved and not failed.success, 'failure preserved'
    assert failed.message == 'coupled model failed at this state', \
        'its own message'
    assert failed.pump_flags == ['pump_outside_envelope'], 'its own flags'
    assert failed.pumped and not failed.self_flowing, 'its own pump state'
    assert failed.pump_depth_m == pytest.approx(1800.0), 'its own depth'
    assert np.isnan(failed.mass_flow_kgs), 'no flow rate invented'
    assert not failed.interpolated_across_failure, 'a solve is not filled in'

    held = filled[29]
    assert held.success and not held.solved, 'filled-in step stays usable'
    assert held.interpolated_across_failure, 'nearest solve failed'
    assert 't = 30.00 yr' in held.message, 'the failed solve is named'
    for attribute in TimestepResult.INTERPOLATED_FIELDS:
        expected = getattr(solved[15], attribute)
        actual = getattr(held, attribute)
        if np.isnan(expected):
            assert np.isnan(actual), f'{attribute} stays NaN'
        else:
            assert actual == pytest.approx(expected), \
                f'{attribute} flat-held from the last good solve'
    assert held.pump_flags == [] and not held.pumped and held.self_flowing, \
        'flags follow the nearest successful solve, not the failure'

    early = filled[7]
    assert not early.interpolated_across_failure, 'nearest solve succeeded'
    assert early.message == 'interpolated between coupled-model solves', \
        'plain interpolation message'
    assert early.mass_flow_kgs == pytest.approx(70.0 - 10.0 * 7.0 / 15.0), \
        'interpolated between the two good solves'
    assert not any(ts.interpolated_across_failure for ts in filled[:23]), \
        'steps nearer to t = 15 than to t = 30 are plain interpolations'
    assert all(ts.interpolated_across_failure for ts in filled[23:30]), \
        'steps nearer to the failed solve carry the marker'

    profile = ProductionProfile(depth_m=3500.0, timesteps=filled)
    assert profile.n_failed == 1, 'the failure is counted'
    assert profile.n_solved == 3, 'the failure still counts as a solve'
    assert not profile.any_pumped, 'the failed pumped solve is not usable'
    assert profile.summary()['pump_flags'] == [], \
        'flags of the failed solve are not reported as usable'


def test_interpolated_across_failure_is_a_plain_field():
    """The marker is never interpolated or copied from a neighbour."""
    from superhot_wellbore.client.results import TimestepResult

    assert 'interpolated_across_failure' not in \
        TimestepResult.INTERPOLATED_FIELDS, 'not interpolated'
    assert 'interpolated_across_failure' not in \
        TimestepResult.COPIED_FIELDS, 'not copied'
    assert TimestepResult().interpolated_across_failure is False, 'default'


def test_failed_solve_without_any_success_stays_failed():
    """With nothing to interpolate from, a failure remains a failure."""
    from superhot_wellbore.client.results import TimestepResult

    solved = {0: TimestepResult(time_yr=0.0, solved=True, success=False,
                                message='boom', pumped=True,
                                pump_flags=['no_liquid_intake'])}
    filled = interpolate_timesteps([0.0, 1.0], solved)
    assert not filled[0].success and filled[0].message == 'boom', \
        'failure preserved'
    assert filled[0].pumped and filled[0].pump_flags == ['no_liquid_intake'], \
        'the failure keeps its own pump state and flags'
    assert not filled[1].success, 'unsolved step has nothing to use'
    assert not filled[1].interpolated_across_failure, \
        'nothing was interpolated, so nothing was carried across'


# ====================================================================
# CHOKED PRESCRIBED-FLOW SOLVES
# ====================================================================

def test_choked_prescribed_flow_solve_is_flagged(monkeypatch):
    """A choked march under prescribed flow raises 'choked_flow'.

    The core returns success=True with choked=True; the wellhead values
    above the choke point are then approximate. Under control='flow' the
    client says so on the per-timestep flag channel that GEOPHIRES
    surfaces and can enforce; a wellhead pressure solve is not flagged.
    """
    from superhot_wellbore.client import PUMP_FLAGS
    from superhot_wellbore.client import client as client_module
    from superhot_wellbore.client.config import CoupledWellboreRequest

    choked = {'success': True, 'mass_flow_kgs': 60.0, 'whp_MPa': 12.0,
              'T_surface_C': 320.0, 'h_surface_MJkg': 2.7,
              'T_feedzone_C': 400.0, 'h_feedzone_MJkg': 2.8,
              'P_bh_MPa': 20.0, 'dP_reservoir_MPa': 10.0, 'choked': True}
    monkeypatch.setattr(client_module.core, 'coupled_model',
                        lambda **kwargs: dict(choked))
    monkeypatch.setattr(client_module.core, 'solve_flow_for_whp',
                        lambda **kwargs: dict(choked, converged=True))
    monkeypatch.setattr(client_module, 'power_cycle',
                        type('PC', (), {
                            'power_cycle_analysis':
                                staticmethod(lambda raw, params: None),
                            'dry_steam_specific_work':
                                staticmethod(lambda whp, params=None: 0.5)}))

    def solve(control):
        request = CoupledWellboreRequest.from_dict({
            'well': {'depth_m': 3500},
            'operating': {'control': control, 'target_whp_MPa': 10.0,
                          'mass_flow_kgs': 60.0},
            'pump': {'mode': 'never'},
        })
        result, _ = client_module.CoupledWellboreClient(request).solve_state(
            30.0, 450.0)
        return result

    assert 'choked_flow' in PUMP_FLAGS, 'a known flag'

    flow = solve('flow')
    assert flow.success and flow.choked, 'a usable, choked solve'
    assert 'choked_flow' in flow.pump_flags, 'flagged under prescribed flow'
    assert 'choke limited' in flow.message, 'the message still says why'

    whp = solve('whp')
    assert whp.success and whp.choked, 'a usable, choked solve'
    assert 'choked_flow' not in whp.pump_flags, \
        'a wellhead pressure solve is not flagged'


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
                            P_pump_intake_MPa=1.9)}
    filled = interpolate_timesteps([0.0, 1.0, 2.0], solved)
    middle = filled[1]
    assert not middle.solved and middle.success, 'interpolated step'
    assert middle.pump_depth_m == pytest.approx(700.0), 'pump depth'
    assert middle.dP_pump_MPa == pytest.approx(6.0), 'pump rise'
    assert middle.pump_power_MWe == pytest.approx(0.5), 'pump power'
    assert middle.T_pump_intake_C == pytest.approx(192.8), 'intake T'
    assert middle.P_pump_intake_MPa == pytest.approx(1.8), 'intake P'
    assert np.isnan(middle.self_flow_whp_MPa), 'NaN stays NaN'
    # The dry-steam work is derived from whp_MPa per timestep, not interpolated:
    # see test_dry_steam_work_is_not_interpolated.


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
