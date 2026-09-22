# -*- coding: utf-8 -*-
"""
Production Pump Stage
=====================

Covers :mod:`superhot_wellbore.client.pump`. The fast tests replace
the wellbore march with an analytic liquid column (hydrostatic head
plus a constant friction gradient, no heat loss) so that the intake
rule, the envelope, the pressure search and the energy bookkeeping
can be checked against numbers worked out by hand; the equation of
state stays real. The slow tests run the actual march on the CATF
cells that motivated the pump - a 200 C well at 5 km that dies on the
way up and a 250 C well that self-flows two-phase - and on the SHR-4
superhot state, which must come out exactly as the unpumped core
model returns it.

Author: superhot-wellbore GEOPHIRES client
"""

import math
import warnings

import CoolProp.CoolProp as CP
import numpy as np
import pytest

from superhot_wellbore import reservoir as core
from superhot_wellbore.client import client as client_module
from superhot_wellbore.client import pump as pump_module
from superhot_wellbore.client.client import CoupledWellboreClient
from superhot_wellbore.client.config import PumpConfig, CoupledWellboreRequest
from superhot_wellbore.client.pump import (PUMP_FLAGS,
                                           saturation_pressure_MPa,
                                           solve_pumped_state,
                                           wellhead_state)
from superhot_wellbore.wellbore_physics import fluid_properties_Ph


# ====================================================================
# ANALYTIC COLUMN
# ====================================================================

GRAVITY = 9.81
RHO = 880.0             # kg/m3, a ~190 C liquid
FRICTION_PA_M = 200.0   # Pa/m, constant friction gradient
DEPTH = 3000
DZ = 10
MDOT = 60.0
T_LIQUID_C = 190.0


def _h_Jkg(T_C, P_MPa):
    """Liquid enthalpy [J/kg] at (T, P)."""
    return CP.PropsSI('H', 'T', T_C + 273.15, 'P', P_MPa * 1e6, 'Water')


def _rock(depth):
    """Linear rock profile on the march grid, int keys."""
    return {z: 10.0 + (T_LIQUID_C - 10.0) * z / DEPTH
            for z in range(0, depth + DZ, DZ)}


class AnalyticMarch:
    """
    Stand-in for wellbore_simulate: an isenthalpic liquid column.

    Reproduces the layout of the real march (profiles of
    (current_depth, value) from the feedzone to the surface, the
    stored pressure being the one after each step, termination when
    the pressure runs out) with dP/dz = rho g + friction, constant.
    Every call is recorded so that a test can inspect the segment
    the pump stage asked for.
    """

    def __init__(self):
        self.calls = []

    def __call__(self, P_bottom_MPa, h_Jkg, mdot, rock, well_params):
        self.calls.append({'P': P_bottom_MPa, 'h': h_Jkg, 'mdot': mdot,
                           'rock': dict(rock),
                           'well_params': dict(well_params)})
        depth = well_params['depth_m']
        dz = well_params['delta_z_m']
        gradient = (RHO * GRAVITY + FRICTION_PA_M) * dz / 1e6
        pressure = P_bottom_MPa
        profiles = [[] for _ in range(6)]
        for z in range(0, depth + dz, dz):
            current = depth - z
            pressure_new = pressure - gradient
            if pressure_new <= 0:
                break
            pressure = pressure_new
            T_C = fluid_properties_Ph(pressure, h_Jkg)['temperature_K'] \
                - 273.15
            profiles[0].append((current, T_C))
            profiles[1].append((current, pressure))
            profiles[2].append((current, h_Jkg * 1e-6))
            profiles[3].append((current, 1.0))
            profiles[4].append((current, np.nan))
            profiles[5].append((current, rock[current]))
        return tuple(profiles) + (False,)


@pytest.fixture
def analytic_march(monkeypatch):
    """Install the analytic column in place of the real march."""
    march = AnalyticMarch()
    monkeypatch.setattr(pump_module, 'wellbore_simulate', march)
    return march


def _solve(P_fz_MPa, pump_cfg, h_Jkg=None, depth=DEPTH):
    """Run the pump stage on the analytic column."""
    if h_Jkg is None:
        h_Jkg = _h_Jkg(T_LIQUID_C, P_fz_MPa)
    return solve_pumped_state(
        P_fz_MPa=P_fz_MPa, h_fz_Jkg=h_Jkg, mdot=MDOT,
        rock_temperatures=_rock(depth),
        well_params={'depth_m': depth, 'delta_z_m': DZ,
                     'diameter_m': 0.217},
        pump_cfg=pump_cfg, P_farfield_MPa=P_fz_MPa + 8.0,
        T_feedzone_C=T_LIQUID_C)


def _npsh_ok(P_MPa, h_MJkg, margin):
    """The intake rule applied to one profile point."""
    props = fluid_properties_Ph(P_MPa, h_MJkg * 1e6)
    T_C = props['temperature_K'] - 273.15
    return (props['phase'] == 'single_phase_liquid'
            and P_MPa >= saturation_pressure_MPa(T_C) + margin)


# A feedzone pressure from which the analytic column dies before the
# surface: 3000 m of 880 kg/m3 liquid needs ~26.5 MPa.
P_DIES = 20.0
# A feedzone pressure from which it reaches the surface above 1 MPa.
P_SELF_FLOWS = 29.0


# ====================================================================
# SELF-FLOW VERDICT
# ====================================================================

def test_self_flow_above_the_floor_is_left_alone(analytic_march):
    """A column that reaches the surface above the floor is not pumped."""
    raw = _solve(P_SELF_FLOWS, PumpConfig(mode='auto'))
    assert raw['success'] and not raw['pumped'], 'not pumped'
    assert raw['self_flowing'], 'self-flowing'
    assert raw['self_flow_whp_MPa'] == pytest.approx(raw['whp_MPa']), \
        'the self-flow WHP is the WHP'
    assert raw['whp_MPa'] >= 1.0, 'above the floor'
    assert raw['pump_depth_m'] == 0.0 and raw['pump_power_MWe'] == 0.0, \
        'no pump quantities'
    assert raw['wellhead_phase'] == 'single_phase_liquid', \
        'liquid wellhead, got ' + raw['wellhead_phase']
    assert np.isnan(raw['wellhead_quality']), 'no quality for a liquid'
    assert raw['pump_flags'] == [], 'no flags'
    assert len(analytic_march.calls) == 1, 'a single march'


def test_self_flow_below_the_floor_is_pumped_in_auto(analytic_march):
    """Reaching the surface below the floor counts as not self-flowing."""
    raw = _solve(P_SELF_FLOWS, PumpConfig(mode='auto',
                                          min_self_flow_whp_MPa=5.0))
    assert raw['pumped'] and not raw['self_flowing'], 'pumped'
    assert 'self_flow_below_floor' in raw['pump_flags'], raw['pump_flags']
    assert np.isfinite(raw['self_flow_whp_MPa']), \
        'the unpumped WHP is still reported'


def test_target_whp_raises_the_floor(analytic_march):
    """A caller's target wellhead pressure acts as the self-flow floor."""
    raw = _solve(P_SELF_FLOWS, PumpConfig(mode='auto', target_whp_MPa=5.0))
    assert raw['pumped'], 'pumped to the target'
    assert raw['whp_MPa'] == pytest.approx(5.0, abs=0.01), \
        'WHP meets the target, got ' + str(raw['whp_MPa'])


def test_mode_never_reports_but_does_not_pump(analytic_march):
    """With the pump off, a low wellhead pressure is only flagged."""
    raw = _solve(P_SELF_FLOWS, PumpConfig(mode='never',
                                          min_self_flow_whp_MPa=5.0))
    assert raw['success'] and not raw['pumped'], 'unpumped'
    assert raw['pump_flags'] == ['self_flow_below_floor'], raw['pump_flags']
    dead = _solve(P_DIES, PumpConfig(mode='never'))
    assert not dead['success'] and not dead['pumped'], \
        'a column that dies stays a failure'
    assert np.isnan(dead['self_flow_whp_MPa']), 'no self-flow WHP'
    assert dead['P_bh_MPa'] == P_DIES, 'the feedzone state is reported'


def test_mode_never_matches_coupled_model(monkeypatch, analytic_march):
    """mode 'never' returns exactly what reservoir.coupled_model returns."""
    monkeypatch.setattr(core, 'wellbore_simulate', analytic_march)
    P_res, T_res = 30.0, 300.0
    # A high transmissivity keeps the Darcy drawdown small enough for
    # the analytic column to reach the surface from the feedzone.
    rp = {'transmissivity_md_m': 5000.0, 'drainage_radius_m': 500.0}
    wp = {'depth_m': DEPTH, 'delta_z_m': DZ, 'diameter_m': 0.217}
    reference = core.coupled_model(MDOT, P_res, T_res, _rock(DEPTH), rp, wp)
    assert reference['success'], 'the reference column reaches the surface'
    bh = core.bottomhole_pressure(MDOT, P_res, T_res, rp, wp)
    raw = solve_pumped_state(
        bh['P_bh_MPa'], bh['h_feedzone_Jkg'], MDOT, _rock(DEPTH), wp,
        PumpConfig(mode='never'), P_res, bh['T_feedzone_C'],
        dP_reservoir_MPa=bh['dP_reservoir_MPa'])
    for key, value in reference.items():
        assert raw[key] == value, f'{key}: {raw[key]!r} != {value!r}'


# ====================================================================
# INTAKE RULE
# ====================================================================

def test_intake_is_the_shallowest_liquid_point(analytic_march):
    """The intake is the shallowest depth with a liquid column below it."""
    cfg = PumpConfig(mode='auto')
    raw = _solve(P_DIES, cfg)
    assert raw['pumped'], 'the column needs a pump'

    full = analytic_march.calls[0]
    profiles = analytic_march(full['P'], full['h'], MDOT, full['rock'],
                              full['well_params'])
    pressures = dict(profiles[1])
    enthalpies = dict(profiles[2])
    z_p = raw['pump_depth_m']
    assert z_p % DZ == 0, 'intake on the march grid'
    assert _npsh_ok(pressures[z_p], enthalpies[z_p], cfg.npsh_margin_MPa), \
        'the intake point satisfies the NPSH rule'
    assert not _npsh_ok(pressures[z_p - DZ], enthalpies[z_p - DZ],
                        cfg.npsh_margin_MPa), \
        'the next point up does not, so the intake is the shallowest'
    assert all(_npsh_ok(pressures[z], enthalpies[z], cfg.npsh_margin_MPa)
               for z in pressures if z > z_p), \
        'every point below the intake is liquid with the margin'
    assert raw['P_pump_intake_MPa'] == pytest.approx(pressures[z_p]), \
        'intake pressure read off the unpumped profile'
    assert raw['T_pump_intake_C'] == pytest.approx(T_LIQUID_C, abs=2.0), \
        'intake temperature, got ' + str(raw['T_pump_intake_C'])


def test_two_phase_sandface_has_no_intake(analytic_march):
    """A feedzone that is itself two-phase cannot feed a pump."""
    P_fz = 1.0   # MPa: the 190 C liquid enthalpy is two-phase here
    raw = _solve(P_fz, PumpConfig(mode='auto'),
                 h_Jkg=_h_Jkg(T_LIQUID_C, 5.0))
    assert not raw['pumped'], 'nothing to pump'
    assert 'no_liquid_intake' in raw['pump_flags'], raw['pump_flags']
    assert 'two_phase_at_sandface' in raw['pump_flags'], raw['pump_flags']

    enforced = _solve(P_fz, PumpConfig(mode='auto', envelope='enforce'),
                      h_Jkg=_h_Jkg(T_LIQUID_C, 5.0))
    assert not enforced['success'], 'enforce fails the solve'
    assert 'no liquid pump intake' in enforced['message'], \
        enforced['message']


# ====================================================================
# SEGMENT COMPOSITION AND ENERGY BOOKKEEPING
# ====================================================================

def test_upper_segment_is_marched_on_its_own_grid(analytic_march):
    """The pumped segment gets a rock dict keyed 0..z_p on integers."""
    raw = _solve(P_DIES, PumpConfig(mode='auto'))
    z_p = int(raw['pump_depth_m'])
    segments = analytic_march.calls[1:]
    assert segments, 'the upper segment was marched'
    for call in segments:
        assert call['well_params']['depth_m'] == z_p, \
            'segment depth is the intake depth'
        assert isinstance(call['well_params']['depth_m'], int), \
            'segment depth is an int'
        keys = sorted(call['rock'])
        assert keys == list(range(0, z_p + DZ, DZ)), \
            'rock dict covers 0..z_p on the march grid'
        assert all(isinstance(k, int) for k in keys), 'integer keys'
        assert call['mdot'] == MDOT, 'same flow rate'


def test_pump_pressure_meets_the_target_within_tolerance(analytic_march):
    """The secant search lands on P_sat(T_intake) + margin."""
    cfg = PumpConfig(mode='auto', tolerance_MPa=0.01)
    raw = _solve(P_DIES, cfg)
    target = (saturation_pressure_MPa(raw['T_pump_intake_C'])
              + cfg.npsh_margin_MPa)
    assert abs(raw['whp_MPa'] - target) <= cfg.tolerance_MPa, \
        f"WHP {raw['whp_MPa']} vs target {target}"
    # The analytic column makes the answer explicit: the segment
    # loses the constant gradient over z_p plus one extra step
    # (the march stores the pressure after each step).
    z_p = raw['pump_depth_m']
    loss = (RHO * GRAVITY + FRICTION_PA_M) * (z_p + DZ) / 1e6
    assert raw['dP_pump_MPa'] == pytest.approx(
        target - raw['P_pump_intake_MPa'] + loss, abs=cfg.tolerance_MPa), \
        'pump rise equals target minus intake pressure plus column loss'
    assert len(analytic_march.calls) <= 8, \
        'the search converges in a few marches, used ' + \
        str(len(analytic_march.calls))


def test_pump_power_and_discharge_enthalpy(analytic_march):
    """Pump power is m dP / (rho eta) and h2 = h1 + dP / (rho eta)."""
    cfg = PumpConfig(mode='auto', efficiency=0.75)
    raw = _solve(P_DIES, cfg)
    intake = fluid_properties_Ph(raw['P_pump_intake_MPa'],
                                 _h_Jkg(T_LIQUID_C, P_DIES))
    rho1 = intake['density_kgm3']
    dP = raw['dP_pump_MPa']
    assert raw['pump_power_MWe'] == pytest.approx(
        MDOT * dP / (rho1 * cfg.efficiency), abs=1e-9), \
        'pump power per well'
    last = analytic_march.calls[-1]
    h1 = _h_Jkg(T_LIQUID_C, P_DIES)
    assert last['h'] == pytest.approx(h1 + dP * 1e6 / (rho1 * cfg.efficiency),
                                      rel=1e-12), \
        'discharge enthalpy of the last segment march'
    assert last['P'] == pytest.approx(raw['P_pump_intake_MPa'] + dP,
                                      rel=1e-12), \
        'discharge pressure of the last segment march'


def test_stitched_profile_is_continuous_except_at_the_pump(analytic_march):
    """The profile runs feedzone -> intake, then intake -> surface."""
    raw = _solve(P_DIES, PumpConfig(mode='auto'))
    depths = [d for d, _ in raw['profiles'][1]]
    pressures = [p for _, p in raw['profiles'][1]]
    z_p = raw['pump_depth_m']
    assert depths[0] == DEPTH and depths[-1] == 0, 'feedzone to surface'
    assert depths.count(z_p) == 2, 'the intake depth appears twice'
    jump = depths.index(z_p)
    assert pressures[jump + 1] > pressures[jump], \
        'pressure jumps up across the pump'
    steps = np.diff(pressures)
    steps = np.delete(steps, jump)
    assert np.all(steps < 0), 'pressure falls monotonically elsewhere'
    assert np.ptp(steps) < 1e-9, 'the analytic column has one gradient'
    assert raw['whp_MPa'] == pressures[-1], 'WHP is the last point'


# ====================================================================
# ENVELOPE
# ====================================================================

def test_envelope_flags(analytic_march):
    """Depth and temperature limits are flagged but modelled."""
    raw = _solve(P_DIES, PumpConfig(mode='auto', max_depth_m=500.0,
                                    max_intake_temperature_C=150.0))
    assert raw['success'] and raw['pumped'], 'still modelled'
    assert raw['pump_depth_m'] > 500.0, 'the intake is deeper than 500 m'
    for flag in ('depth_limit', 'temperature_limit',
                 'pump_outside_envelope'):
        assert flag in raw['pump_flags'], f'{flag} in {raw["pump_flags"]}'
    assert set(raw['pump_flags']) <= set(PUMP_FLAGS), 'known flags only'


def test_envelope_enforced_fails_and_names_the_limit(analytic_march):
    """Under 'enforce' a pump outside the envelope fails the solve."""
    raw = _solve(P_DIES, PumpConfig(mode='auto', envelope='enforce',
                                    max_intake_temperature_C=150.0))
    assert not raw['success'] and raw['pumped'], 'failed, pump required'
    assert 'temperature_limit' in raw['pump_flags'], raw['pump_flags']
    assert 'depth_limit' not in raw['pump_flags'], 'depth is fine'
    assert 'intake temperature' in raw['message'], raw['message']
    assert '150' in raw['message'], 'the limit is named: ' + raw['message']
    assert f"{raw['T_pump_intake_C']:.1f}" in raw['message'], \
        'the value is named: ' + raw['message']
    assert np.isnan(raw['whp_MPa']), 'no wellhead state'
    assert len(analytic_march.calls) == 1, 'no segment march was run'


def test_envelope_omit_leaves_a_self_flowing_well_unpumped(analytic_march):
    """Under 'omit' a pump outside the envelope is not installed when
    the well reaches the surface on its own; the flags say why."""
    # 28 MPa at the feedzone reaches the surface at about 1.5 MPa on the
    # analytic column: below a 2 MPa floor, so a pump is wanted, and its
    # 190 C liquid intake is hotter than the 150 C limit.
    raw = _solve(28.0, PumpConfig(mode='auto', envelope='omit',
                                  min_self_flow_whp_MPa=2.0,
                                  max_intake_temperature_C=150.0))
    assert raw['success'] and not raw['pumped'], 'self-flowing, unpumped'
    assert raw['pump_depth_m'] == 0.0 and raw['pump_power_MWe'] == 0.0
    assert 0.0 < raw['whp_MPa'] < 2.0, 'the unpumped wellhead, below the floor'
    assert raw['whp_MPa'] == raw['self_flow_whp_MPa']
    assert not raw['self_flowing'], 'the floor is not met'
    for flag in ('self_flow_below_floor', 'temperature_limit',
                 'pump_outside_envelope'):
        assert flag in raw['pump_flags'], f'{flag} in {raw["pump_flags"]}'
    assert 'message' not in raw or not raw['message'], 'no failure message'
    assert len(analytic_march.calls) == 1, 'no segment march was run'


def test_envelope_omit_fails_a_well_that_does_not_reach_the_surface(analytic_march):
    """Under 'omit' a well that needs the pump to reach the surface
    still fails when that pump is outside the envelope."""
    raw = _solve(P_DIES, PumpConfig(mode='auto', envelope='omit',
                                    max_intake_temperature_C=150.0))
    assert not raw['success'] and raw['pumped'], 'failed, pump required'
    assert 'temperature_limit' in raw['pump_flags'], raw['pump_flags']
    assert 'intake temperature' in raw['message'], raw['message']
    assert np.isnan(raw['whp_MPa']), 'no wellhead state'


def test_inside_the_envelope_nothing_is_flagged(analytic_march):
    """A pump within the limits carries no flags."""
    raw = _solve(P_DIES, PumpConfig(mode='auto', envelope='enforce'))
    assert raw['success'] and raw['pumped'], 'pumped'
    assert raw['pump_flags'] == [], raw['pump_flags']


# ====================================================================
# HELPERS
# ====================================================================

def test_saturation_pressure_continues_as_the_critical_pressure():
    """P_sat follows CoolProp below T_crit and is P_crit above it."""
    assert saturation_pressure_MPa(100.0) == pytest.approx(0.101418,
                                                           rel=1e-4), \
        '1 atm at 100 C'
    assert saturation_pressure_MPa(400.0) == pytest.approx(22.064), \
        'critical pressure above T_crit'


def test_wellhead_state_labels():
    """Phase labels and quality from (P, h) at the wellhead."""
    h_liquid = _h_Jkg(150.0, 2.0) * 1e-6
    phase, quality = wellhead_state(2.0, h_liquid)
    assert phase == 'single_phase_liquid' and np.isnan(quality)
    h_f = CP.PropsSI('H', 'P', 2e6, 'Q', 0, 'Water')
    h_g = CP.PropsSI('H', 'P', 2e6, 'Q', 1, 'Water')
    phase, quality = wellhead_state(2.0, (h_f + 0.3 * (h_g - h_f)) * 1e-6)
    assert phase == 'two_phase' and quality == pytest.approx(0.3, abs=1e-6)
    phase, quality = wellhead_state(2.0, _h_Jkg(400.0, 2.0) * 1e-6)
    assert phase == 'single_phase_vapor' and np.isnan(quality)
    phase, quality = wellhead_state(26.0, _h_Jkg(375.0, 26.0) * 1e-6)
    assert phase == 'supercritical' and np.isnan(quality)
    assert wellhead_state(float('nan'), 2.0) == ('', pytest.approx(
        np.nan, nan_ok=True))


# ====================================================================
# CLIENT ROUTE
# ====================================================================

def test_client_routes_held_flow_through_the_pump_stage(monkeypatch):
    """Under mode 'auto' a prescribed flow goes through pump.py."""
    seen = []

    def fake_bottomhole_pressure(mdot, P_res, T_res, rp=None, wp=None,
                                 max_iterations=3):
        return {'P_bh_MPa': P_res - 8.0, 'h_feedzone_Jkg': 0.85e6,
                'T_feedzone_C': T_res - 1.0, 'dP_reservoir_MPa': 8.0}

    def fake_solve_pumped_state(**kwargs):
        seen.append(kwargs)
        return {'success': True, 'mass_flow_kgs': kwargs['mdot'],
                'whp_MPa': 1.6, 'T_surface_C': 190.0,
                'h_surface_MJkg': 0.81, 'T_feedzone_C': 199.0,
                'h_feedzone_MJkg': 0.85, 'P_bh_MPa': 41.0,
                'dP_reservoir_MPa': 8.0, 'choked': False,
                'pumped': True, 'pump_depth_m': 610.0,
                'P_pump_intake_MPa': 1.7, 'T_pump_intake_C': 191.8,
                'dP_pump_MPa': 5.4, 'pump_power_MWe': 0.46,
                'self_flow_whp_MPa': float('nan'), 'self_flowing': False,
                'wellhead_phase': 'single_phase_liquid',
                'wellhead_quality': float('nan'),
                'dry_steam_work_MJkg': 0.5, 'pump_flags': ['depth_limit']}

    monkeypatch.setattr(client_module.core, 'bottomhole_pressure',
                        fake_bottomhole_pressure)
    monkeypatch.setattr(client_module.pump_module, 'solve_pumped_state',
                        fake_solve_pumped_state)
    monkeypatch.setattr(client_module.core, 'coupled_model',
                        lambda **kwargs: pytest.fail('coupled_model called'))

    request = CoupledWellboreRequest.from_dict({
        'reservoir': {'P_reservoir_MPa': 49.0, 'T_reservoir_C': 200.0},
        'well': {'depth_m': 5120},
        'operating': {'control': 'flow', 'mass_flow_kgs': 60.0},
        'pump': {'mode': 'auto', 'max_depth_m': 1500.0},
    })
    result = CoupledWellboreClient(request).solve_steady_state()

    assert len(seen) == 1, 'one pump-stage solve'
    call = seen[0]
    assert call['P_fz_MPa'] == pytest.approx(41.0), 'Darcy feedzone pressure'
    assert call['h_fz_Jkg'] == pytest.approx(0.85e6), 'feedzone enthalpy'
    assert call['dP_reservoir_MPa'] == pytest.approx(8.0), \
        'the Darcy drawdown is passed through'
    assert call['pump_cfg'] is request.pump, 'the request pump section'
    assert call['power_params']['T_ambient_C'] == 25.0, 'power params'
    assert result.pumped and result.pump_depth_m == 610.0, \
        'pump quantities translated'
    assert result.pump_power_MWe == pytest.approx(0.46), 'pump power'
    assert result.wellhead_phase == 'single_phase_liquid', 'phase'
    assert result.pump_flags == ['depth_limit'], 'flags as a list'
    assert result.dry_steam_work_MJkg == pytest.approx(0.5), 'dry steam'
    assert result.success, 'usable'


def test_client_reports_an_enforced_failure_message(monkeypatch):
    """The pump stage's message reaches the TimestepResult."""
    monkeypatch.setattr(
        client_module.core, 'bottomhole_pressure',
        lambda *a, **k: {'P_bh_MPa': 41.0, 'h_feedzone_Jkg': 0.85e6,
                         'T_feedzone_C': 199.0, 'dP_reservoir_MPa': 8.0})
    monkeypatch.setattr(
        client_module.pump_module, 'solve_pumped_state',
        lambda **kwargs: {'success': False, 'mass_flow_kgs': 60.0,
                          'whp_MPa': float('nan'), 'pumped': True,
                          'pump_flags': ['depth_limit',
                                         'pump_outside_envelope'],
                          'message': 'production pump required outside '
                                     'the envelope: intake depth 1610 m '
                                     'exceeds the maximum 1500 m'})
    request = CoupledWellboreRequest.from_dict({
        'reservoir': {'T_reservoir_C': 200.0}, 'well': {'depth_m': 5000},
        'operating': {'control': 'flow', 'mass_flow_kgs': 60.0},
        'pump': {'envelope': 'enforce'},
        'solver': {'strict': True},
    })
    client = CoupledWellboreClient(request)
    result = client.solve_steady_state()
    assert not result.success and result.pumped, 'failure with the pump'
    assert 'exceeds the maximum 1500 m' in result.message, result.message
    with pytest.raises(RuntimeError, match='exceeds the maximum'):
        client.solve_profile()


# ====================================================================
# REAL SOLVES
# ====================================================================

def _catf_request(T_C, depth_m, P_res_MPa, PI_kg_s_bar=0.7482,
                  flow_kgs=60.0, **pump):
    """A CATF-style FOAK request: PI-equivalent transmissivity, 8.535 in."""
    diameter_m = 8.535 * 0.0254
    rho = CP.PropsSI('D', 'T', T_C + 273.15, 'P', P_res_MPa * 1e6, 'Water')
    mu = CP.PropsSI('V', 'T', T_C + 273.15, 'P', P_res_MPa * 1e6, 'Water')
    drainage_radius_m = 500.0
    kb_m3 = ((PI_kg_s_bar / 1e5) * mu
             * math.log(drainage_radius_m / (diameter_m / 2))
             / (2.0 * math.pi * rho))
    return CoupledWellboreRequest.from_dict({
        'reservoir': {'P_reservoir_MPa': P_res_MPa, 'T_reservoir_C': T_C,
                      'transmissivity_md_m': kb_m3 / 9.869233e-16,
                      'drainage_radius_m': drainage_radius_m},
        'well': {'depth_m': depth_m, 'diameter_m': diameter_m,
                 'delta_z_m': 10},
        'rock_temperature': {'mode': 'linear', 'T_surface_C': 12.0},
        'operating': {'control': 'flow', 'mass_flow_kgs': flow_kgs},
        'power_cycle': {'T_ambient_C': 10.0, 'T_reject_C': 63.0},
        'pump': dict({'mode': 'auto'}, **pump),
    })


@pytest.mark.slow
def test_foak_200C_well_is_pumped():
    """CATF FOAK 200 C at 5120 m (49.3 MPa): dies unpumped, pumped ORC feed."""
    request = _catf_request(200.0, 5120, 49.3)
    result, raw = CoupledWellboreClient(request).solve_state(49.3, 200.0)
    assert result.success, result.message
    assert result.pumped and not result.self_flowing, 'pumped'
    assert np.isnan(result.self_flow_whp_MPa), \
        'the unpumped column does not reach the surface'
    assert raw['dP_reservoir_MPa'] == pytest.approx(8.0, abs=0.3), \
        'PI drawdown of 8 MPa at 60 kg/s, got ' + str(raw['dP_reservoir_MPa'])
    assert 500 <= result.pump_depth_m <= 700, \
        'intake depth ' + str(result.pump_depth_m)
    assert 185 <= result.T_pump_intake_C <= 200, \
        'intake temperature ' + str(result.T_pump_intake_C)
    target = saturation_pressure_MPa(result.T_pump_intake_C) + 0.3447
    assert result.whp_MPa == pytest.approx(target, abs=0.02), \
        f'WHP {result.whp_MPa} vs P_sat(T_intake) + margin {target}'
    assert 185 <= result.T_wellhead_C <= 195, \
        'wellhead temperature ' + str(result.T_wellhead_C)
    assert 0.35 <= result.pump_power_MWe <= 0.60, \
        'pump power ' + str(result.pump_power_MWe)
    assert result.wellhead_phase == 'single_phase_liquid', \
        result.wellhead_phase
    assert result.pump_flags == [], result.pump_flags
    assert np.isfinite(result.dry_steam_work_MJkg), 'dry-steam work'


@pytest.mark.slow
def test_foak_250C_well_self_flows_two_phase():
    """CATF FOAK 250 C at 6490 m (61.4 MPa): self-flows ~1.8 MPa two-phase."""
    request = _catf_request(250.0, 6490, 61.4)
    result, raw = CoupledWellboreClient(request).solve_state(61.4, 250.0)
    assert result.success, result.message
    assert result.self_flowing and not result.pumped, 'self-flowing'
    assert result.whp_MPa == pytest.approx(1.8, abs=0.3), \
        'WHP ' + str(result.whp_MPa)
    assert result.wellhead_phase == 'two_phase', result.wellhead_phase
    assert 0.03 <= result.wellhead_quality <= 0.10, \
        'quality ' + str(result.wellhead_quality)
    assert result.self_flow_whp_MPa == result.whp_MPa, 'same WHP'
    assert result.pump_depth_m == 0.0 and result.pump_power_MWe == 0.0, \
        'no pump'
    assert np.isfinite(result.dry_steam_work_MJkg), 'dry-steam work'
    assert result.pump_flags == [], result.pump_flags


@pytest.mark.slow
def test_shr4_state_is_unchanged_by_the_pump_stage():
    """450 C / 30 MPa / 3.5 km: the dict equals coupled_model's exactly."""
    base = {
        'reservoir': {'P_reservoir_MPa': 30.0, 'T_reservoir_C': 450.0,
                      'transmissivity_md_m': 1000.0},
        'well': {'depth_m': 3500},
        'operating': {'control': 'flow', 'mass_flow_kgs': 60.0},
    }
    client = CoupledWellboreClient(CoupledWellboreRequest.from_dict(base))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        reference = core.coupled_model(
            60.0, 30.0, 450.0, client._rock_temperatures(450.0),
            client.request.reservoir.to_reservoir_params(),
            client._well_params())
    result, raw = client.solve_state(30.0, 450.0)
    assert reference['success'] and result.success, 'both succeed'
    assert not raw['pumped'] and raw['self_flowing'], 'self-flowing'
    for key, value in reference.items():
        assert raw[key] == value, f'{key}: {raw[key]!r} != {value!r}'
    assert raw['wellhead_phase'] == 'single_phase_vapor', \
        raw['wellhead_phase']
    never = CoupledWellboreClient(CoupledWellboreRequest.from_dict(
        dict(base, pump={'mode': 'never'})))
    result_never, raw_never = never.solve_state(30.0, 450.0)
    assert raw_never == reference, "mode 'never' is coupled_model itself"
    assert result_never.power_MWe == result.power_MWe, 'same power'
