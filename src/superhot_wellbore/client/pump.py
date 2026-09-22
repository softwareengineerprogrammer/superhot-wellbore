# -*- coding: utf-8 -*-
"""
Production Pump Stage
=====================

Two-segment production well for flow rates the reservoir cannot lift
to the surface on its own. The core modules (reservoir.py,
wellbore_physics.py) describe a self-flowing well: the feedzone
state is marched upward and the solve fails when the pressure runs
out before the wellhead. Cooler and deeper wells - a 200 C reservoir
at 5 km, say - need a downhole pump for the flow rates a
techno-economic study prescribes, and this module adds that pump on
top of the unmodified core.

Method
------
1.  The unpumped march (wellbore_simulate from the feedzone) tells
    whether the well self-flows: it reaches the surface AND the
    wellhead pressure is at least the self-flow floor (or the
    caller's target). A self-flowing well is returned unchanged,
    two-phase or not.

2.  Otherwise the pump intake is the shallowest depth on the unpumped
    profile from which the column is single-phase liquid with
    P >= P_sat(T) + NPSH margin all the way down to the feedzone.
    Because the march is upstream-independent, the state at any
    depth of the unpumped profile IS the state of the lower segment,
    so no second march is needed below the pump.

3.  The pump raises the pressure by dP at constant efficiency,
    h2 = h1 + dP / (rho1 * eta), and the upper segment is marched
    from the intake to the surface with the same physics. dP is
    solved so that the wellhead pressure meets the target: the
    caller's target, or P_sat(T_intake) + NPSH margin, which keeps
    the wellhead stream liquid because the fluid only cools on the
    way up.

4.  Pump power per well is m dP / (rho1 eta), the same expression
    GEOPHIRES uses for its production pumps.

The intake depth and temperature are compared with an ESP envelope
(maximum setting depth, maximum intake temperature). Under the 'flag'
policy a pump outside the envelope is modelled and reported in
pump_flags; under 'enforce' the solve fails with the reason in the
message, so that a study never silently swaps its physics.

The returned dict has the shape of reservoir.coupled_model()'s so
that the client and power_cycle.power_cycle_analysis() accept it
unchanged, plus additive pump keys (see solve_pumped_state).

Author: superhot-wellbore GEOPHIRES client
"""

import numpy as np
import CoolProp.CoolProp as CP

from .. import power_cycle
from ..wellbore_physics import (DEFAULT_WELL_PARAMS, GRAVITY, P_CRIT_MPA,
                                T_CRIT_K, fluid_properties_Ph,
                                wellbore_simulate)


# ====================================================================
# CONSTANTS
# ====================================================================

#: Flags a pumped solve can report (subset appears in pump_flags)
PUMP_FLAGS = ('temperature_limit', 'depth_limit', 'self_flow_below_floor',
              'two_phase_at_sandface', 'pump_outside_envelope',
              'no_liquid_intake',
              # raised by the client, not here: a prescribed-flow march
              # reached the local sound speed, so the wellhead values
              # above the choke point are approximate
              'choked_flow')

#: Wellhead phase labels
WELLHEAD_PHASES = ('single_phase_liquid', 'two_phase',
                   'single_phase_vapor', 'supercritical')

#: Additive keys of the dict returned by solve_pumped_state
PUMP_RESULT_KEYS = ('pumped', 'pump_depth_m', 'P_pump_intake_MPa',
                    'T_pump_intake_C', 'dP_pump_MPa', 'pump_power_MWe',
                    'self_flow_whp_MPa', 'self_flowing', 'wellhead_phase',
                    'wellhead_quality', 'dry_steam_work_MJkg', 'pump_flags',
                    'message')


def pump_result_defaults():
    """
    Values the additive keys take when nothing was pumped or solved.

    A fresh dict each time, since pump_flags is a list.
    """
    return {
        'pumped': False,
        'pump_depth_m': 0.0,
        'P_pump_intake_MPa': np.nan,
        'T_pump_intake_C': np.nan,
        'dP_pump_MPa': 0.0,
        'pump_power_MWe': 0.0,
        'self_flow_whp_MPa': np.nan,
        'self_flowing': False,
        'wellhead_phase': '',
        'wellhead_quality': np.nan,
        'dry_steam_work_MJkg': np.nan,
        'pump_flags': [],
        'message': '',
    }


# Maximum number of upper-segment marches in the pump pressure search
_MAX_PUMP_ITERATIONS = 40


# ====================================================================
# THERMODYNAMIC HELPERS
# ====================================================================

def saturation_pressure_MPa(T_C):
    """
    Vapour pressure of water at T_C [MPa].

    Above the critical temperature there is no saturation curve; the
    critical pressure is returned so that the NPSH rule demands a
    dense (liquid-like) supercritical state there.
    """
    T_K = float(T_C) + 273.15
    if T_K >= T_CRIT_K:
        return P_CRIT_MPA
    return CP.PropsSI('P', 'T', T_K, 'Q', 0, 'Water') / 1e6


def wellhead_state(P_MPa, h_MJkg):
    """
    Phase label and vapour quality of a wellhead state.

    Returns
    -------
    (str, float)
        One of WELLHEAD_PHASES (a state above the critical pressure is
        'supercritical' whatever its density) and the vapour quality,
        NaN unless the state is two-phase.
    """
    if not (np.isfinite(P_MPa) and np.isfinite(h_MJkg)):
        return '', np.nan
    if P_MPa >= P_CRIT_MPA:
        return 'supercritical', np.nan
    props = fluid_properties_Ph(float(P_MPa), float(h_MJkg) * 1e6)
    phase = props['phase']
    if phase == 'two_phase':
        return phase, float(props['saturation'])
    if phase not in WELLHEAD_PHASES:
        # 'held_from_previous' after an EOS failure: classify by density
        phase = ('single_phase_liquid'
                 if props['density_kgm3'] >= 322.0
                 else 'single_phase_vapor')
    return phase, np.nan


# ====================================================================
# MARCH HELPERS
# ====================================================================

def _well_params(well_params):
    """Merged well parameters, as wellbore_simulate() sees them."""
    params = dict(DEFAULT_WELL_PARAMS)
    if well_params is not None:
        params.update(well_params)
    return params


def _march(P_MPa, h_Jkg, mdot, rock_temperatures, well_params):
    """
    Run wellbore_simulate() and say whether it reached the surface.

    Returns
    -------
    (tuple, bool, bool)
        The six profiles plus the choked flag as a 7-tuple (the layout
        reservoir.coupled_model() stores under 'profiles'), whether
        the march reached the surface, and the choked flag.
    """
    *profiles_list, choked = wellbore_simulate(
        P_MPa, h_Jkg, mdot, rock_temperatures, well_params)
    profiles = tuple(profiles_list)
    delta_z = _well_params(well_params)['delta_z_m']
    # The same rule as reservoir.coupled_model(): a march that stopped
    # more than two steps short of the surface did not reach it.
    reached = bool(profiles[0]) and profiles[0][-1][0] <= 2 * delta_z
    return profiles + (choked,), reached, bool(choked)


def _surface(profiles):
    """(whp_MPa, T_surface_C, h_surface_MJkg) from the last point."""
    return (profiles[1][-1][1], profiles[0][-1][1], profiles[2][-1][1])


def _find_intake(profiles, npsh_margin_MPa):
    """
    Shallowest pump intake with a liquid column below it.

    Walks the unpumped profile from the feedzone upward and keeps the
    shallowest point at which every point from the feedzone up to it
    is single-phase liquid with P >= P_sat(T) + npsh_margin_MPa.

    Returns
    -------
    (int or None, float, float, float, list)
        Profile index of the intake (None if even the feedzone fails
        the test), and the intake depth [m], pressure [MPa] and
        temperature [C], plus the flags raised.
    """
    temperature_profile, pressure_profile, enthalpy_profile = profiles[:3]
    intake = None
    for index, (depth, P) in enumerate(pressure_profile):
        h_Jkg = enthalpy_profile[index][1] * 1e6
        props = fluid_properties_Ph(P, h_Jkg)
        T_C = props['temperature_K'] - 273.15
        liquid = (props['phase'] == 'single_phase_liquid'
                  and P >= saturation_pressure_MPa(T_C) + npsh_margin_MPa)
        if not liquid:
            break
        intake = (index, float(depth), float(P), float(T_C))

    if intake is None:
        return None, np.nan, np.nan, np.nan, ['no_liquid_intake',
                                              'two_phase_at_sandface']
    return intake + ([],)


# ====================================================================
# RESULT ASSEMBLY
# ====================================================================

def _fail_dict(mdot, choked=False):
    """The failure dict of reservoir.coupled_model(), plus pump keys."""
    result = {
        'mass_flow_kgs': mdot,
        'whp_MPa': np.nan, 'P_bh_MPa': np.nan,
        'h_feedzone_MJkg': np.nan, 'T_feedzone_C': np.nan,
        'h_surface_MJkg': np.nan, 'T_surface_C': np.nan,
        'dP_reservoir_MPa': np.nan, 'choked': choked,
        'profiles': None, 'success': False,
    }
    result.update(pump_result_defaults())
    return result


def _unpumped_dict(mdot, P_fz_MPa, h_fz_Jkg, T_feedzone_C,
                   dP_reservoir_MPa, profiles, reached, choked):
    """
    The dict reservoir.coupled_model() would return for this march.

    The values of the shared keys are computed exactly as
    coupled_model() computes them, so that a request with the pump
    switched off reproduces the core result bit for bit.
    """
    if not reached:
        result = _fail_dict(mdot, choked)
        result['P_bh_MPa'] = P_fz_MPa
        result['h_feedzone_MJkg'] = h_fz_Jkg * 1e-6
        result['T_feedzone_C'] = T_feedzone_C
        result['dP_reservoir_MPa'] = dP_reservoir_MPa
        return result

    whp, T_surface_C, h_surface_MJkg = _surface(profiles)
    result = {
        'mass_flow_kgs': mdot,
        'whp_MPa': whp,
        'P_bh_MPa': P_fz_MPa,
        'h_feedzone_MJkg': h_fz_Jkg * 1e-6,
        'T_feedzone_C': T_feedzone_C,
        'h_surface_MJkg': h_surface_MJkg,
        'T_surface_C': T_surface_C,
        'dP_reservoir_MPa': dP_reservoir_MPa,
        'choked': choked,
        'profiles': profiles,
        'success': True,
    }
    result.update(pump_result_defaults())
    return result


def _add_wellhead_diagnostics(result, power_params):
    """Fill wellhead_phase/quality and the dry-steam work of a result."""
    whp = result['whp_MPa']
    phase, quality = wellhead_state(whp, result['h_surface_MJkg'])
    result['wellhead_phase'] = phase
    result['wellhead_quality'] = quality
    if np.isfinite(whp) and whp < P_CRIT_MPA:
        result['dry_steam_work_MJkg'] = power_cycle.dry_steam_specific_work(
            whp, power_params)
    else:
        result['dry_steam_work_MJkg'] = np.nan
    return result


# ====================================================================
# PUMP PRESSURE SEARCH
# ====================================================================

def _solve_pump_pressure(evaluate, target_MPa, dP0_MPa, max_dP_MPa,
                         tolerance_MPa, max_iterations=_MAX_PUMP_ITERATIONS):
    """
    Pump pressure rise that brings the upper-segment WHP to the target.

    Secant steps from the hydrostatic estimate dP0, safeguarded by a
    bisection bracket on [0, max_dP_MPa]. A march that does not reach
    the surface counts as 'below target'.

    Parameters
    ----------
    evaluate : callable
        dP [MPa] -> wellhead pressure [MPa] of the upper segment, or
        None when that march does not reach the surface.
    target_MPa : float
        Wellhead pressure to meet [MPa].
    dP0_MPa : float
        Starting guess [MPa].
    max_dP_MPa : float
        Upper bound of the search [MPa].
    tolerance_MPa : float
        Acceptable |WHP - target| [MPa].

    Returns
    -------
    (float, float or None, bool, int)
        dP [MPa], the wellhead pressure it produced (None if that
        march failed), whether the tolerance was met, and the number
        of marches used.
    """
    lo, hi = 0.0, float(max_dP_MPa)
    f_lo = f_hi = None
    x = min(max(float(dP0_MPa), lo), hi)
    previous = None
    whp = None
    residual = None
    for iteration in range(1, max_iterations + 1):
        whp = evaluate(x)
        residual = None if whp is None else whp - target_MPa
        if residual is not None and abs(residual) <= tolerance_MPa:
            return x, whp, True, iteration

        if residual is None or residual < 0:
            lo, f_lo = x, residual
        else:
            hi, f_hi = x, residual
        if hi - lo <= 1e-6:
            break

        # Secant step: through the previous point when both residuals
        # are known, otherwise with the unit slope of a liquid column
        # (the wellhead pressure rises one for one with the pump).
        x_new = None
        if residual is not None:
            if (previous is not None and previous[1] is not None
                    and previous[0] != x):
                slope = (residual - previous[1]) / (x - previous[0])
                if slope > 0:
                    x_new = x - residual / slope
            else:
                x_new = x - residual
        if x_new is None or not lo < x_new < hi:
            if f_lo is not None and f_hi is not None:
                x_new = lo - f_lo * (hi - lo) / (f_hi - f_lo)
            else:
                x_new = 0.5 * (lo + hi)
            if not lo < x_new < hi:
                x_new = 0.5 * (lo + hi)
        previous = (x, residual)
        x = x_new

    return x, whp, False, max_iterations if residual is None else iteration


# ====================================================================
# PUBLIC ENTRY POINT
# ====================================================================

def solve_pumped_state(P_fz_MPa, h_fz_Jkg, mdot, rock_temperatures,
                       well_params, pump_cfg, P_farfield_MPa,
                       T_feedzone_C, power_params=None,
                       dP_reservoir_MPa=None):
    """
    Wellhead state of a prescribed flow rate, pumped when necessary.

    Parameters
    ----------
    P_fz_MPa : float
        Feedzone (flowing bottomhole) pressure [MPa].
    h_fz_Jkg : float
        Feedzone specific enthalpy [J/kg].
    mdot : float
        Mass flow rate [kg/s].
    rock_temperatures : dict
        {depth_m: T_C} on the march grid, covering [0, depth].
    well_params : dict
        Well parameters of wellbore_physics.wellbore_simulate();
        'depth_m' and 'delta_z_m' must be ints.
    pump_cfg : config.PumpConfig
        Pump policy and envelope.
    P_farfield_MPa : float
        Far-field reservoir pressure [MPa], only used for the reported
        reservoir drawdown when dP_reservoir_MPa is not given.
    T_feedzone_C : float
        Feedzone temperature [C], reported as T_feedzone_C.
    power_params : dict or None
        power_cycle parameters for the dry-steam work diagnostic.
    dP_reservoir_MPa : float or None
        Reservoir drawdown to report [MPa]; P_farfield_MPa - P_fz_MPa
        when None.

    Returns
    -------
    dict
        The keys of reservoir.coupled_model() (mass_flow_kgs, whp_MPa,
        P_bh_MPa, h_feedzone_MJkg, T_feedzone_C, h_surface_MJkg,
        T_surface_C, dP_reservoir_MPa, choked, profiles, success),
        where the wellhead values are those of the pumped upper
        segment when pumped, plus:

        pumped : bool
        pump_depth_m : float         intake depth [m], 0 if unpumped
        P_pump_intake_MPa : float    intake pressure [MPa]
        T_pump_intake_C : float      intake temperature [C]
        dP_pump_MPa : float          pump pressure rise [MPa]
        pump_power_MWe : float       m dP / (rho1 eta) per well [MW]
        self_flow_whp_MPa : float    unpumped wellhead pressure [MPa],
                                     NaN if the unpumped march did not
                                     reach the surface
        self_flowing : bool          unpumped WHP met the floor/target
        wellhead_phase : str         one of WELLHEAD_PHASES
        wellhead_quality : float     vapour quality, NaN if not two-phase
        dry_steam_work_MJkg : float  power_cycle.dry_steam_specific_work
                                     at the WHP, NaN above P_crit
        pump_flags : list of str     subset of PUMP_FLAGS
        message : str                reason for a failed solve

        'profiles' of a pumped state is the unpumped profile from the
        feedzone to the intake followed by the pumped upper segment
        (absolute depths), so the intake depth appears twice: before
        and after the pump.
    """
    if dP_reservoir_MPa is None:
        dP_reservoir_MPa = P_farfield_MPa - P_fz_MPa
    wp = _well_params(well_params)
    delta_z = int(wp['delta_z_m'])
    mode = pump_cfg.mode

    if P_fz_MPa < 0.5:
        # reservoir.coupled_model() refuses to march from here
        return _fail_dict(mdot)

    # ---- 1. Unpumped march and the self-flow verdict ----------------
    profiles, reached, choked = _march(P_fz_MPa, h_fz_Jkg, mdot,
                                       rock_temperatures, well_params)
    result = _unpumped_dict(mdot, P_fz_MPa, h_fz_Jkg, T_feedzone_C,
                            dP_reservoir_MPa, profiles, reached, choked)
    flags = []
    whp_self = profiles[1][-1][1] if reached else np.nan
    floor = pump_cfg.min_self_flow_whp_MPa
    if pump_cfg.target_whp_MPa is not None:
        floor = max(floor, pump_cfg.target_whp_MPa)
    self_flowing = bool(reached and whp_self >= floor)
    if reached and not self_flowing:
        flags.append('self_flow_below_floor')

    result['self_flow_whp_MPa'] = float(whp_self)
    result['self_flowing'] = self_flowing
    if reached:
        _add_wellhead_diagnostics(result, power_params)

    if mode == 'never' or (mode == 'auto' and self_flowing):
        result['pump_flags'] = list(flags)
        return result

    # ---- 2. Intake search on the unpumped profile -------------------
    index, z_p, P1, T1, intake_flags = _find_intake(
        profiles, pump_cfg.npsh_margin_MPa)
    flags.extend(intake_flags)
    if index is None:
        result['pump_flags'] = list(flags)
        if pump_cfg.envelope == 'enforce':
            result['success'] = False
            result['message'] = (
                f'no liquid pump intake: the column is not single-phase '
                f'liquid with P >= P_sat + {pump_cfg.npsh_margin_MPa:g} '
                f'MPa at the feedzone ({P_fz_MPa:.2f} MPa, '
                f'{T_feedzone_C:.1f} C)')
        return result

    # ---- 3. Envelope --------------------------------------------------
    violations = []
    if z_p > pump_cfg.max_depth_m:
        flags.append('depth_limit')
        violations.append(f'intake depth {z_p:.0f} m exceeds the maximum '
                          f'{pump_cfg.max_depth_m:g} m')
    if T1 > pump_cfg.max_intake_temperature_C:
        flags.append('temperature_limit')
        violations.append(f'intake temperature {T1:.1f} C exceeds the '
                          f'maximum {pump_cfg.max_intake_temperature_C:g} C')
    if violations:
        flags.append('pump_outside_envelope')
        if pump_cfg.envelope == 'enforce':
            result.update({
                'success': False, 'pumped': True,
                'pump_depth_m': float(z_p),
                'P_pump_intake_MPa': float(P1),
                'T_pump_intake_C': float(T1),
                'pump_flags': list(flags),
                'message': ('production pump required outside the '
                            'envelope: ' + '; '.join(violations)),
            })
            return result

    # ---- 4. Intake state and target -----------------------------------
    h1_Jkg = profiles[2][index][1] * 1e6
    intake = fluid_properties_Ph(P1, h1_Jkg)
    rho1 = float(intake['density_kgm3'])
    eta = float(pump_cfg.efficiency)
    if pump_cfg.target_whp_MPa is not None:
        target = float(pump_cfg.target_whp_MPa)
    else:
        target = saturation_pressure_MPa(T1) + pump_cfg.npsh_margin_MPa

    # ---- 5. Pump pressure rise --------------------------------------
    z_p_int = int(round(z_p))
    rock_segment = {depth: rock_temperatures[depth]
                    for depth in range(0, z_p_int + delta_z, delta_z)}
    wp_segment = dict(wp)
    wp_segment['depth_m'] = z_p_int
    segment = {}

    def evaluate(dP):
        """Upper-segment wellhead pressure for a pump rise dP [MPa]."""
        h2 = h1_Jkg + dP * 1e6 / (rho1 * eta)
        if z_p_int == 0:
            # The intake is at the surface: no column to march.
            segment['profiles'] = None
            segment['choked'] = False
            segment['surface'] = (P1 + dP, h2)
            return P1 + dP
        seg_profiles, seg_reached, seg_choked = _march(
            P1 + dP, h2, mdot, rock_segment, wp_segment)
        if not seg_reached:
            return None
        segment['profiles'] = seg_profiles
        segment['choked'] = seg_choked
        segment['surface'] = None
        return seg_profiles[1][-1][1]

    dP0 = target + rho1 * GRAVITY * z_p / 1e6 - P1
    dP, whp, converged, n_marches = _solve_pump_pressure(
        evaluate, target, dP0, pump_cfg.max_dP_MPa, pump_cfg.tolerance_MPa)

    result.update({
        'pumped': True,
        'pump_depth_m': float(z_p),
        'P_pump_intake_MPa': float(P1),
        'T_pump_intake_C': float(T1),
        'dP_pump_MPa': float(dP),
        'pump_power_MWe': float(mdot * dP / (rho1 * eta)),
        'pump_flags': list(flags),
    })
    if not converged or whp is None:
        result.update({
            'success': False,
            'whp_MPa': np.nan, 'T_surface_C': np.nan,
            'h_surface_MJkg': np.nan, 'wellhead_phase': '',
            'wellhead_quality': np.nan, 'dry_steam_work_MJkg': np.nan,
            'message': (f'pump pressure search did not converge in '
                        f'{n_marches} marches (last dP {dP:.2f} MPa, '
                        f'target WHP {target:.2f} MPa)'),
        })
        return result

    # ---- 6. Assemble the pumped state ---------------------------------
    if segment['profiles'] is None:
        P_wh, h_wh_Jkg = segment['surface']
        wellhead = fluid_properties_Ph(P_wh, h_wh_Jkg)
        T_wh = wellhead['temperature_K'] - 273.15
        h_wh_MJkg = h_wh_Jkg * 1e-6
        stitched = tuple(list(p[:index + 1]) for p in profiles[:6])
        seg_choked = False
    else:
        seg_profiles = segment['profiles']
        P_wh, T_wh, h_wh_MJkg = _surface(seg_profiles)
        stitched = tuple(list(p[:index + 1]) + list(q)
                         for p, q in zip(profiles[:6], seg_profiles[:6]))
        seg_choked = segment['choked']

    result.update({
        'whp_MPa': float(P_wh),
        'T_surface_C': float(T_wh),
        'h_surface_MJkg': float(h_wh_MJkg),
        'choked': bool(seg_choked),
        'profiles': stitched + (bool(seg_choked),),
        'success': True,
    })
    _add_wellhead_diagnostics(result, power_params)
    return result
