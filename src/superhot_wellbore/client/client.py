# -*- coding: utf-8 -*-
"""
Superhot Wellbore Client
=========================

Stable facade over the superhot-wellbore core modules, built so that
an external program - GEOPHIRES in particular - can obtain a
production history without knowing anything about reservoir.py,
wellbore_physics.py or their parameter dictionaries.

The client does three things the raw modules leave to the caller:

    1. Resolves the well geometry. When no depth is given it is
       derived from the initial reservoir pressure with
       reservoir.depth_for_pressure(), and it is always snapped to a
       whole number of wellbore integration steps so that the rock
       temperature profile lines up with the simulation grid.

    2. Walks the reservoir state through time. The superhot-wellbore
       model is steady state, while a GEOPHIRES reservoir model must
       return a production history. The client evaluates the decline
       laws in DeclineConfig and re-solves the coupled
       reservoir-wellbore model at the resulting (P, T) states, so
       the history is produced by the physics rather than by a
       curve fit. Because each solve costs seconds, at most
       SolverConfig.max_solve_points solves are performed and the
       remaining timesteps are linearly interpolated.

    3. Isolates the caller from solver failures. Runtime warnings
       from the near-critical equation of state are suppressed,
       failed states are reported through notes and interpolated from
       the neighbouring successful solves instead of raising (unless
       SolverConfig.strict is set), and every result carries a
       success flag.

Typical use::

    from superhot_wellbore.client import (
        SuperhotRequest, SuperhotWellboreClient)

    request = SuperhotRequest.from_dict({
        'reservoir': {'P_reservoir_MPa': 30.0, 'T_reservoir_C': 450.0,
                      'transmissivity_md_m': 1000.0},
        'operating': {'control': 'whp', 'target_whp_MPa': 10.0},
        'decline': {'temperature_mode': 'linear_percent',
                    'temperature_rate_per_year': 0.5},
        'time': {'plant_lifetime_yr': 30, 'timesteps_per_year': 4},
    })
    profile = SuperhotWellboreClient(request).solve_profile()

Author: superhot-wellbore GEOPHIRES client
"""

import warnings

import numpy as np

from .. import power_cycle
from .. import reservoir as core
from . import results as results_module
from .config import SuperhotRequest
from .results import (ProductionProfile, TimestepResult,
                      interpolate_timesteps)


# ====================================================================
# HELPERS
# ====================================================================

def _as_float(value):
    """Convert a possibly missing numeric value to float, None -> NaN."""
    if value is None:
        return float('nan')
    try:
        return float(value)
    except (TypeError, ValueError):
        return float('nan')


def _is_finite(value):
    """True if value is a finite number."""
    return value is not None and np.isfinite(value)


# ====================================================================
# CLIENT
# ====================================================================

class SuperhotWellboreClient:
    """
    Run the coupled superhot reservoir-wellbore model for GEOPHIRES.

    Parameters
    ----------
    request : SuperhotRequest or dict or None
        Scenario description. A dict is passed through
        SuperhotRequest.from_dict(); None uses all defaults.

    Attributes
    ----------
    request : SuperhotRequest
        The validated scenario.
    notes : list of str
        Human-readable remarks collected while solving (depth
        derivation, failed timesteps, interpolation). They are copied
        into every ProductionProfile the client returns.
    """

    def __init__(self, request=None):
        if isinstance(request, dict):
            request = SuperhotRequest.from_dict(request)
        self.request = (request or SuperhotRequest()).validate()
        self.notes = []
        self._depth_m = None
        self._rock_cache = {}

    # ----------------------------------------------------------------
    # Well geometry
    # ----------------------------------------------------------------

    @property
    def depth_m(self):
        """
        Resolved feedzone depth [m].

        Either the user-supplied depth or one derived from the initial
        reservoir pressure, in both cases snapped to a whole number of
        wellbore integration steps.
        """
        if self._depth_m is None:
            self._depth_m = self._resolve_depth()
        return self._depth_m

    def _resolve_depth(self):
        """Derive and snap the well depth, recording what was done."""
        well = self.request.well
        depth = well.depth_m

        if depth is None:
            depth = core.depth_for_pressure(
                self.request.reservoir.P_reservoir_MPa)
            self.notes.append(
                f'Well depth derived from the initial reservoir '
                f'pressure ({self.request.reservoir.P_reservoir_MPa:g} '
                f'MPa) as {depth:.0f} m')

        # The wellbore model marches on an integer depth grid, so the
        # depth must be a whole number of integration steps.
        step = int(round(float(well.delta_z_m)))
        snapped = max(int(round(float(depth) / step)), 1) * step
        if abs(snapped - depth) > 1e-9:
            self.notes.append(
                f'Well depth snapped from {depth:.1f} m to '
                f'{snapped:.1f} m, a whole number of {step:g} m '
                f'wellbore integration steps')
        return float(snapped)

    def _well_params(self):
        """well_params dict for the resolved depth."""
        return self.request.well.to_well_params(self.depth_m)

    # ----------------------------------------------------------------
    # Formation temperature profile
    # ----------------------------------------------------------------

    def _rock_temperatures(self, T_reservoir_C):
        """
        Formation temperature profile for a given reservoir temperature.

        Results are cached because the linear profile has to be
        rebuilt whenever the reservoir temperature declines, while the
        boiling-point and user profiles are time invariant.
        """
        config = self.request.rock_temperature
        well_params = self._well_params()

        if config.mode == 'linear':
            key = ('linear', round(float(T_reservoir_C), 4))
        else:
            key = (config.mode,)

        if key not in self._rock_cache:
            if config.mode == 'linear':
                profile = core.rock_temperature_linear(
                    well_params, T_reservoir_C, config.T_surface_C)
            elif config.mode == 'boiling':
                profile = core.rock_temperature_boiling(
                    well_params, config.surface_pressure_MPa,
                    config.T_surface_C)
            else:
                profile = core.rock_temperature_from_user(
                    well_params, config.user_profile())
            self._rock_cache[key] = profile

        return self._rock_cache[key]

    # ----------------------------------------------------------------
    # Single coupled-model solve
    # ----------------------------------------------------------------

    def solve_state(self, P_reservoir_MPa, T_reservoir_C,
                    previous_solution=None):
        """
        Solve the coupled model at one far-field reservoir state.

        Parameters
        ----------
        P_reservoir_MPa : float
            Far-field reservoir pressure [MPa].
        T_reservoir_C : float
            Far-field reservoir temperature [C].
        previous_solution : dict or None
            Output of an earlier solve, used as a starting point by
            reservoir.solve_flow_for_whp().

        Returns
        -------
        (TimestepResult, dict)
            The translated result and the raw dict returned by
            reservoir.py, the latter being reusable as
            previous_solution and as input to
            power_cycle.power_cycle_analysis().
        """
        operating = self.request.operating
        solver = self.request.solver
        reservoir_params = self.request.reservoir.to_reservoir_params()
        well_params = self._well_params()
        rock_temperatures = self._rock_temperatures(T_reservoir_C)

        raw = None
        with warnings.catch_warnings():
            # The near-critical equation of state warns freely; the
            # success flags in the returned dict are authoritative.
            warnings.simplefilter('ignore', RuntimeWarning)
            try:
                if operating.control == 'whp':
                    raw = core.solve_flow_for_whp(
                        target_whp_MPa=operating.target_whp_MPa,
                        P_reservoir_MPa=P_reservoir_MPa,
                        T_reservoir_C=T_reservoir_C,
                        rock_temperatures=rock_temperatures,
                        reservoir_params=reservoir_params,
                        well_params=well_params,
                        previous_solution=previous_solution,
                        tolerance_MPa=solver.tolerance_MPa,
                        verbose=solver.verbose)
                else:
                    raw = core.coupled_model(
                        mass_flow_rate=operating.mass_flow_kgs,
                        P_reservoir_MPa=P_reservoir_MPa,
                        T_reservoir_C=T_reservoir_C,
                        rock_temperatures=rock_temperatures,
                        reservoir_params=reservoir_params,
                        well_params=well_params)
            except Exception as exc:
                result = TimestepResult(
                    P_reservoir_MPa=float(P_reservoir_MPa),
                    T_reservoir_C=float(T_reservoir_C),
                    solved=True,
                    message=f'{type(exc).__name__}: {exc}')
                return result, None

        result = self._translate(raw, P_reservoir_MPa, T_reservoir_C)
        if result.success:
            result.power_MWe, result.cycle = self._power_diagnostic(raw)
        return result, raw

    def _translate(self, raw, P_reservoir_MPa, T_reservoir_C):
        """Convert a reservoir.py output dict into a TimestepResult."""
        result = TimestepResult(
            P_reservoir_MPa=float(P_reservoir_MPa),
            T_reservoir_C=float(T_reservoir_C),
            solved=True)

        if not raw:
            result.message = 'the coupled model returned no result'
            return result

        result.mass_flow_kgs = _as_float(
            raw.get('mass_flow_kgs', raw.get('flow_rate_kg_s')))
        result.whp_MPa = _as_float(raw.get('whp_MPa'))
        result.T_wellhead_C = _as_float(raw.get('T_surface_C'))
        result.h_wellhead_MJkg = _as_float(raw.get('h_surface_MJkg'))
        result.T_feedzone_C = _as_float(raw.get('T_feedzone_C'))
        result.h_feedzone_MJkg = _as_float(raw.get('h_feedzone_MJkg'))
        result.P_bh_MPa = _as_float(raw.get('P_bh_MPa'))
        result.dP_reservoir_MPa = _as_float(
            raw.get('dP_reservoir_MPa'))
        result.choked = bool(raw.get('choked', False))
        result.converged = bool(
            raw.get('converged', raw.get('success', False)))

        result.success = bool(raw.get('success', False))
        if result.success:
            # Guard against a nominally successful solve that did not
            # produce the quantities GEOPHIRES needs.
            missing = [name for name, value in (
                ('mass flow rate', result.mass_flow_kgs),
                ('wellhead pressure', result.whp_MPa),
                ('wellhead temperature', result.T_wellhead_C),
                ('feedzone temperature', result.T_feedzone_C))
                if not _is_finite(value)]
            if missing:
                result.success = False
                result.message = ('the coupled model succeeded but did '
                                  'not report ' + ', '.join(missing))
            elif result.mass_flow_kgs <= 0:
                result.success = False
                result.message = ('the coupled model reported a '
                                  'non-positive mass flow rate')
        else:
            result.message = ('the coupled model did not reach the '
                              'surface at this reservoir state')

        if result.choked and not result.message:
            result.message = ('choke limited: the wellbore reached the '
                              'local sound speed before the target '
                              'wellhead pressure')
        return result

    def _power_diagnostic(self, raw):
        """
        Gross power for reference only, via power_cycle.py.

        GEOPHIRES computes electricity generation with its own surface
        plant model; this value exists so that the two can be
        compared.
        """
        if not raw:
            return float('nan'), ''
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                analysis = power_cycle.power_cycle_analysis(raw)
        except Exception:
            return float('nan'), ''
        if not analysis or not analysis.get('success', False):
            return float('nan'), ''
        return (_as_float(analysis.get('power_MWe')),
                analysis.get('cycle') or '')

    # ----------------------------------------------------------------
    # Steady state
    # ----------------------------------------------------------------

    def solve_steady_state(self):
        """
        Solve the coupled model at the initial reservoir state.

        Returns
        -------
        TimestepResult
            Solution at t = 0.
        """
        result, _ = self.solve_state(
            self.request.reservoir.P_reservoir_MPa,
            self.request.reservoir.T_reservoir_C)
        return result

    # ----------------------------------------------------------------
    # Production history
    # ----------------------------------------------------------------

    def solve_profile(self, time_yr=None):
        """
        Solve the coupled model along the declining reservoir path.

        Parameters
        ----------
        time_yr : sequence of float or None
            Time vector [yr]. When None the vector is built from the
            request's TimeConfig, which mirrors the GEOPHIRES
            discretisation. Pass GEOPHIRES' own
            Reservoir.timevector here to guarantee an exact
            element-by-element match.

        Returns
        -------
        ProductionProfile
            One entry per element of the time vector.

        Raises
        ------
        RuntimeError
            If SolverConfig.strict is set and any solve fails.
        """
        solver = self.request.solver

        if time_yr is None:
            times = self.request.time.time_vector_yr()
        else:
            times = np.asarray(time_yr, dtype=float)
        if times.size == 0:
            raise ValueError('The time vector must not be empty')

        pressures = self.request.decline.pressures_MPa(
            times, self.request.reservoir.P_reservoir_MPa)
        temperatures = self.request.decline.temperatures_C(
            times, self.request.reservoir.T_reservoir_C)

        indices = self._solve_indices(times.size)
        notes = list(self.notes)
        solved = {}
        previous = None

        for index in indices:
            result, raw = self.solve_state(
                pressures[index], temperatures[index],
                previous_solution=previous)
            result.time_yr = float(times[index])
            solved[index] = result

            if result.success:
                previous = raw if solver.reuse_previous_solution else None
            else:
                previous = None
                message = (f'Coupled model failed at t = '
                           f'{times[index]:.2f} yr '
                           f'(P = {pressures[index]:.2f} MPa, '
                           f'T = {temperatures[index]:.1f} C): '
                           f'{result.message}')
                if solver.strict:
                    raise RuntimeError(message)
                notes.append(message)

            if solver.verbose:
                print(f'  t = {result.time_yr:6.2f} yr  '
                      f'P_res = {pressures[index]:5.2f} MPa  '
                      f'T_res = {temperatures[index]:6.1f} C  ->  '
                      f'm = {result.mass_flow_kgs:6.2f} kg/s  '
                      f'WHP = {result.whp_MPa:5.2f} MPa  '
                      f'T_wh = {result.T_wellhead_C:6.1f} C  '
                      f'{"" if result.success else "(failed)"}')

        timesteps = interpolate_timesteps(times, solved,
                                          pressures, temperatures)

        n_solved = len(indices)
        if n_solved < times.size:
            notes.append(
                f'Coupled model solved at {n_solved} of '
                f'{times.size} timesteps; the remaining values were '
                f'linearly interpolated (solver.max_solve_points = '
                f'{solver.max_solve_points})')
        if not any(ts.success for ts in timesteps):
            notes.append('No timestep produced a usable solution; '
                          'check the reservoir state, transmissivity '
                          'and target wellhead pressure')

        profile = ProductionProfile(request=self.request,
                                    depth_m=self.depth_m,
                                    timesteps=timesteps,
                                    notes=notes)
        if profile.any_choked:
            profile.notes.append(
                'At least one timestep is choke limited, so the '
                'reported wellhead pressure exceeds the target')
        return profile

    def _solve_indices(self, n_timesteps):
        """
        Time indices at which the coupled model is actually solved.

        The first and last timesteps are always included so that the
        interpolation never extrapolates.
        """
        limit = self.request.solver.max_solve_points
        if limit is None or limit <= 0 or limit >= n_timesteps:
            return list(range(n_timesteps))
        limit = max(int(limit), 2)
        picked = np.linspace(0, n_timesteps - 1, limit)
        return sorted({int(round(index)) for index in picked})


# ====================================================================
# CONVENIENCE
# ====================================================================

def solve_profile(request, time_yr=None):
    """
    Solve a production history in one call.

    Parameters
    ----------
    request : SuperhotRequest or dict
        Scenario description.
    time_yr : sequence of float or None
        Optional explicit time vector [yr].

    Returns
    -------
    ProductionProfile
    """
    return SuperhotWellboreClient(request).solve_profile(time_yr=time_yr)


def solve_steady_state(request):
    """
    Solve the initial steady state in one call.

    Parameters
    ----------
    request : SuperhotRequest or dict
        Scenario description.

    Returns
    -------
    TimestepResult
    """
    return SuperhotWellboreClient(request).solve_steady_state()


# Re-exported so that callers can reach the result types through the
# client module alone.
GEOPHIRES_RESERVOIR_MODEL_UPP = results_module.GEOPHIRES_RESERVOIR_MODEL_UPP
