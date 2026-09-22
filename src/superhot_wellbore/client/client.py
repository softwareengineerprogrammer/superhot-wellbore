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
       failed states are reported through notes and kept in the
       history with success=False instead of raising (unless
       SolverConfig.strict is set), and every result carries a
       success flag. Only the successful solves feed the
       interpolation; a filled-in timestep that lies nearer to a
       failed solve than to a successful one says so
       (TimestepResult.interpolated_across_failure).

    4. Pumps wells that do not self-flow. A prescribed flow rate the
       reservoir cannot lift to the surface (or lifts below the
       self-flow floor) goes through the production pump stage of
       pump.py, governed by the request's PumpConfig; the wellhead
       values of such a timestep are those of the pumped upper
       segment and the pump depth, pressure rise and power are
       reported alongside.

    5. Accepts a prescribed feedzone state. With
       ReservoirConfig.inflow = 'prescribed' the sandface pressure
       and enthalpy come from DeclineConfig.feedzone_profile instead
       of the Darcy model, so an external reservoir simulator can
       drive the wellbore directly.

Typical use::

    from superhot_wellbore.client import (
        CoupledWellboreRequest, CoupledWellboreClient)

    request = CoupledWellboreRequest.from_dict({
        'reservoir': {'P_reservoir_MPa': 30.0, 'T_reservoir_C': 450.0,
                      'transmissivity_md_m': 1000.0},
        'operating': {'control': 'whp', 'target_whp_MPa': 10.0},
        'decline': {'temperature_mode': 'linear_percent',
                    'temperature_rate_per_year': 0.5},
        'time': {'plant_lifetime_yr': 30, 'timesteps_per_year': 4},
    })
    profile = CoupledWellboreClient(request).solve_profile()

Author: superhot-wellbore GEOPHIRES client
"""

import warnings

import numpy as np

from .. import power_cycle
from .. import reservoir as core
from ..wellbore_physics import fluid_properties_Ph
from . import pump as pump_module
from . import results as results_module
from .config import CoupledWellboreRequest
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

class CoupledWellboreClient:
    """
    Run the coupled superhot reservoir-wellbore model for GEOPHIRES.

    Parameters
    ----------
    request : CoupledWellboreRequest or dict or None
        Scenario description. A dict is passed through
        CoupledWellboreRequest.from_dict(); None uses all defaults.

    Attributes
    ----------
    request : CoupledWellboreRequest
        The validated scenario.
    notes : list of str
        Human-readable remarks collected while solving (depth
        derivation, failed timesteps, interpolation). They are copied
        into every ProductionProfile the client returns.
    """

    def __init__(self, request=None):
        if isinstance(request, dict):
            request = CoupledWellboreRequest.from_dict(request)
        self.request = (request or CoupledWellboreRequest()).validate()
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
                    previous_solution=None, mass_flow_kgs=None,
                    feedzone_state=None):
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
        mass_flow_kgs : float or None
            Prescribe this flow rate instead of following the
            request's operating control, and compute the resulting
            wellhead pressure. Used to hold the flow rate constant
            along a history (OperatingConfig.hold = 'flow') and to
            follow DeclineConfig.mass_flow_profile.
        feedzone_state : (float, float) or None
            Prescribed feedzone pressure [MPa] and enthalpy [MJ/kg]
            for ReservoirConfig.inflow = 'prescribed'; the Darcy
            inflow model is then bypassed. Requires a flow rate.

        Returns
        -------
        (TimestepResult, dict)
            The translated result and the raw dict returned by
            reservoir.py or pump.py, the latter being reusable as
            previous_solution and as input to
            power_cycle.power_cycle_analysis().

        Notes
        -----
        Prescribed-flow solves go through the production pump stage
        (pump.solve_pumped_state) unless PumpConfig.mode is 'never'
        with Darcy inflow, in which case reservoir.coupled_model()
        is called exactly as before. A wellhead pressure solve
        (control = 'whp' without a held flow) never pumps.
        """
        operating = self.request.operating
        solver = self.request.solver
        pump_cfg = self.request.pump
        prescribed_inflow = self.request.reservoir.inflow == 'prescribed'
        reservoir_params = self.request.reservoir.to_reservoir_params()
        well_params = self._well_params()
        rock_temperatures = self._rock_temperatures(T_reservoir_C)

        if mass_flow_kgs is None and operating.control == 'flow':
            mass_flow_kgs = operating.mass_flow_kgs
        if prescribed_inflow and feedzone_state is None:
            feedzone_state = self._feedzone_state_at(0.0)

        raw = None
        with warnings.catch_warnings():
            # The near-critical equation of state warns freely; the
            # success flags in the returned dict are authoritative.
            warnings.simplefilter('ignore', RuntimeWarning)
            try:
                if mass_flow_kgs is None:
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
                elif prescribed_inflow:
                    raw = self._solve_prescribed_inflow(
                        mass_flow_kgs, P_reservoir_MPa, feedzone_state,
                        rock_temperatures, well_params)
                elif pump_cfg.mode == 'never':
                    raw = core.coupled_model(
                        mass_flow_rate=mass_flow_kgs,
                        P_reservoir_MPa=P_reservoir_MPa,
                        T_reservoir_C=T_reservoir_C,
                        rock_temperatures=rock_temperatures,
                        reservoir_params=reservoir_params,
                        well_params=well_params)
                else:
                    raw = self._solve_darcy_pumped(
                        mass_flow_kgs, P_reservoir_MPa, T_reservoir_C,
                        rock_temperatures, reservoir_params, well_params)
            except Exception as exc:
                result = TimestepResult(
                    P_reservoir_MPa=float(P_reservoir_MPa),
                    T_reservoir_C=float(T_reservoir_C),
                    solved=True,
                    message=f'{type(exc).__name__}: {exc}')
                return result, None

        result = self._translate(raw, P_reservoir_MPa, T_reservoir_C)
        if result.success:
            (result.power_MWe, result.cycle, result.eta_utilization,
             result.exergy_rate_MW) = self._power_cycle(raw)
        return result, raw

    def _solve_darcy_pumped(self, mass_flow_kgs, P_reservoir_MPa,
                            T_reservoir_C, rock_temperatures,
                            reservoir_params, well_params):
        """
        Darcy inflow followed by the pump stage.

        The feedzone state is what reservoir.coupled_model() computes
        (reservoir.bottomhole_pressure); the march from there is
        delegated to pump.solve_pumped_state, which reproduces
        coupled_model()'s values for a self-flowing well.
        """
        bh = core.bottomhole_pressure(
            mass_flow_kgs, P_reservoir_MPa, T_reservoir_C,
            reservoir_params, well_params)
        return pump_module.solve_pumped_state(
            P_fz_MPa=bh['P_bh_MPa'],
            h_fz_Jkg=bh['h_feedzone_Jkg'],
            mdot=mass_flow_kgs,
            rock_temperatures=rock_temperatures,
            well_params=well_params,
            pump_cfg=self.request.pump,
            P_farfield_MPa=P_reservoir_MPa,
            T_feedzone_C=bh['T_feedzone_C'],
            power_params=self.request.power_cycle.to_params(),
            dP_reservoir_MPa=bh['dP_reservoir_MPa'])

    def _solve_prescribed_inflow(self, mass_flow_kgs, P_reservoir_MPa,
                                 feedzone_state, rock_temperatures,
                                 well_params):
        """
        Pump stage from a prescribed feedzone state, no Darcy model.

        The feedzone temperature is derived from the prescribed
        (P, h) with the same equation of state the march uses.
        """
        P_fz_MPa, h_fz_MJkg = (float(v) for v in feedzone_state)
        h_fz_Jkg = h_fz_MJkg * 1e6
        feedzone = fluid_properties_Ph(P_fz_MPa, h_fz_Jkg)
        return pump_module.solve_pumped_state(
            P_fz_MPa=P_fz_MPa,
            h_fz_Jkg=h_fz_Jkg,
            mdot=mass_flow_kgs,
            rock_temperatures=rock_temperatures,
            well_params=well_params,
            pump_cfg=self.request.pump,
            P_farfield_MPa=P_reservoir_MPa,
            T_feedzone_C=feedzone['temperature_K'] - 273.15,
            power_params=self.request.power_cycle.to_params())

    def _feedzone_state_at(self, time_yr):
        """Prescribed (P_fz_MPa, h_fz_MJkg) at one time, or None."""
        states = self.request.decline.feedzone_states([float(time_yr)])
        if states is None:
            return None
        return float(states[0][0]), float(states[1][0])

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

        # Pump stage keys (pump.solve_pumped_state); absent from a
        # plain reservoir.py dict, which describes a self-flowing well.
        result.pumped = bool(raw.get('pumped', False))
        result.pump_depth_m = _as_float(raw.get('pump_depth_m', 0.0))
        result.P_pump_intake_MPa = _as_float(raw.get('P_pump_intake_MPa'))
        result.T_pump_intake_C = _as_float(raw.get('T_pump_intake_C'))
        result.dP_pump_MPa = _as_float(raw.get('dP_pump_MPa', 0.0))
        result.pump_power_MWe = _as_float(raw.get('pump_power_MWe', 0.0))
        result.self_flow_whp_MPa = _as_float(raw.get('self_flow_whp_MPa'))
        result.self_flowing = bool(raw.get('self_flowing', False))
        result.wellhead_phase = str(raw.get('wellhead_phase') or '')
        result.wellhead_quality = _as_float(raw.get('wellhead_quality'))
        result.dry_steam_work_MJkg = _as_float(
            raw.get('dry_steam_work_MJkg'))
        result.pump_flags = [str(flag) for flag in raw.get('pump_flags')
                             or []]

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
        elif raw.get('message'):
            result.message = str(raw['message'])
        else:
            result.message = ('the coupled model did not reach the '
                              'surface at this reservoir state')

        if result.choked:
            if (self.request.operating.control == 'flow'
                    and 'choked_flow' not in result.pump_flags):
                # A choked prescribed-flow march: the wellhead values
                # above the choke point are approximate. Reported on
                # the per-timestep flag channel (pump.PUMP_FLAGS) so
                # that the caller can see and enforce it.
                result.pump_flags.append('choked_flow')
            if not result.message:
                result.message = ('choke limited: the wellbore reached '
                                  'the local sound speed before the '
                                  'target wellhead pressure')
        return result

    def _power_cycle(self, raw):
        """
        Surface power cycle of a solved state, via power_cycle.py with
        the request's PowerCycleConfig.

        Returns
        -------
        (power_MWe, cycle, eta_utilization, exergy_rate_MW)
            Gross turbine power [MWe], selected cycle ('binary' or
            'flash'), utilization efficiency [-] and exergetic power
            of the wellhead stream [MW]; NaN and '' when the analysis
            fails.
        """
        failed = (float('nan'), '', float('nan'), float('nan'))
        if not raw:
            return failed
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                analysis = power_cycle.power_cycle_analysis(
                    raw, self.request.power_cycle.to_params())
        except Exception:
            return failed
        if not analysis or not analysis.get('success', False):
            return failed
        return (_as_float(analysis.get('power_MWe')),
                analysis.get('cycle') or '',
                _as_float(analysis.get('eta_utilization')),
                _as_float(analysis.get('exergy_rate_MW')))

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

        decline = self.request.decline
        pressures = decline.pressures_MPa(
            times, self.request.reservoir.P_reservoir_MPa)
        temperatures = decline.temperatures_C(
            times, self.request.reservoir.T_reservoir_C)
        # Prescribed flow rate series (mass_flow_profile), if any, and
        # prescribed feedzone states (inflow = 'prescribed'), if any.
        flows = decline.mass_flows_kgs(times)
        feedzone_states = decline.feedzone_states(times)

        indices = self._solve_indices(times.size)
        notes = list(self.notes)
        solved = {}
        previous = None
        operating = self.request.operating
        hold_flow = (operating.control == 'whp'
                     and operating.hold == 'flow')
        held_flow_kgs = None

        for index in indices:
            flow_kgs = held_flow_kgs
            if flows is not None:
                flow_kgs = float(flows[index])
            feedzone_state = None
            if feedzone_states is not None:
                feedzone_state = (float(feedzone_states[0][index]),
                                  float(feedzone_states[1][index]))
            result, raw = self.solve_state(
                pressures[index], temperatures[index],
                previous_solution=previous,
                mass_flow_kgs=flow_kgs,
                feedzone_state=feedzone_state)
            result.time_yr = float(times[index])
            solved[index] = result

            if result.success:
                previous = raw if solver.reuse_previous_solution else None
                if hold_flow and held_flow_kgs is None:
                    held_flow_kgs = result.mass_flow_kgs
                    if len(indices) > 1:
                        notes.append(
                            f'Flow rate of {held_flow_kgs:.2f} kg/s '
                            f'solved at t = {times[index]:.2f} yr for '
                            f'a wellhead pressure of '
                            f'{operating.target_whp_MPa:g} MPa is held '
                            f'constant over the history; the wellhead '
                            f'pressure responds to the reservoir '
                            f'decline')
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
        # The dry-steam turbine work is a function of the wellhead
        # pressure alone, and is undefined (NaN) at a supercritical
        # wellhead. Interpolating it linearly spreads that NaN over
        # every interpolated timestep between a supercritical solve
        # and a sub-critical one, which leaves a sub-critical
        # wellhead without the number its flash plant needs. Derive
        # it from each timestep's own pressure instead.
        power_params = self.request.power_cycle.to_params()
        for ts in timesteps:
            ts.dry_steam_work_MJkg = power_cycle.dry_steam_specific_work(
                ts.whp_MPa, power_params)

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
        if profile.any_pumped:
            pumped = [ts for ts in profile.timesteps
                      if ts.success and ts.pumped]
            profile.notes.append(
                f'Production pump used at {len(pumped)} of '
                f'{profile.n_timesteps} timesteps (intake depth up to '
                f'{max(ts.pump_depth_m for ts in pumped):.0f} m, '
                f'pump power up to '
                f'{max(ts.pump_power_MWe for ts in pumped):.3f} MW per '
                f'well); the wellhead values are those of the pumped '
                f'well')
            flags = sorted({flag for ts in pumped for flag in ts.pump_flags})
            if flags:
                profile.notes.append(
                    'Pump flags raised: ' + ', '.join(flags))
        return profile

    def _solve_indices(self, n_timesteps):
        """
        Time indices at which the coupled model is actually solved.

        The first and last timesteps are always included so that the
        interpolation never extrapolates. A steady reservoir (no
        decline) is solved once, since every timestep has the same
        state.
        """
        if self.request.decline.is_steady():
            return [0]
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
    request : CoupledWellboreRequest or dict
        Scenario description.
    time_yr : sequence of float or None
        Optional explicit time vector [yr].

    Returns
    -------
    ProductionProfile
    """
    return CoupledWellboreClient(request).solve_profile(time_yr=time_yr)


def solve_steady_state(request):
    """
    Solve the initial steady state in one call.

    Parameters
    ----------
    request : CoupledWellboreRequest or dict
        Scenario description.

    Returns
    -------
    TimestepResult
    """
    return CoupledWellboreClient(request).solve_steady_state()


# Re-exported so that callers can reach the result types through the
# client module alone.
GEOPHIRES_RESERVOIR_MODEL_UPP = results_module.GEOPHIRES_RESERVOIR_MODEL_UPP
