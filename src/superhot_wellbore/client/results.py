# -*- coding: utf-8 -*-
"""
Client Results and Their GEOPHIRES Mapping
===========================================

Result containers returned by the client, plus the translation of a
superhot production history into GEOPHIRES input parameters and
output arrays.

Two distinct temperatures are carried through the results, and
keeping them apart is essential when feeding GEOPHIRES:

    reservoir_output_temperature_C
        Feedzone temperature after Darcy drawdown and isenthalpic
        expansion into the well. This is the quantity a GEOPHIRES
        reservoir model reports as the reservoir output temperature
        (Reservoir.Tresoutput).

    production_temperature_C
        Wellhead temperature after wellbore heat loss, friction and
        flashing. This is what GEOPHIRES calls the produced
        temperature (WellBores.ProducedTemperature) and normally
        derives itself with Ramey's model; here it comes from the
        superhot wellbore simulator instead.

The difference between the two is reported as the production
wellbore temperature drop, which lets a plain (unmodified) GEOPHIRES
run reproduce the wellhead state with Ramey's model switched off.

Which of the two is written into a GEOPHIRES temperature profile is
selected with the profile_temperature argument:

    'wellhead' (default)
        Write the wellhead temperature and set the production
        wellbore temperature drop to zero. GEOPHIRES then reproduces
        the superhot wellhead state exactly. This matters because
        superhot wells lose considerably more than the 50 C that
        GEOPHIRES accepts as a wellbore temperature drop, so routing
        the drop through GEOPHIRES would truncate it.

    'feedzone'
        Write the feedzone temperature, the physically stricter
        reading of 'reservoir output temperature', and let GEOPHIRES
        apply the reported drop. Use it when the reservoir and
        wellbore contributions must stay visible separately in the
        GEOPHIRES output, and expect the drop to be clamped.

Author: superhot-wellbore GEOPHIRES client
"""

import dataclasses
from dataclasses import dataclass, field
from typing import Any, List

import numpy as np

from ..wellbore_physics import P_CRIT_MPA
from . import units
from .pump import wellhead_state


# ====================================================================
# GEOPHIRES CONSTANTS
# ====================================================================

# 'User-Provided Temperature Profile' reservoir model
GEOPHIRES_RESERVOIR_MODEL_UPP = 5

# Bounds enforced by GEOPHIRES on the parameters written below
# (see geophires_x/Reservoir.py and geophires_x/WellBores.py)
GEOPHIRES_TMAX_LIMITS_C = (50.0, 600.0)
GEOPHIRES_TEMPDROP_LIMITS_C = (-5.0, 50.0)
GEOPHIRES_WELLDIAM_LIMITS_INCH = (1.0, 30.0)
GEOPHIRES_FLOWRATE_LIMITS_KGS = (1.0, 500.0)


#: Temperatures that can be written into a GEOPHIRES profile
PROFILE_TEMPERATURES = ('wellhead', 'feedzone')


def _clamp(value, limits, label, notes=None):
    """Clamp value into limits, appending an explanatory note."""
    low, high = limits
    if value is None or not np.isfinite(value):
        return value
    clamped = min(max(value, low), high)
    if notes is not None and clamped != value:
        note = (f'{label} clamped from {value:.4g} to {clamped:.4g} '
                f'to respect the GEOPHIRES range [{low:g}, {high:g}]')
        if note not in notes:
            notes.append(note)
    return clamped


def _clean(value):
    """Replace non-finite floats with None so that JSON stays valid."""
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


# ====================================================================
# SINGLE TIMESTEP
# ====================================================================

@dataclass
class TimestepResult:
    """
    Coupled reservoir-wellbore solution at one point in time.

    Units follow the superhot-wellbore convention: MPa, MJ/kg,
    degrees C, kg/s.

    Attributes
    ----------
    time_yr : float
        Time since start of production [yr].
    P_reservoir_MPa, T_reservoir_C : float
        Far-field reservoir state used for this solve.
    mass_flow_kgs : float
        Produced mass flow rate [kg/s].
    whp_MPa : float
        Wellhead pressure [MPa].
    T_wellhead_C, h_wellhead_MJkg : float
        Wellhead temperature [C] and specific enthalpy [MJ/kg].
    T_feedzone_C, h_feedzone_MJkg : float
        Feedzone temperature [C] and specific enthalpy [MJ/kg].
    P_bh_MPa : float
        Flowing bottomhole pressure [MPa].
    dP_reservoir_MPa : float
        Reservoir (Darcy) drawdown [MPa].
    power_MWe : float
        Gross turbine power from power_cycle.py [MWe], computed with
        the request's PowerCycleConfig. The GEOPHIRES superhot power
        cycle surface plant uses it; other GEOPHIRES surface plants
        report it for comparison only.
    cycle : str
        Power cycle selected by power_cycle.py ('binary' or 'flash'),
        empty when the power cycle analysis was not computed.
    eta_utilization : float
        Utilization efficiency of the power cycle at wellhead
        conditions [-].
    exergy_rate_MW : float
        Exergetic power of the wellhead stream relative to the
        ambient dead state [MW].
    choked : bool
        True if the wellbore reached the local sound speed.
    converged : bool
        True if the flow solver met the wellhead pressure target.
    success : bool
        True if a usable solution is available at this time.
    solved : bool
        True if the coupled model was actually run at this time;
        False if the values were interpolated between solves.
    message : str
        Diagnostic message, empty when nothing noteworthy happened.
    interpolated_across_failure : bool
        True on an interpolated (unsolved) timestep whose nearest
        solve in time failed: its values were carried across that
        failure from the successful solves and are an estimate the
        caller may want to distrust. Always False on a solved
        timestep.
    pumped : bool
        True if the production pump stage (pump.py) lifted the flow;
        the wellhead values then belong to the pumped upper segment.
    pump_depth_m : float
        Pump intake (setting) depth [m], 0 when not pumped.
    P_pump_intake_MPa, T_pump_intake_C : float
        Fluid state at the pump intake, NaN when not pumped.
    dP_pump_MPa : float
        Pump pressure rise [MPa], 0 when not pumped.
    pump_power_MWe : float
        Pump power per well [MW], m dP / (rho eta), 0 when not pumped.
    self_flow_whp_MPa : float
        Wellhead pressure of the unpumped well [MPa], NaN when the
        unpumped march did not reach the surface.
    self_flowing : bool
        True if the unpumped well reached the surface at or above the
        self-flow floor (PumpConfig.min_self_flow_whp_MPa).
    wellhead_phase : str
        'single_phase_liquid', 'two_phase', 'single_phase_vapor' or
        'supercritical' (wellhead pressure at or above 22.064 MPa);
        empty when unknown.
    wellhead_quality : float
        Vapour mass fraction at the wellhead [-], NaN unless the
        wellhead is two-phase.
    dry_steam_work_MJkg : float
        Gross specific turbine work of saturated steam expanded from
        the wellhead pressure [MJ/kg] (power_cycle.
        dry_steam_specific_work), NaN above the critical pressure.
    pump_flags : list of str
        Pump stage flags, a subset of pump.PUMP_FLAGS.
    """

    time_yr: float = 0.0
    P_reservoir_MPa: float = float('nan')
    T_reservoir_C: float = float('nan')
    mass_flow_kgs: float = float('nan')
    whp_MPa: float = float('nan')
    T_wellhead_C: float = float('nan')
    h_wellhead_MJkg: float = float('nan')
    T_feedzone_C: float = float('nan')
    h_feedzone_MJkg: float = float('nan')
    P_bh_MPa: float = float('nan')
    dP_reservoir_MPa: float = float('nan')
    power_MWe: float = float('nan')
    cycle: str = ''
    eta_utilization: float = float('nan')
    exergy_rate_MW: float = float('nan')
    choked: bool = False
    converged: bool = False
    success: bool = False
    solved: bool = False
    message: str = ''
    interpolated_across_failure: bool = False
    pumped: bool = False
    pump_depth_m: float = 0.0
    P_pump_intake_MPa: float = float('nan')
    T_pump_intake_C: float = float('nan')
    dP_pump_MPa: float = 0.0
    pump_power_MWe: float = 0.0
    self_flow_whp_MPa: float = float('nan')
    self_flowing: bool = False
    wellhead_phase: str = ''
    wellhead_quality: float = float('nan')
    dry_steam_work_MJkg: float = float('nan')
    pump_flags: List[str] = field(default_factory=list)

    #: Numeric fields that can be interpolated between solved states
    INTERPOLATED_FIELDS = ('mass_flow_kgs', 'whp_MPa', 'T_wellhead_C',
                           'h_wellhead_MJkg', 'T_feedzone_C',
                           'h_feedzone_MJkg', 'P_bh_MPa',
                           'dP_reservoir_MPa', 'power_MWe',
                           'eta_utilization', 'exergy_rate_MW',
                           'pump_depth_m', 'P_pump_intake_MPa',
                           'T_pump_intake_C', 'dP_pump_MPa',
                           'pump_power_MWe', 'self_flow_whp_MPa',
                           'wellhead_quality')
    # dry_steam_work_MJkg is NOT interpolated: it is a function of the
    # wellhead pressure alone and is undefined at a supercritical
    # wellhead, so interpolating it would spread that NaN into
    # sub-critical timesteps. The client derives it per timestep from
    # whp_MPa after interpolation.

    #: Flag fields copied from the nearest successful solve
    COPIED_FIELDS = ('cycle', 'choked', 'converged', 'pumped',
                     'self_flowing', 'wellhead_phase')

    def to_dict(self):
        """Return a JSON-safe dict (non-finite floats become None)."""
        return {k: _clean(v)
                for k, v in dataclasses.asdict(self).items()}


# ====================================================================
# PRODUCTION HISTORY
# ====================================================================

@dataclass
class ProductionProfile:
    """
    Production history over the plant lifetime.

    This is the object handed to GEOPHIRES, either in process (by the
    GEOPHIRES coupled inflow-wellbore production wellbore model) or
    through the files written by export.py.
    """

    request: Any = None
    depth_m: float = float('nan')
    timesteps: List[TimestepResult] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    # ----------------------------------------------------------------
    # Series accessors
    # ----------------------------------------------------------------

    def _series(self, attribute):
        """Collect one attribute across all timesteps as a list."""
        return [getattr(ts, attribute) for ts in self.timesteps]

    @property
    def time_yr(self):
        """Time vector [yr]."""
        return self._series('time_yr')

    @property
    def reservoir_output_temperature_C(self):
        """Feedzone temperature history [C] (GEOPHIRES Tresoutput)."""
        return self._series('T_feedzone_C')

    @property
    def production_temperature_C(self):
        """Wellhead temperature history [C] (ProducedTemperature)."""
        return self._series('T_wellhead_C')

    @property
    def mass_flow_kgs(self):
        """Produced mass flow rate history [kg/s]."""
        return self._series('mass_flow_kgs')

    @property
    def whp_MPa(self):
        """Wellhead pressure history [MPa]."""
        return self._series('whp_MPa')

    @property
    def h_wellhead_MJkg(self):
        """Wellhead specific enthalpy history [MJ/kg]."""
        return self._series('h_wellhead_MJkg')

    @property
    def dP_reservoir_MPa(self):
        """Reservoir drawdown history [MPa]."""
        return self._series('dP_reservoir_MPa')

    @property
    def power_MWe(self):
        """Gross turbine power history of the power cycle [MWe]."""
        return self._series('power_MWe')

    @property
    def cycle(self):
        """Power cycle selected at each timestep ('binary' or 'flash')."""
        return self._series('cycle')

    @property
    def eta_utilization(self):
        """Utilization efficiency history of the power cycle [-]."""
        return self._series('eta_utilization')

    @property
    def exergy_rate_MW(self):
        """Exergetic power history of the wellhead stream [MW]."""
        return self._series('exergy_rate_MW')

    @property
    def P_reservoir_MPa(self):
        """Far-field reservoir pressure history [MPa]."""
        return self._series('P_reservoir_MPa')

    @property
    def T_reservoir_C(self):
        """Far-field reservoir temperature history [C]."""
        return self._series('T_reservoir_C')

    @property
    def pumped(self):
        """Whether the production pump lifted the flow, per timestep."""
        return self._series('pumped')

    @property
    def pump_power_MWe(self):
        """Production pump power history per well [MW]."""
        return self._series('pump_power_MWe')

    @property
    def pump_depth_m(self):
        """Production pump intake depth history [m]."""
        return self._series('pump_depth_m')

    @property
    def dP_pump_MPa(self):
        """Production pump pressure rise history [MPa]."""
        return self._series('dP_pump_MPa')

    @property
    def self_flow_whp_MPa(self):
        """Unpumped wellhead pressure history [MPa] (NaN allowed)."""
        return self._series('self_flow_whp_MPa')

    @property
    def self_flowing(self):
        """Whether the unpumped well met the self-flow floor, per step."""
        return self._series('self_flowing')

    @property
    def wellhead_phase(self):
        """Wellhead phase label history."""
        return self._series('wellhead_phase')

    @property
    def wellhead_quality(self):
        """Wellhead vapour quality history [-] (NaN if single phase)."""
        return self._series('wellhead_quality')

    @property
    def dry_steam_work_MJkg(self):
        """Dry-steam specific turbine work history [MJ/kg]."""
        return self._series('dry_steam_work_MJkg')

    @property
    def pump_flags(self):
        """Pump flags per timestep (list of lists of str)."""
        return self._series('pump_flags')

    # ----------------------------------------------------------------
    # Summary statistics
    # ----------------------------------------------------------------

    @property
    def n_timesteps(self):
        """Number of timesteps in the history."""
        return len(self.timesteps)

    @property
    def n_solved(self):
        """Number of timesteps where the coupled model was run."""
        return sum(1 for ts in self.timesteps if ts.solved)

    @property
    def n_failed(self):
        """Number of timesteps without a usable solution."""
        return sum(1 for ts in self.timesteps if not ts.success)

    @property
    def any_choked(self):
        """True if any timestep hit the choked-flow limit."""
        return any(ts.choked for ts in self.timesteps)

    @property
    def any_pumped(self):
        """True if the production pump is used at any usable timestep."""
        return any(ts.pumped for ts in self.timesteps if ts.success)

    @property
    def self_flowing_fraction(self):
        """
        Fraction of usable timesteps at which the well self-flows.

        NaN when no timestep is usable.
        """
        usable = [ts for ts in self.timesteps if ts.success]
        if not usable:
            return float('nan')
        return sum(1 for ts in usable if ts.self_flowing) / len(usable)

    @property
    def dominant_wellhead_phase(self):
        """
        Wellhead phase label holding the most usable timesteps.

        Empty when no timestep is usable or none reports a phase.
        """
        counts = {}
        for ts in self.timesteps:
            if ts.success and ts.wellhead_phase:
                counts[ts.wellhead_phase] = counts.get(ts.wellhead_phase,
                                                       0) + 1
        if not counts:
            return ''
        return max(counts, key=lambda phase: (counts[phase], phase))

    @property
    def initial(self):
        """First timestep with a usable solution, or None."""
        for ts in self.timesteps:
            if ts.success:
                return ts
        return None

    def _mean(self, attribute):
        """Mean of an attribute over successful timesteps."""
        values = [getattr(ts, attribute) for ts in self.timesteps
                  if ts.success and np.isfinite(getattr(ts, attribute))]
        if not values:
            return float('nan')
        return float(np.mean(values))

    def _max(self, attribute):
        """Maximum of an attribute over successful timesteps."""
        values = [getattr(ts, attribute) for ts in self.timesteps
                  if ts.success and np.isfinite(getattr(ts, attribute))]
        if not values:
            return float('nan')
        return float(np.max(values))

    @property
    def mean_wellbore_temperature_drop_C(self):
        """Mean feedzone-to-wellhead temperature drop [C]."""
        drops = [ts.T_feedzone_C - ts.T_wellhead_C
                 for ts in self.timesteps
                 if ts.success and np.isfinite(ts.T_feedzone_C)
                 and np.isfinite(ts.T_wellhead_C)]
        if not drops:
            return float('nan')
        return float(np.mean(drops))

    def summary(self):
        """
        Return a compact dict of headline numbers.

        Useful for logging and for the command-line client.
        """
        first = self.initial
        return {
            'name': getattr(self.request, 'name', None),
            'depth_m': _clean(self.depth_m),
            'n_timesteps': self.n_timesteps,
            'n_solved': self.n_solved,
            'n_failed': self.n_failed,
            'any_choked': self.any_choked,
            'initial_mass_flow_kgs': _clean(
                first.mass_flow_kgs if first else float('nan')),
            'initial_whp_MPa': _clean(
                first.whp_MPa if first else float('nan')),
            'initial_production_temperature_C': _clean(
                first.T_wellhead_C if first else float('nan')),
            'initial_reservoir_output_temperature_C': _clean(
                first.T_feedzone_C if first else float('nan')),
            'mean_mass_flow_kgs': _clean(self._mean('mass_flow_kgs')),
            'mean_production_temperature_C': _clean(
                self._mean('T_wellhead_C')),
            'mean_wellbore_temperature_drop_C': _clean(
                self.mean_wellbore_temperature_drop_C),
            'initial_power_cycle': first.cycle if first else None,
            'mean_power_MWe': _clean(self._mean('power_MWe')),
            'any_pumped': self.any_pumped,
            'self_flowing_fraction': _clean(self.self_flowing_fraction),
            'max_pump_depth_m': _clean(self._max('pump_depth_m')),
            'mean_pump_power_MWe': _clean(self._mean('pump_power_MWe')),
            'initial_self_flow_whp_MPa': _clean(
                first.self_flow_whp_MPa if first else float('nan')),
            'dominant_wellhead_phase': self.dominant_wellhead_phase,
            'pump_flags': sorted({flag for ts in self.timesteps
                                  if ts.success for flag in ts.pump_flags}),
        }

    # ----------------------------------------------------------------
    # Serialisation
    # ----------------------------------------------------------------

    def to_dict(self):
        """Return a JSON-safe dict describing the whole profile."""
        request_dict = None
        if self.request is not None and hasattr(self.request, 'to_dict'):
            request_dict = self.request.to_dict()
        return {
            'request': request_dict,
            'summary': self.summary(),
            'notes': list(self.notes),
            'geophires_parameters': self.to_geophires_parameters(),
            'series': {
                'time_yr': self.time_yr,
                'production_temperature_C': [
                    _clean(v) for v in self.production_temperature_C],
                'reservoir_output_temperature_C': [
                    _clean(v)
                    for v in self.reservoir_output_temperature_C],
                'mass_flow_kgs': [_clean(v)
                                  for v in self.mass_flow_kgs],
                'whp_MPa': [_clean(v) for v in self.whp_MPa],
                'pump_power_MWe': [_clean(v)
                                   for v in self.pump_power_MWe],
                'pump_depth_m': [_clean(v) for v in self.pump_depth_m],
                'wellhead_phase': list(self.wellhead_phase),
            },
            'timesteps': [ts.to_dict() for ts in self.timesteps],
        }

    # ----------------------------------------------------------------
    # GEOPHIRES mapping
    # ----------------------------------------------------------------

    def temperature_profile_rows(self, profile_temperature='wellhead'):
        """
        Rows for a GEOPHIRES 'Reservoir Output File Name' file.

        Parameters
        ----------
        profile_temperature : str
            'wellhead' or 'feedzone', see the module docstring.

        Returns
        -------
        list of (float, float)
            (time [yr], temperature [C]) pairs, with failed timesteps
            dropped.
        """
        if profile_temperature not in PROFILE_TEMPERATURES:
            raise ValueError(
                f'Unknown profile_temperature '
                f'{profile_temperature!r}. Valid values: '
                f'{", ".join(PROFILE_TEMPERATURES)}')
        attribute = ('T_wellhead_C' if profile_temperature == 'wellhead'
                     else 'T_feedzone_C')
        rows = []
        for ts in self.timesteps:
            value = getattr(ts, attribute)
            if ts.success and np.isfinite(value):
                rows.append((float(ts.time_yr), float(value)))
        return rows

    def to_geophires_parameters(self, profile_temperature='wellhead',
                               extra=None):
        """
        Translate the profile into GEOPHIRES input parameters.

        All values are expressed in the units GEOPHIRES expects: kPa
        for pressure, km for reservoir depth, inches for well
        diameter, degrees C for temperature and kg/s for flow rate.
        Values outside the ranges GEOPHIRES accepts are clamped and
        the adjustment is recorded in self.notes.

        Parameters
        ----------
        profile_temperature : str
            Which temperature the accompanying profile file holds,
            'wellhead' or 'feedzone'. It determines the production
            wellbore temperature drop reported here, so it must match
            the value passed to temperature_profile_rows().
        extra : dict or None
            Additional 'Parameter Name' -> value entries appended to
            the result, overriding anything derived here.

        Returns
        -------
        dict
            Mapping of GEOPHIRES parameter names to values.
        """
        request = self.request
        first = self.initial
        notes = self.notes

        params = {}
        params['Reservoir Model'] = GEOPHIRES_RESERVOIR_MODEL_UPP

        if request is not None:
            T_surface_C = request.rock_temperature.T_surface_C
            params['Surface Temperature'] = T_surface_C
            params['Plant Lifetime'] = int(round(
                request.time.plant_lifetime_yr))
            params['Time steps per year'] = int(
                request.time.timesteps_per_year)

            T_max = max(request.reservoir.T_reservoir_C + 5.0, 400.0)
            params['Maximum Temperature'] = _clamp(
                T_max, GEOPHIRES_TMAX_LIMITS_C, 'Maximum Temperature',
                notes)

            params['Reservoir Hydrostatic Pressure'] = units.mpa_to_kpa(
                request.reservoir.P_reservoir_MPa)

            diameter_inch = units.m_to_inch(request.well.diameter_m)
            params['Production Well Diameter'] = _clamp(
                diameter_inch, GEOPHIRES_WELLDIAM_LIMITS_INCH,
                'Production Well Diameter', notes)
        else:
            T_surface_C = None

        if np.isfinite(self.depth_m):
            params['Reservoir Depth'] = units.m_to_km(self.depth_m)
            if T_surface_C is not None and request is not None:
                params['Gradient 1'] = units.gradient_C_per_km(
                    request.reservoir.T_reservoir_C, T_surface_C,
                    self.depth_m)

        # The superhot-wellbore model describes a single self-flowing
        # production well. Injection is outside its scope, so the
        # doublet counts below are a starting point that a user may
        # want to revisit together with the injection temperature.
        params['Number of Production Wells'] = 1
        params['Number of Injection Wells'] = 1

        if first is not None:
            params['Production Flow Rate per Well'] = _clamp(
                first.mass_flow_kgs, GEOPHIRES_FLOWRATE_LIMITS_KGS,
                'Production Flow Rate per Well', notes)
            params['Production Wellhead Pressure'] = units.mpa_to_kpa(
                first.whp_MPa)

        # The wellbore simulator has already accounted for heat loss,
        # friction and flashing, so Ramey's model is always disabled.
        # With a wellhead profile the drop has already been applied
        # and must not be applied twice.
        params['Ramey Production Wellbore Model'] = False
        if profile_temperature == 'wellhead':
            params['Production Wellbore Temperature Drop'] = 0.0
        else:
            drop = self.mean_wellbore_temperature_drop_C
            if np.isfinite(drop):
                params['Production Wellbore Temperature Drop'] = _clamp(
                    drop, GEOPHIRES_TEMPDROP_LIMITS_C,
                    'Production Wellbore Temperature Drop', notes)

        if extra:
            params.update(extra)

        return params


# ====================================================================
# INTERPOLATION SUPPORT
# ====================================================================

def _quality_for_interpolation(qualities, phases):
    """
    Wellhead qualities with single-phase NaNs replaced by 0 or 1.

    A liquid wellhead is the x = 0 limit of a two-phase one and a
    vapour wellhead the x = 1 limit, so a timestep between a
    two-phase solve and a single-phase solve gets a quality on the
    line between them instead of NaN. A supercritical wellhead has no
    quality; its NaN is kept and propagates.
    """
    filled = []
    for quality, phase in zip(qualities, phases):
        if np.isfinite(quality):
            filled.append(float(quality))
        elif phase == 'single_phase_liquid':
            filled.append(0.0)
        elif phase == 'single_phase_vapor':
            filled.append(1.0)
        else:
            filled.append(float('nan'))
    return filled


def _match_phase_to_pressure(result):
    """
    Re-derive the copied wellhead phase of an interpolated timestep
    whose pressure lies on the other side of the critical pressure.

    'supercritical' is a pressure regime (see pump.wellhead_state), so
    a label copied from the nearest solve contradicts the interpolated
    wellhead pressure when the pressure crosses P_crit between the
    solves: a wellhead drifting down through 22.064 MPa would keep
    'supercritical' at 22.0639 MPa, where it is a compressed liquid
    (or a two-phase or vapour state) with no dense-supercritical
    treatment left for it. The phase (and quality) of such a step are
    those of its own interpolated pressure and enthalpy; any other
    copied label is kept, as documented in interpolate_timesteps.
    """
    above_critical = (np.isfinite(result.whp_MPa)
                      and result.whp_MPa >= P_CRIT_MPA)
    if (result.wellhead_phase == 'supercritical') == above_critical:
        return
    phase, quality = wellhead_state(result.whp_MPa, result.h_wellhead_MJkg)
    if phase:
        result.wellhead_phase = phase
        result.wellhead_quality = quality


def interpolate_timesteps(times_yr, solved_results,
                          P_reservoir_MPa=None, T_reservoir_C=None):
    """
    Fill a full time vector from a subset of solved states.

    Parameters
    ----------
    times_yr : sequence of float
        Complete time vector [yr].
    solved_results : dict
        Mapping of time-vector index -> TimestepResult for the
        timesteps where the coupled model was actually run.
    P_reservoir_MPa, T_reservoir_C : sequence of float or None
        Far-field reservoir state at every time. When provided, the
        interpolated timesteps report the true declining state
        instead of that of the nearest solve.

    Returns
    -------
    list of TimestepResult
        One entry per element of times_yr. Solved entries are returned
        exactly as solved, whether they succeeded or failed: a failed
        solve keeps success=False together with its own message, pump
        flags and pump state, so that a well that stops delivering
        stays visible to the caller. Entries that were not solved are
        linearly interpolated from the successful solves only; flags
        (cycle, choked, converged, pumped, self_flowing,
        wellhead_phase, pump_flags) are taken from the nearest
        successful solve, except a wellhead phase on the other side of
        the critical pressure from the entry's own interpolated
        pressure, which is re-derived (_match_phase_to_pressure). An
        unsolved entry whose nearest solve in time failed is still
        filled in (it is a usable estimate) but carries
        interpolated_across_failure=True and names the failed solve in
        its message, leaving the decision to the caller.
        When no solve succeeded, nothing is interpolated and every
        unsolved entry reports success=False.
    """
    times = np.asarray(times_yr, dtype=float)
    solved_indices = sorted(solved_results)
    good_indices = [i for i in solved_indices if solved_results[i].success]
    any_failed = len(good_indices) < len(solved_indices)

    profile = []
    for index, t in enumerate(times):
        if index in solved_results:
            # A solve is reported as it was solved, failed or not:
            # rebuilding a failure from its neighbours would hide it.
            profile.append(solved_results[index])
            continue

        result = TimestepResult(time_yr=float(t))
        if not good_indices:
            result.message = 'no successful solve available'
            profile.append(result)
            continue

        good_times = times[good_indices]
        for attribute in TimestepResult.INTERPOLATED_FIELDS:
            values = [getattr(solved_results[i], attribute)
                      for i in good_indices]
            if attribute == 'wellhead_quality':
                values = _quality_for_interpolation(
                    values, [solved_results[i].wellhead_phase
                             for i in good_indices])
            setattr(result, attribute,
                    float(np.interp(t, good_times, values)))

        nearest = good_indices[int(np.argmin(np.abs(good_times - t)))]
        reference = solved_results[nearest]
        if P_reservoir_MPa is not None:
            result.P_reservoir_MPa = float(P_reservoir_MPa[index])
        else:
            result.P_reservoir_MPa = reference.P_reservoir_MPa
        if T_reservoir_C is not None:
            result.T_reservoir_C = float(T_reservoir_C[index])
        else:
            result.T_reservoir_C = reference.T_reservoir_C
        for attribute in TimestepResult.COPIED_FIELDS:
            setattr(result, attribute, getattr(reference, attribute))
        result.pump_flags = list(reference.pump_flags)
        _match_phase_to_pressure(result)
        if result.wellhead_phase != 'two_phase':
            # Quality is only defined for a two-phase wellhead
            result.wellhead_quality = float('nan')
        result.success = True
        result.message = 'interpolated between coupled-model solves'

        if any_failed:
            # The nearest solve of any outcome: when it failed, the
            # values above were carried across that failure and the
            # caller must be able to tell.
            solved_times = times[solved_indices]
            nearest_solve = solved_indices[
                int(np.argmin(np.abs(solved_times - t)))]
            if not solved_results[nearest_solve].success:
                result.interpolated_across_failure = True
                result.message = (
                    f'interpolated from the successful coupled-model '
                    f'solves across a failed solve at t = '
                    f'{times[nearest_solve]:.2f} yr (nearest successful '
                    f'solve at t = {times[nearest]:.2f} yr)')
        profile.append(result)

    return profile
