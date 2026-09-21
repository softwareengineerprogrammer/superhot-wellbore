# -*- coding: utf-8 -*-
"""
Client Configuration for the GEOPHIRES Integration
===================================================

Declarative, JSON-serialisable description of a superhot single-well
scenario. The configuration objects defined here are the only inputs
the client needs; they are translated into the parameter dictionaries
expected by reservoir.py and wellbore_physics.py.

The configuration is split into sections so that a request can be
written as a single JSON document:

    {
      "name": "iddp1_like",
      "reservoir":        {"P_reservoir_MPa": 30.0, ...},
      "well":             {"depth_m": 2100, ...},
      "rock_temperature": {"mode": "linear", ...},
      "operating":        {"control": "whp", ...},
      "decline":          {"temperature_mode": "linear_percent", ...},
      "time":             {"plant_lifetime_yr": 30, ...},
      "solver":           {"max_solve_points": 8, ...},
      "power_cycle":      {"T_ambient_C": 10.0, ...},
      "pump":             {"mode": "auto", ...}
    }

Units follow the superhot-wellbore inter-module convention: MPa,
MJ/kg, degrees C, metres, kg/s. Conversion to GEOPHIRES units happens
in units.py and results.py, never here.

Time behaviour
--------------
A GEOPHIRES reservoir model must return a production temperature
history over the plant lifetime, while the superhot-wellbore model is
steady state. The DeclineConfig section closes that gap: it defines
how reservoir pressure and temperature evolve with time, and the
client re-solves the coupled reservoir-wellbore model along that
path (see client.py). With the default 'none' modes the reservoir
state is constant and the resulting history is flat, which is the
honest representation of a purely steady-state model.

Production pumping
------------------
A prescribed flow rate that the well cannot deliver to the surface,
or delivers below a minimum wellhead pressure, is lifted by a
production pump described by the PumpConfig section (see pump.py).
With the default mode 'auto' the pump is only used when the well
does not self-flow, so a self-flowing superhot well is solved
exactly as before.

Prescribed inflow
-----------------
The Darcy inflow model can be bypassed by setting
ReservoirConfig.inflow = 'prescribed' and tabulating the feedzone
state (pressure and enthalpy) and, optionally, the flow rate against
time in DeclineConfig. This lets an external reservoir simulator
supply the sandface state directly; it requires a prescribed flow
rate (OperatingConfig.control = 'flow').

Author: superhot-wellbore GEOPHIRES client
"""

import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


# ====================================================================
# DEFAULTS
# ====================================================================

# GEOPHIRES defaults, mirrored so that a client-generated profile
# lines up with the GEOPHIRES time discretisation
# (Reservoir.timevector = linspace(0, lifetime, timesteps_per_year *
# lifetime); see geophires_x/Reservoir.py).
DEFAULT_PLANT_LIFETIME_YR = 30
DEFAULT_TIMESTEPS_PER_YEAR = 4

# Decline modes recognised by DeclineConfig
DECLINE_MODES = ('none', 'linear_percent', 'linear_absolute',
                 'exponential', 'explicit')

# Rock temperature profile modes recognised by RockTemperatureConfig
ROCK_TEMPERATURE_MODES = ('linear', 'boiling', 'user')

# Flow control modes recognised by OperatingConfig
CONTROL_MODES = ('whp', 'flow')

# Quantities that OperatingConfig can hold constant over a history
HOLD_MODES = ('whp', 'flow')

# Inflow models recognised by ReservoirConfig
INFLOW_MODES = ('darcy', 'prescribed')

# Production pump modes recognised by PumpConfig
PUMP_MODES = ('never', 'auto', 'always')

# Pump envelope policies recognised by PumpConfig
PUMP_ENVELOPES = ('flag', 'enforce')

# NPSH margin GEOPHIRES keeps between the pump intake pressure and the
# vapour pressure of the produced water, in MPa (WellBores.py: 344.7 kPa)
DEFAULT_PUMP_NPSH_MARGIN_MPA = 0.3447


# ====================================================================
# SECTION HELPERS
# ====================================================================

def _from_dict(cls, data):
    """
    Build a configuration dataclass from a plain dict.

    Unknown keys raise ValueError rather than being silently
    ignored, so that typos in a JSON request are reported instead
    of quietly falling back to defaults.
    """
    if data is None:
        return cls()
    if isinstance(data, cls):
        return data
    if not isinstance(data, dict):
        raise ValueError(f'{cls.__name__} must be given as a mapping, '
                         f'got {type(data).__name__}')
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f'Unknown {cls.__name__} key(s): '
                         f'{", ".join(unknown)}. '
                         f'Valid keys: {", ".join(sorted(known))}')
    return cls(**data)


def _decline_series(mode, rate_per_year, table, times_yr,
                    initial_value, floor, label):
    """
    Evaluate a decline law on a time vector.

    Parameters
    ----------
    mode : str
        One of DECLINE_MODES.
    rate_per_year : float
        Decline rate. Interpretation depends on mode:
        'linear_percent'  - percent of the initial value per year
        'linear_absolute' - absolute units per year
        'exponential'     - exponential decay constant [1/yr]
    table : list or None
        Explicit [[time_yr, value], ...] table for mode 'explicit'.
    times_yr : ndarray
        Times at which to evaluate the law [yr].
    initial_value : float
        Value at t = 0.
    floor : float
        Lower bound applied to the result.
    label : str
        Quantity name, used in error messages.

    Returns
    -------
    ndarray
        Values at each time, clipped from below at floor.
    """
    t = np.asarray(times_yr, dtype=float)

    if mode == 'none':
        values = np.full(t.shape, float(initial_value))
    elif mode == 'linear_percent':
        values = initial_value * (1.0 - rate_per_year / 100.0 * t)
    elif mode == 'linear_absolute':
        values = initial_value - rate_per_year * t
    elif mode == 'exponential':
        values = initial_value * np.exp(-rate_per_year * t)
    elif mode == 'explicit':
        if not table:
            raise ValueError(f"{label} decline mode 'explicit' requires "
                             f'a [[time_yr, value], ...] table')
        arr = np.asarray(table, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f'{label} decline table must be a list of '
                             f'[time_yr, value] pairs')
        order = np.argsort(arr[:, 0])
        values = np.interp(t, arr[order, 0], arr[order, 1])
    else:
        raise ValueError(f'Unknown {label} decline mode {mode!r}. '
                         f'Valid modes: {", ".join(DECLINE_MODES)}')

    return np.maximum(values, float(floor))


def _profile_table(table, n_columns, label):
    """
    Validate and sort a [[time_yr, value, ...], ...] table.

    Parameters
    ----------
    table : list or None
        Rows of n_columns numbers, the first being the time [yr].
    n_columns : int
        Expected number of entries per row.
    label : str
        Table name, used in error messages.

    Returns
    -------
    ndarray or None
        The table sorted by time, or None when no table was given.
    """
    if table is None:
        return None
    arr = np.asarray(table, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != n_columns or arr.shape[0] == 0:
        raise ValueError(f'{label} must be a non-empty list of rows '
                         f'with {n_columns} numbers each')
    if not np.all(np.isfinite(arr)):
        raise ValueError(f'{label} must contain only finite numbers')
    return arr[np.argsort(arr[:, 0])]


def _table_varies(table, n_columns, label):
    """True if any value column of a profile table changes with time."""
    arr = _profile_table(table, n_columns, label)
    if arr is None or arr.shape[0] < 2:
        return False
    return bool(np.any(np.ptp(arr[:, 1:], axis=0) > 0.0))


# ====================================================================
# CONFIGURATION SECTIONS
# ====================================================================

@dataclass
class ReservoirConfig:
    """
    Initial reservoir state and permeability structure.

    Attributes
    ----------
    P_reservoir_MPa : float
        Far-field reservoir pressure at t = 0 [MPa].
    T_reservoir_C : float
        Far-field reservoir temperature at t = 0 [C].
    transmissivity_md_m : float or None
        Permeability-thickness product k*b [md.m]. Takes precedence
        over permeability_md / thickness_m when provided.
    permeability_md : float or None
        Reservoir permeability [md]. Used with thickness_m when
        transmissivity_md_m is None.
    thickness_m : float or None
        Feedzone thickness [m].
    drainage_radius_m : float
        Distance to the constant-pressure outer boundary [m].
    inflow : str
        'darcy'      - feedzone state from the radial Darcy drawdown
                       of reservoir.py (the default)
        'prescribed' - feedzone pressure and enthalpy tabulated
                       against time in DeclineConfig.feedzone_profile;
                       no Darcy model, so the transmissivity is not
                       needed. P_reservoir_MPa then only sets the
                       far-field pressure the reported reservoir
                       drawdown refers to, and T_reservoir_C the
                       formation temperature profile.
    """

    P_reservoir_MPa: float = 30.0
    T_reservoir_C: float = 450.0
    transmissivity_md_m: Optional[float] = 1000.0
    permeability_md: Optional[float] = None
    thickness_m: Optional[float] = None
    drainage_radius_m: float = 500.0
    inflow: str = 'darcy'

    def to_reservoir_params(self):
        """
        Build the reservoir_params dict expected by reservoir.py.

        Returns
        -------
        dict
            Keys 'drainage_radius_m' plus either 'transmissivity_md_m'
            or 'permeability_md' and 'thickness_m'.
        """
        params = {'drainage_radius_m': self.drainage_radius_m}
        if self.transmissivity_md_m is not None:
            params['transmissivity_md_m'] = self.transmissivity_md_m
        else:
            params['permeability_md'] = self.permeability_md
            params['thickness_m'] = self.thickness_m
        return params

    def validate(self):
        """Raise ValueError if the section is inconsistent."""
        if self.P_reservoir_MPa <= 0:
            raise ValueError('P_reservoir_MPa must be positive')
        if self.T_reservoir_C <= 0:
            raise ValueError('T_reservoir_C must be positive')
        if self.drainage_radius_m <= 0:
            raise ValueError('drainage_radius_m must be positive')
        if self.inflow not in INFLOW_MODES:
            raise ValueError(
                f'Unknown inflow mode {self.inflow!r}. Valid modes: '
                f'{", ".join(INFLOW_MODES)}')
        if self.inflow == 'darcy' and self.transmissivity_md_m is None:
            if self.permeability_md is None or self.thickness_m is None:
                raise ValueError(
                    'Provide either transmissivity_md_m, or both '
                    'permeability_md and thickness_m')


@dataclass
class WellConfig:
    """
    Production well geometry and heat-loss properties.

    Attributes
    ----------
    depth_m : float or None
        Feedzone (total vertical) depth [m]. When None, the depth is
        derived from the initial reservoir pressure with
        reservoir.depth_for_pressure().
    diameter_m : float
        Casing inner diameter [m]. Default 0.217 m is the 9 5/8"
        production casing of IDDP-1.
    delta_z_m : float
        Vertical integration step of the wellbore model [m]. Must be a
        whole number of metres, see to_well_params().
    roughness_m : float
        Absolute casing roughness [m].
    heat_loss_factor : float
        Overall wellbore heat transfer coefficient U [W/m/K].
    """

    depth_m: Optional[float] = None
    diameter_m: float = 0.217
    delta_z_m: float = 10.0
    roughness_m: float = 4.6e-5
    heat_loss_factor: float = 2.5

    def to_well_params(self, depth_m=None):
        """
        Build the well_params dict expected by wellbore_physics.py.

        Parameters
        ----------
        depth_m : float or None
            Resolved well depth [m]. Falls back to self.depth_m.

        Returns
        -------
        dict
            Keys 'depth_m', 'diameter_m', 'delta_z_m', 'roughness_m',
            'heat_loss_factor'.

        Notes
        -----
        The depth and the integration step are emitted as ints:
        wellbore_physics.wellbore_simulate() marches the well with
        range(0, depth_m + delta_z_m, delta_z_m), which only accepts
        integers. Passing floats makes the wellbore model fail with
        no diagnostic beyond a missing solution.
        """
        resolved = self.depth_m if depth_m is None else depth_m
        if resolved is None:
            raise ValueError('Well depth has not been resolved yet')
        return {
            'depth_m': int(round(float(resolved))),
            'diameter_m': float(self.diameter_m),
            'delta_z_m': int(round(float(self.delta_z_m))),
            'roughness_m': float(self.roughness_m),
            'heat_loss_factor': float(self.heat_loss_factor),
        }

    def validate(self):
        """Raise ValueError if the section is inconsistent."""
        if self.depth_m is not None and self.depth_m <= 0:
            raise ValueError('depth_m must be positive when provided')
        if self.diameter_m <= 0:
            raise ValueError('diameter_m must be positive')
        if self.delta_z_m <= 0:
            raise ValueError('delta_z_m must be positive')
        if abs(self.delta_z_m - round(self.delta_z_m)) > 1e-9:
            raise ValueError(
                f'delta_z_m must be a whole number of metres, got '
                f'{self.delta_z_m!r}; the wellbore model marches the '
                f'well on an integer depth grid')
        if (self.depth_m is not None
                and abs(self.depth_m - round(self.depth_m)) > 1e-9):
            raise ValueError(
                f'depth_m must be a whole number of metres, got '
                f'{self.depth_m!r}; the wellbore model marches the '
                f'well on an integer depth grid')
        if self.roughness_m < 0:
            raise ValueError('roughness_m must not be negative')
        if self.heat_loss_factor < 0:
            raise ValueError('heat_loss_factor must not be negative')


@dataclass
class RockTemperatureConfig:
    """
    Formation temperature profile used for wellbore heat loss.

    Attributes
    ----------
    mode : str
        'linear'  - linear T(z) from surface to reservoir temperature
        'boiling' - boiling-point-with-depth profile
        'user'    - explicit {depth_m: T_C} profile
    T_surface_C : float
        Surface (ground) temperature [C].
    surface_pressure_MPa : float
        Reference surface pressure for the boiling-point profile [MPa].
    profile_C : dict or None
        Explicit depth [m] -> temperature [C] mapping for mode 'user'.
        JSON keys are strings and are converted to floats on load.
    """

    mode: str = 'linear'
    T_surface_C: float = 10.0
    surface_pressure_MPa: float = 0.1
    profile_C: Optional[Dict[float, float]] = None

    def validate(self):
        """Raise ValueError if the section is inconsistent."""
        if self.mode not in ROCK_TEMPERATURE_MODES:
            raise ValueError(
                f'Unknown rock temperature mode {self.mode!r}. Valid '
                f'modes: {", ".join(ROCK_TEMPERATURE_MODES)}')
        if self.mode == 'user' and not self.profile_C:
            raise ValueError("rock temperature mode 'user' requires "
                             'profile_C')

    def user_profile(self):
        """Return profile_C with float keys, or None."""
        if not self.profile_C:
            return None
        return {float(z): float(T) for z, T in self.profile_C.items()}


@dataclass
class OperatingConfig:
    """
    Well operating point.

    Attributes
    ----------
    control : str
        'whp'  - solve for the flow rate that delivers
                 target_whp_MPa at the wellhead
        'flow' - prescribe mass_flow_kgs and compute the resulting
                 wellhead pressure
    target_whp_MPa : float
        Target wellhead pressure [MPa] for control = 'whp'.
    mass_flow_kgs : float or None
        Prescribed mass flow rate [kg/s] for control = 'flow'.
    hold : str
        Quantity held constant along a declining production history
        when control = 'whp':
        'whp'  - re-solve for target_whp_MPa at every reservoir
                 state, so the flow rate responds to the decline
        'flow' - solve for the flow rate at the initial state only,
                 then hold that flow rate and let the wellhead
                 pressure respond. This is how GEOPHIRES operates a
                 well, which carries a single flow rate per well.
        Ignored when control = 'flow', which already holds the flow.
    """

    control: str = 'whp'
    target_whp_MPa: float = 10.0
    mass_flow_kgs: Optional[float] = None
    hold: str = 'whp'

    def validate(self):
        """Raise ValueError if the section is inconsistent."""
        if self.control not in CONTROL_MODES:
            raise ValueError(
                f'Unknown control mode {self.control!r}. Valid modes: '
                f'{", ".join(CONTROL_MODES)}')
        if self.control == 'whp' and self.target_whp_MPa <= 0:
            raise ValueError('target_whp_MPa must be positive')
        if self.control == 'flow':
            if self.mass_flow_kgs is None or self.mass_flow_kgs <= 0:
                raise ValueError("control 'flow' requires a positive "
                                 'mass_flow_kgs')
        if self.hold not in HOLD_MODES:
            raise ValueError(
                f'Unknown hold mode {self.hold!r}. Valid modes: '
                f'{", ".join(HOLD_MODES)}')


@dataclass
class DeclineConfig:
    """
    Time evolution of the far-field reservoir state.

    The client re-solves the coupled reservoir-wellbore model at the
    declining (P, T) states described here, which is what turns the
    steady-state superhot-wellbore model into the production history
    a GEOPHIRES reservoir model must supply.

    Attributes
    ----------
    temperature_mode, pressure_mode : str
        One of DECLINE_MODES. 'none' keeps the quantity constant.
    temperature_rate_per_year : float
        Temperature decline rate. Percent of the initial temperature
        per year ('linear_percent'), C/yr ('linear_absolute'), or
        decay constant in 1/yr ('exponential').
    pressure_rate_per_year : float
        Pressure decline rate, same interpretation as above with MPa
        in place of C.
    temperature_profile : list or None
        Explicit [[time_yr, T_C], ...] table for mode 'explicit'.
    pressure_profile : list or None
        Explicit [[time_yr, P_MPa], ...] table for mode 'explicit'.
    min_temperature_C : float
        Lower bound on reservoir temperature [C].
    min_pressure_MPa : float
        Lower bound on reservoir pressure [MPa].
    feedzone_profile : list or None
        Explicit [[time_yr, P_feedzone_MPa, h_feedzone_MJkg], ...]
        table of the sandface state, used when
        ReservoirConfig.inflow = 'prescribed'. Interpolated linearly
        in time and held at the end values outside the table.
    mass_flow_profile : list or None
        Explicit [[time_yr, mass_flow_kgs], ...] table of the
        produced flow rate per well, overriding
        OperatingConfig.mass_flow_kgs at every timestep. Requires
        OperatingConfig.control = 'flow'.
    """

    temperature_mode: str = 'none'
    temperature_rate_per_year: float = 0.0
    temperature_profile: Optional[List[List[float]]] = None
    min_temperature_C: float = 200.0

    pressure_mode: str = 'none'
    pressure_rate_per_year: float = 0.0
    pressure_profile: Optional[List[List[float]]] = None
    min_pressure_MPa: float = 5.0

    feedzone_profile: Optional[List[List[float]]] = None
    mass_flow_profile: Optional[List[List[float]]] = None

    def temperatures_C(self, times_yr, T_initial_C):
        """Reservoir temperature [C] at each time in times_yr."""
        return _decline_series(self.temperature_mode,
                               self.temperature_rate_per_year,
                               self.temperature_profile,
                               times_yr, T_initial_C,
                               self.min_temperature_C, 'temperature')

    def pressures_MPa(self, times_yr, P_initial_MPa):
        """Reservoir pressure [MPa] at each time in times_yr."""
        return _decline_series(self.pressure_mode,
                               self.pressure_rate_per_year,
                               self.pressure_profile,
                               times_yr, P_initial_MPa,
                               self.min_pressure_MPa, 'pressure')

    def feedzone_states(self, times_yr):
        """
        Prescribed feedzone state at each time in times_yr.

        Returns
        -------
        (ndarray, ndarray) or None
            Feedzone pressure [MPa] and enthalpy [MJ/kg] at each time,
            or None when no feedzone_profile is given.
        """
        table = _profile_table(self.feedzone_profile, 3,
                               'feedzone_profile')
        if table is None:
            return None
        t = np.asarray(times_yr, dtype=float)
        return (np.interp(t, table[:, 0], table[:, 1]),
                np.interp(t, table[:, 0], table[:, 2]))

    def mass_flows_kgs(self, times_yr, default_kgs=None):
        """
        Prescribed flow rate at each time in times_yr.

        Returns
        -------
        ndarray or None
            Flow rate [kg/s] at each time from mass_flow_profile, the
            constant default_kgs when there is no table, or None when
            neither is given.
        """
        table = _profile_table(self.mass_flow_profile, 2,
                               'mass_flow_profile')
        t = np.asarray(times_yr, dtype=float)
        if table is None:
            if default_kgs is None:
                return None
            return np.full(t.shape, float(default_kgs))
        return np.interp(t, table[:, 0], table[:, 1])

    def is_steady(self):
        """
        True if the reservoir state does not vary with time.

        Neither pressure nor temperature declines, and neither the
        prescribed feedzone state nor the prescribed flow rate
        changes between the rows of its table.
        """
        return (self.temperature_mode == 'none'
                and self.pressure_mode == 'none'
                and not _table_varies(self.feedzone_profile, 3,
                                      'feedzone_profile')
                and not _table_varies(self.mass_flow_profile, 2,
                                      'mass_flow_profile'))

    def validate(self):
        """Raise ValueError if the section is inconsistent."""
        for mode, label in ((self.temperature_mode, 'temperature'),
                            (self.pressure_mode, 'pressure')):
            if mode not in DECLINE_MODES:
                raise ValueError(
                    f'Unknown {label} decline mode {mode!r}. Valid '
                    f'modes: {", ".join(DECLINE_MODES)}')
        if self.temperature_mode == 'explicit' and not self.temperature_profile:
            raise ValueError("temperature_mode 'explicit' requires "
                             'temperature_profile')
        if self.pressure_mode == 'explicit' and not self.pressure_profile:
            raise ValueError("pressure_mode 'explicit' requires "
                             'pressure_profile')
        if self.min_temperature_C <= 0:
            raise ValueError('min_temperature_C must be positive')
        if self.min_pressure_MPa <= 0:
            raise ValueError('min_pressure_MPa must be positive')
        feedzone = _profile_table(self.feedzone_profile, 3,
                                  'feedzone_profile')
        if feedzone is not None and np.any(feedzone[:, 1:] <= 0):
            raise ValueError('feedzone_profile pressures and enthalpies '
                             'must be positive')
        flows = _profile_table(self.mass_flow_profile, 2,
                               'mass_flow_profile')
        if flows is not None and np.any(flows[:, 1] <= 0):
            raise ValueError('mass_flow_profile flow rates must be '
                             'positive')


@dataclass
class TimeConfig:
    """
    Time discretisation, mirroring the GEOPHIRES time vector.

    Attributes
    ----------
    plant_lifetime_yr : float
        Project lifetime [yr] ('Plant Lifetime' in GEOPHIRES).
    timesteps_per_year : int
        Number of timesteps per year ('Time steps per year').
    """

    plant_lifetime_yr: float = DEFAULT_PLANT_LIFETIME_YR
    timesteps_per_year: int = DEFAULT_TIMESTEPS_PER_YEAR

    def n_timesteps(self):
        """Number of points in the time vector."""
        return max(int(round(self.timesteps_per_year
                             * self.plant_lifetime_yr)), 2)

    def time_vector_yr(self):
        """
        Time vector [yr] matching the GEOPHIRES discretisation.

        GEOPHIRES builds its reservoir time vector as
        linspace(0, lifetime, timesteps_per_year * lifetime); the same
        expression is used here so that a client profile can be
        assigned to Reservoir.Tresoutput element by element.
        """
        return np.linspace(0.0, float(self.plant_lifetime_yr),
                           self.n_timesteps())

    def time_step_yr(self):
        """Spacing of the time vector [yr]."""
        n = self.n_timesteps()
        return float(self.plant_lifetime_yr) / (n - 1)

    def validate(self):
        """Raise ValueError if the section is inconsistent."""
        if self.plant_lifetime_yr <= 0:
            raise ValueError('plant_lifetime_yr must be positive')
        if int(self.timesteps_per_year) < 1:
            raise ValueError('timesteps_per_year must be at least 1')


@dataclass
class SolverConfig:
    """
    Numerical controls for the coupled-model solves.

    Attributes
    ----------
    tolerance_MPa : float
        Wellhead pressure tolerance passed to
        reservoir.solve_flow_for_whp() [MPa].
    max_solve_points : int
        Maximum number of full coupled-model solves. When the time
        vector is longer than this, the coupled model is solved on an
        evenly spaced subset of times and the results are linearly
        interpolated onto the full time vector. Each solve costs
        seconds, so this bounds the runtime of a long project life.
        Set to 0 or a negative value to solve at every timestep.
    reuse_previous_solution : bool
        Pass the previous converged solution to
        solve_flow_for_whp() as a starting point, which markedly
        speeds up a sequence of similar states.
    strict : bool
        Raise RuntimeError when a coupled-model solve fails instead
        of interpolating across the failure.
    verbose : bool
        Print solver progress.
    """

    tolerance_MPa: float = 0.15
    max_solve_points: int = 8
    reuse_previous_solution: bool = True
    strict: bool = False
    verbose: bool = False

    def validate(self):
        """Raise ValueError if the section is inconsistent."""
        if self.tolerance_MPa <= 0:
            raise ValueError('tolerance_MPa must be positive')


@dataclass
class PowerCycleConfig:
    """
    Surface power cycle parameters for power_cycle.power_cycle_analysis().

    The gross turbine power of every solved state is reported in
    TimestepResult.power_MWe together with the cycle selected
    ('binary' or 'flash'), the utilization efficiency and the exergetic
    power of the wellhead stream. The defaults are those of
    power_cycle.DEFAULT_POWER_PARAMS; see that module for the meaning
    and the modelling assumptions (gross turbine power, fixed flash
    and working-fluid pressures, pure water properties).

    Attributes
    ----------
    T_ambient_C : float
        Dead-state (ambient) temperature for exergy [C].
    T_reject_C : float
        Geofluid rejection temperature of the binary heat exchanger [C].
    T_wf_inlet_C : float
        Working fluid heat exchanger inlet temperature [C].
    delta_T_pinch_C : float
        Heat exchanger pinch point temperature difference [C].
    P_wf_MPa : float
        Working fluid pressure of the binary cycle [MPa].
    P_condenser_MPa : float
        Condenser pressure [MPa].
    eta_turbine_dry : float
        Dry isentropic turbine efficiency [-].
    P_flash_MPa : float
        Flash separation pressure of the flash cycle [MPa].
    x_exit_min : float
        Minimum acceptable turbine exit quality [-].
    superheat_margin_Jkg : float
        Enthalpy margin above saturated vapour for selecting the
        binary cycle [J/kg].
    """

    T_ambient_C: float = 25.0
    T_reject_C: float = 60.0
    T_wf_inlet_C: float = 40.0
    delta_T_pinch_C: float = 5.0
    P_wf_MPa: float = 1.0
    P_condenser_MPa: float = 0.01
    eta_turbine_dry: float = 0.85
    P_flash_MPa: float = 1.0
    x_exit_min: float = 0.85
    superheat_margin_Jkg: float = 50.0e3

    def to_params(self):
        """Return the params dict expected by power_cycle_analysis()."""
        return dataclasses.asdict(self)

    def validate(self):
        """Raise ValueError if the section is inconsistent."""
        for name in ('delta_T_pinch_C', 'P_wf_MPa', 'P_condenser_MPa',
                     'P_flash_MPa'):
            if getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive')
        if not 0 < self.eta_turbine_dry <= 1:
            raise ValueError('eta_turbine_dry must be in (0, 1]')
        if not 0 <= self.x_exit_min <= 1:
            raise ValueError('x_exit_min must be in [0, 1]')
        if self.T_reject_C <= self.T_wf_inlet_C:
            raise ValueError('T_reject_C must exceed T_wf_inlet_C')


@dataclass
class PumpConfig:
    """
    Production pump for wells that do not self-flow (see pump.py).

    The pump stage applies to prescribed-flow solves (control =
    'flow', a held flow rate, or a mass_flow_profile). A wellhead
    pressure solve (control = 'whp') finds the flow rate the well
    delivers by itself, so it never pumps.

    Attributes
    ----------
    mode : str
        'never'  - never pump; a well that does not reach the surface
                   fails as before
        'auto'   - pump only when the unpumped well does not reach the
                   surface, or reaches it below the self-flow floor
        'always' - pump whenever a liquid intake exists
    efficiency : float
        Pump (hydraulic times motor) efficiency [-].
    npsh_margin_MPa : float
        Margin the intake pressure must keep above the vapour pressure
        of the fluid at the intake temperature [MPa]; the same rule
        selects the pumped wellhead pressure so the wellhead stream
        stays liquid.
    min_self_flow_whp_MPa : float
        Wellhead pressure below which a self-flowing well is pumped
        anyway [MPa] (mode 'auto').
    max_depth_m : float
        Deepest acceptable pump setting depth [m].
    max_intake_temperature_C : float
        Hottest acceptable pump intake temperature [C].
    envelope : str
        'flag'    - model the pump outside max_depth_m /
                    max_intake_temperature_C and report it in
                    pump_flags
        'enforce' - fail the solve when the pump is needed outside
                    the envelope or when no liquid intake exists
    target_whp_MPa : float or None
        Pumped wellhead pressure [MPa]. None selects the vapour
        pressure at the intake temperature plus npsh_margin_MPa.
    tolerance_MPa : float
        Convergence tolerance on the pumped wellhead pressure [MPa].
    max_dP_MPa : float
        Upper bound of the pump pressure rise searched [MPa].
    """

    mode: str = 'auto'
    efficiency: float = 0.80
    npsh_margin_MPa: float = DEFAULT_PUMP_NPSH_MARGIN_MPA
    min_self_flow_whp_MPa: float = 1.0
    max_depth_m: float = 1500.0
    max_intake_temperature_C: float = 250.0
    envelope: str = 'flag'
    target_whp_MPa: Optional[float] = None
    tolerance_MPa: float = 0.01
    max_dP_MPa: float = 60.0

    def validate(self):
        """Raise ValueError if the section is inconsistent."""
        if self.mode not in PUMP_MODES:
            raise ValueError(
                f'Unknown pump mode {self.mode!r}. Valid modes: '
                f'{", ".join(PUMP_MODES)}')
        if self.envelope not in PUMP_ENVELOPES:
            raise ValueError(
                f'Unknown pump envelope policy {self.envelope!r}. '
                f'Valid policies: {", ".join(PUMP_ENVELOPES)}')
        if not 0 < self.efficiency <= 1:
            raise ValueError('efficiency must be in (0, 1]')
        if self.npsh_margin_MPa < 0:
            raise ValueError('npsh_margin_MPa must not be negative')
        if self.min_self_flow_whp_MPa < 0:
            raise ValueError('min_self_flow_whp_MPa must not be '
                             'negative')
        for name in ('max_depth_m', 'max_intake_temperature_C',
                     'tolerance_MPa', 'max_dP_MPa'):
            if getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive')
        if self.target_whp_MPa is not None and self.target_whp_MPa <= 0:
            raise ValueError('target_whp_MPa must be positive when '
                             'provided')


# ====================================================================
# TOP-LEVEL REQUEST
# ====================================================================

@dataclass
class CoupledWellboreRequest:
    """
    Complete description of a superhot single-well GEOPHIRES scenario.

    Use from_dict()/to_dict() to move a request through JSON, which
    is how the command-line client exchanges scenarios with external
    tools such as GEOPHIRES.
    """

    name: str = 'well'
    reservoir: ReservoirConfig = field(default_factory=ReservoirConfig)
    well: WellConfig = field(default_factory=WellConfig)
    rock_temperature: RockTemperatureConfig = field(
        default_factory=RockTemperatureConfig)
    operating: OperatingConfig = field(default_factory=OperatingConfig)
    decline: DeclineConfig = field(default_factory=DeclineConfig)
    time: TimeConfig = field(default_factory=TimeConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    power_cycle: PowerCycleConfig = field(default_factory=PowerCycleConfig)
    pump: PumpConfig = field(default_factory=PumpConfig)

    # ----------------------------------------------------------------
    # Serialisation
    # ----------------------------------------------------------------

    @classmethod
    def from_dict(cls, data):
        """
        Build a request from a nested dict (e.g. parsed JSON).

        Unknown section names or keys raise ValueError.
        """
        if data is None:
            return cls()
        if isinstance(data, cls):
            return data
        sections = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(data) - sections)
        if unknown:
            raise ValueError(f'Unknown request section(s): '
                             f'{", ".join(unknown)}. Valid sections: '
                             f'{", ".join(sorted(sections))}')
        return cls(
            name=data.get('name', 'well'),
            reservoir=_from_dict(ReservoirConfig, data.get('reservoir')),
            well=_from_dict(WellConfig, data.get('well')),
            rock_temperature=_from_dict(RockTemperatureConfig,
                                        data.get('rock_temperature')),
            operating=_from_dict(OperatingConfig, data.get('operating')),
            decline=_from_dict(DeclineConfig, data.get('decline')),
            time=_from_dict(TimeConfig, data.get('time')),
            solver=_from_dict(SolverConfig, data.get('solver')),
            power_cycle=_from_dict(PowerCycleConfig,
                                   data.get('power_cycle')),
            pump=_from_dict(PumpConfig, data.get('pump')),
        )

    def to_dict(self):
        """Return a plain nested dict suitable for json.dump()."""
        return dataclasses.asdict(self)

    # ----------------------------------------------------------------
    # Validation
    # ----------------------------------------------------------------

    def validate(self):
        """
        Validate every section.

        Returns
        -------
        CoupledWellboreRequest
            self, so that validation can be chained.
        """
        self.reservoir.validate()
        self.well.validate()
        self.rock_temperature.validate()
        self.operating.validate()
        self.decline.validate()
        self.time.validate()
        self.solver.validate()
        self.power_cycle.validate()
        self.pump.validate()

        # Rules that span sections
        if self.reservoir.inflow == 'prescribed':
            if self.operating.control != 'flow':
                raise ValueError(
                    "inflow 'prescribed' requires operating.control "
                    "'flow': without the Darcy model there is no "
                    'inflow relation to solve a wellhead pressure '
                    'against')
            if not self.decline.feedzone_profile:
                raise ValueError("inflow 'prescribed' requires "
                                 'decline.feedzone_profile')
        if (self.decline.mass_flow_profile
                and self.operating.control != 'flow'):
            raise ValueError("decline.mass_flow_profile requires "
                             "operating.control 'flow'")
        return self
