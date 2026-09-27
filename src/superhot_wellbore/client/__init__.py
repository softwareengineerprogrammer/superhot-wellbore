# -*- coding: utf-8 -*-
"""
superhot-wellbore Client
=========================

Stable interface through which an external program obtains a
production history from the superhot-wellbore coupled
reservoir-wellbore model. GEOPHIRES uses it for its coupled
inflow-wellbore production wellbore model ('Production Wellbore Model, 2',
geophires_x/CoupledWellBores.py).

The core modules solve a steady-state, single self-flowing well: given
a far-field reservoir state (pressure, temperature, transmissivity)
and a well, they return the mass flow rate and the wellhead
conditions. A techno-economic simulator such as GEOPHIRES instead
expects a production history over a plant lifetime. This package
bridges the two.

From steady state to a history
------------------------------
The steady-state model is turned into a history by re-solving it: the
DeclineConfig section describes how reservoir pressure and temperature
evolve, and the coupled model is solved again at each of those states
(bounded by SolverConfig.max_solve_points, with linear interpolation
in between). Leaving the decline modes at 'none' yields a flat
history, which is the faithful reading of a purely steady-state model.

Quick start
-----------
In Python::

    from superhot_wellbore.client import (
        CoupledWellboreRequest, CoupledWellboreClient)

    request = CoupledWellboreRequest.from_dict({
        'reservoir': {'P_reservoir_MPa': 30.0, 'T_reservoir_C': 450.0},
        'operating': {'target_whp_MPa': 10.0},
        'decline': {'temperature_mode': 'linear_percent',
                    'temperature_rate_per_year': 0.5},
    })
    profile = CoupledWellboreClient(request).solve_profile()
    print(profile.summary())

Pass the caller's own time vector to solve_profile(time_yr=...) to get
one result per element of it, which is how GEOPHIRES calls it.

Production pumping
------------------
A prescribed flow rate that the well cannot deliver to the surface on
its own is lifted by the production pump stage (pump.py), configured
through the request's 'pump' section (PumpConfig). With the default
mode 'auto' a self-flowing well is untouched; a well that does not
reach the surface, or reaches it below the self-flow floor, gets a
pump at the shallowest depth with a liquid column below it, and the
timestep reports the pump depth, pressure rise and power alongside
the pumped wellhead state.

Prescribed inflow
-----------------
An external reservoir simulator can bypass the Darcy inflow model
(ReservoirConfig.inflow = 'prescribed') and tabulate the feedzone
pressure and enthalpy, and optionally the flow rate, against time in
the 'decline' section.

File-based path
---------------
For GEOPHIRES versions without the built-in superhot production
wellbore model, the command-line client (cli.py, export.py) writes a
'time, temperature' profile plus a matching GEOPHIRES input deck
fragment, which GEOPHIRES reads with its 'User-Provided Temperature
Profile' reservoir model::

    superhot-geophires run --set reservoir.T_reservoir_C=475

Units
-----
The client speaks the superhot-wellbore convention (MPa, MJ/kg,
degrees C, m, kg/s). Conversion to the GEOPHIRES convention (kPa, km,
inch, C/km) happens only in units.py and results.py.

Author: superhot-wellbore client
"""

from .. import __version__
from ..power_cycle import is_dense_supercritical
from .client import (CoupledWellboreClient, solve_profile,
                     solve_steady_state)
from .config import (CONTROL_MODES, DECLINE_MODES, HOLD_MODES,
                     INFLOW_MODES, PUMP_ENVELOPES, PUMP_MODES,
                     ROCK_TEMPERATURE_MODES, DeclineConfig,
                     OperatingConfig, PowerCycleConfig, PumpConfig,
                     ReservoirConfig, RockTemperatureConfig, SolverConfig,
                     CoupledWellboreRequest, TimeConfig, WellConfig)
from .export import (export_all, geophires_input_text,
                     write_geophires_input, write_profile_json,
                     write_request_json, write_temperature_profile)
from .pump import PUMP_FLAGS, WELLHEAD_PHASES, solve_pumped_state
from .results import (GEOPHIRES_RESERVOIR_MODEL_UPP,
                      PROFILE_TEMPERATURES, ProductionProfile,
                      TimestepResult)

__all__ = [
    '__version__',
    # Client
    'CoupledWellboreClient',
    'solve_profile',
    'solve_steady_state',
    # Configuration
    'CoupledWellboreRequest',
    'ReservoirConfig',
    'WellConfig',
    'RockTemperatureConfig',
    'OperatingConfig',
    'DeclineConfig',
    'TimeConfig',
    'SolverConfig',
    'PowerCycleConfig',
    'PumpConfig',
    'DECLINE_MODES',
    'ROCK_TEMPERATURE_MODES',
    'CONTROL_MODES',
    'HOLD_MODES',
    'INFLOW_MODES',
    'PUMP_MODES',
    'PUMP_ENVELOPES',
    # Pump stage
    'solve_pumped_state',
    'PUMP_FLAGS',
    'WELLHEAD_PHASES',
    # Power cycle
    'is_dense_supercritical',
    # Results
    'ProductionProfile',
    'TimestepResult',
    'PROFILE_TEMPERATURES',
    'GEOPHIRES_RESERVOIR_MODEL_UPP',
    # Export
    'export_all',
    'geophires_input_text',
    'write_temperature_profile',
    'write_geophires_input',
    'write_profile_json',
    'write_request_json',
]
