# -*- coding: utf-8 -*-
"""
GEOPHIRES Client for the superhot-wellbore Model
=================================================

Client that lets the superhot-wellbore coupled reservoir-wellbore
model act as a reservoir model in GEOPHIRES.

The superhot-wellbore modules solve a steady-state, single self-flowing
well: given a far-field reservoir state (pressure, temperature,
transmissivity) and a well, they return the mass flow rate and the
wellhead conditions. GEOPHIRES expects a reservoir model to report a
production temperature history over the plant lifetime, and then adds
its own wellbore, surface plant and economics. This package bridges
the two.

Two integration paths
---------------------
In process (adapter.py)
    SuperhotWellboreReservoir is a GEOPHIRES-X reservoir model. It
    reads its parameters from an ordinary GEOPHIRES input file, solves
    the coupled model over GEOPHIRES' own time vector, and supplies
    the reservoir output temperature together with the production
    well state (wellhead temperature, flow rate, wellhead pressure),
    since the superhot model simulates the well itself. Electricity
    generation and economics remain GEOPHIRES' job.

File based (cli.py, export.py)
    A command-line client writes a 'time, temperature' profile plus a
    matching GEOPHIRES input deck fragment, which GEOPHIRES reads with
    its built-in 'User-Provided Temperature Profile' reservoir model.
    Nothing has to import anything, in either direction.

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

    from geophires_client import SuperhotRequest, SuperhotWellboreClient

    request = SuperhotRequest.from_dict({
        'reservoir': {'P_reservoir_MPa': 30.0, 'T_reservoir_C': 450.0},
        'operating': {'target_whp_MPa': 10.0},
        'decline': {'temperature_mode': 'linear_percent',
                    'temperature_rate_per_year': 0.5},
    })
    profile = SuperhotWellboreClient(request).solve_profile()
    print(profile.summary())

On the command line::

    python -m geophires_client run --set reservoir.T_reservoir_C=475

Units
-----
The client speaks the superhot-wellbore convention (MPa, MJ/kg,
degrees C, m, kg/s). Conversion to the GEOPHIRES convention (kPa, km,
inch, C/km) happens only in units.py and results.py.

Author: superhot-wellbore GEOPHIRES client
"""

from .client import (SuperhotWellboreClient, solve_profile,
                     solve_steady_state)
from .config import (CONTROL_MODES, DECLINE_MODES,
                     ROCK_TEMPERATURE_MODES, DeclineConfig,
                     OperatingConfig, ReservoirConfig,
                     RockTemperatureConfig, SolverConfig,
                     SuperhotRequest, TimeConfig, WellConfig)
from .export import (export_all, geophires_input_text,
                     write_geophires_input, write_profile_json,
                     write_request_json, write_temperature_profile)
from .results import (GEOPHIRES_RESERVOIR_MODEL_UPP,
                      PROFILE_TEMPERATURES, ProductionProfile,
                      TimestepResult)

__version__ = '1.0.0'

__all__ = [
    # Client
    'SuperhotWellboreClient',
    'solve_profile',
    'solve_steady_state',
    # Configuration
    'SuperhotRequest',
    'ReservoirConfig',
    'WellConfig',
    'RockTemperatureConfig',
    'OperatingConfig',
    'DeclineConfig',
    'TimeConfig',
    'SolverConfig',
    'DECLINE_MODES',
    'ROCK_TEMPERATURE_MODES',
    'CONTROL_MODES',
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

# adapter.py is deliberately not imported here: it is the only module
# that touches GEOPHIRES, and the file-based path must stay usable
# without it. Import it explicitly when needed:
#
#     from geophires_client.adapter import SuperhotWellboreReservoir
