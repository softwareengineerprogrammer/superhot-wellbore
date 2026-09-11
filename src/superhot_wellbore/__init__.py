# -*- coding: utf-8 -*-
"""
superhot-wellbore
==================

Coupled reservoir-wellbore-power cycle model for single-well power
output from superhot geothermal systems, accompanying:

    Scott, S.W. (2026), Thermo-hydraulic drivers of superhot
    geothermal well performance, Geothermics, 141, 103784.
    https://doi.org/10.1016/j.geothermics.2026.103784

Modules
-------
wellbore_physics
    Wellbore pressure and enthalpy gradient integration (Eqs. 4-5 in
    the manuscript).
reservoir
    Radial Darcy flow model, depth-pressure scaling and
    reservoir-wellbore coupling via bisection.
power_cycle
    Binary and flash power cycle analysis with Baumann wet-stage
    efficiency.
client
    Stable interface exposing the coupled model as a production
    history, used by the GEOPHIRES superhot production wellbore model.

The submodules are not imported here: they pull in the IAPWS-95
equation of state, which is expensive to load. Import what you need::

    from superhot_wellbore.reservoir import solve_flow_for_whp
    from superhot_wellbore.power_cycle import power_cycle_analysis
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version('superhot-wellbore')
except PackageNotFoundError:  # pragma: no cover - not installed
    __version__ = '0.0.0.dev0'

__all__ = ['__version__']
