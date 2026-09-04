# -*- coding: utf-8 -*-
"""
Unit Conversions Between superhot-wellbore and GEOPHIRES
=========================================================

The superhot-wellbore modules (reservoir.py, wellbore_physics.py,
power_cycle.py) share an inter-module boundary convention of MPa for
pressure, MJ/kg for enthalpy, degrees C for temperature and metres
for length.

GEOPHIRES uses a different set of preferred units for the same
quantities:

    Pressure                 kPa    ("Production Wellhead Pressure",
                                      "Reservoir Hydrostatic Pressure")
    Reservoir depth          km     ("Reservoir Depth")
    Well diameter            inch   ("Production Well Diameter")
    Geothermal gradient      C/km   ("Gradient 1")
    Temperature              C      (same as superhot-wellbore)
    Mass flow rate           kg/s   (same as superhot-wellbore)

Every value that crosses the boundary between the two frameworks
passes through one of the helpers below, so the conversion factors
are defined in exactly one place. All helpers pass None through
unchanged, which keeps optional configuration fields simple to
handle.

Author: superhot-wellbore GEOPHIRES client
"""


# ====================================================================
# CONVERSION FACTORS
# ====================================================================

MPA_TO_KPA = 1.0e3
KPA_TO_MPA = 1.0e-3
MJ_PER_KG_TO_KJ_PER_KG = 1.0e3
KJ_PER_KG_TO_MJ_PER_KG = 1.0e-3
M_TO_KM = 1.0e-3
KM_TO_M = 1.0e3
M_TO_INCH = 39.3700787401575
INCH_TO_M = 1.0 / M_TO_INCH
KELVIN_OFFSET = 273.15


# ====================================================================
# PRESSURE
# ====================================================================

def mpa_to_kpa(pressure_MPa):
    """Convert pressure from MPa (superhot-wellbore) to kPa (GEOPHIRES)."""
    if pressure_MPa is None:
        return None
    return pressure_MPa * MPA_TO_KPA


def kpa_to_mpa(pressure_kPa):
    """Convert pressure from kPa (GEOPHIRES) to MPa (superhot-wellbore)."""
    if pressure_kPa is None:
        return None
    return pressure_kPa * KPA_TO_MPA


# ====================================================================
# ENTHALPY
# ====================================================================

def mj_per_kg_to_kj_per_kg(enthalpy_MJkg):
    """Convert specific enthalpy from MJ/kg to kJ/kg."""
    if enthalpy_MJkg is None:
        return None
    return enthalpy_MJkg * MJ_PER_KG_TO_KJ_PER_KG


def kj_per_kg_to_mj_per_kg(enthalpy_kJkg):
    """Convert specific enthalpy from kJ/kg to MJ/kg."""
    if enthalpy_kJkg is None:
        return None
    return enthalpy_kJkg * KJ_PER_KG_TO_MJ_PER_KG


# ====================================================================
# LENGTH
# ====================================================================

def m_to_km(length_m):
    """Convert length from m to km (GEOPHIRES 'Reservoir Depth')."""
    if length_m is None:
        return None
    return length_m * M_TO_KM


def km_to_m(length_km):
    """Convert length from km to m."""
    if length_km is None:
        return None
    return length_km * KM_TO_M


def m_to_inch(length_m):
    """Convert length from m to inch (GEOPHIRES well diameters)."""
    if length_m is None:
        return None
    return length_m * M_TO_INCH


def inch_to_m(length_inch):
    """Convert length from inch to m."""
    if length_inch is None:
        return None
    return length_inch * INCH_TO_M


# ====================================================================
# TEMPERATURE
# ====================================================================

def celsius_to_kelvin(temperature_C):
    """Convert temperature from degrees C to K."""
    if temperature_C is None:
        return None
    return temperature_C + KELVIN_OFFSET


def kelvin_to_celsius(temperature_K):
    """Convert temperature from K to degrees C."""
    if temperature_K is None:
        return None
    return temperature_K - KELVIN_OFFSET


# ====================================================================
# DERIVED QUANTITIES
# ====================================================================

def gradient_C_per_km(T_deep_C, T_surface_C, depth_m):
    """
    Average geothermal gradient for the GEOPHIRES 'Gradient 1' input.

    Parameters
    ----------
    T_deep_C : float
        Temperature at depth [C].
    T_surface_C : float
        Surface temperature [C].
    depth_m : float
        Depth at which T_deep_C applies [m].

    Returns
    -------
    float or None
        Average gradient [C/km], or None if depth is not positive.
    """
    if depth_m is None or depth_m <= 0:
        return None
    return (T_deep_C - T_surface_C) / m_to_km(depth_m)
