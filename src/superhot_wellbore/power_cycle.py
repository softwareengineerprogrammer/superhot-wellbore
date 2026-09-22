# -*- coding: utf-8 -*-
"""
Power Cycle Library for Superhot Geothermal Systems
=====================================================

Surface power conversion module for geothermal fluids, computing
net electrical power output from wellhead conditions using
simplified single-flash and binary (heat-exchanger) steam cycles.

Implements the power cycle model described in:

    Scott, S.W. (2026). Thermo-hydraulic drivers of superhot
    geothermal well performance. Geothermics 141, 103784.
    https://doi.org/10.1016/j.geothermics.2026.103784

    The turbine expansion equations in that paper (Eqs. 6-12)
    follow the standard geothermal power plant thermodynamic
    analysis of DiPippo (2012, Chs. 5, 7, and 8). All equations
    in this module are cited to their textbook origins below.

Modeling assumptions
--------------------
This module provides a simplified upper-bound estimate of gross
turbine power output. The key simplifications are:

    1. No parasitic loads: pump work, cooling tower fans, and gas
       extraction power are neglected. The output therefore
       represents gross turbine power, not net plant output. For
       typical geothermal plants, parasitic loads are 5-15% of
       gross (Mines, 2016, Sec. 13.3.1).

    2. No generator losses: the turbine-to-generator efficiency
       (typically 95-98%) is not applied.

    3. Simplified binary cycle: the working fluid is pure water
       rather than an organic fluid (isobutane, isopentane, etc.).
       This is motivated by the finding that water-based cycles
       may outperform traditional ORC fluids at superhot
       temperatures (Dichter, 2025). The heat exchanger is modeled
       as a simple counter-flow balance with a configurable pinch
       point approach (default 5 K); no detailed pinch analysis
       is performed.

    4. Fixed flash pressure: the flash cycle uses a single flash
       to a fixed pressure (default 1.0 MPa) rather than
       optimizing the separator pressure for maximum specific
       output (cf. DiPippo, 2012, Sec. 5.5-5.6).

    5. Pure water properties: all thermodynamic properties are
       evaluated for pure water via CoolProp (Bell et al., 2014).
       Effects of dissolved salts and noncondensable gases on
       steam properties are neglected.


Interface with coupled_model()
------------------------------
power_cycle_analysis() takes the output dict from
reservoir.coupled_model() (or reservoir.solve_flow_for_whp())
directly as its first argument:

    cm = coupled_model(mdot, P_res, T_res, rock_temps, rp, wp)
    pc = power_cycle_analysis(cm)

The required keys in the input dict are:

    'mass_flow_kgs'    : float - Mass flow rate [kg/s]
    'whp_MPa'          : float - Wellhead pressure [MPa]
    'h_surface_MJkg'   : float - Surface enthalpy [MJ/kg]

Optional keys used when present:

    'h_feedzone_MJkg'  : float - Feedzone enthalpy [MJ/kg]
    'P_bh_MPa'         : float - Bottomhole pressure [MPa]

When h_feedzone_MJkg and P_bh_MPa are both available, feedzone
exergy is computed at the correct (h_fz, P_bh) state. If only
h_feedzone_MJkg is present, P_whp is used as an approximation
for the entropy lookup.

The wellhead temperature is not passed explicitly; it is derived
internally from (h, P) via CoolProp to guarantee thermodynamic
consistency.

Cycle selection
---------------
The cycle is selected automatically based on the wellhead fluid
state (using enthalpy-pressure, not temperature-pressure):

    h_surface >= h_sat,vapor(WHP) + margin  -->  binary cycle
    h_surface <  h_sat,vapor(WHP) + margin  -->  flash cycle

Above the critical pressure (22.064 MPa) the saturation curve ends
and h_sat,vapor(WHP) does not exist, but the wellhead fluid is
single-phase and can still be routed. The boundary is continued as
the critical-point enthalpy h_crit = 2.084 MJ/kg (the endpoint of
the saturated-vapor curve), so selection is continuous across
P_crit: a dense, liquid-like supercritical fluid flashes, a
vapor-like one goes to the binary cycle. Deep superhot wells can
reach the wellhead in this state (e.g. 375 C at 26 MPa).

Binary cycle (superheated or single-phase vapor inlet):
    Geothermal steam transfers heat to a secondary working fluid
    (water at P_wf = 1.0 MPa) via a counter-flow heat exchanger.
    The working fluid mass flow rate is determined from the
    first-law energy balance across the heat exchanger (DiPippo,
    2016, Eqs. 8.9-8.10, 8.13, 8.16; Mines, 2016, Eq. 13.2):

        m_gf * (h_gf,in - h_gf,out) = m_wf * (h_wf,out - h_wf,in)

    The working fluid is then expanded through a two-stage turbine
    (dry expansion from superheated to saturated vapor, then wet
    expansion to condenser pressure).

Single-flash cycle (two-phase inlet):
    The geothermal mixture flashes isenthalpically to P_flash
    (default 1.0 MPa). The wellhead pressure must exceed the flash
    pressure. Saturated vapor is separated and expanded through a
    wet-steam turbine to condenser pressure.

Turbine expansion model
-----------------------
Turbine expansion follows the standard geothermal power plant
thermodynamic analysis described in DiPippo (2012, Chs. 5 and 7).
For superheated inlet conditions, the expansion is split into a
dry stage and a wet stage following the treatment for superheated
inlet steam in DiPippo (2012, Ch. 7, Sec. 7.4.1, Fig. 7.12).

Dry-stage work (superheated region):

    W_dry = m * eta_td * (h_in - h_g(P_sat))

where P_sat is the pressure at which the isentropic expansion
path crosses the saturation curve (x = 1), and eta_td is the
dry turbine isentropic efficiency (DiPippo, 2012, Eq. 7.14).

Wet-stage work (two-phase region):

    W_wet = m * (h_g(P_sat) - h_out)

where h_out is the actual turbine outlet enthalpy computed from
the DiPippo implicit equation that self-consistently couples the
Baumann efficiency correction with the enthalpy balance
(DiPippo, 2012, Eqs. 5.15-5.16 and 7.10-7.11):

    h_out = [h_in - A * (1 - h_f / (h_g - h_f))]
            / [1 + A / (h_g - h_f)]

    A = 0.425 * (h_in - h_out,is)

Here h_f, h_g are saturated liquid and vapor enthalpies at
condenser pressure, h_out,is is the isentropic outlet enthalpy,
and 0.425 = eta_td / 2 for eta_td = 0.85. Because the DiPippo
equation embeds the Baumann efficiency penalty through the
coefficient A, h_out is the actual (not isentropic) outlet
enthalpy and no additional efficiency factor is applied to the
wet-stage work.

The Baumann rule (Baumann, 1921; DiPippo, 2012, Eq. 5.12) states
that 1% average moisture causes approximately 1% drop in turbine
efficiency:

    eta_tw = eta_td * (x_in + x_out) / 2

For a saturated vapor inlet (x_in = 1), this simplifies to:

    eta_tw = eta_td * (1 + x_out) / 2      [DiPippo, 2012, Eq. 7.7]

Flash cycle thermodynamics
--------------------------
The isenthalpic flash and separation processes follow standard
single-flash plant analysis (DiPippo, 2012, Ch. 5, Sec. 5.4):

  - Flashing:    h_1 = h_2 (constant enthalpy; Eq. 5.6)
  - Separation:  x = (h_2 - h_f) / (h_g - h_f) (lever rule; Eq. 5.7)
  - Steam mass:  m_s = x * m_total (Eq. 5.10)

Future extensions
-----------------
Separator pressure optimization:
    The current flash cycle uses a fixed separator pressure. An
    optimized separator temperature can be estimated from the
    equal-temperature-split rule, T_sep = (T_res + T_cond) / 2
    (DiPippo, 2012, Eq. 5.47), or determined numerically by
    maximizing specific power output over a range of separator
    pressures (DiPippo, 2012, Sec. 5.5).

Organic working fluids:
    The binary cycle currently uses water as the working fluid.
    Extension to organic fluids (isobutane, isopentane, R245fa)
    would require CoolProp property calls with the appropriate
    fluid name and careful treatment of retrograde vs.
    non-retrograde expansion behavior (Mines, 2016, Sec. 13.2).
    For superhot applications, Dichter (2025) argues that water-
    based cycles may be competitive or superior due to the high
    source temperatures.

Parasitic load estimation:
    A more complete net power model would subtract pump work
    (feed pump, cooling water pumps), cooling tower fan power,
    and noncondensable gas extraction power. These are typically
    5-15% of gross turbine output (Mines, 2016, Sec. 13.3.1;
    DiPippo, 2016, Sec. 5.4.4).

Double-flash and hybrid cycles:
    Extension to double-flash cycles follows the same framework
    with two separator pressures, each determined by the
    equal-temperature-split rule (DiPippo, 2012, Ch. 6).

Efficiency and performance metrics
----------------------------------
The module computes several standard performance metrics for
both cycle types, following the framework in DiPippo (2016,
Secs. 5.4.7 and 8.2.5) and Mines (2016, Sec. 13.3):

Specific exergy of the incoming geofluid (DiPippo, 2016,
Eq. 5.29; Mines, 2016, Eq. 13.3):

    e = (h_in - h_0) - T_0 * (s_in - s_0)

where T_0 is the ambient (dead-state) temperature in Kelvin,
and h_0, s_0 are saturated liquid properties at T_0 (DiPippo,
2016, p. 203).

Utilization (exergetic) efficiency (DiPippo, 2016, Eqs. 5.31,
8.20; Mines, 2016, Eq. 13.6):

    eta_u = W_net / (m_gf * e)

This metric is computed identically for both binary and flash
cycles, using wellhead conditions as the geofluid inlet state.
It measures how effectively the thermodynamic availability
(exergy) of the geofluid is converted to power.

Thermal efficiency (DiPippo, 2016, Eq. 8.17; Mines, 2016,
Eq. 13.1):

    eta_th = W_net / Q_in

where Q_in = m_gf * (h_gf,in - h_gf,out) is the heat extracted
from the geofluid (Mines, 2016, Eq. 13.2). This metric is
computed for the binary cycle only. For the flash cycle,
thermal efficiency is not conventionally defined because flash
plants are not closed thermodynamic cycles (DiPippo, 2016,
Sec. 5.4.7).

Specific output (Mines, 2016, Sec. 13.3.1):

    w = W_net / m_gf

The ratio of power output to geofluid mass flow rate, also
called brine effectiveness. This metric is independent of
resource conditions and provides a direct measure of how much
power is extracted per unit of produced geofluid.

Default parameters
------------------
    T_reject_C          = 60     Geofluid rejection temperature [C]
    T_ambient_C         = 25     Dead-state (ambient) temperature [C]
    T_wf_inlet_C        = 40     Working fluid HX inlet temp [C]
    delta_T_pinch_C     = 5      HX pinch point temperature diff [C]
    P_wf_MPa            = 1.0    Working fluid turbine inlet pressure [MPa]
    P_condenser_MPa     = 0.01   Condenser pressure (= 10 kPa) [MPa]
    eta_turbine_dry     = 0.85   Dry isentropic turbine efficiency [-]
    P_flash_MPa         = 1.0    Flash separation pressure [MPa]
    x_exit_min          = 0.85   Minimum acceptable exit quality [-]
    superheat_margin_Jkg = 50e3  Cycle selection margin [J/kg]

Units
-----
Inter-module boundary convention: MJ/kg for enthalpy, MPa for
pressure, degrees C for temperature. This applies to all values
passed between reservoir.py, wellbore_physics.py, and this module.
See MODULE_INTERFACE.md for the full specification.

Internally, this module uses J/kg for enthalpy and J/(kg.K) for
entropy, matching CoolProp's native SI output. No unit conversions
are applied to CoolProp return values. The only conversions occur
at the module boundary: MJ/kg * 1e6 -> J/kg on entry, and
J/kg -> MWe (/ 1e6), MJ/kg (/ 1e6) on output.

References
----------
    - Scott, S.W. (2026). Thermo-hydraulic drivers of superhot
      geothermal well performance. Geothermics 141, 103784.
      https://doi.org/10.1016/j.geothermics.2026.103784
    - DiPippo, R. (2012). Geothermal Power Plants: Principles,
      Applications, Case Studies and Environmental Impact, 3rd ed.,
      Butterworth-Heinemann. [Eqs. 5.6-5.17, 5.29-5.31, 5.47, 7.3-7.15]
    - DiPippo, R. (2016). Geothermal Power Plants: Principles,
      Applications, Case Studies and Environmental Impact, 4th ed.,
      Butterworth-Heinemann. [Eqs. 8.9-8.10, 8.13, 8.16-8.20]
    - Baumann, K. (1921). Some recent developments in large steam
      turbine practice. J. Inst. Elect. Eng., 59, 565-623.
    - Mines, G.L. (2016). Binary geothermal energy conversion
      systems. In: Geothermal Power Generation, DiPippo, R. (Ed.),
      Elsevier, Ch. 13, 351-387. [Eqs. 13.1-13.6]
    - Phair, K. and DiPippo, R. (2016). Direct steam geothermal
      energy conversion systems: dry steam and superheated steam
      plants. In: Geothermal Power Generation, Elsevier, Ch. 11,
      291-319.
    - Dichter, D.W. (2025). Water-based geothermal binary cycles.
      Proceedings, 50th Workshop on Geothermal Reservoir Engineering,
      Stanford University, Stanford, CA, SGP-TR-229.
      [Working fluid selection for superhot systems]
    - Bell, I.H., Wronski, J., Quoilin, S., Lemort, V. (2014).
      Pure and pseudo-pure fluid thermophysical property evaluation
      and the open-source thermophysical property library CoolProp.
      Ind. Eng. Chem. Res. 53(6), 2498-2508.

Modules
-------
1. Cycle selection    - Automatic binary/flash dispatch based on WHP state
2. Binary cycle       - HX energy balance + two-stage turbine expansion
3. Flash cycle        - Isenthalpic flash + separation + wet turbine
4. Turbine model      - Dry + wet stage with DiPippo/Baumann corrections
5. Efficiency metrics - Exergy, utilization efficiency, thermal efficiency

Author: Samuel W. Scott
"""

import warnings

import numpy as np
import CoolProp.CoolProp as CP
from scipy.optimize import fsolve, brentq, bisect


# ====================================================================
# DEFAULT PARAMETERS
# ====================================================================

DEFAULT_POWER_PARAMS = {
    'T_reject_C': 60.0,            # Geofluid rejection temperature [C]
    'T_ambient_C': 25.0,           # Dead-state (ambient) temperature [C]
    'T_wf_inlet_C': 40.0,          # Working fluid HX inlet temp [C]
    'delta_T_pinch_C': 5.0,        # HX pinch point temperature diff [C]
    'P_wf_MPa': 1.0,               # Working fluid pressure [MPa]
    'P_condenser_MPa': 0.01,       # Condenser pressure [MPa]
    'eta_turbine_dry': 0.85,       # Dry isentropic turbine efficiency [-]
    'P_flash_MPa': 1.0,            # Flash separation pressure [MPa]
    'x_exit_min': 0.85,            # Min acceptable exit quality [-]
    'superheat_margin_Jkg': 50.0e3,  # Cycle selection margin [J/kg]
}

# Numerical tolerance for detecting superheated vs saturated
# conditions inside the turbine expansion model. This is NOT a
# physical parameter -- it guards against floating-point equality
# when the flash cycle passes h_g as the turbine inlet enthalpy.
_TURBINE_SUPERHEAT_TOL_Jkg = 1.0e3  # 1 kJ/kg

# IAPWS-95 critical point of water (Wagner and Pruss, 2002).
# The saturation curve ends here; CoolProp PQ lookups fail at or
# above _P_CRIT_MPa. _H_CRIT_Jkg is the enthalpy at (T_crit,
# rho_crit), i.e. the endpoint of the saturated-vapor curve.
_P_CRIT_MPa = 22.064
_H_CRIT_Jkg = 2.0843e6


# ====================================================================
# PUBLIC API
# ====================================================================

def power_cycle_analysis(coupled_result, params=None):
    """
    Compute gross turbine power from geothermal wellhead conditions.

    Takes the output dict from reservoir.coupled_model() or
    reservoir.solve_flow_for_whp() directly. Automatically selects
    binary or flash cycle based on the wellhead fluid state.

    Parameters
    ----------
    coupled_result : dict
        Output from reservoir.coupled_model() or solve_flow_for_whp().
        Required keys:
            'mass_flow_kgs'   : float - Mass flow rate [kg/s]
            'whp_MPa'         : float - Wellhead pressure [MPa]
            'h_surface_MJkg'  : float - Surface enthalpy [MJ/kg]
        Optional keys (used if present):
            'h_feedzone_MJkg' : float - Feedzone enthalpy [MJ/kg]
            'P_bh_MPa'        : float - Bottomhole pressure [MPa]
    params : dict or None
        Power cycle parameters. If None, DEFAULT_POWER_PARAMS is
        used. Any missing keys are filled from defaults.

    Returns
    -------
    dict with keys:
        'power_MWe'           : float - Gross turbine power [MWe]
        'cycle'               : str   - 'binary' or 'flash'
        'x_exit'              : float - Turbine exit quality [-]
        'h_exit_MJkg'         : float - Turbine exit enthalpy [MJ/kg]
        'eta_utilization'     : float - Utilization efficiency at WHP [-]
        'eta_thermal'         : float - Thermal efficiency [-]
                                        (NaN for flash cycle)
        'specific_output_MJkg': float - Power per unit geofluid [MJ/kg]
        'exergy_rate_MW'      : float - Exergetic power at WHP [MW]
        'eta_utilization_fz'  : float - Utilization efficiency at
                                        feedzone [-] (NaN if
                                        h_feedzone_MJkg not in input)
        'exergy_rate_fz_MW'   : float - Exergetic power at feedzone [MW]
                                        (NaN if not available)
        'success'             : bool  - True if calculation completed

    Notes
    -----
    Utilization efficiency and specific output are computed
    identically for both cycle types. Thermal efficiency is only
    defined for the binary cycle (DiPippo, 2016, Sec. 5.4.7).
    """
    pp = _merge_params(params)
    fail = {'power_MWe': np.nan, 'cycle': None, 'x_exit': np.nan,
            'h_exit_MJkg': np.nan, 'eta_utilization': np.nan,
            'eta_thermal': np.nan, 'specific_output_MJkg': np.nan,
            'exergy_rate_MW': np.nan, 'eta_utilization_fz': np.nan,
            'exergy_rate_fz_MW': np.nan, 'success': False}

    # Extract required fields
    mass_flow_kgs = coupled_result.get('mass_flow_kgs', 0.0)
    P_whp_MPa = coupled_result.get('whp_MPa', 0.0)
    h_surface_MJkg = coupled_result.get('h_surface_MJkg', np.nan)

    if mass_flow_kgs <= 0 or P_whp_MPa <= 0 or np.isnan(h_surface_MJkg):
        return fail

    # Extract optional fields
    h_feedzone_MJkg = coupled_result.get('h_feedzone_MJkg')
    P_bh_MPa = coupled_result.get('P_bh_MPa')

    h_in_Jkg = h_surface_MJkg * 1e6

    try:
        # Derive temperature from (h, P) for thermodynamic consistency
        T_surface_K = CP.PropsSI('T', 'H', h_in_Jkg,
                                 'P', P_whp_MPa * 1e6, 'Water')
        T_surface_C = T_surface_K - 273.15

        if T_surface_C < 120:
            return fail

        # Saturated vapor enthalpy at WHP for cycle selection
        # (continued through the critical point for supercritical
        # wellhead pressure; see _cycle_selection_enthalpy).
        h_sat_vap = _cycle_selection_enthalpy(P_whp_MPa)

        # Cycle selection based on H-P state
        is_superheated = h_in_Jkg >= (h_sat_vap + pp['superheat_margin_Jkg'])

        if is_superheated:
            result = _binary_cycle(mass_flow_kgs, h_in_Jkg,
                                   T_surface_C, P_whp_MPa, pp)
        else:
            result = _flash_cycle(mass_flow_kgs, h_in_Jkg,
                                  P_whp_MPa, pp)

        if not result.get('success', False):
            return result

        # Efficiency metrics at wellhead conditions
        Q_in_MW = result.pop('Q_in_MW', None)
        metrics = _efficiency_metrics(
            result['power_MWe'], mass_flow_kgs,
            h_in_Jkg, P_whp_MPa, pp['T_ambient_C'],
            Q_in_MW=Q_in_MW)
        result.update(metrics)

        # Feedzone-based exergy (optional).
        # Uses P_bh for entropy lookup when available, giving the
        # correct thermodynamic state at the feedzone. Falls back
        # to P_whp if P_bh is not in the input dict.
        if h_feedzone_MJkg is not None and np.isfinite(h_feedzone_MJkg):
            h_fz_Jkg = h_feedzone_MJkg * 1e6
            P_fz = P_bh_MPa if (P_bh_MPa is not None
                                 and np.isfinite(P_bh_MPa)) else P_whp_MPa
            fz = _efficiency_metrics(
                result['power_MWe'], mass_flow_kgs,
                h_fz_Jkg, P_fz, pp['T_ambient_C'])
            result['eta_utilization_fz'] = fz['eta_utilization']
            result['exergy_rate_fz_MW'] = fz['exergy_rate_MW']
        else:
            result['eta_utilization_fz'] = np.nan
            result['exergy_rate_fz_MW'] = np.nan

        return result

    except Exception as e:
        warnings.warn(f"power_cycle_analysis failed: {e}",
                      RuntimeWarning, stacklevel=2)
        return fail


def dry_steam_specific_work(P_MPa, params=None):
    """
    Gross specific turbine work of saturated steam expanded from P.

    One kg/s of saturated vapour at P_MPa is expanded to the
    condenser pressure by _two_stage_turbine(), i.e. through the same
    DiPippo/Baumann wet-stage model the flash cycle uses once its
    separated steam enters the turbine. This is the dry-steam share of
    a two-phase wellhead stream: a caller that separates the wellhead
    mixture at WHP can price the steam fraction with this number and
    the liquid fraction with any liquid-water correlation.

    Parameters
    ----------
    P_MPa : float
        Turbine inlet (separator) pressure [MPa]. Must lie above the
        condenser pressure and below the critical pressure, where
        saturated vapour exists.
    params : dict or None
        Power cycle parameters, merged over DEFAULT_POWER_PARAMS
        (P_condenser_MPa, eta_turbine_dry and x_exit_min are used).

    Returns
    -------
    float
        Gross specific work [MJ/kg of steam], or NaN when P_MPa is
        not a sub-critical pressure above the condenser pressure or
        when the expansion fails.
    """
    pp = _merge_params(params)
    try:
        P = float(P_MPa)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(P) or P >= _P_CRIT_MPa or P <= pp['P_condenser_MPa']:
        return np.nan
    try:
        h_g = CP.PropsSI('H', 'P', P * 1e6, 'Q', 1, 'Water')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            result = _two_stage_turbine(1.0, h_g, P, pp)
    except Exception:
        return np.nan
    if not result.get('success', False):
        return np.nan
    # power_MWe of a 1 kg/s stream is numerically the specific work in MJ/kg
    return float(result['power_MWe'])


# ====================================================================
# CYCLE SELECTION
# ====================================================================

def _cycle_selection_enthalpy(P_MPa):
    """
    Vapor-like / liquid-like boundary enthalpy [J/kg] at pressure P.

    Below the critical pressure this is the saturated vapor enthalpy
    h_g(P), the threshold used by the cycle selection rule (see the
    module docstring). Above it there is no saturation curve: a PQ
    lookup raises, but the wellhead fluid is single-phase and must
    still be routed to a cycle. The boundary is continued as the
    critical-point enthalpy h_crit, the endpoint of the saturated-
    vapor curve, so selection is continuous across P_crit.

    Together with the default superheat margin (50 kJ/kg) this
    tracks the pseudo-critical (Widom) line to within a few kelvin
    over 22-30 MPa, so a dense, liquid-like supercritical fluid
    (e.g. 375 C at 26 MPa, h = 1.83 MJ/kg) flashes, exactly as the
    same enthalpy would at 21.9 MPa, while a vapor-like fluid
    (e.g. 500 C at 25 MPa, h = 3.17 MJ/kg) goes to the binary cycle.

    Parameters
    ----------
    P_MPa : float
        Wellhead pressure [MPa].

    Returns
    -------
    float
        Boundary enthalpy [J/kg].
    """
    if P_MPa >= _P_CRIT_MPa:
        return _H_CRIT_Jkg
    try:
        return CP.PropsSI('H', 'P', P_MPa * 1e6, 'Q', 1, 'Water')
    except ValueError:
        # Numerical band just below P_crit where the PQ flash does
        # not converge; h_g is within 0.3% of h_crit there.
        return _H_CRIT_Jkg


def is_dense_supercritical(P_MPa, h_MJkg):
    """
    True for a dense, liquid-like supercritical wellhead state.

    A wellhead at or above the critical pressure is single-phase, but
    the cycle selection rule still splits it at the boundary enthalpy
    of _cycle_selection_enthalpy (the critical-point enthalpy h_crit):
    below it the fluid is compressed-liquid-like and this module
    flashes it, above it the fluid is vapor-like and goes to the
    binary cycle. GEOPHIRES uses this helper to send a compressed-
    liquid-like supercritical wellhead to its liquid-water plant
    correlations instead of the flash cycle, so that both programs
    draw the dense-vs-vapor-like line in the same place. The superheat
    margin that the cycle selection adds on top of the boundary is not
    applied here.

    Parameters
    ----------
    P_MPa : float
        Wellhead pressure [MPa].
    h_MJkg : float
        Wellhead specific enthalpy [MJ/kg].

    Returns
    -------
    bool
        True when P_MPa >= P_crit (22.064 MPa) and h_MJkg lies below
        the cycle-selection enthalpy at that pressure; False otherwise,
        including for a sub-critical pressure or a non-finite input.
    """
    if not (np.isfinite(P_MPa) and np.isfinite(h_MJkg)):
        return False
    if P_MPa < _P_CRIT_MPa:
        return False
    return bool(float(h_MJkg) * 1e6 < _cycle_selection_enthalpy(float(P_MPa)))


# ====================================================================
# BINARY CYCLE (superheated inlet)
# ====================================================================

def _binary_cycle(m_gf, h_gf_in, T_gf_C, P_gf_MPa, pp):
    """
    Binary heat-exchanger cycle for superheated geothermal fluid.

    The geofluid inlet enthalpy h_gf_in is passed directly
    from the wellbore simulator output rather than recomputed from
    (T, P), ensuring thermodynamic consistency. Temperature is
    used only for the pinch point approximation (working fluid
    outlet = T_gf - delta_T_pinch).

    The working fluid mass flow rate is determined from the
    first-law energy balance across the heat exchanger (DiPippo,
    2016, Eqs. 8.9-8.10, 8.13, 8.16; Mines, 2016, Eq. 13.2):

        m_gf * (h_gf,in - h_gf,out) = m_wf * (h_wf,out - h_wf,in)

    Returns dict matching power_cycle_analysis() output format.
    """
    fail = {'power_MWe': np.nan, 'cycle': 'binary', 'x_exit': np.nan,
            'h_exit_MJkg': np.nan, 'success': False}

    T_gf_K = T_gf_C + 273.15
    T_reject_K = pp['T_reject_C'] + 273.15
    T_wf_in_K = pp['T_wf_inlet_C'] + 273.15
    T_wf_out_K = T_gf_K - pp['delta_T_pinch_C']  # pinch point
    P_wf = pp['P_wf_MPa']

    try:
        # Geofluid outlet enthalpy at rejection temperature
        h_gf_out = CP.PropsSI('H', 'T', T_reject_K,
                              'P', P_gf_MPa * 1e6, 'Water')

        # Working fluid state points
        h_wf_in = CP.PropsSI('H', 'T', T_wf_in_K,
                             'P', P_wf * 1e6, 'Water')
        h_wf_out = CP.PropsSI('H', 'T', T_wf_out_K,
                              'P', P_wf * 1e6, 'Water')

        dh_wf = h_wf_out - h_wf_in
        if dh_wf <= 0:
            return fail

        # HX energy balance (DiPippo, 2016, Eqs. 8.9-8.10)
        m_wf = m_gf * (h_gf_in - h_gf_out) / dh_wf
        if m_wf <= 0:
            return fail

        # Heat input to cycle (Mines, 2016, Eq. 13.2)
        Q_in_MW = m_gf * (h_gf_in - h_gf_out) / 1e6  # MW

        # Turbine expansion
        result = _two_stage_turbine(m_wf, h_wf_out, P_wf, pp)
        result['cycle'] = 'binary'
        result['Q_in_MW'] = Q_in_MW
        return result

    except Exception:
        return fail


# ====================================================================
# FLASH CYCLE (two-phase inlet)
# ====================================================================

def _flash_cycle(m_total, h_total_Jkg, P_whp_MPa, pp):
    """
    Single-flash steam cycle for two-phase wellhead fluid.

    The geofluid flashes isenthalpically to P_flash (DiPippo,
    2012, Eq. 5.6). Separated vapor fraction is determined from
    the lever rule (Eq. 5.7) and enters the turbine as saturated
    steam at P_flash. The wellhead pressure must exceed the flash
    pressure for the cycle to be feasible.

    Returns dict matching power_cycle_analysis() output format.
    """
    fail = {'power_MWe': np.nan, 'cycle': 'flash', 'x_exit': np.nan,
            'h_exit_MJkg': np.nan, 'success': False}

    P_flash = pp['P_flash_MPa']

    # WHP must exceed flash pressure for separation to work
    if P_whp_MPa < P_flash:
        warnings.warn(
            f"WHP ({P_whp_MPa:.2f} MPa) is below flash pressure "
            f"({P_flash:.2f} MPa). Flash cycle infeasible.",
            RuntimeWarning, stacklevel=3)
        return fail

    try:
        # Saturation properties at flash pressure
        h_f = CP.PropsSI('H', 'P', P_flash * 1e6, 'Q', 0,
                         'Water')
        h_g = CP.PropsSI('H', 'P', P_flash * 1e6, 'Q', 1,
                         'Water')

        if h_total_Jkg <= h_f:
            return fail  # subcooled liquid, no steam

        # Separator quality via lever rule (DiPippo, 2012, Eq. 5.7)
        if h_total_Jkg >= h_g:
            x_flash = 1.0
        else:
            x_flash = (h_total_Jkg - h_f) / (h_g - h_f)

        # Steam mass flow to turbine (DiPippo, 2012, Eq. 5.10)
        m_steam = m_total * x_flash
        if m_steam < 0.1:
            return fail

        # Steam enters turbine as saturated vapor at P_flash
        result = _two_stage_turbine(m_steam, h_g, P_flash, pp)
        result['cycle'] = 'flash'
        return result

    except Exception:
        return fail


# ====================================================================
# TWO-STAGE TURBINE EXPANSION
# ====================================================================

def _two_stage_turbine(m_wf, h_in, P_in_MPa, pp):
    """
    Two-stage turbine expansion: dry stage + wet stage.

    For superheated inlet steam, the expansion is split into a dry
    stage through the superheated region and a wet stage through
    the two-phase region, following DiPippo (2012, Ch. 7,
    Sec. 7.4.1, Fig. 7.12):

    Stage 1 (dry, Eq. 7.14):
        Isentropic expansion from superheated inlet to the pressure
        P_sat where the expansion path first intersects the
        saturated vapor curve (x = 1). The dry turbine efficiency
        eta_td is applied directly.

    Stage 2 (wet, Eqs. 7.10-7.11 / 5.15-5.16):
        Expansion from P_sat to P_condenser through the two-phase
        region. The actual outlet enthalpy h_out is determined by
        the DiPippo implicit equation, which self-consistently
        embeds the Baumann wet-efficiency correction through the
        coefficient A = eta_td / 2. No additional efficiency factor
        is applied to the wet-stage work because h_out already
        reflects irreversibilities.

    For saturated or two-phase inlet (e.g., flash cycle), the dry
    stage is bypassed (W_dry = 0) and the entire expansion occurs
    in the wet stage.

    Parameters
    ----------
    m_wf : float
        Working fluid mass flow rate [kg/s].
    h_in : float
        Turbine inlet enthalpy [J/kg].
    P_in_MPa : float
        Turbine inlet pressure [MPa].
    pp : dict
        Merged power parameters.

    Returns
    -------
    dict with 'power_MWe', 'x_exit', 'h_exit_MJkg', 'success'.
    """
    fail = {'power_MWe': np.nan, 'x_exit': np.nan,
            'h_exit_MJkg': np.nan, 'success': False}

    P_cond = pp['P_condenser_MPa']
    eta_td = pp['eta_turbine_dry']
    x_min = pp['x_exit_min']

    try:
        # Inlet state
        s_in = CP.PropsSI('S', 'H', h_in, 'P', P_in_MPa * 1e6,
                          'Water')

        # Check if inlet is superheated or two-phase
        h_sat_vap_inlet = CP.PropsSI('H', 'P', P_in_MPa * 1e6, 'Q', 1,
                                     'Water')

        if h_in > h_sat_vap_inlet + _TURBINE_SUPERHEAT_TOL_Jkg:
            # --- SUPERHEATED INLET: dry + wet stages ---
            # (DiPippo, 2012, Ch. 7, Sec. 7.4.1, Fig. 7.12)
            P_sat = _find_saturation_pressure(s_in, P_in_MPa, P_cond)

            h_sat = CP.PropsSI('H', 'P', P_sat * 1e6, 'Q', 1,
                               'Water')
            # Dry-stage work (DiPippo, 2012, Eq. 7.14)
            w_dry = eta_td * (h_in - h_sat)  # J/kg
        else:
            # --- TWO-PHASE INLET: wet stage only ---
            P_sat = P_in_MPa
            h_sat = h_in
            w_dry = 0.0

        # --- WET STAGE ---
        # Isentropic outlet state at condenser pressure
        # (DiPippo, 2012, Eq. 5.14 / 7.9)
        h_is_out = CP.PropsSI('H', 'S', s_in,
                              'P', P_cond * 1e6, 'Water')

        # Saturation properties at condenser pressure
        h_f = CP.PropsSI('H', 'P', P_cond * 1e6, 'Q', 0,
                         'Water')
        h_g = CP.PropsSI('H', 'P', P_cond * 1e6, 'Q', 1,
                         'Water')

        # Actual outlet enthalpy from DiPippo implicit equation
        # (DiPippo, 2012, Eqs. 5.15-5.16 / 7.10-7.11).
        # h_sat (the wet-stage inlet) is passed, since the dry
        # stage has already handled expansion from h_in to h_sat.
        # The equation embeds Baumann losses through A = eta_td/2,
        # so h_c is the actual outlet enthalpy.
        h_c = _dipippo_outlet_enthalpy(h_sat, h_is_out, h_f, h_g,
                                       eta_td)

        # Exit quality from lever rule
        if h_g > h_f:
            x_exit_raw = (h_c - h_f) / (h_g - h_f)
            x_exit = np.clip(x_exit_raw, 0.0, 1.0)
        else:
            x_exit = 1.0

        # Wet-stage work (DiPippo, 2012, Eq. 7.15).
        # No additional efficiency factor: h_c already reflects
        # Baumann losses via the implicit equation.
        w_wet = h_sat - h_c  # J/kg

        # --- TOTAL ---
        w_total_specific = w_dry + w_wet   # J/kg
        power_W = m_wf * w_total_specific
        power_MWe = power_W / 1e6

        # --- EXIT QUALITY CHECK ---
        # Typical geothermal turbines operate with up to 14%
        # moisture in the exhaust (Phair and DiPippo, 2016,
        # Sec. 11.5.3.2), corresponding to x_exit >= 0.86.
        if x_exit < x_min:
            warnings.warn(
                f"Turbine exit quality x={x_exit:.3f} is below "
                f"minimum {x_min:.2f}. Excessive moisture may cause "
                f"blade erosion in practice.",
                RuntimeWarning, stacklevel=3)

        return {
            'power_MWe': max(power_MWe, 0.0),
            'x_exit': x_exit,
            'h_exit_MJkg': h_c / 1e6,
            'success': True,
        }

    except Exception:
        return fail


# ====================================================================
# DIPIPPO IMPLICIT EQUATION FOR WET TURBINE OUTLET ENTHALPY
# ====================================================================

def _dipippo_outlet_enthalpy(h_in, h_is_out, h_f, h_g, eta_td):
    """
    Solve the DiPippo implicit equation for the actual turbine
    outlet enthalpy during wet-steam expansion.

    This equation self-consistently couples the Baumann efficiency
    correction (which depends on outlet quality) with the enthalpy
    balance. For a saturated vapor inlet (x_in = 1), the working
    equation is (DiPippo, 2012, Eq. 5.15):

        h_out = [h_in - A * (1 - h_f / h_fg)] / [1 + A / h_fg]

    where:

        A   = 0.425 * (h_in - h_is_out)    (Eq. 5.16)
        h_fg = h_g - h_f                    (latent heat at P_cond)

    The coefficient 0.425 = eta_td / 2 for eta_td = 0.85 arises
    from incorporating the Baumann rule (Eq. 5.12) into the
    isentropic efficiency definition (Eq. 5.9). The equation is
    implicit in h_out because h_out appears on both sides through
    the quality dependence; it is solved iteratively.

    For a wet inlet (x_in < 1), the modified form (Eq. 5.17)
    should be used; this function handles both cases via the
    general residual form.

    Parameters
    ----------
    h_in : float
        Enthalpy entering the wet stage [J/kg]. This is h_g(P_sat)
        for a superheated-inlet turbine, or h_g(P_flash) for a
        flash-cycle turbine.
    h_is_out : float
        Isentropic outlet enthalpy at condenser pressure [J/kg].
    h_f : float
        Saturated liquid enthalpy at condenser pressure [J/kg].
    h_g : float
        Saturated vapor enthalpy at condenser pressure [J/kg].
    eta_td : float
        Dry turbine isentropic efficiency [-].

    Returns
    -------
    float
        Actual outlet enthalpy h_out [J/kg].

    References
    ----------
    Baumann (1921); DiPippo (2012), Eqs. 5.12-5.17, 7.7-7.11.
    """
    A = eta_td / 2.0  # = 0.425 for eta_td = 0.85
    h_fg = h_g - h_f  # latent heat at condenser pressure

    if h_fg <= 0:
        # Degenerate case near critical point
        return h_in - eta_td * (h_in - h_is_out)

    def residual(h_c):
        num = h_in - A * (h_in - h_c) * (1.0 - h_f / h_fg)
        den = 1.0 + A * (h_in - h_is_out) / h_fg
        return h_c - num / den

    h_guess = 0.5 * (h_is_out + h_f)

    try:
        h_c = fsolve(residual, h_guess, full_output=False)[0]
        return h_c
    except Exception:
        # Fallback: simple isentropic efficiency without Baumann
        return h_in - eta_td * (h_in - h_is_out)


# ====================================================================
# BAUMANN RULE
# ====================================================================

def _baumann_efficiency(eta_td, x_outlet):
    """
    Wet isentropic turbine efficiency from the Baumann rule.

    The Baumann rule (Baumann, 1921) states that 1% average
    moisture causes approximately 1% drop in turbine efficiency.
    For a saturated vapor inlet (x_in = 1), the wet efficiency is
    (DiPippo, 2012, Eq. 7.7):

        eta_tw = eta_td * (1 + x_out) / 2

    For dry steam (x = 1), eta_tw = eta_td. For 10% moisture
    (x = 0.90), eta_tw = 0.85 * 1.90 / 2 = 0.808.

    The more general form for a two-phase inlet uses the average
    of inlet and outlet qualities (DiPippo, 2012, Eq. 5.12):

        eta_tw = eta_td * (x_in + x_out) / 2

    Parameters
    ----------
    eta_td : float
        Dry turbine isentropic efficiency [-].
    x_outlet : float
        Vapor quality at turbine outlet (0 to 1) [-].

    Returns
    -------
    float
        Wet isentropic efficiency [-].

    References
    ----------
    Baumann, K. (1921). J. Inst. Elect. Eng., 59, 565-623.
    DiPippo, R. (2012). Eqs. 5.12, 7.7.
    """
    x_clamped = np.clip(x_outlet, 0.0, 1.0)
    return eta_td * (1.0 + x_clamped) / 2.0


# ====================================================================
# SATURATION PRESSURE FINDER
# ====================================================================

def _find_saturation_pressure(s_in, P_in_MPa, P_out_MPa):
    """
    Find the pressure at which an isentropic expansion from P_in
    first crosses the saturated vapor curve (x = 1).

    This defines the boundary between the dry and wet expansion
    stages for a superheated inlet (DiPippo, 2012, Ch. 7,
    Fig. 7.12). Uses Brent's method with bisection fallback.

    Parameters
    ----------
    s_in : float
        Inlet entropy [J/(kg.K)].
    P_in_MPa : float
        Inlet (upper) pressure bound [MPa].
    P_out_MPa : float
        Outlet (lower) pressure bound [MPa].

    Returns
    -------
    float
        Saturation pressure P_sat [MPa].
    """
    P_crit = 22.064  # MPa
    P_lo = max(P_out_MPa, 0.01)
    P_hi = min(P_in_MPa, P_crit * 0.98)

    if P_hi <= P_lo:
        return 0.5 * (P_lo + P_hi)

    def f(P):
        """h_isentropic(P) - h_sat_vapor(P)."""
        try:
            h_is = CP.PropsSI('H', 'S', s_in, 'P', P * 1e6,
                              'Water')
            h_sv = CP.PropsSI('H', 'P', P * 1e6, 'Q', 1,
                              'Water')
            return h_is - h_sv
        except Exception:
            return np.inf

    try:
        f_lo, f_hi = f(P_lo), f(P_hi)
        if np.isinf(f_lo) or np.isinf(f_hi) or f_lo * f_hi > 0:
            return P_lo + 0.6 * (P_hi - P_lo)
        return brentq(f, P_lo, P_hi, xtol=1e-5)
    except Exception:
        try:
            return bisect(f, P_lo, P_hi, xtol=1e-4)
        except Exception:
            return P_lo + 0.6 * (P_hi - P_lo)


# ====================================================================
# EFFICIENCY AND PERFORMANCE METRICS
# ====================================================================

def _efficiency_metrics(W_MWe, m_gf_kgs, h_gf_in, P_gf_MPa,
                        T_ambient_C, Q_in_MW=None):
    """
    Compute thermodynamic performance metrics for the power cycle.

    All metrics use wellhead conditions as the geofluid inlet state,
    providing a consistent basis for comparing binary and flash
    cycles.

    Parameters
    ----------
    W_MWe : float
        Gross turbine power output [MW].
    m_gf_kgs : float
        Total geofluid mass flow rate at wellhead [kg/s].
    h_gf_in : float
        Geofluid specific enthalpy at wellhead [J/kg].
    P_gf_MPa : float
        Geofluid pressure at wellhead [MPa].
    T_ambient_C : float
        Dead-state (ambient) temperature [C].
    Q_in_MW : float or None
        Heat input to power cycle [MW]. Required for thermal
        efficiency; set by _binary_cycle, None for flash.

    Returns
    -------
    dict with keys:
        'eta_utilization'      : float - Utilization efficiency [-]
        'eta_thermal'          : float - Thermal efficiency [-] (NaN
                                         for flash)
        'specific_output_MJkg' : float - Power per unit geofluid [MJ/kg]
        'exergy_rate_MW'       : float - Geofluid exergetic power [MW]

    References
    ----------
    DiPippo (2012), Eqs. 5.29-5.31; DiPippo (2016), Eq. 8.17, 8.20;
    Mines (2016), Eqs. 13.1-13.3, 13.6.
    """
    T_0_K = T_ambient_C + 273.15

    try:
        # Dead-state properties: saturated liquid at T_0
        # (DiPippo, 2016, p. 203)
        h_0 = CP.PropsSI('H', 'T', T_0_K, 'Q', 0,
                          'Water')
        s_0 = CP.PropsSI('S', 'T', T_0_K, 'Q', 0,
                          'Water')

        # Geofluid inlet entropy at wellhead conditions
        s_in = CP.PropsSI('S', 'H', h_gf_in,
                          'P', P_gf_MPa * 1e6,
                          'Water')

        # Specific exergy (DiPippo, 2012, Eq. 5.29; Mines, 2016,
        # Eq. 13.3)
        e_Jkg = ((h_gf_in - h_0)
                  - T_0_K * (s_in - s_0))  # J/kg

        # Exergetic power (DiPippo, 2012, Eq. 5.30)
        E_dot_MW = m_gf_kgs * e_Jkg / 1e6  # MW

        # Utilization efficiency (DiPippo, 2012, Eq. 5.31;
        # DiPippo, 2016, Eq. 8.20; Mines, 2016, Eq. 13.6)
        if E_dot_MW > 0:
            eta_u = W_MWe / E_dot_MW
        else:
            eta_u = np.nan

        # Specific output (Mines, 2016, Sec. 13.3.1)
        if m_gf_kgs > 0:
            specific_output = W_MWe / m_gf_kgs  # MJ/kg
        else:
            specific_output = np.nan

        # Thermal efficiency (DiPippo, 2016, Eq. 8.17; Mines, 2016,
        # Eq. 13.1). Only defined for binary cycles; flash plants
        # are not closed thermodynamic cycles (DiPippo, 2012,
        # Sec. 5.4.7).
        if Q_in_MW is not None and Q_in_MW > 0:
            eta_th = W_MWe / Q_in_MW
        else:
            eta_th = np.nan

    except Exception:
        eta_u = np.nan
        eta_th = np.nan
        specific_output = np.nan
        E_dot_MW = np.nan

    return {
        'eta_utilization': eta_u,
        'eta_thermal': eta_th,
        'specific_output_MJkg': specific_output,
        'exergy_rate_MW': E_dot_MW,
    }


# ====================================================================
# HELPER
# ====================================================================

def _merge_params(user_params):
    """Merge user params with defaults; return complete dict."""
    pp = dict(DEFAULT_POWER_PARAMS)
    if user_params is not None:
        pp.update(user_params)
    return pp
