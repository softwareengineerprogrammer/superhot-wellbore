# -*- coding: utf-8 -*-
"""
wellbore_physics.py -- Steady-state wellbore simulator
======================================================

Steady-state wellbore simulator for single-phase and two-phase upflow
in geothermal wells, with robust thermodynamic property evaluation
near the critical point of water. 

Implements the wellbore model (Eqs. 4-5) described in: 

    Scott, S.W. (2026). Thermo-hydraulic drivers of superhot
    geothermal well performance. Geothermics 141, 103784.
    https://doi.org/10.1016/j.geothermics.2026.103784

Public interface
----------------
Wellbore simulator:
    wellbore_simulate(P_bottom_MPa, feedzone_enthalpy, mass_flow_rate,
                      rock_temperatures, well_params)
        Bottom-to-surface forward-Euler marching model. Returns depth
        profiles of T, P, h, v, x, T_rock and a choked-flow flag.

Equation of state (used by this module and by reservoir.py):
    fluid_properties_Ph(pressure_MPa, enthalpy_Jkg, prev_props)
        Primary thermodynamic lookup from (P, h). Routes to IAPWS97
        near the critical point (|P - P_crit| < 0.1 MPa), CoolProp
        elsewhere. For viscosity, IAPWS97 has a known gap near
        rho_crit; when this occurs, CoolProp's IAPWS-95 viscosity
        is used instead. Returns T, rho, mu, sound speed, phase,
        and vapor quality.
    fluid_properties_TP(temperature_K, pressure_MPa, fallback_enthalpy)
        Convenience wrapper for callers that have (T, P) rather than
        (P, h). Used by reservoir.py for Darcy flow calculations.

Governing equations (Eqs. 4-5)
------------------------------
The simulator solves two coupled ODEs -- momentum (Eq. 4) and
energy (Eq. 5) -- by forward-Euler marching from feedzone to
surface. The formulation follows standard geothermal wellbore
practice as reviewed by Tonkin et al. (2021).

Momentum conservation (Eq. 4):

    dP/dz = -rho*g - (f/2D)*rho*v^2

This is the steady-state, constant-area, homogeneous-flow
momentum equation with hydrostatic and friction terms. It
corresponds to the Nathenson (1974) simplification of
Tonkin et al. (2021, Eq. 102):

    rho*u*du/ds + dP/ds + (2/R)*tau - rho*g*dz/ds = 0

with the convective acceleration term (rho*u*du/ds) set to zero.
This approximation is justified because: (i) in single-phase
superhot wells the density gradient is relatively gradual (no flash front),
so the acceleration term is small compared to gravity and friction;
and (ii) the effect of fluid acceleration on *energy* is still
captured through the kinetic energy term in Eq. 5 (see below).
The friction term uses the Darcy-Weisbach formulation with the
Swamee-Jain (1976) explicit approximation to the Colebrook-White
equation (see friction_factor() docstring). For two-phase
conditions, a homogeneous mixture density and viscosity are used
in the single-phase friction formula rather than a two-phase
friction multiplier (see "Modeling assumptions" below).

Energy conservation (Eq. 5):

    dh/dz = -g + (mdot/A)^2 * (1/rho^3) * (drho/dz)
            - (U/mdot) * (T - T_rock)

This is the steady-state Bernoulli equation with heat loss,
derived from the first law of thermodynamics for a constant-area,
constant-mass-flow system. The three terms represent gravitational
potential energy, kinetic energy change, and conductive heat loss
to the surrounding rock. The parent equation is Tonkin et al.
(2021, Eq. 65) for single-phase flow:

    d/dt[rho*(e + u^2/2)] + (1/A)*d/ds[A*rho*u*(h + u^2/2)]
        - rho*u*g*dz/ds + (2/R)*Qheat + qener = 0

or equivalently, Tonkin et al. (2021, Eq. 99) for homogeneous
two-phase flow (substituting uv = ul = u). At steady state and
with no mass sources (qener = 0), Eq. 65/99 reduces to the
well-known mechanical energy balance (see enthalpy_gradient()
docstring for the full derivation):

    dh/dz + d(v^2/2)/dz + g + (U/mdot)*(T - T_rock) = 0

NOTE ON FRICTION IN THE ENERGY EQUATION: Friction does not appear
as an explicit term in Eq. 5. This is a property of the enthalpy
formulation, not an approximation. Since h = e + P/rho, pressure
work done against friction is converted to internal energy and
remains in the enthalpy. That is, the frictional pressure drop
in Eq. 4 reduces the flow-work component of enthalpy but
increases internal energy by the same amount (viscous dissipation
heats the fluid), so the net effect on specific enthalpy is zero.
Friction affects the pressure profile (Eq. 4) but not the
enthalpy profile (Eq. 5) -- these are coupled only through the
equation of state rho(P, h). This is the standard result for
pipe flow; see, e.g., Bird, Stewart & Lightfoot (2002, Ch. 15)
or Tonkin et al. (2021, discussion below Eq. 74).

Modeling assumptions
--------------------
This simulator is designed for superhot geothermal wells producing
from reservoirs at temperatures exceeding ~375 C. Under these
conditions, fluid enters the wellbore as single-phase supercritical
water or superheated vapor and may partially condense into the
two-phase envelope during ascent (particularly at high reservoir
pressures, > ~25 MPa). When two-phase conditions occur, the liquid
saturation is typically small (high vapor fraction, annular/mist
flow regime), making the homogeneous equilibrium approximation
reasonable. Tonkin et al. (2021) note that the homogeneous
formulation is "considered reasonably accurate for cases with high
vapour saturation (known as mist flow) where the entrained liquid
droplets are travelling at approximately the same velocity as the
vapour phase or for wells with very high mass flow rates."

The key approximations are:
    1. Homogeneous flow: liquid and vapor travel at the same velocity
       (no phase slip). This is valid for high-quality (high vapor
       fraction) flow but becomes inaccurate at low vapor quality
       where buoyancy-driven slip is significant. Slip does not affect
       bulk enthalpy but does affect the pressure gradient.
    2. Convective acceleration neglected in the momentum equation:
       the rho*u*du/ds term is omitted from Eq. 4, following
       Nathenson (1974). This is appropriate when density changes
       are gradual (no abrupt flash front), as is the case for
       superhot single-phase or high-quality two-phase flow. The
       kinetic energy effect on enthalpy is still captured in Eq. 5.
    3. Homogeneous friction model: the Darcy-Weisbach friction factor
       is computed from the mixture Reynolds number rather than using
       a two-phase friction multiplier (e.g., Lockhart-Martinelli).
       This is consistent with the homogeneous flow assumption and
       is the "mixture method" described by Tonkin et al. (2021).
    4. Constant heat-loss coefficient: the conductive heat transfer
       between the wellbore and surrounding rock is parameterized by
       a single coefficient U [W/m/K] that is assumed constant along
       the entire well. See "Future extensions" below.
    5. Choked flow detection: when the fluid velocity exceeds the
       local speed of sound, the simulation flags the condition but
       continues marching. The resulting profiles above the choke
       point are approximate (the real system would back-pressure
       and reduce the inflow to the maximum choked flow rate).
       The choked flag is returned so that upstream solvers
       (e.g., solve_flow_for_whp) can find the maximum non-choked
       flow rate iteratively.

Future extensions
-----------------
Phase slip:
    To account for differential phase velocities, the momentum equation
    (pressure_gradient) would need to include the slip momentum flux
    term gamma (Tonkin et al., 2021, Eq. 70). The drift-flux model
    of Shi et al. (2005) is the most common approach in geothermal
    wellbore simulators. This requires solving for the void fraction
    Sv from the drift-flux relation (Tonkin et al., 2021, Eq. 56)
    rather than computing it from the homogeneous assumption. The
    pressure gradient would then use the profile-adjusted mixture
    density rho*_mix instead of the homogeneous rho_mix, and the
    friction term would use a two-phase friction multiplier rather
    than a single-phase Darcy-Weisbach factor. The energy equation
    is less affected because slip does not change bulk enthalpy
    (Tonkin et al., 2021), though the kinetic energy term would need
    separate treatment of phase velocities.

Convective acceleration:
    Adding the rho*u*du/ds term to the momentum equation would give
    the full non-conservative form (Tonkin et al., 2021, Eq. 102).
    This matters primarily for flashing wells with abrupt density
    changes. For superhot wells, Tonkin et al. (2021, Section 7.1)
    showed that the approximate momentum flux can overestimate the
    pressure loss due to acceleration, with errors up to ~40% in
    wellhead pressure in extreme cases. For our target application
    (single-phase or high-quality two-phase), the effect is small.

Depth- or temperature-dependent heat-loss coefficient:
    The current model uses a constant U [W/m/K]. For a more realistic
    treatment, U could be made a function of depth or local rock
    temperature. The enthalpy_gradient() function would need to accept
    U(z) or U(T_rock) instead of a scalar. Physically, U depends on
    the thermal conductivity of the formation, well completion
    (casing, cement, insulation), and time since well start-up. The
    default value (2.5 W/m/K) is derived by Albertsson et al. (2003)
    from the analytical solution of Carslaw and Jaeger (1959) for
    radial heat conduction from a cylindrical source, applied to a
    9 5/8" well in Icelandic basalt after 1 year of production. A
    time-dependent model (e.g., Ramey, 1962) or a coupled radial
    heat equation (e.g., Garcia-Valladares et al., 2006) would
    provide a more complete treatment, particularly for transient
    or ultra-deep scenarios.

Default parameters
------------------
DEFAULT_WELL_PARAMS contains the reference values from
Scott (2026, Table 1):

    diameter_m       = 0.217   Internal diameter [m] (IDDP-1)
    depth_m          = 2100    Well depth [m]
    delta_z_m        = 10      Vertical step size [m]
    heat_loss_factor = 2.5    Heat-loss coefficient U [W/m/K]
    roughness_m      = 0.046e-3  Pipe roughness [m], commercial steel

References
----------
Scott, S.W. (2026). Thermo-hydraulic drivers of superhot geothermal
    well performance. Geothermics 141, 103784.
    https://doi.org/10.1016/j.geothermics.2026.103784
Tonkin, R.A., O'Sullivan, M.J., O'Sullivan, J.P. (2021). A review
    of mathematical models for geothermal wellbore simulation.
    Geothermics 97, 102255.
Nathenson, M. (1974). Flashing flow in hot-water geothermal wells.
    GRC Transactions 4, 223-226.
Swamee, P.K. and Jain, A.K. (1976). Explicit equations for pipe-flow
    problems. J. Hydraul. Div. 102(5), 657-664.
Bird, R.B., Stewart, W.E., Lightfoot, E.N. (2002). Transport
    Phenomena, 2nd ed. John Wiley & Sons.
Shi, H. et al. (2005). Drift-flux modeling of two-phase flow in
    wellbores. SPE Journal 10, 24-33.
Wagner, W. and Pruss, A. (2002). The IAPWS Formulation 1995 for
    the thermodynamic properties of ordinary water substance for
    general and scientific use. J. Phys. Chem. Ref. Data 31, 387-535.
Wagner, W. et al. (2000). The IAPWS Industrial Formulation 1997
    for the thermodynamic properties of water and steam. J. Eng. Gas
    Turbines Power 122, 150-184.
Bell, I.H., Wronski, J., Quoilin, S., Lemort, V. (2014). Pure
    and pseudo-pure fluid thermophysical property evaluation and the
    open-source thermophysical property library CoolProp. Ind. Eng.
    Chem. Res. 53(6), 2498-2508.
Albertsson, A., Bjarnason, J.O., Gunnarsson, T. (2003). IDDP
    Feasibility Report Part 3: Fluid Handling and Evaluation.
    (Heat-loss coefficient derived from Carslaw and Jaeger, 1959.)
Ingason, K., Kristjansson, V., Einarsson, K. (2014). Design and
    development of the discharge system of IDDP-1. Geothermics 49,
    58-65.

Modules
-------
1. Fluid properties  - P-h and T-P lookups (IAPWS97 near-critical, CoolProp elsewhere)
2. Hydraulics        - Reynolds number, friction factor, pressure gradient
3. Energy balance    - Enthalpy gradient with kinetic energy correction
4. Simulator         - Bottom-to-surface marching wellbore model

Author: Samuel W. Scott
"""

import warnings
import numpy as np
import CoolProp.CoolProp as CP
from iapws import IAPWS97
from math import pi


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
GRAVITY = 9.81              # Gravitational acceleration [m/s2]
P_CRIT_MPA = 22.064         # Critical pressure of water [MPa]
T_CRIT_K = 647.096          # Critical temperature of water [K]
RHO_CRIT_KGM3 = 322.0      # Critical density of water [kg/m3]

# Viscosity of water near the critical point. Last-resort fallback
# when BOTH IAPWS97 and CoolProp fail for viscosity at
# |P - P_crit| < 0.1 MPa and rho ~ rho_crit. In normal operation,
# the code first tries CoolProp's IAPWS-95 viscosity (which handles
# these conditions correctly); this constant is only reached if
# CoolProp also fails. The value 4.0e-5 Pa.s is representative of
# CoolProp's IAPWS-95 result at h ~ 2100 kJ/kg, P ~ 22 MPa
# (mu = 3.8-3.9e-5). At the critical point itself, mu = 4.2e-5 Pa.s
# (Huber et al., 2009, J. Phys. Chem. Ref. Data 38, 101-125).
MU_NEAR_CRITICAL_PAS = 4.0e-5

# ---------------------------------------------------------------------------
# Default well parameters
# ---------------------------------------------------------------------------
# These defaults can be imported and selectively overridden by scripts.
# Diameter based on IDDP-1 production casing (Ingason et al., 2014).
# Heat-loss coefficient from Albertsson et al. (2003), derived from
# Carslaw and Jaeger (1959) for a 9 5/8" well in Icelandic basalt
# after 1 year of production.
DEFAULT_WELL_PARAMS = {
    'diameter_m': 0.217,         # Internal diameter [m] (IDDP-1)
    'depth_m': 2100,             # Well depth [m]
    'delta_z_m': 10,             # Vertical step size [m]
    'heat_loss_factor': 2.5,     # Heat-loss coefficient U [W/m/K]
    'roughness_m': 0.046e-3,     # Pipe roughness [m], commercial steel
}


# ===================================================================
# 1. FLUID PROPERTIES
# ===================================================================

def _two_phase_properties(quality, rho_f, rho_g, mu_dyn_f, mu_dyn_g):
    """
    Compute homogeneous two-phase mixture density and dynamic viscosity.

    Density
    -------
    The bulk density is computed from the mass-weighted specific volume:

        1/rho_bulk = x/rho_g + (1-x)/rho_f

    This is equivalent to the volume-weighted form rho = alpha*rho_g +
    (1-alpha)*rho_f, where alpha is the homogeneous void fraction.

    Dynamic viscosity
    -----------------
    Following the standard approach for two-phase reservoir/wellbore
    flow (e.g., Horne, 2016), the bulk kinematic viscosity is computed
    as a mass-weighted average of the phase kinematic viscosities:

        nu_bulk = x * nu_g + (1-x) * nu_f

    where nu = mu/rho is kinematic viscosity [m2/s]. The bulk dynamic
    viscosity is then:

        mu_bulk = rho_bulk * nu_bulk

    Note: this is algebraically equivalent to the volume-weighted
    dynamic viscosity form (alpha*mu_g + (1-alpha)*mu_f), since
    alpha*rho_g = rho_bulk*x by construction. The mass-weighted
    kinematic form is used here for physical transparency.

    Parameters
    ----------
    quality   : float - Vapor mass fraction x [0-1]
    rho_f     : float - Saturated liquid density [kg/m3]
    rho_g     : float - Saturated vapor density [kg/m3]
    mu_dyn_f  : float - Saturated liquid dynamic viscosity [Pa.s]
    mu_dyn_g  : float - Saturated vapor dynamic viscosity [Pa.s]

    Returns
    -------
    tuple of (density, mu_dyn, alpha)
        density : float - Bulk mixture density [kg/m3]
        mu_dyn  : float - Bulk dynamic viscosity [Pa.s]
        alpha   : float - Void fraction [-]
    """
    x = quality

    # Bulk density from mass-weighted specific volume
    density = 1.0 / (x / rho_g + (1.0 - x) / rho_f)

    # Kinematic viscosities of each phase [m2/s]
    nu_kin_f = mu_dyn_f / rho_f
    nu_kin_g = mu_dyn_g / rho_g

    # Mass-weighted bulk kinematic viscosity
    nu_kin_bulk = x * nu_kin_g + (1.0 - x) * nu_kin_f

    # Convert back to dynamic viscosity
    mu_dyn_bulk = density * nu_kin_bulk

    # Void fraction (for reference / profile output)
    alpha = x * rho_f / (x * rho_f + (1.0 - x) * rho_g)

    return density, mu_dyn_bulk, alpha


def fluid_properties_Ph(pressure_MPa, enthalpy_Jkg, prev_props=None):
    """
    Compute fluid properties from pressure and specific enthalpy.

    This is the primary thermodynamic lookup. It uses the P-h formulation
    rather than P-T to avoid the singularity at the critical point where
    the saturation curve terminates and T(P) becomes multi-valued in the
    two-phase region.

    The routing logic is:
        1. If P is within the near-critical/pseudocritical zone
           (21.5 < P < 27 MPa), use IAPWS97 for T, rho, and sound
           speed. IAPWS97 uses explicit backward equations T(P,h)
           and v(P,h) -- direct polynomial evaluations with no
           iteration. CoolProp (IAPWS-95) must iteratively invert
           the Helmholtz free energy surface, which has extreme
           curvature along the pseudocritical ridge (where cp
           diverges and compressibility peaks). This curvature makes
           the iterative solver sensitive to initial guesses and can
           produce step-to-step oscillations in T and rho that
           propagate into the enthalpy and pressure marching scheme.
           IAPWS97 sacrifices marginal accuracy (~0.01-0.1% vs
           IAPWS-95) for deterministic, iteration-free evaluation
           that eliminates these numerical artifacts.
           For viscosity, IAPWS97 has a known gap at rho ~ rho_crit
           where it returns mu=None. When this occurs, the function
           tries CoolProp's IAPWS-95 viscosity correlation (which
           handles these conditions correctly). If CoolProp also
           fails, a constant fallback MU_NEAR_CRITICAL_PAS = 4.0e-5
           Pa.s is used (see module-level constant for documentation).
           The same CoolProp-first strategy applies to saturated
           phase viscosities (mu_f, mu_g) for two-phase states in
           the near-critical zone.
        2. Otherwise, use CoolProp (IAPWS-95 via Helmholtz EOS). For
           subcritical pressures, first check whether enthalpy falls
           between h_f(P) and h_g(P) to detect two-phase conditions
           and compute mixture properties explicitly.
        3. If both EOS calls fail and prev_props is provided, return
           the previous step's properties unchanged (with success=False
           and a RuntimeWarning). No additional smoothing is applied;
           the simulator uses these values directly.
           If prev_props is None (first step), a RuntimeError is raised.

    Phase classification:
        - 'two_phase'          : h_f(P) <= h <= h_g(P) at subcritical P
        - 'single_phase_vapor' : rho < rho_crit (322 kg/m3)
        - 'single_phase_liquid': rho >= rho_crit (322 kg/m3)

    Parameters
    ----------
    pressure_MPa : float
        Pressure [MPa].
    enthalpy_Jkg : float
        Specific enthalpy [J/kg].
    prev_props : dict or None
        Property dict from the previous marching step, used as fallback.
        Expected keys: temperature_K, density_kgm3, viscosity_Pas,
        pressure_MPa, sound_speed_ms.

    Returns
    -------
    dict
        phase          : str   - 'two_phase', 'single_phase_vapor', or
                                 'single_phase_liquid'
        temperature_K  : float - Temperature [K]
        density_kgm3   : float - Density [kg/m3]
        viscosity_Pas  : float - Dynamic viscosity [Pa.s]
                                 (CoolProp key 'V' = dynamic viscosity)
        sound_speed_ms : float - Speed of sound [m/s], NaN for two-phase
        saturation     : float - Vapor quality [0-1], NaN if single-phase
        success        : bool  - True if EOS call succeeded
        pressure_MPa   : float - Echo of input pressure

    Raises
    ------
    RuntimeError
        If both EOS calls fail and no prev_props is available.
    """
    pressure_Pa = pressure_MPa * 1e6

    # Route to IAPWS97 in the near-critical and pseudocritical zone.
    # Below P_crit: 0.5 MPa buffer captures the saturation boundary.
    # Above P_crit: extends to 27 MPa to cover the pseudocritical
    # ridge where CoolProp's Helmholtz solver oscillates due to
    # extreme curvature of the free energy surface (cp divergence,
    # high compressibility). IAPWS97's explicit backward equations
    # are iteration-free and produce smooth, deterministic results.
    near_critical = (P_CRIT_MPA - 0.5) < pressure_MPa < (P_CRIT_MPA + 5.0)

    def _classify_single_phase(rho):
        """Assign phase from density relative to critical density."""
        if rho < RHO_CRIT_KGM3:
            return 'single_phase_vapor'
        else:
            return 'single_phase_liquid'

    def _make_result(phase, T, rho, mu, sat, sound_speed=np.nan):
        """Build the standard return dict."""
        return {
            'phase': phase,
            'temperature_K': T,
            'density_kgm3': rho,
            'viscosity_Pas': mu,
            'sound_speed_ms': sound_speed,
            'saturation': sat,
            'success': True,
            'pressure_MPa': pressure_MPa,
        }

    # ------------------------------------------------------------------
    # Path A: Near-critical -- route to IAPWS97
    # ------------------------------------------------------------------
    if near_critical:
        try:
            fluid = IAPWS97(P=pressure_MPa, h=enthalpy_Jkg / 1000.0)
            if fluid.T is not None and fluid.rho is not None:
                temperature = fluid.T
                density = fluid.rho
                sound_speed = fluid.w if fluid.w is not None else np.nan

                if fluid.mu is not None:
                    viscosity = fluid.mu
                else:
                    # IAPWS97 returned mu=None (known gap at rho ~ rho_crit).
                    # Try CoolProp's IAPWS-95 viscosity correlation, which
                    # handles these conditions correctly.
                    try:
                        viscosity = CP.PropsSI('V', 'P', pressure_Pa, 'H',
                                               enthalpy_Jkg, 'Water')
                    except Exception:
                        # Both IAPWS97 and CoolProp failed for viscosity.
                        # Fall back to a constant representative of
                        # near-critical conditions (see module docstring).
                        viscosity = MU_NEAR_CRITICAL_PAS
                        warnings.warn(
                            f"Both IAPWS97 and CoolProp failed for viscosity "
                            f"at P={pressure_MPa:.3f} MPa, "
                            f"h={enthalpy_Jkg/1e6:.3f} MJ/kg; using "
                            f"mu={viscosity:.1e} Pa.s (constant fallback)",
                            RuntimeWarning, stacklevel=2)

                # Check for two-phase at subcritical P
                if pressure_MPa < P_CRIT_MPA:
                    try:
                        sat_liq = IAPWS97(P=pressure_MPa, x=0)
                        sat_vap = IAPWS97(P=pressure_MPa, x=1)
                        h_f = sat_liq.h * 1000.0  # J/kg
                        h_g = sat_vap.h * 1000.0

                        if h_f <= enthalpy_Jkg <= h_g:
                            quality = (enthalpy_Jkg - h_f) / (h_g - h_f)
                            rho_f = sat_liq.rho
                            rho_g = sat_vap.rho
                            mu_f = sat_liq.mu
                            mu_g = sat_vap.mu

                            # Viscosity fallback for saturated phases.
                            # IAPWS97 can return mu=None for one or both
                            # phases near the critical point. Try CoolProp
                            # first, then fall back to constants.
                            if mu_f is None or mu_g is None:
                                try:
                                    if mu_f is None:
                                        mu_f = CP.PropsSI('V', 'P', pressure_Pa,
                                                          'Q', 0, 'Water')
                                    if mu_g is None:
                                        mu_g = CP.PropsSI('V', 'P', pressure_Pa,
                                                          'Q', 1, 'Water')
                                except Exception:
                                    # CoolProp also failed; use constants
                                    # from IAPWS-95 at P = 21.9-22.05 MPa.
                                    mu_f = mu_f if mu_f is not None else 4.7e-5
                                    mu_g = mu_g if mu_g is not None else 3.7e-5
                                    warnings.warn(
                                        f"Both IAPWS97 and CoolProp failed "
                                        f"for saturated viscosity at "
                                        f"P={pressure_MPa:.3f} MPa; using "
                                        f"mu_f={mu_f:.1e}, mu_g={mu_g:.1e} "
                                        f"Pa.s (constant fallback)",
                                        RuntimeWarning, stacklevel=2)

                            density, viscosity, _ = _two_phase_properties(
                                quality, rho_f, rho_g, mu_f, mu_g)
                            return _make_result('two_phase', sat_liq.T,
                                                density, viscosity, quality)
                        # Single-phase subcritical
                        return _make_result(
                            _classify_single_phase(density), temperature,
                            density, viscosity, np.nan, sound_speed)
                    except Exception:
                        pass  # Fall through to single-phase classification

                # Supercritical P or subcritical single-phase
                return _make_result(
                    _classify_single_phase(density), temperature,
                    density, viscosity, np.nan, sound_speed)
        except Exception:
            pass  # Fall through to CoolProp

    # ------------------------------------------------------------------
    # Path B: General conditions -- CoolProp (IAPWS-95)
    # ------------------------------------------------------------------
    try:
        if pressure_Pa < P_CRIT_MPA * 1e6:
            # Subcritical: check for two-phase
            try:
                h_f = CP.PropsSI('H', 'P', pressure_Pa, 'Q', 0, 'Water')
                h_g = CP.PropsSI('H', 'P', pressure_Pa, 'Q', 1, 'Water')

                if h_f <= enthalpy_Jkg <= h_g:
                    quality = (enthalpy_Jkg - h_f) / (h_g - h_f)
                    T_sat = CP.PropsSI('T', 'P', pressure_Pa, 'Q', 0, 'Water')
                    rho_f = CP.PropsSI('D', 'P', pressure_Pa, 'Q', 0, 'Water')
                    rho_g = CP.PropsSI('D', 'P', pressure_Pa, 'Q', 1, 'Water')
                    mu_f = CP.PropsSI('V', 'P', pressure_Pa, 'Q', 0, 'Water')
                    mu_g = CP.PropsSI('V', 'P', pressure_Pa, 'Q', 1, 'Water')

                    density, viscosity, _ = _two_phase_properties(
                        quality, rho_f, rho_g, mu_f, mu_g)
                    return _make_result('two_phase', T_sat,
                                        density, viscosity, quality)
                else:
                    # Single-phase subcritical
                    temperature = CP.PropsSI('T', 'P', pressure_Pa, 'H',
                                             enthalpy_Jkg, 'Water')
                    density = CP.PropsSI('D', 'P', pressure_Pa, 'H',
                                         enthalpy_Jkg, 'Water')
                    viscosity = CP.PropsSI('V', 'P', pressure_Pa, 'H',
                                           enthalpy_Jkg, 'Water')
                    sound_speed = CP.PropsSI('A', 'P', pressure_Pa, 'H',
                                             enthalpy_Jkg, 'Water')
                    return _make_result(
                        _classify_single_phase(density), temperature,
                        density, viscosity, np.nan, sound_speed)

            except Exception:
                # Saturation lookup failed; try direct P-H
                temperature = CP.PropsSI('T', 'P', pressure_Pa, 'H',
                                         enthalpy_Jkg, 'Water')
                density = CP.PropsSI('D', 'P', pressure_Pa, 'H',
                                     enthalpy_Jkg, 'Water')
                viscosity = CP.PropsSI('V', 'P', pressure_Pa, 'H',
                                       enthalpy_Jkg, 'Water')
                sound_speed = CP.PropsSI('A', 'P', pressure_Pa, 'H',
                                         enthalpy_Jkg, 'Water')
                return _make_result(
                    _classify_single_phase(density), temperature,
                    density, viscosity, np.nan, sound_speed)
        else:
            # P >= P_crit: single-phase, classify by density
            temperature = CP.PropsSI('T', 'P', pressure_Pa, 'H',
                                     enthalpy_Jkg, 'Water')
            density = CP.PropsSI('D', 'P', pressure_Pa, 'H',
                                 enthalpy_Jkg, 'Water')
            viscosity = CP.PropsSI('V', 'P', pressure_Pa, 'H',
                                   enthalpy_Jkg, 'Water')
            sound_speed = CP.PropsSI('A', 'P', pressure_Pa, 'H',
                                     enthalpy_Jkg, 'Water')
            return _make_result(
                _classify_single_phase(density), temperature,
                density, viscosity, np.nan, sound_speed)

    except Exception:
        pass  # Fall through to fallback

    # ------------------------------------------------------------------
    # Path C: Both EOS failed -- return prev_props unchanged
    # ------------------------------------------------------------------
    if prev_props is not None:
        warnings.warn(
            f"EOS failed at P={pressure_MPa:.3f} MPa, "
            f"h={enthalpy_Jkg/1e6:.3f} MJ/kg; holding properties from "
            f"previous step (rho={prev_props['density_kgm3']:.1f} kg/m3, "
            f"T={prev_props['temperature_K']-273.15:.1f} C, "
            f"mu={prev_props['viscosity_Pas']:.2e} Pa.s)",
            RuntimeWarning, stacklevel=2)
        return {
            'phase': prev_props.get('phase', 'held_from_previous'),
            'temperature_K': prev_props['temperature_K'],
            'density_kgm3': prev_props['density_kgm3'],
            'viscosity_Pas': prev_props['viscosity_Pas'],
            'sound_speed_ms': prev_props.get('sound_speed_ms', np.nan),
            'saturation': np.nan,
            'success': False,
            'pressure_MPa': pressure_MPa,
        }
    else:
        raise RuntimeError(
            f"EOS failed at P={pressure_MPa:.3f} MPa, "
            f"h={enthalpy_Jkg/1e6:.3f} MJ/kg and no previous state "
            f"available. Check initial conditions.")


def fluid_properties_TP(temperature_K, pressure_MPa, fallback_enthalpy=None):
    """
    Compute fluid properties from temperature and pressure.

    This is a convenience utility for external callers (e.g., reservoir
    models, validation scripts) that have T and P rather than P and h.
    It is NOT used by wellbore_simulate(), which uses the P-h formulation
    exclusively because enthalpy is the integrated variable.

    CoolProp's T-P solver uses a different numerical path than the P-h
    solver and can succeed in some edge cases where P-h fails (and vice
    versa). If CoolProp raises a saturation-boundary error (T-P state
    within ~1e-4 of the saturation curve), this function falls back to
    fluid_properties_Ph using the provided fallback_enthalpy.

    Parameters
    ----------
    temperature_K : float
        Temperature [K].
    pressure_MPa : float
        Pressure [MPa].
    fallback_enthalpy : float or None
        Enthalpy [J/kg] to pass to fluid_properties_Ph() if CoolProp's
        T-P solver fails at the saturation boundary.

    Returns
    -------
    tuple of (density, dynamic_viscosity, enthalpy, sound_speed)
        Units: kg/m3, Pa.s, J/kg, m/s
    """
    pressure_Pa = pressure_MPa * 1e6

    try:
        density = CP.PropsSI('D', 'T', temperature_K, 'P', pressure_Pa, 'Water')
        viscosity = CP.PropsSI('V', 'T', temperature_K, 'P', pressure_Pa, 'Water')
        enthalpy = CP.PropsSI('H', 'T', temperature_K, 'P', pressure_Pa, 'Water')
        sound_speed = CP.PropsSI('A', 'T', temperature_K, 'P', pressure_Pa, 'Water')
        return density, viscosity, enthalpy, sound_speed

    except Exception as e:
        if ("saturation pressure" in str(e).lower()
                or "within 1e-4" in str(e)):
            # Near saturation boundary: route through P-h lookup.
            # Sound speed is set to a nominal liquid-water value because
            # fluid_properties_Ph does not return sound speed for two-phase
            # states (returns NaN). This fallback is only used by
            # fluid_properties_TP, which is called by reservoir.py for
            # Darcy flow calculations -- not by wellbore_simulate(), which
            # uses fluid_properties_Ph directly and gets sound speed from
            # the EOS. The value 1500 m/s (approximate liquid water at
            # ambient conditions) is not used in any physics calculation;
            # it exists only to fill the return tuple.
            result = fluid_properties_Ph(pressure_MPa, fallback_enthalpy, None)
            sound_speed = 1500.0
            return (result['density_kgm3'], result['viscosity_Pas'],
                    fallback_enthalpy, sound_speed)
        else:
            raise


# ===================================================================
# 2. HYDRAULICS
# ===================================================================

def reynolds_number(velocity, diameter, density, viscosity):
    """
    Reynolds number for internal pipe flow.

        Re = rho * v * D / mu

    This is the standard definition for flow in circular pipes
    (see, e.g., Bird, Stewart & Lightfoot, 2002, Ch. 6). For
    two-phase conditions, the homogeneous mixture density and
    viscosity from _two_phase_properties() are used, consistent
    with the "mixture method" for two-phase friction (Tonkin et al.,
    2021). No hardcoded fallbacks or guards are applied here;
    the caller is responsible for ensuring rho > 0 and mu > 0.

    Typical values for superhot geothermal wells:
        Re ~ 10^6 to 10^7  (fully turbulent)

    Parameters
    ----------
    velocity  : float - Mean flow velocity [m/s]
    diameter  : float - Internal pipe diameter [m]
    density   : float - Fluid density [kg/m3] (mixture density if two-phase)
    viscosity : float - Dynamic viscosity [Pa.s] (mixture viscosity if two-phase)

    Returns
    -------
    float - Reynolds number [-]
    """
    return density * velocity * diameter / viscosity


def friction_factor(Re, diameter, epsilon=0.046e-3):
    """
    Darcy-Weisbach friction factor from the Swamee-Jain (1976) explicit
    approximation to the Colebrook-White equation.

    The implicit Colebrook-White equation is:

        1/sqrt(f) = -2 log10( eps/(3.7 D) + 2.51/(Re sqrt(f)) )

    The Swamee-Jain explicit approximation avoids iteration:

        f = 0.25 / [ log10( eps/(3.7 D) + 5.74/Re^0.9 ) ]^2

    This is accurate to within ~1% of the implicit Colebrook solution
    for 5000 < Re < 10^8 and 10^-6 < eps/D < 10^-2 (Swamee and Jain,
    1976), which covers the full range of geothermal wellbore conditions.

    Guards
    ------
    Two numerical guards handle edge cases outside the Swamee-Jain
    validity range:
        - Re < 10: returns the laminar Hagen-Poiseuille result
          f = 64/Re, capped at 0.1 for Re -> 0 to prevent
          division-by-zero. This is the exact analytical result for
          fully-developed laminar pipe flow (Bird, Stewart & Lightfoot,
          2002, Eq. 6.1-4). In practice, Re < 10 does not occur during
          normal wellbore simulation (Re ~ 10^6-10^7), but can arise
          transiently during solver initialization or at near-zero
          flow rates.
        - log10(...) ~ 0: if the Swamee-Jain log argument produces
          a near-zero denominator, the log value is perturbed to
          1e-10 to avoid division by zero. This is a degenerate case
          that occurs at Re ~ 7.5 for typical roughness values and
          has no physical significance.

    Reference: Swamee, P.K. and Jain, A.K. (1976). Explicit equations
    for pipe-flow problems. J. Hydraul. Div. 102(5), 657-664.

    Parameters
    ----------
    Re       : float - Reynolds number [-]
    diameter : float - Internal pipe diameter [m]
    epsilon  : float - Absolute roughness [m], default 0.046 mm

    Returns
    -------
    float - Darcy friction factor f_D [-]
    """
    eps_over_D = epsilon / diameter
    # Guard: Swamee-Jain is valid for Re > 5000. Below that,
    # use laminar formula. Also protect against log10(...) = 0
    # which causes division by zero at Re ~ 7.5.
    if Re < 10:
        # Laminar: f = 64/Re (or cap at f = 0.1 for Re -> 0)
        return max(64.0 / max(Re, 1.0), 0.1)
    log_arg = eps_over_D / 3.7 + 5.74 / Re**0.9
    log_val = np.log10(log_arg)
    if abs(log_val) < 1e-10:
        # Degenerate case: perturb slightly
        log_val = 1e-10
    f_D = 0.25 / log_val**2
    return f_D


def pressure_gradient(density, velocity, viscosity, diameter,
                      epsilon=0.046e-3):
    """
    Pressure gradient for steady upward flow in a vertical wellbore.

    From Scott (2026), Eq. (4):

        dP/dz = -rho*g - (f / 2D) * rho * v^2

    where z is positive upward. Since pressure DECREASES going up, dP/dz
    is negative. For the marching scheme it is convenient to return the
    magnitude of the pressure DROP per meter (a positive quantity):

        |dP/dz| = rho*g + (f / 2D) * rho * v^2
                   -----   -------------------
                  gravity       friction

    Relationship to Tonkin et al. (2021)
    ------------------------------------
    The full steady-state homogeneous momentum equation is
    Tonkin et al. (2021, Eq. 102):

        rho*u*du/ds + dP/ds + (2/R)*tau - rho*g*dz/ds = 0

    Eq. 4 omits the convective acceleration term rho*u*du/ds,
    following Nathenson (1974). This simplification retains only the
    hydrostatic and friction terms. For the superhot systems targeted
    here (predominantly single-phase supercritical or high-quality
    two-phase flow), density changes are gradual and the acceleration
    term is small compared to gravity and friction. Tonkin et al.
    (2021, Section 7.1) showed that omitting convective acceleration
    overestimates wellhead pressure by ~35% in their multi-feed test
    case (compared to the exact model), but that case involves
    aggressive flashing that is not typical of superhot production.
    The kinetic energy effect on *enthalpy* is captured separately
    in Eq. 5 (enthalpy_gradient).

    Derivation
    ----------
    The momentum balance for steady, fully-developed, one-dimensional
    flow in a constant-area vertical pipe (z positive upward) is:

        dP/dz = -rho * g  -  tau_w * (perimeter / A)

    Wall shear stress in terms of the Darcy friction factor
    (see, e.g., Bird, Stewart & Lightfoot, 2002, Ch. 6):

        tau_w = (f/8) * rho * v^2

    For a circular pipe, perimeter/A = 4/D, so:

        tau_w * (4/D) = (f/2D) * rho * v^2

    Combining:

        dP/dz = -rho*g - (f/2D)*rho*v^2

    Parameters
    ----------
    density   : float - Fluid density [kg/m3]
    velocity  : float - Mean flow velocity [m/s]
    viscosity : float - Dynamic viscosity [Pa.s]
    diameter  : float - Internal pipe diameter [m]
    epsilon   : float - Absolute roughness [m], default 0.046 mm

    Returns
    -------
    tuple of (dp_total, dp_gravity, dp_friction)
        All in Pa/m (positive = pressure drop per meter going UP).

    Note: returns |dP/dz| (positive magnitude), not the signed derivative.
    The caller subtracts: P_new = P_old - |dP/dz| * dz.
    """
    Re = reynolds_number(velocity, diameter, density, viscosity)
    f_D = friction_factor(Re, diameter, epsilon)

    dp_gravity = GRAVITY * density
    dp_friction = f_D * (density / 2.0) * (velocity**2) / diameter
    dp_total = dp_gravity + dp_friction

    return dp_total, dp_gravity, dp_friction


# ===================================================================
# 3. ENERGY BALANCE
# ===================================================================

def enthalpy_gradient(temperature_K, T_rock_K, mass_flow_rate,
                      density, drho_dz, area,
                      heat_loss_coeff=2.5):
    """
    Specific enthalpy gradient dh/dz for steady upward flow.

    From Scott (2026), Eq. (5):

        dh/dz = -g + (mdot/A)^2 * (1/rho^3) * (drho/dz)
                - (U/mdot) * (T - T_rock)

    The three terms correspond to:
        1. Gravitational potential energy: -g (always negative)
        2. Kinetic energy: as density decreases going up (drho/dz < 0),
           the fluid accelerates and enthalpy is converted to KE, making
           this term negative. (Positive only if density increases going
           up, which is non-physical for normal superhot well production.)
        3. Conductive heat loss: -(U/mdot)*(T - T_rock), negative when
           the fluid is hotter than the surrounding rock.

    In normal superhot well operation all three terms are negative, so
    dh/dz < 0 (enthalpy decreases going up). The simulator warns if
    dh/dz > 0 at any step.

    Relationship to Tonkin et al. (2021)
    ------------------------------------
    This equation is the steady-state, no-source reduction of the
    single-phase energy conservation equation (Tonkin et al., 2021,
    Eq. 65):

        d/dt[rho*(e + u^2/2)]
            + (1/A)*d/ds[A*rho*u*(h + u^2/2)]
            - rho*u*g*dz/ds + (2/R)*Qheat + qener = 0

    At steady state (d/dt = 0), with no mass/energy sources
    (qener = 0) and constant cross-sectional area (A = const),
    Eq. 65 simplifies to:

        d/ds[rho*u*(h + u^2/2)] - rho*u*g*dz/ds + (2/R)*Qheat = 0

    Since rho*u*A = mdot = const (mass conservation), dividing
    through by rho*u gives:

        dh/ds + d(u^2/2)/ds - g*dz/ds + (2/R)*Qheat/(rho*u) = 0

    Replacing the wellbore coordinate s with elevation z (for a
    vertical well, ds = dz) and expressing Qheat in terms of the
    heat-loss coefficient U yields Eq. 5.

    The same result follows from Tonkin et al. (2021, Eq. 99)
    for homogeneous two-phase flow (substituting uv = ul = u).

    Why friction does not appear
    ----------------------------
    Friction does not appear as an explicit term in Eq. 5. This is
    exact, not an approximation. Since specific enthalpy h = e + P/rho,
    the frictional pressure drop (Eq. 4) reduces the P/rho component
    of enthalpy but the viscous dissipation heats the fluid by the same
    amount, increasing internal energy e. The net effect on h is zero.
    Tonkin et al. (2021, discussion below Eq. 74) make the same
    observation: the thermal energy equation (their Eq. 74) is
    approximate because it omits a friction dissipation term, but the
    total enthalpy formulation (Eq. 65/99) is exact and does not
    require such a term. See also Bird, Stewart & Lightfoot (2002,
    Ch. 15) for the general result in pipe flow.

    Derivation of the kinetic energy term
    --------------------------------------
    For constant mass flow rate mdot and constant cross-sectional area A,
    the mean velocity is:

        v = mdot / (rho * A)

    The specific kinetic energy is v^2/2, and its gradient:

        d(v^2/2)/dz = v * dv/dz

    Differentiating v with respect to z (mdot and A constant):

        dv/dz = d/dz[ mdot / (rho * A) ]
              = -(mdot / A) * (1/rho^2) * drho/dz

    Therefore:

        v * dv/dz = [ mdot/(rho*A) ] * [ -(mdot/A) * (1/rho^2) * drho/dz ]
                  = -(mdot/A)^2 * (1/rho^3) * drho/dz

    The steady-state energy equation is:

        dh/dz + d(v^2/2)/dz + g + (U/mdot)*(T - T_rock) = 0

    Substituting and rearranging:

        dh/dz = -g + (mdot/A)^2 * (1/rho^3) * drho/dz
                - (U/mdot) * (T - T_rock)

    Parameters
    ----------
    temperature_K    : float - Fluid temperature [K]
    T_rock_K         : float - Rock temperature at this depth [K]
    mass_flow_rate   : float - Mass flow rate [kg/s]
    density          : float - Fluid density [kg/m3]
    drho_dz          : float - Density gradient [kg/m3/m] (finite diff.)
    area             : float - Pipe cross-sectional area [m2]
    heat_loss_coeff  : float - Wellbore heat-loss coefficient U [W/m/K].
                               Default 2.5 W/m/K, from Albertsson et al.
                               (2003), derived from Carslaw and Jaeger
                               (1959) for a 9 5/8" well in Icelandic
                               basalt after 1 year of production.

    Returns
    -------
    float - dh/dz [J/kg/m]
    """
    gravity_term = -GRAVITY

    kinetic_term = ((mass_flow_rate / area)**2
                    * (1.0 / density**3)
                    * drho_dz)

    heat_loss_term = (-(heat_loss_coeff / mass_flow_rate)
                      * (temperature_K - T_rock_K))

    return gravity_term + kinetic_term + heat_loss_term


# ===================================================================
# 4. WELLBORE SIMULATOR
# ===================================================================

def wellbore_simulate(P_bottom_MPa, feedzone_enthalpy, mass_flow_rate,
                      rock_temperatures, well_params=None):
    """
    Steady-state bottom-to-surface wellbore marching model.

    Integrates the pressure gradient (Eq. 4) and enthalpy gradient (Eq. 5)
    upward from the feedzone to the surface using a forward-difference
    (Euler) scheme with step size delta_z.

    At each step:
        1. Evaluate fluid properties from (P, h) via fluid_properties_Ph().
        2. Compute velocity from mass continuity: v = mdot / (rho * A).
        3. If EOS failed, properties are held from the previous step
           (fluid_properties_Ph returns prev_props unchanged). If EOS
           succeeded and P is within 0.5 MPa of P_crit, apply light
           smoothing (10% prev, 90% new) to damp step-to-step
           variations where properties change most rapidly.
        4. Compute pressure drop and subtract (pressure decreases upward).
        5. Compute enthalpy gradient and integrate.

    Parameters
    ----------
    P_bottom_MPa : float
        Bottomhole pressure [MPa].
    feedzone_enthalpy : float
        Feedzone specific enthalpy [J/kg]. Together with P_bottom_MPa,
        this fully determines the initial thermodynamic state (T, rho,
        mu) via the P-h equation of state.
    mass_flow_rate : float
        Mass flow rate [kg/s].
    rock_temperatures : dict
        Mapping {depth_m: temperature_C} for the surrounding rock.
        Depth keys must cover [0, well_depth] at intervals of delta_z.
    well_params : dict or None
        Wellbore geometry and heat-loss parameters. Any keys provided
        override the corresponding values in DEFAULT_WELL_PARAMS.
        Keys:
            diameter_m      : float - Internal diameter [m] (default 0.217)
            depth_m         : float - Well depth [m] (default 2100)
            delta_z_m       : float - Vertical step size [m] (default 10)
            heat_loss_factor: float - U [W/m/K] (default 2.5)
            roughness_m     : float - Pipe roughness [m] (default 0.046 mm)

    Returns
    -------
    tuple of six lists of (depth_m, value) tuples, plus a boolean:
        (temperature_profile,   - [(depth, T_C), ...]
         pressure_profile,      - [(depth, P_MPa), ...]
         enthalpy_profile,      - [(depth, h_MJkg), ...]
         velocity_profile,      - [(depth, v_ms), ...]
         saturation_profile,    - [(depth, x), ...]
         rock_temp_profile,     - [(depth, T_rock_C), ...]
         choked)                - bool: True if v > c_sound anywhere

    The choked flag indicates that the prescribed mass flow rate
    exceeds what the wellbore can sustain in steady-state flow.
    The profiles are still returned (the simulation does not
    terminate), but the results above the choke point are
    approximate because the continuity equation is violated
    at supersonic velocities.
    """
    # Unpack well parameters: start from defaults, override with user values
    params = dict(DEFAULT_WELL_PARAMS)
    if well_params is not None:
        params.update(well_params)
    diameter = params['diameter_m']
    well_depth = params['depth_m']
    delta_z = params['delta_z_m']
    U = params['heat_loss_factor']
    roughness = params['roughness_m']

    area = pi * (diameter / 2.0)**2

    # Initialize from feedzone conditions using the same P-h lookup
    # as the marching loop. The feedzone enthalpy is known from the
    # reservoir model, so there is no need to compute it from T-P.
    pressure = P_bottom_MPa
    enthalpy = feedzone_enthalpy

    init = fluid_properties_Ph(pressure, enthalpy, prev_props=None)
    temperature = init['temperature_K']

    # Profile storage
    temperature_profile = []
    rock_temp_profile = []
    pressure_profile = []
    enthalpy_profile = []
    velocity_profile = []
    saturation_profile = []

    # Choked flow tracking: set to True if velocity exceeds the
    # local sound speed at any point during the upward march. This
    # indicates that the prescribed mass flow rate exceeds the
    # maximum steady-state throughput of the wellbore at that
    # cross-section.
    choked = False

    # State for marching
    prev_props = {
        'phase': init['phase'],
        'temperature_K': temperature,
        'density_kgm3': init['density_kgm3'],
        'viscosity_Pas': init['viscosity_Pas'],
        'sound_speed_ms': init['sound_speed_ms'],
        'pressure_MPa': pressure,
    }
    density_previous = None

    # March from bottom to surface
    for z in range(0, well_depth + delta_z, delta_z):
        current_depth = well_depth - z

        # --- Fluid properties from P-h ---
        fluid = fluid_properties_Ph(pressure, enthalpy, prev_props)
        density = fluid['density_kgm3']

        # --- Density gradient (backward finite difference) ---
        if density_previous is not None:
            drho_dz = (density - density_previous) / delta_z
        else:
            drho_dz = 0.0

        # --- Velocity from mass continuity ---
        velocity = mass_flow_rate / (density * area)

        # --- Choked flow check ---
        sound_speed = fluid['sound_speed_ms']
        if not np.isnan(sound_speed) and velocity > sound_speed:
            if not choked:
                # Only warn on the first occurrence
                warnings.warn(
                    f"Velocity ({velocity:.1f} m/s) exceeds sound speed "
                    f"({sound_speed:.1f} m/s) at depth {current_depth} m. "
                    f"Flow is choked; steady-state assumptions are violated.",
                    RuntimeWarning, stacklevel=2)
            choked = True

        # --- Property smoothing for numerical stability ---
        # Three cases:
        #   1. EOS failed (success=False): fluid_properties_Ph already
        #      returned the previous step's properties unchanged.
        #      Use them directly -- no additional smoothing needed.
        #   2. EOS succeeded and P is within 0.5 MPa of P_crit:
        #      apply light smoothing (10% prev, 90% new) to damp
        #      step-to-step variations where thermodynamic properties
        #      change most rapidly. This is independent of the wider
        #      IAPWS97 routing zone (21.5-27 MPa); IAPWS97's backward
        #      equations are smooth internally and don't need blending.
        #   3. Normal: use EOS output directly.
        in_smoothing_zone = abs(pressure - P_CRIT_MPA) < 0.5
        if fluid['success'] and in_smoothing_zone:
            alpha = 0.1
            density = (alpha * prev_props['density_kgm3']
                       + (1 - alpha) * fluid['density_kgm3'])
            viscosity = (alpha * prev_props['viscosity_Pas']
                         + (1 - alpha) * fluid['viscosity_Pas'])
            temperature = (alpha * prev_props['temperature_K']
                           + (1 - alpha) * fluid['temperature_K'])
        else:
            density = fluid['density_kgm3']
            viscosity = fluid['viscosity_Pas']
            temperature = fluid['temperature_K']

        saturation = fluid['saturation']

        # --- Pressure step (Eq. 4) ---
        # pressure_gradient() returns the MAGNITUDE of the pressure drop
        # per meter going up (always positive), so we SUBTRACT it.
        # dP/dz = -rho*g - f/(2D)*rho*v^2  <-- negative (P decreases upward)
        # |dP/dz| = rho*g + f/(2D)*rho*v^2  <-- this is what the function returns
        dp_total, _, _ = pressure_gradient(density, velocity,
                                           viscosity, diameter,
                                           roughness)
        pressure_new = pressure - (dp_total * delta_z) / 1e6  # Pa -> MPa

        if pressure_new > pressure:
            warnings.warn(
                f"Pressure INCREASED going up at depth {current_depth} m "
                f"({pressure:.3f} -> {pressure_new:.3f} MPa). "
                f"This is non-physical for steady upward flow.",
                RuntimeWarning, stacklevel=2)

        if pressure_new <= 0:
            warnings.warn(
                f"Pressure dropped to {pressure_new:.3f} MPa at depth "
                f"{current_depth} m. Terminating simulation -- the well "
                f"cannot sustain flow at these conditions (check mass flow "
                f"rate, well depth, or reservoir pressure).",
                RuntimeWarning, stacklevel=2)
            break

        pressure = pressure_new

        # --- Enthalpy step (Eq. 5) ---
        # enthalpy_gradient() returns the SIGNED derivative dh/dz, which
        # is normally negative (gravity, KE, and heat loss all reduce h
        # going up). We add dh/dz * dz directly.
        T_rock_K = rock_temperatures[current_depth] + 273.15

        dh_dz = enthalpy_gradient(
            temperature, T_rock_K, mass_flow_rate,
            density, drho_dz, area,
            heat_loss_coeff=U,
        )
        enthalpy_new = enthalpy + dh_dz * delta_z

        if enthalpy_new > enthalpy:
            warnings.warn(
                f"Enthalpy INCREASED going up at depth {current_depth} m "
                f"({enthalpy/1e6:.4f} -> {enthalpy_new/1e6:.4f} MJ/kg). "
                f"Check heat-loss coefficient or initial conditions.",
                RuntimeWarning, stacklevel=2)
        enthalpy = enthalpy_new

        # --- Update state ---
        density_previous = density
        prev_props = {
            'phase': fluid['phase'],
            'temperature_K': temperature,
            'density_kgm3': density,
            'viscosity_Pas': viscosity,
            'sound_speed_ms': sound_speed,
            'pressure_MPa': pressure,
        }

        # --- Store profiles ---
        temperature_profile.append((current_depth, temperature - 273.15))
        rock_temp_profile.append((current_depth, T_rock_K - 273.15))
        pressure_profile.append((current_depth, pressure))
        enthalpy_profile.append((current_depth, enthalpy * 1e-6))
        velocity_profile.append((current_depth, velocity))
        saturation_profile.append((current_depth, saturation))

    return (temperature_profile, pressure_profile, enthalpy_profile,
            velocity_profile, saturation_profile, rock_temp_profile,
            choked)
