# -*- coding: utf-8 -*-
"""
Wellbore Physics
================

Covers :mod:`superhot_wellbore.wellbore_physics`: the friction
correlation, the homogeneous two-phase mixture rules, the phase
routing of the equation of state, the enthalpy gradient terms, and
the marching solver itself. The solver never announces a wrong
answer -- it returns a full set of profiles either way -- so these
tests verify the profiles against independent CoolProp lookups at
every step: T must equal T(P, h), v must equal mdot/(rho*A), and
the profiles must stay monotonic and smooth through the two-phase
dome and through the critical pressure.

All reference values are computed from CoolProp or derived from
first principles; none are hand-typed constants.

Ported from the self-test block that used to live at the bottom of
wellbore_physics.py; the expected values, tolerances and literature
citations are carried over unchanged.

Author: Samuel W. Scott
"""

import warnings

import CoolProp.CoolProp as CP
import numpy as np
import pytest
from iapws import IAPWS97
from scipy.optimize import fsolve

from superhot_wellbore import wellbore_physics
from superhot_wellbore.wellbore_physics import (DEFAULT_WELL_PARAMS, GRAVITY,
                                                P_CRIT_MPA,
                                                TwoPhaseChokeWarning,
                                                _hem_sound_speed,
                                                _two_phase_properties,
                                                enthalpy_gradient,
                                                fluid_properties_Ph,
                                                friction_factor,
                                                wellbore_simulate)


# ====================================================================
# HELPERS
# ====================================================================

def _colebrook_iterative(Re, eps_D):
    """Solve the implicit Colebrook-White equation via fsolve."""
    def residual(f):
        return (1.0 / np.sqrt(f)
                + 2.0 * np.log10(eps_D / 3.7
                                 + 2.51 / (Re * np.sqrt(f))))
    return fsolve(residual, 0.02)[0]


def _saturated_reference(P_MPa):
    """
    Saturated liquid and vapor density and dynamic viscosity at a
    pressure, straight from CoolProp.
    """
    P_Pa = P_MPa * 1e6
    return (CP.PropsSI('D', 'P', P_Pa, 'Q', 0, 'Water'),
            CP.PropsSI('D', 'P', P_Pa, 'Q', 1, 'Water'),
            CP.PropsSI('V', 'P', P_Pa, 'Q', 0, 'Water'),
            CP.PropsSI('V', 'P', P_Pa, 'Q', 1, 'Water'))


def _linear_rock_temperatures(depth_m, T_bottom_C, step_m=10):
    """Linear rock temperature profile from 10 C at surface."""
    depths = list(range(0, depth_m + step_m, step_m))
    return {d: 10 + (T_bottom_C - 10) * d / depth_m for d in depths}


def _well_area(well_params):
    """Flow area [m2] of a well from its internal diameter."""
    return np.pi * (well_params['diameter_m'] / 2)**2


def _simulate(P_bottom_MPa, h_bottom_Jkg, mass_flow, well_params,
              T_bottom_C):
    """
    Run a full wellbore simulation and return the profiles, the
    choked flag and every warning it raised.
    """
    rock = _linear_rock_temperatures(well_params['depth_m'], T_bottom_C,
                                     well_params['delta_z_m'])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        profiles = wellbore_simulate(P_bottom_MPa, h_bottom_Jkg,
                                     mass_flow, rock, well_params)
    T_prof, P_prof, h_prof, v_prof, s_prof, rock_prof, choked = profiles
    return {
        'T': T_prof, 'P': P_prof, 'h': h_prof, 'v': v_prof,
        'saturation': s_prof, 'rock': rock_prof, 'choked': choked,
        'warnings': caught,
        'well_params': well_params,
        'mass_flow': mass_flow,
    }


def _values(profile):
    """Strip the depth coordinate off a (depth, value) profile."""
    return [point[1] for point in profile]


def _is_decreasing(values):
    """True when a profile never increases from step to step."""
    return all(values[i] >= values[i+1] for i in range(len(values)-1))


def _max_step_jump(values):
    """Largest step-to-step change in a profile."""
    return max(abs(values[i] - values[i+1])
               for i in range(len(values)-1))


def _critical_crossing_index(P_values):
    """Index of the step where P drops below the critical pressure."""
    for i in range(len(P_values)-1):
        if P_values[i] >= P_CRIT_MPA and P_values[i+1] < P_CRIT_MPA:
            return i
    return None


def _max_temperature_error_K(simulation, i_start=None, i_end=None):
    """
    Largest |T_reported - T(P, h)| over a window of the profile,
    falling back to the IAPWS97 route where CoolProp refuses.
    """
    T_prof, P_prof, h_prof = (simulation['T'], simulation['P'],
                              simulation['h'])
    i_start = 0 if i_start is None else i_start
    i_end = len(T_prof) if i_end is None else i_end
    max_err = 0
    for i in range(i_start, i_end):
        T_reported = T_prof[i][1] + 273.15  # K
        P_i = P_prof[i][1] * 1e6  # Pa
        h_i = h_prof[i][1] * 1e6  # J/kg
        try:
            T_from_Ph = CP.PropsSI('T', 'P', P_i, 'H', h_i, 'Water')
            max_err = max(max_err, abs(T_reported - T_from_Ph))
        except Exception:
            # IAPWS97 region -- check via that route
            try:
                fl = IAPWS97(P=P_prof[i][1], h=h_prof[i][1]*1e3)
                if fl.T is not None:
                    max_err = max(max_err, abs(T_reported - fl.T))
            except Exception:
                pass
    return max_err


# ====================================================================
# SIMULATION FIXTURES
# ====================================================================

@pytest.fixture(scope='module')
def dome_simulation():
    """
    Fluid starts as single-phase vapor just above h_g at the bottom
    of the well, then enters the two-phase dome as enthalpy decreases
    during ascent (due to gravity and heat loss) while h_g(P) remains
    high at intermediate pressures.

    Conditions: P_bottom = 8 MPa, h = 2780 kJ/kg (T ~ 299 C).
    At 8 MPa, h_g = 2759 kJ/kg, so h > h_g -> single-phase vapor.
    As the fluid rises, h drops ~50-80 kJ/kg over 2000 m, while
    h_g at intermediate P (6-7 MPa) is ~2773-2785 kJ/kg.
    When h drops below h_g, the fluid enters the two-phase dome.
    This is the scenario that occurs in real superhot wells when
    reservoir pressure is high enough to push surface conditions
    into the two-phase envelope.
    """
    wp_dome = dict(DEFAULT_WELL_PARAMS)
    wp_dome['depth_m'] = 2000
    wp_dome['heat_loss_factor'] = 2.5
    h_dome = 2.780e6  # J/kg, just above h_g(8 MPa) = 2759 kJ/kg
    P_dome = 8.0      # MPa
    return _simulate(P_dome, h_dome, 10.0, wp_dome, 300)


@pytest.fixture(scope='module')
def iddp_simulation():
    """
    Reservoir: T=530 C, P=16.4 MPa (Ingason et al., 2014).
    Well depth 2100 m, 20 kg/s. Single-phase vapor throughout.
    """
    wp_iddp = dict(DEFAULT_WELL_PARAMS)
    wp_iddp['depth_m'] = 2100
    h_iddp = CP.PropsSI('H', 'T', 530 + 273.15, 'P', 16.4e6, 'Water')
    return _simulate(16.4, h_iddp, 20.0, wp_iddp, 530)


@pytest.fixture(scope='module')
def deep_simulation():
    """
    Reservoir: T=500 C, P=25 MPa, depth=4500 m, 20 kg/s.
    Bottomhole P > P_crit, surface P < P_crit. This forces the
    marching loop through the near-critical zone where IAPWS97
    routing and smoothing are active.
    """
    wp_deep = dict(DEFAULT_WELL_PARAMS)
    wp_deep['depth_m'] = 4500
    h_deep = CP.PropsSI('H', 'T', 500 + 273.15, 'P', 25e6, 'Water')
    return _simulate(25.0, h_deep, 20.0, wp_deep, 500)


@pytest.fixture(scope='module')
def choked_simulation():
    """
    A small-diameter well (D = 0.10 m) with high mass flow (80 kg/s)
    and superhot vapor (~500 C, 16.4 MPa).
    """
    wp_choked = dict(DEFAULT_WELL_PARAMS)
    wp_choked['depth_m'] = 2100
    wp_choked['diameter_m'] = 0.10  # small-diameter well
    h_choked = CP.PropsSI('H', 'T', 530 + 273.15, 'P', 16.4e6, 'Water')
    return _simulate(16.4, h_choked, 80.0, wp_choked, 530)


@pytest.fixture(scope='module')
def negative_pressure_simulation():
    """
    A 3000 m well (within the paper's 2000-5000 m range) with
    standard diameter (0.217 m) and a very high mass flow rate
    (120 kg/s) at P_bottom = 15 MPa.
    """
    wp_negP = dict(DEFAULT_WELL_PARAMS)
    wp_negP['depth_m'] = 3000
    h_negP = CP.PropsSI('H', 'T', 450 + 273.15, 'P', 15e6, 'Water')
    return _simulate(15.0, h_negP, 120.0, wp_negP, 450)


# ====================================================================
# 1. SWAMEE-JAIN ACCURACY VS ITERATIVE COLEBROOK-WHITE
# ====================================================================

@pytest.mark.parametrize('Re_val', [1e4, 5e4, 1e5, 5e5, 1e6, 5e6, 1e7],
                         ids=['Re=1e+04', 'Re=5e+04', 'Re=1e+05',
                              'Re=5e+05', 'Re=1e+06', 'Re=5e+06',
                              'Re=1e+07'])
def test_swamee_jain_matches_colebrook_white(Re_val):
    """
    The Swamee-Jain (1976) explicit approximation should agree with
    the iterative Colebrook-White solution to within 1% for
    Re > 5000 and 1e-6 < eps/D < 1e-2.
    """
    D_ref = 0.217        # IDDP-1 casing ID [m]
    eps_ref = 0.046e-3   # commercial steel roughness [m]
    f_ref = _colebrook_iterative(Re_val, eps_ref / D_ref)
    f_sj = friction_factor(Re_val, D_ref, eps_ref)
    err_pct = abs(f_sj - f_ref) / f_ref * 100
    assert err_pct < 1.0, f'Swamee-Jain error = {err_pct:.2f}%'


# ====================================================================
# 2. TWO-PHASE MIXTURE PROPERTIES
# ====================================================================

class TestTwoPhaseProperties:
    """
    Verify density and viscosity formulas against independent
    calculations at 5 MPa, where CoolProp provides exact saturated
    phase properties. 5 MPa is well within the subcritical range.
    """

    @pytest.mark.parametrize('x_val', [0.01, 0.1, 0.5, 0.9, 0.99],
                             ids=['x=0.01', 'x=0.10', 'x=0.50',
                                  'x=0.90', 'x=0.99'])
    def test_mixture_density(self, x_val):
        """Density: 1/rho = x/rho_g + (1-x)/rho_f."""
        rho_f, rho_g, mu_f, mu_g = _saturated_reference(5.0)
        rho_mix, mu_mix, alpha = _two_phase_properties(
            x_val, rho_f, rho_g, mu_f, mu_g)

        rho_indep = 1.0 / (x_val / rho_g + (1 - x_val) / rho_f)
        assert rho_mix == pytest.approx(rho_indep, abs=1e-6)

    @pytest.mark.parametrize('x_val', [0.01, 0.1, 0.5, 0.9, 0.99],
                             ids=['x=0.01', 'x=0.10', 'x=0.50',
                                  'x=0.90', 'x=0.99'])
    def test_mixture_viscosity(self, x_val):
        """Viscosity: mu = rho * (x*nu_g + (1-x)*nu_f)."""
        rho_f, rho_g, mu_f, mu_g = _saturated_reference(5.0)
        rho_mix, mu_mix, alpha = _two_phase_properties(
            x_val, rho_f, rho_g, mu_f, mu_g)

        nu_f = mu_f / rho_f  # kinematic
        nu_g = mu_g / rho_g
        rho_indep = 1.0 / (x_val / rho_g + (1 - x_val) / rho_f)
        nu_indep = x_val * nu_g + (1 - x_val) * nu_f
        mu_indep = rho_indep * nu_indep
        assert mu_mix == pytest.approx(mu_indep, rel=1e-12)


# ====================================================================
# 3. PHASE DETECTION AT SATURATION BOUNDARIES
# ====================================================================

class TestPhaseDetection:
    """
    At each pressure, test states just inside each phase region:
    h = h_f - 1 kJ/kg (compressed liquid), h = (h_f+h_g)/2
    (two-phase), h = h_g + 1 kJ/kg (superheated vapor).
    Tests up to 21.5 MPa, which is within 0.56 MPa of P_crit
    and exercises the IAPWS97 routing at the highest pressure.
    """

    @staticmethod
    def _saturation_enthalpies(P_val):
        """h_f and h_g [J/kg] at a pressure."""
        P_Pa = P_val * 1e6
        return (CP.PropsSI('H', 'P', P_Pa, 'Q', 0, 'Water'),
                CP.PropsSI('H', 'P', P_Pa, 'Q', 1, 'Water'))

    @pytest.mark.parametrize('P_val', [1.0, 5.0, 10.0, 15.0, 20.0, 21.5],
                             ids=['P= 1.0 MPa', 'P= 5.0 MPa',
                                  'P=10.0 MPa', 'P=15.0 MPa',
                                  'P=20.0 MPa', 'P=21.5 MPa'])
    def test_below_h_f_is_liquid(self, P_val):
        """h < h_f -> single_phase_liquid."""
        h_f, h_g = self._saturation_enthalpies(P_val)
        r_liq = fluid_properties_Ph(P_val, h_f - 1000)
        assert r_liq['phase'] == 'single_phase_liquid', \
            f"got {r_liq['phase']}"

    @pytest.mark.parametrize('P_val', [1.0, 5.0, 10.0, 15.0, 20.0, 21.5],
                             ids=['P= 1.0 MPa', 'P= 5.0 MPa',
                                  'P=10.0 MPa', 'P=15.0 MPa',
                                  'P=20.0 MPa', 'P=21.5 MPa'])
    def test_inside_the_dome_is_two_phase(self, P_val):
        """h_f < h < h_g -> two_phase."""
        h_f, h_g = self._saturation_enthalpies(P_val)
        r_2p = fluid_properties_Ph(P_val, (h_f + h_g) / 2)
        assert r_2p['phase'] == 'two_phase', f"got {r_2p['phase']}"

    @pytest.mark.parametrize('P_val', [1.0, 5.0, 10.0, 15.0, 20.0, 21.5],
                             ids=['P= 1.0 MPa', 'P= 5.0 MPa',
                                  'P=10.0 MPa', 'P=15.0 MPa',
                                  'P=20.0 MPa', 'P=21.5 MPa'])
    def test_above_h_g_is_vapor(self, P_val):
        """h > h_g -> single_phase_vapor."""
        h_f, h_g = self._saturation_enthalpies(P_val)
        r_vap = fluid_properties_Ph(P_val, h_g + 1000)
        assert r_vap['phase'] == 'single_phase_vapor', \
            f"got {r_vap['phase']}"


# ====================================================================
# 4. SIMULATION ENTERING TWO-PHASE ZONE
# ====================================================================

@pytest.mark.slow
class TestSimulationEnteringTwoPhaseZone:
    """The ascent from single-phase vapor into the two-phase dome."""

    def test_not_choked(self, dome_simulation):
        """Not choked at 10 kg/s in 0.217 m well."""
        assert dome_simulation['choked'] is False

    def test_pressure_monotonically_decreasing(self, dome_simulation):
        """P monotonically decreasing."""
        assert _is_decreasing(_values(dome_simulation['P']))

    def test_enthalpy_monotonically_decreasing(self, dome_simulation):
        """h monotonically decreasing."""
        assert _is_decreasing(_values(dome_simulation['h']))

    def test_trajectory_enters_two_phase_zone(self, dome_simulation):
        """Verify the trajectory enters two-phase."""
        s_vals = _values(dome_simulation['saturation'])
        has_two_phase = any(not np.isnan(x) and 0 < x < 1
                            for x in s_vals)
        assert has_two_phase

    def test_starts_as_single_phase_vapor(self, dome_simulation):
        """Verify it starts single-phase (saturation = NaN at bottom)."""
        s_vals = _values(dome_simulation['saturation'])
        assert np.isnan(s_vals[0]), \
            f'saturation at bottom = {s_vals[0]}'

    def test_temperature_matches_saturation_temperature(self,
                                                        dome_simulation):
        """
        Check that T, P, h are consistent where fluid is two-phase:
        T should equal T_sat(P) in the two-phase region.
        """
        T_prof = dome_simulation['T']
        P_prof = dome_simulation['P']
        s_prof = dome_simulation['saturation']
        max_Tsat_err = 0
        n_2p_steps = 0
        for i in range(len(s_prof)):
            sat_val = s_prof[i][1]
            if not np.isnan(sat_val) and 0 < sat_val < 1:
                n_2p_steps += 1
                P_i_Pa = P_prof[i][1] * 1e6
                try:
                    T_sat = CP.PropsSI('T', 'P', P_i_Pa, 'Q', 0,
                                       'Water') - 273.15
                    T_err = abs(T_prof[i][1] - T_sat)
                    if T_err > max_Tsat_err:
                        max_Tsat_err = T_err
                except Exception:
                    pass
        assert max_Tsat_err < 2.0 or n_2p_steps == 0, \
            (f'{n_2p_steps} two-phase steps, max T_sat error = '
             f'{max_Tsat_err:.1f} C')

    def test_smooth_through_phase_transition(self, dome_simulation):
        """Profile smoothness through the phase transition."""
        max_T_jump = _max_step_jump(_values(dome_simulation['T']))
        assert max_T_jump < 5.0, f'max T jump = {max_T_jump:.2f} C'


# ====================================================================
# 5. NEAR-CRITICAL EOS ROBUSTNESS
# ====================================================================

def test_near_critical_grid_all_succeed():
    """
    Test P-h states in a grid spanning the IAPWS97 routing zone
    (21.5 to 27 MPa). This covers the critical pressure (+/- 0.5 MPa
    below, +5 MPa above) and the pseudocritical ridge where CoolProp's
    Helmholtz solver can oscillate. All calls must succeed without
    falling back to prev_props interpolation.
    """
    P_crit_band = [21.6, 21.9, 22.0, 22.1, 22.5,
                   23.0, 24.0, 25.0, 26.0, 27.0]
    h_range = [1.5e6, 1.8e6, 2.1e6, 2.5e6, 3.0e6, 3.5e6]
    n_success = 0
    n_total = 0
    for P_val in P_crit_band:
        for h_val in h_range:
            n_total += 1
            with warnings.catch_warnings(record=True):
                warnings.simplefilter('always')
                r = fluid_properties_Ph(P_val, h_val)
            if r['success']:
                n_success += 1
    assert n_success == n_total, f'{n_total - n_success} failures'


# ====================================================================
# 6. ENTHALPY GRADIENT: TERM BY TERM
# ====================================================================

# (T_K, T_rock_K, mdot, rho, drho_dz)
# Low-density vapor with large density gradient (rapid expansion),
# moderate density with low flow (dominated by heat loss), and a
# high flow rate (friction-dominated, small KE correction). The
# default heat-loss coefficient U = 2.5 W/m/K comes from Albertsson
# et al. (2003), derived from Carslaw and Jaeger (1959) for a
# 9 5/8" well in Icelandic basalt after 1 year of production.
_GRADIENT_CONDITIONS = [
    (800, 500, 20.0, 50.0, -1.0),
    (700, 500, 5.0, 200.0, -0.1),
    (650, 600, 100.0, 100.0, -0.5),
]
_GRADIENT_IDS = ['rapid expansion, low rho', 'heat-loss dominated',
                 'high flow rate']


class TestEnthalpyGradient:
    """
    Verify against manual calculation at three different conditions.
    All reference values are computed here, not hardcoded.
    """

    @pytest.mark.parametrize('T, Tr, mdot, rho, drho',
                             _GRADIENT_CONDITIONS, ids=_GRADIENT_IDS)
    def test_matches_manual_calculation(self, T, Tr, mdot, rho, drho):
        """Gravity, kinetic energy and heat loss terms add up."""
        A_well = _well_area(DEFAULT_WELL_PARAMS)
        U_default = DEFAULT_WELL_PARAMS['heat_loss_factor']
        dhdz = enthalpy_gradient(T, Tr, mdot, rho, drho, A_well,
                                 U_default)
        grav = -GRAVITY
        ke = (mdot / A_well)**2 * (1.0 / rho**3) * drho
        hl = -(U_default / mdot) * (T - Tr)
        manual = grav + ke + hl
        assert dhdz == pytest.approx(manual, abs=1e-10)

    @pytest.mark.parametrize('T, Tr, mdot, rho, drho',
                             _GRADIENT_CONDITIONS, ids=_GRADIENT_IDS)
    def test_gradient_is_negative(self, T, Tr, mdot, rho, drho):
        """Enthalpy always falls with height in these conditions."""
        A_well = _well_area(DEFAULT_WELL_PARAMS)
        U_default = DEFAULT_WELL_PARAMS['heat_loss_factor']
        dhdz = enthalpy_gradient(T, Tr, mdot, rho, drho, A_well,
                                 U_default)
        assert dhdz < 0, f'dh/dz = {dhdz:.3f}'


# ====================================================================
# 7. EOS FAILURE HANDLING
# ====================================================================

_PREV_PROPS = {'phase': 'single_phase_vapor',
               'temperature_K': 600.0, 'density_kgm3': 123.456,
               'viscosity_Pas': 4.567e-5, 'sound_speed_ms': 400.0,
               'pressure_MPa': 10.0}


def _failed_lookup_with_prev_props():
    """Force an EOS failure while previous properties are available."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        r = fluid_properties_Ph(-1.0, -1.0, prev_props=dict(_PREV_PROPS))
    return r, caught


class TestEosFailureHandling:
    """An EOS failure must be loud, and must not invent numbers."""

    def test_no_prev_props_raises_runtime_error(self):
        """
        No prev_props available -> must raise RuntimeError (not
        silently return hardcoded values).
        """
        with pytest.raises(RuntimeError):
            fluid_properties_Ph(-1.0, -1.0, prev_props=None)

    def test_density_returned_unchanged(self):
        """With prev_props -> returns those values EXACTLY (no
        smoothing)."""
        r_fail, caught = _failed_lookup_with_prev_props()
        assert r_fail['density_kgm3'] == _PREV_PROPS['density_kgm3']

    def test_viscosity_returned_unchanged(self):
        """The stale viscosity is passed straight through."""
        r_fail, caught = _failed_lookup_with_prev_props()
        assert r_fail['viscosity_Pas'] == _PREV_PROPS['viscosity_Pas']

    def test_success_flag_is_false(self):
        """The caller can tell the lookup failed."""
        r_fail, caught = _failed_lookup_with_prev_props()
        assert r_fail['success'] is False

    def test_warning_issued(self):
        """The fallback is announced, not silent."""
        r_fail, caught = _failed_lookup_with_prev_props()
        assert len(caught) > 0


# ====================================================================
# 8. PROFILE CONSISTENCY: IDDP-1-LIKE SIMULATION
# ====================================================================

@pytest.mark.slow
class TestIddpProfileConsistency:
    """
    Reservoir: T=530 C, P=16.4 MPa (Ingason et al., 2014).
    Well depth 2100 m, 20 kg/s. Single-phase vapor throughout.
    We verify that the output profiles are internally consistent
    at every marching step -- not just that the code ran.

    NOTE: no check for T_fluid >= T_rock. The coupled reservoir model
    simulates adiabatic depressurization of fluid flowing from the
    reservoir to the wellbore, which can cool the fluid below the
    surrounding rock temperature at the feedzone.
    """

    def test_no_warnings_during_simulation(self, iddp_simulation):
        """A well inside the design envelope raises nothing."""
        caught = iddp_simulation['warnings']
        assert len(caught) == 0, f'{len(caught)} warnings'

    def test_pressure_monotonically_decreasing(self, iddp_simulation):
        """P monotonically decreasing."""
        assert _is_decreasing(_values(iddp_simulation['P']))

    def test_enthalpy_monotonically_decreasing(self, iddp_simulation):
        """h monotonically decreasing."""
        assert _is_decreasing(_values(iddp_simulation['h']))

    def test_temperature_pressure_enthalpy_consistency(self,
                                                       iddp_simulation):
        """
        At each step, the reported T should match T(P, h) from
        CoolProp. This verifies that the marching loop's T, P, and h
        are coherent (not drifting apart due to bugs in the update
        order).
        """
        max_T_err = _max_temperature_error_K(iddp_simulation)
        assert max_T_err < 5.0, f'max error = {max_T_err:.2f} K'

    def test_mass_continuity(self, iddp_simulation):
        """
        v = mdot / (rho * A). Compute rho from P, h independently and
        check that the reported velocity is consistent.
        """
        A_well = _well_area(iddp_simulation['well_params'])
        mdot = iddp_simulation['mass_flow']
        max_v_err_pct = 0
        for i in range(len(iddp_simulation['v'])):
            P_i = iddp_simulation['P'][i][1] * 1e6
            h_i = iddp_simulation['h'][i][1] * 1e6
            v_reported = iddp_simulation['v'][i][1]
            try:
                rho_i = CP.PropsSI('D', 'P', P_i, 'H', h_i, 'Water')
                v_expected = mdot / (rho_i * A_well)
                v_err = abs(v_reported - v_expected) / v_expected * 100
                if v_err > max_v_err_pct:
                    max_v_err_pct = v_err
            except Exception:
                pass
        assert max_v_err_pct < 5.0, \
            f'max error = {max_v_err_pct:.1f}%'

    def test_pressure_drop_is_physically_reasonable(self,
                                                    iddp_simulation):
        """
        Total dP over 2100 m should be roughly rho_avg * g * depth.
        For superheated vapor at ~100 kg/m3, expect ~2 MPa
        hydrostatic.
        """
        P_vals = _values(iddp_simulation['P'])
        h_vals = _values(iddp_simulation['h'])
        dP_total = P_vals[0] - P_vals[-1]
        rho_avg = 0.5 * (
            CP.PropsSI('D', 'P', P_vals[0]*1e6, 'H', h_vals[0]*1e6,
                       'Water')
            + CP.PropsSI('D', 'P', P_vals[-1]*1e6, 'H', h_vals[-1]*1e6,
                         'Water'))
        dP_hydrostatic = rho_avg * GRAVITY * 2100 / 1e6
        assert 0.5 * dP_hydrostatic < dP_total < 2.0 * dP_hydrostatic, \
            (f'pressure drop = {dP_total:.2f} MPa, hydrostatic est. = '
             f'{dP_hydrostatic:.2f} MPa')

    def test_profile_smoothness(self, iddp_simulation):
        """
        No step-to-step temperature jump > 5 C (for dz=10 m steps,
        a gradient of 0.5 C/m would be extreme for a superhot well).
        """
        max_T_jump = _max_step_jump(_values(iddp_simulation['T']))
        assert max_T_jump < 5.0, f'max T jump = {max_T_jump:.2f} C/step'


# ====================================================================
# 9. PROFILE CONSISTENCY: DEEP WELL CROSSING P_CRIT
# ====================================================================

@pytest.mark.slow
class TestDeepWellCrossingCriticalPressure:
    """
    Reservoir: T=500 C, P=25 MPa, depth=4500 m, 20 kg/s.
    Bottomhole P > P_crit, surface P < P_crit. This forces the
    marching loop through the near-critical zone where IAPWS97
    routing and smoothing are active. Profiles must remain smooth
    and self-consistent through the transition.
    """

    def test_pressure_monotonically_decreasing(self, deep_simulation):
        """P monotonically decreasing."""
        assert _is_decreasing(_values(deep_simulation['P']))

    def test_enthalpy_monotonically_decreasing(self, deep_simulation):
        """h monotonically decreasing."""
        assert _is_decreasing(_values(deep_simulation['h']))

    def test_pressure_crosses_critical_pressure(self, deep_simulation):
        """The profile really does span the critical pressure."""
        Pd = _values(deep_simulation['P'])
        assert Pd[0] > P_CRIT_MPA and Pd[-1] < P_CRIT_MPA, \
            f'P_bh={Pd[0]:.1f}, P_wh={Pd[-1]:.1f} MPa'

    def test_temperature_smooth_through_critical_pressure(
            self, deep_simulation):
        """
        Find the step where P crosses P_crit, and check that T and rho
        don't have discontinuities there.
        """
        Pd = _values(deep_simulation['P'])
        Td = _values(deep_simulation['T'])
        crit_idx = _critical_crossing_index(Pd)
        assert crit_idx is not None, 'found critical crossing index'
        T_jump_at_crit = abs(Td[crit_idx] - Td[crit_idx+1])
        assert T_jump_at_crit < 3.0, \
            f'jump = {T_jump_at_crit:.2f} C'

    def test_overall_profile_smoothness(self, deep_simulation):
        """Max T jump < 5 C/step over the whole well."""
        max_T_jump_deep = _max_step_jump(_values(deep_simulation['T']))
        assert max_T_jump_deep < 5.0, \
            f'max T jump = {max_T_jump_deep:.2f} C/step'

    def test_no_eos_failures(self, deep_simulation):
        """The near-critical zone never falls back to prev_props."""
        eos_warns = [x for x in deep_simulation['warnings']
                     if 'EOS failed' in str(x.message)]
        assert len(eos_warns) == 0, \
            f'{[str(x.message)[:60] for x in eos_warns[:3]]}'

    def test_temperature_pressure_enthalpy_consistency_near_critical(
            self, deep_simulation):
        """Check T(P,h) agreement in a 20-step window around P_crit."""
        Pd = _values(deep_simulation['P'])
        crit_idx = _critical_crossing_index(Pd)
        assert crit_idx is not None, 'found critical crossing index'
        i_start = max(0, crit_idx - 10)
        i_end = min(len(deep_simulation['T']), crit_idx + 10)
        max_T_err_crit = _max_temperature_error_K(deep_simulation,
                                                  i_start, i_end)
        assert max_T_err_crit < 5.0, \
            f'max error = {max_T_err_crit:.2f} K'


# ====================================================================
# 10. CHOKED FLOW DETECTION
# ====================================================================

@pytest.mark.slow
class TestChokedFlowDetection:
    """
    A small-diameter well (D = 0.10 m) with high mass flow (80 kg/s)
    and superhot vapor (~500 C, 16.4 MPa) produces very high
    velocities at the surface where density is low (~30-50 kg/m3).
    With A = pi*(0.05)^2 = 0.00785 m2 and rho ~ 30 kg/m3:
      v = 80 / (30 * 0.00785) ~ 340 m/s
    At lower surface density or higher flow, v can exceed the speed
    of sound (~500-600 m/s), triggering the choked flow warning.
    """

    def test_choked_flow_warning_triggered(self, choked_simulation):
        """The user is told the well is choking."""
        choked_warns = [x for x in choked_simulation['warnings']
                        if 'choked' in str(x.message).lower()]
        assert len(choked_warns) > 0, \
            f'{len(choked_warns)} choked warnings'

    def test_choked_flag_is_true(self, choked_simulation):
        """Choked flag is True."""
        assert choked_simulation['choked'] is True

    def test_profile_still_valid(self, choked_simulation):
        """The well should still produce a valid profile despite
        choking."""
        Pc = _values(choked_simulation['P'])
        assert len(Pc) > 1, 'a profile was returned'
        assert _is_decreasing(Pc), 'P decreasing'


# ====================================================================
# 11. NEGATIVE PRESSURE GUARD
# ====================================================================

@pytest.mark.slow
class TestNegativePressureGuard:
    """
    A 3000 m well (within the paper's 2000-5000 m range) with
    standard diameter (0.217 m) and a very high mass flow rate
    (120 kg/s) at P_bottom = 15 MPa. At this flow rate, friction
    dominates: dp_friction ~ 7600 Pa/m, so the total pressure drop
    over 3000 m exceeds the available 15 MPa. The simulator should
    detect P <= 0 and terminate early with a warning.
    """

    def test_negative_pressure_warning_triggered(
            self, negative_pressure_simulation):
        """The early termination is announced."""
        negP_warns = [x for x in negative_pressure_simulation['warnings']
                      if 'Terminating' in str(x.message)]
        assert len(negP_warns) > 0

    def test_simulation_terminated_early(self,
                                         negative_pressure_simulation):
        """Fewer steps than the full 3000/10 + 1."""
        full_steps = 3000 // 10 + 1
        n_steps = len(negative_pressure_simulation['T'])
        assert n_steps < full_steps, \
            f'{n_steps} of {full_steps} steps'

    def test_all_stored_pressures_positive(self,
                                           negative_pressure_simulation):
        """All stored pressures should be positive."""
        Pn = _values(negative_pressure_simulation['P'])
        assert len(Pn) > 0, 'a profile was returned'
        assert all(p > 0 for p in Pn)


# ====================================================================
# 12. MARCH BOOKKEEPING
# ====================================================================

# Constant gradients for the bookkeeping march: 1 kPa/m and
# -20 J/kg/m, so a 2000 m superheated-vapour well at 16 MPa and
# 3.0 MJ/kg stays far from the dome and from P_crit.
_CONSTANT_DP_DZ_PA_M = 1000.0
_CONSTANT_DH_DZ_JKG_M = -20.0


@pytest.fixture
def constant_gradient_march(monkeypatch):
    """
    A march whose momentum and energy gradients are constants, so
    that forward Euler is exact and the profiles are known in closed
    form: P(d) = P_bh - |dP/dz| (depth - d), h(d) = h_fz +
    dh/dz (depth - d).
    """
    monkeypatch.setattr(
        wellbore_physics, 'pressure_gradient',
        lambda *args, **kwargs: (_CONSTANT_DP_DZ_PA_M,
                                 _CONSTANT_DP_DZ_PA_M, 0.0))
    monkeypatch.setattr(
        wellbore_physics, 'enthalpy_gradient',
        lambda *args, **kwargs: _CONSTANT_DH_DZ_JKG_M)
    wp = dict(DEFAULT_WELL_PARAMS)
    wp['depth_m'] = 2000
    return _simulate(16.0, 3.0e6, 20.0, wp, 400)


class TestMarchIntegratesTheWellDepth:
    """
    The march takes depth/delta_z steps from the feedzone to the
    wellhead, and each profile entry is the state at its own depth.
    A march that integrates depth + delta_z (one step above the
    surface) stores every P and h one step ahead of its depth label
    and reports a wellhead pressure one step too low.
    """

    def test_depths_run_from_feedzone_to_wellhead(
            self, constant_gradient_march):
        """One entry per grid depth, feedzone first, wellhead last."""
        depth = constant_gradient_march['well_params']['depth_m']
        delta_z = constant_gradient_march['well_params']['delta_z_m']
        expected = list(range(depth, -delta_z, -delta_z))
        for name in ('T', 'P', 'h', 'v', 'saturation', 'rock'):
            depths = [d for d, _ in constant_gradient_march[name]]
            assert depths == expected, name

    def test_wellhead_pressure_is_one_well_depth_of_gradient(
            self, constant_gradient_march):
        """P_wh = P_bh - |dP/dz| * depth, not * (depth + delta_z)."""
        depth = constant_gradient_march['well_params']['depth_m']
        whp = constant_gradient_march['P'][-1][1]
        assert whp == pytest.approx(
            16.0 - _CONSTANT_DP_DZ_PA_M * depth / 1e6, abs=1e-9)

    def test_wellhead_enthalpy_is_one_well_depth_of_gradient(
            self, constant_gradient_march):
        """h_wh = h_fz + dh/dz * depth."""
        depth = constant_gradient_march['well_params']['depth_m']
        h_wh = constant_gradient_march['h'][-1][1]
        assert h_wh == pytest.approx(
            (3.0e6 + _CONSTANT_DH_DZ_JKG_M * depth) * 1e-6, abs=1e-12)

    def test_every_entry_holds_the_state_at_its_depth(
            self, constant_gradient_march):
        """P and h at each depth label match the closed form."""
        depth = constant_gradient_march['well_params']['depth_m']
        for (d, P), (_, h) in zip(constant_gradient_march['P'],
                                  constant_gradient_march['h']):
            assert P == pytest.approx(
                16.0 - _CONSTANT_DP_DZ_PA_M * (depth - d) / 1e6,
                abs=1e-9), f'P at {d} m'
            assert h == pytest.approx(
                (3.0e6 + _CONSTANT_DH_DZ_JKG_M * (depth - d)) * 1e-6,
                abs=1e-12), f'h at {d} m'


def _assert_state_matches_eos(simulation):
    """
    Every stored T, v and x is the equation of state's value at the
    stored (P, h) of the same entry, to round-off. Restricted to
    entries outside the +/-0.5 MPa smoothing band around P_crit,
    where the march deliberately blends T with the previous step.
    """
    area = _well_area(simulation['well_params'])
    mdot = simulation['mass_flow']
    n_checked = 0
    for (d, T_C), (_, P), (_, h), (_, v), (_, x) in zip(
            simulation['T'], simulation['P'], simulation['h'],
            simulation['v'], simulation['saturation']):
        if abs(P - P_CRIT_MPA) < 0.5:
            continue
        eos = fluid_properties_Ph(P, h * 1e6)
        assert eos['success'], f'EOS at {d} m'
        assert T_C == pytest.approx(eos['temperature_K'] - 273.15,
                                    abs=1e-6), f'T at {d} m'
        assert v == pytest.approx(mdot / (eos['density_kgm3'] * area),
                                  rel=1e-9), f'v at {d} m'
        if np.isnan(eos['saturation']):
            assert np.isnan(x), f'x at {d} m'
        else:
            assert x == pytest.approx(eos['saturation'], abs=1e-9), \
                f'x at {d} m'
        n_checked += 1
    assert n_checked > 0, 'no entries checked'


@pytest.mark.slow
class TestStoredStateIsSelfConsistent:
    """
    The march stores T, v and x from the state it is about to step
    and P, h from the state it stepped to would leave T, v and x one
    step behind P and h. The existing T(P, h) and continuity checks
    allow 5 K and 5%, which a one-step lag (about 0.07 K per 10 m
    step in these wells) passes; these checks are to round-off.
    """

    def test_profile_starts_at_the_feedzone_state(self,
                                                  iddp_simulation):
        """The first entry is the bottomhole (P, h) the march began at."""
        depth = iddp_simulation['well_params']['depth_m']
        h_fz = CP.PropsSI('H', 'T', 530 + 273.15, 'P', 16.4e6, 'Water')
        assert iddp_simulation['P'][0] == (depth, 16.4)
        assert iddp_simulation['h'][0][0] == depth
        assert iddp_simulation['h'][0][1] == pytest.approx(h_fz * 1e-6,
                                                           abs=1e-12)

    def test_single_phase_state_matches_eos(self, iddp_simulation):
        """Superheated vapour, CoolProp route."""
        _assert_state_matches_eos(iddp_simulation)

    def test_two_phase_state_matches_eos(self, dome_simulation):
        """Through the dome: T = T_sat(P) and x = x(P, h) at each depth."""
        _assert_state_matches_eos(dome_simulation)


# ====================================================================
# 13. TWO-PHASE CHOKE DETECTION
# ====================================================================

def _saturation_line_hem_sound_speed(P_MPa, x):
    """
    c_HEM from saturation-line derivatives, independently of the
    P-S isentrope that _hem_sound_speed() differences: in the dome,
    s = s_f + x s_fg held constant fixes dx/dP, v = v_f + x v_fg then
    gives (dv/dP)_s, and c^2 = -v^2 / (dv/dP)_s.
    """
    P = P_MPa * 1e6
    dP = 1e-5 * P

    def saturation(p):
        return np.array([1.0 / CP.PropsSI('D', 'P', p, 'Q', 0, 'Water'),
                         1.0 / CP.PropsSI('D', 'P', p, 'Q', 1, 'Water'),
                         CP.PropsSI('S', 'P', p, 'Q', 0, 'Water'),
                         CP.PropsSI('S', 'P', p, 'Q', 1, 'Water')])

    v_f, v_g, s_f, s_g = saturation(P)
    dv_f, dv_g, ds_f, ds_g = (saturation(P + dP) - saturation(P - dP)) / (2 * dP)
    dx_dP = -(ds_f + x * (ds_g - ds_f)) / (s_g - s_f)
    dv_dP = dv_f + x * (dv_g - dv_f) + (v_g - v_f) * dx_dP
    v = v_f + x * (v_g - v_f)
    return np.sqrt(-v * v / dv_dP)


@pytest.mark.parametrize('P_MPa, x', [(0.2, 0.05), (1.0, 0.01), (1.0, 0.5),
                                      (1.0, 0.9), (5.0, 0.1), (10.0, 0.3),
                                      (20.0, 0.5)])
def test_hem_sound_speed_matches_saturation_line_derivative(P_MPa, x):
    """The isentrope difference agrees with the saturation-line form."""
    P_Pa = P_MPa * 1e6
    h_f = CP.PropsSI('H', 'P', P_Pa, 'Q', 0, 'Water')
    h_g = CP.PropsSI('H', 'P', P_Pa, 'Q', 1, 'Water')
    c = _hem_sound_speed(P_MPa, h_f + x * (h_g - h_f))
    assert c == pytest.approx(_saturation_line_hem_sound_speed(P_MPa, x),
                              rel=1e-3)


def test_hem_sound_speed_is_below_both_phase_sound_speeds():
    """
    Phase change makes the equilibrium mixture far softer than either
    phase: at 1 MPa and x = 0.5, c_HEM is about 320 m/s against
    about 1400 m/s for the saturated liquid and 500 m/s for the
    saturated vapor.
    """
    P_Pa = 1e6
    h_f = CP.PropsSI('H', 'P', P_Pa, 'Q', 0, 'Water')
    h_g = CP.PropsSI('H', 'P', P_Pa, 'Q', 1, 'Water')
    c = _hem_sound_speed(1.0, 0.5 * (h_f + h_g))
    assert c < CP.PropsSI('A', 'P', P_Pa, 'Q', 1, 'Water')
    assert c < CP.PropsSI('A', 'P', P_Pa, 'Q', 0, 'Water')


def _flashing_march(mass_flow, diagnostics=None):
    """
    A liquid feed that flashes on the way up and self-flows to the
    surface: 300 C compressed liquid at 10 MPa (T_sat = 311 C) at the
    bottom of a 2000 m, 0.217 m well, which starts to boil about 200 m
    above the feedzone, near 8.4 MPa.
    """
    wp = dict(DEFAULT_WELL_PARAMS)
    wp['depth_m'] = 2000
    h_fz = CP.PropsSI('H', 'T', 300 + 273.15, 'P', 10e6, 'Water')
    rock = _linear_rock_temperatures(wp['depth_m'], 300, wp['delta_z_m'])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        profiles = wellbore_simulate(10.0, h_fz, mass_flow, rock, wp,
                                     diagnostics=diagnostics)
    return profiles, caught


@pytest.fixture(scope='module')
def supersonic_flashing_march():
    """
    At 102.5 kg/s the mixture reaches the wellhead at about 1.3 times
    the HEM sound speed; 102 kg/s stays just below it (about 0.97).
    """
    diagnostics = {}
    profiles, caught = _flashing_march(102.5, diagnostics)
    return profiles, caught, diagnostics


@pytest.mark.slow
class TestTwoPhaseChokeDetection:
    """
    The EOS gives no sound speed for two-phase states, so a flashing
    well never set the choked flag, however fast the mixture moved.
    The march now compares the mixture velocity with the HEM sound
    speed, reports reaching it and sets the choked flag, as a
    single-phase choke does, without changing the march.
    """

    def test_flashing_well_self_flows_to_the_surface(
            self, supersonic_flashing_march):
        """The case is the one targeted: liquid feed, two-phase wellhead."""
        profiles, _, _ = supersonic_flashing_march
        P_prof, s_prof = profiles[1], profiles[4]
        assert np.isnan(s_prof[0][1]), 'single-phase liquid feed'
        assert P_prof[-1][0] == 0 and P_prof[-1][1] > 0, \
            'reaches the surface with a positive WHP'
        assert 0 < s_prof[-1][1] < 1, 'two-phase at the wellhead'

    def test_reaching_the_hem_sound_speed_is_flagged(
            self, supersonic_flashing_march):
        """Flag, depth and Mach number in the diagnostics dict."""
        _, _, diagnostics = supersonic_flashing_march
        assert diagnostics['two_phase_choked'] is True
        assert diagnostics['two_phase_mach_max'] >= 1.0
        assert diagnostics['two_phase_choke_depth_m'] is not None

    def test_reaching_the_hem_sound_speed_is_warned_once(
            self, supersonic_flashing_march):
        """One TwoPhaseChokeWarning, a RuntimeWarning subclass."""
        _, caught, _ = supersonic_flashing_march
        hem = [w for w in caught
               if issubclass(w.category, TwoPhaseChokeWarning)]
        assert len(hem) == 1
        assert issubclass(TwoPhaseChokeWarning, RuntimeWarning)

    def test_choked_flag_is_set(self, supersonic_flashing_march):
        """
        A two-phase choke sets the same choked flag as a single-phase
        one, which the flow solvers and the client act on.
        """
        profiles, _, _ = supersonic_flashing_march
        assert profiles[-1] is True

    def test_detection_does_not_change_the_march(
            self, supersonic_flashing_march):
        """Same profiles with and without the diagnostics dict."""
        profiles, _, _ = supersonic_flashing_march
        plain, _ = _flashing_march(102.5)
        for with_diag, without in zip(profiles[:6], plain[:6]):
            np.testing.assert_array_equal(np.array(with_diag),
                                          np.array(without))

    def test_subsonic_flashing_well_is_not_flagged(self):
        """At 90 kg/s the same well peaks near Mach 0.26."""
        diagnostics = {}
        profiles, caught = _flashing_march(90.0, diagnostics)
        assert diagnostics['two_phase_choked'] is False
        assert diagnostics['two_phase_choke_depth_m'] is None
        assert 0 < diagnostics['two_phase_mach_max'] < 0.5
        assert not any(issubclass(w.category, TwoPhaseChokeWarning)
                       for w in caught)
        assert profiles[-1] is False, 'not choked'

    def test_single_phase_well_has_no_two_phase_mach(self):
        """No two-phase step, nothing to compare: Mach is NaN."""
        diagnostics = {}
        wp = dict(DEFAULT_WELL_PARAMS)
        h = CP.PropsSI('H', 'T', 530 + 273.15, 'P', 16.4e6, 'Water')
        rock = _linear_rock_temperatures(wp['depth_m'], 530,
                                         wp['delta_z_m'])
        wellbore_simulate(16.4, h, 20.0, rock, wp, diagnostics=diagnostics)
        assert diagnostics['two_phase_choked'] is False
        assert np.isnan(diagnostics['two_phase_mach_max'])
