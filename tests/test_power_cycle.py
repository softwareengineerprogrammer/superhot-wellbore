# -*- coding: utf-8 -*-
"""
Surface Power Cycle
===================

Covers :mod:`superhot_wellbore.power_cycle`: cycle selection from the
wellhead state, the First and Second Law bounds that any reported
efficiency has to respect, the DiPippo/Baumann turbine expansion, and
the exergy bookkeeping. The module is a chain of thermodynamic state
lookups, so a wrong branch or a dropped term does not raise -- it just
returns a plausible-looking power number. These tests pin the branch
points and the algebra against CoolProp and against the textbook
equations they were taken from.

Ported from the self-test block that used to live at the bottom of
power_cycle.py; the expected values, tolerances and literature
citations are carried over unchanged.

Author: Samuel W. Scott
"""

import warnings

import CoolProp.CoolProp as CP
import numpy as np
import pytest

from superhot_wellbore.power_cycle import (DEFAULT_POWER_PARAMS,
                                           _baumann_efficiency,
                                           _dipippo_outlet_enthalpy,
                                           _flash_cycle, _merge_params,
                                           _two_stage_turbine,
                                           dry_steam_specific_work,
                                           power_cycle_analysis)


# ====================================================================
# HELPERS
# ====================================================================

def _h_MJkg(T_C, P_MPa):
    """Compute enthalpy [MJ/kg] from (T, P) for test inputs."""
    return CP.PropsSI('H', 'T', T_C + 273.15,
                      'P', P_MPa * 1e6, 'Water') / 1e6


def _mock_cm(mass_flow, P_whp, h_surface_MJkg,
             h_feedzone_MJkg=None, P_bh_MPa=None):
    """Build a mock coupled_model output dict for testing."""
    d = {
        'mass_flow_kgs': mass_flow,
        'whp_MPa': P_whp,
        'h_surface_MJkg': h_surface_MJkg,
        'success': True,
    }
    if h_feedzone_MJkg is not None:
        d['h_feedzone_MJkg'] = h_feedzone_MJkg
    if P_bh_MPa is not None:
        d['P_bh_MPa'] = P_bh_MPa
    return d


def _h_saturated_MJkg(P_MPa, quality):
    """Saturated enthalpy [MJ/kg] at a pressure and quality."""
    return CP.PropsSI('H', 'P', P_MPa * 1e6, 'Q', quality, 'Water') / 1e6


def _h_two_phase_MJkg(P_MPa):
    """Enthalpy [MJ/kg] midway between h_f and h_g at a pressure."""
    return 0.5 * (_h_saturated_MJkg(P_MPa, 0)
                  + _h_saturated_MJkg(P_MPa, 1))


def _moisture_warnings(caught):
    """Filter recorded warnings down to the moisture warning."""
    return [x for x in caught
            if 'quality' in str(x.message).lower()
            or 'moisture' in str(x.message).lower()]


# ====================================================================
# 1. CYCLE SELECTION LOGIC
# ====================================================================

class TestCycleSelection:
    """
    Superheated/supercritical vapor at the wellhead should route to
    the binary cycle. Two-phase mixture should route to flash.
    The threshold is h_sat_vapor(WHP) + superheat_margin_Jkg.
    """

    def test_superheated_inlet_selects_binary(self):
        """Superheated: 480 C at 10 MPa -> well above saturation."""
        r = power_cycle_analysis(
            _mock_cm(30.0, 10.0, _h_MJkg(480.0, 10.0)))
        assert r['cycle'] == 'binary', f"got {r['cycle']}"

    def test_two_phase_inlet_selects_flash(self):
        """Two-phase: h midway between h_f and h_g at 5 MPa."""
        r = power_cycle_analysis(
            _mock_cm(50.0, 5.0, _h_two_phase_MJkg(5.0)))
        assert r['cycle'] == 'flash', f"got {r['cycle']}"


class TestSupercriticalWellhead:
    """
    Above the critical pressure (22.064 MPa) there is no saturated
    vapor state to compare against, but the wellhead fluid is
    single-phase and the analysis must still run. A deep superhot
    well (450 C reservoir at 10-12 km) reaches the wellhead as a
    dense, liquid-like supercritical fluid at 24-26 MPa and ~375 C:
    it flashes like the same enthalpy just below P_crit would. A
    vapor-like supercritical fluid routes to the binary cycle.
    """

    P_CRIT = 22.064

    @staticmethod
    def _analyse(cm):
        """Run the analysis, failing the test on any failure warning."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            r = power_cycle_analysis(cm)
        failures = [x for x in caught if 'failed' in str(x.message)]
        assert not failures, [str(x.message) for x in failures]
        return r

    def test_liquid_like_supercritical_wellhead_flashes(self):
        """375 C at 26 MPa: h ~ 1.83 MJ/kg, rho ~ 520 kg/m3."""
        r = self._analyse(_mock_cm(60.0, 26.0, _h_MJkg(375.0, 26.0)))
        assert r['success'], r
        assert r['cycle'] == 'flash', f"got {r['cycle']}"
        assert np.isfinite(r['power_MWe']) and r['power_MWe'] > 0
        assert 0 < r['eta_utilization'] < 1
        assert np.isfinite(r['exergy_rate_MW']) and r['exergy_rate_MW'] > 0

    def test_vapor_like_supercritical_wellhead_selects_binary(self):
        """500 C at 25 MPa: h ~ 3.17 MJ/kg, rho ~ 90 kg/m3."""
        r = self._analyse(_mock_cm(30.0, 25.0, _h_MJkg(500.0, 25.0)))
        assert r['success'], r
        assert r['cycle'] == 'binary', f"got {r['cycle']}"
        assert np.isfinite(r['power_MWe']) and r['power_MWe'] > 0

    @pytest.mark.parametrize('h_MJkg, expected_cycle', [
        (1.9, 'flash'),
        (3.0, 'binary'),
    ])
    def test_selection_is_continuous_across_critical_pressure(
            self, h_MJkg, expected_cycle):
        """
        The same wellhead enthalpy 1% below and 1% above P_crit must
        select the same cycle and give practically the same power.
        """
        below = self._analyse(_mock_cm(60.0, 0.99 * self.P_CRIT, h_MJkg))
        above = self._analyse(_mock_cm(60.0, 1.01 * self.P_CRIT, h_MJkg))
        assert below['success'] and above['success']
        assert below['cycle'] == expected_cycle, below['cycle']
        assert above['cycle'] == expected_cycle, above['cycle']
        assert above['power_MWe'] == pytest.approx(below['power_MWe'],
                                                   rel=0.02)

    def test_selection_enthalpy_matches_saturation_below_critical(self):
        """Below P_crit the helper is exactly h_g(P): no behaviour change."""
        from superhot_wellbore.power_cycle import _cycle_selection_enthalpy
        for P in (1.0, 10.0, 21.0, 22.0):
            assert _cycle_selection_enthalpy(P) == pytest.approx(
                _h_saturated_MJkg(P, 1) * 1e6)

    def test_selection_enthalpy_continuous_at_critical_pressure(self):
        """The continuation joins h_g(P) at the critical point."""
        from superhot_wellbore.power_cycle import _cycle_selection_enthalpy
        just_below = _cycle_selection_enthalpy(self.P_CRIT - 1e-4)
        at = _cycle_selection_enthalpy(self.P_CRIT)
        assert at == pytest.approx(just_below, rel=0.005)
        assert _cycle_selection_enthalpy(30.0) == at


# ====================================================================
# 2. FIRST LAW CONSISTENCY (BINARY CYCLE)
# ====================================================================

class TestFirstLawConsistency:
    """
    For a binary cycle, the thermal efficiency must satisfy:
      0 < eta_th < 1  (cannot produce more work than heat input)
    Typical geothermal binary plants: eta_th ~ 10-25% (DiPippo,
    2016, Sec. 8.2.5; Mines, 2016, Sec. 13.3.2).
    """

    @pytest.mark.parametrize('T_C, P_MPa', [
        (400, 5.0),
        (480, 10.0),
        (550, 15.0),
    ], ids=['400C/5MPa', '480C/10MPa', '550C/15MPa'])
    def test_thermal_efficiency_between_zero_and_one(self, T_C, P_MPa):
        """0 < eta_th < 1 for every superheated binary inlet."""
        r = power_cycle_analysis(
            _mock_cm(30.0, P_MPa, _h_MJkg(T_C, P_MPa)))
        assert r['success'], 'analysis succeeded'
        assert r['cycle'] == 'binary', f"got {r['cycle']}"
        assert 0 < r['eta_thermal'] < 1.0, \
            f'eta_th = {r["eta_thermal"]:.3f}'

    @pytest.mark.parametrize('T_C, P_MPa', [
        (400, 5.0),
        (480, 10.0),
        (550, 15.0),
    ], ids=['400C/5MPa', '480C/10MPa', '550C/15MPa'])
    def test_specific_output_equals_power_over_mass_flow(self, T_C, P_MPa):
        """Specific output must equal power / mass flow."""
        r = power_cycle_analysis(
            _mock_cm(30.0, P_MPa, _h_MJkg(T_C, P_MPa)))
        assert r['success'], 'analysis succeeded'
        assert r['cycle'] == 'binary', f"got {r['cycle']}"
        w_check = r['power_MWe'] / 30.0
        assert r['specific_output_MJkg'] == pytest.approx(w_check,
                                                         abs=1e-6), \
            f'w = W/m = {w_check:.4f} MJ/kg'


# ====================================================================
# 3. SECOND LAW BOUNDS (BOTH CYCLES)
# ====================================================================

def _second_law_case(label):
    """Build one of the Second Law operating points by label."""
    if label == 'binary 480C/10MPa':
        return _mock_cm(30.0, 10.0, _h_MJkg(480, 10.0))
    if label == 'flash h=2-phase/5MPa':
        return _mock_cm(50.0, 5.0, _h_two_phase_MJkg(5.0))
    return _mock_cm(20.0, 4.0, _h_MJkg(440, 4.0))


class TestSecondLawBounds:
    """
    Utilization efficiency must satisfy 0 < eta_u < 1 for any
    feasible operating condition. It can never exceed 1 because
    that would violate the Second Law of thermodynamics.
    """

    @pytest.mark.parametrize('label', [
        'binary 480C/10MPa',
        'flash h=2-phase/5MPa',
        'binary 440C/4MPa',
    ])
    def test_utilization_efficiency_between_zero_and_one(self, label):
        """0 < eta_u < 1 at every feasible operating point."""
        r = power_cycle_analysis(_second_law_case(label))
        assert r['success'], 'analysis succeeded'
        assert 0 < r['eta_utilization'] < 1.0, \
            f'eta_u = {r["eta_utilization"]:.3f}'

    @pytest.mark.parametrize('label', [
        'binary 480C/10MPa',
        'flash h=2-phase/5MPa',
        'binary 440C/4MPa',
    ])
    def test_exergy_rate_is_positive(self, label):
        """The inlet exergy rate must be strictly positive."""
        r = power_cycle_analysis(_second_law_case(label))
        assert r['success'], 'analysis succeeded'
        assert r['exergy_rate_MW'] > 0, \
            f'E = {r["exergy_rate_MW"]:.2f} MW'


# ====================================================================
# 4. DIPIPPO EQUATION: OUTLET ENTHALPY BOUNDS
# ====================================================================

def _dipippo_expansion(P_in_MPa):
    """Expand saturated vapor at P_in and return the state set."""
    P_cond = DEFAULT_POWER_PARAMS['P_condenser_MPa']
    eta_td = DEFAULT_POWER_PARAMS['eta_turbine_dry']
    h_in = CP.PropsSI('H', 'P', P_in_MPa * 1e6, 'Q', 1, 'Water')
    s_in = CP.PropsSI('S', 'P', P_in_MPa * 1e6, 'Q', 1, 'Water')
    h_is = CP.PropsSI('H', 'S', s_in, 'P', P_cond * 1e6, 'Water')
    h_f = CP.PropsSI('H', 'P', P_cond * 1e6, 'Q', 0, 'Water')
    h_g = CP.PropsSI('H', 'P', P_cond * 1e6, 'Q', 1, 'Water')
    h_c = _dipippo_outlet_enthalpy(h_in, h_is, h_f, h_g, eta_td)
    return h_in, h_is, h_f, h_g, h_c


class TestDiPippoOutletEnthalpy:
    """
    The actual turbine outlet enthalpy h_c from the DiPippo
    implicit equation must satisfy h_is < h_c < h_in, where
    h_is is the isentropic outlet enthalpy. This verifies that
    the Baumann correction produces a physically reasonable
    result between the ideal (isentropic) and no-work limits.
    """

    @pytest.mark.parametrize('P_in', [1.0, 5.0, 10.0],
                             ids=['1 MPa sat vapor',
                                  '5 MPa sat vapor',
                                  '10 MPa sat vapor'])
    def test_outlet_enthalpy_between_isentropic_and_inlet(self, P_in):
        """h_is < h_c < h_in for every inlet pressure."""
        h_in, h_is, h_f, h_g, h_c = _dipippo_expansion(P_in)
        assert h_is < h_c < h_in, \
            (f'h_is={h_is/1e3:.1f}, h_c={h_c/1e3:.1f}, '
             f'h_in={h_in/1e3:.1f}')

    @pytest.mark.parametrize('P_in', [1.0, 5.0, 10.0],
                             ids=['1 MPa sat vapor',
                                  '5 MPa sat vapor',
                                  '10 MPa sat vapor'])
    def test_exit_quality_between_zero_and_one(self, P_in):
        """Exit quality should be between 0 and 1."""
        h_in, h_is, h_f, h_g, h_c = _dipippo_expansion(P_in)
        x_c = (h_c - h_f) / (h_g - h_f)
        assert 0 < x_c < 1, f'x = {x_c:.3f}'


# ====================================================================
# 5. FLASH CYCLE: MONOTONIC POWER WITH ENTHALPY
# ====================================================================

def test_flash_power_increases_with_enthalpy():
    """
    At fixed WHP and flash pressure, increasing the inlet enthalpy
    increases the separator steam fraction (Eq. 5.7) and therefore
    the turbine mass flow and power output. This tests the entire
    flash path from lever rule through turbine expansion.
    """
    P_flash_test = 5.0  # MPa (WHP for the flash cases)
    P_flash_sep = DEFAULT_POWER_PARAMS['P_flash_MPa']  # 1.0 MPa
    h_f_fl = _h_saturated_MJkg(P_flash_sep, 0)
    h_g_fl = _h_saturated_MJkg(P_flash_sep, 1)

    # Enthalpy values spanning the two-phase dome at 1 MPa flash
    h_vals = np.linspace(h_f_fl + 0.05, h_g_fl - 0.05, 6)
    powers = []
    for h_val in h_vals:
        r = power_cycle_analysis(_mock_cm(50.0, P_flash_test, h_val))
        if r['success']:
            powers.append(r['power_MWe'])
        else:
            powers.append(np.nan)

    valid = [p for p in powers if not np.isnan(p)]
    assert all(valid[i] <= valid[i+1] for i in range(len(valid)-1)), \
        f'powers = {[f"{p:.2f}" for p in valid]}'


# ====================================================================
# 6. BINARY CYCLE: HIGHER TEMPERATURE -> MORE POWER
# ====================================================================

def test_binary_power_increases_with_temperature():
    """
    At fixed pressure and mass flow, increasing the wellhead
    temperature (and therefore enthalpy) should increase the
    turbine power output, because more heat is transferred to
    the working fluid in the heat exchanger.
    """
    P_bin = 10.0  # MPa
    m_bin = 30.0  # kg/s
    temps = [380, 420, 460, 500, 540]
    bin_powers = []
    for T in temps:
        r = power_cycle_analysis(
            _mock_cm(m_bin, P_bin, _h_MJkg(T, P_bin)))
        if r['success'] and r['cycle'] == 'binary':
            bin_powers.append(r['power_MWe'])
        else:
            bin_powers.append(np.nan)

    valid_bp = [p for p in bin_powers if not np.isnan(p)]
    assert len(valid_bp) >= 3, f'{len(valid_bp)} binary points'
    assert all(valid_bp[i] <= valid_bp[i+1]
               for i in range(len(valid_bp)-1)), \
        f'powers = {[f"{p:.2f}" for p in valid_bp]}'


# ====================================================================
# 7. EDGE CASES AND INFEASIBLE CONDITIONS
# ====================================================================

class TestEdgeCases:
    """Infeasible wellhead states must fail instead of guessing."""

    def test_zero_mass_flow_fails(self):
        """Zero mass flow -> should fail."""
        r = power_cycle_analysis(_mock_cm(0.0, 10.0, 3.0))
        assert not r['success']

    def test_wellhead_pressure_below_flash_pressure_fails(self):
        """WHP below flash pressure -> flash should fail."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            r = power_cycle_analysis(_mock_cm(50.0, 0.5, 2.5))
        assert not r['success'], \
            f"got success={r['success']}, cycle={r['cycle']}"

    def test_subcooled_liquid_fails(self):
        """Subcooled liquid: h well below h_f at flash pressure."""
        h_f_1MPa = _h_saturated_MJkg(1.0, 0)
        h_sub = h_f_1MPa * 0.5  # clearly subcooled
        r = power_cycle_analysis(_mock_cm(50.0, 5.0, h_sub))
        assert not r['success'], \
            f'h={h_sub:.3f} MJ/kg, h_f(1MPa)={h_f_1MPa:.3f}'

    def test_cold_fluid_fails(self):
        """Very cold fluid (T < 120 C) -> should fail."""
        r = power_cycle_analysis(
            _mock_cm(50.0, 5.0, _h_MJkg(100.0, 5.0)))
        assert not r['success']

    def test_missing_mass_flow_key_fails(self):
        """Missing required key -> should fail gracefully."""
        r = power_cycle_analysis({'whp_MPa': 10.0,
                                  'h_surface_MJkg': 3.0})
        assert not r['success']


# ====================================================================
# 8. BAUMANN RULE: WET TURBINE EFFICIENCY
# ====================================================================

class TestBaumannRule:
    """
    The Baumann rule (DiPippo, 2012, Eq. 5.12) gives eta_tw for
    a saturated vapor inlet (x_in = 1) as:
      eta_tw = eta_td * (1 + x_out) / 2
    At x_out = 1: eta_tw = eta_td (no penalty for dry exhaust)
    At x_out = 0.85: eta_tw = 0.85 * 1.85/2 = 0.786
    """

    def test_dry_exhaust_keeps_dry_efficiency(self):
        """x=1.0 -> eta = eta_td."""
        assert _baumann_efficiency(0.85, 1.0) == pytest.approx(0.85,
                                                               abs=1e-10)

    def test_quality_085_gives_0786(self):
        """x=0.85 -> eta = 0.786."""
        assert _baumann_efficiency(0.85, 0.85) == pytest.approx(
            0.85 * 1.85 / 2, abs=1e-10)

    def test_saturated_liquid_exhaust_halves_efficiency(self):
        """x=0.0 -> eta = eta_td/2."""
        assert _baumann_efficiency(0.85, 0.0) == pytest.approx(0.425,
                                                               abs=1e-10)


# ====================================================================
# 9. FEEDZONE EXERGY WITH P_BH
# ====================================================================

class TestFeedzoneExergy:
    """
    When both h_feedzone_MJkg and P_bh_MPa are provided, feedzone
    exergy should use the correct (h_fz, P_bh) state. When only
    h_feedzone is given, it should fall back to P_whp for entropy.
    """

    @staticmethod
    def _states():
        """Wellhead and feedzone enthalpies for the exergy cases."""
        return _h_MJkg(440, 4.0), _h_MJkg(500, 25.0)

    def test_feedzone_exergy_with_bottomhole_pressure(self):
        """With P_bh: full feedzone state."""
        h_wh, h_fz = self._states()
        r_full = power_cycle_analysis(
            _mock_cm(48.0, 4.0, h_wh,
                     h_feedzone_MJkg=h_fz, P_bh_MPa=25.0))
        assert not np.isnan(r_full['eta_utilization_fz']), \
            f"eta_u_fz = {r_full['eta_utilization_fz']}"

    def test_feedzone_exergy_without_bottomhole_pressure(self):
        """Without P_bh: falls back to P_whp."""
        h_wh, h_fz = self._states()
        r_no_pbh = power_cycle_analysis(
            _mock_cm(48.0, 4.0, h_wh, h_feedzone_MJkg=h_fz))
        assert not np.isnan(r_no_pbh['eta_utilization_fz'])

    def test_bottomhole_pressure_changes_feedzone_exergy(self):
        """The two should differ because entropy depends on pressure."""
        h_wh, h_fz = self._states()
        r_full = power_cycle_analysis(
            _mock_cm(48.0, 4.0, h_wh,
                     h_feedzone_MJkg=h_fz, P_bh_MPa=25.0))
        r_no_pbh = power_cycle_analysis(
            _mock_cm(48.0, 4.0, h_wh, h_feedzone_MJkg=h_fz))
        assert abs(r_full['exergy_rate_fz_MW']
                   - r_no_pbh['exergy_rate_fz_MW']) > 0.01, \
            (f"E_fz(P_bh)={r_full['exergy_rate_fz_MW']:.2f}, "
             f"E_fz(P_whp)={r_no_pbh['exergy_rate_fz_MW']:.2f}")

    def test_both_exergy_rates_positive_and_finite(self):
        """Wellhead and feedzone exergy rates stay in (0, 500) MW."""
        h_wh, h_fz = self._states()
        r_full = power_cycle_analysis(
            _mock_cm(48.0, 4.0, h_wh,
                     h_feedzone_MJkg=h_fz, P_bh_MPa=25.0))
        assert 0 < r_full['exergy_rate_MW'] < 500
        assert 0 < r_full['exergy_rate_fz_MW'] < 500

    def test_without_feedzone_enthalpy_metrics_are_nan(self):
        """Without feedzone enthalpy -> fz metrics are NaN."""
        h_wh, h_fz = self._states()
        r = power_cycle_analysis(_mock_cm(48.0, 4.0, h_wh))
        assert np.isnan(r['eta_utilization_fz'])


# ====================================================================
# 10. THERMAL EFFICIENCY IS NAN FOR FLASH, REAL FOR BINARY
# ====================================================================

class TestThermalEfficiencyDefinition:
    """
    Flash plants are not closed thermodynamic cycles, so thermal
    efficiency is not conventionally defined (DiPippo, 2012,
    Sec. 5.4.7). Binary cycles should have real eta_th.
    """

    def test_binary_thermal_efficiency_is_real(self):
        """Binary: eta_th is real."""
        r_bin = power_cycle_analysis(
            _mock_cm(30.0, 10.0, _h_MJkg(480, 10.0)))
        assert not np.isnan(r_bin['eta_thermal']), \
            f"eta_th = {r_bin['eta_thermal']}"

    def test_flash_thermal_efficiency_is_nan(self):
        """Flash: eta_th is NaN."""
        r_fl = power_cycle_analysis(
            _mock_cm(50.0, 5.0, _h_two_phase_MJkg(5.0)))
        assert np.isnan(r_fl['eta_thermal'])


# ====================================================================
# 12. TURBINE EXIT QUALITY WARNING THRESHOLD
# ====================================================================

def _expand_saturated_vapor(P_in_MPa, pp=None):
    """Run the two-stage turbine on saturated vapor at P_in."""
    h_in = CP.PropsSI('H', 'P', P_in_MPa * 1e6, 'Q', 1, 'Water')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        r = _two_stage_turbine(1.0, h_in, P_in_MPa,
                               pp if pp is not None
                               else DEFAULT_POWER_PARAMS)
    return r, caught


class TestTurbineExitQualityWarning:
    """
    This section does two things:
      (1) reports outlet qualities for several representative cases
      (2) verifies that the moisture warning is triggered when the
          exit quality drops below x_exit_min = 0.85

    The realistic cases below use saturated-vapor turbine inlet
    conditions at several pressures, which correspond to the wet-stage
    test cases already used above. A deliberately degraded case is
    then created by tightening the quality threshold so that the
    warning mechanism itself can be tested deterministically.
    """

    @pytest.mark.parametrize('P_in', [1.0, 5.0, 10.0],
                             ids=['sat. vapor at 1 MPa',
                                  'sat. vapor at 5 MPa',
                                  'sat. vapor at 10 MPa'])
    def test_turbine_calculation_succeeded(self, P_in):
        """The two-stage expansion converges at every pressure."""
        r, caught = _expand_saturated_vapor(P_in)
        assert r['success']

    @pytest.mark.parametrize('P_in', [1.0, 5.0, 10.0],
                             ids=['sat. vapor at 1 MPa',
                                  'sat. vapor at 5 MPa',
                                  'sat. vapor at 10 MPa'])
    def test_exit_quality_in_range(self, P_in):
        """x_exit stays inside the two-phase dome."""
        r, caught = _expand_saturated_vapor(P_in)
        assert 0.0 < r['x_exit'] < 1.0, f"x_exit = {r['x_exit']:.3f}"

    def test_no_moisture_warning_at_the_cycle_inlet_pressure(self):
        """
        No moisture warning at the only inlet pressure the cycles
        actually use.

        DEFAULT_POWER_PARAMS sets P_flash_MPa = P_wf_MPa = 1.0, so
        both the flash and the binary path enter the turbine at
        1.0 MPa. Expanding saturated vapor from there to the 0.01 MPa
        condenser leaves x_exit = 0.886, above x_exit_min = 0.85.
        """
        r, caught = _expand_saturated_vapor(1.0)
        moist_warns = _moisture_warnings(caught)
        assert len(moist_warns) == 0, \
            f'warnings = {[str(x.message) for x in moist_warns]}'

    @pytest.mark.parametrize('P_in', [5.0, 10.0],
                             ids=['sat. vapor at 5 MPa',
                                  'sat. vapor at 10 MPa'])
    def test_moisture_warning_at_elevated_inlet_pressure(self, P_in):
        """
        The warning does fire for wetter expansions.

        Saturated vapor at 5 / 10 MPa expands to x_exit = 0.844 /
        0.812, genuinely below x_exit_min = 0.85, so warning about
        blade erosion is the correct behaviour. These inlet states
        are reachable only by driving _two_stage_turbine directly;
        neither cycle can produce them (see the test above).
        """
        r, caught = _expand_saturated_vapor(P_in)
        moist_warns = _moisture_warnings(caught)
        assert len(moist_warns) > 0, \
            f"x_exit = {r['x_exit']:.3f} but no warning was raised"

    def test_forced_moisture_threshold_warning_triggered(self):
        """
        Deterministic warning test: use the 1 MPa saturated-vapor
        case, but temporarily tighten the threshold above its actual
        x_exit so that the warning must fire.
        """
        r_base, _ = _expand_saturated_vapor(1.0)
        pp_warn = dict(DEFAULT_POWER_PARAMS)
        pp_warn['x_exit_min'] = min(0.999, r_base['x_exit'] + 0.01)

        r_warn, caught = _expand_saturated_vapor(1.0, pp_warn)
        moist_warns = _moisture_warnings(caught)
        assert len(moist_warns) > 0, \
            f'warnings = {[str(x.message) for x in moist_warns]}'

    def test_forced_warning_case_still_returns_success(self):
        """The moisture warning is advisory, not a failure."""
        r_base, _ = _expand_saturated_vapor(1.0)
        pp_warn = dict(DEFAULT_POWER_PARAMS)
        pp_warn['x_exit_min'] = min(0.999, r_base['x_exit'] + 0.01)

        r_warn, caught = _expand_saturated_vapor(1.0, pp_warn)
        assert r_warn['success']


# ====================================================================
# 8. DRY-STEAM SPECIFIC WORK
# ====================================================================

class TestDrySteamSpecificWork:
    """
    dry_steam_specific_work(P) is the gross work of 1 kg of saturated
    steam expanded from P through _two_stage_turbine(). It exists so
    that a caller can price the steam fraction of a two-phase wellhead
    separately from its liquid; it must therefore agree with the flash
    cycle when the flash cycle is fed pure saturated vapour at the
    same pressure, grow with the inlet pressure over the range where
    the extra pressure ratio outweighs the extra exhaust moisture, and
    refuse pressures where saturated vapour does not exist.
    """

    def test_rises_with_pressure_then_peaks(self):
        """
        More inlet pressure, more work, up to a broad maximum.

        Saturated steam expanded to 0.01 MPa gains work with inlet
        pressure until the exhaust moisture penalty (Baumann) takes
        over; with the default 0.85 dry efficiency the maximum sits
        between 5 and 15 MPa (about 0.59 MJ/kg at 7-10 MPa), and the
        work falls off towards the critical pressure.
        """
        pressures = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 7.0])
        work = np.array([dry_steam_specific_work(P) for P in pressures])
        assert np.all(np.isfinite(work)), work
        assert np.all(np.diff(work) > 0), f'not monotone: {work}'
        assert 0.3 < work[0] < 0.5 and 0.5 < work[-1] < 0.7, \
            f'plausible MJ/kg range: {work}'
        peak = max(dry_steam_specific_work(P) for P in (5.0, 7.0, 10.0, 15.0))
        assert dry_steam_specific_work(20.0) < peak, 'falls off above'
        assert dry_steam_specific_work(22.0) < dry_steam_specific_work(20.0)

    def test_nan_at_and_above_the_critical_pressure(self):
        """No saturated vapour at or above P_crit, no work."""
        for P in (22.064, 25.0, 30.0):
            assert np.isnan(dry_steam_specific_work(P)), P

    def test_nan_at_or_below_the_condenser_pressure(self):
        """Nothing to expand at the condenser pressure."""
        pp = _merge_params(None)
        assert np.isnan(dry_steam_specific_work(pp['P_condenser_MPa']))
        assert np.isnan(dry_steam_specific_work(0.0))
        assert np.isnan(dry_steam_specific_work(float('nan')))
        assert np.isnan(dry_steam_specific_work(None))

    def test_equals_the_flash_cycle_on_saturated_vapour_at_1_MPa(self):
        """Saturated vapour at P_flash flashes to itself: same work."""
        pp = _merge_params(None)
        h_g = CP.PropsSI('H', 'P', 1e6, 'Q', 1, 'Water')
        r = _flash_cycle(10.0, h_g, 1.0, pp)
        assert r['success'], r
        assert dry_steam_specific_work(1.0) == pytest.approx(
            r['power_MWe'] / 10.0, rel=1e-9), \
            'specific work equals the flash cycle output per kg'

    def test_honours_the_turbine_parameters(self):
        """A lower dry efficiency and a higher condenser pressure cut work."""
        base = dry_steam_specific_work(5.0)
        assert dry_steam_specific_work(5.0, {'eta_turbine_dry': 0.7}) < base
        assert dry_steam_specific_work(5.0, {'P_condenser_MPa': 0.05}) < base
