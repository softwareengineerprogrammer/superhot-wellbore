# -*- coding: utf-8 -*-
"""
Unit Conversions
================

GEOPHIRES speaks kPa, km, kJ/kg and inches while the coupled model
works in MPa, m, MJ/kg and metres. Every number that crosses that
boundary goes through :mod:`superhot_wellbore.geophires_client.units`,
so a wrong factor there is silent: the deck still runs, it just
describes a different well. These tests pin the factors down.

Author: superhot-wellbore GEOPHIRES client
"""

import pytest

from superhot_wellbore.geophires_client import units


@pytest.mark.parametrize('label, function, arguments, expected, tolerance', [
    ('MPa to kPa', units.mpa_to_kpa, (10.0,), 10000.0, 1e-9),
    ('kPa to MPa', units.kpa_to_mpa, (10000.0,), 10.0, 1e-9),
    ('m to km', units.m_to_km, (3500.0,), 3.5, 1e-9),
    ('MJ/kg to kJ/kg', units.mj_per_kg_to_kj_per_kg, (2.77,), 2770.0, 1e-9),
    ('m to inch', units.m_to_inch, (0.217,), 8.5433, 1e-4),
    ('gradient', units.gradient_C_per_km, (450.0, 10.0, 3500.0), 125.714, 1e-3),
])
def test_conversion(label, function, arguments, expected, tolerance):
    """Each conversion reproduces its hand-computed value."""
    actual = function(*arguments)
    assert actual == pytest.approx(expected, abs=tolerance), label


def test_none_passes_through():
    """A missing value stays missing instead of becoming a zero."""
    assert units.mpa_to_kpa(None) is None, 'None passes through'
