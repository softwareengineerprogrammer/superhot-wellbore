# -*- coding: utf-8 -*-
"""
Exported Files
==============

The exported files are the actual hand-off to GEOPHIRES: a
temperature profile it reads back, an input deck that points at that
profile and a JSON record of the run. GEOPHIRES parses these by
position and by spelling, so a stray column, a missing row or a
Python-style boolean breaks the run far away from here. These tests
read the written files back and check their shape.

Author: superhot-wellbore GEOPHIRES client
"""

import json
import os

from superhot_wellbore.client import export


def test_temperature_profile_file(synthetic_profile, tmp_path):
    """The profile has one time-temperature row per timestep."""
    paths = export.export_all(synthetic_profile, str(tmp_path))

    with open(paths['temperature_profile'], encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    data_lines = [line for line in lines
                  if not line.startswith('#')]
    assert len(data_lines) == synthetic_profile.n_timesteps, \
        'one profile row per timestep'

    first = data_lines[0].split(',')
    assert (len(first) == 2
            and float(first[0]) == 0.0
            and abs(float(first[1]) - 317.0) < 1e-6), \
        'profile row is time and temperature'


def test_geophires_input_deck(synthetic_profile, tmp_path):
    """The deck selects the profile model and finds its file."""
    paths = export.export_all(synthetic_profile, str(tmp_path))

    with open(paths['geophires_input'], encoding='utf-8') as f:
        deck = [line.strip() for line in f if line.strip()]
    parameters = {}
    for line in deck:
        if line.startswith(('#', '--', '*')):
            continue
        fields = line.split(',')
        if len(fields) >= 2:
            parameters[fields[0].strip()] = fields[1].strip()

    assert parameters.get('Reservoir Model') == '5', \
        'deck selects the profile reservoir model'
    assert parameters.get('Reservoir Output File Name') == \
        os.path.basename(paths['temperature_profile']), \
        'deck points at the profile file'
    assert parameters.get('Ramey Production Wellbore Model') == 'False', \
        'deck writes booleans GEOPHIRES understands'


def test_profile_json(synthetic_profile, tmp_path):
    """The JSON record keeps the request and every timestep."""
    paths = export.export_all(synthetic_profile, str(tmp_path))

    with open(paths['profile_json'], encoding='utf-8') as f:
        document = json.load(f)

    assert document['request']['name'] == 'synthetic', \
        'JSON result carries the request'
    assert len(document['timesteps']) == synthetic_profile.n_timesteps, \
        'JSON result carries every timestep'
    assert len(document['series']['production_temperature_C']) == \
        synthetic_profile.n_timesteps, 'JSON result carries the series'
