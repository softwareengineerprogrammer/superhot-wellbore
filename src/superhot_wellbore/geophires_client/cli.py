# -*- coding: utf-8 -*-
"""
Command-Line Client
====================

File-based integration path: solve a superhot scenario and write the
files GEOPHIRES reads, without either program importing the other.

    superhot-geophires template > case.json
    superhot-geophires run --request case.json --output-dir out
    superhot-geophires run --set reservoir.T_reservoir_C=500
    superhot-geophires steady --set operating.target_whp_MPa=12

The same commands are available as
'python -m superhot_wellbore.geophires_client ...'.

The 'run' command writes three files into the output directory: the
GEOPHIRES temperature profile, a GEOPHIRES input deck fragment and
the full JSON result (see export.py).

Author: superhot-wellbore GEOPHIRES client
"""

import argparse
import json
import os
import sys

from . import export
from .client import SuperhotWellboreClient
from .config import SuperhotRequest
from .results import PROFILE_TEMPERATURES

DEFAULT_OUTPUT_DIRECTORY = 'superhot_geophires'


# ====================================================================
# REQUEST ASSEMBLY
# ====================================================================

def _load_request(path):
    """Load a request from a JSON file, or from stdin when path is -."""
    if path is None:
        return SuperhotRequest()
    if path == '-':
        return SuperhotRequest.from_dict(json.load(sys.stdin))
    with open(path, encoding='utf-8') as handle:
        return SuperhotRequest.from_dict(json.load(handle))


def _parse_value(text):
    """Parse an override value as JSON, falling back to a string."""
    try:
        return json.loads(text)
    except ValueError:
        return text


def _apply_overrides(request, assignments):
    """
    Apply 'section.key=value' overrides to a request.

    Values are parsed as JSON, so numbers, booleans, null and nested
    structures all work; anything else is kept as a string.
    """
    for assignment in assignments or ():
        if '=' not in assignment:
            raise ValueError(
                f'Malformed override {assignment!r}, expected '
                f'section.key=value')
        target, text = assignment.split('=', 1)
        value = _parse_value(text)

        if '.' not in target:
            if target != 'name':
                raise ValueError(
                    f'Malformed override target {target!r}, expected '
                    f'section.key or name')
            request.name = str(value)
            continue

        section_name, key = target.split('.', 1)
        if not hasattr(request, section_name):
            raise ValueError(f'Unknown request section '
                             f'{section_name!r}')
        section = getattr(request, section_name)
        if not hasattr(section, key):
            raise ValueError(f'Unknown key {key!r} in section '
                             f'{section_name!r}')
        setattr(section, key, value)
    return request


def _build_request(arguments):
    """Load, override and validate the request for a command."""
    request = _load_request(getattr(arguments, 'request', None))
    _apply_overrides(request, getattr(arguments, 'set', None))
    if getattr(arguments, 'name', None):
        request.name = arguments.name
    if getattr(arguments, 'verbose', False):
        request.solver.verbose = True
    return request.validate()


# ====================================================================
# COMMANDS
# ====================================================================

def command_template(arguments):
    """Print or write an example request."""
    example = SuperhotRequest.from_dict({
        'name': 'superhot_example',
        'reservoir': {
            'P_reservoir_MPa': 30.0,
            'T_reservoir_C': 450.0,
            'transmissivity_md_m': 1000.0,
            'drainage_radius_m': 500.0,
        },
        'well': {
            'depth_m': None,
            'diameter_m': 0.217,
            'delta_z_m': 10.0,
        },
        'rock_temperature': {'mode': 'linear', 'T_surface_C': 10.0},
        'operating': {'control': 'whp', 'target_whp_MPa': 10.0},
        'decline': {
            'temperature_mode': 'linear_percent',
            'temperature_rate_per_year': 0.5,
            'pressure_mode': 'linear_percent',
            'pressure_rate_per_year': 0.5,
        },
        'time': {'plant_lifetime_yr': 30, 'timesteps_per_year': 4},
        'solver': {'max_solve_points': 8},
    })

    text = json.dumps(example.to_dict(), indent=2)
    if arguments.output:
        export.write_request_json(example, arguments.output)
        print(f'Wrote {arguments.output}')
    else:
        print(text)
    return 0


def command_run(arguments):
    """Solve a production history and write the GEOPHIRES files."""
    request = _build_request(arguments)
    client = SuperhotWellboreClient(request)

    print(f'Solving "{request.name}": '
          f'P_res = {request.reservoir.P_reservoir_MPa:g} MPa, '
          f'T_res = {request.reservoir.T_reservoir_C:g} C, '
          f'depth = {client.depth_m:g} m', file=sys.stderr)

    profile = client.solve_profile()

    if arguments.stdout:
        print(json.dumps(profile.to_dict(), indent=2))
        return 0 if profile.initial is not None else 1

    if profile.initial is None:
        print('No usable solution was found:', file=sys.stderr)
        for note in profile.notes:
            print(f'  - {note}', file=sys.stderr)
        return 1

    paths = export.export_all(
        profile, arguments.output_dir, name=request.name,
        profile_temperature=arguments.profile_temperature)

    _print_summary(profile)
    print('')
    for label, path in paths.items():
        print(f'{label}: {path}')
    print('')
    print('Run GEOPHIRES with the generated deck fragment, for '
          'example:')
    print(f'  cd {arguments.output_dir} && python -m geophires_x '
          f'{os.path.basename(paths["geophires_input"])}')
    return 0


def command_steady(arguments):
    """Solve the initial steady state only."""
    request = _build_request(arguments)
    client = SuperhotWellboreClient(request)
    result = client.solve_steady_state()
    print(json.dumps(result.to_dict(), indent=2))
    return 0 if result.success else 1


def _print_summary(profile):
    """Print the headline numbers of a profile."""
    print('')
    print('Superhot production history')
    print('-' * 60)
    for key, value in profile.summary().items():
        if isinstance(value, float):
            print(f'  {key:42s} {value:12.4g}')
        else:
            print(f'  {key:42s} {value!s:>12}')
    if profile.notes:
        print('')
        print('Notes')
        print('-' * 60)
        for note in profile.notes:
            print(f'  - {note}')


# ====================================================================
# ARGUMENT PARSING
# ====================================================================

def build_parser():
    """Build the argument parser of the command-line client."""
    parser = argparse.ArgumentParser(
        prog='superhot-geophires',
        description='Solve a superhot single-well scenario with the '
                    'superhot-wellbore model and export it for '
                    'GEOPHIRES.')
    subparsers = parser.add_subparsers(dest='command')

    def add_request_arguments(subparser):
        subparser.add_argument(
            '--request', metavar='FILE',
            help='JSON request file, or - to read stdin. Defaults to '
                 'the built-in defaults.')
        subparser.add_argument(
            '--set', action='append', metavar='SECTION.KEY=VALUE',
            help='Override a request value, e.g. '
                 '--set reservoir.T_reservoir_C=500. Repeatable.')
        subparser.add_argument('--name', help='Scenario name.')
        subparser.add_argument(
            '--verbose', action='store_true',
            help='Print solver progress.')

    template = subparsers.add_parser(
        'template', help='Print an example request in JSON.')
    template.add_argument('--output', metavar='FILE',
                          help='Write the example to a file.')
    template.set_defaults(handler=command_template)

    run = subparsers.add_parser(
        'run', help='Solve a production history and write the '
                    'GEOPHIRES files.')
    add_request_arguments(run)
    run.add_argument('--output-dir', metavar='DIR',
                     default=DEFAULT_OUTPUT_DIRECTORY,
                     help=f'Output directory (default '
                          f'{DEFAULT_OUTPUT_DIRECTORY}).')
    run.add_argument('--profile-temperature',
                     choices=PROFILE_TEMPERATURES, default='wellhead',
                     help='Temperature written into the GEOPHIRES '
                          'profile (default wellhead).')
    run.add_argument('--stdout', action='store_true',
                     help='Print the JSON result instead of writing '
                          'files.')
    run.set_defaults(handler=command_run)

    steady = subparsers.add_parser(
        'steady', help='Solve the initial steady state only.')
    add_request_arguments(steady)
    steady.set_defaults(handler=command_steady)

    return parser


def main(argv=None):
    """Entry point of the command-line client."""
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if getattr(arguments, 'handler', None) is None:
        parser.print_help()
        return 2
    try:
        return arguments.handler(arguments)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 1
