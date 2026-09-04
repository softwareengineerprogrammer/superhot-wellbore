# -*- coding: utf-8 -*-
"""
Access to the superhot-wellbore Core Modules
=============================================

The superhot-wellbore repository is a flat collection of top-level
modules (reservoir.py, wellbore_physics.py, power_cycle.py) rather
than an installed package. This module locates them once, so that the
client works both when it is run from the repository root and when it
is imported from a program living elsewhere, such as GEOPHIRES.

Importing the client package therefore never requires the caller to
manipulate sys.path.
"""

import importlib
import importlib.util
import os
import sys

#: Repository root, i.e. the directory holding reservoir.py
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ensure_importable():
    """Add the repository root to sys.path if the core modules need it."""
    try:
        found = importlib.util.find_spec('reservoir') is not None
    except (ImportError, ValueError):
        found = False
    if not found and REPO_ROOT not in sys.path:
        sys.path.append(REPO_ROOT)


_ensure_importable()

try:
    reservoir = importlib.import_module('reservoir')
    wellbore_physics = importlib.import_module('wellbore_physics')
    power_cycle = importlib.import_module('power_cycle')
except ImportError as exc:  # pragma: no cover - environment problem
    raise ImportError(
        'Could not import the superhot-wellbore core modules '
        f'(reservoir, wellbore_physics, power_cycle) from {REPO_ROOT}. '
        'Install the dependencies listed in requirements.txt and make '
        'sure the repository is intact.') from exc
