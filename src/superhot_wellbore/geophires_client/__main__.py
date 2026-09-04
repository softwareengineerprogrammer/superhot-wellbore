# -*- coding: utf-8 -*-
"""
Module entry point: python -m superhot_wellbore.geophires_client
"""

import sys

from .cli import main

if __name__ == '__main__':
    sys.exit(main())
