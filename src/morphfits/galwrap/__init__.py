"""Modules for GalWrap, a morphology fitter program designed on wrapping GALFIT.
"""

# Imports


import yaml

from .. import ROOT, DATA_ROOT


# Constants


## Paths


GALWRAP_DATA_ROOT = DATA_ROOT / "galwrap"
"""Path to root directory of GalWrap data standards.
"""


## Dictionaries


GALWRAP_PATH_NAMES = yaml.safe_load(open(GALWRAP_DATA_ROOT / "paths.yaml"))
"""Dict of standardized path names and recognized alternative names.
"""
