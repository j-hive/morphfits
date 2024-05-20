"""Modules and subpackages for the MorphFITS program.
"""

# Imports


from importlib import resources
from pathlib import Path
import yaml


# Constants


## Paths


ROOT = Path(resources.files(__package__) / ".." / "..").resolve()
"""Path to root directory of this repository.
"""


CONFIG_ROOT = ROOT / "config"
"""Path to root of the configuration directory.
"""


DATA_ROOT = ROOT / "data"
"""Path to root of the data standards directory.
"""


LOG_ROOT = ROOT / "logs"
"""Path to root of the logs directory.
"""


## Dictionaries


PATH_NAMES = yaml.safe_load(open(DATA_ROOT / "galwrap_path_names.yaml"))
"""Dict of standardized path names and recognized alternative names.
"""
