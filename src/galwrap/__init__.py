"""GalWrap.
"""

# Imports


from importlib import resources
from pathlib import Path
import yaml

from . import utils


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


## Dictionaries


PATH_NAMES = yaml.safe_load(open(DATA_ROOT / "path_names.yaml"))
"""Dict of standardized path names and recognized alternative names.
"""
