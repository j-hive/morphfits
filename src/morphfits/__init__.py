"""Modules and subpackages for the MorphFITS program.
"""

# Imports


from importlib import resources
from pathlib import Path


# Constants


## Paths


ROOT = Path(resources.files(__package__) / ".." / "..").resolve()
"""Path to root directory of this repository.
"""


DATA_ROOT = ROOT / "data"
"""Path to root of the data standards directory.
"""


EXAMPLES_ROOT = ROOT / "examples"
"""Path to root of the examples directory.
"""


LOGS_ROOT = ROOT / "logs"
"""Path to root of the logs directory.
"""
