"""A morphology fitting data pipeline for JWST data, by J-HIVE.

[Documentation WIP]

References
----------
1. [J-HIVE](https://j-hive.org/)
2. [J-HIVE Organization Page](https://github.com/j-hive)
3. [MorphFITS Repository](https://github.com/j-hive/morphfits)
"""

# Imports


from importlib import resources
from pathlib import Path


# Constants


ROOT = Path(resources.files(__package__) / ".." / "..").resolve()
"""Path to root directory of MorphFITS repository.
"""


DATA_ROOT = ROOT / "data"
"""Path to root MorphFITS data standards directory.
"""


EXAMPLES_ROOT = ROOT / "examples"
"""Path to root MorphFITS examples directory.
"""
