"""Configure settings and fixtures for pytest.
"""

# Imports


from pathlib import Path
import pytest

from morphfits.galwrap.setup import config


# Fixtures


## Path


@pytest.fixture
def test_root() -> Path:
    return (Path(__file__).parent).resolve()


@pytest.fixture
def test_data_path(test_root: Path) -> Path:
    return test_root / "test-data"


## Object


@pytest.fixture
def galwrap_config(test_data_path: Path):
    return config.create_config(test_data_path / "config.yaml")
