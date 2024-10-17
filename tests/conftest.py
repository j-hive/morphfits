"""Base pytest configuration settings and fixtures.
"""

# Imports


from pathlib import Path
import pytest

from morphfits import config


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
def configuration(test_data_path: Path):
    morphfits_config = config.create_config(
        config_path=test_data_path / "config.yaml", download=True
    )
    return morphfits_config
