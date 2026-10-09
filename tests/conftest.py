import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"
sys.path.insert(0, str(ROOT / "scripts"))

import make_fixtures  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _fixtures():
    """Generate the synthetic fixtures if they are not on disk yet."""
    if any(not (FIXTURES / f"{name}.png").is_file() for name in make_fixtures.FIXTURES):
        make_fixtures.main(FIXTURES)


@pytest.fixture(scope="session")
def fixture_path():
    return lambda name: FIXTURES / f"{name}.png"


@pytest.fixture(scope="session")
def preprocessed(fixture_path):
    """Load + preprocess a fixture: returns (Preprocessed, Config)."""
    from pixelforge import Config, io, preprocess

    def load(name, **overrides):
        config = Config(**overrides)
        loaded = io.load(fixture_path(name))
        return preprocess.run(loaded.rgb, loaded.alpha, config), config

    return load
