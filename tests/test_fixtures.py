import make_fixtures   # scripts/ is on sys.path (conftest.py)
import numpy as np
import pytest
from PIL import Image


@pytest.fixture(scope="module")
def regenerated(tmp_path_factory):
    outdir = tmp_path_factory.mktemp("fixtures")
    make_fixtures.main(outdir)
    return outdir


@pytest.mark.parametrize("name", make_fixtures.FIXTURES)
def test_committed_fixtures_match_make_fixtures(fixture_path, regenerated, name):
    """The committed PNGs are what scripts/make_fixtures.py draws (pixels, not bytes)."""
    fresh_path = regenerated / f"{name}.png"
    with Image.open(fixture_path(name)) as committed, Image.open(fresh_path) as fresh:
        assert committed.mode == fresh.mode and committed.size == fresh.size
        assert np.array_equal(np.asarray(committed), np.asarray(fresh)), (
            f"tests/fixtures/{name}.png differs from scripts/make_fixtures.py")
