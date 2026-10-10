import importlib.metadata

import pytest

import pixelforge


def test_version_is_single_sourced():
    try:
        installed = importlib.metadata.version("pixelforge")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("pixelforge is not installed (no distribution metadata)")
    assert pixelforge.__version__ == installed
