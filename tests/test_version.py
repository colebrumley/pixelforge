import importlib.metadata
import re
from pathlib import Path

import pytest

import pixelforge


def test_version_is_single_sourced():
    try:
        installed = importlib.metadata.version("pixelforge")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("pixelforge is not installed (no distribution metadata)")
    assert pixelforge.__version__ == installed


def test_version_matches_pyproject_and_cargo():
    """pyproject.toml and the root Cargo.toml carry static copies of pixelforge/version.py."""
    root = Path(__file__).resolve().parent.parent
    source = re.search(r'^__version__ = "([^"]+)"$',
                       (root / "pixelforge" / "version.py").read_text(), re.M)
    project = re.search(r'^\[project\]$.*?^version = "([^"]+)"$',
                        (root / "pyproject.toml").read_text(), re.M | re.S)
    cargo = re.search(r'^\[workspace\.package\]$.*?^version = "([^"]+)"$',
                      (root / "Cargo.toml").read_text(), re.M | re.S)
    assert source and project and cargo
    assert source.group(1) == pixelforge.__version__
    assert project.group(1) == source.group(1)
    assert cargo.group(1) == source.group(1)
