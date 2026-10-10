"""Output regression guard: pixels per fixture/method/preset against tests/golden.json."""

import json

import pytest
import update_golden   # scripts/ is on sys.path (conftest.py)

GOLDEN = json.loads(update_golden.GOLDEN_PATH.read_text())
# kopf on the 256-px fixtures takes 2-5 s per case.
_SLOW = {f"{fixture}/kopf/{preset}" for fixture in ("circle_alpha", "gradient6")
         for preset in update_golden.GOLDEN_PRESETS}


def test_golden_file_lists_every_case():
    assert sorted(GOLDEN) == sorted(update_golden.golden_keys())
    assert update_golden.GOLDEN_PATH.read_text() == update_golden.dumps(GOLDEN)


@pytest.mark.parametrize("key", [pytest.param(k, marks=pytest.mark.slow) if k in _SLOW else k
                                 for k in update_golden.golden_keys()])
def test_output_matches_golden(key):
    actual = update_golden.compute_entry(*key.split("/"))
    expected = GOLDEN[key]
    hint = ("If this output change is intentional, run `uv run python scripts/update_golden.py` "
            "and commit tests/golden.json with the change.")
    assert actual["config_hash"] == expected["config_hash"], (
        f"{key}: config hash changed (a default, a preset or the version changed). {hint}")
    assert actual["pixels_sha256"] == expected["pixels_sha256"], (
        f"{key}: output pixels changed. {hint}")
