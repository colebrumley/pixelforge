"""tests/oracle/ stays reproducible: re-recording it gives the committed bytes."""

import json

import record_oracles

ORACLE_DIR = record_oracles.DEFAULT_DIR
MAX_BYTES = 2 * 1024 * 1024


def _files(root):
    return {p.relative_to(root).as_posix(): p for p in sorted(root.rglob("*")) if p.is_file()}


def test_oracle_is_reproducible(tmp_path):
    out = tmp_path / "oracle"
    record_oracles.record(out)
    committed, recorded = _files(ORACLE_DIR), _files(out)
    assert sorted(recorded) == sorted(committed)
    differing = [name for name, path in recorded.items()
                 if path.read_bytes() != committed[name].read_bytes()]
    assert not differing, (f"re-recorded oracle files differ: {differing}; run "
                           "scripts/record_oracles.py and review the diff")


def test_oracle_size_budget():
    total = sum(p.stat().st_size for p in _files(ORACLE_DIR).values())
    assert total < MAX_BYTES, f"tests/oracle/ is {total} bytes (budget {MAX_BYTES})"


def test_oracle_json_round_trips_floats():
    for path in _files(ORACLE_DIR).values():
        if path.suffix == ".json":
            json.loads(path.read_text(encoding="ascii"))
    reprs = json.loads((ORACLE_DIR / "float_repr.json").read_text())["repr"]
    assert len(reprs) == len(set(reprs)) == 10_000
    assert all(repr(float(r)) == r for r in reprs)
