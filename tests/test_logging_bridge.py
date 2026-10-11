"""Rust `log` records reach Python's `logging` as children of the `pixelforge` logger."""

import logging

from pixelforge import _native


def test_record_is_forwarded(caplog):
    with caplog.at_level("DEBUG", logger="pixelforge"):
        _native.log_probe("pixelforge::probe", "debug", "kopf iter 1/3 probe")
    records = [r for r in caplog.records if r.name == "pixelforge.probe"]
    assert len(records) == 1
    assert records[0].levelno == logging.DEBUG
    assert records[0].getMessage() == "kopf iter 1/3 probe"


def test_disabled_level_is_dropped(caplog):
    with caplog.at_level("WARNING", logger="pixelforge"):
        _native.log_probe("pixelforge::probe", "info", "quiet")
        _native.log_probe("pixelforge::probe", "warning", "loud")
    messages = [r.getMessage() for r in caplog.records if r.name == "pixelforge.probe"]
    assert messages == ["loud"]
