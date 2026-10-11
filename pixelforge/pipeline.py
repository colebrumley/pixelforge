"""Pipeline orchestration (Section 5). The only module the CLI calls.

The pipeline runs natively (`pixelforge._native`, from crates/pixelforge/src/pipeline.rs):
every stage, the stats, the stage log lines (logger ``pixelforge.pipeline``) and the output
files ``Result.save`` writes. This module converts the arguments and results, and keeps the
two things that need Python: the ``rembg`` package for ``remove_bg`` (the native library
runs an external ``rembg`` command instead) and the Python library versions recorded in
``_meta.json``. The former Python orchestration is kept as tests/pipeline_reference.py.
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import _native, io, preprocess
from .config import Config

LAB_DECIMALS = 6   # precision of the LAB values in _palette.json


@dataclass
class Result:
    image: np.ndarray                 # (H, W, 4) uint8 RGBA at native pixel-art resolution
    palette: np.ndarray               # (K, 3) uint8
    indices: np.ndarray               # (H, W) int, -1 for transparent
    config: Config
    input_sha256: str
    palette_lab: np.ndarray           # (K, 3) float64
    stats: dict = field(default_factory=dict)
    tileset: np.ndarray | None = None  # (h, w, 4) uint8 RGBA tileset sheet
    tilemap: dict | None = None
    tileset_indices: np.ndarray | None = None

    def png_text(self) -> dict:
        text = {io.META_CONFIG_KEY: self.config.canonical_json(),
                io.META_INPUT_KEY: self.input_sha256}
        provenance = self.config.provenance()
        if provenance["palette"] is not None:
            text[io.META_PALETTE_SHA_KEY] = provenance["palette_sha256"]
            text[io.META_PALETTE_KEY] = "\n".join(provenance["palette"])
        return text

    def save(self, prefix) -> dict:
        """Write <prefix>.png, _preview.png, _palette.json, _palette.hex, _meta.json and, for
        tilesets, _tileset.png, _tilemap.json, .tmj (Tiled) and _tilemap.csv. Returns
        {name: path}."""
        versions = None
        if self.config.remove_bg:
            # The matting model is outside the determinism contract; record what ran.
            versions = {name: package_version(name) for name in ("rembg", "onnxruntime")}
        tileset_indices = None
        if self.tilemap is not None:
            tileset_indices = np.ascontiguousarray(self.tileset_indices, dtype=np.int64)
        paths = _native.result_save(
            os.fspath(prefix), np.ascontiguousarray(self.indices, dtype=np.int64),
            _rgb8(self.palette),
            np.ascontiguousarray(self.palette_lab, dtype=np.float64).reshape(-1, 3),
            self.stats, self.config.to_dict(), self.input_sha256,
            tileset_indices=tileset_indices, tilemap=self.tilemap,
            environment=environment(), remove_bg_versions=versions,
            palette_lines=_palette_lines(self.config))
        return dict(paths)


def _rgb8(palette_rgb8) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(palette_rgb8, dtype=np.uint8).reshape(-1, 3))


def _palette_lines(config: Config) -> list[str] | None:
    """The Config's cached palette lines, if it has read its palette already."""
    cached = config.__dict__.get("_palette_lines")
    return None if cached is None else list(cached)


def environment() -> dict:
    """Interpreter, numerical library versions, platform and native thread count (which
    affects speed only); recorded, never hashed."""
    native = _native.environment()
    return {"python": platform.python_version(), "numpy": package_version("numpy"),
            "scipy": package_version("scipy"), "scikit_image": package_version("scikit-image"),
            "pillow": package_version("Pillow"),
            "platform": native["platform"], "threads": native["threads"]}


def package_version(name: str) -> str | None:
    """Installed version of a distribution, or None if it is not installed."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def output_paths(prefix, tileset: bool) -> dict:
    """{name: Path} of every file Result.save(prefix) writes, in write order."""
    return {name: Path(path)
            for name, path in _native.output_paths(os.fspath(prefix), bool(tileset)).items()}


def uses_gerstner_palette(config: Config) -> bool:
    """True when gerstner's own converged palette and indices are the final ones."""
    return _native.uses_gerstner_palette(config.to_dict())


def _result(out: dict, config: Config) -> Result:
    return Result(image=out["image"], palette=out["palette"], indices=out["indices"],
                  config=config, input_sha256=out["input_sha256"],
                  palette_lab=out["palette_lab"], stats=out["stats"], tileset=out["tileset"],
                  tilemap=out["tilemap"], tileset_indices=out["tileset_indices"])


def run(image, config: Config, max_pixels: int | None = None) -> Result:
    """Load and convert one image.

    ``image`` is a path (str or os.PathLike), a ``PIL.Image.Image``, or an (H, W, 3) /
    (H, W, 4) numpy array of uint8 or of float in [0, 1] (see ``io.from_image``). For paths
    ``input_sha256`` hashes the file bytes; for in-memory images it hashes shape and RGBA
    bytes. ``max_pixels`` defaults to ``io.MAX_INPUT_PIXELS`` and applies to every kind.
    """
    if isinstance(image, (str, os.PathLike)) and not config.remove_bg:
        out = _native.pipeline_run(os.fspath(image), config.to_dict(), max_pixels,
                                   palette_lines=_palette_lines(config))
        return _result(out, config)
    start = time.perf_counter()
    budget = io.MAX_INPUT_PIXELS if max_pixels is None else max_pixels
    if isinstance(image, (str, os.PathLike)):
        loaded = io.load(image, budget)
    else:
        loaded = io.from_image(image, budget)
    return run_loaded(loaded, config, {"load": time.perf_counter() - start})


def run_loaded(loaded: io.Loaded, config: Config, timings: dict | None = None) -> Result:
    """Convert an already loaded image (``io.load`` / ``io.from_image`` / ``io.from_rgba``)."""
    timings = dict(timings or {})
    alpha = loaded.alpha
    if config.remove_bg:
        # The rembg package runs here (the native library would run the `rembg` command);
        # its time counts towards "preprocess", as when preprocess.run called it.
        config.validate_palette()   # before any expensive stage
        start = time.perf_counter()
        alpha = preprocess.remove_background(loaded.rgb, loaded.alpha)
        timings["preprocess"] = time.perf_counter() - start
    out = _native.pipeline_run_loaded(
        np.ascontiguousarray(loaded.rgb, dtype=np.float64),
        np.ascontiguousarray(alpha, dtype=np.uint8), loaded.sha256, config.to_dict(),
        timings, alpha_matted=config.remove_bg, palette_lines=_palette_lines(config))
    return _result(out, config)
