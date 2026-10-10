"""Command line interface (Section 10)."""

from __future__ import annotations

import functools
import json
import math
import sys
from dataclasses import fields
from pathlib import Path

import click
import numpy as np

from . import color, io, palette, pipeline, postprocess, preprocess
from .config import METHODS, PRESETS, Config

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff")
BATCH_PIXELS_PER_IMAGE = 50_000
SHARED_PALETTE_NAME = "shared_palette.hex"

_CLICK_TYPES = {"int": int, "float": float, "str": str}


class ConfigError(Exception):
    """Config validation failure → exit code 2."""


def config_options(command):
    """Expose every Config field as a --kebab-case flag (None = not given)."""
    for f in reversed(fields(Config)):
        flag = "--" + f.name.replace("_", "-")
        base = f.type.split("|")[0].strip()
        if f.name == "preset":
            option = click.option(flag, f.name, type=click.Choice(sorted(PRESETS)), default=None)
        elif f.name == "method":
            option = click.option(flag, f.name, type=click.Choice(METHODS), default=None)
        elif base == "bool":
            option = click.option(f"{flag}/--no-{flag[2:]}", f.name, default=None)
        else:
            option = click.option(flag, f.name, type=_CLICK_TYPES[base], default=None)
        command = option(command)
    return click.option("--config", "config_path", type=click.Path(dir_okay=False),
                        default=None, help="JSON file of Config fields; flags override it.")(command)


def build_config(config_path, flags: dict) -> Config:
    """defaults ← preset ← --config JSON ← explicit flags."""
    values = {}
    try:
        if config_path is not None:
            loaded = json.loads(Path(config_path).read_text())
            if not isinstance(loaded, dict):
                raise ValueError(f"{config_path} must contain a JSON object of Config fields")
            values.update(loaded)
        given = {name: value for name, value in flags.items() if value is not None}
        if "preset" in given and config_path is not None:
            shadowed = sorted(set(values) & set(PRESETS[given["preset"]]) - set(given))
            if shadowed:
                click.echo(f"warning: --preset {given['preset']} does not override "
                           f"{', '.join(shadowed)} from {config_path}; the JSON values win",
                           err=True)
        values.update(given)
        config = Config(**values)
        if config.palette_name is not None:
            palette.load_palette(config.palette_name)
    except (ValueError, OSError) as exc:
        raise ConfigError(str(exc)) from exc
    return config


def handle_errors(func):
    """Exit 2 on config validation errors, 1 on anything else; message on stderr."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ConfigError as exc:
            click.echo(f"config error: {exc}", err=True)
            sys.exit(2)
        except click.exceptions.Exit:
            raise
        except Exception as exc:  # noqa: BLE001 - the CLI reports every failure the same way
            click.echo(f"error: {exc}", err=True)
            sys.exit(1)
    return wrapper


def _emit(payload: dict) -> None:
    click.echo(json.dumps(payload, sort_keys=True, separators=(",", ":")))


@click.group()
@click.version_option(package_name="pixelforge")
def cli():
    """Deterministic conversion of images into 16-bit-style pixel art."""


@cli.command()
@click.argument("input_path", metavar="INPUT", type=click.Path(dir_okay=False))
@click.option("-o", "--outdir", type=click.Path(file_okay=False), default="out",
              show_default=True)
@config_options
@handle_errors
def convert(input_path, outdir, config_path, **flags):
    """Convert one image."""
    config = build_config(config_path, flags)
    result = pipeline.run(input_path, config)
    outputs = result.save(Path(outdir) / Path(input_path).stem)
    _emit({"outputs": outputs, "stats": result.stats})


def shared_palette(paths: list[Path], config: Config) -> np.ndarray:
    """One palette (K, 3) uint8 from the union of all images' preprocessed LAB pixels."""
    samples = []
    for path in paths:
        loaded = io.load(path)
        pre = preprocess.run(loaded.rgb, loaded.alpha, config)
        pixels = pre.lab[pre.mask]
        stride = max(1, math.ceil(len(pixels) / BATCH_PIXELS_PER_IMAGE))   # fixed, not random
        samples.append(pixels[::stride])
    palette_lab = palette.build(np.concatenate(samples), config)
    # A named palette is used as-is by the pipeline, so finish it here once.
    if config.palette_ramps:
        palette_lab = palette.regularize_ramps(palette_lab)
    palette_lab = postprocess.apply_saturation(palette_lab, config.saturation_beta)
    return color.lab_to_rgb8(palette_lab)


@cli.command()
@click.argument("input_dir", type=click.Path(file_okay=False))
@click.option("-o", "--outdir", type=click.Path(file_okay=False), default="out",
              show_default=True)
@config_options
@handle_errors
def batch(input_dir, outdir, config_path, **flags):
    """Convert every image in a directory with ONE shared palette."""
    config = build_config(config_path, flags)
    paths = sorted(p for p in Path(input_dir).iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise RuntimeError(f"no images found in {input_dir}")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if config.palette_name is None:
        palette_path = outdir / SHARED_PALETTE_NAME
        palette.write_hex(palette_path, shared_palette(paths, config))
        config = config.replace(palette_name=str(palette_path))
        _emit({"shared_palette": str(palette_path)})
    for path in paths:
        result = pipeline.run(path, config)
        _emit({"input": str(path), "outputs": result.save(outdir / path.stem),
               "stats": result.stats})


def side_by_side(images: list[np.ndarray], gap: int) -> np.ndarray:
    """RGBA images in a row on a transparent canvas."""
    height = max(im.shape[0] for im in images)
    width = sum(im.shape[1] for im in images) + gap * (len(images) - 1)
    sheet = np.zeros((height, width, 4), dtype=np.uint8)
    x = 0
    for im in images:
        sheet[:im.shape[0], x:x + im.shape[1]] = im
        x += im.shape[1] + gap
    return sheet


@cli.command()
@click.argument("input_path", metavar="INPUT", type=click.Path(dir_okay=False))
@click.option("-o", "--outdir", type=click.Path(file_okay=False), default="out",
              show_default=True)
@config_options
@handle_errors
def compare(input_path, outdir, config_path, **flags):
    """Run box, kopf and gerstner with the same config; write a side-by-side PNG."""
    from PIL import Image

    flags.pop("method", None)
    base = build_config(config_path, flags)
    stem = Path(input_path).stem
    previews, report = [], {}
    for method in METHODS:
        config = base.replace(method=method)
        result = pipeline.run(input_path, config)
        report[method] = {"outputs": result.save(Path(outdir) / f"{stem}_{method}"),
                          "stats": result.stats}
        previews.append(io.upscale_nearest(result.image, config.scale_preview))
    sheet_path = Path(outdir) / f"{stem}_compare.png"
    Image.fromarray(side_by_side(previews, gap=base.scale_preview), mode="RGBA").save(sheet_path)
    _emit({"compare": str(sheet_path), "methods": report})


@cli.command()
def palettes():
    """List the bundled palettes."""
    for name in palette.bundled_palettes():
        click.echo(f"{name}\t{len(palette.load_palette(name))} colors")


def main():
    cli()


if __name__ == "__main__":
    main()
