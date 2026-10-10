"""Command line interface (Section 10)."""

from __future__ import annotations

import functools
import json
import math
import os
import sys
import traceback
from dataclasses import fields
from pathlib import Path

import click
import numpy as np

from . import color, io, palette, pipeline, postprocess, preprocess
from .config import METHODS, PRESET_LONGEST_EDGE, PRESETS, Config
from .errors import ConfigError

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff")
BATCH_PIXELS_PER_IMAGE = 50_000
SHARED_PALETTE_NAME = "shared_palette.hex"
OUTPUT_STEM_SUFFIXES = ("_preview", "_tileset", "_compare")   # our own image outputs
MAX_CONFIG_FILE_BYTES = 1024 * 1024

_CLICK_TYPES = {"int": int, "float": float, "str": str}


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
    command = click.option(
        "--max-input-pixels", "max_input_pixels", type=click.IntRange(min=1), default=None,
        help=f"Refuse inputs with more pixels than this [default: {io.MAX_INPUT_PIXELS}].",
    )(command)
    return click.option("--config", "config_path", type=click.Path(dir_okay=False),
                        default=None, help="JSON file of Config fields; flags override it.")(command)


def build_config(config_path, flags: dict) -> Config:
    """defaults ← preset ← --config JSON ← explicit flags."""
    values = {}
    try:
        if config_path is not None:
            loaded = json.loads(_read_config_text(config_path),
                                parse_constant=_reject_json_constant)
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
        config.validate_palette()
    except (ValueError, OSError) as exc:
        raise ConfigError(str(exc)) from exc
    return config


def _reject_json_constant(name: str):
    raise ValueError(f"config JSON must not contain {name}")


def _read_config_text(config_path) -> str:
    """A regular file of at most MAX_CONFIG_FILE_BYTES, decoded as UTF-8."""
    try:
        with io.open_regular(config_path) as f:
            data = f.read(MAX_CONFIG_FILE_BYTES + 1)
    except preprocess.PixelforgeError as exc:
        raise ValueError(str(exc)) from None
    if len(data) > MAX_CONFIG_FILE_BYTES:
        raise ValueError(f"{config_path} is larger than {MAX_CONFIG_FILE_BYTES} bytes")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError(f"{config_path} is not a UTF-8 text file") from None


def _debug() -> bool:
    ctx = click.get_current_context(silent=True)
    flag = ctx is not None and ctx.find_root().params.get("debug", False)
    return bool(flag) or os.environ.get("PIXELFORGE_DEBUG", "") not in ("", "0")


def handle_errors(func):
    """Exit 2 on config validation errors, 1 on anything else; message on stderr.

    With --debug or PIXELFORGE_DEBUG=1 the full traceback is printed instead (same exit code).
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except click.exceptions.Exit:
            raise
        except Exception as exc:  # noqa: BLE001 - the CLI reports every failure the same way
            prefix = "config error" if isinstance(exc, ConfigError) else "error"
            message = traceback.format_exc().rstrip() if _debug() else str(exc)
            click.echo(f"{prefix}: {message or type(exc).__name__}", err=True)
            sys.exit(2 if isinstance(exc, ConfigError) else 1)
    return wrapper


force_option = click.option("--force", is_flag=True, default=False,
                            help="Allow outputs that overwrite an input file.")


def check_collisions(inputs: list[Path], outputs: list[Path], force: bool) -> None:
    """ConfigError if any output path resolves to an input file (unless force)."""
    if force:
        return
    resolved = {Path(p).resolve(): Path(p) for p in inputs if Path(p).is_file()}
    hits = sorted(f"{out} would overwrite input {resolved[Path(out).resolve()]}"
                  for out in outputs if Path(out).resolve() in resolved)
    if hits:
        raise ConfigError("; ".join(hits) + " (use another --outdir, or --force)")


def _emit(payload: dict) -> None:
    click.echo(json.dumps(payload, sort_keys=True, separators=(",", ":")))


@click.group()
@click.version_option(package_name="pixelforge")
@click.option("--debug", is_flag=True, default=False,
              help="Print full tracebacks on errors (also: PIXELFORGE_DEBUG=1).")
def cli(debug):
    """Deterministic conversion of images into 16-bit-style pixel art."""


@cli.command()
@click.argument("input_path", metavar="INPUT", type=click.Path(dir_okay=False))
@click.option("-o", "--outdir", type=click.Path(file_okay=False), default="out",
              show_default=True)
@force_option
@config_options
@handle_errors
def convert(input_path, outdir, force, config_path, max_input_pixels, **flags):
    """Convert one image."""
    config = build_config(config_path, flags)
    prefix = Path(outdir) / Path(input_path).stem
    check_collisions([Path(input_path)],
                     list(pipeline.output_paths(prefix, config.tileset).values()), force)
    result = pipeline.run(input_path, config, max_input_pixels)
    outputs = result.save(prefix)
    _emit({"outputs": outputs, "stats": result.stats})


def shared_palette(paths: list[Path], config: Config,
                   max_pixels: int | None = None) -> np.ndarray:
    """One palette (K, 3) uint8 from the union of all images' preprocessed LAB pixels."""
    samples = []
    for path in paths:
        loaded = io.load(path, io.MAX_INPUT_PIXELS if max_pixels is None else max_pixels)
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


def shared_scale(paths: list[Path], config: Config,
                 max_pixels: int | None = None) -> Config:
    """`config` with one `scale` and `canvas` for every frame, unless they are pinned.

    The union (largest width, largest height) of the frames' subject boxes is mapped to the
    preset's longest edge (or to the one given out_width/out_height), outline margin inside,
    so every frame keeps the same input-to-output ratio and all share one canvas.
    """
    pinned = config.out_width is not None and config.out_height is not None
    if config.scale is not None or config.canvas is not None or pinned:
        return config
    width = height = 0
    for path in paths:
        loaded = io.load(path, io.MAX_INPUT_PIXELS if max_pixels is None else max_pixels)
        y0, y1, x0, x1 = preprocess.subject_bbox(loaded.rgb, loaded.alpha, config)
        width, height = max(width, x1 - x0), max(height, y1 - y0)
    m2 = 2 * preprocess.outline_margin(config)
    if config.out_width is not None:
        scale = width / (config.out_width - m2)
    elif config.out_height is not None:
        scale = height / (config.out_height - m2)
    else:
        scale = max(width, height) / (PRESET_LONGEST_EDGE[config.preset] - m2)
    canvas = [max(8, round(v / scale) + m2) for v in (width, height)]
    if config.tileset:
        canvas = [math.ceil(v / config.tile_size) * config.tile_size for v in canvas]
    return config.replace(out_width=None, out_height=None, scale=scale,
                          canvas=f"{canvas[0]}x{canvas[1]}")


@cli.command()
@click.argument("input_dir", type=click.Path(file_okay=False))
@click.option("-o", "--outdir", type=click.Path(file_okay=False), default="out",
              show_default=True)
@force_option
@config_options
@handle_errors
def batch(input_dir, outdir, force, config_path, max_input_pixels, **flags):
    """Convert every image in a directory with ONE shared palette and ONE scale."""
    config = build_config(config_path, flags)
    paths = []
    for p in sorted(Path(input_dir).iterdir()):
        if not (p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES):
            continue
        if p.stem.endswith(OUTPUT_STEM_SUFFIXES):
            click.echo(f"skipping {p}: looks like a pixelforge output", err=True)
            continue
        paths.append(p)
    if not paths:
        raise RuntimeError(f"no images found in {input_dir}")
    outdir = Path(outdir)
    by_stem: dict[str, list[Path]] = {}
    for p in paths:
        by_stem.setdefault(p.stem, []).append(p)
    clashes = [", ".join(str(p) for p in group) for group in by_stem.values() if len(group) > 1]
    if clashes and not force:
        raise ConfigError("inputs share an output name (later file would win): "
                          + "; ".join(clashes) + " (rename them, or --force)")
    outputs = [outdir / SHARED_PALETTE_NAME] if config.palette_name is None else []
    for stem in by_stem:
        outputs += pipeline.output_paths(outdir / stem, config.tileset).values()
    check_collisions(paths, outputs, force)
    outdir.mkdir(parents=True, exist_ok=True)
    header = {}
    # DEVIATION: Section 10 — one scale and canvas for the whole set (not per frame), so that
    # animation frames cropped to their own alpha keep a constant size.
    scaled = shared_scale(paths, config, max_input_pixels)
    if scaled is not config:
        config = scaled
        header.update(scale=config.scale, canvas=config.canvas)
    if config.palette_name is None:
        palette_path = outdir / SHARED_PALETTE_NAME
        shared = shared_palette(paths, config, max_input_pixels)
        palette.write_hex(palette_path, shared)
        # The colors themselves, not the path: frame metadata must not depend on --outdir.
        config = config.replace(palette_name=palette.inline_name(shared))
        header["shared_palette"] = str(palette_path)
    if header:
        _emit(header)
    for path in paths:
        result = pipeline.run(path, config, max_input_pixels)
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
@force_option
@config_options
@handle_errors
def compare(input_path, outdir, force, config_path, max_input_pixels, **flags):
    """Run box, kopf and gerstner with the same config; write a side-by-side PNG."""
    from PIL import Image

    flags.pop("method", None)
    base = build_config(config_path, flags)
    stem = Path(input_path).stem
    outputs = [Path(outdir) / f"{stem}_compare.png"]
    for method in METHODS:
        outputs += pipeline.output_paths(Path(outdir) / f"{stem}_{method}", base.tileset).values()
    check_collisions([Path(input_path)], outputs, force)
    previews, report = [], {}
    for method in METHODS:
        config = base.replace(method=method)
        result = pipeline.run(input_path, config, max_input_pixels)
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
