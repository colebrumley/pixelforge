"""Pipeline orchestration (Section 5). The only module the CLI calls."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import color, downscale, io, palette, postprocess, preprocess, quantize, tiles
from .config import Config
from .version import __version__


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
        return {io.META_CONFIG_KEY: self.config.canonical_json(),
                io.META_INPUT_KEY: self.input_sha256}

    def save(self, prefix) -> dict:
        """Write <prefix>.png, _preview.png, _palette.json, _palette.hex, _meta.json and, for
        tilesets, _tileset.png, _tilemap.json, .tmj (Tiled) and _tilemap.csv. Returns
        {name: path}."""
        paths = output_paths(prefix, tileset=self.tilemap is not None)
        Path(prefix).parent.mkdir(parents=True, exist_ok=True)
        text = self.png_text()
        layout = self.config.transparent_index
        first = layout == "first"
        io.save_png(paths["image"], self.indices, self.palette, text, transparent_index=layout)
        io.save_png(paths["preview"], self.indices, self.palette, text,
                    scale=self.config.scale_preview, transparent_index=layout)
        used = np.zeros(len(self.palette), dtype=bool)
        used[np.unique(self.indices[self.indices >= 0])] = True
        entries = [{"hex": color.rgb8_to_hex(rgb), "lab": [float(v) for v in lab],
                    "used": bool(u)}
                   for rgb, lab, u in zip(self.palette, self.palette_lab, used)]
        if first:
            entries.insert(0, {"hex": None, "transparent": True,
                               "used": bool((self.indices < 0).any())})
        paths["palette"].write_text(json.dumps(entries, indent=2) + "\n")
        palette.write_hex(paths["palette_hex"], self.palette, transparent_first=first)
        meta = {"version": __version__, "config": self.config.to_dict(),
                "config_hash": self.config.hash(), "input_sha256": self.input_sha256,
                "width": int(self.indices.shape[1]), "height": int(self.indices.shape[0]),
                "stats": self.stats}
        paths["meta"].write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
        if self.tilemap is not None:
            # With "last", an opaque image has no transparent entry while a tileset whose
            # last row is padded gets one appended; the K colors keep their indices.
            io.save_png(paths["tileset"], self.tileset_indices, self.palette, text,
                        transparent_index=layout)
            paths["tilemap"].write_text(json.dumps(self.tilemap, separators=(",", ":")) + "\n")
            sheet_h, sheet_w = self.tileset_indices.shape
            tmj = tiles.tiled_map(self.tilemap, self.stats["tiles"], paths["tileset"].name,
                                  (sheet_w, sheet_h), Path(prefix).name)
            paths["tilemap_tmj"].write_text(json.dumps(tmj, indent=1) + "\n")
            paths["tilemap_csv"].write_text(tiles.tilemap_csv(self.tilemap))
        return {name: str(path) for name, path in paths.items()}


def output_paths(prefix, tileset: bool) -> dict:
    """{name: Path} of every file Result.save(prefix) writes, in write order."""
    prefix = Path(prefix)

    def sibling(suffix: str) -> Path:
        return prefix.parent / (prefix.name + suffix)

    paths = {"image": sibling(".png"), "preview": sibling("_preview.png"),
             "palette": sibling("_palette.json"), "palette_hex": sibling("_palette.hex"),
             "meta": sibling("_meta.json")}
    if tileset:
        paths["tileset"] = sibling("_tileset.png")
        paths["tilemap"] = sibling("_tilemap.json")
        paths["tilemap_tmj"] = sibling(".tmj")
        paths["tilemap_csv"] = sibling("_tilemap.csv")
    return paths


def uses_gerstner_palette(config: Config) -> bool:
    """True when gerstner's own converged palette and indices are the final ones."""
    return (config.method == "gerstner" and config.palette_name is None
            and palette.resolve_source(config) == "mcda")


def run(image, config: Config, max_pixels: int | None = None) -> Result:
    """Load and convert one image.

    ``image`` is a path (str or os.PathLike), a ``PIL.Image.Image``, or an (H, W, 3) /
    (H, W, 4) numpy array of uint8 or of float in [0, 1] (see ``io.from_image``). For paths
    ``input_sha256`` hashes the file bytes; for in-memory images it hashes shape and RGBA
    bytes. ``max_pixels`` defaults to ``io.MAX_INPUT_PIXELS`` and applies to every kind.
    """
    start = time.perf_counter()
    budget = io.MAX_INPUT_PIXELS if max_pixels is None else max_pixels
    if isinstance(image, (str, os.PathLike)):
        loaded = io.load(image, budget)
    else:
        loaded = io.from_image(image, budget)
    return run_loaded(loaded, config, {"load": time.perf_counter() - start})


def run_loaded(loaded: io.Loaded, config: Config, timings: dict | None = None) -> Result:
    """Convert an already loaded image (``io.load`` / ``io.from_image`` / ``io.from_rgba``)."""
    config.validate_palette()   # before any expensive stage
    timings = dict(timings or {})
    clock = time.perf_counter

    t = clock()
    pre = preprocess.run(loaded.rgb, loaded.alpha, config)
    timings["preprocess"] = clock() - t

    t = clock()
    small = downscale.get(config.method).run(pre.lab, pre.mask, pre.target_width,
                                             pre.target_height, config)
    if pre.pad_out:
        # Seamless mode: drop the wrapped context again.
        crop = (slice(pre.pad_out, -pre.pad_out), slice(pre.pad_out, -pre.pad_out))
        small.small_lab, small.small_mask = small.small_lab[crop], small.small_mask[crop]
        if small.indices is not None:
            small.indices, small.mean_lab = small.indices[crop], small.mean_lab[crop]
    timings["downscale"] = clock() - t
    if not small.small_mask.any():
        raise preprocess.PixelforgeError("no opaque pixels left after downscaling")

    t = clock()
    own_palette = uses_gerstner_palette(config)
    if own_palette:
        palette_lab = small.palette_lab
        indices = np.where(small.small_mask, small.indices, -1)
        if config.dither != "none":
            # DEVIATION: Section 8.2 — small_lab is the palette color per pixel, so dithering it
            # rarely moves a pixel. Dither the smoothed superpixel means against the annealer's
            # colors before β instead; palette_lab (index-aligned) already carries β once.
            indices = quantize.run(small.mean_lab, small.small_mask,
                                   small.palette_lab_unsaturated, config)
    else:
        # box / kopf, or gerstner with an external palette: quantize the (mean) colors.
        source = small.mean_lab if small.mean_lab is not None else small.small_lab
        palette_lab = palette.build(source[small.small_mask], config)
        indices = quantize.run(source, small.small_mask, palette_lab, config)
    timings["palette_quantize"] = clock() - t

    t = clock()
    post_stats: dict = {}
    pad = pre.canvas_pad()
    if any(pad[0] + pad[1]):
        # The outline ring is drawn inside the requested canvas: the image was downscaled to
        # the inner size and gets a transparent margin here, so the size never changes later.
        # With `canvas` the same padding also centers the fitted image on the canvas.
        indices = np.pad(indices, pad, mode="constant", constant_values=-1)
    indices, palette_lab = postprocess.run(indices, palette_lab, config, saturated=own_palette,
                                           fixed_palette=config.palette_name is not None,
                                           outline_margin=pre.outline_margin, stats=post_stats)
    timings["postprocess"] = clock() - t

    tile_result = None
    if config.tileset:
        t = clock()
        tile_result = tiles.extract(indices, palette_lab, config)
        # DEVIATION: Section 9 — near-duplicate tiles were merged, so the output image is
        # re-rendered from the tileset and is exactly what the tilemap draws.
        rendered = tiles.reconstruct(tile_result.tiles, tile_result.tilemap)
        tile_px_changed = int(np.count_nonzero(rendered != indices))
        indices = rendered
        timings["tiles"] = clock() - t

    palette_rgb8 = color.lab_to_rgb8(palette_lab)
    stats = {"iterations": int(small.stats.get("iterations", 0)),
             "final_palette_size": int(len(palette_rgb8)),
             "colors_used": int(len(np.unique(indices[indices >= 0]))),
             "colors_unused": int(len(palette_rgb8) - len(np.unique(indices[indices >= 0]))),
             "config_hash": config.hash(),
             "background_keyed": bool(pre.background_keyed),
             "fit": pre.fit, "content_size": [int(pre.inner_width), int(pre.inner_height)],
             "outline_index": post_stats["outline_index"],
             "outline_margin": int(pre.outline_margin),
             "outline_clipped": bool(post_stats["outline_clipped"]),
             "method": dict(small.stats),
             "timings": {name: round(seconds, 4) for name, seconds in timings.items()}}
    result = Result(image=io.indices_to_rgba(indices, palette_rgb8), palette=palette_rgb8,
                    indices=indices, config=config, input_sha256=loaded.sha256,
                    palette_lab=palette_lab, stats=stats)
    if tile_result is not None:
        result.tileset_indices = tile_result.tileset_indices()
        result.tileset = io.indices_to_rgba(result.tileset_indices, palette_rgb8)
        result.tilemap = tile_result.tilemap
        stats["tiles"] = int(len(tile_result.tiles))
        stats["tiles_rerender_px_changed"] = tile_px_changed
    return result
