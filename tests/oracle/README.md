# Oracle fixtures

Reference outputs recorded from the Python implementation (numpy, scipy, scikit-image,
Pillow) before it was removed (0.2.0), read by the Rust tests in `crates/pixelforge/tests/`
(`*_oracle.rs`). Do not edit by hand.

They were recorded by `tests/oracle/record_oracles.py`, which is in the git history up to
227b1df. It needs the Python package (removed in 0.2.0; in the git history up to 1fff901)
and the Python stack at the versions in `versions.json`, so the files cannot be re-recorded
from this tree. The recorder was deterministic (fixed inputs, `numpy.random.default_rng`
with fixed seeds).

The recording version (`versions.json`, `pixelforge`) is part of every recorded
`config_hash`; the Rust tests check those hashes at that version and the canonical JSON as
recorded.

## Format

All data files are JSON, readable with serde_json alone. Every float is written as Python's
`repr` (shortest round-trip digits), so parsing it as f64 gives back the recorded value bit
for bit. The layout is `json.dumps(sort_keys=True, indent=1)`, except that a list or object
whose items are all scalars is written on one line. Conventions:

- Arrays are nested lists in row-major order. Images of shape (H, W, 3) are stored as H
  rows of W*3 values (r, g, b interleaved), so pixel (y, x) channel c is `rows[y][3*x + c]`.
- Boolean masks are lists of row strings of `0`/`1` (`1` = opaque / true).
- Values JSON cannot represent (only in `config_errors.json` kwargs) are tagged:
  `{"$float": "nan" | "inf" | "-inf"}`, and `{"$int": "<decimal>"}` for integers outside
  [-2^63, 2^64).

## Files

- `versions.json`: the numpy, scipy, scikit-image and Pillow versions that produced the
  files, and `pixelforge.__version__` (part of every `Config.hash()`).
- `palettes/`: `four.hex`, `three.gpl` and `pälette ✓.hex`, palette files the config corpus
  refers to by repo-root-relative path (`tests/oracle/palettes/...`). The non-ASCII name
  exercises `ensure_ascii` escaping in the canonical JSON.
- `config_corpus.json`: `cases` of `{kwargs, canonical_json, hash, to_dict}` for
  `Config(**kwargs)`: `Config()`, both presets, every field at one or more non-default
  values (every bundled palette, inline `hex:` palettes, the three palette files), float
  repr stress values (1e-05, 5e-324, 1e16, 123456789.123456789, 4.999999999999999,
  0.30000000000000004, ...), int/float coercions, seed 0 and 2^63-1, and a few combined
  configs. A case with `replace` (a list of keyword objects) is
  `Config(**kwargs).replace(**r1).replace(**r2)...`. Cases with a `palette_name` also hold
  `palette_sha256` and `palette` (the resolved `rrggbb` lines). `hash` is the SHA-256 of
  `canonical_json + "\n" + version`, plus `"\npalette:" + palette_sha256` with a palette.
  Palette paths resolve from the repository root.
- `config_errors.json`: `cases` of `{kwargs, python, message}`: invalid keyword arguments,
  the call as Python source, and the exact `str(ConfigError)`. Covers unknown fields, bad
  presets, None for non-optional fields, wrong types (bool for int, 1.5 for int, str, list),
  integers beyond 64 bits, non-finite floats, and every bound and choice in
  `Config._validate`, in its order. Unreachable and so not recorded: `_coerce`'s
  "unsupported field type" (no field has another type) and the `seed` entry of
  `UPPER_BOUNDS` (2^63 already fails the 64-bit check in `_coerce`).
- `lab_lattice.json`: `axis` = `np.linspace(0, 1, 10)`; `rgb`, the 1,000 lattice colours (r
  outermost, then g, then b); `lab` = skimage `rgb2lab(rgb)` (D65). The three LAB files
  are recorded from scikit-image itself, not from `pixelforge.color`,
  which is tested against them.
- `lab_roundtrip.json`: `lab`, 200 inputs: L in `linspace(0, 100, 8)` × a and b in
  `linspace(-128, 127, 5)` (L outermost; many are out of gamut); `lab_to_rgb` (floats
  clipped to [0, 1]) and `lab_to_rgb8` (`np.round`, half to even).
- `rgb8_lab.json`: `rgb8`, the 256 greys then the 8 cube corners (r, g, b in {0, 255}, r
  outermost); `lab` = `rgb8_to_lab(rgb8)`.
- `edt_indices.json`: `cases` of `{name, mask, distances, indices_row, indices_col}` from
  `scipy.ndimage.distance_transform_edt(~mask, return_distances=True, return_indices=True)`:
  for each pixel, the distance to and the coordinates of the opaque pixel it copies its
  colour from (the transparent fill), which pins scipy's tie-breaking. Masks: five 48×48
  random masks at opaque densities 0.1, 0.3, 0.5, 0.7, 0.9
  (`default_rng(20261010).random((48, 48)) < density`, drawn in that order);
  `circle_alpha.png` alpha >= 128 at rows 20..84, cols 20..84; a single opaque pixel at
  (17, 30) in 48×48; and an all-opaque 16×16.
- `skimage_bilateral.json`: `cases` of `skimage.restoration.denoise_bilateral(rgb,
  sigma_color, sigma_spatial, channel_axis=-1, mode="edge")` clipped to [0, 1] (the
  pre-Phase-4 `preprocess.denoise` with `denoise="bilateral"`). The input is
  `filled_rgb8 / 255.0`: the fixture crop at (`top`, `left`) of `size` × `size`, with
  transparent pixels (`mask` = alpha >= 128 within the crop) given the colour of the opaque
  pixel `scipy.ndimage.distance_transform_edt(~mask, return_indices=True)` picks. Crops: `noisy_gradient.png` and
  `circle_alpha.png` at the default sigmas (0.08, 2.0) at 48×48 and at sigma_spatial 0.5 at
  32×32 (same corner; smaller to stay within the size budget), and a 24×24 corner crop of
  `noisy_gradient.png` at sigma_color 0.5, sigma_spatial 16. `output` is size rows of
  size*3 values.
- `ndimage_filters.json`: `arrays`: `random_17x23` (`default_rng(20261011).random((17,
  23))`), `ramp_40` (`linspace(-1, 2, 40)`), `constant_6x9` (all 0.3). `filters`:
  `{func, input, params, output}` for `scipy.ndimage.gaussian_filter1d` (sigma 1.5 and 3.9,
  default truncate 4) and `uniform_filter1d` (sizes 3, 5, 9, 10) along every axis with
  modes `nearest` and `wrap`, and `uniform_filter` / `median_filter` with size 3, mode
  `nearest`. `samples`, `percentile` (`np.percentile(x, (1.0, 99.0, 50.0))`, linear) and
  `median` (`np.median`, axis None, or 0 for `border_10x3`) on odd, even, duplicate, size-1
  and size-2 arrays (the random ones from the same generator, after `random_17x23`).
- `float_repr.json`: `repr`, 10,000 distinct finite floats as Python `repr` strings (the
  string is the ground truth: parse it as f64, format it back, compare): powers of ten
  1e-320..1e308, short decimals, random mantissas at binary exponents -1074..1023, random
  bit patterns, integers as floats, subnormals, the extremes, negatives and -0.0.
  `json_dumps_equals_repr`: `json.dumps(x) == repr(x)` held for every one;
  `canonical_json_of_list`: SHA-256 of `json.dumps(floats, separators=(",", ":"))`.
- `outputs_small/`: two full pipeline runs saved with `Result.save`; `<prefix>_config.json`
  holds the fixture and the Config kwargs. `two_color_sprite`: sprite, box, 16×16, no
  denoise, no keying, on `two_color.png`. `checker_tiles_background`: background (tileset),
  box, 32×32, no denoise, no keying, on `checker_tiles.png`. The text outputs are kept as
  written (`_palette.json`, `_palette.hex`, `.tmj`, `_tilemap.json`, `_tilemap.csv`), except
  `_meta.json`, whose `environment` and `stats.timings` keys are removed. PNG bytes are not
  committed; `<name>.png.json` holds the decoded PNG: `mode`, `size`, `palette` (PLTE as a
  flat list), `transparency`, `text` (`io.read_png_text`), `pixels` (palette indices) and
  `pixels_sha256` (SHA-256 of the decoded index bytes); for `_preview.png` (the 8× nearest
  upscale) the hash only.

## Recorded with

| Library | Version |
| --- | --- |
| Pillow | 12.3.0 |
| numpy | 2.4.6 |
| pixelforge | 0.1.0 |
| scikit-image | 0.26.0 |
| scipy | 1.17.1 |
