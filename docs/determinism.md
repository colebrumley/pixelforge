# Determinism

The same input and config give the same output bytes on every run, with two limits:

- `--remove-bg` (rembg) is outside the guarantee and is never on by default.
- Across CPU architectures, identity is best-effort at the last bits of a float64. Rounding
  to 8-bit colors has absorbed those bits in every case tested.

## How it holds

pixelforge has no unseeded randomness and no dependence on thread scheduling or the wall
clock, and it uses float64 throughout.

- The program is self-contained, so byte identity across machines does not depend on the
  versions of any installed library. Every transcendental function comes from the pure-Rust
  `libm` crate, and rounding is half to even.
- Threads split the work into a fixed partition merged in a fixed order, so
  `PIXELFORGE_THREADS` changes speed only.

## How it is tested

`cargo test` (`crates/pixelforge/tests/determinism.rs` and `cli.rs`) enforces identical
output bytes for every method and preset, in-process and across fresh processes with
different thread counts. CI checks `tests/golden.json` (pixel hashes per fixture, method and
preset) on Linux x86-64 and macOS arm64.

## Telling outputs apart

Each output PNG carries `tEXt` chunks:

- `pixelforge:config`: the canonical config JSON.
- `pixelforge:input_sha256`: the hash of the input file bytes.
- `pixelforge:palette_sha256` and `pixelforge:palette` (its `rrggbb` colors), when a palette
  is given.

The config hash covers the config, the version and the palette's colors, so editing a
palette file changes it. `_meta.json` also records the platform, the thread count and the
pixelforge version (not hashed), to tell which of these differ when two outputs do. Editor
saves add their own chunk and metadata (see [the edit log](editor-guide.md#the-edit-log)).

## Differences from the Python version

pixelforge was a Python program until 0.2.0. Against the last Python version (which still
reported 0.1.0):

- Outputs from PNG, BMP, TIFF, GIF and WEBP inputs are pixel-identical, except one gerstner
  case that sits on a near-tie (10 pixels of one 256×256 test output).
- JPEG inputs decode within a few levels of Pillow's libjpeg, so outputs from JPEG sources
  differ.
- Seeded gerstner runs (`--seed`) draw from a different random stream.
- Every config hash changed with the version.

[DESIGN.md](../DESIGN.md#implementation) has the details.
