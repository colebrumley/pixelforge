# pixelforge (Python binding)

An optional Python package over the pixelforge Rust library: the same pipeline as the
`pixelforge` command, callable from Python with numpy arrays in and out. The CLI does not
need it; install the CLI with `cargo install --git https://github.com/colebrumley/pixelforge pixelforge`.

## Install

Not on PyPI. Building needs a stable Rust toolchain (`rustup`) and Python 3.11 or newer;
the only Python dependency is numpy. Straight from GitHub:

```bash
pip install 'git+https://github.com/colebrumley/pixelforge#subdirectory=crates/pixelforge-py'
```

or from a checkout of the repository:

```bash
pip install ./crates/pixelforge-py
# or
uv pip install ./crates/pixelforge-py
```

maturin (the build backend) compiles the extension module `pixelforge._native` from this
crate and the library in `crates/pixelforge`. Pillow is optional: with it installed, `run`
also accepts `PIL.Image.Image` inputs.

## Use

```python
from pixelforge import Config, run

cfg = Config(preset="sprite", method="gerstner", palette_size=12, out_height=48)
res = run("hero.png", cfg)
res.save("out/hero")   # out/hero.png, hero_preview.png, hero_palette.json, ...
res.image              # numpy RGBA (H, W, 4) uint8
res.palette            # numpy (K, 3) uint8
res.indices            # (H, W) palette index, -1 = transparent

run("photo.png", cfg.replace(enhance=0.5))
run(rgba, cfg)                     # (H, W, 3|4) numpy array: uint8, or float in [0, 1]
run(Image.open("hero.png"), cfg)   # a PIL image (needs Pillow)
```

- `Config(**fields)`: field defaults, then the preset's overrides, then the arguments;
  immutable, every field an attribute. `to_dict()`, `canonical_json()`, `hash()` (the
  `config_hash` the CLI writes), `replace(**changes)`, `validate_palette()`,
  `provenance()`, `Config.field_names()`. `PRESETS` lists each preset's overrides.
- `run(image, config, max_pixels=None)` for a path, an array or a PIL image;
  `load(path)` / `from_image(image)` and `run_loaded(loaded, config)` split loading from
  converting. `max_pixels` defaults to `MAX_INPUT_PIXELS` (24 million).
- `Result`: `image`, `palette`, `indices`, `palette_lab`, `stats`, `input_sha256`,
  `tileset`, `tileset_indices`, `tilemap`, and `save(prefix)`, which writes the files
  `pixelforge convert` writes, byte for byte (except `_meta.json`'s timings).
- Errors: `ConfigError` (also a `ValueError`) for invalid configs and unreadable palettes,
  `PixelforgeError` (its base class) for everything else pixelforge reports; `OSError`
  when an input cannot be opened.
- Progress messages go to the `pixelforge` logger (silent unless configured).
  `PIXELFORGE_THREADS=N` limits the worker threads (speed only, never the output).

## Test

```bash
uv venv /tmp/pf-venv
uv pip install --python /tmp/pf-venv/bin/python ./crates/pixelforge-py pytest
cargo build --release -p pixelforge       # for the CLI comparison (skipped without it)
/tmp/pf-venv/bin/python -m pytest -q crates/pixelforge-py/tests
```

CI runs these smoke tests on Linux (`.github/workflows/ci.yml`, job `binding`).
