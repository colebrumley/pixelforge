# Library

The same pipeline the CLI runs is available as a Rust crate and, optionally, as a Python
package.

## Rust

Depend on the `pixelforge` crate without the CLI:

```toml
[dependencies]
pixelforge = { git = "https://github.com/colebrumley/pixelforge", default-features = false }
```

```rust
use std::path::Path;

use pixelforge::config::{Config, Value};
use pixelforge::io::ArrayData;
use pixelforge::pipeline::{self, Input};

fn main() -> Result<(), pipeline::Error> {
    let cfg = Config::new(&[
        ("preset".into(), Value::str("sprite")),
        ("method".into(), Value::str("gerstner")),
        ("palette_size".into(), Value::Int(12)),
        ("out_height".into(), Value::Int(48)),
    ])?;
    let out = pipeline::run(Input::Path(Path::new("hero.png")), &cfg, None)?;
    out.save(Path::new("out/hero"))?; // out/hero.png, hero_preview.png, hero_palette.json, ...
    // out.image: RGBA8 (h * w * 4); out.palette: [u8; 3] per entry;
    // out.indices: palette index per pixel, -1 = transparent; out.stats: the stats JSON

    let photo = cfg.replace(&[
        ("enhance".into(), Value::Float(0.5)),
        ("ink".into(), Value::Float(0.6)), // darkens thin features into lines
    ])?;
    let rgba: Vec<u8> = (0..64 * 64 * 4).map(|i| (i % 251) as u8).collect();
    let array = Input::Array { shape: &[64, 64, 4], data: ArrayData::U8(&rgba) };
    pipeline::run(array, &photo, None)?; // (H, W, 3|4): u8, or f64 in [0, 1]
    Ok(())
}
```

`Config::new` applies the field defaults, then the preset, then the given fields.
`config::PRESETS` lists the presets. `Config::replace` with a new `preset` re-applies it to
every field not set explicitly.

`pipeline::run(input, &config, max_pixels)` applies the same pixel budget (`None`: 24
million) to every input kind. `input_sha256` hashes the file bytes for a path, and the shape
plus RGBA bytes for in-memory images.

Errors:

- `pipeline::Error::Config` for invalid configs, returned by `Config::new`, or for an
  unreadable `palette_name` file at the start of `run`. The CLI exits 2.
- `Error::Pixelforge` and `Error::Io` for everything else pixelforge reports, such as
  undecodable, oversized or fully transparent input. The CLI exits 1.

Log messages go through the `log` crate, with targets under `pixelforge`.

## Python

The optional Python binding wraps the same pipeline. Install it with

```bash
pip install 'git+https://github.com/colebrumley/pixelforge#subdirectory=crates/pixelforge-py'
```

Building it needs the Rust toolchain. Then:

```python
from pixelforge import Config, run

cfg = Config(preset="sprite", method="gerstner", palette_size=12, out_height=48)
res = run("hero.png", cfg)   # or a numpy (H, W, 3|4) array, or a PIL image with Pillow
res.save("out/hero")         # the files `pixelforge convert` writes, byte for byte
res.image                    # numpy RGBA (H, W, 4) uint8; res.palette (K, 3); res.indices (H, W)
```

Its errors are `ConfigError` (also a `ValueError`) and `PixelforgeError` (its base class).
[crates/pixelforge-py/README.md](../crates/pixelforge-py/README.md) has the rest.
