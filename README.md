# pixelforge

Deterministic conversion of arbitrary raster images (especially AI-generated ones) into
16-bit-style pixel art: sprites and backgrounds/tiles. Same input + same config =
byte-identical PNG, every run.

See [REQUIREMENTS.md](REQUIREMENTS.md) for the full specification this prototype was built
from, and [gallery/index.html](gallery/index.html) for a side-by-side comparison of the three
downscalers (generate it with `python scripts/gallery.py`; the `gallery/` directory is not
checked in).

## Install

Python 3.11+.

```bash
uv sync                      # creates .venv with pixelforge + pytest
uv run pytest -q
```

or with pip:

```bash
pip install -e . pytest
pytest -q
```

Background removal is optional: `pip install 'pixelforge[bg]'` adds `rembg`. Without it,
`--remove-bg` fails with a clear message and the sprite preset simply leaves it off.

Independently of `rembg`, the sprite preset keys out a *flat* opaque background (`key_bg`):
if at least 90% of the image's border pixels share one color, every pixel of that color
connected to the border becomes transparent. Image generators return sprites on a plain
backdrop rather than with alpha, and an opaque backdrop otherwise swallows most of the
palette. Regions of the backdrop color enclosed by the subject are kept. Turn it off with
`--no-key-bg`; adjust the color match with `--key-bg-tolerance` (per-channel sRGB distance,
default 0.08). `stats.background_keyed` in the output says whether it happened.

## CLI

```bash
pixelforge convert tests/fixtures/circle_alpha.png --preset sprite
pixelforge convert tests/fixtures/noisy_gradient.png --preset background
pixelforge compare tests/fixtures/circle_alpha.png -o out
```

```
pixelforge convert INPUT [-o OUTDIR] [--preset sprite|background] [--method box|kopf|gerstner]
                   [--out-width N] [--out-height N] [--palette-size K] [--palette-name NAME]
                   [... every Config field as --kebab-case ...] [--config config.json]
pixelforge batch INPUT_DIR [-o OUTDIR] [same flags]   # one shared palette for all images
pixelforge compare INPUT [-o OUTDIR]                  # box, kopf, gerstner side by side
pixelforge palettes                                   # lists bundled palettes
```

Config precedence is defaults ← preset ← `--config` JSON ← explicit flags. Boolean fields
take `--flag/--no-flag`. `convert` prints the output paths and stats as one JSON line. Exit
code 0 on success, 2 on a config validation error, 1 on any other error (message on stderr).

`convert` writes into `OUTDIR` (default `out/`), named after the input file:

| file | content |
| --- | --- |
| `NAME.png` | native-resolution result (palette-indexed PNG, RGBA above 256 colors) |
| `NAME_preview.png` | nearest-neighbor upscale by `scale_preview` |
| `NAME_palette.json` | list of `{"hex": "#rrggbb", "lab": [L, a, b]}` |
| `NAME_palette.hex` | the same palette as `.hex`, reusable with `--palette-name` |
| `NAME_meta.json` | stats, timings, config, config hash, input hash |
| `NAME_tileset.png`, `NAME_tilemap.json` | only with `tileset` (background preset) |

`batch` builds one palette from all images (at most 50 000 pixels per image, fixed stride),
writes it to `OUTDIR/shared_palette.hex` and converts every image against it, which keeps
animation frames from flickering.

## Library

```python
from pixelforge import Config, run, PRESETS

cfg = Config(preset="sprite", method="gerstner", palette_size=12, out_height=48)
res = run("hero.png", cfg)
res.save("out/hero")        # out/hero.png, hero_preview.png, hero_palette.json, ...
res.image                   # numpy RGBA (H, W, 4) uint8
res.palette                 # numpy (K, 3) uint8
res.indices                 # (H, W) palette index, -1 = transparent
```

`Config(...)` applies field defaults, then the preset, then the keyword arguments.

## Methods

**box** — the naive baseline. Every output pixel is the mean CIELAB color of the input pixels
in its cell; the palette is then built with median cut. Fast, but thin features dissolve into
the background and edges that do not fall on cell boundaries turn into blended colors.

**kopf** — content-adaptive downscaling. Each output pixel owns an anisotropic Gaussian
kernel with a spatial mean/covariance and a color mean/variance. A constrained EM lets the
kernels shift (up to a quarter pixel), reshape and become color-selective so that they align
with image features, while constraints keep them local, bounded in size and free of
staircase artifacts. Edges stay sharp and thin lines survive at full contrast.

> J. Kopf, A. Shamir, P. Peers. "Content-Adaptive Image Downscaling." ACM Trans. Graphics
> 32(6), SIGGRAPH Asia 2013. https://johanneskopf.de/publications/downscaling

**gerstner** — joint superpixel + palette optimization. The input is segmented into exactly
one superpixel per output pixel with a modified SLIC whose color term uses the superpixel's
*assigned palette color*, while the palette itself grows from one color to K by
mass-constrained deterministic annealing: the temperature is lowered step by step and a
palette color splits in two when its pair of sub-clusters separates. Segmentation and palette
therefore converge together, and the result is already indexed.

> T. Gerstner, D. DeCarlo, M. Alexa, A. Finkelstein, Y. Gingold, A. Nealen. "Pixelated Image
> Abstraction with Integrated User Constraints." Computers & Graphics, 2013.
> https://cragl.cs.gmu.edu/pixelate/

Also used: R. Achanta et al., "SLIC Superpixels" (modified as in Gerstner), and K. Rose,
"Deterministic annealing for clustering, compression, classification, regression, and related
optimization problems", Proc. IEEE, 1998 (the MCDA source).

After downscaling, the image is quantized (optionally with selective ordered dithering) and
post-processed: hue-shifted palette ramps, saturation, orphan-pixel removal, jaggy cleanup
and an outline for sprites; backgrounds can be cut into a de-duplicated tileset + tilemap.

Timings on an Apple M1 Max, 1024×1024 → 64×64: kopf ≈ 36 s (50 iterations), gerstner ≈ 25 s,
box well under a second. `pixelforge convert … --preset background` on a 256×256 input
(256×256 output, 65 536 kernels) takes about 30 s. Gerstner at 256×256 output with 32 colors
is the slow corner: 5 s to 2 minutes depending on how many annealing iterations the image
needs.

## Determinism contract

1. No unseeded randomness anywhere. `random`, `numpy.random` and `os.urandom` are not
   called. The only RNG is `numpy.random.default_rng(config.seed)`, used in exactly one place
   (the MCDA sub-cluster perturbation in `downscale/gerstner.py`) and only when `config.seed`
   is set; the default perturbation is deterministic without an RNG.
2. No dependence on dict/set iteration order of unordered inputs, thread scheduling or
   wall-clock time. No multithreading inside numerical kernels. (Timings are recorded in
   `*_meta.json` only; they never influence a result.)
3. All arrays are `float64` internally. A given quantity is always reduced by the same numpy
   routine. Final color conversion rounds with `np.round(...).astype(np.uint8)` after
   clipping to [0, 255], never truncation.
4. `Config.hash()` is the SHA-256 of the canonical JSON of the config (sorted keys, no
   whitespace) and the pixelforge version string. Every output PNG carries the `tEXt` chunks
   `pixelforge:config` (the canonical JSON) and `pixelforge:input_sha256` (SHA-256 of the
   input file bytes).
5. Running the CLI twice on the same input and config produces byte-identical PNGs;
   `pytest tests/test_determinism.py` enforces this for every method and preset.

Byte-identical output across *different machines* additionally assumes the same versions of
numpy, scipy, scikit-image and Pillow (`uv.lock` pins them).

## Gallery

```bash
python scripts/gallery.py            # → gallery/index.html (about 5 minutes)
python scripts/gallery.py --quick    # background preset at 64 px instead of 256 px
```

Runs box/kopf/gerstner × sprite/background on every fixture plus every PNG in an optional
`samples/` directory and writes [gallery/index.html](gallery/index.html) with timings and
iteration counts under each cell.

## Known deviations

The algorithmic deviations are marked with `# DEVIATION:` comments in the code.

**Kopf (Section 6).** Implemented as specified, the algorithm cannot pass
`test_thin_line_stays_connected`, so the E-step and two constraints differ:

- *No per-kernel normalization.* `w_k(i) /= Σ w_k(i)` rescales each kernel relative to its own
  best-fitting pixels. A kernel on a thin feature (less than half of its cell) then never
  prefers the feature over the background, and the darkest kernel in reach takes all feature
  pixels from its neighbors: the 2-px test line came out as isolated dark blobs with gaps.
  Kernels instead compete on their absolute bilateral fit, computed in log space so the
  initial σ = 1e-4 cannot underflow. The "sum == 0 → uniform over R_k" guard is replaced by
  re-seeding a starved kernel (one that no pixel prefers) with the color of the input pixel
  under its center.
- *`s` and `f` are divided by Σγ_k.* As literal sums they scale with the number of input
  pixels per kernel, so the thresholds 0.2 and 0.08 would fire for every kernel on every
  iteration, σ would saturate at its cap and the method would degenerate into a box filter.
- *Orientation term `o`.* `Σ ∇(γ_k / (γ_k + γ_n))` summed over the window telescopes to the
  ratio on the window's border, where both γ are numerically zero, so the literal estimate is
  noise: it fired for thousands of kernels per iteration on a plain checkerboard and kept
  blurring a result that was exact after five iterations. It is replaced by the direction
  between the two kernels' M-step centroids, a robust measure of the same quantity (the net
  normal of the boundary between the two kernels' pixel sets).
- *Laplacian smoothing* averages displacements from the grid position rather than raw
  positions. Identical for interior kernels; it stops border kernels from being dragged
  inward every iteration.
- *`small_mask`* is computed from spatial-only responsibilities, because transparent pixels
  are excluded from the EM and the specified ratio would always be 1.

With the spec's convergence criterion (|Δμ| and |Δν| < `kopf_tol` and no σ change) most real
images run to `kopf_max_iters`; the simple fixtures converge in 30–48 iterations. The
R_k half-width stays at the specified 2 output units (the 1024² → 64² run takes ≈ 36 s).

**Gerstner (Section 7).**

- Annealing also stops when the final temperature is reached and a convergence event
  produces no split, even if fewer than K colors exist. Otherwise an image with fewer than K
  separable colors would cool forever until `g_max_iters`. `stats.final_palette_size` reports
  the actual count.
- A sub-cluster pair only splits if it has also moved further apart than the 2δ it was
  placed at. The perturbation alone (1.0) already exceeds `g_eps_cluster` (0.25), so the
  spec's test also splits pairs that are still collapsing back together. Those colors
  coincide and never separate once K is reached: a single-color sprite ended with 16 palette
  entries of which 4 were distinct.
- Transparent input pixels are never assigned, so `small_mask` attributes each one to its
  spatially nearest superpixel to compute the opaque fraction, and the prior P(p_s) is
  uniform over the superpixels that hold opaque pixels.

**Palette and post-processing (Section 8).**

- "Median cut" picks the box with the largest summed squared error (not the most pixels)
  and cuts it where the two halves' summed squared error is smallest (not at the median).
  The spec's rule keeps halving a dominant color while the few pixels of every small feature
  share one leftover box and are averaged into a color none of them has. It also never
  splits a box spanning less than ΔE 2.3 (one just-noticeable difference), so it returns
  fewer than K colors when the image has fewer distinguishable ones.
- `regularize_ramps` reaches the 6° maximum at an L distance of 30 from the bin's mean L.
- Ramp regularization and saturation are skipped when a named palette is used.
- With `tileset=True` the outline pass does not pad the canvas.

**Pipeline, tiles, preprocessing (Sections 5 and 9).**

- When gerstner is combined with `palette_name` or `palette_source="median_cut"`, its
  smoothed superpixel mean colors are quantized against that palette.
- After near-duplicate tiles are merged, the output image is re-rendered from the tileset so
  it matches the tilemap exactly.
- Seamless padding is 2 output pixels per side, i.e. `2·rx` × `2·ry` input pixels per axis.
- Preprocessing fills transparent pixels with the nearest opaque color before denoising,
  uses `mode="edge"` for the bilateral filter, and repeats inputs that are smaller than the
  output by an integer factor.
- `convert` writes a fifth file, `NAME_palette.hex`.
- `Config` has two fields that are not in the spec, `key_bg` and `key_bg_tolerance` (see
  Install), and the sprite preset sets `key_bg=True`. The gallery runs the fixtures with
  `key_bg=False` because they are test patterns, not sprites; images in `samples/` use the
  preset as-is.

**Known weakness.** Gerstner can still end with a few near-duplicate palette entries on an
image dominated by one color (13 distinct colors out of 16 on the slime test sprite, where
it was 4 before the split fix).

**Tests (Section 12).**

- The box baseline does not produce a *broken* line on `line_diag.png`; it produces an
  unbroken but gray one (L ≈ 66, nothing below L = 50). The test asserts exactly that, and
  that the Kopf line is dark (L < 50) and 8-connected along the whole diagonal.
- `test_determinism.py` runs the background preset at 64×64 instead of 256×256 to keep the
  suite at about two minutes.
- `test_converges` uses an 8×8 output.

## Troubleshooting

On macOS, files inside a hidden `.venv` can end up with the `hidden` flag, and Python then
ignores the `.pth` file an editable install relies on (`ModuleNotFoundError: No module named
'pixelforge'` from the `pixelforge` command). Either install a regular copy with
`uv sync --no-editable`, or run `python -m pixelforge …` from the repository root. The tests
and scripts do not depend on the editable install.
