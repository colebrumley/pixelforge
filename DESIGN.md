# pixelforge design notes

Details behind the [README](README.md): how the three downscalers work, how the Rust
implementation is organised and keeps its numerics reproducible, the determinism contract,
timings, and where the implementation deliberately deviates from
[REQUIREMENTS.md](REQUIREMENTS.md).

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

Before downscaling, an optional pre-pass for photos (`enhance`, `ink`; off by default)
exaggerates features about one output pixel in size, which the downscalers would otherwise
average away.

After downscaling, the image is quantized (optionally with selective ordered dithering) and
post-processed: hue-shifted palette ramps, saturation, orphan-pixel removal, jaggy cleanup
and an outline for sprites; backgrounds can be cut into a de-duplicated tileset + tilemap.

## Implementation

pixelforge is a Cargo workspace. `crates/pixelforge` is the library and, behind its default
`cli` feature, the `pixelforge` program; `crates/pixelforge-py` is the optional Python
binding (PyO3 + maturin) over the same library; `xtask/` is the developer tooling. The
library has no Python and no numerical crates: images are plain slices with explicit
`(height, width)` dimensions. Modules of `crates/pixelforge/src/`:

- `config` (`config.rs`, `config/`): the field table (name, type, default, help, choices,
  bound per field; the presets) that the `Config`, its canonical JSON, its hash and the CLI
  flags are all built from; validation and its messages; the palette file loaders.
- `color`: sRGB ↔ CIELAB, ΔE, nearest palette entry (brute force, ties to the lowest index),
  hex helpers.
- `io`: input decoding (the six formats, EXIF orientation, 16-bit rescaling, the pixel
  budget checked from the header, regular files only), the input hash, array inputs, and the
  palette-indexed PNG writer with its `tEXt` chunks.
- `preprocess`, with `mask`, `layout`, `filters` and `enhance`: alpha mask, background
  keying and fringe unmixing, transparent filling (exact Euclidean distance transform), crop,
  output size and `fit`, integer repeat, box pre-reduction, the `enhance`/`ink` pass,
  bilateral or median denoise, seamless padding.
- `boxfilter`: the box downscaler.
- `kopf` + `kopf_run`: kopf's per-pixel E-step/M-step and C-step sums, and the control flow
  around them (kernel algebra, shape constraints, ghost and starved kernels, convergence).
- `gerstner` + `gerstner_run` + `annealer`: gerstner's SLIC assignment, superpixel bilateral
  filter and statistics; the iteration loop; the palette annealer (MCDA), also used directly
  for `palette_source="mcda"`.
- `downscale`: the result type every downscaler returns and the grid helper kopf and
  gerstner share.
- `palette`: bundled palettes, median cut, hue ramps, the palette build and `.hex` text.
- `quantize`: mapping to the palette with optional Bayer dithering.
- `postprocess`: saturation, orphan removal, jaggy cleanup, outline.
- `tiles`: tileset extraction with near-duplicate merging, the tilemap, Tiled `.tmj` and CSV.
- `pipeline`: `run` / `run_loaded` (load → preprocess → downscale → palette and quantize →
  post-process → tiles), `Output::save` and every output file, `environment()`.
- `cli` (`cli/`, `main.rs`): `convert`, `batch`, `compare`, `palettes`, `--config` loading,
  logging and exit codes; `sheet`: `compare`'s side-by-side and proof sheets and `--sweep`
  parsing.
- `npsum`: numpy's float64 summation orders, for the sums the Python version compared.
- `pyjson`, `pyrepr`, `pyunicode`: Python's `json.dumps` layouts, `round(x, n)`, float and
  string `repr` and the `str` behaviours the config parser needs, so the JSON files,
  canonical config JSON (and with it every config hash) and error messages are byte-identical
  to the Python version's.
- `parallel`: the thread count and the fixed-partition band runner.

**Threads.** Work runs on `PIXELFORGE_THREADS` threads (default: the available
parallelism) only where the result cannot depend on the thread count. The rule: the work is
split into a fixed partition (bands of rows whose boundaries depend on the image size only),
each band computes into its own buffer, and the buffers are merged in band order after all
bands finish; threads pull band numbers from a counter, which decides when a band runs,
never what it computes.

- kopf's three per-pixel passes reduce across pixels into per-kernel sums, so their bands of
  `BAND_ROWS` (32) input rows are part of the numerics: changing the band height would change
  the rounding of the sums.
- gerstner's SLIC assignment and superpixel bilateral filter (8-row bands) and the
  preprocessing bilateral denoise (32-row bands) compute each output element from the inputs
  alone, so their bands only split the work.
- gerstner's superpixel statistics are single-threaded on purpose: they sum each
  superpixel's pixels in row-major order, as `np.bincount` did, and per-band partial sums would
  change the rounding. Everything else is single-threaded.

**Numerics.** Every float routine was ported formula for formula from the last Python
version (9f53516, which reported 0.1.0) and compared with it on the fixtures, the samples
and a 48-case snapshot matrix before the Python code was removed
(`docs/rust-migration-plan.md` has the per-stage record).
The rules that make the results the same on every machine:

- float64 throughout. Every transcendental function (`exp`, `expm1`, `log`, `pow`, `cbrt`,
  `sin`, `cos`, `atan2`, `hypot`) comes from the pure-Rust `libm` crate, never from the
  platform's C library, whose last bit differs between macOS and Linux; `f64::sqrt` is exact
  and allowed. `crates/pixelforge/tests/determinism_guard.rs` fails the build on any platform
  transcendental, `powi`, `mul_add` or `.round()` in the sources.
- Rounding to integers is `f64::round_ties_even` (half to even, as `np.round` and Python's
  `round()`), never `f64::round`.
- No fast-math, no `target-cpu=native`, no compiler FMA contraction. An explicit `libm::fma`
  (correctly rounded in software, identical everywhere) is used only where the reference
  fused: kopf's 2×2 covariance eigendecomposition reproduces LAPACK's `dsteqr`/`dlaev2` path
  with the two fused multiply-adds Accelerate performs, `np.hypot` is
  `sqrt(fma(small, small, big²))`, and `1.1 ** n` is a table of correctly rounded powers.
- Where the Python version summed with `np.sum`/`mean`/`cumsum` and then compared the sums
  (median cut's box errors, the ramp bins' mean L, tile de-duplication means), `npsum`
  reproduces numpy's pairwise summation bit for bit; `np.bincount` sums (box cells,
  superpixel statistics) and the pre-reduction's block means keep numpy's order.
- scipy's `distance_transform_edt` feature indices (which opaque pixel a transparent one
  copies, ties included), `uniform_filter1d`, `gaussian_filter1d`, the 3×3 `median_filter`,
  `np.percentile`, and scikit-image 0.26's `denoise_bilateral` and colour conversion are
  reproduced as written, lookup tables and padding modes included.
- The only random numbers are a SplitMix64 stream seeded with `config.seed`.

Where the output differs from that Python version:

- Colour conversion agrees with scikit-image to 5.7e-14 in LAB. That moved one case of the
  48-case snapshot matrix: `noisy_gradient`, background preset, gerstner, 10 of 65,536
  pixels (ΔE 4.03), which sit on a near-tie in the annealer that a one-ulp jitter of
  scikit-image's own LAB moves the same way.
- gerstner's covariance and mean products were BLAS matrix products, whose summation order
  cannot be reproduced; they are sequential sums now, and the 3×3 eigenvectors come from a
  Jacobi solver. Principal axes move by at most 2.8e-13 and palettes by at most 3.3e-11; no
  index, mask cell or 8-bit value changed. Hue ramps move by up to 7e-14 (libm's
  `hypot`/`atan2`/`cos`/`sin` differ from numpy's by one ulp on a few percent of arguments).
- The Gaussian blur and the bilateral denoise match scipy and scikit-image built without FMA
  to a few ulps (≤ 2.2e-16 and ≤ 7.8e-16); the macOS arm64 wheels fused multiply-adds, so
  against those 4 pixels of a 1024² photo differ by up to 5.4e-6 where the colour distance
  crosses a lookup bin. No 8-bit value changed.
- JPEG is decoded by `zune-jpeg` (through the `image` crate) instead of Pillow's
  libjpeg-turbo: within 7 levels on up to half the pixels at 4:2:0 chroma, 3 levels at
  4:4:4, 1 for greyscale. PNG, BMP, TIFF, GIF and lossless WEBP decode to Pillow's pixels
  bit for bit, and lossy WEBP matched libwebp on every file tested.
- Seeded gerstner runs draw from SplitMix64 instead of numpy's PCG64: still deterministic,
  still different from the unseeded run, but not the Python version's output for the same
  seed.
- PNGs are written as 8-bit indexed (Pillow wrote 1/2/4-bit for small palettes): the bytes
  differ, the decoded pixels, palette, transparency and text do not. `compare`'s proof-sheet
  labels use an embedded bitmap font, so the labels and the sheet size differ by a few pixels;
  every cell is the same.
- Every config hash changed with the version (0.2.0); `tests/golden.json`'s pixel hashes did
  not.

## Determinism contract

1. No unseeded randomness anywhere. The only random generator is a SplitMix64 stream seeded
   with `config.seed`, used in exactly one place (the MCDA sub-cluster perturbation in
   `annealer.rs`) and only when `config.seed` is set; the default perturbation is
   deterministic without one.
2. No dependence on hash-map iteration order, thread scheduling or wall-clock time.
   Threads only run work split into a fixed partition merged in partition order
   ([Implementation](#implementation)), so `PIXELFORGE_THREADS` affects speed only;
   `crates/pixelforge/tests/determinism.rs` checks the threaded kernels bitwise at 1, 2, 3
   and 8 threads, and `cli.rs` whole runs. (Timings are recorded in `*_meta.json` only; they
   never influence a result.)
3. All arithmetic is float64 under the numerics rules above: `libm` for every
   transcendental function, no FMA except an explicit `libm::fma`, fixed summation orders.
   Final color conversion clips to [0, 255] and rounds half to even, never truncates.
4. `Config::hash()` is the SHA-256 of the canonical JSON of the config (sorted keys, no
   whitespace), the pixelforge version string and, when `palette_name` is set, the SHA-256
   of the resolved palette's `rrggbb` lines, so rewriting a palette file changes the hash.
   Every output PNG carries the `tEXt` chunks `pixelforge:config` (the canonical JSON) and
   `pixelforge:input_sha256` (SHA-256 of the input file bytes), plus, with a palette,
   `pixelforge:palette_sha256` and `pixelforge:palette` (the lines, newline-joined).
   `batch` passes its shared palette as an inline `palette_name="hex:rrggbb,…"`, not as the
   path of `shared_palette.hex`, so frame metadata never depends on `--outdir`.
5. Nothing in a config depends on the environment: `remove_bg` is `false` unless asked for,
   whether or not rembg is installed. A `remove_bg=true` run depends on the external `rembg`
   command and its model, which are outside this contract; `_meta.json` records
   `rembg --version`.
6. Running the CLI twice on the same input and config produces byte-identical PNGs;
   `cargo test` enforces this for every method and preset, in-process
   (`crates/pixelforge/tests/determinism.rs`) and across fresh processes at different
   `PIXELFORGE_THREADS` (`cli.rs`), comparing every output file (`_meta.json` minus
   `stats.timings` and `environment`). `crates/pixelforge/tests/golden.rs` pins the decoded
   pixels and `config_hash` per fixture, method and preset to `tests/golden.json`; after an
   intentional output change, regenerate it with `cargo xtask golden` and commit the diff
   with the change. CI runs the whole suite and `cargo xtask golden --check` on Linux x86-64
   and macOS arm64 against the same file.

Byte-identical output across *different machines* needs only the same pixelforge version:
the program is self-contained (its decoders and numerics are compiled in, and it calls no
system library for either), so no installed library version takes part in a result.
`_meta.json` records the platform, thread count and version under `environment`. Across CPU
architectures it is best-effort: float64 results could differ in the last bits between
instruction sets, which the 8-bit rounding absorbs in every case tested (the goldens agree on
x86-64 and arm64), and `_palette.json` rounds LAB to 6 decimals for the same reason.

## Timings

Whole runs (`total_s`, load to post-processing, the minimum of 3 runs) on an Apple M1 Max
(10 cores) at a load average of 5–7, from `cargo xtask bench --repeat 3` at 0.2.0
(`bench/aaec276.json`). The first two columns are the Python baselines `bench/0edd462.json`
(pure numpy) and `bench/9f53516.json` (Python with the Rust hot loops), both taken under a
load average of 12–24, so their ratios are approximate:

| case | pure Python | Python + Rust hot loops | Rust 0.2.0 | vs pure | vs hot loops |
| --- | ---: | ---: | ---: | ---: | ---: |
| portrait 1024² → 256², box | 1.64 s | 1.64 s | 0.30 s | 5.5× | 5.6× |
| portrait 1024² → 256², kopf (24 iterations) | 33.9 s | 4.48 s | 1.70 s | 20× | 2.6× |
| portrait 1024² → 256², gerstner (139 iterations) | 31.0 s | 6.76 s | 3.82 s | 8.1× | 1.8× |
| knight 456×528 → 44×64, kopf (50 iterations) | 8.05 s | 0.67 s | 0.32 s | 25× | 2.1× |
| knight 456×528 → 44×64, gerstner (61 iterations) | 2.72 s | 0.46 s | 0.19 s | 14× | 2.4× |
| `gradient6` 256² → 256², gerstner (299 iterations) | 28.4 s | 12.85 s | 7.77 s | 3.7× | 1.7× |
| `noisy_gradient` 256² → 256², gerstner (97 iterations) | 8.43 s | 3.20 s | 1.93 s | 4.4× | 1.7× |
| `circle_alpha` 256² → 64², kopf (13 iterations) | 1.01 s | 0.14 s | 0.05 s | 20× | 2.8× |

Over the 30 cases of the fixture matrix plus the two photos, the Rust binary is 3.7–32×
faster than the pure-Python version and 1.6–7× faster than the Python version that already
had the Rust hot loops; what it gained in the second step is everything outside the kernels
(decoding, preprocessing, the annealer, palette and quantization, output files). The
portrait and the knight sprite are photos outside the repository; the others are fixtures.

Threads (`PIXELFORGE_THREADS`, same machine and load, minimum of 3; the output is identical
at every count):

| case | 1 thread | 2 | 4 | 10 (default) |
| --- | ---: | ---: | ---: | ---: |
| portrait, kopf | 9.55 s | 4.98 s | 2.76 s | 1.64 s |
| portrait, gerstner | 8.14 s | 5.53 s | 4.18 s | 3.83 s |
| portrait, box | 0.62 s | 0.43 s | 0.34 s | 0.29 s |
| knight, kopf | 1.35 s | 0.74 s | 0.44 s | 0.33 s |

kopf scales with the cores (its per-pixel sums are banded); gerstner less so, because its
annealer and per-iteration statistics run on one thread by design (the statistics sum in
pixel order to stay bitwise equal to the reference). `cargo xtask bench --compare OLD.json`
compares a new run with an earlier file (`--case PATH:background` adds a photo of your own).

Gerstner at 256×256 output with 32 colors is the slow corner: from seconds to a minute or
more depending on how many annealing iterations the image needs. Without `-v` such runs print
nothing until the result; `-v` logs per-stage timings and `-vv` each iteration (logging only
reads values, it never changes an output).

## Known deviations

The algorithmic deviations are marked with `// DEVIATION:` comments in the Rust modules
(`crates/pixelforge/src/`): `kopf_run.rs`, `annealer.rs`, `gerstner_run.rs`,
`postprocess.rs`, `tiles.rs`, `layout.rs`, `preprocess.rs`, `pipeline.rs` and
`cli/commands/batch.rs`.

**Kopf (Section 6).** Implemented as specified, the algorithm cannot pass
`test_thin_line_stays_connected` (`crates/pixelforge/tests/downscalers.rs`), so the E-step
and two constraints differ:

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
- *Truncated windows skip the shape constraints.* A kernel whose R_k window holds transparent
  pixels or reaches past the image border has a lopsided footprint, so its one-sided variance
  and centroid direction are set by the cut and fire on every iteration: on `circle_alpha`
  256→32 a 2–4-kernel band along the silhouette (340 kernels) fired forever, 248 kernels ended
  at the σ cap, and a dark line 4 px inside the edge came out fainter than with box. Such
  kernels keep their σ, and the orientation test also needs a non-zero overlap `f` (no shared
  pixels, no edge between the pair). No kernel now reaches the cap there, and the run
  converges in 14 iterations.
- *Ghost kernels.* A kernel whose own output cell is less than half opaque (the box filter's
  rule) takes no part in the E-step, the shape constraints or the starved re-seeding, and is
  transparent in the output. Left in, it kept its tiny initial σ, claimed a dark line 2–3 px
  inside the silhouette, and was then dropped from `small_mask`, so the line vanished (13 of
  24 placements at 256→32 showed it on every row; now all 24 do). Its opaque neighbors take
  those pixels instead.
- *Laplacian smoothing* averages displacements from the grid position rather than raw
  positions. Identical for interior kernels; it stops border kernels from being dragged
  inward every iteration.
- *`small_mask`* is the cell rule above (identical to box's mask), because transparent
  pixels are excluded from the EM and the specified ratio would always be 1. A coverage
  measured with spatial-only responsibilities kept cells only 19 % opaque and flickered row
  by row on cells exactly half opaque; it is still used for the color of starved kernels.

- *Convergence.* The spec's test (max |Δμ| and max |Δν| < `kopf_tol`, no σ change) was never
  met on real content: border truncation and non-integer ratios keep a few centroids
  drifting (a solid 100×100 → 64×64 plateaued at 1.1e-3), and 0.5–1 % of the kernels keep
  growing σ every iteration. Kopf now stops when the RMS |Δμ| over fed kernels is below
  10·`kopf_tol` output pixels, the RMS |Δν| (unit-cube color) below `kopf_tol`, and σ
  changed on fewer than 2 % of the kernels not already at the cap. A solid image converges
  in 3 iterations, `line_diag` in 5, noisy content in about 30, with a mean ΔE below 0.1
  from the 50-iteration result.

The R_k half-width stays at the specified 2 output units.

**Gerstner (Section 7).**

- The start temperature is 1.1·2·σ, with σ the standard deviation along the first principal
  axis, not 1.1·2·σ². Rose's Tc = 2λ is for a squared-distance kernel; the association uses
  exp(−‖·‖/T), whose critical temperature scales with the spread. With the variance a ΔE-100
  checker spent 14 of 32 iterations cooling before anything happened.
- Annealing also stops when the final temperature is reached and a convergence event
  produces no split and no pair whose separation grew by more than `g_eps_cluster` over the
  last 3 events, even if fewer than K colors exist (backstop: T < 10⁻³·`g_T_final`).
  Otherwise an image with fewer than K separable colors would cool forever until
  `g_max_iters`. `stats.final_palette_size` reports the actual count.
- A sub-cluster pair only splits once it is three times further apart than the 2δ it was
  placed at, or one standard deviation of its own points (along their principal axis) apart
  if that is less, and a pair that is moving apart but not there yet is left alone rather
  than re-centered. The perturbation alone (1.0) already exceeds `g_eps_cluster` (0.25), so
  the spec's test also splits pairs that are still collapsing back together. Those colors
  coincide and never separate once K is reached: a single-color sprite ended with 16 palette
  entries of which 4 were distinct. Letting growth accumulate matters for small clusters,
  which diverge slowly: with re-centering, a thin line holding 1.5% of the pixels never split
  off. The spread cap matters for close colors: regions at L 50 and L 54 settle ≈ 2.8 apart
  after the bilateral filter and never reached 3·2δ.
- When more pairs are ready to split than palette slots remain, the slots go to the pairs
  whose split removes the most error, not to the lowest index.
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
- With a named palette the outline color (auto or explicit `#rrggbb`) snaps to the nearest
  palette entry (CIE76, ties → lowest index); the palette is never extended or changed, so a
  `_palette.hex` fed back via `palette_name` reproduces itself. Otherwise a new entry is
  appended, or, in a full 256-entry palette, the nearest entry is overwritten with it.
  `stats.outline_index` names the entry used (`null` if no outline was drawn).
- The canvas is always exactly the resolved output size; the outline never grows it. With
  an outline and `tileset=False`, a 1-px margin per side is always reserved inside the
  requested size (`stats.outline_margin = 1`), whether or not the silhouette reaches the
  edge: the image is downscaled to (W−2)×(H−2), padded with 1 transparent pixel per side
  before post-processing, and the ring is drawn without further padding. A derived dimension
  keeps the aspect ratio of that inner area (rounded, then the margin is added back; never
  below 8), so the sprite preset gives exactly 64 px on the longest edge.
- With `tileset=True` nothing is reserved or padded (the canvas must stay a multiple of
  `tile_size`); where the silhouette touches the edge the outline is clipped and
  `stats.outline_clipped` is true.
- Orphan removal merges a region only if its ΔE to the replacement is below `orphan_max_delta`
  (25), other orphans do not vote, and passes stop early on no change or a repeated image.

**Pipeline, tiles, preprocessing (Sections 5 and 9).**

- The mask stays binary (alpha ≥ `alpha_threshold`), but every downscaler weights each opaque
  pixel's color by alpha / 255 (`Preprocessed.weight`): box per cell, kopf in γ before the
  M-step, gerstner in the superpixel means and centers and in the palette prior, which is
  each superpixel's mean weight (uniform, as specified, for an opaque input). A cell made
  only of fringe pixels above the threshold still keeps the fringe's color.
- When gerstner is combined with `palette_name` or `palette_source="median_cut"`, its
  smoothed superpixel mean colors are quantized against that palette.
- With gerstner's own palette and `dither` other than `"none"`, the smoothed superpixel means
  (not the per-pixel palette colors, which dithering cannot move) are dithered against the
  annealer's colors before β, and the indices are then used with the saturated palette. The
  `"auto"` local-std criterion is computed on those means, and only over opaque cells.
- Two tiles (any flip) are near-duplicates only if the mean ΔE over the tile is below
  `tile_dedupe_tolerance` and every pixel's ΔE is below `TILE_DEDUPE_MAX_PIXEL_DELTA` (10,
  fixed; a transparent/opaque pair costs 100). The spec's mean-only test let a few very
  different pixels (stars, highlights) merge into a plain tile and vanish. A tile joins the
  nearest qualifying entry (lowest mean ΔE, then lowest id, then flip order), not the first.
- After near-duplicate tiles are merged, the output image is re-rendered from the tileset so
  it matches the tilemap exactly; `stats.tiles_rerender_px_changed` counts the pixels this
  changed (0 when `tile_dedupe_tolerance=0`).
- The tileset sheet is `tileset_columns` tiles wide (0, the default, = `ceil(sqrt(n))`; the
  spec's fixed 16 left small sets mostly empty); only the last row is padded. Besides the
  spec's `NAME_tilemap.json` (unchanged schema), tilesets also write `NAME.tmj`, a Tiled JSON
  map with an embedded tileset and the standard GID flip bits (0x80000000 horizontal,
  0x40000000 vertical), and `NAME_tilemap.csv` (GIDs without flip bits).
- `transparent_index` ("last" | "first") places the PNG's transparent palette entry. With
  "first" it is always index 0 (even for an opaque image) in the image, preview and tileset,
  so all three share one PLTE; a 256-color palette then no longer fits and is written as
  RGBA. With "last" (the spec) the entry is appended only when a PNG has transparent pixels,
  so an opaque image's tileset with a padded last row carries one extra trailing entry; the K
  color indices agree either way. `_palette.json` marks each entry `"used"` and
  `stats.colors_unused` counts the unused ones; they are kept so named palettes stay whole.
- With `tileset=True` the spec rounds both derived dims up to a multiple of `tile_size`, which
  stretched the image (160×96 at width 64 became 64×48, +26 % vertically) and could exceed
  the target edge (`tile_size=24` gave 264). Now the target edge (preset longest edge or the
  one given `out_width`/`out_height`) is rounded *down* to a multiple, the other axis is
  derived from it and the canvas is rounded up; `fit` decides how the image fills it:
  `"pad"` (default) centers it, aspect kept, with transparent pixels, `"stretch"` is the
  spec behaviour (warns when the aspect changes by > 2 %), `"crop"` crops the input centered
  to the canvas aspect before downscaling, so the downscalers stay untouched. Seamless mode
  always stretches (padding or cropping would break the wrap). `tile_size` larger than the
  longest derived edge is a config error. `stats.fit` and `stats.content_size` record it.
- Seamless padding is 2 output pixels per side, i.e. `2·rx` × `2·ry` input pixels per axis.
- Preprocessing fills transparent pixels with the nearest opaque color before denoising,
  uses `mode="edge"` for the bilateral filter, and repeats inputs that are smaller than the
  output by an integer factor.
- Inputs at least 2·`prereduce_max_ratio` (default 8) times the output on both axes are
  first box-reduced by the integer factor ⌊min(h/out_h, w/out_w) / prereduce_max_ratio⌋:
  an exact area mean of the sRGB image (transparent pixels filled with the nearest opaque
  color first), a block being opaque when at least half its pixels are, and the trailing
  rows/columns that do not fill a block dropped. Denoise and the downscalers then run on the
  reduced image, so the `denoise_sigma_*` values are in reduced-image pixels. Kopf's kernels
  span 4×4 output pixels, so at these ratios this is nearly lossless; together with the kopf
  changes it cuts 1024² → 64² from 209 s to 29 s. `stats.prereduce_factor` reports the
  factor; 0 disables it.
- `convert` writes a fifth file, `NAME_palette.hex`.
- The sprite preset has `remove_bg=False` instead of the spec's `None` ("True if rembg is
  installed"), so the same command gives the same config hash on every machine.
- `scale` and `canvas` are not in the spec. Without them `batch` sizes each frame from its own
  crop box, so frames of one animation came out at different scales; it now sets one `scale`
  from the union of the frames' crop boxes and a shared `canvas` (`shared_scale` in
  `cli/commands/batch.rs`).
- Input loading (not in the spec) only decodes PNG, JPEG, GIF, WEBP, BMP and TIFF, whatever
  the file name says (the Python version restricted Pillow to these plugins so that content
  under a misleading name could not reach others; EPS would run Ghostscript). It applies
  the EXIF orientation, rescales 16-bit grayscale to 8 bit instead of clipping it, uses the
  first frame of animations (with a warning), refuses non-regular files and enforces a pixel
  budget (`io::MAX_INPUT_PIXELS`) before decoding. A small input
  whose integer repeat would exceed that budget is refused.
- `Config` has three fields that are not in the spec, `key_bg`, `key_bg_tolerance` and
  `key_bg_fringe` (see Install), and the sprite preset sets `key_bg=True`. After keying, up to
  `key_bg_fringe` passes estimate each edge pixel's subject coverage as
  `max_c|p−bg| / max_c|fg−bg|` (fg sampled just beyond a 3-px fringe) and key out those below
  0.5, so the anti-aliased halo does not take palette entries. The gallery runs the fixtures with
  `key_bg=False` because they are test patterns, not sprites; images in `samples/` use the
  preset as-is.
- `Config` has five more fields that are not in the spec, `enhance`, `enhance_black`,
  `enhance_white`, `enhance_radius` and `ink` (`enhance.rs`), all off by default. The
  downscalers are faithful to the source, and on a photo that loses whatever is smaller than
  an output pixel: gerstner dropped an eye and the mouth of a 128-px portrait. The pre-pass
  runs after pre-reduction and before denoising, with every radius in output pixels:
  - a levels stretch of the subject's brightness (largest channel) between its 1st and 99th
    percentiles. Brightness rather than luminance, because a saturated color reaches its
    ceiling far below luminance 1 and would clip flat. The gain is capped at 5 (a 1/255 step
    in a flat subject otherwise became black and white);
  - an unsharp mask on luminance of sigma `enhance_radius`;
  - for `ink`, a darkening where a difference of Gaussians (0.5 and 1.6 output pixels) on
    the fully stretched image finds a thin dark feature, so `ink` acts the same at any
    `enhance`.

  Each pixel's RGB is scaled by one factor, and a brightening eases toward the luminance at
  which its largest channel reaches 1, so hue never shifts. Blurs are normalized by alpha
  weight and, from sigma 4 px up, are three running means per axis, so the cost does not grow
  with the radius (`prereduce_max_ratio=0` on a 2048-px input took 300 s with Gaussian
  kernels). In seamless mode the pass runs before the wrap padding with periodic filters.
  `batch` pools the frames' brightness and fixes `enhance_black`/`enhance_white` for all of
  them. The right strength depends on the photo, hence `compare --prepass`; when the backdrop
  is not keyed it is part of the "subject" and of the percentiles. It is not in any preset
  because sources that already have outlines and full contrast lose their mid-tone shading.

**Tests (Section 12).**

- The box baseline does not produce a *broken* line on `line_diag.png`; it produces an
  unbroken but gray one (L ≈ 66, nothing below L = 50). The test asserts exactly that, and
  that the Kopf line is dark (L < 50) and 8-connected along the whole diagonal.
- `crates/pixelforge/tests/determinism.rs` runs the background preset at 64×64 instead of
  256×256. Its kopf cases at preset size, and kopf on the 256-px golden fixtures, are
  `#[ignore]`d as slow; 32-px variants with `kopf_max_iters=10` run in their place under a
  plain `cargo test`. CI runs everything (`-- --include-ignored`).
- `test_converges` uses an 8×8 output.
