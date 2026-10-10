# pixelforge design notes

Details behind the [README](README.md): how the three downscalers work, the determinism
contract, and where the implementation deliberately deviates from [REQUIREMENTS.md](REQUIREMENTS.md).

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
- With `tileset=True` the spec rounds both derived dims up to a multiple of `tile_size`, which
  stretched the image (160×96 at width 64 became 64×48, +26 % vertically) and could exceed
  the target edge (`tile_size=24` gave 264). Now the target edge (preset longest edge or the
  one given `out_width`/`out_height`) is rounded *down* to a multiple, the other axis is
  derived from it and the canvas is rounded up; `fit` decides how the image fills it:
  `"pad"` (default) centers it, aspect kept, with transparent pixels, `"stretch"` is the
  spec behaviour (warns when the aspect changes by > 2 %), `"crop"` crops the input centered
  to the canvas aspect before downscaling, so the downscalers stay untouched. Seamless mode
  always stretches (padding or cropping would break the wrap). `tile_size` larger than the
  longest derived edge is a `ValueError`. `stats.fit` and `stats.content_size` record it.
- Seamless padding is 2 output pixels per side, i.e. `2·rx` × `2·ry` input pixels per axis.
- Preprocessing fills transparent pixels with the nearest opaque color before denoising,
  uses `mode="edge"` for the bilateral filter, and repeats inputs that are smaller than the
  output by an integer factor.
- `convert` writes a fifth file, `NAME_palette.hex`.
- `scale` and `canvas` are not in the spec. Without them `batch` sizes each frame from its own
  crop box, so frames of one animation came out at different scales; it now sets one `scale`
  from the union of the frames' crop boxes and a shared `canvas` (`cli.shared_scale`).
- Input loading (not in the spec) only enables Pillow's PNG, JPEG, GIF, WEBP, BMP and TIFF
  decoders, so content under a misleading name cannot reach other plugins (EPS would run
  Ghostscript). It applies the EXIF orientation, rescales 16-bit grayscale to 8 bit instead
  of clipping it, uses the first frame of animations (with a warning), refuses non-regular
  files and enforces a pixel budget (`io.MAX_INPUT_PIXELS`) before decoding. A small input
  whose integer repeat would exceed that budget is refused.
- `Config` has three fields that are not in the spec, `key_bg`, `key_bg_tolerance` and
  `key_bg_fringe` (see Install), and the sprite preset sets `key_bg=True`. After keying, up to
  `key_bg_fringe` passes estimate each edge pixel's subject coverage as
  `max_c|p−bg| / max_c|fg−bg|` (fg sampled just beyond a 3-px fringe) and key out those below
  0.5, so the anti-aliased halo does not take palette entries. The gallery runs the fixtures with
  `key_bg=False` because they are test patterns, not sprites; images in `samples/` use the
  preset as-is.

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
