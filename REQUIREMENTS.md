# pixelforge — Requirements for a Python Prototype

Deterministic conversion of arbitrary raster images (especially AI-generated ones) into
16-bit-style pixel art: sprites and backgrounds/tiles.

This document is the complete specification. An agent should be able to build the
prototype from this file alone, with no further clarification. Where the spec says
MUST, it is a hard acceptance criterion. Where it says SHOULD, implement it unless it
blocks a MUST. Where it says MAY, it is optional and should be left out if time is
short. Section 12 lists the acceptance tests in order; the build is done when all of
them pass.

---

## 1. Goals and non-goals

### Goals

1. Given any input image and a config, produce a pixel-art PNG at a small target
   resolution with a small palette. Same input + same config = byte-identical output,
   on any machine, every run.
2. Implement the two research methods that existing converters do not: the
   Kopf–Shamir–Peers content-adaptive downscaler (SIGGRAPH Asia 2013) and the
   Gerstner et al. joint superpixel + palette optimization ("Pixelated Image
   Abstraction", Computers & Graphics 2013). Both are specified in full in Sections 6
   and 7 of this document; the agent does not need the papers.
3. Provide a naive baseline (box downsample + median-cut) so the research methods can
   be compared against it on the same inputs.
4. Produce pixel-artist-grade output through post-processing passes (orphan removal,
   outline, selective ordered dither, palette ramp saturation) that existing tools skip.
5. Support two presets, `sprite` and `background`, plus `tileset` extraction for
   backgrounds.
6. Ship as an installable Python package with a CLI, a library API, tests, and a
   comparison gallery script.

### Non-goals (prototype)

- No GUI. No web UI. No WASM. No Rust.
- No generative models. No network calls at runtime.
- No animation/video input. (Shared-palette batch mode across still frames is in scope;
  see Section 9.)
- No GPU requirement. Everything runs on CPU with numpy. Target: a 1024×1024 input
  converts to 64×64 in under 60 s on a laptop for the slowest method.

---

## 2. Stack and project layout

- Python 3.11+. Dependencies: `numpy`, `Pillow`, `scipy` (for `ndimage` and
  `cKDTree`), `scikit-image` (for `color.rgb2lab`/`lab2rgb`, `restoration.denoise_bilateral`,
  `segmentation.slic` is NOT used — SLIC is implemented by hand per Section 7), `click`
  for the CLI, `pytest` for tests. Optional: `rembg` for background removal, behind an
  extra `[bg]`; if not installed, the `--remove-bg` flag errors with a clear message.
- Package manager: `uv` if available, otherwise `pip` with `pyproject.toml`. Use
  `pyproject.toml` with a `[project]` table; no `setup.py`.
- Layout:

```
pixelforge/
  pyproject.toml
  README.md
  REQUIREMENTS.md            # this file, copied in verbatim
  pixelforge/
    __init__.py
    config.py                # Config dataclass, presets, validation, hashing
    color.py                 # sRGB<->CIELAB, palette distance helpers
    io.py                    # load/save, alpha handling, metadata embedding
    preprocess.py            # normalize, denoise, crop, bg removal
    downscale/
      __init__.py
      box.py                 # baseline
      kopf.py                # content-adaptive (Section 6)
      gerstner.py            # superpixel+MCDA joint method (Section 7)
    palette.py               # median-cut, MCDA standalone, hardware palettes, ramps
    quantize.py              # map image to palette, ordered dither
    postprocess.py           # orphan removal, jaggy cleanup, outline, saturation
    tiles.py                 # tileset/tilemap extraction, seamless mode
    pipeline.py              # orchestrates stages; the only thing the CLI calls
    cli.py
  tests/
    fixtures/                # generated synthetic test images (see Section 12)
    test_determinism.py
    test_color.py
    test_box.py
    test_kopf.py
    test_gerstner.py
    test_palette.py
    test_postprocess.py
    test_tiles.py
    test_pipeline.py
  scripts/
    make_fixtures.py         # generates tests/fixtures deterministically
    gallery.py               # runs all methods on all fixtures -> gallery/index.html
```

---

## 3. Determinism contract (MUST)

1. No unseeded randomness anywhere. `random`, `numpy.random`, and `os.urandom` MUST NOT
   be called. The only permitted RNG is `numpy.random.default_rng(config.seed)` and it
   MUST only be used in the one place noted in Section 7 (MCDA sub-cluster
   perturbation), and only when `config.seed` is set; the default perturbation is
   deterministic without an RNG.
2. No dependence on dict/set iteration order of unordered inputs, thread scheduling,
   or wall-clock time. No multithreading inside numerical kernels unless results are
   bitwise identical to the single-threaded path (so: do not use multithreading).
3. Floating-point: all arrays are `float64` internally. Reductions use numpy's default
   pairwise summation on contiguous arrays; do not switch between `np.sum` and manual
   loops for the same quantity in different code paths. Final color conversion rounds
   with `np.round(...).astype(np.uint8)` after clipping to [0, 255], never truncation.
4. `Config.hash()` returns the SHA-256 of the canonical JSON of the config (sorted keys,
   no whitespace) and the pixelforge version string. The output PNG MUST carry a
   `tEXt` chunk `pixelforge:config` (the canonical JSON) and `pixelforge:input_sha256`
   (SHA-256 of the input file bytes).
5. Acceptance: running the CLI twice on the same input and config MUST produce
   byte-identical PNGs. `pytest tests/test_determinism.py` enforces this for every
   method and preset.

---

## 4. Config

A single frozen dataclass `Config` in `config.py`. All fields have defaults. The CLI
exposes every field as a flag with the same name (underscores → hyphens). Presets are
plain dicts that override defaults; CLI order is: defaults ← preset ← explicit flags.

```python
@dataclass(frozen=True)
class Config:
    # --- target ---
    preset: str = "sprite"            # "sprite" | "background"
    out_width: int | None = None      # if None, derived from out_height and aspect
    out_height: int | None = None     # if both None: sprite→64 on longest edge, background→256 on longest edge
    # --- preprocess ---
    remove_bg: bool = False           # sprite preset default True if rembg installed, else False
    crop_to_alpha: bool = True        # crop to alpha bbox before downscale (only if alpha present)
    alpha_threshold: int = 128        # input alpha < this → transparent
    denoise: str = "bilateral"        # "none" | "bilateral" | "median"
    denoise_sigma_color: float = 0.08 # bilateral, in [0,1] sRGB units
    denoise_sigma_spatial: float = 2.0
    # --- downscale ---
    method: str = "gerstner"          # "box" | "kopf" | "gerstner"
    kopf_max_iters: int = 50
    kopf_tol: float = 1e-3
    # --- palette ---
    palette_size: int = 16            # K. Ignored if palette_name set.
    palette_name: str | None = None   # "nes" | "gameboy" | "pico8" | "snes15" | "genesis" | path to .hex/.gpl
    palette_source: str = "auto"      # "auto": gerstner→MCDA, others→median_cut; or force "median_cut" | "mcda"
    palette_ramps: bool = True        # post-cluster hue-ramp regularization (Section 8.1)
    saturation_beta: float = 1.1      # Gerstner β; multiply a,b by this after palette convergence
    # --- gerstner ---
    g_m: float = 45.0                 # SLIC compactness
    g_alpha: float = 0.7              # temperature decay
    g_T_final: float = 1.0
    g_eps_palette: float = 1.0        # total palette change (LAB) below which "converged"
    g_eps_cluster: float = 0.25       # sub-cluster separation (LAB) to split
    g_laplacian: float = 0.4          # superpixel center smoothing fraction
    g_bilateral_sigma_color: float = 10.0   # on the wout×hout mean-color image, LAB units
    g_bilateral_sigma_spatial: float = 1.5
    g_max_iters: int = 2000           # safety cap on total iterations
    # --- dither ---
    dither: str = "auto"              # "none" | "bayer4" | "bayer8" | "auto"
    dither_strength: float = 0.5      # 0..1, scales the threshold matrix amplitude
    dither_variance_threshold: float = 4.0  # "auto": dither only where local LAB std < this
    # --- postprocess ---
    remove_orphans: bool = True
    orphan_min_region: int = 2        # connected regions (8-conn) with < this many px get merged
    fix_jaggies: bool = True
    outline: str = "auto"             # "none" | "auto" | "#rrggbb"  (sprite default "auto", background "none")
    outline_darken: float = 0.55      # auto outline = darkest palette color blended toward black by this
    # --- tiles (background only) ---
    tile_size: int = 16
    tileset: bool = False             # emit tileset.png + tilemap.json
    tile_dedupe_tolerance: float = 2.0  # mean LAB distance between tiles to consider identical
    seamless: bool = False            # wrap-around filtering so output tiles seamlessly
    # --- misc ---
    seed: int | None = None           # only used for optional MCDA jitter; None = fully analytic
    scale_preview: int = 8            # nearest-neighbor upscale factor for *_preview.png
```

Presets:

```python
PRESETS = {
  "sprite":     dict(remove_bg=None, crop_to_alpha=True, method="gerstner", palette_size=16,
                     dither="none", outline="auto", tileset=False, denoise="bilateral"),
  "background": dict(remove_bg=False, crop_to_alpha=False, method="kopf", palette_size=32,
                     dither="auto", outline="none", tileset=True, denoise="bilateral"),
}
```

`remove_bg=None` means "True if rembg importable else False", resolved at config build
time and stored as a bool in the canonical JSON.

Validation (raise `ValueError` with a specific message): `out_width`/`out_height` ≥ 8;
`palette_size` in [2, 256]; `method` in the allowed set; `tile_size` divides both output
dims when `tileset=True`; `g_alpha` in (0,1).

---

## 5. Pipeline (pipeline.py)

`run(input_path, config) -> Result` where `Result` has: `image` (H×W×4 uint8 RGBA at
native pixel-art resolution), `palette` (K×3 uint8), `indices` (H×W int, -1 for
transparent), `tileset`/`tilemap` (optional), `stats` (dict: iterations, timings per
stage, final palette size, config hash).

Stages run in this exact order:

1. **Load** (`io.load`): open with Pillow, convert to RGBA, alpha → float mask.
   Record input SHA-256.
2. **Preprocess** (`preprocess.run`):
   a. If `remove_bg`: run rembg, replace alpha.
   b. Binarize alpha with `alpha_threshold` → `mask` (bool). Pixels outside mask are
      excluded from every statistic below (palette, superpixels, kernels).
   c. If `crop_to_alpha` and the mask is not all-true: crop to the mask bounding box
      padded by 1 input pixel.
   d. Denoise RGB (not alpha) per `denoise`. Bilateral uses
      `skimage.restoration.denoise_bilateral(img, sigma_color, sigma_spatial, channel_axis=-1)`.
   e. Convert to CIELAB (`color.rgb2lab`, D65, sRGB input in [0,1]). All downstream
      math is in LAB. L in [0,100], a/b roughly [-128,127].
   f. Resolve output dims: if neither given, longest edge → preset default; preserve
      aspect ratio, round the other dim with `round()`, min 8. If `tileset`, round
      both dims up to a multiple of `tile_size`.
3. **Downscale + palette** (`downscale.<method>.run`): returns `small_lab`
   (hout×wout×3), `small_mask` (hout×wout bool), and for gerstner also the converged
   `palette_lab` and `indices`. For box/kopf, the palette is built afterwards by
   `palette.build(small_lab[small_mask], config)`.
4. **Quantize** (`quantize.run`): map every opaque output pixel to the nearest palette
   color in LAB (Euclidean, i.e. CIE76 ΔE). If dithering applies, add the ordered
   threshold offset before nearest-color lookup (Section 8.2). Gerstner skips this
   (it already outputs indices) unless `dither != "none"`, in which case dither is
   applied to `small_lab` against the gerstner palette.
5. **Postprocess** (`postprocess.run`): ramp regularization, saturation, orphan
   removal, jaggy fix, outline. Operates on `indices` + `palette_lab`.
6. **Tiles** (`tiles.run`) if `tileset`.
7. **Save** (`io.save`): `out.png` (native res, RGBA, palette-indexed PNG if ≤256
   colors, else RGBA), `out_preview.png` (nearest-neighbor ×`scale_preview`),
   `out_palette.json` (list of `#rrggbb` + LAB), `out_meta.json` (stats, config,
   input hash), and if tileset: `out_tileset.png`, `out_tilemap.json`.

---

## 6. Content-adaptive downscaling (Kopf, Shamir, Peers 2013) — `downscale/kopf.py`

This is a constrained EM over one anisotropic Gaussian kernel per output pixel. The
kernels move and reshape to align with image features so edges stay sharp after
downscaling. Implement exactly this; the symbols match the paper's supplementary
pseudocode.

### Coordinate convention (important)

All spatial quantities in this section — kernel grid positions, μ, Σ, input pixel
positions `pi`, the clamp box, and the `Rk` window — are expressed in **output-pixel
units**. That is, input pixel `(x, y)` has position `pi = ((x + 0.5)/rx, (y + 0.5)/ry)`
and each output pixel is a unit square. With this convention the paper's constants
(Σ init `1/3`, singular-value clamp `[0.05, 0.1]`, clamp box `±1/4`, `Rk` half-width
`2`, directional-variance threshold `0.2`) are self-consistent and need no rescaling.
Do not implement this in input-pixel units.

### Symbols

- Input image size `(wi, hi)`, output `(wo, ho)`, ratios `rx = wi/wo`, `ry = hi/ho`.
- Kernel index `k ∈ [0, wo·ho)`, grid position `(xk, yk) = ((k mod wo) + 0.5, floor(k/wo) + 0.5)`
  (output units; the center of output pixel k).
- Per kernel: spatial mean `μk` (2-vector), spatial covariance `Σk` (2×2), color mean
  `νk` (3-vector, unit-cube color — see Normalization), color variance `σk` (scalar).
- `Rk` = set of input pixels whose position satisfies `|px − xk| < 2` and `|py − yk| < 2`
  (a window of `4rx × 4ry` input pixels). Kernel values are zero outside `Rk`.
- `pi` = input pixel position in output units (above); `ci` = its color.
- `N4k`, `N8k` = 4- and 8-neighbor kernel indices (in the output grid), excluding
  out-of-bounds.

### Color normalization

The paper's `σk = 1e-4` initial color variance and `ν = (0.5,0.5,0.5)` assume colors
in a unit cube. Convert LAB to a unit cube before this algorithm:
`c = (L/100, (a+128)/255, (b+128)/255)`. Convert back after.

### Algorithm

```
INITIALIZE:
  for all k:
    μk = (xk, yk)
    Σk = [[1/3, 0], [0, 1/3]]             # used directly as the covariance matrix (output units²)
    νk = (0.5, 0.5, 0.5)
    σk = 1e-4

loop up to kopf_max_iters:
  E-STEP:
    for all k, for i in Rk:
      wk(i) = exp( -0.5 (pi-μk)ᵀ Σk⁻¹ (pi-μk)  -  ||ci-νk||² / (2 σk²) )
    normalize each kernel: wk(i) /= Σ_{i∈Rk} wk(i)          (guard: if sum==0, set uniform over Rk)
    normalize per pixel:   γk(i) = wk(i) / Σ_n wn(i)          (sum over all kernels n with i∈Rn; guard div-by-0 → 0)
  M-STEP:
    for all k:
      wsum = Σ_i γk(i)
      Σk  = (1/wsum) Σ_i γk(i) (pi-μk)(pi-μk)ᵀ      # uses the OLD μk, computed before updating μk
      μk  = (1/wsum) Σ_i γk(i) pi
      νk  = (1/wsum) Σ_i γk(i) ci
  C-STEP:
    # spatial: Laplacian smoothing then clamp to a box around the grid position
    μ'k = mean of μn over n ∈ N4k                            (computed from the M-step μ for all k first, then applied)
    μk  = clampBox( 0.5 μk + 0.5 μ'k,  box centered (xk, yk) with half-size (1/4, 1/4) )
    # covariance: clamp singular values (output units²)
    U,S,Vᵀ = SVD(Σk);  S11 = clamp(S11, 0.05, 0.1);  S22 = clamp(S22, 0.05, 0.1);  Σk = U S Vᵀ
    # shape constraints: detect dominant kernels and staircasing
    for all k, for n ∈ N8k:
      d = (xn - xk, yn - yk), normalized to unit length
      s = Σ_{i∈Rk} γk(i) · max(0, (pi-μk)ᵀ d)²              # directional variance toward n
      f = Σ_{i∈Rk} γk(i) γn(i)                               # overlap/edge strength
      o = Σ_{i∈Rk} ∇( γk(i) / (γk(i)+γn(i)) )               # orientation: np.gradient over the Rk window, summed → 2-vector
      if s > 0.2  or  (f < 0.08 and angle(d, o) > 25°):
         σk *= 1.1 ;  σn *= 1.1
  CONVERGENCE: stop if max over k of |Δμk| < kopf_tol and |Δνk| < kopf_tol and no σ changed in C-step.
OUTPUT: small_lab[k] = unnormalize(νk). small_mask[k] = (Σ_i γk(i)·mask_i / Σ_i γk(i)) ≥ 0.5.
```

### Implementation notes

- Vectorize over kernels: precompute for each kernel its `Rk` window slice; because
  all windows are the same size (`ceil(4rx) × ceil(4ry)`), stack them into an array
  of shape `(K, Wy, Wx)` using `numpy.lib.stride_tricks.sliding_window_view` on a
  padded input, or an explicit gather with precomputed index arrays. Memory for a
  1024² → 64² run: K=4096, window ≈ 64×64 → 16.7M floats per array; acceptable.
- Per-pixel normalization (`γ`) needs the sum over overlapping kernels; accumulate
  `wk` into a full-resolution buffer with `np.add.at` on the gather indices, then
  divide.
- Masked (transparent) input pixels: exclude from `Rk` (set their `wk` to 0).
- The σ growth in the C-step makes the kernel more spatially-dominated (less
  color-selective); cap σk at 1.0 to avoid runaway.
- Expected behavior test: a 256×256 image of a 2-px-wide diagonal black line on white
  downscaled to 32×32 must produce a connected 8-connected dark line (no gaps) in
  the output, where the box baseline produces a broken gray one. This is
  `tests/test_kopf.py::test_thin_line_stays_connected`.

---

## 7. Joint superpixel + palette optimization (Gerstner et al. 2013) — `downscale/gerstner.py`

This method segments the input into exactly `wo·ho` superpixels (one per output
pixel) and simultaneously builds a `K`-color palette by deterministic annealing. Each
iteration refines superpixels against the *current palette colors*, not raw means,
which is what makes the segmentation respect the final palette.

### Symbols

- Input pixels `p_i`, `i ∈ [1, M]`, with LAB color `c_i` (unnormalized LAB; L in
  0..100) and position `(x_i, y_i)`.
- Superpixels `p_s`, `s ∈ [1, N]`, `N = wo·ho`; each has center `(x_s, y_s)`, mean
  color `m_s`, smoothed color `m'_s`, and assigned palette index `k(s)`.
- Palette colors `c_k`, `k ∈ [1, K]`, each represented by two sub-clusters `c_k1`,
  `c_k2`; `c_k = (c_k1 + c_k2)/2` while sub-clusters exist.
- Temperature `T`.

### Algorithm 1

```
INITIALIZE:
  superpixel centers on a regular grid: (x_s, y_s) = ((sx+0.5)·rx, (sy+0.5)·ry)
  assign every input pixel to nearest center in (x,y)
  palette = [mean LAB of all masked input pixels]  (K_current = 1), with sub-clusters c_11 = c_12 = that mean
  P(c_1) = 1
  T = 1.1 · Tc, where Tc = 2 · (variance of input pixel colors along their first principal
      component axis in LAB)   # compute via np.linalg.eigh on the 3×3 covariance; largest eigenvalue
  P(p_s) = 1/N for all s (uniform prior; importance map not in scope)

iterate (count total iterations; stop at g_max_iters as a safety):
  1. SUPERPIXEL REFINEMENT (one modified-SLIC step):
       for each input pixel i, among superpixels s whose center is within a 2rx × 2ry window of p_i:
         d(p_i, p_s) = || c_i − c_{k(s)} ||  +  m · sqrt(N/M) · || (x_i,y_i) − (x_s,y_s) ||
         (color term uses the superpixel's ASSIGNED PALETTE COLOR, not m_s; m = g_m = 45)
       assign i to argmin_s d.  (Ties → lowest s.)
       Any superpixel with zero assigned pixels keeps its previous center and color.
       m_s = mean LAB of assigned pixels
       new center (x_s,y_s) = mean position of assigned pixels
       LAPLACIAN SMOOTHING: move each center g_laplacian (=0.4) of the way toward the mean
         of its 4-connected neighbors' centers (neighbors defined by the INITIAL grid, fixed forever).
         Compute all targets from the pre-smoothing centers, then apply.
       COLOR SMOOTHING: build a wo×ho×3 image of m_s (superpixel s at its output grid position);
         bilateral-filter it (sigma_color = g_bilateral_sigma_color in LAB units,
         sigma_spatial = g_bilateral_sigma_spatial, window = 2·ceil(2·sigma_spatial)+1,
         implemented directly in numpy — NOT skimage, whose bilateral assumes [0,1] ranges).
         Result is m'_s. Transparent output pixels (mask) are excluded from the filter.
  2. ASSOCIATE:
       for all s, k:  P(c_k | p_s) ∝ P(c_k) · exp( − || m'_s − c_k || / T ), normalized over k
         (compute in log space; subtract the max before exp for stability)
       P(c_k) = Σ_s P(c_k | p_s) P(p_s)
       k(s) = argmax_k P(c_k | p_s)       (ties → lowest k)
       (While sub-clusters exist, do the association against sub-clusters, i.e. treat each
        of the 2·K_current sub-clusters as its own "k" here and in REFINE; the displayed
        palette color c_k is the average of its pair. This is required for splitting to work.)
  3. REFINE:
       c_k = Σ_s m'_s · P(c_k | p_s) P(p_s) / P(c_k)      (for each sub-cluster)
  4. CONVERGENCE CHECK:
       if Σ_k || c_k(new) − c_k(old) || < g_eps_palette:
          T = g_alpha · T
          EXPAND (if K_current < K):
             for each palette color k: if || c_k1 − c_k2 || > g_eps_cluster:
                split: c_k1 and c_k2 each become their own palette color with their own new sub-cluster pair
                K_current += 1  (stop splitting when K_current == K; process k in ascending order)
          if K_current == K: drop sub-clusters; each color is a single cluster from now on
          else: for each color, set both sub-clusters to c_k, then perturb:
                c_k1 = c_k + δ·v,  c_k2 = c_k − δ·v,  where v is the unit first principal
                component of the LAB colors of the superpixels currently assigned to k
                (fallback: (1,0,0) if fewer than 2 superpixels), δ = 0.5 (LAB units).
                If config.seed is set, δ is multiplied by rng.uniform(0.9, 1.1) — this is
                the ONLY permitted RNG use in the codebase.
          if T ≤ g_T_final (=1.0) and K_current == K: BREAK
       else: continue

POST: apply saturation β to the palette (a,b *= saturation_beta), then
  output indices[s] = k(s), palette_lab = c_k, small_lab = c_{k(s)}.
  small_mask[s] = fraction of masked pixels in superpixel s ≥ 0.5.
```

### Implementation notes

- SLIC assignment is the hot loop. Vectorize: for each superpixel, consider the input
  pixels in its 2rx×2ry window; compute distances into a full-res `best_d` and
  `best_s` buffer, updating with `np.minimum`-style masks. Iterate over superpixels
  in ascending `s` so ties resolve deterministically (strict `<` comparison keeps the
  earlier s).
- `sqrt(N/M)` is constant; precompute.
- Masked input pixels are never assigned and never contribute.
- Keep `T` decay and splitting exactly as written; the paper's choice of
  `T_f = 1` is to avoid `exp` underflow with LAB units.
- Expected convergence: 100–400 total iterations for K=16. Record `stats["iterations"]`.
- Test (`tests/test_gerstner.py::test_two_color_image_gives_two_colors`): a 128×128
  image, left half pure red, right half pure blue, `K=2`, out 8×8 → palette has
  exactly 2 colors within ΔE 3 of red and blue, and columns 0–3 index one, 4–7 the
  other.
- Test (`test_palette_grows_to_K`): a smooth 6-hue gradient image, `K=6`, out
  16×16 → final `K_current == 6`.

---

## 8. Palette, quantization, post-processing

### 8.1 Palette (`palette.py`)

- `median_cut(lab_pixels, K) -> K×3`: classic median cut on LAB with the box split
  along the axis of largest range, split at the median (lower median for even counts),
  boxes chosen by largest pixel count. Deterministic by construction. Final color per
  box = mean.
- `mcda(lab_pixels, K, config) -> K×3`: the palette half of Section 7 applied directly
  to pixel colors (no superpixels) — used when `palette_source="mcda"` with box/kopf.
  Reuse the same functions as gerstner.py.
- Hardware palettes: ship as `.hex` text (one `rrggbb` per line) under
  `pixelforge/palettes/`: `nes` (54 colors, the standard 2C02 palette), `gameboy` (4
  greens: `0f380f 306230 8bac0f 9bbc0f`), `pico8` (16), `snes15` (a generic 15-color
  ramped palette; 1 slot reserved for transparency), `genesis` (a generic 16-color
  Mega Drive-style palette). Users may pass a path to `.hex` or GIMP `.gpl`.
  When a named palette is used, skip clustering and quantize directly to it.
- `regularize_ramps(palette_lab) -> palette_lab` (when `palette_ramps=True`): group
  palette colors by hue angle in LAB (`atan2(b, a)`) into bins 30° wide (achromatic
  colors with chroma < 8 go in a separate bin). Within each bin sort by L. For each
  adjacent pair in the sorted bin, nudge hue so that darker colors shift up to 6°
  toward blue/purple (hue angle 270°) and lighter ones up to 6° toward yellow (90°),
  proportional to their L distance from the bin's mean L, max 6°. This is the
  "hue-shifted ramp" convention pixel artists use. Keep L and chroma unchanged. Do
  not merge or add colors.

### 8.2 Quantize and dither (`quantize.py`)

- Nearest color: `scipy.spatial.cKDTree(palette_lab).query(pixels)`. Ties are
  resolved by cKDTree deterministically (lowest index); acceptable.
- Ordered dither: Bayer matrices `B4` (4×4) and `B8` (8×8), normalized to
  `(B/n² − 0.5)`. For output pixel `(x,y)`, perturb `L` by
  `dither_strength · 12 · B[y mod n, x mod n]` and `a`,`b` by half that, then take the
  nearest palette color. (12 ≈ a typical L step between ramp colors.)
- `dither="auto"`: compute local LAB std over a 3×3 window on `small_lab`; dither
  only where std < `dither_variance_threshold` (smooth gradients), not on edges or
  detailed regions.

### 8.3 Post-processing (`postprocess.py`)

All passes work on the index image `idx` (−1 = transparent) and are themselves
deterministic (raster order, fixed rules). Order: ramps → saturation → orphans →
jaggies → outline.

- **Saturation**: already applied in gerstner; for box/kopf apply here (`a,b *= β`).
  Clip to the sRGB gamut after conversion by clipping RGB to [0,1].
- **Orphan removal** (`remove_orphans`): label 8-connected regions of equal index
  (`scipy.ndimage.label` per index value). For any region with fewer than
  `orphan_min_region` pixels, reassign all its pixels to the index that is most
  frequent among its 8-neighbors (ties → the neighbor color nearest in LAB to the
  orphan's color; further ties → lowest index). Transparent pixels are never
  reassigned and never count as neighbors. Repeat until no change (max 5 passes).
- **Jaggy fix** (`fix_jaggies`): for each pixel, look at the 3×3 neighborhood. If the
  pixel and exactly one diagonal neighbor share an index A, and the two pixels
  orthogonally between them share a different index B, and both of those B pixels
  also have B on their outer side (i.e. the A pixels form a lone diagonal "step"
  across a B line), leave it. Only remove "doubles": a 2×1 or 1×2 run of A along a
  diagonal staircase where the staircase elsewhere is 1-px steps; replace the
  pixel of the double farther from the line's centroid with B. Implement
  conservatively; if unsure whether a pattern is a double, leave it. Test: a
  clean 1-px 45° line is unchanged; a 1-px line with one injected 2-px double has
  the double removed.
- **Outline** (`outline`): if `"auto"` or a color: for every transparent pixel that
  has at least one opaque 4-neighbor, set it to the outline color (added to the
  palette if not present; if that would exceed 256 colors, replace the nearest
  existing color). Auto color = darkest palette color (lowest L) blended toward
  LAB black by `outline_darken`. Then re-run orphan removal once (outline can
  create 1-px nubs). Outline is drawn *outside* the silhouette, so the sprite
  grows by up to 1 px per side; the canvas is padded by 1 before this pass if any
  opaque pixel touches the edge.

---

## 9. Tiles and seamless mode (`tiles.py`)

- `seamless=True`: before preprocessing, pad the input by wrapping (`np.pad(mode="wrap")`)
  by `2·max(rx,ry)` pixels on all sides; run the chosen downscale; crop the output
  back to `wo×ho`. For Kopf and Gerstner this means kernels/superpixels at the border
  see the wrapped content; this is what makes the tile repeat cleanly. For box, it's
  equivalent to no-op (fine).
- `tileset=True` (requires `tile_size | wo` and `tile_size | ho`): cut `idx` into
  `tile_size × tile_size` tiles in raster order. Hash each tile's index array
  (`sha256` of bytes). Exact-duplicate tiles merge. Then near-duplicates: for each
  pair of distinct tiles, compute mean ΔE between corresponding pixels (via palette
  LAB); if below `tile_dedupe_tolerance`, merge the later tile into the earlier one.
  Also test the 3 flips/rotations (hflip, vflip, both) and record the transform
  in the map if a match is found. Output `out_tileset.png` (tiles in a row-major
  grid, 16 tiles wide) and `out_tilemap.json`:

```json
{"tile_size":16,"width_tiles":16,"height_tiles":16,
 "tileset_columns":16,"tiles":[[tile_id, flip_h, flip_v], ...]}
```

- **Batch mode** (`cli batch`): given a directory of images and a config, build ONE
  palette from the union of all images' preprocessed LAB pixels (subsample every
  image to at most 50 000 pixels by a fixed stride, not random), then run each image
  with `palette_name` set to that shared palette (written to `shared_palette.hex`).
  This is what keeps animation frames from flickering.

---

## 10. CLI (`cli.py`)

```
pixelforge convert INPUT [-o OUTDIR] [--preset sprite|background] [--method box|kopf|gerstner]
                   [--out-width N] [--out-height N] [--palette-size K] [--palette-name NAME]
                   [... every Config field as --kebab-case ...] [--config config.json]
pixelforge batch INPUT_DIR [-o OUTDIR] [same flags]
pixelforge compare INPUT [-o OUTDIR]        # runs box, kopf, gerstner with the same config; writes a side-by-side PNG
pixelforge palettes                          # lists bundled palettes
```

`--config` loads a JSON of Config fields; explicit flags override it. `convert` prints
the output paths and `stats` as one JSON line on stdout. Exit code 0 on success, 2 on
config validation error, 1 on any other error, with the message on stderr.

---

## 11. Library API

```python
from pixelforge import Config, run, PRESETS
cfg = Config(preset="sprite", method="gerstner", palette_size=12, out_height=48)
res = run("hero.png", cfg)
res.save("out/hero")        # writes out/hero.png, hero_preview.png, hero_palette.json, hero_meta.json
res.image                   # numpy RGBA
res.palette                 # numpy K×3 uint8
```

---

## 12. Fixtures, tests, and acceptance criteria

### Fixtures (`scripts/make_fixtures.py`, deterministic, no RNG; writes PNGs)

1. `line_diag.png` 256×256: white, 2-px black diagonal line corner to corner.
2. `two_color.png` 128×128: left half (255,0,0), right half (0,0,255).
3. `gradient6.png` 256×256: six vertical bands, each a smooth gradient of one hue.
4. `circle_alpha.png` 256×256 RGBA: orange disc radius 100 with soft (anti-aliased,
   4-px feathered) alpha edge on transparent background, with a 20-px darker
   "eye" circle inside.
5. `checker_tiles.png` 256×256: a 16×16 checker pattern of two colors where every
   other tile row is hflipped — exercises tile dedupe with flips.
6. `noisy_gradient.png` 256×256: horizontal gray gradient plus a deterministic
   high-frequency pattern `(sin(37x) + cos(53y))·8` to imitate AI noise.
7. `seamless_src.png` 128×128: a smooth 2D sine color field with period = 128 so it
   wraps.

### Tests (all MUST pass; `pytest -q`)

- `test_color.py`: sRGB→LAB→sRGB round trip max error ≤ 1/255 for 1000 fixed colors
  (a deterministic lattice, not random); pure white → L=100±0.01.
- `test_determinism.py`: for each method × each preset × fixtures 1,4,6: run twice,
  assert PNG bytes identical. Also assert `Config.hash()` stable across runs.
- `test_box.py`: box downscale of a solid color image returns that color exactly.
- `test_kopf.py`: `test_thin_line_stays_connected` (Section 6); `test_converges`
  (iterations < kopf_max_iters on `two_color.png`); `test_mask_respected` (no opaque
  output pixel outside the disc's support in `circle_alpha.png` at 32×32).
- `test_gerstner.py`: `test_two_color_image_gives_two_colors`, `test_palette_grows_to_K`
  (Section 7); `test_iteration_cap` (hits `g_max_iters` gracefully with a tiny cap and
  still returns a valid result).
- `test_palette.py`: median cut on a 4-color image with K=4 recovers all 4 within
  ΔE 1; `regularize_ramps` preserves L and chroma to 1e-6 and changes hue by ≤ 6°;
  every bundled palette file parses and has the expected count.
- `test_postprocess.py`: orphan removal removes a single stray pixel in a flat field
  and nothing else; jaggy test (Section 8.3); outline adds exactly the 4-neighbor ring
  on a 5×5 square sprite.
- `test_tiles.py`: `checker_tiles.png` → tileset has exactly 2 tiles (or 1 with flip
  recorded) and the tilemap reconstructs the index image exactly; seamless run on
  `seamless_src.png` has mean ΔE between left/right edge columns and top/bottom rows
  < 3× the mean ΔE between arbitrary adjacent interior columns.
- `test_pipeline.py`: CLI `convert` on `circle_alpha.png` with sprite preset exits 0,
  writes all five files, the PNG has `pixelforge:config` metadata, and the output
  has ≤ `palette_size + 1` distinct opaque colors (the +1 is the outline).

### Gallery (`scripts/gallery.py`)

Runs box/kopf/gerstner × sprite/background on every fixture plus every PNG in an
optional `samples/` directory, writes `gallery/index.html` with a table: input,
box, kopf, gerstner (each shown at `scale_preview`), with timings and iteration
counts under each cell. This is how a human judges the quality gap. The build is
not done until the gallery renders and `README.md` links to it.

### Definition of done

1. `uv run pytest -q` (or `pip install -e . && pytest -q`) passes with zero failures
   and zero skips other than the `rembg` extra.
2. `pixelforge convert tests/fixtures/circle_alpha.png --preset sprite` and
   `pixelforge convert tests/fixtures/noisy_gradient.png --preset background` both
   succeed in under 60 s each.
3. `python scripts/gallery.py` renders without error.
4. `README.md` contains: install, the three CLI examples above, a one-paragraph
   description of each method with a citation line for Kopf et al. 2013 and
   Gerstner et al. 2013, and the determinism contract from Section 3.

---

## 13. Build order for an autonomous agent

Follow this order; each step has a test that proves it before moving on.

1. Scaffold package, `pyproject.toml`, `Config` + presets + validation + `hash()`.
   Write `make_fixtures.py` and generate fixtures. (`test_color`, config tests.)
2. `io.py`, `preprocess.py`, `color.py`. (`test_color`.)
3. `downscale/box.py`, `palette.median_cut`, `quantize.py` nearest-color, `pipeline.py`
   end-to-end with box only, `cli.py convert`. (`test_box`, `test_pipeline`.)
4. `postprocess.py` all passes. (`test_postprocess`.)
5. `downscale/kopf.py`. (`test_kopf`.) Profile; if the 1024²→64² run exceeds 60 s,
   reduce the `Rk` half-width from 2 to 1.5 output units and note it in README; do not drop the C-step.
6. `downscale/gerstner.py` including MCDA, then `palette.mcda` reusing it.
   (`test_gerstner`, `test_palette`.)
7. `quantize.py` dither + auto mode, `palette.regularize_ramps`, hardware palettes.
8. `tiles.py` seamless + tileset, `cli batch`, `cli compare`. (`test_tiles`.)
9. `test_determinism.py` across everything. Fix any nondeterminism found.
10. `gallery.py`, README, final full test run.

Do not stop at a partial step. If a MUST cannot be met, implement the closest
behavior, mark it with a `# DEVIATION:` comment that names the section of this
document, and list every deviation in the README under "Known deviations".

---

## 14. References (for the README; the agent does not need to fetch these)

- J. Kopf, A. Shamir, P. Peers. "Content-Adaptive Image Downscaling." ACM Trans.
  Graphics 32(6), SIGGRAPH Asia 2013. https://johanneskopf.de/publications/downscaling
- T. Gerstner, D. DeCarlo, M. Alexa, A. Finkelstein, Y. Gingold, A. Nealen.
  "Pixelated Image Abstraction with Integrated User Constraints." Computers &
  Graphics, 2013. https://cragl.cs.gmu.edu/pixelate/
- R. Achanta et al. "SLIC Superpixels." (used as modified in Gerstner.)
- K. Rose. "Deterministic annealing for clustering, compression, classification,
  regression, and related optimization problems." Proc. IEEE, 1998 (the MCDA source).
