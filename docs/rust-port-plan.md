# Rust port of the hot loops

Status: planning. Branch `claude/performance-optimization-e3bbdd`.

## Why

Measured on an Apple M1 Max, after the two hash-preserving numpy fixes on this branch:

| Input | Method | Downscale | Where it goes |
|---|---|---|---|
| portrait 1024² → 256² | kopf | ≈34 s, 24 iterations | 16 kernel slots × ~60 full-image numpy passes + ~315 block sums per iteration |
| portrait 1024² → 256² | gerstner | ≈29 s, 139 iterations | SLIC `_assign` 15 s, `bilateral_filter` 6.5 s, annealer 2.4 s |
| knight 456×528 → 56×64 | kopf | ≈8 s, 50 iterations (cap) | same shape, smaller arrays |

Everything is memory-bandwidth-bound numpy over arrays 16× the image. A fused per-pixel
loop removes the intermediates entirely. A Rust prototype of kopf's E-step + M-step
(`docs/prototypes/kopf_em_step/`: `lib.rs`, `Cargo.toml`, and the ctypes benchmark/check scripts), called through ctypes on the real preprocessed inputs:

| Input | Python E+M | Rust, 1 thread | Rust, 8 threads |
|---|---|---|---|
| portrait | 843 ms | 162 ms (5.2×) | 28 ms (30×) |
| knight | 86 ms | 11.6 ms (7.4×) | 2.8 ms (30×) |

Agreement with the Python sums: max |Δ| / max|sum| ≈ 1e-15 per quantity, derived μ/Σ/ν
updates within 4e-15, identical set of fed kernels, bitwise reproducible run to run on
8 threads. The E+M step is ~55 % of a kopf iteration; the C-step and the gerstner loops
have the same structure. Expected end state: portrait kopf ≈1 s threaded / ≈5 s single
thread, gerstner similar, with skimage's 1.35 s bilateral denoise in preprocess as the floor.

## Decisions

1. **Required native extension, no runtime fallback.** Two implementations cannot be
   bit-identical (summation order), and the contract is byte-identical PNGs. The pure-Python
   kernels are kept as a *reference* for tests and benchmarks, never selected at runtime.
2. **PyO3 + maturin**, crate at `rust/` (or `native/`), Python module `pixelforge._native`.
   `abi3-py311` so one wheel per platform covers 3.11–3.13. Orchestration, config, I/O,
   preprocess, palette, postprocess, tiles stay in Python.
3. **Determinism rules for Rust code** (amend DESIGN.md §Determinism and REQUIREMENTS.md §3.2):
   - float64 only; no `fast-math`, no `target-cpu=native`; rustc does not contract to FMA on
     its own, so leave it that way (no `mul_add` unless used unconditionally).
   - `exp` from the pure-Rust `libm` crate, not `f64::exp` (which calls the platform libm and
     can differ in the last bit between macOS and Linux). `sqrt` is IEEE-exact everywhere.
   - Parallelism only over a **fixed partition that does not depend on the thread count**:
     e.g. bands of 4 output-kernel rows, each with a private accumulator, merged in band
     order. Threads pick up bands; 1 thread and 16 threads produce the same bits. No atomics,
     no dynamic chunking, no reductions whose order depends on scheduling. Thread count via
     `PIXELFORGE_THREADS` (default: available parallelism); it affects speed only.
   - No RNG, no hash-map iteration order in anything numeric.
4. **Golden hashes are regenerated once** per phase that changes numerics
   (`scripts/update_golden.py`), with a documented reason in CHANGELOG.md. The new goldens
   must pass on both CI platforms (macOS arm64 and Linux x86-64), which is the cross-platform
   check the project already relies on.
5. **The Python version after the two fixes is the baseline** for behavior (tolerance
   comparison of sums and of final indices) and for performance (same inputs, same machine).

## Phases

Each phase is one self-contained task for an Opus 5.5 subagent at medium effort; the Fable
session reviews results and runs the acceptance checks. Phases 2–4 each end with green CI,
regenerated goldens where noted, and a benchmark line.

### Phase 0 — Baseline capture (Python, no Rust)

- `scripts/bench.py`: runs box/kopf/gerstner on `tests/fixtures/*` and on two full-size
  inputs (the knight sprite and the portrait background), prints the pipeline's per-stage
  timings as a table, writes `bench/<git-sha>.json`. Deterministic; no RNG.
- `scripts/reference_outputs.py`: saves `indices`, `palette`, and (for kopf) the final
  `nu`, `mu`, `sigma`, `weight_sum` of the Python kernels as `.npz` per case, for tolerance
  comparisons in later phases. Store under `tests/reference/` only if small; otherwise
  generate on demand in the test from the kept Python reference code.
- Move the current numpy kopf/gerstner kernels to `pixelforge/downscale/_reference/` once
  the native versions land (Phase 2/3); until then they are the production code.

Acceptance: `bench.py` reproduces the table above within noise.

### Phase 1 — Scaffold

- `rust/Cargo.toml`, `rust/src/lib.rs` with a `#[pymodule] _native` exposing one trivial
  function; `pyproject.toml` switches `build-system` to maturin (`python-source = "."`,
  `module-name = "pixelforge._native"`), `uv sync --locked` builds it in a dev checkout.
- CI: add a `wheels` job (maturin-action) building macOS arm64 and manylinux x86-64 abi3
  wheels, run the test matrix against the built wheel, upload wheels as workflow artifacts
  and, on tags, as GitHub release assets. Keep the existing lint/test jobs.
- README: install from the release wheel URL for users without a Rust toolchain; the git
  URL install now needs `cargo`. `uv.lock` updated.

Acceptance: CI green on all four matrix entries with the extension imported in a test.

### Phase 2 — kopf

Signatures (all arrays C-contiguous float64 unless noted; Python passes views, Rust writes
into preallocated outputs):

1. `kopf_em_step(hi, wi, ho, wo, channels[3,hi,wi], mask[u8], pixel_weight|None, mu_x, mu_y,
   inv_a, inv_b, inv_c, nu[ho,wo,3], inv_color, ghost[u8]) -> sums[ho,wo,9]` plus, for the
   C-step, the per-slot γ is *not* materialised: the function also returns the eight
   one-sided/two-sided directional sums (`plus`, `both` for 4 axes) and the four overlap
   sums `f`, computed in the same pass using the constrained μ passed in — so the step takes
   `old_mu`, `move` (μ − old μ) and returns everything the C-step needs. (The prototype only
   does the first 9 sums; extend it.)
2. `kopf_spatial_refill(...)` for the final starved-kernel recolouring (spatial-only γ).
3. Python `kopf.run` keeps INITIALIZE, the vectorised M/C-step algebra on (ho, wo) arrays
   (Laplacian smoothing, clamps, eigh, σ growth, convergence test, logging).

Tests: compare each Rust function against the kept Python reference on the fixtures with
`max|Δ|/max|ref| < 1e-12` per quantity; full-pipeline comparison of final `indices` against
the Python baseline reporting the number of differing pixels and mean ΔE (expect a handful
of pixels at most, ΔE differences only at quantisation boundaries); thread-count invariance
(`PIXELFORGE_THREADS=1` vs `8` byte-identical PNGs). Regenerate goldens. Benchmark.

Acceptance: portrait kopf ≤ 6 s single thread, ≤ 1.5 s at 8 threads; knight ≤ 1 s.

### Phase 3 — gerstner

1. `slic_assign(...)`: the body of `_assign` as one pass per pixel over the (2r+1)² candidate
   cells with the same tie rule (ascending s, strict `<`) and the same window test; returns
   `best_s`. Both the colour-aware and the spatial-only (`windowed=False`) variants.
2. `bilateral_filter(...)`: fused per-pixel window loop with `libm::exp`.
3. Optional: the annealer's `d2`/`logp` block in `PaletteAnnealer.step` if it shows up after
   1 and 2 (it is 2.4 s on the portrait today).

Same test pattern as Phase 2; regenerate goldens.

Acceptance: portrait gerstner ≤ 5 s single thread, ≤ 2 s at 8 threads.

### Phase 4 — Threading and hardening

- Fixed-band parallelism (Decision 3) in every kernel; a test asserts bitwise equality of
  outputs for thread counts 1, 2, 3, 8 on all fixtures.
- `PIXELFORGE_THREADS` documented; `_meta.json` records the thread count under
  `environment` (recorded, never hashed, like numpy version today).
- Memory: no (16, hi, wi) γ array any more; check peak RSS on the 1024² case drops
  accordingly and raise `io.MAX_INPUT_PIXELS` only if the user wants it.

### Phase 5 — Docs and release

- DESIGN.md: determinism contract amendments, new timings table, a section on the native
  layer and how the Python reference is used in tests.
- README: install paths, toolchain note, `PIXELFORGE_THREADS`.
- CHANGELOG.md entries per phase. Version bump (this changes `Config.hash()`, which is why
  the goldens change anyway).

### Later, separately decided

- Replace skimage's `denoise_bilateral` (1.35 s on 1024², now the floor) with the Rust
  bilateral filter. It changes output relative to skimage and swaps a dependency's numerics,
  so it is a product decision, not part of this port.

## Risks

- **Cross-platform bits.** `libm` exp is deterministic across platforms; numpy's SIMD exp
  evidently is too for these inputs (shared goldens pass on both CI OSes today). After the
  port the Rust goldens must pass on both; if they do not, the culprit is almost certainly a
  stray `f64::exp`/`powf` or an FMA contraction, both avoidable.
- **Toolchain for contributors and git-URL installs.** Mitigated by CI wheels; documented.
- **Behavioural drift.** All numerics are the same formulas; tolerance tests against the kept
  Python reference catch mistakes, and the golden regeneration is reviewed diff by diff
  (`scripts/update_golden.py` output + differing-pixel counts).
- **Scope creep into the algebra.** The C-step vectorised algebra, convergence logic and all
  DEVIATION comments stay in Python; only the per-pixel loops move.
