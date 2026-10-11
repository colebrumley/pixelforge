# pixelforge integration tests

`cargo test -p pixelforge` runs them; `cargo test -p pixelforge -- --include-ignored` adds
the slow group (each `#[ignore]` carries a `// slow` comment).

The Python test suite these files were ported from was removed in 0.2.0 with the Python
package; the tables below map it for the record (the files are in the git history up to
1fff901).
`common/mod.rs` holds shared helpers: fixtures read from `tests/fixtures/*.png`, Config
construction, pipeline runs on arrays, temporary directories.

## Python test file → Rust file (Phase 8, part A)

| Python file | Rust file | Notes |
|---|---|---|
| `test_cli.py` | `cli.rs` | ported in Phase 7 |
| `test_pipeline.py` (CLI cases) | `cli.rs` | ported in Phase 7 |
| `test_pipeline.py` (library cases) | `pipeline.rs` | log records through a per-thread `log::Log` collector |
| `test_config.py` | `config.rs` | `config_oracle.rs` holds the recorded corpus |
| `test_color.py` | `color.rs` | `color_oracle.rs` holds the skimage lattice |
| `test_palette.py` | `palette.rs` | |
| `test_postprocess.py` | `postprocess.rs` | |
| `test_tiles.py` | `tiles.rs` | |
| `test_preprocess.py` | `preprocess.rs` | |
| `test_io.py` | `io.rs` | input files written byte by byte (no Pillow) |
| `test_box.py`, `test_kopf.py`, `test_gerstner.py` | `downscalers.rs` | shared names prefixed `box_`, `kopf_`, `gerstner_` |
| `test_branches.py` | spread out (below) | |
| `test_determinism.py` (in-process cases) | `determinism.rs` | kopf at preset size `#[ignore]` |
| `test_determinism.py` (subprocess cases) | `cli.rs` | ported in Phase 7, `PIXELFORGE_THREADS` 1/2/3/8 |
| `test_golden.py` | `golden.rs` | reads `tests/golden.json`; the 4 kopf 256-px cases `#[ignore]` |
| `test_native.py` (`thread_count` cases) | `parallel.rs` | the env-mutating cases in one test |
| `test_version.py` | `version.rs` | |

`test_branches.py`:
- `test_mcda_palette_source_with_box_and_kopf` → `pipeline.rs`
- `test_save_png_falls_back_to_rgba_at_256_colors_plus_transparency` → `io.rs`
- `test_median_denoise` and `test_remove_bg_missing_dependency_message` → `preprocess.rs`
- `test_seamless_with_gerstner` → `downscalers.rs`
- `test_batch_on_empty_directory` → `cli.rs`
- `test_bayer8_dither` and `test_zero_dither_strength_is_no_dither` → `quantize.rs`
- `test_tiles_extract_rejects_sizes_not_divisible_by_the_tile` → `tiles.rs`

Python tests that seed numpy's `default_rng` use a local splitmix64 or xorshift generator
with the same seed instead. The assertions and thresholds are unchanged.

## Not ported, deliberately

These files compare the Rust port against the Python code and go away with it:
- the `tests/test_*_native.py` files
- the `tests/*_reference.py` files
- `test_field_table.py`, `test_oracle.py`, `test_cli_parity.py`
- `test_logging_bridge.py`, `test_npsum.py`, `test_pyjson_native.py`
- `test_downscale_native.py`
- `test_native.py`, except its `thread_count` cases

`test_fixtures.py` and `test_bench_scripts.py` belong to `xtask`.

These individual cases or assertions are not ported, or only in part:

| Python test | Reason |
|---|---|
| `test_config.py::test_round_trip_hashable_and_frozen` (frozen dataclass, `hash()`) | Rust mutation is compile-time controlled; equality and `Config::hash()` are checked instead |
| `test_config.py::test_presets_are_read_only` (TypeError on mutation) | `PRESETS` is a `const`; the values are checked |
| `test_config.py::test_config_error_is_value_error_and_pixelforge_error` (class hierarchy) | it checks the conversion into `pipeline::Error::Config` and the `std::error::Error` impl instead |
| `test_color.py::test_native_conversion_errors`, `test_rgb8_to_lab_accepts_other_dtypes` | numpy dtype, shape and contiguity checks of the binding; the Rust API takes flat slices |
| `test_color.py::test_native_keeps_leading_shape`, `test_python_wrappers_accept_non_contiguous_and_lists`, `test_native_nearest_index_errors` | only the value checks and the empty-palette panic are ported; the binding's shape and dtype messages are not |
| `test_kopf.py::test_axis_selections_are_slices_when_downscaling`, `test_reduce_slices_match_fancy_indexing`, `test_contiguity_fallback_for_gapped_kernels` | numpy-reference internals (`_Axis`, `_Layout.reduce`) with no Rust counterpart |
| `test_gerstner.py::test_semi_transparent_superpixels_get_less_palette_mass` | Python records the annealer's prior through a monkeypatch on the reference; Rust's prior is internal. The run and `palette_prior` are checked |
| `test_gerstner.py::test_shared_grid_helpers` | the `_laplacian_smooth` and `_Grid` checks target private Rust helpers; `neighbor_mean4` is checked |
| `test_gerstner.py::test_reference_bilateral_filter_weight_sharing_is_bitwise` | runs against Rust's `gerstner::bilateral_filter` and a Rust port of the reference loop, not numpy |
| `test_branches.py::test_remove_bg_missing_dependency_message` | `rembg_available()` (Python import probe) has no counterpart; the message and the `run` error are checked (the latter in `pipeline_oracle.rs`) |
| `test_pipeline.py::test_run_accepts_array_and_pil_image` | PIL `Image` input is Python-only; path, uint8 and float arrays are ported |
| `test_pipeline.py::test_unknown_palette_fails_fast` | the monkeypatch proving no stage ran becomes "no pipeline log record before the error" |
| `test_pipeline.py::test_downscaler_receives_alpha_weight` | no hook into the downscaler call; the `Preprocessed.weight` the pipeline passes is checked instead |
| `test_pipeline.py::test_meta_environment_and_rounded_palette_lab` | the Rust `environment` keys are `platform`, `threads` and `native`; the `PIXELFORGE_THREADS=3` part is in `cli.rs` (`meta_environment_threads_from_env`) |
| `test_io.py::test_preprocessing_errors_still_importable_from_preprocess` | a shared class object has no Rust equivalent; it checks that one `PixelforgeError` converts into both modules' errors |
| `test_version.py::test_version_is_single_sourced` | installed distribution metadata is Python-only |
