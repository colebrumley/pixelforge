# Development

```bash
git clone https://github.com/colebrumley/pixelforge && cd pixelforge
cargo test --workspace -- --include-ignored   # full suite, including the slow kopf cases
cargo test --workspace                        # without them
```

The `xtask` crate holds the developer tooling. `cargo xtask <command> --help` lists each
command's options.

| command | what it does |
| --- | --- |
| `cargo xtask fixtures --check` | compare the committed `tests/fixtures/` with the synthetic test images drawn from code (`cargo xtask fixtures DIR` writes them) |
| `cargo xtask golden` | regenerate `tests/golden.json` after an intentional output change; commit the diff with the change. `--check` only reports whether it is up to date |
| `cargo xtask bench` | time every method and preset on a few fixtures, print the per-stage timings and write `bench/<git sha>.json` (`--compare bench/OLD.json` for speedups) |
| `cargo xtask snapshot save`, then `cargo xtask snapshot compare` after a change | per case, whether the output is identical, how many pixels differ and their ΔE (exit 1 if any differ) |
| `cargo xtask gallery` | render every method × preset on the fixtures plus any PNGs in `samples/` into `gallery/index.html` (`--quick` for smaller backgrounds) |
| `cargo xtask help-record` | record each command's `--help` in `tests/cli_help/`, which the CLI tests compare with the live output |

Further reading:

- [DESIGN.md](../DESIGN.md): how each method works, the implementation and its numerics, the
  full determinism contract, timings, and every deliberate deviation from the spec.
- [crates/pixelforge/README.md](../crates/pixelforge/README.md): the crate layout and the
  numerical rules for contributors.
- [editor.md](editor.md): the editor's requirements and design.
