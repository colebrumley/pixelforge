# pixelforge integration tests

`cargo test -p pixelforge` runs them; `cargo test -p pixelforge -- --include-ignored` adds
the slow group (each `#[ignore]` carries a `// slow` comment).

`common/mod.rs` holds shared helpers: fixtures read from `tests/fixtures/*.png`, Config
construction, pipeline runs on arrays, temporary directories.

The `*_oracle.rs` files compare against the reference outputs in `tests/oracle/` (see its
README). Tests that need random inputs use a local splitmix64 or xorshift generator with a
fixed seed.
