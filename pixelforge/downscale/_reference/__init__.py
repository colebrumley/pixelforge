"""Pure-numpy reference implementations of the downscalers' hot loops.

These are the implementations the native kernels in `pixelforge._native` replaced. They are
kept for tests (tolerance comparisons of the native per-pixel sums) and benchmarks only and
are never selected at runtime: two implementations cannot be bit-identical (summation order,
`exp`), and the output contract is byte-identical PNGs (docs/rust-port-plan.md, Decision 1).
"""
