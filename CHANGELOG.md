# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- MIT `LICENSE` and package metadata (license expression, author, classifiers, project URLs).
- `MANIFEST.in` so the sdist ships tests, fixtures, scripts, DESIGN.md and REQUIREMENTS.md.
- Ruff configuration and `ruff` in the `dev` dependency group.
- GitHub Actions CI: lint job plus tests on Ubuntu (Python 3.11/3.12/3.13) and macOS arm64 (3.12).
- This changelog.

### Removed

- Duplicate `dev` optional extra; use the `dev` dependency group (`uv sync`).

<!-- Later commits on this branch are appended here by the final integrator. -->

## [0.1.0]

- Initial prototype: box, kopf and gerstner downscalers, palette and post-processing pipeline,
  tileset/tilemap extraction, bundled hardware palettes, `pixelforge` CLI.
