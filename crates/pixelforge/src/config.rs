//! The Config field table: the single source of truth that the Rust `Config`, its canonical
//! JSON, the CLI flags and `--sweep` are built from (docs/rust-migration-plan.md, Decision 8).
//!
//! Transcribed from `pixelforge/config.py`: the `Config` dataclass fields in declaration
//! order (name, annotated type, default), `FIELD_HELP`, `FIELD_CHOICES`, `UPPER_BOUNDS`,
//! `PRESETS` and `PRESET_LONGEST_EDGE`. While the Python package exists,
//! `tests/test_field_table.py` compares this table with it field by field through
//! `pixelforge._native.config_fields()` / `config_presets()`, so the two cannot drift.
//!
//! Range checks other than the upper bounds (lower bounds, open intervals, cross-field rules)
//! live in `Config._validate` and move to Rust with the Config itself (Phase 2).

/// The base type of a field (its Python annotation without `| None`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldKind {
    Bool,
    Int,
    Float,
    Str,
}

impl FieldKind {
    /// The Python annotation of the base type: "bool", "int", "float" or "str".
    pub fn python_name(self) -> &'static str {
        match self {
            FieldKind::Bool => "bool",
            FieldKind::Int => "int",
            FieldKind::Float => "float",
            FieldKind::Str => "str",
        }
    }
}

/// A field value: a default or a preset override.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Value {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(&'static str),
}

impl Value {
    /// Whether this value fits a field of `kind` (`Value::None` only if `optional`).
    pub fn fits(&self, kind: FieldKind, optional: bool) -> bool {
        matches!(
            (self, kind),
            (Value::Bool(_), FieldKind::Bool)
                | (Value::Int(_), FieldKind::Int)
                | (Value::Float(_), FieldKind::Float)
                | (Value::Str(_), FieldKind::Str)
        ) || (optional && *self == Value::None)
    }
}

/// One Config field.
#[derive(Clone, Copy, Debug)]
pub struct FieldSpec {
    pub name: &'static str,
    pub kind: FieldKind,
    /// The annotation includes `| None`.
    pub optional: bool,
    pub default: Value,
    /// The one-line `--help` text.
    pub help: &'static str,
    /// The allowed values of an enum-like string field (`outline` also takes "#rrggbb" and
    /// has no entry here, as in `FIELD_CHOICES`).
    pub choices: Option<&'static [&'static str]>,
    /// Inclusive upper bound (`UPPER_BOUNDS`). An integer because every bound is one, and
    /// `seed`'s 2^63 − 1 is not representable as an f64.
    pub upper_bound: Option<i64>,
}

impl FieldSpec {
    /// The Python annotation, e.g. "int | None".
    pub fn python_type(&self) -> String {
        let base = self.kind.python_name();
        if self.optional {
            format!("{base} | None")
        } else {
            base.to_string()
        }
    }
}

pub const METHODS: &[&str] = &["box", "kopf", "gerstner"];
pub const DENOISERS: &[&str] = &["none", "bilateral", "median"];
pub const DITHERS: &[&str] = &["none", "bayer4", "bayer8", "auto"];
pub const PALETTE_SOURCES: &[&str] = &["auto", "median_cut", "mcda"];
pub const TRANSPARENT_INDICES: &[&str] = &["last", "first"];
pub const FITS: &[&str] = &["pad", "stretch", "crop"];
/// The preset names, in `PRESETS` order.
pub const PRESET_NAMES: &[&str] = &["sprite", "background"];

/// Every Config field, in the dataclass's declaration order.
pub const FIELDS: &[FieldSpec] = &[
    FieldSpec {
        name: "preset",
        kind: FieldKind::Str,
        optional: false,
        default: Value::Str("sprite"),
        help: "Starting values for the other fields: sprite or background.",
        choices: Some(PRESET_NAMES),
        upper_bound: None,
    },
    FieldSpec {
        name: "out_width",
        kind: FieldKind::Int,
        optional: true,
        default: Value::None,
        help: "Output width in pixels; height follows the aspect if not given.",
        choices: None,
        upper_bound: Some(4096),
    },
    FieldSpec {
        name: "out_height",
        kind: FieldKind::Int,
        optional: true,
        default: Value::None,
        help: "Output height in pixels; width follows the aspect if not given.",
        choices: None,
        upper_bound: Some(4096),
    },
    FieldSpec {
        name: "scale",
        kind: FieldKind::Float,
        optional: true,
        default: Value::None,
        help: "Fixed downscale factor: output = round(cropped input / scale).",
        choices: None,
        upper_bound: Some(4096),
    },
    FieldSpec {
        name: "canvas",
        kind: FieldKind::Str,
        optional: true,
        default: Value::None,
        help: "Fit the subject centered inside a WxH canvas (outline drawn inside).",
        choices: None,
        upper_bound: None,
    },
    FieldSpec {
        name: "remove_bg",
        kind: FieldKind::Bool,
        optional: false,
        default: Value::Bool(false),
        help: "Remove the background with rembg AI matting (opt-in, needs the extra).",
        choices: None,
        upper_bound: None,
    },
    FieldSpec {
        name: "key_bg",
        kind: FieldKind::Bool,
        optional: false,
        default: Value::Bool(false),
        help: "Key out a flat opaque background color.",
        choices: None,
        upper_bound: None,
    },
    FieldSpec {
        name: "key_bg_tolerance",
        kind: FieldKind::Float,
        optional: false,
        default: Value::Float(0.08),
        help: "Max per-channel sRGB distance from the background color, 0..1.",
        choices: None,
        upper_bound: None,
    },
    FieldSpec {
        name: "key_bg_fringe",
        kind: FieldKind::Int,
        optional: false,
        default: Value::Int(3),
        help: "Passes keying out the anti-aliased background edge (0 = off).",
        choices: None,
        upper_bound: None,
    },
    FieldSpec {
        name: "crop_to_alpha",
        kind: FieldKind::Bool,
        optional: false,
        default: Value::Bool(true),
        help: "Crop to the alpha bounding box before downscaling.",
        choices: None,
        upper_bound: None,
    },
    FieldSpec {
        name: "alpha_threshold",
        kind: FieldKind::Int,
        optional: false,
        default: Value::Int(128),
        help: "Input alpha below this is transparent (0..255).",
        choices: None,
        upper_bound: None,
    },
    FieldSpec {
        name: "denoise",
        kind: FieldKind::Str,
        optional: false,
        default: Value::Str("bilateral"),
        help: "Denoise filter applied before downscaling.",
        choices: Some(DENOISERS),
        upper_bound: None,
    },
    FieldSpec {
        name: "denoise_sigma_color",
        kind: FieldKind::Float,
        optional: false,
        default: Value::Float(0.08),
        help: "Bilateral color sigma in [0, 1] sRGB units.",
        choices: None,
        upper_bound: None,
    },
    FieldSpec {
        name: "denoise_sigma_spatial",
        kind: FieldKind::Float,
        optional: false,
        default: Value::Float(2.0),
        help: "Denoise spatial sigma in input pixels (after pre-reduction).",
        choices: None,
        upper_bound: Some(16),
    },
    FieldSpec {
        name: "prereduce_max_ratio",
        kind: FieldKind::Int,
        optional: false,
        default: Value::Int(8),
        help: "Box-reduce inputs larger than this many times the output first (0 = never).",
        choices: None,
        upper_bound: None,
    },
    FieldSpec {
        name: "enhance",
        kind: FieldKind::Float,
        optional: false,
        default: Value::Float(0.0),
        help: "Exaggerate features before downscaling: levels stretch plus local contrast, 0..4 (0 = off; for photos, try 0.5 to 1).",
        choices: None,
        upper_bound: Some(4),
    },
    FieldSpec {
        name: "enhance_black",
        kind: FieldKind::Float,
        optional: true,
        default: Value::None,
        help: "Brightness (largest channel, 0..1) the levels stretch maps to black; unset = the subject's 1st percentile.",
        choices: None,
        upper_bound: None,
    },
    FieldSpec {
        name: "enhance_white",
        kind: FieldKind::Float,
        optional: true,
        default: Value::None,
        help: "Brightness (largest channel, 0..1) the levels stretch maps to white; unset = the subject's 99th percentile.",
        choices: None,
        upper_bound: None,
    },
    FieldSpec {
        name: "enhance_radius",
        kind: FieldKind::Float,
        optional: false,
        default: Value::Float(2.0),
        help: "Feature size the local contrast acts on, in output pixels.",
        choices: None,
        upper_bound: Some(16),
    },
    FieldSpec {
        name: "ink",
        kind: FieldKind::Float,
        optional: false,
        default: Value::Float(0.0),
        help: "Darken thin dark features (lines, lashes, creases) before downscaling, 0..1.",
        choices: None,
        upper_bound: Some(1),
    },
    FieldSpec {
        name: "method",
        kind: FieldKind::Str,
        optional: false,
        default: Value::Str("gerstner"),
        help: "Downscaling algorithm.",
        choices: Some(METHODS),
        upper_bound: None,
    },
    FieldSpec {
        name: "kopf_max_iters",
        kind: FieldKind::Int,
        optional: false,
        default: Value::Int(50),
        help: "Iteration cap for the kopf downscaler.",
        choices: None,
        upper_bound: Some(1000),
    },
    FieldSpec {
        name: "kopf_tol",
        kind: FieldKind::Float,
        optional: false,
        default: Value::Float(1e-3),
        help: "kopf convergence tolerance (RMS color change in the unit cube).",
        choices: None,
        upper_bound: None,
    },
    FieldSpec {
        name: "palette_size",
        kind: FieldKind::Int,
        optional: false,
        default: Value::Int(16),
        help: "Number of palette colors K; ignored with --palette-name.",
        choices: None,
        upper_bound: None,
    },
    FieldSpec {
        name: "palette_name",
        kind: FieldKind::Str,
        optional: true,
        default: Value::None,
        help: "Bundled palette name, a .hex/.gpl path or hex:rrggbb,...",
        choices: None,
        upper_bound: None,
    },
    FieldSpec {
        name: "palette_source",
        kind: FieldKind::Str,
        optional: false,
        default: Value::Str("auto"),
        help: "Palette builder (auto = mcda for gerstner, median_cut otherwise).",
        choices: Some(PALETTE_SOURCES),
        upper_bound: None,
    },
    FieldSpec {
        name: "palette_ramps",
        kind: FieldKind::Bool,
        optional: false,
        default: Value::Bool(true),
        help: "Regularize the palette into hue ramps after clustering.",
        choices: None,
        upper_bound: None,
    },
    FieldSpec {
        name: "saturation_beta",
        kind: FieldKind::Float,
        optional: false,
        default: Value::Float(1.1),
        help: "Multiply palette a*, b* (saturation) by this after convergence.",
        choices: None,
        upper_bound: Some(5),
    },
    FieldSpec {
        name: "g_m",
        kind: FieldKind::Float,
        optional: false,
        default: Value::Float(45.0),
        help: "gerstner SLIC compactness.",
        choices: None,
        upper_bound: None,
    },
    FieldSpec {
        name: "g_alpha",
        kind: FieldKind::Float,
        optional: false,
        default: Value::Float(0.7),
        help: "gerstner annealing temperature decay per step, in (0, 1).",
        choices: None,
        upper_bound: None,
    },
    FieldSpec {
        name: "g_T_final",
        kind: FieldKind::Float,
        optional: false,
        default: Value::Float(1.0),
        help: "gerstner final annealing temperature.",
        choices: None,
        upper_bound: None,
    },
    FieldSpec {
        name: "g_eps_palette",
        kind: FieldKind::Float,
        optional: false,
        default: Value::Float(1.0),
        help: "gerstner total palette change (LAB) below which a step has converged.",
        choices: None,
        upper_bound: None,
    },
    FieldSpec {
        name: "g_eps_cluster",
        kind: FieldKind::Float,
        optional: false,
        default: Value::Float(0.25),
        help: "gerstner sub-cluster separation (LAB) needed to split a color.",
        choices: None,
        upper_bound: None,
    },
    FieldSpec {
        name: "g_laplacian",
        kind: FieldKind::Float,
        optional: false,
        default: Value::Float(0.4),
        help: "gerstner superpixel center smoothing fraction.",
        choices: None,
        upper_bound: None,
    },
    FieldSpec {
        name: "g_bilateral_sigma_color",
        kind: FieldKind::Float,
        optional: false,
        default: Value::Float(10.0),
        help: "gerstner bilateral color sigma on the mean-color image (LAB).",
        choices: None,
        upper_bound: None,
    },
    FieldSpec {
        name: "g_bilateral_sigma_spatial",
        kind: FieldKind::Float,
        optional: false,
        default: Value::Float(1.5),
        help: "gerstner bilateral spatial sigma, in output pixels.",
        choices: None,
        upper_bound: None,
    },
    FieldSpec {
        name: "g_max_iters",
        kind: FieldKind::Int,
        optional: false,
        default: Value::Int(2000),
        help: "Safety cap on gerstner's total iterations.",
        choices: None,
        upper_bound: Some(10000),
    },
    FieldSpec {
        name: "dither",
        kind: FieldKind::Str,
        optional: false,
        default: Value::Str("auto"),
        help: "Ordered dithering; auto dithers only smooth regions.",
        choices: Some(DITHERS),
        upper_bound: None,
    },
    FieldSpec {
        name: "dither_strength",
        kind: FieldKind::Float,
        optional: false,
        default: Value::Float(0.5),
        help: "Dither threshold amplitude, 0..1.",
        choices: None,
        upper_bound: None,
    },
    FieldSpec {
        name: "dither_variance_threshold",
        kind: FieldKind::Float,
        optional: false,
        default: Value::Float(4.0),
        help: "auto dither: only where local LAB std is below this.",
        choices: None,
        upper_bound: None,
    },
    FieldSpec {
        name: "remove_orphans",
        kind: FieldKind::Bool,
        optional: false,
        default: Value::Bool(true),
        help: "Merge tiny isolated color regions into their neighbors.",
        choices: None,
        upper_bound: None,
    },
    FieldSpec {
        name: "orphan_min_region",
        kind: FieldKind::Int,
        optional: false,
        default: Value::Int(2),
        help: "Regions (8-connected) with fewer pixels than this get merged.",
        choices: None,
        upper_bound: None,
    },
    FieldSpec {
        name: "orphan_max_delta",
        kind: FieldKind::Float,
        optional: false,
        default: Value::Float(25.0),
        help: "Only merge orphans whose replacement is within this LAB distance.",
        choices: None,
        upper_bound: None,
    },
    FieldSpec {
        name: "fix_jaggies",
        kind: FieldKind::Bool,
        optional: false,
        default: Value::Bool(true),
        help: "Clean up staircase artifacts on diagonal edges.",
        choices: None,
        upper_bound: None,
    },
    FieldSpec {
        name: "outline",
        kind: FieldKind::Str,
        optional: false,
        default: Value::Str("auto"),
        help: "Silhouette outline: \"none\", \"auto\" (darkened palette color) or \"#rrggbb\".",
        choices: None,
        upper_bound: None,
    },
    FieldSpec {
        name: "outline_darken",
        kind: FieldKind::Float,
        optional: false,
        default: Value::Float(0.55),
        help: "auto outline: blend of the darkest palette color toward black, 0..1.",
        choices: None,
        upper_bound: None,
    },
    FieldSpec {
        name: "tile_size",
        kind: FieldKind::Int,
        optional: false,
        default: Value::Int(16),
        help: "Tile edge in pixels; only applies with --tileset.",
        choices: None,
        upper_bound: Some(512),
    },
    FieldSpec {
        name: "tileset",
        kind: FieldKind::Bool,
        optional: false,
        default: Value::Bool(false),
        help: "Also write a de-duplicated tileset and tilemap.",
        choices: None,
        upper_bound: None,
    },
    FieldSpec {
        name: "fit",
        kind: FieldKind::Str,
        optional: false,
        default: Value::Str("pad"),
        help: "How a derived size meets tile_size: pad, stretch or crop.",
        choices: Some(FITS),
        upper_bound: None,
    },
    FieldSpec {
        name: "tile_dedupe_tolerance",
        kind: FieldKind::Float,
        optional: false,
        default: Value::Float(2.0),
        help: "Mean LAB distance below which two tiles count as identical.",
        choices: None,
        upper_bound: None,
    },
    FieldSpec {
        name: "seamless",
        kind: FieldKind::Bool,
        optional: false,
        default: Value::Bool(false),
        help: "Wrap-around filtering so the output tiles seamlessly.",
        choices: None,
        upper_bound: None,
    },
    FieldSpec {
        name: "tileset_columns",
        kind: FieldKind::Int,
        optional: false,
        default: Value::Int(0),
        help: "Tiles per tileset row (0 = ceil(sqrt(tile count))).",
        choices: None,
        upper_bound: None,
    },
    FieldSpec {
        name: "transparent_index",
        kind: FieldKind::Str,
        optional: false,
        default: Value::Str("last"),
        help: "PNG palette slot used for transparency.",
        choices: Some(TRANSPARENT_INDICES),
        upper_bound: None,
    },
    FieldSpec {
        name: "seed",
        kind: FieldKind::Int,
        optional: true,
        default: Value::None,
        help: "Seed for the optional MCDA jitter (unset = fully analytic).",
        choices: None,
        upper_bound: Some(9223372036854775807),
    },
    FieldSpec {
        name: "scale_preview",
        kind: FieldKind::Int,
        optional: false,
        default: Value::Int(8),
        help: "Nearest-neighbor upscale factor for *_preview.png.",
        choices: None,
        upper_bound: Some(64),
    },
];

/// Per-preset overrides of the field defaults, applied as: defaults ← preset ← explicit
/// values. remove_bg is opt-in in every preset (never environment-derived).
pub const PRESETS: &[(&str, &[(&str, Value)])] = &[
    (
        "sprite",
        &[
            ("remove_bg", Value::Bool(false)),
            ("key_bg", Value::Bool(true)),
            ("crop_to_alpha", Value::Bool(true)),
            ("method", Value::Str("gerstner")),
            ("palette_size", Value::Int(16)),
            ("dither", Value::Str("none")),
            ("outline", Value::Str("auto")),
            ("tileset", Value::Bool(false)),
            ("denoise", Value::Str("bilateral")),
        ],
    ),
    (
        "background",
        &[
            ("remove_bg", Value::Bool(false)),
            ("key_bg", Value::Bool(false)),
            ("crop_to_alpha", Value::Bool(false)),
            ("method", Value::Str("kopf")),
            ("palette_size", Value::Int(32)),
            ("dither", Value::Str("auto")),
            ("outline", Value::Str("none")),
            ("tileset", Value::Bool(true)),
            ("denoise", Value::Str("bilateral")),
        ],
    ),
];

/// Longest output edge when neither out_width nor out_height is given.
pub const PRESET_LONGEST_EDGE: &[(&str, i64)] = &[("sprite", 64), ("background", 256)];

/// The field named `name`.
pub fn field(name: &str) -> Option<&'static FieldSpec> {
    FIELDS.iter().find(|f| f.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_unique() {
        for (i, f) in FIELDS.iter().enumerate() {
            assert!(
                FIELDS[..i].iter().all(|g| g.name != f.name),
                "duplicate field {}",
                f.name
            );
        }
    }

    #[test]
    fn defaults_match_kinds() {
        for f in FIELDS {
            assert!(
                f.default.fits(f.kind, f.optional),
                "{}: {:?}",
                f.name,
                f.default
            );
        }
    }

    #[test]
    fn choice_defaults_are_choices() {
        for f in FIELDS {
            if let Some(choices) = f.choices {
                assert_eq!(f.kind, FieldKind::Str, "{}", f.name);
                match f.default {
                    Value::Str(s) => assert!(choices.contains(&s), "{}: {s}", f.name),
                    other => panic!("{}: choice field with default {other:?}", f.name),
                }
            }
        }
    }

    #[test]
    fn defaults_respect_upper_bounds() {
        for f in FIELDS {
            if let Some(bound) = f.upper_bound {
                assert!(
                    matches!(f.kind, FieldKind::Int | FieldKind::Float),
                    "{}",
                    f.name
                );
                match f.default {
                    Value::Int(v) => assert!(v <= bound, "{}", f.name),
                    Value::Float(v) => assert!(v <= bound as f64, "{}", f.name),
                    Value::None => {}
                    other => panic!("{}: bounded field with default {other:?}", f.name),
                }
            }
        }
    }

    #[test]
    fn presets_name_fields_with_fitting_values() {
        let names: Vec<&str> = PRESETS.iter().map(|(n, _)| *n).collect();
        assert_eq!(names, PRESET_NAMES);
        for (preset, values) in PRESETS {
            for (key, value) in *values {
                let f = field(key).unwrap_or_else(|| panic!("{preset}: unknown field {key}"));
                assert!(value.fits(f.kind, f.optional), "{preset}.{key}: {value:?}");
                if let (Some(choices), Value::Str(s)) = (f.choices, value) {
                    assert!(choices.contains(s), "{preset}.{key}: {s}");
                }
            }
        }
        let edges: Vec<&str> = PRESET_LONGEST_EDGE.iter().map(|(n, _)| *n).collect();
        assert_eq!(edges, PRESET_NAMES);
    }

    #[test]
    fn preset_field_choices_are_the_preset_names() {
        assert_eq!(field("preset").unwrap().choices, Some(PRESET_NAMES));
        assert!(
            matches!(field("preset").unwrap().default, Value::Str(s) if PRESET_NAMES.contains(&s))
        );
    }
}
