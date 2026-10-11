//! Port of tests/test_palette.py: median cut and mcda, hue ramps, bundled / file / inline
//! palettes and their errors.

mod common;

use common::*;
use pixelforge::{color, palette};

const EXPECTED_COUNTS: [(&str, usize); 5] = [
    ("gameboy", 4),
    ("genesis", 16),
    ("nes", 54),
    ("pico8", 16),
    ("snes15", 15),
];

fn lab_of(rgb8: &[[u8; 3]]) -> Vec<[f64; 3]> {
    let flat: Vec<u8> = rgb8.iter().flatten().copied().collect();
    color::rgb8_to_lab(&flat)
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect()
}

/// `color.delta_e(found, c).min()`.
fn min_delta(found: &[[f64; 3]], c: &[f64; 3]) -> f64 {
    found
        .iter()
        .map(|f| {
            let d: f64 = (0..3).map(|i| (f[i] - c[i]).powi(2)).sum();
            d.sqrt()
        })
        .fold(f64::INFINITY, f64::min)
}

fn four_color_pixels() -> (Vec<[f64; 3]>, Vec<[f64; 3]>) {
    let colors = lab_of(&[[20, 20, 30], [230, 60, 50], [60, 200, 90], [250, 240, 200]]);
    let counts = [700, 100, 150, 50]; // deliberately unbalanced
    let mut pixels = Vec::new();
    for (c, &n) in colors.iter().zip(&counts) {
        pixels.extend(std::iter::repeat_n(*c, n));
    }
    (colors, pixels)
}

#[test]
fn test_median_cut_recovers_four_colors() {
    let (colors, pixels) = four_color_pixels();
    let found = palette::median_cut(&pixels, 4).unwrap();
    assert_eq!(found.len(), 4);
    for c in &colors {
        assert!(min_delta(&found, c) < 1.0);
    }
}

#[test]
fn test_median_cut_stops_at_distinct_colors() {
    let (_, pixels) = four_color_pixels();
    assert_eq!(palette::median_cut(&pixels, 16).unwrap().len(), 4);
}

#[test]
fn test_mcda_recovers_four_colors() {
    let (colors, pixels) = four_color_pixels();
    let found = palette::mcda(&pixels, 4, &config(&[])).unwrap();
    assert_eq!(found.len(), 4);
    for c in &colors {
        assert!(min_delta(&found, c) < 1.0);
    }
}

fn hue_degrees(c: &[f64; 3]) -> f64 {
    c[2].atan2(c[1]).to_degrees()
}

#[test]
#[allow(non_snake_case)]
fn test_regularize_ramps_preserves_L_and_chroma() {
    let mut rgb = Vec::new();
    for r in [10u8, 90, 170, 250] {
        for g in [20u8, 130, 240] {
            for b in [0u8, 128, 255] {
                rgb.push([r, g, b]);
            }
        }
    }
    let before = lab_of(&rgb);
    let after = palette::regularize_ramps(&before);
    assert_eq!(after.len(), before.len());
    let mut max_change: f64 = 0.0;
    for (p, q) in before.iter().zip(&after) {
        assert!((q[0] - p[0]).abs() <= 1e-6);
        let chroma_before = p[1].hypot(p[2]);
        let chroma_after = q[1].hypot(q[2]);
        assert!((chroma_after - chroma_before).abs() <= 1e-6);
        if chroma_before > 1e-6 {
            let change =
                ((hue_degrees(q) - hue_degrees(p) + 180.0).rem_euclid(360.0) - 180.0).abs();
            max_change = max_change.max(change);
        }
    }
    assert!(max_change <= 6.0 + 1e-9);
    assert!(max_change > 0.5); // the pass does something
}

#[test]
fn test_ramp_direction() {
    // Two reds in one hue bin: the dark one leans toward purple, the light one toward yellow.
    let lab = [[30.0, 50.0, 20.0], [75.0, 50.0, 20.0]];
    let out = palette::regularize_ramps(&lab);
    let original = 20.0f64.atan2(50.0).to_degrees();
    assert!(hue_degrees(&out[0]) < original && original < hue_degrees(&out[1]));
}

#[test]
fn test_bundled_palettes() {
    for (name, count) in EXPECTED_COUNTS {
        let rgb8 = palette::load_palette(name).unwrap();
        assert_eq!(rgb8.len(), count, "{name}");
        let mut unique = rgb8.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), count, "{name}");
    }
}

#[test]
fn test_bundled_palette_list_and_gameboy_values() {
    let names: Vec<&str> = EXPECTED_COUNTS.iter().map(|(n, _)| *n).collect();
    assert_eq!(palette::bundled_palettes(), names);
    let hex: Vec<String> = palette::load_palette("gameboy")
        .unwrap()
        .into_iter()
        .map(color::rgb8_to_hex)
        .collect();
    assert_eq!(hex, ["#0f380f", "#306230", "#8bac0f", "#9bbc0f"]);
}

fn load(path: &std::path::Path) -> Result<Vec<[u8; 3]>, palette::PaletteError> {
    palette::load_palette(path.to_str().unwrap())
}

#[test]
fn test_load_hex_and_gpl_files() {
    let tmp = TempDir::new("load_hex_and_gpl");
    let hex_path = tmp.join("p.hex");
    std::fs::write(&hex_path, "#ff0000\n00ff00\n\n0000ff\n").unwrap();
    assert_eq!(
        load(&hex_path).unwrap(),
        vec![[255, 0, 0], [0, 255, 0], [0, 0, 255]]
    );
    let gpl_path = tmp.join("p.gpl");
    std::fs::write(
        &gpl_path,
        "GIMP Palette\nName: test\nColumns: 2\n#\n255 0 0\tred\n  0 128 255 sky\n",
    )
    .unwrap();
    assert_eq!(load(&gpl_path).unwrap(), vec![[255, 0, 0], [0, 128, 255]]);
    assert!(palette::load_palette("no-such-palette").is_err());
}

/// 94% shades of one green plus four small features, like a single-color sprite.
fn dominant_color_pixels() -> (Vec<[f64; 3]>, Vec<[f64; 3]>) {
    // np.linspace(-1.0, 1.0, 940)
    let n = 940;
    let step = 2.0 / (n - 1) as f64;
    let mut pixels: Vec<[f64; 3]> = (0..n)
        .map(|i| {
            let t = if i == n - 1 {
                1.0
            } else {
                -1.0 + i as f64 * step
            };
            [
                70.0 + 6.0 * t,
                -55.0 + 4.0 * (7.0 * t).sin(),
                60.0 + 4.0 * (5.0 * t).cos(),
            ]
        })
        .collect();
    let features = lab_of(&[[250, 250, 240], [20, 40, 20], [240, 90, 120], [40, 110, 40]]);
    for (f, n) in features.iter().zip([20, 20, 10, 10]) {
        pixels.extend(std::iter::repeat_n(*f, n));
    }
    (pixels, features)
}

#[test]
fn test_median_cut_keeps_small_features_of_a_dominant_color_image() {
    let (pixels, features) = dominant_color_pixels();
    let found = palette::median_cut(&pixels, 8).unwrap();
    assert_eq!(found.len(), 8);
    // Splitting by pixel count at the median averaged these into colors ΔE 24-90 away.
    for f in &features {
        assert!(min_delta(&found, f) < 1.0);
    }
}

#[test]
fn test_mcda_does_not_spend_the_palette_on_coincident_colors() {
    let (pixels, features) = dominant_color_pixels();
    let found = palette::mcda(&pixels, 8, &config(&[])).unwrap();
    assert_eq!(found.len(), 8);
    // With the spec's split test the palette held coincident entries and missed every
    // feature by ΔE 14-49.
    for f in &features {
        assert!(min_delta(&found, f) < 1.0);
    }
    let flat: Vec<f64> = found.iter().flatten().copied().collect();
    let distances = color::palette_distance_matrix(&flat);
    let k = found.len();
    let min_off_diagonal = (0..k * k)
        .filter(|i| i / k != i % k)
        .map(|i| distances[i])
        .fold(f64::INFINITY, f64::min);
    assert!(min_off_diagonal > 3.0);
}

#[test]
fn test_bad_palette_line_reports_number_not_content() {
    let tmp = TempDir::new("bad_palette_line");
    let path = tmp.join("secret.hex");
    std::fs::write(&path, "ff0000\nhunter2-password\n").unwrap();
    let msg = load(&path).unwrap_err().0;
    assert!(
        msg.contains("line 2") && msg.contains("secret.hex"),
        "{msg}"
    );
    assert!(!msg.contains("hunter2"), "{msg}");
}

#[test]
fn test_palette_files_are_size_suffix_and_type_checked() {
    let tmp = TempDir::new("palette_files_checked");
    let big = tmp.join("big.hex");
    std::fs::write(&big, "ff0000\n".repeat(10000)).unwrap();
    assert!(load(&big).unwrap_err().0.contains("larger than"));
    let binary = tmp.join("blob.gpl");
    std::fs::write(&binary, b"\xff\xfe\x00\x80".repeat(8)).unwrap();
    assert!(load(&binary)
        .unwrap_err()
        .0
        .contains("not a UTF-8 text file"));
    let other = tmp.join("colors.txt");
    std::fs::write(&other, "ff0000\n").unwrap();
    assert!(load(&other).unwrap_err().0.contains("unknown palette"));
    let folder = tmp.join("dir.hex");
    std::fs::create_dir(&folder).unwrap();
    assert!(load(&folder).unwrap_err().0.contains("not a regular file"));
    let high = tmp.join("high.gpl");
    std::fs::write(&high, "GIMP Palette\n300 0 0\n").unwrap();
    assert!(load(&high).unwrap_err().0.contains("line 2"));
}

#[test]
fn test_mcda_single_color() {
    let pixels = vec![[60.0, 10.0, -20.0]; 500];
    assert_eq!(palette::mcda(&pixels, 8, &config(&[])).unwrap().len(), 1);
}

#[test]
fn test_inline_palette_round_trips() {
    let rgb8 = [[255u8, 0, 0], [0, 128, 255]];
    let name = palette::inline_name(&rgb8);
    assert_eq!(name, "hex:ff0000,0080ff");
    assert_eq!(palette::load_palette(&name).unwrap(), rgb8);
    for bad in ["hex:", "hex:ff00", "hex:ff0000,zz0000"] {
        assert!(palette::load_palette(bad).is_err(), "{bad}");
    }
}
