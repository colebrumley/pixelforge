//! sRGB <-> CIELAB conversion and palette distance helpers.
//!
//! The conversions follow scikit-image 0.26 formula for formula (`rgb2lab` = `rgb2xyz` then
//! `xyz2lab`, `lab2rgb` = `lab2xyz` then `xyz2rgb`, D65 reference white, 2° observer), with
//! `pow` and `cbrt` from the `libm` crate so the bits do not depend on the platform. They
//! agree with skimage to about 1e-13 (the last bits of the platform `pow`/`cbrt` and of the
//! BLAS matrix product differ); see `tests/color_oracle.rs`.
//!
//! All functions take interleaved triples (`[c0, c1, c2, c0, c1, c2, ...]`) and panic if a
//! slice length is not a multiple of 3.

/// sRGB (linear) -> XYZ, the sRGB specification matrix skimage uses (`xyz_from_rgb`).
const XYZ_FROM_RGB: [[f64; 3]; 3] = [
    [0.412453, 0.357580, 0.180423],
    [0.212671, 0.715160, 0.072169],
    [0.019334, 0.119193, 0.950227],
];

/// XYZ -> sRGB (linear): skimage's `rgb_from_xyz = scipy.linalg.inv(xyz_from_rgb)`, its
/// float64 values copied bit for bit (they differ from `numpy.linalg.inv` in the last bit).
const RGB_FROM_XYZ: [[f64; 3]; 3] = [
    [3.2404813432005266, -1.5371515162713183, -0.4985363261688878],
    [-0.9692549499965682, 1.8759900014898907, 0.04155592655829283],
    [
        0.055646639135177166,
        -0.20404133836651123,
        1.0573110696453443,
    ],
];

/// D65, 2° observer, as skimage's `_illuminants["D65"]["2"]` (the old `lab_ref_white`).
const WHITE: [f64; 3] = [0.95047, 1.0, 1.08883];

fn check_triples(name: &str, len: usize) {
    assert!(
        len.is_multiple_of(3),
        "{name}: length {len} is not a multiple of 3 (interleaved triples expected)"
    );
}

#[inline]
fn mat_vec(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    // `arr @ matrix.T`: out[i] = v0*m[i][0] + v1*m[i][1] + v2*m[i][2], summed left to right.
    let mut out = [0.0; 3];
    for (o, row) in out.iter_mut().zip(m) {
        *o = v[0] * row[0] + v[1] * row[1] + v[2] * row[2];
    }
    out
}

#[inline]
fn srgb_to_linear(c: f64) -> f64 {
    if c > 0.04045 {
        libm::pow((c + 0.055) / 1.055, 2.4)
    } else {
        c / 12.92
    }
}

#[inline]
fn linear_to_srgb(c: f64) -> f64 {
    if c > 0.0031308 {
        1.055 * libm::pow(c, 1.0 / 2.4) - 0.055
    } else {
        c * 12.92
    }
}

#[inline]
fn lab_f(t: f64) -> f64 {
    if t > 0.008856 {
        libm::cbrt(t)
    } else {
        7.787 * t + 16.0 / 116.0
    }
}

#[inline]
fn lab_f_inv(t: f64) -> f64 {
    if t > 0.2068966 {
        libm::pow(t, 3.0)
    } else {
        (t - 16.0 / 116.0) / 7.787
    }
}

#[inline]
fn rgb_to_lab_one(rgb: [f64; 3]) -> [f64; 3] {
    let lin = [
        srgb_to_linear(rgb[0]),
        srgb_to_linear(rgb[1]),
        srgb_to_linear(rgb[2]),
    ];
    let xyz = mat_vec(&XYZ_FROM_RGB, lin);
    let fx = lab_f(xyz[0] / WHITE[0]);
    let fy = lab_f(xyz[1] / WHITE[1]);
    let fz = lab_f(xyz[2] / WHITE[2]);
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

#[inline]
fn lab_to_rgb_one(lab: [f64; 3]) -> [f64; 3] {
    let y = (lab[0] + 16.0) / 116.0;
    let x = lab[1] / 500.0 + y;
    let mut z = y - lab[2] / 200.0;
    if z < 0.0 {
        z = 0.0; // skimage clips negative z (and warns)
    }
    let xyz = [
        lab_f_inv(x) * WHITE[0],
        lab_f_inv(y) * WHITE[1],
        lab_f_inv(z) * WHITE[2],
    ];
    let lin = mat_vec(&RGB_FROM_XYZ, xyz);
    [
        linear_to_srgb(lin[0]).clamp(0.0, 1.0),
        linear_to_srgb(lin[1]).clamp(0.0, 1.0),
        linear_to_srgb(lin[2]).clamp(0.0, 1.0),
    ]
}

/// sRGB in [0, 1] -> CIELAB (D65); L in [0, 100]. Values outside [0, 1] are converted as
/// skimage converts them (no clipping).
pub fn rgb_to_lab(rgb: &[f64]) -> Vec<f64> {
    check_triples("rgb", rgb.len());
    let mut out = Vec::with_capacity(rgb.len());
    for c in rgb.chunks_exact(3) {
        out.extend_from_slice(&rgb_to_lab_one([c[0], c[1], c[2]]));
    }
    out
}

/// CIELAB -> sRGB in [0, 1], clipped to the gamut.
pub fn lab_to_rgb(lab: &[f64]) -> Vec<f64> {
    check_triples("lab", lab.len());
    let mut out = Vec::with_capacity(lab.len());
    for c in lab.chunks_exact(3) {
        out.extend_from_slice(&lab_to_rgb_one([c[0], c[1], c[2]]));
    }
    out
}

/// 8-bit sRGB -> CIELAB: `rgb_to_lab(rgb8 / 255.0)`.
pub fn rgb8_to_lab(rgb8: &[u8]) -> Vec<f64> {
    check_triples("rgb8", rgb8.len());
    let mut out = Vec::with_capacity(rgb8.len());
    for c in rgb8.chunks_exact(3) {
        out.extend_from_slice(&rgb_to_lab_one([
            f64::from(c[0]) / 255.0,
            f64::from(c[1]) / 255.0,
            f64::from(c[2]) / 255.0,
        ]));
    }
    out
}

/// One channel in [0, 1] -> 8 bits: `np.round(np.clip(v * 255, 0, 255))`, half to even.
/// NaN maps to 0.
#[inline]
pub fn unit_to_u8(v: f64) -> u8 {
    let s = (v * 255.0).clamp(0.0, 255.0);
    // In [0, 255] after the clamp, so the cast is exact (NaN casts to 0).
    s.round_ties_even() as u8
}

/// CIELAB -> 8-bit sRGB: `lab_to_rgb`, times 255, clipped, rounded half to even.
pub fn lab_to_rgb8(lab: &[f64]) -> Vec<u8> {
    check_triples("lab", lab.len());
    let mut out = Vec::with_capacity(lab.len());
    for c in lab.chunks_exact(3) {
        let rgb = lab_to_rgb_one([c[0], c[1], c[2]]);
        out.extend(rgb.iter().map(|&v| unit_to_u8(v)));
    }
    out
}

#[inline]
fn dist2(a: &[f64], b: &[f64]) -> f64 {
    let d0 = a[0] - b[0];
    let d1 = a[1] - b[1];
    let d2 = a[2] - b[2];
    d0 * d0 + d1 * d1 + d2 * d2
}

/// CIE76 ΔE (Euclidean distance in LAB) between corresponding triples of `a` and `b`.
pub fn delta_e(a: &[f64], b: &[f64]) -> Vec<f64> {
    check_triples("a", a.len());
    assert_eq!(a.len(), b.len(), "delta_e: a and b differ in length");
    a.chunks_exact(3)
        .zip(b.chunks_exact(3))
        .map(|(p, q)| dist2(p, q).sqrt())
        .collect()
}

/// Index of the nearest palette colour (CIE76) for every pixel triple: brute force over the
/// squared distance, strict `<`, so ties go to the lowest index. A pixel whose distance is
/// NaN for every entry gets index 0. Panics if `palette` is empty and `pixels` is not.
pub fn nearest_index(pixels: &[f64], palette: &[f64]) -> Vec<i64> {
    check_triples("pixels", pixels.len());
    check_triples("palette", palette.len());
    if pixels.is_empty() {
        return Vec::new();
    }
    assert!(!palette.is_empty(), "nearest_index: empty palette");
    pixels
        .chunks_exact(3)
        .map(|p| {
            let mut best = 0usize;
            let mut best_d = f64::INFINITY;
            for (k, q) in palette.chunks_exact(3).enumerate() {
                let d = dist2(p, q);
                if d < best_d {
                    best_d = d;
                    best = k;
                }
            }
            best as i64
        })
        .collect()
}

/// The K×K matrix of CIE76 distances between palette entries, row-major.
pub fn palette_distance_matrix(palette: &[f64]) -> Vec<f64> {
    check_triples("palette", palette.len());
    let mut out = Vec::with_capacity(palette.len() / 3 * palette.len() / 3);
    for p in palette.chunks_exact(3) {
        for q in palette.chunks_exact(3) {
            out.push(dist2(p, q).sqrt());
        }
    }
    out
}

/// `"#rrggbb"` (surrounding whitespace and leading `#`s ignored, either case) -> 8-bit RGB.
pub fn hex_to_rgb8(text: &str) -> Result<[u8; 3], String> {
    let t = text.trim().trim_start_matches('#');
    let err = || format!("not a hex color: '{t}'");
    if t.len() != 6 || !t.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(err());
    }
    let mut out = [0u8; 3];
    for (i, o) in out.iter_mut().enumerate() {
        *o = u8::from_str_radix(&t[2 * i..2 * i + 2], 16).map_err(|_| err())?;
    }
    Ok(out)
}

/// 8-bit RGB -> `"#rrggbb"` (lower case).
pub fn rgb8_to_hex(rgb: [u8; 3]) -> String {
    format!("#{:02x}{:02x}{:02x}", rgb[0], rgb[1], rgb[2])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ties_go_to_lowest_index() {
        // Equidistant from entries 0 and 1 (both 25 away).
        let two = [0.0, 0.0, 0.0, 10.0, 0.0, 0.0];
        assert_eq!(nearest_index(&[5.0, 0.0, 0.0], &two), vec![0]);
        // Three entries all 100 away; the first wins.
        let three = [10.0, 0.0, 0.0, -10.0, 0.0, 0.0, 0.0, 10.0, 0.0];
        assert_eq!(nearest_index(&[0.0, 0.0, 0.0], &three), vec![0]);
        // A duplicated exact match: the lower of the two.
        let dup = [9.0, 9.0, 9.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
        assert_eq!(nearest_index(&[1.0, 2.0, 3.0], &dup), vec![1]);
        assert_eq!(nearest_index(&[], &dup), Vec::<i64>::new());
    }

    #[test]
    fn rounding_is_half_to_even() {
        assert_eq!(unit_to_u8(0.5 / 255.0), 0);
        assert_eq!(unit_to_u8(2.5 / 255.0), 2);
        assert_eq!(unit_to_u8(3.5 / 255.0), 4);
        assert_eq!(unit_to_u8(-1.0), 0);
        assert_eq!(unit_to_u8(2.0), 255);
        assert_eq!(unit_to_u8(f64::NAN), 0);
    }

    #[test]
    fn hex_helpers() {
        assert_eq!(hex_to_rgb8(" #1a2B3c\n"), Ok([0x1a, 0x2b, 0x3c]));
        assert!(hex_to_rgb8("#12345").is_err());
        assert!(hex_to_rgb8("#12345g").is_err());
        assert_eq!(rgb8_to_hex([0x1a, 0x2b, 0x3c]), "#1a2b3c");
    }

    #[test]
    fn distances() {
        let d = delta_e(
            &[0.0, 3.0, 4.0, 1.0, 1.0, 1.0],
            &[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        );
        assert_eq!(d, vec![5.0, 0.0]);
        let m = palette_distance_matrix(&[0.0, 0.0, 0.0, 0.0, 3.0, 4.0]);
        assert_eq!(m, vec![0.0, 5.0, 5.0, 0.0]);
    }
}
