import numpy as np
import pytest

from pixelforge import Config, color, palette

EXPECTED_COUNTS = {"nes": 54, "gameboy": 4, "pico8": 16, "snes15": 15, "genesis": 16}


def _four_color_pixels():
    colors = color.rgb8_to_lab(np.array([[20, 20, 30], [230, 60, 50], [60, 200, 90],
                                         [250, 240, 200]]))
    counts = [700, 100, 150, 50]   # deliberately unbalanced
    return colors, np.concatenate([np.repeat(c[None], n, axis=0) for c, n in zip(colors, counts)])


def test_median_cut_recovers_four_colors():
    colors, pixels = _four_color_pixels()
    found = palette.median_cut(pixels, 4)
    assert found.shape == (4, 3)
    for c in colors:
        assert color.delta_e(found, c).min() < 1.0


def test_median_cut_stops_at_distinct_colors():
    _, pixels = _four_color_pixels()
    assert len(palette.median_cut(pixels, 16)) == 4


def test_mcda_recovers_four_colors():
    colors, pixels = _four_color_pixels()
    found = palette.mcda(pixels, 4, Config())
    assert found.shape == (4, 3)
    for c in colors:
        assert color.delta_e(found, c).min() < 1.0


def test_regularize_ramps_preserves_L_and_chroma():
    rgb = np.array([[r, g, b] for r in (10, 90, 170, 250) for g in (20, 130, 240)
                    for b in (0, 128, 255)], dtype=np.uint8)
    before = color.rgb8_to_lab(rgb)
    after = palette.regularize_ramps(before)
    assert after.shape == before.shape
    assert np.abs(after[:, 0] - before[:, 0]).max() <= 1e-6
    chroma_before = np.hypot(before[:, 1], before[:, 2])
    chroma_after = np.hypot(after[:, 1], after[:, 2])
    assert np.abs(chroma_after - chroma_before).max() <= 1e-6
    hue_before = np.degrees(np.arctan2(before[:, 2], before[:, 1]))
    hue_after = np.degrees(np.arctan2(after[:, 2], after[:, 1]))
    change = np.abs((hue_after - hue_before + 180.0) % 360.0 - 180.0)
    chromatic = chroma_before > 1e-6
    assert change[chromatic].max() <= 6.0 + 1e-9
    assert change[chromatic].max() > 0.5   # the pass does something


def test_ramp_direction():
    # Two reds in one hue bin: the dark one leans toward purple, the light one toward yellow.
    lab = np.array([[30.0, 50.0, 20.0], [75.0, 50.0, 20.0]])
    out = palette.regularize_ramps(lab)
    hue = np.degrees(np.arctan2(out[:, 2], out[:, 1]))
    original = np.degrees(np.arctan2(20.0, 50.0))
    assert hue[0] < original < hue[1]


@pytest.mark.parametrize("name,count", sorted(EXPECTED_COUNTS.items()))
def test_bundled_palettes(name, count):
    rgb8 = palette.load_palette(name)
    assert rgb8.shape == (count, 3) and rgb8.dtype == np.uint8
    assert len(np.unique(rgb8, axis=0)) == count


def test_bundled_palette_list_and_gameboy_values():
    assert palette.bundled_palettes() == sorted(EXPECTED_COUNTS)
    assert [color.rgb8_to_hex(c) for c in palette.load_palette("gameboy")] == \
        ["#0f380f", "#306230", "#8bac0f", "#9bbc0f"]


def test_load_hex_and_gpl_files(tmp_path):
    hex_path = tmp_path / "p.hex"
    hex_path.write_text("#ff0000\n00ff00\n\n0000ff\n")
    assert palette.load_palette(str(hex_path)).tolist() == [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
    gpl_path = tmp_path / "p.gpl"
    gpl_path.write_text("GIMP Palette\nName: test\nColumns: 2\n#\n255 0 0\tred\n  0 128 255 sky\n")
    assert palette.load_palette(str(gpl_path)).tolist() == [[255, 0, 0], [0, 128, 255]]
    with pytest.raises(ValueError):
        palette.load_palette("no-such-palette")


def _dominant_color_pixels():
    """94% shades of one green plus four small features, like a single-color sprite."""
    t = np.linspace(-1.0, 1.0, 940)
    green = np.stack([70 + 6 * t, -55 + 4 * np.sin(7 * t), 60 + 4 * np.cos(5 * t)], axis=1)
    features = color.rgb8_to_lab(np.array([[250, 250, 240], [20, 40, 20], [240, 90, 120],
                                           [40, 110, 40]]))
    counts = (20, 20, 10, 10)
    pixels = np.concatenate([green] + [np.repeat(f[None], n, axis=0)
                                       for f, n in zip(features, counts)])
    return pixels, features


def test_median_cut_keeps_small_features_of_a_dominant_color_image():
    pixels, features = _dominant_color_pixels()
    found = palette.median_cut(pixels, 8)
    assert len(found) == 8
    # Splitting by pixel count at the median averaged these into colors ΔE 24-90 away.
    for feature in features:
        assert color.delta_e(found, feature).min() < 1.0


def test_mcda_does_not_spend_the_palette_on_coincident_colors():
    pixels, features = _dominant_color_pixels()
    found = palette.mcda(pixels, 8, Config())
    assert len(found) == 8
    # With the spec's split test the palette held coincident entries and missed every
    # feature by ΔE 14-49.
    for feature in features:
        assert color.delta_e(found, feature).min() < 1.0
    distances = color.palette_distance_matrix(found)
    np.fill_diagonal(distances, np.inf)
    assert distances.min() > 3.0


def test_bad_palette_line_reports_number_not_content(tmp_path):
    path = tmp_path / "secret.hex"
    path.write_text("ff0000\nhunter2-password\n")
    with pytest.raises(ValueError) as info:
        palette.load_palette(str(path))
    assert "line 2" in str(info.value) and "secret.hex" in str(info.value)
    assert "hunter2" not in str(info.value)


def test_palette_files_are_size_suffix_and_type_checked(tmp_path):
    big = tmp_path / "big.hex"
    big.write_text("ff0000\n" * 10000)
    with pytest.raises(ValueError, match="larger than"):
        palette.load_palette(str(big))
    binary = tmp_path / "blob.gpl"
    binary.write_bytes(b"\xff\xfe\x00\x80" * 8)
    with pytest.raises(ValueError, match="not a UTF-8 text file"):
        palette.load_palette(str(binary))
    other = tmp_path / "colors.txt"
    other.write_text("ff0000\n")
    with pytest.raises(ValueError, match="unknown palette"):
        palette.load_palette(str(other))
    folder = tmp_path / "dir.hex"
    folder.mkdir()
    with pytest.raises(ValueError, match="not a regular file"):
        palette.load_palette(str(folder))
    high = tmp_path / "high.gpl"
    high.write_text("GIMP Palette\n300 0 0\n")
    with pytest.raises(ValueError, match="line 2"):
        palette.load_palette(str(high))


def test_mcda_single_color():
    assert len(palette.mcda(np.tile([60.0, 10.0, -20.0], (500, 1)), 8, Config())) == 1
