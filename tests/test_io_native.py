"""Parity of the native image I/O (crates/pixelforge/src/io.rs) with the Pillow version it
replaced (tests/io_reference.py): decoded pixels, hashes, warnings and errors, the PNG
writer's decoded output and text chunks, and the array inputs."""

import os
import struct
import warnings
import zlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import io_reference as ref
from pixelforge import io
from pixelforge.errors import PixelforgeError

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = sorted((ROOT / "tests" / "fixtures").glob("*.png"))
SAMPLES_DIR = Path(os.environ.get("PIXELFORGE_SAMPLES", ROOT / "samples"))
SAMPLES = sorted(SAMPLES_DIR.glob("*.png"))
RNG_SEED = 20261011


def _rgba(loaded) -> np.ndarray:
    rgb8 = np.rint(loaded.rgb * 255.0).astype(np.uint8)
    return np.dstack([rgb8, loaded.alpha])


def _load_both(path, max_pixels=io.MAX_INPUT_PIXELS):
    with warnings.catch_warnings(record=True) as w_ref:
        warnings.simplefilter("always")
        expected = ref.load(path, max_pixels)
    with warnings.catch_warnings(record=True) as w_new:
        warnings.simplefilter("always")
        actual = io.load(path, max_pixels)
    return expected, actual, [str(w.message) for w in w_ref], [str(w.message) for w in w_new]


def assert_same_decode(path):
    expected, actual, w_ref, w_new = _load_both(path)
    assert actual.rgb.dtype == np.float64 and actual.alpha.dtype == np.uint8
    assert actual.rgb.shape == expected.rgb.shape
    np.testing.assert_array_equal(actual.rgb, expected.rgb)
    np.testing.assert_array_equal(actual.alpha, expected.alpha)
    assert actual.sha256 == expected.sha256
    assert w_new == w_ref


def _errors(path, max_pixels=io.MAX_INPUT_PIXELS):
    out = []
    for module in (ref, io):
        with pytest.raises(Exception) as info:
            module.load(path, max_pixels)
        out.append(info.value)
    return out


# --- writers for inputs Pillow cannot produce ------------------------------------------------

def _chunk(kind: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + kind + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))


def _pack_row(values, bits: int) -> bytes:
    values = np.asarray(values).reshape(-1)
    if bits == 16:
        return values.astype(">u2").tobytes()
    if bits == 8:
        return values.astype(np.uint8).tobytes()
    out = bytearray()
    per = 8 // bits
    for i in range(0, len(values), per):
        byte = 0
        for j, v in enumerate(values[i:i + per]):
            byte |= int(v) << (8 - bits * (j + 1))
        out.append(byte)
    return bytes(out)


ADAM7 = [(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2),
         (0, 1, 1, 2)]


def raw_png(path, samples, bits, color_type, plte=None, trns=None, interlace=False,
            before=(), after=()):
    """A PNG with exactly these samples ((H, W, C) ints), filter 0 on every row."""
    samples = np.asarray(samples)
    h, w = samples.shape[:2]

    def rows(img):
        return b"".join(b"\0" + _pack_row(img[y], bits) for y in range(img.shape[0]))

    if interlace:
        raw = b""
        for x0, y0, dx, dy in ADAM7:
            sub = samples[y0::dy, x0::dx]
            if sub.size:
                raw += rows(sub)
    else:
        raw = rows(samples)
    data = b"\x89PNG\r\n\x1a\n" + _chunk(
        b"IHDR", struct.pack(">IIBBBBB", w, h, bits, color_type, 0, 0, int(interlace)))
    for kind, payload in before:
        data += _chunk(kind, payload)
    if plte is not None:
        data += _chunk(b"PLTE", bytes(plte))
    if trns is not None:
        data += _chunk(b"tRNS", bytes(trns))
    data += _chunk(b"IDAT", zlib.compress(raw))
    for kind, payload in after:
        data += _chunk(kind, payload)
    data += _chunk(b"IEND", b"")
    Path(path).write_bytes(data)
    return path


def _lzw_literals(indices) -> bytes:
    """GIF LZW data (minimum code size 8) made of literal codes only, cleared every 200."""
    codes = []
    flat = [int(v) for v in np.asarray(indices).reshape(-1)]
    for i in range(0, len(flat), 200):
        codes.append(256)
        codes.extend(flat[i:i + 200])
    codes.append(257)
    acc = nbits = 0
    out = bytearray()
    for code in codes:
        acc |= code << nbits
        nbits += 9
        while nbits >= 8:
            out.append(acc & 0xFF)
            acc >>= 8
            nbits -= 8
    if nbits:
        out.append(acc & 0xFF)
    blocks = b"".join(bytes([len(out[i:i + 255])]) + bytes(out[i:i + 255])
                      for i in range(0, len(out), 255))
    return b"\x08" + blocks + b"\x00"


def raw_gif(path, screen, frames, global_palette=None):
    """A GIF89a: frames are dicts with left, top, indices (H, W), palette, transparent."""
    def table(palette):
        n = max(2, len(palette) // 3)
        bits = max(1, (n - 1).bit_length())
        return bits, bytes(palette) + b"\0" * (3 * (1 << bits) - len(palette))

    flags = 0
    gct = b""
    if global_palette is not None:
        bits, gct = table(global_palette)
        flags = 0x80 | (bits - 1)
    out = b"GIF89a" + struct.pack("<HHBBB", screen[0], screen[1], flags, 0, 0) + gct
    for f in frames:
        if f.get("transparent") is not None:
            out += b"!\xf9\x04\x01\x00\x00" + bytes([f["transparent"]]) + b"\x00"
        ind = np.asarray(f["indices"])
        lflags, lct = 0, b""
        if f.get("palette") is not None:
            bits, lct = table(f["palette"])
            lflags = 0x80 | (bits - 1)
        out += b"," + struct.pack("<HHHHB", f["left"], f["top"], ind.shape[1], ind.shape[0],
                                  lflags) + lct + _lzw_literals(ind)
    Path(path).write_bytes(out + b";")
    return path


def _exif(orientation: int) -> bytes:
    im = Image.new("RGB", (1, 1))
    exif = im.getexif()
    exif[0x0112] = orientation
    return exif.tobytes()


def _photo(h=24, w=36, seed=0) -> np.ndarray:
    rng = np.random.default_rng(RNG_SEED + seed)
    y, x = np.mgrid[0:h, 0:w]
    base = np.stack([x * 255 // max(1, w - 1), y * 255 // max(1, h - 1), (x + y) % 256], -1)
    noise = rng.integers(0, 40, size=(h, w, 3))
    return ((base + noise) % 256).astype(np.uint8)


# --- decode parity: committed and sample images ----------------------------------------------

@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_fixtures_decode_identically(path):
    assert_same_decode(path)


@pytest.mark.parametrize("path", SAMPLES or [None], ids=lambda p: p.stem if p else "none")
def test_samples_decode_identically(path):
    if path is None:
        pytest.skip(f"no samples in {SAMPLES_DIR}")
    assert_same_decode(path)


# --- decode parity: PNG in every mode ---------------------------------------------------------

PIL_MODES = ["1", "L", "LA", "P", "RGB", "RGBA", "I;16", "I"]


@pytest.mark.filterwarnings("ignore:Saving I mode images as PNG is deprecated")
@pytest.mark.parametrize("mode", PIL_MODES)
def test_pillow_png_modes(tmp_path, mode):
    rgb = _photo()
    if mode == "I;16":
        im = Image.fromarray(rgb[..., 0].astype(np.uint16) * 257 + rgb[..., 1] % 7)
    elif mode == "I":
        values = rgb[..., 0].astype(np.int32) * 300 - 5000
        im = Image.fromarray(values, mode="I")
    elif mode == "P":
        im = Image.fromarray(rgb).quantize(37)
    else:
        im = Image.fromarray(rgb).convert(mode)
    path = tmp_path / f"{mode.replace(';', '')}.png"
    im.save(path)
    assert_same_decode(path)


def test_pillow_png_transparency_variants(tmp_path):
    rgb = _photo(seed=1)
    p = Image.fromarray(rgb).quantize(16)
    p.save(tmp_path / "p_int.png", transparency=3)
    alpha = bytes(range(0, 256, 23))
    p.save(tmp_path / "p_partial.png", transparency=alpha)
    rgb[2:5, 2:5] = (10, 20, 30)
    Image.fromarray(rgb).save(tmp_path / "rgb_key.png", transparency=(10, 20, 30))
    grey = rgb[..., 0].copy()
    grey[0, :4] = 77
    Image.fromarray(grey).save(tmp_path / "l_key.png", transparency=77)
    Image.fromarray(grey > 128).save(tmp_path / "one_key.png", transparency=1)
    for name in ("p_int", "p_partial", "rgb_key", "l_key", "one_key"):
        assert_same_decode(tmp_path / f"{name}.png")


@pytest.mark.parametrize("interlace", [False, True], ids=["plain", "adam7"])
@pytest.mark.parametrize("bits,color_type,channels", [
    (1, 0, 1), (2, 0, 1), (4, 0, 1), (8, 0, 1), (16, 0, 1),
    (8, 2, 3), (16, 2, 3), (1, 3, 1), (2, 3, 1), (4, 3, 1), (8, 3, 1),
    (8, 4, 2), (16, 4, 2), (8, 6, 4), (16, 6, 4),
])
def test_raw_png_bit_depths(tmp_path, bits, color_type, channels, interlace):
    rng = np.random.default_rng(RNG_SEED + bits * 10 + color_type)
    h, w = 11, 13                      # odd sizes exercise every Adam7 pass and row padding
    top = (1 << bits) - 1
    samples = rng.integers(0, top + 1, size=(h, w, channels))
    plte = trns = None
    if color_type == 3:
        n = max(1, (top + 1) * 3 // 4)  # some indices fall outside PLTE: Pillow's grey ramp
        plte = rng.integers(0, 256, size=3 * n).tolist()
        trns = rng.integers(0, 256, size=max(1, n // 2)).tolist()
    path = raw_png(tmp_path / "raw.png", samples, bits, color_type, plte, trns, interlace)
    assert_same_decode(path)


@pytest.mark.parametrize("bits,color_type,key", [
    (1, 0, [0, 1]), (2, 0, [0, 2]), (4, 0, [0, 0]), (8, 0, [0, 77]), (16, 0, [0x30, 0x39]),
    (8, 2, [0, 10, 0, 20, 0, 30]), (16, 2, [0, 10, 0, 20, 0, 30]),
    (16, 2, [10, 0, 20, 0, 30, 0]),
])
def test_raw_png_colour_keys(tmp_path, bits, color_type, key):
    """Grey and RGB tRNS keys, compared by Pillow against the 8-bit values (16-bit grey: the
    key is dropped with the I;16 -> L rescale)."""
    rng = np.random.default_rng(RNG_SEED + bits)
    channels = 1 if color_type == 0 else 3
    top = (1 << bits) - 1
    samples = rng.integers(0, top + 1, size=(6, 7, channels))
    keyval = [key[i] * 256 + key[i + 1] for i in range(0, len(key), 2)]
    samples[0, :3] = np.minimum(keyval, top)
    if bits == 16 and color_type == 2:
        samples[1, :3] = [v << 8 for v in (10, 20, 30)]
    path = raw_png(tmp_path / "key.png", samples, bits, color_type, trns=key)
    assert_same_decode(path)


def test_png_metadata_chunks_and_text_after_idat(tmp_path):
    """Text and EXIF chunks before and after the image data; zTXt and iTXt."""
    samples = _photo(5, 6)
    before = [(b"tEXt", b"Comment\0before"), (b"zTXt", b"z\0\0" + zlib.compress(b"zipped"))]
    after = [(b"iTXt", "i\0\0\0\0\0é".encode()), (b"eXIf", _exif(6)[6:])]
    path = raw_png(tmp_path / "meta.png", samples, 8, 2, before=before, after=after)
    assert_same_decode(path)
    assert io.read_png_text(path) == ref.read_png_text(path)


def test_png_xmp_and_raw_profile_orientation(tmp_path):
    im = Image.fromarray(_photo(6, 10))
    info = PngInfo()
    info.add_itxt("XML:com.adobe.xmp", '<x:xmpmeta tiff:Orientation="6"/>')
    im.save(tmp_path / "xmp.png", pnginfo=info)
    info = PngInfo()
    hexdata = _exif(8).hex()
    info.add_text("Raw profile type exif", f"\nexif\n{len(hexdata) // 2}\n{hexdata}\n")
    im.save(tmp_path / "raw_profile.png", pnginfo=info)
    for name in ("xmp", "raw_profile"):
        expected, actual, _, _ = _load_both(tmp_path / f"{name}.png")
        assert expected.rgb.shape == (10, 6, 3)
        np.testing.assert_array_equal(actual.rgb, expected.rgb)


@pytest.mark.parametrize("orientation", range(0, 10))
def test_png_exif_orientations(tmp_path, orientation):
    im = Image.fromarray(_photo(5, 9))
    path = tmp_path / f"o{orientation}.png"
    im.save(path, exif=_exif(orientation))
    assert_same_decode(path)


def test_apng_counts_frames_and_uses_the_first(tmp_path):
    frames = [Image.fromarray(_photo(8, 8, seed=s)) for s in range(3)]
    frames[0].save(tmp_path / "anim.png", save_all=True, append_images=frames[1:])
    frames[0].save(tmp_path / "default.png", save_all=True, append_images=frames[1:],
                   default_image=True)
    for name in ("anim", "default"):
        expected, actual, w_ref, w_new = _load_both(tmp_path / f"{name}.png")
        assert w_new == w_ref and len(w_new) == 1
        np.testing.assert_array_equal(actual.rgb, expected.rgb)


# --- decode parity: the other lossless formats ------------------------------------------------

@pytest.mark.parametrize("mode", ["1", "L", "P", "RGB", "RGBA"])
def test_bmp(tmp_path, mode):
    rgb = _photo(7, 10, seed=2)
    im = Image.fromarray(rgb).quantize(12) if mode == "P" else Image.fromarray(rgb).convert(mode)
    if mode == "RGBA":
        im.putalpha(Image.fromarray(rgb[..., 2]))
    path = tmp_path / f"{mode}.bmp"
    im.save(path)
    assert_same_decode(path)


@pytest.mark.parametrize("compression", [None, "tiff_lzw", "tiff_adobe_deflate", "packbits"])
@pytest.mark.parametrize("mode", ["1", "L", "LA", "P", "RGB", "RGBA", "I;16", "I", "CMYK"])
def test_pillow_tiff(tmp_path, mode, compression):
    rgb = _photo(9, 14, seed=3)
    if mode == "I;16":
        im = Image.fromarray(rgb[..., 0].astype(np.uint16) * 257 + 3)
    elif mode == "I":
        im = Image.fromarray(rgb[..., 0].astype(np.int32) * 300 - 5000, mode="I")
    elif mode == "P":
        im = Image.fromarray(rgb).quantize(20)
    else:
        im = Image.fromarray(rgb).convert(mode)
    if mode in ("LA", "RGBA"):
        im.putalpha(Image.fromarray(rgb[..., 1]))
    path = tmp_path / f"{mode.replace(';', '')}.tiff"
    try:
        im.save(path, compression=compression)
    except (OSError, ValueError) as exc:
        pytest.skip(f"Pillow cannot write {mode} with {compression}: {exc}")
    assert_same_decode(path)


@pytest.mark.parametrize("kind", ["uint32", "int32", "uint16_rgb", "assoc_alpha", "extra_unspec",
                                  "bigendian16"])
def test_tifffile_variants(tmp_path, kind):
    tifffile = pytest.importorskip("tifffile")
    rng = np.random.default_rng(RNG_SEED + 7)
    path = tmp_path / f"{kind}.tif"
    if kind == "uint32":
        tifffile.imwrite(path, rng.integers(0, 2**32, size=(6, 8), dtype=np.uint32))
    elif kind == "int32":
        tifffile.imwrite(path, rng.integers(-70000, 70000, size=(6, 8), dtype=np.int32))
    elif kind == "uint16_rgb":
        tifffile.imwrite(path, rng.integers(0, 65536, size=(6, 8, 3), dtype=np.uint16),
                         photometric="rgb")
    elif kind == "bigendian16":
        tifffile.imwrite(path, rng.integers(0, 65536, size=(6, 8), dtype=np.uint16),
                         byteorder=">")
    else:
        data = rng.integers(0, 256, size=(6, 8, 4), dtype=np.uint8)
        data[..., :3] = np.minimum(data[..., :3], data[..., 3:])     # valid premultiplied
        extra = "assocalpha" if kind == "assoc_alpha" else "unspecified"
        tifffile.imwrite(path, data, photometric="rgb", extrasamples=[extra])
    try:
        expected = ref.load(path)
    except PixelforgeError as exc:
        with pytest.raises(PixelforgeError) as info:
            io.load(path)
        assert str(info.value).split(" (")[0] == str(exc).split(" (")[0]
        return
    actual = io.load(path)
    np.testing.assert_array_equal(actual.rgb, expected.rgb)
    np.testing.assert_array_equal(actual.alpha, expected.alpha)


def test_tiff_multipage_and_orientation(tmp_path):
    pages = [Image.fromarray(_photo(6, 9, seed=s)) for s in range(3)]
    pages[0].save(tmp_path / "pages.tiff", save_all=True, append_images=pages[1:])
    assert_same_decode(tmp_path / "pages.tiff")
    pages[0].save(tmp_path / "rot.tiff", tiffinfo={274: 6})
    assert_same_decode(tmp_path / "rot.tiff")


def test_gif_single_and_animated_with_transparency(tmp_path):
    rgb = _photo(10, 12, seed=4)
    single = Image.fromarray(rgb).quantize(30)
    single.save(tmp_path / "single.gif", transparency=5)
    frames = [Image.fromarray(_photo(10, 12, seed=s)).quantize(20) for s in range(3)]
    frames[0].save(tmp_path / "anim.gif", save_all=True, append_images=frames[1:],
                   transparency=2, disposal=2)
    Image.fromarray(rgb[..., 0]).save(tmp_path / "grey.gif")
    for name in ("single", "anim", "grey"):
        assert_same_decode(tmp_path / f"{name}.gif")


def test_gif_partial_first_frame_and_out_of_palette_indices(tmp_path):
    rng = np.random.default_rng(RNG_SEED + 5)
    palette = rng.integers(0, 256, size=12).tolist()            # 4 entries
    frame = {"left": 3, "top": 2, "indices": rng.integers(0, 9, size=(5, 6))}
    raw_gif(tmp_path / "opaque.gif", (12, 9), [frame], palette)
    raw_gif(tmp_path / "trans.gif", (12, 9), [dict(frame, transparent=1)], palette)
    local = dict(frame, palette=rng.integers(0, 256, size=24).tolist(), left=8)  # grows screen
    raw_gif(tmp_path / "grow.gif", (10, 9), [local, frame], palette)
    raw_gif(tmp_path / "noglobal.gif", (12, 9), [frame])
    for name in ("opaque", "trans", "grow", "noglobal"):
        assert_same_decode(tmp_path / f"{name}.gif")


def test_webp_lossless_is_bitwise(tmp_path):
    rgb = _photo(17, 23, seed=6)
    Image.fromarray(rgb).save(tmp_path / "rgb.webp", lossless=True)
    rgba = Image.fromarray(rgb).convert("RGBA")
    rgba.putalpha(Image.fromarray(rgb[..., 0]))
    rgba.save(tmp_path / "rgba.webp", lossless=True)
    rgba.save(tmp_path / "rot.webp", lossless=True, exif=_exif(6))
    frames = [Image.fromarray(_photo(8, 8, seed=s)) for s in range(2)]
    frames[0].save(tmp_path / "anim.webp", save_all=True, append_images=frames[1:],
                   lossless=True)
    for name in ("rgb", "rgba", "rot", "anim"):
        assert_same_decode(tmp_path / f"{name}.webp")


def _max_difference(path):
    expected, actual, _, _ = _load_both(path)
    assert actual.rgb.shape == expected.rgb.shape
    diff = np.abs(_rgba(actual).astype(int) - _rgba(expected).astype(int))
    return int(diff.max()), float((diff.max(axis=2) > 0).mean())


def test_webp_lossy_within_one(tmp_path):
    """The plan accepts ±1 against libwebp; measured: identical on fixtures and samples."""
    for p in FIXTURES:
        im = Image.open(p).convert("RGBA")
        for mode in ("RGB", "RGBA"):
            path = tmp_path / f"{p.stem}_{mode}.webp"
            im.convert(mode).save(path, quality=80)
            assert _max_difference(path)[0] <= 1


def test_jpeg_differences_are_small(tmp_path):
    """zune-jpeg against libjpeg-turbo: grey within 1, colour within a few levels (4:2:0
    chroma upsampling); recorded in CHANGELOG.md."""
    for p in FIXTURES:
        im = Image.open(p).convert("RGB")
        im.save(tmp_path / "c.jpg", quality=90)
        im.convert("L").save(tmp_path / "g.jpg", quality=75)
        assert _max_difference(tmp_path / "c.jpg")[0] <= 8
        assert _max_difference(tmp_path / "g.jpg")[0] <= 1


@pytest.mark.parametrize("orientation", range(1, 9))
def test_jpeg_exif_orientations(tmp_path, orientation):
    """Flat 16-pixel blocks (JPEG blocks decode to their colour within noise), so the
    orientation is what decides the comparison."""
    blocks = np.array([[[200, 30, 30], [30, 200, 30], [30, 30, 200]],
                       [[200, 200, 30], [30, 200, 200], [200, 30, 200]]], dtype=np.uint8)
    img = np.repeat(np.repeat(blocks, 16, axis=0), 16, axis=1)
    path = tmp_path / f"o{orientation}.jpg"
    Image.fromarray(img).save(path, quality=95, exif=_exif(orientation), subsampling=0)
    expected, actual, _, _ = _load_both(path)
    assert actual.rgb.shape == expected.rgb.shape
    assert np.abs(_rgba(actual).astype(int) - _rgba(expected).astype(int)).max() <= 4


def test_jpeg_frames_come_from_the_mpf_index(tmp_path):
    Image.fromarray(_photo()).save(tmp_path / "plain.jpg")
    _, _, w_ref, w_new = _load_both(tmp_path / "plain.jpg")
    assert w_new == w_ref == []
    frames = [Image.fromarray(_photo(seed=s)) for s in range(3)]
    frames[0].save(tmp_path / "multi.jpg", format="MPO", save_all=True,
                   append_images=frames[1:])
    expected, actual, w_ref, w_new = _load_both(tmp_path / "multi.jpg")
    assert w_new == w_ref and len(w_new) == 1
    assert actual.rgb.shape == expected.rgb.shape


# --- errors -----------------------------------------------------------------------------------

def test_budget_message_matches(tmp_path):
    path = tmp_path / "big.png"
    Image.new("RGB", (30, 20)).save(path)
    a, b = _errors(path, 599)
    assert type(a) is type(b) is PixelforgeError and str(a) == str(b)


def test_decompression_bomb_message_matches(tmp_path):
    ihdr = struct.pack(">IIBBBBB", 10000, 10000, 8, 0, 0, 0, 0)
    path = tmp_path / "bomb.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", b"")
                     + _chunk(b"IEND", b""))
    a, b = _errors(path, 10**9)
    assert type(a) is type(b) is PixelforgeError and str(a) == str(b)


def test_directory_and_fifo(tmp_path):
    a, b = _errors(tmp_path)
    assert type(a) is type(b) is PixelforgeError and str(a) == str(b)
    if hasattr(os, "mkfifo"):
        fifo = tmp_path / "pipe.png"
        os.mkfifo(fifo)
        a, b = _errors(fifo)
        assert type(a) is type(b) is PixelforgeError and str(a) == str(b)
        with pytest.raises(PixelforgeError, match="not a regular file"):
            io.sha256_file(fifo)


def test_missing_file_is_the_same_oserror(tmp_path):
    a, b = _errors(tmp_path / "missing.png")
    assert type(a) is type(b) is FileNotFoundError and str(a) == str(b)
    with pytest.raises(FileNotFoundError) as info:
        io.sha256_file(tmp_path / "missing.png")
    assert str(info.value) == str(b)


@pytest.mark.parametrize("payload", [
    b"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 10 10\nshowpage\n",
    b"",
    b"\x89PNG\r\n\x1a\n" + b"garbage" * 4,
    b"BM" + b"\0" * 10,
    b"P6\n1 1\n255\n\0\0\0",
])
def test_unsupported_content_under_a_png_name(tmp_path, payload):
    path = tmp_path / "fake.png"
    path.write_bytes(payload)
    a, b = _errors(path)
    assert type(a) is type(b) is PixelforgeError and str(a) == str(b)


@pytest.mark.parametrize("cut", [0.3, 0.6, 0.9])
def test_truncated_png(tmp_path, cut):
    full = tmp_path / "full.png"
    Image.fromarray(_photo(40, 40)).save(full)
    data = full.read_bytes()
    path = tmp_path / "cut.png"
    path.write_bytes(data[:int(len(data) * cut)])
    a, b = _errors(path)
    assert type(a) is type(b) is PixelforgeError
    prefix = f"{path}: could not decode image ("
    assert str(a).startswith(prefix) and str(b).startswith(prefix)


# --- save_png ---------------------------------------------------------------------------------

def _decoded(path):
    with Image.open(path) as im:
        return (im.mode, im.getpalette(), im.info.get("transparency"), im.tobytes(),
                dict(im.text))


def _index_image(h, w, k, transparent, seed=0):
    rng = np.random.default_rng(RNG_SEED + seed)
    idx = rng.integers(0, k, size=(h, w)).astype(np.int64)
    if transparent:
        idx[rng.random((h, w)) < 0.2] = -1
    return idx


TEXTS = {
    "empty": {},
    "three": {"pixelforge:input_sha256": "ab" * 32, "pixelforge:config": '{"a":1,"b":[2]}',
              "pixelforge:palette": "ff0000\n00ff00"},
}


@pytest.mark.parametrize("text", TEXTS, ids=list(TEXTS))
@pytest.mark.parametrize("scale", [1, 8])
@pytest.mark.parametrize("layout", ["last", "first"])
@pytest.mark.parametrize("transparent", [False, True], ids=["opaque", "alpha"])
@pytest.mark.parametrize("colours", [1, 16, 255, 256, 300])
def test_save_png_decodes_like_pillow(tmp_path, colours, transparent, layout, scale, text):
    palette = np.random.default_rng(RNG_SEED + colours).integers(0, 256, size=(colours, 3))
    palette = palette.astype(np.uint8)
    idx = _index_image(7, 5, colours, transparent, seed=colours)
    ref.save_png(tmp_path / "ref.png", idx, palette, TEXTS[text], scale, layout)
    io.save_png(tmp_path / "new" / "deep" / "new.png", idx, palette, TEXTS[text], scale, layout)
    assert _decoded(tmp_path / "new" / "deep" / "new.png") == _decoded(tmp_path / "ref.png")
    for path in (tmp_path / "ref.png", tmp_path / "new" / "deep" / "new.png"):
        assert io.read_png_text(path) == ref.read_png_text(path) == TEXTS[text]


def test_save_png_one_pixel_and_non_latin1_text(tmp_path):
    palette = np.array([[1, 2, 3]], dtype=np.uint8)
    for idx in (np.array([[0]]), np.array([[-1]])):
        for layout in ("last", "first"):
            text = {"k": "naïve ✓"}
            ref.save_png(tmp_path / "ref.png", idx, palette, text, 1, layout)
            io.save_png(tmp_path / "new.png", idx, palette, text, 1, layout)
            assert _decoded(tmp_path / "new.png") == _decoded(tmp_path / "ref.png")


# --- arrays and helpers -----------------------------------------------------------------------

def _from_both(arr, max_pixels=io.MAX_INPUT_PIXELS):
    results = []
    for module in (ref, io):
        try:
            results.append(module.from_image(arr, max_pixels))
        except PixelforgeError as exc:
            results.append(exc)
    return results


@pytest.mark.parametrize("channels", [3, 4])
@pytest.mark.parametrize("dtype", ["uint8", "float64", "float32"])
def test_from_image_arrays(channels, dtype):
    rng = np.random.default_rng(RNG_SEED + channels)
    if dtype == "uint8":
        arr = rng.integers(0, 256, size=(5, 7, channels), dtype=np.uint8)
    else:
        arr = rng.random((5, 7, channels)).astype(dtype)
    expected, actual = _from_both(arr)
    np.testing.assert_array_equal(actual.rgb, expected.rgb)
    np.testing.assert_array_equal(actual.alpha, expected.alpha)
    assert actual.sha256 == expected.sha256


def test_from_image_half_steps_round_to_even():
    halves = (np.arange(256, dtype=np.float64) + 0.5) / 255.0
    arr = np.clip(np.stack([halves, halves[::-1], np.full(256, 0.5)], -1), 0, 1)[None]
    expected, actual = _from_both(arr)
    np.testing.assert_array_equal(actual.rgb, expected.rgb)
    assert actual.sha256 == expected.sha256


def test_from_image_pil_input():
    im = Image.fromarray(_photo(6, 4)).convert("RGBA")
    expected, actual = _from_both(im)
    np.testing.assert_array_equal(actual.rgb, expected.rgb)
    assert actual.sha256 == expected.sha256


@pytest.mark.parametrize("arr", [
    np.zeros((4, 4), dtype=np.uint8),
    np.zeros((4,), dtype=np.uint8),
    np.zeros((4, 4, 2), dtype=np.uint8),
    np.zeros((4, 4, 3), dtype=np.int64),
    np.zeros((4, 4, 3), dtype=bool),
    np.full((4, 4, 3), 1.5),
    np.full((4, 4, 3), np.nan),
    np.full((4, 4, 3), -0.1),
], ids=["2d", "1d", "2ch", "int64", "bool", "above", "nan", "below"])
def test_from_image_errors(arr):
    expected, actual = _from_both(arr)
    assert isinstance(expected, PixelforgeError) and isinstance(actual, PixelforgeError)
    assert str(actual) == str(expected)


def test_from_image_budget():
    arr = np.zeros((20, 30, 3), dtype=np.uint8)
    expected, actual = _from_both(arr, 599)
    assert str(actual) == str(expected)


def test_indices_to_rgba_and_upscale():
    idx = _index_image(6, 5, 4, True)
    palette = np.arange(12, dtype=np.uint8).reshape(4, 3)
    np.testing.assert_array_equal(io.indices_to_rgba(idx, palette),
                                  ref.indices_to_rgba(idx, palette))
    rgba = ref.indices_to_rgba(idx, palette)
    for arr in (idx, rgba, rgba.astype(np.float64)):
        for factor in (1, 3):
            np.testing.assert_array_equal(io.upscale_nearest(arr, factor),
                                          ref.upscale_nearest(arr, factor))
    with pytest.raises(IndexError, match="out of bounds"):
        io.indices_to_rgba(idx + 4, palette)


def test_open_regular_reads_and_refuses(tmp_path):
    path = tmp_path / "a.bin"
    path.write_bytes(b"hello")
    with io.open_regular(path) as f:
        assert f.read() == b"hello"
    with pytest.raises(PixelforgeError, match="is not a regular file"):
        io.open_regular(tmp_path)
    assert io.sha256_file(path) == ref.sha256_file(path)


@pytest.mark.parametrize("fmt", ["png", "gif", "tiff", "bmp", "webp", "jpeg"])
def test_corrupt_inputs_never_crash(tmp_path, fmt):
    """Truncated and byte-flipped files: a PixelforgeError or a decoded image, never a panic."""
    src = tmp_path / f"src.{fmt}"
    im = Image.fromarray(_photo(12, 10, seed=9))
    if fmt == "png":
        im.save(src, exif=_exif(6))
    elif fmt == "gif":
        im.quantize(8).save(src, save_all=True, append_images=[im.quantize(8)])
    elif fmt == "tiff":
        im.save(src, save_all=True, append_images=[im], compression="tiff_lzw")
    else:
        im.save(src)
    data = src.read_bytes()
    rng = np.random.default_rng(RNG_SEED + len(data))
    variants = [data[:n] for n in range(0, len(data), max(1, len(data) // 40))]
    for _ in range(60):
        b = bytearray(data)
        for i in rng.integers(0, len(b), size=3):
            b[i] = int(rng.integers(0, 256))
        variants.append(bytes(b))
    path = tmp_path / "bad.img"
    for v in variants:
        path.write_bytes(v)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                io.load(path)
        except PixelforgeError:
            pass
