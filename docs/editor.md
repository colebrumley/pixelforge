# pixelforge editor: requirements and design

Status: accepted 2026-10-12; built. All four milestones of Section 3.15 are in place: the
`editor` core and `convert --edits`, the window (`pixelforge edit`, `convert --edit`), the
selection, shape, adjust and cleanup tools with the source reference and diff overlay, and
the tiles panel, named-palette picker, plain indexed PNGs and the shortcuts sheet, released
as 0.3.0. The version is part of every config hash, so the release commit regenerated the
`config_hash` entries of `tests/golden.json`; no pixel hash changed (N4). The items deferred
as "later" or "once X exists" are built too, for 0.3.0: `batch --edit` (R4), ordered dither
(R18), the OS clipboard (R30), the aligned source for `edit --source` (Section 3.8) and
per-tile editing (Section 3.9). Re-running the pipeline from the editor stays out (Section
4, decision 5). Section 4 records the decisions.

pixelforge converts an image and stops. When the result has a stray pixel, a broken outline
or a muddy patch, the user has to open another program, which loses the palette, the
metadata and the tileset. The editor closes that gap: a small GUI inside the `pixelforge`
program for touching up a conversion in place.

It is not an Aseprite replacement. There is no animation timeline, no layers and no
free-form RGB painting. The editor works on what pixelforge produces, an indexed image and
its palette, and keeps every output file consistent when the user saves.

## 1. Scenarios

1. **Fix a stray pixel.** Convert a sprite, see one wrong pixel in the eye, click it with the
   right palette color, save. Every output file (`NAME.png`, the preview, the palette files,
   `_meta.json`) is rewritten and still agrees.
2. **Make an area read better.** A face converted flat. Select the face, raise the contrast,
   watch the pixels snap to lighter and darker entries of the existing palette, apply.
3. **Repair an outline.** Draw the missing outline pixels with the outline color. With a
   named hardware palette the editor offers only that palette's colors.
4. **Clean a region the automatic passes missed.** Select it and run orphan removal or the
   jaggy fix on the selection only, with the same code the pipeline uses.
5. **Touch up a background.** Paint over a tile seam. On save the tileset, tilemap, Tiled
   map and CSV are re-extracted from the edited image.
6. **Reproduce the result later.** The edits are recorded. `pixelforge convert INPUT
   --edits NAME_edits.json` gives the edited PNG again from the source, byte for byte.

## 2. Requirements

"Must" is the first release of the editor. "Should" follows once the core works. "Later"
is recorded so the design leaves room for it.

### Entry points

- R1 (must) `pixelforge edit OUTPUT.png` opens an existing pixelforge output. The palette
  comes from the PNG's `PLTE`/`tRNS`, the config and input hash from its text chunks.
  `--source INPUT` gives the editor the original image for reference. `-o OUTDIR` chooses
  where saves go (default: next to the opened file, same stem).
- R2 (must) `pixelforge convert INPUT ... --edit` converts, writes the outputs and prints
  the JSON result exactly as without the flag, then opens the editor on the result with
  the source image already loaded. Saving rewrites the outputs.
- R3 (should) `pixelforge edit` also opens any PNG with at most 256 distinct colors, with
  the palette built from its colors and a default config. It saves the same file set.
- R4 (later; built) `batch --edit` steps through the frames of a batch with their shared
  palette. Each frame keeps its own document, history and save; a palette edit applies to
  every frame (one undo step in each, undone in all of them, refused while another frame
  has later steps), and Save writes every changed frame, so the files keep one palette.

### Document and palette

- R5 (must) The document is an index image (`-1` = transparent, as everywhere in the
  library) plus a palette. Every tool writes palette indices. The saved PNG is
  palette-indexed with the same `transparent_index` rules as `convert`.
- R6 (must) The palette panel lists every entry with its index, hex and pixel count, and
  the transparent slot. Left click picks the primary color, right click the secondary.
- R7 (must) Editing an entry's color recolors every pixel of that index. Adding an entry is
  allowed up to 255 colors plus transparency. Removing an entry is allowed only when it
  is unused (or the user confirms a remap to the nearest remaining color).
- R8 (must) With a named palette (`palette_name` set) an entry can only take a color from
  that palette, and a new entry is picked from its unused colors. The `.hex` and `.json`
  palette files keep the full list, as `convert` does.
- R9 (should) Merge two entries (every pixel of A becomes B, A is removed).
- R10 (should) The config's `palette_size` is shown. Exceeding it shows a warning badge, not
  an error.

### Tools

- R11 (must) Pencil (sizes 1 to 8, square), eraser (paints transparent), eyedropper
  (Alt+click from any tool), flood fill (4-connected; Shift fills every pixel of that
  index), rectangular selection. A drag is one undo step.
- R12 (must) Tools respect the selection: nothing outside it changes.
- R13 (should) Line and rectangle (outline; Shift for filled) with a live preview while
  dragging; move the selection's contents by dragging inside it.
- R14 (should) Replace color: index A to index B, in the selection or the whole image.
- R15 (must) Adjust: brightness, contrast and saturation sliders applied to the selection
  (or the whole image), computed in CIELAB on each pixel's current palette color and
  snapped back to the nearest palette entry. Live preview, Apply and Cancel. A checkbox
  limits the result to the colors already present in the selection, so a contrast boost
  moves pixels within their ramp instead of pulling in foreign colors.
- R16 (should) Ramp shift: move every selected pixel one step lighter or darker, to the
  nearest palette entry of similar hue with higher or lower L. This is the one-keystroke
  way to add a highlight or a shadow.
- R17 (should) Cleanup on the selection: orphan removal and the jaggy fix, using the
  pipeline's `postprocess` functions with the config's parameters.
- R18 (later; built) Ordered dither between two chosen entries inside the selection: an
  8 × 8 Bayer threshold matrix anchored to image coordinates, density 0 to 64, integer
  arithmetic only, previewed live and applied as one `dither` op.

### View

- R19 (must) Nearest-neighbor zoom from 1x to 64x, pan (Space+drag, middle button,
  scroll), zoom to fit, checkerboard behind transparent pixels, pixel grid at 8x and above.
- R20 (must) Status bar: cursor position, index and hex under the cursor, zoom, image size,
  palette count, unsaved marker.
- R21 (must) Tile grid overlay when the config has `tileset` set (`tile_size` squares).
- R22 (should) Source reference: the input image in a side panel. When the editor was
  opened from `convert --edit`, the source is cropped and padded with the pipeline's own
  layout so that it aligns pixel for pixel with the output, and it can be overlaid on the
  canvas at adjustable opacity. From `edit --source` without the geometry, it is shown
  fitted to the panel, unaligned.
- R23 (should) Diff overlay: tint the pixels that differ from the conversion result.
- R24 (should) Tiles panel for backgrounds: tile count after de-duplication, and a preview of
  what the tilemap will draw (see "Tiles" under Design).

### History, saving, provenance

- R25 (must) Unlimited undo and redo within the session, with a savepoint marker.
- R26 (must) Save writes the complete output set of `convert` from the edited document
  through the library's own `Output::save`: `NAME.png`, `_preview.png`, `_palette.json`,
  `_palette.hex`, `_meta.json` and, with a tileset, `_tileset.png`, `_tilemap.json`,
  `.tmj` and `_tilemap.csv`. Save As takes a new prefix. The editor refuses to write over
  the source input, as `convert` does.
- R27 (must) The edited PNG stays a pixelforge output: it keeps the `pixelforge:config`
  and `pixelforge:input_sha256` text chunks and gains `pixelforge:edits_sha256`.
  `_meta.json` gains an `edited` object: op count, pixels changed, palette changed,
  the edits hash and the editor (pixelforge) version. The config hash does not change,
  because the config did not.
- R28 (must) Save also writes `NAME_edits.json`, the edit log: every change as the pixels
  and palette entries it set. `pixelforge convert INPUT ... --edits NAME_edits.json`
  replays the log after conversion, so the edited result is reproducible from the source,
  the config and the log. Replay refuses a log whose image size or palette length does not
  match the conversion.
- R29 (must) Closing with unsaved changes asks first.
- R30 (should) Copy and paste of the selection inside the editor. (later; built) OS
  clipboard: copy also puts the selection there as an RGBA image; paste keeps the exact
  indices of the editor's own copy and snaps any other image to the nearest palette entry
  (alpha below 128 transparent), clipped to the canvas, as one `paste` op.

### Non-functional

- N1 (must) Pure Rust. Builds with the Rust toolchain alone on macOS and Linux, like the
  rest of the program. Windows stays unsupported.
- N2 (must) The editor is a Cargo feature (`gui`) of the `pixelforge` crate. It is on by
  default so that `cargo install --git ...` gives one program with the editor in it, and
  `--no-default-features --features cli` builds the headless CLI as today. The library
  without default features, and therefore the Python binding, never compiles a GUI
  dependency. The cost is build time (a few minutes more on first build) and binary size
  (roughly 10 MB more).
- N3 (must) Determinism. Every edit operation is a pure function of the document and the
  operation. The same log on the same conversion gives the same bytes on every machine.
  The GUI's frame loop, pointer coordinates and timers never enter the numerics. Adjust and
  ramp shift use `color.rs` only (CIELAB conversions, CIE76 nearest index) plus
  arithmetic, so the existing determinism guard applies to the new code unchanged.
- N4 (must) `convert`, `batch` and `compare` without `--edit` or `--edits` are
  byte-identical to today. No golden hash changes.
- N5 (must) Responsive at the sizes pixelforge produces (up to 4096 on an edge). Pixel
  tools update only the pixels they touched. Adjust previews on large selections and tile
  re-extraction run on a worker thread and never block the frame.
- N6 (must) The editor core (document, tools, history, log) has no GUI dependency and is
  tested headlessly. The GUI layer is thin.
- N7 (must) No network access, no telemetry, no files written outside the output prefix.

### Non-goals

Animation (frames, timeline, onion skin), layers, free RGB painting, custom brushes,
text, rotate and scale transforms, a map editor for the tilemap, Aseprite file import or
export, Windows, a web or WASM build. Changing pipeline parameters inside the editor and
re-running the conversion is attractive but out of scope for the first release (see
"Decisions").

## 3. Design

### 3.1 Architecture

Three layers, from pure to impure:

1. `pixelforge::editor` (library module, always compiled, no GUI dependency):
   `Document`, `Op`, `History`, `Selection`, the tools as functions from pointer input to
   an `Op`, `adjust`, `ramp_shift`, `cleanup`, and `EditLog` with its replay.
2. `pixelforge::gui` (behind the `gui` feature): the eframe application. Panels, the canvas
   widget, keyboard and mouse mapping, textures, a worker thread for slow previews. It owns
   one `Document` and one `History` and calls into `editor`.
3. CLI wiring (behind `cli`): the `edit` subcommand, `--edit` on `convert`, `--edits` on
   `convert`. `--edits` needs only layer 1, so replay works in a headless build.

The GUI lives in the same crate because the `pixelforge` binary is defined there and Cargo
does not allow a separate GUI crate to depend on the library while the library's binary
depends on it. The GUI code is written to the crate's determinism rules (no `.round()`, no
platform transcendental functions; zoom levels are powers of two and screen math is integer
or `round_ties_even`), so `tests/determinism_guard.rs` keeps scanning every source file with
nothing allow-listed.

### 3.2 Document model

```rust
pub struct Document {
    pub indices: IndexImage,            // pipeline::IndexImage: Vec<i64>, h, w; -1 transparent
    pub palette_lab: Vec<f64>,          // K*3 CIELAB, the source of truth for colors
    pub palette: Vec<[u8; 3]>,          // lab_to_rgb8(palette_lab), what the PNG stores
    pub fixed: Option<Vec<[u8; 3]>>,    // the named palette's full color list, if any
    pub origin: Origin,                 // what the document was opened from
    pub baseline: IndexImage,           // the conversion result, for the diff overlay
}

pub struct Origin {
    pub config: Config,
    pub input_sha256: String,
    pub stats: Json,                    // the conversion's stats, carried into _meta.json
    pub source: Option<SourceRef>,      // input image for the reference panel
}
```

Opened from `convert --edit`, the document is built from the `Output` in memory, so
`palette_lab` is the pipeline's own and the untouched palette entries in `_palette.json`
are written with the same six decimals as before. Opened from a PNG, `palette_lab` is
`rgb8_to_lab` of the `PLTE` entries, and the LAB values in `_palette.json` may differ in the
last decimals from the original conversion's. `_meta.json` next to the PNG, when present,
supplies `stats`; otherwise `stats` is minimal.

### 3.3 Operations, history and the log

Every tool reduces to one primitive op type:

```rust
pub enum Op {
    /// Pixels set to new indices. `old` is filled in when the op is applied, for undo.
    Pixels { tool: &'static str, changes: Vec<PixelChange> },   // (x, y, old, new)
    PaletteSet { index: usize, old: [u8; 3], new: [u8; 3] },
    PaletteAdd { rgb: [u8; 3] },                                   // appended at K
    PaletteRemove { index: usize, rgb: [u8; 3], remap: Vec<PixelChange> },
}
```

Brushes, fills, lines, rectangles, adjust, ramp shift, cleanup and merge all compute a
`Vec<PixelChange>` and apply it as one `Pixels` op. A drag stroke accumulates into one op
and commits on release. `History` keeps the applied ops in order with an undo pointer and a
savepoint index. Undo applies the `old` values, redo the `new` ones. Memory is proportional
to the pixels touched, not to the image, so a 4096x4096 canvas is fine.

The edit log stores the applied ops after undo is resolved, as pixel and palette changes
with the tool name as annotation, not as tool invocations. Replay is then independent of
the tool algorithms: a future change to flood fill or to the adjust formula cannot change
what an old log reproduces. The log is JSON written with the crate's `pyjson` layouts and
parsed with the CLI's JSON reader:

```json
{"pixelforge": "0.3.0", "width": 64, "height": 64, "palette_len": 17,
 "ops": [{"op": "pixels", "tool": "pencil", "changes": [[12, 30, 3, 5], [13, 30, 3, 5]]},
         {"op": "palette_set", "index": 5, "old": "4a3b2c", "new": "4e3d2e"}]}
```

`edits_sha256` is the SHA-256 of the log's canonical form. Replay checks width, height
and palette length against the conversion, applies each op, and fails with exit 2 on a
mismatch or an index out of range.

### 3.4 Tools

Each tool is a function in `editor::tools` from the document, the selection and the
pointer path to a change list. None of them touches the GUI.

- Pencil: a square of the brush size at each pointer position, with Bresenham between
  successive positions so fast drags leave no gaps. Left button paints the primary index,
  right button the secondary. Eraser is the pencil with index `-1`.
- Fill: 4-connected flood from the clicked pixel over its index, clipped to the selection.
  Shift replaces every pixel of that index inside the selection instead.
- Line and rectangle: Bresenham and the four edges; filled rectangle with Shift. The
  preview during the drag is the same change list drawn on top of the canvas, not applied.
- Selection: a rectangle in pixel coordinates, not persisted. Esc clears it. Moving the
  contents is a `Pixels` op that writes transparency at the origin and the moved indices at
  the destination, clipped to the canvas.
- Replace color: every pixel of index A inside the selection (or the image) to B.

### 3.5 Adjust, ramp shift and cleanup

Adjust works per pixel on the palette color the pixel currently has, so two pixels of the
same index always get the same result:

1. Take `lab = palette_lab[index]`.
2. Brightness: `L += b` (b in −50..50). Contrast: `L = m + c * (L − m)` where `m` is the
   mean L of the selected pixels and `c` in 0.5..2.0. Saturation: `a *= s; b *= s` with
   `s` in 0..2. Clamp L to 0..100.
3. Snap with `color::nearest_index` to the palette, or, with "only colors in the
   selection" checked, to the subset of entries used inside the selection.
4. Transparent pixels never change.

Only +, −, × and ÷ are involved, and the LAB and nearest-index code already exists and is
already covered by the oracle tests, so the result is identical everywhere. The preview is
the change list computed from the current slider values, recomputed when a slider moves. A
selection of up to a few hundred thousand pixels previews within a frame; above that the
computation runs on the worker thread with a generation counter and the canvas shows the
latest finished preview.

Ramp shift finds, for each selected pixel's entry, the palette entries whose `(a, b)` lie
within a fixed chroma distance (ΔE over a and b only, threshold 20) and picks the one with
the smallest L above (lighter) or below (darker) the current L. Pixels with no such
neighbor stay. This matches how the pipeline's ramp regularization groups colors and gives
a one-key highlight or shadow that stays in the palette.

Cleanup cuts the selection's bounding box plus a one-pixel margin out of the index image,
runs `postprocess::remove_orphans` or `postprocess::fix_jaggies` on it with the config's
`orphan_min_region` and `orphan_max_delta`, and keeps only the changes inside the
selection. The margin lets the passes see the neighbors at the selection's edge.

### 3.6 Palette rules

- Free palette (median cut or gerstner): colors are editable and entries can be added up to
  255 colors. The transparent slot is not a palette entry; `Output::save` places it by
  `transparent_index` as today.
- Named palette (`fixed` is `Some`): `PaletteSet` and `PaletteAdd` are only offered from
  the fixed list, and the color picker is the list of that palette's colors with the
  unused ones marked. The outline entry the pipeline snapped stays what it is.
- Removing a used entry is a `PaletteRemove` with a `remap` that moves its pixels to the
  nearest remaining entry (CIE76); the confirmation dialog shows how many pixels move.
  Indices above the removed one shift down, in the pixels and in the op.
- `palette_size` from the config is informational. The panel shows "17 / 16" in a warning
  color when exceeded.

### 3.7 Canvas rendering

The canvas is an egui widget that owns one RGBA texture of the native image with nearest
filtering, drawn at the current zoom. Pixel ops update the texture through partial uploads
of the dirty rectangle. The checkerboard is drawn under it, the pixel grid and tile grid as
line segments restricted to the visible viewport (at 64x a 4096-wide image still only has a
few hundred visible lines), the selection as a marquee, the tool preview as a second
translucent texture over the dirty area, and the diff overlay as a tint drawn where
`indices != baseline`.

Pointer positions map to pixel coordinates by integer division of the offset from the
image origin by the zoom; the zoom is a power of two from 1 to 64 and never fractional.

### 3.8 Source reference

From `convert --edit` the pipeline already knows the input and its `Geometry`: the crop,
the canvas padding, the outline margin and the fit mode. A new helper in `pipeline`,
`source_reference(&Loaded, &Preprocessed, factor)`, renders the input through the same
crop and padding at an integer multiple of the output size, so output pixel `(x, y)` covers
source block `(factor*x .. factor*x+factor, ...)`. The editor shows it in a panel and as an
overlay with an opacity slider. From `edit --source` the geometry is recovered from the
embedded config and the source image's size (built): `SourceRef::aligned` runs
`preprocess::run` with the output's config and, when that gives the output's size, renders
the reference with `source_reference` at the factor `convert --edit` uses. With
`remove_bg` (the external matting is not run again), a plain PNG (default config) or
another size, the panel shows the image fitted, with no overlay.

### 3.9 Tiles

A background conversion is defined by its tileset and tilemap; the image is what the
tilemap draws. Editing the image and then re-extracting tiles may merge an edited tile into
a near-duplicate under `tile_dedupe_tolerance`, changing the user's pixels at save time.
The editor handles it in two parts:

- The tiles panel re-runs `tiles::extract` and `tiles::reconstruct` on the worker thread
  after each edit (debounced) and shows the tile count and the number of pixels the
  de-duplication would change. A toggle shows the reconstructed image on the canvas.
- Save uses the reconstructed image, as `convert` does, and rewrites the tileset files from
  it. The `edited` object in `_meta.json` records `tiles_rerender_px_changed` separately.

Per-tile editing (built; T or View > Per-tile editing) paints every instance of a tile at
once: `editor::tools::mirror_tiles` maps each change through `tiles::extract`'s tilemap at
tolerance 0 (exact copies and flipped copies; a mirrored instance gets the change
mirrored), keeps the copies inside the selection, and the tool's change list and the copies
are one `Pixels` op.

### 3.10 Saving

Save assembles a `pipeline::Output` from the document: the index image, the palette and
LAB palette, the origin's config and input hash, `stats` with the `edited` object added,
and the tileset fields when `config.tileset` is set. Then `Output::save_with` writes every
file the way `convert` does, including the preview at `scale_preview` and the text chunks.
The PNG text chunks come from `Output::png_text` plus `pixelforge:edits_sha256`; `png_text`
gains an optional extras argument for it. The edit log is written next to the PNG. The
history's savepoint moves to the current op.

Save As asks for a new prefix with a native file dialog (`rfd`, which on Linux talks to the
XDG desktop portal and needs no GTK at build time). Plain Save needs no dialog.

### 3.11 GUI framework

eframe/egui 0.36, pinned exactly like every other dependency, with the `glow` (OpenGL)
backend. Reasons:

- Pure Rust, no system toolchain beyond Rust. winit opens X11 or Wayland libraries at run
  time, so a Linux build needs no development headers; macOS needs nothing.
- Immediate mode fits this tool: a dozen panels, one custom canvas, state that lives in
  plain structs. The texture API (nearest filtering, partial updates) is what the canvas
  needs.
- Mature, widely used for exactly this kind of utility, and `egui_kittest` can drive the
  application headlessly in tests.

Rejected: Tauri (needs a JavaScript toolchain and a system WebView), Slint (a separate UI
language and a license to think about), iced (sound, but custom canvas interaction is
more work and the ecosystem is thinner), raw SDL or wgpu (would mean writing widgets).

New dependencies, all optional under `gui`: `eframe` (with `egui`), `rfd`, and `arboard`
(3.7.0, the version egui-winit already uses, without default features but `image-data`;
X11 through `x11rb`, which winit already builds and which opens libxcb at run time, so no
headers) for the OS clipboard. `Cargo.lock` for the default feature set grows; the library's
dependency tree without `gui` does not change, and a CI check builds
`--no-default-features --features cli` to prove it.

### 3.12 Layout

```
┌──────────────────────────────────────────────────────────────────────┐
│ File  Edit  View  Select  Image            hero.png *   17 colors    │
├────┬─────────────────────────────────────────────────┬───────────────┤
│ B  │                                                 │ Palette       │
│ E  │                                                 │ ■ 0  1a1a2e   │
│ I  │                                                 │ ■ 1  3b2f4a   │
│ G  │              canvas (zoomed, grid)              │ ...           │
│ L  │                                                 │ ▢ T  transp.  │
│ R  │                                                 ├───────────────┤
│ M  │                                                 │ Adjust        │
│    │                                                 │ bright ──●──  │
│ 1  │                                                 │ contr. ───●─  │
│ 2  │                                                 │ sat.   ──●──  │
│ 4  │                                                 │ ☐ only sel.   │
│ 8  │                                                 │ [Apply][Cancel]│
│    ├─────────────────────────────────────────────────┼───────────────┤
│    │ source reference (aligned)         overlay ──●──│ Tiles: 61     │
├────┴─────────────────────────────────────────────────┴───────────────┤
│ 23,41  idx 5  #4a3b2c   16x   64x64   unsaved                        │
└──────────────────────────────────────────────────────────────────────┘
```

Shortcuts follow Aseprite where it has one: B pencil, E eraser, I eyedropper, G fill,
L line, R rectangle, M select, X swap primary and secondary, `[` `]` brush size, 1 to 9
pick palette entries, Alt+click eyedropper, Shift modifies fill, line and rectangle,
Space+drag pans, scroll zooms, `+` `-` `0` zoom in, out and fit, Cmd/Ctrl+Z and
Shift+Cmd/Ctrl+Z undo and redo, Cmd/Ctrl+S save, Cmd/Ctrl+Shift+S save as, Cmd/Ctrl+C
and Cmd/Ctrl+V copy and paste, Esc clears the selection, `,` `.` ramp shift darker and
lighter, Tab toggles the source overlay, D toggles the diff overlay, T toggles per-tile
editing, and the left and right arrows step through the frames of `batch --edit`.

### 3.13 Threading

The main thread owns the `Document` and `History`. The worker receives a cloned snapshot
of the selection's pixels (adjust preview) or the whole index image (tile re-extraction)
with a generation number and sends back the change list. Results with a stale generation
are dropped. `PIXELFORGE_THREADS` has no role; the worker is one thread.

### 3.14 Testing

- Unit tests in `editor`: apply then undo restores the document; flood fill, line and
  rectangle against hand-drawn expectations; adjust is deterministic and snapping is
  idempotent; palette removal remaps and reindexes correctly; named palettes reject foreign
  colors; the log round-trips through JSON.
- Replay golden: a recorded log per fixture in `tests/edits/` and the SHA-256 of the
  replayed PNG, checked in CI next to `golden.json`.
- Save round trip: document to `Output::save` to `pixelforge edit` gives the same document.
- CLI tests: `edit` on a non-pixelforge PNG, `--edits` with a mismatching size exits 2,
  `convert --edits` reproduces an edited output byte for byte.
- GUI smoke test with `egui_kittest`: the application opens a fixture, renders a frame,
  paints one pixel through synthetic input, and the document shows the change.
- CI: the existing matrix (Linux x86-64, macOS arm64) builds with the default features and
  with `--no-default-features --features cli`, and runs clippy on both.

### 3.15 Milestones

1. `editor` core: document, ops, history, pencil, eraser, fill, selection, log and
   replay; `--edits` on `convert`; tests. No GUI yet, but every edit is reproducible.
2. The eframe application: canvas, pencil, eraser, eyedropper, fill, palette panel,
   undo and redo, save; `pixelforge edit` and `convert --edit`.
3. Selection tools, line and rectangle, replace color, adjust, ramp shift, cleanup, source
   reference, diff overlay.
4. Tiles panel and reconstructed preview, named-palette picker, opening arbitrary indexed
   PNGs, shortcuts sheet, README and CHANGELOG, 0.3.0.

## 4. Decisions

These were open questions in the proposal and were decided on 2026-10-12.

1. **Default feature.** `gui` is a default feature: one install gives the whole program.
   `--no-default-features --features cli` builds the headless CLI as today.
2. **`convert --edit` and files.** The outputs are written first and the editor opens on
   them, so a closed editor never loses the conversion and the JSON result keeps its
   contract.
3. **Edit log.** The log and `--edits` replay stay in milestone 1. The core is cheap once
   ops are pixel diffs, and it keeps edited outputs reproducible.
4. **Tiles at save time.** The tiles panel shows what de-duplication would change, and save
   keeps the config's `tile_dedupe_tolerance`, as `convert` does. A user who wants exact
   pixels sets the tolerance to 0 in the config.
5. **Re-running the pipeline from the editor** is out of 0.3.0. It needs the log to carry
   colors rather than indices, because the palette changes between runs.
