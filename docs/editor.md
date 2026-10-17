# Editor

The editor touches up a conversion in place: a stray pixel, a broken outline, a muddy
patch. It edits the indexed image and its palette and keeps every output file consistent.
It is not a general paint program: no layers, no animation, no free RGB painting. Its
requirements and design are in the [editor design](design/editor.md).

The editor is the `gui` Cargo feature, on by default. A command-line-only build (see
[Install](../README.md#install)) still replays edit logs with `--edits`.

## Opening files

```bash
pixelforge convert hero.png --preset sprite -o out --edit   # convert, then open the editor
pixelforge edit out/hero.png                                # open an earlier output
pixelforge edit out/hero.png --source hero.png -o fixed     # with the input for reference
pixelforge edit any_indexed.png                             # any PNG with at most 256 colors
pixelforge batch frames/ -o out --edit                      # a batch, frame by frame
```

`convert --edit` writes the outputs and prints the JSON result exactly as without the flag.
Then it opens the window with the input image aligned pixel for pixel with the output.

`edit` reads the palette, config and input hash back from a pixelforge PNG. A PNG with at
most 256 colors that pixelforge did not write opens with a palette built from its colors and
the default config.

`edit --source` aligns the input too when the PNG's config and the input's size give back
the conversion's crop and canvas (not with `--remove-bg`). Otherwise the input is shown
fitted, for reference only.

## Saving

Saving (Cmd/Ctrl+S) rewrites the complete output set of `convert` under the same name (or
another, with Save As), including the tileset files, plus `NAME_edits.json`. Closing with
unsaved changes asks first.

## Tools

- Pencil (B, sizes 1 to 8), eraser (E), eyedropper (I, or Alt+click with any tool), fill (G;
  Shift fills every pixel of that color), line (L) and rectangle (R; Shift fills) with a live
  preview. The right button paints the secondary color. X swaps the two, and 1 to 9 pick
  palette entries.
- Rectangular selection (M). Every tool stays inside it. Drag inside it to move its
  contents, and Esc clears it. Cmd/Ctrl+C copies it, also to the system clipboard as an
  image. Cmd/Ctrl+V pastes, and the pasted pixels become the selection, ready to move. The
  editor's own copy pastes with its exact palette indices. An image copied in another
  program is snapped to the nearest palette entries (alpha below 128 is transparent) and
  clipped to the canvas.
- Unlimited undo and redo. Help > Keyboard shortcuts lists every shortcut.

## Palette

The palette panel shows pixel counts and lets you:

- edit an entry's color (every pixel of it changes),
- add an entry,
- remove one (used entries move to the nearest remaining one),
- merge two entries,
- replace one color with another.

With a named palette (`--palette-name`) entries only take that palette's colors. More
entries than `palette_size` shows a warning, not an error.

## Adjusting a selection

- Adjust: brightness, contrast and saturation in CIELAB on the selection, snapped back to the
  palette, with a live preview. "Only colors in selection" keeps pixels within their ramp.
- Ramp shift (`,` darker, `.` lighter) moves pixels one step along their hue ramp.
- The Image menu runs orphan removal and the jaggy fix on the selection with the pipeline's
  own code.
- Dither… fills the selection's opaque pixels with an ordered (8x8 Bayer) pattern of two
  entries at a chosen density, previewed live. The pattern is anchored to the image, so
  dithering neighboring selections gives one seamless pattern.

## Views

- Zoom 1x to 64x, pixel grid and tile grid.
- A diff overlay (D) of what differs from the conversion.
- The source image in a panel and, when aligned, overlaid at an adjustable opacity (Tab).
- With a tileset, the tiles panel shows the tile count after de-duplication and how many
  pixels de-duplication would change on save. It can also draw what the tilemap draws.
- Per-tile editing (T, or View > Per-tile editing) makes every edit also reach every
  identical copy of the tiles it touches, mirrored copies included, inside the selection.

## Batches

`batch --edit` writes the batch's outputs and JSON, then opens a window that steps through
the frames: left and right arrows, or the Frame menu. The status bar shows "frame 2/5".

Each frame has its own undo history and saves under its own name. The frames share the
batch's palette, so a palette edit applies to every frame, and undoing it undoes it in every
frame. The editor asks you to undo later edits of another frame first. Save writes every
frame with changes, and closing asks while any frame is unsaved.

## The edit log

`NAME_edits.json` records every edit as the pixels and palette entries it set, not as tool
invocations. A later change to a tool cannot change what an old log reproduces. Replay it on
a fresh conversion and the edited PNG comes back byte for byte:

```bash
pixelforge convert hero.png --preset sprite -o out --edits out/hero_edits.json
```

Replay needs the same input and config. It refuses (exit 2) a log whose image size or
palette length does not match the conversion. `--edits` works in the command-line-only
build. Opening an edited output again and saving keeps the earlier edits in the new log, so
it still replays from the input.

The saved PNG stays a pixelforge output. It keeps its `pixelforge:config` and
`pixelforge:input_sha256` text chunks and gains `pixelforge:edits_sha256`. `_meta.json`
gains an `edited` object (op count, pixels changed, palette changed, the edit log's hash,
the pixelforge version). The config hash does not change.
