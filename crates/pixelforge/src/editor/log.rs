//! The edit log, `NAME_edits.json` (docs/design/editor.md, Section 3.3): the applied ops as pixel
//! and palette changes, with the tool name as an annotation.
//!
//! ```json
//! {"pixelforge": "0.3.0", "width": 64, "height": 64, "palette_len": 17,
//!  "ops": [{"op": "pixels", "tool": "pencil", "changes": [[12, 30, 3, 5], [13, 30, 3, 5]]},
//!          {"op": "palette_set", "index": 5, "old": "4a3b2c", "new": "4e3d2e"}]}
//! ```
//!
//! The other ops are `{"op": "palette_add", "rgb": "rrggbb"}` and `{"op": "palette_remove",
//! "index": i, "rgb": "rrggbb", "remap": [[x, y, old, new], ...]}` (`new` in the numbering
//! before the removal). The file is `json.dumps(log)` (one line) plus a newline; `width`,
//! `height` and `palette_len` describe the conversion the log applies to.
//!
//! `edits_sha256` is the SHA-256 of the canonical form: the same object without the
//! `pixelforge` version, compact (`separators=(",", ":")`). A log therefore keeps its hash,
//! and an edited PNG its bytes, when a later version replays it.

use std::path::Path;

use sha2::{Digest, Sha256};

use super::{err, Document, EditError, Op, PixelChange};
use crate::color;
use crate::io;
use crate::pipeline::VERSION;
use crate::pyjson::loads::{self, Json as Loaded};
use crate::pyjson::{dumps, Json, Layout};

/// The log's file name suffix: `<prefix>_edits.json`.
pub const EDITS_SUFFIX: &str = "_edits.json";

/// The edits of one document, from the conversion on.
#[derive(Clone, Debug, PartialEq)]
pub struct EditLog {
    /// The pixelforge version that wrote the log (not hashed).
    pub pixelforge: String,
    pub width: usize,
    pub height: usize,
    /// The conversion's palette length, before any op.
    pub palette_len: usize,
    pub ops: Vec<Op>,
}

fn hex(rgb: [u8; 3]) -> Json {
    Json::Str(crate::palette::rgb8_hex(rgb))
}

fn changes_json(changes: &[PixelChange]) -> Json {
    Json::List(
        changes
            .iter()
            .map(|c| {
                Json::List(vec![
                    Json::Int(c.x as i64),
                    Json::Int(c.y as i64),
                    Json::Int(c.old),
                    Json::Int(c.new),
                ])
            })
            .collect(),
    )
}

fn op_json(op: &Op) -> Json {
    let kind = ("op".to_string(), Json::str(op.kind()));
    Json::Object(match op {
        Op::Pixels { tool, changes } => vec![
            kind,
            ("tool".into(), Json::str(tool)),
            ("changes".into(), changes_json(changes)),
        ],
        Op::PaletteSet {
            index, old, new, ..
        } => vec![
            kind,
            ("index".into(), Json::Int(*index as i64)),
            ("old".into(), hex(*old)),
            ("new".into(), hex(*new)),
        ],
        Op::PaletteAdd { rgb } => vec![kind, ("rgb".into(), hex(*rgb))],
        Op::PaletteRemove {
            index, rgb, remap, ..
        } => vec![
            kind,
            ("index".into(), Json::Int(*index as i64)),
            ("rgb".into(), hex(*rgb)),
            ("remap".into(), changes_json(remap)),
        ],
    })
}

// ---------------------------------------------------------------------------------------
// Parsing

fn field<'a>(pairs: &'a [(String, Loaded)], name: &str, at: &str) -> Result<&'a Loaded, EditError> {
    pairs
        .iter()
        .rev()
        .find(|(k, _)| k == name)
        .map(|(_, v)| v)
        .ok_or_else(|| EditError(format!("edit log: {at} has no '{name}'")))
}

fn int(v: &Loaded, what: &str) -> Result<i64, EditError> {
    match v {
        Loaded::Int(crate::config::Value::Int(i)) => Ok(*i),
        _ => err(format!("edit log: {what} must be an integer")),
    }
}

fn count(v: &Loaded, what: &str) -> Result<usize, EditError> {
    usize::try_from(int(v, what)?)
        .map_err(|_| EditError(format!("edit log: {what} must not be negative")))
}

fn text<'a>(v: &'a Loaded, what: &str) -> Result<&'a str, EditError> {
    match v {
        Loaded::Str(s) => Ok(s),
        _ => err(format!("edit log: {what} must be a string")),
    }
}

fn rgb(v: &Loaded, what: &str) -> Result<[u8; 3], EditError> {
    let t = text(v, what)?;
    if t.starts_with('#') || t.trim() != t {
        return err(format!("edit log: {what} must be 'rrggbb', got '{t}'"));
    }
    color::hex_to_rgb8(t).map_err(|e| EditError(format!("edit log: {what}: {e}")))
}

fn changes(v: &Loaded, what: &str) -> Result<Vec<PixelChange>, EditError> {
    let Loaded::List(items) = v else {
        return err(format!("edit log: {what} must be a list"));
    };
    items
        .iter()
        .map(|item| match item {
            Loaded::List(q) if q.len() == 4 => Ok(PixelChange {
                x: count(&q[0], what)?,
                y: count(&q[1], what)?,
                old: int(&q[2], what)?,
                new: int(&q[3], what)?,
            }),
            _ => err(format!("edit log: {what} entries must be [x, y, old, new]")),
        })
        .collect()
}

fn parse_op(v: &Loaded, n: usize) -> Result<Op, EditError> {
    let at = format!("op {n}");
    let Loaded::Object(pairs) = v else {
        return err(format!("edit log: {at} must be an object"));
    };
    let get = |name: &str| field(pairs, name, &at);
    let what = |name: &str| format!("{at} '{name}'");
    let lab = |c: [u8; 3]| {
        let l = color::rgb8_to_lab(&c);
        [l[0], l[1], l[2]]
    };
    match text(get("op")?, &what("op"))? {
        "pixels" => Ok(Op::Pixels {
            tool: text(get("tool")?, &what("tool"))?.to_string(),
            changes: changes(get("changes")?, &what("changes"))?,
        }),
        "palette_set" => {
            let old = rgb(get("old")?, &what("old"))?;
            Ok(Op::PaletteSet {
                index: count(get("index")?, &what("index"))?,
                old,
                new: rgb(get("new")?, &what("new"))?,
                old_lab: lab(old),
            })
        }
        "palette_add" => Ok(Op::PaletteAdd {
            rgb: rgb(get("rgb")?, &what("rgb"))?,
        }),
        "palette_remove" => {
            let c = rgb(get("rgb")?, &what("rgb"))?;
            Ok(Op::PaletteRemove {
                index: count(get("index")?, &what("index"))?,
                rgb: c,
                lab: lab(c),
                remap: changes(get("remap")?, &what("remap"))?,
            })
        }
        other => err(format!("edit log: {at} has an unknown kind '{other}'")),
    }
}

impl EditLog {
    /// A log written by this version.
    pub fn new(width: usize, height: usize, palette_len: usize, ops: Vec<Op>) -> EditLog {
        EditLog {
            pixelforge: VERSION.to_string(),
            width,
            height,
            palette_len,
            ops,
        }
    }

    /// The log of `doc`: the edits it was opened with, then `session` (a history's applied
    /// ops), against the conversion's size and palette.
    pub fn of(doc: &Document, session: &[Op]) -> EditLog {
        let mut ops = doc.origin.prior_ops.clone();
        ops.extend(session.iter().cloned());
        EditLog::new(
            doc.baseline.w,
            doc.baseline.h,
            doc.baseline_palette.len(),
            ops,
        )
    }

    fn body(&self) -> Vec<(String, Json)> {
        vec![
            ("width".into(), Json::Int(self.width as i64)),
            ("height".into(), Json::Int(self.height as i64)),
            ("palette_len".into(), Json::Int(self.palette_len as i64)),
            (
                "ops".into(),
                Json::List(self.ops.iter().map(op_json).collect()),
            ),
        ]
    }

    /// The log as JSON, keys in file order.
    pub fn to_json(&self) -> Json {
        let mut pairs = vec![("pixelforge".to_string(), Json::str(&self.pixelforge))];
        pairs.extend(self.body());
        Json::Object(pairs)
    }

    /// The file text: `json.dumps(log) + "\n"`.
    pub fn to_text(&self) -> String {
        let layout = Layout {
            indent: None,
            sort_keys: false,
            compact: false,
        };
        dumps(&self.to_json(), layout) + "\n"
    }

    /// The hashed form: compact, without the version.
    pub fn canonical(&self) -> String {
        dumps(&Json::Object(self.body()), Layout::COMPACT)
    }

    /// `edits_sha256`: SHA-256 (lowercase hex) of [`canonical`](Self::canonical).
    pub fn sha256(&self) -> String {
        format!("{:x}", Sha256::digest(self.canonical().as_bytes()))
    }

    /// Parse a log's text. Errors name what is wrong; values are checked against a
    /// document only by [`replay`](Self::replay).
    pub fn parse(text_in: &str) -> Result<EditLog, EditError> {
        let parsed = loads::loads(text_in).map_err(|e| match e {
            loads::JsonError::Value(m) | loads::JsonError::Recursion(m) => {
                EditError(format!("edit log: {m}"))
            }
        })?;
        let Loaded::Object(pairs) = parsed else {
            return err("edit log: not a JSON object");
        };
        let at = "the log";
        let Loaded::List(ops) = field(&pairs, "ops", at)? else {
            return err("edit log: 'ops' must be a list");
        };
        Ok(EditLog {
            pixelforge: text(field(&pairs, "pixelforge", at)?, "'pixelforge'")?.to_string(),
            width: count(field(&pairs, "width", at)?, "'width'")?,
            height: count(field(&pairs, "height", at)?, "'height'")?,
            palette_len: count(field(&pairs, "palette_len", at)?, "'palette_len'")?,
            ops: ops
                .iter()
                .enumerate()
                .map(|(n, op)| parse_op(op, n))
                .collect::<Result<_, _>>()?,
        })
    }

    /// Read and parse a log file (a regular UTF-8 file).
    pub fn read(path: &Path) -> Result<EditLog, EditError> {
        let bytes = io::read_regular(path).map_err(|e| EditError(e.to_string()))?;
        let text = String::from_utf8(bytes)
            .map_err(|_| EditError(format!("{} is not a UTF-8 text file", path.display())))?;
        EditLog::parse(&text).map_err(|e| EditError(format!("{}: {}", path.display(), e.0)))
    }

    /// Write the log file.
    pub fn write(&self, path: &Path) -> std::io::Result<()> {
        std::fs::write(path, self.to_text())
    }

    /// Replay the log on `doc` (the conversion it was recorded on). Errors, with `doc`
    /// unusable, when the image size or palette length differ, or an op does not apply: an
    /// index out of range, or a recorded old value the document does not have.
    pub fn replay(&self, doc: &mut Document) -> Result<(), EditError> {
        let (w, h, k) = (doc.width(), doc.height(), doc.palette.len());
        if (self.width, self.height) != (w, h) {
            return err(format!(
                "the edit log is for a {}x{} image, the conversion is {w}x{h}",
                self.width, self.height
            ));
        }
        if self.palette_len != k {
            return err(format!(
                "the edit log is for a palette of {} colors, the conversion has {k}",
                self.palette_len
            ));
        }
        for (n, op) in self.ops.iter().enumerate() {
            doc.replay(op)
                .map_err(|e| EditError(format!("edit log op {n} ({}): {}", op.kind(), e.0)))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::editor::document::tests::doc;

    fn sample() -> EditLog {
        EditLog::new(
            4,
            2,
            3,
            vec![
                Op::pixels(
                    "pencil",
                    [PixelChange {
                        x: 3,
                        y: 1,
                        old: -1,
                        new: 2,
                    }],
                ),
                Op::PaletteSet {
                    index: 1,
                    old: [0x4a, 0x3b, 0x2c],
                    new: [0x4e, 0x3d, 0x2e],
                    old_lab: [0.0; 3],
                },
                Op::PaletteAdd { rgb: [1, 2, 3] },
                Op::PaletteRemove {
                    index: 0,
                    rgb: [0, 0, 0],
                    lab: [0.0; 3],
                    remap: vec![PixelChange {
                        x: 0,
                        y: 0,
                        old: 0,
                        new: 3,
                    }],
                },
            ],
        )
    }

    #[test]
    fn json_round_trip() {
        let log = sample();
        let text = log.to_text();
        assert!(text.starts_with(&format!(
            "{{\"pixelforge\": \"{VERSION}\", \"width\": 4, \"height\": 2, \"palette_len\": 3, \
             \"ops\": [{{\"op\": \"pixels\", \"tool\": \"pencil\", \"changes\": [[3, 1, -1, 2]]}}, \
             {{\"op\": \"palette_set\", \"index\": 1, \"old\": \"4a3b2c\", \"new\": \"4e3d2e\"}}"
        )));
        let back = EditLog::parse(&text).unwrap();
        assert_eq!(back, log);
        assert_eq!(back.to_text(), text);
        assert_eq!(back.sha256(), log.sha256());
        // The version is not hashed.
        let mut other = log.clone();
        other.pixelforge = "9.9.9".into();
        assert_eq!(other.sha256(), log.sha256());
        assert!(!log.canonical().contains("pixelforge"));
    }

    #[test]
    fn parse_errors() {
        for (text, message) in [
            ("[]", "edit log: not a JSON object"),
            (
                r#"{"pixelforge": "x", "width": 1, "height": 1, "palette_len": 1}"#,
                "edit log: the log has no 'ops'",
            ),
            (
                r#"{"pixelforge": "x", "width": -1, "height": 1, "palette_len": 1, "ops": []}"#,
                "edit log: 'width' must not be negative",
            ),
            (
                r#"{"pixelforge": "x", "width": 1, "height": 1, "palette_len": 1,
                    "ops": [{"op": "pixels", "tool": "t", "changes": [[0, 0, 1]]}]}"#,
                "edit log: op 0 'changes' entries must be [x, y, old, new]",
            ),
            (
                r##"{"pixelforge": "x", "width": 1, "height": 1, "palette_len": 1,
                    "ops": [{"op": "palette_add", "rgb": "#000000"}]}"##,
                "edit log: op 0 'rgb' must be 'rrggbb', got '#000000'",
            ),
            (
                r#"{"pixelforge": "x", "width": 1, "height": 1, "palette_len": 1,
                    "ops": [{"op": "rotate"}]}"#,
                "edit log: op 0 has an unknown kind 'rotate'",
            ),
        ] {
            assert_eq!(EditLog::parse(text).unwrap_err().0, message, "{text}");
        }
    }

    #[test]
    fn replay_checks_size_palette_and_values() {
        let pal = [[0, 0, 0], [0x4a, 0x3b, 0x2c], [9, 9, 9]];
        let mut d = doc(&["0121", "112."], &pal);
        let log = sample();
        log.replay(&mut d).unwrap();
        assert_eq!(d.indices.indices, vec![2, 0, 1, 0, 0, 0, 1, 1]);
        assert_eq!(d.palette, vec![[0x4e, 0x3d, 0x2e], [9, 9, 9], [1, 2, 3]]);

        let mut small = doc(&["01", "12"], &pal);
        let e = log.replay(&mut small).unwrap_err();
        assert_eq!(
            e.0,
            "the edit log is for a 4x2 image, the conversion is 2x2"
        );
        let mut fewer = doc(&["0110", "110."], &pal[..2]);
        assert!(log
            .replay(&mut fewer)
            .unwrap_err()
            .0
            .contains("palette of 3"));
        // The pencil op expects (3, 1) to be transparent.
        let mut other = doc(&["0121", "1120"], &pal);
        assert_eq!(
            log.replay(&mut other).unwrap_err().0,
            "edit log op 0 (pixels): pixel (3, 1) is 0, the log expects -1"
        );
    }
}
