//! `configure_logging`: the library's `log` records (every target under `pixelforge`) and the
//! CLI's own warnings on stderr as `pixelforge: {message}`. WARNING by default, INFO with
//! `-v`, DEBUG with `-vv`, only ERROR with `-q`. stdout carries nothing but the JSON lines.

use std::io::Write;

use log::{LevelFilter, Log, Metadata, Record};

/// The target of the CLI's own messages.
pub const TARGET: &str = "pixelforge";

struct StderrLogger;

static LOGGER: StderrLogger = StderrLogger;

impl Log for StderrLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        let target = metadata.target();
        metadata.level() <= log::max_level()
            && (target == TARGET || target.starts_with("pixelforge::"))
    }

    fn log(&self, record: &Record) {
        if self.enabled(record.metadata()) {
            // Logging must never break a run: a failed write is ignored.
            let _ = writeln!(std::io::stderr().lock(), "pixelforge: {}", record.args());
        }
    }

    fn flush(&self) {
        let _ = std::io::stderr().flush();
    }
}

/// The level for `-v` given `verbose` times, or `-q`.
pub fn level(verbose: u8, quiet: bool) -> LevelFilter {
    if quiet {
        LevelFilter::Error
    } else {
        match verbose {
            0 => LevelFilter::Warn,
            1 => LevelFilter::Info,
            _ => LevelFilter::Debug,
        }
    }
}

/// Install the stderr logger (once per process) at the level of `-v` / `-q`.
pub fn configure(verbose: u8, quiet: bool) {
    let _ = log::set_logger(&LOGGER);
    log::set_max_level(level(verbose, quiet));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levels() {
        assert_eq!(level(0, false), LevelFilter::Warn);
        assert_eq!(level(1, false), LevelFilter::Info);
        assert_eq!(level(2, false), LevelFilter::Debug);
        assert_eq!(level(5, false), LevelFilter::Debug);
        assert_eq!(level(2, true), LevelFilter::Error);
    }
}
