//! Structured, colourised progress logging for the one-command pipeline.
//!
//! One line per event: a dim tag, a marker glyph, a kind label, the message,
//! and a right-aligned dim detail (a timing or a count). Colour is emitted only
//! to a terminal; piped or redirected output is plain. No dependencies.

use std::io::{IsTerminal, Write};
use std::time::Instant;

const DETAIL_COLUMN: usize = 72;

#[derive(Clone, Copy)]
enum Kind {
    Start,
    Step,
    Layer,
    Info,
    Success,
    Warn,
    Error,
    File,
}

impl Kind {
    fn glyph(self) -> &'static str {
        match self {
            Self::Start => "▶",
            Self::Success => "✓",
            Self::Info => "i",
            Self::Warn => "!",
            Self::Error => "✗",
            Self::Step | Self::Layer | Self::File => "·",
        }
    }
    fn label(self) -> &'static str {
        match self {
            Self::Start => "start",
            Self::Step => "step",
            Self::Layer => "layer",
            Self::Info => "info",
            Self::Success => "success",
            Self::Warn => "warn",
            Self::Error => "error",
            Self::File => "file",
        }
    }
    /// SGR colour for the glyph and label.
    fn colour(self) -> &'static str {
        match self {
            Self::Start => "1;37",     // bold white
            Self::Step => "36",        // cyan
            Self::Layer => "35",       // magenta
            Self::Info => "34",        // blue
            Self::Success => "32",     // green
            Self::Warn => "33",        // yellow
            Self::Error => "1;31",     // bold red
            Self::File => "90",        // grey
        }
    }
}

pub struct Log {
    tag: String,
    colour: bool,
    /// When false, the per-step detail lines (`step`/`layer`/`file`) are hidden
    /// and only the stage headers and outcomes (`start`/`info`/`success`/
    /// `warn`/`error`) print. `--log` turns it on for the full trace.
    verbose: bool,
    /// Start of the whole run and of the current stage, for elapsed timing.
    run_at: Instant,
    stage_at: std::cell::Cell<Instant>,
    started: std::cell::Cell<bool>,
}

impl Log {
    pub fn new(tag: impl Into<String>, verbose: bool) -> Self {
        let now = Instant::now();
        Self {
            tag: tag.into(),
            colour: std::io::stdout().is_terminal(),
            verbose,
            run_at: now,
            stage_at: std::cell::Cell::new(now),
            started: std::cell::Cell::new(false),
        }
    }

    /// Elapsed since the previous `stage` call (or run start), reset for the next.
    fn stage_elapsed(&self) -> String {
        let at = self.stage_at.replace(Instant::now());
        format_elapsed(at.elapsed())
    }

    /// Total elapsed since the run began.
    pub fn total_elapsed(&self) -> String {
        format_elapsed(self.run_at.elapsed())
    }

    fn paint(&self, code: &str, text: &str) -> String {
        if self.colour {
            format!("\x1b[{code}m{text}\x1b[0m")
        } else {
            text.to_owned()
        }
    }

    fn emit(&self, kind: Kind, message: &str, detail: Option<&str>) {
        let tag = self.paint("90", &format!("[{}]", self.tag));
        let arrow = self.paint("90", "»");
        let glyph = self.paint(kind.colour(), kind.glyph());
        let label = self.paint(kind.colour(), &format!("{:<7}", kind.label()));
        let left = format!("{tag} {arrow} {glyph} {label} {message}");
        let line = match detail {
            Some(detail) => {
                // Pad so the detail sits at a fixed column; measure the VISIBLE
                // length (strip SGR) so colour does not skew the alignment.
                let visible = visible_len(&left);
                let pad = DETAIL_COLUMN.saturating_sub(visible).max(2);
                format!("{left}{}{}", " ".repeat(pad), self.paint("90", detail))
            }
            None => left,
        };
        let mut out = std::io::stdout().lock();
        let _ = writeln!(out, "{line}");
    }

    /// Begin a stage. The right-aligned detail shows how long the PREVIOUS
    /// stage took, so the trace is timed in both concise and verbose modes.
    pub fn start(&self, message: &str) {
        let elapsed = self.stage_elapsed();
        if self.started.replace(true) {
            self.emit(Kind::Start, message, Some(&elapsed.replace("in ", "prev ")));
        } else {
            self.emit(Kind::Start, message, None);
        }
    }
    pub fn info(&self, message: &str) {
        self.emit(Kind::Info, message, None);
    }
    pub fn step(&self, message: &str, detail: &str) {
        if self.verbose {
            self.emit(Kind::Step, message, Some(detail));
        }
    }
    pub fn layer(&self, message: &str, detail: &str) {
        if self.verbose {
            self.emit(Kind::Layer, message, Some(detail));
        }
    }
    pub fn success(&self, message: &str) {
        self.emit(Kind::Success, message, None);
    }
    pub fn success_detail(&self, message: &str, detail: &str) {
        self.emit(Kind::Success, message, Some(detail));
    }
    pub fn warn(&self, message: &str) {
        self.emit(Kind::Warn, message, None);
    }
    pub fn error(&self, message: &str) {
        self.emit(Kind::Error, message, None);
    }
    pub fn file(&self, message: &str, detail: &str) {
        self.emit(Kind::File, message, Some(detail));
    }

    #[must_use]
    pub fn verbose(&self) -> bool {
        self.verbose
    }

    /// Run `body`, timing it, and log one `step` with the elapsed time as the
    /// right-aligned detail. The body's own `Ok` value is returned.
    pub fn timed<T, E>(
        &self,
        message: &str,
        body: impl FnOnce() -> Result<T, E>,
    ) -> Result<T, E> {
        let at = Instant::now();
        let result = body();
        if result.is_ok() {
            self.step(message, &format_elapsed(at.elapsed()));
        }
        result
    }
}

fn format_elapsed(duration: std::time::Duration) -> String {
    let ms = duration.as_millis();
    if ms < 1000 {
        format!("in {ms} ms")
    } else {
        format!("in {:.1} s", duration.as_secs_f64())
    }
}

/// Length of `text` ignoring SGR escape sequences, for alignment.
fn visible_len(text: &str) -> usize {
    let mut len = 0;
    let mut chars = text.chars();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            for e in chars.by_ref() {
                if e == 'm' {
                    break;
                }
            }
        } else {
            len += 1;
        }
    }
    len
}
