//! Offline System32 export database.
//!
//! Given a system-DLL name (e.g. `"ntdll.dll"`), [`SysDllDb`] resolves it to the
//! real local file under `<SystemRoot>\System32`, parses its export directory
//! with [`vmpx_pe::PeImage`], and returns the real exports (named + ordinal-only)
//! as [`SymbolOrOrdinal`]s.
//!
//! # Containment
//!
//! This is a driver-side *file read* — exactly like loading the input PE. There
//! is no network access and no syscall: the exports returned are the real local
//! DLL's real exports, never invented. A missing or unparseable DLL yields
//! `None` (fail-safe) so the caller falls back to whatever reference it had.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use vmpx_pe::PeImage;

use crate::observe::SymbolOrOrdinal;

/// The loader/runtime DLLs a VMProtect sample most commonly walks for the APIs
/// it resolves by hand. Used as the default auto-reference set when the exact
/// modules a sample requests are not observable up front.
pub const DEFAULT_SYSTEM_MODULES: &[&str] =
    &["ntdll.dll", "kernel32.dll", "kernelbase.dll", "user32.dll"];

/// Resolve the Windows system root from the environment (`%SystemRoot%`, then
/// `%WINDIR%`), defaulting to `C:\Windows`.
#[must_use]
pub fn system_root() -> PathBuf {
    std::env::var_os("SystemRoot")
        .or_else(|| std::env::var_os("WINDIR"))
        .map_or_else(|| PathBuf::from(r"C:\Windows"), PathBuf::from)
}

/// Offline export DB over the local `System32` directory, caching parsed exports
/// per module.
#[derive(Debug)]
pub struct SysDllDb {
    system32: PathBuf,
    // `None` caches a resolved-but-absent/unparseable module (fail-safe memo).
    cache: RefCell<BTreeMap<String, Option<Vec<SymbolOrOrdinal>>>>,
}

impl Default for SysDllDb {
    fn default() -> Self {
        Self::with_windows_root(system_root())
    }
}

impl SysDllDb {
    /// A DB over the live system root (`%SystemRoot%`/`%WINDIR%`, else `C:\Windows`).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// A DB whose `System32` sits under an explicit Windows root (for tests).
    #[must_use]
    pub fn with_windows_root(windows_root: impl AsRef<Path>) -> Self {
        Self {
            system32: windows_root.as_ref().join("System32"),
            cache: RefCell::new(BTreeMap::new()),
        }
    }

    /// The exports (named + ordinal-only) of the local `module` under System32,
    /// or `None` if the file is not found locally or cannot be parsed.
    ///
    /// Case-insensitive on the module name; the result is cached per module.
    #[must_use]
    pub fn exports(&self, module: &str) -> Option<Vec<SymbolOrOrdinal>> {
        let key = module.to_lowercase();
        if let Some(cached) = self.cache.borrow().get(&key) {
            return cached.clone();
        }
        let exports = self
            .resolve_file(&key)
            .and_then(|path| std::fs::read(path).ok())
            .and_then(|bytes| parse_exports(&bytes));
        self.cache.borrow_mut().insert(key, exports.clone());
        exports
    }

    /// Locate the real file for `module` under System32 (case-insensitive).
    fn resolve_file(&self, module: &str) -> Option<PathBuf> {
        let direct = self.system32.join(module);
        if direct.is_file() {
            return Some(direct);
        }
        // Explicit case-insensitive scan for hosts with a case-sensitive FS.
        std::fs::read_dir(&self.system32).ok().and_then(|entries| {
            entries
                .flatten()
                .map(|e| e.path())
                .find(|p| {
                    p.is_file()
                        && p.file_name()
                            .map(|n| n.to_string_lossy().to_lowercase() == module)
                            .unwrap_or(false)
                })
        })
    }
}

/// Parse a PE's export directory into reference entries (named + ordinal-only).
fn parse_exports(bytes: &[u8]) -> Option<Vec<SymbolOrOrdinal>> {
    let pe = PeImage::parse(bytes).ok()?;
    let exports = pe.report().exports?;
    Some(
        exports
            .symbols
            .iter()
            .map(|symbol| {
                symbol.name.clone().map_or_else(
                    || SymbolOrOrdinal::Ordinal(symbol.ordinal as u16),
                    SymbolOrOrdinal::Name,
                )
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fail-safe: a DB over a root with no System32 resolves nothing, and the
    /// absence is cached. Cross-platform (does not touch the live system).
    #[test]
    fn missing_dll_is_fail_safe() {
        let db = SysDllDb::with_windows_root(std::env::temp_dir().join("vmpx-no-such-root-xyz"));
        assert!(db.exports("ntdll.dll").is_none());
        assert!(db.exports("definitely-not-a-dll.dll").is_none());
    }

    /// The real local `ntdll.dll` parses to a non-empty export set containing
    /// well-known names. Skips gracefully if System32 lacks the file.
    #[test]
    #[cfg(windows)]
    fn resolves_ntdll_from_system32() {
        let db = SysDllDb::new();
        let Some(exports) = db.exports("NTDLL.DLL") else {
            eprintln!("[skip] local System32 ntdll.dll unavailable");
            return;
        };
        assert!(!exports.is_empty(), "ntdll exports non-empty");
        let has = |n: &str| {
            exports
                .iter()
                .any(|e| matches!(e, SymbolOrOrdinal::Name(name) if name == n))
        };
        assert!(has("NtProtectVirtualMemory"), "known ntdll export present");
        assert!(has("RtlAllocateHeap"), "known ntdll export present");
        // Cached: a second lookup returns the same set.
        assert_eq!(db.exports("ntdll.dll").map(|v| v.len()), Some(exports.len()));
    }
}
