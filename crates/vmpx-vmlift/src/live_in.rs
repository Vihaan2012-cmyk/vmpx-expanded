//! Per-function live-in (calling-convention) inference.
//!
//! A routine's inputs are the architectural registers it reads before it writes
//! them. The representation is deliberately minimal: the routine's register
//! accesses in execution order, one `(register, Read | Write)` event each.

use std::collections::BTreeSet;

use serde::Serialize;
use vmpx_vm::{RecoveredEntryPathFamily, X64NativeFrameReport, X64NativeRegister};

use crate::collect_path_effects;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AccessKind {
    Read,
    Write,
}

/// One register access. A read-modify-write is a `Read` event followed by a
/// `Write` event.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize)]
pub struct RegisterAccess {
    pub register: X64NativeRegister,
    pub kind: AccessKind,
}

impl RegisterAccess {
    #[must_use]
    pub const fn read(register: X64NativeRegister) -> Self {
        Self {
            register,
            kind: AccessKind::Read,
        }
    }

    #[must_use]
    pub const fn write(register: X64NativeRegister) -> Self {
        Self {
            register,
            kind: AccessKind::Write,
        }
    }
}

/// Registers read before any write to them, in the order given. A register
/// whose first access is a write, or that is never accessed, is not live-in.
/// The result is sorted (register order) and therefore deterministic.
#[must_use]
pub fn infer_live_in(accesses: &[RegisterAccess]) -> BTreeSet<X64NativeRegister> {
    let mut written = BTreeSet::new();
    let mut live_in = BTreeSet::new();
    for access in accesses {
        match access.kind {
            AccessKind::Read => {
                if !written.contains(&access.register) {
                    live_in.insert(access.register);
                }
            }
            AccessKind::Write => {
                written.insert(access.register);
            }
        }
    }
    live_in
}

/// The inferred inputs of one virtualized entry.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct EntryCallingConvention {
    /// Architectural registers the entry's observable effects read as inputs.
    /// A LOWER BOUND: only store operands and call arguments are inspected, so
    /// it can under-report. Never use it to drop replacement inputs.
    pub live_in: BTreeSet<X64NativeRegister>,
    /// Register that held the VM stack cursor (native RSP capture), if proven.
    pub stack_cursor: Option<X64NativeRegister>,
}

/// Adapts a recovered entry path family to [`infer_live_in`].
///
/// Path effects are expressed over the `native_<reg>_initial` symbols, so every
/// such symbol an effect expression mentions is a read of that register's value
/// at entry. Writes need no events: they are already folded into the
/// expressions. The result is the union over all paths.
#[must_use]
pub fn calling_convention(
    family: &RecoveredEntryPathFamily,
    frame: &X64NativeFrameReport,
) -> EntryCallingConvention {
    let mut accesses = Vec::new();
    for path in &family.paths {
        let effects = collect_path_effects(family, path);
        let exprs = effects
            .stores
            .iter()
            .flat_map(|store| [&store.address, &store.value])
            .chain(effects.calls.iter().flat_map(|call| call.arguments.iter()));
        for expr in exprs {
            let names = expr.dependencies();
            accesses.extend(
                X64NativeRegister::ALL
                    .iter()
                    .filter(|register| names.contains(register.symbolic_name()))
                    .map(|register| RegisterAccess::read(*register)),
            );
        }
    }
    EntryCallingConvention {
        live_in: infer_live_in(&accesses),
        stack_cursor: frame.cursor_architectural_register,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use X64NativeRegister::{R8, Rax, Rbx, Rcx, Rdx};

    fn live(accesses: &[RegisterAccess]) -> Vec<X64NativeRegister> {
        infer_live_in(accesses).into_iter().collect()
    }

    #[test]
    fn empty_sequence_has_no_live_in() {
        assert!(live(&[]).is_empty());
    }

    #[test]
    fn read_before_write_is_live_in() {
        let events = [RegisterAccess::read(Rcx), RegisterAccess::write(Rcx)];
        assert_eq!(live(&events), vec![Rcx]);
    }

    #[test]
    fn write_then_read_is_not_live_in() {
        let events = [RegisterAccess::write(Rax), RegisterAccess::read(Rax)];
        assert!(live(&events).is_empty());
    }

    #[test]
    fn read_only_is_live_in() {
        assert_eq!(live(&[RegisterAccess::read(Rdx)]), vec![Rdx]);
    }

    #[test]
    fn write_only_and_untouched_are_not_live_in() {
        let events = [RegisterAccess::write(Rax), RegisterAccess::read(Rbx)];
        assert_eq!(live(&events), vec![Rbx]);
    }

    #[test]
    fn repeated_reads_collapse_to_one_entry() {
        let events = [RegisterAccess::read(Rcx), RegisterAccess::read(Rcx)];
        assert_eq!(live(&events), vec![Rcx]);
    }

    #[test]
    fn a_later_write_does_not_hide_an_earlier_read() {
        let events = [
            RegisterAccess::read(Rax),
            RegisterAccess::write(Rax),
            RegisterAccess::read(Rax),
        ];
        assert_eq!(live(&events), vec![Rax]);
    }

    #[test]
    fn an_earlier_write_shields_all_later_reads() {
        let events = [
            RegisterAccess::write(Rax),
            RegisterAccess::read(Rax),
            RegisterAccess::write(Rax),
            RegisterAccess::read(Rax),
        ];
        assert!(live(&events).is_empty());
    }

    #[test]
    fn registers_are_tracked_independently() {
        let events = [
            RegisterAccess::write(Rax),
            RegisterAccess::read(Rcx),
            RegisterAccess::read(Rax),
            RegisterAccess::read(R8),
            RegisterAccess::write(Rdx),
            RegisterAccess::read(Rdx),
        ];
        assert_eq!(live(&events), vec![Rcx, R8]);
    }

    #[test]
    fn order_matters() {
        let read_first = [RegisterAccess::read(Rax), RegisterAccess::write(Rax)];
        let write_first = [RegisterAccess::write(Rax), RegisterAccess::read(Rax)];
        assert_ne!(infer_live_in(&read_first), infer_live_in(&write_first));
    }

    #[test]
    fn result_is_sorted_regardless_of_event_order() {
        let forward = [
            RegisterAccess::read(R8),
            RegisterAccess::read(Rax),
            RegisterAccess::read(Rdx),
        ];
        let reverse = [forward[2], forward[1], forward[0]];
        assert_eq!(live(&forward), vec![Rax, Rdx, R8]);
        assert_eq!(live(&forward), live(&reverse));
        assert_eq!(live(&forward), live(&forward));
    }
}
