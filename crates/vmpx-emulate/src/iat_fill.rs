//! IAT-fill observer: recover `.idata slot -> module!name` by watching a
//! protector's loader fill its import slots during the contained init run.
//!
//! # Soundness rule
//!
//! A slot resolves ONLY if its final value (read from guest memory after the
//! run) is exactly an address our own offline model handed out:
//!
//! * a loader import stub (the sample's declared imports),
//! * a fake export address from a populated module export table (names come from
//!   the offline export reference, never invented), or
//! * a `GetProcAddress` result for a known module.
//!
//! Anything else (zero, an encrypted blob, an interior pointer, a value that is
//! not an exact known address) stays unresolved with the reason recorded. No
//! guessing, no near-matches.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use crate::cpu::Cpu;
use crate::emu::{Hook, HookAction, Interest, MemEvent};
use crate::memory::{Access, Memory};

const SLOT: u64 = 8;
/// Largest region the observer will snapshot (guards a mis-derived section).
pub(crate) const MAX_REGION: u64 = 16 << 20;

/// Where a known import address came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImportSource {
    /// A loader import-directory stub.
    LoaderStub,
    /// A fake export address from a module's populated export table.
    ExportTable,
    /// A `GetProcAddress` result.
    GetProcAddress,
}

/// A guest address our own model produced for `module!name`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KnownImport {
    pub module: String,
    pub name: String,
    pub source: ImportSource,
}

/// Address -> import, filled by [`crate::WinApiHook`] as it hands addresses out.
pub type AddressBook = Rc<RefCell<BTreeMap<u64, KnownImport>>>;

/// Write counts per slot VA, shared with the post-run resolver.
pub(crate) type WriteLog = Rc<RefCell<BTreeMap<u64, u32>>>;

/// Records every write landing in the watched slot region.
pub(crate) struct IatFillHook {
    lo: u64,
    hi: u64,
    log: WriteLog,
}

impl IatFillHook {
    pub(crate) fn new(lo: u64, hi: u64) -> (Self, WriteLog) {
        let log = WriteLog::default();
        (Self { lo, hi, log: Rc::clone(&log) }, log)
    }
}

impl Hook for IatFillHook {
    fn interest(&self) -> Interest {
        Interest { write: [(self.lo, self.hi), (0, 0)], ..Interest::NONE }
    }

    fn on_mem(&mut self, ev: &MemEvent, _cpu: &mut Cpu, _mem: &mut Memory) -> HookAction {
        if ev.access != Access::Write {
            return HookAction::Continue;
        }
        let end = ev.addr.saturating_add(u64::from(ev.size.max(1)));
        let mut slot = ev.addr & !(SLOT - 1);
        let mut log = self.log.borrow_mut();
        while slot < end {
            if slot >= self.lo && slot + SLOT <= self.hi {
                *log.entry(slot).or_insert(0) += 1;
            }
            slot += SLOT;
        }
        HookAction::Continue
    }
}

/// Why a filled slot was left unresolved.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnresolvedReason {
    /// Final value is zero.
    Zero,
    /// Final value is not exactly a known export/stub address.
    NotKnownExport,
}

/// A slot whose final value is exactly a known import address.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IatSlotImport {
    pub slot_rva: u64,
    pub value: u64,
    pub module: String,
    pub name: String,
    pub source: ImportSource,
    /// Guest writes observed to this slot.
    pub writes: u32,
    /// Final value differs from the pre-run value. `false` means the guest only
    /// re-wrote what the loader had already placed (the mapping then agrees with
    /// the import directory rather than being newly learned).
    pub changed: bool,
}

/// A filled slot that did not resolve.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IatSlotUnresolved {
    pub slot_rva: u64,
    pub value: u64,
    pub writes: u32,
    pub reason: UnresolvedReason,
}

/// The slot -> import mapping recovered from one run.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct IatFill {
    /// Watched `(rva, len)`, or `None` when no region was derived.
    pub region: Option<(u64, u64)>,
    pub resolved: Vec<IatSlotImport>,
    pub unresolved: Vec<IatSlotUnresolved>,
    /// 8-byte slots in the region never written and unchanged.
    pub untouched: usize,
}

/// Read every 8-byte slot of `[lo, hi)` (unreadable slots read as `None`).
pub(crate) fn snapshot(mem: &Memory, lo: u64, hi: u64) -> Vec<Option<u64>> {
    (lo..hi).step_by(SLOT as usize).map(|va| mem.read_uint(va, 8).ok()).collect()
}

/// Resolve the watched region after the run. A slot counts as filled when the
/// guest wrote it or its value changed from `initial`.
pub(crate) fn resolve(
    mem: &Memory,
    image_base: u64,
    (lo, hi): (u64, u64),
    initial: &[Option<u64>],
    log: &WriteLog,
    book: &AddressBook,
) -> IatFill {
    let log = log.borrow();
    let book = book.borrow();
    let mut fill = IatFill { region: Some((lo - image_base, hi - lo)), ..IatFill::default() };
    for (i, va) in (lo..hi).step_by(SLOT as usize).enumerate() {
        let writes = log.get(&va).copied().unwrap_or(0);
        let Some(value) = mem.read_uint(va, 8).ok() else { continue };
        if writes == 0 && initial.get(i).copied().flatten() == Some(value) {
            fill.untouched += 1;
            continue;
        }
        let slot_rva = va - image_base;
        match book.get(&value) {
            Some(k) => fill.resolved.push(IatSlotImport {
                slot_rva,
                value,
                module: k.module.clone(),
                name: k.name.clone(),
                source: k.source,
                writes,
                changed: initial.get(i).copied().flatten() != Some(value),
            }),
            None => fill.unresolved.push(IatSlotUnresolved {
                slot_rva,
                value,
                writes,
                reason: if value == 0 {
                    UnresolvedReason::Zero
                } else {
                    UnresolvedReason::NotKnownExport
                },
            }),
        }
    }
    fill
}
