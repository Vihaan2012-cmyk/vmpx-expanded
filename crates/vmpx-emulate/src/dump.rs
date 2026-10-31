//! Layer 4: detect unpack completion, snapshot the unpacked image and bridge
//! concrete runtime globals back to the static side.
//!
//! # What this provides
//!
//! * [`OepHook`] — a [`Hook`] that watches for the classic unpacker tail
//!   transfer (write-then-execute): it records every page written during the
//!   run and flags the first control transfer whose target lands in one of
//!   those pages as the OEP, recording it via [`ObservedFacts::set_oep`]. It
//!   can wrap an inner hook (e.g. [`WinApiHook`]) so API emulation and OEP
//!   detection share one hook slot and one [`ObservedFacts`]. The explicit
//!   path is just [`Emulator::run_until`]`(oep, max)`.
//! * [`dump_image`] — reconstruct the mapped guest memory as a set of
//!   contiguous [`DumpRegion`]s (bytes + VA + permissions), suitable for
//!   handing to `vmpx-rebuild`/`vmpx-pe` and then the static devirtualizer.
//! * [`capture_globals`] — read caller-specified global slot VAs out of the
//!   post-init [`Memory`] into [`ObservedFacts::globals`]. This resolves the
//!   static "null memory read" stall: the slot is zero in the static image but
//!   holds a concrete value after load-time init.
//!
//! # Containment
//!
//! Everything here is pure in-model reading: iterating guest pages and copying
//! bytes. No host net/syscall/fs/process/FFI. Captured facts are stamped
//! [`Observed`](crate::Observed) by the collector methods, so the static side
//! always treats them as one observed path, never as sound output.

use std::cell::RefCell;
use std::collections::BTreeSet;
use std::rc::Rc;

use crate::cpu::Cpu;
use crate::emu::{FlowEvent, Hook, HookAction, MemEvent};
use crate::memory::{Access, MemFault, Memory, PAGE_SIZE, Perm};
use crate::observe::ObservedFacts;

fn page_base(addr: u64) -> u64 {
    addr & !(PAGE_SIZE - 1)
}

/// Detects unpack completion via write-then-execute and records the OEP.
///
/// Install it as the emulator's single hook; to keep API emulation as well,
/// build a [`WinApiHook`](crate::WinApiHook), take its [`facts`](crate::WinApiHook::facts)
/// handle and pass both to [`OepHook::wrapping`] so the two share one
/// [`ObservedFacts`].
pub struct OepHook {
    facts: Rc<RefCell<ObservedFacts>>,
    /// Page bases written at least once this run.
    written: Rc<RefCell<BTreeSet<u64>>>,
    inner: Option<Box<dyn Hook>>,
    /// Stop the run the moment the OEP is detected (default); otherwise keep
    /// running so later behaviour can still be observed.
    stop_on_oep: bool,
}

impl Default for OepHook {
    fn default() -> Self {
        Self::with_facts(Rc::new(RefCell::new(ObservedFacts::new())))
    }
}

impl OepHook {
    /// Standalone hook with a fresh [`ObservedFacts`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record OEP into an existing (possibly shared) [`ObservedFacts`].
    #[must_use]
    pub fn with_facts(facts: Rc<RefCell<ObservedFacts>>) -> Self {
        Self { facts, written: Rc::new(RefCell::new(BTreeSet::new())), inner: None, stop_on_oep: true }
    }

    /// Wrap an inner hook (e.g. [`WinApiHook`](crate::WinApiHook)) so both run
    /// from one hook slot and share `facts`. The inner hook sees every event
    /// after OEP bookkeeping, unless the run is stopped at the OEP.
    #[must_use]
    pub fn wrapping(inner: impl Hook + 'static, facts: Rc<RefCell<ObservedFacts>>) -> Self {
        Self { inner: Some(Box::new(inner)), ..Self::with_facts(facts) }
    }

    /// Keep running past a detected OEP instead of stopping at it.
    #[must_use]
    pub fn keep_running(mut self) -> Self {
        self.stop_on_oep = false;
        self
    }

    /// Shared handle to the observations (OEP + anything the inner hook records).
    #[must_use]
    pub fn facts(&self) -> Rc<RefCell<ObservedFacts>> {
        Rc::clone(&self.facts)
    }

    /// The OEP detected so far, if any.
    #[must_use]
    pub fn oep(&self) -> Option<u64> {
        self.facts.borrow().oep
    }

    fn mark_written(&self, addr: u64, size: u32) {
        let mut w = self.written.borrow_mut();
        let end = addr.saturating_add(u64::from(size.max(1)));
        let mut p = page_base(addr);
        while p < end {
            w.insert(p);
            match p.checked_add(PAGE_SIZE) {
                Some(n) => p = n,
                None => break,
            }
        }
    }
}

impl Hook for OepHook {
    fn on_mem(&mut self, ev: &MemEvent, cpu: &mut Cpu, mem: &mut Memory) -> HookAction {
        if ev.access == Access::Write {
            self.mark_written(ev.addr, ev.size);
        }
        match self.inner.as_mut() {
            Some(h) => h.on_mem(ev, cpu, mem),
            None => HookAction::Continue,
        }
    }

    fn on_flow(&mut self, ev: &FlowEvent, cpu: &mut Cpu, mem: &mut Memory) -> HookAction {
        if self.facts.borrow().oep.is_none() && self.written.borrow().contains(&page_base(ev.to)) {
            self.facts.borrow_mut().set_oep(ev.to);
            if self.stop_on_oep {
                return HookAction::Stop;
            }
        }
        match self.inner.as_mut() {
            Some(h) => h.on_flow(ev, cpu, mem),
            None => HookAction::Continue,
        }
    }
}

/// A contiguous run of mapped guest pages sharing one permission.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DumpRegion {
    pub va: u64,
    pub perm: Perm,
    pub bytes: Vec<u8>,
}

impl DumpRegion {
    /// End VA (exclusive).
    #[must_use]
    pub fn end(&self) -> u64 {
        self.va + self.bytes.len() as u64
    }

    fn contains(&self, va: u64) -> bool {
        va >= self.va && va < self.end()
    }
}

/// A snapshot of the mapped guest memory: the reconstructable unpacked image.
///
/// Observed data — the static side quarantines it. `regions` are sorted by VA
/// and never overlap; the consumer filters to `image_base` and rebuilds a PE.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DumpImage {
    pub image_base: u64,
    pub regions: Vec<DumpRegion>,
}

impl DumpImage {
    /// The region containing `va`, if mapped.
    #[must_use]
    pub fn region_containing(&self, va: u64) -> Option<&DumpRegion> {
        self.regions.iter().find(|r| r.contains(va))
    }

    /// Read `len` bytes at `va` out of the snapshot (must lie in one region).
    #[must_use]
    pub fn read(&self, va: u64, len: usize) -> Option<Vec<u8>> {
        let r = self.region_containing(va)?;
        let off = (va - r.va) as usize;
        r.bytes.get(off..off + len).map(<[u8]>::to_vec)
    }

    /// Total mapped bytes across all regions.
    #[must_use]
    pub fn total_bytes(&self) -> usize {
        self.regions.iter().map(|r| r.bytes.len()).sum()
    }
}

/// Snapshot every mapped page of `mem`, coalescing adjacent same-permission
/// pages into [`DumpRegion`]s. `image_base` is recorded so the consumer can
/// pick out the unpacked module.
#[must_use]
pub fn dump_image(mem: &Memory, image_base: u64) -> DumpImage {
    let mut regions: Vec<DumpRegion> = Vec::new();
    for (base, perm, data) in mem.pages() {
        match regions.last_mut() {
            Some(last) if last.perm == perm && last.end() == base => {
                last.bytes.extend_from_slice(data);
            }
            _ => regions.push(DumpRegion { va: base, perm, bytes: data.to_vec() }),
        }
    }
    DumpImage { image_base, regions }
}

/// Read each `(va, len)` global slot out of the post-init `mem` and record its
/// concrete bytes into `facts.globals` (stamped [`Observed`](crate::Observed)).
///
/// This is the runtime-global bridge: a slot that is zero in the static image
/// but non-zero after load-time init is captured here so the static side can
/// resolve the stalled "null memory read".
///
/// # Errors
/// [`MemFault`] if any slot is unmapped or unreadable; slots before the
/// faulting one are still recorded.
pub fn capture_globals(
    mem: &Memory,
    facts: &mut ObservedFacts,
    slots: &[(u64, usize)],
) -> Result<(), MemFault> {
    for &(va, len) in slots {
        let mut buf = vec![0u8; len];
        mem.read(va, &mut buf)?;
        facts.record_global(va, buf);
    }
    Ok(())
}
