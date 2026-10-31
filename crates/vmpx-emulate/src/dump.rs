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

use thiserror::Error;
use vmpx_pe::{PeError, PeImage};
use vmpx_rebuild::{LayoutError, LoaderValidationError, relayout_conventional, validate_for_loader};

use crate::cpu::Cpu;
use crate::emu::{FlowEvent, Hook, HookAction, MemEvent};
use crate::memory::{Access, MemFault, Memory, PAGE_SIZE, Perm};
use crate::observe::ObservedFacts;

const COFF_HEADER_SIZE: u64 = 20;
const SECTION_HEADER_SIZE: u64 = 40;
const IMAGE_SCN_CNT_INITIALIZED_DATA: u32 = 0x0000_0040;
const IMAGE_SCN_CNT_UNINITIALIZED_DATA: u32 = 0x0000_0080;
const IMAGE_SCN_MEM_EXECUTE: u32 = 0x2000_0000;
const IMAGE_SCN_MEM_READ: u32 = 0x4000_0000;
const IMAGE_SCN_MEM_WRITE: u32 = 0x8000_0000;

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
        // Write-then-execute OEP: a transfer INTO a page written this run. The
        // transfer must leave its own page — an unpacker hands control to a
        // freshly-unpacked destination region, never a same-page branch. This
        // excludes a stub that self-patches a byte in its own code page (e.g.
        // UPX writes a `cld` into its TLS-dispatch stub) and then branches
        // within it, which would otherwise read as a spurious OEP inside the
        // packer stub rather than in the decompressed image.
        if self.facts.borrow().oep.is_none()
            && page_base(ev.from) != page_base(ev.to)
            && self.written.borrow().contains(&page_base(ev.to))
        {
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

/// Why a dumped-module PE could not be reconstructed.
#[derive(Debug, Error)]
pub enum RebuildModuleError {
    #[error("size_of_image {0:#X} is zero or too large to reconstruct")]
    BadSizeOfImage(u64),

    #[error("no dumped region covers image_base {image_base:#X}: the PE headers are absent")]
    MissingHeaders { image_base: u64 },

    #[error("the dumped headers at image_base did not parse as a PE: {0}")]
    HeadersUnparseable(#[source] PeError),

    #[error("conventional relayout of the reconstructed image failed: {0}")]
    Relayout(#[source] LayoutError),

    #[error("the reconstructed PE failed loader validation: {0}")]
    Validation(#[source] LoaderValidationError),
}

/// Observed in-memory permissions → PE section characteristics, replacing the
/// R/W/X bits. When `has_content`, the section also reclassifies as initialized
/// data: an unpacker's decompression target (e.g. UPX marks `UPX0` as
/// `CNT_UNINITIALIZED_DATA`) now holds real bytes, and the conventional
/// relayout would otherwise emit a `CNT_UNINITIALIZED_DATA`/zero-`SizeOfRawData`
/// section as zero pages, discarding the unpacked image.
fn perm_to_characteristics(base: u32, perm: Perm, has_content: bool) -> u32 {
    let mut chars = base & !(IMAGE_SCN_MEM_READ | IMAGE_SCN_MEM_WRITE | IMAGE_SCN_MEM_EXECUTE);
    if has_content {
        chars &= !IMAGE_SCN_CNT_UNINITIALIZED_DATA;
        chars |= IMAGE_SCN_CNT_INITIALIZED_DATA;
    }
    if perm.contains(Perm::R) {
        chars |= IMAGE_SCN_MEM_READ;
    }
    if perm.contains(Perm::W) {
        chars |= IMAGE_SCN_MEM_WRITE;
    }
    if perm.contains(Perm::X) {
        chars |= IMAGE_SCN_MEM_EXECUTE;
    }
    chars
}

fn write_u32_le(buf: &mut [u8], offset: u64, value: u32) {
    if let Ok(start) = usize::try_from(offset)
        && let Some(slot) = buf.get_mut(start..start + 4)
    {
        slot.copy_from_slice(&value.to_le_bytes());
    }
}

/// Reconstruct a conventional, loader-valid standalone PE from the MODULE
/// portion of a detonation memory snapshot — the "dumped DLL".
///
/// # Provenance
///
/// This is an OBSERVED artifact: the bytes reflect ONE run's memory state, not
/// a sound static result. Regions outside `[image_base, image_base +
/// size_of_image)` — the fake sysdll/TEB/stub/heap/stack pools — are dropped;
/// only the module is rebuilt.
///
/// # Reconstruction
///
/// The dumped PE headers at `image_base` are reused verbatim. Each module
/// region's bytes are laid into a flat buffer at `va - image_base`, every
/// section is relaid at its own RVA (`PointerToRawData == VirtualAddress`) with
/// its observed in-memory permissions, and the result is handed through
/// [`relayout_conventional`] and [`validate_for_loader`] so the output reparses
/// and is loader-valid. Nothing is fabricated: a dump that lacks the header
/// region, or headers that do not parse, fails with a clear error rather than
/// inventing structure.
///
/// # Errors
///
/// [`RebuildModuleError`] for a bad `size_of_image`, an absent header region,
/// unparseable headers, a failed conventional relayout, or an image that does
/// not pass loader validation.
pub fn rebuild_module_pe(
    dump: &DumpImage,
    size_of_image: u64,
) -> Result<Vec<u8>, RebuildModuleError> {
    let base = dump.image_base;
    let module_end = base
        .checked_add(size_of_image)
        .ok_or(RebuildModuleError::BadSizeOfImage(size_of_image))?;
    let len = usize::try_from(size_of_image)
        .ok()
        .filter(|n| *n != 0)
        .ok_or(RebuildModuleError::BadSizeOfImage(size_of_image))?;

    // The headers must be present: the region covering image_base.
    if dump.region_containing(base).is_none() {
        return Err(RebuildModuleError::MissingHeaders { image_base: base });
    }

    // Flat RVA-layout buffer: every MODULE region's bytes at `va - image_base`.
    // Pools below image_base or at/after module_end are skipped.
    let mut image = vec![0u8; len];
    for region in &dump.regions {
        if region.va < base || region.va >= module_end {
            continue;
        }
        let off = (region.va - base) as usize;
        let avail = len - off;
        let take = region.bytes.len().min(avail);
        image[off..off + take].copy_from_slice(&region.bytes[..take]);
    }

    // Read the section table from the dumped headers, then rewrite each section
    // into RVA layout (raw == VA) carrying its observed in-memory permissions.
    // (header_offset, size_of_raw_data, pointer_to_raw_data, characteristics)
    let edits: Vec<(u64, u32, u32, u32)> = {
        let pe = PeImage::parse(&image).map_err(RebuildModuleError::HeadersUnparseable)?;
        let section_table = pe.nt_headers_offset.0
            + 4
            + COFF_HEADER_SIZE
            + u64::from(pe.coff.size_of_optional_header);
        pe.sections
            .iter()
            .enumerate()
            .map(|(index, section)| {
                let va = section.virtual_address.0;
                let remaining = size_of_image.saturating_sub(u64::from(va));
                let raw_size =
                    u32::try_from(u64::from(section.virtual_size).min(remaining)).unwrap_or(u32::MAX);
                let chars = dump
                    .region_containing(base + u64::from(va))
                    .map_or(section.characteristics, |region| {
                        // Observed content means the section was written this
                        // run (unpacked into), so it must be emitted as real
                        // bytes rather than zero-fill.
                        let has_content = dump
                            .read(base + u64::from(va), raw_size as usize)
                            .is_some_and(|b| b.iter().any(|&x| x != 0));
                        perm_to_characteristics(section.characteristics, region.perm, has_content)
                    });
                let header_offset = section_table + index as u64 * SECTION_HEADER_SIZE;
                (header_offset, raw_size, va, chars)
            })
            .collect()
    };
    for (header_offset, raw_size, raw_ptr, chars) in edits {
        write_u32_le(&mut image, header_offset + 16, raw_size);
        write_u32_le(&mut image, header_offset + 20, raw_ptr);
        write_u32_le(&mut image, header_offset + 36, chars);
    }

    let (bytes, _) = relayout_conventional(&image).map_err(RebuildModuleError::Relayout)?;
    validate_for_loader(&bytes).map_err(RebuildModuleError::Validation)?;
    Ok(bytes)
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

#[cfg(test)]
mod rebuild_tests {
    use super::*;
    use vmpx_pe::PeImage;

    const BASE: u64 = 0x1_8000_0000;
    const SIZE_OF_IMAGE: u64 = 0x4000;

    fn put(d: &mut [u8], at: usize, bytes: &[u8]) {
        d[at..at + bytes.len()].copy_from_slice(bytes);
    }

    /// One header page describing a PE32+ DLL with `.text` (RX, holds the entry)
    /// and `.data` (RW). Raw pointers are the original small file offsets; the
    /// rebuild rewrites them to RVA layout.
    fn header_page() -> Vec<u8> {
        let mut d = vec![0u8; 0x1000];
        put(&mut d, 0, &0x5A4Du16.to_le_bytes()); // MZ
        put(&mut d, 0x3C, &0x80u32.to_le_bytes()); // e_lfanew
        put(&mut d, 0x80, &0x0000_4550u32.to_le_bytes()); // PE\0\0
        put(&mut d, 0x84, &0x8664u16.to_le_bytes()); // machine x64
        put(&mut d, 0x86, &2u16.to_le_bytes()); // 2 sections
        put(&mut d, 0x94, &0xF0u16.to_le_bytes()); // SizeOfOptionalHeader
        put(&mut d, 0x96, &0x2022u16.to_le_bytes()); // characteristics (DLL)

        let o = 0x98usize; // optional header
        put(&mut d, o, &0x20Bu16.to_le_bytes()); // PE32+ magic
        put(&mut d, o + 16, &0x1000u32.to_le_bytes()); // AddressOfEntryPoint
        put(&mut d, o + 24, &BASE.to_le_bytes()); // ImageBase
        put(&mut d, o + 32, &0x1000u32.to_le_bytes()); // SectionAlignment
        put(&mut d, o + 36, &0x200u32.to_le_bytes()); // FileAlignment
        put(&mut d, o + 56, &(SIZE_OF_IMAGE as u32).to_le_bytes()); // SizeOfImage
        put(&mut d, o + 60, &0x200u32.to_le_bytes()); // SizeOfHeaders
        put(&mut d, o + 68, &3u16.to_le_bytes()); // Subsystem
        put(&mut d, o + 108, &16u32.to_le_bytes()); // NumberOfRvaAndSizes

        // Section table at 0x188 = 0x98 + 0xF0. Original (file) raw geometry.
        for (i, (name, rva, raw, chars)) in [
            (&b".text"[..], 0x1000u32, 0x200u32, 0x6000_0020u32),
            (b".data", 0x2000, 0x400, 0xC000_0040),
        ]
        .into_iter()
        .enumerate()
        {
            let at = 0x188 + i * 40;
            put(&mut d, at, name);
            put(&mut d, at + 8, &0x200u32.to_le_bytes()); // VirtualSize
            put(&mut d, at + 12, &rva.to_le_bytes()); // RVA
            put(&mut d, at + 16, &0x200u32.to_le_bytes()); // SizeOfRawData
            put(&mut d, at + 20, &raw.to_le_bytes()); // PointerToRawData
            put(&mut d, at + 36, &chars.to_le_bytes()); // Characteristics
        }
        d
    }

    /// A module snapshot (headers + `.text` + `.data`) plus two unrelated pool
    /// regions (a fake sysdll pool below the base, a heap pool at/after the
    /// module end) that the rebuild must filter out.
    fn module_dump() -> DumpImage {
        let mut text = vec![0u8; 0x1000];
        text[0] = 0xC3; // ret at the entry (RVA 0x1000)
        DumpImage {
            image_base: BASE,
            regions: vec![
                DumpRegion { va: 0x1000_0000, perm: Perm::RW, bytes: vec![0xEEu8; 0x1000] },
                DumpRegion { va: BASE, perm: Perm::R, bytes: header_page() },
                DumpRegion { va: BASE + 0x1000, perm: Perm::RX, bytes: text },
                DumpRegion { va: BASE + 0x2000, perm: Perm::RW, bytes: vec![0u8; 0x1000] },
                DumpRegion { va: BASE + SIZE_OF_IMAGE, perm: Perm::RW, bytes: vec![0xAAu8; 0x1000] },
            ],
        }
    }

    #[test]
    fn rebuilds_a_loadable_pe_filtered_to_the_module() {
        let dump = module_dump();
        let bytes = rebuild_module_pe(&dump, SIZE_OF_IMAGE).expect("rebuild succeeds");

        let pe = PeImage::parse(&bytes).expect("rebuilt PE reparses");
        let validation = validate_for_loader(&bytes).expect("loader-valid");
        assert!(validation.entry_executable);
        assert_eq!(pe.sections.len(), 2);
        assert_eq!(pe.optional.size_of_image, SIZE_OF_IMAGE as u32);
        assert_eq!(pe.optional.address_of_entry_point.0, 0x1000);
        // Observed in-memory perms land on the sections.
        let text = &pe.sections[0];
        assert!(text.characteristics & IMAGE_SCN_MEM_EXECUTE != 0, ".text stays executable");
        let data = &pe.sections[1];
        assert!(data.characteristics & IMAGE_SCN_MEM_EXECUTE == 0, ".data is not executable");
        assert!(data.characteristics & IMAGE_SCN_MEM_WRITE != 0, ".data stays writable");
        // The 16KiB module filtered out the two pool regions: a conventional
        // file is far smaller than base..pool span.
        assert!(bytes.len() < 0x1_0000, "no pool bytes leaked into the file");
    }

    #[test]
    fn missing_header_region_is_a_clean_error() {
        let mut dump = module_dump();
        dump.regions.retain(|r| r.va != BASE); // drop the headers
        let err = rebuild_module_pe(&dump, SIZE_OF_IMAGE).expect_err("no headers => error");
        assert!(matches!(err, RebuildModuleError::MissingHeaders { image_base } if image_base == BASE));
    }
}
