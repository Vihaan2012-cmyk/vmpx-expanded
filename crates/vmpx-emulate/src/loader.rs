//! Layer 1: map a parsed `vmpx-pe` image into guest [`Memory`] and build a
//! minimal x64 user-mode environment (stack, TEB/PEB stubs, import-stub IAT)
//! so a single export can be invoked under the [`Emulator`].
//!
//! Containment is unchanged: nothing here reaches host memory. Imports are
//! given sentinel addresses only (Layer 2 emulates behaviour); a call into the
//! read-only stub region stops the run with an exec fault whose address the
//! caller resolves via [`LoadedImage::resolve_stub`].

use std::collections::BTreeMap;

use vmpx_pe::PeImage;

use crate::cpu::{Cpu, RSP};
use crate::emu::Emulator;
use crate::memory::{MapError, MemFault, Memory, PAGE_SIZE, Perm};

// Section characteristics bits controlling mapped permissions.
const SCN_MEM_EXECUTE: u32 = 0x2000_0000;
const SCN_MEM_READ: u32 = 0x4000_0000;
const SCN_MEM_WRITE: u32 = 0x8000_0000;

// Win64 integer argument registers, in order (RCX, RDX, R8, R9).
const ARG_REGS: [usize; 4] = [1, 2, 8, 9];

// Fixed guest layout, placed far from any plausible image base so the sparse
// map never collides with the mapped image.
const STACK_REGION: u64 = 0x0000_0010_0000_0000;
const STUB_REGION: u64 = 0x0000_0020_0000_0000;
const TEB_REGION: u64 = 0x0000_0030_0000_0000;
const DEFAULT_STACK_SIZE: u64 = 256 * 1024;
// Unmapped address used as the synthetic return target; `run_until` detects it
// before executing there, so it never needs to be mapped.
const RETURN_SENTINEL: u64 = 0x0000_0040_DEAD_0000;

// TEB/PEB field offsets (x64).
const TEB_PEB_PTR: u64 = 0x60;
const PEB_BEING_DEBUGGED: u64 = 0x02;
const PEB_IMAGE_BASE: u64 = 0x10;
const PEB_LDR: u64 = 0x18;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoadError {
    Map(MapError),
    Fault(MemFault),
    Layout(&'static str),
}

impl From<MapError> for LoadError {
    fn from(e: MapError) -> Self {
        LoadError::Map(e)
    }
}
impl From<MemFault> for LoadError {
    fn from(e: MemFault) -> Self {
        LoadError::Fault(e)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LoadOptions {
    pub stack_size: u64,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self { stack_size: DEFAULT_STACK_SIZE }
    }
}

/// A mapped image plus its execution environment. Holds [`Memory`] until an
/// emulator is built from it ([`LoadedImage::prepare`]).
pub struct LoadedImage {
    pub memory: Memory,
    pub image_base: u64,
    /// Entry-point VA (`None` when the image declares no entry point).
    pub entry: Option<u64>,
    /// Export name -> VA (forwarders excluded).
    pub exports: BTreeMap<String, u64>,
    /// Stub VA -> (dll, function); the sentinel written into each IAT slot.
    pub import_stubs: BTreeMap<u64, (String, String)>,
    pub gs_base: u64,
    pub stack_base: u64,
    pub stack_top: u64,
    /// 16-byte-aligned RSP before the synthetic return address is pushed.
    pub initial_rsp: u64,
}

fn round_up(v: u64) -> u64 {
    v.div_ceil(PAGE_SIZE) * PAGE_SIZE
}

fn section_perm(ch: u32) -> Perm {
    let mut p = 0u8;
    if ch & SCN_MEM_READ != 0 {
        p |= Perm::R.0;
    }
    if ch & SCN_MEM_WRITE != 0 {
        p |= Perm::W.0;
    }
    if ch & SCN_MEM_EXECUTE != 0 {
        p |= Perm::X.0;
    }
    if p == 0 {
        p = Perm::R.0;
    }
    Perm(p)
}

/// Map `pe` at its preferred base and build the minimal environment.
///
/// Relocations: the image is always mapped at its preferred `image_base`, so
/// the relocation delta is zero and nothing is patched. (A non-preferred base
/// would require applying the base-relocation table; not needed here.)
///
/// # Errors
/// On unaligned/overlapping maps, out-of-range raw data, or structural limits.
pub fn load(pe: &PeImage, opts: &LoadOptions) -> Result<LoadedImage, LoadError> {
    let report = pe.report();
    let image_base = report.optional.image_base.0;
    let file = pe.data();
    let mut mem = Memory::new();

    // Headers, clamped so they never overlap the first section.
    let first_rva = report
        .sections
        .iter()
        .map(|s| u64::from(s.virtual_address.0))
        .min()
        .unwrap_or(u64::MAX);
    let header_len = round_up(u64::from(report.optional.size_of_headers)).min(first_rva);
    if header_len > 0 {
        mem.map(image_base, header_len, Perm::R)?;
        let n = (report.optional.size_of_headers as usize).min(file.len());
        mem.poke(image_base, &file[..n])?;
    }

    // Sections at image_base + RVA; raw bytes copied, virtual tail left zero.
    for s in &report.sections {
        let rva = u64::from(s.virtual_address.0);
        if rva & (PAGE_SIZE - 1) != 0 {
            return Err(LoadError::Layout("section RVA is not page aligned"));
        }
        let va = image_base
            .checked_add(rva)
            .ok_or(LoadError::Layout("section VA overflows"))?;
        let span = round_up(u64::from(s.mapped_span()));
        if span == 0 {
            continue;
        }
        mem.map(va, span, section_perm(s.characteristics))?;
        let raw_ptr = s.pointer_to_raw_data.0 as usize;
        let raw_len = (s.size_of_raw_data as usize).min(s.mapped_span() as usize);
        if raw_len > 0 {
            let end = raw_ptr
                .checked_add(raw_len)
                .ok_or(LoadError::Layout("section raw range overflows"))?;
            if end > file.len() {
                return Err(LoadError::Layout("section raw range exceeds file"));
            }
            mem.poke(va, &file[raw_ptr..end])?;
        }
    }

    // Exports (skip forwarders and null RVAs).
    let mut exports = BTreeMap::new();
    if let Some(dir) = report.exports.as_ref() {
        for sym in &dir.symbols {
            if sym.forwarder.is_some() || sym.rva.0 == 0 {
                continue;
            }
            if let Some(name) = &sym.name {
                exports.insert(name.clone(), image_base + u64::from(sym.rva.0));
            }
        }
    }
    let entry = (report.optional.address_of_entry_point.0 != 0)
        .then(|| image_base + u64::from(report.optional.address_of_entry_point.0));

    // Import stubs: one sentinel per imported symbol in a read-only region.
    let mut import_stubs = BTreeMap::new();
    let total_imports: usize = report.imports.iter().map(|m| m.entries.len()).sum();
    if total_imports > 0 {
        let region_len = round_up((total_imports as u64) * 8);
        mem.map(STUB_REGION, region_len, Perm::R)?;
        let mut slot = 0u64;
        for module in &report.imports {
            for sym in &module.entries {
                let stub_va = STUB_REGION + slot * 8;
                let func = sym
                    .name
                    .clone()
                    .unwrap_or_else(|| format!("#{}", sym.ordinal.unwrap_or(0)));
                let iat_va = image_base + u64::from(sym.iat_rva.0);
                mem.poke(iat_va, &stub_va.to_le_bytes())?;
                import_stubs.insert(stub_va, (module.name.clone(), func));
                slot += 1;
            }
        }
    }

    // TEB/PEB stubs (one page); gs: base = TEB.
    let teb = TEB_REGION;
    let peb = TEB_REGION + 0x400;
    let ldr = TEB_REGION + 0x800;
    mem.map(teb, PAGE_SIZE, Perm::RW)?;
    mem.poke(teb + TEB_PEB_PTR, &peb.to_le_bytes())?;
    mem.poke(peb + PEB_BEING_DEBUGGED, &[0u8])?;
    mem.poke(peb + PEB_IMAGE_BASE, &image_base.to_le_bytes())?;
    mem.poke(peb + PEB_LDR, &ldr.to_le_bytes())?;
    // PEB_LDR_DATA: three module lists, each an empty (self-linked) LIST_ENTRY.
    for list in [ldr + 0x10, ldr + 0x20, ldr + 0x30] {
        mem.poke(list, &list.to_le_bytes())?;
        mem.poke(list + 8, &list.to_le_bytes())?;
    }

    // Stack region; RSP near top, 16-aligned, with slack below for red zone.
    let stack_base = STACK_REGION;
    let stack_size = round_up(opts.stack_size.max(PAGE_SIZE));
    let stack_top = stack_base + stack_size;
    mem.map(stack_base, stack_size, Perm::RW)?;
    let initial_rsp = (stack_top - 0x200) & !0xF;

    Ok(LoadedImage {
        memory: mem,
        image_base,
        entry,
        exports,
        import_stubs,
        gs_base: teb,
        stack_base,
        stack_top,
        initial_rsp,
    })
}

impl LoadedImage {
    #[must_use]
    pub fn export(&self, name: &str) -> Option<u64> {
        self.exports.get(name).copied()
    }

    /// Resolve a stub VA (e.g. a fault address) to its (dll, function).
    #[must_use]
    pub fn resolve_stub(&self, addr: u64) -> Option<&(String, String)> {
        self.import_stubs.get(&addr)
    }

    /// The synthetic return target; pass to `run_until` to stop on return.
    #[must_use]
    pub fn return_sentinel(&self) -> u64 {
        RETURN_SENTINEL
    }

    /// Build an [`Emulator`] positioned at `rip` with Win64 integer `args`.
    ///
    /// A return sentinel is pushed so the function's final `ret` lands on
    /// [`LoadedImage::return_sentinel`]. Consumes the image's [`Memory`].
    ///
    /// # Errors
    /// If the stack write for the return sentinel faults.
    pub fn prepare(&mut self, rip: u64, args: &[u64]) -> Result<Emulator, LoadError> {
        let mut mem = std::mem::take(&mut self.memory);
        let sp = self.initial_rsp - 8;
        mem.poke(sp, &RETURN_SENTINEL.to_le_bytes())?;

        let mut cpu = Cpu::new(rip);
        cpu.gs_base = self.gs_base;
        cpu.regs[RSP] = sp;
        for (i, &v) in args.iter().take(ARG_REGS.len()).enumerate() {
            cpu.regs[ARG_REGS[i]] = v;
        }
        Ok(Emulator::new(cpu, mem))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perms_track_characteristics() {
        // W-only section is writable but not executable or readable.
        let w = section_perm(SCN_MEM_WRITE);
        assert!(w.contains(Perm::W));
        assert!(!w.contains(Perm::X));
        assert!(!w.contains(Perm::R));
        // Code section: read + execute, not write.
        let rx = section_perm(SCN_MEM_READ | SCN_MEM_EXECUTE);
        assert!(rx.contains(Perm::RX));
        assert!(!rx.contains(Perm::W));
        // No access bits default to readable so the page is still usable.
        assert_eq!(section_perm(0), Perm::R);
    }
}
