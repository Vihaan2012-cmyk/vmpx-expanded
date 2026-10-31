//! WP-D6 detonation driver: load a sample, install the composite observation
//! hooks, run to OEP (or a bounded budget), and hand back the [`ObservedFacts`]
//! plus an optional image snapshot.
//!
//! # Containment
//!
//! [`detonate`] only calls the Layer-1..5 public API ([`load`]/[`crate::LoadedImage::prepare`]/
//! [`crate::Emulator`], [`WinApiHook`], [`OepHook`], [`VmObserveHook`]). The run
//! is pure in-process software interpretation — no host net/syscall/fs/process/
//! FFI path exists. Anything unmodeled stops the run; nothing is guessed.
//!
//! # Provenance
//!
//! The returned facts are [`crate::Observed`]-stamped by their collectors. They
//! reflect ONE concrete path and must stay on the experimental/quarantined tier
//! on the static side — never sound output.

use std::collections::BTreeMap;
use std::rc::Rc;

use vmpx_pe::PeImage;

use crate::cpu::Cpu;
use crate::dump::{DumpImage, OepHook, capture_globals, dump_image};
use crate::emu::{FlowEvent, Hook, HookAction, Interest, MemEvent, Stop};
use crate::iat_fill::{self, IatFill, IatFillHook};
use crate::loader::{LoadError, LoadOptions, load};
use crate::memory::Memory;
use crate::observe::{ObservedFacts, SymbolOrOrdinal};
use crate::vm_discover::{DiscoveredVm, SurveyHook};
use crate::vm_observe::{VmInstanceSpec, VmObservations, VmObserveHook};
use crate::winapi::WinApiHook;
use crate::xplm::{RecordedCallback, Xplm};
use std::cell::RefCell;

const DEFAULT_MAX_STEPS: u64 = 2_000_000;

// COFF characteristic: the image is a DLL.
const IMAGE_FILE_DLL: u16 = 0x2000;
// DllMain reason code for process attach.
const DLL_PROCESS_ATTACH: u64 = 1;

/// Which guest region to watch for the loader filling import slots.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum IatRegion {
    /// The `.idata` section, else the section holding the import directory.
    #[default]
    Auto,
    /// An explicit `(rva, len)`.
    Rva { rva: u64, len: u64 },
    /// No IAT-fill observation.
    Off,
}

/// Knobs for one detonation. `Default` runs the entry point to OEP with a dump.
#[derive(Clone, Debug)]
pub struct DetonateOptions {
    /// Instruction budget before the run stops with [`Stop::StepLimit`].
    pub max_steps: u64,
    /// Guest stack size (bytes).
    pub stack_size: u64,
    /// Win64 integer args passed to the entry point (RCX, RDX, R8, R9).
    pub args: Vec<u64>,
    /// `(va, len)` global slots to snapshot out of post-run memory.
    pub capture_globals: Vec<(u64, usize)>,
    /// VM dispatcher VA to watch for a handler trace (enables VM observation).
    pub vm_dispatcher: Option<u64>,
    /// Multi-instance VM observation: dispatchers keyed by RVA, resolved through
    /// the load base. Takes precedence over `vm_dispatcher` when non-empty.
    pub vm_instances: Vec<VmInstanceSpec>,
    /// Discover the VM dispatch stubs and VIP location structurally (a survey
    /// run, then the observed run). Ignored when `vm_instances` is non-empty.
    pub vm_discover: bool,
    /// Loop-head / recursion-entry VA whose back-edge visits are counted.
    pub vm_loop_head: Option<u64>,
    /// Route XPLM/XPWidgets imports to the recording host-SDK model (never
    /// forwarded anywhere); unmodeled XPLM calls stop the run.
    pub xplm: bool,
    /// OFFLINE export reference `{ module -> [name|ordinal] }`: fills each
    /// resolved module's fake export table so a protector resolving imports by
    /// manually walking the export table finds them (and the walk is captured).
    /// Empty = every fake module exports nothing (the prior behaviour).
    pub export_reference: BTreeMap<String, Vec<SymbolOrOrdinal>>,
    /// Snapshot the mapped image after the run.
    pub dump: bool,
    /// Resolve architecturally-undefined flags to their deterministic hardware
    /// value instead of fail-closing (see [`crate::Emulator::resolve_undefined_flags`]).
    /// Default `true`: a concrete CPU run has a definite flag value, so a
    /// `cmov`/`jcc` consuming e.g. a multi-bit-rotate's undefined OF proceeds
    /// rather than stopping. Observed/quarantined tier — never sound static
    /// output. Set `false` to reproduce the sound fail-closed stop.
    pub resolve_undefined_flags: bool,
    /// Region watched for import-slot fills (see [`DetonationResult::iat_fill`]).
    pub iat_region: IatRegion,
}

impl Default for DetonateOptions {
    fn default() -> Self {
        Self {
            max_steps: DEFAULT_MAX_STEPS,
            stack_size: LoadOptions::default().stack_size,
            args: Vec::new(),
            capture_globals: Vec::new(),
            vm_dispatcher: None,
            vm_loop_head: None,
            vm_instances: Vec::new(),
            vm_discover: false,
            xplm: false,
            export_reference: BTreeMap::new(),
            dump: true,
            resolve_undefined_flags: true,
            iat_region: IatRegion::Auto,
        }
    }
}

/// Build an export reference `{ module -> [name|ordinal] }` from a PE's import
/// directory, grouping each imported symbol under the module it comes from.
///
/// For the real direct-export-walk capture, this is derived from the UNPROTECTED
/// original (its imports are exactly the functions the protected twin resolves
/// by walking those modules' export tables). In production the same shape is a
/// bundled OFFLINE system-DLL export list — static data, never a network fetch.
#[must_use]
pub fn export_reference_from_imports(
    pe: &PeImage,
) -> BTreeMap<String, Vec<SymbolOrOrdinal>> {
    let mut reference: BTreeMap<String, Vec<SymbolOrOrdinal>> = BTreeMap::new();
    for module in &pe.report().imports {
        let key = module.name.to_lowercase();
        let entry = reference.entry(key).or_default();
        for symbol in &module.entries {
            if let Some(name) = &symbol.name {
                entry.push(SymbolOrOrdinal::Name(name.clone()));
            } else if let Some(ordinal) = symbol.ordinal {
                entry.push(SymbolOrOrdinal::Ordinal(ordinal));
            }
        }
    }
    reference
}

/// The named exports of a PE's export directory, as reference entries.
///
/// This is how a bundled OFFLINE system-DLL export list is turned into a
/// reference entry for the module the protector resolves and walks (e.g. the
/// `ntdll.dll` / `user32.dll` the protector's own runtime imports from). Reading
/// a static reference DLL is a file read, never a network fetch.
#[must_use]
pub fn export_reference_from_exports(pe: &PeImage) -> Vec<SymbolOrOrdinal> {
    pe.report().exports.map_or_else(Vec::new, |exports| {
        exports
            .symbols
            .iter()
            .filter_map(|symbol| symbol.name.clone().map(SymbolOrOrdinal::Name))
            .collect()
    })
}

/// Why a detonation could not even start.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DetonateError {
    /// The image failed to map.
    Load(LoadError),
    /// The image declares no entry point to run from.
    NoEntry,
}

impl From<LoadError> for DetonateError {
    fn from(e: LoadError) -> Self {
        DetonateError::Load(e)
    }
}

/// The result of one contained run.
#[derive(Clone, Debug)]
pub struct DetonationResult {
    /// Observed facts (imports, OEP, captured globals, loop depth) — all
    /// [`crate::Observed`]-stamped, experimental-tier only.
    pub observed: ObservedFacts,
    /// Image snapshot, when `opts.dump` was set.
    pub dump: Option<DumpImage>,
    /// Why the run ended. [`Stop::Hook`] with `observed.oep` set means the OEP
    /// was reached; any other stop is the next blocker to report.
    pub stop: Stop,
    /// Retired instruction count.
    pub steps: usize,
    /// Guest rip when the run ended.
    pub rip: u64,
    /// VM-dispatcher observations, when a dispatcher/loop-head was configured.
    pub vm: Option<VmObservations>,
    /// Recorded host-SDK activity, when `opts.xplm` was set.
    pub xplm: Option<XplmSummary>,
    /// VMs found by the structural survey, when `opts.vm_discover` was set.
    pub discovered: Vec<DiscoveredVm>,
    /// Import-slot fills observed in `opts.iat_region`, mapped to `module!name`
    /// only on an exact known-address match.
    pub iat_fill: IatFill,
}

/// What the recording host-SDK model captured during the run.
#[derive(Clone, Debug, Default)]
pub struct XplmSummary {
    /// Callbacks the plugin registered, in registration order.
    pub callbacks: Vec<RecordedCallback>,
    /// Modeled-call counts by function name.
    pub call_counts: BTreeMap<String, u64>,
    /// XPLM imports hit with no stub: `(dll, func, count)`.
    pub unlisted: Vec<(String, String, u64)>,
}

/// Absolute `[lo, hi)` VA range to watch, per `region`.
fn iat_range(pe: &PeImage, image_base: u64, region: IatRegion) -> Option<(u64, u64)> {
    let report = pe.report();
    let (rva, len) = match region {
        IatRegion::Off => return None,
        IatRegion::Rva { rva, len } => (rva, len),
        IatRegion::Auto => {
            let import_rva = report
                .directories
                .iter()
                .find(|d| d.index == 1 && d.size != 0)
                .map(|d| d.address);
            let sec = report
                .sections
                .iter()
                .find(|s| s.name.eq_ignore_ascii_case(".idata"))
                .or_else(|| {
                    import_rva.and_then(|r| {
                        report.sections.iter().find(|s| s.contains_mapped_rva(vmpx_pe::Rva(r)))
                    })
                })?;
            (u64::from(sec.virtual_address.0), u64::from(sec.mapped_span()))
        }
    };
    let len = len & !7;
    (len != 0 && len <= iat_fill::MAX_REGION).then(|| (image_base + rva, image_base + rva + len))
}

/// A hook that fans every event out to several inner hooks. `Stop` from any
/// short-circuits; `Redirect` from any is honoured (the redirecting hook has
/// already set `rip`); otherwise `Continue`.
struct MultiHook {
    hooks: Vec<Box<dyn Hook>>,
}

fn combine(acc: HookAction, next: HookAction) -> HookAction {
    match (acc, next) {
        (HookAction::Stop, _) | (_, HookAction::Stop) => HookAction::Stop,
        (HookAction::Redirect, _) | (_, HookAction::Redirect) => HookAction::Redirect,
        _ => HookAction::Continue,
    }
}

impl Hook for MultiHook {
    fn interest(&self) -> Interest {
        self.hooks.iter().fold(Interest::NONE, |acc, h| acc.union(h.interest()))
    }

    fn on_mem(&mut self, ev: &MemEvent, cpu: &mut Cpu, mem: &mut Memory) -> HookAction {
        let mut acc = HookAction::Continue;
        for h in &mut self.hooks {
            acc = combine(acc, h.on_mem(ev, cpu, mem));
            if acc == HookAction::Stop {
                return acc;
            }
        }
        acc
    }

    fn on_flow(&mut self, ev: &FlowEvent, cpu: &mut Cpu, mem: &mut Memory) -> HookAction {
        let mut acc = HookAction::Continue;
        for h in &mut self.hooks {
            acc = combine(acc, h.on_flow(ev, cpu, mem));
            if acc == HookAction::Stop {
                return acc;
            }
        }
        acc
    }
}

/// Load `pe`, install a composite hook (import emulation + OEP detection, and VM
/// observation when a dispatcher/loop-head is configured) over one shared
/// [`ObservedFacts`], run the entry point, and return what was observed.
///
/// The composite is [`OepHook::wrapping`] a [`WinApiHook`] (both share one
/// `ObservedFacts`); a [`VmObserveHook`] is folded in via a small `MultiHook`
/// only when `opts.vm_dispatcher`/`opts.vm_loop_head` is set. The run stops at
/// the OEP (first transfer into a written page) or at the first unmodeled
/// instruction/API, whichever comes first.
///
/// # Errors
/// [`DetonateError::Load`] if the image cannot be mapped, [`DetonateError::NoEntry`]
/// if it declares no entry point.
pub fn detonate(pe: &PeImage, opts: &DetonateOptions) -> Result<DetonationResult, DetonateError> {
    if !(opts.vm_discover && opts.vm_instances.is_empty()) {
        return detonate_with(pe, opts, None);
    }
    // Survey run: watch every threaded indirect transfer, no dump.
    let survey_hook = SurveyHook::new();
    let survey = survey_hook.survey();
    let survey_opts = DetonateOptions {
        dump: false,
        vm_dispatcher: None,
        vm_loop_head: None,
        capture_globals: Vec::new(),
        ..opts.clone()
    };
    detonate_with(pe, &survey_opts, Some(Box::new(survey_hook)))?;
    let report = pe.report();
    let base = report.optional.image_base.0;
    let image = (base, base + u64::from(report.optional.size_of_image));
    let found = survey.borrow().discover(image);
    let vm_instances = found
        .iter()
        .zip(1u32..)
        .flat_map(|(d, id)| d.specs(base, id, (0, image.1 - base)))
        .collect();
    let mut res =
        detonate_with(pe, &DetonateOptions { vm_instances, vm_discover: false, ..opts.clone() }, None)?;
    res.discovered = found;
    Ok(res)
}

fn detonate_with(
    pe: &PeImage,
    opts: &DetonateOptions,
    extra: Option<Box<dyn Hook>>,
) -> Result<DetonationResult, DetonateError> {
    let mut img = load(pe, &LoadOptions { stack_size: opts.stack_size })?;
    let image_base = img.image_base;
    let entry = img.entry.ok_or(DetonateError::NoEntry)?;

    // A DLL's entry point is DllMain(hinstDLL, fdwReason, lpvReserved). With no
    // explicit args, default to the PROCESS_ATTACH path — the attach branch is
    // the one that runs the protector's startup/unpack to OEP (a DETACH or null
    // reason returns almost immediately). Dynamic load => lpvReserved = NULL.
    let args = if opts.args.is_empty() && pe.report().coff.characteristics & IMAGE_FILE_DLL != 0 {
        vec![image_base, DLL_PROCESS_ATTACH, 0]
    } else {
        opts.args.clone()
    };

    // Import emulation + OEP detection share one ObservedFacts.
    let mut winapi =
        WinApiHook::from_image(&img).with_export_reference(opts.export_reference.clone());
    let xplm = opts.xplm.then(|| Rc::new(RefCell::new(Xplm::new())));
    if let Some(x) = &xplm {
        winapi = winapi.with_xplm(Rc::clone(x));
    }
    let facts = winapi.facts();
    let book = winapi.address_book();
    let oep = OepHook::wrapping(winapi, Rc::clone(&facts));

    // Fold in VM observation only when a dispatcher/loop-head is configured.
    let (hook, vm_handle): (Box<dyn Hook>, _) = if let Some(extra) = extra {
        (Box::new(MultiHook { hooks: vec![Box::new(oep), extra] }), None)
    } else if opts.vm_dispatcher.is_some()
            || opts.vm_loop_head.is_some()
            || !opts.vm_instances.is_empty()
        {
            let vm = if opts.vm_instances.is_empty() {
                VmObserveHook::new(opts.vm_dispatcher, opts.vm_loop_head)
            } else {
                VmObserveHook::with_instances(
                    image_base,
                    opts.vm_instances.iter().copied(),
                    opts.vm_loop_head,
                )
            };
            let handle = vm.observations();
            let multi = MultiHook { hooks: vec![Box::new(oep), Box::new(vm)] };
            (Box::new(multi), Some(handle))
        } else {
            (Box::new(oep), None)
        };

    // IAT-fill observer: fold into the hook set when a slot region is derived.
    let iat_range = iat_range(pe, image_base, opts.iat_region);
    let mut iat_watch = None;
    let hook: Box<dyn Hook> = match iat_range {
        Some((lo, hi)) => {
            let (watcher, log) = IatFillHook::new(lo, hi);
            iat_watch = Some(log);
            Box::new(MultiHook { hooks: vec![hook, Box::new(watcher)] })
        }
        None => hook,
    };

    // Stop cleanly when the entry returns to the synthetic sentinel: for a DLL
    // whose DllMain simply returns (no write-then-execute OEP handoff), this is
    // a legitimate terminus (`Stop::Reached`) rather than an exec fault on the
    // unmapped sentinel. An OEP transfer, when it happens, stops the run earlier
    // via `OepHook` (`Stop::Hook`, with `observed.oep` set).
    let sentinel = img.return_sentinel();
    let mut emu = img.prepare(entry, &args)?;
    let iat_initial = iat_range.map(|(lo, hi)| iat_fill::snapshot(&emu.mem, lo, hi));
    emu.hook = Some(hook);
    // Detonation is a concrete CPU run: by default resolve architecturally-
    // undefined flags to their deterministic hardware value instead of
    // fail-closing, so a `cmov`/`jcc` that consumes e.g. the OF left undefined
    // by a multi-bit rotate/shift/bt proceeds as it would on real silicon. This
    // is an observed/quarantined-tier concretization confined to this contained,
    // in-model run; it never affects sound static output.
    emu.resolve_undefined_flags = opts.resolve_undefined_flags;
    // Only the retired count is consumed; skip the per-step trace allocation.
    emu.record_trace = false;
    let stop = emu.run_until(sentinel, opts.max_steps);
    let steps = emu.steps() as usize;

    let mut observed = facts.borrow().clone();
    if !opts.capture_globals.is_empty() {
        // Best-effort: slots before any unreadable one are still captured.
        let _ = capture_globals(&emu.mem, &mut observed, &opts.capture_globals);
    }
    let vm = vm_handle.map(|h| {
        h.borrow().seed_loop_depth(&mut observed);
        h.borrow().seed_branch_targets(&mut observed);
        h.borrow().clone()
    });

    let xplm = xplm.map(|x| {
        let x = x.borrow();
        XplmSummary {
            callbacks: x.callbacks().to_vec(),
            call_counts: x.call_counts().clone(),
            unlisted: x
                .unlisted()
                .into_iter()
                .map(|((d, f), n)| (d.clone(), f.clone(), n))
                .collect(),
        }
    });

    let dump = opts.dump.then(|| dump_image(&emu.mem, image_base));

    let rip = emu.cpu.rip;
    let iat_fill = match (iat_range, iat_watch, iat_initial) {
        (Some(range), Some(log), Some(initial)) => {
            iat_fill::resolve(&emu.mem, image_base, range, &initial, &log, &book)
        }
        _ => IatFill::default(),
    };

    Ok(DetonationResult {
        observed,
        dump,
        stop,
        steps,
        rip,
        vm,
        xplm,
        discovered: Vec::new(),
        iat_fill,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observe::SymbolOrOrdinal;
    use iced_x86::code_asm::*;

    const IMAGE_BASE: u64 = 0x1_4000_0000;
    const ENTRY_RVA: u64 = 0x1000;
    const IAT_LOADLIB: u64 = IMAGE_BASE + 0x2050;
    const IAT_GETPROC: u64 = IMAGE_BASE + 0x2058;
    const NAME_U: u64 = IMAGE_BASE + 0x20E0;
    const NAME_F: u64 = IMAGE_BASE + 0x20E4;
    const OEP_TARGET: u64 = IMAGE_BASE + 0x2100;

    fn put(d: &mut [u8], at: usize, bytes: &[u8]) {
        d[at..at + bytes.len()].copy_from_slice(bytes);
    }

    /// Entry code: LoadLibraryA("u"); GetProcAddress(that, "F"); then write a
    /// page and jump into it (the synthetic OEP the write-then-execute detector
    /// catches).
    fn entry_code() -> Vec<u8> {
        let mut a = CodeAssembler::new(64).unwrap();
        a.mov(rcx, NAME_U).unwrap();
        a.mov(rax, IAT_LOADLIB).unwrap();
        a.mov(rax, qword_ptr(rax)).unwrap(); // rax = import stub VA
        a.call(rax).unwrap();
        a.mov(rcx, rax).unwrap(); // hmodule
        a.mov(rdx, NAME_F).unwrap();
        a.mov(rax, IAT_GETPROC).unwrap();
        a.mov(rax, qword_ptr(rax)).unwrap();
        a.call(rax).unwrap();
        a.mov(rbx, OEP_TARGET).unwrap();
        a.mov(qword_ptr(rbx), rbx).unwrap(); // write marks the page
        a.jmp(rbx).unwrap(); // transfer into the written page == OEP
        a.assemble(IMAGE_BASE + ENTRY_RVA).unwrap()
    }

    /// DllMain-style entry that records its `hinst` (RCX) and `reason` (RDX) into
    /// two `.idata` globals, then performs the write-then-execute OEP transfer.
    /// Used to assert the detonate driver's default DLL attach args.
    fn arg_capture_code() -> Vec<u8> {
        let mut a = CodeAssembler::new(64).unwrap();
        a.mov(rax, IMAGE_BASE + 0x20E0).unwrap();
        a.mov(qword_ptr(rax), rcx).unwrap(); // [0x20E0] = hinstDLL
        a.mov(rax, IMAGE_BASE + 0x20E8).unwrap();
        a.mov(qword_ptr(rax), rdx).unwrap(); // [0x20E8] = fdwReason
        a.mov(rbx, OEP_TARGET).unwrap();
        a.mov(qword_ptr(rbx), rbx).unwrap();
        a.jmp(rbx).unwrap();
        a.assemble(IMAGE_BASE + ENTRY_RVA).unwrap()
    }

    /// Reads the TEB/PEB the way a protector does and stores each result in a global:
    /// `gs:[0x60]` (PEB), `gs:[eax=0x30]` (TEB self, 32-bit addressing), TEB
    /// StackBase, PEB ImageBaseAddress, PEB->Ldr.
    fn teb_peb_code() -> Vec<u8> {
        let mut a = CodeAssembler::new(64).unwrap();
        let store = |a: &mut CodeAssembler, slot: u64| {
            a.mov(rbx, IMAGE_BASE + slot).unwrap();
            a.mov(qword_ptr(rbx), rax).unwrap();
        };
        a.mov(rax, qword_ptr(0x60).gs()).unwrap();
        store(&mut a, 0x2140);
        a.mov(rdx, rax).unwrap(); // peb
        a.mov(rax, qword_ptr(rdx + 0x10)).unwrap();
        store(&mut a, 0x2148);
        a.mov(rax, qword_ptr(rdx + 0x18)).unwrap();
        store(&mut a, 0x2150);
        a.mov(eax, 0x30).unwrap();
        a.mov(rax, qword_ptr(eax).gs()).unwrap();
        store(&mut a, 0x2158);
        a.mov(rax, qword_ptr(rax + 8)).unwrap();
        store(&mut a, 0x2160);
        a.mov(rbx, OEP_TARGET).unwrap();
        a.mov(qword_ptr(rbx), rbx).unwrap();
        a.jmp(rbx).unwrap();
        a.assemble(IMAGE_BASE + ENTRY_RVA).unwrap()
    }

    /// A minimal PE32+ DLL: `.text` (RX) holds [`entry_code`]; `.idata` (RW)
    /// holds a KERNEL32 import table (`LoadLibraryA`, `GetProcAddress`) plus two
    /// argument strings. The parser must accept it and `load` must map it.
    fn synth_pe() -> Vec<u8> {
        synth_pe_with_entry(&entry_code())
    }

    fn synth_pe_with_entry(entry: &[u8]) -> Vec<u8> {
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
        put(&mut d, o + 16, &(ENTRY_RVA as u32).to_le_bytes()); // AddressOfEntryPoint
        put(&mut d, o + 24, &IMAGE_BASE.to_le_bytes()); // ImageBase
        put(&mut d, o + 32, &0x1000u32.to_le_bytes()); // SectionAlignment
        put(&mut d, o + 36, &0x200u32.to_le_bytes()); // FileAlignment
        put(&mut d, o + 56, &0x4000u32.to_le_bytes()); // SizeOfImage
        put(&mut d, o + 60, &0x200u32.to_le_bytes()); // SizeOfHeaders
        put(&mut d, o + 68, &3u16.to_le_bytes()); // Subsystem
        put(&mut d, o + 108, &16u32.to_le_bytes()); // NumberOfRvaAndSizes
        put(&mut d, o + 112 + 8, &0x2000u32.to_le_bytes()); // import dir RVA
        put(&mut d, o + 112 + 12, &40u32.to_le_bytes()); // import dir size

        // Section headers at 0x188 = 0x98 + 0xF0.
        for (i, (name, rva, raw, flags)) in [
            (&b".text"[..], 0x1000u32, 0x400u32, 0x6000_0020u32), // RX code
            (b".idata", 0x2000, 0x600, 0xC000_0040),              // RW data
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
            put(&mut d, at + 36, &flags.to_le_bytes()); // Characteristics
        }

        // .text raw at file 0x400.
        put(&mut d, 0x400, entry);

        // .idata raw at file 0x600 (RVA 0x2000).
        let idata = 0x600usize;
        put(&mut d, idata, &0x2030u32.to_le_bytes()); // OriginalFirstThunk (ILT)
        put(&mut d, idata + 12, &0x20C0u32.to_le_bytes()); // Name
        put(&mut d, idata + 16, &0x2050u32.to_le_bytes()); // FirstThunk (IAT)
        // ILT + IAT (two entries each) point at the hint/name blobs.
        for (i, hint_rva) in [0x2080u64, 0x2098].into_iter().enumerate() {
            put(&mut d, idata + 0x30 + 8 * i, &hint_rva.to_le_bytes()); // ILT
            put(&mut d, idata + 0x50 + 8 * i, &hint_rva.to_le_bytes()); // IAT
        }
        put(&mut d, idata + 0x80 + 2, b"LoadLibraryA\0"); // RVA 0x2080 (hint=0)
        put(&mut d, idata + 0x98 + 2, b"GetProcAddress\0"); // RVA 0x2098
        put(&mut d, idata + 0xC0, b"kernel32.dll\0"); // RVA 0x20C0
        put(&mut d, idata + 0xE0, b"u\0"); // RVA 0x20E0
        put(&mut d, idata + 0xE4, b"F\0"); // RVA 0x20E4
        d
    }

    #[test]
    fn detonate_captures_import_and_oep_on_synthetic_pe() {
        let data = synth_pe();
        let pe = PeImage::parse(&data).expect("synthetic PE parses");
        let res = detonate(&pe, &DetonateOptions::default()).expect("detonation starts");

        // OEP reached: the write-then-execute detector stopped the run.
        assert_eq!(res.stop, Stop::Hook, "run stopped at the OEP transfer");
        assert_eq!(res.observed.oep, Some(OEP_TARGET), "OEP == written target");

        // The GetProcAddress resolution was captured (LoadLibraryA records none).
        assert_eq!(res.observed.imports.len(), 1, "one import captured");
        let imp = &res.observed.imports[0];
        assert_eq!(imp.module, "u");
        assert_eq!(imp.symbol, SymbolOrOrdinal::Name("F".into()));

        // A snapshot came back and contains the unpacked OEP page.
        let dump = res.dump.expect("dump produced");
        assert!(dump.region_containing(OEP_TARGET).is_some(), "OEP page in dump");
        assert!(res.steps >= 8, "retired the entry block");
    }

    /// Informational probe of a real sample's reach/next-blocker. Optionally
    /// derives an export reference from an UNPROTECTED original so the
    /// direct-export-walk import capture runs:
    /// `VMPX_DETONATE_SAMPLE=<protected.pe> VMPX_IMPORT_REFERENCE=<original.pe> \
    ///   cargo test -p vmpx-emulate --lib detonate_probe -- --ignored --nocapture`
    #[test]
    #[ignore = "needs VMPX_DETONATE_SAMPLE"]
    fn detonate_probe_real_sample() {
        let path = std::env::var("VMPX_DETONATE_SAMPLE").unwrap();
        let data = std::fs::read(&path).unwrap();
        let pe = PeImage::parse(&data).expect("parse sample");
        // Reference bytes must outlive the parsed original.
        let reference_data = std::env::var("VMPX_IMPORT_REFERENCE")
            .ok()
            .map(|p| std::fs::read(p).expect("read reference"));
        let mut export_reference = reference_data
            .as_ref()
            .map(|bytes| {
                let original = PeImage::parse(bytes).expect("parse reference");
                export_reference_from_imports(&original)
            })
            .unwrap_or_default();
        // Experiment: merge full EXPORT tables of listed system DLLs (offline),
        // keyed by file stem, so the protector's runtime walk of them resolves.
        let sys_bytes: Vec<(String, Vec<u8>)> = std::env::var("VMPX_SYS_DLLS")
            .ok()
            .into_iter()
            .flat_map(|list| list.split(',').map(str::to_owned).collect::<Vec<_>>())
            .filter(|p| !p.is_empty())
            .map(|p| {
                let key = std::path::Path::new(&p)
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .to_lowercase();
                (key, std::fs::read(&p).expect("read sys dll"))
            })
            .collect();
        for (key, bytes) in &sys_bytes {
            let pe = PeImage::parse(bytes).expect("parse sys dll");
            let names = export_reference_from_exports(&pe);
            eprintln!("[probe] merged {} exports for {key}", names.len());
            export_reference.insert(key.clone(), names);
        }
        // VMPX_IAT_SYSDB=1 (or a comma list of extra modules): offline System32
        // export DB for the default loader set, so a manual export walk resolves.
        if let Ok(extra) = std::env::var("VMPX_IAT_SYSDB") {
            let db = crate::SysDllDb::new();
            let imported = pe.report().imports;
            let mods = crate::DEFAULT_SYSTEM_MODULES
                .iter()
                .map(|m| (*m).to_owned())
                .chain(extra.split(',').filter(|m| m.ends_with(".dll")).map(str::to_owned))
                .chain(imported.iter().map(|m| m.name.to_lowercase()));
            for m in mods {
                if let Some(names) = db.exports(&m) {
                    export_reference.insert(m, names);
                }
            }
        }
        let ref_syms: usize = export_reference.values().map(Vec::len).sum();
        let max_steps = std::env::var("VMPX_DETONATE_MAXSTEPS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_MAX_STEPS);
        // VMPX_RESOLVE_UNDEF=0 reproduces the sound fail-closed stop; default on.
        let resolve_undefined_flags =
            std::env::var("VMPX_RESOLVE_UNDEF").map_or(true, |v| v != "0");
        // VMPX_VM_DISPATCH_RVA=<hex>[:reg] observes one dispatcher (VIP reg default RDI).
        let vm_instances = std::env::var("VMPX_VM_DISPATCH_RVA")
            .ok()
            .map(|v| {
                let (rva, reg) = v.split_once(':').unwrap_or((&v, "rdi"));
                let reg = match reg {
                    "rsi" => iced_x86::Register::RSI,
                    "rbx" => iced_x86::Register::RBX,
                    _ => iced_x86::Register::RDI,
                };
                VmInstanceSpec {
                    at: crate::vm_observe::DispatchAt::Arrival,
                    dispatcher_rva: u64::from_str_radix(rva.trim_start_matches("0x"), 16).unwrap(),
                    instance: crate::vm_observe::VmInstance {
                        id: 0,
                        vip: reg.into(),
                        bytecode_region: None,
                    },
                }
            })
            .into_iter()
            .collect();
        let opts = DetonateOptions {
            vm_instances,
            export_reference,
            max_steps,
            resolve_undefined_flags,
            xplm: std::env::var("VMPX_XPLM").is_ok_and(|v| v == "1"),
            vm_discover: std::env::var("VMPX_VM_DISCOVER").is_ok_and(|v| v == "1"),
            dump: true,
            ..DetonateOptions::default()
        };
        let res = detonate(&pe, &opts).expect("detonation starts");
        eprintln!(
            "sample={path} resolve_undef={}\n  reference modules={} symbols={ref_syms}\n  steps={} stop={:?} rip={:#x}\n  oep={:?}\n  imports captured={}",
            opts.resolve_undefined_flags,
            opts.export_reference.len(),
            res.steps,
            res.stop,
            res.rip,
            res.observed.oep,
            res.observed.imports.len(),
        );
        for d in &res.discovered {
            eprintln!(
                "  discovered: dispatch_reg={:?} events={} entries={} vip={:?}",
                d.dispatch_reg,
                d.events,
                d.entries.len(),
                d.vip
            );
            for e in d.entries.iter().take(3) {
                eprintln!("    site {:#x} hits={}", e.va, e.hits);
            }
        }
        if let Some(vm) = &res.vm {
            eprintln!(
                "  vm: handlers={} edges={} unattributed={} branch_targets={}",
                vm.handlers.len(),
                vm.edges.len(),
                vm.unattributed,
                vm.branch_targets.len()
            );
            let mut per: BTreeMap<u32, (usize, u64, u64, usize, u64)> = BTreeMap::new();
            for e in &vm.edges {
                let p = per.entry(e.instance).or_insert((0, u64::MAX, 0, 0, 0));
                p.0 += 1;
                p.1 = p.1.min(e.from_vip.min(e.to_vip));
                p.2 = p.2.max(e.from_vip.max(e.to_vip));
                p.3 += usize::from(e.from_vip.abs_diff(e.to_vip) > 0x100);
                p.4 += e.dispatch_count;
            }
            let base = pe.report().optional.image_base.0;
            let sec = |va: u64| {
                pe.report()
                    .sections
                    .iter()
                    .find(|s| {
                        let lo = base + u64::from(s.virtual_address.0);
                        (lo..lo + u64::from(s.virtual_size)).contains(&va)
                    })
                    .map_or("-".to_owned(), |s| s.name.clone())
            };
            for (id, (n, lo, hi, far, disp)) in &per {
                eprintln!(
                    "    inst {id}: edges={n} dispatches={disp} vip=[{lo:#x},{hi:#x}] in [{},{}] far_jumps={far}",
                    sec(*lo),
                    sec(*hi)
                );
            }
            for e in vm.edges.iter().take(12) {
                eprintln!("    {e:?}");
            }
        }
        if let Some(x) = &res.xplm {
            eprintln!(
                "  xplm: callbacks={} calls={:?} unlisted={:?}",
                x.callbacks.len(),
                x.call_counts,
                x.unlisted
            );
        }
        for imp in res.observed.imports.iter().take(30) {
            eprintln!("    {} :: {:?}", imp.module, imp.symbol);
        }
        for sec in &pe.report().sections {
            eprintln!("  sec {:8} rva={:#x} vsize={:#x}", sec.name, sec.virtual_address.0, sec.virtual_size);
        }
        eprintln!("  image_base={:#x} import modules={:?}", pe.report().optional.image_base.0, pe.report().imports.iter().map(|m| (m.name.clone(), m.entries.len())).collect::<Vec<_>>());
        report_iat_fill(&pe, &res.iat_fill, res.dump.as_ref());
    }

    /// Print the IAT-fill mapping, plus how many of the slots `.text` actually
    /// calls/jumps through (`call/jmp [rip+disp]`, linear decode) resolved.
    fn report_iat_fill(pe: &PeImage, fill: &IatFill, dump: Option<&DumpImage>) {
        use crate::memory::Perm;
        use iced_x86::{Decoder, DecoderOptions, FlowControl};
        let report = pe.report();
        let base = report.optional.image_base.0;
        let size = u64::from(report.optional.size_of_image);
        let Some((r_rva, r_len)) = fill.region else {
            eprintln!("  iat_fill: no region");
            return;
        };
        let mut referenced = std::collections::BTreeSet::new();
        // Scan the POST-RUN image: the on-disk code is packed.
        let image_end = base + size;
        for reg in dump.iter().flat_map(|d| &d.regions) {
            if !reg.perm.contains(Perm::X) || reg.va < base || reg.va >= image_end {
                continue;
            }
            let mut dec = Decoder::with_ip(64, &reg.bytes, reg.va, DecoderOptions::NONE);
            while dec.can_decode() {
                let i = dec.decode();
                if matches!(i.flow_control(), FlowControl::IndirectCall | FlowControl::IndirectBranch)
                    && i.is_ip_rel_memory_operand()
                {
                    let rva = i.ip_rel_memory_address().wrapping_sub(base);
                    if (r_rva..r_rva + r_len).contains(&rva) {
                        referenced.insert(rva & !7);
                    }
                }
            }
        }
        let learned = fill.resolved.iter().filter(|s| s.changed).count();
        eprintln!("  resolved: value-changed-by-guest={learned} rewritten-same-as-loader={}", fill.resolved.len() - learned);
        let resolved: std::collections::BTreeSet<u64> =
            fill.resolved.iter().map(|s| s.slot_rva).collect();
        let unresolved: std::collections::BTreeSet<u64> =
            fill.unresolved.iter().map(|s| s.slot_rva).collect();
        eprintln!(
            "  iat_fill region rva={r_rva:#x} len={r_len:#x}: resolved={} unresolved={} untouched={}",
            fill.resolved.len(),
            fill.unresolved.len(),
            fill.untouched
        );
        eprintln!(
            "  text-referenced slots={} -> resolved={} unresolved={} never-filled={}",
            referenced.len(),
            referenced.intersection(&resolved).count(),
            referenced.intersection(&unresolved).count(),
            referenced.difference(&resolved).count() - referenced.intersection(&unresolved).count(),
        );
        let (mut zero, mut in_image, mut in_model, mut other) = (0, 0, 0, 0);
        for u in &fill.unresolved {
            if u.value == 0 {
                zero += 1;
            } else if u.value.wrapping_sub(base) < size {
                in_image += 1;
            } else if (0x20_0000_0000..0x80_0000_0000).contains(&u.value) {
                in_model += 1;
            } else {
                other += 1;
            }
        }
        eprintln!("  unresolved values: zero={zero} in-image={in_image} in-model-region(not exact)={in_model} other={other}");
        let mut by_module: BTreeMap<&str, usize> = BTreeMap::new();
        for s in &fill.resolved {
            *by_module.entry(s.module.as_str()).or_default() += 1;
        }
        eprintln!("  resolved by module: {by_module:?}");
        for s in fill.resolved.iter().take(15) {
            eprintln!("    {:#x} -> {}!{} ({:?}, writes={})", s.slot_rva, s.module, s.name, s.source, s.writes);
        }
        for u in fill.unresolved.iter().take(15) {
            eprintln!("    {:#x} = {:#x} unresolved {:?} (writes={})", u.slot_rva, u.value, u.reason, u.writes);
        }
    }

    #[test]
    fn dll_defaults_to_process_attach_args() {
        // With no explicit args, a DLL entry is driven as DllMain(hinst=image_base,
        // reason=DLL_PROCESS_ATTACH, reserved=0). The entry records RCX/RDX; the
        // captured globals must show the attach arguments.
        let data = synth_pe_with_entry(&arg_capture_code());
        let pe = PeImage::parse(&data).unwrap();
        let opts = DetonateOptions {
            capture_globals: vec![(IMAGE_BASE + 0x20E0, 8), (IMAGE_BASE + 0x20E8, 8)],
            ..DetonateOptions::default()
        };
        let res = detonate(&pe, &opts).expect("detonation starts");
        assert_eq!(res.stop, Stop::Hook, "reached the OEP transfer");

        let g = &res.observed.globals;
        let read = |va: u64| -> u64 {
            let b = &g.iter().find(|x| x.va == va).expect("captured").bytes;
            u64::from_le_bytes(b[..8].try_into().unwrap())
        };
        assert_eq!(read(IMAGE_BASE + 0x20E0), IMAGE_BASE, "hinstDLL == image base");
        assert_eq!(read(IMAGE_BASE + 0x20E8), 1, "fdwReason == DLL_PROCESS_ATTACH");
    }

    const FILL_A: u64 = IMAGE_BASE + 0x2110;

    /// Resolve LoadLibraryA("u") + GetProcAddress(.., "F"), then "fill" three
    /// `.idata` slots: the GetProcAddress result, a loader import stub, and a
    /// value that is no known address; then transfer to the OEP.
    fn fill_code() -> Vec<u8> {
        let mut a = CodeAssembler::new(64).unwrap();
        a.mov(rcx, NAME_U).unwrap();
        a.mov(rax, IAT_LOADLIB).unwrap();
        a.mov(rax, qword_ptr(rax)).unwrap();
        a.call(rax).unwrap();
        a.mov(rcx, rax).unwrap();
        a.mov(rdx, NAME_F).unwrap();
        a.mov(rax, IAT_GETPROC).unwrap();
        a.mov(rax, qword_ptr(rax)).unwrap();
        a.call(rax).unwrap();
        a.mov(rdi, FILL_A).unwrap();
        a.mov(qword_ptr(rdi), rax).unwrap();
        a.mov(rsi, IAT_LOADLIB).unwrap();
        a.mov(rsi, qword_ptr(rsi)).unwrap();
        a.mov(qword_ptr(rdi + 8), rsi).unwrap();
        a.mov(rdx, 0x1234u64).unwrap();
        a.mov(qword_ptr(rdi + 16), rdx).unwrap();
        a.mov(rbx, OEP_TARGET).unwrap();
        a.mov(qword_ptr(rbx), rbx).unwrap();
        a.jmp(rbx).unwrap();
        a.assemble(IMAGE_BASE + ENTRY_RVA).unwrap()
    }

    #[test]
    fn iat_fill_resolves_only_exact_known_addresses() {
        let data = synth_pe_with_entry(&fill_code());
        let pe = PeImage::parse(&data).unwrap();
        let res = detonate(&pe, &DetonateOptions::default()).expect("detonation starts");
        let fill = &res.iat_fill;
        assert_eq!(fill.region, Some((0x2000, 0x200)), "derived from the .idata section");

        let rva = |va: u64| va - IMAGE_BASE;
        let by_rva = |r: u64| fill.resolved.iter().find(|s| s.slot_rva == r);

        let gpa = by_rva(rva(FILL_A)).expect("GetProcAddress result resolves");
        assert_eq!((gpa.module.as_str(), gpa.name.as_str()), ("u", "F"));
        assert!(gpa.changed && gpa.writes == 1);
        assert_eq!(gpa.source, crate::ImportSource::GetProcAddress);

        let stub = by_rva(rva(FILL_A) + 8).expect("loader stub resolves");
        assert_eq!((stub.module.as_str(), stub.name.as_str()), ("kernel32.dll", "LoadLibraryA"));
        assert_eq!(stub.source, crate::ImportSource::LoaderStub);

        // A non-export value stays unresolved, with the reason recorded.
        let bad = fill.unresolved.iter().find(|s| s.slot_rva == rva(FILL_A) + 16).unwrap();
        assert_eq!((bad.value, bad.reason), (0x1234, crate::UnresolvedReason::NotKnownExport));
        assert!(by_rva(rva(FILL_A) + 16).is_none());
        assert!(fill.untouched > 0, "unwritten slots are not reported as filled");

        let off = detonate(&pe, &DetonateOptions { iat_region: IatRegion::Off, ..Default::default() })
            .unwrap();
        assert_eq!(off.iat_fill, IatFill::default());
    }

    #[test]
    fn guest_reads_valid_teb_and_peb() {
        let data = synth_pe_with_entry(&teb_peb_code());
        let pe = PeImage::parse(&data).unwrap();
        let slots = [0x2140u64, 0x2148, 0x2150, 0x2158, 0x2160];
        let opts = DetonateOptions {
            capture_globals: slots.iter().map(|s| (IMAGE_BASE + s, 8)).collect(),
            ..DetonateOptions::default()
        };
        let res = detonate(&pe, &opts).expect("detonation starts");
        assert_eq!(res.stop, Stop::Hook, "reached the OEP transfer, no fault");
        let g = &res.observed.globals;
        let read = |s: u64| -> u64 {
            let b = &g.iter().find(|x| x.va == IMAGE_BASE + s).expect("captured").bytes;
            u64::from_le_bytes(b[..8].try_into().unwrap())
        };
        let (peb, image_base, ldr, teb, stack_base) =
            (read(0x2140), read(0x2148), read(0x2150), read(0x2158), read(0x2160));
        assert_ne!(peb, 0, "gs:[0x60] is a non-null PEB");
        assert_eq!(peb, teb + 0x400, "PEB hangs off the TEB the guest sees at gs:[0x30]");
        assert_eq!(image_base, IMAGE_BASE, "PEB.ImageBaseAddress");
        assert_ne!(ldr, 0, "PEB.Ldr");
        assert_ne!(stack_base, 0, "TEB.StackBase");
    }

    #[test]
    fn detonate_reports_blocker_when_entry_hits_unmodeled() {
        // Budget of 1 step: the run cannot reach OEP, so no OEP is observed and
        // the stop is a clearly-reported step-limit blocker (not a panic).
        let data = synth_pe();
        let pe = PeImage::parse(&data).unwrap();
        let res = detonate(&pe, &DetonateOptions { max_steps: 1, ..DetonateOptions::default() })
            .unwrap();
        assert_eq!(res.stop, Stop::StepLimit);
        assert_eq!(res.observed.oep, None, "OEP not reached under the tiny budget");
    }
}
