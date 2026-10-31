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
use crate::emu::{FlowEvent, Hook, HookAction, MemEvent, Stop};
use crate::loader::{LoadError, LoadOptions, load};
use crate::memory::Memory;
use crate::observe::{ObservedFacts, SymbolOrOrdinal};
use crate::vm_observe::{VmObservations, VmObserveHook};
use crate::winapi::WinApiHook;

const DEFAULT_MAX_STEPS: u64 = 2_000_000;

// COFF characteristic: the image is a DLL.
const IMAGE_FILE_DLL: u16 = 0x2000;
// DllMain reason code for process attach.
const DLL_PROCESS_ATTACH: u64 = 1;

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
    /// Loop-head / recursion-entry VA whose back-edge visits are counted.
    pub vm_loop_head: Option<u64>,
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
            export_reference: BTreeMap::new(),
            dump: true,
            resolve_undefined_flags: true,
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
    /// VM-dispatcher observations, when a dispatcher/loop-head was configured.
    pub vm: Option<VmObservations>,
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
    let winapi =
        WinApiHook::from_image(&img).with_export_reference(opts.export_reference.clone());
    let facts = winapi.facts();
    let oep = OepHook::wrapping(winapi, Rc::clone(&facts));

    // Fold in VM observation only when a dispatcher/loop-head is configured.
    let (hook, vm_handle): (Box<dyn Hook>, _) =
        if opts.vm_dispatcher.is_some() || opts.vm_loop_head.is_some() {
            let vm = VmObserveHook::new(opts.vm_dispatcher, opts.vm_loop_head);
            let handle = vm.observations();
            let multi = MultiHook { hooks: vec![Box::new(oep), Box::new(vm)] };
            (Box::new(multi), Some(handle))
        } else {
            (Box::new(oep), None)
        };

    // Stop cleanly when the entry returns to the synthetic sentinel: for a DLL
    // whose DllMain simply returns (no write-then-execute OEP handoff), this is
    // a legitimate terminus (`Stop::Reached`) rather than an exec fault on the
    // unmapped sentinel. An OEP transfer, when it happens, stops the run earlier
    // via `OepHook` (`Stop::Hook`, with `observed.oep` set).
    let sentinel = img.return_sentinel();
    let mut emu = img.prepare(entry, &args)?;
    emu.hook = Some(hook);
    // Detonation is a concrete CPU run: by default resolve architecturally-
    // undefined flags to their deterministic hardware value instead of
    // fail-closing, so a `cmov`/`jcc` that consumes e.g. the OF left undefined
    // by a multi-bit rotate/shift/bt proceeds as it would on real silicon. This
    // is an observed/quarantined-tier concretization confined to this contained,
    // in-model run; it never affects sound static output.
    emu.resolve_undefined_flags = opts.resolve_undefined_flags;
    let stop = emu.run_until(sentinel, opts.max_steps);
    let steps = emu.trace.len();

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

    let dump = opts.dump.then(|| dump_image(&emu.mem, image_base));

    Ok(DetonationResult { observed, dump, stop, steps, vm })
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
        let ref_syms: usize = export_reference.values().map(Vec::len).sum();
        let max_steps = std::env::var("VMPX_DETONATE_MAXSTEPS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_MAX_STEPS);
        // VMPX_RESOLVE_UNDEF=0 reproduces the sound fail-closed stop; default on.
        let resolve_undefined_flags =
            std::env::var("VMPX_RESOLVE_UNDEF").map_or(true, |v| v != "0");
        let opts = DetonateOptions {
            export_reference,
            max_steps,
            resolve_undefined_flags,
            ..DetonateOptions::default()
        };
        let res = detonate(&pe, &opts).expect("detonation starts");
        eprintln!(
            "sample={path} resolve_undef={}\n  reference modules={} symbols={ref_syms}\n  steps={} stop={:?}\n  oep={:?}\n  imports captured={}",
            opts.resolve_undefined_flags,
            opts.export_reference.len(),
            res.steps,
            res.stop,
            res.observed.oep,
            res.observed.imports.len(),
        );
        for imp in res.observed.imports.iter().take(30) {
            eprintln!("    {} :: {:?}", imp.module, imp.symbol);
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
