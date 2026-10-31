//! Layer 2: a [`Hook`] that emulates the minimum Win32 surface a packed sample
//! touches early (memory, loader, time/anti-debug) and captures dynamically
//! resolved imports into [`ObservedFacts`].
//!
//! # Containment
//!
//! Every modeled API is a pure in-model computation. Fake handles and proc
//! addresses are deterministic sentinel values; allocations map real *guest*
//! pages. Nothing here opens a socket, file, process or FFI boundary, and no
//! host OS call is made. An API that is not modeled does not get guessed: the
//! run stops ([`HookAction::Stop`]) so no unsound state leaks into the trace.
//!
//! # Install
//!
//! ```ignore
//! let hook = WinApiHook::from_image(&img);   // clones img.import_stubs
//! let facts = hook.facts();                   // shared handle to observations
//! let mut emu = img.prepare(rip, args)?;
//! emu.hook = Some(Box::new(hook));
//! emu.run_until(sentinel, MAX);
//! let imports = &facts.borrow().imports;      // ObservedImport bindings
//! ```

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use crate::cpu::Cpu;
use crate::emu::{FlowEvent, FlowKind, Hook, HookAction, MemEvent};
use crate::loader::LoadedImage;
use crate::memory::{Access, MapError, Memory, Perm};
use crate::ntdll::{NtDispatch, NtState};
use crate::observe::{ObservedFacts, SymbolOrOrdinal};

// Win64 integer-argument registers (RCX, RDX, R8, R9) and RAX/RSP by index
// into `Cpu::regs`.
const ARG: [usize; 4] = [1, 2, 8, 9];
const RAX: usize = 0;
const RSP: usize = 4;

// Deterministic in-model address pools, each far from the loader's regions and
// from each other so a map never collides.
const ALLOC_BASE: u64 = 0x0000_0050_0000_0000; // VirtualAlloc/HeapAlloc pages
const PROC_BASE: u64 = 0x0000_0060_0000_0000; // GetProcAddress result sentinels
const HMODULE_BASE: u64 = 0x0000_0070_0000_0000; // LoadLibrary/GetModuleHandle
const PROCESS_HEAP: u64 = 0x0000_0070_FFFF_0000; // GetProcessHeap sentinel

// Each minted HMODULE gets this much private address space; a faithful, readable
// PE-header image is mapped there so the protector can walk the module it just
// resolved (VMProtect dereferences GetModuleHandle results to parse headers).
// Sized to hold a full system-DLL export table (ntdll exports ~2.5k functions).
const MODULE_IMAGE_SIZE: u64 = 0x8_0000;

// Within a module image: the export directory + its name/ordinal/address arrays
// and the name-string blob live in [EXPORT_RVA, STUB_REGION_RVA); each exported
// function's address points into the stub region above it. The data-directory
// export Size spans exactly the former range so a function RVA in the stub region
// is read as a real address, never mistaken for a forwarder string.
const EXPORT_RVA: u32 = 0x1000;
const MODULE_NAME_RVA: u32 = 0x1100;
const EXPORT_DATA_RVA: u32 = 0x1200; // arrays + name strings bump from here
const STUB_REGION_RVA: u32 = 0x4_0000; // per-function fake addresses from here
const STUB_STRIDE: u32 = 0x10;
const EXPORT_DIR_SIZE: u32 = STUB_REGION_RVA - EXPORT_RVA;

// Fixed FILETIME (100ns ticks) written by GetSystemTimeAsFileTime.
const FIXED_FILETIME: u64 = 0x01D0_0000_0000_0000;

const PAGE: u64 = 0x1000;

fn round_up(v: u64) -> u64 {
    v.div_ceil(PAGE) * PAGE
}

/// What a dispatched API resolved to.
enum Dispatch {
    /// Set RAX to this and resume.
    Ret(u64),
    /// Modeled void call; RAX left at 0.
    Void,
    /// Not modeled, or a fault while modeling it: stop the run (never guess).
    Stop,
}

/// Emulates the early Win32 surface and records resolved imports.
///
/// Dispatch is keyed on the `(dll, func)` the called address maps to in a copy
/// of the loader's `import_stubs`; a transfer whose target is not a known stub
/// is left to execute normally ([`HookAction::Continue`]).
pub struct WinApiHook {
    stubs: BTreeMap<u64, (String, String)>,
    facts: Rc<RefCell<ObservedFacts>>,
    alloc_next: u64,
    hmodule_next: u64,
    /// Base of the sample's own mapped image, returned for a `GetModuleHandle(NULL)`
    /// so the protector walks the real (already-mapped) headers. 0 = unknown.
    image_base: u64,
    /// Fake HMODULE -> lowercase module name (for `GetProcAddress` resolution).
    handle_names: BTreeMap<u64, String>,
    /// Lowercase module name -> fake HMODULE (stable across repeated lookups).
    name_handles: BTreeMap<String, u64>,
    /// (module, symbol) -> fake proc VA (stable across repeated lookups).
    procs: BTreeMap<(String, String), u64>,
    /// Lowercase module name -> exports to publish in that module's fake export
    /// table, from an OFFLINE reference (names are never invented). A module
    /// with no entry here exports nothing (the original empty-directory shape).
    export_reference: BTreeMap<String, Vec<SymbolOrOrdinal>>,
    /// Read-watches over each populated export address table: a read of a slot
    /// is the protector resolving that export via the direct walk.
    export_watches: Vec<ExportWatch>,
    /// (module, symbol-key) already recorded from the walk, to dedupe the
    /// repeated reads a binary search performs.
    walk_recorded: BTreeSet<(String, String)>,
    /// Resolved export stub VA -> (module, symbol). Populated for every module
    /// whose fake export table was published; a guest `call`/`jmp` to one of
    /// these VAs is the protector invoking a directly-walked export (notably the
    /// ntdll `Zw*` syscall stubs it uses to unpack). Dispatched like an IAT thunk.
    export_stub_addrs: BTreeMap<u64, (String, String)>,
    /// In-model NT object + memory state for the emulated ntdll syscall stubs.
    nt: NtState,
}

/// A read-watch over one module's `AddressOfFunctions` array. When the protector
/// reads a slot (to obtain a function RVA and compute its VA), we know which
/// export it resolved and record it as an [`ObservedImport`].
struct ExportWatch {
    module: String,
    /// VA of the `AddressOfFunctions` array.
    functions_va: u64,
    /// functions-array index -> the export published at that slot, when any.
    slots: Vec<Option<ResolvedExport>>,
}

/// An export published at a functions-array slot and the fake VA the protector
/// obtains when it reads that slot (`module_base + stub_rva`).
struct ResolvedExport {
    symbol: SymbolOrOrdinal,
    va: u64,
}

/// Computed export-table layout returned by [`map_module_image`] so the hook can
/// register the matching read-watch.
struct ExportLayout {
    functions_va: u64,
    slots: Vec<Option<ResolvedExport>>,
}

fn symbol_key(symbol: &SymbolOrOrdinal) -> String {
    match symbol {
        SymbolOrOrdinal::Name(n) => n.clone(),
        SymbolOrOrdinal::Ordinal(o) => format!("#{o}"),
    }
}

impl WinApiHook {
    #[must_use]
    pub fn new(stubs: BTreeMap<u64, (String, String)>) -> Self {
        Self {
            stubs,
            facts: Rc::new(RefCell::new(ObservedFacts::new())),
            alloc_next: ALLOC_BASE,
            hmodule_next: HMODULE_BASE,
            image_base: 0,
            handle_names: BTreeMap::new(),
            name_handles: BTreeMap::new(),
            procs: BTreeMap::new(),
            export_reference: BTreeMap::new(),
            export_watches: Vec::new(),
            walk_recorded: BTreeSet::new(),
            export_stub_addrs: BTreeMap::new(),
            nt: NtState::new(),
        }
    }

    /// Build a hook from a loaded image, copying its import-stub table and
    /// recording the image base (returned for `GetModuleHandle(NULL)`).
    #[must_use]
    pub fn from_image(img: &LoadedImage) -> Self {
        let mut h = Self::new(img.import_stubs.clone());
        h.image_base = img.image_base;
        h
    }

    /// Attach an OFFLINE export reference: `{ module -> [name|ordinal] }`. Each
    /// resolved module's fake export directory is populated from its entry so a
    /// protector that resolves imports by manually walking the export table
    /// (a classic protector anti-hook) finds the functions it is looking for.
    /// Keys are matched case-insensitively. Names come only from this reference.
    #[must_use]
    pub fn with_export_reference(
        mut self,
        reference: BTreeMap<String, Vec<SymbolOrOrdinal>>,
    ) -> Self {
        self.export_reference =
            reference.into_iter().map(|(k, v)| (k.to_lowercase(), v)).collect();
        self
    }

    /// Shared handle to the observations collected during the run. Clone it
    /// before installing the hook, then read it after the run.
    #[must_use]
    pub fn facts(&self) -> Rc<RefCell<ObservedFacts>> {
        Rc::clone(&self.facts)
    }

    fn arg(cpu: &Cpu, i: usize) -> u64 {
        cpu.regs[ARG[i]]
    }

    /// Bump-allocate `size` bytes of fresh RW guest pages; returns the base VA,
    /// or 0 (NULL, the Win32 failure value) if the map fails.
    fn alloc(&mut self, mem: &mut Memory, size: u64) -> u64 {
        let len = round_up(size.max(1));
        let va = self.alloc_next;
        if mem.map(va, len, Perm::RW).is_err() {
            return 0;
        }
        self.alloc_next = va + len;
        va
    }

    /// Resolve `name` to an HMODULE, mapping a readable module image behind it.
    ///
    /// A Win32 HMODULE *is* the module's load base, and protectors (VMProtect in
    /// particular) dereference it to parse the PE headers of the module they just
    /// resolved. So the handle must point at a readable, structurally faithful
    /// image, not a bare sentinel:
    ///
    /// * `GetModuleHandle(NULL)` ("<self>") returns the sample's real, already
    ///   mapped base — the protector then walks the genuine headers.
    /// * Any other module gets a fresh base with a synthetic-but-valid PE header
    ///   ([`Self::map_module_image`]) mapped once on first resolution.
    ///
    /// Returns 0 (NULL, the Win32 failure value) if the backing map fails.
    fn module_handle(&mut self, mem: &mut Memory, name: &str) -> u64 {
        if name == "<self>" && self.image_base != 0 {
            return self.image_base;
        }
        let key = name.to_lowercase();
        if let Some(&h) = self.name_handles.get(&key) {
            return h;
        }
        let h = self.hmodule_next;
        let exports = self.export_reference.get(&key).cloned().unwrap_or_default();
        match map_module_image(mem, h, &key, &exports) {
            Ok(Some(layout)) => {
                for export in layout.slots.iter().flatten() {
                    self.export_stub_addrs
                        .insert(export.va, (key.clone(), symbol_key(&export.symbol)));
                }
                self.export_watches.push(ExportWatch {
                    module: key.clone(),
                    functions_va: layout.functions_va,
                    slots: layout.slots,
                });
            }
            Ok(None) => {}
            Err(_) => return 0,
        }
        self.hmodule_next += MODULE_IMAGE_SIZE;
        self.name_handles.insert(key.clone(), h);
        self.handle_names.insert(h, key);
        h
    }

    /// A read landing in a watched `AddressOfFunctions` slot is the protector
    /// resolving that export through the direct walk: record the function it
    /// just obtained the address of (deduped across the binary search's reads).
    fn note_export_read(&mut self, addr: u64) {
        for watch in &self.export_watches {
            if addr < watch.functions_va {
                continue;
            }
            let idx = ((addr - watch.functions_va) / 4) as usize;
            let Some(Some(export)) = watch.slots.get(idx) else {
                continue;
            };
            let key = (watch.module.clone(), symbol_key(&export.symbol));
            if self.walk_recorded.insert(key) {
                self.facts.borrow_mut().record_import(
                    export.va,
                    watch.module.clone(),
                    export.symbol.clone(),
                );
            }
            return;
        }
    }

    /// Deterministic fake proc VA for `(module, symbol)`.
    fn proc_addr(&mut self, module: &str, symbol: &str) -> u64 {
        let key = (module.to_string(), symbol.to_string());
        let n = self.procs.len() as u64;
        *self.procs.entry(key).or_insert(PROC_BASE + n * 8)
    }

    fn dispatch(&mut self, dll: &str, func: &str, cpu: &Cpu, mem: &mut Memory) -> Dispatch {
        match (dll.to_lowercase().as_str(), func) {
            // ---- Memory ------------------------------------------------------
            // VirtualAlloc(lpAddress, dwSize, flAllocationType, flProtect)
            (_, "VirtualAlloc") => {
                let size = Self::arg(cpu, 1);
                Dispatch::Ret(self.alloc(mem, size))
            }
            // HeapAlloc(hHeap, dwFlags, dwBytes) — pages are already zeroed.
            (_, "HeapAlloc") => {
                let size = Self::arg(cpu, 2);
                Dispatch::Ret(self.alloc(mem, size))
            }
            // VirtualProtect(lpAddress, dwSize, flNewProtect, lpflOldProtect):
            // report success; write a plausible old protection if asked. We do
            // not re-permission pages (allocations are already RW).
            (_, "VirtualProtect") => {
                let old_ptr = Self::arg(cpu, 3);
                if old_ptr != 0 && mem.write_uint(old_ptr, 4, 0x40).is_err() {
                    return Dispatch::Stop;
                }
                Dispatch::Ret(1)
            }
            // Frees are a no-op in the bump model; report success.
            (_, "VirtualFree" | "HeapFree") => Dispatch::Ret(1),
            (_, "GetProcessHeap") => Dispatch::Ret(PROCESS_HEAP),

            // ---- Loader ------------------------------------------------------
            (_, "GetModuleHandleA" | "LoadLibraryA") => {
                let p = Self::arg(cpu, 0);
                let name = if p == 0 { "<self>".to_string() } else {
                    match read_cstr(mem, p) {
                        Some(s) => s,
                        None => return Dispatch::Stop,
                    }
                };
                Dispatch::Ret(self.module_handle(mem, &name))
            }
            (_, "GetModuleHandleW" | "LoadLibraryW") => {
                let p = Self::arg(cpu, 0);
                let name = if p == 0 { "<self>".to_string() } else {
                    match read_wstr(mem, p) {
                        Some(s) => s,
                        None => return Dispatch::Stop,
                    }
                };
                Dispatch::Ret(self.module_handle(mem, &name))
            }
            // GetProcAddress(hModule, lpProcName) — the import bridge.
            (_, "GetProcAddress") => {
                let hmodule = Self::arg(cpu, 0);
                let name_or_ord = Self::arg(cpu, 1);
                let module =
                    self.handle_names.get(&hmodule).cloned().unwrap_or_else(|| "<unknown>".into());
                // lpProcName < 0x10000 with zero high bits is an ordinal
                // (MAKEINTRESOURCE); otherwise a NUL-terminated name pointer.
                let symbol = if name_or_ord != 0 && name_or_ord < 0x1_0000 {
                    SymbolOrOrdinal::Ordinal(name_or_ord as u16)
                } else {
                    match read_cstr(mem, name_or_ord) {
                        Some(s) => SymbolOrOrdinal::Name(s),
                        None => return Dispatch::Stop,
                    }
                };
                let sym_str = match &symbol {
                    SymbolOrOrdinal::Name(s) => s.clone(),
                    SymbolOrOrdinal::Ordinal(o) => format!("#{o}"),
                };
                let va = self.proc_addr(&module, &sym_str);
                self.facts.borrow_mut().record_import(va, module, symbol);
                Dispatch::Ret(va)
            }

            // ---- Time / anti-debug ------------------------------------------
            (_, "GetSystemTimeAsFileTime") => {
                let p = Self::arg(cpu, 0);
                if p != 0 && mem.write_uint(p, 8, FIXED_FILETIME).is_err() {
                    return Dispatch::Stop;
                }
                Dispatch::Void
            }
            (_, "IsDebuggerPresent") => Dispatch::Ret(0),
            // CheckRemoteDebuggerPresent(hProcess, pbDebuggerPresent) -> BOOL ok
            (_, "CheckRemoteDebuggerPresent") => {
                let p = Self::arg(cpu, 1);
                if p != 0 && mem.write_uint(p, 4, 0).is_err() {
                    return Dispatch::Stop;
                }
                Dispatch::Ret(1)
            }

            // Unmodeled: stop rather than guess (see module docs).
            _ => Dispatch::Stop,
        }
    }
}

impl Hook for WinApiHook {
    fn on_mem(&mut self, ev: &MemEvent, _cpu: &mut Cpu, _mem: &mut Memory) -> HookAction {
        if ev.access == Access::Read && !self.export_watches.is_empty() {
            self.note_export_read(ev.addr);
        }
        HookAction::Continue
    }

    fn on_flow(&mut self, ev: &FlowEvent, cpu: &mut Cpu, mem: &mut Memory) -> HookAction {
        // A protector reaches a stub by `call`/`jmp` or the `push addr; ret`
        // idiom (the popped target is the stub); all three are dispatched.
        if !matches!(ev.kind, FlowKind::Call | FlowKind::Jmp | FlowKind::Ret) {
            return HookAction::Continue;
        }
        // An IAT import thunk (the loader's STUB_REGION sentinels).
        if let Some((dll, func)) = self.stubs.get(&ev.to).cloned() {
            return self.dispatch_win32(&dll, &func, ev, cpu, mem);
        }
        // A directly-walked export stub (e.g. the ntdll Zw* the protector
        // resolves to unpack). ntdll syscall stubs go to the NT emulator; any
        // other walked export reuses the Win32 surface (unmodeled -> Stop).
        if let Some((dll, func)) = self.export_stub_addrs.get(&ev.to).cloned() {
            if dll == "ntdll.dll" && is_nt_syscall(&func) {
                let sab = Self::stack_arg_base(ev, cpu);
                return match self.nt.dispatch(&func, cpu, mem, sab) {
                    NtDispatch::Status(s) => {
                        cpu.regs[RAX] = s;
                        Self::resume(ev, cpu, mem)
                    }
                    NtDispatch::Stop => HookAction::Stop,
                };
            }
            return self.dispatch_win32(&dll, &func, ev, cpu, mem);
        }
        HookAction::Continue // not an import thunk
    }
}

/// An ntdll export is a syscall stub when its name is an `Nt*`/`Zw*` pair.
fn is_nt_syscall(func: &str) -> bool {
    func.starts_with("Nt") || func.starts_with("Zw")
}

impl WinApiHook {
    /// Dispatch a Win32 API by `(dll, func)` and return into the guest.
    fn dispatch_win32(
        &mut self,
        dll: &str,
        func: &str,
        ev: &FlowEvent,
        cpu: &mut Cpu,
        mem: &mut Memory,
    ) -> HookAction {
        match self.dispatch(dll, func, cpu, mem) {
            Dispatch::Stop => HookAction::Stop,
            Dispatch::Ret(v) => {
                cpu.regs[RAX] = v;
                Self::resume(ev, cpu, mem)
            }
            Dispatch::Void => Self::resume(ev, cpu, mem),
        }
    }

    /// Guest address of the 5th NT argument's stack slot. At a `Call`
    /// interception the return address is not yet pushed, so the 5th arg sits at
    /// `[RSP+0x20]` (above the 0x20 shadow space); a `Jmp` thunk has already
    /// pushed a return address, shifting it to `[RSP+0x28]`.
    fn stack_arg_base(ev: &FlowEvent, cpu: &Cpu) -> u64 {
        // Offset to arg5 = (RSP at stub entry) + 0x28. A `Call` has not pushed
        // the return address yet (entry RSP = RSP-8 => +0x20); a `Jmp` thunk has
        // (entry RSP = RSP => +0x28); a `push;ret` pops the target first (entry
        // RSP = RSP+8 => +0x30).
        let shadow = match ev.kind {
            FlowKind::Jmp => 0x28,
            FlowKind::Ret => 0x30,
            _ => 0x20,
        };
        cpu.regs[RSP].wrapping_add(shadow)
    }

    /// Return into the guest after an emulated API.
    ///
    /// * `Call`: the emulator skips its implicit return-address push on
    ///   `Redirect`, so RSP is already balanced for a call+ret pair — resume at
    ///   the fall-through (`ev.next`), which is the return address.
    /// * `Jmp` (tail-call thunk): the caller's return address sits at `[RSP]`;
    ///   emulate the API's `ret` by popping it.
    fn resume(ev: &FlowEvent, cpu: &mut Cpu, mem: &mut Memory) -> HookAction {
        match ev.kind {
            FlowKind::Call => {
                cpu.rip = ev.next;
                HookAction::Redirect
            }
            FlowKind::Jmp => {
                let sp = cpu.regs[RSP];
                match mem.read_uint(sp, 8) {
                    Ok(ra) => {
                        cpu.regs[RSP] = sp.wrapping_add(8);
                        cpu.rip = ra;
                        HookAction::Redirect
                    }
                    Err(_) => HookAction::Stop,
                }
            }
            // `push target; ret`: [RSP] = target (popped by this ret), [RSP+8] =
            // the real return address the emulated stub's own `ret` lands on.
            FlowKind::Ret => {
                let sp = cpu.regs[RSP];
                match mem.read_uint(sp.wrapping_add(8), 8) {
                    Ok(ra) => {
                        cpu.regs[RSP] = sp.wrapping_add(16);
                        cpu.rip = ra;
                        HookAction::Redirect
                    }
                    Err(_) => HookAction::Stop,
                }
            }
            _ => HookAction::Continue,
        }
    }
}

/// Map a readable, structurally faithful PE-header image for a resolved module
/// at `base`, label it `name`, and populate its export directory from the
/// OFFLINE `exports` reference.
///
/// This is the shape the OS loader guarantees behind every HMODULE: a DOS stub
/// (`MZ` + `e_lfanew`), NT headers (`PE\0\0`, a PE32+ optional header whose
/// `ImageBase` is `base`), and a valid `IMAGE_EXPORT_DIRECTORY`. When `exports` is
/// empty the directory is well-formed but empty (the module exports nothing);
/// otherwise its name/ordinal/address tables are filled so a protector that
/// resolves imports by manually walking the export table (parse DOS->NT->export
/// dir->binary-search names) resolves each one.
///
/// Each exported function gets a deterministic fake address inside the module's
/// own stub region (`base + STUB_REGION_RVA + index*STUB_STRIDE`); the export
/// data-directory `Size` is set so those addresses sit *outside* it and are read
/// as real addresses, not forwarder strings. Names come only from the reference
/// — nothing is invented — and everything is self-contained (no host access).
///
/// Returns the [`ExportLayout`] (for the read-watch) when a populated table was
/// written, `None` for the empty case, or [`MapError`] if the backing map fails.
fn map_module_image(
    mem: &mut Memory,
    base: u64,
    name: &str,
    exports: &[SymbolOrOrdinal],
) -> Result<Option<ExportLayout>, MapError> {
    const E_LFANEW: u32 = 0x40;
    const EMPTY_ARR_RVA: u32 = EXPORT_DATA_RVA;

    let plan = plan_exports(exports);
    // Map only what this module needs (headers + any populated export table and
    // stubs), rounded to a page and capped at the per-module stride. An empty
    // module needs just its headers and the (empty) export directory.
    let map_len = plan.as_ref().map_or(0x2000, |p| round_up(u64::from(p.image_end)));
    let map_len = map_len.min(MODULE_IMAGE_SIZE);
    mem.map(base, map_len, Perm::R)?;
    let mut w = |off: u64, bytes: &[u8]| {
        // Within a region we just mapped; poke ignores perms and only faults if
        // unmapped, which cannot happen here.
        let _ = mem.poke(base + off, bytes);
    };

    // DOS header.
    w(0x00, &0x5A4Du16.to_le_bytes()); // "MZ"
    w(0x3C, &E_LFANEW.to_le_bytes());

    // NT headers.
    let nt = u64::from(E_LFANEW);
    w(nt, &0x0000_4550u32.to_le_bytes()); // "PE\0\0"
    // IMAGE_FILE_HEADER.
    w(nt + 4, &0x8664u16.to_le_bytes()); // Machine = AMD64
    w(nt + 6, &0u16.to_le_bytes()); // NumberOfSections
    w(nt + 20, &0xF0u16.to_le_bytes()); // SizeOfOptionalHeader
    w(nt + 22, &0x2022u16.to_le_bytes()); // Characteristics (DLL | executable)
    // IMAGE_OPTIONAL_HEADER64.
    let opt = nt + 24;
    w(opt, &0x20Bu16.to_le_bytes()); // Magic = PE32+
    w(opt + 24, &base.to_le_bytes()); // ImageBase
    w(opt + 56, &(map_len as u32).to_le_bytes()); // SizeOfImage
    w(opt + 108, &16u32.to_le_bytes()); // NumberOfRvaAndSizes

    // Module name string (for a protector that reads the export Name field).
    let mut nm = name.as_bytes().to_vec();
    nm.push(0);
    w(u64::from(MODULE_NAME_RVA), &nm);

    let Some(plan) = plan else {
        // Empty (or over-large): keep the well-formed, zero-export directory.
        w(opt + 112, &EXPORT_RVA.to_le_bytes());
        w(opt + 116, &0x28u32.to_le_bytes());
        let exp = u64::from(EXPORT_RVA);
        w(exp + 0x0C, &MODULE_NAME_RVA.to_le_bytes()); // Name
        w(exp + 0x10, &1u32.to_le_bytes()); // Base (ordinal base)
        w(exp + 0x14, &0u32.to_le_bytes()); // NumberOfFunctions
        w(exp + 0x18, &0u32.to_le_bytes()); // NumberOfNames
        w(exp + 0x1C, &EMPTY_ARR_RVA.to_le_bytes()); // AddressOfFunctions
        w(exp + 0x20, &EMPTY_ARR_RVA.to_le_bytes()); // AddressOfNames
        w(exp + 0x24, &EMPTY_ARR_RVA.to_le_bytes()); // AddressOfNameOrdinals
        return Ok(None);
    };

    // Data directory[0] = export table, sized to span the directory and its
    // arrays/strings but NOT the stub region above them.
    w(opt + 112, &EXPORT_RVA.to_le_bytes());
    w(opt + 116, &EXPORT_DIR_SIZE.to_le_bytes());

    // Functions array: RVA of each function's stub (or 0 for an empty slot).
    for (i, slot) in plan.functions.iter().enumerate() {
        w(u64::from(plan.functions_rva) + (i as u64) * 4, &slot.to_le_bytes());
    }
    // Name array (sorted ascending by bytes) + parallel name-ordinal array.
    for (i, (name_rva, ordinal_index)) in plan.names.iter().enumerate() {
        w(u64::from(plan.names_rva) + (i as u64) * 4, &name_rva.to_le_bytes());
        w(u64::from(plan.ordinals_rva) + (i as u64) * 2, &ordinal_index.to_le_bytes());
    }
    // Name strings blob.
    for (rva, bytes) in &plan.strings {
        w(u64::from(*rva), bytes);
    }

    // IMAGE_EXPORT_DIRECTORY.
    let exp = u64::from(EXPORT_RVA);
    w(exp + 0x0C, &MODULE_NAME_RVA.to_le_bytes()); // Name
    w(exp + 0x10, &1u32.to_le_bytes()); // Base (ordinal base)
    w(exp + 0x14, &plan.function_count.to_le_bytes()); // NumberOfFunctions
    w(exp + 0x18, &plan.name_count.to_le_bytes()); // NumberOfNames
    w(exp + 0x1C, &plan.functions_rva.to_le_bytes()); // AddressOfFunctions
    w(exp + 0x20, &plan.names_rva.to_le_bytes()); // AddressOfNames
    w(exp + 0x24, &plan.ordinals_rva.to_le_bytes()); // AddressOfNameOrdinals

    Ok(Some(ExportLayout {
        functions_va: base + u64::from(plan.functions_rva),
        slots: plan
            .functions
            .iter()
            .enumerate()
            .map(|(i, &stub_rva)| {
                plan.slot_symbol(i).map(|symbol| ResolvedExport {
                    symbol,
                    va: base + u64::from(stub_rva),
                })
            })
            .collect(),
    }))
}

/// A planned export table: the RVAs and byte arrays to write, plus the symbol at
/// each functions-array slot (for the read-watch). `None` from [`plan_exports`]
/// means nothing to publish (empty reference) or a table too large to fit the
/// module image — both fall back to the empty directory.
struct ExportPlan {
    functions_rva: u32,
    names_rva: u32,
    ordinals_rva: u32,
    function_count: u32,
    name_count: u32,
    /// First RVA past everything written (headers excluded): the module image
    /// need only be mapped this far.
    image_end: u32,
    /// functions[i] = stub RVA (0 = unused slot).
    functions: Vec<u32>,
    /// Sorted name RVA + its functions-array index (nameOrdinal).
    names: Vec<(u32, u16)>,
    /// (RVA, NUL-terminated bytes) for each name string.
    strings: Vec<(u32, Vec<u8>)>,
    /// functions-array index -> symbol, for the read-watch.
    slot_symbols: Vec<Option<SymbolOrOrdinal>>,
}

impl ExportPlan {
    fn slot_symbol(&self, index: usize) -> Option<SymbolOrOrdinal> {
        self.slot_symbols.get(index).cloned().flatten()
    }
}

/// Compute an [`ExportPlan`] from a module's reference exports.
///
/// Ordinal base is 1. Ordinal-imported exports occupy `functions[ordinal - 1]`
/// (so a walk's `functions[ordinal - Base]` lands on them); name-imported
/// exports are placed in the slots above the highest such ordinal, and each
/// gets a sorted name-table entry pointing at its slot. Names are sorted by
/// byte value so a protector's binary search over `AddressOfNames` resolves.
fn plan_exports(exports: &[SymbolOrOrdinal]) -> Option<ExportPlan> {
    if exports.is_empty() {
        return None;
    }
    let mut names: Vec<String> = Vec::new();
    let mut ordinals: Vec<u16> = Vec::new();
    for e in exports {
        match e {
            SymbolOrOrdinal::Name(n) => names.push(n.clone()),
            SymbolOrOrdinal::Ordinal(o) => ordinals.push(*o),
        }
    }
    // Slots below `named_base` are reserved for ordinal exports at index
    // `ordinal - 1`; named exports fill the slots from `named_base` up.
    let named_base = ordinals.iter().map(|o| u32::from(*o)).max().unwrap_or(0);
    let function_count = named_base + u32::try_from(names.len()).ok()?;

    let mut slot_symbols: Vec<Option<SymbolOrOrdinal>> = vec![None; function_count as usize];
    for o in &ordinals {
        if *o >= 1 {
            slot_symbols[(*o - 1) as usize] = Some(SymbolOrOrdinal::Ordinal(*o));
        }
    }
    // Named exports sorted by bytes so the name table is binary-searchable; each
    // keeps its functions-array slot index.
    let mut named_sorted: Vec<String> = names;
    named_sorted.sort();
    let mut name_slots: Vec<(String, u32)> = Vec::new();
    for (k, n) in named_sorted.iter().enumerate() {
        let slot = named_base + u32::try_from(k).ok()?;
        slot_symbols[slot as usize] = Some(SymbolOrOrdinal::Name(n.clone()));
        name_slots.push((n.clone(), slot));
    }

    // Lay out: functions array, names array, ordinals array, name strings.
    let name_count = u32::try_from(name_slots.len()).ok()?;
    let functions_rva = EXPORT_DATA_RVA;
    let names_rva = functions_rva + function_count * 4;
    let ordinals_rva = names_rva + name_count * 4;
    let mut cursor = ordinals_rva + name_count * 2;

    let mut strings: Vec<(u32, Vec<u8>)> = Vec::new();
    let mut names_table: Vec<(u32, u16)> = Vec::new();
    for (n, slot) in &name_slots {
        let mut bytes = n.as_bytes().to_vec();
        bytes.push(0);
        let rva = cursor;
        cursor += u32::try_from(bytes.len()).ok()?;
        names_table.push((rva, u16::try_from(*slot).ok()?));
        strings.push((rva, bytes));
    }
    // The arrays and strings must not reach into the stub region, and the stub
    // region must fit the module image. Otherwise fall back to empty (sound).
    if cursor > STUB_REGION_RVA {
        return None;
    }
    let stub_end = u64::from(STUB_REGION_RVA)
        + u64::from(function_count) * u64::from(STUB_STRIDE);
    if stub_end > MODULE_IMAGE_SIZE {
        return None;
    }
    let image_end = u32::try_from(stub_end).ok()?;

    let functions: Vec<u32> = (0..function_count)
        .map(|i| {
            if slot_symbols[i as usize].is_some() {
                STUB_REGION_RVA + i * STUB_STRIDE
            } else {
                0
            }
        })
        .collect();

    Some(ExportPlan {
        functions_rva,
        names_rva,
        ordinals_rva,
        function_count,
        name_count,
        image_end,
        functions,
        names: names_table,
        strings,
        slot_symbols,
    })
}

/// Read a NUL-terminated ASCII/ANSI string from guest memory (bounded).
fn read_cstr(mem: &Memory, mut addr: u64) -> Option<String> {
    let mut bytes = Vec::new();
    for _ in 0..4096 {
        let b = mem.read_uint(addr, 1).ok()? as u8;
        if b == 0 {
            return Some(String::from_utf8_lossy(&bytes).into_owned());
        }
        bytes.push(b);
        addr = addr.wrapping_add(1);
    }
    None
}

/// Read a NUL-terminated UTF-16LE string from guest memory (bounded).
fn read_wstr(mem: &Memory, mut addr: u64) -> Option<String> {
    let mut units = Vec::new();
    for _ in 0..4096 {
        let u = mem.read_uint(addr, 2).ok()? as u16;
        if u == 0 {
            return Some(String::from_utf16_lossy(&units));
        }
        units.push(u);
        addr = addr.wrapping_add(2);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Resolve `target` exactly as a protector's direct export walk does: parse
    /// DOS -> NT -> export directory, binary-search `AddressOfNames`, follow the
    /// parallel `AddressOfNameOrdinals` into `AddressOfFunctions`, and compute the
    /// function VA. Returns `(resolved_va, functions_slot_va)`.
    fn walk_export_by_name(mem: &Memory, base: u64, target: &str) -> Option<(u64, u64)> {
        let rd32 = |va: u64| mem.read_uint(va, 4).ok().map(|v| v as u32);
        let rd16 = |va: u64| mem.read_uint(va, 2).ok().map(|v| v as u16);
        assert_eq!(mem.read_uint(base, 2).ok()?, 0x5A4D, "MZ");
        let e_lfanew = rd32(base + 0x3C)?;
        let nt = base + u64::from(e_lfanew);
        assert_eq!(rd32(nt)?, 0x0000_4550, "PE\\0\\0");
        let opt = nt + 24;
        let export_dir = base + u64::from(rd32(opt + 112)?);
        let number_of_names = rd32(export_dir + 0x18)?;
        let functions_rva = rd32(export_dir + 0x1C)?;
        let names_rva = rd32(export_dir + 0x20)?;
        let ordinals_rva = rd32(export_dir + 0x24)?;

        // Binary search the (sorted) name table.
        let name_at = |i: u32| -> Option<String> {
            let name_rva = rd32(base + u64::from(names_rva) + u64::from(i) * 4)?;
            read_cstr(mem, base + u64::from(name_rva))
        };
        let (mut lo, mut hi) = (0i64, i64::from(number_of_names) - 1);
        let mut found: Option<u32> = None;
        while lo <= hi {
            let mid = ((lo + hi) / 2) as u32;
            match name_at(mid)?.as_str().cmp(target) {
                std::cmp::Ordering::Equal => {
                    found = Some(mid);
                    break;
                }
                std::cmp::Ordering::Less => lo = i64::from(mid) + 1,
                std::cmp::Ordering::Greater => hi = i64::from(mid) - 1,
            }
        }
        let name_index = found?;
        let ordinal_index = rd16(base + u64::from(ordinals_rva) + u64::from(name_index) * 2)?;
        let functions_slot_va = base + u64::from(functions_rva) + u64::from(ordinal_index) * 4;
        let func_rva = rd32(functions_slot_va)?;
        Some((base + u64::from(func_rva), functions_slot_va))
    }

    /// A populated fake module is walkable end to end, the walk resolves the
    /// planned deterministic stub VA, and a read of the resolved functions slot
    /// is recorded once (deduped) as an ObservedImport.
    #[test]
    fn populated_export_table_is_walkable_and_recorded() {
        let base = HMODULE_BASE;
        let exports = vec![
            SymbolOrOrdinal::Name("GetTickCount".into()),
            SymbolOrOrdinal::Name("CreateFileW".into()),
            SymbolOrOrdinal::Name("Sleep".into()),
        ];
        let mut mem = Memory::new();
        let layout = map_module_image(&mut mem, base, "kernel32.dll", &exports)
            .expect("module image maps")
            .expect("exports populated");

        let (resolved_va, functions_slot_va) =
            walk_export_by_name(&mem, base, "CreateFileW").expect("name resolves via walk");
        let planned = layout
            .slots
            .iter()
            .flatten()
            .find(|e| matches!(&e.symbol, SymbolOrOrdinal::Name(n) if n == "CreateFileW"))
            .expect("CreateFileW present in layout");
        assert_eq!(resolved_va, planned.va, "walk resolves the planned stub VA");

        // Drive the read-watch directly (as on_mem would during a run).
        let mut hook = WinApiHook::new(BTreeMap::new());
        let facts = hook.facts();
        hook.export_watches.push(ExportWatch {
            module: "kernel32.dll".into(),
            functions_va: layout.functions_va,
            slots: layout.slots,
        });
        hook.note_export_read(functions_slot_va);
        hook.note_export_read(functions_slot_va); // repeated reads dedupe

        let facts = facts.borrow();
        assert_eq!(facts.imports.len(), 1, "one import recorded, deduped");
        assert_eq!(facts.imports[0].module, "kernel32.dll");
        assert_eq!(facts.imports[0].symbol, SymbolOrOrdinal::Name("CreateFileW".into()));
        assert_eq!(facts.imports[0].stub_va_or_slot, resolved_va);
    }

    /// An empty reference keeps the well-formed zero-export directory and
    /// registers no watch (the prior behaviour is unchanged).
    #[test]
    fn empty_reference_keeps_empty_directory() {
        let mut mem = Memory::new();
        let layout =
            map_module_image(&mut mem, HMODULE_BASE, "kernel32.dll", &[]).expect("maps");
        assert!(layout.is_none(), "no export layout for an empty reference");
        // NumberOfFunctions == 0 in the directory.
        let e_lfanew = mem.read_uint(HMODULE_BASE + 0x3C, 4).unwrap();
        let export_dir =
            HMODULE_BASE + mem.read_uint(HMODULE_BASE + e_lfanew + 24 + 112, 4).unwrap();
        assert_eq!(mem.read_uint(export_dir + 0x14, 4).unwrap(), 0);
    }

    /// An ordinal-imported export lands at `functions[ordinal - 1]` so a walk's
    /// `functions[ordinal - Base]` (Base = 1) resolves it.
    #[test]
    fn ordinal_export_lands_at_ordinal_slot() {
        let mut mem = Memory::new();
        let exports = vec![SymbolOrOrdinal::Ordinal(7), SymbolOrOrdinal::Name("A".into())];
        let layout = map_module_image(&mut mem, HMODULE_BASE, "mod.dll", &exports)
            .expect("maps")
            .expect("populated");
        let slot7 = layout.slots.get(6).and_then(|s| s.as_ref()).expect("slot 6 (ordinal 7)");
        assert_eq!(slot7.symbol, SymbolOrOrdinal::Ordinal(7));
        let functions_rva = STUB_REGION_RVA + 6 * STUB_STRIDE;
        assert_eq!(slot7.va, HMODULE_BASE + u64::from(functions_rva));
    }
}
