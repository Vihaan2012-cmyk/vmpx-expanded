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
const HEAP_BASE: u64 = 0x0000_0070_FFF0_0000; // HeapCreate handle pool
const STDHANDLE_BASE: u64 = 0x0000_0070_FFE0_0000; // GetStdHandle sentinels

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
// Fixed monotonic clock readings (ms tick / QPC counter).
const FIXED_TICK: u64 = 0x0100_0000;
const FIXED_QPC: u64 = 0x0000_0001_0000_0000;
// Fixed per-process pointer-encoding cookie (Encode/DecodePointer involution).
const POINTER_COOKIE: u64 = 0x5A5A_5A5A_5A5A_5A5A;
// Deterministic process command line returned by GetCommandLineA/W.
const CMDLINE: &str = "program.exe";

const PAGE: u64 = 0x1000;

// Fixed user-mode address of KUSER_SHARED_DATA (constant on every Windows x64).
const KUSD_BASE: u64 = 0x7FFE_0000;

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
    heap_next: u64,
    /// Heap block base VA -> requested byte size (for HeapSize/HeapReAlloc).
    heap_sizes: BTreeMap<u64, u64>,
    /// Fiber/thread local storage slots: allocated index -> stored pointer. One
    /// index space each; a Get returns exactly what the matching Set stored.
    fls_slots: BTreeMap<u64, u64>,
    fls_next: u64,
    tls_slots: BTreeMap<u64, u64>,
    tls_next: u64,
    /// Cached guest VAs of the ANSI / wide command-line strings (0 = not built).
    cmdline_a: u64,
    cmdline_w: u64,
    /// Cached guest VAs of the ANSI / wide environment blocks (0 = not built).
    env_a: u64,
    env_w: u64,
    /// Thread last-error code (Get/SetLastError).
    last_error: u64,
    /// Whether the fixed `KUSER_SHARED_DATA` page has been mapped yet.
    kusd_ready: bool,
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
            heap_next: HEAP_BASE,
            heap_sizes: BTreeMap::new(),
            fls_slots: BTreeMap::new(),
            fls_next: 1,
            tls_slots: BTreeMap::new(),
            tls_next: 1,
            cmdline_a: 0,
            cmdline_w: 0,
            env_a: 0,
            env_w: 0,
            last_error: 0,
            kusd_ready: false,
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

    /// Map and populate the fixed `KUSER_SHARED_DATA` page with deterministic,
    /// plausible values (Windows 10, no kernel debugger, single-processor). The
    /// page is read-only; only non-zero fields are written (the map is zeroed).
    fn map_kuser_shared_data(&mut self, mem: &mut Memory) -> Result<(), MapError> {
        mem.map(KUSD_BASE, PAGE, Perm::R)?;
        self.kusd_ready = true;
        let mut w = |off: u64, bytes: &[u8]| {
            let _ = mem.poke(KUSD_BASE + off, bytes);
        };
        w(0x004, &0x0FA0_0000u32.to_le_bytes()); // TickCountMultiplier
        // SystemTime / InterruptTime (KSYSTEM_TIME: Low, High1, High2), fixed and
        // consistent with GetSystemTimeAsFileTime's FIXED_FILETIME.
        let (lo, hi) = (FIXED_FILETIME as u32, (FIXED_FILETIME >> 32) as u32);
        for base in [0x008u64, 0x014] {
            w(base, &lo.to_le_bytes());
            w(base + 4, &hi.to_le_bytes());
            w(base + 8, &hi.to_le_bytes());
        }
        // NtSystemRoot (WCHAR[260]) at 0x030.
        let root: Vec<u8> =
            "C:\\Windows".encode_utf16().chain([0]).flat_map(u16::to_le_bytes).collect();
        w(0x030, &root);
        w(0x260, &19041u32.to_le_bytes()); // NtBuildNumber
        w(0x264, &1u32.to_le_bytes()); // NtProductType = NtProductWinNt
        w(0x268, &1u32.to_le_bytes()); // ProductTypeIsValid
        w(0x26C, &10u32.to_le_bytes()); // NtMajorVersion
        w(0x270, &0u32.to_le_bytes()); // NtMinorVersion
        // KdDebuggerEnabled (0x2D4) stays 0: no kernel debugger.
        Ok(())
    }

    /// The deterministic filesystem path reported for a module handle: the
    /// process image for NULL/self, otherwise a System32 path from the resolved
    /// module's name.
    fn module_path(&self, hmod: u64) -> String {
        if hmod == 0 || hmod == self.image_base {
            "C:\\program.exe".to_string()
        } else if let Some(name) = self.handle_names.get(&hmod) {
            format!("C:\\Windows\\System32\\{name}")
        } else {
            "C:\\program.exe".to_string()
        }
    }

    /// Shared body of `GetModuleFileNameA/W`: copy the module path (truncated to
    /// `nsize` chars, always NUL-terminated) and return the char count written.
    fn get_module_filename(
        &self,
        hmod: u64,
        dst: u64,
        nsize: u64,
        wide: bool,
        mem: &mut Memory,
    ) -> Dispatch {
        let nsize = nsize as u32 as usize;
        if dst == 0 || nsize == 0 {
            return Dispatch::Ret(0);
        }
        let path = self.module_path(hmod);
        let elem = if wide { 2 } else { 1 };
        // Characters of the path (ANSI bytes or UTF-16 units), then a NUL.
        let units: Vec<u64> = if wide {
            path.encode_utf16().map(u64::from).collect()
        } else {
            path.bytes().map(u64::from).collect()
        };
        // Copy at most nsize-1 chars, then a NUL terminator (Win10 semantics).
        let copy = units.len().min(nsize - 1);
        for (i, &u) in units.iter().take(copy).enumerate() {
            if mem.write_uint(dst + (i * elem) as u64, elem, u).is_err() {
                return Dispatch::Stop;
            }
        }
        if mem.write_uint(dst + (copy * elem) as u64, elem, 0).is_err() {
            return Dispatch::Stop;
        }
        Dispatch::Ret(copy as u64)
    }

    /// Bump-allocate a buffer and write `bytes` into it; returns the base VA, or
    /// `None` if the map/write fails.
    fn write_buf(&mut self, mem: &mut Memory, bytes: &[u8]) -> Option<u64> {
        let va = self.alloc(mem, bytes.len() as u64);
        if va == 0 || mem.poke(va, bytes).is_err() {
            return None;
        }
        Some(va)
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

    /// Read 0-based argument `i`: the first four from registers, `i >= 4` from
    /// the guest stack starting at `sab` (the 5th argument's slot).
    fn stack_arg(cpu: &Cpu, mem: &Memory, sab: u64, i: usize) -> Option<u64> {
        if i < 4 {
            Some(cpu.regs[ARG[i]])
        } else {
            mem.read_uint(sab + (i as u64 - 4) * 8, 8).ok()
        }
    }

    fn dispatch(&mut self, dll: &str, func: &str, cpu: &Cpu, mem: &mut Memory, sab: u64) -> Dispatch {
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
                let va = self.alloc(mem, size);
                if va != 0 {
                    self.heap_sizes.insert(va, size);
                }
                Dispatch::Ret(va)
            }
            // HeapReAlloc(hHeap, dwFlags, lpMem, dwBytes): a fresh block with the
            // old contents copied in (the bump allocator never grows in place).
            (_, "HeapReAlloc") => {
                let old_ptr = Self::arg(cpu, 2);
                let size = Self::arg(cpu, 3);
                if old_ptr == 0 {
                    let va = self.alloc(mem, size);
                    if va != 0 {
                        self.heap_sizes.insert(va, size);
                    }
                    return Dispatch::Ret(va);
                }
                let Some(&old_size) = self.heap_sizes.get(&old_ptr) else {
                    return Dispatch::Stop; // unknown block: cannot copy soundly
                };
                let va = self.alloc(mem, size);
                if va == 0 {
                    return Dispatch::Ret(0);
                }
                let copy = old_size.min(size);
                for i in 0..copy {
                    let Ok(b) = mem.read_uint(old_ptr + i, 1) else {
                        return Dispatch::Stop;
                    };
                    if mem.write_uint(va + i, 1, b).is_err() {
                        return Dispatch::Stop;
                    }
                }
                self.heap_sizes.insert(va, size);
                Dispatch::Ret(va)
            }
            // HeapSize(hHeap, dwFlags, lpMem) -> SIZE_T: the recorded block size,
            // or (SIZE_T)-1 for a block we did not allocate.
            (_, "HeapSize") => {
                let p = Self::arg(cpu, 2);
                Dispatch::Ret(self.heap_sizes.get(&p).copied().unwrap_or(u64::MAX))
            }
            // VirtualProtect(lpAddress, dwSize, flNewProtect, lpflOldProtect):
            // honor the stub's own reprotect. A packer (e.g. UPX) flips the
            // R-only image header page to RW to patch a few header bytes, then
            // restores it; emulating this as a no-op left the write faulting.
            // We apply exactly the requested R/W/X to the covered pages and
            // report the real prior protection. An out-of-image protect over
            // unmapped pages still fails (FALSE) — nothing is silently widened.
            (_, "VirtualProtect") => {
                let addr = Self::arg(cpu, 0);
                let size = Self::arg(cpu, 1);
                let new_prot = Self::arg(cpu, 2);
                let old_ptr = Self::arg(cpu, 3);
                let old = mem.perm_at(addr).map_or(PAGE_READONLY, perm_to_page);
                if old_ptr != 0 && mem.write_uint(old_ptr, 4, u64::from(old)).is_err() {
                    return Dispatch::Stop;
                }
                match mem.protect(addr, size, page_to_perm(new_prot)) {
                    Ok(()) => Dispatch::Ret(1),
                    Err(_) => Dispatch::Ret(0),
                }
            }
            // Frees are a no-op in the bump model; report success.
            (_, "VirtualFree" | "HeapFree") => Dispatch::Ret(1),
            (_, "GetProcessHeap") => Dispatch::Ret(PROCESS_HEAP),
            // HeapCreate(flOptions, dwInitialSize, dwMaximumSize) -> HANDLE: mint
            // a distinct non-null heap sentinel (HeapAlloc ignores the heap arg
            // and bump-allocates real guest pages regardless).
            (_, "HeapCreate") => {
                let h = self.heap_next;
                self.heap_next += PAGE;
                Dispatch::Ret(h)
            }
            (_, "HeapDestroy") => Dispatch::Ret(1),
            // HeapSetInformation(...) -> BOOL: policy hint (e.g. enable LFH); a
            // no-op success that does not change the bump allocator's behaviour.
            (_, "HeapSetInformation") => Dispatch::Ret(1),

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
            // GetProcessAffinityMask(h, *proc, *sys) -> BOOL: a single-processor
            // system (both masks = 1).
            (_, "GetProcessAffinityMask") => {
                let p_proc = Self::arg(cpu, 1);
                let p_sys = Self::arg(cpu, 2);
                if (p_proc != 0 && mem.write_uint(p_proc, 8, 1).is_err())
                    || (p_sys != 0 && mem.write_uint(p_sys, 8, 1).is_err())
                {
                    return Dispatch::Stop;
                }
                Dispatch::Ret(1)
            }

            // SetThreadAffinityMask(h, mask) -> previous mask (non-zero success).
            (_, "SetThreadAffinityMask") => Dispatch::Ret(1),
            // Scheduling yields are no-ops in deterministic single-threaded time.
            (_, "Sleep") => Dispatch::Void,
            (_, "SleepEx" | "SwitchToThread") => Dispatch::Ret(0),

            // ---- Process / thread identity ----------------------------------
            // Pseudo-handles (current process = -1, current thread = -2) and the
            // fixed in-model ids, kept consistent with the NT info queries.
            (_, "GetCurrentProcess") => Dispatch::Ret(0xFFFF_FFFF_FFFF_FFFF),
            (_, "GetCurrentThread") => Dispatch::Ret(0xFFFF_FFFF_FFFF_FFFE),
            (_, "GetCurrentProcessId") => Dispatch::Ret(crate::ntdll::FAKE_PID),
            (_, "GetCurrentThreadId") => Dispatch::Ret(crate::ntdll::FAKE_TID),

            // Fixed monotonic clocks: a constant reading makes any elapsed-time
            // anti-debug delta zero (the no-debugger-present path). Deterministic.
            (_, "GetTickCount") => Dispatch::Ret(FIXED_TICK),
            (_, "GetTickCount64") => Dispatch::Ret(FIXED_TICK),
            // QueryPerformanceCounter(*lpPerformanceCount) -> BOOL; fixed counter.
            (_, "QueryPerformanceCounter") => {
                let p = Self::arg(cpu, 0);
                if p != 0 && mem.write_uint(p, 8, FIXED_QPC).is_err() {
                    return Dispatch::Stop;
                }
                Dispatch::Ret(1)
            }
            // QueryPerformanceFrequency(*lpFrequency) -> BOOL; fixed 10 MHz.
            (_, "QueryPerformanceFrequency") => {
                let p = Self::arg(cpu, 0);
                if p != 0 && mem.write_uint(p, 8, 10_000_000).is_err() {
                    return Dispatch::Stop;
                }
                Dispatch::Ret(1)
            }

            // ---- Code-page conversion ---------------------------------------
            // WideCharToMultiByte(cp, flags, lpWide, cchWide, lpMB, cbMB,
            //   lpDefaultChar, lpUsedDefaultChar). UTF-8 (65001) converts exactly;
            // any other code page is treated as Latin-1 (identity for 0x00-0xFF,
            // default char '?' above) — deterministic and faithful for the ASCII
            // the CRT startup converts. cch<0 = NUL-terminated (terminator and
            // its count included); cb==0 = return the required byte count.
            (_, "WideCharToMultiByte") => {
                let cp = Self::arg(cpu, 0);
                let src = Self::arg(cpu, 2);
                let cch = Self::arg(cpu, 3) as u32 as i32;
                let (Some(dst), Some(cb), Some(p_used)) = (
                    Self::stack_arg(cpu, mem, sab, 4),
                    Self::stack_arg(cpu, mem, sab, 5),
                    Self::stack_arg(cpu, mem, sab, 7),
                ) else {
                    return Dispatch::Stop;
                };
                let cb = u64::from(cb as u32); // cbMultiByte is a 32-bit int
                let Some(units) = read_wide_units(mem, src, cch) else {
                    return Dispatch::Stop;
                };
                let (bytes, used_default) = wide_to_mb(cp, &units);
                let need = bytes.len() as u64;
                if cb != 0 {
                    if need > cb {
                        return Dispatch::Ret(0); // insufficient buffer
                    }
                    if mem.write(dst, &bytes).is_err() {
                        return Dispatch::Stop;
                    }
                }
                if p_used != 0 && mem.write_uint(p_used, 4, u64::from(used_default)).is_err() {
                    return Dispatch::Stop;
                }
                Dispatch::Ret(need)
            }
            // MultiByteToWideChar(cp, flags, lpMB, cbMB, lpWide, cchWide).
            (_, "MultiByteToWideChar") => {
                let cp = Self::arg(cpu, 0);
                let src = Self::arg(cpu, 2);
                let cb = Self::arg(cpu, 3) as u32 as i32;
                let (Some(dst), Some(cch)) =
                    (Self::stack_arg(cpu, mem, sab, 4), Self::stack_arg(cpu, mem, sab, 5))
                else {
                    return Dispatch::Stop;
                };
                let cch = u64::from(cch as u32); // cchWideChar is a 32-bit int
                let Some(bytes) = read_mb_bytes(mem, src, cb) else {
                    return Dispatch::Stop;
                };
                let units = mb_to_wide(cp, &bytes);
                let need = units.len() as u64;
                if cch != 0 {
                    if need > cch {
                        return Dispatch::Ret(0);
                    }
                    for (i, u) in units.iter().enumerate() {
                        if mem.write_uint(dst + (i as u64) * 2, 2, u64::from(*u)).is_err() {
                            return Dispatch::Stop;
                        }
                    }
                }
                Dispatch::Ret(need)
            }

            // GetStringTypeW(dwInfoType, lpSrcStr, cchSrc, lpCharType) -> BOOL:
            // one WORD of character-type flags per source char. Only CT_CTYPE1
            // (type 1) is modeled; other info types fail closed. cchSrc<0 =
            // NUL-terminated (terminator not classified).
            (_, "GetStringTypeW") => {
                if Self::arg(cpu, 0) != 1 {
                    return Dispatch::Stop; // only CT_CTYPE1 modeled
                }
                let src = Self::arg(cpu, 1);
                let cch = Self::arg(cpu, 2) as u32 as i32;
                let p_out = Self::arg(cpu, 3);
                let Some(mut units) = read_wide_units(mem, src, cch) else {
                    return Dispatch::Stop;
                };
                if cch < 0 {
                    units.pop(); // drop the terminator
                }
                for (i, u) in units.iter().enumerate() {
                    if mem.write_uint(p_out + (i as u64) * 2, 2, u64::from(ctype1(*u))).is_err() {
                        return Dispatch::Stop;
                    }
                }
                Dispatch::Ret(1)
            }

            // LCMapStringW(Locale, dwMapFlags, lpSrc, cchSrc, lpDest, cchDest).
            // Only simple case mapping (LCMAP_UPPERCASE/LOWERCASE) is modeled;
            // sort keys and other locale transforms fail closed. cchSrc<0 =
            // NUL-terminated; cchDest==0 = return the required length.
            (_, "LCMapStringW") => {
                const LCMAP_LOWERCASE: u64 = 0x0100;
                const LCMAP_UPPERCASE: u64 = 0x0200;
                let flags = Self::arg(cpu, 1);
                let src = Self::arg(cpu, 2);
                let cch = Self::arg(cpu, 3) as u32 as i32;
                let (Some(dst), Some(cch_dest)) =
                    (Self::stack_arg(cpu, mem, sab, 4), Self::stack_arg(cpu, mem, sab, 5))
                else {
                    return Dispatch::Stop;
                };
                let cch_dest = u64::from(cch_dest as u32);
                let upper = flags & LCMAP_UPPERCASE != 0;
                if !upper && flags & LCMAP_LOWERCASE == 0 {
                    return Dispatch::Stop; // only case mapping modeled
                }
                let Some(units) = read_wide_units(mem, src, cch) else {
                    return Dispatch::Stop;
                };
                let mapped: Vec<u16> = units.iter().map(|&u| map_case(u, upper)).collect();
                let need = mapped.len() as u64;
                if cch_dest != 0 {
                    if need > cch_dest {
                        return Dispatch::Ret(0);
                    }
                    for (i, u) in mapped.iter().enumerate() {
                        if mem.write_uint(dst + (i as u64) * 2, 2, u64::from(*u)).is_err() {
                            return Dispatch::Stop;
                        }
                    }
                }
                Dispatch::Ret(need)
            }

            // EncodePointer/DecodePointer: a mutually-inverse transform over a
            // per-process cookie. We own both sides, so a fixed-cookie XOR is a
            // sound, deterministic involution (Decode(Encode(p)) == p); the exact
            // encoded bits are never observed outside this pair.
            (_, "EncodePointer" | "DecodePointer") => {
                Dispatch::Ret(Self::arg(cpu, 0) ^ POINTER_COOKIE)
            }

            // ---- Process environment ----------------------------------------
            // A deterministic command line, materialized once into guest memory;
            // the same pointer is returned on repeat calls. Fail closed if the
            // backing buffer cannot be mapped.
            (_, "GetCommandLineA") => {
                if self.cmdline_a == 0 {
                    let mut bytes = CMDLINE.as_bytes().to_vec();
                    bytes.push(0);
                    match self.write_buf(mem, &bytes) {
                        Some(va) => self.cmdline_a = va,
                        None => return Dispatch::Stop,
                    }
                }
                Dispatch::Ret(self.cmdline_a)
            }
            (_, "GetCommandLineW") => {
                if self.cmdline_w == 0 {
                    let mut bytes: Vec<u8> =
                        CMDLINE.encode_utf16().flat_map(u16::to_le_bytes).collect();
                    bytes.extend_from_slice(&[0, 0]);
                    match self.write_buf(mem, &bytes) {
                        Some(va) => self.cmdline_w = va,
                        None => return Dispatch::Stop,
                    }
                }
                Dispatch::Ret(self.cmdline_w)
            }

            // An empty (double-NUL-terminated) environment block: a legal,
            // deterministic environment with no variables. Freed as a no-op.
            (_, "GetEnvironmentStringsW") => {
                if self.env_w == 0 {
                    match self.write_buf(mem, &[0, 0, 0, 0]) {
                        Some(va) => self.env_w = va,
                        None => return Dispatch::Stop,
                    }
                }
                Dispatch::Ret(self.env_w)
            }
            (_, "GetEnvironmentStrings" | "GetEnvironmentStringsA") => {
                if self.env_a == 0 {
                    match self.write_buf(mem, &[0, 0]) {
                        Some(va) => self.env_a = va,
                        None => return Dispatch::Stop,
                    }
                }
                Dispatch::Ret(self.env_a)
            }
            (_, "FreeEnvironmentStringsW" | "FreeEnvironmentStringsA") => Dispatch::Ret(1),
            // GetModuleFileNameA/W(hModule, lpFilename, nSize) -> DWORD: a
            // deterministic module path (the process image for NULL/self, a
            // System32 path for a resolved module). Truncates to nSize (the copy
            // is always NUL-terminated); returns the char count written.
            (_, "GetModuleFileNameA") => {
                self.get_module_filename(Self::arg(cpu, 0), Self::arg(cpu, 1), Self::arg(cpu, 2), false, mem)
            }
            (_, "GetModuleFileNameW") => {
                self.get_module_filename(Self::arg(cpu, 0), Self::arg(cpu, 1), Self::arg(cpu, 2), true, mem)
            }
            // GetStdHandle(nStdHandle) -> HANDLE: deterministic per-stream
            // sentinel (-10/-11/-12 => in/out/err); any other id is invalid.
            (_, "GetStdHandle") => {
                let h = match Self::arg(cpu, 0) as u32 as i32 {
                    -10 => STDHANDLE_BASE,
                    -11 => STDHANDLE_BASE + 4,
                    -12 => STDHANDLE_BASE + 8,
                    _ => 0xFFFF_FFFF_FFFF_FFFF, // INVALID_HANDLE_VALUE
                };
                Dispatch::Ret(h)
            }
            // Fixed locale code pages: ANSI = Windows-1252, OEM = 437; every
            // code page the sample asks about is reported valid.
            (_, "GetACP") => Dispatch::Ret(1252),
            (_, "GetOEMCP") => Dispatch::Ret(437),
            (_, "IsValidCodePage") => Dispatch::Ret(1),
            // GetCPInfo(CodePage, lpCPInfo) -> BOOL. CPINFO is 18 bytes:
            // MaxCharSize (u32), DefaultChar[2], LeadByte[12]. Our modeled code
            // pages are single-byte (UTF-8 reports MaxCharSize 4); no DBCS lead
            // bytes, default char '?'.
            (_, "GetCPInfo") => {
                let cp = Self::arg(cpu, 0);
                let p = Self::arg(cpu, 1);
                let mut info = [0u8; 18];
                let max = if cp == CP_UTF8 { 4u32 } else { 1 };
                info[0..4].copy_from_slice(&max.to_le_bytes());
                info[4] = b'?'; // DefaultChar[0]
                if p != 0 && mem.write(p, &info).is_err() {
                    return Dispatch::Stop;
                }
                Dispatch::Ret(1)
            }

            // Thread last-error, stored in-model (Get returns what Set stored).
            (_, "GetLastError") => Dispatch::Ret(self.last_error),
            (_, "SetLastError") => {
                self.last_error = Self::arg(cpu, 0) & 0xFFFF_FFFF;
                Dispatch::Void
            }
            // SetHandleCount(uNumber) -> UINT: a legacy no-op that returns its
            // argument unchanged.
            (_, "SetHandleCount") => Dispatch::Ret(Self::arg(cpu, 0)),
            // GetFileType(hFile) -> DWORD: the std sentinels are character
            // devices (FILE_TYPE_CHAR = 2); anything else is FILE_TYPE_UNKNOWN.
            (_, "GetFileType") => {
                let h = Self::arg(cpu, 0);
                let ty = if (STDHANDLE_BASE..STDHANDLE_BASE + 0x10).contains(&h) { 2 } else { 0 };
                Dispatch::Ret(ty)
            }
            // GetStartupInfoA/W(lpStartupInfo) -> void: a fresh process's startup
            // info is all-zero (dwFlags clear => the CRT uses GetStdHandle, no
            // inherited handle table in lpReserved2). STARTUPINFO is 104 bytes on
            // x64 (same size for A/W — only the string-pointer type differs).
            (_, "GetStartupInfoA" | "GetStartupInfoW") => {
                let p = Self::arg(cpu, 0);
                if p != 0 && mem.write(p, &[0u8; 104]).is_err() {
                    return Dispatch::Stop;
                }
                Dispatch::Void
            }

            // ---- Fiber / thread local storage -------------------------------
            // Backed by real slot maps so Get returns what Set stored; indices
            // are allocated monotonically (1-based). FLS callbacks are ignored
            // (no fiber teardown in this run).
            (_, "FlsAlloc") => {
                let idx = self.fls_next;
                self.fls_next += 1;
                self.fls_slots.insert(idx, 0);
                Dispatch::Ret(idx)
            }
            (_, "FlsSetValue") => {
                self.fls_slots.insert(Self::arg(cpu, 0), Self::arg(cpu, 1));
                Dispatch::Ret(1)
            }
            (_, "FlsGetValue") => {
                Dispatch::Ret(self.fls_slots.get(&Self::arg(cpu, 0)).copied().unwrap_or(0))
            }
            (_, "FlsFree") => {
                self.fls_slots.remove(&Self::arg(cpu, 0));
                Dispatch::Ret(1)
            }
            (_, "TlsAlloc") => {
                let idx = self.tls_next;
                self.tls_next += 1;
                self.tls_slots.insert(idx, 0);
                Dispatch::Ret(idx)
            }
            (_, "TlsSetValue") => {
                self.tls_slots.insert(Self::arg(cpu, 0), Self::arg(cpu, 1));
                Dispatch::Ret(1)
            }
            (_, "TlsGetValue") => {
                Dispatch::Ret(self.tls_slots.get(&Self::arg(cpu, 0)).copied().unwrap_or(0))
            }
            (_, "TlsFree") => {
                self.tls_slots.remove(&Self::arg(cpu, 0));
                Dispatch::Ret(1)
            }

            // ---- Critical sections ------------------------------------------
            // Single-threaded, deterministic run: every lock is uncontended, so
            // the whole family is a faithful no-op. Init variants return BOOL ok;
            // enter/leave/delete are void; try-enter always acquires.
            (_, "InitializeCriticalSectionAndSpinCount" | "InitializeCriticalSectionEx"
                | "TryEnterCriticalSection") => Dispatch::Ret(1),
            (_, "InitializeCriticalSection" | "EnterCriticalSection" | "LeaveCriticalSection"
                | "DeleteCriticalSection") => Dispatch::Void,

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
    fn on_mem(&mut self, ev: &MemEvent, _cpu: &mut Cpu, mem: &mut Memory) -> HookAction {
        // The fixed KUSER_SHARED_DATA page is provided by the OS at a constant
        // address; map it in-model on first touch (on_mem runs before the read,
        // so the faulting access then succeeds).
        if !self.kusd_ready
            && (KUSD_BASE..KUSD_BASE + PAGE).contains(&ev.addr)
            && self.map_kuser_shared_data(mem).is_err()
        {
            return HookAction::Stop;
        }
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

/// Whether a resolved import is a networking API that the contained, air-gapped
/// emulator must refuse outright (belt-and-suspenders over the no-FFI design).
///
/// Matched by MODULE (the sockets/HTTP/DNS/TLS DLLs) or by FUNCTION-name pattern
/// (the Winsock, WinINet, WinHTTP, DNS and URL-download surfaces), so a network
/// call is un-serviceable regardless of which DLL happens to export it.
fn is_network_api(module: &str, func: &str) -> bool {
    const NET_MODULES: &[&str] = &[
        "ws2_32.dll",
        "wsock32.dll",
        "mswsock.dll",
        "wininet.dll",
        "winhttp.dll",
        "dnsapi.dll",
        "iphlpapi.dll",
        "urlmon.dll",
        "ncrypt.dll",
        "schannel.dll",
        "secur32.dll",
        "sspicli.dll",
        "wsdapi.dll",
    ];
    if NET_MODULES.contains(&module.to_lowercase().as_str()) {
        return true;
    }
    // Function-name prefixes covering the Winsock/HTTP/DNS/URL surfaces.
    const NET_PREFIXES: &[&str] =
        &["WSA", "Internet", "WinHttp", "Dns", "URLDownload", "HttpOpen", "HttpSend", "gethostby"];
    if NET_PREFIXES.iter().any(|p| func.starts_with(p)) {
        return true;
    }
    // Exact Winsock verbs (plain BSD-socket names that carry no distinctive prefix).
    const NET_EXACT: &[&str] = &[
        "socket",
        "connect",
        "send",
        "sendto",
        "recv",
        "recvfrom",
        "bind",
        "listen",
        "accept",
        "closesocket",
        "shutdown",
        "getaddrinfo",
        "GetAddrInfoW",
        "freeaddrinfo",
        "getnameinfo",
        "select",
        "ioctlsocket",
    ];
    NET_EXACT.contains(&func)
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
        // CONTAINMENT: a hard, un-bypassable network denylist, checked BEFORE any
        // API modeling. These are untrusted real-world samples; no networking API
        // is ever serviceable. A resolved network import stops the run (reason:
        // "network access blocked: <module>!<func> — contained, air-gapped").
        // This only guards; nothing here can reach the host (no FFI/syscall path).
        if is_network_api(dll, func) {
            return HookAction::Stop;
        }
        let sab = Self::stack_arg_base(ev, cpu);
        match self.dispatch(dll, func, cpu, mem, sab) {
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

// Windows memory-protection constant we report as a page's prior protection.
const PAGE_READONLY: u32 = 0x02;

/// Map a Windows `PAGE_*` protection constant to a guest [`Perm`]. The low byte
/// selects the base protection; GUARD/NOCACHE/WRITECOMBINE modifiers are
/// ignored. Unknown/`PAGE_NOACCESS` map to no access.
fn page_to_perm(flags: u64) -> Perm {
    match flags & 0xff {
        0x02 => Perm::R,          // PAGE_READONLY
        0x04 | 0x08 => Perm::RW,  // PAGE_READWRITE / PAGE_WRITECOPY
        0x10 => Perm::X,          // PAGE_EXECUTE
        0x20 => Perm::RX,         // PAGE_EXECUTE_READ
        0x40 | 0x80 => Perm::RWX, // PAGE_EXECUTE_READWRITE / _WRITECOPY
        _ => Perm(0),             // PAGE_NOACCESS (0x01) / unknown
    }
}

/// Map a guest [`Perm`] back to the nearest Windows `PAGE_*` constant, used to
/// report `lpflOldProtect`.
fn perm_to_page(perm: Perm) -> u32 {
    match perm.0 & 7 {
        1 => 0x02,         // R   -> PAGE_READONLY
        2 | 3 => 0x04,     // W/RW -> PAGE_READWRITE
        4 => 0x10,         // X   -> PAGE_EXECUTE
        5 => 0x20,         // RX  -> PAGE_EXECUTE_READ
        6 | 7 => 0x40,     // WX/RWX -> PAGE_EXECUTE_READWRITE
        _ => 0x01,         // none -> PAGE_NOACCESS
    }
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

/// Windows UTF-8 code page.
const CP_UTF8: u64 = 65001;
/// Bound on a single conversion's element count (guards a bogus length).
const CONV_MAX: usize = 0x10_0000;

/// Read the UTF-16 input for a code-page conversion. `cch < 0` means the source
/// is NUL-terminated and the terminating `0x0000` is included in the result
/// (so its byte/char count is counted); otherwise read exactly `cch` units.
fn read_wide_units(mem: &Memory, addr: u64, cch: i32) -> Option<Vec<u16>> {
    let mut units = Vec::new();
    if cch < 0 {
        let mut a = addr;
        for _ in 0..CONV_MAX {
            let u = mem.read_uint(a, 2).ok()? as u16;
            units.push(u);
            if u == 0 {
                return Some(units);
            }
            a = a.wrapping_add(2);
        }
        None
    } else {
        let n = cch as usize;
        if n > CONV_MAX {
            return None;
        }
        for i in 0..n {
            units.push(mem.read_uint(addr + (i as u64) * 2, 2).ok()? as u16);
        }
        Some(units)
    }
}

/// Read the multi-byte input for a code-page conversion (`cb < 0` = NUL-terminated,
/// terminator included).
fn read_mb_bytes(mem: &Memory, addr: u64, cb: i32) -> Option<Vec<u8>> {
    let mut bytes = Vec::new();
    if cb < 0 {
        let mut a = addr;
        for _ in 0..CONV_MAX {
            let b = mem.read_uint(a, 1).ok()? as u8;
            bytes.push(b);
            if b == 0 {
                return Some(bytes);
            }
            a = a.wrapping_add(1);
        }
        None
    } else {
        let n = cb as usize;
        if n > CONV_MAX {
            return None;
        }
        for i in 0..n {
            bytes.push(mem.read_uint(addr + i as u64, 1).ok()? as u8);
        }
        Some(bytes)
    }
}

/// UTF-16 -> multi-byte. UTF-8 converts exactly (lossy-substituting unpaired
/// surrogates); any other code page is Latin-1 (identity 0x00-0xFF, `?` above,
/// flagging the default-char substitution).
fn wide_to_mb(code_page: u64, units: &[u16]) -> (Vec<u8>, bool) {
    if code_page == CP_UTF8 {
        (String::from_utf16_lossy(units).into_bytes(), false)
    } else {
        let mut out = Vec::with_capacity(units.len());
        let mut used_default = false;
        for &u in units {
            if u <= 0xFF {
                out.push(u as u8);
            } else {
                out.push(b'?');
                used_default = true;
            }
        }
        (out, used_default)
    }
}

/// Multi-byte -> UTF-16, the inverse of [`wide_to_mb`].
fn mb_to_wide(code_page: u64, bytes: &[u8]) -> Vec<u16> {
    if code_page == CP_UTF8 {
        String::from_utf8_lossy(bytes).encode_utf16().collect()
    } else {
        bytes.iter().map(|&b| u16::from(b)).collect()
    }
}

/// Simple (1:1) case mapping of one UTF-16 code unit, as `LCMapStringW` with
/// `LCMAP_UPPERCASE`/`LCMAP_LOWERCASE` performs for the common case. A unit whose Unicode
/// case expansion is not a single BMP scalar is left unchanged (sound: no
/// length change, exact for ASCII/Latin).
fn map_case(u: u16, upper: bool) -> u16 {
    let Some(ch) = char::from_u32(u32::from(u)) else {
        return u;
    };
    let mapped: Vec<char> =
        if upper { ch.to_uppercase().collect() } else { ch.to_lowercase().collect() };
    match mapped.as_slice() {
        [c] if (*c as u32) <= 0xFFFF => *c as u16,
        _ => u,
    }
}

/// `CT_CTYPE1` character-type flags for one UTF-16 code unit (`GetStringTypeW`).
/// The ASCII range is classified exactly; other BMP scalars use Rust's Unicode
/// properties (deterministic best effort).
fn ctype1(u: u16) -> u16 {
    const UPPER: u16 = 0x0001;
    const LOWER: u16 = 0x0002;
    const DIGIT: u16 = 0x0004;
    const SPACE: u16 = 0x0008;
    const PUNCT: u16 = 0x0010;
    const CNTRL: u16 = 0x0020;
    const BLANK: u16 = 0x0040;
    const XDIGIT: u16 = 0x0080;
    const ALPHA: u16 = 0x0100;
    let Some(ch) = char::from_u32(u32::from(u)) else {
        return 0;
    };
    let mut t = 0u16;
    if ch.is_ascii() {
        let b = u as u8;
        if ch.is_ascii_uppercase() {
            t |= UPPER | ALPHA;
        }
        if ch.is_ascii_lowercase() {
            t |= LOWER | ALPHA;
        }
        if ch.is_ascii_digit() {
            t |= DIGIT;
        }
        if ch.is_ascii_hexdigit() {
            t |= XDIGIT;
        }
        if matches!(b, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r') {
            t |= SPACE;
        }
        if matches!(b, b' ' | b'\t') {
            t |= BLANK;
        }
        if ch.is_ascii_punctuation() {
            t |= PUNCT;
        }
        if ch.is_ascii_control() {
            t |= CNTRL;
        }
    } else {
        if ch.is_alphabetic() {
            t |= ALPHA;
            if ch.is_uppercase() {
                t |= UPPER;
            }
            if ch.is_lowercase() {
                t |= LOWER;
            }
        }
        if ch.is_numeric() {
            t |= DIGIT;
        }
        if ch.is_whitespace() {
            t |= SPACE;
        }
        if ch.is_control() {
            t |= CNTRL;
        }
    }
    t
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

    /// A guest `VirtualProtect` flipping a mapped read-only page to RW is
    /// honored: the subsequent write succeeds and the real prior protection is
    /// reported, then a restoring call makes it read-only again. This is the
    /// real-UPX case — the stub reprotects its own image-header page to patch a
    /// byte, then restores it.
    #[test]
    fn virtual_protect_reprotects_a_mapped_page() {
        const PAGE: u64 = 0x1_4000_0000;
        let mut mem = Memory::new();
        mem.map(PAGE, 0x1000, Perm::R).unwrap();
        mem.map(PAGE + 0x1000, 0x1000, Perm::RW).unwrap(); // lpflOldProtect lives here
        let old_ptr = PAGE + 0x1000;
        // A write to the R-only page faults before the reprotect.
        assert!(mem.write_uint(PAGE + 0x10, 1, 0xAB).is_err());

        let mut hook = WinApiHook::new(BTreeMap::new());
        let mut cpu = Cpu::new(0);
        cpu.regs[ARG[0]] = PAGE; // lpAddress
        cpu.regs[ARG[1]] = 0x1000; // dwSize
        cpu.regs[ARG[2]] = 0x04; // flNewProtect = PAGE_READWRITE
        cpu.regs[ARG[3]] = old_ptr;

        let d = hook.dispatch("kernel32.dll", "VirtualProtect", &cpu, &mut mem, 0);
        assert!(matches!(d, Dispatch::Ret(1)), "reports success");
        assert_eq!(mem.read_uint(old_ptr, 4).unwrap(), 0x02, "old protection = PAGE_READONLY");
        // The page is now writable: the patch the stub would make succeeds.
        assert!(mem.write_uint(PAGE + 0x10, 1, 0xAB).is_ok());

        // Restore the old protection: the page is read-only again.
        cpu.regs[ARG[2]] = 0x02;
        let d = hook.dispatch("kernel32.dll", "VirtualProtect", &cpu, &mut mem, 0);
        assert!(matches!(d, Dispatch::Ret(1)));
        assert!(mem.write_uint(PAGE + 0x10, 1, 0xCD).is_err(), "restored to read-only");
    }

    /// `VirtualProtect` over an unmapped range fails (FALSE) rather than
    /// silently mapping memory: a wild protect never widens the model.
    #[test]
    fn virtual_protect_over_unmapped_fails() {
        let mut mem = Memory::new();
        let mut hook = WinApiHook::new(BTreeMap::new());
        let mut cpu = Cpu::new(0);
        cpu.regs[ARG[0]] = 0x1_4000_0000; // unmapped
        cpu.regs[ARG[1]] = 0x1000;
        cpu.regs[ARG[2]] = 0x40;
        cpu.regs[ARG[3]] = 0;
        let d = hook.dispatch("kernel32.dll", "VirtualProtect", &cpu, &mut mem, 0);
        assert!(matches!(d, Dispatch::Ret(0)), "fails closed");
        assert!(mem.perm_at(0x1_4000_0000).is_none(), "no page was created");
    }

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

    const T_STACK: u64 = 0x0000_0010_0000_0000;
    const T_DATA: u64 = 0x0000_0020_0000_0000;

    fn harness() -> (WinApiHook, Cpu, Memory, u64) {
        let mut mem = Memory::new();
        mem.map(T_STACK, 0x8000, Perm::RW).unwrap();
        mem.map(T_DATA, 0x8000, Perm::RW).unwrap();
        let mut cpu = Cpu::new(0);
        cpu.regs[RSP] = T_STACK + 0x4000;
        let sab = cpu.regs[RSP] + 0x20; // Call convention
        (WinApiHook::new(BTreeMap::new()), cpu, mem, sab)
    }

    fn set_args(cpu: &mut Cpu, args: [u64; 4]) {
        for (i, &v) in args.iter().enumerate() {
            cpu.regs[ARG[i]] = v;
        }
    }

    fn ret_of(d: Dispatch) -> u64 {
        match d {
            Dispatch::Ret(v) => v,
            Dispatch::Void => panic!("expected Ret, got Void"),
            Dispatch::Stop => panic!("expected Ret, got Stop"),
        }
    }

    #[test]
    fn current_ids_match_nt_info_values() {
        let (mut h, cpu, mut mem, s) = harness();
        assert_eq!(
            ret_of(h.dispatch("kernel32.dll", "GetCurrentProcessId", &cpu, &mut mem, s)),
            crate::ntdll::FAKE_PID
        );
        assert_eq!(
            ret_of(h.dispatch("kernel32.dll", "GetCurrentThreadId", &cpu, &mut mem, s)),
            crate::ntdll::FAKE_TID
        );
        assert_eq!(
            ret_of(h.dispatch("kernel32.dll", "GetCurrentProcess", &cpu, &mut mem, s)),
            0xFFFF_FFFF_FFFF_FFFF
        );
    }

    #[test]
    fn encode_decode_pointer_is_an_involution() {
        let (mut h, mut cpu, mut mem, s) = harness();
        let p = 0x1234_5678_9abc_def0;
        set_args(&mut cpu, [p, 0, 0, 0]);
        let enc = ret_of(h.dispatch("kernel32.dll", "EncodePointer", &cpu, &mut mem, s));
        assert_ne!(enc, p);
        set_args(&mut cpu, [enc, 0, 0, 0]);
        let dec = ret_of(h.dispatch("kernel32.dll", "DecodePointer", &cpu, &mut mem, s));
        assert_eq!(dec, p, "Decode(Encode(p)) == p");
    }

    #[test]
    fn fls_tls_get_returns_set() {
        let (mut h, mut cpu, mut mem, s) = harness();
        let idx = ret_of(h.dispatch("kernel32.dll", "FlsAlloc", &cpu, &mut mem, s));
        set_args(&mut cpu, [idx, 0xCAFE, 0, 0]);
        assert_eq!(ret_of(h.dispatch("kernel32.dll", "FlsSetValue", &cpu, &mut mem, s)), 1);
        set_args(&mut cpu, [idx, 0, 0, 0]);
        assert_eq!(ret_of(h.dispatch("kernel32.dll", "FlsGetValue", &cpu, &mut mem, s)), 0xCAFE);
    }

    #[test]
    fn widechar_multibyte_roundtrips_and_sizes() {
        let (mut h, mut cpu, mut mem, s) = harness();
        // Wide "Hi\0" at T_DATA.
        for (i, u) in [u16::from(b'H'), u16::from(b'i'), 0].iter().enumerate() {
            mem.write_uint(T_DATA + (i as u64) * 2, 2, u64::from(*u)).unwrap();
        }
        let dst = T_DATA + 0x100;
        // cch = -1 (NUL-terminated); cb = 0 => sizing.
        set_args(&mut cpu, [0, 0, T_DATA, u64::from(u32::MAX)]);
        mem.write_uint(s, 8, dst).unwrap(); // arg5 lpMultiByteStr
        mem.write_uint(s + 8, 8, 0).unwrap(); // arg6 cbMultiByte = 0 (size)
        mem.write_uint(s + 24, 8, 0).unwrap(); // arg8 lpUsedDefaultChar
        assert_eq!(ret_of(h.dispatch("k", "WideCharToMultiByte", &cpu, &mut mem, s)), 3);
        // Now with a real buffer.
        mem.write_uint(s + 8, 8, 0x100).unwrap();
        assert_eq!(ret_of(h.dispatch("k", "WideCharToMultiByte", &cpu, &mut mem, s)), 3);
        assert_eq!(&mem_bytes(&mem, dst, 3), b"Hi\0");
        // Round-trip back to wide.
        let wout = T_DATA + 0x200;
        set_args(&mut cpu, [0, 0, dst, u64::from(u32::MAX)]);
        mem.write_uint(s, 8, wout).unwrap();
        mem.write_uint(s + 8, 8, 0x80).unwrap();
        assert_eq!(ret_of(h.dispatch("k", "MultiByteToWideChar", &cpu, &mut mem, s)), 3);
        assert_eq!(mem.read_uint(wout, 2).unwrap(), u64::from(b'H'));
    }

    fn mem_bytes(mem: &Memory, addr: u64, n: usize) -> Vec<u8> {
        (0..n).map(|i| mem.read_uint(addr + i as u64, 1).unwrap() as u8).collect()
    }

    #[test]
    fn string_type_classifies_and_fails_closed() {
        let (mut h, mut cpu, mut mem, s) = harness();
        for (i, u) in [u16::from(b'A'), u16::from(b'5'), 0].iter().enumerate() {
            mem.write_uint(T_DATA + (i as u64) * 2, 2, u64::from(*u)).unwrap();
        }
        let out = T_DATA + 0x100;
        set_args(&mut cpu, [1, T_DATA, u64::from(u32::MAX), out]); // CT_CTYPE1, NUL-term
        assert_eq!(ret_of(h.dispatch("k", "GetStringTypeW", &cpu, &mut mem, s)), 1);
        assert_ne!(mem.read_uint(out, 2).unwrap() & 0x0100, 0, "'A' is C1_ALPHA");
        assert_ne!(mem.read_uint(out + 2, 2).unwrap() & 0x0004, 0, "'5' is C1_DIGIT");
        // Unsupported info type fails closed.
        set_args(&mut cpu, [2, T_DATA, u64::from(u32::MAX), out]);
        assert!(matches!(
            h.dispatch("k", "GetStringTypeW", &cpu, &mut mem, s),
            Dispatch::Stop
        ));
    }

    #[test]
    fn lcmap_uppercases_and_fails_closed_on_sortkey() {
        let (mut h, mut cpu, mut mem, s) = harness();
        for (i, u) in [u16::from(b'a'), u16::from(b'b'), 0].iter().enumerate() {
            mem.write_uint(T_DATA + (i as u64) * 2, 2, u64::from(*u)).unwrap();
        }
        let out = T_DATA + 0x100;
        set_args(&mut cpu, [0, 0x200 /*UPPERCASE*/, T_DATA, u64::from(u32::MAX)]);
        mem.write_uint(s, 8, out).unwrap();
        mem.write_uint(s + 8, 8, 0x80).unwrap();
        assert_eq!(ret_of(h.dispatch("k", "LCMapStringW", &cpu, &mut mem, s)), 3);
        assert_eq!(mem.read_uint(out, 2).unwrap(), u64::from(b'A'));
        // Sort key (no case flag) is not modeled: fail closed.
        set_args(&mut cpu, [0, 0x400 /*SORTKEY*/, T_DATA, u64::from(u32::MAX)]);
        assert!(matches!(h.dispatch("k", "LCMapStringW", &cpu, &mut mem, s), Dispatch::Stop));
    }

    #[test]
    fn heap_size_tracks_allocations() {
        let (mut h, mut cpu, mut mem, s) = harness();
        set_args(&mut cpu, [0, 0, 0x30, 0]); // HeapAlloc(,,0x30)
        let va = ret_of(h.dispatch("k", "HeapAlloc", &cpu, &mut mem, s));
        assert_ne!(va, 0);
        set_args(&mut cpu, [0, 0, va, 0]);
        assert_eq!(ret_of(h.dispatch("k", "HeapSize", &cpu, &mut mem, s)), 0x30);
        set_args(&mut cpu, [0, 0, 0xdead_0000, 0]); // unknown block
        assert_eq!(ret_of(h.dispatch("k", "HeapSize", &cpu, &mut mem, s)), u64::MAX);
    }

    #[test]
    fn kuser_shared_data_is_populated() {
        let (mut h, _cpu, mut mem, _s) = harness();
        h.map_kuser_shared_data(&mut mem).expect("maps");
        assert_eq!(mem.read_uint(KUSD_BASE + 0x030, 2).unwrap(), u64::from(b'C'), "NtSystemRoot");
        assert_eq!(mem.read_uint(KUSD_BASE + 0x26C, 4).unwrap(), 10, "NtMajorVersion");
        assert!(mem.perm_at(KUSD_BASE).is_some(), "page mapped");
    }

    #[test]
    fn module_filename_writes_path_and_length() {
        let (mut h, mut cpu, mut mem, s) = harness();
        let dst = T_DATA + 0x100;
        set_args(&mut cpu, [0, dst, 0x100, 0]); // hModule=NULL, nSize large
        let n = ret_of(h.dispatch("k", "GetModuleFileNameW", &cpu, &mut mem, s));
        assert_eq!(n, "C:\\program.exe".encode_utf16().count() as u64);
        assert_eq!(mem.read_uint(dst + n * 2, 2).unwrap(), 0, "NUL-terminated");
        // Truncation: nSize = 4 => 3 chars + NUL, returns 3.
        set_args(&mut cpu, [0, dst, 4, 0]);
        assert_eq!(ret_of(h.dispatch("k", "GetModuleFileNameW", &cpu, &mut mem, s)), 3);
    }

    #[test]
    fn network_apis_are_denied_before_modeling() {
        // Module- and function-name matching both trip the denylist.
        assert!(is_network_api("ws2_32.dll", "connect"));
        assert!(is_network_api("WININET.dll", "InternetOpenA"));
        assert!(is_network_api("ntdll.dll", "WSASocketW")); // by function pattern
        assert!(is_network_api("winhttp.dll", "WinHttpConnect"));
        assert!(!is_network_api("kernel32.dll", "GetProcAddress"));
        assert!(!is_network_api("kernel32.dll", "HeapAlloc"));

        // A resolved network stub stops the run at the Win32 dispatch chokepoint,
        // before any modeling — and nothing touches the host (in-model only).
        let (mut h, mut cpu, mut mem, _s) = harness();
        let ev = FlowEvent { kind: FlowKind::Call, from: 0, to: 0, next: 0 };
        assert!(
            matches!(
                h.dispatch_win32("ws2_32.dll", "connect", &ev, &mut cpu, &mut mem),
                HookAction::Stop
            ),
            "network access blocked"
        );
    }

    #[test]
    fn unmodeled_api_stops() {
        let (mut h, cpu, mut mem, s) = harness();
        assert!(matches!(h.dispatch("k", "NoSuchApi", &cpu, &mut mem, s), Dispatch::Stop));
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
