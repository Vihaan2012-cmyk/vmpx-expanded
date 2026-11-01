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

use crate::iat_fill::{AddressBook, ImportSource, KnownImport};
use crate::cpu::Cpu;
use crate::emu::{FlowEvent, FlowKind, Hook, HookAction, Interest, MemEvent};
use crate::loader::LoadedImage;
use crate::memory::{Access, MapError, Memory, Perm};
use crate::ntdll::{NtDispatch, NtState};
use crate::observe::{ObservedFacts, SymbolOrOrdinal};
use crate::xplm::{Call as XCall, Dispatch as XDispatch, GuestMem, Xplm};

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
use crate::loader::PROCESS_HEAP; // GetProcessHeap == PEB.ProcessHeap
const HEAP_BASE: u64 = 0x0000_0070_FFF0_0000; // HeapCreate handle pool
const STDHANDLE_BASE: u64 = 0x0000_0070_FFE0_0000; // GetStdHandle sentinels
const REGHANDLE_BASE: u64 = 0x0000_0070_FFC0_0000; // RegOpenKeyEx handles
const SNAPHANDLE_BASE: u64 = 0x0000_0070_FFB0_0000; // Toolhelp snapshot handles

// Win32 error codes returned/stored by the host-identity models.
const ERROR_FILE_NOT_FOUND: u64 = 2;
const ERROR_NO_MORE_FILES: u64 = 18;
const ERROR_BAD_LENGTH: u64 = 24;
const ERROR_INVALID_PARAMETER: u64 = 87;
const ERROR_BUFFER_OVERFLOW: u64 = 111;
const ERROR_MORE_DATA: u64 = 234;
const ERROR_UNSUPPORTED_TYPE: u64 = 1630;
const INVALID_HANDLE_VALUE: u64 = 0xFFFF_FFFF_FFFF_FFFF;

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
const STUB_STRIDE: u32 = 0x20;
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

const ERROR_INVALID_HANDLE: u64 = 6;
const ERROR_MOD_NOT_FOUND: u64 = 126;
const ERROR_INSUFFICIENT_BUFFER: u64 = 122;
const ERROR_NOT_FOUND: u64 = 1168;

/// Whether `name` is a sandbox, analysis-hook or VM-guest module that a clean
/// bare-metal, no-debugger process never has loaded (case-insensitive, path and
/// `.dll` extension ignored).
fn is_analysis_artifact_module(name: &str) -> bool {
    const EXACT: &[&str] = &[
        "sbiedll", "sbiedllx", "dbghelp", "api_log", "dir_watch", "pstorec", "vmcheck", "wpespy",
        "cmdvrt32", "cmdvrt64", "snxhk", "avghookx", "avghooka", "sxin", "sf2", "vmguestlib",
    ];
    const PREFIX: &[&str] = &["vbox", "vmware", "vmhgfs", "vm3d", "vmdum", "vmtools", "vmmouse"];
    let base = name.rsplit(['\\', '/']).next().unwrap_or(name).to_lowercase();
    let stem = base.strip_suffix(".dll").unwrap_or(&base);
    EXACT.contains(&stem) || PREFIX.iter().any(|p| stem.starts_with(p))
}

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
    /// Inclusive `[lo, hi]` key span of `stubs` / `export_stub_addrs`; a flow
    /// target outside both is provably not a thunk, skipping the map lookups.
    stub_span: (u64, u64),
    export_span: (u64, u64),
    /// Half-open `[lo, hi)` union of the watched `AddressOfFunctions` slot
    /// ranges; a read outside it cannot hit any watch.
    watch_span: (u64, u64),
    /// In-model NT object + memory state for the emulated ntdll syscall stubs.
    nt: NtState,
    /// Real syscall numbers by `Nt*`/`Zw*` export name (offline System32 ntdll).
    syscall_numbers: BTreeMap<String, u32>,
    /// Placeholder syscall number per `Nt`/`Zw`-stripped name, for exports with no
    /// real number (recorded in the audit log).
    syscall_placeholders: BTreeMap<String, u32>,
    /// Syscall number -> `Nt*` function name, for `syscall` dispatch.
    syscall_names: BTreeMap<u32, String>,
    /// Backed ntdll stub VA -> its syscall number (pristine-bytes check).
    nt_stub_nr: BTreeMap<u64, u32>,
    /// Recording host-SDK model; XPLM/XPWidgets imports route here (never forwarded).
    xplm: Option<Rc<RefCell<Xplm>>>,
    /// Every import address this hook has handed out (see [`AddressBook`]).
    addr_book: AddressBook,
    /// Anti-analysis probes seen (see [`WinApiHook::audit`]).
    audit: Rc<RefCell<BTreeMap<String, u64>>>,
    /// Open registry key handle -> canonical lowercase key path.
    reg_handles: BTreeMap<u64, String>,
    reg_next: u64,
    /// Open Toolhelp snapshot handle -> snapshot.
    snapshots: BTreeMap<u64, Snapshot>,
    snap_next: u64,
    /// Last filter passed to `SetUnhandledExceptionFilter` (never invoked).
    unhandled_filter: u64,
}

/// A module as reported by the module enumeration APIs.
struct ModEntry {
    base: u64,
    size: u64,
    entry: u64,
    name: String,
    path: String,
}

enum SnapKind {
    Procs(Vec<crate::hostmodel::ProcEntry>),
    Mods(Vec<ModEntry>),
    /// `(thread id, owner pid)` of the contained process's threads.
    Threads(Vec<(u64, u64)>),
}

struct Snapshot {
    kind: SnapKind,
    /// Index of the entry the next `*Next` returns (`*First` resets to 0).
    cursor: usize,
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
        let stub_span = span_of(stubs.keys().copied());
        let addr_book: AddressBook = Rc::new(RefCell::new(
            stubs
                .iter()
                .map(|(&va, (m, f))| {
                    let known = KnownImport {
                        module: m.to_lowercase(),
                        name: f.clone(),
                        source: ImportSource::LoaderStub,
                    };
                    (va, known)
                })
                .collect(),
        ));
        Self {
            stubs,
            stub_span,
            export_span: EMPTY_SPAN,
            watch_span: (u64::MAX, 0),
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
            syscall_numbers: BTreeMap::new(),
            syscall_placeholders: BTreeMap::new(),
            syscall_names: BTreeMap::new(),
            nt_stub_nr: BTreeMap::new(),
            xplm: None,
            addr_book,
            audit: Rc::new(RefCell::new(BTreeMap::new())),
            reg_handles: BTreeMap::new(),
            reg_next: REGHANDLE_BASE,
            snapshots: BTreeMap::new(),
            snap_next: SNAPHANDLE_BASE,
            unhandled_filter: 0,
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

    /// Route XPLM/XPWidgets imports to this recording model.
    #[must_use]
    pub fn with_xplm(mut self, xplm: Rc<RefCell<Xplm>>) -> Self {
        self.xplm = Some(xplm);
        self
    }

    /// Attach the real `Nt*`/`Zw*` syscall numbers (from an offline System32 ntdll)
    /// used to back the ntdll export stubs with canonical `syscall` code.
    #[must_use]
    pub fn with_syscall_numbers(mut self, numbers: BTreeMap<String, u32>) -> Self {
        self.syscall_numbers = numbers;
        self
    }

    /// Shared handle to the observations collected during the run. Clone it
    /// before installing the hook, then read it after the run.
    #[must_use]
    pub fn facts(&self) -> Rc<RefCell<ObservedFacts>> {
        Rc::clone(&self.facts)
    }

    /// Shared handle to every import address handed out so far (loader stubs,
    /// export-table addresses, `GetProcAddress` results).
    #[must_use]
    pub fn address_book(&self) -> AddressBook {
        Rc::clone(&self.addr_book)
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
        let path = self.module_path(hmod);
        Self::copy_str_trunc(&path, dst, nsize, wide, mem)
    }

    /// Copy `s` (truncated to `nsize` chars, always NUL-terminated); returns the
    /// char count written.
    fn copy_str_trunc(s: &str, dst: u64, nsize: u64, wide: bool, mem: &mut Memory) -> Dispatch {
        let nsize = nsize as u32 as usize;
        if dst == 0 || nsize == 0 {
            return Dispatch::Ret(0);
        }
        let path = s;
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
                if key == "ntdll.dll" && self.install_nt_stubs(mem, h, &layout.slots).is_err() {
                    return 0;
                }
                for export in layout.slots.iter().flatten() {
                    self.export_span.0 = self.export_span.0.min(export.va);
                    self.export_span.1 = self.export_span.1.max(export.va);
                    self.export_stub_addrs
                        .insert(export.va, (key.clone(), symbol_key(&export.symbol)));
                    self.addr_book.borrow_mut().insert(
                        export.va,
                        KnownImport {
                            module: key.clone(),
                            name: symbol_key(&export.symbol),
                            source: ImportSource::ExportTable,
                        },
                    );
                }
                let wlo = layout.functions_va;
                let whi = wlo.saturating_add(4 * layout.slots.len() as u64);
                self.watch_span = (self.watch_span.0.min(wlo), self.watch_span.1.max(whi));
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

    fn fail(&mut self, err: u64) -> Dispatch {
        self.last_error = err;
        Dispatch::Ret(0)
    }

    /// `GetModuleHandle`/`LoadLibrary` front-end: sandbox / analysis / VM-guest
    /// modules are absent (NULL, `ERROR_MOD_NOT_FOUND`), as on a clean bare-metal
    /// host; everything else resolves via [`Self::module_handle`].
    fn lookup_module(&mut self, mem: &mut Memory, name: &str, api: &str) -> u64 {
        if is_analysis_artifact_module(name) {
            self.last_error = ERROR_MOD_NOT_FOUND;
            self.note(&format!("{api}({}) -> NULL", name.to_lowercase()));
            return 0;
        }
        self.module_handle(mem, name)
    }

    /// Audit an ntdll syscall that probes for a debugger / VM / sandbox.
    fn note_nt(&self, func: &str, cpu: &Cpu, status: u64) {
        let class = match func.get(2..).unwrap_or(func) {
            "QuerySystemInformation" => Self::arg(cpu, 0),
            "QueryInformationProcess" | "QueryInformationThread" | "SetInformationThread"
            | "SetInformationProcess" => Self::arg(cpu, 1),
            "GetContextThread" | "SetContextThread" => 0,
            "Close" if status != 0 => 0,
            _ => return,
        };
        self.note(&format!("{func} class={class} -> {status:#x}"));
    }

    /// Record an anti-analysis probe and the clean answer it received.
    fn note(&self, what: &str) {
        *self.audit.borrow_mut().entry(what.to_string()).or_insert(0) += 1;
    }

    /// Shared handle to the anti-analysis audit log: each distinct probe the
    /// guest made (API or NT info class) and the clean answer it got, with a
    /// call count. Clone it before installing the hook.
    #[must_use]
    pub fn audit(&self) -> Rc<RefCell<BTreeMap<String, u64>>> {
        Rc::clone(&self.audit)
    }

    /// The sample's own module: header-derived size and entry point.
    fn self_entry(&self, mem: &Memory) -> Option<ModEntry> {
        let b = self.image_base;
        if b == 0 {
            return None;
        }
        let nt = b + mem.read_uint(b + 0x3C, 4).ok()?;
        Some(ModEntry {
            base: b,
            size: mem.read_uint(nt + 24 + 56, 4).ok()?,
            entry: b + mem.read_uint(nt + 24 + 16, 4).ok()?,
            name: "program.exe".to_string(),
            path: "C:\\program.exe".to_string(),
        })
    }

    /// Info for a module handle we modeled (self or a minted fake module).
    fn module_info(&self, mem: &Memory, hmod: u64) -> Option<ModEntry> {
        if hmod == 0 || hmod == self.image_base {
            return self.self_entry(mem);
        }
        let name = self.handle_names.get(&hmod)?;
        Some(ModEntry {
            base: hmod,
            size: MODULE_IMAGE_SIZE,
            entry: 0, // synthetic module: no real entry point
            name: name.clone(),
            path: format!("C:\\Windows\\System32\\{name}"),
        })
    }

    /// The loaded-module list: the sample, ntdll/kernel32/kernelbase (minted on
    /// demand), then every other module the sample resolved, in load order.
    /// `None` if the sample image is unknown or a backing map fails.
    fn module_entries(&mut self, mem: &mut Memory) -> Option<Vec<ModEntry>> {
        let mut out = vec![self.self_entry(mem)?];
        for n in ["ntdll.dll", "kernel32.dll", "kernelbase.dll"] {
            if self.module_handle(mem, n) == 0 {
                return None;
            }
        }
        let mut hs: Vec<u64> = self.handle_names.keys().copied().collect();
        let rank = |n: &str| ["ntdll.dll", "kernel32.dll", "kernelbase.dll"].iter().position(|x| *x == n);
        hs.sort_by_key(|h| (rank(&self.handle_names[h]).unwrap_or(3), *h));
        for h in hs {
            out.push(self.module_info(mem, h)?);
        }
        Some(out)
    }

    /// Whether a process handle is the current process.
    fn is_self_process(h: u64) -> bool {
        h == u64::MAX
    }

    /// Registry key path for an open or predefined HKEY.
    fn reg_path(&self, hkey: u64) -> Option<String> {
        crate::hostmodel::predefined_root(hkey)
            .map(str::to_string)
            .or_else(|| self.reg_handles.get(&hkey).cloned())
    }

    /// Read an optional ANSI/wide string argument (NULL = empty); `None` on a
    /// bad pointer.
    fn opt_str(mem: &Memory, p: u64, wide: bool) -> Option<String> {
        if p == 0 {
            Some(String::new())
        } else if wide {
            read_wstr(mem, p)
        } else {
            read_cstr(mem, p)
        }
    }

    /// Shared tail of `RegQueryValueEx`/`RegGetValue`: write type/data/size and
    /// return the Win32 error code. `Err` = guest fault (stop).
    fn reg_copy_out(
        mem: &mut Memory,
        v: crate::hostmodel::RegVal,
        wide: bool,
        (p_type, p_data, p_cb): (u64, u64, u64),
    ) -> Result<u64, ()> {
        let bytes = v.bytes(wide);
        if p_type != 0 {
            mem.write_uint(p_type, 4, u64::from(v.reg_type())).map_err(|_| ())?;
        }
        if p_data != 0 && p_cb == 0 {
            return Ok(ERROR_INVALID_PARAMETER);
        }
        let mut status = 0;
        if p_data != 0 {
            let cap = mem.read_uint(p_cb, 4).map_err(|_| ())?;
            if cap < bytes.len() as u64 {
                status = ERROR_MORE_DATA;
            } else {
                mem.write(p_data, &bytes).map_err(|_| ())?;
            }
        }
        if p_cb != 0 {
            mem.write_uint(p_cb, 4, bytes.len() as u64).map_err(|_| ())?;
        }
        Ok(status)
    }

    /// Shared body of `GetUserName*`/`GetComputerName*`: copy `s` to `buf` and
    /// update the in/out char count at `pcb`. `succ_with_nul` selects whether
    /// the success count includes the terminator (GetUserName) or not.
    fn name_inout(
        &mut self,
        mem: &mut Memory,
        s: &str,
        wide: bool,
        (buf, pcb): (u64, u64),
        succ_with_nul: bool,
        small_err: u64,
    ) -> Dispatch {
        if pcb == 0 {
            return self.fail(ERROR_INVALID_PARAMETER);
        }
        let Ok(cap) = mem.read_uint(pcb, 4) else {
            return Dispatch::Stop;
        };
        let len = s.len() as u64; // names are ASCII
        if buf == 0 || cap < len + 1 {
            if mem.write_uint(pcb, 4, len + 1).is_err() {
                return Dispatch::Stop;
            }
            return self.fail(small_err);
        }
        let bytes: Vec<u8> = if wide {
            s.encode_utf16().chain([0]).flat_map(u16::to_le_bytes).collect()
        } else {
            s.bytes().chain([0]).collect()
        };
        let reported = if succ_with_nul { len + 1 } else { len };
        if mem.write(buf, &bytes).is_err() || mem.write_uint(pcb, 4, reported).is_err() {
            return Dispatch::Stop;
        }
        Dispatch::Ret(1)
    }

    /// `RegOpenKeyExA/W(hKey, lpSubKey, ulOptions, samDesired, phkResult)`.
    fn reg_open(&mut self, mem: &mut Memory, cpu: &Cpu, sab: u64, wide: bool) -> Dispatch {
        let Some(base) = self.reg_path(Self::arg(cpu, 0)) else {
            return Dispatch::Ret(ERROR_INVALID_HANDLE);
        };
        let (Some(sub), Some(phk)) =
            (Self::opt_str(mem, Self::arg(cpu, 1), wide), Self::stack_arg(cpu, mem, sab, 4))
        else {
            return Dispatch::Stop;
        };
        let path = crate::hostmodel::join_path(&base, &sub);
        if phk != 0 && mem.write_uint(phk, 8, 0).is_err() {
            return Dispatch::Stop;
        }
        if !crate::hostmodel::key_exists(&path) {
            return Dispatch::Ret(ERROR_FILE_NOT_FOUND);
        }
        let h = self.reg_next;
        self.reg_next += 4;
        self.reg_handles.insert(h, path);
        if phk != 0 && mem.write_uint(phk, 8, h).is_err() {
            return Dispatch::Stop;
        }
        Dispatch::Ret(0)
    }

    /// `RegQueryValueExA/W(hKey, lpValueName, lpReserved, lpType, lpData, lpcbData)`.
    fn reg_query(&mut self, mem: &mut Memory, cpu: &Cpu, sab: u64, wide: bool) -> Dispatch {
        let (Some(p_data), Some(p_cb)) =
            (Self::stack_arg(cpu, mem, sab, 4), Self::stack_arg(cpu, mem, sab, 5))
        else {
            return Dispatch::Stop;
        };
        let Some(name) = Self::opt_str(mem, Self::arg(cpu, 1), wide) else {
            return Dispatch::Stop;
        };
        let Some(path) = self.reg_path(Self::arg(cpu, 0)) else {
            return Dispatch::Ret(ERROR_INVALID_HANDLE);
        };
        let Some(v) = crate::hostmodel::value(&path, &name.to_lowercase()) else {
            return Dispatch::Ret(ERROR_FILE_NOT_FOUND);
        };
        match Self::reg_copy_out(mem, v, wide, (Self::arg(cpu, 3), p_data, p_cb)) {
            Ok(status) => Dispatch::Ret(status),
            Err(()) => Dispatch::Stop,
        }
    }

    /// `RegGetValueA/W(hkey, lpSubKey, lpValue, dwFlags, pdwType, pvData, pcbData)`.
    fn reg_get_value(&mut self, mem: &mut Memory, cpu: &Cpu, sab: u64, wide: bool) -> Dispatch {
        const RRF_RT_REG_SZ: u64 = 0x2;
        const RRF_RT_REG_DWORD: u64 = 0x10;
        let (Some(p_type), Some(p_data), Some(p_cb)) = (
            Self::stack_arg(cpu, mem, sab, 4),
            Self::stack_arg(cpu, mem, sab, 5),
            Self::stack_arg(cpu, mem, sab, 6),
        ) else {
            return Dispatch::Stop;
        };
        let flags = Self::arg(cpu, 3) as u32 as u64;
        if flags & !0xFFFF != 0 {
            return Dispatch::Stop; // WOW64/NOEXPAND/ZEROONFAILURE not modeled
        }
        if flags == 0 {
            return Dispatch::Ret(ERROR_INVALID_PARAMETER);
        }
        let (Some(sub), Some(name)) = (
            Self::opt_str(mem, Self::arg(cpu, 1), wide),
            Self::opt_str(mem, Self::arg(cpu, 2), wide),
        ) else {
            return Dispatch::Stop;
        };
        let Some(base) = self.reg_path(Self::arg(cpu, 0)) else {
            return Dispatch::Ret(ERROR_INVALID_HANDLE);
        };
        let path = crate::hostmodel::join_path(&base, &sub);
        let Some(v) = crate::hostmodel::value(&path, &name.to_lowercase()) else {
            return Dispatch::Ret(ERROR_FILE_NOT_FOUND);
        };
        let allowed = match v {
            crate::hostmodel::RegVal::Sz(_) => RRF_RT_REG_SZ,
            crate::hostmodel::RegVal::Dword(_) => RRF_RT_REG_DWORD,
        };
        if flags & allowed == 0 {
            return Dispatch::Ret(ERROR_UNSUPPORTED_TYPE);
        }
        match Self::reg_copy_out(mem, v, wide, (p_type, p_data, p_cb)) {
            Ok(status) => Dispatch::Ret(status),
            Err(()) => Dispatch::Stop,
        }
    }

    /// `CreateToolhelp32Snapshot(dwFlags, th32ProcessID)`: process (`TH32CS_SNAPPROCESS`)
    /// or module (`SNAPMODULE`/`SNAPMODULE32`) snapshots of the current process only;
    /// thread/heap snapshots and flag mixes fail closed.
    fn create_snapshot(&mut self, mem: &mut Memory, cpu: &Cpu) -> Dispatch {
        let flags = Self::arg(cpu, 0) as u32 & !0x8000_0000; // drop TH32CS_INHERIT
        let pid = Self::arg(cpu, 1) as u32 as u64;
        let kind = if flags == 0x2 {
            let mut procs = crate::hostmodel::SYSTEM_PROCESSES.to_vec();
            procs.push((crate::ntdll::FAKE_PID, crate::hostmodel::PARENT_PID, "program.exe", 4));
            procs.sort_by_key(|p| p.0);
            SnapKind::Procs(procs)
        } else if flags == 0x4 {
            SnapKind::Threads(vec![(crate::ntdll::FAKE_TID, crate::ntdll::FAKE_PID)])
        } else if flags & 0x18 != 0 && flags & !0x18 == 0 {
            if pid != 0 && pid != crate::ntdll::FAKE_PID {
                self.last_error = ERROR_INVALID_PARAMETER;
                return Dispatch::Ret(INVALID_HANDLE_VALUE);
            }
            match self.module_entries(mem) {
                Some(m) => SnapKind::Mods(m),
                None => return Dispatch::Stop,
            }
        } else {
            self.note(&format!("UNMODELED snapshot flags={flags:#x} pid={pid:#x} (run stopped)"));
            return Dispatch::Stop;
        };
        let h = self.snap_next;
        self.snap_next += 4;
        self.snapshots.insert(h, Snapshot { kind, cursor: 0 });
        Dispatch::Ret(h)
    }

    /// `Process32First/Next[W]` and `Module32First/Next[W]`.
    fn snap_step(
        &mut self,
        mem: &mut Memory,
        cpu: &Cpu,
        (modules, wide, first): (bool, bool, bool),
    ) -> Dispatch {
        let (h, out) = (Self::arg(cpu, 0), Self::arg(cpu, 1));
        let size: u64 = match (modules, wide) {
            (false, true) => 568,
            (false, false) => 304,
            (true, true) => 1080,
            (true, false) => 568,
        };
        let le = &mut self.last_error;
        let err = |le: &mut u64, e: u64| {
            *le = e;
            Dispatch::Ret(0)
        };
        let Some(snap) = self.snapshots.get_mut(&h) else {
            return err(le, ERROR_INVALID_HANDLE);
        };
        if matches!(snap.kind, SnapKind::Mods(_)) != modules || matches!(snap.kind, SnapKind::Threads(_)) {
            return err(le, ERROR_INVALID_HANDLE);
        }
        let Ok(dw_size) = mem.read_uint(out, 4) else {
            return Dispatch::Stop;
        };
        if dw_size != size {
            return err(le, ERROR_BAD_LENGTH);
        }
        if first {
            snap.cursor = 0;
        }
        let idx = snap.cursor;
        let mut b = vec![0u8; size as usize];
        let put_str = |b: &mut [u8], off: usize, s: &str| {
            if wide {
                for (i, u) in s.encode_utf16().enumerate() {
                    b[off + i * 2..off + i * 2 + 2].copy_from_slice(&u.to_le_bytes());
                }
            } else {
                b[off..off + s.len()].copy_from_slice(s.as_bytes());
            }
        };
        b[0..4].copy_from_slice(&(size as u32).to_le_bytes());
        match &snap.kind {
            SnapKind::Procs(p) => {
                let Some(&(pid, parent, name, threads)) = p.get(idx) else {
                    return err(le, ERROR_NO_MORE_FILES);
                };
                b[4..8].copy_from_slice(&1u32.to_le_bytes()); // cntUsage
                b[8..12].copy_from_slice(&(pid as u32).to_le_bytes());
                b[28..32].copy_from_slice(&threads.to_le_bytes());
                b[32..36].copy_from_slice(&(parent as u32).to_le_bytes());
                b[36..40].copy_from_slice(&8u32.to_le_bytes()); // pcPriClassBase
                put_str(&mut b, 44, name);
            }
            SnapKind::Threads(_) => return err(le, ERROR_INVALID_HANDLE),
            SnapKind::Mods(m) => {
                let Some(e) = m.get(idx) else {
                    return err(le, ERROR_NO_MORE_FILES);
                };
                b[4..8].copy_from_slice(&1u32.to_le_bytes()); // th32ModuleID
                b[8..12].copy_from_slice(&(crate::ntdll::FAKE_PID as u32).to_le_bytes());
                b[12..16].copy_from_slice(&1u32.to_le_bytes()); // GlblcntUsage
                b[16..20].copy_from_slice(&1u32.to_le_bytes()); // ProccntUsage
                b[24..32].copy_from_slice(&e.base.to_le_bytes());
                b[32..36].copy_from_slice(&(e.size as u32).to_le_bytes());
                b[40..48].copy_from_slice(&e.base.to_le_bytes()); // hModule
                put_str(&mut b, 48, &e.name);
                put_str(&mut b, if wide { 560 } else { 304 }, &e.path);
            }
        }
        snap.cursor = idx + 1;
        if mem.write(out, &b).is_err() {
            return Dispatch::Stop;
        }
        Dispatch::Ret(1)
    }

    /// `Thread32First/Next` over a thread snapshot (THREADENTRY32, 28 bytes).
    fn thread_step(&mut self, mem: &mut Memory, cpu: &Cpu, first: bool) -> Dispatch {
        let (h, out) = (Self::arg(cpu, 0), Self::arg(cpu, 1));
        let fail = |this: &mut Self, e: u64| {
            this.last_error = e;
            Dispatch::Ret(0)
        };
        let Some(snap) = self.snapshots.get_mut(&h) else {
            return fail(self, ERROR_INVALID_HANDLE);
        };
        let SnapKind::Threads(t) = &snap.kind else {
            return fail(self, ERROR_INVALID_HANDLE);
        };
        let Ok(dw_size) = mem.read_uint(out, 4) else {
            return Dispatch::Stop;
        };
        if dw_size != 28 {
            return fail(self, ERROR_BAD_LENGTH);
        }
        if first {
            snap.cursor = 0;
        }
        let Some(&(tid, pid)) = t.get(snap.cursor) else {
            return fail(self, ERROR_NO_MORE_FILES);
        };
        snap.cursor += 1;
        let mut b = [0u8; 28];
        b[0..4].copy_from_slice(&28u32.to_le_bytes());
        b[4..8].copy_from_slice(&1u32.to_le_bytes()); // cntUsage
        b[8..12].copy_from_slice(&(tid as u32).to_le_bytes());
        b[12..16].copy_from_slice(&(pid as u32).to_le_bytes());
        b[16..20].copy_from_slice(&8u32.to_le_bytes()); // tpBasePri
        if mem.write(out, &b).is_err() {
            return Dispatch::Stop;
        }
        Dispatch::Ret(1)
    }

    /// Back every ntdll `Nt*`/`Zw*` export with canonical Win10 x64 syscall-stub
    /// code (`mov r10,rcx; mov eax,N; test [KUSD+0x308],1; jne; syscall; ret;
    /// int 2e; ret`) so a protector reading the export bytes (API-hook integrity
    /// checks) sees real code. `N` is the real number when known, else a stable
    /// recorded placeholder.
    fn install_nt_stubs(
        &mut self,
        mem: &mut Memory,
        base: u64,
        slots: &[Option<ResolvedExport>],
    ) -> Result<(), MapError> {
        if self.syscall_numbers.is_empty() {
            self.note("SYSCALL_NUMBERS_UNAVAILABLE (placeholder numbers)");
        }
        for export in slots.iter().flatten() {
            let SymbolOrOrdinal::Name(name) = &export.symbol else { continue };
            if !is_nt_syscall(name) {
                continue;
            }
            let nr = match self.syscall_numbers.get(name) {
                Some(&n) => n,
                None => {
                    let stem = name[2..].to_string();
                    let next = SYSCALL_PLACEHOLDER_BASE + self.syscall_placeholders.len() as u32;
                    let n = *self.syscall_placeholders.entry(stem).or_insert(next);
                    if !self.syscall_numbers.is_empty() {
                        self.note(&format!("SYSCALL_PLACEHOLDER {name}"));
                    }
                    n
                }
            };
            mem.poke(export.va, &nt_stub_bytes(nr)).map_err(|_| MapError::Overlap)?;
            self.nt_stub_nr.insert(export.va, nr);
            if !self.syscall_names.contains_key(&nr) || name.starts_with("Nt") {
                self.syscall_names.insert(nr, name.clone());
            }
        }
        let stub_va = base + u64::from(STUB_REGION_RVA);
        let len = round_up(slots.len() as u64 * u64::from(STUB_STRIDE));
        mem.protect(stub_va, len, Perm::RX).map_err(|_| MapError::Overlap)
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
        let va = *self.procs.entry(key).or_insert(PROC_BASE + n * 8);
        if module != "<unknown>" {
            self.addr_book.borrow_mut().insert(
                va,
                KnownImport {
                    module: module.to_owned(),
                    name: symbol.to_owned(),
                    source: ImportSource::GetProcAddress,
                },
            );
        }
        va
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
            // LocalAlloc/GlobalAlloc(uFlags, dwBytes): fixed-memory blocks from
            // the same zeroed bump pool; Free returns NULL on success.
            (_, "LocalAlloc" | "GlobalAlloc") => {
                let size = Self::arg(cpu, 1);
                let va = self.alloc(mem, size);
                if va != 0 {
                    self.heap_sizes.insert(va, size);
                }
                Dispatch::Ret(va)
            }
            (_, "LocalFree" | "GlobalFree") => Dispatch::Ret(0),
            (_, "FreeLibrary") => Dispatch::Ret(1),
            // GetUserObjectInformationW(h, nIndex, pvInfo, nLength, lpnLengthNeeded):
            // the interactive window station "WinSta0" (UOI_NAME=2) / visible
            // flags (UOI_FLAGS=1); other indices fail.
            (_, "GetUserObjectInformationW") => {
                let Some(needed) = Self::stack_arg(cpu, mem, sab, 4) else {
                    return Dispatch::Stop;
                };
                let (index, buf, len) = (Self::arg(cpu, 1), Self::arg(cpu, 2), Self::arg(cpu, 3));
                let payload: Vec<u8> = match index {
                    2 => "WinSta0\0".encode_utf16().flat_map(u16::to_le_bytes).collect(),
                    1 => [0u32, 0, 1].iter().flat_map(|v| v.to_le_bytes()).collect(),
                    _ => return Dispatch::Ret(0),
                };
                if needed != 0 && mem.write_uint(needed, 4, payload.len() as u64).is_err() {
                    return Dispatch::Stop;
                }
                if len < payload.len() as u64 || mem.write(buf, &payload).is_err() {
                    return Dispatch::Ret(0);
                }
                Dispatch::Ret(1)
            }
            // Fixed non-null window-station handle (interactive WinSta0).
            (_, "GetProcessWindowStation") => Dispatch::Ret(0x0000_0070_FFD0_0010),
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
                Dispatch::Ret(self.lookup_module(mem, &name, func))
            }
            (_, "GetModuleHandleW" | "LoadLibraryW") => {
                let p = Self::arg(cpu, 0);
                let name = if p == 0 { "<self>".to_string() } else {
                    match read_wstr(mem, p) {
                        Some(s) => s,
                        None => return Dispatch::Stop,
                    }
                };
                Dispatch::Ret(self.lookup_module(mem, &name, func))
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
            (_, "GetSystemInfo" | "GetNativeSystemInfo") => {
                // SYSTEM_INFO (48 bytes): x64, 4 KiB pages, 64 KiB granularity, 4 CPUs.
                let mut b = [0u8; 48];
                b[0..2].copy_from_slice(&9u16.to_le_bytes());
                b[4..8].copy_from_slice(&0x1000u32.to_le_bytes());
                b[8..16].copy_from_slice(&0x1_0000u64.to_le_bytes());
                b[16..24].copy_from_slice(&0x7FFF_FFFE_FFFFu64.to_le_bytes());
                b[24..32].copy_from_slice(&0xFu64.to_le_bytes());
                b[32..36].copy_from_slice(&4u32.to_le_bytes());
                b[36..40].copy_from_slice(&8664u32.to_le_bytes());
                b[40..44].copy_from_slice(&0x1_0000u32.to_le_bytes());
                b[44..46].copy_from_slice(&6u16.to_le_bytes());
                if mem.write(Self::arg(cpu, 0), &b).is_err() {
                    return Dispatch::Stop;
                }
                Dispatch::Ret(0)
            }
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

            (_, "IsDebuggerPresent") => {
                self.note("IsDebuggerPresent -> FALSE");
                Dispatch::Ret(0)
            }
            // ---- Host identity (clean, deterministic; see hostmodel) ---------
            // User/computer name: a generic benign user and host.
            (_, "GetUserNameA" | "GetUserNameW") => {
                let (buf, pcb) = (Self::arg(cpu, 0), Self::arg(cpu, 1));
                let wide = func.ends_with('W');
                self.name_inout(mem, crate::hostmodel::USER_NAME, wide, (buf, pcb), true, ERROR_INSUFFICIENT_BUFFER)
            }
            (_, "GetComputerNameA" | "GetComputerNameW") => {
                let (buf, pcb) = (Self::arg(cpu, 0), Self::arg(cpu, 1));
                let wide = func.ends_with('W');
                self.name_inout(mem, crate::hostmodel::COMPUTER_NAME, wide, (buf, pcb), false, ERROR_BUFFER_OVERFLOW)
            }
            // GetComputerNameExA/W(NameType, buf, *size): host-name formats give the
            // host; the DNS-domain formats give "" (a workgroup machine). The
            // remaining formats fail closed.
            (_, "GetComputerNameExA" | "GetComputerNameExW") => {
                let s = match Self::arg(cpu, 0) as u32 {
                    0 | 1 | 3 | 4 | 5 | 7 => crate::hostmodel::COMPUTER_NAME,
                    2 | 6 => "",
                    _ => return Dispatch::Stop,
                };
                let (buf, pcb) = (Self::arg(cpu, 1), Self::arg(cpu, 2));
                let wide = func.ends_with('W');
                self.name_inout(mem, s, wide, (buf, pcb), false, ERROR_MORE_DATA)
            }

            // Registry: a read-only closed-world model. Writes, enumeration and
            // key creation are not modeled (they stop the run).
            (_, "RegOpenKeyExA") => self.reg_open(mem, cpu, sab, false),
            (_, "RegOpenKeyExW") => self.reg_open(mem, cpu, sab, true),
            (_, "RegQueryValueExA") => self.reg_query(mem, cpu, sab, false),
            (_, "RegQueryValueExW") => self.reg_query(mem, cpu, sab, true),
            (_, "RegGetValueA") => self.reg_get_value(mem, cpu, sab, false),
            (_, "RegGetValueW") => self.reg_get_value(mem, cpu, sab, true),
            // RegCloseKey returns the error code directly; predefined roots close
            // as a no-op success.
            (_, "RegCloseKey") => {
                let h = Self::arg(cpu, 0);
                let known = crate::hostmodel::predefined_root(h).is_some()
                    || self.reg_handles.remove(&h).is_some();
                Dispatch::Ret(if known { 0 } else { ERROR_INVALID_HANDLE })
            }

            // Toolhelp: current-process process/module snapshots only.
            (_, "CreateToolhelp32Snapshot") => self.create_snapshot(mem, cpu),
            (_, "Process32FirstW") => self.snap_step(mem, cpu, (false, true, true)),
            (_, "Process32NextW") => self.snap_step(mem, cpu, (false, true, false)),
            (_, "Process32First") => self.snap_step(mem, cpu, (false, false, true)),
            (_, "Process32Next") => self.snap_step(mem, cpu, (false, false, false)),
            (_, "Module32FirstW") => self.snap_step(mem, cpu, (true, true, true)),
            (_, "Module32NextW") => self.snap_step(mem, cpu, (true, true, false)),
            (_, "Module32First") => self.snap_step(mem, cpu, (true, false, true)),
            (_, "Module32Next") => self.snap_step(mem, cpu, (true, false, false)),
            (_, "Thread32First") => self.thread_step(mem, cpu, true),
            (_, "Thread32Next") => self.thread_step(mem, cpu, false),

            // PSAPI, current process only. EnumProcessModulesEx(.., dwFilterFlag):
            // default/64-bit/all filters list every module; the 32-bit filter is
            // unmodeled.
            (_, "EnumProcessModules" | "K32EnumProcessModules" | "EnumProcessModulesEx"
                | "K32EnumProcessModulesEx") => {
                if !Self::is_self_process(Self::arg(cpu, 0)) {
                    return Dispatch::Stop;
                }
                if func.contains("Ex") && !matches!(Self::stack_arg(cpu, mem, sab, 4), Some(0 | 2 | 3)) {
                    return Dispatch::Stop;
                }
                let (lph, cb, p_need) = (Self::arg(cpu, 1), Self::arg(cpu, 2) as u32 as u64, Self::arg(cpu, 3));
                let Some(mods) = self.module_entries(mem) else {
                    return Dispatch::Stop;
                };
                for (i, m) in mods.iter().enumerate().take((cb / 8) as usize) {
                    if mem.write_uint(lph + i as u64 * 8, 8, m.base).is_err() {
                        return Dispatch::Stop;
                    }
                }
                if p_need != 0 && mem.write_uint(p_need, 4, mods.len() as u64 * 8).is_err() {
                    return Dispatch::Stop;
                }
                Dispatch::Ret(1)
            }
            (_, "GetModuleFileNameExA" | "GetModuleFileNameExW" | "K32GetModuleFileNameExA"
                | "K32GetModuleFileNameExW" | "GetModuleBaseNameA" | "GetModuleBaseNameW"
                | "K32GetModuleBaseNameA" | "K32GetModuleBaseNameW") => {
                if !Self::is_self_process(Self::arg(cpu, 0)) {
                    return Dispatch::Stop;
                }
                let Some(m) = self.module_info(mem, Self::arg(cpu, 1)) else {
                    return self.fail(ERROR_INVALID_HANDLE);
                };
                let s = if func.contains("BaseName") { m.name } else { m.path };
                Self::copy_str_trunc(&s, Self::arg(cpu, 2), Self::arg(cpu, 3), func.ends_with('W'), mem)
            }
            // GetModuleInformation(hProcess, hModule, lpmodinfo, cb): MODULEINFO
            // { lpBaseOfDll, SizeOfImage, EntryPoint } (24 bytes).
            (_, "GetModuleInformation" | "K32GetModuleInformation") => {
                if !Self::is_self_process(Self::arg(cpu, 0)) {
                    return Dispatch::Stop;
                }
                if (Self::arg(cpu, 3) as u32) < 24 {
                    return self.fail(ERROR_INSUFFICIENT_BUFFER);
                }
                let hmod = Self::arg(cpu, 1);
                let Some(m) = self.module_info(mem, hmod).filter(|_| hmod != 0) else {
                    return self.fail(ERROR_INVALID_HANDLE);
                };
                let mut b = [0u8; 24];
                b[0..8].copy_from_slice(&m.base.to_le_bytes());
                b[8..12].copy_from_slice(&(m.size as u32).to_le_bytes());
                b[16..24].copy_from_slice(&m.entry.to_le_bytes());
                if mem.write(Self::arg(cpu, 2), &b).is_err() {
                    return Dispatch::Stop;
                }
                Dispatch::Ret(1)
            }

            // GetModuleHandleExA/W(dwFlags, lpModuleName | address, phModule):
            // consistent with GetModuleHandle; FROM_ADDRESS resolves an address
            // inside a modeled module. PIN/UNCHANGED_REFCOUNT are no-ops.
            (_, "GetModuleHandleExA" | "GetModuleHandleExW") => {
                let flags = Self::arg(cpu, 0) as u32;
                let (p, phm) = (Self::arg(cpu, 1), Self::arg(cpu, 2));
                if flags & !0x7 != 0 {
                    return Dispatch::Stop;
                }
                if phm == 0 {
                    return self.fail(ERROR_INVALID_PARAMETER);
                }
                let h = if flags & 0x4 != 0 {
                    let mut hit = self
                        .self_entry(mem)
                        .filter(|m| (m.base..m.base + m.size).contains(&p))
                        .map(|m| m.base);
                    if hit.is_none() {
                        hit = self
                            .handle_names
                            .keys()
                            .copied()
                            .find(|&b| (b..b + MODULE_IMAGE_SIZE).contains(&p));
                    }
                    hit.unwrap_or(0)
                } else {
                    let name = if p == 0 {
                        "<self>".to_string()
                    } else {
                        match Self::opt_str(mem, p, func.ends_with('W')) {
                            Some(s) => s,
                            None => return Dispatch::Stop,
                        }
                    };
                    self.lookup_module(mem, &name, func)
                };
                if mem.write_uint(phm, 8, h).is_err() {
                    return Dispatch::Stop;
                }
                if h == 0 {
                    return self.fail(ERROR_MOD_NOT_FOUND);
                }
                Dispatch::Ret(1)
            }

            // SetUnhandledExceptionFilter(lpFilter) -> previous filter. Stored and
            // never invoked; not an anti-debug signal. UnhandledExceptionFilter
            // (no debugger => EXCEPTION_EXECUTE_HANDLER) cannot run a stored
            // guest filter, so that case fails closed.
            (_, "SetUnhandledExceptionFilter") => {
                let prev = self.unhandled_filter;
                self.unhandled_filter = Self::arg(cpu, 0);
                Dispatch::Ret(prev)
            }
            (_, "UnhandledExceptionFilter") => {
                if self.unhandled_filter != 0 {
                    Dispatch::Stop
                } else {
                    Dispatch::Ret(1)
                }
            }
            // DebugBreak raises a breakpoint exception we do not deliver: stop.
            (_, "DebugBreak") => Dispatch::Stop,

            // CheckRemoteDebuggerPresent(hProcess, pbDebuggerPresent) -> BOOL ok
            (_, "CheckRemoteDebuggerPresent") => {
                let p = Self::arg(cpu, 1);
                if p != 0 && mem.write_uint(p, 4, 0).is_err() {
                    return Dispatch::Stop;
                }
                self.note("CheckRemoteDebuggerPresent -> FALSE");
                Dispatch::Ret(1)
            }
            // OutputDebugStringA/W: with no debugger attached the text goes nowhere.
            (_, "OutputDebugStringA" | "OutputDebugStringW") => {
                self.note(func);
                Dispatch::Void
            }
            // CloseHandle(h) -> BOOL. A handle we do not own (incl. NULL and
            // INVALID_HANDLE_VALUE) is FALSE / ERROR_INVALID_HANDLE with NO
            // exception (a debugger would raise EXCEPTION_INVALID_HANDLE).
            (_, "CloseHandle") => {
                let h = Self::arg(cpu, 0);
                if self.snapshots.remove(&h).is_some() {
                    return Dispatch::Ret(1);
                }
                let std = (STDHANDLE_BASE..STDHANDLE_BASE + 0x10).contains(&h);
                if std || self.nt.owns_handle(h) {
                    let _ = self.nt.dispatch("NtClose", cpu, mem, sab);
                    return Dispatch::Ret(1);
                }
                self.last_error = ERROR_INVALID_HANDLE;
                self.note("CloseHandle(invalid) -> FALSE, no exception");
                Dispatch::Ret(0)
            }
            // Get/SetThreadContext(hThread, lpContext) -> BOOL: only the debug
            // registers are modeled (read as zero; a set is a no-op).
            (_, "GetThreadContext" | "SetThreadContext") => {
                let get = func == "GetThreadContext";
                if !crate::ntdll::debug_context(mem, Self::arg(cpu, 1), get) {
                    return Dispatch::Stop;
                }
                self.note(if get {
                    "GetThreadContext(DRx) -> Dr0-7 = 0"
                } else {
                    "SetThreadContext(DRx) -> accepted no-op"
                });
                Dispatch::Ret(1)
            }
            // EnumSystemFirmwareTables(sig, pBuf, size) /
            // GetSystemFirmwareTable(sig, id, pBuf, size) -> bytes (or required size).
            (_, "EnumSystemFirmwareTables" | "GetSystemFirmwareTable") => {
                let enumerate = func == "EnumSystemFirmwareTables";
                let sig = Self::arg(cpu, 0) as u32;
                let (id, buf, size) = if enumerate {
                    (0, Self::arg(cpu, 1), Self::arg(cpu, 2) as u32)
                } else {
                    (Self::arg(cpu, 1) as u32, Self::arg(cpu, 2), Self::arg(cpu, 3) as u32)
                };
                let data = match crate::ntdll::firmware_table(sig, u64::from(!enumerate), id) {
                    crate::ntdll::FwResult::Data(d) => d,
                    crate::ntdll::FwResult::NotFound => {
                        self.last_error = ERROR_NOT_FOUND;
                        return Dispatch::Ret(0);
                    }
                    crate::ntdll::FwResult::Unmodeled => return Dispatch::Stop,
                };
                let provider: String = sig.to_be_bytes().iter().map(|&b| char::from(b)).collect();
                self.note(&format!("{func}({provider}) -> bare-metal/empty"));
                if (data.len() as u32) > size || buf == 0 {
                    self.last_error = ERROR_INSUFFICIENT_BUFFER;
                } else if mem.write(buf, &data).is_err() {
                    return Dispatch::Stop;
                }
                Dispatch::Ret(data.len() as u64)
            }

            // Unmodeled: stop rather than guess (see module docs).
            _ => Dispatch::Stop,
        }
    }
}

/// Empty span: `lo > hi`, so no address is inside it.
const EMPTY_SPAN: (u64, u64) = (u64::MAX, 0);

fn span_of(keys: impl Iterator<Item = u64>) -> (u64, u64) {
    keys.fold(EMPTY_SPAN, |(lo, hi), k| (lo.min(k), hi.max(k)))
}

impl Hook for WinApiHook {
    fn interest(&self) -> Interest {
        let kusd = if self.kusd_ready { (0, 0) } else { (KUSD_BASE, KUSD_BASE + PAGE) };
        // Flow spans are inclusive in `on_flow`; widen to half-open (a span ending
        // at u64::MAX cannot be, so fall back to every transfer).
        let incl = |s: (u64, u64)| if s.0 > s.1 { (0, 0) } else { (s.0, s.1.saturating_add(1)) };
        let (st, ex) = (incl(self.stub_span), incl(self.export_span));
        Interest {
            read: [kusd, self.watch_span],
            write: [kusd, (0, 0)],
            flow: [st, ex],
            flow_all: st.1 == u64::MAX || ex.1 == u64::MAX,
            ..Interest::NONE
        }
    }

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
        if ev.access == Access::Read
            && ev.addr >= self.watch_span.0
            && ev.addr < self.watch_span.1
        {
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
        let to = ev.to;
        let in_span = |s: (u64, u64)| to >= s.0 && to <= s.1;
        if !in_span(self.stub_span) && !in_span(self.export_span) {
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
                // A stub the guest has patched (an API hook) runs as real code;
                // its `syscall` is dispatched by `on_syscall`.
                if let Some(&nr) = self.nt_stub_nr.get(&ev.to) {
                    let mut cur = [0u8; NT_STUB_LEN];
                    if mem.read(ev.to, &mut cur).is_err() || cur != nt_stub_bytes(nr)[..NT_STUB_LEN] {
                        return HookAction::Continue;
                    }
                }
                let sab = Self::stack_arg_base(ev, cpu);
                return match self.nt.dispatch(&func, cpu, mem, sab) {
                    NtDispatch::Status(s) => {
                        self.note_nt(&func, cpu, s);
                        cpu.regs[RAX] = s;
                        Self::resume(ev, cpu, mem)
                    }
                    NtDispatch::Stop => {
                        self.note(&format!("UNMODELED {func} (run stopped)"));
                        HookAction::Stop
                    }
                };
            }
            return self.dispatch_win32(&dll, &func, ev, cpu, mem);
        }
        HookAction::Continue // not an import thunk
    }

    /// `syscall`: map `eax` back to its `Nt*` handler, run it, and return as the
    /// stub's `syscall` would (RAX = NTSTATUS, RCX = next rip, R11 = rflags).
    fn on_syscall(&mut self, cpu: &mut Cpu, mem: &mut Memory) -> Option<HookAction> {
        let nr = cpu.regs[RAX] as u32;
        let Some(func) = self.syscall_names.get(&nr).cloned() else {
            self.note(&format!("UNMODELED syscall {nr:#x} (run stopped)"));
            return Some(HookAction::Stop);
        };
        // Kernel ABI: arg0 is in R10; stack args start above ret + shadow space.
        cpu.regs[ARG[0]] = cpu.regs[R10];
        let sab = cpu.regs[RSP].wrapping_add(0x28);
        Some(match self.nt.dispatch(&func, cpu, mem, sab) {
            NtDispatch::Status(s) => {
                self.note_nt(&func, cpu, s);
                cpu.regs[RAX] = s;
                cpu.regs[ARG[0]] = cpu.rip.wrapping_add(2);
                cpu.regs[R11] = cpu.rflags;
                HookAction::Continue
            }
            NtDispatch::Stop => {
                self.note(&format!("UNMODELED {func} (run stopped)"));
                HookAction::Stop
            }
        })
    }
}

const NT_STUB_LEN: usize = 24;
const SYSCALL_PLACEHOLDER_BASE: u32 = 0x7000;
const R10: usize = 10;
const R11: usize = 11;

/// Canonical Win10 x64 `Nt*` stub for syscall number `nr`, padded to
/// [`STUB_STRIDE`] with `int3`.
fn nt_stub_bytes(nr: u32) -> [u8; STUB_STRIDE as usize] {
    let mut b = [0xCCu8; STUB_STRIDE as usize];
    let n = nr.to_le_bytes();
    let code: [u8; NT_STUB_LEN] = [
        0x4C, 0x8B, 0xD1, // mov r10, rcx
        0xB8, n[0], n[1], n[2], n[3], // mov eax, nr
        0xF6, 0x04, 0x25, 0x08, 0x03, 0xFE, 0x7F, 0x01, // test byte [7FFE0308h], 1
        0x75, 0x03, // jne +3
        0x0F, 0x05, // syscall
        0xC3, // ret
        0xCD, 0x2E, // int 2Eh
        0xC3, // ret
    ];
    b[..NT_STUB_LEN].copy_from_slice(&code);
    b
}

/// `Memory` as the XPLM model's guest-memory interface.
struct MemAdapter<'a> {
    mem: &'a mut Memory,
    alloc_next: &'a mut u64,
}

impl GuestMem for MemAdapter<'_> {
    fn read(&self, addr: u64, buf: &mut [u8]) -> bool {
        self.mem.read(addr, buf).is_ok()
    }

    fn write(&mut self, addr: u64, data: &[u8]) -> bool {
        self.mem.write(addr, data).is_ok()
    }

    fn alloc(&mut self, len: usize) -> Option<u64> {
        let va = *self.alloc_next;
        let n = round_up((len as u64).max(1));
        self.mem.map(va, n, Perm::RW).ok()?;
        *self.alloc_next = va + n;
        Some(va)
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
            self.note(&format!("NETWORK-BLOCKED {dll}!{func} (run stopped)"));
            return HookAction::Stop;
        }
        let sab = Self::stack_arg_base(ev, cpu);
        if func.starts_with("MessageBox") {
            let wide = func.ends_with('W');
            let rd = |p| if wide { read_wstr(mem, p) } else { read_cstr(mem, p) };
            let (text, cap) = (rd(Self::arg(cpu, 1)), rd(Self::arg(cpu, 2)));
            self.note(&format!("UNMODELED {func} text={text:?} caption={cap:?} (run stopped)"));
            return HookAction::Stop;
        }
        if let Some(x) = self.xplm.clone() {
            let call = XCall {
                args: [Self::arg(cpu, 0), Self::arg(cpu, 1), Self::arg(cpu, 2), Self::arg(cpu, 3)],
                rsp: sab.wrapping_sub(0x28),
            };
            let mut gm = MemAdapter { mem, alloc_next: &mut self.alloc_next };
            match x.borrow_mut().dispatch(dll, func, &call, &mut gm) {
                XDispatch::NotXplm => {}
                XDispatch::Unlisted => return HookAction::Stop,
                XDispatch::Handled(r) => {
                    cpu.regs[RAX] = r.rax;
                    if let Some(v) = r.xmm0 {
                        cpu.xmm[0] = u128::from(v);
                    }
                    return Self::resume(ev, cpu, mem);
                }
            }
        }
        match self.dispatch(dll, func, cpu, mem, sab) {
            Dispatch::Stop => {
                self.note(&format!("UNMODELED {dll}!{func} (run stopped)"));
                HookAction::Stop
            }
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
    fn user_object_information_reports_interactive_winsta() {
        let (mut h, mut cpu, mut mem, sab) = harness();
        set_args(&mut cpu, [0x1, 2, T_DATA, 0x40]);
        mem.write_uint(sab, 8, T_DATA + 0x100).unwrap();
        assert_eq!(ret_of(h.dispatch("user32.dll", "GetUserObjectInformationW", &cpu, &mut mem, sab)), 1);
        assert_eq!(mem.read_uint(T_DATA + 0x100, 4).unwrap(), 16);
        assert_eq!(mem.read_uint(T_DATA, 2).unwrap(), u64::from(b'W'));
        // Unknown index fails; a too-small buffer fails.
        set_args(&mut cpu, [0x1, 99, T_DATA, 0x40]);
        assert_eq!(ret_of(h.dispatch("user32.dll", "GetUserObjectInformationW", &cpu, &mut mem, sab)), 0);
        set_args(&mut cpu, [0x1, 2, T_DATA, 4]);
        assert_eq!(ret_of(h.dispatch("user32.dll", "GetUserObjectInformationW", &cpu, &mut mem, sab)), 0);
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
    // ---- anti-debug / anti-VM clean answers ---------------------------------

    fn call(h: &mut WinApiHook, f: &str, cpu: &Cpu, mem: &mut Memory, s: u64) -> Dispatch {
        h.dispatch("kernel32.dll", f, cpu, mem, s)
    }

    #[test]
    fn sandbox_and_vm_modules_are_absent() {
        let (mut h, mut cpu, mut mem, s) = harness();
        let name = T_DATA + 0x100;
        for (api, n) in [
            ("GetModuleHandleA", "SbieDll.dll"),
            ("GetModuleHandleA", "sbiedll"),
            ("LoadLibraryA", "C:\\x\\dbghelp.dll"),
            ("GetModuleHandleA", "vboxhook.dll"),
            ("LoadLibraryA", "VMwareGuestLib.dll"),
            ("GetModuleHandleA", "SxIn.dll"),
        ] {
            mem.poke(name, format!("{n}\0").as_bytes()).unwrap();
            set_args(&mut cpu, [name, 0, 0, 0]);
            assert_eq!(ret_of(call(&mut h, api, &cpu, &mut mem, s)), 0, "{n} must be absent");
            assert_eq!(h.last_error, ERROR_MOD_NOT_FOUND);
        }
        // An ordinary system module still resolves.
        mem.poke(name, b"kernel32.dll\0").unwrap();
        assert_ne!(ret_of(call(&mut h, "GetModuleHandleA", &cpu, &mut mem, s)), 0);
        // The wide variants honour the same list.
        let w: Vec<u8> = "sbiedll.dll\0".encode_utf16().flat_map(u16::to_le_bytes).collect();
        mem.poke(name, &w).unwrap();
        assert_eq!(ret_of(call(&mut h, "LoadLibraryW", &cpu, &mut mem, s)), 0);
    }

    #[test]
    fn debug_registers_read_zero_and_set_is_noop() {
        let (mut h, mut cpu, mut mem, s) = harness();
        let ctx = T_DATA + 0x400;
        mem.write(ctx, &[0xAAu8; 0x4D0]).unwrap(); // dirty CONTEXT
        mem.write_uint(ctx + 0x30, 4, 0x0010_0010).unwrap(); // CONTEXT_DEBUG_REGISTERS
        set_args(&mut cpu, [0xFFFF_FFFF_FFFF_FFFE, ctx, 0, 0]);
        assert_eq!(ret_of(call(&mut h, "GetThreadContext", &cpu, &mut mem, s)), 1);
        for off in (0x48..0x78).step_by(8) {
            assert_eq!(mem.read_uint(ctx + off, 8).unwrap(), 0, "Dr slot {off:#x}");
        }
        assert_eq!(mem.read_uint(ctx + 0x78, 8).unwrap(), 0xAAAA_AAAA_AAAA_AAAA, "only DRx written");
        // SetThreadContext with armed breakpoints is accepted and changes nothing.
        mem.write_uint(ctx + 0x48, 8, 0x1234).unwrap();
        mem.write_uint(ctx + 0x70, 8, 0x401).unwrap();
        assert_eq!(ret_of(call(&mut h, "SetThreadContext", &cpu, &mut mem, s)), 1);
        // A request needing integer/control state is unmodeled, never half-answered.
        mem.write_uint(ctx + 0x30, 4, 0x0010_0003).unwrap();
        assert!(matches!(call(&mut h, "GetThreadContext", &cpu, &mut mem, s), Dispatch::Stop));
    }

    #[test]
    fn firmware_tables_carry_no_vm_signature() {
        let (mut h, mut cpu, mut mem, s) = harness();
        let buf = T_DATA + 0x800;
        const RSMB: u64 = 0x5253_4D42;
        // GetSystemFirmwareTable('RSMB', 0, buf, 0x1000)
        set_args(&mut cpu, [RSMB, 0, buf, 0x1000]);
        let n = ret_of(call(&mut h, "GetSystemFirmwareTable", &cpu, &mut mem, s));
        assert!(n > 8, "SMBIOS present");
        let mut raw = vec![0u8; n as usize];
        mem.read(buf, &mut raw).unwrap();
        let text = String::from_utf8_lossy(&raw).to_lowercase();
        for vm in ["vmware", "virtualbox", "vbox", "qemu", "xen", "innotek", "kvm", "parallels", "bochs", "hyper-v", "virtual"] {
            assert!(!text.contains(vm), "SMBIOS must not mention {vm}");
        }
        // Too small: the required size is returned, nothing is written.
        set_args(&mut cpu, [RSMB, 0, buf + 0x200, 4]);
        assert_eq!(ret_of(call(&mut h, "GetSystemFirmwareTable", &cpu, &mut mem, s)), n);
        assert_eq!(h.last_error, ERROR_INSUFFICIENT_BUFFER);
        // FIRM / ACPI enumerate to nothing; an ACPI table lookup is not-found.
        set_args(&mut cpu, [0x4649_524D, buf, 0x100, 0]);
        assert_eq!(ret_of(call(&mut h, "EnumSystemFirmwareTables", &cpu, &mut mem, s)), 0);
        set_args(&mut cpu, [0x4143_5049, 0x5041_4346, buf, 0x100]);
        assert_eq!(ret_of(call(&mut h, "GetSystemFirmwareTable", &cpu, &mut mem, s)), 0);
        // An unknown provider is unmodeled -> explicit stop.
        set_args(&mut cpu, [0x5858_5858, 0, buf, 0x100]);
        assert!(matches!(call(&mut h, "GetSystemFirmwareTable", &cpu, &mut mem, s), Dispatch::Stop));
    }

    #[test]
    fn debugger_probes_answer_clean() {
        let (mut h, mut cpu, mut mem, s) = harness();
        assert_eq!(ret_of(call(&mut h, "IsDebuggerPresent", &cpu, &mut mem, s)), 0);
        let out = T_DATA + 0x40;
        mem.write_uint(out, 4, 0xFFFF).unwrap();
        set_args(&mut cpu, [0xFFFF_FFFF_FFFF_FFFF, out, 0, 0]);
        assert_eq!(ret_of(call(&mut h, "CheckRemoteDebuggerPresent", &cpu, &mut mem, s)), 1);
        assert_eq!(mem.read_uint(out, 4).unwrap(), 0);
        set_args(&mut cpu, [T_DATA, 0, 0, 0]);
        assert!(matches!(call(&mut h, "OutputDebugStringA", &cpu, &mut mem, s), Dispatch::Void));
        assert!(matches!(call(&mut h, "OutputDebugStringW", &cpu, &mut mem, s), Dispatch::Void));
    }

    #[test]
    fn close_handle_invalid_is_false_without_exception() {
        let (mut h, mut cpu, mut mem, s) = harness();
        for bad in [0u64, 0xFFFF_FFFF_FFFF_FFFF, 0xDEAD_BEEF] {
            set_args(&mut cpu, [bad, 0, 0, 0]);
            assert_eq!(ret_of(call(&mut h, "CloseHandle", &cpu, &mut mem, s)), 0);
            assert_eq!(h.last_error, ERROR_INVALID_HANDLE);
        }
        // A handle we minted closes fine, and only once.
        h.nt.handles_for_test(0x44);
        set_args(&mut cpu, [0x44, 0, 0, 0]);
        assert_eq!(ret_of(call(&mut h, "CloseHandle", &cpu, &mut mem, s)), 1);
        assert_eq!(ret_of(call(&mut h, "CloseHandle", &cpu, &mut mem, s)), 0);
    }

    // ---- host identity models ------------------------------------------------
    const T_IMG: u64 = 0x0000_0030_0000_0000;

    fn put_w(mem: &mut Memory, at: u64, s: &str) {
        for (i, u) in s.encode_utf16().chain([0]).enumerate() {
            mem.write_uint(at + i as u64 * 2, 2, u64::from(u)).unwrap();
        }
    }

    fn rd_w(mem: &Memory, at: u64) -> String {
        read_wstr(mem, at).unwrap()
    }

    /// `harness` plus a mapped sample image with a minimal PE header.
    fn img_harness() -> (WinApiHook, Cpu, Memory, u64) {
        let (mut h, cpu, mut mem, s) = harness();
        mem.map(T_IMG, 0x1000, Perm::RW).unwrap();
        mem.write_uint(T_IMG + 0x3C, 4, 0x40).unwrap();
        mem.write_uint(T_IMG + 0x40 + 24 + 16, 4, 0x1000).unwrap(); // entry RVA
        mem.write_uint(T_IMG + 0x40 + 24 + 56, 4, 0x5000).unwrap(); // SizeOfImage
        h.image_base = T_IMG;
        (h, cpu, mem, s)
    }

    #[test]
    fn user_and_computer_names_are_benign() {
        let (mut h, mut cpu, mut mem, s) = harness();
        let (buf, pcb) = (T_DATA, T_DATA + 0x400);
        mem.write_uint(pcb, 4, 64).unwrap();
        set_args(&mut cpu, [buf, pcb, 0, 0]);
        assert_eq!(ret_of(h.dispatch("advapi32.dll", "GetUserNameW", &cpu, &mut mem, s)), 1);
        assert_eq!(rd_w(&mem, buf), "user");
        assert_eq!(mem.read_uint(pcb, 4).unwrap(), 5, "count includes the NUL");
        // Too small: fails with the required size + ERROR_INSUFFICIENT_BUFFER.
        mem.write_uint(pcb, 4, 2).unwrap();
        assert_eq!(ret_of(h.dispatch("advapi32.dll", "GetUserNameW", &cpu, &mut mem, s)), 0);
        assert_eq!(mem.read_uint(pcb, 4).unwrap(), 5);
        assert_eq!(h.last_error, ERROR_INSUFFICIENT_BUFFER);
        // ANSI computer name; success count excludes the NUL.
        mem.write_uint(pcb, 4, 64).unwrap();
        assert_eq!(ret_of(h.dispatch("kernel32.dll", "GetComputerNameA", &cpu, &mut mem, s)), 1);
        assert_eq!(read_cstr(&mem, buf).unwrap(), "DESKTOP-GENERIC");
        assert_eq!(mem.read_uint(pcb, 4).unwrap(), 15);
        // Ex: DNS hostname = host, DNS domain = empty, unmodeled format stops.
        set_args(&mut cpu, [1, buf, pcb, 0]);
        mem.write_uint(pcb, 4, 64).unwrap();
        assert_eq!(ret_of(h.dispatch("kernel32.dll", "GetComputerNameExW", &cpu, &mut mem, s)), 1);
        assert_eq!(rd_w(&mem, buf), "DESKTOP-GENERIC");
        set_args(&mut cpu, [2, buf, pcb, 0]);
        mem.write_uint(pcb, 4, 64).unwrap();
        assert_eq!(ret_of(h.dispatch("kernel32.dll", "GetComputerNameExW", &cpu, &mut mem, s)), 1);
        assert_eq!(rd_w(&mem, buf), "");
        set_args(&mut cpu, [99, buf, pcb, 0]);
        assert!(matches!(h.dispatch("kernel32.dll", "GetComputerNameExW", &cpu, &mut mem, s), Dispatch::Stop));
    }

    #[test]
    fn registry_hides_sandbox_keys_and_serves_benign_values() {
        const HKLM: u64 = 0xFFFF_FFFF_8000_0002;
        let (mut h, mut cpu, mut mem, s) = harness();
        let (sub, phk) = (T_DATA, T_DATA + 0x800);
        mem.write_uint(s, 8, phk).unwrap(); // arg5 phkResult
        for key in [
            "SOFTWARE\\Oracle\\VirtualBox Guest Additions",
            "SOFTWARE\\VMware, Inc.\\VMware Tools",
            "SYSTEM\\CurrentControlSet\\Services\\VBoxGuest",
            "SOFTWARE\\Sandboxie",
        ] {
            put_w(&mut mem, sub, key);
            set_args(&mut cpu, [HKLM, sub, 0, 0x20019]);
            assert_eq!(ret_of(h.dispatch("advapi32.dll", "RegOpenKeyExW", &cpu, &mut mem, s)), 2, "{key}");
            assert_eq!(mem.read_uint(phk, 8).unwrap(), 0);
        }
        // A normal key opens; a value reads back; the handle closes once.
        put_w(&mut mem, sub, "SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion");
        set_args(&mut cpu, [HKLM, sub, 0, 0x20019]);
        assert_eq!(ret_of(h.dispatch("advapi32.dll", "RegOpenKeyExW", &cpu, &mut mem, s)), 0);
        let hk = mem.read_uint(phk, 8).unwrap();
        assert_ne!(hk, 0);
        let (name, data, cb) = (T_DATA + 0x100, T_DATA + 0x200, T_DATA + 0x300);
        put_w(&mut mem, name, "ProductName");
        mem.write_uint(cb, 4, 0x100).unwrap();
        mem.write_uint(s, 8, data).unwrap(); // arg5 lpData
        mem.write_uint(s + 8, 8, cb).unwrap(); // arg6 lpcbData
        set_args(&mut cpu, [hk, name, 0, T_DATA + 0x380]); // lpType
        assert_eq!(ret_of(h.dispatch("advapi32.dll", "RegQueryValueExW", &cpu, &mut mem, s)), 0);
        assert_eq!(rd_w(&mem, data), "Windows 10 Pro");
        assert_eq!(mem.read_uint(T_DATA + 0x380, 4).unwrap(), 1, "REG_SZ");
        // Missing value -> NOT_FOUND; too-small buffer -> MORE_DATA with size.
        put_w(&mut mem, name, "NoSuchValue");
        assert_eq!(ret_of(h.dispatch("advapi32.dll", "RegQueryValueExW", &cpu, &mut mem, s)), 2);
        put_w(&mut mem, name, "ProductName");
        mem.write_uint(cb, 4, 4).unwrap();
        assert_eq!(ret_of(h.dispatch("advapi32.dll", "RegQueryValueExW", &cpu, &mut mem, s)), 234);
        assert_eq!(mem.read_uint(cb, 4).unwrap(), 30);
        set_args(&mut cpu, [hk, 0, 0, 0]);
        assert_eq!(ret_of(h.dispatch("advapi32.dll", "RegCloseKey", &cpu, &mut mem, s)), 0);
        assert_eq!(ret_of(h.dispatch("advapi32.dll", "RegCloseKey", &cpu, &mut mem, s)), 6);
    }

    #[test]
    fn reg_get_value_dword_and_type_filter() {
        const HKLM: u64 = 0xFFFF_FFFF_8000_0002;
        let (mut h, mut cpu, mut mem, s) = harness();
        let (sub, val, data, cb) = (T_DATA, T_DATA + 0x100, T_DATA + 0x200, T_DATA + 0x300);
        put_w(&mut mem, sub, "SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion");
        put_w(&mut mem, val, "CurrentMajorVersionNumber");
        mem.write_uint(cb, 4, 4).unwrap();
        mem.write_uint(s, 8, 0).unwrap(); // pdwType
        mem.write_uint(s + 8, 8, data).unwrap();
        mem.write_uint(s + 16, 8, cb).unwrap();
        set_args(&mut cpu, [HKLM, sub, val, 0x10]); // RRF_RT_REG_DWORD
        assert_eq!(ret_of(h.dispatch("advapi32.dll", "RegGetValueW", &cpu, &mut mem, s)), 0);
        assert_eq!(mem.read_uint(data, 4).unwrap(), 10);
        set_args(&mut cpu, [HKLM, sub, val, 0x2]); // wants REG_SZ
        assert_eq!(ret_of(h.dispatch("advapi32.dll", "RegGetValueW", &cpu, &mut mem, s)), 1630);
        // ANSI path with a sandbox subkey.
        mem.write(sub, b"SOFTWARE\\Oracle\\VirtualBox Guest Additions\0").unwrap();
        mem.write(val, b"Version\0").unwrap();
        set_args(&mut cpu, [HKLM, sub, val, 0xFFFF]);
        assert_eq!(ret_of(h.dispatch("advapi32.dll", "RegGetValueA", &cpu, &mut mem, s)), 2);
    }

    #[test]
    fn toolhelp_process_list_is_clean() {
        let (mut h, mut cpu, mut mem, s) = img_harness();
        set_args(&mut cpu, [2, 0, 0, 0]);
        let snap = ret_of(h.dispatch("kernel32.dll", "CreateToolhelp32Snapshot", &cpu, &mut mem, s));
        assert_ne!(snap, INVALID_HANDLE_VALUE);
        let pe = T_DATA;
        let mut names = Vec::new();
        let mut api = "Process32FirstW";
        loop {
            mem.write_uint(pe, 4, 568).unwrap();
            set_args(&mut cpu, [snap, pe, 0, 0]);
            if ret_of(h.dispatch("kernel32.dll", api, &cpu, &mut mem, s)) == 0 {
                break;
            }
            api = "Process32NextW";
            names.push((mem.read_uint(pe + 8, 4).unwrap(), rd_w(&mem, pe + 44)));
        }
        assert_eq!(h.last_error, ERROR_NO_MORE_FILES);
        assert!(names.contains(&(crate::ntdll::FAKE_PID, "program.exe".to_string())));
        for (_, n) in &names {
            let n = n.to_lowercase();
            for bad in ["dbg", "ida", "wireshark", "procmon", "vbox", "vmware", "sandbox", "fiddler", "hacker"] {
                assert!(!n.contains(bad), "{n}");
            }
        }
        // Wrong dwSize is rejected; the snapshot closes; a heap snapshot is unmodeled.
        mem.write_uint(pe, 4, 100).unwrap();
        set_args(&mut cpu, [snap, pe, 0, 0]);
        assert_eq!(ret_of(h.dispatch("kernel32.dll", "Process32FirstW", &cpu, &mut mem, s)), 0);
        assert_eq!(h.last_error, ERROR_BAD_LENGTH);
        set_args(&mut cpu, [snap, 0, 0, 0]);
        assert_eq!(ret_of(h.dispatch("kernel32.dll", "CloseHandle", &cpu, &mut mem, s)), 1);
        set_args(&mut cpu, [1, 0, 0, 0]);
        assert!(matches!(h.dispatch("kernel32.dll", "CreateToolhelp32Snapshot", &cpu, &mut mem, s), Dispatch::Stop));
    }

    #[test]
    fn get_system_info_fills_a_fixed_x64_struct() {
        let (mut h, mut cpu, mut mem, s) = img_harness();
        set_args(&mut cpu, [T_DATA, 0, 0, 0]);
        h.dispatch("kernel32.dll", "GetSystemInfo", &cpu, &mut mem, s);
        assert_eq!(mem.read_uint(T_DATA, 2).unwrap(), 9);
        assert_eq!(mem.read_uint(T_DATA + 4, 4).unwrap(), 0x1000);
        assert_eq!(mem.read_uint(T_DATA + 32, 4).unwrap(), 4);
    }

    #[test]
    fn thread_snapshot_lists_only_the_contained_thread() {
        let (mut h, mut cpu, mut mem, s) = img_harness();
        let te = T_DATA;
        set_args(&mut cpu, [4, 0, 0, 0]);
        let snap = ret_of(h.dispatch("kernel32.dll", "CreateToolhelp32Snapshot", &cpu, &mut mem, s));
        mem.write_uint(te, 4, 28).unwrap();
        set_args(&mut cpu, [snap, te, 0, 0]);
        assert_eq!(ret_of(h.dispatch("kernel32.dll", "Thread32First", &cpu, &mut mem, s)), 1);
        assert_eq!(mem.read_uint(te + 8, 4).unwrap(), crate::ntdll::FAKE_TID);
        assert_eq!(mem.read_uint(te + 12, 4).unwrap(), crate::ntdll::FAKE_PID);
        assert_eq!(ret_of(h.dispatch("kernel32.dll", "Thread32Next", &cpu, &mut mem, s)), 0);
        assert_eq!(h.last_error, ERROR_NO_MORE_FILES);
        mem.write_uint(te, 4, 100).unwrap();
        assert_eq!(ret_of(h.dispatch("kernel32.dll", "Thread32First", &cpu, &mut mem, s)), 0);
        assert_eq!(h.last_error, ERROR_BAD_LENGTH);
    }

    #[test]
    fn enum_process_modules_and_module_snapshot_agree() {
        let (mut h, mut cpu, mut mem, s) = img_harness();
        let (list, need, out) = (T_DATA, T_DATA + 0x100, T_DATA + 0x200);
        set_args(&mut cpu, [u64::MAX, list, 0x100, need]);
        assert_eq!(ret_of(h.dispatch("psapi.dll", "EnumProcessModules", &cpu, &mut mem, s)), 1);
        let n = (mem.read_uint(need, 4).unwrap() / 8) as usize;
        let mods: Vec<u64> = (0..n).map(|i| mem.read_uint(list + i as u64 * 8, 8).unwrap()).collect();
        assert_eq!(mods[0], T_IMG, "sample first");
        assert_eq!(mods.len(), 4, "sample + ntdll + kernel32 + kernelbase");
        // Consistent with GetModuleHandle.
        mem.write(out, b"KERNEL32.DLL\0").unwrap();
        set_args(&mut cpu, [out, 0, 0, 0]);
        let k32 = ret_of(h.dispatch("kernel32.dll", "GetModuleHandleA", &cpu, &mut mem, s));
        assert!(mods.contains(&k32));
        // GetModuleFileNameExW names the module; GetModuleInformation sizes it.
        set_args(&mut cpu, [u64::MAX, k32, out, 0x100]);
        assert!(ret_of(h.dispatch("psapi.dll", "GetModuleFileNameExW", &cpu, &mut mem, s)) > 0);
        assert_eq!(rd_w(&mem, out), "C:\\Windows\\System32\\kernel32.dll");
        set_args(&mut cpu, [u64::MAX, T_IMG, out, 24]);
        assert_eq!(ret_of(h.dispatch("psapi.dll", "GetModuleInformation", &cpu, &mut mem, s)), 1);
        assert_eq!(mem.read_uint(out, 8).unwrap(), T_IMG);
        assert_eq!(mem.read_uint(out + 8, 4).unwrap(), 0x5000);
        assert_eq!(mem.read_uint(out + 16, 8).unwrap(), T_IMG + 0x1000);
        // Too small a buffer still reports the needed size.
        set_args(&mut cpu, [u64::MAX, list, 8, need]);
        assert_eq!(ret_of(h.dispatch("psapi.dll", "EnumProcessModules", &cpu, &mut mem, s)), 1);
        assert_eq!(mem.read_uint(need, 4).unwrap(), 32);
        // A foreign process handle is unmodeled.
        set_args(&mut cpu, [0x1234, list, 0x100, need]);
        assert!(matches!(h.dispatch("psapi.dll", "EnumProcessModules", &cpu, &mut mem, s), Dispatch::Stop));
        // Module snapshot lists the same set.
        set_args(&mut cpu, [0x8, 0, 0, 0]);
        let snap = ret_of(h.dispatch("kernel32.dll", "CreateToolhelp32Snapshot", &cpu, &mut mem, s));
        let (me, mut api, mut seen) = (T_DATA + 0x400, "Module32FirstW", Vec::new());
        loop {
            mem.write_uint(me, 4, 1080).unwrap();
            set_args(&mut cpu, [snap, me, 0, 0]);
            if ret_of(h.dispatch("kernel32.dll", api, &cpu, &mut mem, s)) == 0 {
                break;
            }
            api = "Module32NextW";
            seen.push(mem.read_uint(me + 24, 8).unwrap());
        }
        assert_eq!(seen, mods);
    }

    #[test]
    fn module_handle_ex_and_artifact_modules() {
        let (mut h, mut cpu, mut mem, s) = img_harness();
        let (name, out) = (T_DATA, T_DATA + 0x100);
        put_w(&mut mem, name, "ntdll.dll");
        set_args(&mut cpu, [0, name, out, 0]);
        assert_eq!(ret_of(h.dispatch("kernel32.dll", "GetModuleHandleExW", &cpu, &mut mem, s)), 1);
        mem.write(name, b"ntdll.dll\0").unwrap();
        set_args(&mut cpu, [name, 0, 0, 0]);
        let plain = ret_of(h.dispatch("kernel32.dll", "GetModuleHandleA", &cpu, &mut mem, s));
        assert_eq!(mem.read_uint(out, 8).unwrap(), plain, "Ex agrees with GetModuleHandle");
        // FROM_ADDRESS inside the sample resolves to its base; outside any module fails.
        set_args(&mut cpu, [4, T_IMG + 0x10, out, 0]);
        assert_eq!(ret_of(h.dispatch("kernel32.dll", "GetModuleHandleExW", &cpu, &mut mem, s)), 1);
        assert_eq!(mem.read_uint(out, 8).unwrap(), T_IMG);
        set_args(&mut cpu, [4, T_DATA, out, 0]);
        assert_eq!(ret_of(h.dispatch("kernel32.dll", "GetModuleHandleExW", &cpu, &mut mem, s)), 0);
        // Sandbox/VM modules are never present, by either path.
        for m in ["SbieDll.dll", "vboxmrxnp.dll", "C:\\x\\VBoxHook.dll"] {
            let mut b = m.as_bytes().to_vec();
            b.push(0);
            mem.write(name, &b).unwrap();
            set_args(&mut cpu, [name, 0, 0, 0]);
            assert_eq!(ret_of(h.dispatch("kernel32.dll", "GetModuleHandleA", &cpu, &mut mem, s)), 0, "{m}");
            assert_eq!(h.last_error, ERROR_MOD_NOT_FOUND);
            assert_eq!(ret_of(h.dispatch("kernel32.dll", "LoadLibraryA", &cpu, &mut mem, s)), 0, "{m}");
        }
    }

    #[test]
    fn unhandled_exception_filter_stores_previous_and_debugbreak_stops() {
        let (mut h, mut cpu, mut mem, s) = harness();
        set_args(&mut cpu, [0, 0, 0, 0]);
        assert_eq!(ret_of(h.dispatch("kernel32.dll", "UnhandledExceptionFilter", &cpu, &mut mem, s)), 1);
        set_args(&mut cpu, [0xAAAA, 0, 0, 0]);
        assert_eq!(ret_of(h.dispatch("kernel32.dll", "SetUnhandledExceptionFilter", &cpu, &mut mem, s)), 0);
        set_args(&mut cpu, [0xBBBB, 0, 0, 0]);
        assert_eq!(ret_of(h.dispatch("kernel32.dll", "SetUnhandledExceptionFilter", &cpu, &mut mem, s)), 0xAAAA);
        // Cannot run a stored guest filter: fail closed.
        assert!(matches!(h.dispatch("kernel32.dll", "UnhandledExceptionFilter", &cpu, &mut mem, s), Dispatch::Stop));
        assert!(matches!(h.dispatch("kernel32.dll", "DebugBreak", &cpu, &mut mem, s), Dispatch::Stop));
    }

    // ---- ntdll syscall-stub fidelity ----------------------------------------

    const NT_PROTECT_NR: u32 = 0x50;
    const T_CODE: u64 = 0x0000_0030_0000_0000;

    fn ntdll_hook() -> WinApiHook {
        let names = ["NtProtectVirtualMemory", "ZwProtectVirtualMemory", "NtMadeUpCall"];
        let reference = BTreeMap::from([(
            "ntdll.dll".to_string(),
            names.iter().map(|n| SymbolOrOrdinal::Name((*n).into())).collect(),
        )]);
        let numbers = BTreeMap::from([
            ("NtProtectVirtualMemory".to_string(), NT_PROTECT_NR),
            ("ZwProtectVirtualMemory".to_string(), NT_PROTECT_NR),
            ("NtMadeUpCall".to_string(), 0x1234),
        ]);
        WinApiHook::new(BTreeMap::new())
            .with_export_reference(reference)
            .with_syscall_numbers(numbers)
    }

    fn export_va(h: &WinApiHook, name: &str) -> u64 {
        *h.export_stub_addrs.iter().find(|(_, (_, f))| f == name).map(|(va, _)| va).unwrap()
    }

    /// Run `NtProtectVirtualMemory(-1, &base, &size, PAGE_READONLY, &old)` against
    /// `target` (the export VA, or an RX copy of its bytes) and return
    /// (rax, old protection written, resulting perm, stop).
    fn run_nt_protect(via_copy: bool) -> (u64, u64, Option<Perm>, crate::emu::Stop) {
        use iced_x86::code_asm::*;
        let mut h = ntdll_hook();
        let mut mem = Memory::new();
        mem.map(T_STACK, 0x8000, Perm::RW).unwrap();
        mem.map(T_DATA, 0x8000, Perm::RW).unwrap();
        mem.map(T_CODE, 0x1000, Perm::RWX).unwrap();
        h.module_handle(&mut mem, "ntdll.dll");
        let export = export_va(&h, "NtProtectVirtualMemory");
        let target = if via_copy {
            let mut stub = [0u8; 32];
            mem.read(export, &mut stub).unwrap();
            mem.poke(T_CODE + 0x800, &stub).unwrap();
            T_CODE + 0x800
        } else {
            export
        };
        // [T_DATA]=region base, [T_DATA+8]=size, [T_DATA+16]=old protect.
        let region = T_DATA + 0x2000;
        mem.write_uint(T_DATA, 8, region).unwrap();
        mem.write_uint(T_DATA + 8, 8, 0x1000).unwrap();
        let mut a = CodeAssembler::new(64).unwrap();
        a.sub(rsp, 0x28).unwrap();
        a.mov(rcx, -1i64).unwrap();
        a.mov(rdx, T_DATA).unwrap();
        a.mov(r8, T_DATA + 8).unwrap();
        a.mov(r9d, 2).unwrap();
        a.mov(rax, T_DATA + 16).unwrap();
        a.mov(qword_ptr(rsp + 0x20), rax).unwrap();
        a.mov(rax, target).unwrap();
        a.call(rax).unwrap();
        a.add(rsp, 0x28).unwrap();
        a.nop().unwrap();
        let code = a.assemble(T_CODE).unwrap();
        mem.poke(T_CODE, &code).unwrap();
        let end = T_CODE + code.len() as u64 - 1;
        let mut cpu = Cpu::new(T_CODE);
        cpu.regs[RSP] = T_STACK + 0x4000;
        let mut emu = crate::emu::Emulator::new(cpu, mem);
        emu.hook = Some(Box::new(h));
        let stop = emu.run_until(end, 1000);
        let old = emu.mem.read_uint(T_DATA + 16, 4).unwrap();
        (emu.cpu.regs[RAX], old, emu.mem.perm_at(region), stop)
    }

    #[test]
    fn ntdll_exports_have_real_stub_bytes() {
        let mut h = ntdll_hook();
        let mut mem = Memory::new();
        h.module_handle(&mut mem, "ntdll.dll");
        for name in ["NtProtectVirtualMemory", "ZwProtectVirtualMemory"] {
            let mut b = [0u8; 32];
            mem.read(export_va(&h, name), &mut b).unwrap();
            assert_eq!(b[..4], [0x4C, 0x8B, 0xD1, 0xB8], "{name} mov r10,rcx; mov eax");
            assert_eq!(u32::from_le_bytes(b[4..8].try_into().unwrap()), NT_PROTECT_NR);
            assert!(b[8..].windows(2).any(|w| w == [0x0F, 0x05]), "{name} has syscall");
            assert!(b.iter().any(|&x| x != 0));
        }
    }

    #[test]
    fn syscall_path_matches_name_keyed_handler() {
        let by_name = run_nt_protect(false);
        let by_syscall = run_nt_protect(true);
        assert_eq!(by_name, by_syscall);
        assert_eq!(by_syscall.0, 0, "STATUS_SUCCESS");
        assert_eq!(by_syscall.1, 0x04, "old protection was PAGE_READWRITE");
        assert_eq!(by_syscall.2, Some(Perm::R), "region is now read-only");
        assert_eq!(by_syscall.3, crate::emu::Stop::Reached);
    }

    #[test]
    fn unmodeled_and_unknown_syscalls_fail_closed() {
        use crate::emu::Stop;
        let run = |nr: u32| {
            let mut h = ntdll_hook();
            let mut mem = Memory::new();
            mem.map(T_CODE, 0x1000, Perm::RWX).unwrap();
            h.module_handle(&mut mem, "ntdll.dll");
            // eax = nr; syscall
            let mut code = vec![0xB8];
            code.extend(nr.to_le_bytes());
            code.extend([0x0F, 0x05, 0x90]);
            mem.poke(T_CODE, &code).unwrap();
            let mut emu = crate::emu::Emulator::new(Cpu::new(T_CODE), mem);
            emu.hook = Some(Box::new(h));
            emu.run_until(T_CODE + 7, 100)
        };
        assert_eq!(run(0x1234), Stop::Hook, "known stub, no NT handler");
        assert_eq!(run(0x9999), Stop::Hook, "number not backed by any stub");
        // No hook at all: unsupported, never a guessed return.
        let mut mem = Memory::new();
        mem.map(T_CODE, 0x1000, Perm::RWX).unwrap();
        mem.poke(T_CODE, &[0x0F, 0x05]).unwrap();
        let mut emu = crate::emu::Emulator::new(Cpu::new(T_CODE), mem);
        assert!(matches!(emu.run(10), Stop::Unsupported { .. }));
    }
}
