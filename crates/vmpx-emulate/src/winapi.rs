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
use std::collections::BTreeMap;
use std::rc::Rc;

use crate::cpu::Cpu;
use crate::emu::{FlowEvent, FlowKind, Hook, HookAction};
use crate::loader::LoadedImage;
use crate::memory::{MapError, Memory, Perm};
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
const MODULE_IMAGE_SIZE: u64 = 0x1_0000;

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
        if map_module_image(mem, h, &key).is_err() {
            return 0;
        }
        self.hmodule_next += MODULE_IMAGE_SIZE;
        self.name_handles.insert(key.clone(), h);
        self.handle_names.insert(h, key);
        h
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
    fn on_flow(&mut self, ev: &FlowEvent, cpu: &mut Cpu, mem: &mut Memory) -> HookAction {
        if !matches!(ev.kind, FlowKind::Call | FlowKind::Jmp) {
            return HookAction::Continue;
        }
        let Some((dll, func)) = self.stubs.get(&ev.to).cloned() else {
            return HookAction::Continue; // not an import thunk
        };

        match self.dispatch(&dll, &func, cpu, mem) {
            Dispatch::Stop => HookAction::Stop,
            Dispatch::Ret(v) => {
                cpu.regs[RAX] = v;
                Self::resume(ev, cpu, mem)
            }
            Dispatch::Void => Self::resume(ev, cpu, mem),
        }
    }
}

impl WinApiHook {
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
            _ => HookAction::Continue,
        }
    }
}

/// Map a readable, structurally faithful PE-header image for a resolved module
/// at `base` and label its export directory with `name`.
///
/// This is the minimal shape the OS loader guarantees behind every HMODULE: a
/// DOS stub (`MZ` + `e_lfanew`), NT headers (`PE\0\0`, a PE32+ optional header
/// whose `ImageBase` is `base`), and a valid IMAGE_EXPORT_DIRECTORY. A module
/// resolved here exports *nothing* — the directory is well-formed but empty.
/// That is deliberate and sound: the module is faithfully present and its
/// headers parse, but we do not fabricate export addresses we do not have. A
/// protector that walks the table finds no entry and resolves NULL; any later
/// use of that NULL faults and stops the run rather than proceeding on a guessed
/// address. Everything is deterministic and self-contained (no host access).
fn map_module_image(mem: &mut Memory, base: u64, name: &str) -> Result<(), MapError> {
    const E_LFANEW: u32 = 0x40;
    const EXPORT_RVA: u32 = 0x1000;
    const NAME_RVA: u32 = 0x1100;
    const EMPTY_ARR_RVA: u32 = 0x1200;

    mem.map(base, MODULE_IMAGE_SIZE, Perm::R)?;
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
    w(opt + 56, &(MODULE_IMAGE_SIZE as u32).to_le_bytes()); // SizeOfImage
    w(opt + 108, &16u32.to_le_bytes()); // NumberOfRvaAndSizes
    // Data directory[0] = export table.
    w(opt + 112, &EXPORT_RVA.to_le_bytes());
    w(opt + 116, &0x28u32.to_le_bytes()); // size (one IMAGE_EXPORT_DIRECTORY)

    // IMAGE_EXPORT_DIRECTORY at EXPORT_RVA: well-formed, zero exports.
    let exp = u64::from(EXPORT_RVA);
    w(exp + 0x0C, &NAME_RVA.to_le_bytes()); // Name
    w(exp + 0x10, &1u32.to_le_bytes()); // Base (ordinal base)
    w(exp + 0x14, &0u32.to_le_bytes()); // NumberOfFunctions
    w(exp + 0x18, &0u32.to_le_bytes()); // NumberOfNames
    w(exp + 0x1C, &EMPTY_ARR_RVA.to_le_bytes()); // AddressOfFunctions
    w(exp + 0x20, &EMPTY_ARR_RVA.to_le_bytes()); // AddressOfNames
    w(exp + 0x24, &EMPTY_ARR_RVA.to_le_bytes()); // AddressOfNameOrdinals

    // Module name string (for a protector that reads the export Name field).
    let mut nm = name.as_bytes().to_vec();
    nm.push(0);
    w(u64::from(NAME_RVA), &nm);
    Ok(())
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
