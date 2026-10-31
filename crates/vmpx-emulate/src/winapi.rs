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
use crate::memory::{Memory, Perm};
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
            handle_names: BTreeMap::new(),
            name_handles: BTreeMap::new(),
            procs: BTreeMap::new(),
        }
    }

    /// Build a hook from a loaded image, copying its import-stub table.
    #[must_use]
    pub fn from_image(img: &LoadedImage) -> Self {
        Self::new(img.import_stubs.clone())
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

    /// Deterministic fake HMODULE for `name` (lowercased), allocating one on
    /// first sight and remembering the reverse mapping.
    fn module_handle(&mut self, name: &str) -> u64 {
        let key = name.to_lowercase();
        if let Some(&h) = self.name_handles.get(&key) {
            return h;
        }
        let h = self.hmodule_next;
        self.hmodule_next += 0x1_0000;
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
                Dispatch::Ret(self.module_handle(&name))
            }
            (_, "GetModuleHandleW" | "LoadLibraryW") => {
                let p = Self::arg(cpu, 0);
                let name = if p == 0 { "<self>".to_string() } else {
                    match read_wstr(mem, p) {
                        Some(s) => s,
                        None => return Dispatch::Stop,
                    }
                };
                Dispatch::Ret(self.module_handle(&name))
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
