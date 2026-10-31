//! In-model emulation of the ntdll `Nt*`/`Zw*` syscall stubs a `VMProtect`
//! runtime calls to map and unpack its payload.
//!
//! # Containment
//!
//! Every handler is a pure in-model computation over the emulator's [`Memory`].
//! Section objects and views are in-emulator byte buffers mapped into guest
//! pages — no host NT call, file, socket, process or FFI boundary is ever
//! touched. Memory a call reserves/commits/reprotects maps real *guest* pages.
//! A class, handle or file we cannot model SOUNDLY fails closed (an error
//! NTSTATUS, or [`NtDispatch::Stop`]) rather than being guessed: a wrong
//! mapping would corrupt the unpack, so only what the call specifies is mapped.
//!
//! # ABI
//!
//! The System-V-unlike Win64 NT stub takes args in RCX/RDX/R8/R9 and the rest
//! on the stack; the handler returns an NTSTATUS in RAX and the stub `ret`s.
//! Dispatch happens from [`crate::winapi`] when a guest `call`/`jmp` targets a
//! resolved ntdll export stub VA, reusing the Win32-stub return mechanism.

use crate::cpu::Cpu;
use crate::memory::{Memory, Perm};

// GPR indices into `Cpu::regs` (match winapi.rs).
const ARG: [usize; 4] = [1, 2, 8, 9];

// --- NTSTATUS -------------------------------------------------------------
const STATUS_SUCCESS: u64 = 0x0000_0000;
const STATUS_INFO_LENGTH_MISMATCH: u64 = 0xC000_0004;
const STATUS_INVALID_INFO_CLASS: u64 = 0xC000_0003;
const STATUS_OBJECT_NAME_NOT_FOUND: u64 = 0xC000_0034;
const STATUS_INVALID_HANDLE: u64 = 0xC000_0008;
const STATUS_PORT_NOT_SET: u64 = 0xC000_0353;
const STATUS_BUFFER_TOO_SMALL: u64 = 0xC000_0023;
const STATUS_NOT_FOUND: u64 = 0xC000_0225;

// --- Win32 page protection constants --------------------------------------
const PAGE_NOACCESS: u64 = 0x01;
const PAGE_READONLY: u64 = 0x02;
const PAGE_READWRITE: u64 = 0x04;
const PAGE_EXECUTE: u64 = 0x10;
const PAGE_EXECUTE_READ: u64 = 0x20;
const PAGE_EXECUTE_READWRITE: u64 = 0x40;

// --- MEMORY_BASIC_INFORMATION State/Type ----------------------------------
const MEM_COMMIT: u64 = 0x0000_1000;
const MEM_FREE: u64 = 0x0001_0000;
const MEM_PRIVATE: u64 = 0x0002_0000;

const PAGE: u64 = 0x1000;

// Deterministic plausible identifiers reported by the info queries. The process
// id is shared between ProcessBasicInformation and the thread's CLIENT_ID so a
// cross-check sees a consistent view.
pub(crate) const FAKE_PID: u64 = 0x0000_01A4;
pub(crate) const FAKE_TID: u64 = 0x0000_01A8;
const STATUS_PENDING: u64 = 0x0000_0103;

// Deterministic in-model address pools, far from the loader regions and from
// winapi.rs's own pools (ALLOC_BASE 0x50.., HMODULE 0x70..) so maps never
// collide.
const NT_ALLOC_BASE: u64 = 0x0000_0080_0000_0000; // NtAllocateVirtualMemory
const NT_VIEW_BASE: u64 = 0x0000_0090_0000_0000; // NtMapViewOfSection views
const NT_HANDLE_BASE: u64 = 0x0000_0000_0000_4000; // minted object handles

// Cap a single section/allocation so a bogus size cannot exhaust the map.
const MAX_REGION: u64 = 0x1000_0000; // 256 MiB

fn round_up(v: u64) -> u64 {
    v.div_ceil(PAGE) * PAGE
}

/// Win32 `PAGE_*` protection -> guest [`Perm`]. Only the low byte (the access
/// bits) is honoured; `PAGE_GUARD`/`PAGE_NOCACHE`/`PAGE_WRITECOMBINE` modifiers
/// are ignored. Execute implies read (classic x64 paging has no exec-only).
fn perm_of(protect: u64) -> Perm {
    match protect & 0xff {
        PAGE_NOACCESS => Perm(0),
        PAGE_READONLY => Perm::R,
        PAGE_EXECUTE | PAGE_EXECUTE_READ => Perm::RX,
        PAGE_EXECUTE_READWRITE | 0x80 => Perm::RWX,
        // READWRITE (0x04), WRITECOPY (0x08) or anything unrecognized -> RW.
        _ => Perm::RW,
    }
}

/// Guest [`Perm`] -> the `PAGE_*` value NT reports (for `OldProtect` / MBI).
fn protect_of(perm: Perm) -> u64 {
    let (r, w, x) = (perm.contains(Perm::R), perm.contains(Perm::W), perm.contains(Perm::X));
    match (x, w, r) {
        (true, true, _) => PAGE_EXECUTE_READWRITE,
        (true, false, _) => PAGE_EXECUTE_READ,
        (false, true, _) => PAGE_READWRITE,
        (false, false, true) => PAGE_READONLY,
        (false, false, false) => PAGE_NOACCESS,
    }
}

/// What a dispatched NT stub resolved to.
pub(crate) enum NtDispatch {
    /// Set RAX to this NTSTATUS and `ret`.
    Status(u64),
    /// Not modeled, or a fault while modeling it: stop the run (never guess).
    Stop,
}

/// A section object: a flat byte buffer that views map into guest memory.
struct Section {
    bytes: Vec<u8>,
    /// Default page protection requested at creation (used when a view passes 0).
    protect: u64,
}

/// In-model NT object + memory state, owned by the `WinApiHook`.
#[derive(Default)]
pub(crate) struct NtState {
    sections: std::collections::BTreeMap<u64, Section>,
    /// Live non-section handles (files etc.) we minted, for `NtClose` validation.
    handles: std::collections::BTreeSet<u64>,
    next_handle: u64,
    alloc_next: u64,
    view_next: u64,
}

/// Reads NT call arguments: the first four from registers, the rest from the
/// guest stack starting at `stack_arg_base` (the 5th argument's slot).
struct Args<'a> {
    cpu: &'a Cpu,
    mem: &'a Memory,
    stack_arg_base: u64,
}

impl Args<'_> {
    fn reg(&self, i: usize) -> u64 {
        self.cpu.regs[ARG[i]]
    }
    /// 0-based argument `i`; `i >= 4` reads `[stack_arg_base + (i-4)*8]`.
    fn get(&self, i: usize) -> Option<u64> {
        if i < 4 {
            Some(self.reg(i))
        } else {
            self.mem.read_uint(self.stack_arg_base + (i as u64 - 4) * 8, 8).ok()
        }
    }
}

impl NtState {
    pub(crate) fn new() -> Self {
        Self {
            next_handle: NT_HANDLE_BASE,
            alloc_next: NT_ALLOC_BASE,
            view_next: NT_VIEW_BASE,
            ..Self::default()
        }
    }

    fn mint_handle(&mut self) -> u64 {
        let h = self.next_handle;
        self.next_handle += 4;
        h
    }

    /// Dispatch a resolved ntdll stub by its exported name. `Nt`/`Zw` share one
    /// implementation (strip the 2-char prefix). `stack_arg_base` is the guest
    /// address of the 5th argument's slot (the caller computes it per flow kind).
    pub(crate) fn dispatch(
        &mut self,
        func: &str,
        cpu: &Cpu,
        mem: &mut Memory,
        stack_arg_base: u64,
    ) -> NtDispatch {
        let base = func.get(2..).unwrap_or(func); // drop "Nt"/"Zw"
        match base {
            "AllocateVirtualMemory" => self.allocate(cpu, mem, stack_arg_base),
            "ProtectVirtualMemory" => Self::protect_vm(cpu, mem, stack_arg_base),
            "QueryVirtualMemory" => Self::query_vm(cpu, mem, stack_arg_base),
            "CreateSection" => self.create_section(cpu, mem, stack_arg_base),
            "MapViewOfSection" => self.map_view(cpu, mem, stack_arg_base),
            "OpenSection" => NtDispatch::Status(STATUS_OBJECT_NAME_NOT_FOUND),
            "Close" => self.close(cpu),
            "QueryInformationProcess" => Self::query_process(cpu, mem, stack_arg_base),
            "QueryInformationThread" => Self::query_thread(cpu, mem, stack_arg_base),
            "QuerySystemInformation" => Self::query_system(cpu, mem, stack_arg_base),
            // Debug registers read as zero; a set is accepted as a no-op (no DRx
            // breakpoint is ever armed). Anything beyond DEBUG_REGISTERS is unmodeled.
            "GetContextThread" | "SetContextThread" => {
                if debug_context(mem, cpu.regs[ARG[1]], base == "GetContextThread") {
                    NtDispatch::Status(STATUS_SUCCESS)
                } else {
                    NtDispatch::Stop
                }
            }
            "OpenFile" => Self::open_file(cpu, mem, stack_arg_base),
            // No in-model registry: opening any key fails closed (NOT_FOUND),
            // never fabricating a handle or key contents.
            "OpenKey" | "OpenKeyEx" => Self::open_key(cpu, mem),
            "RaiseHardError" => Self::raise_hard_error(cpu, mem, stack_arg_base),
            // A view unmap, thread/process info set, or sleep: a benign success.
            "UnmapViewOfSection"
            | "SetInformationThread"
            | "SetInformationProcess"
            | "DelayExecution" => NtDispatch::Status(STATUS_SUCCESS),
            _ => NtDispatch::Stop,
        }
    }

    /// `NtAllocateVirtualMemory(h, *Base, ZeroBits, *RegionSize, Type, Protect)`.
    fn allocate(&mut self, cpu: &Cpu, mem: &mut Memory, sab: u64) -> NtDispatch {
        let a = Args { cpu, mem: &*mem, stack_arg_base: sab };
        let p_base = a.reg(1);
        let p_size = a.reg(3);
        let Some(protect) = a.get(5) else {
            return NtDispatch::Stop;
        };
        let (Ok(req_base), Ok(size)) = (mem.read_uint(p_base, 8), mem.read_uint(p_size, 8)) else {
            return NtDispatch::Stop;
        };
        if size == 0 || size > MAX_REGION {
            return NtDispatch::Status(STATUS_INVALID_INFO_CLASS);
        }
        let perm = perm_of(protect);
        let (va, len) = if req_base == 0 {
            let len = round_up(size);
            let va = self.alloc_next;
            if mem.map(va, len, perm).is_err() {
                return NtDispatch::Stop;
            }
            self.alloc_next += len;
            (va, len)
        } else {
            let va = req_base & !(PAGE - 1);
            let len = round_up(req_base + size) - va;
            if ensure_mapped(mem, va, len, perm).is_err() {
                return NtDispatch::Stop;
            }
            (va, len)
        };
        if mem.write_uint(p_base, 8, va).is_err() || mem.write_uint(p_size, 8, len).is_err() {
            return NtDispatch::Stop;
        }
        NtDispatch::Status(STATUS_SUCCESS)
    }

    /// `NtProtectVirtualMemory(h, *Base, *RegionSize, NewProtect, *OldProtect)`.
    fn protect_vm(cpu: &Cpu, mem: &mut Memory, sab: u64) -> NtDispatch {
        let a = Args { cpu, mem: &*mem, stack_arg_base: sab };
        let p_base = a.reg(1);
        let p_size = a.reg(2);
        let new_protect = a.reg(3);
        let Some(p_old) = a.get(4) else {
            return NtDispatch::Stop;
        };
        let (Ok(req_base), Ok(size)) = (mem.read_uint(p_base, 8), mem.read_uint(p_size, 8)) else {
            return NtDispatch::Stop;
        };
        let va = req_base & !(PAGE - 1);
        let len = round_up(req_base + size) - va;
        let old = mem.perm_at(va).map_or(PAGE_NOACCESS, protect_of);
        if mem.protect(va, len, perm_of(new_protect)).is_err() {
            return NtDispatch::Stop;
        }
        if p_old != 0 && mem.write_uint(p_old, 4, old).is_err() {
            return NtDispatch::Stop;
        }
        // NT writes the page-aligned base/size back to the in/out pointers.
        let _ = mem.write_uint(p_base, 8, va);
        let _ = mem.write_uint(p_size, 8, len);
        NtDispatch::Status(STATUS_SUCCESS)
    }

    /// `NtQueryVirtualMemory(h, Base, Class, Buffer, Length, *ReturnLength)`.
    fn query_vm(cpu: &Cpu, mem: &mut Memory, sab: u64) -> NtDispatch {
        let a = Args { cpu, mem: &*mem, stack_arg_base: sab };
        let addr = a.reg(1);
        let class = a.reg(2);
        let buf = a.reg(3);
        let (Some(length), Some(p_ret)) = (a.get(4), a.get(5)) else {
            return NtDispatch::Stop;
        };
        // Only MemoryBasicInformation (class 0) is modeled.
        if class != 0 {
            return NtDispatch::Status(STATUS_INVALID_INFO_CLASS);
        }
        if length < 48 {
            return NtDispatch::Status(STATUS_INFO_LENGTH_MISMATCH);
        }
        let mbi = basic_memory_info(mem, addr);
        if mem.write(buf, &mbi).is_err() {
            return NtDispatch::Stop;
        }
        if p_ret != 0 {
            let _ = mem.write_uint(p_ret, 8, 48);
        }
        NtDispatch::Status(STATUS_SUCCESS)
    }

    /// `NtCreateSection(*Handle, Access, *ObjAttr, *MaxSize, Protect, Attr, File)`.
    fn create_section(&mut self, cpu: &Cpu, mem: &mut Memory, sab: u64) -> NtDispatch {
        let a = Args { cpu, mem: &*mem, stack_arg_base: sab };
        let p_handle = a.reg(0);
        let p_maxsize = a.reg(3);
        let (Some(page_protect), Some(file_handle)) = (a.get(4), a.get(6)) else {
            return NtDispatch::Stop;
        };
        // Only anonymous (pagefile-backed) sections are modeled soundly; a
        // file/image-backed section needs file bytes we do not have -> fail closed.
        if file_handle != 0 {
            return NtDispatch::Status(STATUS_INVALID_HANDLE);
        }
        let size = if p_maxsize != 0 {
            match mem.read_uint(p_maxsize, 8) {
                Ok(v) => v,
                Err(_) => return NtDispatch::Stop,
            }
        } else {
            0
        };
        if size == 0 || size > MAX_REGION {
            return NtDispatch::Status(STATUS_INVALID_INFO_CLASS);
        }
        let h = self.mint_handle();
        self.sections.insert(h, Section { bytes: vec![0u8; size as usize], protect: page_protect });
        if mem.write_uint(p_handle, 8, h).is_err() {
            return NtDispatch::Stop;
        }
        NtDispatch::Status(STATUS_SUCCESS)
    }

    /// `NtMapViewOfSection(Section, Process, *Base, ZeroBits, Commit, *Offset,
    /// *ViewSize, Inherit, AllocType, Win32Protect)`.
    fn map_view(&mut self, cpu: &Cpu, mem: &mut Memory, sab: u64) -> NtDispatch {
        let a = Args { cpu, mem: &*mem, stack_arg_base: sab };
        let section = a.reg(0);
        let p_base = a.reg(2);
        let (Some(p_offset), Some(p_viewsize), Some(win32_protect)) =
            (a.get(5), a.get(6), a.get(9))
        else {
            return NtDispatch::Stop;
        };
        let Some(sec) = self.sections.get(&section) else {
            return NtDispatch::Status(STATUS_INVALID_HANDLE);
        };
        let sec_len = sec.bytes.len() as u64;
        let protect = if win32_protect != 0 { win32_protect } else { sec.protect };
        let offset = read_opt(mem, p_offset).unwrap_or(0).min(sec_len);
        let want = read_opt(mem, p_viewsize).filter(|&v| v != 0).unwrap_or(sec_len - offset);
        let view_len = round_up(want.min(sec_len - offset).max(1));
        let req_base = mem.read_uint(p_base, 8).unwrap_or(0);
        let va = if req_base == 0 {
            let va = self.view_next;
            self.view_next += round_up(view_len);
            va
        } else {
            req_base & !(PAGE - 1)
        };
        if ensure_mapped(mem, va, view_len, perm_of(protect)).is_err() {
            return NtDispatch::Stop;
        }
        // Copy the section's bytes into the mapped view (tail stays zero).
        let copy = (sec_len - offset).min(view_len) as usize;
        let bytes = self.sections[&section].bytes[offset as usize..offset as usize + copy].to_vec();
        if mem.poke(va, &bytes).is_err() {
            return NtDispatch::Stop;
        }
        if mem.write_uint(p_base, 8, va).is_err() {
            return NtDispatch::Stop;
        }
        if p_viewsize != 0 {
            let _ = mem.write_uint(p_viewsize, 8, view_len);
        }
        NtDispatch::Status(STATUS_SUCCESS)
    }

    /// `NtClose(Handle)`. Drops any object we minted; a pseudo-handle (current
    /// process/thread) closes as a no-op success. An unknown handle returns
    /// `STATUS_INVALID_HANDLE` with NO exception: the no-debugger behaviour (a
    /// debugger would turn it into `EXCEPTION_INVALID_HANDLE`).
    fn close(&mut self, cpu: &Cpu) -> NtDispatch {
        let h = cpu.regs[ARG[0]];
        if self.sections.remove(&h).is_some() || self.handles.remove(&h) || is_pseudo_handle(h) {
            NtDispatch::Status(STATUS_SUCCESS)
        } else {
            NtDispatch::Status(STATUS_INVALID_HANDLE)
        }
    }

    #[cfg(test)]
    pub(crate) fn handles_for_test(&mut self, h: u64) {
        self.handles.insert(h);
    }

    /// Whether `h` is a live handle we minted (for the Win32 `CloseHandle`).
    pub(crate) fn owns_handle(&self, h: u64) -> bool {
        self.sections.contains_key(&h) || self.handles.contains(&h)
    }

    /// `NtQueryInformationProcess(h, Class, Buffer, Length, *ReturnLength)`.
    fn query_process(cpu: &Cpu, mem: &mut Memory, sab: u64) -> NtDispatch {
        let a = Args { cpu, mem: &*mem, stack_arg_base: sab };
        let class = a.reg(1);
        let buf = a.reg(2);
        let length = a.reg(3);
        let p_ret = a.get(4).unwrap_or(0);
        let peb = mem.read_uint(cpu.gs_base + 0x60, 8).unwrap_or(0);
        let (bytes, status): (Vec<u8>, u64) = match class {
            // ProcessBasicInformation: 48 bytes; only `PebBaseAddress` matters.
            0 => {
                if length < 48 {
                    return NtDispatch::Status(STATUS_INFO_LENGTH_MISMATCH);
                }
                let mut b = vec![0u8; 48];
                b[8..16].copy_from_slice(&peb.to_le_bytes()); // PebBaseAddress
                b[32..40].copy_from_slice(&FAKE_PID.to_le_bytes()); // UniqueProcessId
                (b, STATUS_SUCCESS)
            }
            // ProcessDebugPort (7): 0 == no debugger.
            7 => (vec![0u8; 8], STATUS_SUCCESS),
            // ProcessDebugObjectHandle (30): no debug object.
            30 => (vec![0u8; 8], STATUS_PORT_NOT_SET),
            // ProcessDebugFlags (31): 1 == not being debugged (no inherit).
            31 => (1u32.to_le_bytes().to_vec(), STATUS_SUCCESS),
            // ProcessBreakOnTermination (29): not a critical process.
            29 => (0u32.to_le_bytes().to_vec(), STATUS_SUCCESS),
            _ => return NtDispatch::Status(STATUS_INVALID_INFO_CLASS),
        };
        if buf != 0 && (length as usize) >= bytes.len() && mem.write(buf, &bytes).is_err() {
            return NtDispatch::Stop;
        }
        if p_ret != 0 {
            let _ = mem.write_uint(p_ret, 4, bytes.len() as u64);
        }
        NtDispatch::Status(status)
    }

    /// `NtQueryInformationThread(h, Class, Buffer, Length, *ReturnLength)`.
    fn query_thread(cpu: &Cpu, mem: &mut Memory, sab: u64) -> NtDispatch {
        let a = Args { cpu, mem: &*mem, stack_arg_base: sab };
        let class = a.reg(1);
        let buf = a.reg(2);
        let length = a.reg(3);
        let p_ret = a.get(4).unwrap_or(0);
        let bytes: Vec<u8> = match class {
            // ThreadBasicInformation (0): 48-byte `THREAD_BASIC_INFORMATION`; the
            // protector reads `TebBaseAddress` (gs base) and the `CLIENT_ID`.
            0 => {
                if length < 48 {
                    return NtDispatch::Status(STATUS_INFO_LENGTH_MISMATCH);
                }
                let mut b = vec![0u8; 48];
                b[0..4].copy_from_slice(&(STATUS_PENDING as u32).to_le_bytes()); // ExitStatus
                b[8..16].copy_from_slice(&cpu.gs_base.to_le_bytes()); // TebBaseAddress
                b[16..24].copy_from_slice(&FAKE_PID.to_le_bytes()); // ClientId.UniqueProcess
                b[24..32].copy_from_slice(&FAKE_TID.to_le_bytes()); // ClientId.UniqueThread
                b[32..40].copy_from_slice(&1u64.to_le_bytes()); // AffinityMask
                b
            }
            // ThreadHideFromDebugger (17): the query form returns the BOOLEAN state.
            17 => vec![0u8],
            _ => return NtDispatch::Status(STATUS_INVALID_INFO_CLASS),
        };
        if buf != 0 && (length as usize) >= bytes.len() && mem.write(buf, &bytes).is_err() {
            return NtDispatch::Stop;
        }
        if p_ret != 0 {
            let _ = mem.write_uint(p_ret, 4, bytes.len() as u64);
        }
        NtDispatch::Status(STATUS_SUCCESS)
    }

    /// `NtQuerySystemInformation(Class, Buffer, Length, *ReturnLength)`.
    fn query_system(cpu: &Cpu, mem: &mut Memory, sab: u64) -> NtDispatch {
        let a = Args { cpu, mem: &*mem, stack_arg_base: sab };
        let class = a.reg(0);
        let buf = a.reg(1);
        let length = a.reg(2);
        let p_ret = a.reg(3);
        // Anti-debug / anti-VM classes answer as a clean bare-metal host with no
        // kernel debugger; any other class is an explicit INVALID_INFO_CLASS.
        let bytes: Vec<u8> = match class {
            // SystemKernelDebuggerInformation (35): { KernelDebuggerEnabled,
            // KernelDebuggerNotPresent } = { FALSE, TRUE }.
            35 => vec![0x00, 0x01],
            // SystemKernelDebuggerInformationEx (149): { DebuggerAllowed,
            // DebuggerEnabled, DebuggerPresent } all FALSE.
            149 => vec![0, 0, 0],
            // SystemFirmwareTableInformation (76).
            76 => return Self::query_firmware(mem, buf, length, p_ret),
            _ => return NtDispatch::Status(STATUS_INVALID_INFO_CLASS),
        };
        if p_ret != 0 {
            let _ = mem.write_uint(p_ret, 4, bytes.len() as u64);
        }
        if (length as usize) < bytes.len() {
            return NtDispatch::Status(STATUS_INFO_LENGTH_MISMATCH);
        }
        if buf != 0 && mem.write(buf, &bytes).is_err() {
            return NtDispatch::Stop;
        }
        NtDispatch::Status(STATUS_SUCCESS)
    }

    /// `SystemFirmwareTableInformation`: a `SYSTEM_FIRMWARE_TABLE_INFORMATION`
    /// header {Provider, Action, TableID, TableBufferLength} then the table bytes.
    fn query_firmware(mem: &mut Memory, buf: u64, length: u64, p_ret: u64) -> NtDispatch {
        if length < FW_HEADER {
            return NtDispatch::Status(STATUS_INFO_LENGTH_MISMATCH);
        }
        let (Ok(provider), Ok(action), Ok(id)) =
            (mem.read_uint(buf, 4), mem.read_uint(buf + 4, 4), mem.read_uint(buf + 8, 4))
        else {
            return NtDispatch::Stop;
        };
        let payload = match firmware_table(provider as u32, action, id as u32) {
            FwResult::Data(d) => d,
            FwResult::NotFound => return NtDispatch::Status(STATUS_NOT_FOUND),
            FwResult::Unmodeled => return NtDispatch::Stop,
        };
        let need = FW_HEADER + payload.len() as u64;
        if mem.write_uint(buf + 12, 4, payload.len() as u64).is_err() {
            return NtDispatch::Stop;
        }
        if p_ret != 0 {
            let _ = mem.write_uint(p_ret, 4, need);
        }
        if length < need {
            return NtDispatch::Status(STATUS_BUFFER_TOO_SMALL);
        }
        if mem.write(buf + FW_HEADER, &payload).is_err() {
            return NtDispatch::Stop;
        }
        NtDispatch::Status(STATUS_SUCCESS)
    }

    /// `NtOpenFile(*Handle, Access, *ObjAttr, *IoStatus, Share, Options)`.
    ///
    /// We have no host filesystem and cannot soundly fabricate file contents, so
    /// opening any file fails closed. (A section the protector then tries to back
    /// with the returned handle also fails closed in `create_section`.)
    fn open_file(cpu: &Cpu, mem: &mut Memory, _sab: u64) -> NtDispatch {
        let p_handle = cpu.regs[ARG[0]];
        if p_handle != 0 {
            let _ = mem.write_uint(p_handle, 8, 0);
        }
        NtDispatch::Status(STATUS_OBJECT_NAME_NOT_FOUND)
    }

    /// `NtRaiseHardError(Status, NumberOfParameters, UnicodeStringParameterMask,
    /// Parameters, ValidResponseOptions, *Response)`. The hard error is reported
    /// as handled: write `ResponseReturnToCaller` (0) and succeed so a
    /// continuable (informational) error does not abort the run.
    fn raise_hard_error(cpu: &Cpu, mem: &mut Memory, sab: u64) -> NtDispatch {
        let status = cpu.regs[ARG[0]] as u32;
        // An error-severity hard error (top two bits set, 0xC0000000+) is meant
        // to terminate the process: fail closed rather than fabricate a resume.
        // A continuable (success/informational/warning) error returns to the
        // caller with ResponseReturnToCaller.
        if status >= 0xC000_0000 {
            return NtDispatch::Stop;
        }
        let a = Args { cpu, mem: &*mem, stack_arg_base: sab };
        if let Some(p_response) = a.get(5).filter(|&p| p != 0) {
            let _ = mem.write_uint(p_response, 4, 0); // ResponseReturnToCaller
        }
        NtDispatch::Status(STATUS_SUCCESS)
    }

    /// `NtOpenKey(*KeyHandle, DesiredAccess, *ObjectAttributes)`. No in-model
    /// registry: no handle is fabricated and the open fails closed.
    fn open_key(cpu: &Cpu, mem: &mut Memory) -> NtDispatch {
        let p_handle = cpu.regs[ARG[0]];
        if p_handle != 0 {
            let _ = mem.write_uint(p_handle, 8, 0);
        }
        NtDispatch::Status(STATUS_OBJECT_NAME_NOT_FOUND)
    }
}

/// `SYSTEM_FIRMWARE_TABLE_INFORMATION` header size; the table bytes follow it.
const FW_HEADER: u64 = 16;
// Firmware provider signatures (MSVC multi-char literals, first char = MSB).
const FW_FIRM: u32 = 0x4649_524D; // 'FIRM'
const FW_RSMB: u32 = 0x5253_4D42; // 'RSMB'
const FW_ACPI: u32 = 0x4143_5049; // 'ACPI'

/// Outcome of a firmware-table request.
pub(crate) enum FwResult {
    Data(Vec<u8>),
    /// The provider is known but has no such table.
    NotFound,
    /// A provider we do not model: fail closed rather than invent content.
    Unmodeled,
}

/// A minimal bare-metal SMBIOS (`RawSMBIOSData`): one Type-1 System Information
/// structure naming a generic desktop board, then the Type-127 terminator. No
/// hypervisor / virtual-machine vendor string appears anywhere.
fn smbios_bare_metal() -> Vec<u8> {
    let mut t = vec![1u8, 27, 0x01, 0x00, 1, 2, 3, 4];
    t.extend_from_slice(&[
        0x5A, 0xC3, 0x11, 0x7E, 0x90, 0x2B, 0x4D, 0x18, 0xA6, 0x3F, 0x0C, 0xD2, 0x71, 0xE8, 0x44,
        0x09,
    ]);
    t.extend_from_slice(&[6, 0, 0]); // wake-up type (power switch), SKU, family
    for s in ["ASUSTeK COMPUTER INC.", "System Product Name", "System Version", "System Serial Number"]
    {
        t.extend_from_slice(s.as_bytes());
        t.push(0);
    }
    t.push(0);
    t.extend_from_slice(&[127, 4, 0x02, 0x00, 0, 0]);
    let mut raw = vec![0u8, 3, 3, 0];
    raw.extend_from_slice(&(t.len() as u32).to_le_bytes());
    raw.extend_from_slice(&t);
    raw
}

/// Firmware table `id` of `provider` (`action`: 0 = enumerate, 1 = get). RSMB
/// exposes the single bare-metal SMBIOS table (id 0); FIRM and ACPI expose no
/// tables (empty, never a VM signature).
pub(crate) fn firmware_table(provider: u32, action: u64, id: u32) -> FwResult {
    match (provider, action) {
        (FW_RSMB, 0) => FwResult::Data(0u32.to_le_bytes().to_vec()),
        (FW_RSMB, 1) if id == 0 => FwResult::Data(smbios_bare_metal()),
        (FW_FIRM | FW_ACPI, 0) => FwResult::Data(Vec::new()),
        (FW_RSMB | FW_FIRM | FW_ACPI, 1) => FwResult::NotFound,
        _ => FwResult::Unmodeled,
    }
}

// x64 `CONTEXT`: ContextFlags, and Dr0-Dr3, Dr6, Dr7 contiguous from 0x48.
const CTX_FLAGS: u64 = 0x30;
const CTX_DR0: u64 = 0x48;
const CTX_DR_LEN: usize = 0x30;
const CONTEXT_AMD64: u64 = 0x10_0000;
const CONTEXT_DEBUG_REGISTERS: u64 = CONTEXT_AMD64 | 0x10;

/// Service a Get/SetThreadContext that requests only `CONTEXT_DEBUG_REGISTERS`:
/// a get returns Dr0-Dr7 as zero, a set is a no-op. `false` for any request
/// that needs more of the context (unmodeled; the caller fails closed).
pub(crate) fn debug_context(mem: &mut Memory, ctx: u64, get: bool) -> bool {
    let Ok(flags) = mem.read_uint(ctx + CTX_FLAGS, 4) else {
        return false;
    };
    if flags & CONTEXT_AMD64 == 0 || flags & !CONTEXT_DEBUG_REGISTERS != 0 {
        return false;
    }
    !get || flags & 0x10 == 0 || mem.write(ctx + CTX_DR0, &[0u8; CTX_DR_LEN]).is_ok()
}

/// The current-process (-1) / current-thread (-2) pseudo-handles.
fn is_pseudo_handle(h: u64) -> bool {
    h == u64::MAX || h == u64::MAX - 1
}

/// Read a `u64` through `ptr` when non-null, else `None`.
fn read_opt(mem: &Memory, ptr: u64) -> Option<u64> {
    if ptr == 0 {
        None
    } else {
        mem.read_uint(ptr, 8).ok()
    }
}

/// Ensure `[base, base+len)` is mapped with at least `perm`: map the unmapped
/// pages and reprotect the already-mapped ones (an NT commit/reprotect).
fn ensure_mapped(mem: &mut Memory, base: u64, len: u64, perm: Perm) -> Result<(), ()> {
    let mut a = base;
    let end = base.checked_add(len).ok_or(())?;
    while a < end {
        if mem.perm_at(a).is_some() {
            mem.protect(a, PAGE, perm).map_err(|_| ())?;
        } else {
            mem.map(a, PAGE, perm).map_err(|_| ())?;
        }
        a = a.checked_add(PAGE).ok_or(())?;
    }
    Ok(())
}

/// Build a 48-byte x64 `MEMORY_BASIC_INFORMATION` for `addr` from the live page
/// map: the contiguous run of same-permission mapped pages (committed/private),
/// or the free gap up to the next mapping.
fn basic_memory_info(mem: &Memory, addr: u64) -> [u8; 48] {
    let page = addr & !(PAGE - 1);
    let mut b = [0u8; 48];
    if let Some(perm) = mem.perm_at(page) {
        // Extend down and up over pages with the same permission.
        let mut lo = page;
        while lo >= PAGE && mem.perm_at(lo - PAGE) == Some(perm) {
            lo -= PAGE;
        }
        let mut hi = page + PAGE;
        while mem.perm_at(hi) == Some(perm) {
            hi += PAGE;
        }
        let protect = protect_of(perm);
        b[0..8].copy_from_slice(&lo.to_le_bytes()); // BaseAddress
        b[8..16].copy_from_slice(&lo.to_le_bytes()); // AllocationBase
        b[16..20].copy_from_slice(&(protect as u32).to_le_bytes()); // AllocationProtect
        b[24..32].copy_from_slice(&(hi - lo).to_le_bytes()); // RegionSize
        b[32..36].copy_from_slice(&(MEM_COMMIT as u32).to_le_bytes()); // State
        b[36..40].copy_from_slice(&(protect as u32).to_le_bytes()); // Protect
        b[40..44].copy_from_slice(&(MEM_PRIVATE as u32).to_le_bytes()); // Type
    } else {
        // Free region: bound it by the next mapped page above (best-effort).
        let next = mem.pages().map(|(base, _, _)| base).find(|&base| base > page);
        let size = next.map_or(PAGE, |n| n - page);
        b[0..8].copy_from_slice(&page.to_le_bytes()); // BaseAddress
        b[24..32].copy_from_slice(&size.to_le_bytes()); // RegionSize
        b[32..36].copy_from_slice(&(MEM_FREE as u32).to_le_bytes()); // State
        b[36..40].copy_from_slice(&(PAGE_NOACCESS as u32).to_le_bytes()); // Protect
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::Cpu;

    const STACK: u64 = 0x0000_0010_0000_0000;
    const STACK_LEN: u64 = 0x4000;
    const RSP: u64 = STACK + 0x2000;

    /// Build a CPU with NT args in RCX/RDX/R8/R9 and (optionally) on the stack.
    fn cpu_with(reg: [u64; 4], stack: &[u64], mem: &mut Memory) -> (Cpu, u64) {
        let mut cpu = Cpu::new(0);
        cpu.regs[ARG[0]] = reg[0];
        cpu.regs[ARG[1]] = reg[1];
        cpu.regs[ARG[2]] = reg[2];
        cpu.regs[ARG[3]] = reg[3];
        cpu.regs[4] = RSP; // RSP index
        let sab = RSP + 0x20;
        for (i, &v) in stack.iter().enumerate() {
            mem.write_uint(sab + i as u64 * 8, 8, v).unwrap();
        }
        (cpu, sab)
    }

    fn fresh_mem() -> Memory {
        let mut mem = Memory::new();
        mem.map(STACK, STACK_LEN, Perm::RW).unwrap();
        mem
    }

    #[test]
    fn allocate_maps_a_usable_region() {
        let mut mem = fresh_mem();
        // out-params: *BaseAddress and *RegionSize live on the stack scratch.
        let p_base = STACK + 0x100;
        let p_size = STACK + 0x108;
        mem.write_uint(p_base, 8, 0).unwrap(); // let NT choose the base
        mem.write_uint(p_size, 8, 0x3000).unwrap();
        let (cpu, sab) = cpu_with(
            [0xFFFF_FFFF_FFFF_FFFF, p_base, 0, p_size],
            &[0x3000, PAGE_EXECUTE_READWRITE], // AllocationType, Protect
            &mut mem,
        );
        let mut nt = NtState::new();
        let r = nt.dispatch("NtAllocateVirtualMemory", &cpu, &mut mem, sab);
        assert!(matches!(r, NtDispatch::Status(STATUS_SUCCESS)));
        let base = mem.read_uint(p_base, 8).unwrap();
        assert_eq!(base, NT_ALLOC_BASE);
        // The region is mapped RWX and both readable/writable/executable.
        assert_eq!(mem.perm_at(base), Some(Perm::RWX));
        mem.write_uint(base, 8, 0xdead_beef).unwrap();
        assert_eq!(mem.read_uint(base, 8).unwrap(), 0xdead_beef);
    }

    #[test]
    fn protect_changes_page_perms() {
        let mut mem = fresh_mem();
        let region = 0x0000_0001_0000_0000;
        mem.map(region, 0x2000, Perm::RW).unwrap();
        let p_base = STACK + 0x100;
        let p_size = STACK + 0x108;
        let p_old = STACK + 0x110;
        mem.write_uint(p_base, 8, region).unwrap();
        mem.write_uint(p_size, 8, 0x2000).unwrap();
        let (cpu, sab) =
            cpu_with([0xFFFF_FFFF_FFFF_FFFF, p_base, p_size, PAGE_EXECUTE_READ], &[p_old], &mut mem);
        let mut nt = NtState::new();
        let r = nt.dispatch("NtProtectVirtualMemory", &cpu, &mut mem, sab);
        assert!(matches!(r, NtDispatch::Status(STATUS_SUCCESS)));
        // Pages are now RX, and the old protect (RW) was written back.
        assert_eq!(mem.perm_at(region), Some(Perm::RX));
        assert_eq!(mem.perm_at(region + PAGE), Some(Perm::RX));
        assert_eq!(mem.read_uint(p_old, 4).unwrap(), PAGE_READWRITE);
    }

    #[test]
    fn map_view_copies_section_bytes_into_guest_memory() {
        let mut mem = fresh_mem();
        // Create a 0x2000-byte anonymous section.
        let p_handle = STACK + 0x100;
        let p_maxsize = STACK + 0x108;
        mem.write_uint(p_maxsize, 8, 0x2000).unwrap();
        let (cpu, sab) = cpu_with(
            [p_handle, 0, 0, p_maxsize],
            &[PAGE_READWRITE, 0, 0], // PageProtect, AllocAttr, FileHandle=0
            &mut mem,
        );
        let mut nt = NtState::new();
        assert!(matches!(
            nt.dispatch("NtCreateSection", &cpu, &mut mem, sab),
            NtDispatch::Status(STATUS_SUCCESS)
        ));
        let handle = mem.read_uint(p_handle, 8).unwrap();
        // Seed section bytes directly so we can prove the view reflects them.
        nt.sections.get_mut(&handle).unwrap().bytes[..4].copy_from_slice(&[1, 2, 3, 4]);

        let p_base = STACK + 0x120;
        let p_view = STACK + 0x128;
        mem.write_uint(p_base, 8, 0).unwrap(); // NT chooses base
        mem.write_uint(p_view, 8, 0).unwrap(); // whole section
        let (cpu, sab) = cpu_with(
            [handle, 0, p_base, 0],
            &[0, 0, p_view, 0, 0, PAGE_READWRITE], // Commit, *Offset, *ViewSize, Inherit, AllocType, Win32Protect
            &mut mem,
        );
        assert!(matches!(
            nt.dispatch("NtMapViewOfSection", &cpu, &mut mem, sab),
            NtDispatch::Status(STATUS_SUCCESS)
        ));
        let va = mem.read_uint(p_base, 8).unwrap();
        assert_eq!(va, NT_VIEW_BASE);
        assert_eq!(mem.read_uint(va, 4).unwrap(), 0x0403_0201, "section bytes readable in view");
        assert_eq!(mem.read_uint(p_view, 8).unwrap(), 0x2000);
    }

    #[test]
    fn open_file_fails_closed() {
        let mut mem = fresh_mem();
        let p_handle = STACK + 0x100;
        mem.write_uint(p_handle, 8, 0xdead).unwrap();
        let (cpu, sab) = cpu_with([p_handle, 0, 0, 0], &[0, 0], &mut mem);
        let mut nt = NtState::new();
        let r = nt.dispatch("NtOpenFile", &cpu, &mut mem, sab);
        assert!(matches!(r, NtDispatch::Status(STATUS_OBJECT_NAME_NOT_FOUND)));
        assert_eq!(mem.read_uint(p_handle, 8).unwrap(), 0, "no handle fabricated");
        // A section backed by a (non-existent) file handle also fails closed.
        let p_h2 = STACK + 0x108;
        let p_max = STACK + 0x110;
        mem.write_uint(p_max, 8, 0x1000).unwrap();
        let (cpu, sab) = cpu_with([p_h2, 0, 0, p_max], &[PAGE_READWRITE, 0, 0x1234], &mut mem);
        let r = nt.dispatch("NtCreateSection", &cpu, &mut mem, sab);
        assert!(matches!(r, NtDispatch::Status(STATUS_INVALID_HANDLE)));
    }

    #[test]
    fn open_key_fails_closed() {
        let mut mem = fresh_mem();
        let p_handle = STACK + 0x100;
        mem.write_uint(p_handle, 8, 0xdead).unwrap();
        let (cpu, sab) = cpu_with([p_handle, 0, 0, 0], &[], &mut mem);
        let mut nt = NtState::new();
        let r = nt.dispatch("NtOpenKey", &cpu, &mut mem, sab);
        assert!(matches!(r, NtDispatch::Status(STATUS_OBJECT_NAME_NOT_FOUND)));
        assert_eq!(mem.read_uint(p_handle, 8).unwrap(), 0, "no key handle fabricated");
    }

    #[test]
    fn raise_hard_error_continues_or_fails_by_severity() {
        let mut mem = fresh_mem();
        let p_resp = STACK + 0x100;
        mem.write_uint(p_resp, 8, 0xFF).unwrap();
        // Informational (continuable) status: return to caller, Response = 0.
        let (cpu, sab) =
            cpu_with([0x5000_0018, 4, 0, 0], &[0 /*opts*/, p_resp /*response*/], &mut mem);
        let mut nt = NtState::new();
        let r = nt.dispatch("NtRaiseHardError", &cpu, &mut mem, sab);
        assert!(matches!(r, NtDispatch::Status(STATUS_SUCCESS)));
        assert_eq!(mem.read_uint(p_resp, 4).unwrap(), 0, "ResponseReturnToCaller");
        // Error-severity (0xC0000000+) hard error fails closed.
        let (cpu, sab) = cpu_with([0xC000_0005, 0, 0, 0], &[0, 0], &mut mem);
        assert!(matches!(
            nt.dispatch("NtRaiseHardError", &cpu, &mut mem, sab),
            NtDispatch::Stop
        ));
    }

    #[test]
    fn query_vm_reports_committed_region() {
        let mut mem = fresh_mem();
        let region = 0x0000_0002_0000_0000;
        mem.map(region, 0x3000, Perm::RX).unwrap();
        let buf = STACK + 0x100;
        let p_ret = STACK + 0x140;
        let (cpu, sab) =
            cpu_with([0xFFFF_FFFF_FFFF_FFFF, region + 0x1000, 0, buf], &[48, p_ret], &mut mem);
        let mut nt = NtState::new();
        let r = nt.dispatch("NtQueryVirtualMemory", &cpu, &mut mem, sab);
        assert!(matches!(r, NtDispatch::Status(STATUS_SUCCESS)));
        assert_eq!(mem.read_uint(buf, 8).unwrap(), region, "BaseAddress = region start");
        assert_eq!(mem.read_uint(buf + 24, 8).unwrap(), 0x3000, "RegionSize");
        assert_eq!(mem.read_uint(buf + 32, 4).unwrap(), MEM_COMMIT, "State = COMMIT");
        assert_eq!(mem.read_uint(buf + 36, 4).unwrap(), PAGE_EXECUTE_READ, "Protect = RX");
    }
    fn sysinfo(class: u64, len: u64, buf: u64, p_ret: u64, mem: &mut Memory) -> u64 {
        let (cpu, sab) = cpu_with([class, buf, len, p_ret], &[], mem);
        match NtState::new().dispatch("NtQuerySystemInformation", &cpu, mem, sab) {
            NtDispatch::Status(s) => s,
            NtDispatch::Stop => u64::MAX,
        }
    }

    #[test]
    fn kernel_debugger_not_present() {
        let mut mem = fresh_mem();
        let (buf, p_ret) = (STACK + 0x100, STACK + 0x140);
        mem.write(buf, &[0xFF; 8]).unwrap();
        assert_eq!(sysinfo(35, 2, buf, p_ret, &mut mem), STATUS_SUCCESS);
        assert_eq!(mem.read_uint(buf, 1).unwrap(), 0, "KernelDebuggerEnabled");
        assert_eq!(mem.read_uint(buf + 1, 1).unwrap(), 1, "KernelDebuggerNotPresent");
        assert_eq!(mem.read_uint(p_ret, 4).unwrap(), 2);
        // The Ex form: allowed / enabled / present are all FALSE.
        mem.write(buf, &[0xFF; 8]).unwrap();
        assert_eq!(sysinfo(149, 3, buf, p_ret, &mut mem), STATUS_SUCCESS);
        assert_eq!(mem.read_uint(buf, 2).unwrap(), 0);
        assert_eq!(mem.read_uint(buf + 2, 1).unwrap(), 0);
        // A short buffer is a length mismatch, never a truncated lie.
        assert_eq!(sysinfo(35, 1, buf, p_ret, &mut mem), STATUS_INFO_LENGTH_MISMATCH);
    }

    #[test]
    fn firmware_query_is_bare_metal() {
        let mut mem = fresh_mem();
        let (buf, p_ret) = (STACK + 0x200, STACK + 0x140);
        // Header: Provider 'RSMB', Action = get (1), TableID 0.
        for (off, v) in [(0u64, FW_RSMB), (4, 1), (8, 0)] {
            mem.write_uint(buf + off, 4, u64::from(v)).unwrap();
        }
        assert_eq!(sysinfo(76, 0x1000, buf, p_ret, &mut mem), STATUS_SUCCESS);
        let n = mem.read_uint(buf + 12, 4).unwrap() as usize;
        assert!(n > 8);
        assert_eq!(mem.read_uint(p_ret, 4).unwrap() as usize, 16 + n);
        let mut raw = vec![0u8; n];
        mem.read(buf + 16, &mut raw).unwrap();
        let text = String::from_utf8_lossy(&raw).to_lowercase();
        for vm in ["vmware", "virtualbox", "vbox", "qemu", "xen", "innotek", "kvm", "hyper-v", "virtual"] {
            assert!(!text.contains(vm), "no {vm} signature");
        }
        // Too small: BUFFER_TOO_SMALL with the required size reported.
        assert_eq!(sysinfo(76, 20, buf, p_ret, &mut mem), STATUS_BUFFER_TOO_SMALL);
        assert_eq!(mem.read_uint(p_ret, 4).unwrap() as usize, 16 + n);
        // ACPI enumerate is empty; an unknown provider fails closed (Stop).
        mem.write_uint(buf, 4, u64::from(FW_ACPI)).unwrap();
        mem.write_uint(buf + 4, 4, 0).unwrap();
        assert_eq!(sysinfo(76, 0x100, buf, p_ret, &mut mem), STATUS_SUCCESS);
        assert_eq!(mem.read_uint(buf + 12, 4).unwrap(), 0);
        mem.write_uint(buf, 4, 0x5858_5858).unwrap();
        assert_eq!(sysinfo(76, 0x100, buf, p_ret, &mut mem), u64::MAX);
    }

    #[test]
    fn nt_context_debug_registers_zero() {
        let mut mem = fresh_mem();
        let ctx = STACK + 0x400;
        mem.write(ctx, &[0x55u8; 0x100]).unwrap();
        mem.write_uint(ctx + 0x30, 4, 0x0010_0010).unwrap();
        let (cpu, sab) = cpu_with([0xFFFF_FFFF_FFFF_FFFE, ctx, 0, 0], &[], &mut mem);
        let mut nt = NtState::new();
        assert!(matches!(
            nt.dispatch("NtGetContextThread", &cpu, &mut mem, sab),
            NtDispatch::Status(STATUS_SUCCESS)
        ));
        for off in (0x48..0x78).step_by(8) {
            assert_eq!(mem.read_uint(ctx + off, 8).unwrap(), 0);
        }
        assert!(matches!(
            nt.dispatch("ZwSetContextThread", &cpu, &mut mem, sab),
            NtDispatch::Status(STATUS_SUCCESS)
        ));
        mem.write_uint(ctx + 0x30, 4, 0x0010_0001).unwrap(); // CONTROL: unmodeled
        assert!(matches!(nt.dispatch("NtGetContextThread", &cpu, &mut mem, sab), NtDispatch::Stop));
    }

    #[test]
    fn debugger_info_classes_and_invalid_close() {
        let mut mem = fresh_mem();
        let buf = STACK + 0x100;
        let mut nt = NtState::new();
        let q = |class: u64, mem: &mut Memory, nt: &mut NtState| {
            mem.write(buf, &[0xFF; 8]).unwrap();
            let (cpu, sab) = cpu_with([0xFFFF_FFFF_FFFF_FFFF, class, buf, 8], &[0], mem);
            match nt.dispatch("NtQueryInformationProcess", &cpu, mem, sab) {
                NtDispatch::Status(s) => s,
                NtDispatch::Stop => u64::MAX,
            }
        };
        assert_eq!(q(7, &mut mem, &mut nt), STATUS_SUCCESS); // ProcessDebugPort
        assert_eq!(mem.read_uint(buf, 8).unwrap(), 0);
        assert_eq!(q(30, &mut mem, &mut nt), STATUS_PORT_NOT_SET); // DebugObjectHandle
        assert_eq!(mem.read_uint(buf, 8).unwrap(), 0);
        assert_eq!(q(31, &mut mem, &mut nt), STATUS_SUCCESS); // DebugFlags
        assert_eq!(mem.read_uint(buf, 4).unwrap(), 1, "1 == not debugged");
        // NtClose of a bogus handle: INVALID_HANDLE status, no exception/stop.
        let (cpu, sab) = cpu_with([0xDEAD_BEEF, 0, 0, 0], &[], &mut mem);
        assert!(matches!(
            nt.dispatch("NtClose", &cpu, &mut mem, sab),
            NtDispatch::Status(STATUS_INVALID_HANDLE)
        ));
    }
}
