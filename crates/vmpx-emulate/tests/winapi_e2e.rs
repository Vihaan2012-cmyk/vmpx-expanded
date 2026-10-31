//! Layer-2 end-to-end: drive hand-assembled guest code that calls import
//! thunks through the [`WinApiHook`], and assert deterministic returns, a clean
//! resume, balanced RSP, the captured import binding, a usable allocation, and
//! the stop on an unmodeled API.

use std::collections::BTreeMap;

use iced_x86::code_asm::*;
use vmpx_emulate::{
    Cpu, Emulator, HookAction, Memory, ObservedFacts, Perm, Stop, SymbolOrOrdinal, WinApiHook,
};

const CODE: u64 = 0x1000;
// Above 0x10000 so string pointers are not mistaken for MAKEINTRESOURCE ordinals.
const DATA: u64 = 0x2_0000;
const STACK_TOP: u64 = 0x0000_0010_0000_0000 + 0x40_000;
const STUB_LOADLIB: u64 = 0x0000_0020_0000_0000;
const STUB_GETPROC: u64 = 0x0000_0020_0000_0008;
const STUB_ALLOC: u64 = 0x0000_0020_0000_0010;
const STUB_UNMODELED: u64 = 0x0000_0020_0000_0018;
const STUB_GETMODULE: u64 = 0x0000_0020_0000_0020;
const RSP_START: u64 = STACK_TOP - 0x100;

/// Build an emulator over a one-page RX code image, a writable data/stack
/// region, and the four import stubs, with `rip`/`rsp` primed and a
/// `WinApiHook` installed. Returns the emulator and a handle to its facts.
fn harness(code: &[u8], data: &[(u64, Vec<u8>)]) -> (Emulator, std::rc::Rc<std::cell::RefCell<ObservedFacts>>) {
    let mut mem = Memory::new();
    mem.map(CODE, 0x1000, Perm::RX).unwrap();
    mem.poke(CODE, code).unwrap();
    mem.map(DATA, 0x1000, Perm::RW).unwrap();
    for (addr, bytes) in data {
        mem.poke(*addr, bytes).unwrap();
    }
    mem.map(STACK_TOP - 0x40_000, 0x40_000, Perm::RW).unwrap();

    let mut stubs = BTreeMap::new();
    stubs.insert(STUB_LOADLIB, ("kernel32.dll".into(), "LoadLibraryA".into()));
    stubs.insert(STUB_GETPROC, ("kernel32.dll".into(), "GetProcAddress".into()));
    stubs.insert(STUB_ALLOC, ("kernel32.dll".into(), "VirtualAlloc".into()));
    stubs.insert(STUB_UNMODELED, ("secret.dll".into(), "TotallyUnknownApi".into()));
    stubs.insert(STUB_GETMODULE, ("kernel32.dll".into(), "GetModuleHandleA".into()));

    let hook = WinApiHook::new(stubs);
    let facts = hook.facts();

    let mut cpu = Cpu::new(CODE);
    cpu.regs[4] = RSP_START; // RSP
    let mut emu = Emulator::new(cpu, mem);
    emu.hook = Some(Box::new(hook));
    (emu, facts)
}

/// `LoadLibraryA("user32.dll")`; `GetProcAddress(that, "MessageBoxA")`.
#[test]
fn getprocaddress_bridge_records_import_and_returns_deterministically() {
    let user32 = b"user32.dll\0".to_vec();
    let msgbox = b"MessageBoxA\0".to_vec();
    let name_user32 = DATA;
    let name_msgbox = DATA + 0x20;

    let mut a = CodeAssembler::new(64).unwrap();
    a.mov(rcx, name_user32).unwrap();
    a.mov(rax, STUB_LOADLIB).unwrap();
    a.call(rax).unwrap();
    a.mov(rcx, rax).unwrap(); // hmodule
    a.mov(rdx, name_msgbox).unwrap();
    a.mov(rax, STUB_GETPROC).unwrap();
    a.call(rax).unwrap();
    let code = a.assemble(CODE).unwrap();
    let end = CODE + code.len() as u64;

    let (mut emu, facts) =
        harness(&code, &[(name_user32, user32), (name_msgbox, msgbox)]);
    assert_eq!(emu.run_until(end, 1000), Stop::Reached, "ran to end of block");

    // Deterministic proc VA returned, RSP balanced across both call thunks.
    assert_ne!(emu.cpu.regs[0], 0, "GetProcAddress returned non-null");
    assert_eq!(emu.cpu.regs[4], RSP_START, "RSP balanced after two call thunks");

    let f = facts.borrow();
    assert_eq!(f.imports.len(), 1, "one GetProcAddress resolution recorded");
    let imp = &f.imports[0];
    assert_eq!(imp.module, "user32.dll");
    assert_eq!(imp.symbol, SymbolOrOrdinal::Name("MessageBoxA".into()));
    assert_eq!(imp.stub_va_or_slot, emu.cpu.regs[0], "recorded VA == returned proc VA");
}

/// `VirtualAlloc` returns a mapped, writable region.
#[test]
fn virtualalloc_returns_usable_mapped_region() {
    let mut a = CodeAssembler::new(64).unwrap();
    a.mov(rcx, 0u64).unwrap(); // lpAddress
    a.mov(rdx, 0x2000u64).unwrap(); // dwSize (2 pages)
    a.mov(rax, STUB_ALLOC).unwrap();
    a.call(rax).unwrap();
    let code = a.assemble(CODE).unwrap();
    let end = CODE + code.len() as u64;

    let (mut emu, _facts) = harness(&code, &[]);
    assert_eq!(emu.run_until(end, 1000), Stop::Reached);

    let va = emu.cpu.regs[0];
    assert_ne!(va, 0, "VirtualAlloc returned non-null");
    // The returned region is real guest memory: write then read it back.
    emu.mem.write_uint(va + 0x1234, 8, 0xDEAD_BEEF_CAFE_F00D).unwrap();
    assert_eq!(emu.mem.read_uint(va + 0x1234, 8).unwrap(), 0xDEAD_BEEF_CAFE_F00D);
}

/// An unmodeled (dll,func) hits the documented default: stop, never guess.
#[test]
fn unmodeled_import_stops_the_run() {
    let mut a = CodeAssembler::new(64).unwrap();
    a.mov(rax, STUB_UNMODELED).unwrap();
    a.call(rax).unwrap();
    let code = a.assemble(CODE).unwrap();

    let (mut emu, _facts) = harness(&code, &[]);
    assert_eq!(emu.run(1000), Stop::Hook, "unmodeled API stops the run");
}

/// A jmp-based (tail-call) thunk returns via the stack: RSP unwinds by 8 and
/// rip lands on the pushed return address.
#[test]
fn jmp_thunk_returns_via_stack() {
    let ret_target = 0x3000u64;
    let mut a = CodeAssembler::new(64).unwrap();
    a.mov(rax, STUB_GETPROC).unwrap();
    a.jmp(rax).unwrap();
    let code = a.assemble(CODE).unwrap();

    // GetProcAddress(hmodule=0 -> "<unknown>", ordinal 7).
    let (mut emu, facts) = harness(&code, &[]);
    emu.cpu.regs[1] = 0; // RCX hmodule
    emu.cpu.regs[2] = 7; // RDX ordinal
    // Pre-push a return address the tail-call thunk should unwind to.
    emu.cpu.regs[4] = RSP_START - 8;
    emu.mem.write_uint(RSP_START - 8, 8, ret_target).unwrap();

    assert_eq!(emu.run_until(ret_target, 1000), Stop::Reached, "returned to caller");
    assert_eq!(emu.cpu.regs[4], RSP_START, "RSP unwound by 8");

    let f = facts.borrow();
    assert_eq!(f.imports[0].symbol, SymbolOrOrdinal::Ordinal(7));
    assert_eq!(f.imports[0].module, "<unknown>");
}

/// Regression: a `GetModuleHandle` result is a *dereferenceable* module base,
/// not a bare sentinel. VMProtect resolves a module (here "ntdll.dll") and then
/// walks the returned HMODULE's PE headers; before the loader mapped a faithful
/// header behind each handle, that walk faulted (a read of the sentinel module
/// base — the real-sample `0x70_0000_0000` exec/read fault). Now the handle
/// points at a valid DOS+NT header whose fields the guest reads cleanly.
#[test]
fn module_handle_is_a_walkable_pe_image() {
    let ntdll = b"ntdll.dll\0".to_vec();
    let name_ntdll = DATA;

    let mut a = CodeAssembler::new(64).unwrap();
    a.mov(rcx, name_ntdll).unwrap();
    a.mov(rax, STUB_GETMODULE).unwrap();
    a.call(rax).unwrap();
    a.mov(rbx, rax).unwrap(); // save the HMODULE (= module base)
    a.movzx(rdx, word_ptr(rax)).unwrap(); // DOS e_magic "MZ"
    a.mov(esi, dword_ptr(rax + 0x3c)).unwrap(); // e_lfanew (zero-extends to rsi)
    a.mov(r8, rbx).unwrap();
    a.add(r8, rsi).unwrap(); // r8 = base + e_lfanew
    a.mov(r9d, dword_ptr(r8)).unwrap(); // NT signature "PE\0\0"
    let code = a.assemble(CODE).unwrap();
    let end = CODE + code.len() as u64;

    let (mut emu, _facts) = harness(&code, &[(name_ntdll, ntdll)]);
    // The whole block retires with no fault on the handle dereference.
    assert_eq!(emu.run_until(end, 1000), Stop::Reached, "walked the module header");
    assert_ne!(emu.cpu.regs[3], 0, "GetModuleHandle returned a non-null base");
    assert_eq!(emu.cpu.regs[2], 0x5A4D, "read DOS 'MZ' at module base");
    assert_eq!(emu.cpu.regs[9], 0x0000_4550, "read 'PE\\0\\0' at base + e_lfanew");
}

/// Sanity: a transfer to a non-stub address is left alone (Continue).
#[test]
fn non_stub_transfer_is_untouched() {
    use vmpx_emulate::{FlowEvent, FlowKind, Hook};
    let mut hook = WinApiHook::new(BTreeMap::new());
    let ev = FlowEvent { kind: FlowKind::Call, from: 0x1000, to: 0x2000, next: 0x1005 };
    let mut cpu = Cpu::new(0x1000);
    let mut mem = Memory::new();
    assert_eq!(hook.on_flow(&ev, &mut cpu, &mut mem), HookAction::Continue);
}
