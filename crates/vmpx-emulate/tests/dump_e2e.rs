//! Layer-4 end-to-end: hand-assembled guests exercise OEP detection
//! (write-then-execute), the memory snapshot round-trip, the runtime-global
//! capture bridge, and OEP detection composed over a `WinApiHook`.

use std::collections::BTreeMap;

use iced_x86::code_asm::*;
use vmpx_emulate::{
    Cpu, Emulator, Memory, ObservedFacts, OepHook, Perm, Stop, SymbolOrOrdinal, WinApiHook,
    capture_globals, dump_image,
};

const CODE: u64 = 0x1000;
const TARGET: u64 = 0x5000; // RWX page the guest unpacks into then jumps to
const SLOT: u64 = 0x8000; // "null" global slot, zero statically
const STACK_TOP: u64 = 0x0000_0010_0000_0000 + 0x40_000;
const RSP_START: u64 = STACK_TOP - 0x100;
const STUB_GETPROC: u64 = 0x0000_0020_0000_0008;

const PATTERN: u64 = 0xAABB_CCDD_1122_3344;

/// Map an RX code page, an RWX unpack target, an RW data page and a stack;
/// prime rip/rsp. Hook is left for the caller to install.
fn harness(code: &[u8]) -> Emulator {
    let mut mem = Memory::new();
    mem.map(CODE, 0x1000, Perm::RX).unwrap();
    mem.poke(CODE, code).unwrap();
    mem.map(TARGET, 0x1000, Perm::RWX).unwrap();
    mem.map(SLOT & !0xFFF, 0x1000, Perm::RW).unwrap();
    mem.map(STACK_TOP - 0x40_000, 0x40_000, Perm::RW).unwrap();

    let mut cpu = Cpu::new(CODE);
    cpu.regs[4] = RSP_START; // RSP
    Emulator::new(cpu, mem)
}

/// `mov rax, PATTERN; mov rbx, TARGET; mov [rbx], rax; jmp rbx`.
fn write_then_execute() -> Vec<u8> {
    let mut a = CodeAssembler::new(64).unwrap();
    a.mov(rax, PATTERN).unwrap();
    a.mov(rbx, TARGET).unwrap();
    a.mov(qword_ptr(rbx), rax).unwrap();
    a.jmp(rbx).unwrap();
    a.assemble(CODE).unwrap()
}

#[test]
fn write_then_execute_detects_oep_and_dumps_the_page() {
    let mut emu = harness(&write_then_execute());
    let hook = OepHook::new();
    let facts = hook.facts();
    emu.hook = Some(Box::new(hook));

    // The run stops the instant control enters the freshly written page.
    assert_eq!(emu.run(1000), Stop::Hook, "OEP hook stops at the tail jump");
    assert_eq!(facts.borrow().oep, Some(TARGET), "OEP recorded = written target");

    // The snapshot round-trips the bytes the guest unpacked into that page.
    let dump = dump_image(&emu.mem, CODE);
    let region = dump.region_containing(TARGET).expect("target page mapped in dump");
    assert!(region.perm.contains(Perm::X), "unpacked page carries its perms");
    assert_eq!(
        dump.read(TARGET, 8),
        Some(PATTERN.to_le_bytes().to_vec()),
        "written bytes survive the dump round-trip",
    );
}

#[test]
fn capture_globals_reads_back_a_runtime_written_slot() {
    // `mov rbx, SLOT; mov [rbx], rax` — rax carries the post-init value.
    let mut a = CodeAssembler::new(64).unwrap();
    a.mov(rbx, SLOT).unwrap();
    a.mov(qword_ptr(rbx), rax).unwrap();
    let code = a.assemble(CODE).unwrap();
    let end = CODE + code.len() as u64;

    let mut emu = harness(&code);
    emu.cpu.regs[0] = PATTERN; // RAX: the value init will store into the slot
    // Statically the slot is zero — the stall the static side hits.
    assert_eq!(emu.mem.read_uint(SLOT, 8).unwrap(), 0, "slot zero before the run");

    assert_eq!(emu.run_until(end, 1000), Stop::Reached);

    let mut facts = ObservedFacts::new();
    capture_globals(&emu.mem, &mut facts, &[(SLOT, 8)]).unwrap();
    assert_eq!(facts.globals.len(), 1);
    assert_eq!(facts.globals[0].va, SLOT);
    assert_eq!(facts.globals[0].bytes, PATTERN.to_le_bytes().to_vec());
}

#[test]
fn oep_detection_composes_over_winapi_hook() {
    // GetProcAddress(hmodule=0, ordinal 7) through the inner WinApiHook, then
    // the write-then-execute tail jump detected by the outer OepHook.
    let mut a = CodeAssembler::new(64).unwrap();
    a.mov(rcx, 0u64).unwrap();
    a.mov(rdx, 7u64).unwrap();
    a.mov(rax, STUB_GETPROC).unwrap();
    a.call(rax).unwrap();
    a.mov(rax, PATTERN).unwrap();
    a.mov(rbx, TARGET).unwrap();
    a.mov(qword_ptr(rbx), rax).unwrap();
    a.jmp(rbx).unwrap();
    let code = a.assemble(CODE).unwrap();

    let mut emu = harness(&code);
    let mut stubs = BTreeMap::new();
    stubs.insert(STUB_GETPROC, ("kernel32.dll".into(), "GetProcAddress".into()));
    let winapi = WinApiHook::new(stubs);
    let facts = winapi.facts();
    emu.hook = Some(Box::new(OepHook::wrapping(winapi, facts.clone())));

    assert_eq!(emu.run(1000), Stop::Hook, "stops at the unpack jump");

    let f = facts.borrow();
    assert_eq!(f.oep, Some(TARGET), "OEP recorded by outer hook");
    assert_eq!(f.imports.len(), 1, "import recorded by inner hook, same facts");
    assert_eq!(f.imports[0].symbol, SymbolOrOrdinal::Ordinal(7));
}
