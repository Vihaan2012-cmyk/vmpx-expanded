//! Deterministic undefined-flag MODE (detonation-only concretization).
//!
//! Default (mode off): a consumer of a flag left architecturally-undefined by a
//! multi-bit shift/rotate stops the run — the sound fail-closed behaviour.
//! Mode on: the producer resolves the flag to its deterministic hardware value,
//! so the following `jcc`/`cmov` reads a concrete bit and proceeds.

use iced_x86::code_asm::*;
use vmpx_emulate::{Cpu, Emulator, Memory, OF, Perm, Stop};

const CODE: u64 = 0x1000;
const STACK_TOP: u64 = 0x9000;

fn emu(bytes: &[u8]) -> Emulator {
    let mut mem = Memory::new();
    mem.map(CODE, 0x1000, Perm::RWX).unwrap();
    mem.map(0x8000, 0x1000, Perm::RW).unwrap();
    mem.poke(CODE, bytes).unwrap();
    let mut cpu = Cpu::new(CODE);
    cpu.regs[4] = STACK_TOP;
    Emulator::new(cpu, mem)
}

fn asm(f: impl FnOnce(&mut CodeAssembler)) -> Vec<u8> {
    let mut a = CodeAssembler::new(64).unwrap();
    f(&mut a);
    a.assemble(CODE).unwrap()
}

/// `shl eax,3` leaves OF undefined; the following `jo` consumes it. A marker
/// `mov r8d, 0xBEEF` after the join point proves the run got past the consumer.
fn shift_then_jo() -> Vec<u8> {
    asm(|a| {
        let mut l = a.create_label();
        a.mov(eax, 1).unwrap();
        a.shl(eax, 3u32).unwrap();
        a.jo(l).unwrap();
        a.set_label(&mut l).unwrap();
        a.mov(r8d, 0xBEEFu32).unwrap();
        a.nop().unwrap();
    })
}

#[test]
fn mode_off_still_stops_on_undefined_consumer() {
    let code = shift_then_jo();
    let mut e = emu(&code);
    assert!(!e.resolve_undefined_flags, "default is fail-closed");
    let s = e.run(10);
    assert!(matches!(s, Stop::UndefinedFlags { mask, .. } if mask == OF), "{s:?}");
    // Only mov + shl retired; the jo never resolved.
    assert_eq!(e.trace.len(), 2);
}

#[test]
fn mode_on_resolves_and_proceeds() {
    let code = shift_then_jo();
    let mut e = emu(&code);
    e.resolve_undefined_flags = true;
    let s = e.run_until(CODE + code.len() as u64 - 1, 20);
    assert_eq!(s, Stop::Reached, "ran past the OF consumer to the end");
    // shl eax,3 of 1 => result 8, msb=0, cf=0 => deterministic OF=0 => jo not
    // taken, falls through to the marker.
    assert_eq!(e.cpu.regs[8] & 0xffff_ffff, 0xBEEF);
    assert_eq!(e.cpu.undef_flags & OF, 0, "OF concretized, not left undefined");
}

/// A multi-bit `rol` also leaves OF undefined; mode-on resolves it so a `cmovo`
/// reads a concrete bit instead of stopping.
#[test]
fn mode_on_resolves_rotate_of_for_cmov() {
    let code = asm(|a| {
        a.mov(eax, 0x8000_0000u32).unwrap();
        a.mov(ecx, 0x1234u32).unwrap();
        a.rol(eax, 3u32).unwrap(); // multi-bit: OF architecturally undefined
        a.cmovo(edx, ecx).unwrap(); // consumes OF
        a.nop().unwrap();
    });
    // Mode off: the cmovo stops.
    let mut off = emu(&code);
    assert!(matches!(off.run(10), Stop::UndefinedFlags { mask, .. } if mask == OF));
    // Mode on: resolves and runs to the end.
    let mut on = emu(&code);
    on.resolve_undefined_flags = true;
    assert_eq!(on.run_until(CODE + code.len() as u64 - 1, 20), Stop::Reached);
    assert_eq!(on.cpu.undef_flags & OF, 0);
}
