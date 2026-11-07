//! The checksum accelerator must be invisible: a folded run has to reach the
//! same CPU state as the pure interpreter, not merely a plausible one.
use iced_x86::Register;
use iced_x86::code_asm::*;
use vmpx_emulate::{Cpu, Emulator, Memory, Perm, Stop};

const CODE: u64 = 0x1_0000;
const TABLE: u64 = 0x2_0000;
const INPUT: u64 = 0x3_0000;
const ITERATIONS: u64 = 4096;

/// The body observed in systems.xpl at rva 0xf04f107: the recurrence plus the
/// flag noise the protector interleaves, split across a `jmp` exactly as the
/// real loop is, and ending in a `hlt` so the run has a definite stop.
fn program() -> Vec<u8> {
    let mut a = CodeAssembler::new(64).expect("asm");
    let mut head = a.create_label();
    let mut tail = a.create_label();
    a.set_label(&mut head).expect("head");
    a.movzx(esi, byte_ptr(rbx)).expect("movzx");
    a.xor(esi, r15d).expect("xor");
    a.clc().expect("clc");
    a.and(esi, 0xFF).expect("and");
    a.mov(esi, dword_ptr(rdx + rsi * 4)).expect("mov");
    a.stc().expect("stc");
    a.shr(r15d, 8u32).expect("shr");
    a.cmp(ax, 0x6F76).expect("cmp");
    a.cmc().expect("cmc");
    a.clc().expect("clc2");
    a.xor(r15d, esi).expect("xor2");
    a.inc(rbx).expect("inc");
    a.xor(r15d, 0xFEFD_F493u32 as i32).expect("xor3");
    a.jmp(tail).expect("jmp");
    a.set_label(&mut tail).expect("tail");
    a.dec(r13).expect("dec");
    a.jne(head).expect("jne");
    a.hlt().expect("hlt");
    a.assemble(CODE).expect("assemble")
}

fn fresh(accelerate: bool) -> Emulator {
    let mut mem = Memory::new();
    let code = program();
    mem.map(CODE & !0xFFF, 0x1000, Perm::RX).expect("map code");
    mem.poke(CODE, &code).expect("write code");
    mem.map(TABLE, 0x1000, Perm::R).expect("map table");
    for index in 0..256u64 {
        #[allow(clippy::cast_possible_truncation)]
        let value = (index as u32).wrapping_mul(0x0100_1001) ^ 0xA5A5_1234;
        mem.poke(TABLE + index * 4, &value.to_le_bytes()).expect("write table");
    }
    mem.map(INPUT, 0x2000, Perm::R).expect("map input");
    for offset in 0..ITERATIONS {
        #[allow(clippy::cast_possible_truncation)]
        let byte = (offset.wrapping_mul(37).wrapping_add(11) & 0xFF) as u8;
        mem.poke(INPUT + offset, &[byte]).expect("write input");
    }

    let mut cpu = Cpu::new(CODE);
    cpu.set_reg(Register::R15D, 0x1234_5678);
    cpu.set_reg(Register::RBX, INPUT);
    cpu.set_reg(Register::RDX, TABLE);
    cpu.set_reg(Register::R13, ITERATIONS);

    let mut emu = Emulator::new(cpu, mem);
    emu.record_trace = false;
    emu.accelerate_checksums = accelerate;
    emu
}

fn state(emu: &Emulator) -> (Vec<u64>, u64, u64) {
    let regs = [
        Register::R15, Register::RBX, Register::RDX, Register::R13, Register::RSI,
        Register::RAX, Register::RCX, Register::RSP, Register::RBP, Register::R8,
    ];
    (
        regs.iter().map(|r| emu.cpu.reg(*r).unwrap_or_default()).collect(),
        emu.cpu.rip,
        emu.cpu.rflags,
    )
}

#[test]
fn a_folded_checksum_loop_lands_in_the_same_state_as_the_interpreter() {
    let mut interpreted = fresh(false);
    let plain = interpreted.run(10_000_000);
    let mut folded = fresh(true);
    let fast = folded.run(10_000_000);

    assert_eq!(plain, fast, "both runs must stop the same way");
    assert!(matches!(plain, Stop::Unsupported { .. } | Stop::DecodeError { .. }),
        "the `hlt` after the loop should stop the run, got {plain:?}");
    assert_eq!(state(&interpreted), state(&folded), "folding changed the CPU state");

    // The fold must actually have happened, and have skipped nearly the whole
    // loop -- otherwise this test would pass while measuring nothing.
    assert!(folded.prof.folded > 0, "no fold occurred");
    assert_eq!(folded.prof.folded_iterations, ITERATIONS - 1);
    assert_eq!(folded.prof.fold_declined, 0);
    assert!(
        folded.steps() < interpreted.steps() / 100,
        "folded run took {} steps vs {} interpreted; expected a ~17x+ cut",
        folded.steps(),
        interpreted.steps()
    );
}

#[test]
fn acceleration_is_off_by_default_so_sound_paths_are_unchanged() {
    let emu = fresh(false);
    assert!(!emu.accelerate_checksums);
    let mut on = fresh(true);
    let mut off = fresh(false);
    on.run(10_000_000);
    off.run(10_000_000);
    assert_eq!(off.prof.folded, 0);
    assert!(on.prof.folded > 0);
}

#[test]
fn an_unreadable_input_byte_declines_the_fold_instead_of_inventing_a_value() {
    // The input mapping is 0x2000 long but the counter asks for far more, so the
    // walk runs off the end. The fold must stop at the boundary and hand over to
    // the interpreter, which then faults exactly where it would have anyway.
    let mut folded = fresh(true);
    folded.cpu.set_reg(Register::R13, 0x8000);
    let mut interpreted = fresh(false);
    interpreted.cpu.set_reg(Register::R13, 0x8000);

    let fast = folded.run(10_000_000);
    let plain = interpreted.run(10_000_000);
    assert_eq!(plain, fast, "the fold must not change the stop reason");
    assert_eq!(state(&interpreted), state(&folded), "the fold must not change state");
    assert!(folded.prof.fold_declined > 0, "the boundary should have declined");
}
