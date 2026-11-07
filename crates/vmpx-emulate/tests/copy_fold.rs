//! The copy accelerator must be invisible: a folded run has to reach the same
//! CPU state AND the same guest memory as the pure interpreter.
use iced_x86::Register;
use iced_x86::code_asm::*;
use vmpx_emulate::{Cpu, Emulator, Memory, Perm, Stop};

const CODE: u64 = 0x1_0000;
const SOURCE: u64 = 0x3_0000;
const DEST: u64 = 0x5_0000;
const COUNT: u64 = 4096;

/// The body observed in systems.xpl at rva 0xefd3649: a byte load, three
/// counter/pointer updates, a byte store, and the two exits.
fn program() -> Vec<u8> {
    let mut a = CodeAssembler::new(64).expect("asm");
    let mut head = a.create_label();
    let mut exit = a.create_label();
    a.set_label(&mut head).expect("head");
    a.mov(r11b, byte_ptr(rax)).expect("load");
    a.inc(r14).expect("inc dest");
    a.inc(rax).expect("inc src");
    a.sub(ebp, 1).expect("sub");
    a.mov(byte_ptr(r14 + rdx - 1), r11b).expect("store");
    a.je(exit).expect("je");
    a.cmp(r14, -1).expect("cmp");
    a.jb(head).expect("jb");
    a.set_label(&mut exit).expect("exit");
    a.hlt().expect("hlt");
    a.assemble(CODE).expect("assemble")
}

/// `dest_perm` lets a test make the destination unwritable.
fn fresh(accelerate: bool, count: u64, dest_perm: Perm) -> Emulator {
    let mut mem = Memory::new();
    mem.map(CODE & !0xFFF, 0x1000, Perm::RX).expect("map code");
    mem.poke(CODE, &program()).expect("write code");
    mem.map(SOURCE, 0x2000, Perm::R).expect("map source");
    for offset in 0..0x2000u64 {
        #[allow(clippy::cast_possible_truncation)]
        let byte = (offset.wrapping_mul(31).wrapping_add(7) & 0xFF) as u8;
        mem.poke(SOURCE + offset, &[byte]).expect("fill source");
    }
    mem.map(DEST, 0x2000, dest_perm).expect("map dest");

    let mut cpu = Cpu::new(CODE);
    cpu.set_reg(Register::RAX, SOURCE);
    // `mov [r14+rdx-1]`: with r14 = 0 and rdx = DEST, iteration k stores at
    // DEST + k, so the destination walks forward from DEST.
    cpu.set_reg(Register::R14, 0);
    cpu.set_reg(Register::RDX, DEST);
    cpu.set_reg(Register::EBP, count);

    let mut emu = Emulator::new(cpu, mem);
    emu.record_trace = false;
    emu.accelerate_checksums = accelerate;
    emu
}

fn state(emu: &Emulator) -> (Vec<u64>, u64, u64) {
    let regs = [
        Register::RAX, Register::R14, Register::RDX, Register::RBP, Register::R11,
        Register::RCX, Register::RSI, Register::RSP,
    ];
    (
        regs.iter().map(|r| emu.cpu.reg(*r).unwrap_or_default()).collect(),
        emu.cpu.rip,
        emu.cpu.rflags,
    )
}

fn written(emu: &Emulator, len: u64) -> Vec<u8> {
    (0..len).map(|i| emu.mem.read_uint(DEST + i, 1).unwrap_or(0xAA) as u8).collect()
}

#[test]
fn a_folded_copy_loop_matches_the_interpreter_in_state_and_memory() {
    let mut interpreted = fresh(false, COUNT, Perm::RW);
    let plain = interpreted.run(10_000_000);
    let mut folded = fresh(true, COUNT, Perm::RW);
    let fast = folded.run(10_000_000);

    assert_eq!(plain, fast, "both runs must stop the same way");
    assert_eq!(state(&interpreted), state(&folded), "folding changed the CPU state");
    assert_eq!(
        written(&interpreted, COUNT),
        written(&folded, COUNT),
        "folding changed the copied bytes"
    );
    // The copy must really have happened, not just matched an empty result.
    let expected: Vec<u8> = (0..COUNT)
        .map(|i| {
            #[allow(clippy::cast_possible_truncation)]
            let byte = (i.wrapping_mul(31).wrapping_add(7) & 0xFF) as u8;
            byte
        })
        .collect();
    assert_eq!(written(&folded, COUNT), expected, "the bytes copied are wrong");

    assert!(folded.prof.copies_folded > 0, "no copy fold occurred");
    assert_eq!(folded.prof.copied_bytes, COUNT - 1);
    assert!(
        folded.steps() < interpreted.steps() / 100,
        "folded {} steps vs {} interpreted",
        folded.steps(),
        interpreted.steps()
    );
}

#[test]
fn an_unwritable_destination_faults_exactly_as_the_interpreter_does() {
    let mut interpreted = fresh(false, COUNT, Perm::R);
    let plain = interpreted.run(10_000_000);
    let mut folded = fresh(true, COUNT, Perm::R);
    let fast = folded.run(10_000_000);
    assert_eq!(plain, fast, "the fold must not change the stop reason");
    assert_eq!(state(&interpreted), state(&folded), "the fold must not change state");
    assert_eq!(folded.prof.copied_bytes, 0, "nothing may be written to a read-only page");
}

#[test]
fn running_off_the_end_of_the_source_declines_rather_than_inventing_bytes() {
    // The source mapping is 0x2000 but the counter asks for more.
    let count = 0x4000;
    let mut interpreted = fresh(false, count, Perm::RW);
    let plain = interpreted.run(50_000_000);
    let mut folded = fresh(true, count, Perm::RW);
    let fast = folded.run(50_000_000);
    assert_eq!(plain, fast, "the fold must not change the stop reason");
    assert_eq!(state(&interpreted), state(&folded), "the fold must not change state");
    assert_eq!(written(&interpreted, 0x2000), written(&folded, 0x2000));
    assert!(folded.prof.fold_declined > 0, "the boundary should have declined");
}

#[test]
fn a_copy_whose_source_and_destination_overlap_still_matches_byte_for_byte() {
    // Overlapping ranges are where a bulk copy would diverge from a byte loop,
    // so the fold must stay byte-at-a-time in its effects.
    let overlap_dest = SOURCE + 1;
    let build = |accelerate: bool| {
        let mut mem = Memory::new();
        mem.map(CODE & !0xFFF, 0x1000, Perm::RX).expect("map code");
        mem.poke(CODE, &program()).expect("write code");
        mem.map(SOURCE, 0x2000, Perm::RW).expect("map source");
        for offset in 0..0x2000u64 {
            #[allow(clippy::cast_possible_truncation)]
            let byte = (offset.wrapping_mul(31).wrapping_add(7) & 0xFF) as u8;
            mem.poke(SOURCE + offset, &[byte]).expect("fill");
        }
        let mut cpu = Cpu::new(CODE);
        cpu.set_reg(Register::RAX, SOURCE);
        cpu.set_reg(Register::R14, 0);
        cpu.set_reg(Register::RDX, overlap_dest);
        cpu.set_reg(Register::EBP, 512);
        let mut emu = Emulator::new(cpu, mem);
        emu.record_trace = false;
        emu.accelerate_checksums = accelerate;
        emu
    };
    let mut interpreted = build(false);
    let plain = interpreted.run(10_000_000);
    let mut folded = build(true);
    let fast = folded.run(10_000_000);
    assert_eq!(plain, fast);
    let region = |emu: &Emulator| {
        (0..0x600u64).map(|i| emu.mem.read_uint(SOURCE + i, 1).unwrap_or(0) as u8).collect::<Vec<_>>()
    };
    assert_eq!(region(&interpreted), region(&folded), "overlapping copy diverged");
    assert!(folded.prof.copies_folded > 0);
}
