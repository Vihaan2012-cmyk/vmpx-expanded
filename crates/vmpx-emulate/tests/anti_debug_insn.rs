//! Instruction-level anti-debug / anti-VM clean answers: Trap Flag, `cpuid`
//! hypervisor leaves, `rdtsc`, and software-breakpoint instructions.

use iced_x86::Mnemonic;
use iced_x86::code_asm::*;
use vmpx_emulate::{Cpu, Emulator, Memory, Perm, Stop};

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

#[test]
fn pushfq_observes_trap_flag_clear() {
    let code = asm(|a| {
        a.pushfq().unwrap();
        a.pop(rax).unwrap();
        a.nop().unwrap();
    });
    let mut e = emu(&code);
    assert_eq!(e.run_until(CODE + code.len() as u64 - 1, 100), Stop::Reached);
    assert_eq!(e.cpu.regs[0] & 0x100, 0, "TF reads 0");
}

#[test]
fn popfq_that_would_set_tf_fails_closed_without_injecting_a_single_step() {
    // The classic trick: pushfq; or [rsp], 0x100; popfq; nop. Real hardware would
    // single-step into the process's own handler; no exception model exists, so
    // the run stops explicitly and the architectural state is untouched.
    let code = asm(|a| {
        a.pushfq().unwrap();
        a.or(qword_ptr(rsp), 0x100).unwrap();
        a.popfq().unwrap();
        a.nop().unwrap();
    });
    let mut e = emu(&code);
    let stop = e.run(100);
    assert!(matches!(stop, Stop::Unsupported { mnemonic: Mnemonic::Popfq, .. }), "{stop:?}");
    assert_eq!(e.cpu.rflags & 0x100, 0, "TF never latched");
    assert_eq!(e.cpu.regs[4], STACK_TOP - 8, "popfq did not consume the stack slot");
}

#[test]
fn popfq_without_tf_still_works() {
    let code = asm(|a| {
        a.pushfq().unwrap();
        a.popfq().unwrap();
        a.nop().unwrap();
    });
    let mut e = emu(&code);
    assert_eq!(e.run_until(CODE + code.len() as u64 - 1, 100), Stop::Reached);
}

#[test]
fn cpuid_reports_no_hypervisor() {
    let code = asm(|a| {
        a.mov(eax, 1).unwrap();
        a.cpuid().unwrap();
        a.mov(esi, ecx).unwrap(); // leaf 1 ECX
        a.mov(eax, 0x4000_0000u32).unwrap();
        a.cpuid().unwrap(); // hypervisor vendor leaf
        a.nop().unwrap();
    });
    let mut e = emu(&code);
    assert_eq!(e.run_until(CODE + code.len() as u64 - 1, 100), Stop::Reached);
    assert_eq!(e.cpu.regs[6] & (1 << 31), 0, "CPUID.1:ECX[31] hypervisor-present bit clear");
    assert_eq!(e.cpu.regs[0] & 0xffff_ffff, 0, "no hypervisor max leaf");
    assert_eq!(e.cpu.regs[3] | e.cpu.regs[1] | e.cpu.regs[2], 0, "no hypervisor vendor id");
    assert_eq!(e.audit.cpuid.len(), 2, "both leaves audited");
}

#[test]
fn software_breakpoint_instructions_are_explicit_stops() {
    for (bytes, m) in [(vec![0xCCu8], Mnemonic::Int3), (vec![0xCD, 0x03], Mnemonic::Int)] {
        let mut e = emu(&bytes);
        let stop = e.run(10);
        assert!(matches!(stop, Stop::Unsupported { mnemonic, .. } if mnemonic == m), "{stop:?}");
    }
}

#[test]
fn rdtsc_is_deterministic_and_monotonic() {
    let code = asm(|a| {
        a.rdtsc().unwrap();
        a.mov(esi, eax).unwrap();
        a.rdtsc().unwrap();
        a.nop().unwrap();
    });
    let run = || {
        let mut e = emu(&code);
        assert_eq!(e.run_until(CODE + code.len() as u64 - 1, 100), Stop::Reached);
        (e.cpu.regs[6], e.cpu.regs[0], e.audit.rdtsc)
    };
    let (t0, t1, n) = run();
    assert!(t1 > t0, "monotonic");
    assert_eq!(n, 2);
    assert_eq!(run(), (t0, t1, 2), "repeatable");
}
