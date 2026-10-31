//! Instruction-level tests for the widened x64 ISA (Layer-3): exact results and
//! flags for mul/div, rotates, bit ops, setcc/cmov, string ops, flag transfers,
//! and the fail-closed paths (divide-by-zero, unmodeled XMM).
#![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]

use iced_x86::code_asm::*;
use vmpx_emulate::{CF, Cpu, Emulator, Memory, OF, Perm, Stop, ZF};

const CODE: u64 = 0x1000;
const DATA: u64 = 0x2000;
const STACK_TOP: u64 = 0x9000;

fn emu_bytes(bytes: &[u8]) -> Emulator {
    let mut mem = Memory::new();
    mem.map(CODE, 0x1000, Perm::RWX).unwrap();
    mem.map(DATA, 0x2000, Perm::RW).unwrap();
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

/// Run `code` to its last byte and return the emulator for inspection.
fn run(code: &[u8]) -> Emulator {
    let mut e = emu_bytes(code);
    let end = CODE + code.len() as u64 - 1;
    assert_eq!(e.run_until(end, 500), Stop::Reached, "trace {:?}", e.trace.last());
    e
}

#[test]
fn mul_imul_one_operand() {
    // unsigned: RAX = 0x1_0000_0000 * 0x4 -> RDX:RAX = 0x4_0000_0000.
    let e = run(&asm(|a| {
        a.mov(rax, 0x1_0000_0000u64).unwrap();
        a.mov(rbx, 4u64).unwrap();
        a.mul(rbx).unwrap();
        a.nop().unwrap();
    }));
    assert_eq!(e.cpu.regs[0], 0x4_0000_0000);
    assert_eq!(e.cpu.regs[2], 0);
    assert!(!e.cpu.flag(CF) && !e.cpu.flag(OF));

    // signed byte: AL = -2 * 3 = -6 (0xFA), high half sign-extends, no overflow.
    let e = run(&asm(|a| {
        a.mov(al, 0xFEu32).unwrap(); // -2
        a.mov(bl, 3u32).unwrap();
        a.imul(bl).unwrap();
        a.nop().unwrap();
    }));
    assert_eq!(e.cpu.regs[0] & 0xffff, 0xFFFA); // AX = -6
    assert!(!e.cpu.flag(CF));
}

#[test]
fn imul_two_and_three_operand() {
    // 2-operand reg,reg: 6 * 7 = 42, fits -> no overflow.
    let e = run(&asm(|a| {
        a.mov(eax, 6).unwrap();
        a.mov(ebx, 7).unwrap();
        a.imul_2(eax, ebx).unwrap();
        a.nop().unwrap();
    }));
    assert_eq!(e.cpu.regs[0], 42);
    assert!(!e.cpu.flag(CF) && !e.cpu.flag(OF));

    // 3-operand reg,reg,imm with 32-bit overflow sets CF=OF.
    let e = run(&asm(|a| {
        a.mov(ebx, 0x1000_0000).unwrap();
        a.imul_3(eax, ebx, 0x40i32).unwrap(); // 0x4_0000_0000 -> truncates to 0
        a.nop().unwrap();
    }));
    assert_eq!(e.cpu.regs[0], 0);
    assert!(e.cpu.flag(CF) && e.cpu.flag(OF));
}

#[test]
fn div_idiv_quotient_remainder() {
    // unsigned 64-bit: 100 / 7 = 14 r 2.
    let e = run(&asm(|a| {
        a.xor(edx, edx).unwrap();
        a.mov(rax, 100u64).unwrap();
        a.mov(rbx, 7u64).unwrap();
        a.div(rbx).unwrap();
        a.nop().unwrap();
    }));
    assert_eq!((e.cpu.regs[0], e.cpu.regs[2]), (14, 2));

    // signed 32-bit: -100 / 7 = -14 r -2 (cdq sign-extends EAX into EDX).
    let e = run(&asm(|a| {
        a.mov(eax, -100).unwrap();
        a.cdq().unwrap();
        a.mov(ebx, 7).unwrap();
        a.idiv(ebx).unwrap();
        a.nop().unwrap();
    }));
    assert_eq!(e.cpu.regs[0] as i32, -14);
    assert_eq!(e.cpu.regs[2] as i32, -2);
}

#[test]
fn divide_by_zero_fails_closed() {
    let code = asm(|a| {
        a.xor(edx, edx).unwrap();
        a.mov(eax, 10).unwrap();
        a.xor(ebx, ebx).unwrap();
        a.div(ebx).unwrap();
    });
    let mut e = emu_bytes(&code);
    let s = e.run(10);
    assert!(matches!(s, Stop::DivideError { .. }), "{s:?}");
    // The div did not retire; state rolled back to before it.
    assert_eq!(e.cpu.regs[0], 10);
}

#[test]
fn adc_sbb_128bit_chain() {
    // (0xFFFF_FFFF_FFFF_FFFF + 1) as a 128-bit add carried through adc.
    let e = run(&asm(|a| {
        a.mov(rax, u64::MAX).unwrap();
        a.mov(rdx, 0u64).unwrap();
        a.add(rax, 1).unwrap(); // sets CF
        a.adc(rdx, 0).unwrap(); // RDX picks up the carry
        a.nop().unwrap();
    }));
    assert_eq!((e.cpu.regs[0], e.cpu.regs[2]), (0, 1));

    // sbb borrow: 0 - 0 - 1(borrow) = -1.
    let e = run(&asm(|a| {
        a.mov(rax, 0u64).unwrap();
        a.stc().unwrap();
        a.sbb(rax, 0).unwrap();
        a.nop().unwrap();
    }));
    assert_eq!(e.cpu.regs[0], u64::MAX);
    assert!(e.cpu.flag(CF));
}

#[test]
fn rotates_with_carry() {
    // rol: 0x8000_0000_0000_0001 rol 1 -> 0x0000_0000_0000_0003, CF=1.
    let e = run(&asm(|a| {
        a.mov(rax, 0x8000_0000_0000_0001u64).unwrap();
        a.rol(rax, 1u32).unwrap();
        a.nop().unwrap();
    }));
    assert_eq!(e.cpu.regs[0], 3);
    assert!(e.cpu.flag(CF));

    // rcr: CF rotates into the MSB. rax=0, CF=1, rcr 1 -> MSB set.
    let e = run(&asm(|a| {
        a.mov(rax, 0u64).unwrap();
        a.stc().unwrap();
        a.rcr(rax, 1u32).unwrap();
        a.nop().unwrap();
    }));
    assert_eq!(e.cpu.regs[0], 0x8000_0000_0000_0000);
    assert!(!e.cpu.flag(CF)); // the 0 bit rotated out
}

#[test]
fn bit_test_and_modify() {
    // bts sets bit 5 and reports its old value in CF (0).
    let e = run(&asm(|a| {
        a.mov(eax, 0).unwrap();
        a.bts(eax, 5u32).unwrap();
        a.nop().unwrap();
    }));
    assert_eq!(e.cpu.regs[0], 0x20);
    assert!(!e.cpu.flag(CF));

    // btr clears bit 3, CF = old bit (1).
    let e = run(&asm(|a| {
        a.mov(eax, 0xFF).unwrap();
        a.btr(eax, 3u32).unwrap();
        a.nop().unwrap();
    }));
    assert_eq!(e.cpu.regs[0], 0xF7);
    assert!(e.cpu.flag(CF));
}

#[test]
fn bit_scan_forward_reverse() {
    let e = run(&asm(|a| {
        a.mov(rax, 0x100u64).unwrap();
        a.bsf(rbx, rax).unwrap(); // lowest set bit = 8
        a.bsr(rcx, rax).unwrap(); // highest set bit = 8
        a.nop().unwrap();
    }));
    assert_eq!((e.cpu.regs[3], e.cpu.regs[1]), (8, 8));
    assert!(!e.cpu.flag(ZF));

    // zero source sets ZF; destination left unchanged by convention.
    let e = run(&asm(|a| {
        a.mov(rbx, 0x1234u64).unwrap();
        a.xor(eax, eax).unwrap();
        a.bsf(rbx, rax).unwrap();
        a.nop().unwrap();
    }));
    assert!(e.cpu.flag(ZF));
    assert_eq!(e.cpu.regs[3], 0x1234);
}

#[test]
fn setcc_and_cmov() {
    // cmp 5, 9 -> below/less; setl AL = 1, setg BL = 0.
    let e = run(&asm(|a| {
        a.mov(eax, 5).unwrap();
        a.cmp(eax, 9).unwrap();
        a.setl(cl).unwrap();
        a.setg(dl).unwrap();
        a.nop().unwrap();
    }));
    assert_eq!(e.cpu.regs[1] & 0xff, 1);
    assert_eq!(e.cpu.regs[2] & 0xff, 0);

    // cmov: pick the larger of 5 and 9 via cmovl.
    let e = run(&asm(|a| {
        a.mov(eax, 5).unwrap();
        a.mov(ebx, 9).unwrap();
        a.cmp(eax, ebx).unwrap();
        a.cmovl(eax, ebx).unwrap(); // 5 < 9 so move
        a.nop().unwrap();
    }));
    assert_eq!(e.cpu.regs[0], 9);
}

#[test]
fn xchg_cmpxchg_xadd() {
    // xchg swaps two registers.
    let e = run(&asm(|a| {
        a.mov(rax, 1u64).unwrap();
        a.mov(rbx, 2u64).unwrap();
        a.xchg(rax, rbx).unwrap();
        a.nop().unwrap();
    }));
    assert_eq!((e.cpu.regs[0], e.cpu.regs[3]), (2, 1));

    // cmpxchg: RAX matches dest -> store src, ZF=1.
    let e = run(&asm(|a| {
        a.mov(rax, 5u64).unwrap(); // accumulator
        a.mov(rbx, 5u64).unwrap(); // dest
        a.mov(rcx, 9u64).unwrap(); // src
        a.cmpxchg(rbx, rcx).unwrap();
        a.nop().unwrap();
    }));
    assert_eq!(e.cpu.regs[3], 9);
    assert!(e.cpu.flag(ZF));

    // cmpxchg mismatch -> RAX loaded with dest, ZF=0.
    let e = run(&asm(|a| {
        a.mov(rax, 1u64).unwrap();
        a.mov(rbx, 5u64).unwrap();
        a.mov(rcx, 9u64).unwrap();
        a.cmpxchg(rbx, rcx).unwrap();
        a.nop().unwrap();
    }));
    assert_eq!((e.cpu.regs[0], e.cpu.regs[3]), (5, 5));
    assert!(!e.cpu.flag(ZF));

    // xadd: dest += src, src = old dest.
    let e = run(&asm(|a| {
        a.mov(rax, 10u64).unwrap();
        a.mov(rbx, 3u64).unwrap();
        a.xadd(rax, rbx).unwrap();
        a.nop().unwrap();
    }));
    assert_eq!((e.cpu.regs[0], e.cpu.regs[3]), (13, 10));
}

#[test]
fn pushfq_popfq_roundtrip() {
    // Set flags via cmp (equal -> ZF), save, clobber, restore.
    let e = run(&asm(|a| {
        a.mov(eax, 7).unwrap();
        a.cmp(eax, 7).unwrap(); // ZF=1
        a.pushfq().unwrap();
        a.add(eax, 1).unwrap(); // ZF=0 now
        a.popfq().unwrap(); // ZF restored to 1
        a.nop().unwrap();
    }));
    assert!(e.cpu.flag(ZF));
    assert_eq!(e.cpu.regs[4], STACK_TOP); // stack balanced
}

#[test]
fn leave_and_bswap_and_cdqe() {
    // leave: RSP=RBP; RBP=pop. Set up a frame manually.
    let e = run(&asm(|a| {
        a.mov(rbp, 0x8800u64).unwrap();
        a.mov(rcx, 0x1234u64).unwrap();
        a.mov(qword_ptr(rbp), rcx).unwrap(); // saved RBP at [RBP]
        a.leave().unwrap();
        a.nop().unwrap();
    }));
    assert_eq!(e.cpu.regs[5], 0x1234); // RBP popped
    assert_eq!(e.cpu.regs[4], 0x8808); // RSP = old RBP + 8

    // bswap reverses byte order.
    let e = run(&asm(|a| {
        a.mov(eax, 0x1122_3344u32 as i32).unwrap();
        a.bswap(eax).unwrap();
        a.nop().unwrap();
    }));
    assert_eq!(e.cpu.regs[0], 0x4433_2211);

    // cdqe sign-extends EAX into RAX.
    let e = run(&asm(|a| {
        a.mov(eax, -1).unwrap();
        a.cdqe().unwrap();
        a.nop().unwrap();
    }));
    assert_eq!(e.cpu.regs[0], u64::MAX);
}

#[test]
fn shld_shrd_double_shift() {
    // shld: bring in high bits of src. dest=0xF000_0000, src=0x8000_0000, by 4.
    let e = run(&asm(|a| {
        a.mov(eax, 0xF000_0000u32 as i32).unwrap();
        a.mov(ebx, 0x8000_0000u32 as i32).unwrap();
        a.shld(eax, ebx, 4u32).unwrap();
        a.nop().unwrap();
    }));
    // (0xF000_0000 << 4) | (0x8000_0000 >> 28) = 0x0000_0000 | 0x8 = 0x8.
    assert_eq!(e.cpu.regs[0], 0x8);
}

#[test]
fn rep_movsq_copies_block_respecting_df() {
    // rep movsq (F3 48 A5): copy RCX qwords from [RSI] to [RDI], DF=0.
    let mut e = emu_bytes(&[0xF3, 0x48, 0xA5, 0x90]);
    // Source block at DATA, dest at DATA+0x100.
    for i in 0..3u64 {
        e.mem.write_uint(DATA + i * 8, 8, 0x1000 + i).unwrap();
    }
    e.cpu.regs[6] = DATA; // RSI
    e.cpu.regs[7] = DATA + 0x100; // RDI
    e.cpu.regs[1] = 3; // RCX
    assert_eq!(e.run_until(CODE + 3, 100), Stop::Reached);
    for i in 0..3u64 {
        assert_eq!(e.mem.read_uint(DATA + 0x100 + i * 8, 8).unwrap(), 0x1000 + i);
    }
    assert_eq!(e.cpu.regs[1], 0); // RCX drained
    assert_eq!(e.cpu.regs[6], DATA + 24); // RSI advanced (DF=0)
}

#[test]
fn repne_scasb_finds_byte() {
    // repne scasb (F2 AE): scan [RDI] for AL, stopping when equal.
    let mut e = emu_bytes(&[0xF2, 0xAE, 0x90]);
    let bytes = [1u8, 2, 3, 0xAA, 5];
    e.mem.write(DATA, &bytes).unwrap();
    e.cpu.regs[0] = 0xAA; // AL = search target
    e.cpu.regs[7] = DATA; // RDI
    e.cpu.regs[1] = 5; // RCX
    assert_eq!(e.run_until(CODE + 2, 100), Stop::Reached);
    // Found 0xAA at index 3: RDI advanced past it (DATA+4), RCX = 5-4 = 1.
    assert_eq!(e.cpu.regs[7], DATA + 4);
    assert_eq!(e.cpu.regs[1], 1);
    assert!(e.cpu.flag(ZF));
}

#[test]
fn std_sets_direction_for_string_ops() {
    // std; rep movsq backward. DF=1 means RSI/RDI decrement.
    let mut e = emu_bytes(&[0xFD, 0xF3, 0x48, 0xA5, 0x90]); // std; rep movsq; nop
    e.mem.write_uint(DATA + 16, 8, 0xCAFE).unwrap();
    e.cpu.regs[6] = DATA + 16; // RSI
    e.cpu.regs[7] = DATA + 0x100; // RDI
    e.cpu.regs[1] = 1; // RCX
    assert_eq!(e.run_until(CODE + 4, 100), Stop::Reached);
    assert_eq!(e.mem.read_uint(DATA + 0x100, 8).unwrap(), 0xCAFE);
    assert_eq!(e.cpu.regs[6], DATA + 8); // RSI decremented (DF=1)
}

#[test]
fn flag_byte_transfer_lahf_sahf() {
    // sahf then a conditional: set CF via AH bit0, confirm with setc.
    let e = run(&asm(|a| {
        a.mov(ah, 0x01).unwrap(); // CF bit in the low flag byte
        a.sahf().unwrap();
        a.setb(cl).unwrap(); // CF -> CL
        a.nop().unwrap();
    }));
    assert_eq!(e.cpu.regs[1] & 0xff, 1);
}

#[test]
fn undefined_flag_consumer_still_stops() {
    // xor leaves AF undefined; a jcc chain that reads OF after a multi-bit shift
    // must stop rather than trust a guess. shl eax,3 leaves OF undefined; jo reads it.
    let code = asm(|a| {
        let mut l = a.create_label();
        a.mov(eax, 1).unwrap();
        a.shl(eax, 3u32).unwrap();
        a.jo(l).unwrap();
        a.set_label(&mut l).unwrap();
        a.nop().unwrap();
    });
    let mut e = emu_bytes(&code);
    let s = e.run(10);
    assert!(matches!(s, Stop::UndefinedFlags { mask, .. } if mask == OF), "{s:?}");
}

#[test]
fn xmm_move_fails_closed() {
    // movaps xmm0, xmm1 (0F 28 C1): XMM state is deferred, so it must stop
    // cleanly as unsupported rather than guess.
    let mut e = emu_bytes(&[0x0f, 0x28, 0xc1]);
    let before = e.cpu.clone();
    let s = e.step().unwrap_err();
    assert!(matches!(s, Stop::Unsupported { .. }), "{s:?}");
    assert_eq!(e.cpu, before); // rolled back
    assert!(e.trace.is_empty());
}

#[test]
fn carry_flag_controls() {
    // clc/stc/cmc exact carry manipulation.
    let e = run(&asm(|a| {
        a.stc().unwrap();
        a.cmc().unwrap(); // CF: 1 -> 0
        a.nop().unwrap();
    }));
    assert!(!e.cpu.flag(CF));
}

#[test]
fn rdtsc_is_monotonic_and_deterministic() {
    // Two reads with work between them: the second tick is strictly larger
    // (EAX low dword; EDX stays 0 within the budget). Capture the first in R8D.
    let code = asm(|a| {
        a.rdtsc().unwrap();
        a.mov(r8d, eax).unwrap();
        a.nop().unwrap();
        a.nop().unwrap();
        a.rdtsc().unwrap();
        a.nop().unwrap();
    });
    let e1 = run(&code);
    let first = e1.cpu.regs[8] & 0xffff_ffff;
    let second = e1.cpu.regs[0] & 0xffff_ffff;
    assert!(second > first, "tsc monotonic: {second} > {first}");
    assert_eq!(e1.cpu.regs[2] & 0xffff_ffff, 0, "EDX high dword 0 in-budget");
    // Deterministic: an independent run yields identical ticks (no host clock).
    let e2 = run(&code);
    assert_eq!(e2.cpu.regs[8] & 0xffff_ffff, first);
    assert_eq!(e2.cpu.regs[0] & 0xffff_ffff, second);
}

#[test]
fn cpuid_fixed_signature() {
    // Leaf 0: "GenuineIntel" vendor string in EBX/EDX/ECX, max leaf in EAX.
    let e = run(&asm(|a| {
        a.mov(eax, 0).unwrap();
        a.cpuid().unwrap();
        a.nop().unwrap();
    }));
    assert_eq!(e.cpu.regs[3] & 0xffff_ffff, 0x756e_6547); // EBX "Genu"
    assert_eq!(e.cpu.regs[1] & 0xffff_ffff, 0x6c65_746e); // ECX "ntel"
    assert_eq!(e.cpu.regs[2] & 0xffff_ffff, 0x4965_6e69); // EDX "ineI"
    assert_eq!(e.cpu.regs[0] & 0xffff_ffff, 0x16); // EAX max standard leaf
    // Leaf 1: SSE2 advertised (EDX bit 26), no AVX/SSE4 (ECX == 0).
    let e = run(&asm(|a| {
        a.mov(eax, 1).unwrap();
        a.cpuid().unwrap();
        a.nop().unwrap();
    }));
    assert_ne!(e.cpu.regs[2] & (1 << 26), 0, "SSE2 feature bit set");
    assert_eq!(e.cpu.regs[1] & 0xffff_ffff, 0, "ECX feature set empty (no AVX)");
}

#[test]
fn shrd_16bit_overwidth_is_undefined_not_unsupported() {
    // `shrd dx, cx, 30` (66 0f ac ca 1e): the imm masks (mod 32) to 30 >= 16, a
    // 16-bit count past the operand width. The matching VMProtect form used to
    // fail closed as Unsupported; now it retires with the destination unchanged
    // and the status flags left undefined (bsf/bsr convention).
    let mut e = emu_bytes(&[0x66, 0x0f, 0xac, 0xca, 0x1e, 0x90]);
    e.cpu.regs[2] = 0x1234; // DX
    e.cpu.regs[1] = 0xFFFF; // CX
    assert_eq!(e.run_until(CODE + 5, 10), Stop::Reached, "retired, not Unsupported");
    assert_eq!(e.cpu.regs[2] & 0xffff, 0x1234, "destination preserved");
    // A consumer of a status flag it left undefined must then stop.
    let mut e = emu_bytes(&[0x66, 0x0f, 0xac, 0xca, 0x1e, 0x70, 0x00]); // + jo
    assert!(matches!(e.run(10), Stop::UndefinedFlags { .. }));
}

#[test]
fn bswap_16bit_is_undefined_not_unsupported() {
    // `bswap ax` (66 0f c8) is architecturally undefined; model it as a no-op
    // (destination unchanged) rather than stopping. 32-bit bswap still swaps.
    let mut e = emu_bytes(&[0x66, 0x0f, 0xc8, 0x90]);
    e.cpu.regs[0] = 0x1234; // AX
    assert_eq!(e.run_until(CODE + 3, 10), Stop::Reached);
    assert_eq!(e.cpu.regs[0] & 0xffff, 0x1234, "16-bit bswap leaves dest unchanged");
    let e = run(&asm(|a| {
        a.mov(eax, 0x1122_3344u32).unwrap();
        a.bswap(eax).unwrap();
        a.nop().unwrap();
    }));
    assert_eq!(e.cpu.regs[0] & 0xffff_ffff, 0x4433_2211);
}

#[test]
fn shift_past_width_defines_carry_as_zero() {
    // `shl al, cl` with CL=9 (> 8-bit width): result 0 and CF is a defined 0 on
    // real silicon, so a CF consumer (adc) must NOT stop on an undefined flag.
    let e = run(&asm(|a| {
        a.mov(al, 0xFFu32).unwrap();
        a.mov(cl, 9u32).unwrap();
        a.shl(al, cl).unwrap();
        a.mov(bl, 5u32).unwrap();
        a.adc(bl, 0u32).unwrap(); // bl = 5 + CF; stops if CF were undefined
        a.nop().unwrap();
    }));
    assert_eq!(e.cpu.regs[0] & 0xff, 0, "shifted fully out");
    assert!(!e.cpu.flag(CF), "CF defined as 0");
    assert_eq!(e.cpu.regs[3] & 0xff, 5, "adc saw CF=0");
}

#[test]
fn cmc_propagates_undefined_carry() {
    // `bsf` leaves CF undefined; `cmc` is a pure CF->!CF transform, so it must
    // propagate the undefinedness (not invent a value) and not stop. A later
    // real consumer (jc) still stops.
    let code = asm(|a| {
        a.mov(eax, 1u32).unwrap();
        a.bsf(ebx, eax).unwrap(); // CF undefined
        a.cmc().unwrap(); // propagates: no stop
        a.stc().unwrap(); // redefines CF
        a.nop().unwrap();
    });
    // Runs to completion: cmc did not fail closed, stc re-defined CF.
    let mut e = emu_bytes(&code);
    assert_eq!(e.run_until(CODE + code.len() as u64 - 1, 50), Stop::Reached);
    assert!(e.cpu.flag(CF));
    // But a consumer reading CF straight after the propagating cmc stops.
    let code = asm(|a| {
        let mut l = a.create_label();
        a.mov(eax, 1u32).unwrap();
        a.bsf(ebx, eax).unwrap();
        a.cmc().unwrap();
        a.jc(l).unwrap(); // reads the still-undefined CF
        a.set_label(&mut l).unwrap();
        a.nop().unwrap();
    });
    let mut e = emu_bytes(&code);
    assert!(matches!(e.run(50), Stop::UndefinedFlags { mask, .. } if mask == CF));
}
