//! SSE/XMM subset: known vectors plus the fail-closed path.
use iced_x86::Mnemonic as M;
use iced_x86::code_asm::*;
use vmpx_emulate::{Cpu, Emulator, Memory, Perm, Stop};

const CODE: u64 = 0x1000;
const DATA: u64 = 0x2000;

fn emu(code: &[u8]) -> Emulator {
    let mut mem = Memory::new();
    mem.map(CODE, 0x1000, Perm::RWX).unwrap();
    mem.map(DATA, 0x1000, Perm::RW).unwrap();
    mem.poke(CODE, code).unwrap();
    Emulator::new(Cpu::new(CODE), mem)
}

fn asm(f: impl FnOnce(&mut CodeAssembler)) -> Vec<u8> {
    let mut a = CodeAssembler::new(64).unwrap();
    f(&mut a);
    a.nop().unwrap();
    a.assemble(CODE).unwrap()
}

fn run(code: &[u8], setup: impl FnOnce(&mut Emulator)) -> Emulator {
    let mut e = emu(code);
    setup(&mut e);
    let end = CODE + code.len() as u64 - 1;
    assert_eq!(e.run_until(end, 1000), Stop::Reached, "trace {:?}", e.trace.last());
    e
}

const A: u128 = 0x0000_0004_0000_0003_0000_0002_ffff_ffff;
const B: u128 = 0x0000_0001_0000_0001_0000_0001_0000_0001;

#[test]
fn pxor_zeroing_idiom() {
    let e = run(&asm(|a| a.pxor(xmm3, xmm3).unwrap()), |e| e.cpu.xmm[3] = A);
    assert_eq!(e.cpu.xmm[3], 0);
}

#[test]
fn movdqu_roundtrip_and_movdqa_alignment() {
    let code = asm(|a| {
        a.mov(rbx, DATA + 3).unwrap();
        a.movdqu(xmm0, xmmword_ptr(rbx)).unwrap();
        a.movdqu(xmmword_ptr(rbx + 0x40), xmm0).unwrap();
        a.movdqu(xmm1, xmmword_ptr(rbx + 0x40)).unwrap();
    });
    let e = run(&code, |e| {
        e.mem.write(DATA + 3, &A.to_le_bytes()).unwrap();
    });
    assert_eq!(e.cpu.xmm[0], A);
    assert_eq!(e.cpu.xmm[1], A);
    let mut buf = [0u8; 16];
    e.mem.read(DATA + 0x43, &mut buf).unwrap();
    assert_eq!(u128::from_le_bytes(buf), A);

    // movdqa on a misaligned address faults and leaves xmm0 untouched.
    let code = asm(|a| {
        a.mov(rbx, DATA + 8).unwrap();
        a.movdqa(xmm0, xmmword_ptr(rbx)).unwrap();
    });
    let mut e = emu(&code);
    e.cpu.xmm[0] = 7;
    assert!(matches!(e.run(10), Stop::Fault { .. }));
    assert_eq!(e.cpu.xmm[0], 7);
}

#[test]
fn paddd_psubd_lane_math() {
    let e = run(
        &asm(|a| {
            a.movdqa(xmm2, xmm0).unwrap();
            a.paddd(xmm0, xmm1).unwrap();
            a.psubd(xmm2, xmm1).unwrap();
        }),
        |e| {
            e.cpu.xmm[0] = A;
            e.cpu.xmm[1] = B;
        },
    );
    // lanes (low->high): ffffffff+1 wraps to 0; 2+1; 3+1; 4+1.
    assert_eq!(e.cpu.xmm[0], 0x0000_0005_0000_0004_0000_0003_0000_0000);
    assert_eq!(e.cpu.xmm[2], 0x0000_0003_0000_0002_0000_0001_ffff_fffe);
}

#[test]
fn compare_unpack_shuffle_shift() {
    let e = run(
        &asm(|a| {
            a.pcmpeqd(xmm0, xmm1).unwrap();
            a.pshufd(xmm2, xmm3, 0x1b).unwrap(); // reverse dwords
            a.punpcklbw(xmm4, xmm5).unwrap();
            a.psrld(xmm6, 4).unwrap();
            a.pslldq(xmm7, 4).unwrap();
            a.movq(rax, xmm3).unwrap();
            a.movd(xmm8, eax).unwrap();
        }),
        |e| {
            e.cpu.xmm[0] = A;
            e.cpu.xmm[1] = (A & !(0xffff_ffffu128 << 64)) | (9u128 << 64);
            e.cpu.xmm[3] = 0x0000_0004_0000_0003_0000_0002_0000_0001;
            e.cpu.xmm[4] = 0x0807_0605_0403_0201;
            e.cpu.xmm[5] = 0x1817_1615_1413_1211;
            e.cpu.xmm[6] = 0xf0f0_f0f0_0000_0010_0000_0001_8000_0000;
            e.cpu.xmm[7] = 0x4433_2211;
        },
    );
    // dword 2 differs (3 vs 9); the others match.
    assert_eq!(e.cpu.xmm[0], 0xffff_ffff_0000_0000_ffff_ffff_ffff_ffff);
    assert_eq!(e.cpu.xmm[2], 0x0000_0001_0000_0002_0000_0003_0000_0004);
    assert_eq!(e.cpu.xmm[4], 0x1808_1707_1606_1505_1404_1303_1202_1101);
    assert_eq!(e.cpu.xmm[6], 0x0f0f_0f0f_0000_0001_0000_0000_0800_0000);
    assert_eq!(e.cpu.xmm[7], 0x4433_2211_0000_0000);
    assert_eq!(e.cpu.regs[0], 0x0000_0002_0000_0001);
    assert_eq!(e.cpu.xmm[8], 1);
}

#[test]
fn addr32_override_wraps_to_32_bits() {
    // mov rax, gs:[ebx+8] (65 67 48 8B 43 08): high bits of rbx are ignored.
    let mut e = emu(&[0x65, 0x67, 0x48, 0x8b, 0x43, 0x08]);
    e.cpu.gs_base = DATA;
    e.cpu.regs[3] = 0xdead_beef_0000_0010;
    e.mem.write_uint(DATA + 0x18, 8, 0x1122_3344_5566_7788).unwrap();
    e.step().unwrap();
    assert_eq!(e.cpu.regs[0], 0x1122_3344_5566_7788);
}

#[test]
fn unmodeled_sse_fails_closed() {
    // addss xmm0, xmm1
    let mut e = emu(&[0xf3, 0x0f, 0x58, 0xc1]);
    let s = e.step().unwrap_err();
    assert_eq!(s, Stop::UnmodeledSse { rip: CODE, mnemonic: M::Addss });
    assert_eq!(s.to_string(), "unmodeled SSE: addss at 0x1000");
    // vpxor xmm0, xmm1, xmm2 (VEX) is not silently treated as pxor.
    let mut e = emu(&[0xc5, 0xf1, 0xef, 0xc2]);
    assert!(matches!(e.step(), Err(Stop::UnmodeledSse { .. })));
}
