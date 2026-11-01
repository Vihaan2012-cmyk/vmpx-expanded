//! SSE/SSE2 float + move/shuffle subset: hand-computed vectors, NaN/MXCSR rules, fail-closed.
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

fn run_raw(bytes: &[u8], setup: impl FnOnce(&mut Emulator)) -> Emulator {
    let mut code = bytes.to_vec();
    code.push(0x90);
    let mut e = emu(&code);
    setup(&mut e);
    assert_eq!(e.run_until(CODE + code.len() as u64 - 1, 1000), Stop::Reached, "trace {:?}", e.trace.last());
    e
}

fn run(f: impl FnOnce(&mut CodeAssembler), setup: impl FnOnce(&mut Emulator)) -> Emulator {
    let code = asm(f);
    let mut e = emu(&code);
    setup(&mut e);
    assert_eq!(e.run_until(CODE + code.len() as u64 - 1, 1000), Stop::Reached, "trace {:?}", e.trace.last());
    e
}

fn stop_of(bytes: &[u8], setup: impl FnOnce(&mut Emulator)) -> Stop {
    let mut e = emu(bytes);
    setup(&mut e);
    e.step().unwrap_err()
}

fn d(v: f64) -> u128 {
    u128::from(v.to_bits())
}
fn s(v: f32) -> u128 {
    u128::from(v.to_bits())
}

const HI: u128 = 0xdead_beef_cafe_f00d << 64;
const QNAN64: u64 = 0xfff8_0000_0000_0000;
const QNAN32: u32 = 0xffc0_0000;
const ZF: u64 = 1 << 6;
const PF: u64 = 1 << 2;
const CF: u64 = 1;

#[test]
fn movsd_movss_forms() {
    // movsd xmm0,[rbx] (zero-extends); movsd xmm1,xmm2 (merges low); movsd [rbx+16],xmm3
    // F2 0F 10 03 / F2 0F 10 CA / F2 0F 11 5B 10
    let e = run_raw(&[0xf2, 0x0f, 0x10, 0x03, 0xf2, 0x0f, 0x10, 0xca, 0xf2, 0x0f, 0x11, 0x5b, 0x10], |e| {
        e.cpu.regs[3] = DATA;
        e.mem.write(DATA, &1.5f64.to_le_bytes()).unwrap();
        e.cpu.xmm[0] = u128::MAX;
        e.cpu.xmm[1] = HI | 0x1111;
        e.cpu.xmm[2] = 0xaaaa_bbbb_cccc_dddd_u128 << 64 | d(7.0);
        e.cpu.xmm[3] = u128::MAX;
    });
    assert_eq!(e.cpu.xmm[0], d(1.5));
    assert_eq!(e.cpu.xmm[1], HI | d(7.0));
    let mut b = [0u8; 8];
    e.mem.read(DATA + 16, &mut b).unwrap();
    assert_eq!(b, u64::MAX.to_le_bytes());
    let mut b = [0u8; 9];
    e.mem.read(DATA + 16, &mut b).unwrap();
    assert_eq!(b[8], 0, "store wrote only 8 bytes");

    // movss xmm0,[rbx]; movss xmm1,xmm2; movss [rbx+16],xmm3
    let e = run_raw(&[0xf3, 0x0f, 0x10, 0x03, 0xf3, 0x0f, 0x10, 0xca, 0xf3, 0x0f, 0x11, 0x5b, 0x10], |e| {
        e.cpu.regs[3] = DATA;
        e.mem.write(DATA, &2.5f32.to_le_bytes()).unwrap();
        e.cpu.xmm[0] = u128::MAX;
        e.cpu.xmm[1] = HI | 0x1111_2222;
        e.cpu.xmm[2] = s(9.0);
        e.cpu.xmm[3] = 0x1234_5678;
    });
    assert_eq!(e.cpu.xmm[0], s(2.5));
    assert_eq!(e.cpu.xmm[1], HI | s(9.0));
    assert_eq!(e.mem.read_uint(DATA + 16, 4).unwrap(), 0x1234_5678);
    assert_eq!(e.mem.read_uint(DATA + 20, 4).unwrap(), 0);
}

#[test]
fn scalar_double_arith() {
    let e = run(
        |a| {
            a.addsd(xmm0, xmm1).unwrap(); // 1.5 + 2.25
            a.subsd(xmm2, xmm1).unwrap(); // 10 - 2.25
            a.mulsd(xmm3, xmm1).unwrap(); // 4 * 2.25
            a.divsd(xmm4, xmm1).unwrap(); // 4.5 / 2.25
            a.sqrtsd(xmm5, xmm6).unwrap(); // sqrt(2)
            a.minsd(xmm7, xmm1).unwrap();
            a.maxsd(xmm8, xmm1).unwrap();
        },
        |e| {
            e.cpu.xmm[0] = HI | d(1.5);
            e.cpu.xmm[1] = d(2.25);
            e.cpu.xmm[2] = d(10.0);
            e.cpu.xmm[3] = d(4.0);
            e.cpu.xmm[4] = d(4.5);
            e.cpu.xmm[5] = HI | 99;
            e.cpu.xmm[6] = d(2.0);
            e.cpu.xmm[7] = d(3.0);
            e.cpu.xmm[8] = d(3.0);
        },
    );
    assert_eq!(e.cpu.xmm[0], HI | d(3.75), "upper lane preserved");
    assert_eq!(e.cpu.xmm[2], d(7.75));
    assert_eq!(e.cpu.xmm[3], d(9.0));
    assert_eq!(e.cpu.xmm[4], d(2.0));
    assert_eq!(e.cpu.xmm[5], HI | 0x3ff6_a09e_667f_3bcd, "sqrt(2) correctly rounded");
    assert_eq!(e.cpu.xmm[7], d(2.25));
    assert_eq!(e.cpu.xmm[8], d(3.0));
}

#[test]
fn scalar_single_arith_and_rounding() {
    let e = run(
        |a| {
            a.addss(xmm0, xmm1).unwrap();
            a.addss(xmm2, xmm3).unwrap();
            a.divss(xmm4, xmm5).unwrap();
        },
        |e| {
            e.cpu.xmm[0] = HI | s(1.5);
            e.cpu.xmm[1] = s(2.25);
            // 2^24 + 3 is a tie between 2^24+2 and 2^24+4; nearest-even picks 2^24+4
            e.cpu.xmm[2] = s(16_777_216.0);
            e.cpu.xmm[3] = s(3.0);
            e.cpu.xmm[4] = s(1.0);
            e.cpu.xmm[5] = s(3.0);
        },
    );
    assert_eq!(e.cpu.xmm[0], HI | s(3.75));
    assert_eq!(e.cpu.xmm[2], s(16_777_220.0));
    assert_eq!(e.cpu.xmm[4], 0x3eaa_aaab, "1/3 in f32 = 0x3eaaaaab (RNE)");
}

#[test]
fn nan_and_special_rules() {
    let q1 = 0x7ff8_0000_0000_1111u64;
    let q2 = 0x7ff8_0000_0000_2222u64;
    let snan = 0x7ff0_0000_0000_0001u64;
    let e = run(
        |a| {
            a.addsd(xmm0, xmm1).unwrap(); // both NaN -> first, quiet
            a.addsd(xmm2, xmm3).unwrap(); // SNaN first -> quieted
            a.addsd(xmm4, xmm5).unwrap(); // 1 + qnan -> qnan (second)
            a.subsd(xmm6, xmm7).unwrap(); // inf - inf -> default QNaN
            a.mulsd(xmm8, xmm9).unwrap(); // 0 * inf -> default QNaN
            a.divsd(xmm10, xmm11).unwrap(); // 1 / 0 -> +inf
            a.minsd(xmm12, xmm13).unwrap(); // NaN operand -> second
            a.maxsd(xmm14, xmm15).unwrap(); // +0 vs -0 -> second
        },
        |e| {
            e.cpu.xmm[0] = u128::from(q1);
            e.cpu.xmm[1] = u128::from(q2);
            e.cpu.xmm[2] = u128::from(snan);
            e.cpu.xmm[3] = d(1.0);
            e.cpu.xmm[4] = d(1.0);
            e.cpu.xmm[5] = u128::from(q2);
            e.cpu.xmm[6] = d(f64::INFINITY);
            e.cpu.xmm[7] = d(f64::INFINITY);
            e.cpu.xmm[8] = d(0.0);
            e.cpu.xmm[9] = d(f64::INFINITY);
            e.cpu.xmm[10] = d(1.0);
            e.cpu.xmm[11] = d(0.0);
            e.cpu.xmm[12] = u128::from(q1);
            e.cpu.xmm[13] = d(5.0);
            e.cpu.xmm[14] = d(0.0);
            e.cpu.xmm[15] = d(-0.0);
        },
    );
    assert_eq!(e.cpu.xmm[0], u128::from(q1));
    assert_eq!(e.cpu.xmm[2], u128::from(snan | 1 << 51));
    assert_eq!(e.cpu.xmm[4], u128::from(q2));
    assert_eq!(e.cpu.xmm[6], u128::from(QNAN64));
    assert_eq!(e.cpu.xmm[8], u128::from(QNAN64));
    assert_eq!(e.cpu.xmm[10], d(f64::INFINITY));
    assert_eq!(e.cpu.xmm[12], d(5.0));
    assert_eq!(e.cpu.xmm[14], d(-0.0));

    // sqrtss(-1) -> default single QNaN
    let e = run(|a| a.sqrtss(xmm0, xmm1).unwrap(), |e| e.cpu.xmm[1] = s(-1.0));
    assert_eq!(e.cpu.xmm[0], u128::from(QNAN32));
}

#[test]
fn packed_arith_and_cmp() {
    let e = run(
        |a| {
            a.addps(xmm0, xmm1).unwrap();
            a.mulpd(xmm2, xmm3).unwrap();
            a.cmpps(xmm4, xmm5, 1).unwrap();
            a.cmpsd_3(xmm6, xmm7, 0).unwrap();
            a.cmppd(xmm8, xmm9, 3).unwrap();
        },
        |e| {
            let ps = |v: [f32; 4]| v.iter().enumerate().fold(0u128, |m, (i, x)| m | (s(*x) << (32 * i)));
            e.cpu.xmm[0] = ps([1.0, 2.0, 3.0, 4.0]);
            e.cpu.xmm[1] = ps([0.5, 0.5, 0.5, 0.5]);
            e.cpu.xmm[2] = d(3.0) | d(-2.0) << 64;
            e.cpu.xmm[3] = d(1.5) | d(4.0) << 64;
            e.cpu.xmm[4] = ps([1.0, 5.0, f32::NAN, 2.0]);
            e.cpu.xmm[5] = ps([2.0, 5.0, 1.0, 1.0]);
            e.cpu.xmm[6] = HI | d(2.0);
            e.cpu.xmm[7] = d(2.0);
            e.cpu.xmm[8] = d(f64::NAN) | d(1.0) << 64;
            e.cpu.xmm[9] = d(1.0) | d(2.0) << 64;
        },
    );
    let ps = |v: [f32; 4]| v.iter().enumerate().fold(0u128, |m, (i, x)| m | (s(*x) << (32 * i)));
    assert_eq!(e.cpu.xmm[0], ps([1.5, 2.5, 3.5, 4.5]));
    assert_eq!(e.cpu.xmm[2], d(4.5) | d(-8.0) << 64);
    // lanes: 1<2 true, 5<5 false, NaN<1 false, 2<1 false
    assert_eq!(e.cpu.xmm[4], 0xffff_ffff);
    assert_eq!(e.cpu.xmm[6], HI | u128::from(u64::MAX));
    assert_eq!(e.cpu.xmm[8], u128::from(u64::MAX));
}

#[test]
fn comis_flags() {
    let flags = |a: f64, b: f64, ucomi: bool| {
        let code: &[u8] = if ucomi { &[0x66, 0x0f, 0x2e, 0xc1] } else { &[0x66, 0x0f, 0x2f, 0xc1] };
        let e = run_raw(code, |e| {
            e.cpu.xmm[0] = d(a);
            e.cpu.xmm[1] = d(b);
            e.cpu.rflags |= (1 << 11) | (1 << 7) | (1 << 4); // OF/SF/AF must be cleared
            e.cpu.undef_flags = 0xffff;
        });
        assert_eq!(e.cpu.rflags & ((1 << 11) | (1 << 7) | (1 << 4)), 0);
        assert_eq!(e.cpu.undef_flags & (ZF | PF | CF), 0, "flags now defined");
        e.cpu.rflags & (ZF | PF | CF)
    };
    assert_eq!(flags(1.0, 2.0, false), CF);
    assert_eq!(flags(2.0, 2.0, false), ZF);
    assert_eq!(flags(3.0, 2.0, true), 0);
    assert_eq!(flags(f64::NAN, 2.0, true), ZF | PF | CF);
    // comiss: 0F 2F C1
    let e = run_raw(&[0x0f, 0x2f, 0xc1], |e| {
        e.cpu.xmm[0] = s(1.0);
        e.cpu.xmm[1] = s(-1.0);
    });
    assert_eq!(e.cpu.rflags & (ZF | PF | CF), 0);
}

#[test]
fn conversions() {
    // cvtsi2sd xmm0,eax (-5); cvtsi2sd xmm1,rcx (2^53+1 -> RNE 2^53); cvtsi2ss xmm2,edx
    let e = run(
        |a| {
            a.cvtsi2sd(xmm0, eax).unwrap();
            a.cvtsi2sd(xmm1, rcx).unwrap();
            a.cvtsi2ss(xmm2, edx).unwrap();
        },
        |e| {
            e.cpu.regs[0] = 0xffff_fffb;
            e.cpu.regs[1] = (1 << 53) + 1;
            e.cpu.regs[2] = 16_777_217; // 2^24+1 -> RNE 2^24 in f32
            e.cpu.xmm[0] = HI;
        },
    );
    assert_eq!(e.cpu.xmm[0], HI | 0xc014_0000_0000_0000);
    assert_eq!(e.cpu.xmm[1], 0x4340_0000_0000_0000);
    assert_eq!(e.cpu.xmm[2], s(16_777_216.0));

    // truncating vs rounding double -> int
    let e = run(
        |a| {
            a.cvttsd2si(eax, xmm0).unwrap(); // -2.9 -> -2
            a.cvtsd2si(ebx, xmm1).unwrap(); // 2.5 -> 2 (ties-even)
            a.cvtsd2si(ecx, xmm2).unwrap(); // 3.5 -> 4
            a.cvttsd2si(edx, xmm3).unwrap(); // NaN -> indefinite
            a.cvttsd2si(rsi, xmm3).unwrap(); // NaN -> 64-bit indefinite
            a.cvttsd2si(edi, xmm4).unwrap(); // 3e9 out of i32 -> indefinite
            a.cvttsd2si(r8, xmm4).unwrap(); // 3e9 fits i64
            a.cvttss2si(r9d, xmm5).unwrap(); // -7.75f -> -7
        },
        |e| {
            e.cpu.xmm[0] = d(-2.9);
            e.cpu.xmm[1] = d(2.5);
            e.cpu.xmm[2] = d(3.5);
            e.cpu.xmm[3] = d(f64::NAN);
            e.cpu.xmm[4] = d(3.0e9);
            e.cpu.xmm[5] = s(-7.75);
        },
    );
    assert_eq!(e.cpu.regs[0], 0xffff_fffe);
    assert_eq!(e.cpu.regs[3], 2);
    assert_eq!(e.cpu.regs[1], 4);
    assert_eq!(e.cpu.regs[2], 0x8000_0000);
    assert_eq!(e.cpu.regs[6], 1 << 63);
    assert_eq!(e.cpu.regs[7], 0x8000_0000);
    assert_eq!(e.cpu.regs[8], 3_000_000_000);
    assert_eq!(e.cpu.regs[9], 0xffff_fff9);

    // float <-> double
    let e = run(
        |a| {
            a.cvtss2sd(xmm0, xmm1).unwrap();
            a.cvtsd2ss(xmm2, xmm3).unwrap();
            a.cvtss2sd(xmm4, xmm5).unwrap();
        },
        |e| {
            e.cpu.xmm[0] = HI | 0x77;
            e.cpu.xmm[1] = s(1.5);
            e.cpu.xmm[2] = HI | 0x77;
            e.cpu.xmm[3] = d(0.1);
            e.cpu.xmm[5] = 0x7f80_0001; // single SNaN -> quiet double NaN, payload kept
        },
    );
    assert_eq!(e.cpu.xmm[0], HI | d(1.5));
    assert_eq!(e.cpu.xmm[2] & u128::from(u32::MAX), 0x3dcc_cccd, "0.1 -> f32 (RNE)");
    assert_eq!(e.cpu.xmm[2] >> 32, (HI | 0x77) >> 32, "bits 127:32 preserved");
    assert_eq!(e.cpu.xmm[4], 0x7ff8_0000_2000_0000);
}

#[test]
fn moves_halves_and_masks() {
    let lo = 0x1111_2222_3333_4444u64;
    let hi = 0x5555_6666_7777_8888u64;
    let both = u128::from(lo) | u128::from(hi) << 64;
    // movhps xmm0,[rbx]; movlps xmm1,[rbx]; movhps [rbx+16],xmm2; movhlps xmm3,xmm4; movlhps xmm5,xmm6
    let e = run(
        |a| {
            a.mov(rbx, DATA).unwrap();
            a.movhps(xmm0, qword_ptr(rbx)).unwrap();
            a.movlps(xmm1, qword_ptr(rbx)).unwrap();
            a.movhps(qword_ptr(rbx + 16), xmm2).unwrap();
            a.movhlps(xmm3, xmm4).unwrap();
            a.movlhps(xmm5, xmm6).unwrap();
            a.movmskps(eax, xmm7).unwrap();
            a.movmskpd(ecx, xmm7).unwrap();
        },
        |e| {
            e.mem.write(DATA, &0xabcd_ef01_2345_6789u64.to_le_bytes()).unwrap();
            e.cpu.xmm[0] = both;
            e.cpu.xmm[1] = both;
            e.cpu.xmm[2] = both;
            e.cpu.xmm[3] = both;
            e.cpu.xmm[4] = 0x9999_aaaa_bbbb_cccc_u128 << 64 | 0xdddd;
            e.cpu.xmm[5] = both;
            e.cpu.xmm[6] = 0x0123_4567_89ab_cdef_u128 | 0xffff_u128 << 64;
            // dword sign bits: lane0 = 1, lane1 = 0, lane2 = 1, lane3 = 1 -> 0b1101
            e.cpu.xmm[7] = 0x8000_0000 | 0x8000_0000_u128 << 64 | 0xc000_0000_u128 << 96;
        },
    );
    assert_eq!(e.cpu.xmm[0], u128::from(lo) | 0xabcd_ef01_2345_6789_u128 << 64);
    assert_eq!(e.cpu.xmm[1], 0xabcd_ef01_2345_6789_u128 | u128::from(hi) << 64);
    assert_eq!(e.mem.read_uint(DATA + 16, 8).unwrap(), hi);
    assert_eq!(e.cpu.xmm[3], u128::from(0x9999_aaaa_bbbb_cccc_u64) | u128::from(hi) << 64);
    assert_eq!(e.cpu.xmm[5], u128::from(lo) | 0x0123_4567_89ab_cdef_u128 << 64);
    assert_eq!(e.cpu.regs[0], 0b1101);
    assert_eq!(e.cpu.regs[1], 0b10, "pd sign bits: qword0 bit63 clear, qword1 set");
}

#[test]
fn shuffles_and_integer_extras() {
    let ps = |v: [u32; 4]| v.iter().enumerate().fold(0u128, |m, (i, x)| m | (u128::from(*x) << (32 * i)));
    let e = run(
        |a| {
            a.shufps(xmm0, xmm1, 0b11_00_01_10).unwrap();
            a.unpcklps(xmm2, xmm3).unwrap();
            a.unpckhps(xmm4, xmm5).unwrap();
            a.shufpd(xmm6, xmm7, 0b10).unwrap();
            a.pshuflw(xmm8, xmm9, 0x1b).unwrap();
            a.pshufhw(xmm10, xmm11, 0x1b).unwrap();
            a.pmuludq(xmm12, xmm13).unwrap();
            a.pmullw(xmm14, xmm15).unwrap();
        },
        |e| {
            e.cpu.xmm[0] = ps([10, 11, 12, 13]);
            e.cpu.xmm[1] = ps([20, 21, 22, 23]);
            e.cpu.xmm[2] = ps([1, 2, 3, 4]);
            e.cpu.xmm[3] = ps([5, 6, 7, 8]);
            e.cpu.xmm[4] = ps([1, 2, 3, 4]);
            e.cpu.xmm[5] = ps([5, 6, 7, 8]);
            e.cpu.xmm[6] = 0xa_u128 | 0xb_u128 << 64;
            e.cpu.xmm[7] = 0xc_u128 | 0xd_u128 << 64;
            e.cpu.xmm[9] = 0x0008_0007_0006_0005_0004_0003_0002_0001;
            e.cpu.xmm[11] = 0x0008_0007_0006_0005_0004_0003_0002_0001;
            e.cpu.xmm[12] = 0xffff_ffff_u128 | 7u128 << 64;
            e.cpu.xmm[13] = 0xffff_ffff_u128 | 3u128 << 64;
            e.cpu.xmm[14] = 0x0002_0003_ffff_0100;
            e.cpu.xmm[15] = 0x0004_0005_0002_0100;
        },
    );
    // imm 0b11_00_01_10: dst[2], dst[1], src[0], src[3]
    assert_eq!(e.cpu.xmm[0], ps([12, 11, 20, 23]));
    assert_eq!(e.cpu.xmm[2], ps([1, 5, 2, 6]));
    assert_eq!(e.cpu.xmm[4], ps([3, 7, 4, 8]));
    // shufpd imm 0b10: low = dst[0]=a, high = src[1]=d
    assert_eq!(e.cpu.xmm[6], 0xa_u128 | 0xd_u128 << 64);
    // pshuflw reverses low 4 words; high half copied from src
    assert_eq!(e.cpu.xmm[8], 0x0008_0007_0006_0005_0001_0002_0003_0004);
    assert_eq!(e.cpu.xmm[10], 0x0005_0006_0007_0008_0004_0003_0002_0001);
    // 0xffffffff * 0xffffffff = 0xfffffffe00000001 ; 7*3 = 21
    assert_eq!(e.cpu.xmm[12], 0xffff_fffe_0000_0001_u128 | 21u128 << 64);
    // words: 0x0100*0x0100 = 0x10000 -> 0; 0xffff*2 = 0x1fffe -> 0xfffe; 3*5 = 15; 2*4 = 8
    assert_eq!(e.cpu.xmm[14], 0x0008_000f_fffe_0000);

    // packuswb: words (-1, 0x100, 0x7f, 0xff) saturate to (0, 255, 0x7f, 255)
    let e = run(
        |a| {
            a.packuswb(xmm0, xmm1).unwrap();
            a.pinsrw(xmm2, ebx, 3).unwrap();
            a.pextrw(eax, xmm3, 5).unwrap();
        },
        |e| {
            e.cpu.xmm[0] = 0x00ff_007f_0100_ffff;
            e.cpu.xmm[1] = 0x8000_7fff_0001_0000;
            e.cpu.regs[3] = 0xabcd_1234;
            e.cpu.xmm[2] = u128::MAX;
            e.cpu.xmm[3] = 0x0007_0006_beef_0004_0003_0002_0001_0000;
        },
    );
    assert_eq!(e.cpu.xmm[0], 0xff7f_ff00 | 0x00ff_0100_u128 << 64);
    assert_eq!(e.cpu.xmm[2], u128::MAX & !(0xffff_u128 << 48) | 0x1234_u128 << 48);
    assert_eq!(e.cpu.regs[0], 0xbeef);
}

#[test]
fn nondefault_mxcsr_fails_closed() {
    // addsd xmm0,xmm1
    let add = [0xf2, 0x0f, 0x58, 0xc1];
    for m in [0x1f80 | 0x2000, 0x1f80 | 0x8000, 0x1f80 | 0x40, 0x1f80 & !0x80] {
        let st = stop_of(&add, |e| e.cpu.mxcsr = m);
        assert_eq!(st, Stop::UnmodeledSse { rip: CODE, mnemonic: M::Addsd }, "mxcsr {m:#x}");
    }
    // sticky exception flag bits alone do not change the environment
    let mut e = emu(&[0xf2, 0x0f, 0x58, 0xc1]);
    e.cpu.mxcsr = 0x1f80 | 0x3f;
    e.step().unwrap();
}

#[test]
fn still_unmodeled_ops_fail_closed() {
    // pshufb xmm0,xmm1 (SSSE3, not advertised by CPUID), cvtdq2ps, vaddsd
    for (b, mn) in [
        (&[0x66u8, 0x0f, 0x38, 0x00, 0xc1][..], M::Pshufb),
        (&[0x0f, 0x5b, 0xc1][..], M::Cvtdq2ps),
        (&[0xc5, 0xfb, 0x58, 0xc1][..], M::Vaddsd),
    ] {
        let st = stop_of(b, |_| {});
        assert!(matches!(st, Stop::UnmodeledSse { mnemonic, .. } if mnemonic == mn), "{mn:?}: {st:?}");
    }
}

/// CRT-startup shape: zero a block with `pxor`+`movdqu`/`movups` stores in a loop,
/// then copy it with `movdqu` load/store pairs and `movaps`.
#[test]
fn steps_past_crt_style_vector_loops() {
    let e = run(
        |a| {
            let mut top = a.create_label();
            a.mov(rdi, DATA + 0x100).unwrap();
            a.mov(rsi, DATA).unwrap();
            a.mov(ecx, 8u32).unwrap();
            a.pxor(xmm0, xmm0).unwrap();
            a.set_label(&mut top).unwrap();
            a.movdqu(xmmword_ptr(rdi), xmm0).unwrap(); // zero fill, unaligned
            a.add(rdi, 16).unwrap();
            a.dec(ecx).unwrap();
            a.jnz(top).unwrap();
            // copy 32 bytes from DATA (offset 1: misaligned) to DATA+0x200
            a.movdqu(xmm1, xmmword_ptr(rsi + 1)).unwrap();
            a.movdqu(xmm2, xmmword_ptr(rsi + 17)).unwrap();
            a.movups(xmmword_ptr(rsi + 0x201), xmm1).unwrap();
            a.movaps(xmmword_ptr(rsi + 0x220), xmm2).unwrap();
        },
        |e| {
            for i in 0..0x180u64 {
                e.mem.write(DATA + i, &[0xff]).unwrap();
            }
            for i in 0..48u8 {
                e.mem.write(DATA + u64::from(i), &[i]).unwrap();
            }
        },
    );
    let rd = |addr: u64, n: usize| {
        let mut b = vec![0u8; n];
        e.mem.read(addr, &mut b).unwrap();
        b
    };
    assert_eq!(rd(DATA + 0x100, 128), vec![0u8; 128]);
    assert_eq!(rd(DATA + 0x201, 16), (1..17).collect::<Vec<u8>>());
    assert_eq!(rd(DATA + 0x220, 16), (17..33).collect::<Vec<u8>>());
}
