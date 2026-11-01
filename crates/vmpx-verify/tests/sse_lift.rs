//! Differential test of the native SSE lifter against the contained software
//! emulator (`vmpx_emulate::sse_fp`, the semantic reference): every snippet is
//! lifted, evaluated at concrete inputs, and compared to the emulator's
//! registers, XMM lanes and ZF/PF/CF/OF/SF/AF.

use std::collections::BTreeMap;

use iced_x86::code_asm::*;
use vmpx_disasm::{
    Algebra, AlgebraError, BinOp, Flag, FlagState, Gpr, Resize, UnOp, Width, lift_bytes_with,
};
use vmpx_emulate::{Cpu, Emulator, Memory, Perm, Stop};
use vmpx_mutation::VmAlgebra;
use vmpx_mutation::cfg_recovery::CfgOptions;
use vmpx_vm::{BitWidth, Expr};

const CODE: u64 = 0x1000;
const DATA: u64 = 0x2000;

/// `VmAlgebra` with memory loads left as `memld(addr)` so a test can resolve them.
struct TestAlgebra;

impl Algebra for TestAlgebra {
    type Expr = Expr;
    fn constant(&self, w: Width, v: u64) -> Expr {
        VmAlgebra.constant(w, v)
    }
    fn input(&self, w: Width, n: &str) -> Expr {
        VmAlgebra.input(w, n)
    }
    fn load(&self, w: Width, a: Expr) -> Expr {
        VmAlgebra.intrinsic(w, "memld", vec![a])
    }
    fn unary(&self, op: UnOp, v: Expr) -> Result<Expr, AlgebraError> {
        VmAlgebra.unary(op, v)
    }
    fn binary(&self, op: BinOp, l: Expr, r: Expr) -> Result<Expr, AlgebraError> {
        VmAlgebra.binary(op, l, r)
    }
    fn resize(&self, k: Resize, t: Width, v: Expr) -> Result<Expr, AlgebraError> {
        VmAlgebra.resize(k, t, v)
    }
    fn intrinsic(&self, w: Width, n: &str, a: Vec<Expr>) -> Expr {
        VmAlgebra.intrinsic(w, n, a)
    }
    fn width(&self, e: &Expr) -> Width {
        VmAlgebra.width(e)
    }
    fn as_const(&self, e: &Expr) -> Option<u64> {
        VmAlgebra.as_const(e)
    }
}

const SPECIAL64: [u64; 14] = [
    0x0000_0000_0000_0000, // +0
    0x8000_0000_0000_0000, // -0
    0x3FF0_0000_0000_0000, // 1.0
    0xBFF8_0000_0000_0000, // -1.5
    0x4059_0000_0000_0000, // 100.0
    0x0000_0000_0000_0001, // min denormal
    0x7FEF_FFFF_FFFF_FFFF, // max
    0x7FF0_0000_0000_0000, // +inf
    0xFFF0_0000_0000_0000, // -inf
    0x7FF8_0000_0000_00AB, // qNaN payload
    0x7FF0_0000_0000_0001, // sNaN
    0xFFF8_0000_0000_0000, // default NaN
    0x43E0_0000_0000_0000, // 2^63
    0xC3E0_0000_0000_0000, // -2^63
];

const SPECIAL32: [u32; 12] = [
    0x0000_0000,
    0x8000_0000,
    0x3F80_0000,
    0xBFC0_0000,
    0x42C8_0000,
    0x0000_0001,
    0x7F7F_FFFF,
    0x7F80_0000,
    0xFF80_0000,
    0x7FC0_00AB,
    0x7F80_0001,
    0xFFC0_0000,
];

const FLAG_MASK: u64 = 0x8D5; // CF PF AF ZF SF OF

fn eval(e: &Expr, env: &BTreeMap<String, Expr>, mem: &[u8; 64]) -> u64 {
    try_eval(e, env, mem).unwrap_or_else(|| panic!("did not fold to a constant: {e:?}"))
}

fn try_eval(e: &Expr, env: &BTreeMap<String, Expr>, mem: &[u8; 64]) -> Option<u64> {
    let s = e.substitute(env).expect("substitute").simplify();
    let r = s
        .rewrite_intrinsics(&mut |w, name, args| {
            if name != "memld" {
                return None;
            }
            let addr = match args {
                [a] => match a.simplify().kind {
                    vmpx_vm::ExprKind::Const { value } => value,
                    _ => return None,
                },
                _ => return None,
            };
            let off = usize::try_from(addr - DATA).ok()?;
            let n = (w.bits() / 8) as usize;
            let mut v = 0u64;
            for i in 0..n {
                v |= u64::from(mem[off + i]) << (8 * i);
            }
            Some(Expr::constant(w, v))
        })
        .expect("rewrite")
        .simplify();
    match r.kind {
        vmpx_vm::ExprKind::Const { value } => Some(value),
        _ => None,
    }
}

struct Case {
    gpr: [u64; 16],
    xmm: [u128; 16],
    mem: [u8; 64],
}

fn check(name: &str, build: impl FnOnce(&mut CodeAssembler), cases: &[Case]) {
    let mut a = CodeAssembler::new(64).unwrap();
    build(&mut a);
    a.nop().unwrap();
    let bytes = a.assemble(CODE).unwrap();
    let lifted = lift_bytes_with(
        &TestAlgebra,
        &bytes[..bytes.len() - 1],
        CODE,
        CfgOptions::PROVEN_FP.into(),
    )
    .unwrap_or_else(|e| panic!("{name}: lift failed: {e}"));
    for (ci, c) in cases.iter().enumerate() {
        let mut mem = Memory::new();
        mem.map(CODE, 0x1000, Perm::RWX).unwrap();
        mem.map(DATA, 0x1000, Perm::RW).unwrap();
        mem.poke(CODE, &bytes).unwrap();
        mem.write(DATA, &c.mem).unwrap();
        let mut e = Emulator::new(Cpu::new(CODE), mem);
        e.cpu.regs = c.gpr;
        e.cpu.xmm = c.xmm;
        e.cpu.rflags = 0x202;
        let stop = e.run_until(CODE + bytes.len() as u64 - 1, 1000);
        assert_eq!(stop, Stop::Reached, "{name}: emulator stopped: {:?}", e.trace.last());

        let mut env: BTreeMap<String, Expr> = BTreeMap::new();
        for (i, g) in Gpr::ALL.iter().enumerate() {
            env.insert(g.input_name(), Expr::constant(BitWidth::W64, c.gpr[i]));
        }
        for n in 0..16 {
            env.insert(format!("xmm{n}_lo_in"), Expr::constant(BitWidth::W64, c.xmm[n] as u64));
            env.insert(
                format!("xmm{n}_hi_in"),
                Expr::constant(BitWidth::W64, (c.xmm[n] >> 64) as u64),
            );
        }
        for f in ["cf", "pf", "af", "zf", "sf", "of"] {
            env.insert(format!("{f}_in"), Expr::constant(BitWidth::W8, 0));
        }
        for (i, g) in Gpr::ALL.iter().enumerate() {
            if *g == Gpr::Rsp {
                continue;
            }
            let got = eval(&lifted.regs[g], &env, &c.mem);
            assert_eq!(got, e.cpu.regs[i], "{name} case {ci}: {} lifted {got:#x} emu {:#x}", g.name(), e.cpu.regs[i]);
        }
        for n in 0..16u8 {
            let (lo, hi) = &lifted.xmm[&n];
            let got = u128::from(eval(lo, &env, &c.mem)) | u128::from(eval(hi, &env, &c.mem)) << 64;
            assert_eq!(got, e.cpu.xmm[n as usize], "{name} case {ci}: xmm{n} lifted {got:#x} emu {:#x}", e.cpu.xmm[n as usize]);
        }
        for (flag, st) in &lifted.flags {
            let FlagState::Value(v) = st else { continue };
            let bit = match flag {
                Flag::Cf => 0,
                Flag::Pf => 2,
                Flag::Af => 4,
                Flag::Zf => 6,
                Flag::Sf => 7,
                Flag::Of => 11,
            };
            if matches!(flag, Flag::Af | Flag::Sf | Flag::Of | Flag::Cf | Flag::Pf | Flag::Zf)
                && (FLAG_MASK >> bit) & 1 == 1
            {
                // Integer flag intrinsics (cf_add, ...) are opaque here; SSE flags fold.
                let Some(got) = try_eval(v, &env, &c.mem) else { continue };
                let want = (e.cpu.rflags >> bit) & 1;
                assert_eq!(got, want, "{name} case {ci}: flag {flag:?}");
            }
        }
        // MXCSR-independent: stores into DATA are checked through the writes list.
        for w in &lifted.writes {
            let addr = eval(&w.address, &env, &c.mem);
            let val = eval(&w.value, &env, &c.mem);
            let mut buf = [0u8; 8];
            e.mem.read(addr, &mut buf[..w.width.bytes() as usize]).unwrap();
            let have = u64::from_le_bytes(buf);
            assert_eq!(have, val, "{name} case {ci}: store at {addr:#x}");
        }
    }
}

fn base_gpr() -> [u64; 16] {
    let mut g = [0u64; 16];
    for (i, v) in g.iter_mut().enumerate() {
        *v = 0x0123_4567_89AB_CDEFu64.rotate_left(i as u32 * 7) ^ (i as u64 * 0x1111);
    }
    g[4] = 0x2f08; // rsp
    g[3] = DATA; // rbx
    g[6] = DATA + 16; // rsi
    g
}

fn base_xmm() -> [u128; 16] {
    let mut x = [0u128; 16];
    for (i, v) in x.iter_mut().enumerate() {
        *v = (0xA5A5_0000_1111_2222u128 << 64 | i as u128) ^ (i as u128 * 0x0101_0101);
    }
    x
}

/// Cases pairing every special f64 with every other in `xmm0`/`xmm1` and as memory.
fn cases64() -> Vec<Case> {
    let mut v = Vec::new();
    for a in SPECIAL64 {
        for b in SPECIAL64 {
            let mut c = Case { gpr: base_gpr(), xmm: base_xmm(), mem: [0; 64] };
            c.xmm[0] = (0xDEAD_BEEF_CAFE_F00Du128 << 64) | u128::from(a);
            c.xmm[1] = (0x0123_4567_89AB_CDEFu128 << 64) | u128::from(b);
            c.mem[..8].copy_from_slice(&b.to_le_bytes());
            c.mem[8..16].copy_from_slice(&a.to_le_bytes());
            v.push(c);
        }
    }
    v
}

fn cases32() -> Vec<Case> {
    let mut v = Vec::new();
    for a in SPECIAL32 {
        for b in SPECIAL32 {
            let mut c = Case { gpr: base_gpr(), xmm: base_xmm(), mem: [0; 64] };
            c.xmm[0] = (0xDEAD_BEEF_CAFE_F00Du128 << 64) | (0x1234_5678u128 << 32) | u128::from(a);
            c.xmm[1] = (0x0123_4567_89AB_CDEFu128 << 64) | (0x9ABC_DEF0u128 << 32) | u128::from(b);
            c.mem[..4].copy_from_slice(&b.to_le_bytes());
            v.push(c);
        }
    }
    v
}

fn cases_int() -> Vec<Case> {
    let ints: [u64; 8] = [0, 1, u64::MAX, 7, 0x8000_0000, 0x7FFF_FFFF, 1 << 63, 0xFFFF_FFFF_8000_0000];
    ints.iter()
        .map(|&i| {
            let mut c = Case { gpr: base_gpr(), xmm: base_xmm(), mem: [0; 64] };
            c.gpr[0] = i;
            c.gpr[2] = i ^ 0x55;
            c.mem[..8].copy_from_slice(&i.to_le_bytes());
            c
        })
        .collect()
}

#[test]
fn scalar_double_arithmetic() {
    let cases_v = cases64();
    check("addsd rr", |a| a.addsd(xmm0, xmm1).unwrap(), &cases_v);
    check("subsd rr", |a| a.subsd(xmm0, xmm1).unwrap(), &cases_v);
    check("mulsd rr", |a| a.mulsd(xmm0, xmm1).unwrap(), &cases_v);
    check("divsd rr", |a| a.divsd(xmm0, xmm1).unwrap(), &cases_v);
    check("minsd rr", |a| a.minsd(xmm0, xmm1).unwrap(), &cases_v);
    check("maxsd rr", |a| a.maxsd(xmm0, xmm1).unwrap(), &cases_v);
    check("sqrtsd rr", |a| a.sqrtsd(xmm0, xmm1).unwrap(), &cases_v);
    check("addsd rm", |a| a.addsd(xmm0, qword_ptr(rbx)).unwrap(), &cases_v);
    check("divsd rm", |a| a.divsd(xmm0, qword_ptr(rbx)).unwrap(), &cases_v);
    check("sqrtsd rm", |a| a.sqrtsd(xmm0, qword_ptr(rbx)).unwrap(), &cases_v);
    check("chain", |a| {
        a.mulsd(xmm0, xmm1).unwrap();
        a.addsd(xmm0, xmm1).unwrap();
        a.sqrtsd(xmm0, xmm0).unwrap();
    }, &cases_v);
}

#[test]
fn scalar_single_arithmetic() {
    let cases_v = cases32();
    check("addss", |a| a.addss(xmm0, xmm1).unwrap(), &cases_v);
    check("subss", |a| a.subss(xmm0, xmm1).unwrap(), &cases_v);
    check("mulss", |a| a.mulss(xmm0, xmm1).unwrap(), &cases_v);
    check("divss", |a| a.divss(xmm0, xmm1).unwrap(), &cases_v);
    check("minss", |a| a.minss(xmm0, xmm1).unwrap(), &cases_v);
    check("maxss", |a| a.maxss(xmm0, xmm1).unwrap(), &cases_v);
    check("sqrtss", |a| a.sqrtss(xmm0, xmm1).unwrap(), &cases_v);
    check("mulss rm", |a| a.mulss(xmm0, dword_ptr(rbx)).unwrap(), &cases_v);
}

#[test]
fn compares_and_flags() {
    let cases_v = cases64();
    check("ucomisd", |a| a.ucomisd(xmm0, xmm1).unwrap(), &cases_v);
    check("comisd", |a| a.comisd(xmm0, xmm1).unwrap(), &cases_v);
    check("ucomisd rm", |a| a.ucomisd(xmm0, qword_ptr(rbx)).unwrap(), &cases_v);
    let cases_v = cases32();
    check("ucomiss", |a| a.ucomiss(xmm0, xmm1).unwrap(), &cases_v);
    check("comiss", |a| a.comiss(xmm0, xmm1).unwrap(), &cases_v);
    let cases_v = cases64();
    for pred in 0..8u32 {
        check(&format!("cmpsd {pred}"), |a| a.cmpsd_3(xmm0, xmm1, pred).unwrap(), &cases_v);
    }
    let cases_v = cases32();
    for pred in 0..8u32 {
        check(&format!("cmpss {pred}"), |a| a.cmpss(xmm0, xmm1, pred).unwrap(), &cases_v);
    }
}

#[test]
fn conversions() {
    let cases_v = cases_int();
    check("cvtsi2sd r32", |a| a.cvtsi2sd(xmm0, eax).unwrap(), &cases_v);
    check("cvtsi2sd r64", |a| a.cvtsi2sd(xmm0, rax).unwrap(), &cases_v);
    check("cvtsi2ss r32", |a| a.cvtsi2ss(xmm0, edx).unwrap(), &cases_v);
    check("cvtsi2ss r64", |a| a.cvtsi2ss(xmm0, rdx).unwrap(), &cases_v);
    check("cvtsi2sd m32", |a| a.cvtsi2sd(xmm0, dword_ptr(rbx)).unwrap(), &cases_v);
    check("cvtsi2sd m64", |a| a.cvtsi2sd(xmm0, qword_ptr(rbx)).unwrap(), &cases_v);
    let cases_v = cases64();
    check("cvttsd2si r32", |a| a.cvttsd2si(eax, xmm0).unwrap(), &cases_v);
    check("cvttsd2si r64", |a| a.cvttsd2si(rax, xmm1).unwrap(), &cases_v);
    check("cvtsd2ss", |a| a.cvtsd2ss(xmm0, xmm1).unwrap(), &cases_v);
    check("cvtsd2ss m", |a| a.cvtsd2ss(xmm0, qword_ptr(rbx)).unwrap(), &cases_v);
    let cases_v = cases32();
    check("cvttss2si r32", |a| a.cvttss2si(eax, xmm0).unwrap(), &cases_v);
    check("cvttss2si r64", |a| a.cvttss2si(rax, xmm1).unwrap(), &cases_v);
    check("cvtss2sd", |a| a.cvtss2sd(xmm0, xmm1).unwrap(), &cases_v);
    check("cvtss2sd m", |a| a.cvtss2sd(xmm0, dword_ptr(rbx)).unwrap(), &cases_v);
}

#[test]
fn moves_and_lanes() {
    let cases_v = cases64();
    check("movsd rr", |a| a.movsd_2(xmm0, xmm1).unwrap(), &cases_v);
    check("movsd rm", |a| a.movsd_2(xmm0, qword_ptr(rbx)).unwrap(), &cases_v);
    check("movsd mr", |a| a.movsd_2(qword_ptr(rsi), xmm1).unwrap(), &cases_v);
    check("movss rr", |a| a.movss(xmm0, xmm1).unwrap(), &cases_v);
    check("movss rm", |a| a.movss(xmm0, dword_ptr(rbx)).unwrap(), &cases_v);
    check("movss mr", |a| a.movss(dword_ptr(rsi), xmm1).unwrap(), &cases_v);
    check("movaps rr", |a| a.movaps(xmm2, xmm0).unwrap(), &cases_v);
    check("movaps rm", |a| a.movaps(xmm2, xmmword_ptr(rbx)).unwrap(), &cases_v);
    check("movups mr", |a| a.movups(xmmword_ptr(rsi), xmm1).unwrap(), &cases_v);
    check("movapd rr", |a| a.movapd(xmm2, xmm1).unwrap(), &cases_v);
    check("movdqa rr", |a| a.movdqa(xmm2, xmm1).unwrap(), &cases_v);
    check("movdqu rm", |a| a.movdqu(xmm2, xmmword_ptr(rbx)).unwrap(), &cases_v);
    check("movq rr", |a| a.movq(xmm2, xmm1).unwrap(), &cases_v);
    check("movq rm", |a| a.movq(xmm2, qword_ptr(rbx)).unwrap(), &cases_v);
    check("movq mr", |a| a.movq(qword_ptr(rsi), xmm1).unwrap(), &cases_v);
    check("movq x,r64", |a| a.movq(xmm2, rax).unwrap(), &cases_v);
    check("movq r64,x", |a| a.movq(rax, xmm1).unwrap(), &cases_v);
    check("movd x,r32", |a| a.movd(xmm2, eax).unwrap(), &cases_v);
    check("movd r32,x", |a| a.movd(eax, xmm1).unwrap(), &cases_v);
    check("movd x,m32", |a| a.movd(xmm2, dword_ptr(rbx)).unwrap(), &cases_v);
    check("movlps rm", |a| a.movlps(xmm0, qword_ptr(rbx)).unwrap(), &cases_v);
    check("movlpd mr", |a| a.movlpd(qword_ptr(rsi), xmm0).unwrap(), &cases_v);
    check("movhps rm", |a| a.movhps(xmm0, qword_ptr(rbx)).unwrap(), &cases_v);
    check("movhpd mr", |a| a.movhpd(qword_ptr(rsi), xmm0).unwrap(), &cases_v);
    check("movhlps", |a| a.movhlps(xmm0, xmm1).unwrap(), &cases_v);
    check("movlhps", |a| a.movlhps(xmm0, xmm1).unwrap(), &cases_v);
    check("unpcklpd", |a| a.unpcklpd(xmm0, xmm1).unwrap(), &cases_v);
    check("unpckhpd", |a| a.unpckhpd(xmm0, xmm1).unwrap(), &cases_v);
    for imm in 0..4u32 {
        check(&format!("shufpd {imm}"), |a| a.shufpd(xmm0, xmm1, imm).unwrap(), &cases_v);
    }
}

#[test]
fn bitwise_sign_mask_idioms() {
    let cases_v = cases64();
    check("xorps zero", |a| a.xorps(xmm0, xmm0).unwrap(), &cases_v);
    check("pxor zero", |a| a.pxor(xmm1, xmm1).unwrap(), &cases_v);
    check("xorpd rm", |a| a.xorpd(xmm0, xmmword_ptr(rbx)).unwrap(), &cases_v);
    check("andpd rr", |a| a.andpd(xmm0, xmm1).unwrap(), &cases_v);
    check("andnpd rr", |a| a.andnpd(xmm0, xmm1).unwrap(), &cases_v);
    check("orpd rm", |a| a.orpd(xmm0, xmmword_ptr(rbx)).unwrap(), &cases_v);
    check("andps rr", |a| a.andps(xmm0, xmm1).unwrap(), &cases_v);
    check("pand", |a| a.pand(xmm0, xmm1).unwrap(), &cases_v);
    check("por", |a| a.por(xmm0, xmm1).unwrap(), &cases_v);
    check("pandn", |a| a.pandn(xmm0, xmm1).unwrap(), &cases_v);
    check("fabs via andnpd mask", |a| {
        a.movapd(xmm2, xmm1).unwrap();
        a.andnpd(xmm2, xmm0).unwrap();
    }, &cases_v);
}

#[test]
fn spill_and_reload_through_the_frame() {
    let cases_v = cases64();
    check("movaps spill/reload", |a| {
        a.sub(rsp, 0x28).unwrap();
        a.movaps(xmmword_ptr(rsp + 0x10), xmm6).unwrap();
        a.movaps(xmm6, xmm0).unwrap();
        a.addsd(xmm6, xmm1).unwrap();
        a.movaps(xmm0, xmm6).unwrap();
        a.movaps(xmm6, xmmword_ptr(rsp + 0x10)).unwrap();
        a.add(rsp, 0x28).unwrap();
    }, &cases_v);
}

fn lift_err(build: impl FnOnce(&mut CodeAssembler)) -> String {
    let mut a = CodeAssembler::new(64).unwrap();
    build(&mut a);
    let bytes = a.assemble(CODE).unwrap();
    lift_bytes_with(&TestAlgebra, &bytes, CODE, CfgOptions::PROVEN_FP.into())
        .expect_err("must fail closed")
        .to_string()
}

#[test]
fn unmodeled_forms_fail_closed() {
    for text in [
        lift_err(|a| a.addpd(xmm0, xmm1).unwrap()),
        lift_err(|a| a.mulps(xmm0, xmm1).unwrap()),
        lift_err(|a| a.cvtsd2si(eax, xmm0).unwrap()),
        lift_err(|a| a.cvtss2si(eax, xmm0).unwrap()),
        lift_err(|a| a.shufps(xmm0, xmm1, 3).unwrap()),
        lift_err(|a| a.ldmxcsr(dword_ptr(rbx)).unwrap()),
        lift_err(|a| a.vaddsd(xmm0, xmm1, xmm2).unwrap()),
        lift_err(|a| a.roundsd(xmm0, xmm1, 1).unwrap()),
        lift_err(|a| a.movmskpd(eax, xmm0).unwrap()),
        lift_err(|a| a.unpcklps(xmm0, xmm1).unwrap()),
        lift_err(|a| a.movss(xmm0, xmm1).and_then(|()| a.pshufd(xmm0, xmm1, 1)).unwrap()),
    ] {
        assert!(text.contains("not modeled") || text.contains("unsupported"), "{text}");
    }
}

#[test]
fn xmm_is_off_unless_requested() {
    let mut a = CodeAssembler::new(64).unwrap();
    a.addsd(xmm0, xmm1).unwrap();
    let bytes = a.assemble(CODE).unwrap();
    assert!(lift_bytes_with(&TestAlgebra, &bytes, CODE, CfgOptions::PROVEN.into()).is_err());
}

#[test]
fn string_movsd_is_not_sse() {
    let err = lift_bytes_with(&TestAlgebra, &[0xA5], CODE, CfgOptions::PROVEN_FP.into())
        .expect_err("string movsd is not modeled");
    assert!(err.to_string().contains("not modeled"), "{err}");
}
