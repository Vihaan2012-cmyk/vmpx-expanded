//! SSE/SSE2 floating-point + remaining move/shuffle subset. Child of `emu`.
//!
//! Scalar `{add,sub,mul,div,min,max,sqrt}{ss,sd}`, packed `{add,sub,mul,div}{ps,pd}`,
//! `movss/movsd`, `mov{l,h}{ps,pd}`, `movhlps/movlhps`, `movmsk{ps,pd}`,
//! `cmp{ss,sd,ps,pd}`, `(u)comis{s,d}`, `cvt{ss2sd,sd2ss,si2ss,si2sd,(t)ss2si,(t)sd2si}`,
//! `shuf{ps,pd}`, `unpck{l,h}{ps,pd}`, `pshuf{l,h}w`, `pmuludq`, `pmullw`,
//! `packuswb`, `pinsrw`, `pextrw`.
//!
//! Exactness: arithmetic runs on host IEEE-754 (round-to-nearest-even); NaN
//! results follow the x86 rules explicitly (first-operand NaN wins and is
//! quieted, invalid operations give the default QNaN), so they do not depend on
//! the host. Every float op fails closed unless MXCSR is the default
//! environment (round-nearest, FTZ/DAZ off, all exceptions masked).
//!
//! Not modeled: the sticky MXCSR exception flags (IE/ZE/OE/UE/PE are never
//! set), so `stmxcsr`/`fetestexcept` observe a stale value.

#![allow(clippy::float_cmp, clippy::neg_cmp_op_on_partial_ord, clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_possible_wrap, clippy::cast_precision_loss)]

#[allow(clippy::wildcard_imports)]
use super::*;
use super::sse::{lanes, unpack};
use iced_x86::MemorySize;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Op {
    Add,
    Sub,
    Mul,
    Div,
    Min,
    Max,
    Sqrt,
}

/// One IEEE format, operated on as zero-extended bit patterns.
trait Fp {
    const BYTES: u32;
    fn bin(op: Op, a: u64, b: u64) -> u64;
    fn cmp(pred: u8, a: u64, b: u64) -> bool;
    /// `(u)comis` status flags (ZF/PF/CF).
    fn comis(a: u64, b: u64) -> u64;
}

macro_rules! fp_impl {
    ($t:ident, $f:ty, $u:ty, $bytes:expr, $quiet:expr, $dnan:expr) => {
        struct $t;
        impl Fp for $t {
            const BYTES: u32 = $bytes;
            fn bin(op: Op, a: u64, b: u64) -> u64 {
                let (a, b) = (a as $u, b as $u);
                let (x, y) = (<$f>::from_bits(a), <$f>::from_bits(b));
                let r: $u = match op {
                    // Either-NaN / both-zero / unordered all return the second operand.
                    Op::Min => if x < y { a } else { b },
                    Op::Max => if x > y { a } else { b },
                    Op::Sqrt if y.is_nan() => b | $quiet,
                    Op::Sqrt if y < 0.0 => $dnan,
                    Op::Sqrt => y.sqrt().to_bits(),
                    _ if x.is_nan() => a | $quiet,
                    _ if y.is_nan() => b | $quiet,
                    _ => {
                        let r = match op {
                            Op::Add => x + y,
                            Op::Sub => x - y,
                            Op::Mul => x * y,
                            _ => x / y,
                        };
                        if r.is_nan() { $dnan } else { r.to_bits() }
                    }
                };
                u64::from(r)
            }
            fn cmp(pred: u8, a: u64, b: u64) -> bool {
                let (x, y) = (<$f>::from_bits(a as $u), <$f>::from_bits(b as $u));
                let un = x.is_nan() || y.is_nan();
                match pred {
                    0 => x == y,
                    1 => x < y,
                    2 => x <= y,
                    3 => un,
                    4 => x != y,
                    5 => !(x < y),
                    6 => !(x <= y),
                    _ => !un,
                }
            }
            fn comis(a: u64, b: u64) -> u64 {
                let (x, y) = (<$f>::from_bits(a as $u), <$f>::from_bits(b as $u));
                if x.is_nan() || y.is_nan() {
                    ZF | PF | CF
                } else if x < y {
                    CF
                } else if x == y {
                    ZF
                } else {
                    0
                }
            }
        }
    };
}
fp_impl!(F32, f32, u32, 4, 1 << 22, 0xffc0_0000);
fp_impl!(F64, f64, u64, 8, 1 << 51, 0xfff8_0000_0000_0000);

/// `cvt(t)ss2si`/`cvt(t)sd2si` core: out-of-range and NaN give the integer indefinite.
fn to_int(x: f64, wide: bool, nearest: bool) -> u64 {
    let t = if nearest { x.round_ties_even() } else { x.trunc() };
    if wide {
        if x.is_nan() || !(-9_223_372_036_854_775_808.0..9_223_372_036_854_775_808.0).contains(&t) {
            return 1 << 63;
        }
        (t as i64) as u64
    } else {
        if x.is_nan() || !(-2_147_483_648.0..2_147_483_648.0).contains(&t) {
            return 0x8000_0000;
        }
        u64::from(t as i32 as u32)
    }
}

/// `cvtss2sd` payload-preserving widen (a NaN is quieted).
fn widen(b: u32) -> u64 {
    let f = f32::from_bits(b);
    if f.is_nan() {
        (u64::from(b >> 31) << 63) | 0x7ff8_0000_0000_0000 | (u64::from(b & 0x007f_ffff) << 29)
    } else {
        f64::from(f).to_bits()
    }
}

/// `cvtsd2ss` narrow (round-to-nearest-even; a NaN keeps its top payload bits, quieted).
fn narrow(b: u64) -> u64 {
    let d = f64::from_bits(b);
    if d.is_nan() {
        ((b >> 63) << 31) | 0x7fc0_0000 | ((b >> 29) & 0x007f_ffff)
    } else {
        u64::from((d as f32).to_bits())
    }
}

fn sat_u8(w: u64) -> u64 {
    let s = w as u16 as i16;
    if s < 0 { 0 } else if s > 255 { 255 } else { s as u64 }
}

impl Emulator {
    fn fp_env(&self) -> Result<(), Stop> {
        // RC (13-14), FTZ (15), DAZ (6) clear and all exception masks (7-12) set.
        if self.cpu.mxcsr & 0xe040 == 0 && self.cpu.mxcsr & 0x1f80 == 0x1f80 {
            Ok(())
        } else {
            Err(self.unmodeled())
        }
    }

    fn xdst_idx(&self, ins: &Instruction) -> Result<usize, Stop> {
        if ins.op_kind(0) != OpKind::Register {
            return Err(self.unmodeled());
        }
        self.xmm_idx(ins.op0_register())
    }

    /// Replace the low `bytes` of xmm `ix`, preserving the rest.
    fn set_low(&mut self, ix: usize, bytes: u32, v: u128) {
        let m = (1u128 << (bytes * 8)) - 1;
        self.cpu.xmm[ix] = (self.cpu.xmm[ix] & !m) | (v & m);
    }

    fn fp_scalar<W: Fp>(&mut self, ins: &Instruction, op: Op) -> Result<(), Stop> {
        self.fp_env()?;
        let ix = self.xdst_idx(ins)?;
        let b = self.xsrc(ins, 1, W::BYTES, false)? as u64;
        let a = self.cpu.xmm[ix] as u64 & mask(W::BYTES);
        self.set_low(ix, W::BYTES, u128::from(W::bin(op, a, b)));
        Ok(())
    }

    fn fp_packed<W: Fp>(&mut self, ins: &Instruction, op: Op) -> Result<(), Stop> {
        self.fp_env()?;
        let ix = self.xdst_idx(ins)?;
        let b = self.xsrc(ins, 1, 16, true)?;
        self.cpu.xmm[ix] = lanes(self.cpu.xmm[ix], b, W::BYTES * 8, |x, y| W::bin(op, x, y));
        Ok(())
    }

    fn fp_cmp<W: Fp>(&mut self, ins: &Instruction, scalar: bool) -> Result<(), Stop> {
        self.fp_env()?;
        let ix = self.xdst_idx(ins)?;
        let pred = ins.immediate(2) as u8;
        if ins.op_kind(2) != OpKind::Immediate8 || pred > 7 {
            return Err(self.unmodeled());
        }
        let bits = W::BYTES * 8;
        let f = |x, y| if W::cmp(pred, x, y) { u64::MAX } else { 0 };
        if scalar {
            let b = self.xsrc(ins, 1, W::BYTES, false)? as u64;
            let a = self.cpu.xmm[ix] as u64 & mask(W::BYTES);
            self.set_low(ix, W::BYTES, u128::from(f(a, b)));
        } else {
            let b = self.xsrc(ins, 1, 16, true)?;
            self.cpu.xmm[ix] = lanes(self.cpu.xmm[ix], b, bits, f);
        }
        Ok(())
    }

    fn comis<W: Fp>(&mut self, ins: &Instruction) -> Result<(), Stop> {
        let ix = self.xdst_idx(ins)?;
        let b = self.xsrc(ins, 1, W::BYTES, false)? as u64;
        let a = self.cpu.xmm[ix] as u64 & mask(W::BYTES);
        let all = OF | SF | ZF | AF | PF | CF;
        self.cpu.rflags = (self.cpu.rflags & !all) | W::comis(a, b);
        self.cpu.undef_flags &= !all;
        Ok(())
    }

    /// Width in bytes of a `cvtsi2s{s,d}` integer source.
    fn int_src_bytes(&self, ins: &Instruction) -> Result<u32, Stop> {
        match ins.op_kind(1) {
            OpKind::Register if ins.op1_register().is_gpr64() => Ok(8),
            OpKind::Register if ins.op1_register().is_gpr32() => Ok(4),
            OpKind::Memory if ins.memory_size() == MemorySize::Int64 => Ok(8),
            OpKind::Memory if ins.memory_size() == MemorySize::Int32 => Ok(4),
            _ => Err(self.unmodeled()),
        }
    }

    fn cvt_from_int(&mut self, ins: &Instruction, dbl: bool) -> Result<(), Stop> {
        self.fp_env()?;
        let ix = self.xdst_idx(ins)?;
        let n = self.int_src_bytes(ins)?;
        let raw = self.xsrc(ins, 1, n, false)? as u64;
        let v = if n == 8 { raw as i64 } else { i64::from(raw as u32 as i32) };
        if dbl {
            self.set_low(ix, 8, u128::from((v as f64).to_bits()));
        } else {
            self.set_low(ix, 4, u128::from((v as f32).to_bits()));
        }
        Ok(())
    }

    fn cvt_to_int(&mut self, ins: &Instruction, dbl: bool, nearest: bool) -> Result<(), Stop> {
        self.fp_env()?;
        let r = ins.op0_register();
        if ins.op_kind(0) != OpKind::Register || !(r.is_gpr32() || r.is_gpr64()) {
            return Err(self.unmodeled());
        }
        let n = if dbl { 8 } else { 4 };
        let b = self.xsrc(ins, 1, n, false)? as u64;
        let x = if dbl { f64::from_bits(b) } else { f64::from(f32::from_bits(b as u32)) };
        self.cpu.set_reg(r, to_int(x, r.is_gpr64(), nearest));
        Ok(())
    }

    /// `movlps/movlpd/movhps/movhpd`: one 64-bit half against memory.
    fn mov_half(&mut self, ins: &Instruction, hi: bool) -> Result<(), Stop> {
        let sh = if hi { 64 } else { 0 };
        match (ins.op_kind(0), ins.op_kind(1)) {
            (OpKind::Register, OpKind::Memory) => {
                let ix = self.xdst_idx(ins)?;
                let v = self.xsrc(ins, 1, 8, false)?;
                let keep = self.cpu.xmm[ix] & !(u128::from(u64::MAX) << sh);
                self.cpu.xmm[ix] = keep | (v << sh);
                Ok(())
            }
            (OpKind::Memory, OpKind::Register) => {
                let v = self.xsrc(ins, 1, 16, false)? >> sh;
                self.xdst(ins, 0, 8, false, v)
            }
            _ => Err(self.unmodeled()),
        }
    }

    fn mov_scalar(&mut self, ins: &Instruction, n: u32) -> Result<(), Stop> {
        match (ins.op_kind(0), ins.op_kind(1)) {
            (OpKind::Register, OpKind::Register) => {
                let ix = self.xdst_idx(ins)?;
                let v = self.xsrc(ins, 1, n, false)?;
                self.set_low(ix, n, v);
                Ok(())
            }
            // Load zero-extends; store writes the low `n` bytes.
            (OpKind::Register, OpKind::Memory) | (OpKind::Memory, OpKind::Register) => {
                let v = self.xsrc(ins, 1, n, false)?;
                self.xdst(ins, 0, n, false, v)
            }
            _ => Err(self.unmodeled()),
        }
    }

    pub(super) fn exec_fp(&mut self, ins: &Instruction, mn: Mnemonic) -> Result<(), Stop> {
        use Mnemonic as M;
        match mn {
            M::Movss => self.mov_scalar(ins, 4),
            M::Movsd => self.mov_scalar(ins, 8),
            M::Movlps | M::Movlpd => self.mov_half(ins, false),
            M::Movhps | M::Movhpd => self.mov_half(ins, true),
            M::Movhlps | M::Movlhps => {
                let ix = self.xdst_idx(ins)?;
                let s = self.xsrc(ins, 1, 16, false)?;
                let d = self.cpu.xmm[ix];
                self.cpu.xmm[ix] = if mn == M::Movhlps {
                    (d & !u128::from(u64::MAX)) | (s >> 64)
                } else {
                    (d & u128::from(u64::MAX)) | (s << 64)
                };
                Ok(())
            }
            M::Movmskps | M::Movmskpd => {
                let r = ins.op0_register();
                if ins.op_kind(0) != OpKind::Register || !(r.is_gpr32() || r.is_gpr64()) {
                    return Err(self.unmodeled());
                }
                let v = self.xsrc(ins, 1, 16, true)?;
                let (n, w) = if mn == M::Movmskps { (4, 32) } else { (2, 64) };
                let m = (0..n).fold(0u64, |m, i| m | ((((v >> (i * w + w - 1)) & 1) as u64) << i));
                self.cpu.set_reg(r, m);
                Ok(())
            }
            M::Addss => self.fp_scalar::<F32>(ins, Op::Add),
            M::Subss => self.fp_scalar::<F32>(ins, Op::Sub),
            M::Mulss => self.fp_scalar::<F32>(ins, Op::Mul),
            M::Divss => self.fp_scalar::<F32>(ins, Op::Div),
            M::Minss => self.fp_scalar::<F32>(ins, Op::Min),
            M::Maxss => self.fp_scalar::<F32>(ins, Op::Max),
            M::Sqrtss => self.fp_scalar::<F32>(ins, Op::Sqrt),
            M::Addsd => self.fp_scalar::<F64>(ins, Op::Add),
            M::Subsd => self.fp_scalar::<F64>(ins, Op::Sub),
            M::Mulsd => self.fp_scalar::<F64>(ins, Op::Mul),
            M::Divsd => self.fp_scalar::<F64>(ins, Op::Div),
            M::Minsd => self.fp_scalar::<F64>(ins, Op::Min),
            M::Maxsd => self.fp_scalar::<F64>(ins, Op::Max),
            M::Sqrtsd => self.fp_scalar::<F64>(ins, Op::Sqrt),
            M::Addps => self.fp_packed::<F32>(ins, Op::Add),
            M::Subps => self.fp_packed::<F32>(ins, Op::Sub),
            M::Mulps => self.fp_packed::<F32>(ins, Op::Mul),
            M::Divps => self.fp_packed::<F32>(ins, Op::Div),
            M::Addpd => self.fp_packed::<F64>(ins, Op::Add),
            M::Subpd => self.fp_packed::<F64>(ins, Op::Sub),
            M::Mulpd => self.fp_packed::<F64>(ins, Op::Mul),
            M::Divpd => self.fp_packed::<F64>(ins, Op::Div),
            M::Cmpss => self.fp_cmp::<F32>(ins, true),
            M::Cmpsd => self.fp_cmp::<F64>(ins, true),
            M::Cmpps => self.fp_cmp::<F32>(ins, false),
            M::Cmppd => self.fp_cmp::<F64>(ins, false),
            M::Comiss | M::Ucomiss => self.comis::<F32>(ins),
            M::Comisd | M::Ucomisd => self.comis::<F64>(ins),
            M::Cvtss2sd | M::Cvtsd2ss => {
                self.fp_env()?;
                let ix = self.xdst_idx(ins)?;
                if mn == M::Cvtss2sd {
                    let b = self.xsrc(ins, 1, 4, false)? as u32;
                    self.set_low(ix, 8, u128::from(widen(b)));
                } else {
                    let b = self.xsrc(ins, 1, 8, false)? as u64;
                    self.set_low(ix, 4, u128::from(narrow(b)));
                }
                Ok(())
            }
            M::Cvtsi2ss => self.cvt_from_int(ins, false),
            M::Cvtsi2sd => self.cvt_from_int(ins, true),
            M::Cvttss2si => self.cvt_to_int(ins, false, false),
            M::Cvttsd2si => self.cvt_to_int(ins, true, false),
            M::Cvtss2si => self.cvt_to_int(ins, false, true),
            M::Cvtsd2si => self.cvt_to_int(ins, true, true),
            M::Shufps | M::Shufpd => {
                let ix = self.xdst_idx(ins)?;
                if ins.op_kind(2) != OpKind::Immediate8 {
                    return Err(self.unmodeled());
                }
                let imm = ins.immediate(2) as u32;
                let s = self.xsrc(ins, 1, 16, true)?;
                let d = self.cpu.xmm[ix];
                let pick = |v: u128, sel: u32, w: u32| (v >> (sel * w)) & ((1u128 << w) - 1);
                self.cpu.xmm[ix] = if mn == M::Shufps {
                    pick(d, imm & 3, 32)
                        | pick(d, (imm >> 2) & 3, 32) << 32
                        | pick(s, (imm >> 4) & 3, 32) << 64
                        | pick(s, (imm >> 6) & 3, 32) << 96
                } else {
                    pick(d, imm & 1, 64) | pick(s, (imm >> 1) & 1, 64) << 64
                };
                Ok(())
            }
            M::Unpcklps => self.bin(ins, true, |a, b| unpack(a, b, 32, false)),
            M::Unpckhps => self.bin(ins, true, |a, b| unpack(a, b, 32, true)),
            M::Unpcklpd => self.bin(ins, true, |a, b| unpack(a, b, 64, false)),
            M::Unpckhpd => self.bin(ins, true, |a, b| unpack(a, b, 64, true)),
            M::Pshuflw | M::Pshufhw => {
                let ix = self.xdst_idx(ins)?;
                if ins.op_kind(2) != OpKind::Immediate8 {
                    return Err(self.unmodeled());
                }
                let imm = ins.immediate(2) as u32;
                let s = self.xsrc(ins, 1, 16, true)?;
                let sh = if mn == M::Pshufhw { 64 } else { 0 };
                let half = s >> sh;
                let mut out = 0u128;
                for i in 0..4 {
                    out |= ((half >> (((imm >> (2 * i)) & 3) * 16)) & 0xffff) << (i * 16);
                }
                let keep = s & !(u128::from(u64::MAX) << sh);
                self.cpu.xmm[ix] = keep | (out << sh);
                Ok(())
            }
            M::Pmuludq => self.bin(ins, true, |a, b| lanes(a, b, 64, |x, y| (x & 0xffff_ffff) * (y & 0xffff_ffff))),
            M::Pmullw => self.bin(ins, true, |a, b| lanes(a, b, 16, u64::wrapping_mul)),
            M::Packuswb => self.bin(ins, true, |a, b| {
                let mut out = 0u128;
                for i in 0..8 {
                    out |= u128::from(sat_u8((a >> (i * 16)) as u64)) << (i * 8);
                    out |= u128::from(sat_u8((b >> (i * 16)) as u64)) << (64 + i * 8);
                }
                out
            }),
            M::Pinsrw => {
                let ix = self.xdst_idx(ins)?;
                if ins.op_kind(2) != OpKind::Immediate8 {
                    return Err(self.unmodeled());
                }
                let v = match ins.op_kind(1) {
                    OpKind::Register if ins.op1_register().is_gpr32() => self.xsrc(ins, 1, 4, false)? & 0xffff,
                    OpKind::Memory => self.xsrc(ins, 1, 2, false)? & 0xffff,
                    _ => return Err(self.unmodeled()),
                };
                let sh = (ins.immediate(2) as u32 & 7) * 16;
                self.cpu.xmm[ix] = (self.cpu.xmm[ix] & !(0xffffu128 << sh)) | (v << sh);
                Ok(())
            }
            M::Pextrw => {
                let r = ins.op0_register();
                if ins.op_kind(0) != OpKind::Register || !r.is_gpr32() || ins.op_kind(1) != OpKind::Register {
                    return Err(self.unmodeled());
                }
                let v = self.xsrc(ins, 1, 16, false)?;
                let sh = (ins.immediate(2) as u32 & 7) * 16;
                self.cpu.set_reg(r, ((v >> sh) & 0xffff) as u64);
                Ok(())
            }
            _ => Err(self.unmodeled()),
        }
    }
}
