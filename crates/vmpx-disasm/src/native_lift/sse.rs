//! SSE/SSE2 scalar floating-point slice of the native lifter.
//!
//! XMM registers are `(lo, hi)` 64-bit lanes over the inputs `xmm<n>_lo_in` /
//! `xmm<n>_hi_in`. Floats are their bit patterns; arithmetic, conversions and
//! compares are the reserved `fp_*` intrinsics (see `vmpx_vm::fp`), whose
//! semantics mirror `vmpx_emulate::sse_fp`: default MXCSR (round-to-nearest-even,
//! exceptions masked, no FTZ/DAZ), first-NaN propagation, integer indefinite.
//!
//! Modeled: `movsd movss movq movd movaps movups movapd movupd movdqa movdqu
//! movlps movlpd movhps movhpd movhlps movlhps unpcklpd unpckhpd shufpd`,
//! `{add,sub,mul,div,min,max,sqrt}{ss,sd}`, `cmp{ss,sd}` (predicates 0-7),
//! `(u)comis{s,d}`, `cvtsi2{ss,sd}`, `cvtss2sd cvtsd2ss`, `cvtt{ss,sd}2si`, and the
//! 128-bit bitwise `xorps/pd andps/pd andnps/pd orps/pd pxor pand pandn por`
//! (the fneg/fabs sign-mask idioms). Everything else (packed arithmetic, 32-bit lane
//! shuffles, rounding `cvt{ss,sd}2si`, `ldmxcsr`, AVX) fails closed.
//!
//! Alignment faults of `movaps`-class memory operands are not modeled, exactly as
//! the integer lifter does not model other fault paths.

use super::{
    Algebra, BinOp, Flag, Instruction, Lifter, Mnemonic, OpKind, R, Register, Resize, Width,
    unsupported,
};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Fmt {
    F32,
    F64,
}

impl Fmt {
    const fn width(self) -> Width {
        match self {
            Self::F32 => Width::W32,
            Self::F64 => Width::W64,
        }
    }

    const fn suffix(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F64 => "f64",
        }
    }
}

fn is_xmm_reg(inst: &Instruction, index: u32) -> bool {
    inst.op_kind(index) == OpKind::Register && inst.op_register(index).is_xmm()
}

/// Whether `mnemonic` (on this operand form) belongs to the SSE slice. `movsd`/`cmpsd`
/// share their mnemonic with the string instructions, which have no XMM operand.
pub(super) fn handles(inst: &Instruction, mnemonic: Mnemonic) -> bool {
    use Mnemonic as M;
    match mnemonic {
        M::Movsd | M::Cmpsd => (0..inst.op_count()).any(|i| is_xmm_reg(inst, i)),
        M::Movss
        | M::Movq
        | M::Movd
        | M::Movaps
        | M::Movups
        | M::Movapd
        | M::Movupd
        | M::Movdqa
        | M::Movdqu
        | M::Movlps
        | M::Movlpd
        | M::Movhps
        | M::Movhpd
        | M::Movhlps
        | M::Movlhps
        | M::Unpcklpd
        | M::Unpckhpd
        | M::Shufpd
        | M::Addss
        | M::Subss
        | M::Mulss
        | M::Divss
        | M::Minss
        | M::Maxss
        | M::Sqrtss
        | M::Addsd
        | M::Subsd
        | M::Mulsd
        | M::Divsd
        | M::Minsd
        | M::Maxsd
        | M::Sqrtsd
        | M::Cmpss
        | M::Comiss
        | M::Ucomiss
        | M::Comisd
        | M::Ucomisd
        | M::Cvtsi2ss
        | M::Cvtsi2sd
        | M::Cvtss2sd
        | M::Cvtsd2ss
        | M::Cvttss2si
        | M::Cvttsd2si
        | M::Cvtss2si
        | M::Cvtsd2si
        | M::Xorps
        | M::Xorpd
        | M::Pxor
        | M::Andps
        | M::Andpd
        | M::Pand
        | M::Andnps
        | M::Andnpd
        | M::Pandn
        | M::Orps
        | M::Orpd
        | M::Por => true,
        _ => false,
    }
}

impl<A: Algebra> Lifter<'_, A> {
    fn xmm_idx(reg: Register) -> R<usize> {
        if reg.is_xmm() {
            Ok(reg.number())
        } else {
            unsupported(format!("register {reg:?} is not an XMM register"))
        }
    }

    fn dst_xmm(inst: &Instruction) -> R<usize> {
        if inst.op_kind(0) != OpKind::Register {
            return unsupported("destination is not an XMM register");
        }
        Self::xmm_idx(inst.op_register(0))
    }

    fn set_xmm(&mut self, idx: usize, lo: A::Expr, hi: A::Expr) {
        self.xmm[idx] = (lo, hi);
        self.xmm_written.insert(u8::try_from(idx).unwrap_or(u8::MAX));
    }

    /// Replaces the low `fmt` bits of the low lane, preserving everything above.
    fn set_scalar(&mut self, idx: usize, fmt: Fmt, value: A::Expr) -> R<()> {
        let (lo, hi) = self.xmm[idx].clone();
        let new_lo = match fmt {
            Fmt::F64 => value,
            Fmt::F32 => {
                let kept = self.bin(BinOp::And, lo, self.k(Width::W64, 0xFFFF_FFFF_0000_0000))?;
                let low = self.resize(Resize::ZeroExtend, Width::W64, value)?;
                self.bin(BinOp::Or, kept, low)?
            }
        };
        self.set_xmm(idx, new_lo, hi);
        Ok(())
    }

    fn scalar_of_lane(&self, lo: A::Expr, fmt: Fmt) -> R<A::Expr> {
        self.resize(Resize::Truncate, fmt.width(), lo)
    }

    fn check_mem_size(inst: &Instruction, bytes: u32) -> R<()> {
        let size = inst.memory_size().size();
        if size == bytes as usize {
            Ok(())
        } else {
            unsupported(format!(
                "memory operand is {size} bytes, expected {bytes} for this form"
            ))
        }
    }

    /// Loads `width` from the instruction's memory operand plus `offset` bytes.
    fn load_mem_off(&mut self, inst: &Instruction, width: Width, offset: i64) -> R<A::Expr> {
        let (mut address, mut key) = self.mem_address(inst, true)?;
        if offset != 0 {
            address = match self.alg.as_const(&address) {
                Some(c) => self.k(Width::W64, c.wrapping_add(offset.cast_unsigned())),
                None => self.bin(BinOp::Add, address, self.k(Width::W64, offset.cast_unsigned()))?,
            };
            key.disp = key.disp.wrapping_add(offset);
        }
        self.load_at(address, &key, width)
    }

    fn store_mem_off(&mut self, inst: &Instruction, offset: i64, value: A::Expr) -> R<()> {
        let (mut address, mut key) = self.mem_address(inst, true)?;
        if offset != 0 {
            address = match self.alg.as_const(&address) {
                Some(c) => self.k(Width::W64, c.wrapping_add(offset.cast_unsigned())),
                None => self.bin(BinOp::Add, address, self.k(Width::W64, offset.cast_unsigned()))?,
            };
            key.disp = key.disp.wrapping_add(offset);
        }
        self.store_at(address, key, value);
        Ok(())
    }

    /// A scalar float source: an XMM register's low bits or a memory operand.
    fn sse_scalar_src(&mut self, inst: &Instruction, index: u32, fmt: Fmt) -> R<A::Expr> {
        match inst.op_kind(index) {
            OpKind::Register => {
                let idx = Self::xmm_idx(inst.op_register(index))?;
                let lo = self.xmm[idx].0.clone();
                self.scalar_of_lane(lo, fmt)
            }
            OpKind::Memory => {
                Self::check_mem_size(inst, fmt.width().bytes())?;
                self.load_mem_off(inst, fmt.width(), 0)
            }
            kind => unsupported(format!("SSE source operand kind {kind:?}")),
        }
    }

    /// A full 128-bit source as `(lo, hi)`.
    fn sse_src128(&mut self, inst: &Instruction, index: u32) -> R<(A::Expr, A::Expr)> {
        match inst.op_kind(index) {
            OpKind::Register => {
                let idx = Self::xmm_idx(inst.op_register(index))?;
                Ok(self.xmm[idx].clone())
            }
            OpKind::Memory => {
                Self::check_mem_size(inst, 16)?;
                let lo = self.load_mem_off(inst, Width::W64, 0)?;
                let hi = self.load_mem_off(inst, Width::W64, 8)?;
                Ok((lo, hi))
            }
            kind => unsupported(format!("SSE source operand kind {kind:?}")),
        }
    }

    fn fp(&self, width: Width, name: &str, args: Vec<A::Expr>) -> A::Expr {
        self.alg.intrinsic(width, name, args)
    }

    #[allow(clippy::too_many_lines)]
    pub(super) fn exec_sse(&mut self, inst: &Instruction, mnemonic: Mnemonic) -> R<()> {
        use Mnemonic as M;
        let zero = self.k(Width::W64, 0);
        match mnemonic {
            M::Movaps | M::Movups | M::Movapd | M::Movupd | M::Movdqa | M::Movdqu => {
                Self::expect_ops(inst, 2)?;
                if inst.op_kind(0) == OpKind::Memory {
                    let src = Self::xmm_idx(inst.op_register(1))?;
                    Self::check_mem_size(inst, 16)?;
                    let (lo, hi) = self.xmm[src].clone();
                    self.store_mem_off(inst, 0, lo)?;
                    return self.store_mem_off(inst, 8, hi);
                }
                let dst = Self::dst_xmm(inst)?;
                let (lo, hi) = self.sse_src128(inst, 1)?;
                self.set_xmm(dst, lo, hi);
                Ok(())
            }
            M::Movsd | M::Movss => {
                Self::expect_ops(inst, 2)?;
                let fmt = if mnemonic == M::Movsd { Fmt::F64 } else { Fmt::F32 };
                match (inst.op_kind(0), inst.op_kind(1)) {
                    (OpKind::Register, OpKind::Register) => {
                        let dst = Self::dst_xmm(inst)?;
                        let value = self.sse_scalar_src(inst, 1, fmt)?;
                        self.set_scalar(dst, fmt, value)
                    }
                    (OpKind::Register, OpKind::Memory) => {
                        let dst = Self::dst_xmm(inst)?;
                        let value = self.sse_scalar_src(inst, 1, fmt)?;
                        let lo = self.resize(Resize::ZeroExtend, Width::W64, value)?;
                        self.set_xmm(dst, lo, zero);
                        Ok(())
                    }
                    (OpKind::Memory, OpKind::Register) => {
                        Self::check_mem_size(inst, fmt.width().bytes())?;
                        let src = Self::xmm_idx(inst.op_register(1))?;
                        let lo = self.xmm[src].0.clone();
                        let value = self.scalar_of_lane(lo, fmt)?;
                        self.store_mem_off(inst, 0, value)
                    }
                    _ => unsupported("movs[sd] operand form"),
                }
            }
            M::Movq | M::Movd => {
                Self::expect_ops(inst, 2)?;
                let fmt = if mnemonic == M::Movq { Fmt::F64 } else { Fmt::F32 };
                let width = fmt.width();
                if is_xmm_reg(inst, 0) {
                    let dst = Self::dst_xmm(inst)?;
                    let value = match inst.op_kind(1) {
                        OpKind::Register if inst.op_register(1).is_xmm() => {
                            let src = Self::xmm_idx(inst.op_register(1))?;
                            let lo = self.xmm[src].0.clone();
                            self.scalar_of_lane(lo, fmt)?
                        }
                        OpKind::Register | OpKind::Memory => {
                            if Self::operand_width(inst, 1)? != width {
                                return unsupported("movd/movq source width");
                            }
                            self.read_operand(inst, 1, width)?
                        }
                        kind => return unsupported(format!("movd/movq source kind {kind:?}")),
                    };
                    let lo = self.resize(Resize::ZeroExtend, Width::W64, value)?;
                    self.set_xmm(dst, lo, zero);
                    return Ok(());
                }
                if !is_xmm_reg(inst, 1) {
                    return unsupported("movd/movq without an XMM operand");
                }
                let src = Self::xmm_idx(inst.op_register(1))?;
                let lo = self.xmm[src].0.clone();
                let value = self.scalar_of_lane(lo, fmt)?;
                match inst.op_kind(0) {
                    OpKind::Register | OpKind::Memory => {
                        if Self::operand_width(inst, 0)? != width {
                            return unsupported("movd/movq destination width");
                        }
                        self.write_operand(inst, 0, value)
                    }
                    kind => unsupported(format!("movd/movq destination kind {kind:?}")),
                }
            }
            M::Movlps | M::Movlpd | M::Movhps | M::Movhpd => {
                Self::expect_ops(inst, 2)?;
                let high = matches!(mnemonic, M::Movhps | M::Movhpd);
                match (inst.op_kind(0), inst.op_kind(1)) {
                    (OpKind::Register, OpKind::Memory) => {
                        Self::check_mem_size(inst, 8)?;
                        let dst = Self::dst_xmm(inst)?;
                        let value = self.load_mem_off(inst, Width::W64, 0)?;
                        let (lo, hi) = self.xmm[dst].clone();
                        if high {
                            self.set_xmm(dst, lo, value);
                        } else {
                            self.set_xmm(dst, value, hi);
                        }
                        Ok(())
                    }
                    (OpKind::Memory, OpKind::Register) => {
                        Self::check_mem_size(inst, 8)?;
                        let src = Self::xmm_idx(inst.op_register(1))?;
                        let value = if high {
                            self.xmm[src].1.clone()
                        } else {
                            self.xmm[src].0.clone()
                        };
                        self.store_mem_off(inst, 0, value)
                    }
                    _ => unsupported("movl/movh operand form"),
                }
            }
            M::Movhlps | M::Movlhps => {
                Self::expect_ops(inst, 2)?;
                if !is_xmm_reg(inst, 1) {
                    return unsupported("movhlps/movlhps source");
                }
                let dst = Self::dst_xmm(inst)?;
                let src = Self::xmm_idx(inst.op_register(1))?;
                let (lo, hi) = self.xmm[dst].clone();
                if mnemonic == M::Movhlps {
                    let s = self.xmm[src].1.clone();
                    self.set_xmm(dst, s, hi);
                } else {
                    let s = self.xmm[src].0.clone();
                    self.set_xmm(dst, lo, s);
                }
                Ok(())
            }
            M::Unpcklpd | M::Unpckhpd => {
                Self::expect_ops(inst, 2)?;
                let dst = Self::dst_xmm(inst)?;
                let (lo, hi) = self.xmm[dst].clone();
                let (s_lo, s_hi) = self.sse_src128(inst, 1)?;
                if mnemonic == M::Unpcklpd {
                    self.set_xmm(dst, lo, s_lo);
                } else {
                    self.set_xmm(dst, hi, s_hi);
                }
                Ok(())
            }
            M::Shufpd => {
                Self::expect_ops(inst, 3)?;
                if inst.op_kind(2) != OpKind::Immediate8 {
                    return unsupported("shufpd selector");
                }
                let dst = Self::dst_xmm(inst)?;
                let imm = inst.immediate(2);
                let (d_lo, d_hi) = self.xmm[dst].clone();
                let (s_lo, s_hi) = self.sse_src128(inst, 1)?;
                let lo = if imm & 1 == 0 { d_lo } else { d_hi };
                let hi = if imm & 2 == 0 { s_lo } else { s_hi };
                self.set_xmm(dst, lo, hi);
                Ok(())
            }
            M::Addss
            | M::Subss
            | M::Mulss
            | M::Divss
            | M::Minss
            | M::Maxss
            | M::Addsd
            | M::Subsd
            | M::Mulsd
            | M::Divsd
            | M::Minsd
            | M::Maxsd => {
                Self::expect_ops(inst, 2)?;
                let (fmt, op) = match mnemonic {
                    M::Addss => (Fmt::F32, "add"),
                    M::Subss => (Fmt::F32, "sub"),
                    M::Mulss => (Fmt::F32, "mul"),
                    M::Divss => (Fmt::F32, "div"),
                    M::Minss => (Fmt::F32, "min"),
                    M::Maxss => (Fmt::F32, "max"),
                    M::Addsd => (Fmt::F64, "add"),
                    M::Subsd => (Fmt::F64, "sub"),
                    M::Mulsd => (Fmt::F64, "mul"),
                    M::Divsd => (Fmt::F64, "div"),
                    M::Minsd => (Fmt::F64, "min"),
                    _ => (Fmt::F64, "max"),
                };
                let dst = Self::dst_xmm(inst)?;
                let lo = self.xmm[dst].0.clone();
                let a = self.scalar_of_lane(lo, fmt)?;
                let b = self.sse_scalar_src(inst, 1, fmt)?;
                let name = format!("fp_{op}_{}", fmt.suffix());
                let result = self.fp(fmt.width(), &name, vec![a, b]);
                self.set_scalar(dst, fmt, result)
            }
            M::Sqrtss | M::Sqrtsd => {
                Self::expect_ops(inst, 2)?;
                let fmt = if mnemonic == M::Sqrtss { Fmt::F32 } else { Fmt::F64 };
                let dst = Self::dst_xmm(inst)?;
                let src = self.sse_scalar_src(inst, 1, fmt)?;
                let name = format!("fp_sqrt_{}", fmt.suffix());
                let result = self.fp(fmt.width(), &name, vec![src]);
                self.set_scalar(dst, fmt, result)
            }
            M::Cmpss | M::Cmpsd => {
                Self::expect_ops(inst, 3)?;
                if inst.op_kind(2) != OpKind::Immediate8 {
                    return unsupported("cmps[sd] predicate");
                }
                let fmt = if mnemonic == M::Cmpss { Fmt::F32 } else { Fmt::F64 };
                let pred = inst.immediate(2);
                let dst = Self::dst_xmm(inst)?;
                let lo = self.xmm[dst].0.clone();
                let a = self.scalar_of_lane(lo, fmt)?;
                let b = self.sse_scalar_src(inst, 1, fmt)?;
                let cmp = |this: &Self, p: &str| {
                    this.fp(
                        Width::W8,
                        &format!("fp_cmp_{p}_{}", fmt.suffix()),
                        vec![a.clone(), b.clone()],
                    )
                };
                let one = self.k(Width::W8, 1);
                let bit = match pred {
                    0 => cmp(self, "eq"),
                    1 => cmp(self, "lt"),
                    2 => cmp(self, "le"),
                    3 => cmp(self, "unord"),
                    4 => cmp(self, "ne"),
                    5 => self.bin(BinOp::Xor, cmp(self, "lt"), one)?,
                    6 => self.bin(BinOp::Xor, cmp(self, "le"), one)?,
                    7 => cmp(self, "ord"),
                    _ => return unsupported("cmps[sd] predicate above 7"),
                };
                let wide = self.resize(Resize::ZeroExtend, fmt.width(), bit)?;
                let mask = self.un(super::UnOp::Neg, wide)?;
                self.set_scalar(dst, fmt, mask)
            }
            M::Comiss | M::Ucomiss | M::Comisd | M::Ucomisd => {
                Self::expect_ops(inst, 2)?;
                let fmt = if matches!(mnemonic, M::Comiss | M::Ucomiss) {
                    Fmt::F32
                } else {
                    Fmt::F64
                };
                let dst = Self::dst_xmm(inst)?;
                let lo = self.xmm[dst].0.clone();
                let a = self.scalar_of_lane(lo, fmt)?;
                let b = self.sse_scalar_src(inst, 1, fmt)?;
                let cmp = |this: &Self, p: &str| {
                    this.fp(
                        Width::W8,
                        &format!("fp_cmp_{p}_{}", fmt.suffix()),
                        vec![a.clone(), b.clone()],
                    )
                };
                let unord = cmp(self, "unord");
                let eq = cmp(self, "eq");
                let lt = cmp(self, "lt");
                let zf = self.bin(BinOp::Or, unord.clone(), eq)?;
                let cf = self.bin(BinOp::Or, unord.clone(), lt)?;
                self.set_flag(Flag::Zf, zf);
                self.set_flag(Flag::Pf, unord);
                self.set_flag(Flag::Cf, cf);
                for flag in [Flag::Of, Flag::Sf, Flag::Af] {
                    let off = self.k(Width::W8, 0);
                    self.set_flag(flag, off);
                }
                Ok(())
            }
            M::Cvtsi2ss | M::Cvtsi2sd => {
                Self::expect_ops(inst, 2)?;
                let to = if mnemonic == M::Cvtsi2ss { Fmt::F32 } else { Fmt::F64 };
                let dst = Self::dst_xmm(inst)?;
                let width = Self::operand_width(inst, 1)?;
                if !matches!(width, Width::W32 | Width::W64) {
                    return unsupported("cvtsi2s[sd] integer source width");
                }
                let src = self.read_operand(inst, 1, width)?;
                let name = format!("fp_from_i{}_to_{}", width.bits(), to.suffix());
                let result = self.fp(to.width(), &name, vec![src]);
                self.set_scalar(dst, to, result)
            }
            M::Cvtss2sd | M::Cvtsd2ss => {
                Self::expect_ops(inst, 2)?;
                let (from, to) = if mnemonic == M::Cvtss2sd {
                    (Fmt::F32, Fmt::F64)
                } else {
                    (Fmt::F64, Fmt::F32)
                };
                let dst = Self::dst_xmm(inst)?;
                let src = self.sse_scalar_src(inst, 1, from)?;
                let name = format!("fp_cvt_{}_to_{}", from.suffix(), to.suffix());
                let result = self.fp(to.width(), &name, vec![src]);
                self.set_scalar(dst, to, result)
            }
            M::Cvttss2si | M::Cvttsd2si => {
                Self::expect_ops(inst, 2)?;
                let from = if mnemonic == M::Cvttss2si { Fmt::F32 } else { Fmt::F64 };
                if inst.op_kind(0) != OpKind::Register {
                    return unsupported("cvtt[sd]2si destination");
                }
                let width = Self::operand_width(inst, 0)?;
                if !matches!(width, Width::W32 | Width::W64) {
                    return unsupported("cvtt[sd]2si destination width");
                }
                let src = self.sse_scalar_src(inst, 1, from)?;
                let name = format!("fp_to_i{}_from_{}", width.bits(), from.suffix());
                let result = self.fp(width, &name, vec![src]);
                self.write_operand(inst, 0, result)
            }
            M::Cvtss2si | M::Cvtsd2si => unsupported(
                "round-to-nearest cvt[sd]2si has no fp vocabulary (only the truncating form)",
            ),
            M::Xorps
            | M::Xorpd
            | M::Pxor
            | M::Andps
            | M::Andpd
            | M::Pand
            | M::Andnps
            | M::Andnpd
            | M::Pandn
            | M::Orps
            | M::Orpd
            | M::Por => {
                Self::expect_ops(inst, 2)?;
                let dst = Self::dst_xmm(inst)?;
                let xor = matches!(mnemonic, M::Xorps | M::Xorpd | M::Pxor);
                if xor && is_xmm_reg(inst, 1) && inst.op_register(1) == inst.op_register(0) {
                    let hi = self.k(Width::W64, 0);
                    self.set_xmm(dst, zero, hi);
                    return Ok(());
                }
                let (d_lo, d_hi) = self.xmm[dst].clone();
                let (s_lo, s_hi) = self.sse_src128(inst, 1)?;
                let lane = |this: &mut Self, d: A::Expr, s: A::Expr| -> R<A::Expr> {
                    match mnemonic {
                        M::Xorps | M::Xorpd | M::Pxor => this.bin(BinOp::Xor, d, s),
                        M::Andps | M::Andpd | M::Pand => this.bin(BinOp::And, d, s),
                        M::Orps | M::Orpd | M::Por => this.bin(BinOp::Or, d, s),
                        _ => {
                            let inverted = this.un(super::UnOp::Not, d)?;
                            this.bin(BinOp::And, inverted, s)
                        }
                    }
                };
                let lo = lane(self, d_lo, s_lo)?;
                let hi = lane(self, d_hi, s_hi)?;
                self.set_xmm(dst, lo, hi);
                Ok(())
            }
            other => unsupported(format!("SSE mnemonic {other:?} is not modeled")),
        }
    }
}
