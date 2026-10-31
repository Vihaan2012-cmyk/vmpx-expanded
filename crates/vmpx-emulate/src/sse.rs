//! SSE/SSE2 integer + move subset. Child of `emu` to reach its private state.
//!
//! Modeled: `movdqu/movdqa/movups/movaps/movupd/movapd`, `movq`, `movd`,
//! `pxor/pand/pandn/por` (+ `xorps/andps/andnps/orps` and `pd` twins),
//! `padd{b,w,d,q}`, `psub{b,w,d,q}`, `pcmpeq{b,w,d}`, `pcmpgt{b,w,d}`,
//! `punpck{l,h}{bw,wd,dq,qdq}`, `pshufd`, `psrl/psll{w,d,q}`, `psra{w,d}`,
//! `psrldq/pslldq`, `pmovmskb`, `ldmxcsr/stmxcsr`.
//!
//! Anything else touching a vector register (VEX/EVEX, MMX, scalar float, ...)
//! stops with [`Stop::UnmodeledSse`].

#[allow(clippy::wildcard_imports)]
use super::*;

/// `true` if `ins` reads or writes a vector/MMX register, or is an MXCSR op.
pub(super) fn is_vector(ins: &Instruction) -> bool {
    use Mnemonic as M;
    if matches!(ins.mnemonic(), M::Ldmxcsr | M::Stmxcsr) {
        return true;
    }
    (0..ins.op_count()).any(|i| {
        ins.op_kind(i) == OpKind::Register && {
            let r = ins.op_register(i);
            r.is_xmm() || r.is_ymm() || r.is_zmm() || r.is_mm()
        }
    })
}

/// Apply `f` to each `bits`-wide lane pair.
fn lanes(a: u128, b: u128, bits: u32, f: impl Fn(u64, u64) -> u64) -> u128 {
    let m = if bits == 64 { u64::MAX } else { (1u64 << bits) - 1 };
    let mut out = 0u128;
    for i in 0..128 / bits {
        let s = i * bits;
        let (x, y) = ((a >> s) as u64 & m, (b >> s) as u64 & m);
        out |= u128::from(f(x, y) & m) << s;
    }
    out
}

fn sext(v: u64, bits: u32) -> i64 {
    let s = 64 - bits;
    ((v << s) as i64) >> s
}

/// Interleave the low (`hi == false`) or high 64-bit halves in `bits` lanes.
fn unpack(a: u128, b: u128, bits: u32, hi: bool) -> u128 {
    let m = if bits == 64 { u64::MAX } else { (1u64 << bits) - 1 };
    let base = if hi { 64 } else { 0 };
    let mut out = 0u128;
    for i in 0..64 / bits {
        let s = base + i * bits;
        let (x, y) = ((a >> s) as u64 & m, (b >> s) as u64 & m);
        out |= u128::from(x) << (2 * i * bits);
        out |= u128::from(y) << ((2 * i + 1) * bits);
    }
    out
}

#[derive(Clone, Copy)]
enum Sh {
    Srl,
    Sll,
    Sra,
}

fn shift(v: u128, bits: u32, count: u64, kind: Sh) -> u128 {
    lanes(v, 0, bits, |x, _| match kind {
        Sh::Srl => if count >= u64::from(bits) { 0 } else { x >> count },
        Sh::Sll => if count >= u64::from(bits) { 0 } else { x << count },
        Sh::Sra => {
            let c = count.min(u64::from(bits) - 1);
            (sext(x, bits) >> c) as u64
        }
    })
}

impl Emulator {
    fn unmodeled(&self) -> Stop {
        Stop::UnmodeledSse { rip: self.cur.0, mnemonic: self.cur.1 }
    }

    fn xmm_idx(&self, r: Register) -> Result<usize, Stop> {
        if r.is_xmm() && r.number() < 16 { Ok(r.number()) } else { Err(self.unmodeled()) }
    }

    fn load128(&mut self, addr: u64, aligned: bool) -> Result<u128, Stop> {
        if aligned && addr & 15 != 0 {
            return Err(*self.fault(addr, Access::Read));
        }
        if self.gate.wants_read(addr) {
            self.mem_hook(MemEvent { addr, size: 16, access: Access::Read, value: None })?;
        }
        let mut b = [0u8; 16];
        self.mem.read(addr, &mut b).map_err(|f| self.fault(f.addr, f.access))?;
        let v = u128::from_le_bytes(b);
        if self.observe.detail {
            self.det = (Some(addr), Some(v as u64));
        }
        Ok(v)
    }

    fn store128(&mut self, addr: u64, v: u128, aligned: bool) -> Result<(), Stop> {
        if aligned && addr & 15 != 0 {
            return Err(*self.fault(addr, Access::Write));
        }
        if self.gate.wants_write(addr, 16) {
            self.mem_hook(MemEvent { addr, size: 16, access: Access::Write, value: Some(v as u64) })?;
        }
        self.mem.write(addr, &v.to_le_bytes()).map_err(|f| self.fault(f.addr, f.access))?;
        if self.observe.detail {
            self.det = (Some(addr), Some(v as u64));
        }
        Ok(())
    }

    /// Read operand `i` as a vector value of `bytes` (4/8/16) zero-extended to 128 bits.
    fn xsrc(&mut self, ins: &Instruction, i: u32, bytes: u32, aligned: bool) -> Result<u128, Stop> {
        match ins.op_kind(i) {
            OpKind::Register => {
                let r = ins.op_register(i);
                if r.is_xmm() {
                    let v = self.cpu.xmm[self.xmm_idx(r)?];
                    Ok(if bytes == 16 { v } else { v & ((1u128 << (bytes * 8)) - 1) })
                } else if r.is_gpr32() || r.is_gpr64() {
                    Ok(u128::from(self.cpu.reg(r).ok_or_else(|| self.unmodeled())?))
                } else {
                    Err(self.unmodeled())
                }
            }
            OpKind::Memory => {
                let a = self.ea(ins)?;
                if bytes == 16 { self.load128(a, aligned) } else { Ok(u128::from(self.load(a, bytes)?)) }
            }
            _ => Err(self.unmodeled()),
        }
    }

    /// Write `v` to operand `i`. A register destination takes `v` whole (callers
    /// zero-extend); a memory destination stores the low `bytes`.
    fn xdst(&mut self, ins: &Instruction, i: u32, bytes: u32, aligned: bool, v: u128) -> Result<(), Stop> {
        match ins.op_kind(i) {
            OpKind::Register => {
                let r = ins.op_register(i);
                if r.is_xmm() {
                    let ix = self.xmm_idx(r)?;
                    self.cpu.xmm[ix] = v;
                } else if r.is_gpr32() || r.is_gpr64() {
                    self.cpu.set_reg(r, v as u64);
                } else {
                    return Err(self.unmodeled());
                }
                Ok(())
            }
            OpKind::Memory => {
                let a = self.ea(ins)?;
                if bytes == 16 { self.store128(a, v, aligned) } else { Ok(self.store(a, bytes, v as u64)?) }
            }
            _ => Err(self.unmodeled()),
        }
    }

    pub(super) fn exec_sse(&mut self, ins: &Instruction, mn: Mnemonic) -> Result<(), Stop> {
        use Mnemonic as M;
        // Memory forms of the legacy-SSE ops below, except the explicitly
        // unaligned moves, require 16-byte alignment (#GP otherwise).
        let align = !matches!(mn, M::Movdqu | M::Movups | M::Movupd);
        match mn {
            M::Movdqu | M::Movdqa | M::Movups | M::Movaps | M::Movupd | M::Movapd => {
                let v = self.xsrc(ins, 1, 16, align)?;
                self.xdst(ins, 0, 16, align, v)
            }
            M::Movq | M::Movd => {
                let n = if mn == M::Movq { 8 } else { 4 };
                // GPR/memory/xmm source is zero-extended into an xmm destination;
                // an xmm source stores only its low `n` bytes.
                let v = self.xsrc(ins, 1, n, false)?;
                let v = if n == 4 { v & 0xffff_ffff } else { v & u128::from(u64::MAX) };
                self.xdst(ins, 0, n, false, v)
            }
            M::Pxor | M::Xorps | M::Xorpd => self.bin(ins, align, |a, b| a ^ b),
            M::Pand | M::Andps | M::Andpd => self.bin(ins, align, |a, b| a & b),
            M::Pandn | M::Andnps | M::Andnpd => self.bin(ins, align, |a, b| !a & b),
            M::Por | M::Orps | M::Orpd => self.bin(ins, align, |a, b| a | b),
            M::Paddb => self.bin(ins, align, |a, b| lanes(a, b, 8, u64::wrapping_add)),
            M::Paddw => self.bin(ins, align, |a, b| lanes(a, b, 16, u64::wrapping_add)),
            M::Paddd => self.bin(ins, align, |a, b| lanes(a, b, 32, u64::wrapping_add)),
            M::Paddq => self.bin(ins, align, |a, b| lanes(a, b, 64, u64::wrapping_add)),
            M::Psubb => self.bin(ins, align, |a, b| lanes(a, b, 8, u64::wrapping_sub)),
            M::Psubw => self.bin(ins, align, |a, b| lanes(a, b, 16, u64::wrapping_sub)),
            M::Psubd => self.bin(ins, align, |a, b| lanes(a, b, 32, u64::wrapping_sub)),
            M::Psubq => self.bin(ins, align, |a, b| lanes(a, b, 64, u64::wrapping_sub)),
            M::Pcmpeqb => self.bin(ins, align, |a, b| lanes(a, b, 8, |x, y| if x == y { !0 } else { 0 })),
            M::Pcmpeqw => self.bin(ins, align, |a, b| lanes(a, b, 16, |x, y| if x == y { !0 } else { 0 })),
            M::Pcmpeqd => self.bin(ins, align, |a, b| lanes(a, b, 32, |x, y| if x == y { !0 } else { 0 })),
            M::Pcmpgtb => self.bin(ins, align, |a, b| lanes(a, b, 8, |x, y| if sext(x, 8) > sext(y, 8) { !0 } else { 0 })),
            M::Pcmpgtw => self.bin(ins, align, |a, b| lanes(a, b, 16, |x, y| if sext(x, 16) > sext(y, 16) { !0 } else { 0 })),
            M::Pcmpgtd => self.bin(ins, align, |a, b| lanes(a, b, 32, |x, y| if sext(x, 32) > sext(y, 32) { !0 } else { 0 })),
            M::Punpcklbw => self.bin(ins, align, |a, b| unpack(a, b, 8, false)),
            M::Punpcklwd => self.bin(ins, align, |a, b| unpack(a, b, 16, false)),
            M::Punpckldq => self.bin(ins, align, |a, b| unpack(a, b, 32, false)),
            M::Punpcklqdq => self.bin(ins, align, |a, b| unpack(a, b, 64, false)),
            M::Punpckhbw => self.bin(ins, align, |a, b| unpack(a, b, 8, true)),
            M::Punpckhwd => self.bin(ins, align, |a, b| unpack(a, b, 16, true)),
            M::Punpckhdq => self.bin(ins, align, |a, b| unpack(a, b, 32, true)),
            M::Punpckhqdq => self.bin(ins, align, |a, b| unpack(a, b, 64, true)),
            M::Pshufd => {
                let src = self.xsrc(ins, 1, 16, align)?;
                let imm = ins.immediate(2) as u32;
                let mut out = 0u128;
                for i in 0..4 {
                    let sel = (imm >> (2 * i)) & 3;
                    out |= ((src >> (sel * 32)) & 0xffff_ffff) << (i * 32);
                }
                self.xdst(ins, 0, 16, true, out)
            }
            M::Psrlw => self.shf(ins, align, 16, Sh::Srl),
            M::Psrld => self.shf(ins, align, 32, Sh::Srl),
            M::Psrlq => self.shf(ins, align, 64, Sh::Srl),
            M::Psllw => self.shf(ins, align, 16, Sh::Sll),
            M::Pslld => self.shf(ins, align, 32, Sh::Sll),
            M::Psllq => self.shf(ins, align, 64, Sh::Sll),
            M::Psraw => self.shf(ins, align, 16, Sh::Sra),
            M::Psrad => self.shf(ins, align, 32, Sh::Sra),
            M::Psrldq | M::Pslldq => {
                if ins.op_kind(1) != OpKind::Immediate8 {
                    return Err(self.unmodeled());
                }
                let n = ins.immediate(1);
                let v = self.xsrc(ins, 0, 16, true)?;
                let out = match (n > 15, mn) {
                    (true, _) => 0,
                    (_, M::Psrldq) => v >> (n * 8),
                    _ => v << (n * 8),
                };
                self.xdst(ins, 0, 16, true, out)
            }
            M::Pmovmskb => {
                if !ins.op_register(0).is_gpr32() && !ins.op_register(0).is_gpr64() {
                    return Err(self.unmodeled());
                }
                let v = self.xsrc(ins, 1, 16, true)?;
                let mut m = 0u64;
                for i in 0..16 {
                    m |= (((v >> (i * 8 + 7)) & 1) as u64) << i;
                }
                self.xdst(ins, 0, 4, true, u128::from(m))
            }
            M::Ldmxcsr => {
                let a = self.ea(ins)?;
                let v = self.load(a, 4)? as u32;
                // Reserved bits set raise #GP on hardware; do not guess.
                if v & 0xffff_0000 != 0 {
                    return Err(self.unmodeled());
                }
                self.cpu.mxcsr = v;
                Ok(())
            }
            M::Stmxcsr => {
                let a = self.ea(ins)?;
                Ok(self.store(a, 4, u64::from(self.cpu.mxcsr))?)
            }
            _ => Err(self.unmodeled()),
        }
    }

    /// `dst = f(dst, src)` for a 128-bit xmm destination and xmm/m128 source.
    fn bin(&mut self, ins: &Instruction, align: bool, f: impl Fn(u128, u128) -> u128) -> Result<(), Stop> {
        if ins.op_count() != 2 || ins.op_kind(0) != OpKind::Register {
            return Err(self.unmodeled());
        }
        let ix = self.xmm_idx(ins.op0_register())?;
        let b = self.xsrc(ins, 1, 16, align)?;
        self.cpu.xmm[ix] = f(self.cpu.xmm[ix], b);
        Ok(())
    }

    /// Shift by an imm8 or by the low 64 bits of an xmm/m128 count.
    fn shf(&mut self, ins: &Instruction, align: bool, bits: u32, kind: Sh) -> Result<(), Stop> {
        if ins.op_kind(0) != OpKind::Register {
            return Err(self.unmodeled());
        }
        let ix = self.xmm_idx(ins.op0_register())?;
        let count = if ins.op_kind(1) == OpKind::Immediate8 {
            ins.immediate(1)
        } else {
            self.xsrc(ins, 1, 16, align)? as u64
        };
        self.cpu.xmm[ix] = shift(self.cpu.xmm[ix], bits, count, kind);
        Ok(())
    }
}
