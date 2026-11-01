use iced_x86::Register;

pub const CF: u64 = 1;
pub const PF: u64 = 1 << 2;
pub const AF: u64 = 1 << 4;
pub const ZF: u64 = 1 << 6;
pub const SF: u64 = 1 << 7;
pub const OF: u64 = 1 << 11;

pub const RSP: usize = 4;

/// Power-on MXCSR: all exceptions masked, round-to-nearest.
pub const MXCSR_DEFAULT: u32 = 0x1f80;

#[must_use]
pub fn mask(bytes: u32) -> u64 {
    if bytes >= 8 { u64::MAX } else { (1u64 << (bytes * 8)) - 1 }
}

/// x64 user-mode register state. `undef_flags` marks rflags bits whose value
/// the architecture leaves undefined after the last flag-writing instruction.
/// `gs_base` is the linear base added to `gs:`-relative memory operands (the
/// only segment base modeled; see [`crate::Emulator`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Cpu {
    pub regs: [u64; 16],
    pub rip: u64,
    pub rflags: u64,
    pub undef_flags: u64,
    pub gs_base: u64,
    /// Low 128 bits of xmm0-15 (see `sse.rs` for the modeled subset).
    pub xmm: [u128; 16],
    pub mxcsr: u32,
}

impl Cpu {
    /// Serialize full CPU state for a snapshot.
    pub fn save(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        for &r in &self.regs {
            w.write_all(&r.to_le_bytes())?;
        }
        for v in [self.rip, self.rflags, self.undef_flags, self.gs_base] {
            w.write_all(&v.to_le_bytes())?;
        }
        for &x in &self.xmm {
            w.write_all(&x.to_le_bytes())?;
        }
        w.write_all(&self.mxcsr.to_le_bytes())
    }

    /// Reconstruct CPU state written by [`Cpu::save`].
    pub fn load(r: &mut impl std::io::Read) -> std::io::Result<Cpu> {
        let mut c = Cpu {
            regs: [0; 16],
            rip: 0,
            rflags: 0,
            undef_flags: 0,
            gs_base: 0,
            xmm: [0; 16],
            mxcsr: 0,
        };
        let mut b8 = [0u8; 8];
        for r8 in &mut c.regs {
            r.read_exact(&mut b8)?;
            *r8 = u64::from_le_bytes(b8);
        }
        for v in [&mut c.rip, &mut c.rflags, &mut c.undef_flags, &mut c.gs_base] {
            r.read_exact(&mut b8)?;
            *v = u64::from_le_bytes(b8);
        }
        let mut b16 = [0u8; 16];
        for x in &mut c.xmm {
            r.read_exact(&mut b16)?;
            *x = u128::from_le_bytes(b16);
        }
        let mut b4 = [0u8; 4];
        r.read_exact(&mut b4)?;
        c.mxcsr = u32::from_le_bytes(b4);
        Ok(c)
    }
}

enum Loc {
    Full(usize, u32),
    High8(usize),
}

fn locate(r: Register) -> Option<Loc> {
    match r {
        Register::AH => Some(Loc::High8(0)),
        Register::CH => Some(Loc::High8(1)),
        Register::DH => Some(Loc::High8(2)),
        Register::BH => Some(Loc::High8(3)),
        _ if r.is_gpr64() => Some(Loc::Full(r.number(), 8)),
        _ if r.is_gpr32() => Some(Loc::Full(r.number(), 4)),
        _ if r.is_gpr16() => Some(Loc::Full(r.number(), 2)),
        _ if r.is_gpr8() => Some(Loc::Full(r.full_register().number(), 1)),
        _ => None,
    }
}

/// `(regs index, byte width, bit shift)` of a GPR operand, or `None` for a
/// non-GPR. Pre-resolution twin of [`Cpu::reg`]/[`Cpu::set_reg`]'s addressing.
pub(crate) fn locate_reg(r: Register) -> Option<(u8, u8, u8)> {
    Some(match locate(r)? {
        Loc::Full(i, b) => (i as u8, b as u8, 0),
        Loc::High8(i) => (i as u8, 1, 8),
    })
}

impl Cpu {
    #[must_use]
    pub fn new(rip: u64) -> Self {
        Self { regs: [0; 16], rip, rflags: 2, undef_flags: 0, gs_base: 0, xmm: [0; 16], mxcsr: MXCSR_DEFAULT }
    }

    /// Read a general-purpose register; `None` for non-GPRs.
    #[must_use]
    pub fn reg(&self, r: Register) -> Option<u64> {
        match locate(r)? {
            Loc::Full(i, b) => Some(self.regs[i] & mask(b)),
            Loc::High8(i) => Some((self.regs[i] >> 8) & 0xff),
        }
    }

    /// Write a general-purpose register with x64 partial-write rules (32-bit
    /// writes zero-extend; 8/16-bit writes preserve the rest). `false` for non-GPRs.
    pub fn set_reg(&mut self, r: Register, v: u64) -> bool {
        match locate(r) {
            Some(Loc::Full(i, 8)) => self.regs[i] = v,
            Some(Loc::Full(i, 4)) => self.regs[i] = v & 0xffff_ffff,
            Some(Loc::Full(i, b)) => {
                let m = mask(b);
                self.regs[i] = (self.regs[i] & !m) | (v & m);
            }
            Some(Loc::High8(i)) => self.regs[i] = (self.regs[i] & !0xff00) | ((v & 0xff) << 8),
            None => return false,
        }
        true
    }

    #[must_use]
    pub fn flag(&self, f: u64) -> bool {
        self.rflags & f != 0
    }
}
