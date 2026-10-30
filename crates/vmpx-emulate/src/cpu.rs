use iced_x86::Register;

pub const CF: u64 = 1;
pub const PF: u64 = 1 << 2;
pub const AF: u64 = 1 << 4;
pub const ZF: u64 = 1 << 6;
pub const SF: u64 = 1 << 7;
pub const OF: u64 = 1 << 11;

pub const RSP: usize = 4;

#[must_use]
pub fn mask(bytes: u32) -> u64 {
    if bytes >= 8 { u64::MAX } else { (1u64 << (bytes * 8)) - 1 }
}

/// x64 user-mode register state. `undef_flags` marks rflags bits whose value
/// the architecture leaves undefined after the last flag-writing instruction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Cpu {
    pub regs: [u64; 16],
    pub rip: u64,
    pub rflags: u64,
    pub undef_flags: u64,
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

impl Cpu {
    #[must_use]
    pub fn new(rip: u64) -> Self {
        Self { regs: [0; 16], rip, rflags: 2, undef_flags: 0 }
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
