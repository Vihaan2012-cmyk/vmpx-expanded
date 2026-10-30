//! Pure ALU/shift semantics with exact flag results.

use crate::cpu::{AF, CF, OF, PF, SF, ZF, mask};

const ALL: u64 = CF | PF | AF | ZF | SF | OF;

/// Flag update: bits in `def` take `val`; bits in `undef` become undefined;
/// all other flags are preserved.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Flags {
    pub def: u64,
    pub val: u64,
    pub undef: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Alu {
    Add,
    Sub,
    And,
    Or,
    Xor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shift {
    Shl,
    Shr,
    Sar,
}

fn bit(v: u64, n: u32) -> u64 {
    (v >> n) & 1
}

fn szp(res: u64, bits: u32) -> u64 {
    let mut f = 0;
    if res == 0 {
        f |= ZF;
    }
    if bit(res, bits - 1) != 0 {
        f |= SF;
    }
    if (res & 0xff).count_ones().is_multiple_of(2) {
        f |= PF;
    }
    f
}

fn pick(cond: bool, f: u64) -> u64 {
    if cond { f } else { 0 }
}

/// Two-operand ALU op over `bytes` (1/2/4/8) wide operands.
#[must_use]
pub fn arith(op: Alu, a: u64, b: u64, bytes: u32) -> (u64, Flags) {
    let m = mask(bytes);
    let bits = bytes * 8;
    let (a, b) = (a & m, b & m);
    let (res, extra) = match op {
        Alu::Add => {
            let r = a.wrapping_add(b) & m;
            let cf = pick(r < a, CF);
            let of = pick(bit((a ^ r) & (b ^ r), bits - 1) != 0, OF);
            let af = pick((a ^ b ^ r) & 0x10 != 0, AF);
            (r, cf | of | af)
        }
        Alu::Sub => {
            let r = a.wrapping_sub(b) & m;
            let cf = pick(a < b, CF);
            let of = pick(bit((a ^ b) & (a ^ r), bits - 1) != 0, OF);
            let af = pick((a ^ b ^ r) & 0x10 != 0, AF);
            (r, cf | of | af)
        }
        Alu::And | Alu::Or | Alu::Xor => {
            let r = match op {
                Alu::And => a & b,
                Alu::Or => a | b,
                _ => a ^ b,
            };
            return (r, Flags { def: CF | OF | SF | ZF | PF, val: szp(r, bits), undef: AF });
        }
    };
    (res, Flags { def: ALL, val: szp(res, bits) | extra, undef: 0 })
}

/// `inc`/`dec`: like add/sub of 1 but CF is preserved.
#[must_use]
pub fn incdec(a: u64, dec: bool, bytes: u32) -> (u64, Flags) {
    let (r, f) = arith(if dec { Alu::Sub } else { Alu::Add }, a, 1, bytes);
    (r, Flags { def: ALL & !CF, val: f.val & !CF, undef: 0 })
}

#[must_use]
pub fn neg(a: u64, bytes: u32) -> (u64, Flags) {
    arith(Alu::Sub, 0, a, bytes)
}

/// Shift with x64 count masking. `None` when the masked count is zero (no
/// register or flag change).
#[must_use]
pub fn shift(kind: Shift, a: u64, count: u64, bytes: u32) -> Option<(u64, Flags)> {
    let m = mask(bytes);
    let bits = bytes * 8;
    let a = a & m;
    let c = u32::try_from(count & if bytes == 8 { 63 } else { 31 }).ok()?;
    if c == 0 {
        return None;
    }
    let in_range = c <= bits;
    let (res, cf) = match kind {
        Shift::Shl => ((a << c) & m, in_range.then(|| bit(a, bits - c))),
        Shift::Shr => (a >> c, in_range.then(|| bit(a, c - 1))),
        Shift::Sar => {
            let sa = ((a << (64 - bits)) as i64) >> (64 - bits);
            (((sa >> c.min(63)) as u64) & m, in_range.then(|| bit(sa as u64, c - 1)))
        }
    };
    let of = (c == 1).then(|| match kind {
        Shift::Shl => bit(res, bits - 1) ^ cf.unwrap_or(0),
        Shift::Shr => bit(a, bits - 1),
        Shift::Sar => 0,
    });
    let mut def = SF | ZF | PF;
    let mut undef = AF;
    let mut val = szp(res, bits);
    match cf {
        Some(v) => {
            def |= CF;
            val |= pick(v != 0, CF);
        }
        None => undef |= CF,
    }
    match of {
        Some(v) => {
            def |= OF;
            val |= pick(v != 0, OF);
        }
        None => undef |= OF,
    }
    Some((res, Flags { def, val, undef }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_carry_overflow() {
        let (r, f) = arith(Alu::Add, 0xff, 1, 1);
        assert_eq!(r, 0);
        assert_eq!(f.val, CF | ZF | PF | AF);
        let (r, f) = arith(Alu::Add, 0x7f, 1, 1);
        assert_eq!(r, 0x80);
        assert_eq!(f.val & (OF | SF | CF), OF | SF);
    }

    #[test]
    fn sub_borrow() {
        let (r, f) = arith(Alu::Sub, 0, 1, 8);
        assert_eq!(r, u64::MAX);
        assert_eq!(f.val & (CF | SF | ZF | OF), CF | SF);
        let (_, f) = arith(Alu::Sub, 0x8000_0000, 1, 4);
        assert_eq!(f.val & OF, OF);
    }

    #[test]
    fn shifts() {
        let (r, f) = shift(Shift::Shl, 0x8000_0001, 1, 4).unwrap();
        assert_eq!(r, 2);
        assert_eq!(f.val & (CF | OF), CF | OF);
        assert!(shift(Shift::Shl, 1, 32, 4).is_none());
        let (r, f) = shift(Shift::Sar, 0x80, 1, 1).unwrap();
        assert_eq!(r, 0xc0);
        assert_eq!(f.undef, AF);
        let (_, f) = shift(Shift::Shr, 5, 3, 4).unwrap();
        assert_eq!(f.undef, AF | OF);
    }
}
