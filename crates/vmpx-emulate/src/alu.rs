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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rotate {
    Rol,
    Ror,
    Rcl,
    Rcr,
}

/// Sign-extend the low `bytes` of `v` to 64 bits.
fn sext(v: u64, bytes: u32) -> u64 {
    if bytes >= 8 {
        v
    } else {
        let sh = 64 - bytes * 8;
        (((v << sh) as i64) >> sh) as u64
    }
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
    // Per Intel SDM, CF is undefined for SHL/SHR when the count reaches the
    // operand width; for SAR the last bit shifted out is always the sign bit,
    // so CF stays defined for every nonzero count.
    let in_range = c <= bits;
    let (res, cf) = match kind {
        Shift::Shl => ((a << c) & m, in_range.then(|| bit(a, bits - c))),
        Shift::Shr => (a >> c, in_range.then(|| bit(a, c - 1))),
        Shift::Sar => {
            let sa = ((a << (64 - bits)) as i64) >> (64 - bits);
            (((sa >> c.min(63)) as u64) & m, Some(bit(sa as u64, (c - 1).min(63))))
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

/// `adc`: `a + b + carry`, exact flags (same model as `Alu::Add`).
#[must_use]
pub fn adc(a: u64, b: u64, carry: bool, bytes: u32) -> (u64, Flags) {
    let m = mask(bytes);
    let bits = bytes * 8;
    let (a, b) = (a & m, b & m);
    let full = u128::from(a) + u128::from(b) + u128::from(carry);
    let r = (full as u64) & m;
    let cf = pick((full >> bits) & 1 != 0, CF);
    let of = pick(bit((a ^ r) & (b ^ r), bits - 1) != 0, OF);
    let af = pick((a ^ b ^ r) & 0x10 != 0, AF);
    (r, Flags { def: ALL, val: szp(r, bits) | cf | of | af, undef: 0 })
}

/// `sbb`: `a - b - borrow`, exact flags (same model as `Alu::Sub`).
#[must_use]
pub fn sbb(a: u64, b: u64, borrow: bool, bytes: u32) -> (u64, Flags) {
    let m = mask(bytes);
    let bits = bytes * 8;
    let (a, b) = (a & m, b & m);
    let rhs = u128::from(b) + u128::from(borrow);
    let r = (u128::from(a).wrapping_sub(rhs) as u64) & m;
    let cf = pick(u128::from(a) < rhs, CF);
    let of = pick(bit((a ^ b) & (a ^ r), bits - 1) != 0, OF);
    let af = pick((a ^ b ^ r) & 0x10 != 0, AF);
    (r, Flags { def: ALL, val: szp(r, bits) | cf | of | af, undef: 0 })
}

// mul/imul leave SF, ZF, AF, PF architecturally undefined; only CF/OF are set.
const MULDEF: u64 = CF | OF;
const MULUNDEF: u64 = SF | ZF | AF | PF;

/// Unsigned 1-operand `mul`: returns `(low, high)` halves and flags. CF=OF when
/// the high half is non-zero (result does not fit the low half).
#[must_use]
pub fn mul(a: u64, b: u64, bytes: u32) -> (u64, u64, Flags) {
    let m = mask(bytes);
    let bits = bytes * 8;
    let p = u128::from(a & m) * u128::from(b & m);
    let lo = (p as u64) & m;
    let hi = ((p >> bits) as u64) & m;
    (lo, hi, Flags { def: MULDEF, val: pick(hi != 0, MULDEF), undef: MULUNDEF })
}

/// Signed 1-operand `imul`: returns `(low, high)` halves and flags. CF=OF when
/// the full product does not fit a signed `bytes`-wide value.
#[must_use]
pub fn imul1(a: u64, b: u64, bytes: u32) -> (u64, u64, Flags) {
    let m = mask(bytes);
    let bits = bytes * 8;
    let p = i128::from(sext(a, bytes) as i64) * i128::from(sext(b, bytes) as i64);
    let lo = (p as u64) & m;
    let hi = ((p >> bits) as u64) & m;
    let fits = p >= -(1i128 << (bits - 1)) && p < (1i128 << (bits - 1));
    (lo, hi, Flags { def: MULDEF, val: pick(!fits, MULDEF), undef: MULUNDEF })
}

/// Signed 2/3-operand `imul`: truncated result with CF=OF on signed overflow.
#[must_use]
pub fn imul_n(a: u64, b: u64, bytes: u32) -> (u64, Flags) {
    let m = mask(bytes);
    let bits = bytes * 8;
    let p = i128::from(sext(a, bytes) as i64) * i128::from(sext(b, bytes) as i64);
    let r = (p as u64) & m;
    let fits = p >= -(1i128 << (bits - 1)) && p < (1i128 << (bits - 1));
    (r, Flags { def: MULDEF, val: pick(!fits, MULDEF), undef: MULUNDEF })
}

/// `div`/`idiv` of the double-wide dividend `hi:lo` by `divisor`. For byte
/// operands pass `hi = AH`, `lo = AL`. Returns `(quotient, remainder)` or
/// `None` for #DE (divide-by-zero or quotient out of range) — never traps.
/// All status flags are left undefined by the caller.
#[must_use]
pub fn divmod(hi: u64, lo: u64, divisor: u64, bytes: u32, signed: bool) -> Option<(u64, u64)> {
    let m = mask(bytes);
    let bits = bytes * 8;
    if divisor & m == 0 {
        return None;
    }
    let raw = (u128::from(hi & m) << bits) | u128::from(lo & m);
    if signed {
        let sh = 128 - 2 * bits;
        let num = if sh == 0 { raw as i128 } else { ((raw << sh) as i128) >> sh };
        let d = i128::from(sext(divisor, bytes) as i64);
        let q = num / d;
        let r = num % d;
        let qmax = (1i128 << (bits - 1)) - 1;
        let qmin = -(1i128 << (bits - 1));
        if q < qmin || q > qmax {
            return None;
        }
        Some(((q as u64) & m, (r as u64) & m))
    } else {
        let d = u128::from(divisor & m);
        let q = raw / d;
        if q > u128::from(m) {
            return None;
        }
        Some((q as u64, (raw % d) as u64))
    }
}

/// `shld`/`shrd` double-precision shift, exact flags. Requires `0 < cnt < bits`
/// (the caller rejects an out-of-range count, which is architecturally
/// undefined). CF is the last bit shifted out; OF defined only for a 1-bit
/// shift; AF undefined.
#[must_use]
pub fn dshift(left: bool, dest: u64, src: u64, cnt: u32, bytes: u32) -> (u64, Flags) {
    let m = mask(bytes);
    let bits = bytes * 8;
    let (dest, src) = (dest & m, src & m);
    let (res, cf) = if left {
        (((dest << cnt) | (src >> (bits - cnt))) & m, bit(dest, bits - cnt))
    } else {
        (((dest >> cnt) | (src << (bits - cnt))) & m, bit(dest, cnt - 1))
    };
    let of = bit(dest, bits - 1) ^ bit(res, bits - 1);
    let mut def = CF | SF | ZF | PF;
    let mut val = szp(res, bits) | pick(cf != 0, CF);
    let mut undef = AF;
    if cnt == 1 {
        def |= OF;
        val |= pick(of != 0, OF);
    } else {
        undef |= OF;
    }
    (res, Flags { def, val, undef })
}

/// `rol`/`ror`/`rcl`/`rcr`. `None` when the masked count is a no-op (flags
/// unchanged). CF always defined; OF defined only for a 1-bit rotate (else
/// undefined); SF/ZF/AF/PF are never affected.
#[must_use]
pub fn rotate(kind: Rotate, a: u64, count: u64, cf_in: bool, bytes: u32) -> Option<(u64, Flags)> {
    let m = mask(bytes);
    let bits = bytes * 8;
    let a = a & m;
    let masked = (count & if bytes == 8 { 63 } else { 31 }) as u32;
    let (res, cf) = match kind {
        Rotate::Rol | Rotate::Ror => {
            if masked == 0 {
                return None;
            }
            let cnt = masked % bits;
            let res = if cnt == 0 {
                a
            } else if kind == Rotate::Rol {
                ((a << cnt) | (a >> (bits - cnt))) & m
            } else {
                ((a >> cnt) | (a << (bits - cnt))) & m
            };
            let cf = if kind == Rotate::Rol { res & 1 } else { bit(res, bits - 1) };
            (res, cf)
        }
        Rotate::Rcl | Rotate::Rcr => {
            let eff = masked % (bits + 1);
            if eff == 0 {
                return None;
            }
            let mut v = a;
            let mut carry = u64::from(cf_in);
            for _ in 0..eff {
                if kind == Rotate::Rcl {
                    let nc = bit(v, bits - 1);
                    v = ((v << 1) | carry) & m;
                    carry = nc;
                } else {
                    let nc = v & 1;
                    v = (v >> 1) | (carry << (bits - 1));
                    carry = nc;
                }
            }
            (v, carry)
        }
    };
    let of = match kind {
        Rotate::Rol | Rotate::Rcl => bit(res, bits - 1) ^ cf,
        Rotate::Ror | Rotate::Rcr => bit(res, bits - 1) ^ bit(res, bits - 2),
    };
    let mut def = CF;
    let mut undef = 0;
    let mut val = pick(cf != 0, CF);
    if masked == 1 {
        def |= OF;
        val |= pick(of != 0, OF);
    } else {
        undef |= OF;
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

    #[test]
    fn adc_sbb_with_carry() {
        // 0xff + 0 + 1 = 0x100 -> byte 0, carry out.
        let (r, f) = adc(0xff, 0, true, 1);
        assert_eq!(r, 0);
        assert_eq!(f.val & (CF | ZF | AF), CF | ZF | AF);
        // 0 - 0 - borrow = 0xff with CF (borrow out), SF set.
        let (r, f) = sbb(0, 0, true, 1);
        assert_eq!(r, 0xff);
        assert_eq!(f.val & (CF | SF), CF | SF);
        // signed overflow: 0x7f + 0 + 1 = 0x80.
        let (_, f) = adc(0x7f, 0, true, 1);
        assert_eq!(f.val & (OF | SF), OF | SF);
    }

    #[test]
    fn mul_imul_halves_and_flags() {
        // unsigned 8-bit: 0x10 * 0x10 = 0x100 -> AL=0x00, AH=0x01, CF/OF set.
        let (lo, hi, f) = mul(0x10, 0x10, 1);
        assert_eq!((lo, hi), (0, 1));
        assert_eq!(f.val & (CF | OF), CF | OF);
        assert_eq!(f.undef, SF | ZF | AF | PF);
        // unsigned fits: 3 * 4 = 12, no carry.
        let (lo, hi, f) = mul(3, 4, 4);
        assert_eq!((lo, hi), (12, 0));
        assert_eq!(f.val & (CF | OF), 0);
        // signed 1-op: (-1) * (-1) = 1, fits -> no CF/OF.
        let (lo, hi, f) = imul1(0xff, 0xff, 1);
        assert_eq!((lo, hi), (1, 0));
        assert_eq!(f.val & (CF | OF), 0);
        // signed 2-op overflow: 0x40 * 4 = 0x100 doesn't fit signed byte.
        let (r, f) = imul_n(0x40, 4, 1);
        assert_eq!(r, 0);
        assert_eq!(f.val & (CF | OF), CF | OF);
    }

    #[test]
    fn divmod_quotient_remainder_and_de() {
        // 0x0101 / 2 = 0x80 r1 (byte: AX=0x0101).
        assert_eq!(divmod(0x01, 0x01, 2, 1, false), Some((0x80, 1)));
        // 100 / 7 = 14 r2, 32-bit.
        assert_eq!(divmod(0, 100, 7, 4, false), Some((14, 2)));
        // divide by zero -> None (#DE), never a trap.
        assert_eq!(divmod(0, 5, 0, 4, false), None);
        // quotient overflow: 0x1_0000 / 1 doesn't fit a 16-bit quotient -> None.
        assert_eq!(divmod(1, 0, 1, 2, false), None);
        // signed: -14 / 4 = -3 r -2 (trunc toward zero).
        let (q, r) = divmod(0xffff_ffff, 0xffff_fff2, 4, 4, true).unwrap();
        assert_eq!((q as i32, r as i32), (-3, -2));
        // signed overflow: i32::MIN / -1 -> None.
        assert_eq!(divmod(0xffff_ffff, 0x8000_0000, 0xffff_ffff, 4, true), None);
    }

    #[test]
    fn rotate_rol_ror_rcl_rcr() {
        // rol byte 0x80 by 1 -> 0x01, CF=1, OF = msb^cf = 0^1 = 1.
        let (r, f) = rotate(Rotate::Rol, 0x80, 1, false, 1).unwrap();
        assert_eq!(r, 1);
        assert_eq!(f.val & (CF | OF), CF | OF);
        // ror byte 0x01 by 1 -> 0x80, CF=1.
        let (r, f) = rotate(Rotate::Ror, 0x01, 1, false, 1).unwrap();
        assert_eq!(r, 0x80);
        assert_eq!(f.val & CF, CF);
        // rcl byte 0x80 by 1 with CF=0 -> 0x00, carry out = 1.
        let (r, f) = rotate(Rotate::Rcl, 0x80, 1, false, 1).unwrap();
        assert_eq!(r, 0);
        assert_eq!(f.val & CF, CF);
        // rcr byte 0x01 by 1 with CF=1 -> 0x80 (carry rotates into msb), CF=1.
        let (r, f) = rotate(Rotate::Rcr, 0x01, 1, true, 1).unwrap();
        assert_eq!(r, 0x80);
        assert_eq!(f.val & CF, CF);
        // zero masked count is a no-op; multi-bit leaves OF undefined.
        assert!(rotate(Rotate::Rol, 0x12, 0, false, 1).is_none());
        let (_, f) = rotate(Rotate::Rol, 0x12, 3, false, 1).unwrap();
        assert_eq!(f.undef, OF);
    }
}
