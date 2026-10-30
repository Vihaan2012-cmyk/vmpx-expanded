//! Static value-range analysis over [`Expr`].
//!
//! [`ValueRange`] over-approximates the set of values an expression can take as
//! an unsigned interval plus known-zero/known-one bit masks. A range is only
//! returned when every possible value lies inside it; anything not derivable is
//! the full-width [`ValueRange::unbounded`] range. Path facts (comparison
//! guards) are supplied through [`RangeContext`] and keyed by structural
//! expression equality.
//!
//! # Jump-table recovery sketch
//!
//! For a computed dispatch `target = Load(base + index * stride)`:
//! 1. Build a [`RangeContext`] from the dominating guards on the path, e.g. an
//!    unsigned `cmp index, N; ja default` contributes `assume_ule(&index, N)`.
//! 2. `ctx.range_of(&index)` gives the proven index interval, and
//!    `.values(limit)` the finite list of indices when the table is small.
//!    An unbounded result means the table size is unproven: do not enumerate.
//! 3. Substitute each index into the address expression to read one target per
//!    entry, and keep the dispatch edges only for entries inside the proven
//!    range. A masked index (`and index, 0x7`) needs no guard at all.
//! 4. A branch whose condition operand has a one-value range is decided
//!    statically; one whose range excludes the compared constant is dead.

use std::collections::{BTreeSet, HashMap};

use crate::{BinaryOp, BitWidth, Expr, ExprKind, ResizeOp, UnaryOp};

const MAX_DEPTH: usize = 256;
const MAX_SHIFT_CANDIDATES: u64 = 64;

/// Proven bound on an expression's unsigned value within its width.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ValueRange {
    width: BitWidth,
    min: u64,
    max: u64,
    known_zero: u64,
    known_one: u64,
}

impl ValueRange {
    /// Every value of `width`.
    #[must_use]
    pub const fn unbounded(width: BitWidth) -> Self {
        Self {
            width,
            min: 0,
            max: width.mask(),
            known_zero: 0,
            known_one: 0,
        }
    }

    #[must_use]
    pub fn constant(width: BitWidth, value: u64) -> Self {
        let value = width.truncate(value);
        make(width, value, value, !value & width.mask(), value)
    }

    /// Inclusive unsigned interval; an inverted or out-of-width interval is not
    /// a proof, so it yields the unbounded range.
    #[must_use]
    pub fn interval(width: BitWidth, min: u64, max: u64) -> Self {
        if min > max || max > width.mask() {
            return Self::unbounded(width);
        }
        make(width, min, max, 0, 0)
    }

    #[must_use]
    pub const fn width(&self) -> BitWidth {
        self.width
    }

    #[must_use]
    pub const fn min(&self) -> u64 {
        self.min
    }

    #[must_use]
    pub const fn max(&self) -> u64 {
        self.max
    }

    #[must_use]
    pub const fn known_zero(&self) -> u64 {
        self.known_zero
    }

    #[must_use]
    pub const fn known_one(&self) -> u64 {
        self.known_one
    }

    #[must_use]
    pub const fn is_unbounded(&self) -> bool {
        self.min == 0
            && self.max == self.width.mask()
            && self.known_zero == 0
            && self.known_one == 0
    }

    #[must_use]
    pub const fn as_constant(&self) -> Option<u64> {
        if self.min == self.max {
            Some(self.min)
        } else {
            None
        }
    }

    #[must_use]
    pub const fn contains(&self, value: u64) -> bool {
        value >= self.min
            && value <= self.max
            && value & self.known_zero == 0
            && value & self.known_one == self.known_one
    }

    /// Every possible value in ascending order, or `None` when more than
    /// `limit` values fall inside the interval.
    #[must_use]
    pub fn values(&self, limit: u64) -> Option<Vec<u64>> {
        if u128::from(self.max - self.min) >= u128::from(limit) {
            return None;
        }
        Some(
            (self.min..=self.max)
                .filter(|v| self.contains(*v))
                .collect(),
        )
    }

    /// Smallest range containing both operands.
    #[must_use]
    pub fn join(&self, other: &Self) -> Self {
        if self.width != other.width {
            return Self::unbounded(self.width);
        }
        make(
            self.width,
            self.min.min(other.min),
            self.max.max(other.max),
            self.known_zero & other.known_zero,
            self.known_one & other.known_one,
        )
    }

    /// Values in both ranges; `None` when provably disjoint.
    #[must_use]
    pub fn intersect(&self, other: &Self) -> Option<Self> {
        if self.width != other.width {
            return None;
        }
        let min = self.min.max(other.min);
        let max = self.max.min(other.max);
        if min > max {
            return None;
        }
        refine(
            self.width,
            min,
            max,
            self.known_zero | other.known_zero,
            self.known_one | other.known_one,
        )
    }
}

/// Path facts (guards) under which expressions are bounded.
#[derive(Clone, Debug, Default)]
pub struct RangeContext {
    facts: HashMap<Expr, ValueRange>,
}

impl RangeContext {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records that `expr` lies in `range` on this path. Facts of a different
    /// width, or contradicting an earlier fact, are ignored (never unsound).
    pub fn assume(&mut self, expr: &Expr, range: ValueRange) {
        if range.width != expr.width {
            return;
        }
        let merged = match self.facts.get(expr) {
            Some(prior) => match prior.intersect(&range) {
                Some(merged) => merged,
                None => return,
            },
            None => range,
        };
        self.facts.insert(expr.clone(), merged);
    }

    /// Unsigned guard `expr <= max`.
    pub fn assume_ule(&mut self, expr: &Expr, max: u64) {
        self.assume(expr, ValueRange::interval(expr.width, 0, max));
    }

    /// Unsigned guard `expr < bound`.
    pub fn assume_ult(&mut self, expr: &Expr, bound: u64) {
        if let Some(max) = bound.checked_sub(1) {
            self.assume_ule(expr, max);
        }
    }

    /// Unsigned guard `expr >= min`.
    pub fn assume_uge(&mut self, expr: &Expr, min: u64) {
        self.assume(
            expr,
            ValueRange::interval(expr.width, min, expr.width.mask()),
        );
    }

    /// `expr == value`.
    pub fn assume_eq(&mut self, expr: &Expr, value: u64) {
        self.assume(expr, ValueRange::constant(expr.width, value));
    }

    #[must_use]
    pub fn range_of(&self, expr: &Expr) -> ValueRange {
        let mut memo = HashMap::new();
        self.eval(expr, 0, &mut memo)
    }

    fn eval(
        &self,
        expr: &Expr,
        depth: usize,
        memo: &mut HashMap<*const Expr, ValueRange>,
    ) -> ValueRange {
        let width = expr.width;
        if depth >= MAX_DEPTH {
            return ValueRange::unbounded(width);
        }
        let key = std::ptr::from_ref(expr);
        if let Some(done) = memo.get(&key) {
            return *done;
        }
        let structural = self.structural(expr, depth, memo);
        let result = match self.facts.get(expr) {
            Some(fact) => structural.intersect(fact).unwrap_or(structural),
            None => structural,
        };
        memo.insert(key, result);
        result
    }

    fn structural(
        &self,
        expr: &Expr,
        depth: usize,
        memo: &mut HashMap<*const Expr, ValueRange>,
    ) -> ValueRange {
        let width = expr.width;
        match &expr.kind {
            ExprKind::Const { value } => ValueRange::constant(width, *value),
            ExprKind::Register { .. } | ExprKind::Load { .. } | ExprKind::Intrinsic { .. } => {
                ValueRange::unbounded(width)
            }
            ExprKind::Unary { op, value } => {
                let inner = self.eval(value, depth + 1, memo);
                unary_range(*op, &inner)
            }
            ExprKind::Binary { op, left, right } => {
                let left = self.eval(left, depth + 1, memo);
                let right = self.eval(right, depth + 1, memo);
                binary_range(*op, &left, &right)
            }
            ExprKind::Extract { value, lsb } => {
                let inner = self.eval(value, depth + 1, memo);
                extract_range(width, &inner, *lsb)
            }
            ExprKind::Concat { high, low } => {
                let high = self.eval(high, depth + 1, memo);
                let low = self.eval(low, depth + 1, memo);
                concat_range(width, &high, &low)
            }
            ExprKind::Resize { op, value } => {
                let inner = self.eval(value, depth + 1, memo);
                resize_range(*op, width, &inner)
            }
        }
    }
}

/// Range of `expr` with no path facts.
#[must_use]
pub fn value_range(expr: &Expr) -> ValueRange {
    RangeContext::new().range_of(expr)
}

fn make(width: BitWidth, min: u64, max: u64, zero: u64, one: u64) -> ValueRange {
    refine(width, min, max, zero, one).unwrap_or(ValueRange::unbounded(width))
}

/// Tightens interval and bit masks against each other; `None` on contradiction.
fn refine(width: BitWidth, min: u64, max: u64, zero: u64, one: u64) -> Option<ValueRange> {
    let mask = width.mask();
    let mut min = min & mask;
    let mut max = max & mask;
    let mut zero = zero & mask;
    let mut one = one & mask;
    for _ in 0..2 {
        if zero & one != 0 {
            return None;
        }
        min = min.max(one);
        max = max.min(mask & !zero);
        if min > max {
            return None;
        }
        let diff = min ^ max;
        let fixed = if diff == 0 {
            mask
        } else {
            let below = u64::MAX.checked_shr(diff.leading_zeros()).unwrap_or(0);
            mask & !below
        };
        one |= min & fixed;
        zero |= !min & fixed;
    }
    if zero & one != 0 {
        return None;
    }
    min = min.max(one);
    max = max.min(mask & !zero);
    if min > max {
        return None;
    }
    Some(ValueRange {
        width,
        min,
        max,
        known_zero: zero,
        known_one: one,
    })
}

fn from_interval_u128(width: BitWidth, min: u128, max: u128) -> ValueRange {
    let mask = u128::from(width.mask());
    if max > mask || min > max {
        return ValueRange::unbounded(width);
    }
    let (Ok(min), Ok(max)) = (u64::try_from(min), u64::try_from(max)) else {
        return ValueRange::unbounded(width);
    };
    make(width, min, max, 0, 0)
}

fn trailing_zero_bits(range: &ValueRange) -> u32 {
    (!range.known_zero).trailing_zeros().min(range.width.bits())
}

fn low_zero_mask(width: BitWidth, count: u32) -> u64 {
    if count >= 64 {
        width.mask()
    } else {
        ((1_u64 << count) - 1) & width.mask()
    }
}

fn unary_range(op: UnaryOp, value: &ValueRange) -> ValueRange {
    let width = value.width;
    let mask = width.mask();
    match op {
        UnaryOp::Not => make(
            width,
            mask - value.max,
            mask - value.min,
            value.known_one,
            value.known_zero,
        ),
        UnaryOp::Neg => {
            if value.min == 0 && value.max == 0 {
                return ValueRange::constant(width, 0);
            }
            let tz = trailing_zero_bits(value);
            let zero = low_zero_mask(width, tz);
            if value.min == 0 {
                return make(width, 0, mask, zero, 0);
            }
            let modulus = u128::from(mask) + 1;
            let lo = modulus - u128::from(value.max);
            let hi = modulus - u128::from(value.min);
            let base = from_interval_u128(width, lo, hi);
            make(width, base.min, base.max, zero, 0)
        }
    }
}

fn shift_count_mask(width: BitWidth) -> u64 {
    if width == BitWidth::W64 { 0x3F } else { 0x1F }
}

fn sar_eval(value: u64, width: BitWidth, count: u32) -> u64 {
    let value = width.truncate(value);
    let sign = 1_u64 << (width.bits() - 1);
    let extended = if value & sign == 0 {
        value
    } else {
        value | !width.mask()
    };
    let signed = i64::from_ne_bytes(extended.to_ne_bytes());
    width.truncate(signed.wrapping_shr(count).cast_unsigned())
}

fn rol_eval(value: u64, width: BitWidth, count: u32) -> u64 {
    let bits = width.bits();
    let count = count % bits;
    let value = width.truncate(value);
    if count == 0 {
        value
    } else {
        width.truncate((value << count) | (value >> (bits - count)))
    }
}

fn shift_by(op: BinaryOp, x: &ValueRange, count: u32) -> ValueRange {
    let width = x.width;
    let mask = width.mask();
    match op {
        BinaryOp::Shl => {
            let zero = (x.known_zero << count) | low_zero_mask(width, count);
            let one = x.known_one << count;
            let max = u128::from(x.max) << count;
            let min = u128::from(x.min) << count;
            if max <= u128::from(mask) {
                let base = from_interval_u128(width, min, max);
                make(width, base.min, base.max, zero, one)
            } else {
                make(width, 0, mask, zero, one)
            }
        }
        BinaryOp::Shr => {
            let high = mask & !(mask >> count);
            make(
                width,
                x.min >> count,
                x.max >> count,
                (x.known_zero >> count) | high,
                x.known_one >> count,
            )
        }
        BinaryOp::Sar => {
            let sign = 1_u64 << (width.bits() - 1);
            let mut hull: Option<(u64, u64)> = None;
            let mut add = |lo: u64, hi: u64| {
                let (lo, hi) = (sar_eval(lo, width, count), sar_eval(hi, width, count));
                hull = Some(hull.map_or((lo, hi), |(a, b)| (a.min(lo), b.max(hi))));
            };
            if x.min < sign {
                add(x.min, x.max.min(sign - 1));
            }
            if x.max >= sign {
                add(x.min.max(sign), x.max);
            }
            match hull {
                Some((lo, hi)) => make(width, lo, hi, 0, 0),
                None => ValueRange::unbounded(width),
            }
        }
        BinaryOp::Rol | BinaryOp::Ror => {
            let bits = width.bits();
            let n = count % bits;
            let left = if op == BinaryOp::Rol {
                n
            } else {
                (bits - n) % bits
            };
            if left == 0 {
                return *x;
            }
            make(
                width,
                0,
                mask,
                rol_eval(x.known_zero, width, left),
                rol_eval(x.known_one, width, left),
            )
        }
        _ => ValueRange::unbounded(width),
    }
}

fn shift_range(op: BinaryOp, x: &ValueRange, amount: &ValueRange) -> ValueRange {
    let width = x.width;
    let Some(raw) = amount.values(MAX_SHIFT_CANDIDATES) else {
        return ValueRange::unbounded(width);
    };
    let counts: BTreeSet<u32> = raw
        .into_iter()
        .filter_map(|v| u32::try_from(v & shift_count_mask(width)).ok())
        .collect();
    let mut joined: Option<ValueRange> = None;
    for count in counts {
        let one = shift_by(op, x, count);
        joined = Some(joined.map_or(one, |acc| acc.join(&one)));
    }
    joined.unwrap_or(ValueRange::unbounded(width))
}

fn binary_range(op: BinaryOp, a: &ValueRange, b: &ValueRange) -> ValueRange {
    let width = a.width;
    if b.width != width {
        return ValueRange::unbounded(width);
    }
    let mask = width.mask();
    match op {
        BinaryOp::And => make(
            width,
            0,
            a.max.min(b.max),
            a.known_zero | b.known_zero,
            a.known_one & b.known_one,
        ),
        BinaryOp::Or => make(
            width,
            a.min.max(b.min),
            mask,
            a.known_zero & b.known_zero,
            a.known_one | b.known_one,
        ),
        BinaryOp::Xor => make(
            width,
            0,
            mask,
            (a.known_zero & b.known_zero) | (a.known_one & b.known_one),
            (a.known_zero & b.known_one) | (a.known_one & b.known_zero),
        ),
        BinaryOp::Add => {
            let zero = low_zero_mask(width, trailing_zero_bits(a).min(trailing_zero_bits(b)));
            let lo = u128::from(a.min) + u128::from(b.min);
            let hi = u128::from(a.max) + u128::from(b.max);
            let modulus = u128::from(mask) + 1;
            let base = if hi <= u128::from(mask) {
                from_interval_u128(width, lo, hi)
            } else if lo >= modulus && hi < 2 * modulus {
                from_interval_u128(width, lo - modulus, hi - modulus)
            } else {
                ValueRange::unbounded(width)
            };
            make(width, base.min, base.max, zero, 0)
        }
        BinaryOp::Sub => {
            let zero = low_zero_mask(width, trailing_zero_bits(a).min(trailing_zero_bits(b)));
            let modulus = i128::from(mask) + 1;
            let lo = i128::from(a.min) - i128::from(b.max);
            let hi = i128::from(a.max) - i128::from(b.min);
            let base = if lo >= 0 {
                from_interval_u128(width, lo.cast_unsigned(), hi.cast_unsigned())
            } else if hi < 0 {
                from_interval_u128(
                    width,
                    (lo + modulus).cast_unsigned(),
                    (hi + modulus).cast_unsigned(),
                )
            } else {
                ValueRange::unbounded(width)
            };
            make(width, base.min, base.max, zero, 0)
        }
        BinaryOp::Mul => {
            let zero = low_zero_mask(width, trailing_zero_bits(a) + trailing_zero_bits(b));
            let lo = u128::from(a.min) * u128::from(b.min);
            let hi = u128::from(a.max) * u128::from(b.max);
            let base = from_interval_u128(width, lo, hi);
            make(width, base.min, base.max, zero, 0)
        }
        BinaryOp::Shl | BinaryOp::Shr | BinaryOp::Sar | BinaryOp::Rol | BinaryOp::Ror => {
            shift_range(op, a, b)
        }
    }
}

fn extract_range(width: BitWidth, value: &ValueRange, lsb: u32) -> ValueRange {
    if lsb >= 64 {
        return ValueRange::unbounded(width);
    }
    let mask = width.mask();
    let zero = (value.known_zero >> lsb) & mask;
    let one = (value.known_one >> lsb) & mask;
    let (lo, hi) = (value.min >> lsb, value.max >> lsb);
    if hi <= mask {
        make(width, lo, hi, zero, one)
    } else {
        make(width, 0, mask, zero, one)
    }
}

fn concat_range(width: BitWidth, high: &ValueRange, low: &ValueRange) -> ValueRange {
    let shift = low.width.bits();
    if shift >= 64 || high.width.bits() + shift != width.bits() {
        return ValueRange::unbounded(width);
    }
    make(
        width,
        (high.min << shift) | low.min,
        (high.max << shift) | low.max,
        (high.known_zero << shift) | low.known_zero,
        (high.known_one << shift) | low.known_one,
    )
}

fn resize_range(op: ResizeOp, target: BitWidth, value: &ValueRange) -> ValueRange {
    let source = value.width;
    let mask = target.mask();
    match op {
        ResizeOp::Truncate => {
            let zero = value.known_zero & mask;
            let one = value.known_one & mask;
            if value.max <= mask {
                make(target, value.min, value.max, zero, one)
            } else if target.bits() < 64 && value.min >> target.bits() == value.max >> target.bits()
            {
                make(target, value.min & mask, value.max & mask, zero, one)
            } else {
                make(target, 0, mask, zero, one)
            }
        }
        ResizeOp::ZeroExtend => make(
            target,
            value.min,
            value.max,
            value.known_zero | (mask & !source.mask()),
            value.known_one,
        ),
        ResizeOp::SignExtend => {
            let sign = 1_u64 << (source.bits() - 1);
            let ext = mask & !source.mask();
            let (zero, one) = if value.known_zero & sign != 0 {
                (value.known_zero | ext, value.known_one)
            } else if value.known_one & sign != 0 {
                (value.known_zero, value.known_one | ext)
            } else {
                (value.known_zero, value.known_one)
            };
            let (min, max) = if value.max < sign {
                (value.min, value.max)
            } else if value.min >= sign {
                (value.min | ext, value.max | ext)
            } else {
                (value.min, value.max | ext)
            };
            make(target, min, max, zero, one)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ExprRef;

    fn reg(width: BitWidth, name: &str) -> Expr {
        Expr::register(width, name)
    }

    fn k(width: BitWidth, value: u64) -> Expr {
        Expr::constant(width, value)
    }

    fn bin(op: BinaryOp, left: Expr, right: Expr) -> Expr {
        Expr::binary(op, left, right).unwrap()
    }

    fn raw(op: BinaryOp, left: Expr, right: Expr) -> Expr {
        Expr {
            width: left.width,
            kind: ExprKind::Binary {
                op,
                left: ExprRef::new(left),
                right: ExprRef::new(right),
            },
        }
    }

    #[test]
    fn mask_bounds_to_mask_value() {
        let x = reg(BitWidth::W32, "x");
        let r = value_range(&bin(BinaryOp::And, x, k(BitWidth::W32, 0xFF)));
        assert_eq!((r.min(), r.max()), (0, 255));
        assert!(!r.is_unbounded());
    }

    #[test]
    fn non_contiguous_mask_records_known_bits() {
        let x = reg(BitWidth::W32, "x");
        let r = value_range(&bin(BinaryOp::And, x, k(BitWidth::W32, 0b1010)));
        assert_eq!(r.max(), 0b1010);
        assert!(r.contains(0b1000) && !r.contains(0b0100));
        assert_eq!(r.values(16).unwrap(), vec![0, 2, 8, 10]);
    }

    #[test]
    fn unconstrained_reads_are_unbounded() {
        let w = BitWidth::W32;
        assert!(value_range(&reg(w, "x")).is_unbounded());
        assert!(value_range(&Expr::load(w, reg(BitWidth::W64, "p"))).is_unbounded());
        assert!(value_range(&Expr::intrinsic(w, "mystery", vec![reg(w, "x")])).is_unbounded());
        let sum = bin(BinaryOp::Add, reg(w, "x"), k(w, 1));
        assert!(value_range(&sum).is_unbounded());
    }

    #[test]
    fn guard_bounds_value_and_derived_expressions() {
        let w = BitWidth::W32;
        let index = reg(w, "idx");
        let scaled = bin(BinaryOp::Mul, index.clone(), k(w, 4));
        let mut ctx = RangeContext::new();
        assert!(ctx.range_of(&index).is_unbounded());
        ctx.assume_ule(&index, 7);
        let r = ctx.range_of(&index);
        assert_eq!((r.min(), r.max()), (0, 7));
        assert_eq!(r.values(64).unwrap().len(), 8);
        let s = ctx.range_of(&scaled);
        assert_eq!((s.min(), s.max()), (0, 28));
        assert!(s.contains(28) && !s.contains(2));
    }

    #[test]
    fn guards_combine_and_contradictions_are_ignored() {
        let w = BitWidth::W16;
        let x = reg(w, "x");
        let mut ctx = RangeContext::new();
        ctx.assume_uge(&x, 3);
        ctx.assume_ult(&x, 10);
        let r = ctx.range_of(&x);
        assert_eq!((r.min(), r.max()), (3, 9));
        ctx.assume_ule(&x, 1);
        assert_eq!(ctx.range_of(&x), r);
        ctx.assume_ult(&x, 0);
        assert_eq!(ctx.range_of(&x), r);
    }

    #[test]
    fn masked_then_shifted_bound() {
        let w = BitWidth::W32;
        let x = reg(w, "x");
        let masked = bin(BinaryOp::And, x.clone(), k(w, 0xFF));
        let shl = bin(BinaryOp::Shl, masked.clone(), k(w, 4));
        let r = value_range(&shl);
        assert_eq!((r.min(), r.max()), (0, 0xFF0));
        assert!(r.contains(0xAB0) && !r.contains(0xAB1));
        let shr = bin(BinaryOp::Shr, x.clone(), k(w, 24));
        let r = value_range(&shr);
        assert_eq!((r.min(), r.max()), (0, 255));
        let both = bin(BinaryOp::Shr, bin(BinaryOp::And, x, k(w, 0xF0F0)), k(w, 4));
        assert_eq!(value_range(&both).max(), 0x0F0F);
    }

    #[test]
    fn bitfield_extracts_and_resizes() {
        let x = reg(BitWidth::W32, "x");
        let byte = Expr::extract(BitWidth::W8, 8, x.clone()).unwrap();
        assert!(value_range(&byte).is_unbounded());
        let wide = Expr::resize(ResizeOp::ZeroExtend, BitWidth::W32, byte).unwrap();
        let r = value_range(&wide);
        assert_eq!((r.min(), r.max()), (0, 255));
        let narrow = Expr::resize(
            ResizeOp::Truncate,
            BitWidth::W16,
            bin(BinaryOp::And, x, k(BitWidth::W32, 0x3FF)),
        )
        .unwrap();
        assert_eq!(value_range(&narrow).max(), 0x3FF);
        let sext =
            Expr::resize(ResizeOp::SignExtend, BitWidth::W32, reg(BitWidth::W8, "b")).unwrap();
        assert!(value_range(&sext).contains(0xFFFF_FF80));
        assert!(value_range(&sext).contains(0x7F));
    }

    #[test]
    fn concat_of_zero_high_is_bounded_by_low() {
        let low =
            Expr::resize(ResizeOp::ZeroExtend, BitWidth::W16, reg(BitWidth::W8, "b")).unwrap();
        let joined = raw_concat(k(BitWidth::W16, 0), low);
        assert_eq!(value_range(&joined).max(), 255);
    }

    fn raw_concat(high: Expr, low: Expr) -> Expr {
        Expr {
            width: BitWidth::W32,
            kind: ExprKind::Concat {
                high: ExprRef::new(high),
                low: ExprRef::new(low),
            },
        }
    }

    #[test]
    fn constants_and_set_enumeration() {
        let w = BitWidth::W32;
        let r = value_range(&raw(BinaryOp::Add, k(w, 5), k(w, 7)));
        assert_eq!(r.as_constant(), Some(12));
        let wrapped = value_range(&raw(BinaryOp::Sub, k(w, 1), k(w, 2)));
        assert_eq!(wrapped.as_constant(), Some(0xFFFF_FFFF));
        assert!(value_range(&reg(w, "x")).values(1 << 20).is_none());
    }

    #[test]
    fn variable_shift_amount_uses_amount_range() {
        let w = BitWidth::W32;
        let amount = bin(BinaryOp::And, reg(w, "n"), k(w, 3));
        let r = value_range(&raw(BinaryOp::Shl, k(w, 1), amount));
        assert_eq!((r.min(), r.max()), (1, 8));
        assert!([1, 2, 4, 8].iter().all(|v| r.contains(*v)));
    }

    fn samples(r: &ValueRange) -> Vec<u64> {
        r.values(40).unwrap_or_else(|| {
            let mut v = vec![r.min(), r.max(), r.max() / 2, r.min().saturating_add(1)];
            v.retain(|x| r.contains(*x));
            v
        })
    }

    #[test]
    fn binary_transfer_is_sound_against_concrete_folding() {
        let ops = [
            BinaryOp::Add,
            BinaryOp::Sub,
            BinaryOp::Mul,
            BinaryOp::Xor,
            BinaryOp::And,
            BinaryOp::Or,
            BinaryOp::Shl,
            BinaryOp::Shr,
            BinaryOp::Sar,
            BinaryOp::Rol,
            BinaryOp::Ror,
        ];
        for width in [BitWidth::W8, BitWidth::W16, BitWidth::W32, BitWidth::W64] {
            let mask = width.mask();
            let ranges = [
                ValueRange::interval(width, 3, 20),
                ValueRange::interval(width, mask - 9, mask),
                ValueRange::interval(width, (mask >> 1) - 7, (mask >> 1) + 9),
                ValueRange::constant(width, 5),
                ValueRange::constant(width, 0),
                ValueRange::interval(width, 0, 3),
                value_range(&bin(BinaryOp::And, reg(width, "x"), k(width, 0x2A))),
                value_range(&bin(
                    BinaryOp::Shl,
                    bin(BinaryOp::And, reg(width, "x"), k(width, 0x7)),
                    k(width, 2),
                )),
            ];
            for op in ops {
                for a in &ranges {
                    for b in &ranges {
                        let abstract_result = binary_range(op, a, b);
                        for x in samples(a) {
                            for y in samples(b) {
                                let folded = bin(op, k(width, x), k(width, y));
                                let ExprKind::Const { value } = folded.kind else {
                                    panic!("not folded: {op:?}");
                                };
                                assert!(
                                    abstract_result.contains(value),
                                    "{op:?} {width:?} {a:?} {b:?}: {x:#x} op {y:#x} = {value:#x} escapes {abstract_result:?}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn unary_and_resize_transfer_is_sound() {
        let width = BitWidth::W16;
        let ranges = [
            ValueRange::interval(width, 0, 0),
            ValueRange::interval(width, 1, 9),
            ValueRange::interval(width, 0, 9),
            ValueRange::interval(width, 0x7FF0, 0x8010),
            ValueRange::interval(width, 0xFFF0, 0xFFFF),
        ];
        for r in ranges {
            for x in samples(&r) {
                for op in [UnaryOp::Not, UnaryOp::Neg] {
                    let abstract_result = unary_range(op, &r);
                    let folded = Expr::unary(op, k(width, x));
                    let ExprKind::Const { value } = folded.kind else {
                        panic!("not folded");
                    };
                    assert!(abstract_result.contains(value), "{op:?} {x:#x}");
                }
                for (op, target) in [
                    (ResizeOp::ZeroExtend, BitWidth::W32),
                    (ResizeOp::SignExtend, BitWidth::W32),
                    (ResizeOp::Truncate, BitWidth::W8),
                ] {
                    let folded = Expr::resize(op, target, k(width, x)).unwrap();
                    let ExprKind::Const { value } = folded.kind else {
                        panic!("not folded");
                    };
                    assert!(
                        resize_range(op, target, &r).contains(value),
                        "{op:?} {x:#x}"
                    );
                }
            }
        }
    }
}
