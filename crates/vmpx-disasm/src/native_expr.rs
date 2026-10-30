use std::sync::Arc;

use crate::native_lift::{Algebra, AlgebraError, BinOp, Resize, UnOp, Width};

/// Self-contained expression tree used as the default lifter domain.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct NativeExpr(Arc<NativeNode>);

#[derive(Debug, Eq, Hash, PartialEq)]
pub enum NativeNode {
    Const {
        width: Width,
        value: u64,
    },
    Input {
        width: Width,
        name: String,
    },
    Load {
        width: Width,
        address: NativeExpr,
    },
    Unary {
        op: UnOp,
        value: NativeExpr,
    },
    Binary {
        op: BinOp,
        left: NativeExpr,
        right: NativeExpr,
    },
    Resize {
        kind: Resize,
        width: Width,
        value: NativeExpr,
    },
    Intrinsic {
        width: Width,
        name: String,
        args: Vec<NativeExpr>,
    },
}

impl NativeExpr {
    fn new(node: NativeNode) -> Self {
        Self(Arc::new(node))
    }

    #[must_use]
    pub fn node(&self) -> &NativeNode {
        &self.0
    }

    #[must_use]
    pub fn width(&self) -> Width {
        match self.node() {
            NativeNode::Const { width, .. }
            | NativeNode::Input { width, .. }
            | NativeNode::Load { width, .. }
            | NativeNode::Resize { width, .. }
            | NativeNode::Intrinsic { width, .. } => *width,
            NativeNode::Unary { value, .. } => value.width(),
            NativeNode::Binary { left, .. } => left.width(),
        }
    }

    #[must_use]
    pub fn as_const(&self) -> Option<u64> {
        match self.node() {
            NativeNode::Const { value, .. } => Some(*value),
            _ => None,
        }
    }

    /// Concretely evaluates the tree. `input` resolves named inputs, `load`
    /// resolves entry-state memory reads. Returns `None` for an unknown intrinsic.
    pub fn eval(
        &self,
        input: &dyn Fn(&str) -> u64,
        load: &dyn Fn(u64, Width) -> u64,
    ) -> Option<u64> {
        Some(match self.node() {
            NativeNode::Const { value, .. } => *value,
            NativeNode::Input { width, name } => width.truncate(input(name)),
            NativeNode::Load { width, address } => {
                width.truncate(load(address.eval(input, load)?, *width))
            }
            NativeNode::Unary { op, value } => {
                eval_unary(*op, value.width(), value.eval(input, load)?)
            }
            NativeNode::Binary { op, left, right } => eval_binary(
                *op,
                left.width(),
                left.eval(input, load)?,
                right.eval(input, load)?,
            ),
            NativeNode::Resize { kind, width, value } => {
                eval_resize(*kind, value.width(), *width, value.eval(input, load)?)
            }
            NativeNode::Intrinsic { width, name, args } => {
                let mut values = Vec::with_capacity(args.len());
                for arg in args {
                    values.push((arg.width(), arg.eval(input, load)?));
                }
                width.truncate(eval_intrinsic(name, &values)?)
            }
        })
    }
}

pub(crate) fn eval_unary(op: UnOp, width: Width, value: u64) -> u64 {
    width.truncate(match op {
        UnOp::Not => !value,
        UnOp::Neg => value.wrapping_neg(),
    })
}

/// Fixed-width semantics with mathematical shift counts: a shift of at least
/// the width yields 0 (or the sign fill for `Sar`), rotates are taken modulo the width.
pub(crate) fn eval_binary(op: BinOp, width: Width, left: u64, right: u64) -> u64 {
    let bits = u64::from(width.bits());
    let l = width.truncate(left);
    let r = width.truncate(right);
    let result = match op {
        BinOp::Add => l.wrapping_add(r),
        BinOp::Sub => l.wrapping_sub(r),
        BinOp::Mul => l.wrapping_mul(r),
        BinOp::And => l & r,
        BinOp::Or => l | r,
        BinOp::Xor => l ^ r,
        BinOp::Shl => {
            if r >= bits {
                0
            } else {
                l << r
            }
        }
        BinOp::Shr => {
            if r >= bits {
                0
            } else {
                l >> r
            }
        }
        BinOp::Sar => {
            let signed = width.sign_extend(l).cast_signed();
            (signed >> r.min(bits - 1)).cast_unsigned()
        }
        BinOp::Rol | BinOp::Ror => {
            let k = r % bits;
            let k = if op == BinOp::Rol {
                k
            } else {
                (bits - k) % bits
            };
            if k == 0 {
                l
            } else {
                (l << k) | (l >> (bits - k))
            }
        }
    };
    width.truncate(result)
}

pub(crate) fn eval_resize(kind: Resize, from: Width, to: Width, value: u64) -> u64 {
    let v = from.truncate(value);
    to.truncate(match kind {
        Resize::Truncate | Resize::ZeroExtend => v,
        Resize::SignExtend => from.sign_extend(v),
    })
}

fn sign_bit(width: Width, value: u64) -> u64 {
    (width.truncate(value) >> (width.bits() - 1)) & 1
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::many_single_char_names
)]
fn eval_intrinsic(name: &str, args: &[(Width, u64)]) -> Option<u64> {
    let flag = u64::from;
    match (name, args) {
        ("zf", [(w, r)]) => Some(flag(w.truncate(*r) == 0)),
        ("sf", [(w, r)]) => Some(sign_bit(*w, *r)),
        ("pf", [(_, r)]) => Some(flag((r & 0xFF).count_ones() % 2 == 0)),
        ("cf_add", [(w, a), (_, b)]) => {
            let sum = u128::from(w.truncate(*a)) + u128::from(w.truncate(*b));
            Some(flag((sum >> w.bits()) & 1 == 1))
        }
        ("of_add", [(w, a), (_, b)]) => {
            let r = w.truncate(a.wrapping_add(*b));
            Some(sign_bit(*w, (a ^ r) & (b ^ r)))
        }
        ("af_add", [(w, a), (_, b)]) => {
            let r = w.truncate(a.wrapping_add(*b));
            Some(((a ^ b ^ r) >> 4) & 1)
        }
        ("cf_sub", [(w, a), (_, b)]) => Some(flag(w.truncate(*a) < w.truncate(*b))),
        ("of_sub", [(w, a), (_, b)]) => {
            let r = w.truncate(a.wrapping_sub(*b));
            Some(sign_bit(*w, (a ^ b) & (a ^ r)))
        }
        ("af_sub", [(w, a), (_, b)]) => {
            let r = w.truncate(a.wrapping_sub(*b));
            Some(((a ^ b ^ r) >> 4) & 1)
        }
        ("cf_adc", [(w, a), (_, b), (_, c)]) => {
            let sum = u128::from(w.truncate(*a)) + u128::from(w.truncate(*b)) + u128::from(*c & 1);
            Some(flag((sum >> w.bits()) & 1 == 1))
        }
        ("of_adc", [(w, a), (_, b), (_, c)]) => {
            let r = w.truncate(a.wrapping_add(*b).wrapping_add(*c & 1));
            Some(sign_bit(*w, (a ^ r) & (b ^ r)))
        }
        ("af_adc", [(w, a), (_, b), (_, c)]) => {
            let r = w.truncate(a.wrapping_add(*b).wrapping_add(*c & 1));
            Some(((a ^ b ^ r) >> 4) & 1)
        }
        ("cf_sbb", [(w, a), (_, b), (_, c)]) => Some(flag(
            u128::from(w.truncate(*a)) < u128::from(w.truncate(*b)) + u128::from(*c & 1),
        )),
        ("of_sbb", [(w, a), (_, b), (_, c)]) => {
            let r = w.truncate(a.wrapping_sub(*b).wrapping_sub(*c & 1));
            Some(sign_bit(*w, (a ^ b) & (a ^ r)))
        }
        ("af_sbb", [(w, a), (_, b), (_, c)]) => {
            let r = w.truncate(a.wrapping_sub(*b).wrapping_sub(*c & 1));
            Some(((a ^ b ^ r) >> 4) & 1)
        }
        ("mulhi_u", [(w, a), (_, b)]) => {
            let p = u128::from(w.truncate(*a)) * u128::from(w.truncate(*b));
            Some((p >> w.bits()) as u64)
        }
        ("mulhi_s", [(w, a), (_, b)]) => {
            let p = i128::from(w.sign_extend(*a).cast_signed())
                * i128::from(w.sign_extend(*b).cast_signed());
            Some((p >> w.bits()) as u64)
        }
        ("udiv_q" | "udiv_r" | "sdiv_q" | "sdiv_r", [(w, hi), (_, lo), (_, d)]) => {
            eval_div(name, *w, *hi, *lo, *d)
        }
        _ => None,
    }
}

/// `hi:lo / d` at width `w`. `None` where hardware raises #DE (zero divisor or
/// quotient overflow), so the value is simply undefined there.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn eval_div(name: &str, w: Width, hi: u64, lo: u64, d: u64) -> Option<u64> {
    let bits = w.bits();
    let lo = w.truncate(lo);
    if name.starts_with('u') {
        let (hi, d) = (w.truncate(hi), w.truncate(d));
        let dividend = (u128::from(hi) << bits) | u128::from(lo);
        let q = dividend.checked_div(u128::from(d))?;
        let r = dividend.checked_rem(u128::from(d))?;
        if q > u128::from(w.mask()) {
            return None;
        }
        return Some(if name.ends_with('q') {
            q as u64
        } else {
            r as u64
        });
    }
    let hi = i128::from(w.sign_extend(hi).cast_signed());
    let d = i128::from(w.sign_extend(d).cast_signed());
    let dividend = (hi << bits) | i128::from(lo);
    let q = dividend.checked_div(d)?;
    let r = dividend.checked_rem(d)?;
    let half = 1i128 << (bits - 1);
    if q < -half || q >= half {
        return None;
    }
    Some(if name.ends_with('q') {
        q as u64
    } else {
        r as u64
    })
}

/// Default [`Algebra`] with constant folding and a few identity rewrites.
#[derive(Clone, Copy, Debug, Default)]
pub struct NativeAlgebra;

impl Algebra for NativeAlgebra {
    type Expr = NativeExpr;

    fn constant(&self, width: Width, value: u64) -> NativeExpr {
        NativeExpr::new(NativeNode::Const {
            width,
            value: width.truncate(value),
        })
    }

    fn input(&self, width: Width, name: &str) -> NativeExpr {
        NativeExpr::new(NativeNode::Input {
            width,
            name: name.to_owned(),
        })
    }

    fn load(&self, width: Width, address: NativeExpr) -> NativeExpr {
        NativeExpr::new(NativeNode::Load { width, address })
    }

    fn unary(&self, op: UnOp, value: NativeExpr) -> Result<NativeExpr, AlgebraError> {
        if let Some(c) = value.as_const() {
            return Ok(self.constant(value.width(), eval_unary(op, value.width(), c)));
        }
        Ok(NativeExpr::new(NativeNode::Unary { op, value }))
    }

    fn binary(
        &self,
        op: BinOp,
        left: NativeExpr,
        right: NativeExpr,
    ) -> Result<NativeExpr, AlgebraError> {
        let width = left.width();
        if width != right.width() {
            return Err(AlgebraError(format!(
                "width mismatch {:?} vs {:?}",
                width,
                right.width()
            )));
        }
        if let (Some(l), Some(r)) = (left.as_const(), right.as_const()) {
            return Ok(self.constant(width, eval_binary(op, width, l, r)));
        }
        let zero = |e: &NativeExpr| e.as_const() == Some(0);
        let ones = |e: &NativeExpr| e.as_const() == Some(width.mask());
        match op {
            BinOp::Add | BinOp::Or | BinOp::Xor if zero(&right) => return Ok(left),
            BinOp::Add | BinOp::Or | BinOp::Xor if zero(&left) => return Ok(right),
            BinOp::Sub | BinOp::Shl | BinOp::Shr | BinOp::Sar | BinOp::Rol | BinOp::Ror
                if zero(&right) =>
            {
                return Ok(left);
            }
            BinOp::And if zero(&left) || zero(&right) => return Ok(self.constant(width, 0)),
            BinOp::And if ones(&right) => return Ok(left),
            BinOp::And if ones(&left) => return Ok(right),
            BinOp::Mul if right.as_const() == Some(1) => return Ok(left),
            BinOp::Mul if left.as_const() == Some(1) => return Ok(right),
            _ => {}
        }
        Ok(NativeExpr::new(NativeNode::Binary { op, left, right }))
    }

    fn resize(
        &self,
        kind: Resize,
        target: Width,
        value: NativeExpr,
    ) -> Result<NativeExpr, AlgebraError> {
        let from = value.width();
        let valid = match kind {
            Resize::Truncate => target.bits() <= from.bits(),
            Resize::ZeroExtend | Resize::SignExtend => target.bits() >= from.bits(),
        };
        if !valid {
            return Err(AlgebraError(format!(
                "invalid {kind:?} {from:?} -> {target:?}"
            )));
        }
        if from == target {
            return Ok(value);
        }
        if let Some(c) = value.as_const() {
            return Ok(self.constant(target, eval_resize(kind, from, target, c)));
        }
        if let NativeNode::Resize {
            kind: inner_kind,
            value: inner,
            ..
        } = value.node()
        {
            let inner_width = inner.width();
            match (kind, *inner_kind) {
                (Resize::Truncate, Resize::Truncate)
                | (Resize::ZeroExtend, Resize::ZeroExtend)
                | (Resize::SignExtend, Resize::SignExtend) => {
                    return self.resize(kind, target, inner.clone());
                }
                (Resize::Truncate, Resize::ZeroExtend | Resize::SignExtend)
                    if target.bits() <= inner_width.bits() =>
                {
                    return self.resize(Resize::Truncate, target, inner.clone());
                }
                _ => {}
            }
        }
        Ok(NativeExpr::new(NativeNode::Resize {
            kind,
            width: target,
            value,
        }))
    }

    fn intrinsic(&self, width: Width, name: &str, args: Vec<NativeExpr>) -> NativeExpr {
        NativeExpr::new(NativeNode::Intrinsic {
            width,
            name: name.to_owned(),
            args,
        })
    }

    fn width(&self, expr: &NativeExpr) -> Width {
        expr.width()
    }

    fn as_const(&self, expr: &NativeExpr) -> Option<u64> {
        expr.as_const()
    }
}
