//! Dead-term elimination over `Expr`: removes sub-terms that provably cannot
//! affect the value (absorbed, cancelled, or masked away), bottom-up to a fixpoint.

use crate::Problem;
use vmpx_vm::{BinaryOp, Expr, ExprKind, ExprRef};

/// Returns a strictly smaller, value-equivalent expression, or `None` if no
/// dead term was found. Every rule is an exact identity for all inputs.
#[must_use]
pub fn eliminate(problem: &Problem) -> Option<Expr> {
    let mut current = problem.expr.clone();
    loop {
        let next = reduce(&current);
        if next == current {
            break;
        }
        current = next;
    }
    (size(&current) < size(&problem.expr)).then_some(current)
}

fn size(e: &Expr) -> usize {
    1 + match &e.kind {
        ExprKind::Const { .. } | ExprKind::Register { .. } => 0,
        ExprKind::Unary { value, .. }
        | ExprKind::Extract { value, .. }
        | ExprKind::Resize { value, .. } => size(value),
        ExprKind::Load { address } => size(address),
        ExprKind::Binary { left, right, .. }
        | ExprKind::Concat {
            high: left,
            low: right,
        } => size(left) + size(right),
        ExprKind::Intrinsic { arguments, .. } => arguments.iter().map(size).sum(),
    }
}

fn konst(e: &Expr) -> Option<u64> {
    match e.kind {
        ExprKind::Const { value } => Some(value),
        _ => None,
    }
}

/// Dropping a term that contains an opaque intrinsic could drop a side effect.
fn droppable(e: &Expr) -> bool {
    match &e.kind {
        ExprKind::Intrinsic { .. } => false,
        ExprKind::Const { .. } | ExprKind::Register { .. } => true,
        ExprKind::Unary { value, .. }
        | ExprKind::Extract { value, .. }
        | ExprKind::Resize { value, .. } => droppable(value),
        ExprKind::Load { address } => droppable(address),
        ExprKind::Binary { left, right, .. }
        | ExprKind::Concat {
            high: left,
            low: right,
        } => droppable(left) && droppable(right),
    }
}

fn bin(width: vmpx_vm::BitWidth, op: BinaryOp, l: Expr, r: Expr) -> Expr {
    Expr {
        width,
        kind: ExprKind::Binary {
            op,
            left: ExprRef::new(l),
            right: ExprRef::new(r),
        },
    }
}

fn reduce(e: &Expr) -> Expr {
    let w = e.width;
    match &e.kind {
        ExprKind::Const { .. } | ExprKind::Register { .. } => e.clone(),
        ExprKind::Load { address } => Expr {
            width: w,
            kind: ExprKind::Load {
                address: ExprRef::new(reduce(address)),
            },
        },
        ExprKind::Intrinsic { name, arguments } => Expr {
            width: w,
            kind: ExprKind::Intrinsic {
                name: name.clone(),
                arguments: arguments.iter().map(reduce).collect(),
            },
        },
        ExprKind::Extract { value, lsb } => Expr {
            width: w,
            kind: ExprKind::Extract {
                value: ExprRef::new(reduce(value)),
                lsb: *lsb,
            },
        },
        ExprKind::Concat { high, low } => Expr {
            width: w,
            kind: ExprKind::Concat {
                high: ExprRef::new(reduce(high)),
                low: ExprRef::new(reduce(low)),
            },
        },
        ExprKind::Resize { op, value } => Expr {
            width: w,
            kind: ExprKind::Resize {
                op: *op,
                value: ExprRef::new(reduce(value)),
            },
        },
        ExprKind::Unary { op, value } => {
            let v = reduce(value);
            // not(not x) == x, neg(neg x) == x.
            if let ExprKind::Unary {
                op: inner,
                value: x,
            } = &v.kind
                && inner == op
            {
                return (**x).clone();
            }
            Expr {
                width: w,
                kind: ExprKind::Unary {
                    op: *op,
                    value: ExprRef::new(v),
                },
            }
        }
        ExprKind::Binary { op, left, right } => reduce_binary(w, *op, reduce(left), reduce(right)),
    }
}

fn reduce_binary(w: vmpx_vm::BitWidth, op: BinaryOp, l: Expr, r: Expr) -> Expr {
    use BinaryOp::{Add, And, Mul, Or, Rol, Ror, Sar, Shl, Shr, Sub, Xor};
    let mask = w.mask();
    let commutative = matches!(op, Add | Mul | Xor | And | Or);

    // Normalise to (other, constant) for commutative ops.
    let (other, c) = match (konst(&l), konst(&r)) {
        (_, Some(c)) => (Some(&l), Some(c)),
        (Some(c), None) if commutative => (Some(&r), Some(c)),
        _ => (None, None),
    };

    if let (Some(x), Some(c)) = (other, c) {
        match op {
            And | Mul if c == 0 && droppable(x) => return Expr::constant(w, 0),
            Or if c == mask && droppable(x) => return Expr::constant(w, mask),
            Or | Xor | Add | Sub | Shl | Shr | Sar | Rol | Ror if c == 0 => return x.clone(),
            And if c == mask => return x.clone(),
            Mul if c == 1 => return x.clone(),
            And => {
                if let Some(folded) = mask_away(w, x, c) {
                    return folded;
                }
            }
            Or => {
                // (x & m1) | m == m when m1 is a subset of m.
                if let ExprKind::Binary {
                    op: And,
                    left,
                    right,
                } = &x.kind
                    && droppable(x)
                    && let Some(m1) = konst(right).or_else(|| konst(left))
                    && m1 & !c == 0
                {
                    return Expr::constant(w, c);
                }
            }
            _ => {}
        }
    }

    if l == r {
        match op {
            Xor | Sub if droppable(&l) => return Expr::constant(w, 0),
            And | Or => return l,
            _ => {}
        }
    }
    bin(w, op, l, r)
}

/// `x & m` where `x` is itself `inner op m1` and the inner constant provably
/// cannot change any bit that `m` keeps.
fn mask_away(w: vmpx_vm::BitWidth, x: &Expr, m: u64) -> Option<Expr> {
    let ExprKind::Binary { op, left, right } = &x.kind else {
        return None;
    };
    let (core, m1) = match (konst(left), konst(right)) {
        (None, Some(m1)) => (&**left, m1),
        (Some(m1), None) => (&**right, m1),
        _ => return None,
    };
    let keep = |new_mask: u64| {
        let masked = bin(w, BinaryOp::And, core.clone(), Expr::constant(w, new_mask));
        // Fold a zero mask immediately so the result is a constant.
        if new_mask == 0 && droppable(core) {
            Expr::constant(w, 0)
        } else {
            masked
        }
    };
    match op {
        // (x & m1) & m == x & (m1 & m)
        BinaryOp::And => Some(keep(m1 & m)),
        // (x | m1) & m == x & m and (x ^ m1) & m == x & m when m1 & m == 0.
        BinaryOp::Or | BinaryOp::Xor if m1 & m == 0 => Some(keep(m)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Ctx, ProblemClass};
    use std::collections::BTreeMap;
    use vmpx_vm::{BitWidth, UnaryOp};

    const W: BitWidth = BitWidth::W64;

    fn reg(n: &str) -> Expr {
        Expr::register(W, n)
    }
    fn k(v: u64) -> Expr {
        Expr::constant(W, v)
    }
    // Raw builder: `Expr::binary` would pre-simplify and hide what we test.
    fn b(op: BinaryOp, l: Expr, r: Expr) -> Expr {
        bin(W, op, l, r)
    }
    fn run(e: &Expr) -> Option<Expr> {
        eliminate(&Problem::new(
            e.clone(),
            ProblemClass::DeadCode,
            Ctx::default(),
        ))
    }

    fn eval(e: &Expr, env: &BTreeMap<String, u64>) -> u64 {
        match &e.kind {
            ExprKind::Const { value } => *value,
            ExprKind::Register { name } => env[name],
            ExprKind::Unary { op, value } => {
                let v = eval(value, env);
                match op {
                    UnaryOp::Not => !v,
                    UnaryOp::Neg => v.wrapping_neg(),
                }
            }
            ExprKind::Binary { op, left, right } => {
                let (a, c) = (eval(left, env), eval(right, env));
                match op {
                    BinaryOp::Add => a.wrapping_add(c),
                    BinaryOp::Sub => a.wrapping_sub(c),
                    BinaryOp::Mul => a.wrapping_mul(c),
                    BinaryOp::Xor => a ^ c,
                    BinaryOp::And => a & c,
                    BinaryOp::Or => a | c,
                    BinaryOp::Shl => a.wrapping_shl((c & 63) as u32),
                    _ => unreachable!(),
                }
            }
            _ => unreachable!(),
        }
    }

    fn assert_equiv(a: &Expr, b: &Expr) {
        for x in [0_u64, 1, 0xFF, 0xDEAD_BEEF, u64::MAX, 0x8000_0000_0000_0000] {
            for y in [0_u64, 7, 0xF0F0, u64::MAX - 3] {
                let env = BTreeMap::from([("x".to_string(), x), ("y".to_string(), y)]);
                assert_eq!(eval(a, &env), eval(b, &env));
            }
        }
    }

    #[test]
    fn and_zero_absorbs() {
        let e = b(BinaryOp::And, reg("x"), k(0));
        let r = run(&e).unwrap();
        assert_eq!(r, k(0));
        assert_equiv(&e, &r);
    }

    #[test]
    fn xor_self_junk_or_real() {
        let junk = b(BinaryOp::Add, reg("y"), k(5));
        let e = b(
            BinaryOp::Or,
            b(BinaryOp::Xor, junk.clone(), junk),
            b(BinaryOp::Mul, reg("x"), k(3)),
        );
        let r = run(&e).unwrap();
        assert_eq!(r, b(BinaryOp::Mul, reg("x"), k(3)));
        assert_equiv(&e, &r);
    }

    #[test]
    fn add_zero_sub_zero() {
        let e = b(BinaryOp::Sub, b(BinaryOp::Add, reg("x"), k(0)), k(0));
        let r = run(&e).unwrap();
        assert_eq!(r, reg("x"));
        assert_equiv(&e, &r);
    }

    #[test]
    fn nested_cancellation() {
        let inner = b(BinaryOp::Xor, reg("y"), reg("y"));
        let e = b(BinaryOp::Add, b(BinaryOp::Mul, reg("x"), k(7)), inner);
        let r = run(&e).unwrap();
        assert_eq!(r, b(BinaryOp::Mul, reg("x"), k(7)));
        assert_equiv(&e, &r);
    }

    #[test]
    fn masking_away_is_value_preserving() {
        // (x | 0xF0) & 0x0F == x & 0x0F ; (x & 0xFF) & 0x0F == x & 0x0F
        for inner in [BinaryOp::Or, BinaryOp::Xor] {
            let e = b(BinaryOp::And, b(inner, reg("x"), k(0xF0)), k(0x0F));
            let r = run(&e).unwrap();
            assert_eq!(r, b(BinaryOp::And, reg("x"), k(0x0F)));
            assert_equiv(&e, &r);
        }
        let e = b(BinaryOp::And, b(BinaryOp::And, reg("x"), k(0xFF)), k(0x0F));
        let r = run(&e).unwrap();
        assert_equiv(&e, &r);
        // Overlapping mask may matter: leave untouched.
        let e = b(BinaryOp::And, b(BinaryOp::Or, reg("x"), k(0x18)), k(0x0F));
        assert!(run(&e).is_none());
    }

    #[test]
    fn nothing_to_remove_is_none() {
        let e = b(BinaryOp::Add, b(BinaryOp::Mul, reg("x"), reg("y")), k(9));
        assert!(run(&e).is_none());
    }
}
