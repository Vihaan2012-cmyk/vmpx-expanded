//! Instruction-substitution canonicalizer (pass side).
//!
//! Reverses `VMProtect`'s "mutation" of simple ops into longer equivalent shapes by
//! rewriting the lifted [`Expr`] bottom-up to a fixpoint. Every rule is a true
//! wrapping-bitvector identity and never raises the distinct operator count.

use crate::Problem;
use std::collections::{HashMap, HashSet};
use vmpx_vm::{BinaryOp, Expr, ExprKind, UnaryOp};

const MAX_PASSES: usize = 16;

/// Canonicalizes a substituted expression, or `None` if nothing got simpler.
///
/// Rules: `a+a -> a<<1`, `x*2^k -> x<<k`, `x*1 -> x`, `x*0 -> 0`, `~x+1 -> -x`,
/// `a+(-b) -> a-b`, `a-(-b) -> a+b`, `a-(-c) -> a+c` (constants), `0-x -> -x`,
/// `(x<<a)<<b -> x<<(a+b)`, plus the constructor folds (`x+0`, `x<<0`,
/// `--x`, `~~x`, `x^x`, `x-x`).
#[must_use]
pub fn canonicalize(problem: &Problem) -> Option<Expr> {
    let mut current = problem.expr.clone();
    for _ in 0..MAX_PASSES {
        let next = rewrite(&current, &mut HashMap::new());
        if next == current {
            break;
        }
        current = next;
    }
    (current != problem.expr && op_count(&current) <= op_count(&problem.expr)).then_some(current)
}

#[allow(clippy::mutable_key_type)]
fn rewrite(expr: &Expr, memo: &mut HashMap<Expr, Expr>) -> Expr {
    if let Some(done) = memo.get(expr) {
        return done.clone();
    }
    let rebuilt = match &expr.kind {
        ExprKind::Const { .. } | ExprKind::Register { .. } => expr.clone(),
        ExprKind::Unary { op, value } => Expr::unary(*op, rewrite(value, memo)),
        ExprKind::Binary { op, left, right } => {
            let node = bin(*op, rewrite(left, memo), rewrite(right, memo));
            apply_rules(node)
        }
        ExprKind::Load { address } => Expr::load(expr.width, rewrite(address, memo)),
        ExprKind::Intrinsic { name, arguments } => Expr::intrinsic(
            expr.width,
            name.clone(),
            arguments.iter().map(|a| rewrite(a, memo)).collect(),
        ),
        ExprKind::Extract { value, lsb } => {
            Expr::extract(expr.width, *lsb, rewrite(value, memo)).unwrap_or_else(|_| expr.clone())
        }
        ExprKind::Concat { high, low } => {
            Expr::concat(rewrite(high, memo), rewrite(low, memo)).unwrap_or_else(|_| expr.clone())
        }
        ExprKind::Resize { op, value } => {
            Expr::resize(*op, expr.width, rewrite(value, memo)).unwrap_or_else(|_| expr.clone())
        }
    };
    memo.insert(expr.clone(), rebuilt.clone());
    rebuilt
}

fn bin(op: BinaryOp, left: Expr, right: Expr) -> Expr {
    Expr::binary(op, left, right).expect("operands share a width")
}

fn constant(expr: &Expr) -> Option<u64> {
    match expr.kind {
        ExprKind::Const { value } => Some(value),
        _ => None,
    }
}

fn neg_operand(expr: &Expr) -> Option<&Expr> {
    match &expr.kind {
        ExprKind::Unary {
            op: UnaryOp::Neg,
            value,
        } => Some(value),
        _ => None,
    }
}

fn apply_rules(node: Expr) -> Expr {
    let ExprKind::Binary { op, left, right } = &node.kind else {
        return node;
    };
    let width = node.width;
    let (l, r) = ((**left).clone(), (**right).clone());
    match op {
        BinaryOp::Add => rewrite_add(node),
        BinaryOp::Sub => {
            if let Some(b) = neg_operand(&r) {
                return bin(BinaryOp::Add, l, b.clone());
            }
            if constant(&l) == Some(0) {
                return Expr::unary(UnaryOp::Neg, r);
            }
            match constant(&r) {
                Some(c) if c > 1 << (width.bits() - 1) => bin(
                    BinaryOp::Add,
                    l,
                    Expr::constant(width, width.truncate(c.wrapping_neg())),
                ),
                _ => node,
            }
        }
        BinaryOp::Mul => {
            let (x, c) = match (constant(&l), constant(&r)) {
                (_, Some(c)) => (l, c),
                (Some(c), None) => (r, c),
                _ => return node,
            };
            match c {
                0 => Expr::constant(width, 0),
                1 => x,
                c if c.is_power_of_two() => bin(
                    BinaryOp::Shl,
                    x,
                    Expr::constant(width, u64::from(c.trailing_zeros())),
                ),
                _ => node,
            }
        }
        BinaryOp::Shl => {
            let (
                Some(outer),
                ExprKind::Binary {
                    op: BinaryOp::Shl,
                    left: x,
                    right: inner,
                },
            ) = (constant(&r), &l.kind)
            else {
                return node;
            };
            match constant(inner) {
                Some(inner) if inner + outer < u64::from(width.bits()) => bin(
                    BinaryOp::Shl,
                    (**x).clone(),
                    Expr::constant(width, inner + outer),
                ),
                _ => node,
            }
        }
        _ => node,
    }
}

fn flatten_add(expr: &Expr, out: &mut Vec<Expr>) {
    if let ExprKind::Binary {
        op: BinaryOp::Add,
        left,
        right,
    } = &expr.kind
    {
        flatten_add(left, out);
        flatten_add(right, out);
    } else {
        out.push(expr.clone());
    }
}

fn sum(terms: Vec<Expr>) -> Option<Expr> {
    terms
        .into_iter()
        .reduce(|acc, t| bin(BinaryOp::Add, acc, t))
}

fn rewrite_add(node: Expr) -> Expr {
    let mut terms = Vec::new();
    flatten_add(&node, &mut terms);
    if terms.len() == 2 && terms[0] == terms[1] {
        let one = Expr::constant(node.width, 1);
        return bin(BinaryOp::Shl, terms.swap_remove(0), one);
    }
    // ~x + 1 == -x
    let not_at = terms.iter().position(|t| {
        matches!(
            t.kind,
            ExprKind::Unary {
                op: UnaryOp::Not,
                ..
            }
        )
    });
    let one_at = terms.iter().position(|t| constant(t) == Some(1));
    if let (Some(n), Some(o)) = (not_at, one_at) {
        let ExprKind::Unary { value, .. } = &terms[n].kind else {
            return node;
        };
        let negated = Expr::unary(UnaryOp::Neg, (**value).clone());
        let mut rest: Vec<Expr> = terms
            .into_iter()
            .enumerate()
            .filter(|(i, _)| *i != n && *i != o)
            .map(|(_, t)| t)
            .collect();
        rest.push(negated);
        return sum(rest).unwrap_or(node);
    }
    // a + (-b) == a - b
    if terms.len() >= 2
        && let Some(n) = terms.iter().position(|t| neg_operand(t).is_some())
    {
        let b = neg_operand(&terms[n])
            .cloned()
            .unwrap_or_else(|| node.clone());
        terms.remove(n);
        if let Some(rest) = sum(terms) {
            return bin(BinaryOp::Sub, rest, b);
        }
    }
    node
}

/// Distinct operator (non-leaf) nodes; shared subtrees count once.
#[allow(clippy::mutable_key_type)]
fn op_count(root: &Expr) -> usize {
    let mut seen: HashSet<&Expr> = HashSet::new();
    let mut pending = vec![root];
    while let Some(e) = pending.pop() {
        if !seen.insert(e) {
            continue;
        }
        match &e.kind {
            ExprKind::Const { .. } | ExprKind::Register { .. } => {}
            ExprKind::Unary { value, .. }
            | ExprKind::Load { address: value }
            | ExprKind::Extract { value, .. }
            | ExprKind::Resize { value, .. } => pending.push(value),
            ExprKind::Binary { left, right, .. }
            | ExprKind::Concat {
                high: left,
                low: right,
            } => {
                pending.push(left);
                pending.push(right);
            }
            ExprKind::Intrinsic { arguments, .. } => pending.extend(arguments),
        }
    }
    seen.iter()
        .filter(|e| !matches!(e.kind, ExprKind::Const { .. } | ExprKind::Register { .. }))
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BitWidth, Ctx, ProblemClass};
    use std::collections::BTreeMap;
    use vmpx_vm::ExprRef;

    const W: BitWidth = BitWidth::W32;

    fn x() -> Expr {
        Expr::register(W, "x")
    }
    fn y() -> Expr {
        Expr::register(W, "y")
    }
    fn c(v: u64) -> Expr {
        Expr::constant(W, v)
    }
    fn raw_un(op: UnaryOp, v: Expr) -> Expr {
        Expr {
            width: v.width,
            kind: ExprKind::Unary {
                op,
                value: ExprRef::new(v),
            },
        }
    }
    fn problem(expr: Expr) -> Problem {
        Problem::new(expr, ProblemClass::Substitution, Ctx::default())
    }

    fn eval(e: &Expr, env: &BTreeMap<&str, u64>) -> u64 {
        let w = e.width;
        match &e.kind {
            ExprKind::Const { value } => *value,
            ExprKind::Register { name } => w.truncate(env[name.as_str()]),
            ExprKind::Unary { op, value } => {
                let v = eval(value, env);
                w.truncate(match op {
                    UnaryOp::Not => !v,
                    UnaryOp::Neg => v.wrapping_neg(),
                })
            }
            ExprKind::Binary { op, left, right } => {
                let (a, b) = (eval(left, env), eval(right, env));
                w.truncate(match op {
                    BinaryOp::Add => a.wrapping_add(b),
                    BinaryOp::Sub => a.wrapping_sub(b),
                    BinaryOp::Mul => a.wrapping_mul(b),
                    BinaryOp::Xor => a ^ b,
                    BinaryOp::Shl => a << b,
                    _ => unreachable!("not generated"),
                })
            }
            _ => unreachable!("not generated"),
        }
    }

    fn assert_equivalent(before: &Expr, after: &Expr) {
        for (a, b) in [
            (0, 0),
            (1, 2),
            (0xFFFF_FFFF, 0x8000_0000),
            (0x1234_5678, 0x9ABC_DEF0),
        ] {
            let env = BTreeMap::from([("x", a), ("y", b)]);
            assert_eq!(eval(before, &env), eval(after, &env), "{before:?}");
        }
    }

    #[test]
    fn add_self_becomes_shift() {
        let e = bin(BinaryOp::Add, x(), x());
        let out = canonicalize(&problem(e.clone())).expect("canonicalized");
        assert_eq!(out, bin(BinaryOp::Shl, x(), c(1)));
        assert_equivalent(&e, &out);
    }

    #[test]
    fn double_negation_cancels() {
        let e = raw_un(UnaryOp::Neg, raw_un(UnaryOp::Neg, x()));
        assert_eq!(canonicalize(&problem(e)), Some(x()));
    }

    #[test]
    fn mul_power_of_two_becomes_shift() {
        let e = bin(BinaryOp::Mul, x(), c(8));
        let out = canonicalize(&problem(e.clone())).expect("canonicalized");
        assert_eq!(out, bin(BinaryOp::Shl, x(), c(3)));
        assert_equivalent(&e, &out);
    }

    #[test]
    fn not_plus_one_becomes_neg() {
        let e = bin(BinaryOp::Add, Expr::unary(UnaryOp::Not, x()), c(1));
        let out = canonicalize(&problem(e.clone())).expect("canonicalized");
        assert_eq!(out, Expr::unary(UnaryOp::Neg, x()));
        assert_equivalent(&e, &out);
    }

    #[test]
    fn composed_substitution_collapses() {
        // (x*2) - (-(~~y))  ->  (x<<1) + y
        let double_not = raw_un(UnaryOp::Not, raw_un(UnaryOp::Not, y()));
        let e = bin(
            BinaryOp::Sub,
            bin(BinaryOp::Mul, x(), c(2)),
            raw_un(UnaryOp::Neg, double_not),
        );
        let out = canonicalize(&problem(e.clone())).expect("canonicalized");
        assert_eq!(out, bin(BinaryOp::Add, bin(BinaryOp::Shl, x(), c(1)), y()));
        assert!(op_count(&out) < op_count(&e));
        assert_equivalent(&e, &out);
    }

    #[test]
    fn negative_sub_constant_becomes_add() {
        let e = bin(BinaryOp::Sub, x(), c(0xFFFF_FFFF));
        let out = canonicalize(&problem(e.clone())).expect("canonicalized");
        assert_eq!(out, bin(BinaryOp::Add, x(), c(1)));
        assert_equivalent(&e, &out);
    }

    #[test]
    fn already_canonical_is_none() {
        assert!(canonicalize(&problem(bin(BinaryOp::Shl, x(), c(3)))).is_none());
        assert!(canonicalize(&problem(bin(BinaryOp::Add, x(), y()))).is_none());
    }
}
