//! Mixed boolean-arithmetic simplification of rendered expressions.
//!
//! VMProtect rewrites plain arithmetic into MBA identities, so the recovered IR
//! is correct but unreadable: `~(~a + 1)` is `a - 1`, and
//! `(x << 8) + x` is `x * 0x101`. Collapsing those is the difference between
//! output that can be read and output that can only be trusted.
//!
//! The work is done by `vmpx_mutation`'s theorem-grade simplifiers, which are
//! already in this crate's dependencies. [`simplify_general`] never fails and
//! returns its input unchanged when nothing applies; the linear rewrites it
//! adopts internally re-parse their result and check its signature against the
//! input's, so a rewrite that does not preserve meaning is rejected rather than
//! returned.
//!
//! This module adds the part those functions do not do: applying them BOTTOM-UP
//! across a whole function, so a simplifiable subterm buried inside a larger
//! expression is reached, and memoizing by node so a shared DAG is not
//! re-simplified once per parent.
//!
//! Every rewrite is accepted only when it is strictly smaller, so the pass can
//! only ever reduce what is printed.

use std::collections::HashMap;

use vmpx_mutation::general_mba::simplify_general;
use vmpx_vm::{Expr, ExprKind};

/// Node budget for one simplification attempt. MBA simplification is
/// superlinear, and the recovered expressions are large enough that an
/// unbounded attempt dominates the run.
const ATTEMPT_LIMIT: usize = 512;

/// Rebuilds `expr` with `children` in place of its own, preserving width.
/// Returns `None` when the rebuild would be ill-typed, in which case the caller
/// keeps the original.
fn rebuild(expr: &Expr, children: &[Expr]) -> Option<Expr> {
    Some(match (&expr.kind, children) {
        (ExprKind::Const { .. } | ExprKind::Register { .. }, _) => expr.clone(),
        (ExprKind::Unary { op, .. }, [value]) => Expr::unary(*op, value.clone()),
        (ExprKind::Load { .. }, [address]) => Expr::load(expr.width, address.clone()),
        (ExprKind::Binary { op, .. }, [left, right]) => {
            Expr::binary(*op, left.clone(), right.clone()).ok()?
        }
        (ExprKind::Concat { .. }, [high, low]) => Expr::concat(high.clone(), low.clone()).ok()?,
        (ExprKind::Extract { lsb, .. }, [value]) => {
            Expr::extract(expr.width, *lsb, value.clone()).ok()?
        }
        (ExprKind::Resize { op, .. }, [value]) => {
            Expr::resize(*op, expr.width, value.clone()).ok()?
        }
        (ExprKind::Intrinsic { name, .. }, args) => {
            Expr::intrinsic(expr.width, name.clone(), args.to_vec())
        }
        _ => return None,
    })
}

fn children_of(expr: &Expr) -> Vec<Expr> {
    match &expr.kind {
        ExprKind::Const { .. } | ExprKind::Register { .. } => Vec::new(),
        ExprKind::Load { address: value }
        | ExprKind::Unary { value, .. }
        | ExprKind::Extract { value, .. }
        | ExprKind::Resize { value, .. } => vec![(**value).clone()],
        ExprKind::Binary { left, right, .. }
        | ExprKind::Concat { high: left, low: right } => {
            vec![(**left).clone(), (**right).clone()]
        }
        ExprKind::Intrinsic { arguments, .. } => arguments.clone(),
    }
}

/// How hard an expression is to READ, which is not the same as how many nodes
/// it has. `~(-a)` and `a - 1` are both three nodes, but one of them is the
/// thing the programmer wrote. Each bit-twiddling operator costs an extra
/// point, so a rewrite that trades `not`/`neg`/`and`/`or`/`xor` for plain
/// arithmetic is an improvement even at equal size.
fn cost(expr: &Expr) -> usize {
    let mut total = 0usize;
    let mut pending = vec![expr.clone()];
    let mut seen = std::collections::HashSet::new();
    while let Some(current) = pending.pop() {
        if total > ATTEMPT_LIMIT * 4 {
            break;
        }
        total += 1;
        match &current.kind {
            ExprKind::Unary { op, .. } => {
                total += usize::from(matches!(op, vmpx_vm::UnaryOp::Not | vmpx_vm::UnaryOp::Neg));
            }
            ExprKind::Binary { op, .. } => {
                total += usize::from(matches!(
                    op,
                    vmpx_vm::BinaryOp::And | vmpx_vm::BinaryOp::Or | vmpx_vm::BinaryOp::Xor
                ));
            }
            _ => {}
        }
        for child in children_of(&current) {
            if seen.insert(child.clone()) {
                pending.push(child);
            }
        }
    }
    total
}

/// Two's-complement identities the MBA simplifier does not apply, because they
/// do not reduce node count -- only unreadability. Both follow from
/// `~x == -x - 1`:
///
/// - `~(-x)` = `-(-x) - 1` = `x - 1`
/// - `-(~x)` = `-(-x - 1)` = `x + 1`
fn twos_complement(expr: &Expr) -> Option<Expr> {
    use vmpx_vm::{BinaryOp, UnaryOp};
    let ExprKind::Unary { op: outer, value } = &expr.kind else {
        return None;
    };
    let ExprKind::Unary { op: inner, value: x } = &value.kind else {
        return None;
    };
    let one = Expr::constant(expr.width, 1);
    match (outer, inner) {
        (UnaryOp::Not, UnaryOp::Neg) => Expr::binary(BinaryOp::Sub, (**x).clone(), one).ok(),
        (UnaryOp::Neg, UnaryOp::Not) => Expr::binary(BinaryOp::Add, (**x).clone(), one).ok(),
        _ => None,
    }
}

/// One simplification attempt on a node whose children are already simplified.
/// Keeps a result only when it reads better by [`cost`].
fn attempt(expr: &Expr) -> Expr {
    let size = expr.node_count_bounded(ATTEMPT_LIMIT);
    if size >= ATTEMPT_LIMIT || size < 3 {
        // Still worth the cheap structural identity on a small node.
        return twos_complement(expr).unwrap_or_else(|| expr.clone());
    }
    let mut best = expr.clone();
    let mut best_cost = cost(&best);
    for candidate in [Some(simplify_general(expr).expr), twos_complement(expr)]
        .into_iter()
        .flatten()
    {
        // Identities can expose each other: `~(~a + 1)` becomes `~(-a)` first,
        // and only then `a - 1`.
        let candidate = twos_complement(&candidate).unwrap_or(candidate);
        let candidate_cost = cost(&candidate);
        if candidate_cost < best_cost {
            best = candidate;
            best_cost = candidate_cost;
        }
    }
    best
}

/// Simplifies `root` bottom-up, memoizing per node so a shared subtree is
/// simplified once however many parents reference it.
///
/// `memo` is shared across a whole function, which is what keeps the pass
/// linear in DISTINCT nodes rather than in rendered size.
pub fn simplify(root: &Expr, memo: &mut HashMap<Expr, Expr>) -> Expr {
    if let Some(done) = memo.get(root) {
        return done.clone();
    }
    // Explicit post-order: these expressions are deep enough to overflow the
    // stack if this recursed.
    let mut pending: Vec<(Expr, bool)> = vec![(root.clone(), false)];
    while let Some((expr, expanded)) = pending.pop() {
        if memo.contains_key(&expr) {
            continue;
        }
        let children = children_of(&expr);
        if expanded {
            let mapped: Vec<Expr> =
                children.iter().map(|c| memo.get(c).cloned().unwrap_or_else(|| c.clone())).collect();
            let rebuilt = rebuild(&expr, &mapped).unwrap_or_else(|| expr.clone());
            memo.insert(expr, attempt(&rebuilt));
        } else {
            pending.push((expr, true));
            for child in children {
                if !memo.contains_key(&child) {
                    pending.push((child, false));
                }
            }
        }
    }
    memo.get(root).cloned().unwrap_or_else(|| root.clone())
}

/// Applies [`simplify`] to every expression a statement list holds, in place.
///
/// Both the emitted C text and the pipeline's own cross-check are built from
/// these statements, and that check replays the ORIGINAL unoptimized function
/// against them over 48 input vectors -- so every rewrite this makes is
/// validated there rather than taken on trust.
pub fn simplify_stmts(stmts: &mut [crate::opt::Stmt], memo: &mut HashMap<Expr, Expr>) {
    use crate::ir::{Cond, Store};
    use crate::opt::Stmt;

    fn cond(c: &mut Cond, memo: &mut HashMap<Expr, Expr>) {
        match c {
            Cond::Cmp { a, b, .. } => {
                *a = simplify(a, memo);
                *b = simplify(b, memo);
            }
            Cond::Opaque { args, .. } => {
                for arg in args.iter_mut() {
                    *arg = simplify(arg, memo);
                }
            }
        }
    }

    for stmt in stmts {
        match stmt {
            Stmt::Assign(_, e) => *e = simplify(e, memo),
            Stmt::Store(Store { address, value }) => {
                *address = simplify(address, memo);
                *value = simplify(value, memo);
            }
            Stmt::Return(values) => {
                for (_, e) in values.iter_mut() {
                    *e = simplify(e, memo);
                }
            }
            Stmt::If { cond: c, then, els } => {
                cond(c, memo);
                simplify_stmts(then, memo);
                simplify_stmts(els, memo);
            }
            Stmt::While { cond: c, body } | Stmt::DoWhile { body, cond: c } => {
                cond(c, memo);
                simplify_stmts(body, memo);
            }
            Stmt::IfGoto { cond: c, .. } => cond(c, memo),
            Stmt::Loop(body) => simplify_stmts(body, memo),
            Stmt::Break
            | Stmt::Continue
            | Stmt::Label(_)
            | Stmt::Goto(_)
            | Stmt::Comment(_) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use vmpx_vm::{BinaryOp, BitWidth, UnaryOp};

    use super::*;

    fn reg(name: &str) -> Expr {
        Expr::register(BitWidth::W32, name)
    }

    fn k(v: u64) -> Expr {
        Expr::constant(BitWidth::W32, v)
    }

    fn bin(op: BinaryOp, a: Expr, b: Expr) -> Expr {
        Expr::binary(op, a, b).expect("binary")
    }

    #[test]
    fn the_two_s_complement_identity_collapses() {
        // `~(~a + 1)` is `a - 1`: the shape that litters the recovered output.
        let a = reg("a");
        let inner = Expr::unary(UnaryOp::Not, a.clone());
        let expr = Expr::unary(UnaryOp::Not, bin(BinaryOp::Add, inner, k(1)));
        let mut memo = HashMap::new();
        let out = simplify(&expr, &mut memo);
        let expected = bin(BinaryOp::Sub, a, k(1));
        assert_eq!(out, expected, "got {out:?}");
    }

    #[test]
    fn simplification_reaches_a_subterm_buried_in_a_bigger_expression() {
        // The identity is nested under a Load, which `simplify_general` alone
        // would not look through. Bottom-up application is the point.
        let a = reg("a");
        let inner = Expr::unary(UnaryOp::Not, a.clone());
        let folded = Expr::unary(UnaryOp::Not, bin(BinaryOp::Add, inner, k(1)));
        let address = Expr::resize(vmpx_vm::ResizeOp::ZeroExtend, BitWidth::W64, folded)
            .expect("resize");
        let expr = Expr::load(BitWidth::W64, address);
        let mut memo = HashMap::new();
        let out = simplify(&expr, &mut memo);
        let rendered = format!("{out:?}");
        assert!(
            !rendered.contains("Not"),
            "the nested identity should be gone: {rendered}"
        );
    }

    #[test]
    fn a_node_that_does_not_shrink_is_returned_unchanged() {
        let expr = bin(BinaryOp::Add, reg("a"), reg("b"));
        let mut memo = HashMap::new();
        assert_eq!(simplify(&expr, &mut memo), expr);
    }

    #[test]
    fn leaves_survive_and_the_memo_is_reused_across_roots() {
        let a = reg("a");
        let mut memo = HashMap::new();
        assert_eq!(simplify(&a, &mut memo), a);
        let before = memo.len();
        assert_eq!(simplify(&a, &mut memo), a);
        assert_eq!(memo.len(), before, "a second call must hit the memo");
    }
}
