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
        | ExprKind::Concat {
            high: left,
            low: right,
        } => {
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
    let ExprKind::Unary {
        op: inner,
        value: x,
    } = &value.kind
    else {
        return None;
    };
    let one = Expr::constant(expr.width, 1);
    match (outer, inner) {
        (UnaryOp::Not, UnaryOp::Neg) => Expr::binary(BinaryOp::Sub, (**x).clone(), one).ok(),
        (UnaryOp::Neg, UnaryOp::Not) => Expr::binary(BinaryOp::Add, (**x).clone(), one).ok(),
        _ => None,
    }
}

/// Width-change chains that cancel. VMProtect's lifter emits a cast per VM
/// operand, so a single byte read arrives as `(u16)(u8)((u16)x >> 8)` and a
/// 64-bit value is rebuilt one byte at a time. Each rule below is an identity
/// about widths, not a guess:
///
/// - a resize to the width a value already has is nothing
/// - truncating back to the width a value had before it was extended is nothing
/// - two truncations, or two extensions of the same kind, are one
/// - extracting a value's whole width from bit 0 is nothing
fn collapse_casts(expr: &Expr) -> Option<Expr> {
    use vmpx_vm::ResizeOp;

    match &expr.kind {
        ExprKind::Resize { op, value } => {
            // Any resize to the width the value already has.
            if value.width == expr.width {
                return Some((**value).clone());
            }
            let ExprKind::Resize {
                op: inner_op,
                value: inner,
            } = &value.kind
            else {
                return None;
            };
            match (op, inner_op) {
                // Truncate(Extend(x)) back to x's own width is x itself. Only
                // when the target width is exactly x's: a narrower truncation
                // still discards bits, and a wider one is not this rule.
                (ResizeOp::Truncate, ResizeOp::ZeroExtend | ResizeOp::SignExtend)
                    if inner.width == expr.width =>
                {
                    Some((**inner).clone())
                }
                // Two truncations collapse to the narrower one.
                (ResizeOp::Truncate, ResizeOp::Truncate) => {
                    Expr::resize(ResizeOp::Truncate, expr.width, (**inner).clone()).ok()
                }
                // Two extensions of the SAME kind collapse to the wider one.
                // Mixing zero- and sign-extension does not: the inner one
                // decides the bits the outer then copies.
                (ResizeOp::ZeroExtend, ResizeOp::ZeroExtend) => {
                    Expr::resize(ResizeOp::ZeroExtend, expr.width, (**inner).clone()).ok()
                }
                (ResizeOp::SignExtend, ResizeOp::SignExtend) => {
                    Expr::resize(ResizeOp::SignExtend, expr.width, (**inner).clone()).ok()
                }
                _ => None,
            }
        }
        // Extracting the whole of a value, from bit 0, is the value.
        ExprKind::Extract { value, lsb: 0 } if value.width == expr.width => Some((**value).clone()),
        // A zero-extended value has nothing but zeros above its own width, so
        // shifting those bits down leaves zero. This is what makes
        // `(u16)(u8)((u16)byte >> 8)` -- a shape the lifter emits constantly --
        // collapse to a constant instead of surviving as four casts.
        ExprKind::Binary {
            op: vmpx_vm::BinaryOp::Shr,
            left,
            right,
        } => {
            let ExprKind::Const { value: amount } = right.kind else {
                return None;
            };
            let ExprKind::Resize {
                op: ResizeOp::ZeroExtend,
                value: inner,
            } = &left.kind
            else {
                return None;
            };
            (amount >= u64::from(inner.width.bits())).then(|| Expr::constant(expr.width, 0))
        }
        _ => None,
    }
}

/// One simplification attempt on a node whose children are already simplified.
/// Keeps a result only when it reads better by [`cost`].
fn attempt(expr: &Expr) -> Expr {
    // Cast collapsing is cheap and always worth trying, whatever the size.
    let expr = &settle_casts(expr);
    let size = expr.node_count_bounded(ATTEMPT_LIMIT);
    if size >= ATTEMPT_LIMIT || size < 3 {
        // Still worth the cheap structural identities on a small node.
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
        let candidate = settle_casts(&candidate);
        let candidate_cost = cost(&candidate);
        if candidate_cost < best_cost {
            best = candidate;
            best_cost = candidate_cost;
        }
    }
    best
}

/// Applies [`collapse_casts`] until it stops firing: removing one cast can
/// expose the next, as it does in `(u16)(u8)((u16)x >> 8)`.
fn settle_casts(expr: &Expr) -> Expr {
    let mut current = expr.clone();
    // Each step strictly removes a node, so the bound is only a safety net.
    for _ in 0..16 {
        match collapse_casts(&current) {
            Some(next) => current = next,
            None => break,
        }
    }
    current
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
            let mapped: Vec<Expr> = children
                .iter()
                .map(|c| memo.get(c).cloned().unwrap_or_else(|| c.clone()))
                .collect();
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
            Stmt::Break | Stmt::Continue | Stmt::Label(_) | Stmt::Goto(_) | Stmt::Comment(_) => {}
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
        let address =
            Expr::resize(vmpx_vm::ResizeOp::ZeroExtend, BitWidth::W64, folded).expect("resize");
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
    fn a_resize_to_the_width_a_value_already_has_is_removed() {
        use vmpx_vm::ResizeOp;
        let a = Expr::register(BitWidth::W32, "a");
        let same = Expr::resize(ResizeOp::ZeroExtend, BitWidth::W32, a.clone()).expect("resize");
        let mut memo = HashMap::new();
        assert_eq!(simplify(&same, &mut memo), a);
    }

    #[test]
    fn truncating_back_to_the_width_a_value_was_extended_from_is_removed() {
        use vmpx_vm::ResizeOp;
        let a = Expr::register(BitWidth::W8, "a");
        let wide = Expr::resize(ResizeOp::ZeroExtend, BitWidth::W32, a.clone()).expect("zext");
        let back = Expr::resize(ResizeOp::Truncate, BitWidth::W8, wide).expect("trunc");
        let mut memo = HashMap::new();
        assert_eq!(simplify(&back, &mut memo), a);
    }

    #[test]
    fn a_narrower_truncation_of_an_extension_is_kept() {
        // `(u8)((u32)x)` where x is 16 bits still discards eight bits, so the
        // pair is NOT a no-op and must survive.
        use vmpx_vm::ResizeOp;
        let a = Expr::register(BitWidth::W16, "a");
        let wide = Expr::resize(ResizeOp::ZeroExtend, BitWidth::W32, a.clone()).expect("zext");
        let narrow = Expr::resize(ResizeOp::Truncate, BitWidth::W8, wide).expect("trunc");
        let mut memo = HashMap::new();
        let out = simplify(&narrow, &mut memo);
        assert_ne!(out, a, "the truncation to 8 bits is not removable");
        assert_eq!(out.width, BitWidth::W8);
    }

    #[test]
    fn mixed_sign_and_zero_extension_does_not_collapse() {
        // zext(sext(x)) is not sext(x) widened: the inner sign-extension
        // decides the bits the outer copies.
        use vmpx_vm::ResizeOp;
        let a = Expr::register(BitWidth::W8, "a");
        let inner = Expr::resize(ResizeOp::SignExtend, BitWidth::W16, a).expect("sext");
        let outer = Expr::resize(ResizeOp::ZeroExtend, BitWidth::W32, inner.clone()).expect("zext");
        let mut memo = HashMap::new();
        let out = simplify(&outer, &mut memo);
        assert!(
            format!("{out:?}").contains("SignExtend"),
            "the sign-extension must survive: {out:?}"
        );
    }

    #[test]
    fn a_chain_of_casts_settles_in_one_pass() {
        use vmpx_vm::ResizeOp;
        let a = Expr::register(BitWidth::W8, "a");
        let one = Expr::resize(ResizeOp::ZeroExtend, BitWidth::W16, a.clone()).expect("1");
        let two = Expr::resize(ResizeOp::ZeroExtend, BitWidth::W32, one).expect("2");
        let three = Expr::resize(ResizeOp::Truncate, BitWidth::W8, two).expect("3");
        let mut memo = HashMap::new();
        assert_eq!(
            simplify(&three, &mut memo),
            a,
            "three casts cancel to nothing"
        );
    }

    #[test]
    fn shifting_a_zero_extended_value_past_its_own_width_is_zero() {
        use vmpx_vm::ResizeOp;
        // `(u16)byte >> 8`: the high half of a zero-extended byte is all zero.
        let byte = Expr::register(BitWidth::W8, "b");
        let wide = Expr::resize(ResizeOp::ZeroExtend, BitWidth::W16, byte).expect("zext");
        let shifted =
            Expr::binary(BinaryOp::Shr, wide, Expr::constant(BitWidth::W16, 8)).expect("shr");
        let mut memo = HashMap::new();
        assert_eq!(
            simplify(&shifted, &mut memo),
            Expr::constant(BitWidth::W16, 0)
        );
    }

    #[test]
    fn a_shift_that_stays_inside_the_value_is_kept() {
        use vmpx_vm::ResizeOp;
        let byte = Expr::register(BitWidth::W8, "b");
        let wide = Expr::resize(ResizeOp::ZeroExtend, BitWidth::W16, byte).expect("zext");
        let shifted =
            Expr::binary(BinaryOp::Shr, wide, Expr::constant(BitWidth::W16, 4)).expect("shr");
        let mut memo = HashMap::new();
        let out = simplify(&shifted, &mut memo);
        assert_ne!(
            out,
            Expr::constant(BitWidth::W16, 0),
            "a shift of 4 keeps real bits"
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
