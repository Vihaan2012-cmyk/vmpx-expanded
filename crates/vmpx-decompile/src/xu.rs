//! Expression utilities: generic rebuild/rewrite, width cleanups, rotate folding.

use std::collections::HashMap;

use vmpx_vm::{BinaryOp, BitWidth, Expr, ExprKind, ResizeOp};

pub fn w(bits: u32) -> BitWidth {
    BitWidth::from_bits(bits).expect("width")
}

/// Rebuilds `e` with `f` applied to every direct child (children first rebuilt).
pub fn map_children(e: &Expr, f: &mut dyn FnMut(&Expr) -> Expr) -> Expr {
    match &e.kind {
        ExprKind::Const { .. } | ExprKind::Register { .. } => e.clone(),
        ExprKind::Unary { op, value } => Expr::unary(*op, f(value)),
        ExprKind::Binary { op, left, right } => {
            let (l, r) = (f(left), f(right));
            Expr::binary(*op, l, r).unwrap_or_else(|_| e.clone())
        }
        ExprKind::Extract { value, lsb } => {
            Expr::extract(e.width, *lsb, f(value)).unwrap_or_else(|_| e.clone())
        }
        ExprKind::Concat { high, low } => {
            let (h, l) = (f(high), f(low));
            Expr::concat(h, l).unwrap_or_else(|_| e.clone())
        }
        ExprKind::Resize { op, value } => {
            Expr::resize(*op, e.width, f(value)).unwrap_or_else(|_| e.clone())
        }
        ExprKind::Intrinsic { name, arguments } => {
            Expr::intrinsic(e.width, name.clone(), arguments.iter().map(&mut *f).collect())
        }
        ExprKind::Load { address } => Expr::load(e.width, f(address)),
    }
}

/// Top-down memoized rewrite: if `f` returns `Some`, that replaces the node
/// (not descended into); otherwise children are rewritten.
pub fn rewrite(e: &Expr, f: &mut dyn FnMut(&Expr) -> Option<Expr>) -> Expr {
    fn go(
        e: &Expr,
        f: &mut dyn FnMut(&Expr) -> Option<Expr>,
        memo: &mut HashMap<Expr, Expr>,
    ) -> Expr {
        if let Some(d) = memo.get(e) {
            return d.clone();
        }
        let out = if let Some(r) = f(e) {
            r
        } else {
            map_children(e, &mut |c| go(c, f, memo))
        };
        memo.insert(e.clone(), out.clone());
        out
    }
    go(e, f, &mut HashMap::new())
}

/// Bottom-up memoized rewrite: children are rewritten first, then `f` sees the rebuilt node.
pub fn rewrite_up(e: &Expr, f: &mut dyn FnMut(&Expr) -> Option<Expr>) -> Expr {
    fn go(
        e: &Expr,
        f: &mut dyn FnMut(&Expr) -> Option<Expr>,
        memo: &mut HashMap<Expr, Expr>,
    ) -> Expr {
        if let Some(d) = memo.get(e) {
            return d.clone();
        }
        let rebuilt = map_children(e, &mut |c| go(c, f, memo));
        let out = f(&rebuilt).unwrap_or(rebuilt);
        memo.insert(e.clone(), out.clone());
        out
    }
    go(e, f, &mut HashMap::new())
}

/// `Some(inner)` when `e` is `Truncate`/`Extract@0` of `inner` (or `inner` itself at equal width).
pub fn as_trunc(e: &Expr) -> Option<&Expr> {
    match &e.kind {
        ExprKind::Resize { op: ResizeOp::Truncate, value } => Some(value),
        ExprKind::Extract { value, lsb: 0 } => Some(value),
        _ => None,
    }
}

pub fn trunc_to(width: BitWidth, e: &Expr) -> Expr {
    if e.width == width {
        e.clone()
    } else {
        Expr::resize(ResizeOp::Truncate, width, e.clone()).unwrap_or_else(|_| e.clone())
    }
}

/// Strips operations that cannot change the low bits a shift/rotate of width
/// `w` actually reads (a mask covering them, a zero-extension of a byte-or-wider
/// truncation of a `w`-wide value).
pub fn peel_count(c: &Expr, w: BitWidth, rotate: bool) -> Option<Expr> {
    let need: u64 = if rotate { u64::from(w.bits() - 1) } else if w == BitWidth::W64 { 63 } else { 31 };
    Some(fit(w, lowbits(c, need)))
}

fn fit(w: BitWidth, r: Expr) -> Expr {
    if r.width == w {
        r
    } else if r.width.bits() > w.bits() {
        trunc_to(w, &r)
    } else {
        Expr::resize(ResizeOp::ZeroExtend, w, r.clone()).unwrap_or(r)
    }
}

/// An expression of width `w` that agrees with `c` on the bits in `need`
/// (a mask of low bits), with masks / extensions / truncations that cannot
/// affect those bits removed.
fn lowbits(c: &Expr, need: u64) -> Expr {
    match &c.kind {
        ExprKind::Binary { op: BinaryOp::And, left, right } => match (&left.kind, &right.kind) {
            (_, ExprKind::Const { value }) if value & need == need => lowbits(left, need),
            (ExprKind::Const { value }, _) if value & need == need => lowbits(right, need),
            _ => c.clone(),
        },
        ExprKind::Resize { op: ResizeOp::ZeroExtend, value } => lowbits(value, need),
        ExprKind::Resize { op: ResizeOp::Truncate, value } | ExprKind::Extract { value, lsb: 0 }
            if (need + 1).is_power_of_two() && c.width.bits() >= 64 - need.leading_zeros() =>
        {
            lowbits(value, need)
        }
        _ => c.clone(),
    }
}


fn local_rules(e: &Expr) -> Option<Expr> {
    // Trunc(ZExt(x)) / Trunc(Trunc(x)) / Trunc(SExt(x)) / ZExt(Trunc(Const))
    if let Some(inner) = as_trunc(e) {
        match &inner.kind {
            ExprKind::Resize { op: ResizeOp::ZeroExtend | ResizeOp::SignExtend, value }
                if value.width.bits() >= e.width.bits() =>
            {
                return Some(trunc_to(e.width, value));
            }
            ExprKind::Resize { op: ResizeOp::ZeroExtend | ResizeOp::SignExtend, value }
                if value.width == e.width =>
            {
                return Some((**value).clone());
            }
            _ => {}
        }
        if let Some(inner2) = as_trunc(inner) {
            return Some(trunc_to(e.width, inner2));
        }
        // Truncate of a low-bits-only operation distributes: Trunc(a op b) = Trunc(a) op Trunc(b)
        if let ExprKind::Binary { op, left, right } = &inner.kind
            && matches!(
                op,
                BinaryOp::Add
                    | BinaryOp::Sub
                    | BinaryOp::Mul
                    | BinaryOp::Xor
                    | BinaryOp::And
                    | BinaryOp::Or
            )
        {
            let l = trunc_to(e.width, left);
            let r = trunc_to(e.width, right);
            return Expr::binary(*op, l, r).ok();
        }
    }
    // identities the core simplifier leaves behind
    if let ExprKind::Binary { op, left, right } = &e.kind {
        for (a, b) in [(left, right), (right, left)] {
            if let ExprKind::Const { value } = a.kind {
                let ident = match op {
                    BinaryOp::Or | BinaryOp::Xor | BinaryOp::Add => value == 0,
                    BinaryOp::And => value == e.width.mask(),
                    BinaryOp::Mul => value == 1,
                    _ => false,
                };
                if ident {
                    return Some((**b).clone());
                }
            }
        }
    }
    // Shift/rotate counts only use their low bits: peel masks and byte truncations.
    if let ExprKind::Binary { op, left, right } = &e.kind
        && matches!(op, BinaryOp::Shl | BinaryOp::Shr | BinaryOp::Sar | BinaryOp::Rol | BinaryOp::Ror)
        && let Some(peeled) = peel_count(right, e.width, matches!(op, BinaryOp::Rol | BinaryOp::Ror))
        && peeled != **right
    {
        return Expr::binary(*op, (**left).clone(), peeled).ok();
    }
    // And(ZExt(x), c) with c < 2^|x|  ->  ZExt(And(x, c))
    if let ExprKind::Binary { op: BinaryOp::And, left, right } = &e.kind {
        for (a, b) in [(left, right), (right, left)] {
            if let (ExprKind::Resize { op: ResizeOp::ZeroExtend, value }, ExprKind::Const { value: c }) =
                (&a.kind, &b.kind)
                && *c <= value.width.mask()
            {
                let narrow_c = Expr::constant(value.width, *c);
                if let Ok(inner) = Expr::binary(BinaryOp::And, (**value).clone(), narrow_c) {
                    return Expr::resize(ResizeOp::ZeroExtend, e.width, inner).ok();
                }
            }
        }
    }
    // Or(Shl(x,a), Shr(x,b)) with a+b == width -> Rol(x,a)
    if let ExprKind::Binary { op: BinaryOp::Or, left, right } = &e.kind {
        for (l, r) in [(left, right), (right, left)] {
            if let (
                ExprKind::Binary { op: BinaryOp::Shl, left: x1, right: a },
                ExprKind::Binary { op: BinaryOp::Shr, left: x2, right: b },
            ) = (&l.kind, &r.kind)
                && x1 == x2
                && let (ExprKind::Const { value: ca }, ExprKind::Const { value: cb }) =
                    (&a.kind, &b.kind)
                && ca + cb == u64::from(e.width.bits())
                && *ca > 0
                && *cb > 0
            {
                let count = Expr::constant(e.width, *ca);
                return Expr::binary(BinaryOp::Rol, (**x1).clone(), count).ok();
            }
        }
    }
    None
}

/// Width/rotate cleanups, bottom-up, to a fixpoint (bounded).
pub fn cleanup(e: &Expr) -> Expr {
    let mut cur = e.simplify();
    for _ in 0..6 {
        let next = rewrite_up(&cur, &mut |n| local_rules(n)).simplify();
        if next == cur {
            break;
        }
        cur = next;
    }
    cur
}

pub fn is_leaf(e: &Expr) -> bool {
    matches!(e.kind, ExprKind::Const { .. } | ExprKind::Register { .. })
}

/// Number of distinct non-leaf nodes (bounded).
pub fn size(e: &Expr, limit: usize) -> usize {
    e.node_count_bounded(limit)
}
