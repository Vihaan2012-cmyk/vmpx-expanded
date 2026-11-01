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
    // Cat(0, x) is a zero-extension
    if let ExprKind::Concat { high, low } = &e.kind
        && matches!(high.kind, ExprKind::Const { value: 0 })
    {
        return Expr::resize(ResizeOp::ZeroExtend, e.width, (**low).clone()).ok();
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

// ------------------------------------------------------------------- MBA folding

#[derive(Clone, Debug)]
enum B {
    Atom(usize),
    Const(bool),
    Not(Box<B>),
    Bin(BinaryOp, Box<B>, Box<B>),
}

fn b_ops(b: &B) -> usize {
    match b {
        B::Atom(_) | B::Const(_) => 0,
        B::Not(x) => 1 + b_ops(x),
        B::Bin(_, l, r) => 1 + b_ops(l) + b_ops(r),
    }
}

fn b_collect(e: &Expr, atoms: &mut Vec<Expr>, budget: &mut usize, depth: usize) -> B {
    if *budget == 0 || depth == 0 {
        return b_atom(e, atoms);
    }
    *budget -= 1;
    match &e.kind {
        ExprKind::Unary { op: vmpx_vm::UnaryOp::Not, value } => B::Not(Box::new(b_collect(value, atoms, budget, depth - 1))),
        ExprKind::Binary { op: op @ (BinaryOp::And | BinaryOp::Or | BinaryOp::Xor), left, right } => {
            let l = b_collect(left, atoms, budget, depth - 1);
            let r = b_collect(right, atoms, budget, depth - 1);
            B::Bin(*op, Box::new(l), Box::new(r))
        }
        ExprKind::Const { value } if *value == 0 => B::Const(false),
        ExprKind::Const { value } if *value == e.width.mask() => B::Const(true),
        _ => b_atom(e, atoms),
    }
}

fn b_atom(e: &Expr, atoms: &mut Vec<Expr>) -> B {
    if let Some(i) = atoms.iter().position(|a| a == e) {
        return B::Atom(i);
    }
    atoms.push(e.clone());
    B::Atom(atoms.len() - 1)
}

fn b_eval(b: &B, row: u32) -> bool {
    match b {
        B::Atom(i) => (row >> i) & 1 == 1,
        B::Const(v) => *v,
        B::Not(x) => !b_eval(x, row),
        B::Bin(op, l, r) => {
            let (l, r) = (b_eval(l, row), b_eval(r, row));
            match op {
                BinaryOp::And => l & r,
                BinaryOp::Or => l | r,
                _ => l ^ r,
            }
        }
    }
}

/// Cheapest expression tree per truth table over `k` atoms (layered by cost;
/// k = 4 is searched to cost 3, k <= 3 to cost 7).
fn b_best(k: usize) -> &'static HashMap<u16, (usize, B)> {
    use std::sync::OnceLock;
    static TABLES: [OnceLock<HashMap<u16, (usize, B)>>; 5] =
        [OnceLock::new(), OnceLock::new(), OnceLock::new(), OnceLock::new(), OnceLock::new()];
    TABLES[k].get_or_init(|| {
        let rows = 1_u32 << k;
        let mask: u16 = ((1_u32 << rows) - 1) as u16;
        let max_cost = if k >= 4 { 3 } else { 7 };
        let mut best: HashMap<u16, (usize, B)> = HashMap::new();
        let mut layers: Vec<Vec<(u16, B)>> = vec![Vec::new()];
        for i in 0..k {
            let mut t = 0_u16;
            for r in 0..rows {
                if (r >> i) & 1 == 1 {
                    t |= 1 << r;
                }
            }
            best.insert(t, (0, B::Atom(i)));
            layers[0].push((t, B::Atom(i)));
        }
        best.insert(0, (0, B::Const(false)));
        best.insert(mask, (0, B::Const(true)));
        layers[0].push((0, B::Const(false)));
        layers[0].push((mask, B::Const(true)));
        for c in 1..=max_cost {
            let mut layer: Vec<(u16, B)> = Vec::new();
            for (t1, b1) in &layers[c - 1] {
                let nt = !*t1 & mask;
                if !best.contains_key(&nt) {
                    best.insert(nt, (c, B::Not(Box::new(b1.clone()))));
                    layer.push((nt, B::Not(Box::new(b1.clone()))));
                }
            }
            for c1 in 0..c {
                let c2 = c - 1 - c1;
                for (t1, b1) in &layers[c1] {
                    for (t2, b2) in &layers[c2] {
                        for op in [BinaryOp::And, BinaryOp::Or, BinaryOp::Xor] {
                            let t = match op {
                                BinaryOp::And => t1 & t2,
                                BinaryOp::Or => t1 | t2,
                                _ => t1 ^ t2,
                            };
                            if !best.contains_key(&t) {
                                let tree = B::Bin(op, Box::new(b1.clone()), Box::new(b2.clone()));
                                best.insert(t, (c, tree.clone()));
                                layer.push((t, tree));
                            }
                        }
                    }
                }
            }
            layers.push(layer);
        }
        best
    })
}


fn b_build(b: &B, atoms: &[Expr], width: BitWidth) -> Option<Expr> {
    Some(match b {
        B::Atom(i) => atoms[*i].clone(),
        B::Const(v) => Expr::constant(width, if *v { width.mask() } else { 0 }),
        B::Not(x) => Expr::unary(vmpx_vm::UnaryOp::Not, b_build(x, atoms, width)?),
        B::Bin(op, l, r) => Expr::binary(*op, b_build(l, atoms, width)?, b_build(r, atoms, width)?).ok()?,
    })
}

/// Folds bitwise And/Or/Xor/Not trees over at most three atoms to their
/// cheapest equivalent (per-bit truth table, so exact at every width).
pub fn mba_fold(e: &Expr) -> Expr {
    rewrite_up(e, &mut |n| {
        let is_bool = matches!(
            &n.kind,
            ExprKind::Unary { op: vmpx_vm::UnaryOp::Not, .. }
                | ExprKind::Binary { op: BinaryOp::And | BinaryOp::Or | BinaryOp::Xor, .. }
        );
        if !is_bool {
            return None;
        }
        let mut atoms = Vec::new();
        let mut tree = B::Const(false);
        let mut fits = false;
        for depth in (2..=10).rev() {
            atoms.clear();
            let mut budget = 96;
            tree = b_collect(n, &mut atoms, &mut budget, depth);
            if atoms.len() <= 4 {
                fits = true;
                break;
            }
        }
        if !fits || atoms.iter().any(|a| a.width != n.width) {
            return None;
        }
        if atoms.is_empty() {
            return None; // pure constants are folded by the core simplifier
        }
        let cur = b_ops(&tree);
        if cur < 2 {
            return None;
        }
        let k = atoms.len();
        let rows = 1_u32 << k;
        let mut table = 0_u16;
        for r in 0..rows {
            if b_eval(&tree, r) {
                table |= 1 << r;
            }
        }
        let (cost, best) = b_best(k).get(&table)?;
        if *cost >= cur {
            return None;
        }
        b_build(best, &atoms, n.width)
    })
}

pub fn is_leaf(e: &Expr) -> bool {
    matches!(e.kind, ExprKind::Const { .. } | ExprKind::Register { .. })
}

/// Number of distinct non-leaf nodes (bounded).
pub fn size(e: &Expr, limit: usize) -> usize {
    e.node_count_bounded(limit)
}
