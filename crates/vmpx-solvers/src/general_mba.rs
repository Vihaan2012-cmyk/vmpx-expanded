//! General-MBA solver: algebraic rewriting to a fixpoint.
//!
//! Every rule is a true bitvector identity. A rule fires only when its result
//! has strictly fewer distinct AST nodes than the node it replaces, so the node
//! count is a decreasing measure and rewriting terminates; a fuel counter and a
//! pass cap are belt-and-braces. The tournament's input tests and the SMT gate
//! still verify the final proposal.

use std::collections::HashMap;
use std::time::Instant;

use crate::Solver;
use vmpx_mutation::{Candidate, Cost, Problem, ProblemClass, SolverId};
use vmpx_vm::{BinaryOp, Expr, ExprKind, UnaryOp};

const SIZE_CAP: usize = 1 << 16;
const MAX_PASSES: usize = 8;
const FUEL: u64 = 50_000;

/// Simplifies nonlinear / general MBA via algebraic rewrites.
pub struct GeneralMbaSolver;

impl Solver for GeneralMbaSolver {
    fn id(&self) -> SolverId {
        SolverId::new("general_mba")
    }

    fn applies_to(&self, class: ProblemClass) -> bool {
        matches!(class, ProblemClass::NonlinearMba | ProblemClass::LinearMba)
    }

    fn solve(&self, p: &Problem) -> Option<Candidate> {
        let start = Instant::now();
        let mut rw = Rewriter {
            fuel: FUEL,
            steps: 0,
        };
        let mut cur = p.expr.clone();
        for _ in 0..MAX_PASSES {
            let next = rw.rewrite(&cur, &mut HashMap::new());
            if next == cur {
                break;
            }
            cur = next;
        }
        (size(&cur) < size(&p.expr)).then(|| Candidate {
            expr: cur,
            by: self.id(),
            cost: Cost {
                time_ns: u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX),
                steps: rw.steps,
            },
        })
    }
}

struct Rewriter {
    fuel: u64,
    steps: u64,
}

type Memo = HashMap<Expr, Expr>;

impl Rewriter {
    fn rewrite(&mut self, e: &Expr, memo: &mut Memo) -> Expr {
        if let Some(r) = memo.get(e) {
            return r.clone();
        }
        let mut cur = self.rebuild(e, memo);
        while self.fuel > 0 {
            let Some(next) = best_rule(&cur) else { break };
            self.fuel -= 1;
            self.steps += 1;
            cur = self.rebuild(&next, memo);
        }
        memo.insert(e.clone(), cur.clone());
        memo.insert(cur.clone(), cur.clone());
        cur
    }

    /// Rewrites the children, then rebuilds the node (which also const-folds).
    fn rebuild(&mut self, e: &Expr, memo: &mut Memo) -> Expr {
        let w = e.width;
        match &e.kind {
            ExprKind::Const { .. } | ExprKind::Register { .. } => e.clone(),
            ExprKind::Unary { op, value } => Expr::unary(*op, self.rewrite(value, memo)),
            ExprKind::Binary { op, left, right } => {
                let l = self.rewrite(left, memo);
                let r = self.rewrite(right, memo);
                Expr::binary(*op, l, r).unwrap_or_else(|_| e.clone())
            }
            ExprKind::Load { address } => Expr::load(w, self.rewrite(address, memo)),
            ExprKind::Intrinsic { name, arguments } => {
                let args = arguments.iter().map(|a| self.rewrite(a, memo)).collect();
                Expr::intrinsic(w, name.clone(), args)
            }
            ExprKind::Extract { value, lsb } => {
                Expr::extract(w, *lsb, self.rewrite(value, memo)).unwrap_or_else(|_| e.clone())
            }
            ExprKind::Concat { high, low } => {
                let h = self.rewrite(high, memo);
                let l = self.rewrite(low, memo);
                Expr::concat(h, l).unwrap_or_else(|_| e.clone())
            }
            ExprKind::Resize { op, value } => {
                Expr::resize(*op, w, self.rewrite(value, memo)).unwrap_or_else(|_| e.clone())
            }
        }
    }
}

fn size(e: &Expr) -> usize {
    e.node_count_bounded(SIZE_CAP)
}

/// Smallest rule result that is strictly smaller than `e`.
fn best_rule(e: &Expr) -> Option<Expr> {
    let mut out = Vec::new();
    match &e.kind {
        ExprKind::Unary {
            op: UnaryOp::Not,
            value,
        } => not_rules(value, &mut out),
        ExprKind::Unary {
            op: UnaryOp::Neg,
            value,
        } => neg_rules(value, &mut out),
        ExprKind::Binary { op, left, right } => match op {
            BinaryOp::Add => add_rules(e, &mut out),
            BinaryOp::Sub => sub_rules(left, right, &mut out),
            BinaryOp::Mul => mul_rules(left, right, &mut out),
            BinaryOp::And | BinaryOp::Or | BinaryOp::Xor => {
                bitwise_rules(*op, left, right, &mut out)
            }
            _ => {}
        },
        _ => {}
    }
    let limit = size(e);
    out.into_iter()
        .map(|c| (size(&c), c))
        .filter(|(s, _)| *s < limit)
        .min_by_key(|(s, _)| *s)
        .map(|(_, c)| c)
}

// ---------- matching helpers ----------

fn bin_of(e: &Expr) -> Option<(BinaryOp, &Expr, &Expr)> {
    match &e.kind {
        ExprKind::Binary { op, left, right } => Some((*op, &**left, &**right)),
        _ => None,
    }
}

fn args(e: &Expr, op: BinaryOp) -> Option<(&Expr, &Expr)> {
    bin_of(e).and_then(|(o, l, r)| (o == op).then_some((l, r)))
}

fn un_of(e: &Expr, op: UnaryOp) -> Option<&Expr> {
    match &e.kind {
        ExprKind::Unary { op: o, value } if *o == op => Some(&**value),
        _ => None,
    }
}

fn not_of(e: &Expr) -> Option<&Expr> {
    un_of(e, UnaryOp::Not)
}

fn neg_of(e: &Expr) -> Option<&Expr> {
    un_of(e, UnaryOp::Neg)
}

fn konst(e: &Expr) -> Option<u64> {
    match e.kind {
        ExprKind::Const { value } => Some(value),
        _ => None,
    }
}

/// `x` for `2*x`, `x*2`, `x<<1`.
fn twice(e: &Expr) -> Option<&Expr> {
    let (op, l, r) = bin_of(e)?;
    match op {
        BinaryOp::Mul if konst(l) == Some(2) => Some(r),
        BinaryOp::Mul if konst(r) == Some(2) => Some(l),
        BinaryOp::Shl if konst(r) == Some(1) => Some(l),
        _ => None,
    }
}

/// Same unordered argument pair.
fn same(a: (&Expr, &Expr), b: (&Expr, &Expr)) -> bool {
    (a.0 == b.0 && a.1 == b.1) || (a.0 == b.1 && a.1 == b.0)
}

/// `(x op y)` and `(x op ~y)` (any order) combine to `x`.
fn split(a: &Expr, b: &Expr, op: BinaryOp) -> Option<Expr> {
    let (p, q) = args(a, op)?;
    let (r, s) = args(b, op)?;
    for (x, y) in [(p, q), (q, p)] {
        for (x2, y2) in [(r, s), (s, r)] {
            if x == x2 && (not_of(y2) == Some(y) || not_of(y) == Some(y2)) {
                return Some(x.clone());
            }
        }
    }
    None
}

fn mk(op: BinaryOp, a: &Expr, b: &Expr) -> Option<Expr> {
    Expr::binary(op, a.clone(), b.clone()).ok()
}

fn not(x: &Expr) -> Expr {
    Expr::unary(UnaryOp::Not, x.clone())
}

fn neg(x: &Expr) -> Expr {
    Expr::unary(UnaryOp::Neg, x.clone())
}

fn two_times(x: &Expr) -> Option<Expr> {
    mk(BinaryOp::Mul, &Expr::constant(x.width, 2), x)
}

fn flatten(e: &Expr, terms: &mut Vec<Expr>, c: &mut u64) {
    match &e.kind {
        ExprKind::Binary {
            op: BinaryOp::Add,
            left,
            right,
        } => {
            flatten(left, terms, c);
            flatten(right, terms, c);
        }
        ExprKind::Const { value } => *c = c.wrapping_add(*value),
        _ => terms.push(e.clone()),
    }
}

fn sum(w: vmpx_vm::BitWidth, terms: &[Expr], c: u64) -> Option<Expr> {
    let c = w.truncate(c);
    let mut it = terms.iter();
    let Some(first) = it.next() else {
        return Some(Expr::constant(w, c));
    };
    let mut acc = first.clone();
    for t in it {
        acc = mk(BinaryOp::Add, &acc, t)?;
    }
    if c != 0 {
        acc = mk(BinaryOp::Add, &acc, &Expr::constant(w, c))?;
    }
    Some(acc)
}

/// The two terms of a constant-free two-term sum.
fn two_terms(e: &Expr) -> Option<(Expr, Expr)> {
    let (mut t, mut c) = (Vec::new(), 0);
    flatten(e, &mut t, &mut c);
    match (t.len(), e.width.truncate(c)) {
        (2, 0) => {
            let b = t.pop()?;
            let a = t.pop()?;
            Some((a, b))
        }
        _ => None,
    }
}

// ---------- rules ----------

fn not_rules(v: &Expr, out: &mut Vec<Expr>) {
    // ~~x = x
    if let Some(x) = not_of(v) {
        out.push(x.clone());
    }
    // ~(a ^ ~b) = a ^ b
    if let Some((p, q)) = args(v, BinaryOp::Xor) {
        if let Some(r) = not_of(q) {
            out.extend(mk(BinaryOp::Xor, p, r));
        }
        if let Some(r) = not_of(p) {
            out.extend(mk(BinaryOp::Xor, r, q));
        }
    }
    // De Morgan: ~(~a & ~b) = a | b ; ~(~a | ~b) = a & b
    for (op, dual) in [(BinaryOp::And, BinaryOp::Or), (BinaryOp::Or, BinaryOp::And)] {
        if let Some((p, q)) = args(v, op) {
            if let (Some(a), Some(b)) = (not_of(p), not_of(q)) {
                out.extend(mk(dual, a, b));
            }
        }
    }
    // ~(x - 1) = -x ; ~(x + -1) = -x
    let (mut t, mut c) = (Vec::new(), 0);
    flatten(v, &mut t, &mut c);
    if t.len() == 1 && v.width.truncate(c) == v.width.mask() {
        out.push(neg(&t[0]));
    }
    if let Some((x, one)) = args(v, BinaryOp::Sub) {
        if konst(one) == Some(1) {
            out.push(neg(x));
        }
    }
}

fn neg_rules(v: &Expr, out: &mut Vec<Expr>) {
    // -(-x) = x
    if let Some(x) = neg_of(v) {
        out.push(x.clone());
    }
    // -(a - b) = b - a
    if let Some((a, b)) = args(v, BinaryOp::Sub) {
        out.extend(mk(BinaryOp::Sub, b, a));
    }
}

fn mul_rules(a: &Expr, b: &Expr, out: &mut Vec<Expr>) {
    for (k, x) in [(a, b), (b, a)] {
        match konst(k) {
            Some(0) => out.push(Expr::constant(k.width, 0)),
            Some(1) => out.push(x.clone()),
            Some(m) if m == k.width.mask() => out.push(neg(x)),
            _ => {}
        }
    }
}

fn add_pair(a: &Expr, b: &Expr) -> Option<Expr> {
    // x + -x = 0
    if neg_of(b) == Some(a) {
        return Some(Expr::constant(a.width, 0));
    }
    // (x ^ y) + 2*(x & y) = x + y
    if let (Some(xy), Some(t)) = (args(a, BinaryOp::Xor), twice(b)) {
        if let Some(and) = args(t, BinaryOp::And) {
            if same(xy, and) {
                return mk(BinaryOp::Add, xy.0, xy.1);
            }
        }
    }
    // (x | y) + (x & y) = x + y
    if let (Some(or), Some(and)) = (args(a, BinaryOp::Or), args(b, BinaryOp::And)) {
        if same(or, and) {
            return mk(BinaryOp::Add, or.0, or.1);
        }
    }
    // (x & y) + (x ^ y) = x | y   (disjoint bits)
    if let (Some(and), Some(xor)) = (args(a, BinaryOp::And), args(b, BinaryOp::Xor)) {
        if same(and, xor) {
            return mk(BinaryOp::Or, and.0, and.1);
        }
    }
    // (x & y) + (x & ~y) = x
    split(a, b, BinaryOp::And)
}

fn add_rules(e: &Expr, out: &mut Vec<Expr>) {
    let w = e.width;
    let (mut terms, mut c) = (Vec::new(), 0);
    flatten(e, &mut terms, &mut c);
    let c = w.truncate(c);
    for i in 0..terms.len() {
        let rest = |skip: &[usize]| -> Vec<Expr> {
            terms
                .iter()
                .enumerate()
                .filter(|(k, _)| !skip.contains(k))
                .map(|(_, t)| t.clone())
                .collect()
        };
        for j in (i + 1)..terms.len() {
            let r = add_pair(&terms[i], &terms[j]).or_else(|| add_pair(&terms[j], &terms[i]));
            if let Some(r) = r {
                let mut others = rest(&[i, j]);
                others.push(r);
                out.extend(sum(w, &others, c));
            }
        }
        let others = rest(&[i]);
        if let Some(x) = neg_of(&terms[i]) {
            // S + -x = S - x ; -x + -1 = ~x
            if !others.is_empty() || c != 0 {
                if let Some(s) = sum(w, &others, c) {
                    out.extend(mk(BinaryOp::Sub, &s, x));
                }
            }
            if c == w.mask() {
                let mut o = others.clone();
                o.push(not(x));
                out.extend(sum(w, &o, 0));
            }
        }
        // S + ~x + 1 = S - x
        if let (Some(x), 1) = (not_of(&terms[i]), c) {
            if others.is_empty() {
                out.push(neg(x));
            } else if let Some(s) = sum(w, &others, 0) {
                out.extend(mk(BinaryOp::Sub, &s, x));
            }
        }
    }
}

fn sub_pair(a: &Expr, b: &Expr) -> Option<Expr> {
    // (x | y) - (x & y) = x ^ y ; (x | y) - (x ^ y) = x & y
    if let Some(or) = args(a, BinaryOp::Or) {
        if args(b, BinaryOp::And).is_some_and(|and| same(or, and)) {
            return mk(BinaryOp::Xor, or.0, or.1);
        }
        if args(b, BinaryOp::Xor).is_some_and(|xor| same(or, xor)) {
            return mk(BinaryOp::And, or.0, or.1);
        }
        // (x | y) - x = ~x & y
        for (x, y) in [(or.0, or.1), (or.1, or.0)] {
            if b == x {
                return mk(BinaryOp::And, &not(x), y);
            }
        }
    }
    // (x + y) - 2*(x&y) = x ^ y ; - (x|y) = x & y ; - (x^y) = 2*(x&y) ; - (x&y) = x | y
    if let Some((x, y)) = two_terms(a) {
        let xy = (&x, &y);
        if twice(b)
            .and_then(|t| args(t, BinaryOp::And))
            .is_some_and(|and| same(xy, and))
        {
            return mk(BinaryOp::Xor, xy.0, xy.1);
        }
        if args(b, BinaryOp::Or).is_some_and(|or| same(xy, or)) {
            return mk(BinaryOp::And, xy.0, xy.1);
        }
        if args(b, BinaryOp::Xor).is_some_and(|xor| same(xy, xor)) {
            return two_times(&mk(BinaryOp::And, xy.0, xy.1)?);
        }
        if args(b, BinaryOp::And).is_some_and(|and| same(xy, and)) {
            return mk(BinaryOp::Or, xy.0, xy.1);
        }
    }
    // 2*(x|y) - (x^y) = x + y ; 2*(x|y) - (x+y) = x ^ y
    if let Some(or) = twice(a).and_then(|t| args(t, BinaryOp::Or)) {
        if args(b, BinaryOp::Xor).is_some_and(|xor| same(or, xor)) {
            return mk(BinaryOp::Add, or.0, or.1);
        }
        if two_terms(b).is_some_and(|(p, q)| same(or, (&p, &q))) {
            return mk(BinaryOp::Xor, or.0, or.1);
        }
    }
    // x - (x & y) = x & ~y
    if let Some((p, q)) = args(b, BinaryOp::And) {
        for (x, y) in [(p, q), (q, p)] {
            if a == x {
                return mk(BinaryOp::And, x, &not(y));
            }
        }
    }
    None
}

fn sub_rules(a: &Expr, b: &Expr, out: &mut Vec<Expr>) {
    let w = a.width;
    if a == b {
        out.push(Expr::constant(w, 0));
    }
    if konst(a) == Some(0) {
        out.push(neg(b));
    }
    // x - -y = x + y
    if let Some(y) = neg_of(b) {
        out.extend(mk(BinaryOp::Add, a, y));
    }
    // -x - 1 = ~x
    if let (Some(x), Some(1)) = (neg_of(a), konst(b)) {
        out.push(not(x));
    }
    // (S + b) - b = S
    let (mut terms, mut c) = (Vec::new(), 0);
    flatten(a, &mut terms, &mut c);
    if let Some(pos) = terms.iter().position(|t| t == b) {
        terms.remove(pos);
        out.extend(sum(w, &terms, c));
    }
    out.extend(sub_pair(a, b));
}

fn bitwise_pair(op: BinaryOp, a: &Expr, b: &Expr) -> Option<Expr> {
    let w = a.width;
    // Constant / identity / absorbing.
    match (op, konst(a)) {
        (BinaryOp::And | BinaryOp::Or, Some(0)) | (BinaryOp::Xor, Some(0)) => {
            return Some(if op == BinaryOp::And {
                a.clone()
            } else {
                b.clone()
            });
        }
        (BinaryOp::And, Some(m)) if m == w.mask() => return Some(b.clone()),
        (BinaryOp::Or, Some(m)) if m == w.mask() => return Some(a.clone()),
        (BinaryOp::Xor, Some(m)) if m == w.mask() => return Some(not(b)),
        _ => {}
    }
    if not_of(a) == Some(b) {
        return match op {
            BinaryOp::And => Some(Expr::constant(w, 0)),
            _ => Some(Expr::constant(w, w.mask())),
        };
    }
    match op {
        BinaryOp::And => and_pair(a, b),
        BinaryOp::Or => or_pair(a, b),
        _ => xor_pair(a, b),
    }
}

fn and_pair(a: &Expr, b: &Expr) -> Option<Expr> {
    // x & (x | y) = x
    if args(b, BinaryOp::Or).is_some_and(|(p, q)| a == p || a == q) {
        return Some(a.clone());
    }
    // x & (x ^ y) = x & ~y
    if let Some((p, q)) = args(b, BinaryOp::Xor) {
        for (x, y) in [(p, q), (q, p)] {
            if a == x {
                return mk(BinaryOp::And, x, &not(y));
            }
        }
    }
    if let Some(or) = args(a, BinaryOp::Or) {
        // (x | y) & (x ^ y) = x ^ y
        if let Some(xor) = args(b, BinaryOp::Xor) {
            if same(or, xor) {
                return mk(BinaryOp::Xor, xor.0, xor.1);
            }
        }
        // (x | y) & ~(x & y) = x ^ y
        if not_of(b)
            .and_then(|n| args(n, BinaryOp::And))
            .is_some_and(|and| same(or, and))
        {
            return mk(BinaryOp::Xor, or.0, or.1);
        }
        // (x | y) & (~x | ~y) = x ^ y
        if let Some((p, q)) = args(b, BinaryOp::Or) {
            if let (Some(np), Some(nq)) = (not_of(p), not_of(q)) {
                if same(or, (np, nq)) {
                    return mk(BinaryOp::Xor, or.0, or.1);
                }
            }
        }
    }
    // (x | y) & (x | ~y) = x
    split(a, b, BinaryOp::Or)
}

fn or_pair(a: &Expr, b: &Expr) -> Option<Expr> {
    // x | (x & y) = x
    if args(b, BinaryOp::And).is_some_and(|(p, q)| a == p || a == q) {
        return Some(a.clone());
    }
    // x | (x ^ y) = x | y
    if let Some((p, q)) = args(b, BinaryOp::Xor) {
        for (x, y) in [(p, q), (q, p)] {
            if a == x {
                return mk(BinaryOp::Or, x, y);
            }
        }
    }
    // (x & y) | (x ^ y) = x | y
    if let (Some(and), Some(xor)) = (args(a, BinaryOp::And), args(b, BinaryOp::Xor)) {
        if same(and, xor) {
            return mk(BinaryOp::Or, and.0, and.1);
        }
    }
    // (x & y) | (x & ~y) = x
    split(a, b, BinaryOp::And)
}

fn xor_pair(a: &Expr, b: &Expr) -> Option<Expr> {
    // x ^ (x ^ y) = y
    if let Some((p, q)) = args(b, BinaryOp::Xor) {
        if a == p {
            return Some(q.clone());
        }
        if a == q {
            return Some(p.clone());
        }
    }
    if let Some(or) = args(a, BinaryOp::Or) {
        // (x | y) ^ (x & y) = x ^ y
        if args(b, BinaryOp::And).is_some_and(|and| same(or, and)) {
            return mk(BinaryOp::Xor, or.0, or.1);
        }
        // (x | y) ^ x = ~x & y
        for (x, y) in [(or.0, or.1), (or.1, or.0)] {
            if b == x {
                return mk(BinaryOp::And, &not(x), y);
            }
        }
    }
    if let Some(and) = args(a, BinaryOp::And) {
        // (x & y) ^ (x ^ y) = x | y
        if args(b, BinaryOp::Xor).is_some_and(|xor| same(and, xor)) {
            return mk(BinaryOp::Or, and.0, and.1);
        }
        // (x & y) ^ x = x & ~y
        for (x, y) in [(and.0, and.1), (and.1, and.0)] {
            if b == x {
                return mk(BinaryOp::And, x, &not(y));
            }
        }
    }
    // (x & y) ^ (x & ~y) = x
    split(a, b, BinaryOp::And)
}

fn bitwise_rules(op: BinaryOp, a: &Expr, b: &Expr, out: &mut Vec<Expr>) {
    if a == b {
        out.push(if op == BinaryOp::Xor {
            Expr::constant(a.width, 0)
        } else {
            a.clone()
        });
    }
    out.extend(bitwise_pair(op, a, b).or_else(|| bitwise_pair(op, b, a)));
    // ~a & ~b = ~(a | b) ; ~a | ~b = ~(a & b) ; ~a ^ ~b = a ^ b
    if let (Some(p), Some(q)) = (not_of(a), not_of(b)) {
        match op {
            BinaryOp::And => out.extend(mk(BinaryOp::Or, p, q).map(|e| not(&e))),
            BinaryOp::Or => out.extend(mk(BinaryOp::And, p, q).map(|e| not(&e))),
            _ => out.extend(mk(BinaryOp::Xor, p, q)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vmpx_vm::{BitWidth, ExprRef};

    const W: BitWidth = BitWidth::W64;

    fn reg(n: &str) -> Expr {
        Expr::register(W, n)
    }

    fn k(v: u64) -> Expr {
        Expr::constant(W, v)
    }

    /// Builds a node without simplification, so rules see the raw shape.
    fn raw(op: BinaryOp, l: Expr, r: Expr) -> Expr {
        Expr {
            width: W,
            kind: ExprKind::Binary {
                op,
                left: ExprRef::new(l),
                right: ExprRef::new(r),
            },
        }
    }

    fn raw_un(op: UnaryOp, v: Expr) -> Expr {
        Expr {
            width: W,
            kind: ExprKind::Unary {
                op,
                value: ExprRef::new(v),
            },
        }
    }

    fn eval(e: &Expr, x: u64, y: u64) -> u64 {
        match &e.kind {
            ExprKind::Const { value } => *value,
            ExprKind::Register { name } => {
                if name == "x" {
                    x
                } else {
                    y
                }
            }
            ExprKind::Unary { op, value } => {
                let v = eval(value, x, y);
                match op {
                    UnaryOp::Not => !v,
                    UnaryOp::Neg => v.wrapping_neg(),
                }
            }
            ExprKind::Binary { op, left, right } => {
                let (a, b) = (eval(left, x, y), eval(right, x, y));
                match op {
                    BinaryOp::Add => a.wrapping_add(b),
                    BinaryOp::Sub => a.wrapping_sub(b),
                    BinaryOp::Mul => a.wrapping_mul(b),
                    BinaryOp::Xor => a ^ b,
                    BinaryOp::And => a & b,
                    BinaryOp::Or => a | b,
                    BinaryOp::Shl => a.wrapping_shl(u32::try_from(b).unwrap()),
                    _ => unreachable!("not generated"),
                }
            }
            _ => unreachable!("not generated"),
        }
    }

    const SAMPLES: [(u64, u64); 6] = [
        (0, 0),
        (1, u64::MAX),
        (0x0123_4567_89AB_CDEF, 0xFEDC_BA98_7654_3210),
        (u64::MAX, u64::MAX),
        (0x8000_0000_0000_0000, 3),
        (0xDEAD_BEEF, 0xCAFE_F00D_1234_5678),
    ];

    fn solve(e: &Expr) -> Option<Expr> {
        let p = Problem::new(
            e.clone(),
            ProblemClass::NonlinearMba,
            vmpx_mutation::Ctx::default(),
        );
        let c = GeneralMbaSolver.solve(&p)?;
        for (x, y) in SAMPLES {
            assert_eq!(eval(e, x, y), eval(&c.expr, x, y), "{e:?} -> {:?}", c.expr);
        }
        Some(c.expr)
    }

    fn and_xy() -> Expr {
        raw(BinaryOp::And, reg("x"), reg("y"))
    }

    fn xor_xy() -> Expr {
        raw(BinaryOp::Xor, reg("x"), reg("y"))
    }

    #[test]
    fn xor_plus_twice_and_is_add() {
        let two_and = raw(BinaryOp::Mul, k(2), and_xy());
        let e = raw(BinaryOp::Add, xor_xy(), two_and);
        let want = Expr::binary(BinaryOp::Add, reg("x"), reg("y")).unwrap();
        assert_eq!(solve(&e), Some(want));
    }

    #[test]
    fn not_plus_one_is_neg() {
        let e = raw(BinaryOp::Add, raw_un(UnaryOp::Not, reg("x")), k(1));
        assert_eq!(solve(&e), Some(Expr::unary(UnaryOp::Neg, reg("x"))));
    }

    #[test]
    fn absorbing_and_identity_rules_fire() {
        let x = reg("x");
        let ones = k(u64::MAX);
        let cases = [
            (raw(BinaryOp::And, x.clone(), k(0)), k(0)),
            (raw(BinaryOp::Or, x.clone(), k(0)), x.clone()),
            (raw(BinaryOp::Xor, x.clone(), k(0)), x.clone()),
            (raw(BinaryOp::And, x.clone(), ones.clone()), x.clone()),
            (raw(BinaryOp::Or, x.clone(), ones.clone()), ones.clone()),
            (raw(BinaryOp::Xor, x.clone(), x.clone()), k(0)),
            (raw(BinaryOp::And, x.clone(), x.clone()), x.clone()),
            (raw(BinaryOp::Or, x.clone(), x.clone()), x.clone()),
            (raw(BinaryOp::Sub, x.clone(), x.clone()), k(0)),
            (raw(BinaryOp::And, k(0), x.clone()), k(0)),
            (raw(BinaryOp::Or, ones.clone(), x.clone()), ones.clone()),
            (
                raw(BinaryOp::Xor, x.clone(), ones),
                Expr::unary(UnaryOp::Not, x.clone()),
            ),
            (
                raw(BinaryOp::Add, x.clone(), raw_un(UnaryOp::Neg, x.clone())),
                k(0),
            ),
        ];
        for (input, want) in cases {
            assert_eq!(solve(&input), Some(want), "{input:?}");
        }
    }

    #[test]
    fn nested_identities_collapse_to_fixpoint() {
        // ((x^y) + 2*(x&y)) - (x|y) -> (x+y) - (x|y) -> x&y
        let sum = raw(BinaryOp::Add, xor_xy(), raw(BinaryOp::Mul, k(2), and_xy()));
        let e = raw(BinaryOp::Sub, sum, raw(BinaryOp::Or, reg("x"), reg("y")));
        let want = Expr::binary(BinaryOp::And, reg("x"), reg("y")).unwrap();
        assert_eq!(solve(&e), Some(want));

        // x - y obfuscated: (x ^ -y) + 2*(x & -y), then (x|y)-(x&y) = x^y on top.
        let ny = raw_un(UnaryOp::Neg, reg("y"));
        let xor = raw(BinaryOp::Xor, reg("x"), ny.clone());
        let and = raw(BinaryOp::And, reg("x"), ny);
        let diff = raw(BinaryOp::Add, xor, raw(BinaryOp::Mul, k(2), and));
        let or_minus_and = raw(
            BinaryOp::Sub,
            raw(BinaryOp::Or, reg("x"), reg("y")),
            and_xy(),
        );
        let e = raw(BinaryOp::Add, diff, or_minus_and);
        let got = solve(&e).expect("collapses");
        assert!(size(&got) <= 7, "{got:?}");
    }

    #[test]
    fn non_mba_passes() {
        let e = Expr::binary(
            BinaryOp::Add,
            Expr::binary(BinaryOp::Mul, reg("x"), reg("y")).unwrap(),
            k(3),
        )
        .unwrap();
        assert!(solve(&e).is_none());
    }

    #[test]
    fn routing() {
        assert!(GeneralMbaSolver.applies_to(ProblemClass::LinearMba));
        assert!(GeneralMbaSolver.applies_to(ProblemClass::NonlinearMba));
        assert!(!GeneralMbaSolver.applies_to(ProblemClass::DeadCode));
    }
}
