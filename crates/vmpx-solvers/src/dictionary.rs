//! Dictionary / pattern-recognition solver. A table of `(obfuscated shape ->
//! clean form)` templates is matched structurally against the target, bottom-up,
//! to a fixpoint. Template holes bind any subexpression; a hole that occurs
//! twice must bind structurally equal subtrees (expressions are hash-consed, so
//! that check is a pointer comparison). Commutative operators match in both
//! operand orders with backtracking.

use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;
use std::time::Instant;

use crate::Solver;
use vmpx_mutation::{Candidate, Cost, Problem, ProblemClass, SolverId};
use vmpx_vm::{BinaryOp, BitWidth, Expr, ExprKind, UnaryOp};

const MAX_HOLES: usize = 4;
const MAX_PASSES: usize = 8;

/// A template node, used for both patterns and replacements.
#[derive(Clone, Debug)]
enum P {
    /// Matches any subexpression; consistent across occurrences.
    H(usize),
    /// Literal constant (compared after truncation to the node width).
    C(u64),
    /// Pattern only: a constant `2^k` with `k >= 1`, bound to slot.
    Pow2(usize),
    /// Replacement only: the constant `k` where slot is bound to `2^k`.
    Log2(usize),
    Not(Box<P>),
    Neg(Box<P>),
    Bin(BinaryOp, Box<P>, Box<P>),
}

fn h(i: usize) -> P {
    P::H(i)
}
fn k(v: u64) -> P {
    P::C(v)
}
fn not(a: P) -> P {
    P::Not(Box::new(a))
}
fn neg(a: P) -> P {
    P::Neg(Box::new(a))
}
fn bin(op: BinaryOp, a: P, b: P) -> P {
    P::Bin(op, Box::new(a), Box::new(b))
}
fn and(a: P, b: P) -> P {
    bin(BinaryOp::And, a, b)
}
fn or(a: P, b: P) -> P {
    bin(BinaryOp::Or, a, b)
}
fn xor(a: P, b: P) -> P {
    bin(BinaryOp::Xor, a, b)
}
fn add(a: P, b: P) -> P {
    bin(BinaryOp::Add, a, b)
}
fn sub(a: P, b: P) -> P {
    bin(BinaryOp::Sub, a, b)
}
fn mul(a: P, b: P) -> P {
    bin(BinaryOp::Mul, a, b)
}
fn shl(a: P, b: P) -> P {
    bin(BinaryOp::Shl, a, b)
}

const fn commutative(op: BinaryOp) -> bool {
    matches!(
        op,
        BinaryOp::Add | BinaryOp::Mul | BinaryOp::Xor | BinaryOp::And | BinaryOp::Or
    )
}

type Bindings = [Option<Expr>; MAX_HOLES];

fn matches(p: &P, e: &Expr, b: &mut Bindings) -> bool {
    match p {
        P::H(i) => match &b[*i] {
            Some(bound) => bound == e,
            None => {
                b[*i] = Some(e.clone());
                true
            }
        },
        P::C(v) => matches!(e.kind, ExprKind::Const { value } if value == e.width.truncate(*v)),
        P::Pow2(i) => match e.kind {
            ExprKind::Const { value } if value.is_power_of_two() && value != 1 => {
                b[*i] = Some(e.clone());
                true
            }
            _ => false,
        },
        P::Log2(_) => false,
        P::Not(q) | P::Neg(q) => {
            let want = if matches!(p, P::Not(_)) {
                UnaryOp::Not
            } else {
                UnaryOp::Neg
            };
            matches!(&e.kind, ExprKind::Unary { op, value } if *op == want && matches(q, value, b))
        }
        P::Bin(op, pl, pr) => {
            let ExprKind::Binary {
                op: eop,
                left,
                right,
            } = &e.kind
            else {
                return false;
            };
            if eop != op {
                return false;
            }
            let saved = b.clone();
            if matches(pl, left, b) && matches(pr, right, b) {
                return true;
            }
            if commutative(*op) {
                *b = saved;
                if matches(pl, right, b) && matches(pr, left, b) {
                    return true;
                }
            }
            false
        }
    }
}

fn build(p: &P, b: &Bindings, w: BitWidth) -> Option<Expr> {
    Some(match p {
        P::H(i) => b[*i].clone()?,
        P::C(v) => Expr::constant(w, *v),
        P::Pow2(_) => return None,
        P::Log2(i) => match b[*i].as_ref()?.kind {
            ExprKind::Const { value } => Expr::constant(w, u64::from(value.trailing_zeros())),
            _ => return None,
        },
        P::Not(q) => Expr::unary(UnaryOp::Not, build(q, b, w)?),
        P::Neg(q) => Expr::unary(UnaryOp::Neg, build(q, b, w)?),
        P::Bin(op, l, r) => Expr::binary(*op, build(l, b, w)?, build(r, b, w)?).ok()?,
    })
}

struct Rule {
    pat: P,
    rep: P,
}

fn rule(pat: P, rep: P) -> Rule {
    Rule { pat, rep }
}

/// `2*x` in every shape the obfuscators emit it.
fn doubles(x: &P) -> [P; 3] {
    [
        mul(k(2), x.clone()),
        shl(x.clone(), k(1)),
        add(x.clone(), x.clone()),
    ]
}

static TABLE: LazyLock<Vec<Rule>> = LazyLock::new(build_table);

#[allow(clippy::too_many_lines)]
fn build_table() -> Vec<Rule> {
    let (a, b) = (h(0), h(1));
    let mut t = Vec::new();

    // Strength reduction.
    t.push(rule(mul(a.clone(), P::Pow2(1)), shl(a.clone(), P::Log2(1))));

    // Add/sub <-> bitwise MBA identities.
    for d in doubles(&and(a.clone(), b.clone())) {
        t.push(rule(
            add(xor(a.clone(), b.clone()), d.clone()),
            add(a.clone(), b.clone()),
        ));
        t.push(rule(
            sub(add(a.clone(), b.clone()), d),
            xor(a.clone(), b.clone()),
        ));
    }
    for d in doubles(&or(a.clone(), b.clone())) {
        t.push(rule(
            sub(d, xor(a.clone(), b.clone())),
            add(a.clone(), b.clone()),
        ));
    }
    t.push(rule(
        sub(or(a.clone(), b.clone()), and(a.clone(), b.clone())),
        xor(a.clone(), b.clone()),
    ));
    t.push(rule(
        add(or(a.clone(), b.clone()), and(a.clone(), b.clone())),
        add(a.clone(), b.clone()),
    ));
    t.push(rule(
        sub(or(a.clone(), b.clone()), xor(a.clone(), b.clone())),
        and(a.clone(), b.clone()),
    ));
    t.push(rule(
        sub(add(a.clone(), b.clone()), and(a.clone(), b.clone())),
        or(a.clone(), b.clone()),
    ));
    t.push(rule(
        sub(add(a.clone(), b.clone()), or(a.clone(), b.clone())),
        and(a.clone(), b.clone()),
    ));
    t.push(rule(
        add(and(a.clone(), not(b.clone())), b.clone()),
        or(a.clone(), b.clone()),
    ));
    t.push(rule(
        sub(a.clone(), and(a.clone(), b.clone())),
        and(a.clone(), not(b.clone())),
    ));
    t.push(rule(
        sub(or(a.clone(), b.clone()), a.clone()),
        and(b.clone(), not(a.clone())),
    ));

    // Disjoint-split and absorption.
    t.push(rule(
        add(and(a.clone(), not(b.clone())), and(a.clone(), b.clone())),
        a.clone(),
    ));
    t.push(rule(
        or(and(a.clone(), b.clone()), and(not(a.clone()), b.clone())),
        b.clone(),
    ));
    t.push(rule(
        xor(a.clone(), and(a.clone(), b.clone())),
        and(a.clone(), not(b.clone())),
    ));
    t.push(rule(
        xor(a.clone(), or(a.clone(), b.clone())),
        and(b.clone(), not(a.clone())),
    ));
    t.push(rule(
        or(a.clone(), xor(a.clone(), b.clone())),
        or(a.clone(), b.clone()),
    ));
    t.push(rule(
        and(a.clone(), xor(a.clone(), b.clone())),
        and(a.clone(), not(b.clone())),
    ));
    t.push(rule(or(a.clone(), and(a.clone(), b.clone())), a.clone()));
    t.push(rule(and(a.clone(), or(a.clone(), b.clone())), a.clone()));

    // XOR / XNOR spelled out.
    t.push(rule(
        or(
            and(a.clone(), not(b.clone())),
            and(not(a.clone()), b.clone()),
        ),
        xor(a.clone(), b.clone()),
    ));
    t.push(rule(
        and(or(a.clone(), b.clone()), not(and(a.clone(), b.clone()))),
        xor(a.clone(), b.clone()),
    ));
    t.push(rule(
        and(or(a.clone(), b.clone()), or(not(a.clone()), not(b.clone()))),
        xor(a.clone(), b.clone()),
    ));
    t.push(rule(
        or(and(a.clone(), b.clone()), xor(a.clone(), b.clone())),
        or(a.clone(), b.clone()),
    ));
    t.push(rule(
        xor(or(a.clone(), b.clone()), and(a.clone(), b.clone())),
        xor(a.clone(), b.clone()),
    ));
    t.push(rule(
        or(
            and(a.clone(), b.clone()),
            and(not(a.clone()), not(b.clone())),
        ),
        not(xor(a.clone(), b.clone())),
    ));
    t.push(rule(
        xor(not(a.clone()), not(b.clone())),
        xor(a.clone(), b.clone()),
    ));
    t.push(rule(xor(a.clone(), k(u64::MAX)), not(a.clone())));

    // De Morgan.
    t.push(rule(
        and(not(a.clone()), not(b.clone())),
        not(or(a.clone(), b.clone())),
    ));
    t.push(rule(
        or(not(a.clone()), not(b.clone())),
        not(and(a.clone(), b.clone())),
    ));
    t.push(rule(
        not(and(not(a.clone()), not(b.clone()))),
        or(a.clone(), b.clone()),
    ));
    t.push(rule(
        not(or(not(a.clone()), not(b.clone()))),
        and(a.clone(), b.clone()),
    ));

    // Negation / complement arithmetic.
    t.push(rule(add(not(a.clone()), k(1)), neg(a.clone())));
    t.push(rule(not(sub(a.clone(), k(1))), neg(a.clone())));
    t.push(rule(neg(neg(a.clone())), a.clone()));
    t.push(rule(not(not(a.clone())), a.clone()));
    t.push(rule(
        add(neg(a.clone()), b.clone()),
        sub(b.clone(), a.clone()),
    ));
    t.push(rule(
        sub(a.clone(), neg(b.clone())),
        add(a.clone(), b.clone()),
    ));
    t.push(rule(
        add(add(a.clone(), not(b.clone())), k(1)),
        sub(a.clone(), b.clone()),
    ));
    t.push(rule(
        not(add(not(a.clone()), b.clone())),
        sub(a.clone(), b.clone()),
    ));
    t.push(rule(add(a.clone(), not(a.clone())), k(u64::MAX)));
    t.push(rule(or(a.clone(), not(a.clone())), k(u64::MAX)));
    t.push(rule(xor(a.clone(), not(a.clone())), k(u64::MAX)));
    t.push(rule(and(a.clone(), not(a.clone())), k(0)));

    // Cancellation.
    t.push(rule(xor(xor(a.clone(), b.clone()), b.clone()), a.clone()));
    t.push(rule(add(sub(a.clone(), b.clone()), b.clone()), a.clone()));
    t.push(rule(sub(add(a.clone(), b.clone()), b.clone()), a.clone()));
    t.push(rule(sub(a.clone(), sub(a.clone(), b.clone())), b.clone()));
    t
}

/// `(node count with holes as leaves, multiplications)`; strictly decreases on
/// every rule, which bounds the rewriting.
#[cfg(test)]
fn template_cost(p: &P) -> (usize, usize) {
    match p {
        P::H(_) | P::C(_) | P::Pow2(_) | P::Log2(_) => (1, 0),
        P::Not(q) | P::Neg(q) => {
            let (n, m) = template_cost(q);
            (n + 1, m)
        }
        P::Bin(op, l, r) => {
            let (n1, m1) = template_cost(l);
            let (n2, m2) = template_cost(r);
            (n1 + n2 + 1, m1 + m2 + usize::from(*op == BinaryOp::Mul))
        }
    }
}

struct Rewriter {
    memo: HashMap<Expr, Expr>,
    steps: u64,
}

impl Rewriter {
    fn try_rules(&mut self, e: &Expr) -> Option<Expr> {
        for r in TABLE.iter() {
            let mut b: Bindings = Default::default();
            if matches(&r.pat, e, &mut b)
                && let Some(out) = build(&r.rep, &b, e.width)
                && out != *e
            {
                self.steps += 1;
                return Some(out);
            }
        }
        None
    }

    fn rewrite(&mut self, e: &Expr) -> Expr {
        if let Some(done) = self.memo.get(e) {
            return done.clone();
        }
        let mut cur = self.rebuild(e);
        for _ in 0..MAX_PASSES {
            let Some(next) = self.try_rules(&cur) else {
                break;
            };
            cur = self.rebuild(&next);
        }
        self.memo.insert(e.clone(), cur.clone());
        cur
    }

    /// Rewrites children, then reassembles the node if any child changed.
    fn rebuild(&mut self, e: &Expr) -> Expr {
        let w = e.width;
        let rebuilt = match &e.kind {
            ExprKind::Const { .. } | ExprKind::Register { .. } => None,
            ExprKind::Unary { op, value } => {
                let v = self.rewrite(value);
                (v != **value).then(|| Expr::unary(*op, v))
            }
            ExprKind::Binary { op, left, right } => {
                let (l, r) = (self.rewrite(left), self.rewrite(right));
                if l != **left || r != **right {
                    Expr::binary(*op, l, r).ok()
                } else {
                    None
                }
            }
            ExprKind::Load { address } => {
                let a = self.rewrite(address);
                (a != **address).then(|| Expr::load(w, a))
            }
            ExprKind::Intrinsic { name, arguments } => {
                let args: Vec<Expr> = arguments.iter().map(|a| self.rewrite(a)).collect();
                (args != *arguments).then(|| Expr::intrinsic(w, name.clone(), args))
            }
            ExprKind::Extract { value, lsb } => {
                let v = self.rewrite(value);
                if v == **value {
                    None
                } else {
                    Expr::extract(w, *lsb, v).ok()
                }
            }
            ExprKind::Concat { high, low } => {
                let (hi, lo) = (self.rewrite(high), self.rewrite(low));
                if hi != **high || lo != **low {
                    Expr::concat(hi, lo).ok()
                } else {
                    None
                }
            }
            ExprKind::Resize { op, value } => {
                let v = self.rewrite(value);
                if v == **value {
                    None
                } else {
                    Expr::resize(*op, w, v).ok()
                }
            }
        };
        rebuilt.unwrap_or_else(|| e.clone())
    }
}

/// `(distinct nodes, distinct multiplications)` of an expression DAG.
fn measure(root: &Expr) -> (usize, usize) {
    let mut seen: HashSet<Expr> = HashSet::new();
    let mut stack = vec![root.clone()];
    let mut muls = 0;
    while let Some(e) = stack.pop() {
        if !seen.insert(e.clone()) {
            continue;
        }
        match &e.kind {
            ExprKind::Const { .. } | ExprKind::Register { .. } => {}
            ExprKind::Unary { value, .. }
            | ExprKind::Load { address: value }
            | ExprKind::Extract { value, .. }
            | ExprKind::Resize { value, .. } => stack.push((**value).clone()),
            ExprKind::Binary { op, left, right } => {
                muls += usize::from(*op == BinaryOp::Mul);
                stack.push((**left).clone());
                stack.push((**right).clone());
            }
            ExprKind::Concat { high, low } => {
                stack.push((**high).clone());
                stack.push((**low).clone());
            }
            ExprKind::Intrinsic { arguments, .. } => stack.extend(arguments.iter().cloned()),
        }
    }
    (seen.len(), muls)
}

#[derive(Default)]
pub struct DictionarySolver;

impl Solver for DictionarySolver {
    fn id(&self) -> SolverId {
        SolverId::new("dictionary")
    }
    fn applies_to(&self, _class: ProblemClass) -> bool {
        true // patterns span every class
    }
    fn solve(&self, p: &Problem) -> Option<Candidate> {
        let start = Instant::now();
        let mut rw = Rewriter {
            memo: HashMap::new(),
            steps: 0,
        };
        let mut cur = p.expr.clone();
        for _ in 0..MAX_PASSES {
            rw.memo.clear();
            let next = rw.rewrite(&cur);
            if next == cur {
                break;
            }
            cur = next;
        }
        (cur != p.expr && measure(&cur) < measure(&p.expr)).then(|| Candidate {
            expr: cur,
            by: self.id(),
            cost: Cost {
                time_ns: u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX),
                steps: rw.steps,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use vmpx_mutation::Ctx;

    fn reg(n: &str) -> Expr {
        Expr::register(BitWidth::W64, n)
    }
    fn c(v: u64) -> Expr {
        Expr::constant(BitWidth::W64, v)
    }
    fn bn(op: BinaryOp, l: Expr, r: Expr) -> Expr {
        Expr::binary(op, l, r).unwrap()
    }
    fn solve(e: Expr) -> Option<Candidate> {
        DictionarySolver.solve(&Problem::new(e, ProblemClass::LinearMba, Ctx::default()))
    }
    fn eval(e: &Expr, x: u64, y: u64) -> u64 {
        let vals = BTreeMap::from([("x".to_string(), c(x)), ("y".to_string(), c(y))]);
        match e.substitute(&vals).unwrap().simplify().kind {
            ExprKind::Const { value } => value,
            other => panic!("not folded: {other:?}"),
        }
    }
    const SAMPLES: [(u64, u64); 6] = [
        (0, 0),
        (1, 2),
        (0xdead_beef, 0x1234_5678_9abc_def0),
        (u64::MAX, 1),
        (u64::MAX, u64::MAX),
        (0x8000_0000_0000_0000, 3),
    ];
    fn same_values(a: &Expr, b: &Expr) {
        for (x, y) in SAMPLES {
            assert_eq!(eval(a, x, y), eval(b, x, y), "at ({x:#x},{y:#x})");
        }
    }

    #[test]
    fn plus_two_and_mba_is_recognised() {
        let (x, y) = (reg("x"), reg("y"));
        let e = bn(
            BinaryOp::Add,
            bn(BinaryOp::Xor, x.clone(), y.clone()),
            bn(BinaryOp::Mul, c(2), bn(BinaryOp::And, x.clone(), y.clone())),
        );
        let cand = solve(e.clone()).expect("match");
        assert_eq!(cand.expr, bn(BinaryOp::Add, x, y));
        same_values(&e, &cand.expr);
    }

    #[test]
    fn hole_consistency_rejects_distinct_subtrees() {
        let (x, y, z) = (reg("x"), reg("y"), reg("z"));
        // (x|y) - (x&z): the two sides disagree on the second hole.
        let e = bn(
            BinaryOp::Sub,
            bn(BinaryOp::Or, x.clone(), y),
            bn(BinaryOp::And, x, z),
        );
        assert!(solve(e).is_none());
    }

    #[test]
    fn compound_nested_pattern_collapses() {
        let (x, y) = (reg("x"), reg("y"));
        // Two-stage: (x^y)+2*(x&y) -> x+y, then (x+y)-(x&y) -> x|y.
        let inner = bn(
            BinaryOp::Add,
            bn(BinaryOp::Xor, x.clone(), y.clone()),
            bn(BinaryOp::Shl, bn(BinaryOp::And, x.clone(), y.clone()), c(1)),
        );
        let e = bn(
            BinaryOp::Sub,
            inner,
            bn(BinaryOp::And, x.clone(), y.clone()),
        );
        let cand = solve(e.clone()).expect("match");
        assert_eq!(cand.expr, bn(BinaryOp::Or, x.clone(), y.clone()));
        same_values(&e, &cand.expr);
        // Nested: (~x & ~y) rewritten through De Morgan inside a recognised XOR shape.
        let nx = Expr::unary(UnaryOp::Not, x.clone());
        let ny = Expr::unary(UnaryOp::Not, y.clone());
        let e2 = bn(
            BinaryOp::Add,
            bn(BinaryOp::Xor, nx, ny),
            bn(BinaryOp::Mul, bn(BinaryOp::And, x.clone(), y.clone()), c(2)),
        );
        let cand2 = solve(e2.clone()).expect("match");
        assert_eq!(cand2.expr, bn(BinaryOp::Add, x, y));
        same_values(&e2, &cand2.expr);
    }

    #[test]
    fn mul_by_power_of_two_becomes_shift() {
        let x = reg("x");
        let e = bn(BinaryOp::Mul, x.clone(), c(8));
        let cand = solve(e.clone()).expect("match");
        assert_eq!(cand.expr, bn(BinaryOp::Shl, x, c(3)));
        same_values(&e, &cand.expr);
    }

    #[test]
    fn no_match_is_none() {
        let (x, y) = (reg("x"), reg("y"));
        assert!(solve(bn(BinaryOp::Add, x.clone(), y.clone())).is_none());
        assert!(solve(bn(BinaryOp::Mul, x, c(6))).is_none());
    }

    /// Every table entry is a real bitvector identity and strictly shrinks.
    #[test]
    fn every_template_is_a_sound_shrinking_identity() {
        let vals: [u64; 8] = [
            0,
            1,
            2,
            0x7f,
            0x80,
            0xdead_beef,
            u64::MAX,
            0x8000_0000_0000_0000,
        ];
        for (i, r) in TABLE.iter().enumerate() {
            assert!(
                template_cost(&r.rep) < template_cost(&r.pat),
                "rule {i} grows"
            );
            for w in [BitWidth::W8, BitWidth::W32, BitWidth::W64] {
                for &v0 in &vals {
                    for &v1 in &vals {
                        let mut b: Bindings = Default::default();
                        b[0] = Some(Expr::constant(w, v0));
                        b[1] = Some(Expr::constant(w, v1));
                        let slot_is_pow2 = format!("{:?}", r.pat).contains("Pow2");
                        if slot_is_pow2 {
                            b[1] = Some(Expr::constant(w, 1 << (v1 % u64::from(w.bits() - 1) + 1)));
                        }
                        let (p, q) = (build(&concrete(&r.pat), &b, w), build(&r.rep, &b, w));
                        let (Some(p), Some(q)) = (p, q) else {
                            panic!("rule {i} unbuildable")
                        };
                        assert_eq!(p, q, "rule {i} unsound at {w:?} a={v0:#x} b={v1:#x}");
                    }
                }
            }
        }
    }

    /// Pattern -> buildable template (Pow2 becomes a plain hole).
    fn concrete(p: &P) -> P {
        match p {
            P::Pow2(i) => P::H(*i),
            P::Not(q) => not(concrete(q)),
            P::Neg(q) => neg(concrete(q)),
            P::Bin(op, l, r) => bin(*op, concrete(l), concrete(r)),
            other => other.clone(),
        }
    }
}
