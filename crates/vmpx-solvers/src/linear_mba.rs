//! Linear-MBA simplifier (SiMBA-style, up to two variables).
//!
//! A linear MBA is an integer combination of bitwise terms. Evaluated at the
//! boolean points (each variable all-0 or all-1) it is fully determined by its
//! 2^n values, which gives its coordinates in the conjunction basis
//! `{1, x, y, x&y}`. We then search for the sparsest combination over the full
//! bitwise basis (all 14 non-constant 2-variable truth tables plus a constant).

use std::collections::BTreeMap;
use std::time::Instant;

use crate::Solver;
use vmpx_mutation::{Candidate, Cost, Problem, ProblemClass, SolverId};
use vmpx_vm::{BinaryOp, BitWidth, Expr, ExprKind, UnaryOp};

/// Canonicalizes linear mixed boolean-arithmetic to a minimal linear form.
pub struct LinearMbaSolver;

const NODE_LIMIT: usize = 4096;
const MAX_VARS: usize = 2;
const MAX_TERMS: usize = 3;

/// Truth tables are indexed by bit `(b << 1) | a` where `a`/`b` are x/y. The
/// all-zero and all-one tables are folded into the constant term.
fn atom_expr(t: u8, x: &Expr, y: &Expr) -> Expr {
    let not = |e: &Expr| Expr::unary(UnaryOp::Not, e.clone());
    let bin = |op, l: Expr, r: Expr| Expr::binary(op, l, r).expect("same width");
    match t {
        1 => not(&bin(BinaryOp::Or, x.clone(), y.clone())),
        2 => bin(BinaryOp::And, x.clone(), not(y)),
        3 => not(y),
        4 => bin(BinaryOp::And, not(x), y.clone()),
        5 => not(x),
        6 => bin(BinaryOp::Xor, x.clone(), y.clone()),
        7 => not(&bin(BinaryOp::And, x.clone(), y.clone())),
        8 => bin(BinaryOp::And, x.clone(), y.clone()),
        9 => not(&bin(BinaryOp::Xor, x.clone(), y.clone())),
        10 => x.clone(),
        11 => bin(BinaryOp::Or, x.clone(), not(y)),
        12 => y.clone(),
        13 => bin(BinaryOp::Or, not(x), y.clone()),
        14 => bin(BinaryOp::Or, x.clone(), y.clone()),
        _ => unreachable!("truth table {t}"),
    }
}

/// Coordinates `(K, cx, cy, cxy)` in the basis `{1, x, y, x&y}` of a value
/// vector over the points (a,b) = (0,0), (1,0), (0,1), (1,1), where a set
/// variable is the all-ones value (-1 as an integer).
fn coords(v: [i128; 4]) -> [i128; 4] {
    let k = v[0];
    let cx = k - v[1];
    let cy = k - v[2];
    let cxy = k - cx - cy - v[3];
    [k, cx, cy, cxy]
}

fn atom_coords(t: Option<u8>) -> [i128; 4] {
    match t {
        None => coords([1; 4]),
        Some(t) => coords(std::array::from_fn(|i| -i128::from((t >> i) & 1))),
    }
}

fn is_linear(e: &Expr) -> bool {
    match &e.kind {
        ExprKind::Const { .. } | ExprKind::Register { .. } => true,
        ExprKind::Unary { value, .. } => is_linear(value),
        ExprKind::Binary { op, left, right } => match op {
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Xor | BinaryOp::And | BinaryOp::Or => {
                is_linear(left) && is_linear(right)
            }
            BinaryOp::Mul => {
                (left.dependencies().is_empty() && is_linear(right))
                    || (right.dependencies().is_empty() && is_linear(left))
            }
            _ => false,
        },
        _ => false,
    }
}

fn eval_point(e: &Expr, vals: &BTreeMap<String, Expr>) -> Option<u64> {
    match e.substitute(vals).ok()?.simplify().kind {
        ExprKind::Const { value } => Some(value),
        _ => None,
    }
}

fn sign_extend(width: BitWidth, v: u64) -> i128 {
    let bits = width.bits();
    let v = width.truncate(v);
    if bits < 64 && v >> (bits - 1) & 1 == 1 {
        i128::from(v) - (1_i128 << bits)
    } else if bits == 64 {
        i128::from(v as i64)
    } else {
        i128::from(v)
    }
}

type Frac = (i128, i128);

fn norm(n: i128, d: i128) -> Option<Frac> {
    let (mut a, mut b) = (n.checked_abs()?, d.checked_abs()?);
    while b != 0 {
        (a, b) = (b, a % b);
    }
    let g = a.max(1);
    let s = if d < 0 { -1 } else { 1 };
    Some((n / g * s, d / g * s))
}

fn fsub_mul(a: Frac, f: Frac, b: Frac) -> Option<Frac> {
    // a - f*b
    let fb = (f.0.checked_mul(b.0)?, f.1.checked_mul(b.1)?);
    let num = a.0.checked_mul(fb.1)?.checked_sub(fb.0.checked_mul(a.1)?)?;
    norm(num, a.1.checked_mul(fb.1)?)
}

fn fdiv(a: Frac, b: Frac) -> Option<Frac> {
    norm(a.0.checked_mul(b.1)?, a.1.checked_mul(b.0)?)
}

/// Solve `sum_j m_j * cols[j] = rhs` over the integers; `None` if the columns
/// are dependent, the system is inconsistent, or a solution is non-integral.
fn solve(cols: &[[i128; 4]], rhs: [i128; 4]) -> Option<Vec<i128>> {
    let k = cols.len();
    let mut m: Vec<Vec<Frac>> = (0..4)
        .map(|r| {
            cols.iter()
                .map(|c| (c[r], 1))
                .chain(std::iter::once((rhs[r], 1)))
                .collect()
        })
        .collect();
    for c in 0..k {
        let p = (c..4).find(|&r| m[r][c].0 != 0)?;
        m.swap(c, p);
        for r in 0..4 {
            if r != c && m[r][c].0 != 0 {
                let f = fdiv(m[r][c], m[c][c])?;
                for j in c..=k {
                    m[r][j] = fsub_mul(m[r][j], f, m[c][j])?;
                }
            }
        }
    }
    if m[k..4].iter().any(|row| row[k].0 != 0) {
        return None;
    }
    (0..k)
        .map(|c| {
            let q = fdiv(m[c][k], m[c][c])?;
            (q.1 == 1).then_some(q.0)
        })
        .collect()
}

fn subsets(n: usize, k: usize) -> Vec<Vec<usize>> {
    fn rec(start: usize, n: usize, k: usize, cur: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if cur.len() == k {
            out.push(cur.clone());
            return;
        }
        for i in start..n {
            cur.push(i);
            rec(i + 1, n, k, cur, out);
            cur.pop();
        }
    }
    let mut out = Vec::new();
    rec(0, n, k, &mut Vec::new(), &mut out);
    out
}

fn build(width: BitWidth, terms: &[(i128, Option<Expr>)]) -> Option<Expr> {
    let mut acc: Option<Expr> = None;
    for (coef, atom) in terms {
        let c = sign_extend(width, *coef as u64);
        if c == 0 {
            continue;
        }
        let mag = Expr::constant(width, c.unsigned_abs() as u64);
        let term = match atom {
            None => Expr::constant(width, c as u64),
            Some(a) if c.unsigned_abs() == 1 => a.clone(),
            Some(a) => Expr::binary(BinaryOp::Mul, mag, a.clone()).ok()?,
        };
        acc = Some(match acc {
            None if c < 0 && atom.is_some() && c.unsigned_abs() == 1 => {
                Expr::unary(UnaryOp::Neg, term)
            }
            None if c < 0 && atom.is_some() => Expr::binary(
                BinaryOp::Mul,
                Expr::constant(width, c as u64),
                atom.clone().expect("checked"),
            )
            .ok()?,
            None => term,
            Some(prev) if atom.is_some() && c < 0 => {
                Expr::binary(BinaryOp::Sub, prev, term).ok()?
            }
            Some(prev) => Expr::binary(BinaryOp::Add, prev, term).ok()?,
        });
    }
    Some(acc.unwrap_or_else(|| Expr::constant(width, 0)).simplify())
}

fn minimize(e: &Expr) -> Option<Expr> {
    if e.node_count_bounded(NODE_LIMIT) > NODE_LIMIT || !is_linear(e) {
        return None;
    }
    let width = e.width;
    let vars: Vec<String> = e.dependencies().into_iter().collect();
    if vars.len() > MAX_VARS {
        return None;
    }
    let reg = |n: &String| Expr::register(width, n.clone());
    let x = vars.first().map_or_else(|| Expr::constant(width, 0), reg);
    let y = vars.get(1).map_or_else(|| x.clone(), reg);
    let all = width.mask();

    let mut v = [0_i128; 4];
    for (i, slot) in v.iter_mut().enumerate() {
        let mut vals = BTreeMap::new();
        let (a, b) = (i & 1, (i >> 1) & 1);
        for (n, bit) in vars.iter().zip([a, b]) {
            vals.insert(
                n.clone(),
                Expr::constant(width, if bit == 1 { all } else { 0 }),
            );
        }
        // With one variable, y mirrors x, so only the a == b points are real.
        *slot = sign_extend(width, eval_point(e, &vals)?);
    }
    if vars.len() == 1 && (v[1] != v[3] || v[0] != v[2]) {
        return None;
    }
    let target = coords(v);

    // Atom pool: None = constant; Some(t) = truth table t. One variable only
    // admits tables that ignore y.
    let mut pool: Vec<Option<u8>> = vec![None];
    pool.extend(
        (1..=14_u8)
            .filter(|t| vars.len() > 1 || [5, 10].contains(t))
            .map(Some),
    );
    if vars.is_empty() {
        pool.truncate(1);
    }
    let ac: Vec<[i128; 4]> = pool.iter().map(|t| atom_coords(*t)).collect();

    let mut best: Option<Expr> = None;
    let mut best_nodes = usize::MAX;
    for k in 0..=MAX_TERMS.min(pool.len()) {
        for s in subsets(pool.len(), k) {
            let cols: Vec<[i128; 4]> = s.iter().map(|&i| ac[i]).collect();
            let Some(ms) = (if k == 0 {
                (target == [0; 4]).then(Vec::new)
            } else {
                solve(&cols, target)
            }) else {
                continue;
            };
            if ms.contains(&0) {
                continue;
            }
            let terms: Vec<(i128, Option<Expr>)> = s
                .iter()
                .zip(&ms)
                .map(|(&i, &m)| (m, pool[i].map(|t| atom_expr(t, &x, &y))))
                .collect();
            if let Some(cand) = build(width, &terms) {
                let n = cand.node_count_bounded(NODE_LIMIT);
                if n < best_nodes {
                    best_nodes = n;
                    best = Some(cand);
                }
            }
        }
    }
    best
}

impl Solver for LinearMbaSolver {
    fn id(&self) -> SolverId {
        SolverId::new("linear_mba")
    }

    fn applies_to(&self, class: ProblemClass) -> bool {
        class == ProblemClass::LinearMba
    }

    fn solve(&self, p: &Problem) -> Option<Candidate> {
        let start = Instant::now();
        let expr = minimize(&p.expr)?;
        if expr.node_count_bounded(NODE_LIMIT) >= p.expr.node_count_bounded(NODE_LIMIT) {
            return None;
        }
        Some(Candidate {
            expr,
            by: self.id(),
            cost: Cost {
                time_ns: u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX),
                steps: 1,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vmpx_mutation::Ctx;

    const W: BitWidth = BitWidth::W64;

    fn r(n: &str) -> Expr {
        Expr::register(W, n)
    }
    fn k(v: u64) -> Expr {
        Expr::constant(W, v)
    }
    fn b(op: BinaryOp, l: Expr, rr: Expr) -> Expr {
        Expr::binary(op, l, rr).unwrap()
    }
    fn eval(e: &Expr, x: u64, y: u64) -> u64 {
        let vals = BTreeMap::from([("x".to_string(), k(x)), ("y".to_string(), k(y))]);
        eval_point(e, &vals).unwrap()
    }
    fn solve_expr(e: Expr) -> Option<Candidate> {
        LinearMbaSolver.solve(&Problem::new(e, ProblemClass::LinearMba, Ctx::default()))
    }
    const SAMPLES: [(u64, u64); 6] = [
        (0, 0),
        (1, 2),
        (0xdead_beef, 0x1234_5678_9abc_def0),
        (u64::MAX, 1),
        (u64::MAX, u64::MAX),
        (0x8000_0000_0000_0000, 7),
    ];

    #[test]
    fn xor_plus_twice_and_becomes_add() {
        let e = b(
            BinaryOp::Add,
            b(BinaryOp::Xor, r("x"), r("y")),
            b(BinaryOp::Mul, k(2), b(BinaryOp::And, r("x"), r("y"))),
        );
        let c = solve_expr(e.clone()).expect("simplifies");
        assert_eq!(c.expr.node_count_bounded(100), 3, "{:?}", c.expr);
        for (x, y) in SAMPLES {
            assert_eq!(eval(&c.expr, x, y), eval(&e, x, y));
            assert_eq!(eval(&c.expr, x, y), x.wrapping_add(y));
        }
    }

    #[test]
    fn identity_difference_is_zero() {
        let sum = b(BinaryOp::Add, r("x"), r("y"));
        let e = b(
            BinaryOp::Sub,
            b(BinaryOp::Sub, sum, b(BinaryOp::Xor, r("x"), r("y"))),
            b(BinaryOp::Mul, k(2), b(BinaryOp::And, r("x"), r("y"))),
        );
        let c = solve_expr(e).expect("simplifies");
        assert_eq!(c.expr, k(0));
    }

    #[test]
    fn or_via_and_xor_collapses() {
        // (x & y) + (x ^ y) == x | y
        let e = b(
            BinaryOp::Add,
            b(BinaryOp::And, r("x"), r("y")),
            b(BinaryOp::Xor, r("x"), r("y")),
        );
        let c = solve_expr(e.clone()).expect("simplifies");
        for (x, y) in SAMPLES {
            assert_eq!(eval(&c.expr, x, y), x | y);
        }
    }

    #[test]
    fn already_minimal_passes() {
        assert!(solve_expr(b(BinaryOp::Add, r("x"), r("y"))).is_none());
    }

    #[test]
    fn nonlinear_passes() {
        assert!(solve_expr(b(BinaryOp::Mul, r("x"), r("y"))).is_none());
        let shl = b(BinaryOp::Shl, r("x"), k(1));
        assert!(solve_expr(b(BinaryOp::Add, shl, k(0))).is_none());
    }

    #[test]
    fn too_many_vars_passes() {
        let e = b(
            BinaryOp::Add,
            b(BinaryOp::Add, r("x"), r("y")),
            b(BinaryOp::Xor, r("z"), r("x")),
        );
        assert!(solve_expr(e).is_none());
    }
}
