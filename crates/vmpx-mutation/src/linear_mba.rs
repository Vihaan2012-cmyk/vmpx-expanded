//! Linear-MBA simplification by signature vector (SiMBA-style).
//!
//! A *linear MBA* over `w`-bit words is a sum `c0 + Σ ci * fi(x1..xt)` where each
//! `fi` is a bitwise function (`& | ^ ~`) of variables, with integer `ci`, all
//! mod `2^w`.
//!
//! # Soundness (why the signature vector decides equality)
//!
//! Let `m_J(x)` be the word whose bit `k` is 1 iff the set of variables whose bit
//! `k` is 1 is exactly `J`. The `m_J` are disjoint and `Σ_J m_J = -1`. A bitwise
//! `f` acts independently per bit, so `f(x) = Σ_J f(J) * m_J(x)` with
//! `f(J) ∈ {0,1}` the truth-table entry. Evaluate `f` on the *row* where
//! variables in `J` are the all-ones word and the others are 0: that gives
//! `-f(J)`. Linearity then yields, for the whole expression `e` with row value
//! `v_J = e(row J)`: `e(x) = -Σ_J v_J * m_J(x)  (mod 2^w)`. A constant `c` also fits
//! (`-Σ_J (-c)... = c` since `Σ m_J = -1`). So `e` is determined by its `2^t`
//! signature `(v_J)`, and equal signatures imply equal functions; conversely the
//! rows are actual inputs, so differing signatures give a concrete
//! counterexample. This is a proof, independent of any solver.
//!
//! Anything outside the class (multiplying two variables, arithmetic inside a
//! bitwise op, constants that are not 0/-1 inside a bitwise op, loads, resizes,
//! intrinsics, more than [`MAX_VARS`] variables) is rejected, never guessed at.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;

use vmpx_vm::{BinaryOp, BitWidth, Expr, ExprError, ExprKind, UnaryOp};

use crate::bvutil::mask;

/// Signature length is `2^t`; keep it small and decidable.
pub const MAX_VARS: usize = 6;
const MAX_NODES: usize = 4096;

#[derive(Debug)]
pub enum MbaError {
    NotLinear(&'static str),
    TooManyVars(usize),
    Expr(ExprError),
    /// Internal self-check failed (never returns an unverified result).
    SelfCheck,
}

impl fmt::Display for MbaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotLinear(why) => write!(f, "not a linear MBA: {why}"),
            Self::TooManyVars(n) => write!(f, "{n} variables exceeds the {MAX_VARS} cap"),
            Self::Expr(e) => write!(f, "{e}"),
            Self::SelfCheck => f.write_str("simplified form failed its signature self-check"),
        }
    }
}

impl std::error::Error for MbaError {}

impl From<ExprError> for MbaError {
    fn from(e: ExprError) -> Self {
        Self::Expr(e)
    }
}

enum L {
    C(u64),
    V(String),
    Not(Box<L>),
    And(Box<L>, Box<L>),
    Or(Box<L>, Box<L>),
    Xor(Box<L>, Box<L>),
    Add(Box<L>, Box<L>),
    Sub(Box<L>, Box<L>),
    Neg(Box<L>),
    MulC(u64, Box<L>),
}

struct Parser {
    width: BitWidth,
    nodes: usize,
    vars: BTreeSet<String>,
}

impl Parser {
    fn tick(&mut self) -> Result<(), MbaError> {
        self.nodes += 1;
        if self.nodes > MAX_NODES {
            return Err(MbaError::NotLinear("expression too large"));
        }
        Ok(())
    }

    fn lin(&mut self, e: &Expr) -> Result<L, MbaError> {
        self.tick()?;
        if e.width != self.width {
            return Err(MbaError::NotLinear("mixed widths"));
        }
        let m = mask(self.width);
        match &e.kind {
            ExprKind::Const { value } => Ok(L::C(*value & m)),
            ExprKind::Register { name } => {
                self.vars.insert(name.clone());
                Ok(L::V(name.clone()))
            }
            ExprKind::Unary {
                op: UnaryOp::Neg,
                value,
            } => Ok(L::Neg(Box::new(self.lin(value)?))),
            // ~e == -1 - e (mod 2^w), so Not stays inside the linear class.
            ExprKind::Unary {
                op: UnaryOp::Not,
                value,
            } => Ok(L::Sub(Box::new(L::C(m)), Box::new(self.lin(value)?))),
            ExprKind::Binary { op, left, right } => match op {
                BinaryOp::Add => Ok(L::Add(
                    Box::new(self.lin(left)?),
                    Box::new(self.lin(right)?),
                )),
                BinaryOp::Sub => Ok(L::Sub(
                    Box::new(self.lin(left)?),
                    Box::new(self.lin(right)?),
                )),
                BinaryOp::Mul => match (&left.kind, &right.kind) {
                    (ExprKind::Const { value }, _) => {
                        Ok(L::MulC(*value & m, Box::new(self.lin(right)?)))
                    }
                    (_, ExprKind::Const { value }) => {
                        Ok(L::MulC(*value & m, Box::new(self.lin(left)?)))
                    }
                    _ => Err(MbaError::NotLinear("product of non-constants")),
                },
                BinaryOp::Shl => match right.kind {
                    ExprKind::Const { value } if value < u64::from(self.width.bits()) => {
                        Ok(L::MulC(1_u64 << value & m, Box::new(self.lin(left)?)))
                    }
                    _ => Err(MbaError::NotLinear("non-constant or oversized shift")),
                },
                BinaryOp::And | BinaryOp::Or | BinaryOp::Xor => self.bit(e),
                _ => Err(MbaError::NotLinear("unsupported operator")),
            },
            _ => Err(MbaError::NotLinear("unsupported node")),
        }
    }

    fn bit(&mut self, e: &Expr) -> Result<L, MbaError> {
        self.tick()?;
        if e.width != self.width {
            return Err(MbaError::NotLinear("mixed widths"));
        }
        let m = mask(self.width);
        match &e.kind {
            ExprKind::Register { name } => {
                self.vars.insert(name.clone());
                Ok(L::V(name.clone()))
            }
            ExprKind::Const { value } if *value & m == 0 || *value & m == m => {
                Ok(L::C(*value & m))
            }
            ExprKind::Unary {
                op: UnaryOp::Not,
                value,
            } => Ok(L::Not(Box::new(self.bit(value)?))),
            ExprKind::Binary { op, left, right }
                if matches!(op, BinaryOp::And | BinaryOp::Or | BinaryOp::Xor) =>
            {
                let (a, b) = (Box::new(self.bit(left)?), Box::new(self.bit(right)?));
                Ok(match op {
                    BinaryOp::And => L::And(a, b),
                    BinaryOp::Or => L::Or(a, b),
                    _ => L::Xor(a, b),
                })
            }
            _ => Err(MbaError::NotLinear("non-bitwise node inside bitwise op")),
        }
    }
}

fn parse(e: &Expr) -> Result<(L, BTreeSet<String>), MbaError> {
    let mut p = Parser {
        width: e.width,
        nodes: 0,
        vars: BTreeSet::new(),
    };
    let l = p.lin(e)?;
    Ok((l, p.vars))
}

fn eval(l: &L, row: &BTreeMap<&str, u64>, m: u64) -> u64 {
    match l {
        L::C(c) => *c,
        L::V(n) => row[n.as_str()],
        L::Not(a) => !eval(a, row, m) & m,
        L::And(a, b) => eval(a, row, m) & eval(b, row, m),
        L::Or(a, b) => eval(a, row, m) | eval(b, row, m),
        L::Xor(a, b) => eval(a, row, m) ^ eval(b, row, m),
        L::Add(a, b) => eval(a, row, m).wrapping_add(eval(b, row, m)) & m,
        L::Sub(a, b) => eval(a, row, m).wrapping_sub(eval(b, row, m)) & m,
        L::Neg(a) => eval(a, row, m).wrapping_neg() & m,
        L::MulC(c, a) => c.wrapping_mul(eval(a, row, m)) & m,
    }
}

/// `v_J` for every `J` over `vars` (bit `i` of `J` = `vars[i]` is all-ones).
fn signature_of(l: &L, vars: &[String], width: BitWidth) -> Vec<u64> {
    let m = mask(width);
    (0..1_usize << vars.len())
        .map(|j| {
            let row: BTreeMap<&str, u64> = vars
                .iter()
                .enumerate()
                .map(|(i, n)| (n.as_str(), if j >> i & 1 == 1 { m } else { 0 }))
                .collect();
            eval(l, &row, m)
        })
        .collect()
}

fn check_vars(vars: &BTreeSet<String>) -> Result<Vec<String>, MbaError> {
    if vars.len() > MAX_VARS {
        return Err(MbaError::TooManyVars(vars.len()));
    }
    Ok(vars.iter().cloned().collect())
}

/// Signature vector of a linear MBA over its own (sorted) variables.
///
/// # Errors
///
/// Fails closed with [`MbaError`] if `e` is outside the linear class.
pub fn signature(e: &Expr) -> Result<(Vec<String>, Vec<u64>), MbaError> {
    let (l, vars) = parse(e)?;
    let vars = check_vars(&vars)?;
    let sig = signature_of(&l, &vars, e.width);
    Ok((vars, sig))
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LinearCmp {
    /// Proven equal as functions (theorem, no solver).
    Equal,
    /// Not equal; the assignment is a real input on which they differ.
    Differ(BTreeMap<String, u64>),
}

/// Decides equality of two linear MBAs by signature vector.
///
/// # Errors
///
/// Fails closed if either side is outside the class or widths differ.
pub fn compare_linear(a: &Expr, b: &Expr) -> Result<LinearCmp, MbaError> {
    if a.width != b.width {
        return Err(MbaError::NotLinear("width mismatch"));
    }
    let ((la, va), (lb, vb)) = (parse(a)?, parse(b)?);
    let vars = check_vars(&va.union(&vb).cloned().collect())?;
    let (sa, sb) = (
        signature_of(&la, &vars, a.width),
        signature_of(&lb, &vars, b.width),
    );
    let m = mask(a.width);
    Ok(match (0..sa.len()).find(|&j| sa[j] != sb[j]) {
        None => LinearCmp::Equal,
        Some(j) => LinearCmp::Differ(
            vars.iter()
                .enumerate()
                .map(|(i, n)| (n.clone(), if j >> i & 1 == 1 { m } else { 0 }))
                .collect(),
        ),
    })
}

#[derive(Clone, Debug)]
pub struct LinearSimplified {
    pub expr: Expr,
    pub vars: Vec<String>,
    pub signature: Vec<u64>,
}

/// Rewrites a linear MBA to its canonical conjunction-basis form (or a single
/// scaled bitwise term when that is smaller). The result is re-parsed and its
/// signature checked equal to the input's before being returned.
///
/// # Errors
///
/// [`MbaError::NotLinear`]/[`MbaError::TooManyVars`] outside the class;
/// [`MbaError::SelfCheck`] if the verification of the rewrite fails.
pub fn simplify_linear(e: &Expr) -> Result<LinearSimplified, MbaError> {
    let (vars, sig) = signature(e)?;
    let w = e.width;
    let mut best = canonical(w, &vars, &sig)?;
    if let Some(single) = single_term(w, &vars, &sig)?
        && single.node_count_bounded(10_000) < best.node_count_bounded(10_000)
    {
        best = single;
    }
    let (lb, vb) = parse(&best).map_err(|_| MbaError::SelfCheck)?;
    if !vb.iter().all(|v| vars.contains(v)) || signature_of(&lb, &vars, w) != sig {
        return Err(MbaError::SelfCheck);
    }
    Ok(LinearSimplified {
        expr: best,
        vars,
        signature: sig,
    })
}

fn sext(c: u64, w: BitWidth) -> i64 {
    let sh = 64 - w.bits();
    ((c << sh) as i64) >> sh
}

fn and_chain(w: BitWidth, vars: &[String], subset: usize) -> Result<Expr, MbaError> {
    let mut acc: Option<Expr> = None;
    for (i, n) in vars.iter().enumerate() {
        if subset >> i & 1 == 1 {
            let v = Expr::register(w, n.clone());
            acc = Some(match acc {
                None => v,
                Some(a) => Expr::binary(BinaryOp::And, a, v)?,
            });
        }
    }
    acc.ok_or(MbaError::SelfCheck)
}

/// `(negative, |c| * term)` for a coefficient `c`.
fn scaled(w: BitWidth, c: u64, term: Expr) -> Result<(bool, Expr), MbaError> {
    let s = sext(c, w);
    let mag = s.unsigned_abs() & mask(w);
    let t = if mag == 1 {
        term
    } else {
        Expr::binary(BinaryOp::Mul, Expr::constant(w, mag), term)?
    };
    Ok((s < 0, t))
}

fn accumulate(acc: Option<Expr>, neg: bool, t: Expr) -> Result<Expr, MbaError> {
    Ok(match (acc, neg) {
        (None, false) => t,
        (None, true) => Expr::unary(UnaryOp::Neg, t),
        (Some(a), false) => Expr::binary(BinaryOp::Add, a, t)?,
        (Some(a), true) => Expr::binary(BinaryOp::Sub, a, t)?,
    })
}

/// Conjunction-basis form: `v_0 + Σ_{S != {}} a_S * AND(S)`, `a` = Möbius
/// inverse of `-v` over the subset lattice (exact over any `Z/2^w`).
fn canonical(w: BitWidth, vars: &[String], sig: &[u64]) -> Result<Expr, MbaError> {
    let m = mask(w);
    let mut a: Vec<u64> = sig.iter().map(|v| v.wrapping_neg() & m).collect();
    for i in 0..vars.len() {
        for j in 0..a.len() {
            if j >> i & 1 == 1 {
                a[j] = a[j].wrapping_sub(a[j ^ (1 << i)]) & m;
            }
        }
    }
    let mut acc: Option<Expr> = None;
    for (s, &c) in a.iter().enumerate().skip(1) {
        if c != 0 {
            let (neg, t) = scaled(w, c, and_chain(w, vars, s)?)?;
            acc = Some(accumulate(acc, neg, t)?);
        }
    }
    let c0 = sig[0];
    match (acc, c0) {
        (None, _) => Ok(Expr::constant(w, c0)),
        (Some(a), 0) => Ok(a),
        (Some(a), _) => {
            let s = sext(c0, w);
            accumulate(Some(a), s < 0, Expr::constant(w, s.unsigned_abs()))
        }
    }
}

#[derive(Clone)]
enum Bx {
    Var(usize),
    Not(Box<Bx>),
    And(Box<Bx>, Box<Bx>),
    Or(Box<Bx>, Box<Bx>),
    Xor(Box<Bx>, Box<Bx>),
}

impl Bx {
    fn to_expr(&self, w: BitWidth, vars: &[String]) -> Result<Expr, MbaError> {
        Ok(match self {
            Self::Var(i) => Expr::register(w, vars[*i].clone()),
            Self::Not(a) => Expr::unary(UnaryOp::Not, a.to_expr(w, vars)?),
            Self::And(a, b) => {
                Expr::binary(BinaryOp::And, a.to_expr(w, vars)?, b.to_expr(w, vars)?)?
            }
            Self::Or(a, b) => Expr::binary(BinaryOp::Or, a.to_expr(w, vars)?, b.to_expr(w, vars)?)?,
            Self::Xor(a, b) => {
                Expr::binary(BinaryOp::Xor, a.to_expr(w, vars)?, b.to_expr(w, vars)?)?
            }
        })
    }
}

/// Smallest bitwise formula for every truth table over `t <= 3` variables.
fn bitwise_table(t: usize) -> HashMap<u32, (usize, Bx)> {
    let rows = 1_usize << t;
    let full = (1_u32 << rows) - 1;
    let mut best: HashMap<u32, (usize, Bx)> = HashMap::new();
    for i in 0..t {
        let tt = (0..rows).fold(0_u32, |acc, j| acc | (u32::from(j >> i & 1 == 1) << j));
        best.insert(tt, (1, Bx::Var(i)));
    }
    loop {
        let mut snap: Vec<(u32, usize, Bx)> =
            best.iter().map(|(k, (c, e))| (*k, *c, e.clone())).collect();
        snap.sort_unstable_by_key(|x| (x.1, x.0));
        let mut changed = false;
        let mut offer = |tt: u32, cost: usize, e: Bx, best: &mut HashMap<u32, (usize, Bx)>| {
            if best.get(&tt).is_none_or(|(c, _)| cost < *c) {
                best.insert(tt, (cost, e));
                changed = true;
            }
        };
        for (k, c, e) in &snap {
            offer(!k & full, c + 1, Bx::Not(Box::new(e.clone())), &mut best);
        }
        for (ka, ca, ea) in &snap {
            for (kb, cb, eb) in &snap {
                let cost = ca + cb + 1;
                let (x, y) = (Box::new(ea.clone()), Box::new(eb.clone()));
                offer(ka & kb, cost, Bx::And(x.clone(), y.clone()), &mut best);
                offer(ka | kb, cost, Bx::Or(x.clone(), y.clone()), &mut best);
                offer(ka ^ kb, cost, Bx::Xor(x, y), &mut best);
            }
        }
        if !changed {
            return best;
        }
    }
}

/// `k * F(x)` when every non-zero `-v_J` is the same `k` and `F` is a bitwise
/// function of at most 3 variables.
fn single_term(w: BitWidth, vars: &[String], sig: &[u64]) -> Result<Option<Expr>, MbaError> {
    if vars.is_empty() || vars.len() > 3 {
        return Ok(None);
    }
    let m = mask(w);
    let wv: Vec<u64> = sig.iter().map(|v| v.wrapping_neg() & m).collect();
    let nz: BTreeSet<u64> = wv.iter().copied().filter(|v| *v != 0).collect();
    let (Some(&k), 1) = (nz.iter().next(), nz.len()) else {
        return Ok(None);
    };
    let tt = wv
        .iter()
        .enumerate()
        .fold(0_u32, |acc, (j, v)| acc | (u32::from(*v != 0) << j));
    if tt == (1_u32 << wv.len()) - 1 {
        return Ok(None);
    }
    let table = bitwise_table(vars.len());
    let Some((_, f)) = table.get(&tt) else {
        return Ok(None);
    };
    let (neg, t) = scaled(w, k, f.to_expr(w, vars)?)?;
    accumulate(None, neg, t).map(Some)
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: BitWidth = BitWidth::W64;

    fn v(n: &str) -> Expr {
        Expr::register(W, n)
    }
    fn c(x: u64) -> Expr {
        Expr::constant(W, x)
    }
    fn b(op: BinaryOp, l: Expr, r: Expr) -> Expr {
        Expr::binary(op, l, r).unwrap()
    }

    #[test]
    fn xor_identity_reduces_and_signatures_match() {
        // (x | y) - (x & y) == x ^ y
        let obf = b(
            BinaryOp::Sub,
            b(BinaryOp::Or, v("x"), v("y")),
            b(BinaryOp::And, v("x"), v("y")),
        );
        let r = simplify_linear(&obf).unwrap();
        assert_eq!(r.expr, b(BinaryOp::Xor, v("x"), v("y")));
        assert_eq!(
            compare_linear(&obf, &b(BinaryOp::Xor, v("x"), v("y"))).unwrap(),
            LinearCmp::Equal
        );
        assert_eq!(signature(&r.expr).unwrap().1, r.signature);
    }

    #[test]
    fn add_via_xor_and_reduces_to_add() {
        // (x ^ y) + 2*(x & y) == x + y
        let obf = b(
            BinaryOp::Add,
            b(BinaryOp::Xor, v("x"), v("y")),
            b(BinaryOp::Mul, c(2), b(BinaryOp::And, v("x"), v("y"))),
        );
        let r = simplify_linear(&obf).unwrap();
        assert_eq!(
            compare_linear(&r.expr, &b(BinaryOp::Add, v("x"), v("y"))).unwrap(),
            LinearCmp::Equal
        );
        assert!(r.expr.node_count_bounded(100) <= obf.node_count_bounded(100));
    }

    #[test]
    fn not_and_constants_fold() {
        // ~x + 1 == -x ; x - x + 5 == 5
        let neg = b(BinaryOp::Add, Expr::unary(UnaryOp::Not, v("x")), c(1));
        let r = simplify_linear(&neg).unwrap();
        assert_eq!(
            compare_linear(&r.expr, &Expr::unary(UnaryOp::Neg, v("x"))).unwrap(),
            LinearCmp::Equal
        );
        let five = b(BinaryOp::Add, b(BinaryOp::Sub, v("x"), v("x")), c(5));
        assert_eq!(simplify_linear(&five).unwrap().expr, c(5));
    }

    #[test]
    fn differing_functions_yield_a_real_counterexample() {
        let (lhs, rhs) = (b(BinaryOp::Or, v("x"), v("y")), b(BinaryOp::Xor, v("x"), v("y")));
        let LinearCmp::Differ(cex) = compare_linear(&lhs, &rhs).unwrap() else {
            panic!("expected a counterexample");
        };
        let env: BTreeMap<String, u64> = cex;
        let ev = |e: &Expr| crate::bvutil::eval_concrete(e, &env, &BTreeMap::new()).unwrap();
        assert_ne!(ev(&lhs), ev(&rhs));
    }

    #[test]
    fn simplification_agrees_with_concrete_evaluation() {
        let obf = b(
            BinaryOp::Add,
            b(BinaryOp::Mul, c(3), b(BinaryOp::Or, v("x"), v("y"))),
            b(
                BinaryOp::Sub,
                b(BinaryOp::Xor, v("y"), v("z")),
                b(BinaryOp::Mul, c(7), b(BinaryOp::And, v("x"), Expr::unary(UnaryOp::Not, v("z")))),
            ),
        );
        let r = simplify_linear(&obf).unwrap();
        let mut s = 0x9E37_79B9_7F4A_7C15_u64;
        for _ in 0..200 {
            let mut next = || {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                s
            };
            let env: BTreeMap<String, u64> =
                ["x", "y", "z"].iter().map(|n| ((*n).to_owned(), next())).collect();
            let ev = |e: &Expr| crate::bvutil::eval_concrete(e, &env, &BTreeMap::new());
            assert_eq!(ev(&obf), ev(&r.expr));
            assert!(ev(&obf).is_some());
        }
    }

    #[test]
    fn nonlinear_is_rejected_not_guessed() {
        let xy = b(BinaryOp::Mul, v("x"), v("y"));
        assert!(matches!(simplify_linear(&xy), Err(MbaError::NotLinear(_))));
        let masked = b(BinaryOp::And, v("x"), c(0xFF));
        assert!(matches!(simplify_linear(&masked), Err(MbaError::NotLinear(_))));
        let arith_in_bit = b(BinaryOp::And, b(BinaryOp::Add, v("x"), v("y")), v("z"));
        assert!(matches!(simplify_linear(&arith_in_bit), Err(MbaError::NotLinear(_))));
    }
}
