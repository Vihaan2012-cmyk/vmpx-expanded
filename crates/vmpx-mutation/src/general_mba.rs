//! GAMBA-style general-MBA simplification.
//!
//! A non-linear MBA is rewritten by isolating its linear structure: every
//! maximal non-linear sub-expression (a product of non-constants, a shift by a
//! variable, a load, a resize, ...) is abstracted to a fresh variable, the
//! resulting purely linear expression is simplified with
//! [`crate::linear_mba::simplify_linear`], and the atoms are substituted back.
//! The atoms' own children are simplified first, and the whole thing iterates
//! to a fixpoint. A rewrite is kept only if it strictly shrinks the DAG, so a
//! genuinely non-linear core is left exactly as it was.
//!
//! Soundness: each linear step is a signature-theorem identity over the fresh
//! variables, hence true for every value of them, in particular the atoms'
//! values. [`certify_general`] turns that into a checked proof: abstract the
//! atoms of the original and of the result with one shared map and compare the
//! two linear forms by the signature theorem. Without a certificate the result
//! is only as strong as the [`PromotionGate`]'s verdict (see [`judge_general`]).

use std::collections::{BTreeMap, HashMap};

use vmpx_vm::{BinaryOp, BitWidth, Expr, ExprError, ExprKind, UnaryOp};

use crate::bvutil::{mask, registers, walk};
use crate::gate::{EquivalenceBackend, Judgement, PromotionGate, ProofKind, Tier};
use crate::linear_mba::{LinearCmp, MAX_VARS, compare_linear, simplify_linear};
use crate::sampling::judge_sampled;

const SIZE_CAP: usize = 1 << 14;
const MAX_PASSES: usize = 8;
const FUEL: u64 = 20_000;
const CERT_FUEL: u32 = 256;

#[derive(Clone, Debug)]
pub struct GeneralSimplified {
    pub expr: Expr,
    /// Number of linear sub-problems whose rewrite was adopted.
    pub linear_rewrites: usize,
    pub passes: usize,
}

impl GeneralSimplified {
    #[must_use]
    pub fn changed_from(&self, original: &Expr) -> bool {
        self.expr != *original
    }
}

/// Simplifies `e`; never fails (returns it unchanged when nothing applies).
#[must_use]
pub fn simplify_general(e: &Expr) -> GeneralSimplified {
    if e.node_count_bounded(SIZE_CAP) > SIZE_CAP {
        return GeneralSimplified {
            expr: e.clone(),
            linear_rewrites: 0,
            passes: 0,
        };
    }
    let mut prefix = String::from("__gamba");
    if let Some(regs) = registers(e) {
        while regs.keys().any(|n| n.starts_with(&prefix)) {
            prefix.insert(0, '_');
        }
    }
    let mut s = Simplifier {
        prefix,
        memo: HashMap::new(),
        fuel: FUEL,
        rewrites: 0,
    };
    let mut cur = e.clone();
    let mut passes = 0;
    while passes < MAX_PASSES {
        passes += 1;
        let next = s.go(&cur);
        if next == cur {
            break;
        }
        cur = next;
    }
    GeneralSimplified {
        expr: cur,
        linear_rewrites: s.rewrites,
        passes,
    }
}

/// Judges `result` against `original`. `Proven` (signature theorem, no solver)
/// when [`certify_general`] holds; otherwise the sampled [`PromotionGate`]
/// verdict: `Proven` only if the gate proves it, else `Observed` (or `Refuted`
/// on a concrete mismatch).
pub fn judge_general<B: EquivalenceBackend>(
    gate: &mut PromotionGate<B>,
    original: &Expr,
    result: &GeneralSimplified,
) -> Judgement {
    if result.expr != *original && certify_general(original, &result.expr) {
        return Judgement {
            tier: Tier::Proven,
            proof: Some(ProofKind::SignatureTheorem),
            counterexample: None,
            tests_agreed: 0,
        };
    }
    judge_sampled(gate, original, &result.expr)
}

/// Soundness certificate for a GAMBA-style rewrite: `true` only if `original`
/// and `simplified` are equal for every input, proven without a solver.
///
/// Both sides are abstracted with one shared atom map (each maximal non-linear
/// subterm becomes a fresh variable; the same atom is the same variable on both
/// sides) and the abstractions are compared by the linear-MBA signature
/// theorem. The atoms are opaque, so an identity over the fresh variables holds
/// for any values of them, in particular the atoms' real values.
///
/// An atom of `simplified` matches one of `original` if structurally equal, or
/// if congruent with provably-equal children (the simplifier rewrites inside
/// atoms too). Fails closed (`false`) when an atom is unmatched on either side,
/// an atom contains an intrinsic (possibly impure), widths mix, or the
/// abstraction leaves the linear class. `false` is not a refutation: abstract
/// variables are independent while the real atoms are correlated.
#[must_use]
pub fn certify_general(original: &Expr, simplified: &Expr) -> bool {
    Cert { fuel: CERT_FUEL }.equal(original, simplified)
}

struct Cert {
    fuel: u32,
}

impl Cert {
    fn equal(&mut self, a: &Expr, b: &Expr) -> bool {
        (a == b && !has_intrinsic(a)) || self.linear(a, b) || self.congruent(a, b)
    }

    /// Same constructor with pairwise provably-equal (strictly smaller) children.
    fn congruent(&mut self, a: &Expr, b: &Expr) -> bool {
        if a.width != b.width {
            return false;
        }
        match (&a.kind, &b.kind) {
            (ExprKind::Unary { op: o1, value: v1 }, ExprKind::Unary { op: o2, value: v2 }) => {
                o1 == o2 && self.equal(v1, v2)
            }
            (
                ExprKind::Binary { op: o1, left: l1, right: r1 },
                ExprKind::Binary { op: o2, left: l2, right: r2 },
            ) => o1 == o2 && self.equal(l1, l2) && self.equal(r1, r2),
            (ExprKind::Load { address: x }, ExprKind::Load { address: y }) => self.equal(x, y),
            (
                ExprKind::Extract { value: v1, lsb: s1 },
                ExprKind::Extract { value: v2, lsb: s2 },
            ) => s1 == s2 && self.equal(v1, v2),
            (ExprKind::Concat { high: h1, low: l1 }, ExprKind::Concat { high: h2, low: l2 }) => {
                self.equal(h1, h2) && self.equal(l1, l2)
            }
            (ExprKind::Resize { op: o1, value: v1 }, ExprKind::Resize { op: o2, value: v2 }) => {
                o1 == o2 && self.equal(v1, v2)
            }
            _ => false,
        }
    }

    /// Shared-map abstraction of both sides, then the signature theorem.
    fn linear(&mut self, a: &Expr, b: &Expr) -> bool {
        if a.width != b.width || self.fuel == 0 {
            return false;
        }
        self.fuel -= 1;
        let (Some(ra), Some(rb)) = (registers(a), registers(b)) else {
            return false;
        };
        let mut prefix = String::from("__cert");
        while ra.keys().chain(rb.keys()).any(|n| n.starts_with(&prefix)) {
            prefix.insert(0, '_');
        }
        let name = |i: usize, w: BitWidth| Expr::register(w, format!("{prefix}{i}"));

        let mut atoms: Vec<Expr> = Vec::new();
        let mut index: HashMap<Expr, usize> = HashMap::new();
        let mut failed = false;
        let Ok(la) = lift_with(a, a.width, false, &mut |x| {
            failed |= has_intrinsic(x);
            let i = *index.entry(x.clone()).or_insert_with(|| {
                atoms.push(x.clone());
                atoms.len() - 1
            });
            Ok(name(i, x.width))
        }) else {
            return false;
        };
        if failed {
            return false;
        }

        let mut hit = vec![false; atoms.len()];
        let Ok(lb) = lift_with(b, b.width, false, &mut |y| {
            let found = if has_intrinsic(y) {
                None
            } else if let Some(&i) = index.get(y) {
                Some(i)
            } else {
                let i = (0..atoms.len()).find(|&i| self.congruent(&atoms[i], y));
                if let Some(i) = i {
                    index.insert(y.clone(), i);
                }
                i
            };
            if let Some(i) = found {
                hit[i] = true;
                Ok(name(i, y.width))
            } else {
                failed = true;
                Ok(y.clone())
            }
        }) else {
            return false;
        };
        if failed || !hit.iter().all(|h| *h) {
            return false;
        }
        matches!(compare_linear(&la, &lb), Ok(LinearCmp::Equal))
    }
}

fn has_intrinsic(e: &Expr) -> bool {
    let mut found = false;
    walk(e, &mut |n| found |= matches!(n.kind, ExprKind::Intrinsic { .. }));
    found
}

struct Simplifier {
    prefix: String,
    memo: HashMap<Expr, Expr>,
    fuel: u64,
    rewrites: usize,
}

impl Simplifier {
    fn size(e: &Expr) -> usize {
        e.node_count_bounded(SIZE_CAP)
    }

    fn go(&mut self, e: &Expr) -> Expr {
        if matches!(e.kind, ExprKind::Const { .. } | ExprKind::Register { .. }) {
            return e.clone();
        }
        if let Some(hit) = self.memo.get(e) {
            return hit.clone();
        }
        let mut out = self.with_children(e).unwrap_or_else(|_| e.clone());
        if self.fuel > 0 {
            self.fuel -= 1;
            if let Some(whole) = self.lift_simplify(e)
                && Self::size(&whole) < Self::size(&out)
            {
                self.rewrites += 1;
                out = whole;
            }
        }
        if Self::size(&out) > Self::size(e) {
            out = e.clone();
        }
        self.memo.insert(e.clone(), out.clone());
        out
    }

    fn with_children(&mut self, e: &Expr) -> Result<Expr, ExprError> {
        Ok(match &e.kind {
            ExprKind::Const { .. } | ExprKind::Register { .. } => e.clone(),
            ExprKind::Unary { op, value } => Expr::unary(*op, self.go(value)),
            ExprKind::Binary { op, left, right } => {
                Expr::binary(*op, self.go(left), self.go(right))?
            }
            ExprKind::Load { address } => Expr::load(e.width, self.go(address)),
            ExprKind::Extract { value, lsb } => Expr::extract(e.width, *lsb, self.go(value))?,
            ExprKind::Concat { high, low } => Expr::concat(self.go(high), self.go(low))?,
            ExprKind::Resize { op, value } => Expr::resize(*op, e.width, self.go(value))?,
            ExprKind::Intrinsic { name, arguments } => {
                let args = arguments.iter().map(|a| self.go(a)).collect();
                Expr::intrinsic(e.width, name.clone(), args)
            }
        })
    }

    /// Abstract atoms, simplify the linear remainder, substitute back.
    fn lift_simplify(&mut self, e: &Expr) -> Option<Expr> {
        let mut lifter = Lifter {
            width: e.width,
            atoms: Vec::new(),
            by_expr: HashMap::new(),
        };
        let lifted = lifter.lift(self, e, false).ok()?;
        if registers(&lifted)?.len() > MAX_VARS {
            return None;
        }
        let simplified = simplify_linear(&lifted).ok()?;
        let back: BTreeMap<String, Expr> = lifter.atoms.into_iter().collect();
        simplified.expr.substitute(&back).ok()
    }
}

struct Lifter {
    width: BitWidth,
    atoms: Vec<(String, Expr)>,
    by_expr: HashMap<Expr, String>,
}

impl Lifter {
    fn atom(&mut self, s: &mut Simplifier, e: &Expr) -> Result<Expr, ExprError> {
        let inner = s.with_children(e)?;
        if let Some(name) = self.by_expr.get(&inner) {
            return Ok(Expr::register(e.width, name.clone()));
        }
        let name = format!("{}{}", s.prefix, self.atoms.len());
        self.by_expr.insert(inner.clone(), name.clone());
        self.atoms.push((name.clone(), inner));
        Ok(Expr::register(e.width, name))
    }

    fn lift(&mut self, s: &mut Simplifier, e: &Expr, bitwise: bool) -> Result<Expr, ExprError> {
        lift_with(e, self.width, bitwise, &mut |a| self.atom(s, a))
    }
}

/// Atom abstraction shared by the simplifier and the certificate. `bitwise`:
/// inside an `&|^~` tree, where only variables, 0, -1 and bitwise nodes belong
/// to the linear class. Every other maximal subterm goes through `atom`.
fn lift_with(
    e: &Expr,
    width: BitWidth,
    bitwise: bool,
    atom: &mut dyn FnMut(&Expr) -> Result<Expr, ExprError>,
) -> Result<Expr, ExprError> {
    if e.width != width {
        return atom(e);
    }
    let m = mask(width);
    match &e.kind {
        ExprKind::Register { .. } => Ok(e.clone()),
        ExprKind::Const { value } => {
            if !bitwise || *value & m == 0 || *value & m == m {
                Ok(e.clone())
            } else {
                atom(e)
            }
        }
        ExprKind::Unary { op, value } => match (op, bitwise) {
            (UnaryOp::Not, _) | (UnaryOp::Neg, false) => {
                Ok(Expr::unary(*op, lift_with(value, width, bitwise, atom)?))
            }
            _ => atom(e),
        },
        ExprKind::Binary { op, left, right } => match op {
            BinaryOp::And | BinaryOp::Or | BinaryOp::Xor => {
                let l = lift_with(left, width, true, atom)?;
                let r = lift_with(right, width, true, atom)?;
                Expr::binary(*op, l, r)
            }
            BinaryOp::Add | BinaryOp::Sub if !bitwise => {
                let l = lift_with(left, width, false, atom)?;
                let r = lift_with(right, width, false, atom)?;
                Expr::binary(*op, l, r)
            }
            BinaryOp::Mul if !bitwise => match (&left.kind, &right.kind) {
                (ExprKind::Const { .. }, _) => Expr::binary(
                    *op,
                    (**left).clone(),
                    lift_with(right, width, false, atom)?,
                ),
                (_, ExprKind::Const { .. }) => Expr::binary(
                    *op,
                    lift_with(left, width, false, atom)?,
                    (**right).clone(),
                ),
                _ => atom(e),
            },
            BinaryOp::Shl if !bitwise => match right.kind {
                ExprKind::Const { value } if value < u64::from(width.bits()) => Expr::binary(
                    *op,
                    lift_with(left, width, false, atom)?,
                    (**right).clone(),
                ),
                _ => atom(e),
            },
            _ => atom(e),
        },
        _ => atom(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate::{NoSolver, ProofKind, SmtVerdict, Tier};
    use BinaryOp::{Add, And, Mul, Or, Shr, Sub, Xor};

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
    /// (x|y) - (x&y) == x^y
    fn obf_xor(x: Expr, y: Expr) -> Expr {
        b(Sub, b(Or, x.clone(), y.clone()), b(And, x, y))
    }
    fn size(e: &Expr) -> usize {
        e.node_count_bounded(1 << 20)
    }

    #[test]
    fn linear_subparts_of_a_nonlinear_mba_simplify() {
        // (x*y) * ((x|y) - (x&y))  ->  (x*y) * (x^y)
        let core = b(Mul, v("x"), v("y"));
        let obf = b(Mul, core.clone(), obf_xor(v("x"), v("y")));
        assert!(simplify_linear(&obf).is_err());
        let r = simplify_general(&obf);
        assert_eq!(r.expr, b(Mul, core, b(Xor, v("x"), v("y"))));
        assert!(r.linear_rewrites >= 1);
        assert!(size(&r.expr) < size(&obf));
    }

    #[test]
    fn linear_structure_around_an_atom_cancels() {
        // (x*y) + ((x|y)-(x&y)) - (x^y)  ->  x*y
        let obf = b(
            Sub,
            b(Add, b(Mul, v("x"), v("y")), obf_xor(v("x"), v("y"))),
            b(Xor, v("x"), v("y")),
        );
        assert_eq!(simplify_general(&obf).expr, b(Mul, v("x"), v("y")));
    }

    #[test]
    fn nested_atoms_iterate_to_a_fixpoint() {
        // ((x ^ y) + 2*(x&y)) * (z*z)
        let add = b(Add, b(Xor, v("x"), v("y")), b(Mul, c(2), b(And, v("x"), v("y"))));
        let obf = b(Mul, add, b(Mul, v("z"), v("z")));
        let r = simplify_general(&obf);
        assert!(size(&r.expr) < size(&obf));
        assert_eq!(simplify_general(&r.expr).expr, r.expr);
    }

    #[test]
    fn genuinely_nonlinear_core_is_left_intact() {
        for e in [
            b(Mul, v("x"), v("y")),
            b(Xor, b(Mul, v("x"), v("y")), b(Shr, v("x"), v("y"))),
            b(And, v("x"), c(0xFF)),
            b(Add, b(Mul, v("x"), v("y")), v("z")),
        ] {
            let r = simplify_general(&e);
            assert_eq!(r.expr, e, "must not touch an already-minimal nonlinear core");
            assert_eq!(r.linear_rewrites, 0);
        }
    }

    fn hand(expr: Expr) -> GeneralSimplified {
        GeneralSimplified {
            expr,
            linear_rewrites: 0,
            passes: 0,
        }
    }

    #[test]
    fn unmatched_atoms_fall_back_to_observed_or_smt() {
        struct Unsat;
        impl EquivalenceBackend for Unsat {
            fn check_inequality(&self, _: &Expr, _: &Expr) -> SmtVerdict {
                SmtVerdict::Unsat
            }
        }
        // x*x rewritten through a different nonlinear atom
        let x1 = b(Add, v("x"), c(1));
        let cand = b(Sub, b(Sub, b(Mul, x1.clone(), x1), b(Mul, c(2), v("x"))), c(1));
        let orig = b(Mul, v("x"), v("x"));
        assert!(!certify_general(&orig, &cand));
        let mut gate = PromotionGate::<NoSolver>::without_solver();
        let j = judge_general(&mut gate, &orig, &hand(cand.clone()));
        assert_eq!((j.tier, j.proof), (Tier::Observed, None));
        let j = judge_general(&mut PromotionGate::with_backend(Unsat), &orig, &hand(cand));
        assert_eq!((j.tier, j.proof), (Tier::Proven, Some(ProofKind::Smt)));
    }

    #[test]
    fn linear_identity_over_opaque_atoms_is_proven_without_a_solver() {
        // (x*y) + ((a|b)-(a&b)) - (b^a)  ->  x*y
        let obf = b(
            Sub,
            b(Add, b(Mul, v("x"), v("y")), obf_xor(v("a"), v("b"))),
            b(Xor, v("b"), v("a")),
        );
        let r = simplify_general(&obf);
        assert_eq!(r.expr, b(Mul, v("x"), v("y")));
        let mut gate = PromotionGate::<NoSolver>::without_solver();
        let j = judge_general(&mut gate, &obf, &r);
        assert_eq!((j.tier, j.proof), (Tier::Proven, Some(ProofKind::SignatureTheorem)));
    }

    #[test]
    fn rewrites_inside_atoms_are_certified_by_congruence() {
        // (x*y) * ((x|y)-(x&y))  ->  (x*y) * (x^y)
        let obf = b(Mul, b(Mul, v("x"), v("y")), obf_xor(v("x"), v("y")));
        let r = simplify_general(&obf);
        assert_ne!(r.expr, obf);
        assert!(certify_general(&obf, &r.expr));
        let mut gate = PromotionGate::<NoSolver>::without_solver();
        let j = judge_general(&mut gate, &obf, &r);
        assert_eq!((j.tier, j.proof), (Tier::Proven, Some(ProofKind::SignatureTheorem)));
    }

    #[test]
    fn wrong_rewrites_are_never_certified() {
        let (x, y) = (v("x"), v("y"));
        let xy = b(Mul, x.clone(), y.clone());
        let orig = b(Add, xy.clone(), obf_xor(x.clone(), y.clone()));
        // same atom, wrong linear part
        let bad = b(Add, xy, b(And, x.clone(), y.clone()));
        assert!(!certify_general(&orig, &bad));
        let mut gate = PromotionGate::<NoSolver>::without_solver();
        assert_eq!(judge_general(&mut gate, &orig, &hand(bad)).tier, Tier::Refuted);
        // right linear part, wrong atom
        let bad = b(Add, b(Mul, x.clone(), x.clone()), b(Xor, x.clone(), y.clone()));
        assert!(!certify_general(&orig, &bad));
        // dropped atom
        assert!(!certify_general(&orig, &b(Xor, x, y)));
    }

    #[test]
    fn atoms_with_intrinsics_are_not_certified() {
        let io = Expr::intrinsic(W, "rdtsc", vec![v("x")]);
        let obf = b(Add, io.clone(), obf_xor(v("x"), v("y")));
        let cand = b(Add, io, b(Xor, v("x"), v("y")));
        assert!(!certify_general(&obf, &cand));
    }

    #[test]
    fn agrees_with_the_original_on_concrete_inputs() {
        let obf = b(
            Add,
            b(Mul, obf_xor(v("x"), v("y")), b(Mul, v("x"), v("z"))),
            b(Sub, b(Or, v("y"), v("z")), b(And, v("y"), v("z"))),
        );
        let r = simplify_general(&obf);
        let none = BTreeMap::new();
        let mut s = 7_u64;
        for _ in 0..200 {
            let env: BTreeMap<String, u64> = ["x", "y", "z"]
                .iter()
                .map(|n| ((*n).to_owned(), crate::sampling::next_rand(&mut s)))
                .collect();
            let ev = |e: &Expr| crate::bvutil::eval_concrete(e, &env, &none);
            assert!(ev(&obf).is_some());
            assert_eq!(ev(&obf), ev(&r.expr));
        }
        assert!(certify_general(&obf, &r.expr));
    }
}
