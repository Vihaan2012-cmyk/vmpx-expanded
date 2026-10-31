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
//! values. The result as a whole is nevertheless reported as unproven here:
//! confirmation is the [`PromotionGate`]'s job (see [`judge_general`]).

use std::collections::{BTreeMap, HashMap};

use vmpx_vm::{BinaryOp, BitWidth, Expr, ExprError, ExprKind, UnaryOp};

use crate::bvutil::{mask, registers};
use crate::gate::{EquivalenceBackend, Judgement, PromotionGate};
use crate::linear_mba::{MAX_VARS, simplify_linear};
use crate::sampling::judge_sampled;

const SIZE_CAP: usize = 1 << 14;
const MAX_PASSES: usize = 8;
const FUEL: u64 = 20_000;

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

/// Judges `result` against `original` through `gate` on sampled inputs. The
/// answer is `Proven` only if the gate proves it (structurally, by theorem, or
/// by its backend); otherwise `Observed`.
pub fn judge_general<B: EquivalenceBackend>(
    gate: &mut PromotionGate<B>,
    original: &Expr,
    result: &GeneralSimplified,
) -> Judgement {
    judge_sampled(gate, original, &result.expr)
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

    /// `bitwise`: inside an `&|^~` tree, where only variables, 0, -1 and bitwise
    /// nodes belong to the linear class.
    fn lift(&mut self, s: &mut Simplifier, e: &Expr, bitwise: bool) -> Result<Expr, ExprError> {
        if e.width != self.width {
            return self.atom(s, e);
        }
        let m = mask(self.width);
        match &e.kind {
            ExprKind::Register { .. } => Ok(e.clone()),
            ExprKind::Const { value } => {
                if !bitwise || *value & m == 0 || *value & m == m {
                    Ok(e.clone())
                } else {
                    self.atom(s, e)
                }
            }
            ExprKind::Unary { op, value } => match (op, bitwise) {
                (UnaryOp::Not, _) | (UnaryOp::Neg, false) => {
                    Ok(Expr::unary(*op, self.lift(s, value, bitwise)?))
                }
                _ => self.atom(s, e),
            },
            ExprKind::Binary { op, left, right } => match op {
                BinaryOp::And | BinaryOp::Or | BinaryOp::Xor => {
                    let (l, r) = (self.lift(s, left, true)?, self.lift(s, right, true)?);
                    Expr::binary(*op, l, r)
                }
                BinaryOp::Add | BinaryOp::Sub if !bitwise => {
                    let (l, r) = (self.lift(s, left, false)?, self.lift(s, right, false)?);
                    Expr::binary(*op, l, r)
                }
                BinaryOp::Mul if !bitwise => match (&left.kind, &right.kind) {
                    (ExprKind::Const { .. }, _) => {
                        Expr::binary(*op, (**left).clone(), self.lift(s, right, false)?)
                    }
                    (_, ExprKind::Const { .. }) => {
                        Expr::binary(*op, self.lift(s, left, false)?, (**right).clone())
                    }
                    _ => self.atom(s, e),
                },
                BinaryOp::Shl if !bitwise => match right.kind {
                    ExprKind::Const { value } if value < u64::from(self.width.bits()) => {
                        Expr::binary(*op, self.lift(s, left, false)?, (**right).clone())
                    }
                    _ => self.atom(s, e),
                },
                _ => self.atom(s, e),
            },
            _ => self.atom(s, e),
        }
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

    #[test]
    fn result_is_observed_without_a_solver_and_proven_with_unsat() {
        struct Unsat;
        impl EquivalenceBackend for Unsat {
            fn check_inequality(&self, _: &Expr, _: &Expr) -> SmtVerdict {
                SmtVerdict::Unsat
            }
        }
        let obf = b(Mul, b(Mul, v("x"), v("y")), obf_xor(v("x"), v("y")));
        let r = simplify_general(&obf);
        let j = judge_general(&mut PromotionGate::<NoSolver>::without_solver(), &obf, &r);
        assert_eq!((j.tier, j.proof), (Tier::Observed, None));
        let j = judge_general(&mut PromotionGate::with_backend(Unsat), &obf, &r);
        assert_eq!((j.tier, j.proof), (Tier::Proven, Some(ProofKind::Smt)));
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
    }
}
