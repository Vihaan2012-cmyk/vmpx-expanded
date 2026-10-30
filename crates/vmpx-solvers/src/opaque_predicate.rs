//! Opaque-predicate prover (solver).
//!
//! Sound by construction: the condition is first rewritten bottom-up with a few
//! proven algebraic identities (exact constants, or over-approximating
//! "even"/"square" shapes built from fresh unconstrained registers), then
//! `vmpx_vm::value_range` must prove a single value. Anything else passes.

use std::collections::HashMap;
use std::time::Instant;

use crate::Solver;
use vmpx_mutation::{Candidate, Cost, Problem, ProblemClass, SolverId};
use vmpx_vm::{BinaryOp, Expr, ExprKind, ExprRef, UnaryOp, value_range};

const MAX_DEPTH: usize = 256;

/// Proves a branch condition constant and folds the dead arm.
pub struct OpaquePredicateSolver;

impl Solver for OpaquePredicateSolver {
    fn id(&self) -> SolverId {
        SolverId::new("opaque_predicate")
    }

    fn applies_to(&self, class: ProblemClass) -> bool {
        class == ProblemClass::OpaquePredicate
    }

    fn solve(&self, p: &Problem) -> Option<Candidate> {
        if matches!(p.expr.kind, ExprKind::Const { .. }) {
            return None;
        }
        let start = Instant::now();
        let mut rw = Rewriter::default();
        let proven = rw.rewrite(&p.expr, 0);
        let value = value_range(&proven).as_constant()?;
        Some(Candidate {
            expr: Expr::constant(p.expr.width, value),
            by: self.id(),
            cost: Cost {
                time_ns: u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX),
                steps: 1,
            },
        })
    }
}

#[derive(Default)]
struct Rewriter {
    memo: HashMap<*const Expr, Expr>,
    fresh: usize,
}

fn raw(width: vmpx_vm::BitWidth, kind: ExprKind) -> Expr {
    Expr { width, kind }
}

fn bin(op: BinaryOp, l: Expr, r: Expr) -> Expr {
    raw(
        l.width,
        ExprKind::Binary {
            op,
            left: ExprRef::new(l),
            right: ExprRef::new(r),
        },
    )
}

fn pure(e: &Expr) -> bool {
    match &e.kind {
        ExprKind::Const { .. } | ExprKind::Register { .. } => true,
        ExprKind::Load { .. } | ExprKind::Intrinsic { .. } => false,
        ExprKind::Unary { value, .. }
        | ExprKind::Extract { value, .. }
        | ExprKind::Resize { value, .. } => pure(value),
        ExprKind::Binary { left, right, .. }
        | ExprKind::Concat {
            high: left,
            low: right,
        } => pure(left) && pure(right),
    }
}

fn is_not_of(n: &Expr, a: &ExprRef) -> bool {
    matches!(&n.kind, ExprKind::Unary { op: UnaryOp::Not, value } if value == a)
}

fn is_neg_of(n: &Expr, a: &ExprRef) -> bool {
    matches!(&n.kind, ExprKind::Unary { op: UnaryOp::Neg, value } if value == a)
}

fn is_odd_const(e: &Expr) -> bool {
    matches!(e.kind, ExprKind::Const { value } if value & 1 == 1)
}

/// `a * (a +/- odd)` in either operand order: one factor is always even.
fn is_consecutive_product(l: &ExprRef, r: &ExprRef) -> bool {
    let shifted = |a: &ExprRef, other: &Expr| {
        matches!(&other.kind, ExprKind::Binary { op: BinaryOp::Add | BinaryOp::Sub, left, right }
            if left == a && is_odd_const(right))
    };
    shifted(l, r) || shifted(r, l)
}

impl Rewriter {
    fn fresh_reg(&mut self, width: vmpx_vm::BitWidth) -> Expr {
        self.fresh += 1;
        Expr::register(width, format!("$opaque{}", self.fresh))
    }

    /// Exact constant, or a shape whose value set contains the original's.
    fn identity(&mut self, e: &Expr) -> Option<Expr> {
        let w = e.width;
        let ExprKind::Binary { op, left, right } = &e.kind else {
            return None;
        };
        let (l, r) = (&**left, &**right);
        let same = left == right;
        let complement = is_not_of(l, right) || is_not_of(r, left);
        let negated = is_neg_of(l, right) || is_neg_of(r, left);
        let exact = |v: u64| Some(Expr::constant(w, v));
        match op {
            BinaryOp::Xor | BinaryOp::Sub if same && pure(l) => exact(0),
            BinaryOp::And if complement && pure(l) && pure(r) => exact(0),
            BinaryOp::Or | BinaryOp::Xor | BinaryOp::Add if complement && pure(l) && pure(r) => {
                exact(w.mask())
            }
            BinaryOp::Add if negated && pure(l) && pure(r) => exact(0),
            BinaryOp::Add if same && pure(l) => {
                let y = self.fresh_reg(w);
                Some(bin(BinaryOp::Shl, y, Expr::constant(w, 1)))
            }
            BinaryOp::Mul if is_consecutive_product(left, right) && pure(l) && pure(r) => {
                let y = self.fresh_reg(w);
                Some(bin(BinaryOp::Shl, y, Expr::constant(w, 1)))
            }
            // x*x is 0 or 1 mod 4: 4y | b with b in {0, 1}.
            BinaryOp::Mul if same && pure(l) && w.bits() >= 8 => {
                let (y, z) = (self.fresh_reg(w), self.fresh_reg(w));
                let hi = bin(BinaryOp::Shl, y, Expr::constant(w, 2));
                let lo = bin(BinaryOp::And, z, Expr::constant(w, 1));
                Some(bin(BinaryOp::Or, hi, lo))
            }
            _ => None,
        }
    }

    fn rewrite(&mut self, e: &Expr, depth: usize) -> Expr {
        if depth >= MAX_DEPTH {
            return e.clone();
        }
        let key = std::ptr::from_ref(e);
        if let Some(done) = self.memo.get(&key) {
            return done.clone();
        }
        let out = self.identity(e).unwrap_or_else(|| self.descend(e, depth));
        self.memo.insert(key, out.clone());
        out
    }

    fn descend(&mut self, e: &Expr, depth: usize) -> Expr {
        let d = depth + 1;
        let kind = match &e.kind {
            ExprKind::Const { .. }
            | ExprKind::Register { .. }
            | ExprKind::Load { .. }
            | ExprKind::Intrinsic { .. } => return e.clone(),
            ExprKind::Unary { op, value } => ExprKind::Unary {
                op: *op,
                value: ExprRef::new(self.rewrite(value, d)),
            },
            ExprKind::Binary { op, left, right } => ExprKind::Binary {
                op: *op,
                left: ExprRef::new(self.rewrite(left, d)),
                right: ExprRef::new(self.rewrite(right, d)),
            },
            ExprKind::Extract { value, lsb } => ExprKind::Extract {
                value: ExprRef::new(self.rewrite(value, d)),
                lsb: *lsb,
            },
            ExprKind::Concat { high, low } => ExprKind::Concat {
                high: ExprRef::new(self.rewrite(high, d)),
                low: ExprRef::new(self.rewrite(low, d)),
            },
            ExprKind::Resize { op, value } => ExprKind::Resize {
                op: *op,
                value: ExprRef::new(self.rewrite(value, d)),
            },
        };
        raw(e.width, kind)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vmpx_mutation::{BitWidth, Ctx};

    const W: BitWidth = BitWidth::W32;

    fn x() -> Expr {
        Expr::register(W, "x")
    }

    fn k(v: u64) -> Expr {
        Expr::constant(W, v)
    }

    fn solve(e: Expr) -> Option<u64> {
        let p = Problem::new(e, ProblemClass::OpaquePredicate, Ctx::default());
        let c = OpaquePredicateSolver.solve(&p)?;
        assert_eq!(c.by, OpaquePredicateSolver.id());
        match c.expr.kind {
            ExprKind::Const { value } => Some(value),
            _ => panic!("not a constant"),
        }
    }

    #[test]
    fn applies_only_to_opaque_predicate() {
        for class in [
            ProblemClass::LinearMba,
            ProblemClass::NonlinearMba,
            ProblemClass::DeadCode,
            ProblemClass::Substitution,
            ProblemClass::Unknown,
        ] {
            assert!(!OpaquePredicateSolver.applies_to(class));
        }
        assert!(OpaquePredicateSolver.applies_to(ProblemClass::OpaquePredicate));
    }

    #[test]
    fn folds_provably_constant_predicates() {
        // x ^ x (built raw so the constructor's own folding is not what proves it).
        assert_eq!(solve(bin(BinaryOp::Xor, x(), x())), Some(0));
        let bit = bin(BinaryOp::And, x(), k(1));
        assert_eq!(solve(bin(BinaryOp::Sub, bit.clone(), bit.clone())), Some(0));
        // (x & 1) + (x & 1) is even, so its low bit is 0.
        let even = bin(BinaryOp::Add, bit.clone(), bit);
        assert_eq!(solve(bin(BinaryOp::And, even, k(1))), Some(0));
        // x * (x + 1) is even.
        let prod = bin(BinaryOp::Mul, x(), bin(BinaryOp::Add, x(), k(1)));
        assert_eq!(solve(bin(BinaryOp::And, prod, k(1))), Some(0));
        // x * x is 0 or 1 mod 4: bit 1 clear.
        assert_eq!(
            solve(bin(BinaryOp::And, bin(BinaryOp::Mul, x(), x()), k(2))),
            Some(0)
        );
        // x | ~x is all ones; value-range alone: (x & 0) is 0.
        let not_x = Expr::unary(UnaryOp::Not, x());
        assert_eq!(solve(bin(BinaryOp::Or, x(), not_x)), Some(0xFFFF_FFFF));
        assert_eq!(solve(bin(BinaryOp::And, x(), k(0))), Some(0));
    }

    #[test]
    fn input_dependent_conditions_pass() {
        assert_eq!(solve(x()), None);
        assert_eq!(solve(bin(BinaryOp::And, x(), k(1))), None);
        let prod = bin(BinaryOp::Mul, x(), bin(BinaryOp::Add, x(), k(1)));
        assert_eq!(solve(prod.clone()), None);
        // x*(x+2) is not always even; a*(a+1) with unrelated operands neither.
        let bad = bin(BinaryOp::Mul, x(), bin(BinaryOp::Add, x(), k(2)));
        assert_eq!(solve(bin(BinaryOp::And, bad, k(1))), None);
        let y = Expr::register(W, "y");
        let mixed = bin(BinaryOp::Mul, x(), bin(BinaryOp::Add, y, k(1)));
        assert_eq!(solve(bin(BinaryOp::And, mixed, k(1))), None);
        // impure reads are never folded as identical.
        let l = Expr::load(W, Expr::register(BitWidth::W64, "p"));
        assert_eq!(solve(bin(BinaryOp::Xor, l.clone(), l)), None);
        // already constant: nothing to simplify.
        assert_eq!(solve(k(1)), None);
    }
}
