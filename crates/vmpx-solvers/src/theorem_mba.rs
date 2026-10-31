//! Players backed by `vmpx-mutation`'s theorem-grade MBA simplifiers:
//! [`LinearSimplifySolver`] (signature-vector theorem, any linear MBA up to 6
//! variables) and [`GambaSolver`] (non-linear MBA by linear-part isolation).

use std::time::Instant;

use crate::Solver;
use vmpx_mutation::general_mba::simplify_general;
use vmpx_mutation::linear_mba::simplify_linear;
use vmpx_mutation::{Candidate, Cost, Problem, ProblemClass, SolverId};

const NODE_LIMIT: usize = 1 << 16;

fn elapsed_ns(t: Instant) -> u64 {
    u64::try_from(t.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

/// Rewrites a linear MBA to its canonical form; passes on anything else.
pub struct LinearSimplifySolver;

impl Solver for LinearSimplifySolver {
    fn id(&self) -> SolverId {
        SolverId::new("linear_simplify")
    }

    fn applies_to(&self, class: ProblemClass) -> bool {
        matches!(class, ProblemClass::LinearMba | ProblemClass::Unknown)
    }

    fn solve(&self, p: &Problem) -> Option<Candidate> {
        let t = Instant::now();
        let r = simplify_linear(&p.expr).ok()?;
        (r.expr.node_count_bounded(NODE_LIMIT) < p.expr.node_count_bounded(NODE_LIMIT)).then(
            || Candidate {
                expr: r.expr,
                by: self.id(),
                cost: Cost {
                    time_ns: elapsed_ns(t),
                    steps: 1,
                },
            },
        )
    }
}

/// GAMBA-style: simplifies the linear sub-parts of a non-linear MBA.
pub struct GambaSolver;

impl Solver for GambaSolver {
    fn id(&self) -> SolverId {
        SolverId::new("gamba")
    }

    fn applies_to(&self, class: ProblemClass) -> bool {
        matches!(
            class,
            ProblemClass::NonlinearMba | ProblemClass::LinearMba | ProblemClass::Unknown
        )
    }

    fn solve(&self, p: &Problem) -> Option<Candidate> {
        let t = Instant::now();
        let r = simplify_general(&p.expr);
        (r.expr.node_count_bounded(NODE_LIMIT) < p.expr.node_count_bounded(NODE_LIMIT)).then(
            || Candidate {
                expr: r.expr,
                by: self.id(),
                cost: Cost {
                    time_ns: elapsed_ns(t),
                    steps: r.linear_rewrites as u64 + 1,
                },
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vmpx_mutation::{BitWidth, Ctx, Expr};
    use vmpx_vm::BinaryOp::{And, Mul, Or, Sub, Xor};

    const W: BitWidth = BitWidth::W64;

    fn v(n: &str) -> Expr {
        Expr::register(W, n)
    }
    fn b(op: vmpx_vm::BinaryOp, l: Expr, r: Expr) -> Expr {
        Expr::binary(op, l, r).unwrap()
    }
    fn obf_xor() -> Expr {
        b(Sub, b(Or, v("x"), v("y")), b(And, v("x"), v("y")))
    }

    #[test]
    fn linear_solver_simplifies_linear_and_passes_on_nonlinear() {
        let p = Problem::new(obf_xor(), ProblemClass::LinearMba, Ctx::default());
        assert_eq!(LinearSimplifySolver.solve(&p).unwrap().expr, b(Xor, v("x"), v("y")));
        let nl = Problem::new(
            b(Mul, v("x"), obf_xor()),
            ProblemClass::NonlinearMba,
            Ctx::default(),
        );
        assert!(LinearSimplifySolver.solve(&nl).is_none());
        assert_eq!(
            GambaSolver.solve(&nl).unwrap().expr,
            b(Mul, v("x"), b(Xor, v("x"), v("y")))
        );
    }
}
