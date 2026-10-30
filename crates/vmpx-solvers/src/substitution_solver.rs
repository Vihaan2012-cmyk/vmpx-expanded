//! Substitution canonicalizer (solver).

use crate::Solver;
use std::time::Instant;
use vmpx_mutation::{Candidate, Cost, Problem, ProblemClass, SolverId, substitution};

/// Reverses `VMProtect`'s instruction-substitution patterns.
pub struct SubstitutionSolver;

impl Solver for SubstitutionSolver {
    fn id(&self) -> SolverId {
        SolverId::new("substitution")
    }

    fn applies_to(&self, class: ProblemClass) -> bool {
        class == ProblemClass::Substitution
    }

    fn solve(&self, p: &Problem) -> Option<Candidate> {
        if !self.applies_to(p.class) {
            return None;
        }
        let start = Instant::now();
        let expr = substitution::canonicalize(p)?;
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
    use vmpx_mutation::{BitWidth, Ctx, Expr};
    use vmpx_vm::BinaryOp;

    fn problem(class: ProblemClass) -> Problem {
        let x = Expr::register(BitWidth::W64, "x");
        let e = Expr::binary(BinaryOp::Add, x.clone(), x).expect("same width");
        Problem::new(e, class, Ctx::default())
    }

    #[test]
    fn canonicalizes_substitution_problem() {
        let c = SubstitutionSolver
            .solve(&problem(ProblemClass::Substitution))
            .expect("candidate");
        assert_eq!(c.by, SolverId::new("substitution"));
        let x = Expr::register(BitWidth::W64, "x");
        let one = Expr::constant(BitWidth::W64, 1);
        assert_eq!(
            c.expr,
            Expr::binary(BinaryOp::Shl, x, one).expect("same width")
        );
    }

    #[test]
    fn passes_on_other_classes() {
        assert!(!SubstitutionSolver.applies_to(ProblemClass::LinearMba));
        assert!(
            SubstitutionSolver
                .solve(&problem(ProblemClass::LinearMba))
                .is_none()
        );
    }
}
