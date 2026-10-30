//! SP6 solver portfolio: the [`Solver`] trait, the [`Chain`] combinator, and one
//! stub module per solver. Players compete on the ELO ladder in `vmpx-tournament`.
//!
//! Depends on `vmpx-mutation` (contract types) only; never the reverse.

pub mod general_mba;
pub mod linear_mba;
pub mod opaque_predicate;
pub mod substitution_solver;

use vmpx_mutation::{Candidate, Cost, Problem, ProblemClass, SolverId};

/// A deobfuscation player: given a [`Problem`], either proposes a simplified
/// [`Candidate`] or passes (`None`). Scoring/gating happen elsewhere; a solver
/// only proposes.
pub trait Solver {
    /// Stable identity on the ladder.
    fn id(&self) -> SolverId;
    /// Routing hint: should this solver be offered problems of `class`?
    fn applies_to(&self, class: ProblemClass) -> bool;
    /// Propose a candidate, or `None` to pass / give up.
    fn solve(&self, p: &Problem) -> Option<Candidate>;
}

/// A chain is a solver built from solvers: each stage rewrites the running
/// expression and feeds it to the next. Implements [`Solver`], so chains are
/// themselves players. A chain produces a candidate iff at least one stage did.
pub struct Chain(pub Vec<Box<dyn Solver>>);

impl Chain {
    #[must_use]
    pub fn new(stages: Vec<Box<dyn Solver>>) -> Self {
        Self(stages)
    }
}

impl Solver for Chain {
    fn id(&self) -> SolverId {
        let inner = self
            .0
            .iter()
            .map(|s| s.id().0)
            .collect::<Vec<_>>()
            .join("+");
        SolverId::new(format!("chain({inner})"))
    }

    fn applies_to(&self, class: ProblemClass) -> bool {
        self.0.iter().any(|s| s.applies_to(class))
    }

    fn solve(&self, p: &Problem) -> Option<Candidate> {
        let mut current = p.clone();
        let mut cost = Cost::default();
        let mut produced = false;
        for stage in &self.0 {
            if let Some(c) = stage.solve(&current) {
                produced = true;
                cost.time_ns = cost.time_ns.saturating_add(c.cost.time_ns);
                cost.steps = cost.steps.saturating_add(c.cost.steps);
                current.expr = c.expr;
            }
        }
        produced.then(|| Candidate {
            expr: current.expr,
            by: self.id(),
            cost,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vmpx_mutation::{BitWidth, Ctx, Expr};

    struct NoOp(&'static str);
    impl Solver for NoOp {
        fn id(&self) -> SolverId {
            SolverId::new(self.0)
        }
        fn applies_to(&self, _class: ProblemClass) -> bool {
            true
        }
        fn solve(&self, _p: &Problem) -> Option<Candidate> {
            None
        }
    }

    fn problem() -> Problem {
        Problem::new(
            Expr::constant(BitWidth::W64, 42),
            ProblemClass::LinearMba,
            Ctx::default(),
        )
    }

    #[test]
    fn chain_of_two_noops_passes() {
        let chain = Chain::new(vec![Box::new(NoOp("a")), Box::new(NoOp("b"))]);
        assert!(chain.applies_to(ProblemClass::LinearMba));
        assert!(chain.solve(&problem()).is_none());
    }
}
