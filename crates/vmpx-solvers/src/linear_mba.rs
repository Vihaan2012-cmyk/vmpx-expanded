//! Linear-MBA simplifier (solver).

use crate::Solver;
use vmpx_mutation::{Candidate, Problem, ProblemClass, SolverId};

/// Canonicalizes linear mixed boolean-arithmetic to a normal form.
pub struct LinearMbaSolver;

// WAVE2: WP9 — SiMBA-style canonicalization of linear MBA to a normal form.
// Fill `solve` to return the canonicalized candidate.
impl Solver for LinearMbaSolver {
    fn id(&self) -> SolverId {
        SolverId::new("linear_mba")
    }

    fn applies_to(&self, class: ProblemClass) -> bool {
        class == ProblemClass::LinearMba
    }

    fn solve(&self, p: &Problem) -> Option<Candidate> {
        let _ = p;
        None
    }
}
