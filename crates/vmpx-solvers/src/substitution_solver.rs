//! Substitution canonicalizer (solver).

use crate::Solver;
use vmpx_mutation::{Candidate, Problem, ProblemClass, SolverId};

/// Reverses VMProtect's instruction-substitution patterns.
pub struct SubstitutionSolver;

// WAVE2: WP11 — peephole reversal of VMProtect's instruction-substitution
// patterns. Pairs with vmpx_mutation::substitution (pass side). Fill `solve`.
impl Solver for SubstitutionSolver {
    fn id(&self) -> SolverId {
        SolverId::new("substitution")
    }

    fn applies_to(&self, class: ProblemClass) -> bool {
        class == ProblemClass::Substitution
    }

    fn solve(&self, p: &Problem) -> Option<Candidate> {
        let _ = p;
        None
    }
}
