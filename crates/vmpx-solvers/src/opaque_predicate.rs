//! Opaque-predicate prover (solver).

use crate::Solver;
use vmpx_mutation::{Candidate, Problem, ProblemClass, SolverId};

/// Proves a branch condition constant and folds the dead arm.
pub struct OpaquePredicateSolver;

// WAVE2: WP8 — prove a predicate constant via vmpx_vm::value_range + bounded
// symbolic reasoning (BINSEC backward-bounded DSE style), then fold the dead
// arm. Fill `solve` to return the folded candidate; keep `applies_to` as-is.
impl Solver for OpaquePredicateSolver {
    fn id(&self) -> SolverId {
        SolverId::new("opaque_predicate")
    }

    fn applies_to(&self, class: ProblemClass) -> bool {
        class == ProblemClass::OpaquePredicate
    }

    fn solve(&self, p: &Problem) -> Option<Candidate> {
        let _ = p;
        None
    }
}
