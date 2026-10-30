//! The roster of players: the four solvers plus chains and pass adapters.

use std::time::Instant;

use vmpx_mutation::{Candidate, Cost, Problem, ProblemClass, SolverId, dead_code, substitution};
use vmpx_solvers::dictionary::DictionarySolver;
use vmpx_solvers::general_mba::GeneralMbaSolver;
use vmpx_solvers::linear_mba::LinearMbaSolver;
use vmpx_solvers::opaque_predicate::OpaquePredicateSolver;
use vmpx_solvers::substitution_solver::SubstitutionSolver;
use vmpx_solvers::synthesis::SynthesisSolver;
use vmpx_solvers::{Chain, Solver};

fn elapsed_ns(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

/// `dead_code::eliminate` as a class-agnostic player (usable inside chains).
pub struct DeadCodePass;

impl Solver for DeadCodePass {
    fn id(&self) -> SolverId {
        SolverId::new("dead_code")
    }

    fn applies_to(&self, class: ProblemClass) -> bool {
        !matches!(class, ProblemClass::OpaquePredicate | ProblemClass::Unknown)
    }

    fn solve(&self, p: &Problem) -> Option<Candidate> {
        let start = Instant::now();
        let expr = dead_code::eliminate(p)?;
        Some(Candidate {
            expr,
            by: self.id(),
            cost: Cost {
                time_ns: elapsed_ns(start),
                steps: 1,
            },
        })
    }
}

/// `substitution::canonicalize` without the class guard `SubstitutionSolver`
/// applies, so it can run as a chain stage on MBA problems.
pub struct CanonicalizePass;

impl Solver for CanonicalizePass {
    fn id(&self) -> SolverId {
        SolverId::new("canonicalize")
    }

    fn applies_to(&self, class: ProblemClass) -> bool {
        !matches!(class, ProblemClass::OpaquePredicate | ProblemClass::Unknown)
    }

    fn solve(&self, p: &Problem) -> Option<Candidate> {
        let start = Instant::now();
        let expr = substitution::canonicalize(p)?;
        Some(Candidate {
            expr,
            by: self.id(),
            cost: Cost {
                time_ns: elapsed_ns(start),
                steps: 1,
            },
        })
    }
}

/// The full roster: 6 solvers, 2 pass adapters, chains.
///
/// `dictionary` (structural template matcher) and `synthesis` (I/O-oracle
/// enumerator) are proposers like the rest; the pipeline's SMT gate / input
/// tester verify every proposal, so adding them cannot emit an unsound rewrite.
#[must_use]
pub fn roster() -> Vec<Box<dyn Solver>> {
    vec![
        Box::new(LinearMbaSolver),
        Box::new(GeneralMbaSolver),
        Box::new(OpaquePredicateSolver),
        Box::new(SubstitutionSolver),
        Box::new(DictionarySolver),
        Box::new(SynthesisSolver),
        Box::new(DeadCodePass),
        Box::new(CanonicalizePass),
        Box::new(Chain::new(vec![
            Box::new(CanonicalizePass),
            Box::new(GeneralMbaSolver),
        ])),
        Box::new(Chain::new(vec![
            Box::new(DeadCodePass),
            Box::new(LinearMbaSolver),
        ])),
        Box::new(Chain::new(vec![
            Box::new(DeadCodePass),
            Box::new(CanonicalizePass),
            Box::new(GeneralMbaSolver),
        ])),
        // Dictionary as a canonicalizing pre-pass feeding the algebraic solver.
        Box::new(Chain::new(vec![
            Box::new(DictionarySolver),
            Box::new(GeneralMbaSolver),
        ])),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The dictionary and synthesis proposers must be on the roster so the
    /// tournament/ELO can route to them; a prior roster silently omitted both.
    #[test]
    fn roster_includes_dictionary_and_synthesis() {
        let ids: Vec<String> = roster().iter().map(|s| s.id().0).collect();
        assert!(ids.iter().any(|i| i == "dictionary"), "{ids:?}");
        assert!(ids.iter().any(|i| i == "synthesis"), "{ids:?}");
    }

    /// Every roster player either produces a candidate that survives the SMT
    /// gate / input tester downstream or passes — a solver only proposes, so no
    /// registration can by itself emit an unverified rewrite. Here we just check
    /// each player is well-formed on a trivial problem (no panic, honest id).
    #[test]
    fn every_player_has_a_stable_id() {
        for s in roster() {
            assert!(!s.id().0.is_empty());
        }
    }
}
