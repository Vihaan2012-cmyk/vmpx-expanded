//! The roster of players: the four solvers plus chains and pass adapters.

use std::time::Instant;

use vmpx_mutation::{Candidate, Cost, Problem, ProblemClass, SolverId, dead_code, substitution};
use vmpx_solvers::general_mba::GeneralMbaSolver;
use vmpx_solvers::linear_mba::LinearMbaSolver;
use vmpx_solvers::opaque_predicate::OpaquePredicateSolver;
use vmpx_solvers::substitution_solver::SubstitutionSolver;
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

/// The full roster: 4 solvers, 2 pass adapters, 3 chains.
#[must_use]
pub fn roster() -> Vec<Box<dyn Solver>> {
    vec![
        Box::new(LinearMbaSolver),
        Box::new(GeneralMbaSolver),
        Box::new(OpaquePredicateSolver),
        Box::new(SubstitutionSolver),
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
    ]
}
