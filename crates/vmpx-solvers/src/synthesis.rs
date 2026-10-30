//! WAVE4 (fill me): program-synthesis solver, Syntia/Xyntia-style (prior-art
//! survey: Blazytko USENIX'17, Xyntia). Uses I/O behaviour as an oracle to
//! SYNTHESIZE a small equivalent expression — targeting the nonlinear-MBA gap
//! (`x*y` recovery) that the algebraic solvers miss. Inductive: proposes a
//! candidate; the input-tester + SMT gate verify it downstream.

use crate::Solver;
use vmpx_mutation::{Candidate, Problem, ProblemClass, SolverId};

#[derive(Default)]
pub struct SynthesisSolver;

impl Solver for SynthesisSolver {
    fn id(&self) -> SolverId {
        SolverId::new("synthesis")
    }
    fn applies_to(&self, class: ProblemClass) -> bool {
        matches!(class, ProblemClass::NonlinearMba | ProblemClass::LinearMba)
    }
    fn solve(&self, _p: &Problem) -> Option<Candidate> {
        None // WAVE4
    }
}
