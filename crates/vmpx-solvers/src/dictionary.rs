//! WAVE4 (fill me): dictionary / pattern-recognition solver. The deterministic-
//! corpus idea — learn a dictionary of (obfuscated-shape -> clean-form) pairs
//! from known functions, then RECOGNISE those shapes in a target and substitute
//! the clean form. Inductive recognition; the input-tester + gate verify it.

use crate::Solver;
use vmpx_mutation::{Candidate, Problem, ProblemClass, SolverId};

#[derive(Default)]
pub struct DictionarySolver;

impl Solver for DictionarySolver {
    fn id(&self) -> SolverId {
        SolverId::new("dictionary")
    }
    fn applies_to(&self, _class: ProblemClass) -> bool {
        true // patterns span every class
    }
    fn solve(&self, _p: &Problem) -> Option<Candidate> {
        None // WAVE4
    }
}
