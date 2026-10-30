//! Differential input-testing harness — the Scorer (the tournament workhorse).

use crate::Scorer;
use vmpx_mutation::{Candidate, Cost, Score};
use vmpx_vm::Expr;

/// Scores a candidate by agreement with the original over many concrete inputs.
pub struct InputTester;

// WAVE2: WP2 — generate input tuples (random PLUS structured edge cases: 0, 1,
// MAX, sign boundaries, 2^k +/- 1, single-bit), evaluate both Expr trees on each
// (reuse vmpx_vm const-folding / an evaluator), compare per-input. Fill `score`
// to return the real agreement ratio and simplicity (AST node count). Stub
// reports zero agreement so nothing is ever ranked eligible by accident.
impl Scorer for InputTester {
    fn score(&self, original: &Expr, cand: &Candidate) -> Score {
        let _ = (original, cand);
        Score {
            agreement: 0.0,
            simplicity: u64::MAX,
            cost: Cost::default(),
        }
    }
}
