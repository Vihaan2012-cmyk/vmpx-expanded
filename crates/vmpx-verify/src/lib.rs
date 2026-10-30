//! SP6 verification: the [`Scorer`] (differential input-testing, the workhorse)
//! and the [`Gate`] (SMT equivalence, promotion to the sound tier). Two roles,
//! never one: input-testing ranks + feeds the experimental tier; the Gate
//! upgrades an eligible candidate to proven. Depends on `vmpx-mutation` only.

pub mod input_testing;
pub mod smt;

use vmpx_mutation::{Candidate, Score};
use vmpx_vm::Expr;

/// Differential input-testing: run `original` and `cand` over many concrete
/// inputs, return a [`Score`] whose `agreement` ratio gates eligibility.
pub trait Scorer {
    fn score(&self, original: &Expr, cand: &Candidate) -> Score;
}

/// Outcome of a soundness proof attempt.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Verdict {
    /// Proven behavior-equivalent over bitvectors — promote to the sound tier.
    Proven,
    /// Proven NOT equivalent — reject.
    Refuted,
    /// Undecided within budget — stays experimental.
    Unknown,
}

/// SMT / bitvector equivalence gate. Proves `original == candidate`; never a
/// scorer, never a blocker on the experimental tier.
pub trait Gate {
    fn verify(&self, original: &Expr, cand: &Expr) -> Verdict;
}
