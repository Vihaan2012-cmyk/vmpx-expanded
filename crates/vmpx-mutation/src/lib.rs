//! SP6 mutation-deobfuscation core: the parallelization contract (WP1).
//!
//! Holds the shared types every other SP6 crate builds against — [`Problem`],
//! [`ProblemClass`], [`Candidate`], [`SolverId`], [`Cost`], [`Score`], [`Ctx`] —
//! plus the pass-pipeline skeleton and the (stubbed) native passes. The traits
//! live in sibling crates to keep the dependency graph acyclic:
//! `vmpx-solvers` owns `Solver`, `vmpx-verify` owns `Scorer`/`Gate`,
//! `vmpx-tournament` owns `Elo` — all depending on this crate, never the reverse.

pub mod cfg_recovery;
pub mod dead_code;
pub mod substitution;

pub use cfg_recovery::{VmAlgebra, lift_native_block};
pub use vmpx_vm::{BitWidth, Expr};

/// Which family of mutation obfuscation a [`Problem`] belongs to. Drives both
/// solver routing ([`Solver::applies_to`](../vmpx_solvers/trait.Solver.html))
/// and the per-class ELO ladder.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ProblemClass {
    LinearMba,
    NonlinearMba,
    OpaquePredicate,
    DeadCode,
    Substitution,
    Unknown,
}

impl ProblemClass {
    /// Every class except [`ProblemClass::Unknown`], for tournament iteration.
    pub const RANKED: [ProblemClass; 5] = [
        ProblemClass::LinearMba,
        ProblemClass::NonlinearMba,
        ProblemClass::OpaquePredicate,
        ProblemClass::DeadCode,
        ProblemClass::Substitution,
    ];
}

/// Identifies a player (a solver or a chain) on the ELO ladder.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SolverId(pub String);

impl SolverId {
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Cost of producing a [`Candidate`]: the cost-aware half of the ladder so slow
/// winners are not crowned over cheap near-equals.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Cost {
    /// Wall-clock spent in the solver.
    pub time_ns: u64,
    /// Solver-internal work units (rewrite steps, SMT queries, ...).
    pub steps: u64,
}

/// The scorecard the [`Scorer`](../vmpx_verify/trait.Scorer.html) attaches to a
/// candidate. `agreement` must be `1.0` over the differential input-test set for
/// the candidate to be *eligible*; among eligible candidates, rank by
/// (`simplicity`, `cost`). See spec section 4.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Score {
    /// Input-test agreement ratio in `0.0..=1.0`; `1.0` == agrees on every input.
    pub agreement: f64,
    /// AST node / op count — lower is simpler.
    pub simplicity: u64,
    /// Time and effort to produce the candidate.
    pub cost: Cost,
}

impl Score {
    /// Eligibility gate: only candidates that agree on every tested input may be
    /// ranked or promoted. Inductive, never a soundness proof (that is the Gate).
    #[must_use]
    pub fn is_eligible(&self) -> bool {
        self.agreement >= 1.0
    }
}

/// Minimal surrounding context for a [`Problem`]. Grows as the passes need it
/// (register domains, surrounding CFG, assumptions); kept small for WP1.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Ctx {
    /// Dominant operand width of the problem.
    pub width: BitWidth,
}

impl Default for Ctx {
    fn default() -> Self {
        Self {
            width: BitWidth::W64,
        }
    }
}

/// A unit of deobfuscation work: an expression (or code region) to simplify,
/// tagged with its class and context.
#[derive(Clone, Debug)]
pub struct Problem {
    pub expr: Expr,
    pub class: ProblemClass,
    pub ctx: Ctx,
}

impl Problem {
    pub fn new(expr: Expr, class: ProblemClass, ctx: Ctx) -> Self {
        Self { expr, class, ctx }
    }
}

/// A proposed simplification of a [`Problem`], attributed to the player that
/// produced it and tagged with its cost.
#[derive(Clone, Debug)]
pub struct Candidate {
    pub expr: Expr,
    pub by: SolverId,
    pub cost: Cost,
}

/// WAVE3 (integration): wire DCE -> substitution -> MBA -> opaque-fold to a
/// fixpoint, with the tournament selecting a player per sub-problem and the
/// Gate tiering results sound/experimental. Stub returns the input unchanged.
#[must_use]
pub fn run_pipeline(problem: &Problem) -> Option<Candidate> {
    let _ = problem;
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn score_eligibility_requires_full_agreement() {
        let full = Score {
            agreement: 1.0,
            simplicity: 3,
            cost: Cost::default(),
        };
        let partial = Score {
            agreement: 0.999,
            simplicity: 1,
            cost: Cost::default(),
        };
        assert!(full.is_eligible());
        assert!(!partial.is_eligible());
    }

    #[test]
    fn pipeline_stub_is_a_noop() {
        let p = Problem::new(
            Expr::constant(BitWidth::W64, 7),
            ProblemClass::Unknown,
            Ctx::default(),
        );
        assert!(run_pipeline(&p).is_none());
    }
}
