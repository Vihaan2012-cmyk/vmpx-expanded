//! Tier/gate scaffolding: observed -> proven.
//!
//! A candidate is never promoted on I/O agreement alone. [`PromotionGate::judge`]
//! checks recorded tests (any concrete mismatch => `Refuted`), then tries proofs
//! (structural identity, the linear-MBA signature theorem, then an optional
//! [`EquivalenceBackend`] answering "is `a != b` satisfiable at full width?").
//! UNSAT => `Proven`; SAT => `Refuted` only if the model concretely reproduces the
//! difference, and it is appended to the test set (CEGIS feedback); unknown or
//! timeout => stays `Observed`.
//!
//! The backend is a minimal local trait (`vmpx-verify` depends on this crate, so
//! it cannot be reused here); `vmpx_verify::SmtGate` can be adapted to it.
//! No solver is wired by default: [`NoSolver`] always answers `Unknown`.

use std::collections::BTreeMap;

use vmpx_vm::Expr;

use crate::bvutil::eval_concrete;
use crate::linear_mba::{LinearCmp, compare_linear};
use crate::state_model::{HandlerModel, Slot};

/// Named register assignment.
pub type Counterexample = BTreeMap<String, u64>;

/// Confidence ladder, ordered worst to best so a conjunction is the `min`.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Tier {
    /// A concrete input on which original and candidate differ.
    Refuted,
    /// No usable evidence (no tests evaluated, ill-formed pair, ...).
    Unproven,
    /// Agrees on every evaluated test input; no proof (includes solver timeout).
    Observed,
    /// Equivalent over the full bit-width, by a proof.
    Proven,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProofKind {
    /// Identical after simplification.
    Structural,
    /// Linear-MBA signature-vector theorem (see `linear_mba`).
    SignatureTheorem,
    /// Backend returned UNSAT for `a != b`.
    Smt,
}

/// Answer of an SMT inequality query `a != b`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SmtVerdict {
    Unsat,
    Sat(Counterexample),
    /// Timeout, unsupported shape, or no solver.
    Unknown,
}

/// Hook point for a real solver (Z3, ...). Must be sound for `Unsat`.
pub trait EquivalenceBackend {
    fn check_inequality(&self, original: &Expr, candidate: &Expr) -> SmtVerdict;
}

/// Default backend: never proves anything.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoSolver;

impl EquivalenceBackend for NoSolver {
    fn check_inequality(&self, _: &Expr, _: &Expr) -> SmtVerdict {
        SmtVerdict::Unknown
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Judgement {
    pub tier: Tier,
    pub proof: Option<ProofKind>,
    pub counterexample: Option<Counterexample>,
    /// Number of test inputs on which both sides evaluated concretely and agreed.
    pub tests_agreed: usize,
}

impl Judgement {
    const fn of(tier: Tier, tests_agreed: usize) -> Self {
        Self {
            tier,
            proof: None,
            counterexample: None,
            tests_agreed,
        }
    }
}

pub struct PromotionGate<B: EquivalenceBackend = NoSolver> {
    backend: B,
    tests: Vec<Counterexample>,
}

impl PromotionGate<NoSolver> {
    /// Gate with no solver: can reach `Proven` only structurally or by theorem.
    #[must_use]
    pub const fn without_solver() -> Self {
        Self::with_backend(NoSolver)
    }
}

impl<B: EquivalenceBackend> PromotionGate<B> {
    #[must_use]
    pub const fn with_backend(backend: B) -> Self {
        Self {
            backend,
            tests: Vec::new(),
        }
    }

    pub fn add_test(&mut self, input: Counterexample) {
        self.tests.push(input);
    }

    #[must_use]
    pub fn tests(&self) -> &[Counterexample] {
        &self.tests
    }

    /// Judges `candidate` against `original` (register-level expressions).
    pub fn judge(&mut self, original: &Expr, candidate: &Expr) -> Judgement {
        if original.width != candidate.width {
            return Judgement::of(Tier::Unproven, 0);
        }
        let (a, b) = (original.simplify(), candidate.simplify());
        let none = BTreeMap::new();
        let differs = |env: &Counterexample| {
            match (eval_concrete(&a, env, &none), eval_concrete(&b, env, &none)) {
                (Some(x), Some(y)) => Some(x != y),
                _ => None,
            }
        };

        let mut agreed = 0;
        for t in &self.tests {
            match differs(t) {
                Some(true) => return refuted(t.clone(), agreed),
                Some(false) => agreed += 1,
                None => {}
            }
        }

        if a == b {
            return proven(ProofKind::Structural, agreed);
        }
        match compare_linear(&a, &b) {
            Ok(LinearCmp::Equal) => return proven(ProofKind::SignatureTheorem, agreed),
            Ok(LinearCmp::Differ(cex)) if differs(&cex) == Some(true) => {
                self.tests.push(cex.clone());
                return refuted(cex, agreed);
            }
            _ => {}
        }
        match self.backend.check_inequality(&a, &b) {
            SmtVerdict::Unsat => return proven(ProofKind::Smt, agreed),
            SmtVerdict::Sat(cex) if differs(&cex) == Some(true) => {
                self.tests.push(cex.clone());
                return refuted(cex, agreed);
            }
            SmtVerdict::Sat(_) | SmtVerdict::Unknown => {}
        }
        Judgement::of(
            if agreed > 0 { Tier::Observed } else { Tier::Unproven },
            agreed,
        )
    }

    /// Judges whole handlers: tier is the minimum over sp-delta, every output
    /// slot and every memory write; an unmatched structure is `Refuted`/`Unproven`.
    pub fn judge_model(&mut self, original: &HandlerModel, candidate: &HandlerModel) -> Tier {
        if original.sp_delta() != candidate.sp_delta()
            || original.writes().len() != candidate.writes().len()
        {
            return Tier::Refuted;
        }
        let mut tier = Tier::Proven;
        let slots: std::collections::BTreeSet<&Slot> = original
            .outputs()
            .keys()
            .chain(candidate.outputs().keys())
            .collect();
        for slot in slots {
            let side = |m: &HandlerModel| m.outputs().get(slot).cloned().or_else(|| m.input(slot));
            let (Some(x), Some(y)) = (side(original), side(candidate)) else {
                return Tier::Unproven;
            };
            tier = tier.min(self.judge(&x, &y).tier);
        }
        for (x, y) in original.writes().iter().zip(candidate.writes()) {
            tier = tier.min(self.judge(&x.addr, &y.addr).tier);
            tier = tier.min(self.judge(&x.value, &y.value).tier);
        }
        tier
    }
}

fn refuted(cex: Counterexample, agreed: usize) -> Judgement {
    Judgement {
        counterexample: Some(cex),
        ..Judgement::of(Tier::Refuted, agreed)
    }
}

fn proven(proof: ProofKind, agreed: usize) -> Judgement {
    Judgement {
        proof: Some(proof),
        ..Judgement::of(Tier::Proven, agreed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vmpx_vm::{BinaryOp, BitWidth};

    const W: BitWidth = BitWidth::W64;

    fn v(n: &str) -> Expr {
        Expr::register(W, n)
    }
    fn b(op: BinaryOp, l: Expr, r: Expr) -> Expr {
        Expr::binary(op, l, r).unwrap()
    }
    fn env(x: u64, y: u64) -> Counterexample {
        [("x".to_owned(), x), ("y".to_owned(), y)].into()
    }

    /// Nonlinear pair that agrees on small inputs but differs for large `x`.
    fn lookalike() -> (Expr, Expr) {
        let orig = b(BinaryOp::Mul, v("x"), v("y"));
        let cand = b(
            BinaryOp::Xor,
            orig.clone(),
            b(BinaryOp::Shr, v("x"), Expr::constant(W, 40)),
        );
        (orig, cand)
    }

    struct Fixed(SmtVerdict);
    impl EquivalenceBackend for Fixed {
        fn check_inequality(&self, _: &Expr, _: &Expr) -> SmtVerdict {
            self.0.clone()
        }
    }

    #[test]
    fn io_match_alone_is_observed_not_proven() {
        let (orig, cand) = lookalike();
        let mut gate = PromotionGate::without_solver();
        gate.add_test(env(3, 5));
        gate.add_test(env(1000, 77));
        let j = gate.judge(&orig, &cand);
        assert_eq!(j.tier, Tier::Observed);
        assert_eq!(j.proof, None);
        assert_eq!(j.tests_agreed, 2);
    }

    #[test]
    fn no_evidence_is_unproven() {
        let (orig, cand) = lookalike();
        assert_eq!(PromotionGate::without_solver().judge(&orig, &cand).tier, Tier::Unproven);
    }

    #[test]
    fn concrete_mismatch_refutes() {
        let (orig, cand) = lookalike();
        let mut gate = PromotionGate::without_solver();
        gate.add_test(env(1 << 41, 3));
        let j = gate.judge(&orig, &cand);
        assert_eq!(j.tier, Tier::Refuted);
        assert!(j.counterexample.is_some());
    }

    #[test]
    fn linear_mba_identity_is_proven_by_theorem_without_a_solver() {
        let mba = b(
            BinaryOp::Sub,
            b(BinaryOp::Or, v("x"), v("y")),
            b(BinaryOp::And, v("x"), v("y")),
        );
        let j = PromotionGate::without_solver().judge(&mba, &b(BinaryOp::Xor, v("x"), v("y")));
        assert_eq!((j.tier, j.proof), (Tier::Proven, Some(ProofKind::SignatureTheorem)));
        // Close but wrong: refuted with a real counterexample, not merely "observed".
        let wrong = PromotionGate::without_solver().judge(&mba, &b(BinaryOp::Or, v("x"), v("y")));
        assert_eq!(wrong.tier, Tier::Refuted);
    }

    #[test]
    fn solver_verdicts_decide_promotion_and_feed_back() {
        let (orig, cand) = lookalike();
        let mut seeded = PromotionGate::with_backend(Fixed(SmtVerdict::Unsat));
        seeded.add_test(env(3, 5));
        let j = seeded.judge(&orig, &cand);
        assert_eq!((j.tier, j.proof), (Tier::Proven, Some(ProofKind::Smt)));

        let mut unknown = PromotionGate::with_backend(Fixed(SmtVerdict::Unknown));
        unknown.add_test(env(3, 5));
        assert_eq!(unknown.judge(&orig, &cand).tier, Tier::Observed);

        let mut sat = PromotionGate::with_backend(Fixed(SmtVerdict::Sat(env(1 << 45, 9))));
        sat.add_test(env(3, 5));
        assert_eq!(sat.judge(&orig, &cand).tier, Tier::Refuted);
        assert_eq!(sat.tests().len(), 2);

        // A bogus model that does not reproduce a difference must not refute.
        let mut bogus = PromotionGate::with_backend(Fixed(SmtVerdict::Sat(env(1, 1))));
        bogus.add_test(env(3, 5));
        assert_eq!(bogus.judge(&orig, &cand).tier, Tier::Observed);
    }

    #[test]
    fn model_tier_is_the_minimum_over_effects() {
        let mk = |sp: i64, mba: bool| {
            let mut m = HandlerModel::new("h").with_input(Slot::Stack(0), W);
            let x = m.input(&Slot::Stack(0)).unwrap();
            let e = if mba {
                b(BinaryOp::Sub, b(BinaryOp::Or, x.clone(), x.clone()), Expr::constant(W, 0))
            } else {
                x
            };
            m.set_output(Slot::Stack(0), e).unwrap();
            m.set_sp_delta(sp);
            m
        };
        let mut gate = PromotionGate::without_solver();
        assert_eq!(gate.judge_model(&mk(8, false), &mk(8, true)), Tier::Proven);
        assert_eq!(gate.judge_model(&mk(8, false), &mk(16, false)), Tier::Refuted);
    }
}
