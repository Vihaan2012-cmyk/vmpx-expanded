//! The solver tournament for one problem, and `run_pipeline` on top of it.
//!
//! Every solver that applies to the problem's class proposes a candidate. Each
//! candidate is scored by differential input-testing (`InputTester`) and run
//! through the [`PromotionGate`] (sampled tests first, then structural /
//! linear-MBA theorem / optional SMT backend). Refuted or non-eligible
//! candidates are dropped; the rest are ranked by tier (`Proven` before
//! `Observed`), then simplicity, then solver cost, then ELO rating. Results
//! also feed the ELO ladder. Nothing is `Proven` without a proof, and with no
//! backend configured only the theorem/structural proofs are available.

use std::cmp::Reverse;

use crate::{Elo, MatchResult};
use vmpx_mutation::gate::{
    EquivalenceBackend, NoSolver, ProofKind, PromotionGate, SmtVerdict, Tier,
};
use vmpx_mutation::linear_mba::signature;
use vmpx_mutation::sampling::judge_sampled;
use vmpx_mutation::{Candidate, Expr, Problem, ProblemClass, Score};
use vmpx_solvers::Solver;
use vmpx_solvers::local_search::LocalSearchSolver;
use vmpx_solvers::theorem_mba::{GambaSolver, LinearSimplifySolver};
use vmpx_verify::Scorer;
use vmpx_verify::input_testing::InputTester;

/// A candidate that survived scoring and the gate, with its evidence.
#[derive(Clone, Debug)]
pub struct Gated {
    pub candidate: Candidate,
    /// `Proven` or `Observed`; refuted/unproven candidates are never returned.
    pub tier: Tier,
    pub proof: Option<ProofKind>,
    pub score: Score,
}

struct BackendRef<'a>(&'a dyn EquivalenceBackend);

impl EquivalenceBackend for BackendRef<'_> {
    fn check_inequality(&self, original: &Expr, candidate: &Expr) -> SmtVerdict {
        self.0.check_inequality(original, candidate)
    }
}

/// Solvers competing per problem, a scorer, an (optional) SMT backend and the
/// persistent ELO ladder.
pub struct Tournament {
    solvers: Vec<Box<dyn Solver>>,
    backend: Box<dyn EquivalenceBackend>,
    tester: InputTester,
    pub elo: Elo,
}

impl Default for Tournament {
    /// linear-simplify, general-MBA (GAMBA) and local-search; no SMT backend.
    fn default() -> Self {
        Self::with_solvers(vec![
            Box::new(LinearSimplifySolver),
            Box::new(GambaSolver),
            Box::new(LocalSearchSolver::default()),
        ])
    }
}

impl Tournament {
    #[must_use]
    pub fn with_solvers(solvers: Vec<Box<dyn Solver>>) -> Self {
        Self {
            solvers,
            backend: Box::new(NoSolver),
            tester: InputTester::default(),
            elo: Elo::new(),
        }
    }

    /// Attach an SMT backend; without one nothing promotes beyond theorem proofs.
    #[must_use]
    pub fn with_backend(mut self, backend: Box<dyn EquivalenceBackend>) -> Self {
        self.backend = backend;
        self
    }

    /// Runs all applicable solvers on `p` and returns the best gated result.
    pub fn run(&mut self, p: &Problem) -> Option<Gated> {
        let class = match p.class {
            ProblemClass::Unknown if signature(&p.expr).is_ok() => ProblemClass::LinearMba,
            ProblemClass::Unknown => ProblemClass::NonlinearMba,
            c => c,
        };
        let routed = Problem::new(p.expr.clone(), class, p.ctx.clone());

        let mut results = Vec::new();
        let mut entered = Vec::new();
        for solver in self.solvers.iter().filter(|s| s.applies_to(class)) {
            let Some(candidate) = solver.solve(&routed) else {
                continue;
            };
            let score = self.tester.score(&p.expr, &candidate);
            let mut gate = PromotionGate::with_backend(BackendRef(&*self.backend));
            let j = judge_sampled(&mut gate, &p.expr, &candidate.expr);
            let eligible = score.is_eligible() && j.tier >= Tier::Observed;
            results.push(MatchResult {
                player: candidate.by.clone(),
                eligible,
                simplicity: score.simplicity,
                cost: candidate.cost,
            });
            if eligible {
                entered.push(Gated {
                    candidate,
                    tier: j.tier,
                    proof: j.proof,
                    score,
                });
            }
        }
        self.elo.run_round(class, &results);
        let elo = &self.elo;
        entered.into_iter().min_by(|a, b| {
            let key = |g: &Gated| {
                (
                    Reverse(g.tier),
                    g.score.simplicity,
                    g.candidate.cost,
                    Reverse(elo.rating(&g.candidate.by, class).to_bits()),
                )
            };
            key(a).cmp(&key(b)).then_with(|| a.candidate.by.cmp(&b.candidate.by))
        })
    }
}

/// Runs the default tournament (no SMT backend) on `problem` and returns the
/// best gated result, or `None` if no solver produced a verified candidate.
#[must_use]
pub fn run_pipeline(problem: &Problem) -> Option<Gated> {
    Tournament::default().run(problem)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vmpx_mutation::{BitWidth, Cost, Ctx, SolverId};
    use vmpx_vm::BinaryOp::{self, And, Mul, Or, Sub, Xor};

    const W: BitWidth = BitWidth::W64;

    fn v(n: &str) -> Expr {
        Expr::register(W, n)
    }
    fn b(op: BinaryOp, l: Expr, r: Expr) -> Expr {
        Expr::binary(op, l, r).unwrap()
    }
    fn obf_xor() -> Expr {
        b(Sub, b(Or, v("x"), v("y")), b(And, v("x"), v("y")))
    }
    fn nonlinear() -> Expr {
        b(Mul, b(Mul, v("x"), v("y")), obf_xor())
    }

    struct Unsat;
    impl EquivalenceBackend for Unsat {
        fn check_inequality(&self, _: &Expr, _: &Expr) -> SmtVerdict {
            SmtVerdict::Unsat
        }
    }

    /// Always proposes `x | y`: wrong for the problems below.
    struct Liar;
    impl Solver for Liar {
        fn id(&self) -> SolverId {
            SolverId::new("liar")
        }
        fn applies_to(&self, _: ProblemClass) -> bool {
            true
        }
        fn solve(&self, _: &Problem) -> Option<Candidate> {
            Some(Candidate {
                expr: b(Or, v("x"), v("y")),
                by: self.id(),
                cost: Cost::default(),
            })
        }
    }

    #[test]
    fn linear_mba_problem_returns_a_proven_result() {
        let p = Problem::new(obf_xor(), ProblemClass::LinearMba, Ctx::default());
        let g = run_pipeline(&p).expect("a result");
        assert_eq!(g.tier, Tier::Proven);
        assert_eq!(g.proof, Some(ProofKind::SignatureTheorem));
        assert_eq!(g.candidate.expr, b(Xor, v("x"), v("y")));
        assert!(g.score.is_eligible());
    }

    #[test]
    fn solverless_nonlinear_case_is_observed_not_proven() {
        let orig = nonlinear();
        let p = Problem::new(orig.clone(), ProblemClass::NonlinearMba, Ctx::default());
        let g = run_pipeline(&p).expect("a result");
        assert_eq!((g.tier, g.proof), (Tier::Observed, None));
        assert!(g.score.is_eligible());
        assert!(g.candidate.expr.node_count_bounded(1000) < orig.node_count_bounded(1000));
    }

    #[test]
    fn a_backend_unsat_promotes_the_same_nonlinear_result() {
        let p = Problem::new(nonlinear(), ProblemClass::NonlinearMba, Ctx::default());
        let g = Tournament::default()
            .with_backend(Box::new(Unsat))
            .run(&p)
            .expect("a result");
        assert_eq!((g.tier, g.proof), (Tier::Proven, Some(ProofKind::Smt)));
    }

    #[test]
    fn unknown_class_is_inferred_and_refuted_candidates_lose() {
        let p = Problem::new(obf_xor(), ProblemClass::Unknown, Ctx::default());
        let mut t = Tournament::with_solvers(vec![Box::new(Liar), Box::new(LinearSimplifySolver)]);
        let g = t.run(&p).expect("a result");
        assert_eq!(g.candidate.by, SolverId::new("linear_simplify"));
        // The liar was recorded as the loser on the linear ladder.
        assert_eq!(
            t.elo.rank(ProblemClass::LinearMba),
            vec![SolverId::new("linear_simplify"), SolverId::new("liar")]
        );
        // Alone, a refuted candidate yields nothing: fail closed.
        let mut only_liar = Tournament::with_solvers(vec![Box::new(Liar)]);
        assert!(only_liar.run(&p).is_none());
    }

    #[test]
    fn no_applicable_solver_yields_none() {
        let p = Problem::new(obf_xor(), ProblemClass::DeadCode, Ctx::default());
        assert!(run_pipeline(&p).is_none());
    }
}
