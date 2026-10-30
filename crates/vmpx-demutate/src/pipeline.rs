//! `demutate`: route a problem to the top-ELO players for its class and iterate
//! to a fixpoint.

use vmpx_mutation::{Candidate, Cost, Ctx, Expr, Problem, ProblemClass};
use vmpx_solvers::Solver;
use vmpx_tournament::Elo;
use vmpx_verify::input_testing::InputTester;
use vmpx_verify::smt::SmtGate;
use vmpx_verify::{Gate, Scorer, Verdict};

use crate::registry::roster;

const TOP_N: usize = 3;
const MAX_ITERS: usize = 8;

/// Assurance tier of a demutated result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Tier {
    /// Gate-proven equivalent to the input.
    Sound,
    /// Agrees on all tested inputs only; not proven.
    Experimental,
}

#[derive(Clone, Debug)]
pub struct Demutated {
    pub expr: Expr,
    pub tier: Tier,
}

/// Players to try for `class`: top `TOP_N` by ELO that apply, else any applicable.
fn players_for(class: ProblemClass, elo: &Elo) -> Vec<Box<dyn Solver>> {
    let mut all = roster();
    let mut chosen: Vec<Box<dyn Solver>> = Vec::new();
    for id in elo.rank(class).into_iter().take(TOP_N) {
        if let Some(i) = all.iter().position(|s| s.id() == id) {
            chosen.push(all.swap_remove(i));
        }
    }
    if chosen.is_empty() {
        chosen = all.into_iter().filter(|s| s.applies_to(class)).collect();
    }
    chosen
}

/// Simplify `expr`, tiered sound/experimental. `None` if nothing got simpler.
#[must_use]
pub fn demutate_verified(expr: &Expr, class: ProblemClass, elo: &Elo) -> Option<Demutated> {
    let players = players_for(class, elo);
    let tester = InputTester::default();
    let size_of = |e: &Expr| {
        tester
            .score(
                expr,
                &Candidate {
                    expr: e.clone(),
                    by: vmpx_mutation::SolverId::new("size"),
                    cost: Cost::default(),
                },
            )
            .simplicity
    };
    let mut current = expr.clone();
    let mut current_size = size_of(&current);
    for _ in 0..MAX_ITERS {
        let problem = Problem::new(
            current.clone(),
            class,
            Ctx {
                width: current.width,
            },
        );
        let best = players
            .iter()
            .filter(|s| s.applies_to(class))
            .filter_map(|s| s.solve(&problem))
            .map(|c| (tester.score(expr, &c), c))
            .filter(|(sc, _)| sc.is_eligible() && sc.simplicity < current_size)
            .min_by_key(|(sc, _)| sc.simplicity);
        let Some((sc, cand)) = best else { break };
        current = cand.expr;
        current_size = sc.simplicity;
    }
    if current == *expr {
        return None;
    }
    match SmtGate.verify(expr, &current) {
        Verdict::Refuted => None,
        Verdict::Proven => Some(Demutated {
            expr: current,
            tier: Tier::Sound,
        }),
        Verdict::Unknown => Some(Demutated {
            expr: current,
            tier: Tier::Experimental,
        }),
    }
}

/// Simplify `expr` using the top-ELO solvers for `class`, to a fixpoint.
#[must_use]
pub fn demutate(expr: &Expr, class: ProblemClass, elo: &Elo) -> Option<Expr> {
    demutate_verified(expr, class, elo).map(|d| d.expr)
}
