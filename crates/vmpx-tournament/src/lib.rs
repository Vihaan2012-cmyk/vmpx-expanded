//! SP6 tournament: the per-(player, class) ELO ladder that ranks solvers and
//! routes each sub-problem to the top-rated player(s) for its class. Depends on
//! `vmpx-mutation` only.
//!
//! Matchmaking (`run_round`) turns per-problem results into pairwise ELO
//! outcomes; `rank_cost_aware` discounts ratings by observed solver cost.

use std::cmp::Ordering;
use std::collections::HashMap;

use vmpx_mutation::{Cost, ProblemClass, SolverId};

const DEFAULT_RATING: f64 = 1500.0;
const K_FACTOR: f64 = 32.0;

/// One player's result on a single problem.
#[derive(Clone, Debug)]
pub struct MatchResult {
    pub player: SolverId,
    /// Passed input-testing (agreement 1.0).
    pub eligible: bool,
    /// Lower is simpler.
    pub simplicity: u64,
    pub cost: Cost,
}

/// Scalar cost: solver steps plus wall-clock milliseconds.
fn cost_scalar(c: Cost) -> f64 {
    c.steps as f64 + c.time_ns as f64 / 1_000_000.0
}

/// Per-(player, class) ELO ratings. Ratings are honest because the scorer
/// (differential input-testing) decides each matchup; this only bookkeeps.
#[derive(Clone, Debug, Default)]
pub struct Elo {
    ratings: HashMap<(SolverId, ProblemClass), f64>,
    /// (total scalar cost, rounds) per (player, class).
    costs: HashMap<(SolverId, ProblemClass), (f64, u64)>,
}

impl Elo {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Current rating of `player` for `class` (default 1500 if unseen).
    #[must_use]
    pub fn rating(&self, player: &SolverId, class: ProblemClass) -> f64 {
        self.ratings
            .get(&(player.clone(), class))
            .copied()
            .unwrap_or(DEFAULT_RATING)
    }

    /// Record one matchup for `class`: `winner` beat `loser`. Standard ELO
    /// update with a fixed K-factor.
    pub fn record(&mut self, class: ProblemClass, winner: SolverId, loser: SolverId) {
        let rw = self.rating(&winner, class);
        let rl = self.rating(&loser, class);
        let expected_w = 1.0 / (1.0 + 10f64.powf((rl - rw) / 400.0));
        let expected_l = 1.0 - expected_w;
        self.ratings
            .insert((winner, class), rw + K_FACTOR * (1.0 - expected_w));
        self.ratings
            .insert((loser, class), rl + K_FACTOR * (0.0 - expected_l));
    }

    /// Players seen for `class`, best rating first.
    #[must_use]
    pub fn rank(&self, class: ProblemClass) -> Vec<SolverId> {
        let mut seen: Vec<(SolverId, f64)> = self
            .ratings
            .iter()
            .filter(|((_, c), _)| *c == class)
            .map(|((p, _), r)| (p.clone(), *r))
            .collect();
        seen.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        seen.into_iter().map(|(p, _)| p).collect()
    }

    /// Like [`Elo::rank`] but orders by `rating - cost_weight * ln(1 + mean_cost)`,
    /// where `mean_cost` is the player's average scalar cost (steps + ms) over
    /// rounds run via [`Elo::run_round`]. `cost_weight == 0` equals pure ELO.
    #[must_use]
    pub fn rank_cost_aware(&self, class: ProblemClass, cost_weight: f64) -> Vec<SolverId> {
        let mut seen: Vec<(SolverId, f64)> = self
            .ratings
            .iter()
            .filter(|((_, c), _)| *c == class)
            .map(|((p, c), r)| {
                let mean = self
                    .costs
                    .get(&(p.clone(), *c))
                    .map_or(0.0, |(t, n)| t / (*n).max(1) as f64);
                (p.clone(), r - cost_weight * mean.ln_1p())
            })
            .collect();
        seen.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        seen.into_iter().map(|(p, _)| p).collect()
    }

    /// Record every pairwise outcome of one problem. Eligible beats ineligible;
    /// among eligible, lower simplicity wins, then lower cost. Fully tied or
    /// both-ineligible pairs record nothing. Per-player cost is accumulated.
    pub fn run_round(&mut self, class: ProblemClass, results: &[MatchResult]) {
        for r in results {
            let e = self.costs.entry((r.player.clone(), class)).or_default();
            e.0 += cost_scalar(r.cost);
            e.1 += 1;
        }
        for (i, a) in results.iter().enumerate() {
            for b in &results[i + 1..] {
                let ord = match (a.eligible, b.eligible) {
                    (true, false) => Ordering::Less,
                    (false, true) => Ordering::Greater,
                    (false, false) => continue,
                    (true, true) => a
                        .simplicity
                        .cmp(&b.simplicity)
                        .then_with(|| a.cost.cmp(&b.cost)),
                };
                match ord {
                    Ordering::Less => self.record(class, a.player.clone(), b.player.clone()),
                    Ordering::Greater => self.record(class, b.player.clone(), a.player.clone()),
                    Ordering::Equal => {}
                }
            }
        }
    }

    /// Top-rated player for `class`, if any have competed.
    #[must_use]
    pub fn select(&self, class: ProblemClass) -> Option<SolverId> {
        self.rank(class).into_iter().next()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_then_rank_orders_winner_first() {
        let mut elo = Elo::new();
        let a = SolverId::new("a");
        let b = SolverId::new("b");
        elo.record(ProblemClass::LinearMba, a.clone(), b.clone());

        assert!(elo.rating(&a, ProblemClass::LinearMba) > DEFAULT_RATING);
        assert!(elo.rating(&b, ProblemClass::LinearMba) < DEFAULT_RATING);

        let ranked = elo.rank(ProblemClass::LinearMba);
        assert_eq!(ranked, vec![a.clone(), b]);
        assert_eq!(elo.select(ProblemClass::LinearMba), Some(a));
    }

    #[test]
    fn ratings_are_per_class() {
        let mut elo = Elo::new();
        let a = SolverId::new("a");
        let b = SolverId::new("b");
        elo.record(ProblemClass::DeadCode, a.clone(), b);
        // A win in DeadCode must not bleed into another class.
        assert!(elo.rank(ProblemClass::OpaquePredicate).is_empty());
        assert_eq!(
            elo.rating(&a, ProblemClass::OpaquePredicate),
            DEFAULT_RATING
        );
    }

    fn mr(name: &str, eligible: bool, simplicity: u64, steps: u64) -> MatchResult {
        MatchResult {
            player: SolverId::new(name),
            eligible,
            simplicity,
            cost: Cost { time_ns: 0, steps },
        }
    }

    #[test]
    fn repeated_winner_ranks_above_loser() {
        let mut elo = Elo::new();
        for _ in 0..3 {
            elo.run_round(
                ProblemClass::LinearMba,
                &[mr("a", true, 1, 1), mr("b", true, 5, 1)],
            );
        }
        assert_eq!(
            elo.rank(ProblemClass::LinearMba),
            vec![SolverId::new("a"), SolverId::new("b")]
        );
    }

    #[test]
    fn ineligible_loses_regardless_of_simplicity() {
        let mut elo = Elo::new();
        elo.run_round(
            ProblemClass::DeadCode,
            &[mr("bad", false, 1, 1), mr("ok", true, 99, 1)],
        );
        assert_eq!(
            elo.select(ProblemClass::DeadCode),
            Some(SolverId::new("ok"))
        );
    }

    #[test]
    fn eligible_lower_simplicity_wins_then_cost() {
        let mut elo = Elo::new();
        elo.run_round(
            ProblemClass::DeadCode,
            &[mr("big", true, 9, 1), mr("small", true, 2, 100)],
        );
        assert_eq!(
            elo.select(ProblemClass::DeadCode),
            Some(SolverId::new("small"))
        );
        let mut elo = Elo::new();
        elo.run_round(
            ProblemClass::DeadCode,
            &[mr("slow", true, 2, 100), mr("fast", true, 2, 1)],
        );
        assert_eq!(
            elo.select(ProblemClass::DeadCode),
            Some(SolverId::new("fast"))
        );
    }

    #[test]
    fn cost_aware_demotes_expensive_winner() {
        let mut elo = Elo::new();
        elo.run_round(
            ProblemClass::LinearMba,
            &[mr("costly", true, 1, 1_000_000), mr("cheap", true, 2, 1)],
        );
        let c = ProblemClass::LinearMba;
        assert_eq!(elo.rank_cost_aware(c, 0.0), elo.rank(c));
        assert_eq!(elo.rank(c)[0], SolverId::new("costly"));
        assert_eq!(elo.rank_cost_aware(c, 10.0)[0], SolverId::new("cheap"));
    }

    #[test]
    fn run_round_ratings_are_per_class() {
        let mut elo = Elo::new();
        elo.run_round(
            ProblemClass::LinearMba,
            &[mr("a", true, 1, 1), mr("b", true, 2, 1)],
        );
        assert!(elo.rank(ProblemClass::OpaquePredicate).is_empty());
        assert_eq!(
            elo.rating(&SolverId::new("a"), ProblemClass::OpaquePredicate),
            DEFAULT_RATING
        );
    }
}
