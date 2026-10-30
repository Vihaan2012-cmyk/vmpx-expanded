//! SP6 tournament: the per-(player, class) ELO ladder that ranks solvers and
//! routes each sub-problem to the top-rated player(s) for its class. Depends on
//! `vmpx-mutation` only.
//!
//! WAVE2: WP4 extends this with matchmaking over the benchmark corpus and a
//! fuller, cost-aware selection policy. The registry + ratings below are
//! functional today.

use std::collections::HashMap;

use vmpx_mutation::{ProblemClass, SolverId};

const DEFAULT_RATING: f64 = 1500.0;
const K_FACTOR: f64 = 32.0;

/// Per-(player, class) ELO ratings. Ratings are honest because the scorer
/// (differential input-testing) decides each matchup; this only bookkeeps.
#[derive(Clone, Debug, Default)]
pub struct Elo {
    ratings: HashMap<(SolverId, ProblemClass), f64>,
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
}
