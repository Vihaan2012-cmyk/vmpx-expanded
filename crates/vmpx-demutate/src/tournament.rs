//! Tournament runner: every applicable player attempts every benchmark problem;
//! input-testing decides eligibility, ELO ranks per class.

use std::collections::HashMap;
use std::fmt::Write as _;

use vmpx_mutation::{BitWidth, Candidate, Cost, Ctx, Problem, ProblemClass, SolverId};
use vmpx_tournament::{Elo, MatchResult};
use vmpx_verify::input_testing::InputTester;
use vmpx_verify::smt::SmtGate;
use vmpx_verify::{Gate, Scorer, Verdict};

use crate::bench::{BENCH_CLASSES, BenchProblem, generate};
use crate::registry::roster;

#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub seed: u64,
    pub per_class: usize,
    pub width: BitWidth,
    /// Run the SMT gate on eligible candidates (sound vs experimental tag).
    pub gate: bool,
    pub cost_weight: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            seed: 0x5EED,
            per_class: 14,
            width: BitWidth::W8,
            gate: true,
            cost_weight: 10.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    pub attempted: u32,
    pub produced: u32,
    /// Candidate agreed on every tested input.
    pub eligible: u32,
    /// Eligible and simplest among eligible candidates on the problem.
    pub wins: u32,
    /// Eligible and no larger than the known clean form.
    pub recovered: u32,
    /// Eligible and proven equivalent by the gate.
    pub proven: u32,
}

pub struct Report {
    pub elo: Elo,
    pub stats: HashMap<(ProblemClass, SolverId), Stats>,
    pub problems: HashMap<ProblemClass, u32>,
    pub cost_weight: f64,
    pub gate: bool,
}

#[must_use]
pub fn run(cfg: &Config) -> Report {
    let bench = generate(cfg.seed, cfg.per_class, cfg.width);
    run_on(&bench, cfg)
}

#[must_use]
pub fn run_on(bench: &[BenchProblem], cfg: &Config) -> Report {
    let players = roster();
    let tester = InputTester::default();
    let mut report = Report {
        elo: Elo::new(),
        stats: HashMap::new(),
        problems: HashMap::new(),
        cost_weight: cfg.cost_weight,
        gate: cfg.gate,
    };
    for bp in bench {
        *report.problems.entry(bp.class).or_default() += 1;
        let problem = Problem::new(
            bp.obfuscated.clone(),
            bp.class,
            Ctx { width: cfg.width },
        );
        let clean_size = tester
            .score(
                &bp.obfuscated,
                &Candidate {
                    expr: bp.clean.clone(),
                    by: SolverId::new("clean"),
                    cost: Cost::default(),
                },
            )
            .simplicity;

        let mut results = Vec::new();
        let mut scored: Vec<(SolverId, Option<(Candidate, u64)>)> = Vec::new();
        for s in players.iter().filter(|s| s.applies_to(bp.class)) {
            let id = s.id();
            let st = report.stats.entry((bp.class, id.clone())).or_default();
            st.attempted += 1;
            let Some(cand) = s.solve(&problem) else {
                results.push(MatchResult {
                    player: id.clone(),
                    eligible: false,
                    simplicity: u64::MAX,
                    cost: Cost::default(),
                });
                scored.push((id, None));
                continue;
            };
            st.produced += 1;
            let score = tester.score(&bp.obfuscated, &cand);
            let eligible = score.is_eligible();
            if eligible {
                st.eligible += 1;
                if score.simplicity <= clean_size {
                    st.recovered += 1;
                }
                if cfg.gate && SmtGate.verify(&bp.obfuscated, &cand.expr) == Verdict::Proven {
                    st.proven += 1;
                }
            }
            results.push(MatchResult {
                player: id.clone(),
                eligible,
                simplicity: score.simplicity,
                cost: score.cost,
            });
            scored.push((id, eligible.then_some((cand, score.simplicity))));
        }
        let best = scored
            .iter()
            .filter_map(|(_, c)| c.as_ref().map(|(_, s)| *s))
            .min();
        if let Some(best) = best {
            for (id, c) in &scored {
                if c.as_ref().is_some_and(|(_, s)| *s == best) {
                    report.stats.entry((bp.class, id.clone())).or_default().wins += 1;
                }
            }
        }
        report.elo.run_round(bp.class, &results);
    }
    report
}

impl Report {
    fn rows(&self, class: ProblemClass) -> Vec<SolverId> {
        let mut order = self.elo.rank(class);
        let mut rest: Vec<SolverId> = self
            .stats
            .keys()
            .filter(|(c, id)| *c == class && !order.contains(id))
            .map(|(_, id)| id.clone())
            .collect();
        rest.sort();
        order.extend(rest);
        order
    }

    /// Top player per class by ELO. A class with a single applicable player has
    /// no matchups (hence no ratings), so fall back to most wins then solves.
    #[must_use]
    pub fn winner(&self, class: ProblemClass) -> Option<SolverId> {
        self.elo.select(class).or_else(|| {
            self.stats
                .iter()
                .filter(|((c, _), st)| *c == class && st.eligible > 0)
                .max_by(|a, b| {
                    (a.1.wins, a.1.eligible)
                        .cmp(&(b.1.wins, b.1.eligible))
                        .then_with(|| b.0.1.cmp(&a.0.1))
                })
                .map(|((_, id), _)| id.clone())
        })
    }

    #[must_use]
    pub fn render(&self) -> String {
        let mut o = String::new();
        let _ = writeln!(
            o,
            "SP6 solver tournament (ELO is empirical; solved = input-test agreement 1.0{})",
            if self.gate {
                "; proven = SMT gate Proven (sound tier), rest experimental"
            } else {
                "; gate disabled, all results experimental"
            }
        );
        let mut overall: HashMap<SolverId, Stats> = HashMap::new();
        let mut summary = String::new();
        for class in BENCH_CLASSES {
            let n = self.problems.get(&class).copied().unwrap_or(0);
            let _ = writeln!(o, "\n== {class:?} ({n} problems) ==");
            let _ = writeln!(
                o,
                " #  {:<44} {:>7} {:>5} {:>7} {:>5} {:>9} {:>6}",
                "player", "ELO", "ca#", "solved", "wins", "recovered", "proven"
            );
            let cost_rank = self.elo.rank_cost_aware(class, self.cost_weight);
            for (i, id) in self.rows(class).iter().enumerate() {
                let st = self.stats.get(&(class, id.clone())).copied().unwrap_or_default();
                let ca = cost_rank.iter().position(|p| p == id).map_or(0, |p| p + 1);
                let _ = writeln!(
                    o,
                    "{:>2}  {:<44} {:>7.1} {:>5} {:>3}/{:<3} {:>5} {:>9} {:>6}",
                    i + 1,
                    id.as_str(),
                    self.elo.rating(id, class),
                    ca,
                    st.eligible,
                    st.attempted,
                    st.wins,
                    st.recovered,
                    st.proven
                );
                let t = overall.entry(id.clone()).or_default();
                t.attempted += st.attempted;
                t.produced += st.produced;
                t.eligible += st.eligible;
                t.wins += st.wins;
                t.recovered += st.recovered;
                t.proven += st.proven;
            }
            let top = self.winner(class).map_or("-".to_string(), |s| s.0);
            let top_ca = cost_rank.first().map_or_else(|| top.clone(), |s| s.0.clone());
            let _ = writeln!(summary, "  {class:<16?} ELO: {top}   cost-aware: {top_ca}");
        }
        let _ = writeln!(o, "\n== Winner per class ==\n{summary}");
        let _ = writeln!(o, "== Overall solve rates (applicable attempts only) ==");
        let mut all: Vec<_> = overall.into_iter().collect();
        all.sort_by(|a, b| b.1.eligible.cmp(&a.1.eligible).then_with(|| a.0.cmp(&b.0)));
        for (id, st) in all {
            let _ = writeln!(
                o,
                "  {:<44} solved {:>3}/{:<3} ({:>5.1}%)  wins {:>3}  recovered {:>3}  proven {:>3}",
                id.as_str(),
                st.eligible,
                st.attempted,
                100.0 * f64::from(st.eligible) / f64::from(st.attempted.max(1)),
                st.wins,
                st.recovered,
                st.proven
            );
        }
        o
    }
}
