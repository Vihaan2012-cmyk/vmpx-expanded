//! Xyntia-style local-search synthesizer.
//!
//! Iterated hill-climbing over a small bitvector grammar
//! (`~ -`, `+ - * & | ^`, `<< >>` by constant, variables, constants). The
//! fitness is lexicographic `(bit-level Hamming distance to the I/O oracle,
//! size)`, so the search first matches the oracle and then shrinks. It can be
//! seeded with the symbolic expression (start from it and shrink) or run from
//! random trees, and it uses the sampling recipe random + {0, 1, -1, min, max}.
//!
//! Every training-exact candidate is checked by `vmpx-verify`'s `InputTester`
//! on a much larger input set; a divergence becomes a new training sample
//! (CEGIS-style). The output is a CANDIDATE: input agreement is not a proof,
//! so it must still pass the `PromotionGate`.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use crate::Solver;
use vmpx_mutation::sampling::{next_rand, sample_assignments};
use vmpx_mutation::{
    BitWidth, Candidate, Cost, Expr, Problem, ProblemClass, SolverId, eval_concrete, registers,
};
use vmpx_verify::input_testing::InputTester;
use vmpx_vm::{BinaryOp, ExprKind, UnaryOp};

const MAX_VARS: usize = 6;
const NODE_LIMIT: usize = 4096;
const MIN_TRAIN: usize = 4;

#[derive(Clone, Debug)]
pub struct LocalSearchConfig {
    pub max_evals: u64,
    /// Evaluations without improvement before a run restarts from scratch.
    pub stall_limit: u64,
    pub time_budget: Duration,
    pub seed: u64,
    pub train_random: usize,
}

impl Default for LocalSearchConfig {
    fn default() -> Self {
        Self {
            max_evals: 300_000,
            stall_limit: 15_000,
            time_budget: Duration::from_millis(2500),
            seed: 0xC0FF_EE15_BADC_0DE5,
            train_random: 24,
        }
    }
}

/// Verdict of the held-out check on a training-exact candidate.
#[derive(Clone, Debug)]
pub enum Check {
    Accept,
    /// A real input on which the candidate differs; becomes a training sample.
    Counterexample(BTreeMap<String, u64>),
    Reject,
}

#[derive(Clone, Debug)]
pub struct Synth {
    pub expr: Expr,
    pub evals: u64,
}

#[derive(Clone, Copy, PartialEq)]
enum Un {
    Not,
    Neg,
}

#[derive(Clone, Copy, PartialEq)]
enum Bi {
    Add,
    Sub,
    Mul,
    And,
    Or,
    Xor,
}

const BIS: [Bi; 6] = [Bi::Add, Bi::Sub, Bi::Mul, Bi::And, Bi::Or, Bi::Xor];

#[derive(Clone)]
enum T {
    Var(usize),
    C(u64),
    U(Un, Box<T>),
    B(Bi, Box<T>, Box<T>),
    Shl(Box<T>, u32),
    Shr(Box<T>, u32),
}

impl T {
    fn eval(&self, xs: &[u64], m: u64) -> u64 {
        match self {
            Self::Var(i) => xs[*i],
            Self::C(c) => *c,
            Self::U(Un::Not, a) => !a.eval(xs, m) & m,
            Self::U(Un::Neg, a) => a.eval(xs, m).wrapping_neg() & m,
            Self::B(op, a, b) => {
                let (x, y) = (a.eval(xs, m), b.eval(xs, m));
                match op {
                    Bi::Add => x.wrapping_add(y) & m,
                    Bi::Sub => x.wrapping_sub(y) & m,
                    Bi::Mul => x.wrapping_mul(y) & m,
                    Bi::And => x & y,
                    Bi::Or => x | y,
                    Bi::Xor => x ^ y,
                }
            }
            Self::Shl(a, k) => (a.eval(xs, m) << k) & m,
            Self::Shr(a, k) => a.eval(xs, m) >> k,
        }
    }

    fn size(&self) -> usize {
        match self {
            Self::Var(_) | Self::C(_) => 1,
            Self::U(_, a) | Self::Shl(a, _) | Self::Shr(a, _) => 1 + a.size(),
            Self::B(_, a, b) => 1 + a.size() + b.size(),
        }
    }

    fn to_expr(&self, names: &[String], w: BitWidth) -> Option<Expr> {
        let bin = |op, a: &Self, b: Expr| Expr::binary(op, a.to_expr(names, w)?, b).ok();
        Some(match self {
            Self::Var(i) => Expr::register(w, names[*i].clone()),
            Self::C(c) => Expr::constant(w, *c),
            Self::U(op, a) => Expr::unary(
                if *op == Un::Not { UnaryOp::Not } else { UnaryOp::Neg },
                a.to_expr(names, w)?,
            ),
            Self::B(op, a, b) => Expr::binary(
                match op {
                    Bi::Add => BinaryOp::Add,
                    Bi::Sub => BinaryOp::Sub,
                    Bi::Mul => BinaryOp::Mul,
                    Bi::And => BinaryOp::And,
                    Bi::Or => BinaryOp::Or,
                    Bi::Xor => BinaryOp::Xor,
                },
                a.to_expr(names, w)?,
                b.to_expr(names, w)?,
            )
            .ok()?,
            Self::Shl(a, k) => bin(BinaryOp::Shl, a, Expr::constant(w, u64::from(*k)))?,
            Self::Shr(a, k) => bin(BinaryOp::Shr, a, Expr::constant(w, u64::from(*k)))?,
        })
    }

    /// Converts a symbolic expression to the grammar, or `None` if it leaves it.
    fn from_expr(e: &Expr, names: &[String], w: BitWidth) -> Option<Self> {
        if e.width != w {
            return None;
        }
        let m = mask(w);
        Some(match &e.kind {
            ExprKind::Register { name } => Self::Var(names.iter().position(|n| n == name)?),
            ExprKind::Const { value } => Self::C(value & m),
            ExprKind::Unary { op, value } => Self::U(
                if *op == UnaryOp::Not { Un::Not } else { Un::Neg },
                Box::new(Self::from_expr(value, names, w)?),
            ),
            ExprKind::Binary { op, left, right } => {
                let shift = |k: &Expr| match k.kind {
                    ExprKind::Const { value } if value < u64::from(w.bits()) => u32::try_from(value).ok(),
                    _ => None,
                };
                let bi = match op {
                    BinaryOp::Add => Bi::Add,
                    BinaryOp::Sub => Bi::Sub,
                    BinaryOp::Mul => Bi::Mul,
                    BinaryOp::And => Bi::And,
                    BinaryOp::Or => Bi::Or,
                    BinaryOp::Xor => Bi::Xor,
                    BinaryOp::Shl => {
                        return Some(Self::Shl(
                            Box::new(Self::from_expr(left, names, w)?),
                            shift(right)?,
                        ));
                    }
                    BinaryOp::Shr => {
                        return Some(Self::Shr(
                            Box::new(Self::from_expr(left, names, w)?),
                            shift(right)?,
                        ));
                    }
                    _ => return None,
                };
                Self::B(
                    bi,
                    Box::new(Self::from_expr(left, names, w)?),
                    Box::new(Self::from_expr(right, names, w)?),
                )
            }
            _ => return None,
        })
    }
}

const fn mask(w: BitWidth) -> u64 {
    u64::MAX >> (64 - w.bits())
}

struct Rng(u64);

impl Rng {
    fn below(&mut self, n: usize) -> usize {
        next_rand(&mut self.0) as usize % n.max(1)
    }
    fn chance(&mut self, pct: usize) -> bool {
        self.below(100) < pct
    }
}

struct Space {
    nvars: usize,
    w: BitWidth,
    m: u64,
}

impl Space {
    fn leaf(&self, r: &mut Rng) -> T {
        if self.nvars > 0 && r.chance(75) {
            return T::Var(r.below(self.nvars));
        }
        let pool = [
            0,
            1,
            2,
            3,
            4,
            8,
            self.m,
            1 << (self.w.bits() - 1),
            self.m >> 1,
            u64::from(self.w.bits() - 1),
        ];
        T::C(if r.chance(15) {
            next_rand(&mut r.0) & self.m
        } else {
            pool[r.below(pool.len())]
        })
    }

    fn tree(&self, r: &mut Rng, depth: u32) -> T {
        if depth == 0 || r.chance(25) {
            return self.leaf(r);
        }
        match r.below(10) {
            0 => T::U(Un::Not, Box::new(self.tree(r, depth - 1))),
            1 => T::U(Un::Neg, Box::new(self.tree(r, depth - 1))),
            2 => {
                let t = self.tree(r, depth - 1);
                self.shift(r, t)
            }
            _ => T::B(
                BIS[r.below(BIS.len())],
                Box::new(self.tree(r, depth - 1)),
                Box::new(self.tree(r, depth - 1)),
            ),
        }
    }

    fn shift(&self, r: &mut Rng, t: T) -> T {
        let k = 1 + r.below(self.w.bits() as usize - 1) as u32;
        if r.chance(50) {
            T::Shl(Box::new(t), k)
        } else {
            T::Shr(Box::new(t), k)
        }
    }

    fn mutate(&self, t: &T, r: &mut Rng) -> T {
        let mut out = t.clone();
        let mut idx = r.below(t.size());
        let kind = r.below(7);
        self.at(&mut out, &mut idx, &mut |n, r| self.edit(n, kind, r), r);
        out
    }

    fn at(&self, t: &mut T, idx: &mut usize, f: &mut dyn FnMut(&mut T, &mut Rng), r: &mut Rng) {
        if *idx == 0 {
            f(t, r);
            *idx = usize::MAX;
            return;
        }
        *idx -= 1;
        match t {
            T::Var(_) | T::C(_) => {}
            T::U(_, a) | T::Shl(a, _) | T::Shr(a, _) => self.at(a, idx, f, r),
            T::B(_, a, b) => {
                self.at(a, idx, f, r);
                if *idx != usize::MAX {
                    self.at(b, idx, f, r);
                }
            }
        }
    }

    fn edit(&self, n: &mut T, kind: usize, r: &mut Rng) {
        let old = std::mem::replace(n, T::C(0));
        *n = match (kind, old) {
            (0, _) => self.tree(r, 2),
            (1, T::B(_, a, b)) => T::B(BIS[r.below(BIS.len())], a, b),
            (1, T::U(op, a)) => T::U(if op == Un::Not { Un::Neg } else { Un::Not }, a),
            (1 | 2, T::Shl(a, _) | T::Shr(a, _)) => self.shift(r, *a),
            (1, _) => self.leaf(r),
            (2, old) => T::U(if r.chance(50) { Un::Not } else { Un::Neg }, Box::new(old)),
            (3 | 4, T::B(_, a, b)) => if r.chance(50) { *a } else { *b },
            (3 | 4, T::U(_, a) | T::Shl(a, _) | T::Shr(a, _)) => *a,
            (4, _) => self.leaf(r),
            (5, old) => {
                let other = self.leaf(r);
                T::B(BIS[r.below(BIS.len())], Box::new(old), Box::new(other))
            }
            (_, old) => self.shift(r, old),
        };
    }
}

struct Train {
    xs: Vec<Vec<u64>>,
    ys: Vec<u64>,
}

impl Train {
    fn dist(&self, t: &T, m: u64) -> u64 {
        self.xs
            .iter()
            .zip(&self.ys)
            .map(|(x, y)| u64::from(((t.eval(x, m) ^ y) & m).count_ones()))
            .sum()
    }
}

/// Searches for a small expression over `names` (all at `width`) that
/// reproduces `oracle`. `seed` starts the first run from a symbolic expression.
/// `check` vets every training-exact candidate smaller than the best so far.
pub fn synthesize(
    width: BitWidth,
    names: &[String],
    oracle: &dyn Fn(&BTreeMap<String, u64>) -> Option<u64>,
    seed: Option<&Expr>,
    check: &mut dyn FnMut(&Expr) -> Check,
    cfg: &LocalSearchConfig,
) -> Option<Synth> {
    if names.len() > MAX_VARS {
        return None;
    }
    let m = mask(width);
    let vars: BTreeMap<String, BitWidth> = names.iter().map(|n| (n.clone(), width)).collect();
    let mut train = Train {
        xs: Vec::new(),
        ys: Vec::new(),
    };
    let add = |train: &mut Train, env: &BTreeMap<String, u64>| -> Option<()> {
        let y = oracle(env)? & m;
        train.xs.push(names.iter().map(|n| env[n]).collect());
        train.ys.push(y);
        Some(())
    };
    for env in sample_assignments(&vars, cfg.train_random, cfg.seed) {
        let _ = add(&mut train, &env);
    }
    if train.xs.len() < MIN_TRAIN {
        return None;
    }

    let space = Space {
        nvars: names.len(),
        w: width,
        m,
    };
    let mut rng = Rng(cfg.seed);
    let seed_tree = seed.and_then(|e| T::from_expr(e, names, width));
    let mut best: Option<(T, usize)> = None;
    let mut best_size = seed_tree.as_ref().map_or(usize::MAX, T::size);
    let (start, mut evals) = (Instant::now(), 0_u64);

    'runs: for run in 0_u32.. {
        let mut cur = match (&seed_tree, run) {
            (Some(s), 0) => s.clone(),
            _ => space.tree(&mut rng, 3),
        };
        let mut fit = (train.dist(&cur, m), cur.size());
        let mut stall = 0_u64;
        while stall < cfg.stall_limit {
            if evals >= cfg.max_evals || start.elapsed() > cfg.time_budget {
                break 'runs;
            }
            evals += 1;
            let cand = space.mutate(&cur, &mut rng);
            let f = (train.dist(&cand, m), cand.size());
            stall = if f < fit { 0 } else { stall + 1 };
            if f > fit {
                continue;
            }
            (cur, fit) = (cand, f);
            if fit.0 == 0 && fit.1 < best_size {
                let Some(e) = cur.to_expr(names, width) else {
                    continue;
                };
                match check(&e) {
                    Check::Accept => {
                        best_size = fit.1;
                        best = Some((cur.clone(), fit.1));
                    }
                    Check::Counterexample(env) => {
                        if env.len() == names.len() && add(&mut train, &env).is_some() {
                            fit = (train.dist(&cur, m), cur.size());
                        }
                    }
                    Check::Reject => {}
                }
            }
        }
        if best.is_some() {
            break;
        }
    }
    let (t, _) = best?;
    Some(Synth {
        expr: t.to_expr(names, width)?,
        evals,
    })
}

/// Local-search player for the tournament.
#[derive(Default)]
pub struct LocalSearchSolver {
    pub cfg: LocalSearchConfig,
}

impl Solver for LocalSearchSolver {
    fn id(&self) -> SolverId {
        SolverId::new("local_search")
    }

    fn applies_to(&self, class: ProblemClass) -> bool {
        matches!(
            class,
            ProblemClass::NonlinearMba | ProblemClass::LinearMba | ProblemClass::Unknown
        )
    }

    fn solve(&self, p: &Problem) -> Option<Candidate> {
        let started = Instant::now();
        let width = p.expr.width;
        let regs = registers(&p.expr)?;
        if regs.values().any(|w| *w != width) {
            return None;
        }
        let names: Vec<String> = regs.into_keys().collect();
        let none = BTreeMap::new();
        let oracle = |env: &BTreeMap<String, u64>| eval_concrete(&p.expr, env, &none);
        let tester = InputTester {
            random_inputs: 200,
            grid_cap: 300,
        };
        let mut check = |cand: &Expr| {
            let r = tester.test(&p.expr, cand);
            match r.divergence {
                Some(d) => Check::Counterexample(d.into_iter().collect()),
                None if r.total > 0 && r.equal == r.total => Check::Accept,
                None => Check::Reject,
            }
        };
        let s = synthesize(width, &names, &oracle, Some(&p.expr), &mut check, &self.cfg)?;
        if s.expr.node_count_bounded(NODE_LIMIT) >= p.expr.node_count_bounded(NODE_LIMIT) {
            return None;
        }
        Some(Candidate {
            expr: s.expr,
            by: self.id(),
            cost: Cost {
                time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
                steps: s.evals,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vmpx_mutation::Ctx;
    use vmpx_mutation::gate::{
        EquivalenceBackend, NoSolver, PromotionGate, ProofKind, SmtVerdict, Tier,
    };
    use vmpx_mutation::sampling::judge_sampled;

    const W: BitWidth = BitWidth::W64;

    fn v(n: &str) -> Expr {
        Expr::register(W, n)
    }
    fn b(op: BinaryOp, l: Expr, r: Expr) -> Expr {
        Expr::binary(op, l, r).unwrap()
    }
    fn xor() -> Expr {
        b(BinaryOp::Xor, v("x"), v("y"))
    }
    fn names() -> Vec<String> {
        vec!["x".into(), "y".into()]
    }
    fn tester_check(reference: Expr) -> impl FnMut(&Expr) -> Check {
        let t = InputTester {
            random_inputs: 200,
            grid_cap: 300,
        };
        move |c| {
            let r = t.test(&reference, c);
            match r.divergence {
                Some(d) => Check::Counterexample(d.into_iter().collect()),
                None if r.total > 0 && r.equal == r.total => Check::Accept,
                None => Check::Reject,
            }
        }
    }

    struct Unsat;
    impl EquivalenceBackend for Unsat {
        fn check_inequality(&self, _: &Expr, _: &Expr) -> SmtVerdict {
            SmtVerdict::Unsat
        }
    }

    #[test]
    fn recovers_xor_from_pure_io_without_a_seed() {
        let oracle = |env: &BTreeMap<String, u64>| Some(env["x"] ^ env["y"]);
        let s = synthesize(
            W,
            &names(),
            &oracle,
            None,
            &mut tester_check(xor()),
            &LocalSearchConfig::default(),
        )
        .expect("x^y is within reach of the search");
        assert!(InputTester::default().test(&xor(), &s.expr).divergence.is_none());
        assert!(s.expr.node_count_bounded(100) <= 3);
    }

    /// `(x&y)*(x|y) + (x&~y)*(~x&y)` is `x*y`, a nonlinear MBA identity.
    fn mul_identity() -> Expr {
        let not = |e| Expr::unary(UnaryOp::Not, e);
        b(
            BinaryOp::Add,
            b(BinaryOp::Mul, b(BinaryOp::And, v("x"), v("y")), b(BinaryOp::Or, v("x"), v("y"))),
            b(
                BinaryOp::Mul,
                b(BinaryOp::And, v("x"), not(v("y"))),
                b(BinaryOp::And, not(v("x")), v("y")),
            ),
        )
    }

    #[test]
    fn seeded_solver_shrinks_nonlinear_obfuscation_and_is_gated_by_tier() {
        let xy = b(BinaryOp::Mul, v("x"), v("y"));
        // x^y with a hidden nonlinear zero: no linear theorem applies to it.
        let zero = b(BinaryOp::Sub, mul_identity(), xy);
        let obf = b(BinaryOp::Xor, xor(), zero);
        assert!(vmpx_mutation::linear_mba::signature(&obf).is_err());

        let p = Problem::new(obf.clone(), ProblemClass::NonlinearMba, Ctx::default());
        let cand = LocalSearchSolver::default().solve(&p).expect("candidate");
        assert!(cand.expr.node_count_bounded(100) < obf.node_count_bounded(100));
        assert_eq!(cand.by, SolverId::new("local_search"));
        assert!(InputTester::default().test(&obf, &cand.expr).divergence.is_none());

        let j = judge_sampled(&mut PromotionGate::<NoSolver>::without_solver(), &obf, &cand.expr);
        assert_eq!((j.tier, j.proof), (Tier::Observed, None));
        let j = judge_sampled(&mut PromotionGate::with_backend(Unsat), &obf, &cand.expr);
        assert_eq!((j.tier, j.proof), (Tier::Proven, Some(ProofKind::Smt)));
    }

    #[test]
    fn unseeded_search_recovers_x_times_y_from_io() {
        let target = b(BinaryOp::Mul, v("x"), v("y"));
        let oracle = |env: &BTreeMap<String, u64>| Some(env["x"].wrapping_mul(env["y"]));
        let s = synthesize(
            W,
            &names(),
            &oracle,
            None,
            &mut tester_check(target.clone()),
            &LocalSearchConfig::default(),
        )
        .expect("x*y is within reach of the search");
        assert!(InputTester::default().test(&mul_identity(), &s.expr).divergence.is_none());
        assert!(s.expr.node_count_bounded(100) <= 3);
    }

    #[test]
    fn fails_closed_on_unevaluable_and_oversized_problems() {
        let load = Expr::load(W, v("p"));
        let p = Problem::new(load, ProblemClass::NonlinearMba, Ctx::default());
        assert!(LocalSearchSolver::default().solve(&p).is_none());
        let many: Vec<String> = (0..7).map(|i| format!("v{i}")).collect();
        let oracle = |_: &BTreeMap<String, u64>| Some(0);
        let mut ok = |_: &Expr| Check::Accept;
        assert!(synthesize(W, &many, &oracle, None, &mut ok, &LocalSearchConfig::default()).is_none());
    }
}
