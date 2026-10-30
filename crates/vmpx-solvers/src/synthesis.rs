//! Program-synthesis solver (Syntia/Xyntia-style). Treats the input expression
//! as an I/O oracle and enumerates the smallest expression reproducing it,
//! targeting nonlinear MBA (`x*y`) that the algebraic solvers miss. Inductive:
//! the input-tester + SMT gate verify the candidate downstream.
//!
//! Search: bottom-up by node count over {vars, small consts, `~ -`,
//! `+ - * & | ^`, `<< >>` by const}, deduplicated by output vector on the
//! samples (observational equivalence).

use std::collections::{BTreeMap, HashSet};
use std::time::{Duration, Instant};

use crate::Solver;
use vmpx_mutation::{Candidate, Cost, Problem, ProblemClass, SolverId};
use vmpx_vm::{BinaryOp, BitWidth, Expr, ExprKind, UnaryOp};

#[derive(Default)]
pub struct SynthesisSolver;

const MAX_VARS: usize = 2;
const MAX_SIZE: usize = 7;
const MAX_CANDIDATES: usize = 200_000;
const MAX_ATTEMPTS: usize = 3_000_000;
const TIME_BUDGET: Duration = Duration::from_millis(2500);
const INPUT_NODE_LIMIT: usize = 4096;
const CONSTS: [u64; 8] = [0, 1, 2, 3, 4, 8, 16, u64::MAX];
const STRUCTURED: [(u64, u64); 12] = [
    (0, 0),
    (1, 1),
    (1, 2),
    (2, 3),
    (u64::MAX, 1),
    (u64::MAX, u64::MAX),
    (0, u64::MAX),
    (1 << 63, 1),
    (3, 5),
    (7, 0),
    (0xff, 0x0f),
    (0x5555_5555_5555_5555, 0xaaaa_aaaa_aaaa_aaaa),
];
const RANDOM: usize = 12;

const BINOPS: [(BinaryOp, bool); 6] = [
    (BinaryOp::Add, true),
    (BinaryOp::Sub, false),
    (BinaryOp::Mul, true),
    (BinaryOp::And, true),
    (BinaryOp::Or, true),
    (BinaryOp::Xor, true),
];

#[derive(Clone, Copy)]
enum Node {
    Var(usize),
    Const(u64),
    Un(UnaryOp, usize),
    Bin(BinaryOp, usize, usize),
}

fn eval_point(e: &Expr, vals: &BTreeMap<String, Expr>) -> Option<u64> {
    match e.substitute(vals).ok()?.simplify().kind {
        ExprKind::Const { value } => Some(value),
        _ => None,
    }
}

fn samples(n_vars: usize, width: BitWidth) -> Vec<[u64; MAX_VARS]> {
    let mut out: Vec<[u64; MAX_VARS]> = STRUCTURED
        .iter()
        .map(|&(a, b)| [width.truncate(a), width.truncate(b)])
        .collect();
    let mut s = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        width.truncate(s)
    };
    for _ in 0..RANDOM {
        out.push([next(), next()]);
    }
    for p in &mut out {
        p[n_vars.min(MAX_VARS)..].fill(0);
    }
    out
}

fn apply_bin(op: BinaryOp, a: u64, b: u64, w: BitWidth) -> u64 {
    w.truncate(match op {
        BinaryOp::Add => a.wrapping_add(b),
        BinaryOp::Sub => a.wrapping_sub(b),
        BinaryOp::Mul => a.wrapping_mul(b),
        BinaryOp::And => a & b,
        BinaryOp::Or => a | b,
        BinaryOp::Xor => a ^ b,
        BinaryOp::Shl => a << b,
        BinaryOp::Shr => a >> b,
        _ => unreachable!("op not in grammar"),
    })
}

fn build(nodes: &[Node], vars: &[Expr], width: BitWidth, i: usize) -> Option<Expr> {
    Some(match nodes[i] {
        Node::Var(v) => vars[v].clone(),
        Node::Const(c) => Expr::constant(width, c),
        Node::Un(op, a) => Expr::unary(op, build(nodes, vars, width, a)?),
        Node::Bin(op, a, b) => Expr::binary(
            op,
            build(nodes, vars, width, a)?,
            build(nodes, vars, width, b)?,
        )
        .ok()?,
    })
}

fn synthesize(e: &Expr) -> Option<Expr> {
    if e.node_count_bounded(INPUT_NODE_LIMIT) > INPUT_NODE_LIMIT {
        return None;
    }
    let width = e.width;
    let names: Vec<String> = e.dependencies().into_iter().collect();
    if names.len() > MAX_VARS {
        return None;
    }
    let smp = samples(names.len(), width);
    let n = smp.len();
    let mut target = Vec::with_capacity(n);
    for p in &smp {
        let vals = names
            .iter()
            .enumerate()
            .map(|(j, nm)| (nm.clone(), Expr::constant(width, p[j])))
            .collect();
        target.push(eval_point(e, &vals)?);
    }
    let vars: Vec<Expr> = names
        .iter()
        .map(|nm| Expr::register(width, nm.clone()))
        .collect();

    let start = Instant::now();
    let mut nodes: Vec<Node> = Vec::new();
    let mut vals: Vec<u64> = Vec::new();
    let mut by_size: Vec<Vec<usize>> = vec![Vec::new(); MAX_SIZE + 1];
    let mut seen: HashSet<Vec<u64>> = HashSet::new();
    let mut attempts = 0_usize;
    let mut scratch = vec![0_u64; n];

    // Budget check, match check (returns the built expr), else retain if new.
    macro_rules! offer {
        ($node:expr, $size:expr) => {{
            attempts += 1;
            if attempts > MAX_ATTEMPTS
                || nodes.len() >= MAX_CANDIDATES
                || (attempts % 4096 == 0 && start.elapsed() > TIME_BUDGET)
            {
                return None;
            }
            if scratch == target {
                nodes.push($node);
                return build(&nodes, &vars, width, nodes.len() - 1);
            }
            if !seen.contains(&scratch) {
                seen.insert(scratch.clone());
                vals.extend_from_slice(&scratch);
                by_size[$size].push(nodes.len());
                nodes.push($node);
            }
        }};
    }

    for v in 0..vars.len() {
        for (k, s) in scratch.iter_mut().enumerate() {
            *s = smp[k][v];
        }
        offer!(Node::Var(v), 1);
    }
    let mut shift_consts: Vec<(usize, u64)> = Vec::new();
    for c in CONSTS {
        let c = width.truncate(c);
        scratch.fill(c);
        if (1..u64::from(width.bits())).contains(&c) {
            // Keep a const leaf for shift amounts even if its vector was seen.
            shift_consts.push((nodes.len(), c));
        }
        offer!(Node::Const(c), 1);
    }
    shift_consts.retain(|&(i, c)| matches!(nodes.get(i), Some(Node::Const(v)) if *v == c));

    for size in 2..=MAX_SIZE {
        for op in [UnaryOp::Not, UnaryOp::Neg] {
            for ci in 0..by_size[size - 1].len() {
                let a = by_size[size - 1][ci];
                for k in 0..n {
                    let x = vals[a * n + k];
                    scratch[k] = width.truncate(match op {
                        UnaryOp::Not => !x,
                        UnaryOp::Neg => x.wrapping_neg(),
                    });
                }
                offer!(Node::Un(op, a), size);
            }
        }
        // Operand sizes sa + sb = size - 1; sa <= sb, both orders if non-commutative.
        for sa in 1..=(size - 1) / 2 {
            let sb = size - 1 - sa;
            for (op, comm) in BINOPS {
                for ia in 0..by_size[sa].len() {
                    for ib in 0..by_size[sb].len() {
                        if sa == sb && comm && ib < ia {
                            continue;
                        }
                        let (a, b) = (by_size[sa][ia], by_size[sb][ib]);
                        for o in 0..if comm { 1 } else { 2 } {
                            let (l, r) = if o == 0 { (a, b) } else { (b, a) };
                            for k in 0..n {
                                scratch[k] = apply_bin(op, vals[l * n + k], vals[r * n + k], width);
                            }
                            offer!(Node::Bin(op, l, r), size);
                        }
                    }
                }
            }
        }
        // Shifts: any left operand, constant leaf amount.
        for op in [BinaryOp::Shl, BinaryOp::Shr] {
            for ia in 0..by_size[size - 2].len() {
                let a = by_size[size - 2][ia];
                for &(ci, amt) in &shift_consts {
                    for k in 0..n {
                        scratch[k] = apply_bin(op, vals[a * n + k], amt, width);
                    }
                    offer!(Node::Bin(op, a, ci), size);
                }
            }
        }
    }
    None
}

impl Solver for SynthesisSolver {
    fn id(&self) -> SolverId {
        SolverId::new("synthesis")
    }
    fn applies_to(&self, class: ProblemClass) -> bool {
        matches!(class, ProblemClass::NonlinearMba | ProblemClass::LinearMba)
    }
    fn solve(&self, p: &Problem) -> Option<Candidate> {
        let start = Instant::now();
        let expr = synthesize(&p.expr)?;
        if expr.node_count_bounded(INPUT_NODE_LIMIT) >= p.expr.node_count_bounded(INPUT_NODE_LIMIT)
        {
            return None;
        }
        Some(Candidate {
            expr,
            by: self.id(),
            cost: Cost {
                time_ns: u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX),
                steps: 1,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vmpx_mutation::Ctx;

    const W: BitWidth = BitWidth::W64;

    fn r(n: &str) -> Expr {
        Expr::register(W, n)
    }
    fn k(v: u64) -> Expr {
        Expr::constant(W, v)
    }
    fn b(op: BinaryOp, l: Expr, rr: Expr) -> Expr {
        Expr::binary(op, l, rr).unwrap()
    }
    fn not(e: Expr) -> Expr {
        Expr::unary(UnaryOp::Not, e)
    }
    fn eval(e: &Expr, x: u64, y: u64) -> u64 {
        let vals = BTreeMap::from([("x".to_string(), k(x)), ("y".to_string(), k(y))]);
        eval_point(e, &vals).unwrap()
    }
    fn solve(e: Expr, class: ProblemClass) -> Option<Candidate> {
        SynthesisSolver.solve(&Problem::new(e, class, Ctx::default()))
    }
    const CHECK: [(u64, u64); 6] = [
        (0, 0),
        (1, 2),
        (0xdead_beef, 0x1234_5678_9abc_def0),
        (u64::MAX, 1),
        (u64::MAX, u64::MAX),
        (0x8000_0000_0000_0000, 7),
    ];

    #[test]
    fn nonlinear_obfuscation_of_mul() {
        let (x, y) = (r("x"), r("y"));
        let e = b(
            BinaryOp::Add,
            b(
                BinaryOp::Mul,
                b(BinaryOp::And, x.clone(), y.clone()),
                b(BinaryOp::Or, x.clone(), y.clone()),
            ),
            b(
                BinaryOp::Mul,
                b(BinaryOp::And, x.clone(), not(y.clone())),
                b(BinaryOp::And, not(x), y),
            ),
        );
        let c = solve(e.clone(), ProblemClass::NonlinearMba).expect("synthesizes");
        assert_eq!(c.expr.node_count_bounded(100), 3, "{:?}", c.expr);
        for (x, y) in CHECK {
            assert_eq!(eval(&c.expr, x, y), eval(&e, x, y));
            assert_eq!(eval(&c.expr, x, y), x.wrapping_mul(y));
        }
    }

    #[test]
    fn linear_obfuscation_of_add() {
        let e = b(
            BinaryOp::Add,
            b(BinaryOp::Xor, r("x"), r("y")),
            b(BinaryOp::Mul, k(2), b(BinaryOp::And, r("x"), r("y"))),
        );
        let c = solve(e, ProblemClass::LinearMba).expect("synthesizes");
        assert_eq!(c.expr.node_count_bounded(100), 3);
        for (x, y) in CHECK {
            assert_eq!(eval(&c.expr, x, y), x.wrapping_add(y));
        }
    }

    #[test]
    fn bounded_returns_none() {
        let e = b(
            BinaryOp::Add,
            b(BinaryOp::Mul, r("x"), r("y")),
            b(BinaryOp::Xor, r("z"), k(1)),
        );
        assert!(solve(e, ProblemClass::NonlinearMba).is_none());
        // Beyond the size bound: a long hash-like mix of two vars.
        let mut e = r("x");
        for c in [0x9e37_79b9_7f4a_7c15_u64, 0xbf58_476d_1ce4_e5b9, 0x94d0_49bb_1331_11eb] {
            e = b(BinaryOp::Xor, b(BinaryOp::Mul, e, k(c)), r("y"));
            e = b(BinaryOp::Add, e, b(BinaryOp::Shr, r("x"), k(7)));
        }
        let start = Instant::now();
        assert!(solve(e, ProblemClass::NonlinearMba).is_none());
        assert!(start.elapsed() < Duration::from_secs(10));
    }

    #[test]
    fn applies_to_mba_classes() {
        assert!(SynthesisSolver.applies_to(ProblemClass::NonlinearMba));
        assert!(SynthesisSolver.applies_to(ProblemClass::LinearMba));
        assert!(!SynthesisSolver.applies_to(ProblemClass::OpaquePredicate));
    }
}
