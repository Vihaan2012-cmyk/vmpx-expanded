//! Differential input-testing harness — the Scorer (the tournament workhorse).
//!
//! Inductive only: agreement over the tested inputs gates eligibility, it never
//! proves equivalence (that is the Gate's job).

use crate::Scorer;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use vmpx_mutation::{Candidate, Score};
use vmpx_vm::{BitWidth, Expr, ExprKind};

const SEED: u64 = 0x9E37_79B9_7F4A_7C15;
const NODE_LIMIT: usize = 1 << 24;

/// Scores a candidate by agreement with the original over many concrete inputs.
pub struct InputTester {
    /// Random assignments on top of the structured edge grid.
    pub random_inputs: usize,
    /// Max edge-grid assignments (full cross product when it fits).
    pub grid_cap: usize,
}

impl Default for InputTester {
    fn default() -> Self {
        Self {
            random_inputs: 1000,
            grid_cap: 2000,
        }
    }
}

/// Outcome of a differential run.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Report {
    pub total: usize,
    pub equal: usize,
    /// Inputs where either side failed to fold to a constant; never agreement.
    pub inconclusive: usize,
    /// First assignment where both folded to differing constants.
    pub divergence: Option<Vec<(String, u64)>>,
}

impl InputTester {
    /// Runs both expressions over the input set.
    #[must_use]
    pub fn test(&self, original: &Expr, candidate: &Expr) -> Report {
        let mut vars = BTreeMap::new();
        collect_registers(original, &mut vars);
        collect_registers(candidate, &mut vars);
        let mut report = Report::default();
        for assignment in self.inputs(&vars) {
            report.total += 1;
            match (fold(original, &assignment), fold(candidate, &assignment)) {
                (Some(a), Some(b)) if a == b => report.equal += 1,
                (Some(_), Some(_)) => {
                    if report.divergence.is_none() {
                        report.divergence =
                            Some(assignment.iter().map(|(k, v)| (k.clone(), *v)).collect());
                    }
                }
                _ => report.inconclusive += 1,
            }
        }
        report
    }

    #[allow(clippy::cast_possible_truncation)]
    fn inputs(&self, vars: &BTreeMap<String, BitWidth>) -> Vec<BTreeMap<String, u64>> {
        let mut rng = SEED;
        let edges: Vec<(&String, Vec<u64>)> =
            vars.iter().map(|(n, w)| (n, edge_values(*w))).collect();
        let mut out: Vec<BTreeMap<String, u64>> = Vec::new();

        let product = edges
            .iter()
            .try_fold(1usize, |acc, (_, e)| acc.checked_mul(e.len()));
        if product.is_some_and(|p| p <= self.grid_cap) {
            let mut idx = vec![0usize; edges.len()];
            loop {
                out.push(
                    edges
                        .iter()
                        .zip(&idx)
                        .map(|((n, e), i)| ((*n).clone(), e[*i]))
                        .collect(),
                );
                let mut k = 0;
                while k < idx.len() {
                    idx[k] += 1;
                    if idx[k] < edges[k].1.len() {
                        break;
                    }
                    idx[k] = 0;
                    k += 1;
                }
                if k == idx.len() {
                    break;
                }
            }
        } else {
            for _ in 0..self.grid_cap {
                out.push(
                    edges
                        .iter()
                        .map(|(n, e)| ((*n).clone(), e[next(&mut rng) as usize % e.len()]))
                        .collect(),
                );
            }
            // One variable sweeps its edges while the others are random.
            for (n, e) in &edges {
                for v in e {
                    let mut a: BTreeMap<String, u64> = vars
                        .keys()
                        .map(|k| (k.clone(), next(&mut rng)))
                        .collect();
                    a.insert((*n).clone(), *v);
                    out.push(a);
                }
            }
        }
        for _ in 0..self.random_inputs {
            out.push(vars.keys().map(|k| (k.clone(), next(&mut rng))).collect());
        }
        out
    }
}

fn next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// 0, all-ones, and 2^k, 2^k +/- 1 for every bit (covers 1, sign bit, sign +/- 1).
fn edge_values(width: BitWidth) -> Vec<u64> {
    let mut set = BTreeSet::new();
    set.insert(0);
    set.insert(width.truncate(u64::MAX));
    for k in 0..width.bits() {
        let p = 1u64 << k;
        set.insert(width.truncate(p));
        set.insert(width.truncate(p.wrapping_add(1)));
        set.insert(width.truncate(p.wrapping_sub(1)));
    }
    set.into_iter().collect()
}

fn collect_registers(expr: &Expr, out: &mut BTreeMap<String, BitWidth>) {
    let mut seen: HashSet<*const Expr> = HashSet::new();
    let mut stack = vec![expr];
    while let Some(e) = stack.pop() {
        if !seen.insert(std::ptr::from_ref(e)) {
            continue;
        }
        match &e.kind {
            ExprKind::Const { .. } => {}
            ExprKind::Register { name } => {
                let w = out.entry(name.clone()).or_insert(e.width);
                if e.width.bits() > w.bits() {
                    *w = e.width;
                }
            }
            ExprKind::Unary { value, .. }
            | ExprKind::Load { address: value }
            | ExprKind::Extract { value, .. }
            | ExprKind::Resize { value, .. } => stack.push(value),
            ExprKind::Binary { left, right, .. }
            | ExprKind::Concat {
                high: left,
                low: right,
            } => {
                stack.push(left);
                stack.push(right);
            }
            ExprKind::Intrinsic { arguments, .. } => stack.extend(arguments),
        }
    }
}

/// Substitutes constants and folds; `None` when the result is not a constant.
fn fold(expr: &Expr, assignment: &BTreeMap<String, u64>) -> Option<(BitWidth, u64)> {
    let mut widths = BTreeMap::new();
    collect_registers(expr, &mut widths);
    let replacements: BTreeMap<String, Expr> = widths
        .iter()
        .map(|(n, w)| (n.clone(), Expr::constant(*w, assignment[n])))
        .collect();
    let folded = expr.substitute(&replacements).ok()?.simplify();
    match folded.kind {
        ExprKind::Const { value } => Some((folded.width, value)),
        _ => None,
    }
}

impl Scorer for InputTester {
    fn score(&self, original: &Expr, cand: &Candidate) -> Score {
        let r = self.test(original, &cand.expr);
        #[allow(clippy::cast_precision_loss)]
        let agreement = if r.total == 0 {
            0.0
        } else {
            r.equal as f64 / r.total as f64
        };
        Score {
            agreement,
            simplicity: cand.expr.node_count_bounded(NODE_LIMIT) as u64,
            cost: cand.cost,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vmpx_mutation::{Cost, SolverId};
    use vmpx_vm::BinaryOp::{Add, And, Mul, Or, Xor};
    use vmpx_vm::UnaryOp::Not;

    const W: BitWidth = BitWidth::W64;

    fn reg(n: &str) -> Expr {
        Expr::register(W, n)
    }
    fn bin(op: vmpx_vm::BinaryOp, l: Expr, r: Expr) -> Expr {
        Expr::binary(op, l, r).unwrap()
    }
    fn cand(expr: Expr) -> Candidate {
        Candidate {
            expr,
            by: SolverId::new("t"),
            cost: Cost::default(),
        }
    }

    #[test]
    fn identical_agrees() {
        let e = bin(Add, reg("a"), reg("b"));
        let s = InputTester::default().score(&e, &cand(e.clone()));
        assert!((s.agreement - 1.0).abs() < f64::EPSILON);
        assert!(s.is_eligible());
    }

    #[test]
    fn equivalent_rewrites_agree() {
        let t = InputTester::default();
        let a = reg("a");
        let two = Expr::constant(W, 2);
        let s = t.score(&bin(Add, a.clone(), a.clone()), &cand(bin(Mul, a, two)));
        assert!(s.is_eligible());

        let (a, b) = (reg("a"), reg("b"));
        let xor = bin(Xor, a.clone(), b.clone());
        let mba = bin(
            And,
            bin(Or, a.clone(), b.clone()),
            Expr::unary(Not, bin(And, a, b)),
        );
        let r = t.test(&xor, &mba);
        assert_eq!(r.equal, r.total);
        assert!(r.total > 1000);
        assert!(t.score(&xor, &cand(mba)).is_eligible());
    }

    #[test]
    fn non_equivalent_reports_divergence() {
        let (a, b) = (reg("a"), reg("b"));
        let t = InputTester::default();
        let sum = bin(Add, a.clone(), b.clone());
        let or = bin(Or, a, b);
        let r = t.test(&sum, &or);
        assert!(r.divergence.is_some());
        assert_eq!(r.inconclusive, 0);
        let s = t.score(&sum, &cand(or));
        assert!(s.agreement < 1.0 && !s.is_eligible());
    }

    #[test]
    fn unfoldable_input_is_inconclusive_not_agreement() {
        let load = Expr::load(W, reg("p"));
        let t = InputTester::default();
        let r = t.test(&load, &load);
        assert_eq!(r.equal, 0);
        assert_eq!(r.inconclusive, r.total);
        assert!(!t.score(&load, &cand(load.clone())).is_eligible());
    }

    #[test]
    fn simplicity_is_node_count() {
        let t = InputTester::default();
        let a = reg("a");
        let small = t.score(&a, &cand(a.clone()));
        let bigger = t.score(
            &a,
            &cand(bin(Mul, bin(Add, a.clone(), reg("b")), bin(Xor, a.clone(), reg("b")))),
        );
        assert_eq!(small.simplicity, 1);
        // DAG count: shared a and b are counted once (a, b, +, ^, *).
        assert_eq!(bigger.simplicity, 5);
    }
}
