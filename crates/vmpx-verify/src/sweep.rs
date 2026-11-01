//! SAT-sweeping (cut-point) equivalence for two expression DAGs over the same
//! inputs: the standard combinational-equivalence strategy. Random simulation
//! proposes pairs of sub-terms that look equal; each pair is then *proven*
//! equal by the inner backend with all earlier merged classes abstracted to
//! shared free variables, so every query stays one-operation deep. Merged
//! classes are what make a long unrolled computation tractable where a single
//! monolithic query is not.
//!
//! Sound for UNSAT only: abstracting a sub-term to a free variable
//! over-approximates, so an inner UNSAT on the abstraction holds for the real
//! terms. A satisfiable abstraction proves nothing (the free variables are
//! unconstrained), so this backend never reports `Sat`; undecided is `Unknown`.
//! `Load`/`Intrinsic` terms are uninterpreted functions of their operands.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use vmpx_vm::{BitWidth, Expr, ExprKind};

use crate::equiv::{Budget, SmtBackend, SolveOutcome};

const TRIALS: usize = 48;
const CANDIDATES_PER_NODE: usize = 4;
const FULL_CONE_MS: u32 = 8_000;
const MAX_CONE_NODES: usize = 450;
const SWEEP_BUDGET_FACTOR: u64 = 6;

pub struct SweepBackend {
    inner: Box<dyn SmtBackend>,
}

impl SweepBackend {
    #[must_use]
    pub fn new(inner: Box<dyn SmtBackend>) -> Self {
        Self { inner }
    }
}

impl SmtBackend for SweepBackend {
    fn name(&self) -> String {
        format!("sweep+{}", self.inner.name())
    }

    fn check_distinct(&self, a: &Expr, b: &Expr, budget: &Budget) -> SolveOutcome {
        sweep(self.inner.as_ref(), a, b, budget)
    }
}

struct Node {
    expr: Expr,
    kids: Vec<usize>,
}

fn kids_of(e: &Expr) -> Vec<&Expr> {
    match &e.kind {
        ExprKind::Const { .. } | ExprKind::Register { .. } => Vec::new(),
        ExprKind::Unary { value, .. }
        | ExprKind::Extract { value, .. }
        | ExprKind::Resize { value, .. }
        | ExprKind::Load { address: value } => vec![value],
        ExprKind::Binary { left, right, .. }
        | ExprKind::Concat {
            high: left,
            low: right,
        } => vec![left, right],
        ExprKind::Intrinsic { arguments, .. } => arguments.iter().collect(),
    }
}

/// Distinct nodes of both terms, children before parents.
fn collect(a: &Expr, b: &Expr) -> (Vec<Node>, usize, usize) {
    let mut index: HashMap<Expr, usize> = HashMap::new();
    let mut nodes: Vec<Node> = Vec::new();
    let mut roots = [0usize; 2];
    for (slot, root) in [a, b].into_iter().enumerate() {
        let mut stack: Vec<(Expr, bool)> = vec![(root.clone(), false)];
        while let Some((e, expanded)) = stack.pop() {
            if index.contains_key(&e) {
                continue;
            }
            if expanded {
                let kids = kids_of(&e).into_iter().map(|k| index[k]).collect();
                index.insert(e.clone(), nodes.len());
                nodes.push(Node { expr: e, kids });
            } else {
                stack.push((e.clone(), true));
                for k in kids_of(&e) {
                    if !index.contains_key(k) {
                        stack.push((k.clone(), false));
                    }
                }
            }
        }
        roots[slot] = index[root];
    }
    (nodes, roots[0], roots[1])
}

fn mix(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn text_hash(s: &str) -> u64 {
    s.bytes().fold(0xCBF2_9CE4_8422_2325, |h, c| (h ^ u64::from(c)).wrapping_mul(0x100_0000_01B3))
}

/// Value of every node under one random input assignment, or `None` if some
/// node does not fold (then simulation is skipped and nothing is merged).
fn simulate(nodes: &[Node], trial: u64) -> Option<Vec<u64>> {
    let mut values: Vec<u64> = Vec::with_capacity(nodes.len());
    for node in nodes {
        let e = &node.expr;
        let kid = |i: usize| values[node.kids[i]];
        let constant = |w: BitWidth, v: u64| Expr::constant(w, v);
        let value = match &e.kind {
            ExprKind::Const { value } => e.width.truncate(*value),
            ExprKind::Register { name } => e.width.truncate(mix(text_hash(name) ^ mix(trial))),
            ExprKind::Load { .. } => {
                e.width.truncate(mix(0x10AD ^ mix(kid(0)) ^ mix(trial)))
            }
            ExprKind::Intrinsic { name, .. } => {
                let mut h = text_hash(name) ^ mix(trial);
                for i in 0..node.kids.len() {
                    h = mix(h ^ kid(i));
                }
                e.width.truncate(h)
            }
            ExprKind::Unary { op, value } => {
                let rebuilt = Expr::unary(*op, constant(value.width, kid(0)));
                let ExprKind::Const { value } = rebuilt.simplify().kind else { return None };
                value
            }
            ExprKind::Binary { op, left, .. } => {
                let rebuilt =
                    Expr::binary(*op, constant(left.width, kid(0)), constant(left.width, kid(1))).ok()?;
                let ExprKind::Const { value } = rebuilt.simplify().kind else { return None };
                value
            }
            ExprKind::Extract { value, lsb } => {
                let rebuilt = Expr::extract(e.width, *lsb, constant(value.width, kid(0))).ok()?;
                let ExprKind::Const { value } = rebuilt.simplify().kind else { return None };
                value
            }
            ExprKind::Concat { high, low } => {
                let rebuilt =
                    Expr::concat(constant(high.width, kid(0)), constant(low.width, kid(1))).ok()?;
                let ExprKind::Const { value } = rebuilt.simplify().kind else { return None };
                value
            }
            ExprKind::Resize { op, value } => {
                let rebuilt = Expr::resize(*op, e.width, constant(value.width, kid(0))).ok()?;
                let ExprKind::Const { value } = rebuilt.simplify().kind else { return None };
                value
            }
        };
        values.push(value);
    }
    Some(values)
}

/// Structural key of a node over its children's classes.
fn key_of(e: &Expr, classes: &[usize]) -> String {
    let head = match &e.kind {
        ExprKind::Const { value } => format!("c{value}"),
        ExprKind::Register { name } => format!("r{name}"),
        ExprKind::Unary { op, .. } => format!("u{op:?}"),
        ExprKind::Binary { op, .. } => format!("b{op:?}"),
        ExprKind::Load { .. } => "load".to_owned(),
        ExprKind::Intrinsic { name, .. } => format!("i{name}"),
        ExprKind::Extract { lsb, .. } => format!("x{lsb}"),
        ExprKind::Concat { .. } => "cat".to_owned(),
        ExprKind::Resize { op, .. } => format!("z{op:?}"),
    };
    format!("{head}/{}/{classes:?}", e.width.bits())
}

fn local_form(e: &Expr, vars: &[Expr]) -> Option<Expr> {
    Some(match &e.kind {
        ExprKind::Const { .. } | ExprKind::Register { .. } => e.clone(),
        ExprKind::Unary { op, .. } => Expr::unary(*op, vars[0].clone()),
        ExprKind::Binary { op, .. } => Expr::binary(*op, vars[0].clone(), vars[1].clone()).ok()?,
        ExprKind::Extract { lsb, .. } => Expr::extract(e.width, *lsb, vars[0].clone()).ok()?,
        ExprKind::Concat { .. } => Expr::concat(vars[0].clone(), vars[1].clone()).ok()?,
        ExprKind::Resize { op, .. } => Expr::resize(*op, e.width, vars[0].clone()).ok()?,
        ExprKind::Load { .. } => Expr::load(e.width, vars[0].clone()),
        ExprKind::Intrinsic { name, .. } => Expr::intrinsic(e.width, name.clone(), vars.to_vec()),
    })
}

/// The term with every class opened `depth` levels (ignoring shared
/// variables there), below which classes stand for their current term.
fn opened(
    class: usize,
    depth: u32,
    vars: &[Option<Expr>],
    shapes: &[(usize, Vec<usize>)],
    nodes: &[Node],
    memo: &mut HashMap<(usize, u32), Expr>,
) -> Option<Expr> {
    let (n, kids) = &shapes[class];
    if matches!(nodes[*n].expr.kind, ExprKind::Const { .. } | ExprKind::Register { .. }) {
        return vars[class].clone();
    }
    if depth == 0 {
        let mut plain = HashMap::new();
        return current(class, vars, shapes, nodes, &mut plain);
    }
    if let Some(e) = memo.get(&(class, depth)) {
        return Some(e.clone());
    }
    let mut resolved = Vec::with_capacity(kids.len());
    for k in kids {
        resolved.push(opened(*k, depth - 1, vars, shapes, nodes, memo)?);
    }
    let e = local_form(&nodes[*n].expr, &resolved)?;
    memo.insert((class, depth), e.clone());
    Some(e)
}

fn opened_form(
    kids: &[usize],
    node: &Expr,
    depth: u32,
    vars: &[Option<Expr>],
    shapes: &[(usize, Vec<usize>)],
    nodes: &[Node],
    memo: &mut HashMap<(usize, u32), Expr>,
) -> Option<Expr> {
    let mut resolved = Vec::with_capacity(kids.len());
    for k in kids {
        resolved.push(opened(*k, depth, vars, shapes, nodes, memo)?);
    }
    local_form(node, &resolved)
}

/// The term a class currently stands for: its shared variable once it has been
/// proven equal to a second term, else its operation over its children's
/// current terms.
fn current(
    class: usize,
    vars: &[Option<Expr>],
    shapes: &[(usize, Vec<usize>)],
    nodes: &[Node],
    memo: &mut HashMap<usize, Expr>,
) -> Option<Expr> {
    if let Some(v) = &vars[class] {
        return Some(v.clone());
    }
    if let Some(e) = memo.get(&class) {
        return Some(e.clone());
    }
    let (node, kids) = &shapes[class];
    let mut resolved = Vec::with_capacity(kids.len());
    for k in kids {
        resolved.push(current(*k, vars, shapes, nodes, memo)?);
    }
    let e = local_form(&nodes[*node].expr, &resolved)?;
    memo.insert(class, e.clone());
    Some(e)
}

fn sweep(inner: &dyn SmtBackend, a: &Expr, b: &Expr, budget: &Budget) -> SolveOutcome {
    // One sweep makes many small queries; its overall allowance is a multiple of
    // the per-cell budget.
    let deadline = Instant::now()
        + Duration::from_millis(u64::from(budget.timeout_ms) * SWEEP_BUDGET_FACTOR);
    let (nodes, root_a, root_b) = collect(a, b);
    if std::env::var_os("VMPX_SWEEP_TRACE").is_some() {
        eprintln!("[sweep] start: {} nodes, budget {} ms", nodes.len(), budget.timeout_ms);
    }
    let mut sims: Vec<Vec<u64>> = Vec::new();
    for trial in 0..TRIALS as u64 {
        match simulate(&nodes, trial) {
            Some(v) => sims.push(v),
            None => return SolveOutcome::Unknown("sweep: a term does not simulate".to_owned()),
        }
    }
    // class id -> the term parents build on (the expansion until the class is
    // proven equal to a second term, then a shared free variable) and the
    // class's own local form.
    let mut class_var: Vec<Option<Expr>> = Vec::new();
    let mut class_shape: Vec<(usize, Vec<usize>)> = Vec::new();
    let mut strash: HashMap<String, usize> = HashMap::new();
    let mut by_signature: HashMap<(u32, Vec<u64>), Vec<usize>> = HashMap::new();
    let mut node_class: Vec<usize> = Vec::with_capacity(nodes.len());

    for (i, node) in nodes.iter().enumerate() {
        let kid_classes: Vec<usize> = node.kids.iter().map(|k| node_class[*k]).collect();
        let key = key_of(&node.expr, &kid_classes);
        if let Some(c) = strash.get(&key) {
            node_class.push(*c);
            continue;
        }
        let mut memo: HashMap<usize, Expr> = HashMap::new();
        let mut vars: Vec<Expr> = Vec::new();
        for c in &kid_classes {
            let Some(v) = current(*c, &class_var, &class_shape, &nodes, &mut memo) else {
                return SolveOutcome::Unknown("sweep: cannot rebuild a term".to_owned());
            };
            vars.push(v);
        }
        let Some(local) = local_form(&node.expr, &vars) else {
            return SolveOutcome::Unknown("sweep: cannot rebuild a term".to_owned());
        };
        let leaf = matches!(node.expr.kind, ExprKind::Const { .. } | ExprKind::Register { .. });
        let signature: Vec<u64> = sims.iter().map(|s| s[i]).collect();
        let sig_key = (node.expr.width.bits(), signature);
        let mut merged = None;
        if !leaf {
            let candidates = by_signature.get(&sig_key).cloned().unwrap_or_default();
            for c in candidates.into_iter().take(CANDIDATES_PER_NODE) {
                let now = Instant::now();
                if now >= deadline {
                    if std::env::var_os("VMPX_SWEEP_TRACE").is_some() {
                        eprintln!("[sweep] deadline at node {i}");
                    }
                    return SolveOutcome::Timeout;
                }
                let remaining = u32::try_from((deadline - now).as_millis()).unwrap_or(u32::MAX);
                let query = Budget {
                    exhaustive_bits: budget.exhaustive_bits,
                    timeout_ms: remaining.clamp(1, budget.timeout_ms),
                };
                let (shape_node, shape_kids) = &class_shape[c];
                let mut other_kids = Vec::new();
                for k in shape_kids {
                    match current(*k, &class_var, &class_shape, &nodes, &mut memo) {
                        Some(v) => other_kids.push(v),
                        None => return SolveOutcome::Unknown("sweep: cannot rebuild a term".to_owned()),
                    }
                }
                let Some(other_form) = local_form(&nodes[*shape_node].expr, &other_kids) else {
                    continue;
                };
                match inner.check_distinct(&local, &other_form, &query) {
                    SolveOutcome::Unsat => {
                        merged = Some(c);
                        break;
                    }
                    SolveOutcome::Timeout => {
                        if std::env::var_os("VMPX_SWEEP_TRACE").is_some() {
                            eprintln!("[sweep] inner timeout at node {i}");
                        }
                        return SolveOutcome::Timeout;
                    }
                    other => {
                        // The abstraction may hide a relation between a shared
                        // variable and the inputs; retry on the unabstracted
                        // cones before giving up on this pair.
                        let mut full = HashMap::new();
                        let mut retried = SolveOutcome::Unknown("not retried".to_owned());
                        for depth in [1_u32, 2, 4, 1_000] {
                            let a_open = opened_form(&kid_classes, &node.expr, depth, &class_var, &class_shape, &nodes, &mut full);
                            let b_open = opened_form(shape_kids, &nodes[*shape_node].expr, depth, &class_var, &class_shape, &nodes, &mut full);
                            let (Some(x), Some(y)) = (a_open, b_open) else { break };
                            // A cone too big for a one-shot query would only
                            // exhaust the solver; keep sweeping instead.
                            if x.node_count_bounded(MAX_CONE_NODES) > MAX_CONE_NODES
                                || y.node_count_bounded(MAX_CONE_NODES) > MAX_CONE_NODES
                            {
                                break;
                            }
                            let cap = Budget {
                                exhaustive_bits: budget.exhaustive_bits,
                                timeout_ms: query.timeout_ms.min(FULL_CONE_MS),
                            };
                            retried = inner.check_distinct(&x, &y, &cap);
                            if retried == SolveOutcome::Unsat {
                                break;
                            }
                        }
                        if std::env::var_os("VMPX_SWEEP_TRACE").is_some() {
                            eprintln!("[sweep] node {i} candidate {c}: abstract {other:?}; opened {retried:?}");
                        }
                        if retried == SolveOutcome::Unsat {
                            merged = Some(c);
                            break;
                        }
                    }
                }
            }
        }
        let class = if let Some(c) = merged {
            if class_var[c].is_none() {
                class_var[c] = Some(Expr::register(node.expr.width, format!("cp!{c}")));
            }
            c
        } else {
            let c = class_var.len();
            class_var.push(leaf.then(|| node.expr.clone()));
            class_shape.push((i, kid_classes.clone()));
            by_signature.entry(sig_key).or_default().push(c);
            c
        };
        strash.insert(key, class);
        node_class.push(class);
    }
    if std::env::var_os("VMPX_SWEEP_TRACE").is_some() {
        eprintln!(
            "[sweep] {} nodes, {} classes, {} abstracted, roots {} / {}",
            nodes.len(),
            class_var.len(),
            class_var.iter().filter(|x| x.is_some()).count(),
            node_class[root_a],
            node_class[root_b]
        );
    }
    if node_class[root_a] == node_class[root_b] {
        SolveOutcome::Unsat
    } else {
        SolveOutcome::Unknown("sweep: the two roots did not merge".to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equiv::NativeBackend;
    use vmpx_vm::BinaryOp;

    fn reg(n: &str) -> Expr {
        Expr::register(BitWidth::W8, n)
    }
    fn bin(op: BinaryOp, l: Expr, r: Expr) -> Expr {
        Expr::binary(op, l, r).unwrap()
    }

    #[test]
    fn merges_equal_sub_terms_and_proves_the_roots() {
        // x^y spelled two ways, chained twice; 8-bit so the native backend decides each step.
        let xor = |x: &Expr, y: &Expr| bin(BinaryOp::Xor, x.clone(), y.clone());
        let alt = |x: &Expr, y: &Expr| {
            let or = bin(BinaryOp::Or, x.clone(), y.clone());
            let and = bin(BinaryOp::And, x.clone(), y.clone());
            bin(BinaryOp::And, or, Expr::unary(vmpx_vm::UnaryOp::Not, and))
        };
        let (a, b, c) = (reg("a"), reg("b"), reg("c"));
        let left = xor(&xor(&a, &b), &c);
        let right = alt(&alt(&a, &b), &c);
        let sweeper = SweepBackend::new(Box::new(NativeBackend));
        assert_eq!(sweeper.check_distinct(&left, &right, &Budget::default()), SolveOutcome::Unsat);
    }

    #[test]
    fn never_proves_unequal_terms() {
        let (a, b) = (reg("a"), reg("b"));
        let left = bin(BinaryOp::Add, a.clone(), b.clone());
        let right = bin(BinaryOp::Add, a, bin(BinaryOp::Add, b, Expr::constant(BitWidth::W8, 1)));
        let sweeper = SweepBackend::new(Box::new(NativeBackend));
        assert!(!matches!(
            sweeper.check_distinct(&left, &right, &Budget::default()),
            SolveOutcome::Unsat | SolveOutcome::Sat(_)
        ));
    }
}
