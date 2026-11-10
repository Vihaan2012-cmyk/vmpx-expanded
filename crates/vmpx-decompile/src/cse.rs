//! Common-subexpression extraction into named locals.
//!
//! The IR is a hash-consed DAG, so a value computed once and used ten times is
//! ONE node with ten parents. Rendering walks it as a tree, which prints that
//! value ten times and turns a readable function into an unreadable wall: on
//! systems.xpl a single statement repeated `*(u8 *)(~(~a2 + 1))` ten times and
//! an entire `rol32(...)` subtree verbatim.
//!
//! This pass recovers the sharing the renderer loses. It names every subtree
//! that is referenced more than once and is big enough to be worth a line, so
//! the body reads as a sequence of small definitions instead of one expression.
//!
//! It is purely syntactic: the same DAG, printed with its sharing made visible.
//! No expression is rewritten, so nothing here can change what the code means.

use std::collections::{HashMap, HashSet};

use vmpx_vm::{BinaryOp, Expr, ExprKind};

use crate::ir::{Cond, Store};
use crate::opt::Stmt;

/// A branch operand at least this big is named even if read once.
const COND_MIN_NODES: usize = 12;

/// Subtrees worth naming, and the order to define them in.
#[derive(Default)]
pub struct Temps {
    /// Expression -> local name.
    pub map: HashMap<Expr, String>,
    /// `(name, definition)`, dependencies first: a definition may reference any
    /// name defined before it, and never one defined after.
    pub order: Vec<(String, Expr)>,
}

impl Temps {
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }
}

/// Children of a node, in render order.
fn children(expr: &Expr) -> Vec<&Expr> {
    match &expr.kind {
        ExprKind::Const { .. } | ExprKind::Register { .. } => Vec::new(),
        ExprKind::Load { address: value }
        | ExprKind::Unary { value, .. }
        | ExprKind::Extract { value, .. }
        | ExprKind::Resize { value, .. } => vec![value],
        ExprKind::Binary { left, right, .. }
        | ExprKind::Concat {
            high: left,
            low: right,
        } => vec![left, right],
        ExprKind::Intrinsic { arguments, .. } => arguments.iter().collect(),
    }
}

/// Whether `e` is exactly the constant `v`.
fn is_const(e: &Expr, v: u64) -> bool {
    matches!(&e.kind, ExprKind::Const { value } if *value == v)
}

/// A name prefix for a temp, after the dominant protector operation in its
/// definition: a timestamp read is `tsc`, a table checksum is `sum`, a value
/// formed by adding the image base is a pointer `ptr`, everything else `t`.
/// Derived only from terms the expression actually holds, so the name never
/// claims more than the definition shows.
fn role_prefix(expr: &Expr, image_base: u64) -> &'static str {
    if image_base != 0
        && let ExprKind::Binary {
            op: BinaryOp::Add,
            left,
            right,
        } = &expr.kind
        && (is_const(left, image_base) || is_const(right, image_base))
    {
        return "ptr";
    }
    fn scan(e: &Expr, tsc: &mut bool, sum: &mut bool) {
        if let ExprKind::Intrinsic { name, .. } = &e.kind {
            if name.contains("timestamp") {
                *tsc = true;
            }
            if name.contains("checksum") {
                *sum = true;
            }
        }
        for c in children(e) {
            scan(c, tsc, sum);
        }
    }
    let (mut tsc, mut sum) = (false, false);
    scan(expr, &mut tsc, &mut sum);
    if sum {
        "sum"
    } else if tsc {
        "tsc"
    } else {
        "t"
    }
}

/// A read from a fixed address: a global. The address carries no meaning
/// inline -- `*(u64 *)(0x378154c16ULL)` buried mid-expression is just noise --
/// so these are hoisted into named locals even when used only once, which moves
/// every raw address out of the body and into one block of definitions.
fn global_load(expr: &Expr) -> Option<u64> {
    let ExprKind::Load { address } = &expr.kind else {
        return None;
    };
    match address.kind {
        ExprKind::Const { value } => Some(value),
        _ => None,
    }
}

/// Never worth a line of its own, however often it appears: naming these would
/// add noise rather than remove it.
fn trivial(expr: &Expr) -> bool {
    match &expr.kind {
        ExprKind::Const { .. } | ExprKind::Register { .. } => true,
        // A bare cast or negation of a leaf reads better inline.
        ExprKind::Unary { value, .. }
        | ExprKind::Extract { value, .. }
        | ExprKind::Resize { value, .. } => {
            matches!(
                value.kind,
                ExprKind::Const { .. } | ExprKind::Register { .. }
            )
        }
        _ => false,
    }
}

/// Whether a subtree has already been chosen, so a condition operand that is
/// also shared is not named twice.
fn temps_contains(chosen: &[(usize, Expr)], expr: &Expr) -> bool {
    chosen.iter().any(|(_, e)| e == expr)
}

/// Every expression a statement list reads, roots only (children come from the
/// walk). Nested statement lists are included.
fn roots(stmts: &[Stmt], out: &mut Vec<Expr>) {
    roots_and_conditions(stmts, out, &mut Vec::new());
}

/// As [`roots`], also collecting the operands of every branch condition.
///
/// A guard is read once, so the use-count rule leaves it inline -- and a guard
/// is exactly where an inline expression hurts most, because the reader has to
/// hold all of it to see what the branch tests. On vm_c447739 the four longest
/// lines in the function were all `if` conditions of 367-376 characters.
fn roots_and_conditions(stmts: &[Stmt], out: &mut Vec<Expr>, conditions: &mut Vec<Expr>) {
    let mut cond = |c: &Cond, out: &mut Vec<Expr>, conditions: &mut Vec<Expr>| match c {
        Cond::Cmp { a, b, .. } => {
            out.push(a.clone());
            out.push(b.clone());
            conditions.push(a.clone());
            conditions.push(b.clone());
        }
        Cond::Opaque { args, .. } => {
            out.extend(args.iter().cloned());
            conditions.extend(args.iter().cloned());
        }
    };
    for stmt in stmts {
        match stmt {
            Stmt::Assign(_, e) => out.push(e.clone()),
            Stmt::Store(Store { address, value }) => {
                out.push(address.clone());
                out.push(value.clone());
            }
            Stmt::Return(values) => out.extend(values.iter().map(|(_, e)| e.clone())),
            Stmt::If { cond: c, then, els } => {
                cond(c, out, conditions);
                roots_and_conditions(then, out, conditions);
                roots_and_conditions(els, out, conditions);
            }
            Stmt::While { cond: c, body } | Stmt::DoWhile { body, cond: c } => {
                cond(c, out, conditions);
                roots_and_conditions(body, out, conditions);
            }
            Stmt::IfGoto { cond: c, .. } => cond(c, out, conditions),
            Stmt::Loop(body) => roots_and_conditions(body, out, conditions),
            Stmt::Break | Stmt::Continue | Stmt::Label(_) | Stmt::Goto(_) | Stmt::Comment(_) => {}
        }
    }
}

/// How many times each subtree is REFERENCED (parent edges), plus each
/// subtree's node count. The structure walk is memoized so it stays linear on a
/// DAG, while the reference count deliberately is not -- that count is the
/// whole point.
fn survey(roots: &[Expr]) -> (HashMap<Expr, usize>, HashMap<Expr, usize>) {
    let mut uses: HashMap<Expr, usize> = HashMap::new();
    let mut walked: HashSet<Expr> = HashSet::new();
    let mut stack: Vec<Expr> = Vec::new();
    for root in roots {
        *uses.entry(root.clone()).or_default() += 1;
        stack.push(root.clone());
    }
    // Explicit stack: these expressions are deep enough to overflow recursion.
    while let Some(expr) = stack.pop() {
        if !walked.insert(expr.clone()) {
            continue;
        }
        for child in children(&expr) {
            *uses.entry(child.clone()).or_default() += 1;
            stack.push(child.clone());
        }
    }

    // Sizes in true post-order. Discovery order will NOT do: a shared node can
    // be reached from one parent before another parent that also references it,
    // so reversing the discovery order can still size a parent before its child.
    let mut size: HashMap<Expr, usize> = HashMap::new();
    let mut pending: Vec<(Expr, bool)> = roots.iter().map(|r| (r.clone(), false)).collect();
    while let Some((expr, expanded)) = pending.pop() {
        if size.contains_key(&expr) {
            continue;
        }
        if expanded {
            let total = 1 + children(&expr)
                .iter()
                .map(|c| size.get(*c).copied().unwrap_or(1))
                .sum::<usize>();
            size.insert(expr, total);
        } else {
            // Children are pushed AFTER the marker, so they are sized first.
            pending.push((expr.clone(), true));
            for child in children(&expr) {
                if !size.contains_key(child) {
                    pending.push((child.clone(), false));
                }
            }
        }
    }
    (uses, size)
}

/// Picks the subtrees to name. `min_uses` is how many references earn a line,
/// `min_size` how many nodes.
///
/// `taken` are names already in use in the function, so a temporary can never
/// shadow a recovered variable.
#[must_use]
pub fn extract(stmts: &[Stmt], min_uses: usize, min_size: usize, taken: &HashSet<String>) -> Temps {
    extract_with(
        stmts,
        min_uses,
        min_size,
        taken,
        &crate::symbols::Symbols::default(),
    )
}

/// As [`extract`], naming a global that is an import slot after the import
/// itself, so the body says `iat_ws2_32_send` instead of an address.
#[must_use]
pub fn extract_with(
    stmts: &[Stmt],
    min_uses: usize,
    min_size: usize,
    taken: &HashSet<String>,
    symbols: &crate::symbols::Symbols,
) -> Temps {
    let mut rs = Vec::new();
    let mut conditions = Vec::new();
    roots_and_conditions(stmts, &mut rs, &mut conditions);
    let (uses, size) = survey(&rs);

    let mut chosen: Vec<(usize, Expr)> = Vec::new();
    let mut globals: Vec<(u64, Expr)> = Vec::new();
    for (expr, count) in &uses {
        if let Some(address) = global_load(expr) {
            globals.push((address, expr.clone()));
            continue;
        }
        if *count < min_uses || trivial(expr) {
            continue;
        }
        let nodes = size.get(expr).copied().unwrap_or(1);
        if nodes < min_size {
            continue;
        }
        chosen.push((nodes, expr.clone()));
    }
    // A branch operand big enough to dominate its line is named whatever its
    // use count, for the same reason a global is: what costs the reader is the
    // length in place, not the repetition.
    for expr in &conditions {
        if trivial(expr) || temps_contains(&chosen, expr) || global_load(expr).is_some() {
            continue;
        }
        let nodes = size.get(expr).copied().unwrap_or(1);
        if nodes >= COND_MIN_NODES {
            chosen.push((nodes, expr.clone()));
        }
    }
    // Globals first and in address order: they depend on nothing, and sorting
    // by address makes the block scan like a memory map.
    globals.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then_with(|| a.1.width.bits().cmp(&b.1.width.bits()))
    });
    globals.dedup_by(|a, b| a.1 == b.1);
    // Smaller subtrees first: a temporary can only contain strictly smaller
    // subtrees, so size order IS a dependency order. `{expr:?}` breaks ties so
    // the output does not depend on hash iteration order.
    chosen.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then_with(|| format!("{:?}", a.1).cmp(&format!("{:?}", b.1)))
    });

    let free = |stem: &str, count: usize| {
        let mut prefix = stem.to_owned();
        while (0..count).any(|i| taken.contains(&format!("{prefix}{i}"))) {
            prefix.push('_');
        }
        prefix
    };
    let global_prefix = free("g", globals.len());

    let mut temps = Temps::default();
    for (index, (address, expr)) in globals.into_iter().enumerate() {
        let name = symbols
            .identifier_at(address)
            .filter(|n| !taken.contains(n))
            .unwrap_or_else(|| format!("{global_prefix}{index}"));
        temps.map.insert(expr.clone(), name.clone());
        temps.order.push((name, expr));
    }
    let image_base = symbols.image_base();
    let mut used: HashSet<String> = taken.clone();
    used.extend(temps.order.iter().map(|(n, _)| n.clone()));
    for (_, expr) in chosen.into_iter() {
        let role = role_prefix(&expr, image_base);
        let mut n = 0;
        let name = loop {
            let candidate = format!("{role}{n}");
            if !used.contains(&candidate) {
                break candidate;
            }
            n += 1;
        };
        used.insert(name.clone());
        temps.map.insert(expr.clone(), name.clone());
        temps.order.push((name, expr));
    }
    temps
}

#[cfg(test)]
mod tests {
    use vmpx_vm::{BinaryOp, BitWidth};

    use super::*;

    fn reg(name: &str) -> Expr {
        Expr::register(BitWidth::W64, name)
    }

    fn add(a: Expr, b: Expr) -> Expr {
        Expr::binary(BinaryOp::Add, a, b).expect("add")
    }

    #[test]
    fn a_subtree_used_twice_is_named_and_one_used_once_is_not() {
        let shared = add(reg("a"), reg("b"));
        let once = add(reg("e"), reg("f"));
        // Share across two statements: `add(shared, shared)` would be folded by
        // `Expr::binary`, which simplifies on construction.
        let stmts = vec![
            Stmt::Assign("x".to_owned(), add(shared.clone(), reg("c"))),
            Stmt::Assign("y".to_owned(), add(shared.clone(), reg("d"))),
            Stmt::Assign("z".to_owned(), once.clone()),
        ];
        let temps = extract(&stmts, 2, 2, &HashSet::new());
        assert!(
            temps.map.contains_key(&shared),
            "a twice-used subtree earns a name"
        );
        assert!(
            !temps.map.contains_key(&once),
            "a once-used subtree does not"
        );
    }

    #[test]
    fn leaves_are_never_named_however_often_they_appear() {
        let a = reg("a");
        let stmts = vec![
            Stmt::Assign("x".to_owned(), add(a.clone(), reg("b"))),
            Stmt::Assign("y".to_owned(), add(a.clone(), reg("c"))),
        ];
        let temps = extract(&stmts, 2, 1, &HashSet::new());
        assert!(
            !temps.map.contains_key(&a),
            "naming a bare register adds noise"
        );
    }

    #[test]
    fn definitions_come_before_the_definitions_that_use_them() {
        // `inner` is inside `outer`, and both are shared, so `inner` must be
        // defined first or the emitted C would reference a name too early.
        // A `Load` around `inner`: an `add` would be reassociated by the
        // construction-time simplifier and `inner` would stop being a node.
        let inner = add(reg("a"), reg("b"));
        let outer = Expr::load(BitWidth::W64, inner.clone());
        let stmts = vec![
            Stmt::Assign("x".to_owned(), add(outer.clone(), reg("e"))),
            Stmt::Assign("y".to_owned(), add(outer.clone(), reg("f"))),
            Stmt::Assign("z".to_owned(), add(inner.clone(), reg("d"))),
        ];
        let temps = extract(&stmts, 2, 2, &HashSet::new());
        let position = |e: &Expr| temps.order.iter().position(|(_, d)| d == e);
        let (Some(i), Some(o)) = (position(&inner), position(&outer)) else {
            panic!("both shared subtrees should be named");
        };
        assert!(i < o, "inner must be defined before outer");
    }

    #[test]
    fn a_temporary_never_shadows_an_existing_name() {
        let shared = add(reg("a"), reg("b"));
        let stmts = vec![Stmt::Assign(
            "x".to_owned(),
            add(shared.clone(), shared.clone()),
        )];
        let taken: HashSet<String> = ["t0".to_owned()].into_iter().collect();
        let temps = extract(&stmts, 2, 2, &taken);
        assert!(temps.order.iter().all(|(n, _)| !taken.contains(n)));
    }

    #[test]
    fn a_global_is_named_even_when_it_is_read_only_once() {
        let global = Expr::load(BitWidth::W64, Expr::constant(BitWidth::W64, 0x3781_54c1_6));
        let stmts = vec![Stmt::Assign("x".to_owned(), add(global.clone(), reg("a")))];
        let temps = extract(&stmts, 2, 6, &HashSet::new());
        let name = temps
            .map
            .get(&global)
            .expect("a global earns a name on first use");
        assert!(
            name.starts_with('g'),
            "globals get their own prefix, got {name}"
        );
    }

    #[test]
    fn globals_are_defined_before_the_temporaries_that_use_them() {
        let global = Expr::load(BitWidth::W64, Expr::constant(BitWidth::W64, 0x3781_54c1_6));
        let shared = add(global.clone(), reg("a"));
        let stmts = vec![
            Stmt::Assign("x".to_owned(), add(shared.clone(), reg("b"))),
            Stmt::Assign("y".to_owned(), add(shared.clone(), reg("c"))),
        ];
        let temps = extract(&stmts, 2, 2, &HashSet::new());
        let position = |e: &Expr| temps.order.iter().position(|(_, d)| d == e);
        let (Some(g), Some(t)) = (position(&global), position(&shared)) else {
            panic!("both should be named");
        };
        assert!(
            g < t,
            "a global must be defined before anything that reads it"
        );
    }

    #[test]
    fn a_big_branch_condition_is_named_even_though_it_is_read_once() {
        // Build a condition operand large enough to clear COND_MIN_NODES.
        let mut big = add(reg("a"), reg("b"));
        for i in 0..12 {
            big = add(big, reg(&format!("v{i}")));
        }
        let stmts = vec![Stmt::If {
            cond: crate::ir::Cond::Cmp {
                op: crate::ir::Cmp::Eq,
                a: big.clone(),
                b: Expr::constant(BitWidth::W64, 7),
            },
            then: vec![Stmt::Break],
            els: Vec::new(),
        }];
        let temps = extract(&stmts, 2, 6, &HashSet::new());
        assert!(
            temps.map.contains_key(&big),
            "a guard that dominates its line should get a name"
        );
    }

    #[test]
    fn a_small_branch_condition_stays_inline() {
        let small = add(reg("a"), reg("b"));
        let stmts = vec![Stmt::If {
            cond: crate::ir::Cond::Cmp {
                op: crate::ir::Cmp::Eq,
                a: small.clone(),
                b: Expr::constant(BitWidth::W64, 7),
            },
            then: vec![Stmt::Break],
            els: Vec::new(),
        }];
        let temps = extract(&stmts, 2, 6, &HashSet::new());
        assert!(
            !temps.map.contains_key(&small),
            "a short guard reads better inline"
        );
    }

    #[test]
    fn nested_statement_lists_are_searched_too() {
        let shared = add(reg("a"), reg("b"));
        let stmts = vec![Stmt::Loop(vec![
            Stmt::Assign("x".to_owned(), add(shared.clone(), reg("c"))),
            Stmt::Assign("y".to_owned(), add(shared.clone(), reg("d"))),
        ])];
        let temps = extract(&stmts, 2, 2, &HashSet::new());
        assert!(
            temps.map.contains_key(&shared),
            "a loop body is part of the function"
        );
    }
}
