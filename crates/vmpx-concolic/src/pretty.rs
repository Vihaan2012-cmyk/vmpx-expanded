//! Compact printer for `Expr` (depth-limited) for the spike's reports.

use vmpx_vm::{Expr, ExprKind};

#[must_use]
pub fn show(e: &Expr, depth: u32) -> String {
    if depth == 0 {
        return "..".into();
    }
    let d = depth - 1;
    match &e.kind {
        ExprKind::Const { value } => format!("{value:#x}"),
        ExprKind::Register { name } => name.clone(),
        ExprKind::Unary { op, value } => format!("{op:?}({})", show(value, d)),
        ExprKind::Binary { op, left, right } => format!("({} {op:?} {})", show(left, d), show(right, d)),
        ExprKind::Load { address } => format!("load[{}]", show(address, d)),
        ExprKind::Intrinsic { name, arguments } => {
            let a: Vec<String> = arguments.iter().map(|x| show(x, d)).collect();
            format!("{name}({})", a.join(","))
        }
        ExprKind::Extract { value, lsb } => format!("{}[{lsb}+{}]", show(value, d), e.width.bits()),
        ExprKind::Concat { high, low } => format!("{{{}:{}}}", show(high, d), show(low, d)),
        ExprKind::Resize { op, value } => format!("{op:?}{}({})", e.width.bits(), show(value, d)),
    }
}

/// Every distinct Intrinsic / Load leaf term (printed), for diagnosing what a selector reads.
#[must_use]
pub fn leaves(e: &Expr, depth: u32) -> Vec<String> {
    fn go(e: &Expr, depth: u32, seen: &mut std::collections::BTreeSet<String>, budget: &mut u32) {
        if *budget == 0 {
            return;
        }
        *budget -= 1;
        match &e.kind {
            ExprKind::Const { .. } | ExprKind::Register { .. } => {}
            ExprKind::Unary { value, .. } | ExprKind::Extract { value, .. } | ExprKind::Resize { value, .. } => {
                go(value, depth, seen, budget);
            }
            ExprKind::Binary { left, right, .. } | ExprKind::Concat { high: left, low: right } => {
                go(left, depth, seen, budget);
                go(right, depth, seen, budget);
            }
            ExprKind::Load { .. } | ExprKind::Intrinsic { .. } => {
                seen.insert(show(e, depth));
            }
        }
    }
    let mut seen = std::collections::BTreeSet::new();
    let mut budget = 200_000;
    go(e, depth, &mut seen, &mut budget);
    seen.into_iter().collect()
}
