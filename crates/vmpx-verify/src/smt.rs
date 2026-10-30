//! SMT / bitvector equivalence gate — promotion to the sound tier.
//!
//! Pure-Rust, sound-but-limited first version: `Proven` only via structural
//! identity after simplification or a true exhaustive check over every input;
//! `Refuted` only with a concrete counterexample; everything else `Unknown`.
//! A future WP can add a real SMT backend (z3/boolector) behind the same
//! [`Gate`] to decide large input spaces.

use std::collections::{BTreeMap, HashSet};

use crate::{Gate, Verdict};
use vmpx_vm::{BitWidth, Expr, ExprKind, ExprRef};

/// Default cap on exhaustive search: total input bits across all free variables.
pub const DEFAULT_MAX_BITS: u32 = 16;

/// Proves `original == candidate` over bitvectors.
pub struct SmtGate;

impl SmtGate {
    /// Like [`Gate::verify`] with an explicit exhaustive-search cap of
    /// `2^max_bits` total input combinations.
    #[must_use]
    pub fn verify_bounded(&self, original: &Expr, cand: &Expr, max_bits: u32) -> Verdict {
        let (a, b) = (original.simplify(), cand.simplify());
        if a.width != b.width {
            return Verdict::Unknown;
        }
        if a == b {
            return Verdict::Proven;
        }
        let Some(vars) = free_registers(&a, &b) else {
            return Verdict::Unknown;
        };
        let total_bits: u32 = vars.values().map(|w| w.bits()).sum();
        if total_bits > max_bits.min(40) {
            return Verdict::Unknown;
        }
        let names: Vec<(&String, BitWidth)> = vars.iter().map(|(n, w)| (n, *w)).collect();
        for index in 0..(1_u64 << total_bits) {
            let mut rest = index;
            let mut env = BTreeMap::new();
            for (name, width) in &names {
                let value = rest & (u64::MAX >> (64 - width.bits()));
                rest >>= width.bits();
                env.insert((*name).clone(), Expr::constant(*width, value));
            }
            match (fold(&a, &env), fold(&b, &env)) {
                (Some(x), Some(y)) if x == y => {}
                (Some(_), Some(_)) => return Verdict::Refuted,
                _ => return Verdict::Unknown,
            }
        }
        Verdict::Proven
    }
}

impl Gate for SmtGate {
    fn verify(&self, original: &Expr, cand: &Expr) -> Verdict {
        self.verify_bounded(original, cand, DEFAULT_MAX_BITS)
    }
}

/// Substitutes every register and requires a constant result.
fn fold(expr: &Expr, env: &BTreeMap<String, Expr>) -> Option<u64> {
    match expr.substitute(env).ok()?.kind {
        ExprKind::Const { value } => Some(value),
        _ => None,
    }
}

/// Register names with widths across both expressions; `None` on a width clash.
fn free_registers<'a>(a: &'a Expr, b: &'a Expr) -> Option<BTreeMap<String, BitWidth>> {
    let mut vars = BTreeMap::new();
    #[allow(clippy::mutable_key_type)] // hash/eq are pointer-based; the atomic is a cache flag
    let mut seen: HashSet<ExprRef> = HashSet::new();
    let mut pending = vec![a, b];
    while let Some(e) = pending.pop() {
        let mut push = |c: &'a ExprRef, pending: &mut Vec<&'a Expr>| {
            if seen.insert(c.clone()) {
                pending.push(c);
            }
        };
        match &e.kind {
            ExprKind::Const { .. } => {}
            ExprKind::Register { name } => {
                if *vars.entry(name.clone()).or_insert(e.width) != e.width {
                    return None;
                }
            }
            ExprKind::Unary { value, .. }
            | ExprKind::Load { address: value }
            | ExprKind::Extract { value, .. }
            | ExprKind::Resize { value, .. } => push(value, &mut pending),
            ExprKind::Binary { left, right, .. }
            | ExprKind::Concat {
                high: left,
                low: right,
            } => {
                push(left, &mut pending);
                push(right, &mut pending);
            }
            ExprKind::Intrinsic { arguments, .. } => pending.extend(arguments),
        }
    }
    Some(vars)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vmpx_vm::{BinaryOp, UnaryOp};

    fn reg(n: &str, w: BitWidth) -> Expr {
        Expr::register(w, n)
    }
    fn bin(op: BinaryOp, l: Expr, r: Expr) -> Expr {
        Expr::binary(op, l, r).unwrap()
    }

    #[test]
    fn structural_identity_is_proven() {
        let a = bin(BinaryOp::Add, reg("a", BitWidth::W64), reg("b", BitWidth::W64));
        assert_eq!(SmtGate.verify(&a, &a.clone()), Verdict::Proven);
    }

    #[test]
    fn small_width_equivalence_is_proven_exhaustively() {
        let w = BitWidth::W8;
        let (a, b) = (reg("a", w), reg("b", w));
        let lhs = Expr::unary(UnaryOp::Not, bin(BinaryOp::And, a.clone(), b.clone()));
        let rhs = bin(
            BinaryOp::Or,
            Expr::unary(UnaryOp::Not, a),
            Expr::unary(UnaryOp::Not, b),
        );
        assert_eq!(SmtGate.verify(&lhs, &rhs), Verdict::Proven);
    }

    #[test]
    fn non_equivalence_is_refuted() {
        let w = BitWidth::W8;
        let a = reg("a", w);
        let lhs = bin(BinaryOp::Add, a.clone(), a.clone());
        let rhs = bin(BinaryOp::Add, a, Expr::constant(w, 1));
        assert_eq!(SmtGate.verify(&lhs, &rhs), Verdict::Refuted);
    }

    #[test]
    fn large_space_is_unknown_not_proven() {
        let w = BitWidth::W64;
        let a = reg("a", w);
        let lhs = bin(BinaryOp::Mul, a.clone(), Expr::constant(w, 3));
        let rhs = bin(
            BinaryOp::Add,
            bin(BinaryOp::Shl, a.clone(), Expr::constant(w, 1)),
            a,
        );
        assert_eq!(SmtGate.verify(&lhs, &rhs), Verdict::Unknown);
    }
}
