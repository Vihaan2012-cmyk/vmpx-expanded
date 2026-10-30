//! SMT / bitvector equivalence gate — promotion to the sound tier.
//!
//! Pure-Rust core (always available): `Proven` only via structural identity
//! after simplification or a true exhaustive check over every input; `Refuted`
//! only with a concrete counterexample; everything else `Unknown`.
//!
//! With the `smt-z3` feature, inputs too large to enumerate go to Z3 as a
//! fixed-width bitvector query (`a != b`): UNSAT is `Proven`, a SAT model over
//! plain registers is `Refuted`, and anything else (timeout, unknown, models
//! depending on uninterpreted loads/intrinsics, unencodable ops) is `Unknown`.

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
            return smt_check(&a, &b);
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
                _ => return smt_check(&a, &b),
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

#[cfg(not(feature = "smt-z3"))]
fn smt_check(_a: &Expr, _b: &Expr) -> Verdict {
    Verdict::Unknown
}

#[cfg(feature = "smt-z3")]
fn smt_check(a: &Expr, b: &Expr) -> Verdict {
    z3_backend::check(a, b)
}

#[cfg(feature = "smt-z3")]
mod z3_backend {
    //! QF_BV encoding of `vmpx_vm::Expr`, mirroring `evaluate_binary`: shift
    //! counts are masked to 6 bits (64-bit) or 5 bits (narrower), then use
    //! SMT-LIB shifts (count >= width gives 0 / sign fill); rotates use the
    //! masked count modulo the width. Unencodable shapes yield `None`.

    use std::collections::HashMap;

    use z3::ast::{BV, Bool};
    use z3::{Params, SatResult, Solver};

    use super::{Expr, ExprKind, ExprRef, Verdict};
    use vmpx_vm::{BinaryOp, ResizeOp, UnaryOp};

    const TIMEOUT_MS: u32 = 10_000;
    const MAX_DEPTH: u32 = 1000;

    #[allow(clippy::mutable_key_type)] // hash/eq are pointer-based; the atomic is a cache flag
    struct Encoder {
        memo: HashMap<ExprRef, BV>,
        uninterpreted: HashMap<Expr, BV>,
    }

    pub(super) fn check(a: &Expr, b: &Expr) -> Verdict {
        let mut enc = Encoder {
            memo: HashMap::new(),
            uninterpreted: HashMap::new(),
        };
        let (Some(x), Some(y)) = (enc.encode(a, 0), enc.encode(b, 0)) else {
            return Verdict::Unknown;
        };
        if x.get_size() != y.get_size() {
            return Verdict::Unknown;
        }
        let solver = Solver::new();
        let mut params = Params::new();
        params.set_u32("timeout", TIMEOUT_MS);
        solver.set_params(&params);
        let differ: Bool = x.eq(&y).not();
        solver.assert(&differ);
        match solver.check() {
            SatResult::Unsat => Verdict::Proven,
            // A model may give unrelated values to loads/intrinsics that are
            // functionally dependent, so only plain-register models are trusted.
            SatResult::Sat if enc.uninterpreted.is_empty() => Verdict::Refuted,
            _ => Verdict::Unknown,
        }
    }

    impl Encoder {
        fn node(&mut self, r: &ExprRef, depth: u32) -> Option<BV> {
            if let Some(v) = self.memo.get(r) {
                return Some(v.clone());
            }
            let v = self.encode(r, depth + 1)?;
            self.memo.insert(r.clone(), v.clone());
            Some(v)
        }

        fn uninterpreted(&mut self, e: &Expr) -> BV {
            let n = self.uninterpreted.len();
            self.uninterpreted
                .entry(e.clone())
                .or_insert_with(|| BV::new_const(format!("uf!{n}"), e.width.bits()))
                .clone()
        }

        fn encode(&mut self, e: &Expr, depth: u32) -> Option<BV> {
            if depth > MAX_DEPTH {
                return None;
            }
            let bits = e.width.bits();
            let bv = match &e.kind {
                ExprKind::Const { value } => BV::from_u64(e.width.truncate(*value), bits),
                ExprKind::Register { name } => BV::new_const(format!("r!{name}"), bits),
                ExprKind::Unary { op, value } => {
                    let v = self.node(value, depth)?;
                    match op {
                        UnaryOp::Not => v.bvnot(),
                        UnaryOp::Neg => v.bvneg(),
                    }
                }
                ExprKind::Binary { op, left, right } => {
                    let (l, r) = (self.node(left, depth)?, self.node(right, depth)?);
                    if l.get_size() != bits || r.get_size() != bits {
                        return None;
                    }
                    binary(*op, bits, &l, &r)
                }
                ExprKind::Load { .. } | ExprKind::Intrinsic { .. } => self.uninterpreted(e),
                ExprKind::Extract { value, lsb } => {
                    let v = self.node(value, depth)?;
                    let end = lsb.checked_add(bits)?;
                    if end > v.get_size() {
                        return None;
                    }
                    v.extract(end - 1, *lsb)
                }
                ExprKind::Concat { high, low } => {
                    let (h, l) = (self.node(high, depth)?, self.node(low, depth)?);
                    if h.get_size() + l.get_size() != bits {
                        return None;
                    }
                    h.concat(&l)
                }
                ExprKind::Resize { op, value } => {
                    let v = self.node(value, depth)?;
                    let src = v.get_size();
                    match op {
                        ResizeOp::Truncate if bits <= src => v.extract(bits - 1, 0),
                        ResizeOp::ZeroExtend if bits >= src => v.zero_ext(bits - src),
                        ResizeOp::SignExtend if bits >= src => v.sign_ext(bits - src),
                        _ => return None,
                    }
                }
            };
            (bv.get_size() == bits).then_some(bv)
        }
    }

    fn binary(op: BinaryOp, bits: u32, l: &BV, r: &BV) -> BV {
        let mask = if bits == 64 { 0x3F } else { 0x1F };
        let count = r.bvand(BV::from_u64(mask, bits));
        let rotate = count.bvand(BV::from_u64(u64::from(bits - 1), bits));
        let complement = BV::from_u64(u64::from(bits), bits).bvsub(&rotate);
        match op {
            BinaryOp::Add => l.bvadd(r),
            BinaryOp::Sub => l.bvsub(r),
            BinaryOp::Mul => l.bvmul(r),
            BinaryOp::Xor => l.bvxor(r),
            BinaryOp::And => l.bvand(r),
            BinaryOp::Or => l.bvor(r),
            BinaryOp::Shl => l.bvshl(&count),
            BinaryOp::Shr => l.bvlshr(&count),
            BinaryOp::Sar => l.bvashr(&count),
            BinaryOp::Rol => l.bvshl(&rotate).bvor(l.bvlshr(&complement)),
            BinaryOp::Ror => l.bvlshr(&rotate).bvor(l.bvshl(&complement)),
        }
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

    #[cfg(not(feature = "smt-z3"))]
    #[test]
    fn large_space_is_unknown_without_smt_backend() {
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

    #[cfg(feature = "smt-z3")]
    fn mul3_vs_shift_add(tail: u64) -> Verdict {
        let w = BitWidth::W64;
        let a = reg("a", w);
        let lhs = bin(BinaryOp::Mul, a.clone(), Expr::constant(w, 3));
        let rhs = bin(
            BinaryOp::Add,
            bin(BinaryOp::Shl, a.clone(), Expr::constant(w, 1)),
            bin(BinaryOp::Add, a, Expr::constant(w, tail)),
        );
        SmtGate.verify(&lhs, &rhs)
    }

    #[cfg(feature = "smt-z3")]
    #[test]
    fn smt_proves_64_bit_equivalence() {
        assert_eq!(mul3_vs_shift_add(0), Verdict::Proven);
    }

    #[cfg(feature = "smt-z3")]
    #[test]
    fn smt_refutes_64_bit_non_equivalence() {
        assert_eq!(mul3_vs_shift_add(1), Verdict::Refuted);
    }

    #[cfg(feature = "smt-z3")]
    #[test]
    fn smt_rotate_and_shift_masking_match_vm() {
        let w = BitWidth::W64;
        let a = reg("a", w);
        let rol = bin(BinaryOp::Rol, a.clone(), Expr::constant(w, 64 + 5));
        let ror = bin(BinaryOp::Ror, a.clone(), Expr::constant(w, 59));
        assert_eq!(SmtGate.verify(&rol, &ror), Verdict::Proven);
        // count 64 masks to 0 for 64-bit shifts
        let shl = bin(BinaryOp::Shl, a.clone(), Expr::constant(w, 64));
        assert_eq!(SmtGate.verify(&shl, &a), Verdict::Proven);
    }
}
