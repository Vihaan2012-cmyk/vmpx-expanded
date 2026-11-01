//! Path-condition encoding over the bit-vector `Expr` IR and the z3 query.
//!
//! The IR has no comparison operator, so `selector == v` is encoded as a 0/1 "violation" bit
//! (`((d | -d) >> 63)` with `d = selector ^ v` is 1 iff `selector != v`); the conjunction is the
//! OR of the violation bits, and the existing `SmtBackend::check_distinct(violation, 1)` is
//! satisfiable exactly when some assignment drives every literal true.

use std::collections::BTreeMap;

use vmpx_verify::equiv::{Budget, SmtBackend, SolveOutcome};
use vmpx_vm::{BinaryOp, BitWidth, Expr, ResizeOp, UnaryOp};

/// `selector == val` (eq) or `selector != val`.
#[derive(Clone, Debug)]
pub struct Lit {
    pub sel: Expr,
    pub val: u64,
    pub eq: bool,
}

fn c64(v: u64) -> Expr {
    Expr::constant(BitWidth::W64, v)
}

fn bin(op: BinaryOp, l: Expr, r: Expr) -> Expr {
    Expr::binary(op, l, r).expect("w64 binary")
}

fn widen(e: &Expr) -> Expr {
    if e.width == BitWidth::W64 {
        e.clone()
    } else {
        Expr::resize(ResizeOp::ZeroExtend, BitWidth::W64, e.clone()).unwrap_or_else(|_| e.clone())
    }
}

/// 1 iff `sel != v`.
fn ne_bit(sel: &Expr, v: u64) -> Expr {
    let d = bin(BinaryOp::Xor, widen(sel), c64(v));
    let nd = Expr::unary(UnaryOp::Neg, d.clone());
    bin(BinaryOp::Shr, bin(BinaryOp::Or, d, nd), c64(63))
}

/// 0 iff every literal holds, 1 otherwise.
#[must_use]
pub fn violation(lits: &[Lit]) -> Expr {
    let mut acc = c64(0);
    for l in lits {
        let ne = ne_bit(&l.sel, l.val);
        let bad = if l.eq { ne } else { bin(BinaryOp::Xor, ne, c64(1)) };
        acc = bin(BinaryOp::Or, acc, bad);
    }
    acc.simplify()
}

#[derive(Debug)]
pub enum Solved {
    Sat(BTreeMap<String, u64>),
    Unsat,
    /// The condition still contains uninterpreted leaves (symbolic-address loads, unknown intrinsics).
    Uninterpreted(String),
    Other(String),
}

/// Satisfy the conjunction `lits`.
pub fn solve(backend: &dyn SmtBackend, lits: &[Lit], timeout_ms: u32) -> Solved {
    let v = violation(lits);
    let budget = Budget { exhaustive_bits: 0, timeout_ms };
    match backend.check_distinct(&v, &c64(1), &budget) {
        SolveOutcome::Sat(m) => Solved::Sat(m),
        SolveOutcome::Unsat => Solved::Unsat,
        SolveOutcome::Timeout => Solved::Other("timeout".into()),
        SolveOutcome::Unknown(why) if why.contains("uninterpreted") => Solved::Uninterpreted(why),
        SolveOutcome::Unknown(why) => Solved::Other(why),
    }
}

/// The z3 backend when built with `smt-z3`, else the bounded native one (cannot decide wide cells).
#[must_use]
pub fn backend() -> (Box<dyn SmtBackend>, &'static str) {
    #[cfg(feature = "smt-z3")]
    {
        (Box::new(vmpx_verify::smt::Z3Backend), "z3")
    }
    #[cfg(not(feature = "smt-z3"))]
    {
        (Box::new(vmpx_verify::equiv::NativeBackend), "native-bounded (no z3 feature)")
    }
}
