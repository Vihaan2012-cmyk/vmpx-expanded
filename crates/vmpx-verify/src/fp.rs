//! Floating-point cells of the differential (IR vocabulary: `vmpx_vm::fp`).
//!
//! Two sound tiers, then a counterexample search; nothing else may prove:
//!
//! 1. [`canonicalize`]: the IEEE op sequence survives virtualization, so the
//!    recovered tree equals the original's after commuting `fadd`/`fmul`
//!    operands into a canonical order, folding constants (done by
//!    `Expr::intrinsic`) and cancelling `fneg(fneg x)`. Identical canonical
//!    trees compute the same value at every input: bit-exact, no solver. The
//!    one thing commutation can change is which payload a NaN result carries
//!    when BOTH operands of a commuted node are NaN (x86 returns the first).
//! 2. The Z3 backend's floating-point theory (round-nearest-even), see `smt.rs`.
//!    Z3 has one NaN, so its UNSAT is equality modulo NaN payload/sign.
//! 3. [`probe`]: a deterministic search over "interesting" doubles; a hit is
//!    replayed through the exact evaluator before it is believed.
//!
//! NaN policy: non-structural tiers compare floats with NaN merged (Z3 has one
//! NaN), and the counterexample replay applies the same rule, so a divergence
//! where both sides are NaN (different payload or sign) is neither proof nor
//! refutation. Structural identity is exact.
//!
//! No tolerance anywhere: closeness never proves anything. Reassociation
//! (`(a+b)+c` vs `a+(b+c)`) is not an equivalence in IEEE and is left to the
//! solver, which refutes it. Libm calls are pure uninterpreted functions: equal
//! target and equal arguments give equal results; nothing models their insides.

use std::collections::{BTreeMap, HashSet};
use std::hash::{Hash, Hasher};

use vmpx_vm::fp::{FpIntrinsic, FpUnary};
use vmpx_vm::{BitWidth, Expr, ExprKind};

use crate::smt::{fold, free_registers};

/// True if any node of `e` is an `fp_*` intrinsic.
#[must_use]
pub fn mentions_fp(e: &Expr) -> bool {
    let mut seen: HashSet<Expr> = HashSet::new();
    let mut stack = vec![e.clone()];
    while let Some(n) = stack.pop() {
        if !seen.insert(n.clone()) {
            continue;
        }
        match &n.kind {
            ExprKind::Const { .. } | ExprKind::Register { .. } => {}
            ExprKind::Intrinsic { name, arguments } => {
                if FpIntrinsic::parse(name).is_some() {
                    return true;
                }
                stack.extend(arguments.iter().cloned());
            }
            ExprKind::Unary { value, .. }
            | ExprKind::Load { address: value }
            | ExprKind::Extract { value, .. }
            | ExprKind::Resize { value, .. } => stack.push((**value).clone()),
            ExprKind::Binary { left, right, .. }
            | ExprKind::Concat { high: left, low: right } => {
                stack.push((**left).clone());
                stack.push((**right).clone());
            }
        }
    }
    false
}

/// True if the root of `e` is an FP operation producing a float bit pattern.
#[must_use]
pub fn yields_float(e: &Expr) -> bool {
    matches!(&e.kind, ExprKind::Intrinsic { name, .. }
        if FpIntrinsic::parse(name).is_some_and(|op| op.yields_float()))
}

/// True when `x` and `y` are the values of float-yielding roots `a`/`b` and both
/// are NaN: equal under the NaN-merged equality every non-structural tier uses,
/// so such a divergence is neither a proof nor a refutation.
#[must_use]
pub fn both_nan(a: &Expr, b: &Expr, x: u64, y: u64) -> bool {
    if !(yields_float(a) || yields_float(b)) {
        return false;
    }
    let nan = |bits: u64| match a.width {
        BitWidth::W32 => f32::from_bits(bits as u32).is_nan(),
        BitWidth::W64 => f64::from_bits(bits).is_nan(),
        _ => false,
    };
    nan(x) && nan(y)
}

fn order_key(e: &Expr) -> u64 {
    let mut h = std::hash::DefaultHasher::new();
    e.hash(&mut h);
    h.finish()
}

/// Canonical form of an FP expression (see the module docs for what it may and
/// may not change). Falls back to `e` unchanged if rebuilding fails.
#[must_use]
pub fn canonicalize(e: &Expr) -> Expr {
    e.rewrite_intrinsics(&mut |width, name, args| {
        let op = FpIntrinsic::parse(name)?;
        if let (FpIntrinsic::Unary(FpUnary::Neg, _), [only]) = (&op, args)
            && let ExprKind::Intrinsic { name: inner, arguments } = &only.kind
            && inner == name
            && let [original] = arguments.as_slice()
        {
            return Some(original.clone());
        }
        if op.is_commutative()
            && let [x, y] = args
            && order_key(x) > order_key(y)
        {
            return Some(Expr::intrinsic(width, name.to_owned(), vec![y.clone(), x.clone()]));
        }
        None
    })
    .unwrap_or_else(|_| e.clone())
}

const DOUBLES: [f64; 24] = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    0.5,
    2.0,
    3.0,
    0.1,
    0.2,
    0.3,
    10.0,
    -2.5,
    1.0e-3,
    1.0e3,
    123.456,
    std::f64::consts::PI,
    1.0e10,
    1.0e-310,
    1.0e154,
    1.0e300,
    f64::MAX,
    f64::INFINITY,
    f64::NEG_INFINITY,
    f64::MIN_POSITIVE,
];

fn pool(width: BitWidth) -> Vec<u64> {
    match width {
        BitWidth::W64 => DOUBLES.iter().map(|v| v.to_bits()).chain([1, 2, 3, 7]).collect(),
        BitWidth::W32 => DOUBLES
            .iter()
            .map(|v| u64::from((*v as f32).to_bits()))
            .chain([1, 2, 3, 7])
            .collect(),
        _ => (0..8).chain([0x7F, 0x80, 0xFF]).map(|v| width.truncate(v)).collect(),
    }
}

const PROBES: usize = 700;

/// Searches finite, non-NaN inputs for one on which `a` and `b` differ.
/// Returns a candidate model (the caller replays it); `None` if none was found
/// or a term cannot be evaluated (a call). NaN inputs are excluded on purpose:
/// the tiers above treat NaN payloads as equal, so only real value
/// differences are reported.
#[must_use]
pub fn probe(a: &Expr, b: &Expr) -> Option<BTreeMap<String, u64>> {
    let vars = free_registers(a, b)?;
    let names: Vec<(&String, Vec<u64>)> = vars.iter().map(|(n, w)| (n, pool(*w))).collect();
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let grid = names.first().map_or(1, |(_, v)| v.len().pow(names.len().min(2) as u32));
    for i in 0..if names.is_empty() { 1 } else { PROBES } {
        let mut env = BTreeMap::new();
        let mut model = BTreeMap::new();
        let mut rest = i;
        for (name, values) in &names {
            // small spaces are covered exhaustively first, then pseudo-randomly
            let pick = if i < grid {
                let p = rest % values.len();
                rest /= values.len();
                p
            } else {
                (next() % values.len() as u64) as usize
            };
            let value = values[pick];
            env.insert((*name).clone(), Expr::constant(vars[*name], value));
            model.insert((*name).clone(), value);
        }
        match (fold(a, &env), fold(b, &env)) {
            (Some(x), Some(y)) if x != y && !both_nan(a, b, x, y) => return Some(model),
            (Some(_), Some(_)) => {}
            _ => return None,
        }
    }
    None
}
