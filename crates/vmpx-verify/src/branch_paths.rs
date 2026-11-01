//! Multi-path differential: input-dependent branches.
//!
//! The original forks at every branch it cannot decide; each leaf path carries
//! the literals it took. The recovered function already enumerates its own
//! paths under `selector == successor` conditions. Both families partition the
//! input space, so the function is equal exactly when, for every pair
//! (original path, recovered path), the two exits agree on every input that
//! satisfies BOTH conditions. A pair whose joint condition is proven
//! unsatisfiable is vacuous; every other pair is compared with each cell
//! masked by the joint condition (`a & M == b & M`, `M` all-ones iff it holds),
//! so any backend decides it as an ordinary unconditional equality.
//!
//! Soundness lives in three places: the predicates are exact bit-vector
//! translations of the flag semantics (unit-tested against the concrete
//! evaluators; an intrinsic that is not translated stays an uninterpreted free
//! value, which over-approximates), forking never drops an arm, and every bound
//! (fork depth, path count, pair count) fails closed.

use vmpx_vm::{BinaryOp, BitWidth, Expr, ResizeOp, UnaryOp};
use vmpx_vmlift::{CollectedCall, CollectedStore, ExitSemantics, UnrolledFunction};

use crate::differential::{
    AbiPolicy, DiffOutcome, NativePath, differential_obligation, prepare_condition, prepare_recovered,
};
use crate::equiv::{Cell, CellId, CellVerdict, Verifier};

/// Most (original path, recovered path) pairs one function may need.
const MAX_PAIRS: usize = 1024;

fn c(width: BitWidth, v: u64) -> Expr {
    Expr::constant(width, v)
}

fn bin(op: BinaryOp, a: &Expr, b: &Expr) -> Result<Expr, String> {
    Expr::binary(op, a.clone(), b.clone()).map_err(|e| e.to_string())
}

fn not(a: &Expr) -> Expr {
    Expr::unary(UnaryOp::Not, a.clone())
}

fn to_w8(x: &Expr) -> Result<Expr, String> {
    if x.width == BitWidth::W8 {
        return Ok(x.clone());
    }
    Expr::resize(ResizeOp::Truncate, BitWidth::W8, x.clone()).map_err(|e| e.to_string())
}

/// Resizes `x` to `width` (zero-extending or truncating).
fn coerce(x: &Expr, width: BitWidth) -> Result<Expr, String> {
    match x.width.bits().cmp(&width.bits()) {
        std::cmp::Ordering::Equal => Ok(x.clone()),
        std::cmp::Ordering::Greater => Expr::resize(ResizeOp::Truncate, width, x.clone()),
        std::cmp::Ordering::Less => Expr::resize(ResizeOp::ZeroExtend, width, x.clone()),
    }
    .map_err(|e| e.to_string())
}

/// The top bit of `x` as a `W8` 0/1 value.
fn msb(x: &Expr) -> Result<Expr, String> {
    let shifted = bin(BinaryOp::Shr, x, &c(x.width, u64::from(x.width.bits() - 1)))?;
    to_w8(&shifted)
}

/// Bit `k` of `x` as a `W8` 0/1 value.
fn bit(x: &Expr, k: u64) -> Result<Expr, String> {
    let shifted = bin(BinaryOp::Shr, x, &c(x.width, k))?;
    bin(BinaryOp::And, &to_w8(&shifted)?, &c(BitWidth::W8, 1))
}

fn flip(b: &Expr) -> Result<Expr, String> {
    bin(BinaryOp::Xor, b, &c(BitWidth::W8, 1))
}

/// 1 iff `x == 0`.
fn is_zero(x: &Expr) -> Result<Expr, String> {
    let neg = Expr::unary(UnaryOp::Neg, x.clone());
    flip(&msb(&bin(BinaryOp::Or, x, &neg)?)?)
}

/// 1 iff the low byte of `x` has an even number of set bits.
fn even_parity(x: &Expr) -> Result<Expr, String> {
    let mut v = to_w8(x)?;
    for k in [4, 2, 1] {
        let shifted = bin(BinaryOp::Shr, &v, &c(BitWidth::W8, k))?;
        v = bin(BinaryOp::Xor, &v, &shifted)?;
    }
    flip(&bin(BinaryOp::And, &v, &c(BitWidth::W8, 1))?)
}

fn sub_carry(a: &Expr, b: &Expr, res: &Expr) -> Result<Expr, String> {
    let borrow_in = bin(BinaryOp::And, &not(a), b)?;
    let same = bin(BinaryOp::And, &not(&bin(BinaryOp::Xor, a, b)?), res)?;
    msb(&bin(BinaryOp::Or, &borrow_in, &same)?)
}

fn sub_overflow(a: &Expr, b: &Expr, res: &Expr) -> Result<Expr, String> {
    msb(&bin(
        BinaryOp::And,
        &bin(BinaryOp::Xor, a, b)?,
        &bin(BinaryOp::Xor, a, res)?,
    )?)
}

fn add_carry(a: &Expr, b: &Expr, res: &Expr) -> Result<Expr, String> {
    let both = bin(BinaryOp::And, a, b)?;
    let either = bin(BinaryOp::And, &bin(BinaryOp::Or, a, b)?, &not(res))?;
    msb(&bin(BinaryOp::Or, &both, &either)?)
}

fn add_overflow(a: &Expr, b: &Expr, res: &Expr) -> Result<Expr, String> {
    msb(&bin(
        BinaryOp::And,
        &bin(BinaryOp::Xor, a, res)?,
        &bin(BinaryOp::Xor, b, res)?,
    )?)
}

/// Native lifter flag intrinsic (`zf sf pf cf_add of_add cf_sub of_sub` over two
/// operands) as bit-vector operations; `None` leaves it uninterpreted.
fn native_flag(width: BitWidth, name: &str, args: &[Expr]) -> Option<Expr> {
    if width != BitWidth::W8 {
        return None;
    }
    let built = match (name, args) {
        ("zf", [r]) => is_zero(r),
        ("sf", [r]) => msb(r),
        ("pf", [r]) => even_parity(r),
        ("cf_sub", [a, b]) => bin(BinaryOp::Sub, a, b).and_then(|r| sub_carry(a, b, &r)),
        ("of_sub", [a, b]) => bin(BinaryOp::Sub, a, b).and_then(|r| sub_overflow(a, b, &r)),
        ("cf_add", [a, b]) => bin(BinaryOp::Add, a, b).and_then(|r| add_carry(a, b, &r)),
        ("of_add", [a, b]) => bin(BinaryOp::Add, a, b).and_then(|r| add_overflow(a, b, &r)),
        _ => return None,
    };
    built.ok()
}

/// `vm_flags_<add|sub|nor|nand>_w<N>(l, r)`: the RFLAGS word the VM evaluator
/// computes, as bit-vector operations.
fn vm_flags(width: BitWidth, name: &str, args: &[Expr]) -> Option<Expr> {
    let rest = name.strip_prefix("vm_flags_")?;
    let (op, bits) = rest.split_once("_w")?;
    let w = match bits {
        "8" => BitWidth::W8,
        "16" => BitWidth::W16,
        "32" => BitWidth::W32,
        "64" => BitWidth::W64,
        _ => return None,
    };
    let [l, r] = args else { return None };
    if width != BitWidth::W64 {
        return None;
    }
    let build = || -> Result<Expr, String> {
        let (l, r) = (coerce(l, w)?, coerce(r, w)?);
        let (res, cf, of, af) = match op {
            "add" | "sub" => {
                let add = op == "add";
                let res = bin(if add { BinaryOp::Add } else { BinaryOp::Sub }, &l, &r)?;
                let (cf, of) = if add {
                    (add_carry(&l, &r, &res)?, add_overflow(&l, &r, &res)?)
                } else {
                    (sub_carry(&l, &r, &res)?, sub_overflow(&l, &r, &res)?)
                };
                let af = bit(&bin(BinaryOp::Xor, &bin(BinaryOp::Xor, &l, &r)?, &res)?, 4)?;
                (res, cf, of, af)
            }
            "nor" | "nand" => {
                let join = if op == "nor" { BinaryOp::And } else { BinaryOp::Or };
                let zero = c(BitWidth::W8, 0);
                (bin(join, &not(&l), &not(&r))?, zero.clone(), zero.clone(), zero)
            }
            _ => return Err(String::new()),
        };
        let mut word = c(BitWidth::W64, 0x202);
        for (flag, position) in [
            (even_parity(&res)?, 2),
            (is_zero(&res)?, 6),
            (msb(&res)?, 7),
            (cf, 0),
            (af, 4),
            (of, 11),
        ] {
            let wide = Expr::resize(ResizeOp::ZeroExtend, BitWidth::W64, flag).map_err(|e| e.to_string())?;
            let placed = bin(BinaryOp::Shl, &wide, &c(BitWidth::W64, position))?;
            word = bin(BinaryOp::Or, &word, &placed)?;
        }
        Ok(word)
    };
    build().ok()
}

/// Rewrites every translatable native flag and VM flags intrinsic into
/// bit-vector operations. Anything else is left as an uninterpreted term.
///
/// # Errors
/// An expression that cannot be rebuilt.
pub fn expand_flags(e: &Expr) -> Result<Expr, String> {
    e.rewrite_intrinsics(&mut |width, name, args| {
        native_flag(width, name, args).or_else(|| vm_flags(width, name, args))
    })
    .map_err(|x| x.to_string())
}

/// Conjunction of two `W8` 0/1 conditions.
fn and(a: &Expr, b: &Expr) -> Result<Expr, String> {
    bin(BinaryOp::And, a, b)
}

/// `W8` 0/1 condition of one recovered path: every `selector == value` (or `!=`)
/// it took, over the entry symbols.
///
/// # Errors
/// A selector that is not a 64-bit value over entry symbols, or cannot be rebuilt.
pub fn recovered_condition(path: &vmpx_vmlift::UnrolledPath) -> Result<Expr, String> {
    let mut all = c(BitWidth::W8, 1);
    for cond in &path.conditions {
        let selector = prepare_condition(&cond.selector)?;
        if selector.width != BitWidth::W64 {
            return Err("a recovered branch selector is not 64-bit".to_owned());
        }
        if let Some(stray) = selector.dependencies().into_iter().find(|d| !d.starts_with("native_")) {
            return Err(format!("a recovered branch selector depends on `{stray}`"));
        }
        let equal = is_zero(&bin(BinaryOp::Xor, &selector, &c(BitWidth::W64, cond.value))?)?;
        all = and(&all, &if cond.equal { equal } else { flip(&equal)? })?;
    }
    Ok(all)
}

/// Runs `f` over every expression of an exit.
fn map_exit(
    s: &ExitSemantics,
    f: &mut impl FnMut(&Expr) -> Result<Expr, String>,
) -> Result<ExitSemantics, String> {
    let mut out = s.clone();
    for v in out.exit.registers.values_mut() {
        *v = f(v)?;
    }
    for (lo, hi) in out.exit.xmm.values_mut() {
        *lo = f(lo)?;
        *hi = f(hi)?;
    }
    out.exit.rflags = f(&out.exit.rflags)?;
    out.effects.stores = s
        .effects
        .stores
        .iter()
        .map(|st| {
            Ok(CollectedStore { address: f(&st.address)?, value: f(&st.value)?, width: st.width })
        })
        .collect::<Result<_, String>>()?;
    out.effects.calls = s
        .effects
        .calls
        .iter()
        .map(|call| {
            Ok(CollectedCall {
                result_intrinsic: call.result_intrinsic.clone(),
                iat_rva: call.iat_rva,
                arguments: call.arguments.iter().map(&mut *f).collect::<Result<_, String>>()?,
            })
        })
        .collect::<Result<_, String>>()?;
    Ok(out)
}

/// Restricts a cell to the inputs satisfying `condition`, unless the two sides
/// are already identical.
fn restrict(cell: &mut Cell, condition: &Expr) -> Result<(), String> {
    let (a, b) = (cell.vm.simplify(), cell.recovered.simplify());
    if a == b || a.width != b.width {
        return Ok(());
    }
    let wide = coerce(condition, a.width)?;
    let mask = Expr::unary(UnaryOp::Neg, wide);
    cell.vm = bin(BinaryOp::And, &a, &mask)?;
    cell.recovered = bin(BinaryOp::And, &b, &mask)?;
    Ok(())
}

/// Compares every callee-return path of the recovered function against the
/// original's path family, pair by pair under the joint path condition.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn diff_against_original_paths(
    natives: &[NativePath],
    recovered: &UnrolledFunction,
    verifier: &Verifier,
    policy: &AbiPolicy,
    strict_gates: bool,
) -> DiffOutcome {
    if let [only] = natives
        && only.literals.is_empty()
    {
        return crate::differential::diff_against_original(
            &only.function.semantics,
            recovered,
            verifier,
            policy,
            strict_gates,
        );
    }
    let mut out = DiffOutcome::unproven("");
    out.spin_pruned = recovered.spin_arms_pruned;
    let fail = |mut out: DiffOutcome, why: String| {
        out.reason = why;
        out
    };
    let mut originals: Vec<(ExitSemantics, Expr)> = Vec::new();
    for (i, n) in natives.iter().enumerate() {
        let built = n
            .condition()
            .and_then(|cond| expand_flags(&cond))
            .and_then(|cond| Ok((map_exit(&n.function.semantics, &mut expand_flags)?, cond)));
        match built {
            Ok(p) => originals.push(p),
            Err(e) => return fail(out, format!("original path {i}: {e}")),
        }
    }
    let mut recs: Vec<(ExitSemantics, Expr)> = Vec::new();
    let mut why = Vec::new();
    for (index, path) in recovered.paths.iter().enumerate() {
        if path.region_exit {
            out.region_paths += 1;
            continue;
        }
        let built = prepare_recovered(&path.semantics, &path.store_epochs, recovered.entry_sp_offset)
            .and_then(|(sem, notes)| {
                out.frame_stores_dropped += notes.frame_stores_dropped;
                let cond = expand_flags(&recovered_condition(path)?)?;
                Ok((map_exit(&sem, &mut expand_flags)?, cond))
            });
        match built {
            Ok(p) => recs.push(p),
            Err(e) => why.push(format!("path {index}: {e}")),
        }
    }
    if !why.is_empty() {
        return fail(out, why.join("; "));
    }
    if recs.is_empty() {
        return fail(out, "no callee-return path was recovered".to_owned());
    }
    if originals.len().saturating_mul(recs.len()) > MAX_PAIRS {
        return fail(
            out,
            format!("{} original x {} recovered paths exceed the pair budget {MAX_PAIRS}", originals.len(), recs.len()),
        );
    }
    let zero = c(BitWidth::W8, 0);
    let mut obligations = Vec::new();
    let mut vacuous = 0_usize;
    for (oi, (orig, ocond)) in originals.iter().enumerate() {
        let before = obligations.len();
        for (ri, (rec, rcond)) in recs.iter().enumerate() {
            let step = and(ocond, rcond).and_then(|joint| {
                let infeasible = verifier.check_cell(&Cell {
                    id: CellId::Named("joint-path-condition".to_owned()),
                    vm: joint.clone(),
                    recovered: zero.clone(),
                });
                if matches!(infeasible, CellVerdict::Proven(_)) {
                    return Ok(None);
                }
                let mut o = differential_obligation(rec, orig, policy);
                for cell in &mut o.cells {
                    restrict(cell, &joint)?;
                }
                Ok(Some(o))
            });
            match step {
                Ok(Some(o)) => obligations.push(o),
                Ok(None) => vacuous += 1,
                Err(e) => return fail(out, format!("pair (original {oi}, recovered {ri}): {e}")),
            }
        }
        if obligations.len() == before {
            return fail(out, format!("no recovered path can take original path {oi}; the recovered family is incomplete"));
        }
    }
    crate::differential::conclude(out, &obligations, verifier, strict_gates, false)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use vmpx_vm::ExprKind;

    use super::*;
    use crate::smt::fold;

    fn reg(n: &str, w: BitWidth) -> Expr {
        Expr::register(w, n)
    }

    fn rng(seed: &mut u64) -> u64 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        *seed
    }

    fn samples(w: BitWidth) -> Vec<(u64, u64)> {
        let mut seed = 0x9E37_79B9_7F4A_7C15_u64;
        let edge = [0, 1, 2, 0x7F, 0x80, 0xFF, 0x7FFF_FFFF, 0x8000_0000, u64::MAX, i64::MAX as u64, 1 << 63];
        let mut v = Vec::new();
        for a in edge {
            for b in edge {
                v.push((w.truncate(a), w.truncate(b)));
            }
        }
        for _ in 0..400 {
            v.push((w.truncate(rng(&mut seed)), w.truncate(rng(&mut seed))));
        }
        v
    }

    #[test]
    fn native_flag_expansions_match_the_concrete_evaluators() {
        for w in [BitWidth::W8, BitWidth::W16, BitWidth::W32, BitWidth::W64] {
            for (a, b) in samples(w) {
                for name in ["cf_sub", "of_sub", "cf_add", "of_add"] {
                    let args = vec![c(w, a), c(w, b)];
                    let original = Expr::intrinsic(BitWidth::W8, name, args.clone());
                    let expanded = expand_flags(&original).unwrap();
                    assert!(!matches!(expanded.kind, ExprKind::Intrinsic { .. }), "{name} not expanded");
                    let want = crate::differential::eval_flag_for_tests(name, &[(w, a), (w, b)]).unwrap();
                    assert_eq!(fold(&expanded, &BTreeMap::new()), Some(want), "{name} w{} {a:#x} {b:#x}", w.bits());
                }
                for name in ["zf", "sf", "pf"] {
                    let original = Expr::intrinsic(BitWidth::W8, name, vec![c(w, a)]);
                    let expanded = expand_flags(&original).unwrap();
                    let want = crate::differential::eval_flag_for_tests(name, &[(w, a)]).unwrap();
                    assert_eq!(fold(&expanded, &BTreeMap::new()), Some(want), "{name} w{} {a:#x}", w.bits());
                }
            }
        }
    }

    #[test]
    fn vm_flags_expansions_match_the_vm_evaluator() {
        for w in [BitWidth::W8, BitWidth::W16, BitWidth::W32, BitWidth::W64] {
            for (a, b) in samples(w) {
                for op in ["add", "sub", "nor", "nand"] {
                    let name = format!("vm_flags_{op}_w{}", w.bits());
                    // Operands arrive 64-bit wide: the evaluator truncates them to the operation width.
                    let wide = |v: u64| c(BitWidth::W64, v | (0xABCD << 48));
                    let original = Expr::intrinsic(BitWidth::W64, name.clone(), vec![wide(a), wide(b)]);
                    let want = fold(&original.simplify(), &BTreeMap::new()).expect("vm evaluator folds");
                    let expanded = expand_flags(&original).unwrap();
                    assert!(!matches!(expanded.kind, ExprKind::Intrinsic { .. }), "{name} not expanded");
                    assert_eq!(fold(&expanded, &BTreeMap::new()), Some(want), "{name} {a:#x} {b:#x}");
                }
            }
        }
    }

    #[test]
    fn unknown_intrinsics_stay_uninterpreted() {
        let e = Expr::intrinsic(BitWidth::W8, "af_sub", vec![reg("x", BitWidth::W32), reg("y", BitWidth::W32)]);
        assert_eq!(expand_flags(&e).unwrap(), e);
        let three = Expr::intrinsic(
            BitWidth::W8,
            "cf_add",
            vec![reg("x", BitWidth::W32), reg("y", BitWidth::W32), reg("c", BitWidth::W8)],
        );
        assert_eq!(expand_flags(&three).unwrap(), three);
    }
}
