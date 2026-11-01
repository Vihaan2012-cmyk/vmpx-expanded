//! Lowering of recovered VM selector expressions to plain bit-vector terms z3 can decide.
//!
//! The recovered IR carries opaque leaves z3 would treat as uninterpreted (and the verifier then
//! refuses to trust the model): epoch-tagged memory reads, processor/segment probes and the
//! `vm_flags_*` status-flag intrinsics. Here:
//! - `memory_load_epoch_N(<const addr>)` becomes the free "cell" `cell_<addr>_<bytes>`; equal
//!   addresses share a cell regardless of epoch (an APPROXIMATION: sound only if the cell is not
//!   stored to between the reads; the emulator re-run is the oracle that catches a wrong model);
//! - `x86_cpuid_*` / `x86_gs_load_*` / `x86_fs_load_*` become `env_*` registers (machine state the
//!   input cannot steer);
//! - `vm_flags_{add,sub,nor,nand}_wN(l, r)` are expanded to their exact RFLAGS image.

use std::collections::{BTreeMap, BTreeSet};

use vmpx_vm::{BinaryOp, BitWidth, Expr, ExprKind, ResizeOp, UnaryOp};

use crate::pretty::show;

#[derive(Clone, Debug, Default)]
pub struct LowerInfo {
    /// Free memory cells: register name -> (constant address, bytes).
    pub cells: BTreeMap<String, (u64, u32)>,
    /// Environment registers the input cannot steer.
    pub env: BTreeSet<String>,
    /// Leaves that could not be lowered (symbolic-address loads, unknown intrinsics).
    pub stuck: Vec<String>,
}

fn is_env(name: &str) -> bool {
    name.starts_with("x86_cpuid_") || name.starts_with("x86_gs_load_") || name.starts_with("x86_fs_load_")
}

/// # Errors
/// Expression rebuild failure.
pub fn lower(e: &Expr) -> Result<(Expr, LowerInfo), String> {
    let mut info = LowerInfo::default();
    let out = e
        .rewrite_intrinsics(&mut |width, name, args| {
            if name.starts_with("memory_load_epoch_") && args.len() == 1 {
                if let ExprKind::Const { value } = args[0].simplify().kind {
                    let n = format!("cell_{value:x}_{}", width.bytes());
                    info.cells.insert(n.clone(), (value, width.bytes()));
                    return Some(Expr::register(width, n));
                }
                // A read at an address derived from machine state (e.g. gs:[0x60]+0x120, the PEB) is
                // itself machine state; any other symbolic address is real symbolic memory.
                if args[0].dependencies().iter().any(|d| d.starts_with("env_")) {
                    let n = format!("env_{name}@{}", show(&args[0], 4));
                    info.env.insert(n.clone());
                    return Some(Expr::register(width, n));
                }
                info.stuck.push(format!("{name}(symbolic address {})", show(&args[0], 4)));
                return None;
            }
            if is_env(name) {
                let a: Vec<String> = args.iter().map(|x| show(x, 4)).collect();
                let n = format!("env_{name}@{}", a.join(","));
                info.env.insert(n.clone());
                return Some(Expr::register(width, n));
            }
            if let Some(r) = lower_flags(name, width, args) {
                return Some(r);
            }
            info.stuck.push(name.to_owned());
            None
        })
        .map_err(|x| x.to_string())?;
    Ok((out, info))
}

fn c(w: BitWidth, v: u64) -> Expr {
    Expr::constant(w, v)
}

fn bin(op: BinaryOp, l: Expr, r: Expr) -> Option<Expr> {
    Expr::binary(op, l, r).ok()
}

fn bit(x: &Expr, k: u32) -> Option<Expr> {
    let w = x.width;
    bin(BinaryOp::And, bin(BinaryOp::Shr, x.clone(), c(w, u64::from(k)))?, c(w, 1))
}

fn zext(e: Expr, to: BitWidth) -> Option<Expr> {
    if e.width == to { Some(e) } else { Expr::resize(ResizeOp::ZeroExtend, to, e).ok() }
}

/// `vm_flags_<op>_wN(l, r)` as the exact RFLAGS image of the operation (mirrors the evaluator).
fn lower_flags(name: &str, out: BitWidth, args: &[Expr]) -> Option<Expr> {
    let (op, rest) = if let Some(r) = name.strip_prefix("vm_flags_add_w") {
        ("add", r)
    } else if let Some(r) = name.strip_prefix("vm_flags_sub_w") {
        ("sub", r)
    } else if let Some(r) = name.strip_prefix("vm_flags_nor_w") {
        ("nor", r)
    } else if let Some(r) = name.strip_prefix("vm_flags_nand_w") {
        ("nand", r)
    } else {
        return None;
    };
    let bits: u32 = rest.parse().ok()?;
    let [l, r] = args else { return None };
    if l.width.bits() != bits || r.width.bits() != bits {
        return None;
    }
    let w = l.width;
    let not = |x: &Expr| Expr::unary(UnaryOp::Not, x.clone());
    let res = match op {
        "add" => bin(BinaryOp::Add, l.clone(), r.clone())?,
        "sub" => bin(BinaryOp::Sub, l.clone(), r.clone())?,
        "nor" => bin(BinaryOp::And, not(l), not(r))?,
        _ => bin(BinaryOp::Or, not(l), not(r))?,
    };
    // parity of the low byte
    let b = if bits > 8 { bin(BinaryOp::And, res.clone(), c(w, 0xff))? } else { res.clone() };
    let p1 = bin(BinaryOp::Xor, b.clone(), bin(BinaryOp::Shr, b, c(w, 4))?)?;
    let p2 = bin(BinaryOp::Xor, p1.clone(), bin(BinaryOp::Shr, p1, c(w, 2))?)?;
    let p3 = bin(BinaryOp::Xor, p2.clone(), bin(BinaryOp::Shr, p2, c(w, 1))?)?;
    let pf = bin(BinaryOp::Xor, bin(BinaryOp::And, p3, c(w, 1))?, c(w, 1))?;
    let msb = bits - 1;
    let sf = bit(&res, msb)?;
    let neg = Expr::unary(UnaryOp::Neg, res.clone());
    let nonzero = bit(&bin(BinaryOp::Or, res.clone(), neg)?, msb)?;
    let zf = bin(BinaryOp::Xor, nonzero, c(w, 1))?;
    let lr = bin(BinaryOp::Xor, l.clone(), r.clone())?;
    let (cf, af, of) = match op {
        "sub" => {
            let borrow = bin(
                BinaryOp::Or,
                bin(BinaryOp::And, not(l), r.clone())?,
                bin(BinaryOp::And, not(&lr), res.clone())?,
            )?;
            let ov = bin(BinaryOp::And, lr.clone(), bin(BinaryOp::Xor, l.clone(), res.clone())?)?;
            (bit(&borrow, msb)?, bit(&bin(BinaryOp::Xor, lr.clone(), res.clone())?, 4)?, bit(&ov, msb)?)
        }
        "add" => {
            let carry = bin(
                BinaryOp::Or,
                bin(BinaryOp::And, l.clone(), r.clone())?,
                bin(BinaryOp::And, bin(BinaryOp::Or, l.clone(), r.clone())?, not(&res))?,
            )?;
            let ov = bin(BinaryOp::And, not(&lr), bin(BinaryOp::Xor, l.clone(), res.clone())?)?;
            (bit(&carry, msb)?, bit(&bin(BinaryOp::Xor, lr.clone(), res.clone())?, 4)?, bit(&ov, msb)?)
        }
        _ => (c(w, 0), c(w, 0), c(w, 0)),
    };
    let place = |x: Expr, k: u64| bin(BinaryOp::Shl, zext(x, out)?, c(out, k));
    let mut acc = c(out, 0x202);
    for term in [zext(cf, out)?, place(pf, 2)?, place(af, 4)?, place(zf, 6)?, place(sf, 7)?, place(of, 11)?] {
        acc = bin(BinaryOp::Or, acc, term)?;
    }
    Some(acc)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(name: &str, bits: u32, l: u64, r: u64) {
        let w = match bits {
            8 => BitWidth::W8,
            16 => BitWidth::W16,
            32 => BitWidth::W32,
            _ => BitWidth::W64,
        };
        let a = Expr::register(w, "a");
        let b = Expr::register(w, "b");
        let orig = Expr::intrinsic(BitWidth::W64, name, vec![a.clone(), b.clone()]);
        let lowered = lower_flags(name, BitWidth::W64, &[a, b]).expect("lowers");
        let env: BTreeMap<String, Expr> =
            [("a".to_owned(), c(w, l)), ("b".to_owned(), c(w, r))].into_iter().collect();
        let want = orig.substitute(&env).unwrap().simplify();
        let got = lowered.substitute(&env).unwrap().simplify();
        assert_eq!(want, got, "{name} l={l:#x} r={r:#x}");
        assert!(matches!(want.kind, ExprKind::Const { .. }), "reference folded to a constant");
    }

    #[test]
    fn flag_lowering_matches_the_evaluator() {
        let vals = [0u64, 1, 2, 0x7f, 0x80, 0xff, 0x7fff_ffff, 0x8000_0000, 0xffff_ffff, 0x1234_5678, 0xdead_beef, 5, 0xfe0];
        for name in ["vm_flags_sub_w32", "vm_flags_add_w32", "vm_flags_nor_w32", "vm_flags_nand_w32"] {
            for &l in &vals {
                for &r in &vals {
                    check(name, 32, l, r);
                }
            }
        }
        check("vm_flags_sub_w8", 8, 3, 200);
        check("vm_flags_add_w16", 16, 0xffff, 1);
        check("vm_flags_sub_w64", 64, 5, u64::MAX);
    }

    #[test]
    fn cells_are_address_keyed_and_env_is_split_off() {
        let w = BitWidth::W32;
        let a = Expr::intrinsic(w, "memory_load_epoch_5", vec![c(BitWidth::W64, 0x190)]);
        let b = Expr::intrinsic(w, "memory_load_epoch_99", vec![c(BitWidth::W64, 0x190)]);
        let e = Expr::binary(BinaryOp::Xor, a, b).unwrap();
        let (_, info) = lower(&e).unwrap();
        assert_eq!(info.cells.len(), 1, "same address, different epochs, one cell");
        let cpu = Expr::intrinsic(w, "x86_cpuid_eax_65", vec![c(w, 1), c(w, 0)]);
        let (_, info) = lower(&cpu).unwrap();
        assert_eq!(info.env.len(), 1);
    }
}
