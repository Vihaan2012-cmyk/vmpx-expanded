//! Ground-truth differential: the original (unprotected) native function lifted
//! with the native lifter and executed along its input-independent path, against
//! a function recovered from the protected binary, compared by the equivalence
//! gate under an explicit calling-convention policy.
//!
//! Both sides are expressed over the same entry symbols (`native_<reg>_initial`),
//! so the argument mapping is by construction: whatever the original reads from
//! `rcx`/`rdx`/`r8`/`r9` the recovered side reads from the same symbols. Anything
//! this module cannot model (undecidable branch, aliasing it cannot rule out, a
//! flag read before definition, unbalanced stack) is an error, never a guess.

use std::collections::{BTreeMap, HashMap};

use vmpx_disasm::{Flag, FlagState, Gpr};
use vmpx_mutation::cfg_recovery::{CallTarget, NativeCfg, Terminator};
use vmpx_vm::{BinaryOp, BitWidth, Expr, ExprKind, NativeExitState, X64NativeRegister};
use vmpx_vmlift::{CollectedCall, CollectedEffects, CollectedStore, ExitSemantics};

use crate::equiv::{CellId, CellVerdict, EquivVerdict, FunctionReport, Obligation, Verifier};

const RSP_ENTRY: &str = "native_rsp_entry";

fn entry_symbol(r: Gpr) -> Expr {
    Expr::register(BitWidth::W64, entry_name(r))
}

fn entry_name(r: Gpr) -> String {
    if r == Gpr::Rsp {
        return RSP_ENTRY.to_owned();
    }
    format!("native_{}_initial", r.name())
}

/// Entry symbol of one 64-bit XMM lane (`native_xmm<n>_lo_initial`).
#[must_use]
pub fn xmm_entry_symbol(reg: u8, hi: bool) -> Expr {
    Expr::register(BitWidth::W64, xmm_entry_name(reg, hi))
}

fn xmm_entry_name(reg: u8, hi: bool) -> String {
    format!("native_xmm{reg}_{}_initial", if hi { "hi" } else { "lo" })
}

/// The general-purpose register an exit register names.
#[must_use]
pub fn gpr_of(r: X64NativeRegister) -> Gpr {
    native_register(r)
}

fn native_register(r: X64NativeRegister) -> Gpr {
    match r {
        X64NativeRegister::Rax => Gpr::Rax,
        X64NativeRegister::Rbx => Gpr::Rbx,
        X64NativeRegister::Rcx => Gpr::Rcx,
        X64NativeRegister::Rdx => Gpr::Rdx,
        X64NativeRegister::Rbp => Gpr::Rbp,
        X64NativeRegister::Rsi => Gpr::Rsi,
        X64NativeRegister::Rdi => Gpr::Rdi,
        X64NativeRegister::R8 => Gpr::R8,
        X64NativeRegister::R9 => Gpr::R9,
        X64NativeRegister::R10 => Gpr::R10,
        X64NativeRegister::R11 => Gpr::R11,
        X64NativeRegister::R12 => Gpr::R12,
        X64NativeRegister::R13 => Gpr::R13,
        X64NativeRegister::R14 => Gpr::R14,
        X64NativeRegister::R15 => Gpr::R15,
    }
}

/// Splits `address` into (symbolic base, constant offset).
fn split_offset(address: &Expr) -> (Expr, i64) {
    let a = address.simplify();
    if let ExprKind::Binary { op: BinaryOp::Add, left, right } = &a.kind {
        if let ExprKind::Const { value } = right.kind {
            return ((**left).clone(), value.cast_signed());
        }
        if let ExprKind::Const { value } = left.kind {
            return ((**right).clone(), value.cast_signed());
        }
    }
    if let ExprKind::Const { value } = a.kind {
        return (Expr::constant(BitWidth::W64, 0), value.cast_signed());
    }
    (a, 0)
}

fn disjoint(a: &Expr, a_bytes: u32, b: &Expr, b_bytes: u32) -> bool {
    let ((ab, ao), (bb, bo)) = (split_offset(a), split_offset(b));
    ab == bb && (ao + i64::from(a_bytes) <= bo || bo + i64::from(b_bytes) <= ao)
}

/// A store `(address, value)` of `width`, as the memory model sees it.
type Store = (Expr, Expr, BitWidth);

/// Rewrites every `Load(address)` (a read of memory as it was at the start of
/// the block) against the stores that happened before: an exact earlier store
/// forwards its value, a provably disjoint one is skipped, anything else is an
/// error. With no earlier stores the expression is returned unchanged.
fn resolve_loads(e: &Expr, stores: &[Store], after_call: Option<&NativeEnv>) -> Result<Expr, String> {
    if stores.is_empty() && after_call.is_none() {
        return Ok(e.clone());
    }
    let mut memo: HashMap<Expr, Expr> = HashMap::new();
    rebuild(e, stores, after_call, &mut memo)
}

fn rebuild(
    e: &Expr,
    stores: &[Store],
    after_call: Option<&NativeEnv>,
    memo: &mut HashMap<Expr, Expr>,
) -> Result<Expr, String> {
    if let Some(done) = memo.get(e) {
        return Ok(done.clone());
    }
    let err = |x: vmpx_vm::ExprError| x.to_string();
    let out = match &e.kind {
        ExprKind::Const { .. } | ExprKind::Register { .. } => e.clone(),
        ExprKind::Unary { op, value } => Expr::unary(*op, rebuild(value, stores, after_call, memo)?),
        ExprKind::Binary { op, left, right } => {
            Expr::binary(*op, rebuild(left, stores, after_call, memo)?, rebuild(right, stores, after_call, memo)?).map_err(err)?
        }
        ExprKind::Extract { value, lsb } => {
            Expr::extract(e.width, *lsb, rebuild(value, stores, after_call, memo)?).map_err(err)?
        }
        ExprKind::Concat { high, low } => {
            Expr::concat(rebuild(high, stores, after_call, memo)?, rebuild(low, stores, after_call, memo)?).map_err(err)?
        }
        ExprKind::Resize { op, value } => {
            Expr::resize(*op, e.width, rebuild(value, stores, after_call, memo)?).map_err(err)?
        }
        ExprKind::Intrinsic { name, arguments } => Expr::intrinsic(
            e.width,
            name.clone(),
            arguments
                .iter()
                .map(|a| rebuild(a, stores, after_call, memo))
                .collect::<Result<_, _>>()?,
        ),
        ExprKind::Load { address } => {
            let address = rebuild(address, stores, after_call, memo)?;
            load_against(e.width, &address, stores, after_call)?
        }
    };
    memo.insert(e.clone(), out.clone());
    Ok(out)
}

fn load_against(
    width: BitWidth,
    address: &Expr,
    stores: &[Store],
    after_call: Option<&NativeEnv>,
) -> Result<Expr, String> {
    for (store_address, value, store_width) in stores.iter().rev() {
        if store_address.simplify() == address.simplify() && *store_width == width {
            return Ok(value.clone());
        }
        if !disjoint(address, width.bytes(), store_address, store_width.bytes()) {
            return Err(format!(
                "a load may alias an earlier store ({} bytes at {:?})",
                width.bytes(),
                address.simplify()
            ));
        }
    }
    if let Some(env) = after_call {
        // Memory is no longer the function-entry image once a callee has run. Only
        // read-only image bytes and the caller's own frame above the return address
        // (assumed not to escape to the callee) are still the entry values.
        let (base, offset) = split_offset(address);
        let readonly = base == Expr::constant(BitWidth::W64, 0)
            && env.readonly.iter().any(|(lo, hi)| {
                let a = offset.cast_unsigned();
                a >= *lo && a.saturating_add(u64::from(width.bytes())) <= *hi
            });
        let caller_frame = base == entry_symbol(Gpr::Rsp) && offset >= 8;
        if !readonly && !caller_frame {
            return Err(format!(
                "a load of {:?} after a call reads memory the callee may have written",
                address.simplify()
            ));
        }
    }
    Ok(Expr::load(width, address.clone()))
}

/// The image facts the native executor needs to model calls and post-call loads.
#[derive(Clone, Debug, Default)]
pub struct NativeEnv {
    /// Preferred load address of the image (calls are keyed by RVA).
    pub image_base: u64,
    /// Virtual-address ranges of read-only sections (`[lo, hi)`).
    pub readonly: Vec<(u64, u64)>,
}

/// The original function as executed along its input-independent path.
#[derive(Clone, Debug)]
pub struct NativeFunction {
    pub semantics: ExitSemantics,
    pub blocks_executed: usize,
    pub branches_decided: usize,
    /// Stores below the entry stack pointer (the callee's own dead frame).
    pub frame_stores_dropped: usize,
    /// Labeled modeling assumptions the result depends on (calls only).
    pub assumptions: Vec<String>,
}

fn flag_value(e: &Expr) -> Option<u64> {
    let e = e.simplify();
    match &e.kind {
        ExprKind::Const { value } => Some(*value),
        ExprKind::Intrinsic { name, arguments } => {
            let mut args = Vec::new();
            for a in arguments {
                let ExprKind::Const { value } = a.simplify().kind else {
                    return None;
                };
                args.push((a.width, value));
            }
            eval_flag_intrinsic(name, &args)
        }
        _ => None,
    }
}

#[cfg(test)]
pub(crate) fn eval_flag_for_tests(name: &str, args: &[(BitWidth, u64)]) -> Option<u64> {
    eval_flag_intrinsic(name, args)
}

fn truncate(width: BitWidth, v: u64) -> u64 {
    width.truncate(v)
}

fn sign_bit(width: BitWidth, v: u64) -> u64 {
    (truncate(width, v) >> (width.bits() - 1)) & 1
}

fn eval_flag_intrinsic(name: &str, args: &[(BitWidth, u64)]) -> Option<u64> {
    let flag = u64::from;
    match (name, args) {
        ("zf", [(w, r)]) => Some(flag(truncate(*w, *r) == 0)),
        ("sf", [(w, r)]) => Some(sign_bit(*w, *r)),
        ("pf", [(_, r)]) => Some(flag((r & 0xFF).count_ones() % 2 == 0)),
        ("cf_add", [(w, a), (_, b)]) => {
            let sum = u128::from(truncate(*w, *a)) + u128::from(truncate(*w, *b));
            Some(flag((sum >> w.bits()) & 1 == 1))
        }
        ("of_add", [(w, a), (_, b)]) => {
            let r = truncate(*w, a.wrapping_add(*b));
            Some(sign_bit(*w, (a ^ r) & (b ^ r)))
        }
        ("cf_sub", [(w, a), (_, b)]) => Some(flag(truncate(*w, *a) < truncate(*w, *b))),
        ("of_sub", [(w, a), (_, b)]) => {
            let r = truncate(*w, a.wrapping_sub(*b));
            Some(sign_bit(*w, (a ^ b) & (a ^ r)))
        }
        _ => None,
    }
}

const INPUT_DEPENDENT: &str = "depends on the function's inputs";

/// A branch the flags cannot decide, as a `W8` 0/1 predicate and the polarity at
/// which the branch is taken (`taken` iff `predicate == polarity`).
fn branch_literal(cc: &str, flags: &BTreeMap<Flag, Expr>) -> Result<(Expr, bool), String> {
    let get = |f: Flag| -> Result<Expr, String> {
        let e = flags
            .get(&f)
            .ok_or_else(|| format!("branch `{cc}` reads flag {} that is not defined", f.name()))?;
        crate::branch_paths::expand_flags(e)
    };
    let bin = |op: BinaryOp, a: Expr, b: Expr| Expr::binary(op, a, b).map_err(|e| e.to_string());
    let lt = || bin(BinaryOp::Xor, get(Flag::Sf)?, get(Flag::Of)?);
    let (predicate, positive) = match cc {
        "o" => (get(Flag::Of)?, true),
        "no" => (get(Flag::Of)?, false),
        "b" => (get(Flag::Cf)?, true),
        "ae" => (get(Flag::Cf)?, false),
        "e" => (get(Flag::Zf)?, true),
        "ne" => (get(Flag::Zf)?, false),
        "be" => (bin(BinaryOp::Or, get(Flag::Cf)?, get(Flag::Zf)?)?, true),
        "a" => (bin(BinaryOp::Or, get(Flag::Cf)?, get(Flag::Zf)?)?, false),
        "s" => (get(Flag::Sf)?, true),
        "ns" => (get(Flag::Sf)?, false),
        "p" => (get(Flag::Pf)?, true),
        "np" => (get(Flag::Pf)?, false),
        "l" => (lt()?, true),
        "ge" => (lt()?, false),
        "le" => (bin(BinaryOp::Or, get(Flag::Zf)?, lt()?)?, true),
        "g" => (bin(BinaryOp::Or, get(Flag::Zf)?, lt()?)?, false),
        other => return Err(format!("unmodeled condition code `{other}`")),
    };
    Ok((predicate.simplify(), positive))
}

fn condition(cc: &str, flags: &BTreeMap<Flag, Expr>) -> Result<bool, String> {
    let get = |f: Flag| -> Result<u64, String> {
        let e = flags
            .get(&f)
            .ok_or_else(|| format!("branch `{cc}` reads flag {} that is not defined", f.name()))?;
        flag_value(e).ok_or_else(|| format!("branch `{cc}` {INPUT_DEPENDENT}"))
    };
    Ok(match cc {
        "o" => get(Flag::Of)? == 1,
        "no" => get(Flag::Of)? == 0,
        "b" => get(Flag::Cf)? == 1,
        "ae" => get(Flag::Cf)? == 0,
        "e" => get(Flag::Zf)? == 1,
        "ne" => get(Flag::Zf)? == 0,
        "be" => get(Flag::Cf)? == 1 || get(Flag::Zf)? == 1,
        "a" => get(Flag::Cf)? == 0 && get(Flag::Zf)? == 0,
        "s" => get(Flag::Sf)? == 1,
        "ns" => get(Flag::Sf)? == 0,
        "p" => get(Flag::Pf)? == 1,
        "np" => get(Flag::Pf)? == 0,
        "l" => get(Flag::Sf)? != get(Flag::Of)?,
        "ge" => get(Flag::Sf)? == get(Flag::Of)?,
        "le" => get(Flag::Zf)? == 1 || get(Flag::Sf)? != get(Flag::Of)?,
        "g" => get(Flag::Zf)? == 0 && get(Flag::Sf)? == get(Flag::Of)?,
        other => return Err(format!("unmodeled condition code `{other}`")),
    })
}

/// Executes the native CFG from its entry along the path its branches decide,
/// as expressions over the entry registers.
///
/// # Errors
/// An incomplete CFG, an unlifted block, a branch that depends on the inputs,
/// an unmodeled flag/condition, aliasing the memory model cannot rule out, an
/// unbalanced stack at the return, or exceeding `max_blocks`.
pub fn execute_native_function(cfg: &NativeCfg, max_blocks: usize) -> Result<NativeFunction, String> {
    execute_native_function_in(cfg, max_blocks, &NativeEnv::default())
}

/// Bounds on the original's path family. Exceeding any of them is an error:
/// a path is never guessed or dropped.
#[derive(Clone, Copy, Debug)]
pub struct ForkLimits {
    /// Input-dependent branches one path may take (0: none, the legacy mode).
    pub max_forks_per_path: usize,
    /// Leaf paths the function may have.
    pub max_paths: usize,
}

impl ForkLimits {
    pub const NONE: Self = Self { max_forks_per_path: 0, max_paths: 1 };
    pub const DEFAULT: Self = Self { max_forks_per_path: 10, max_paths: 32 };
}

/// One leaf path of the original: the input-dependent branch literals it took
/// (`predicate == polarity`, predicates are `W8` 0/1 bit-vector expressions over
/// the entry symbols) and the exit it reaches. The leaves of one function
/// partition the input space.
#[derive(Clone, Debug)]
pub struct NativePath {
    pub literals: Vec<(Expr, bool)>,
    pub function: NativeFunction,
}

impl NativePath {
    /// The conjunction of the path's literals as a `W8` 0/1 expression.
    ///
    /// # Errors
    /// An expression that cannot be rebuilt.
    pub fn condition(&self) -> Result<Expr, String> {
        let mut all = Expr::constant(BitWidth::W8, 1);
        for (predicate, polarity) in &self.literals {
            let one = Expr::constant(BitWidth::W8, 1);
            let lit = if *polarity {
                predicate.clone()
            } else {
                bin(BinaryOp::Xor, predicate.clone(), one)?
            };
            all = bin(BinaryOp::And, all, lit)?;
        }
        Ok(all)
    }
}

/// [`execute_native_function`] with the image environment that call blocks need.
///
/// A call is modeled as the Win64 ABI defines it and nothing more: the result
/// register(s) and the volatile registers are call-site-specific `fp_call_*`
/// functions of the eight argument registers (`rcx rdx r8 r9`, `xmm0-3` low
/// lanes); callee-saved registers survive; flags are undefined afterwards. The
/// callee's memory effects are *not* modeled, so a later load must hit read-only
/// image bytes, the caller's frame, or a store the function made itself.
///
/// # Errors
/// As [`execute_native_function`], plus a call block without the image base.
pub fn execute_native_function_in(
    cfg: &NativeCfg,
    max_blocks: usize,
    native_env: &NativeEnv,
) -> Result<NativeFunction, String> {
    let mut paths = execute_native_paths(cfg, max_blocks, native_env, ForkLimits::NONE)?;
    Ok(paths.remove(0).function)
}

/// The state of one path being executed.
#[derive(Clone)]
struct Run {
    calls: Vec<CollectedCall>,
    assumptions: Vec<String>,
    regs: BTreeMap<Gpr, Expr>,
    xmm: BTreeMap<u8, (Expr, Expr)>,
    flags: BTreeMap<Flag, Expr>,
    stores: Vec<Store>,
    literals: Vec<(Expr, bool)>,
    ip: u64,
    executed: usize,
    decided: usize,
}

/// [`execute_native_function_in`] that, at a branch the flags cannot decide, forks:
/// the taken arm and the fall-through arm are both executed, each carrying the
/// branch literal, so the leaves partition the inputs. A branch whose predicate
/// repeats a literal already on the path follows the one consistent arm.
/// `max_blocks` bounds the blocks executed over all paths.
///
/// # Errors
/// As [`execute_native_function_in`], plus exceeding `limits` (fork depth or
/// leaf count) on any path, which fails the whole function.
#[allow(clippy::too_many_lines)]
pub fn execute_native_paths(
    cfg: &NativeCfg,
    max_blocks: usize,
    native_env: &NativeEnv,
    limits: ForkLimits,
) -> Result<Vec<NativePath>, String> {
    let first = Run {
        calls: Vec::new(),
        assumptions: Vec::new(),
        regs: Gpr::ALL.iter().map(|g| (*g, entry_symbol(*g))).collect(),
        xmm: (0..16u8)
            .map(|n| (n, (xmm_entry_symbol(n, false), xmm_entry_symbol(n, true))))
            .collect(),
        flags: BTreeMap::new(),
        stores: Vec::new(),
        literals: Vec::new(),
        ip: cfg.entry,
        executed: 0,
        decided: 0,
    };
    let mut pending = vec![first];
    let mut leaves: Vec<NativePath> = Vec::new();
    let mut total_blocks = 0_usize;
    'paths: while let Some(mut run) = pending.pop() {
        loop {
            total_blocks += 1;
            if total_blocks > max_blocks {
                return Err(format!("did not return within {max_blocks} blocks"));
            }
            run.executed += 1;
            let ip = run.ip;
            let block = cfg
                .blocks
                .get(&ip)
                .ok_or_else(|| format!("block {ip:#x} is not in the native CFG"))?;
            let sem = block
                .semantics
                .as_ref()
                .ok_or_else(|| format!("block {ip:#x} was not lifted ({:?})", block.terminator))?;
            let mut env: BTreeMap<String, Expr> = Gpr::ALL
                .iter()
                .map(|g| (g.input_name(), run.regs[g].clone()))
                .collect();
            for (f, e) in &run.flags {
                env.insert(format!("{}_in", f.name()), e.clone());
            }
            for (n, (lo, hi)) in &run.xmm {
                env.insert(format!("xmm{n}_lo_in"), lo.clone());
                env.insert(format!("xmm{n}_hi_in"), hi.clone());
            }
            let after_call = (!run.calls.is_empty()).then_some(native_env);
            let stores = &run.stores;
            let subst = |e: &Expr| -> Result<Expr, String> {
                let substituted = e.substitute(&env).map_err(|x| x.to_string())?.simplify();
                if let Some(stray) = substituted
                    .dependencies()
                    .into_iter()
                    .find(|d| !d.starts_with("native_"))
                {
                    return Err(format!("block {ip:#x} reads `{stray}` before it is defined"));
                }
                resolve_loads(&substituted, stores, after_call)
            };
            let mut next_regs = BTreeMap::new();
            for (g, e) in &sem.regs {
                next_regs.insert(*g, subst(e)?);
            }
            let mut next_xmm = run.xmm.clone();
            for (n, (lo, hi)) in &sem.xmm {
                if sem.xmm_written.contains(n) {
                    next_xmm.insert(*n, (subst(lo)?, subst(hi)?));
                }
            }
            let mut next_flags = run.flags.clone();
            for (f, st) in &sem.flags {
                match st {
                    FlagState::Value(e) => {
                        next_flags.insert(*f, subst(e)?);
                    }
                    FlagState::Clobbered => {
                        next_flags.remove(f);
                    }
                }
            }
            let mut new_stores = Vec::new();
            for w in &sem.writes {
                let width = match w.width {
                    vmpx_disasm::Width::W8 => BitWidth::W8,
                    vmpx_disasm::Width::W16 => BitWidth::W16,
                    vmpx_disasm::Width::W32 => BitWidth::W32,
                    vmpx_disasm::Width::W64 => BitWidth::W64,
                };
                new_stores.push((subst(&w.address)?, subst(&w.value)?, width));
            }
            run.stores.extend(new_stores);
            run.regs = next_regs;
            run.xmm = next_xmm;
            run.flags = next_flags;
            match &block.terminator {
                Terminator::FallThrough(t) | Terminator::Jump(t) => run.ip = *t,
                Terminator::CondJump { cc, taken, not_taken } => {
                    let decision = match condition(cc, &run.flags) {
                        Ok(b) => Some(b),
                        Err(e) if limits.max_forks_per_path > 0 && e.ends_with(INPUT_DEPENDENT) => None,
                        Err(e) => return Err(e),
                    };
                    if let Some(b) = decision {
                        run.ip = if b { *taken } else { *not_taken };
                        run.decided += 1;
                        continue;
                    }
                    let (predicate, polarity) = branch_literal(cc, &run.flags)?;
                    let held = match predicate.kind {
                        ExprKind::Const { value } => Some(value & 1 == 1),
                        _ => run.literals.iter().find(|(p, _)| *p == predicate).map(|(_, h)| *h),
                    };
                    if let Some(held) = held {
                        run.ip = if held == polarity { *taken } else { *not_taken };
                        run.decided += 1;
                        continue;
                    }
                    if run.literals.len() >= limits.max_forks_per_path {
                        return Err(format!(
                            "branch `{cc}` at block {ip:#x} exceeds the fork depth {}",
                            limits.max_forks_per_path
                        ));
                    }
                    if leaves.len() + pending.len() + 2 > limits.max_paths {
                        return Err(format!("the function has more than {} paths", limits.max_paths));
                    }
                    let mut other = run.clone();
                    other.literals.push((predicate.clone(), !polarity));
                    other.ip = *not_taken;
                    pending.push(other);
                    run.literals.push((predicate, polarity));
                    run.ip = *taken;
                }
                Terminator::Call { target, next } => {
                    if native_env.image_base == 0 {
                        return Err(format!("block {ip:#x} calls, but the image base is unknown"));
                    }
                    let (label, iat_rva) = match *target {
                        CallTarget::Import(slot) => {
                            let rva = u32::try_from(slot.wrapping_sub(native_env.image_base))
                                .map_err(|_| format!("import slot {slot:#x} outside the image"))?;
                            (format!("iat_{rva:#x}"), rva)
                        }
                        CallTarget::Direct(va) => {
                            let rva = u32::try_from(va.wrapping_sub(native_env.image_base))
                                .map_err(|_| format!("call target {va:#x} outside the image"))?;
                            (format!("sub_{rva:#x}"), rva | 0x8000_0000)
                        }
                    };
                    apply_call(&mut run.regs, &mut run.xmm, &mut run.flags, &mut run.calls, &label, iat_rva);
                    if run.calls.len() == 1 {
                        run.assumptions.push(
                            "callee memory effects are not modeled; later loads are limited to read-only image bytes, the caller frame and the function's own stores".to_owned(),
                        );
                        run.assumptions.push(
                            "no pointer into the function's frame or the caller frame escapes to a callee".to_owned(),
                        );
                    }
                    run.ip = *next;
                }
                Terminator::Return => {
                    let rsp = run.regs[&Gpr::Rsp].simplify();
                    if rsp != entry_symbol(Gpr::Rsp) {
                        return Err(format!("stack is unbalanced at the return: rsp = {rsp:?}"));
                    }
                    let function = finish(
                        &run.regs,
                        run.xmm,
                        run.stores,
                        run.calls,
                        run.assumptions,
                        run.executed,
                        run.decided,
                    );
                    leaves.push(NativePath { literals: run.literals, function });
                    continue 'paths;
                }
                Terminator::Unsupported(why) => return Err(format!("block {ip:#x}: {why}")),
            }
        }
    }
    Ok(leaves)
}

fn finish(
    regs: &BTreeMap<Gpr, Expr>,
    xmm: BTreeMap<u8, (Expr, Expr)>,
    stores: Vec<Store>,
    calls: Vec<CollectedCall>,
    assumptions: Vec<String>,
    blocks: usize,
    decided: usize,
) -> NativeFunction {
    let rsp = entry_symbol(Gpr::Rsp);
    let mut kept = Vec::new();
    let mut dropped = 0;
    for (address, value, width) in stores {
        let (base, offset) = split_offset(&address);
        if base == rsp && offset < 0 {
            dropped += 1;
        } else {
            kept.push(CollectedStore { address, value, width });
        }
    }
    NativeFunction {
        semantics: ExitSemantics {
            exit: NativeExitState {
                xmm,
                schema_version: 1,
                registers: X64NativeRegister::ALL
                    .iter()
                    .map(|r| (*r, regs[&native_register(*r)].clone()))
                    .collect(),
                rflags: Expr::constant(BitWidth::W64, 0),
                return_address: Expr::constant(BitWidth::W64, 0),
                frame_size: 0,
            },
            effects: CollectedEffects {
                stores: kept,
                calls,
                unsupported: Vec::new(),
            },
        },
        blocks_executed: blocks,
        branches_decided: decided,
        frame_stores_dropped: dropped,
        assumptions,
    }
}

/// The argument registers a Win64 callee may read: `rcx rdx r8 r9`, `xmm0-3` low lanes.
fn call_arguments(regs: &BTreeMap<Gpr, Expr>, xmm: &BTreeMap<u8, (Expr, Expr)>) -> Vec<Expr> {
    let mut args: Vec<Expr> = [Gpr::Rcx, Gpr::Rdx, Gpr::R8, Gpr::R9]
        .iter()
        .map(|g| regs[g].simplify())
        .collect();
    args.extend((0..4u8).map(|n| xmm[&n].0.simplify()));
    args
}

fn opaque(name: String, args: &[Expr]) -> Expr {
    Expr::intrinsic(BitWidth::W64, name, args.to_vec())
}

/// Applies the Win64 effect of one call site to the register state.
fn apply_call(
    regs: &mut BTreeMap<Gpr, Expr>,
    xmm: &mut BTreeMap<u8, (Expr, Expr)>,
    flags: &mut BTreeMap<Flag, Expr>,
    calls: &mut Vec<CollectedCall>,
    label: &str,
    iat_rva: u32,
) {
    let args = call_arguments(regs, xmm);
    let site = format!("{label}_c{}", calls.len());
    let f64_result = format!("fp_call_{site}_f64");
    calls.push(CollectedCall {
        result_intrinsic: f64_result.clone(),
        iat_rva,
        arguments: args.clone(),
    });
    regs.insert(Gpr::Rax, opaque(format!("fp_call_{site}_rax_f64"), &args));
    for (g, tag) in [
        (Gpr::Rcx, "rcx"),
        (Gpr::Rdx, "rdx"),
        (Gpr::R8, "r8"),
        (Gpr::R9, "r9"),
        (Gpr::R10, "r10"),
        (Gpr::R11, "r11"),
    ] {
        regs.insert(g, opaque(format!("fp_call_{site}_{tag}_f64"), &args));
    }
    xmm.insert(
        0,
        (
            opaque(f64_result, &args),
            opaque(format!("fp_call_{site}_xmm0hi_f64"), &args),
        ),
    );
    for n in 1..6u8 {
        xmm.insert(
            n,
            (
                opaque(format!("fp_call_{site}_xmm{n}lo_f64"), &args),
                opaque(format!("fp_call_{site}_xmm{n}hi_f64"), &args),
            ),
        );
    }
    flags.clear();
}

/// Bit position of a flag in RFLAGS.
const fn flag_bit(f: Flag) -> u32 {
    match f {
        Flag::Cf => 0,
        Flag::Pf => 2,
        Flag::Af => 4,
        Flag::Zf => 6,
        Flag::Sf => 7,
        Flag::Of => 11,
    }
}

fn c64(v: u64) -> Expr {
    Expr::constant(BitWidth::W64, v)
}

fn bin(op: BinaryOp, a: Expr, b: Expr) -> Result<Expr, String> {
    Expr::binary(op, a, b).map_err(|e| e.to_string())
}

/// The 0/1 value of RFLAGS bit `bit` as a `W8` expression.
fn flag_in(rflags: &Expr, bit: u32) -> Result<Expr, String> {
    let shifted = bin(BinaryOp::Shr, rflags.clone(), c64(u64::from(bit)))?;
    let byte = Expr::resize(vmpx_vm::ResizeOp::Truncate, BitWidth::W8, shifted)
        .map_err(|e| e.to_string())?;
    bin(BinaryOp::And, byte, Expr::constant(BitWidth::W8, 1))
}

/// Applies a straight-line native segment (lifted over `<reg>_in` / `xmm<n>_*_in` /
/// `<flag>_in` inputs) to the state `prev` it starts from. The segment must leave
/// the stack exactly as it found it and write no memory, and may not clobber a
/// flag the next VM region could read: anything else is an error, never a guess.
///
/// # Errors
/// An unbalanced stack, a memory write, a clobbered flag, or an expression that
/// cannot be rebuilt.
pub fn apply_native_segment(
    prev: &ExitSemantics,
    seg: &vmpx_disasm::BlockSemantics<Expr>,
) -> Result<ExitSemantics, String> {
    if !seg.writes.is_empty() {
        return Err("native segment writes memory; its ordering against the VM frames is not modeled".to_owned());
    }
    let mut env: BTreeMap<String, Expr> = BTreeMap::new();
    for r in X64NativeRegister::ALL {
        let v = prev
            .exit
            .registers
            .get(&r)
            .ok_or_else(|| format!("register {r:?} is not tracked before the native segment"))?;
        env.insert(native_register(r).input_name(), v.clone());
    }
    env.insert(Gpr::Rsp.input_name(), entry_symbol(Gpr::Rsp));
    for n in 0..16u8 {
        let (lo, hi) = prev
            .exit
            .xmm
            .get(&n)
            .cloned()
            .unwrap_or_else(|| (xmm_entry_symbol(n, false), xmm_entry_symbol(n, true)));
        env.insert(format!("xmm{n}_lo_in"), lo);
        env.insert(format!("xmm{n}_hi_in"), hi);
    }
    for f in [Flag::Cf, Flag::Pf, Flag::Af, Flag::Zf, Flag::Sf, Flag::Of] {
        env.insert(format!("{}_in", f.name()), flag_in(&prev.exit.rflags, flag_bit(f))?);
    }
    let subst = |e: &Expr| e.substitute(&env).map(|x| x.simplify()).map_err(|x| x.to_string());
    let rsp = subst(&seg.regs[&Gpr::Rsp])?;
    if rsp != entry_symbol(Gpr::Rsp) {
        return Err(format!("native segment changes the stack pointer: rsp = {rsp:?}"));
    }
    let mut out = prev.clone();
    for r in X64NativeRegister::ALL {
        out.exit.registers.insert(r, subst(&seg.regs[&native_register(r)])?);
    }
    let mut xmm = prev.exit.xmm.clone();
    if xmm.is_empty() {
        xmm = (0..16u8)
            .map(|n| (n, (xmm_entry_symbol(n, false), xmm_entry_symbol(n, true))))
            .collect();
    }
    for (n, (lo, hi)) in &seg.xmm {
        if seg.xmm_written.contains(n) {
            xmm.insert(*n, (subst(lo)?, subst(hi)?));
        }
    }
    out.exit.xmm = xmm;
    if !seg.flags.is_empty() {
        let mut mask = 0u64;
        let mut set: Option<Expr> = None;
        for (f, st) in &seg.flags {
            let FlagState::Value(v) = st else {
                return Err(format!("native segment clobbers flag {}", f.name()));
            };
            mask |= 1 << flag_bit(*f);
            let wide = Expr::resize(vmpx_vm::ResizeOp::ZeroExtend, BitWidth::W64, subst(v)?)
                .map_err(|e| e.to_string())?;
            let placed = bin(BinaryOp::Shl, wide, c64(u64::from(flag_bit(*f))))?;
            set = Some(match set {
                Some(acc) => bin(BinaryOp::Or, acc, placed)?,
                None => placed,
            });
        }
        let kept = bin(BinaryOp::And, prev.exit.rflags.clone(), c64(!mask))?;
        out.exit.rflags = match set {
            Some(bits) => bin(BinaryOp::Or, kept, bits)?.simplify(),
            None => kept.simplify(),
        };
    }
    Ok(out)
}

/// Feeds the state `prev` into a prepared VM region `next` whose exit is expressed
/// over its own entry symbols (`native_<reg>_initial`, `native_rflags_initial`,
/// `native_xmm<n>_*_initial`), yielding the composite exit.
///
/// # Errors
/// `next` stores to memory, or an expression cannot be rebuilt.
pub fn compose_into_next(prev: &ExitSemantics, next: &ExitSemantics) -> Result<ExitSemantics, String> {
    if !next.effects.stores.is_empty() || !next.effects.calls.is_empty() {
        return Err("a chained VM region has stores or calls; composition is not modeled".to_owned());
    }
    let mut env: BTreeMap<String, Expr> = BTreeMap::new();
    for r in X64NativeRegister::ALL {
        let v = prev
            .exit
            .registers
            .get(&r)
            .ok_or_else(|| format!("register {r:?} is not tracked at the chain point"))?;
        env.insert(entry_name(native_register(r)), v.clone());
    }
    env.insert("native_rflags_initial".to_owned(), prev.exit.rflags.clone());
    for n in 0..16u8 {
        let (lo, hi) = prev
            .exit
            .xmm
            .get(&n)
            .cloned()
            .unwrap_or_else(|| (xmm_entry_symbol(n, false), xmm_entry_symbol(n, true)));
        env.insert(xmm_entry_name(n, false), lo);
        env.insert(xmm_entry_name(n, true), hi);
    }
    let subst = |e: &Expr| e.substitute(&env).map(|x| x.simplify()).map_err(|x| x.to_string());
    let mut out = next.clone();
    for v in out.exit.registers.values_mut() {
        *v = subst(v)?;
    }
    out.exit.rflags = subst(&out.exit.rflags)?;
    if out.exit.xmm.is_empty() {
        out.exit.xmm = (0..16u8)
            .map(|n| (n, (xmm_entry_symbol(n, false), xmm_entry_symbol(n, true))))
            .collect();
    }
    for (lo, hi) in out.exit.xmm.values_mut() {
        *lo = subst(lo)?;
        *hi = subst(hi)?;
    }
    Ok(out)
}

/// The observable exit of the two semantics is expression-for-expression equal.
#[must_use]
pub fn same_exit(a: &ExitSemantics, b: &ExitSemantics) -> bool {
    let norm = |s: &ExitSemantics| -> Vec<Expr> {
        let mut v: Vec<Expr> = s.exit.registers.values().map(Expr::simplify).collect();
        for (lo, hi) in s.exit.xmm.values() {
            v.push(lo.simplify());
            v.push(hi.simplify());
        }
        v.push(s.exit.rflags.simplify());
        v
    };
    a.exit.registers.keys().eq(b.exit.registers.keys()) && norm(a) == norm(b)
}

/// Which exit registers the calling convention makes observable.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AbiPolicy {
    pub compared: Vec<X64NativeRegister>,
    /// XMM registers whose lanes are compared (`(register, high lane)`).
    pub compared_xmm: Vec<(u8, bool)>,
}

impl AbiPolicy {
    /// Win64: the return value (`rax`) and the callee-saved registers
    /// (`rbx rbp rsi rdi r12-r15`) must agree; volatile registers (`rcx rdx
    /// r8-r11`), flags and the return-address slot are not part of the contract.
    /// FP: `xmm0` (the return value, low lane) and the callee-saved `xmm6-15` (both
    /// lanes); `xmm1-5` are volatile.
    #[must_use]
    pub fn win64() -> Self {
        use X64NativeRegister as R;
        let mut compared_xmm = vec![(0, false)];
        for n in 6..16u8 {
            compared_xmm.push((n, false));
            compared_xmm.push((n, true));
        }
        Self {
            compared: vec![R::Rax, R::Rbx, R::Rbp, R::Rsi, R::Rdi, R::R12, R::R13, R::R14, R::R15],
            compared_xmm,
        }
    }
}

/// Builds the gate obligation for one recovered path against the original.
/// Stores and calls compare in full; registers per `policy`.
#[must_use]
pub fn differential_obligation(
    recovered: &ExitSemantics,
    original: &ExitSemantics,
    policy: &AbiPolicy,
) -> Obligation {
    // A side that tracks no XMM state leaves every XMM register at its entry value.
    let identity = |s: &ExitSemantics| -> ExitSemantics {
        let mut s = s.clone();
        if s.exit.xmm.is_empty() {
            s.exit.xmm = (0..16u8)
                .map(|n| (n, (xmm_entry_symbol(n, false), xmm_entry_symbol(n, true))))
                .collect();
        }
        s
    };
    let mut o = Obligation::from_exits(&identity(recovered), &identity(original));
    o.cells.retain(|c| match &c.id {
        CellId::Register(r) => policy.compared.contains(r),
        CellId::Xmm { reg, hi } => policy.compared_xmm.contains(&(*reg, *hi)),
        CellId::Rflags | CellId::ReturnAddress => false,
        _ => true,
    });
    o
}

/// Rewrites a recovered expression into the forms the original uses: a double
/// shift of a value by itself is a rotate (32/64-bit only, where every masked
/// count is architecturally defined).
///
/// # Errors
/// An expression the rewrite cannot rebuild.
pub fn canonicalize_recovered(e: &Expr) -> Result<Expr, String> {
    e.rewrite_intrinsics(&mut |width, name, arguments| {
        let (op, rest) = if let Some(r) = name.strip_prefix("x86_double_shift_left_w") {
            (BinaryOp::Rol, r)
        } else if let Some(r) = name.strip_prefix("x86_double_shift_right_w") {
            (BinaryOp::Ror, r)
        } else {
            return None;
        };
        if !(rest.starts_with("32_") || rest.starts_with("64_")) {
            return None;
        }
        let [destination, source, count] = arguments else {
            return None;
        };
        if destination.width != width
            || source.width != width
            || count.width != BitWidth::W8
            || destination.simplify() != source.simplify()
        {
            return None;
        }
        let count = Expr::resize(vmpx_vm::ResizeOp::ZeroExtend, width, count.clone()).ok()?;
        Expr::binary(op, destination.clone(), count).ok()
    })
    .map_err(|x| x.to_string())
}

/// What `prepare_recovered` did to a recovered path, for the report.
#[derive(Clone, Debug, Default)]
pub struct PrepareNotes {
    pub frame_stores_dropped: usize,
    pub reads_resolved: usize,
}

fn prepare_expr(e: &Expr, resolved: &mut usize) -> Result<Expr, String> {
    let rewritten = e
        .rewrite_intrinsics(&mut |width, name, arguments| {
            name.strip_prefix("memory_load_epoch_")?.parse::<u64>().ok()?;
            let [address] = arguments else { return None };
            *resolved += 1;
            Some(Expr::load(width, address.clone()))
        })
        .map_err(|x| x.to_string())?;
    canonicalize_recovered(&rewritten)
}

/// [`prepare_recovered`]'s rewrite of one expression (a recovered branch selector).
///
/// # Errors
/// An expression the rewrite cannot rebuild.
pub fn prepare_condition(e: &Expr) -> Result<Expr, String> {
    prepare_expr(e, &mut 0)
}

/// Brings a recovered path into the original's vocabulary:
/// stores into the VM's own dead frame (`vsp_initial`-relative, below the entry
/// stack pointer) are dropped, on the labeled assumption that no caller pointer
/// aliases the callee's frame; `memory_load_epoch_<n>` reads become loads of
/// initial memory; double shifts of a value by itself become rotates. Nothing
/// observable may still mention the VM stack root, and a remaining store makes
/// load/store ordering unmodeled.
///
/// # Errors
/// A remaining store, or a value that depends on the VM stack root.
pub fn prepare_recovered(
    semantics: &ExitSemantics,
    store_epochs: &[u64],
    entry_sp_offset: i64,
) -> Result<(ExitSemantics, PrepareNotes), String> {
    let root = Expr::register(BitWidth::W64, "vsp_initial");
    let mut notes = PrepareNotes::default();
    for (store, _) in semantics.effects.stores.iter().zip(store_epochs) {
        let (base, offset) = split_offset(&store.address);
        if base == root && offset + i64::from(store.width.bytes()) <= entry_sp_offset {
            notes.frame_stores_dropped += 1;
        } else {
            return Err(
                "recovered function stores outside its frame; load/store ordering is not modeled"
                    .to_owned(),
            );
        }
    }
    let mut resolved = 0_usize;
    let mut fix = |e: &Expr| prepare_expr(e, &mut resolved);
    let mut out = semantics.clone();
    for v in out.exit.registers.values_mut() {
        *v = fix(v)?;
    }
    // The recovered side models only the integer VM; handlers fail closed on any
    // XMM-touching instruction, so an exit that tracks no XMM state leaves every
    // XMM register at its entry value.
    if out.exit.xmm.is_empty() {
        out.exit.xmm = (0..16u8)
            .map(|n| (n, (xmm_entry_symbol(n, false), xmm_entry_symbol(n, true))))
            .collect();
    }
    for (lo, hi) in out.exit.xmm.values_mut() {
        *lo = fix(lo)?;
        *hi = fix(hi)?;
    }
    out.effects.stores.clear();
    if out.exit.registers.values().any(|e| e.dependencies().contains("vsp_initial")) {
        return Err("an observable value depends on the VM stack root".to_owned());
    }
    notes.reads_resolved = resolved;
    Ok((out, notes))
}

/// Verdict of one function's differential.
#[derive(Clone, Debug)]
pub struct DiffOutcome {
    pub verdict: EquivVerdict,
    /// Why the verdict is not PROVEN (empty when proven).
    pub reason: String,
    pub report: Option<FunctionReport>,
    pub paths_compared: usize,
    /// Gate-failure arms that return through staged stack, not compared.
    pub region_paths: usize,
    /// Arms pruned as retry/spin on a failed environment probe.
    pub spin_pruned: usize,
    pub frame_stores_dropped: usize,
    /// Structural fidelity of a PROVEN function (descriptive, not a proof).
    pub fidelity: Option<crate::fidelity::FunctionFidelity>,
}

impl DiffOutcome {
    pub(crate) fn unproven(reason: impl Into<String>) -> Self {
        Self {
            verdict: EquivVerdict::Timeout,
            reason: reason.into(),
            report: None,
            paths_compared: 0,
            region_paths: 0,
            spin_pruned: 0,
            frame_stores_dropped: 0,
            fidelity: None,
        }
    }

    /// Environment-gate arms (region exits, pruned spins) were excluded from a
    /// PROVEN verdict.
    #[must_use]
    pub fn modulo_env_gates(&self) -> bool {
        self.region_paths + self.spin_pruned > 0
    }
}

/// Compares every callee-return path of the recovered function against the
/// original under `policy`. A path that cannot be brought into the original's
/// vocabulary, or an empty family, is unproven. With `strict_gates`, excluded
/// environment-gate arms also make the verdict unproven.
#[must_use]
pub fn diff_against_original(
    original: &ExitSemantics,
    recovered: &vmpx_vmlift::UnrolledFunction,
    verifier: &Verifier,
    policy: &AbiPolicy,
    strict_gates: bool,
) -> DiffOutcome {
    let mut obligations = Vec::new();
    let mut region = 0;
    let mut dropped = 0;
    let mut why = Vec::new();
    for (index, path) in recovered.paths.iter().enumerate() {
        if path.region_exit {
            region += 1;
            continue;
        }
        match prepare_recovered(&path.semantics, &path.store_epochs, recovered.entry_sp_offset) {
            Ok((semantics, notes)) => {
                dropped += notes.frame_stores_dropped;
                obligations.push(differential_obligation(&semantics, original, policy));
            }
            Err(reason) => why.push(format!("path {index}: {reason}")),
        }
    }
    let mut out = DiffOutcome::unproven("");
    out.region_paths = region;
    out.spin_pruned = recovered.spin_arms_pruned;
    out.frame_stores_dropped = dropped;
    if !why.is_empty() {
        out.reason = why.join("; ");
        return out;
    }
    if obligations.is_empty() {
        out.reason = "no callee-return path was recovered".to_owned();
        return out;
    }
    if std::env::var_os("VMPX_DIFF_DUMP").is_some() {
        for (i, o) in obligations.iter().enumerate() {
            for c in o.cells.iter().filter(|c| c.id == CellId::Register(X64NativeRegister::Rax)) {
                eprintln!(
                    "[dump path {i}] nodes recovered={} original={}\n  recovered: {}\n  original:  {}",
                    c.vm.node_count_bounded(1_000_000),
                    c.recovered.node_count_bounded(1_000_000),
                    pretty(&c.vm.simplify(), 9),
                    pretty(&c.recovered.simplify(), 9)
                );
            }
        }
    }
    conclude(out, &obligations, verifier, strict_gates, true)
}

/// Runs the gate over the path obligations and folds the result into `out`.
pub(crate) fn conclude(
    mut out: DiffOutcome,
    obligations: &[Obligation],
    verifier: &Verifier,
    strict_gates: bool,
    with_fidelity: bool,
) -> DiffOutcome {
    let report = verifier.check_function(obligations, true);
    out.paths_compared = obligations.len();
    out.verdict = report.verdict.clone();
    if report.verdict == EquivVerdict::Proven && strict_gates && out.modulo_env_gates() {
        out.verdict = EquivVerdict::Timeout;
        out.reason = format!(
            "strict gates: {} gate-failure arm(s) and {} spin arm(s) are not modeled",
            out.region_paths, out.spin_pruned
        );
    } else if report.verdict == EquivVerdict::Timeout {
        out.reason = classify(&report);
    }
    if with_fidelity && out.verdict == EquivVerdict::Proven {
        out.fidelity = crate::fidelity::assess(obligations, &report);
    }
    out.report = Some(report);
    out
}

fn classify(report: &FunctionReport) -> String {
    for (i, path) in report.paths.iter().enumerate() {
        if let Some(gap) = path.gaps.first() {
            return format!("path {i}: modeling gap: {gap}");
        }
        for (id, verdict) in &path.cells {
            if let CellVerdict::Unproven(u) = verdict {
                return format!("path {i}: cell {id:?} unproven: {u:?}");
            }
        }
    }
    "no paths".to_owned()
}

/// Depth-limited, human-readable rendering of an expression (shared subtrees
/// are printed in full, so keep `depth` small).
#[must_use]
pub fn pretty(e: &Expr, depth: u32) -> String {
    fn go(e: &Expr, d: u32, out: &mut String) {
        if d == 0 {
            out.push('~');
            return;
        }
        match &e.kind {
            ExprKind::Const { value } => out.push_str(&format!("{value:#x}")),
            ExprKind::Register { name } => out.push_str(name),
            ExprKind::Unary { op, value } => {
                out.push_str(&format!("{op:?}("));
                go(value, d - 1, out);
                out.push(')');
            }
            ExprKind::Binary { op, left, right } => {
                out.push_str(&format!("{op:?}{}(", e.width.bits()));
                go(left, d - 1, out);
                out.push(',');
                go(right, d - 1, out);
                out.push(')');
            }
            ExprKind::Load { address } => {
                out.push_str(&format!("Load{}(", e.width.bits()));
                go(address, d - 1, out);
                out.push(')');
            }
            ExprKind::Intrinsic { name, arguments } => {
                out.push_str(name);
                out.push('(');
                for a in arguments {
                    go(a, d - 1, out);
                    out.push(',');
                }
                out.push(')');
            }
            ExprKind::Extract { value, lsb } => {
                out.push_str(&format!("Ext{}@{lsb}(", e.width.bits()));
                go(value, d - 1, out);
                out.push(')');
            }
            ExprKind::Concat { high, low } => {
                out.push_str("Cat(");
                go(high, d - 1, out);
                out.push(',');
                go(low, d - 1, out);
                out.push(')');
            }
            ExprKind::Resize { op, value } => {
                out.push_str(&format!("{op:?}{}(", e.width.bits()));
                go(value, d - 1, out);
                out.push(')');
            }
        }
    }
    let mut out = String::new();
    go(e, depth, &mut out);
    out
}
