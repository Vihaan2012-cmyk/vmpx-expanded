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
use vmpx_mutation::cfg_recovery::{NativeCfg, Terminator};
use vmpx_vm::{BinaryOp, BitWidth, Expr, ExprKind, NativeExitState, X64NativeRegister};
use vmpx_vmlift::{CollectedEffects, CollectedStore, ExitSemantics};

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
fn resolve_loads(e: &Expr, stores: &[Store]) -> Result<Expr, String> {
    if stores.is_empty() {
        return Ok(e.clone());
    }
    let mut memo: HashMap<Expr, Expr> = HashMap::new();
    rebuild(e, stores, &mut memo)
}

fn rebuild(e: &Expr, stores: &[Store], memo: &mut HashMap<Expr, Expr>) -> Result<Expr, String> {
    if let Some(done) = memo.get(e) {
        return Ok(done.clone());
    }
    let err = |x: vmpx_vm::ExprError| x.to_string();
    let out = match &e.kind {
        ExprKind::Const { .. } | ExprKind::Register { .. } => e.clone(),
        ExprKind::Unary { op, value } => Expr::unary(*op, rebuild(value, stores, memo)?),
        ExprKind::Binary { op, left, right } => {
            Expr::binary(*op, rebuild(left, stores, memo)?, rebuild(right, stores, memo)?).map_err(err)?
        }
        ExprKind::Extract { value, lsb } => {
            Expr::extract(e.width, *lsb, rebuild(value, stores, memo)?).map_err(err)?
        }
        ExprKind::Concat { high, low } => {
            Expr::concat(rebuild(high, stores, memo)?, rebuild(low, stores, memo)?).map_err(err)?
        }
        ExprKind::Resize { op, value } => {
            Expr::resize(*op, e.width, rebuild(value, stores, memo)?).map_err(err)?
        }
        ExprKind::Intrinsic { name, arguments } => Expr::intrinsic(
            e.width,
            name.clone(),
            arguments
                .iter()
                .map(|a| rebuild(a, stores, memo))
                .collect::<Result<_, _>>()?,
        ),
        ExprKind::Load { address } => {
            let address = rebuild(address, stores, memo)?;
            load_against(e.width, &address, stores)?
        }
    };
    memo.insert(e.clone(), out.clone());
    Ok(out)
}

fn load_against(width: BitWidth, address: &Expr, stores: &[Store]) -> Result<Expr, String> {
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
    Ok(Expr::load(width, address.clone()))
}

/// The original function as executed along its input-independent path.
#[derive(Clone, Debug)]
pub struct NativeFunction {
    pub semantics: ExitSemantics,
    pub blocks_executed: usize,
    pub branches_decided: usize,
    /// Stores below the entry stack pointer (the callee's own dead frame).
    pub frame_stores_dropped: usize,
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

fn condition(cc: &str, flags: &BTreeMap<Flag, Expr>) -> Result<bool, String> {
    let get = |f: Flag| -> Result<u64, String> {
        let e = flags
            .get(&f)
            .ok_or_else(|| format!("branch `{cc}` reads flag {} that is not defined", f.name()))?;
        flag_value(e).ok_or_else(|| format!("branch `{cc}` depends on the function's inputs"))
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
    let mut regs: BTreeMap<Gpr, Expr> = Gpr::ALL.iter().map(|g| (*g, entry_symbol(*g))).collect();
    let mut flags: BTreeMap<Flag, Expr> = BTreeMap::new();
    let mut stores: Vec<Store> = Vec::new();
    let mut ip = cfg.entry;
    let mut decided = 0;
    for executed in 1..=max_blocks {
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
            .map(|g| (g.input_name(), regs[g].clone()))
            .collect();
        for (f, e) in &flags {
            env.insert(format!("{}_in", f.name()), e.clone());
        }
        let subst = |e: &Expr| -> Result<Expr, String> {
            let substituted = e.substitute(&env).map_err(|x| x.to_string())?.simplify();
            if let Some(stray) = substituted
                .dependencies()
                .into_iter()
                .find(|d| !d.starts_with("native_"))
            {
                return Err(format!("block {ip:#x} reads `{stray}` before it is defined"));
            }
            resolve_loads(&substituted, &stores)
        };
        let mut next_regs = BTreeMap::new();
        for (g, e) in &sem.regs {
            next_regs.insert(*g, subst(e)?);
        }
        let mut next_flags = flags.clone();
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
        stores.extend(new_stores);
        regs = next_regs;
        flags = next_flags;
        match &block.terminator {
            Terminator::FallThrough(t) | Terminator::Jump(t) => ip = *t,
            Terminator::CondJump { cc, taken, not_taken } => {
                ip = if condition(cc, &flags)? { *taken } else { *not_taken };
                decided += 1;
            }
            Terminator::Return => {
                let rsp = regs[&Gpr::Rsp].simplify();
                if rsp != entry_symbol(Gpr::Rsp) {
                    return Err(format!("stack is unbalanced at the return: rsp = {rsp:?}"));
                }
                return Ok(finish(&regs, stores, executed, decided));
            }
            Terminator::Unsupported(why) => return Err(format!("block {ip:#x}: {why}")),
        }
    }
    Err(format!("did not return within {max_blocks} blocks"))
}

fn finish(regs: &BTreeMap<Gpr, Expr>, stores: Vec<Store>, blocks: usize, decided: usize) -> NativeFunction {
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
                calls: Vec::new(),
                unsupported: Vec::new(),
            },
        },
        blocks_executed: blocks,
        branches_decided: decided,
        frame_stores_dropped: dropped,
    }
}

/// Which exit registers the calling convention makes observable.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AbiPolicy {
    pub compared: Vec<X64NativeRegister>,
}

impl AbiPolicy {
    /// Win64: the return value (`rax`) and the callee-saved registers
    /// (`rbx rbp rsi rdi r12-r15`) must agree; volatile registers (`rcx rdx
    /// r8-r11`), flags and the return-address slot are not part of the contract.
    #[must_use]
    pub fn win64() -> Self {
        use X64NativeRegister as R;
        Self {
            compared: vec![R::Rax, R::Rbx, R::Rbp, R::Rsi, R::Rdi, R::R12, R::R13, R::R14, R::R15],
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
    let mut o = Obligation::from_exits(recovered, original);
    o.cells.retain(|c| match &c.id {
        CellId::Register(r) => policy.compared.contains(r),
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
    let mut fix = |e: &Expr| -> Result<Expr, String> {
        let rewritten = e
            .rewrite_intrinsics(&mut |width, name, arguments| {
                name.strip_prefix("memory_load_epoch_")?.parse::<u64>().ok()?;
                let [address] = arguments else { return None };
                resolved += 1;
                Some(Expr::load(width, address.clone()))
            })
            .map_err(|x| x.to_string())?;
        canonicalize_recovered(&rewritten)
    };
    let mut out = semantics.clone();
    for v in out.exit.registers.values_mut() {
        *v = fix(v)?;
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
}

impl DiffOutcome {
    fn timeout(reason: impl Into<String>) -> Self {
        Self {
            verdict: EquivVerdict::Timeout,
            reason: reason.into(),
            report: None,
            paths_compared: 0,
            region_paths: 0,
            spin_pruned: 0,
            frame_stores_dropped: 0,
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
    let mut out = DiffOutcome::timeout("");
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
    let report = verifier.check_function(&obligations, true);
    out.paths_compared = obligations.len();
    out.verdict = report.verdict.clone();
    if report.verdict == EquivVerdict::Proven && strict_gates && out.modulo_env_gates() {
        out.verdict = EquivVerdict::Timeout;
        out.reason = format!(
            "strict gates: {} gate-failure arm(s) and {} spin arm(s) are not modeled",
            region, recovered.spin_arms_pruned
        );
    } else if report.verdict == EquivVerdict::Timeout {
        out.reason = classify(&report);
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
