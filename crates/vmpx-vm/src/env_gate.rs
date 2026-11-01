//! Generalized environment-gate fork.
//!
//! A VM branch whose condition is rooted in a host-environment fingerprint (a
//! CPUID word, an API return such as a volume serial / device id / registry
//! value, a PEB/TEB load, a constant out-of-image probe) cannot be resolved from
//! the program's own data flow. Such gates are compare-then-branch, not
//! decryption keys, so the code behind BOTH arms is plaintext and can be
//! recovered without the real environment by forking both arms.
//!
//! The fork is a conservative superset. Every environment source is replaced by
//! an unconstrained free variable (its arguments, which may carry program
//! state, are dropped), and the branch is resolved over the proven input-domain /
//! flag-fork machinery. A target set is returned only when that set is finite and
//! exact for ALL values of the environment variables; a target that depends on an
//! environment value at full width (an unbounded computed target) is refused and
//! stays dynamic-only. No environment value is ever assumed or fabricated.

use std::cell::RefCell;
use std::collections::{BTreeSet, HashSet};
use std::hash::{Hash, Hasher};

use serde::Serialize;
use vmpx_pe::Va;

use crate::cfg::{
    VmCfgLimits, bind_preferred_base_registers_except, diagnostics_enabled, is_flags_intrinsic,
    resolve_over_inputs_capped, substitute_flags,
};
use crate::{BinaryOp, BitWidth, Expr, ExprError, ExprKind, OPAQUE_CALL_RETURN_INTRINSIC_PREFIX};

/// Distinct flag bits the environment fork may enumerate: every flag there is
/// derived from unconstrained environment variables only, so independent
/// enumeration is a pure superset.
const ENV_MAX_FLAGS: usize = 6;
/// Expressions larger than this are not abstracted (fail closed).
const MAX_NODES: usize = 16_384;
/// Delta-carrier candidates examined per branch.
const MAX_CARRIER_CANDIDATES: usize = 8;
/// Constant addresses below this are null-page probes, never environment.
const MIN_ENV_ADDRESS: u64 = 0x1_0000;

/// A class of host-environment value a gate condition can be rooted in.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EnvironmentSource {
    /// A CPUID result word.
    Cpuid,
    /// The timestamp counter.
    TimestampCounter,
    /// A `gs:`-relative load (TEB / PEB field).
    PebTebLoad,
    /// The return value of a native (API) call.
    ApiReturn,
    /// The status flags left by a native (API) call.
    ApiFlags,
    /// A memory read whose address is a constant outside the image, or derives
    /// only from other environment sources (a PEB walk, a shared-user-data probe).
    EnvironmentRead,
}

impl EnvironmentSource {
    fn of_intrinsic(name: &str) -> Option<Self> {
        if name.starts_with("x86_cpuid_") {
            Some(Self::Cpuid)
        } else if name.starts_with("read_timestamp_counter_") {
            Some(Self::TimestampCounter)
        } else if name.starts_with("x86_gs_load_") {
            Some(Self::PebTebLoad)
        } else if name.starts_with(OPAQUE_CALL_RETURN_INTRINSIC_PREFIX) {
            Some(Self::ApiReturn)
        } else if name.starts_with("native_call_flags_epoch_") {
            Some(Self::ApiFlags)
        } else {
            None
        }
    }
}

/// Which resolution path produced an environment-sourced branch's targets.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EnvironmentGateRoute {
    /// The ordinary proven path resolved it (the environment value only fed a
    /// flag word the flag-fork accepts whole). Not labeled by the walker.
    Primary,
    /// The original CPUID / timestamp gate over-approximation. Labeled.
    LegacyGate,
    /// The generalized environment-source fork. Labeled.
    Generalized,
    /// The branch reads an environment source but no finite fork exists (the
    /// target consumes an environment or program value at full width, or a
    /// program input survives): it stays a stall, dynamic-only.
    Refused,
}

/// One branch whose condition reads a host-environment source.
#[derive(Clone, Debug, Serialize)]
pub struct EnvironmentGateFork {
    pub vip: Va,
    pub route: EnvironmentGateRoute,
    pub sources: Vec<EnvironmentSource>,
    pub targets: Vec<u64>,
}

thread_local! {
    static FORKS: RefCell<Vec<EnvironmentGateFork>> = const { RefCell::new(Vec::new()) };
}

/// Drains the environment-sourced branches recorded on this thread since the
/// last call. Recording is on only while branch-refusal diagnostics are enabled;
/// it never influences recovery.
#[must_use]
pub fn take_environment_gate_forks() -> Vec<EnvironmentGateFork> {
    FORKS.with(|forks| std::mem::take(&mut *forks.borrow_mut()))
}

pub(crate) fn record_environment_fork(
    vip: Va,
    route: EnvironmentGateRoute,
    target: &Expr,
    in_image: &impl Fn(u64) -> bool,
    targets: &[u64],
) {
    if !diagnostics_enabled() {
        return;
    }
    let Ok(abstraction) = abstract_sources(target, in_image) else {
        return;
    };
    let sources = abstraction.sources;
    if sources.is_empty() {
        return;
    }
    FORKS.with(|forks| {
        forks.borrow_mut().push(EnvironmentGateFork {
            vip,
            route,
            sources: sources.into_iter().collect(),
            targets: targets.to_vec(),
        });
    });
}

/// True when `target` reads a host-environment source. Cheap name prefilter
/// first; the full abstraction runs only when a candidate intrinsic is present.
pub(crate) fn has_environment_source(target: &Expr, in_image: &impl Fn(u64) -> bool) -> bool {
    let candidate = target.intrinsic_dependencies().iter().any(|name| {
        EnvironmentSource::of_intrinsic(name).is_some() || name.starts_with("memory_load_epoch_")
    });
    candidate
        && target.node_count_bounded(MAX_NODES) <= MAX_NODES
        && abstract_sources(target, in_image).is_ok_and(|a| !a.sources.is_empty())
}

struct Abstraction {
    expr: Expr,
    sources: BTreeSet<EnvironmentSource>,
    free: BTreeSet<String>,
}

/// Replaces every environment source in `expr` with a free register, dropping
/// the source's arguments, and reports which source classes were present.
fn abstract_sources(expr: &Expr, in_image: &impl Fn(u64) -> bool) -> Result<Abstraction, ExprError> {
    let mut sources = BTreeSet::new();
    let mut free = BTreeSet::new();
    let rewritten = expr.rewrite_intrinsics(&mut |width, name, arguments| {
        if let Some(source) = EnvironmentSource::of_intrinsic(name) {
            sources.insert(source);
            free.insert(name.to_owned());
            return Some(Expr::register(width, name.to_owned()));
        }
        if name.starts_with("memory_load_epoch_") && arguments.len() == 1 {
            let address = arguments[0].simplify();
            if is_environment_address(&address, &free, in_image) {
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                format!("{address:?}").hash(&mut hasher);
                let variable = format!("env_read_{name}_{:x}", hasher.finish());
                sources.insert(EnvironmentSource::EnvironmentRead);
                free.insert(variable.clone());
                return Some(Expr::register(width, variable));
            }
        }
        None
    })?;
    Ok(Abstraction {
        expr: rewritten,
        sources,
        free,
    })
}

/// A read address is environmental when it is a constant outside the image (and
/// off the null page), or derives only from environment variables and constants.
/// Anything touching a program register or an unrelated intrinsic is program
/// state and stays an ordinary (unabstracted) read.
fn is_environment_address(
    address: &Expr,
    free: &BTreeSet<String>,
    in_image: &impl Fn(u64) -> bool,
) -> bool {
    if !address.intrinsic_dependencies().is_empty() {
        return false;
    }
    let dependencies = address.dependencies();
    if dependencies.is_empty() {
        return matches!(
            address.kind,
            ExprKind::Const { value } if value >= MIN_ENV_ADDRESS && !in_image(value)
        );
    }
    dependencies.iter().all(|name| free.contains(name))
}

/// True when every occurrence of `delta` in `expr` is an operand of
/// `delta + <in-image VA constant>`.
fn only_image_addend(
    expr: &Expr,
    delta: &Expr,
    is_image_va: &impl Fn(u64) -> bool,
    seen: &mut HashSet<*const Expr>,
) -> bool {
    if !seen.insert(std::ptr::from_ref(expr)) {
        return true;
    }
    if expr == delta {
        return false;
    }
    let image_const = |side: &Expr| matches!(side.kind, ExprKind::Const { value } if is_image_va(value));
    let mut go = |child: &Expr| only_image_addend(child, delta, is_image_va, seen);
    match &expr.kind {
        ExprKind::Const { .. } | ExprKind::Register { .. } => true,
        ExprKind::Binary {
            op: BinaryOp::Add,
            left,
            right,
        } if (**left == *delta && image_const(right)) || (**right == *delta && image_const(left)) => {
            true
        }
        ExprKind::Unary { value, .. }
        | ExprKind::Load { address: value }
        | ExprKind::Extract { value, .. }
        | ExprKind::Resize { value, .. } => go(value),
        ExprKind::Binary { left, right, .. }
        | ExprKind::Concat {
            high: left,
            low: right,
        } => go(left) && go(right),
        ExprKind::Intrinsic { arguments, .. } => arguments.iter().all(go),
    }
}

/// Binds to 0 each composite subexpression that is the relocation delta: a
/// non-constant 64-bit value `D` that occurs only as `D + <in-image VA>` and
/// whose own registers / intrinsics occur nowhere else. This is the structural
/// proof [`bind_preferred_base_registers`] applies to a bare register, extended
/// to a delta that reaches the branch as a multi-cell concatenation or a stack
/// word. Same labeled assumption (the preferred-base frame); returns the count.
fn bind_delta_carriers(
    expr: &Expr,
    is_image_va: &impl Fn(u64) -> bool,
    environment: &BTreeSet<String>,
) -> (Expr, usize) {
    let image_const = |side: &Expr| matches!(side.kind, ExprKind::Const { value } if is_image_va(value));
    let mut candidates: Vec<Expr> = Vec::new();
    expr.walk_distinct(&mut |node| {
        if candidates.len() >= MAX_CARRIER_CANDIDATES {
            return;
        }
        if let ExprKind::Binary {
            op: BinaryOp::Add,
            left,
            right,
        } = &node.kind
        {
            for (carrier, constant) in [(left, right), (right, left)] {
                if image_const(constant)
                    && carrier.width == BitWidth::W64
                    && !matches!(
                        carrier.kind,
                        ExprKind::Const { .. } | ExprKind::Register { .. }
                    )
                    && carrier.dependencies().is_disjoint(environment)
                    && !candidates.iter().any(|known| known == &**carrier)
                {
                    candidates.push((**carrier).clone());
                }
            }
        }
    });
    let mut current = expr.clone();
    let mut bound = 0;
    for carrier in candidates {
        if !only_image_addend(&current, &carrier, is_image_va, &mut HashSet::new()) {
            continue;
        }
        let Ok(next) = substitute_flags(
            &current,
            &[(&carrier, 0)],
            &mut std::collections::HashMap::new(),
        ) else {
            continue;
        };
        let own_registers = carrier.dependencies();
        let own_intrinsics = carrier.intrinsic_dependencies();
        let next = next.simplify();
        if next.dependencies().is_disjoint(&own_registers)
            && next.intrinsic_dependencies().is_disjoint(&own_intrinsics)
        {
            current = next;
            bound += 1;
        }
    }
    (current, bound)
}

/// Targets, whether input-guarded, and the delta bindings made.
pub(crate) struct GateResolution {
    pub(crate) targets: Vec<u64>,
    pub(crate) guarded: bool,
    pub(crate) bound_registers: BTreeSet<String>,
}

/// Resolves a branch gated on any host-environment source by a sound both-target
/// over-approximation, or `None` when it is not a pure environment gate or the
/// target set is not provably finite.
///
/// Structural trigger: at least one environment source is present; once every
/// source is abstracted and the preferred-base delta is bound, nothing
/// program-derived may remain (every surviving register is an environment
/// variable, every surviving intrinsic a pure flags word). A condition touching
/// a real program input, or a target that reads an environment value at full
/// width, fails here and is never forked.
pub(crate) fn resolve_generalized_environment_gate(
    target: &Expr,
    is_image_va: &impl Fn(u64) -> bool,
    in_image: &impl Fn(u64) -> bool,
    limits: VmCfgLimits,
) -> Option<GateResolution> {
    if target.node_count_bounded(MAX_NODES) > MAX_NODES {
        return None;
    }
    let abstraction = abstract_sources(target, in_image).ok()?;
    if abstraction.sources.is_empty() {
        return None;
    }
    let (bound, mut bound_registers) =
        bind_preferred_base_registers_except(&abstraction.expr, is_image_va, &abstraction.free);
    let (bound, carriers) = bind_delta_carriers(&bound, is_image_va, &abstraction.free);
    for index in 0..carriers {
        bound_registers.insert(format!("delta_carrier_{index}"));
    }
    let bound = bound.simplify();
    if !bound
        .dependencies()
        .iter()
        .all(|name| abstraction.free.contains(name))
    {
        return None;
    }
    if !bound
        .intrinsic_dependencies()
        .iter()
        .all(|name| is_flags_intrinsic(name))
    {
        return None;
    }
    let (targets, guarded) = resolve_over_inputs_capped(&bound, limits, ENV_MAX_FLAGS).ok()?;
    Some(GateResolution {
        targets,
        guarded,
        bound_registers,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::UnaryOp;

    const LO: u64 = 0x1_8009_ED37;
    const HI: u64 = 0x1_8009_EDB0;

    fn is_image(value: u64) -> bool {
        (0x1_8000_0000..0x1_8100_0000).contains(&value)
    }

    fn in_image(value: u64) -> bool {
        (0x1_8000_0000..0x1_8100_0000).contains(&value)
    }

    fn w64(value: u64) -> Expr {
        Expr::constant(BitWidth::W64, value)
    }

    fn not(value: Expr) -> Expr {
        Expr::unary(UnaryOp::Not, value)
    }

    fn or(left: Expr, right: Expr) -> Expr {
        Expr::binary(BinaryOp::Or, left, right).unwrap()
    }

    fn add(left: Expr, right: Expr) -> Expr {
        Expr::binary(BinaryOp::Add, left, right).unwrap()
    }

    /// VMProtect's branch select: `delta + lo` when `bit` is 1, else `delta + hi`.
    fn select(delta: &Expr, bit: Expr, lo: u64, hi: u64) -> Expr {
        let inner = add(bit, w64(u64::MAX));
        let left = not(or(not(inner.clone()), not(add(delta.clone(), w64(lo)))));
        let right = not(or(inner, not(add(delta.clone(), w64(hi)))));
        add(left, right)
    }

    /// ZF-style single bit: `(sub_flags32(value, 0) >> 6) & 1`.
    fn zero_bit(value: Expr) -> Expr {
        let flags = Expr::intrinsic(
            BitWidth::W64,
            "vm_flags_sub_w32".to_owned(),
            vec![
                Expr::resize(crate::ResizeOp::Truncate, BitWidth::W32, value).unwrap(),
                Expr::constant(BitWidth::W32, 0),
            ],
        );
        let shifted = Expr::binary(BinaryOp::Shr, flags, w64(6)).unwrap();
        Expr::binary(BinaryOp::And, shifted, w64(1)).unwrap()
    }

    fn api_return(epoch: u64, argument: Expr) -> Expr {
        Expr::intrinsic(
            BitWidth::W64,
            format!("native_call_return_epoch_{epoch}"),
            vec![argument],
        )
    }

    fn delta_cat() -> Expr {
        let byte = |n: u32| Expr::register(BitWidth::W8, format!("loop_havoc_1801_m{n}"));
        let mut level: Vec<Expr> = (0..8).map(byte).collect();
        while level.len() > 1 {
            level = level
                .chunks(2)
                .map(|pair| Expr::concat(pair[1].clone(), pair[0].clone()).unwrap())
                .collect();
        }
        level.remove(0)
    }

    fn sources_of(target: &Expr) -> BTreeSet<EnvironmentSource> {
        abstract_sources(target, &in_image).unwrap().sources
    }

    fn resolve(target: &Expr) -> Option<GateResolution> {
        resolve_generalized_environment_gate(target, &is_image, &in_image, VmCfgLimits::default())
    }

    #[test]
    fn intrinsic_names_classify_into_source_classes() {
        let class = EnvironmentSource::of_intrinsic;
        assert_eq!(class("x86_cpuid_eax_3"), Some(EnvironmentSource::Cpuid));
        assert_eq!(class("x86_cpuid_subleaf_3"), Some(EnvironmentSource::Cpuid));
        assert_eq!(
            class("read_timestamp_counter_1"),
            Some(EnvironmentSource::TimestampCounter)
        );
        assert_eq!(class("x86_gs_load_73"), Some(EnvironmentSource::PebTebLoad));
        assert_eq!(
            class("native_call_return_epoch_4"),
            Some(EnvironmentSource::ApiReturn)
        );
        assert_eq!(
            class("native_call_flags_epoch_4"),
            Some(EnvironmentSource::ApiFlags)
        );
        // Program-owned values are not environment.
        assert_eq!(class("memory_load_epoch_4"), None);
        assert_eq!(class("fresh_allocation_return_epoch_4"), None);
        assert_eq!(class("vm_flags_sub_w32"), None);
    }

    #[test]
    fn api_return_gate_forks_both_arms_even_with_a_program_pointer_argument() {
        // `if (NtQuery...(buffer) != 0)`: the call's pointer argument is program
        // state, but its return value is the environment source; the argument is
        // dropped by the abstraction.
        let buffer = Expr::register(BitWidth::W64, "native_rcx_initial");
        let target = select(&w64(0), zero_bit(api_return(2, buffer)), LO, HI);
        let gate = resolve(&target).expect("API-return gate forks");
        assert_eq!(gate.targets, vec![LO, HI]);
        assert!(sources_of(&target).contains(&EnvironmentSource::ApiReturn));
    }

    #[test]
    fn gate_with_more_than_two_independent_environment_flags_forks() {
        // Three fingerprints (volume serial, device id, computer name) folded
        // into one decision: more flags than the ordinary flag-fork allows.
        let flag = |epoch: u64| zero_bit(api_return(epoch, w64(0)));
        let bit = Expr::binary(
            BinaryOp::And,
            Expr::binary(BinaryOp::And, flag(1), flag(2)).unwrap(),
            flag(3),
        )
        .unwrap();
        let target = select(&w64(0), bit, LO, HI);
        let gate = resolve(&target).expect("three-flag environment gate forks");
        assert_eq!(gate.targets, vec![LO, HI]);
    }

    #[test]
    fn peb_load_gate_forks_both_arms() {
        let peb_flag = Expr::intrinsic(
            BitWidth::W64,
            "x86_gs_load_9".to_owned(),
            vec![w64(0x60)],
        );
        let target = select(&w64(0), zero_bit(peb_flag), LO, HI);
        let gate = resolve(&target).expect("PEB-sourced gate forks");
        assert_eq!(gate.targets, vec![LO, HI]);
        assert!(sources_of(&target).contains(&EnvironmentSource::PebTebLoad));
    }

    #[test]
    fn constant_out_of_image_probe_is_an_environment_read() {
        // KUSER_SHARED_DATA-style probe.
        let probe = Expr::intrinsic(
            BitWidth::W64,
            "memory_load_epoch_5".to_owned(),
            vec![w64(0x7FFE_0300)],
        );
        let target = select(&w64(0), zero_bit(probe), LO, HI);
        let gate = resolve(&target).expect("out-of-image constant probe forks");
        assert_eq!(gate.targets, vec![LO, HI]);
        assert!(sources_of(&target).contains(&EnvironmentSource::EnvironmentRead));
    }

    #[test]
    fn read_of_in_image_or_program_memory_is_not_environment() {
        // In-image constant: an image read, not the host environment.
        let image_read = Expr::intrinsic(
            BitWidth::W64,
            "memory_load_epoch_5".to_owned(),
            vec![w64(0x1_8000_1000)],
        );
        assert!(resolve(&select(&w64(0), zero_bit(image_read), LO, HI)).is_none());
        // Register-rooted (a caller pointer): program state.
        let program_read = Expr::intrinsic(
            BitWidth::W64,
            "memory_load_epoch_5".to_owned(),
            vec![Expr::register(BitWidth::W64, "native_rcx_initial")],
        );
        assert!(resolve(&select(&w64(0), zero_bit(program_read), LO, HI)).is_none());
        // Null-page probe.
        let null_read = Expr::intrinsic(
            BitWidth::W64,
            "memory_load_epoch_5".to_owned(),
            vec![w64(0x10)],
        );
        assert!(resolve(&select(&w64(0), zero_bit(null_read), LO, HI)).is_none());
    }

    #[test]
    fn peb_walk_chain_is_an_environment_read() {
        let peb = Expr::intrinsic(BitWidth::W64, "x86_gs_load_1".to_owned(), vec![w64(0x60)]);
        let ldr = Expr::intrinsic(
            BitWidth::W64,
            "memory_load_epoch_3".to_owned(),
            vec![add(peb, w64(0x18))],
        );
        let target = select(&w64(0), zero_bit(ldr), LO, HI);
        let gate = resolve(&target).expect("PEB walk forks");
        assert!(sources_of(&target).contains(&EnvironmentSource::EnvironmentRead));
        assert_eq!(gate.targets, vec![LO, HI]);
    }

    #[test]
    fn delta_carried_as_a_multi_cell_concatenation_binds_and_the_gate_forks() {
        // The relocation delta reaches the branch as an 8-cell loop-widened
        // concatenation, not a bare register.
        let delta = delta_cat();
        let target = select(&delta, zero_bit(api_return(1, w64(0))), LO, HI);
        assert!(resolve_over_inputs_capped(&target.simplify(), VmCfgLimits::default(), 2).is_err());
        let gate = resolve(&target).expect("carried delta binds, gate forks");
        assert_eq!(gate.targets, vec![LO, HI]);
        assert!(gate.bound_registers.iter().any(|r| r.starts_with("delta_carrier_")));
    }

    #[test]
    fn carrier_that_is_also_used_as_data_is_not_bound() {
        // The same cells also feed the compare: they are not purely the delta.
        let delta = delta_cat();
        let bit = zero_bit(add(api_return(1, w64(0)), delta.clone()));
        let target = select(&delta, bit, LO, HI);
        assert!(resolve(&target).is_none());
    }

    #[test]
    fn unbounded_computed_target_stays_refused() {
        // Target = environment value + constant: every arm is a different value;
        // no finite fork exists, so it is not forked.
        let value = api_return(7, w64(0));
        let target = add(value, w64(LO));
        assert!(resolve(&target).is_none());
        // A full-width environment value inside the select arithmetic.
        let selected = select(&w64(0), zero_bit(api_return(1, w64(0))), LO, HI);
        let leaky = add(selected, api_return(2, w64(0)));
        assert!(resolve(&leaky).is_none());
    }

    #[test]
    fn program_input_surviving_abstraction_is_never_forked() {
        // The gate also depends on a real program register outside any delta.
        let program = Expr::register(BitWidth::W64, "native_rdx_initial");
        let bit = zero_bit(add(api_return(1, w64(0)), program));
        assert!(resolve(&select(&w64(0), bit, LO, HI)).is_none());
    }

    #[test]
    fn environment_source_detection_distinguishes_env_from_program_conditions() {
        let env = select(&w64(0), zero_bit(api_return(2, w64(0))), LO, HI);
        assert!(has_environment_source(&env, &in_image));
        let program = select(
            &w64(0),
            zero_bit(Expr::register(BitWidth::W64, "native_rdx_initial")),
            LO,
            HI,
        );
        assert!(!has_environment_source(&program, &in_image));
        let image_read = Expr::intrinsic(
            BitWidth::W64,
            "memory_load_epoch_5".to_owned(),
            vec![w64(0x1_8000_1000)],
        );
        assert!(!has_environment_source(
            &select(&w64(0), zero_bit(image_read), LO, HI),
            &in_image
        ));
    }

    #[test]
    fn no_environment_source_means_no_fork() {
        let bit = zero_bit(Expr::register(BitWidth::W64, "native_rdx_initial"));
        assert!(resolve(&select(&w64(0), bit, LO, HI)).is_none());
    }
}
