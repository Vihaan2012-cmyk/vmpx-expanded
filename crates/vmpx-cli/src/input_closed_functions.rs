//! Discovery of virtualized functions whose native exit is a closed function of
//! the replacement's inputs. Recovery only: nothing is reintegrated here.
//!
//! Two shapes are recovered, both fail-closed:
//! * input-closed: one complete path, no branch on an input (also the executable
//!   entry, whose invocation domain is a single path);
//! * branching: the first virtual branch depends on live-in registers, but a
//!   finite input domain is *proven* exhaustive (`infer_branch_input_domains`).

use std::collections::{BTreeMap, BTreeSet};

use vmpx_disasm::DirectJumpTraceLimits;
use vmpx_pe::{PeImage, Rva};
use vmpx_vm::{
    BranchingRecoveryLimits, DomainCertificate, EntryPathRecoveryLimits, Expr, NativeExitState,
    RecoveredEntryPath, RecoveredEntryPathFamily, VmDiscoveryLimits, VmEntryCandidate,
    VmProfileExplorerLimits, X64NativeRegister, analyze_vm_entry_at, analyze_x64_native_frame,
    exit_closure_violation, input_closure_violation, recover_branching_entry_paths,
    recover_entry_paths,
};
use vmpx_vmlift::{CollectedEffects, calling_convention, collect_path_effects};

use crate::{clean::replacement_inputs, vm_inventory::VmInventoryEntry};

/// Inventory entries below this confidence are not attempted.
pub(crate) const MIN_CONFIDENCE: u8 = 60;

/// A virtualized function proven to reach a native exit expressed purely over
/// the symbols a replacement provides.
#[derive(Clone, Debug)]
pub(crate) struct InputClosedFunction {
    pub rva: Rva,
    pub exit: NativeExitState,
    /// Stores and calls the replacement must replay (an exit alone would
    /// silently drop them).
    pub effects: CollectedEffects,
    /// Every symbol the replacement binds (the closedness `allowed` set).
    pub inputs: BTreeSet<String>,
    /// Task 3's inferred live-in registers. Informational only: it
    /// under-reports, so it never narrows `inputs`.
    pub live_in: BTreeSet<X64NativeRegister>,
    /// True when closing this function required folding an image-backed read to
    /// a constant, which holds only while the image's protection does. Such a
    /// recovery rests on `AssumesImageProtectionStable` and is never PROVEN.
    pub assumes_image_protection: bool,
}

/// A virtualized function whose first virtual branch depends on its inputs and
/// whose complete input domain was inferred and proven finite and exhaustive.
/// `paths[i]` runs exactly when `branch_target == paths[i].0`; the targets are
/// the full image of `branch_target`.
#[derive(Clone, Debug)]
pub(crate) struct BranchingFunction {
    pub rva: Rva,
    pub branch_target: Expr,
    pub paths: Vec<BranchPath>,
    /// The recovered family's proof that `paths` cover the whole input domain.
    pub domain_complete: bool,
    pub inputs: BTreeSet<String>,
    pub live_in: BTreeSet<X64NativeRegister>,
    pub certificates: BTreeMap<String, DomainCertificate>,
}

/// One guarded path: runs exactly when `branch_target == target`.
#[derive(Clone, Debug)]
pub(crate) struct BranchPath {
    pub target: u64,
    pub exit: NativeExitState,
    pub effects: CollectedEffects,
}

/// Why an entry is not an input-closed single-path function.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ClosedFailure {
    /// The first virtual branch depends on an input the empty domain lacks.
    BranchesOnInput,
    Rejected(String),
}

impl std::fmt::Display for ClosedFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BranchesOnInput => f.write_str("entry branches on an input"),
            Self::Rejected(reason) => f.write_str(reason),
        }
    }
}

impl From<String> for ClosedFailure {
    fn from(reason: String) -> Self {
        Self::Rejected(reason)
    }
}

/// True when the path family carries no memory the closedness check cannot see:
/// havocs or tainted memory in any retained symbolic state of `path`.
fn memory_is_exact(family: &RecoveredEntryPathFamily, path: &RecoveredEntryPath) -> bool {
    std::iter::once(&family.entry_profile.symbolic)
        .chain(path.chain.segments.iter().flat_map(|segment| {
            segment
                .symbolic
                .iter()
                .chain(segment.boundary_symbolic.iter())
                .chain(segment.transition_pop.iter())
        }))
        .all(|report| report.memory_havocs.is_empty() && !report.memory_tainted)
}

fn recovery_limits() -> EntryPathRecoveryLimits {
    EntryPathRecoveryLimits {
        chain: VmProfileExplorerLimits {
            max_segments: 32,
            max_instructions_per_segment: 2_048,
            max_total_instructions: 8_192,
            ..VmProfileExplorerLimits::default()
        },
        ..EntryPathRecoveryLimits::default()
    }
}

/// The entry candidate and the full set the replacement binds (every
/// architectural register, `vsp_initial`, and the cursor register's rewritten
/// name), not the under-reporting live-in subset: "closed" must mean "depends on
/// nothing the replacement does not provide".
fn entry_inputs(
    pe: &PeImage<'_>,
    rva: Rva,
) -> Result<(VmEntryCandidate, BTreeSet<String>), String> {
    let candidate = analyze_vm_entry_at(pe, rva, VmDiscoveryLimits::default())
        .ok_or_else(|| "no VM entry stub at this RVA".to_owned())?;
    let mut rewrites = BTreeMap::new();
    let inputs = replacement_inputs(pe, &candidate, &mut rewrites)?;
    let allowed = inputs
        .into_iter()
        .map(|(name, _)| name)
        // The entry condition state. Not an architectural GP register, so
        // `replacement_inputs` never listed it, and a routine that preserves its
        // caller's flags -- exit rflags == this symbol -- was refused for
        // depending on something outside the inputs. 135 of 264 closedness
        // refusals on systems.xpl, 125 on the rflags slot itself.
        //
        // The replacement DOES provide it: `build_replacement` captures it with
        // `Operation::ReadFlags` whenever a lowered expression reads it, so
        // admitting it here is backed by a real binding, not an assumption.
        .chain([vmpx_vmlift::ENTRY_FLAGS_SYMBOL.to_owned()])
        // The caller's return address, likewise provided rather than assumed:
        // `build_replacement` captures it as an entry-stack-pointer read plus a
        // load through it. 4 refusals on systems.xpl.
        .chain([vmpx_vmlift::NATIVE_RETURN_ADDRESS_INITIAL.to_owned()])
        .chain(rewrites.into_keys())
        .collect();
    Ok((candidate, allowed))
}

/// `VMPX_CONCRETIZE_IMAGE=1`: fold image-backed opaque intrinsics to constants
/// before the closedness check.
///
/// Opt-in for the same reason `--serve-ro` is. `concretize_with_immutable_image`
/// states that it "deliberately does not infer memory immutability" and that a
/// caller may treat its result as a proof "only after separately establishing
/// that all represented memory epochs preserve the referenced image bytes".
/// This binary imports `VirtualProtect`, so that cannot be established -- a
/// recovery that depends on this rests on `AssumesImageProtectionStable` and can
/// never be PROVEN.
fn concretize_image_reads() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VMPX_CONCRETIZE_IMAGE").is_ok_and(|value| value == "1"))
}

/// Folds image-backed opaque intrinsics in `expression` to constants, returning
/// the rewritten expression and whether anything was resolved.
///
/// Resolves exactly the two shapes that dominate the closedness refusals:
/// `memory_load_epoch_N` at a constant address, and the table-checksum
/// recurrence. Both read bytes the file already contains, so the value is
/// computable -- but only under image immutability, which is why the caller
/// records the assumption.
fn concretize(pe: &PeImage<'_>, expression: &vmpx_vm::Expr) -> (vmpx_vm::Expr, bool) {
    if !concretize_image_reads() {
        return (expression.clone(), false);
    }
    match vmpx_vm::concretize_with_immutable_image(
        pe,
        expression,
        &BTreeMap::new(),
        vmpx_vm::ImageConcretizationLimits::default(),
    ) {
        Ok(report) if report.resolved_intrinsics > 0 => (report.expression, true),
        // Unresolved or over budget: keep the original and let the closedness
        // check refuse it exactly as before.
        _ => (expression.clone(), false),
    }
}

/// Seeds the opaque-state probe substitutes, for the diagnostic alone.
///
/// MEASURED on systems.xpl (2026-11-07): of 241 exits refused for opaque state,
/// all 241 come back VARIES, so this route recovers nothing. An earlier probe
/// that gave every opaque term ONE shared constant reported 146 as independent
/// and drove `recovered` 20 -> 158; that was an artefact -- a shared constant
/// makes `checksum_a ^ checksum_b` fold to zero under every seed. The fills are
/// per-term (see `substitute_opaque_intrinsics`) and the number collapses back
/// to 0. The dependence on distinct integrity terms is real, so these functions
/// are not pure functions of their inputs under any purely static reading.
/// A wide spread matters: on systems.xpl, going from 4 fills to 7 removed 8 of
/// 146 acceptances, so a narrow probe does produce false positives.
const OPAQUE_STATE_FILLS: [u64; 23] = [
    1,
    2,
    3,
    u64::MAX,
    u64::MAX - 1,
    0x5555_5555_5555_5555,
    0xAAAA_AAAA_AAAA_AAAA,
    0x8000_0000_0000_0000,
    0x0000_0000_8000_0000,
    0x0000_0000_0000_0080,
    0x0123_4567_89AB_CDEF,
    0xDEAD_BEEF_CAFE_F00D,
    0xFFFF_FFFF_0000_0000,
    0x0000_0000_FFFF_FFFF,
    0xFF00_FF00_FF00_FF00,
    0x00FF_00FF_00FF_00FF,
    0x7FFF_FFFF_FFFF_FFFF,
    0x0F0F_0F0F_0F0F_0F0F,
    0xF0F0_F0F0_F0F0_F0F0,
    0x1234_5678_9ABC_DEF0,
    0xA5A5_A5A5_A5A5_A5A5,
    0x3141_5926_5358_9793,
    0x2718_2818_2845_9045,
];

/// What closing a function rested on. Each flag marks a recovery that is never
/// PROVEN, so the tier it feeds must stay below N2.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ClosedAssumptions {
    /// Closing required folding an image-backed read: holds only while the
    /// image's protection does (`AssumesImageProtectionStable`).
    pub image_protection: bool,
}

/// The native exit of `path` if it, its stores and its memory are all closed
/// over `allowed`. `assumptions` records what closing it rested on.
fn closed_exit(
    pe: &PeImage<'_>,
    family: &RecoveredEntryPathFamily,
    path: &RecoveredEntryPath,
    allowed: &BTreeSet<String>,
    assumptions: &mut ClosedAssumptions,
) -> Result<(NativeExitState, CollectedEffects), String> {
    let mut exit = path
        .native_exit
        .clone()
        .ok_or_else(|| "no native exit state".to_owned())?;
    for value in exit
        .registers
        .values_mut()
        .chain([&mut exit.rflags, &mut exit.return_address])
    {
        let (folded, resolved) = concretize(pe, value);
        *value = folded;
        assumptions.image_protection |= resolved;
    }
    // Name the TERM, not just the fact. The three causes need different work: an
    // unbound register can be admitted by widening the replacement's input set,
    // while a load or an opaque intrinsic cannot.
    if let Some(why) = exit_closure_violation(&exit, allowed) {
        // An unresolved read is an opaque `memory_load_epoch_N` intrinsic, and a
        // replacement needs only to perform the same read, which `expr_codegen`
        // already lowers. The epoch makes it sound: a replacement emits every
        // store before computing any value, so a materialised read observes all
        // of them -- which is what the VM saw exactly when the read's epoch was
        // already at or above every store's. Reads below that keep refusing.
        let after_every_store = vmpx_vmlift::last_write_epoch(family, path);
        let retried = {
            let mut candidate = exit.clone();
            for value in candidate
                .registers
                .values_mut()
                .chain([&mut candidate.rflags, &mut candidate.return_address])
            {
                if let Ok(materialised) =
                    vmpx_vm::materialize_epoch_loads_from(value, after_every_store)
                {
                    *value = materialised;
                }
            }
            candidate
        };
        let still_open = retried
            .registers
            .values()
            .chain([&retried.rflags, &retried.return_address])
            .find_map(|value| vmpx_vm::input_closure_violation_allowing_loads(value, allowed));
        match still_open {
            None => exit = retried,
            Some(remaining) => {
                // DIAGNOSTIC: is the dependence on opaque state SEMANTIC or only
                // syntactic? VMProtect mixes integrity checksums and `rdtsc`
                // into register values and cancels them out again. Substituting
                // several fills and comparing says whether the exit actually
                // varies. Evidence for a ceiling, never a basis for accepting.
                let probe = |fill: u64| {
                    let mut candidate = exit.clone();
                    for value in candidate
                        .registers
                        .values_mut()
                        .chain([&mut candidate.rflags, &mut candidate.return_address])
                    {
                        if let Ok(substituted) =
                            vmpx_vm::substitute_opaque_intrinsics(value, fill)
                        {
                            *value = substituted;
                        }
                    }
                    candidate
                };
                let reference = probe(0);
                let independent = OPAQUE_STATE_FILLS
                    .iter()
                    .all(|fill| probe(*fill) == reference);
                let closed_after = independent
                    && reference
                        .registers
                        .values()
                        .chain([&reference.rflags, &reference.return_address])
                        .all(|value| input_closure_violation(value, allowed).is_none());
                let probe_verdict = if !independent {
                    "VARIES"
                } else if closed_after {
                    "INDEPENDENT-AND-CLOSED"
                } else {
                    "INDEPENDENT-BUT-OPEN"
                };
                return Err(format!(
                    "exit depends on a symbol outside the replacement inputs: {why} \
                     (after materialising reads at or above store epoch \
                     {after_every_store}: {remaining}; opaque-state probe: {probe_verdict})"
                ));
            }
        }
    }
    // Side effects must be closed too, and nothing may be hidden in memory the
    // exit expressions do not mention.
    let mut effects = collect_path_effects(family, path);
    if !effects.unsupported.is_empty() || !effects.calls.is_empty() {
        return Err("path has unsupported effects or native calls".to_owned());
    }
    for store in &mut effects.stores {
        let (address, address_resolved) = concretize(pe, &store.address);
        let (value, value_resolved) = concretize(pe, &store.value);
        store.address = address;
        store.value = value;
        assumptions.image_protection |= address_resolved || value_resolved;
    }
    if let Some(why) = effects.stores.iter().find_map(|store| {
        input_closure_violation(&store.address, allowed)
            .map(|w| format!("address: {w}"))
            .or_else(|| {
                input_closure_violation(&store.value, allowed).map(|w| format!("value: {w}"))
            })
    }) {
        return Err(format!(
            "a store depends on a symbol outside the replacement inputs: {why}"
        ));
    }
    if !memory_is_exact(family, path) {
        return Err("path has memory havocs or tainted memory".to_owned());
    }
    Ok((exit, effects))
}

/// Recovers `rva` as an input-closed function: an empty input domain, so the
/// entry must be a single complete path that no input steers.
pub(crate) fn recover_input_closed(
    pe: &PeImage<'_>,
    rva: Rva,
) -> Result<InputClosedFunction, ClosedFailure> {
    let (candidate, allowed) = entry_inputs(pe, rva)?;
    // Trivial domain: no input is enumerated, so a branch on one is incomplete.
    let family = recover_entry_paths(pe, &candidate, &BTreeMap::new(), recovery_limits())
        .map_err(|error| error.to_string())?;
    if !family.domain_complete {
        return Err(ClosedFailure::BranchesOnInput);
    }
    if !family.all_paths_reach_vm_exit {
        return Err(ClosedFailure::Rejected(
            "entry paths are not complete".to_owned(),
        ));
    }
    let [path] = family.paths.as_slice() else {
        return Err(ClosedFailure::Rejected(format!(
            "{} paths, exactly one required",
            family.paths.len()
        )));
    };
    if !path.assignments.is_empty() {
        return Err(ClosedFailure::Rejected(
            "path depends on an enumerated input".to_owned(),
        ));
    }
    let mut assumptions = ClosedAssumptions::default();
    let (exit, effects) = closed_exit(pe, &family, path, &allowed, &mut assumptions)?;
    let frame = analyze_x64_native_frame(pe, rva, DirectJumpTraceLimits::default())
        .map_err(|error| error.to_string())?;
    Ok(InputClosedFunction {
        rva,
        exit,
        effects,
        inputs: allowed,
        live_in: calling_convention(&family, &frame).live_in,
        assumes_image_protection: assumptions.image_protection,
    })
}

/// Recovers `rva` as a branching function: only when the first branch's inputs
/// have a proven-exhaustive finite domain, and every resulting path is closed.
pub(crate) fn recover_branching(pe: &PeImage<'_>, rva: Rva) -> Result<BranchingFunction, String> {
    let (candidate, allowed) = entry_inputs(pe, rva)?;
    let guarded = recover_branching_entry_paths(
        pe,
        &candidate,
        BranchingRecoveryLimits {
            paths: recovery_limits(),
            ..BranchingRecoveryLimits::default()
        },
    )
    .map_err(|error| error.to_string())?;
    let family = &guarded.family;
    if !family.domain_complete || !family.all_paths_reach_vm_exit {
        return Err("branch paths are not complete".to_owned());
    }
    let mut assumptions = ClosedAssumptions::default();
    // The replacement evaluates the branch target to pick a path, so the guard
    // itself must be closed over the replacement's inputs.
    let (branch_target, branch_resolved) = concretize(pe, &guarded.branch_target);
    assumptions.image_protection |= branch_resolved;
    if let Some(why) = input_closure_violation(&branch_target, &allowed) {
        return Err(format!(
            "branch target depends on a symbol outside the replacement inputs: {why}"
        ));
    }
    let mut paths = Vec::with_capacity(family.paths.len());
    for (path, target) in family.paths.iter().zip(&guarded.targets) {
        if !path.assignments.is_empty() {
            return Err("branching path was specialized by an input".to_owned());
        }
        let (exit, effects) = closed_exit(pe, family, path, &allowed, &mut assumptions)?;
        paths.push(BranchPath {
            target: *target,
            exit,
            effects,
        });
    }
    let frame = analyze_x64_native_frame(pe, rva, DirectJumpTraceLimits::default())
        .map_err(|error| error.to_string())?;
    Ok(BranchingFunction {
        rva,
        branch_target,
        paths,
        domain_complete: family.domain_complete,
        inputs: allowed,
        live_in: calling_convention(family, &frame).live_in,
        certificates: guarded.certificates.clone(),
    })
}

/// Outcome of recovering an executable (non-DLL) header entry.
#[derive(Debug)]
pub(crate) struct ExeEntryDecision {
    pub function: Option<InputClosedFunction>,
    /// Why the entry remains a blocker, if it does.
    pub blocker: Option<String>,
}

/// An executable's PE entry (CRT start) is invoked once, not with a DllMain-style
/// reason, so its invocation domain is a single path: the same empty-domain
/// input-closed recovery applies, with no input enumerated. `recover` runs only
/// when the entry clears the confidence floor.
pub(crate) fn decide_exe_entry(
    confidence: u8,
    recover: impl FnOnce() -> Result<InputClosedFunction, ClosedFailure>,
) -> ExeEntryDecision {
    if confidence < MIN_CONFIDENCE {
        return ExeEntryDecision {
            function: None,
            blocker: Some(format!(
                "executable entry has VM-entry confidence {confidence}, below {MIN_CONFIDENCE}"
            )),
        };
    }
    match recover() {
        Ok(function) => ExeEntryDecision {
            function: Some(function),
            blocker: None,
        },
        Err(failure) => ExeEntryDecision {
            function: None,
            blocker: Some(format!(
                "executable entry is not recoverable as an input-closed single path: {failure}"
            )),
        },
    }
}

/// Everything recovered from the inventory, plus one diagnostic per entry that
/// was attempted and rejected.
#[derive(Debug, Default)]
pub(crate) struct RecoveredFunctions {
    pub closed: Vec<InputClosedFunction>,
    pub branching: Vec<BranchingFunction>,
    pub diagnostics: Vec<String>,
}

/// Recovers every inventory entry (confidence >= [`MIN_CONFIDENCE`]) as an
/// input-closed function or, when its first branch depends on inputs, as a
/// branching function with a proven-exhaustive domain. `skip` is an executable
/// header entry, decided separately by [`decide_exe_entry`].
pub(crate) fn recover_input_closed_functions(
    pe: &PeImage<'_>,
    inventory: &[VmInventoryEntry],
    skip: Option<Rva>,
) -> RecoveredFunctions {
    let mut recovered = RecoveredFunctions::default();
    for entry in inventory
        .iter()
        .filter(|entry| entry.confidence >= MIN_CONFIDENCE && Some(entry.rva) != skip)
    {
        match recover_input_closed(pe, entry.rva) {
            Ok(function) => recovered.closed.push(function),
            Err(ClosedFailure::BranchesOnInput) => match recover_branching(pe, entry.rva) {
                Ok(function) => recovered.branching.push(function),
                Err(reason) => recovered.diagnostics.push(format!(
                    "{}: branches on an input and no provable finite domain: {reason}",
                    entry.rva
                )),
            },
            Err(ClosedFailure::Rejected(reason)) => recovered
                .diagnostics
                .push(format!("{}: not input-closed: {reason}", entry.rva)),
        }
    }
    recovered
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use vmpx_vm::BitWidth;

    use super::*;

    fn function() -> InputClosedFunction {
        InputClosedFunction {
            rva: Rva(0x1000),
            exit: NativeExitState {
                xmm: std::collections::BTreeMap::new(),
                schema_version: 1,
                registers: BTreeMap::new(),
                rflags: Expr::constant(BitWidth::W64, 0x202),
                return_address: Expr::constant(BitWidth::W64, 0x1_4000_1000),
                frame_size: 136,
            },
            effects: CollectedEffects {
                stores: Vec::new(),
                calls: Vec::new(),
                unsupported: Vec::new(),
            },
            inputs: BTreeSet::new(),
            live_in: BTreeSet::new(),
            assumes_image_protection: false,
        }
    }

    #[test]
    fn exe_entry_that_recovers_clears_the_blocker() {
        let decision = decide_exe_entry(90, || Ok(function()));
        assert!(decision.function.is_some());
        assert_eq!(decision.blocker, None);
    }

    #[test]
    fn exe_entry_that_branches_on_an_input_stays_a_blocker() {
        let decision = decide_exe_entry(90, || Err(ClosedFailure::BranchesOnInput));
        assert!(decision.function.is_none());
        assert!(
            decision
                .blocker
                .expect("blocker")
                .contains("branches on an input")
        );
    }

    #[test]
    fn exe_entry_rejection_reason_is_kept_in_the_blocker() {
        let decision = decide_exe_entry(90, || {
            Err(ClosedFailure::Rejected("path has native calls".to_owned()))
        });
        assert!(decision.function.is_none());
        assert!(
            decision
                .blocker
                .expect("blocker")
                .contains("path has native calls")
        );
    }

    #[test]
    fn exe_entry_below_the_confidence_floor_is_not_attempted() {
        let attempted = Cell::new(false);
        let decision = decide_exe_entry(MIN_CONFIDENCE - 1, || {
            attempted.set(true);
            Ok(function())
        });
        assert!(!attempted.get());
        assert!(decision.function.is_none());
        assert!(decision.blocker.is_some());
    }
}
