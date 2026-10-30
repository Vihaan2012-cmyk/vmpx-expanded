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
    is_expr_input_closed, is_input_closed, recover_branching_entry_paths, recover_entry_paths,
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
        .chain(rewrites.into_keys())
        .collect();
    Ok((candidate, allowed))
}

/// The native exit of `path` if it, its stores and its memory are all closed
/// over `allowed`.
fn closed_exit(
    family: &RecoveredEntryPathFamily,
    path: &RecoveredEntryPath,
    allowed: &BTreeSet<String>,
) -> Result<(NativeExitState, CollectedEffects), String> {
    let exit = path
        .native_exit
        .clone()
        .ok_or_else(|| "no native exit state".to_owned())?;
    if !is_input_closed(&exit, allowed) {
        return Err("exit depends on a symbol outside the replacement inputs".to_owned());
    }
    // Side effects must be closed too, and nothing may be hidden in memory the
    // exit expressions do not mention.
    let effects = collect_path_effects(family, path);
    if !effects.unsupported.is_empty() || !effects.calls.is_empty() {
        return Err("path has unsupported effects or native calls".to_owned());
    }
    if !effects.stores.iter().all(|store| {
        is_expr_input_closed(&store.address, allowed) && is_expr_input_closed(&store.value, allowed)
    }) {
        return Err("a store depends on a symbol outside the replacement inputs".to_owned());
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
    let (exit, effects) = closed_exit(&family, path, &allowed)?;
    let frame = analyze_x64_native_frame(pe, rva, DirectJumpTraceLimits::default())
        .map_err(|error| error.to_string())?;
    Ok(InputClosedFunction {
        rva,
        exit,
        effects,
        inputs: allowed,
        live_in: calling_convention(&family, &frame).live_in,
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
    // The replacement evaluates the branch target to pick a path, so the guard
    // itself must be closed over the replacement's inputs.
    if !is_expr_input_closed(&guarded.branch_target, &allowed) {
        return Err("branch target depends on a symbol outside the replacement inputs".to_owned());
    }
    let mut paths = Vec::with_capacity(family.paths.len());
    for (path, target) in family.paths.iter().zip(&guarded.targets) {
        if !path.assignments.is_empty() {
            return Err("branching path was specialized by an input".to_owned());
        }
        let (exit, effects) = closed_exit(family, path, &allowed)?;
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
        branch_target: guarded.branch_target.clone(),
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
