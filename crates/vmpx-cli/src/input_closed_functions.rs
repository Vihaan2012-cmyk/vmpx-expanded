//! Discovery of virtualized functions whose native exit is a closed function of
//! the replacement's inputs. Recovery only: nothing is reintegrated here.

use std::collections::BTreeSet;

use vmpx_disasm::DirectJumpTraceLimits;
use vmpx_pe::{PeImage, Rva};
use vmpx_vm::{
    EntryPathRecoveryLimits, NativeExitState, RecoveredEntryPathFamily, VmDiscoveryLimits,
    VmProfileExplorerLimits, X64NativeRegister, analyze_vm_entry_at, analyze_x64_native_frame,
    is_expr_input_closed, is_input_closed, recover_entry_paths,
};
use vmpx_vmlift::{calling_convention, collect_path_effects};

use crate::{clean::replacement_inputs, vm_inventory::VmInventoryEntry};

/// Inventory entries below this confidence are not attempted.
pub(crate) const MIN_CONFIDENCE: u8 = 60;

/// A virtualized function proven to reach a native exit expressed purely over
/// the symbols a replacement provides.
#[derive(Clone, Debug)]
pub(crate) struct InputClosedFunction {
    pub rva: Rva,
    /// Consumed by reintegration (SP2 Task 5).
    #[expect(dead_code, reason = "consumed by SP2 Task 5 reintegration")]
    pub exit: NativeExitState,
    /// Every symbol the replacement binds (the closedness `allowed` set).
    pub inputs: BTreeSet<String>,
    /// Task 3's inferred live-in registers. Informational only: it
    /// under-reports, so it never narrows `inputs`.
    pub live_in: BTreeSet<X64NativeRegister>,
}

/// True when the path family carries no memory the closedness check cannot see:
/// havocs or tainted memory in any retained symbolic state.
fn memory_is_exact(family: &RecoveredEntryPathFamily) -> bool {
    let path = &family.paths[0];
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

/// Why `entry` is not an input-closed function, or its recovery.
fn recover_one(pe: &PeImage<'_>, entry: &VmInventoryEntry) -> Result<InputClosedFunction, String> {
    let candidate = analyze_vm_entry_at(pe, entry.rva, VmDiscoveryLimits::default())
        .ok_or_else(|| "no VM entry stub at this RVA".to_owned())?;
    // `allowed` is the full set the replacement binds (every architectural
    // register, `vsp_initial`, and the cursor register's rewritten name), not
    // the under-reporting live-in subset: "closed" must mean "depends on nothing
    // the replacement does not provide".
    let mut rewrites = std::collections::BTreeMap::new();
    let inputs = replacement_inputs(pe, &candidate, &mut rewrites)?;
    let allowed: BTreeSet<String> = inputs
        .into_iter()
        .map(|(name, _)| name)
        .chain(rewrites.into_keys())
        .collect();

    let limits = EntryPathRecoveryLimits {
        chain: VmProfileExplorerLimits {
            max_segments: 32,
            max_instructions_per_segment: 2_048,
            max_total_instructions: 8_192,
            ..VmProfileExplorerLimits::default()
        },
        ..EntryPathRecoveryLimits::default()
    };
    // Trivial domain: no input is enumerated, so a branch on one is incomplete.
    let family = recover_entry_paths(pe, &candidate, &std::collections::BTreeMap::new(), limits)
        .map_err(|error| error.to_string())?;
    if !family.domain_complete || !family.all_paths_reach_vm_exit {
        return Err("entry paths are not complete".to_owned());
    }
    let [path] = family.paths.as_slice() else {
        return Err(format!(
            "{} paths, exactly one required",
            family.paths.len()
        ));
    };
    if !path.assignments.is_empty() {
        return Err("path depends on an enumerated input".to_owned());
    }
    let exit = path
        .native_exit
        .clone()
        .ok_or_else(|| "no native exit state".to_owned())?;
    if !is_input_closed(&exit, &allowed) {
        return Err("exit depends on a symbol outside the replacement inputs".to_owned());
    }
    // Side effects must be closed too, and nothing may be hidden in memory the
    // exit expressions do not mention.
    let effects = collect_path_effects(&family, path);
    if !effects.unsupported.is_empty() || !effects.calls.is_empty() {
        return Err("path has unsupported effects or native calls".to_owned());
    }
    if !effects.stores.iter().all(|store| {
        is_expr_input_closed(&store.address, &allowed)
            && is_expr_input_closed(&store.value, &allowed)
    }) {
        return Err("a store depends on a symbol outside the replacement inputs".to_owned());
    }
    if !memory_is_exact(&family) {
        return Err("path has memory havocs or tainted memory".to_owned());
    }
    let frame = analyze_x64_native_frame(pe, entry.rva, DirectJumpTraceLimits::default())
        .map_err(|error| error.to_string())?;
    Ok(InputClosedFunction {
        rva: entry.rva,
        exit,
        inputs: allowed,
        live_in: calling_convention(&family, &frame).live_in,
    })
}

/// Recovers every inventory entry (confidence >= [`MIN_CONFIDENCE`]) whose single
/// recovered path reaches an input-closed native exit. Also returns one
/// diagnostic per entry that was attempted and rejected.
pub(crate) fn recover_input_closed_functions(
    pe: &PeImage<'_>,
    inventory: &[VmInventoryEntry],
) -> (Vec<InputClosedFunction>, Vec<String>) {
    let mut functions = Vec::new();
    let mut diagnostics = Vec::new();
    for entry in inventory
        .iter()
        .filter(|entry| entry.confidence >= MIN_CONFIDENCE)
    {
        match recover_one(pe, entry) {
            Ok(function) => functions.push(function),
            Err(reason) => diagnostics.push(format!("{}: not input-closed: {reason}", entry.rva)),
        }
    }
    (functions, diagnostics)
}
