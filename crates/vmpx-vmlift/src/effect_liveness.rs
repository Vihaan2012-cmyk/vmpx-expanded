use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;
use vmpx_vm::{
    Expr, FRESH_ALLOCATION_RETURN_INTRINSIC_PREFIX, NativeCallMemoryDecision, RecoveredEntryPath,
    RecoveredEntryPathFamily, SymbolicNativeCall, SymbolicVmExecutionReport, X64NativeRegister,
    symbolic_pointer_offset,
};

const VSP_ROOT: &str = "vsp_initial";
const WIN64_NONVOLATILE: [X64NativeRegister; 8] = [
    X64NativeRegister::Rbx,
    X64NativeRegister::Rbp,
    X64NativeRegister::Rsi,
    X64NativeRegister::Rdi,
    X64NativeRegister::R12,
    X64NativeRegister::R13,
    X64NativeRegister::R14,
    X64NativeRegister::R15,
];

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PathEffectLivenessReport {
    pub assignments: BTreeMap<String, u64>,
    pub vm_stack_writes_removed: usize,
    pub dead_allocation_writes_removed: usize,
    pub dead_allocation_calls_removed: usize,
    pub observable_writes: usize,
    pub observable_calls: usize,
    pub memory_havocs: usize,
    pub nonvolatile_registers_preserved: bool,
    pub returns_to_original_caller: bool,
    pub issues: Vec<String>,
    pub ready_for_native_replacement: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct EntryEffectLivenessReport {
    pub schema_version: u32,
    pub paths: Vec<PathEffectLivenessReport>,
    pub domain_complete: bool,
    pub all_paths_reach_vm_exit: bool,
    pub ready_for_native_replacement: bool,
}

fn path_reports<'a>(
    family: &'a RecoveredEntryPathFamily,
    path: &'a RecoveredEntryPath,
) -> Vec<&'a SymbolicVmExecutionReport> {
    let mut reports = vec![&family.entry_profile.symbolic];
    for segment in &path.chain.segments {
        reports.extend(segment.symbolic.iter());
        reports.extend(segment.boundary_symbolic.iter());
        reports.extend(segment.transition_pop.iter());
    }
    reports
}

fn fresh_names(reports: &[&SymbolicVmExecutionReport]) -> BTreeSet<String> {
    reports
        .iter()
        .flat_map(|report| report.fresh_allocations.iter())
        .flat_map(|allocation| allocation.allocation.intrinsic_dependencies())
        .filter(|name| name.starts_with(FRESH_ALLOCATION_RETURN_INTRINSIC_PREFIX))
        .collect()
}

fn call_result_fresh_names(call: &SymbolicNativeCall) -> BTreeSet<String> {
    call.applied_effect
        .result
        .intrinsic_dependencies()
        .into_iter()
        .filter(|name| name.starts_with(FRESH_ALLOCATION_RETURN_INTRINSIC_PREFIX))
        .collect()
}

fn call_uses_name(call: &SymbolicNativeCall, name: &str) -> bool {
    call.invocation
        .target
        .value
        .intrinsic_dependencies()
        .contains(name)
        || call
            .invocation
            .arguments
            .iter()
            .any(|argument| argument.value.intrinsic_dependencies().contains(name))
}

fn allocation_escapes(
    name: &str,
    path: &RecoveredEntryPath,
    reports: &[&SymbolicVmExecutionReport],
) -> bool {
    let Some(exit) = path.native_exit.as_ref() else {
        return true;
    };
    let returns_to_original_caller = exit.return_address
        == Expr::register(vmpx_vm::BitWidth::W64, "native_return_address_initial");
    let output_registers = if returns_to_original_caller {
        WIN64_NONVOLATILE
            .into_iter()
            .chain([X64NativeRegister::Rax])
            .collect::<Vec<_>>()
    } else {
        X64NativeRegister::ALL.to_vec()
    };
    if output_registers.iter().any(|register| {
        exit.registers
            .get(register)
            .is_some_and(|value| value.intrinsic_dependencies().contains(name))
    }) || (!returns_to_original_caller
        && (exit.rflags.intrinsic_dependencies().contains(name)
            || exit.return_address.intrinsic_dependencies().contains(name)))
    {
        return true;
    }
    for report in reports {
        for write in &report.external_writes {
            let address_names = write.address.intrinsic_dependencies();
            if !address_names.contains(name)
                && (write.value.intrinsic_dependencies().contains(name)
                    || address_names.contains(name))
            {
                return true;
            }
        }
        for call in &report.native_calls {
            if !call_result_fresh_names(call).contains(name) && call_uses_name(call, name) {
                return true;
            }
        }
    }
    false
}

fn preserves_nonvolatile(path: &RecoveredEntryPath, issues: &mut Vec<String>) -> bool {
    let Some(exit) = path.native_exit.as_ref() else {
        issues.push("path has no exact native exit state".to_owned());
        return false;
    };
    let mut preserved = true;
    for register in WIN64_NONVOLATILE {
        let expected = Expr::register(vmpx_vm::BitWidth::W64, register.symbolic_name());
        if exit.registers.get(&register) != Some(&expected) {
            issues.push(format!(
                "native nonvolatile {register:?} is not proven equal to its entry value"
            ));
            preserved = false;
        }
    }
    preserved
}

fn audit_path(
    family: &RecoveredEntryPathFamily,
    path: &RecoveredEntryPath,
) -> PathEffectLivenessReport {
    let reports = path_reports(family, path);
    let fresh = fresh_names(&reports);
    let escaping = fresh
        .iter()
        .filter(|name| allocation_escapes(name, path, &reports))
        .cloned()
        .collect::<BTreeSet<_>>();
    let dead = fresh
        .difference(&escaping)
        .cloned()
        .collect::<BTreeSet<_>>();
    let mut vm_stack_writes_removed = 0_usize;
    let mut dead_allocation_writes_removed = 0_usize;
    let mut dead_allocation_calls_removed = 0_usize;
    let mut observable_writes = 0_usize;
    let mut observable_calls = 0_usize;
    let mut memory_havocs = 0_usize;
    let mut issues = Vec::new();

    for report in &reports {
        memory_havocs = memory_havocs.saturating_add(report.memory_havocs.len());
        for write in &report.external_writes {
            if symbolic_pointer_offset(&write.address, VSP_ROOT).is_some() {
                vm_stack_writes_removed = vm_stack_writes_removed.saturating_add(1);
            } else if write
                .address
                .intrinsic_dependencies()
                .iter()
                .any(|name| dead.contains(name))
            {
                dead_allocation_writes_removed = dead_allocation_writes_removed.saturating_add(1);
            } else {
                observable_writes = observable_writes.saturating_add(1);
            }
        }
        for call in &report.native_calls {
            let result_names = call_result_fresh_names(call);
            let dead_producer = !result_names.is_empty()
                && result_names.iter().all(|name| dead.contains(name))
                && matches!(
                    call.applied_effect.memory,
                    NativeCallMemoryDecision::PreserveTrackedAndCreateFreshAllocation { .. }
                );
            if dead_producer {
                dead_allocation_calls_removed = dead_allocation_calls_removed.saturating_add(1);
            } else {
                observable_calls = observable_calls.saturating_add(1);
            }
        }
    }
    let returns_to_original_caller = path.native_exit.as_ref().is_some_and(|exit| {
        exit.return_address
            == Expr::register(vmpx_vm::BitWidth::W64, "native_return_address_initial")
    });
    if path.native_exit.is_some() && !returns_to_original_caller {
        issues.push(
            "VM exit transfers to a native continuation instead of the original caller; continuation semantics require recovery"
                .to_owned(),
        );
    }
    if !escaping.is_empty() {
        issues.push(format!(
            "fresh allocations escape native outputs or observable effects: {escaping:?}"
        ));
    }
    if observable_writes != 0 {
        issues.push(format!(
            "{observable_writes} application-observable memory writes require lowering"
        ));
    }
    if observable_calls != 0 {
        issues.push(format!(
            "{observable_calls} application-observable native calls require lowering"
        ));
    }
    if memory_havocs != 0 {
        issues.push(format!(
            "{memory_havocs} opaque memory havocs prevent sound replacement"
        ));
    }
    let nonvolatile_registers_preserved = preserves_nonvolatile(path, &mut issues);
    let ready_for_native_replacement = path.native_exit.is_some()
        && issues.is_empty()
        && observable_writes == 0
        && observable_calls == 0
        && memory_havocs == 0
        && nonvolatile_registers_preserved;
    PathEffectLivenessReport {
        assignments: path.assignments.clone(),
        vm_stack_writes_removed,
        dead_allocation_writes_removed,
        dead_allocation_calls_removed,
        observable_writes,
        observable_calls,
        memory_havocs,
        nonvolatile_registers_preserved,
        returns_to_original_caller,
        issues,
        ready_for_native_replacement,
    }
}

/// Classifies effects retained by exact entry paths and proves whether a normal
/// Win64 replacement may discard VM-local stack state and non-escaping fresh
/// VM contexts. Unknown/global effects and ABI changes remain hard blockers.
#[must_use]
pub fn audit_entry_effect_liveness(family: &RecoveredEntryPathFamily) -> EntryEffectLivenessReport {
    let paths = family
        .paths
        .iter()
        .map(|path| audit_path(family, path))
        .collect::<Vec<_>>();
    let ready_for_native_replacement = family.domain_complete
        && family.all_paths_reach_vm_exit
        && !paths.is_empty()
        && paths.iter().all(|path| path.ready_for_native_replacement);
    EntryEffectLivenessReport {
        schema_version: 2,
        paths,
        domain_complete: family.domain_complete,
        all_paths_reach_vm_exit: family.all_paths_reach_vm_exit,
        ready_for_native_replacement,
    }
}
