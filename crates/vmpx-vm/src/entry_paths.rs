use std::collections::BTreeMap;

use serde::Serialize;
use thiserror::Error;
use vmpx_pe::{PeImage, Va};

use crate::{
    DispatchExtractionCandidate, DomainCertificate, EntryProfileRecoveryError,
    EntryProfileRecoveryLimits, Expr, ExprError, ExprKind, InputDomainLimits, InputDomainRefusal,
    NativeExitState, NativeExitStateError, RecoveredEntryProfile, SymbolicDomainError,
    SymbolicDomainLimits, SymbolicDomainReport, VmEntryCandidate, VmProfileChainReport,
    VmProfileChainStop, VmProfileExecutionState, VmProfileExplorerError, VmProfileExplorerLimits,
    enumerate_symbolic_value, explore_vm_profile_chain, infer_branch_input_domains,
    recover_entry_profile, recover_native_exit_state,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EntryPathRecoveryLimits {
    pub entry: EntryProfileRecoveryLimits,
    pub domain: SymbolicDomainLimits,
    pub chain: VmProfileExplorerLimits,
    pub max_paths: usize,
}

impl Default for EntryPathRecoveryLimits {
    fn default() -> Self {
        Self {
            entry: EntryProfileRecoveryLimits::default(),
            domain: SymbolicDomainLimits::default(),
            chain: VmProfileExplorerLimits::default(),
            max_paths: 4_096,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RecoveredEntryPath {
    pub assignments: BTreeMap<String, u64>,
    pub chain: VmProfileChainReport,
    pub native_exit: Option<NativeExitState>,
}

#[derive(Clone, Debug)]
pub struct RecoveredEntryPathFamily {
    pub entry_profile: RecoveredEntryProfile,
    pub branch_enumeration: SymbolicDomainReport,
    pub paths: Vec<RecoveredEntryPath>,
    pub domain_complete: bool,
    pub all_paths_reach_vm_exit: bool,
}

#[derive(Clone, Debug, Eq, Error, PartialEq, Serialize)]
pub enum EntryPathRecoveryError {
    #[error("entry profile recovery failed: {0}")]
    Entry(String),
    #[error("entry profile has no symbolic branch target")]
    MissingBranchTarget,
    #[error("entry branch has {count} dispatch candidates; exactly one is required")]
    AmbiguousBranchDispatch { count: usize },
    #[error("entry branch dispatch does not establish a handler anchor")]
    MissingHandlerAnchor,
    #[error("entry branch domain enumeration failed: {0}")]
    Domain(String),
    #[error("entry branch enumerated {count} paths, exceeding limit {limit}")]
    PathLimit { count: usize, limit: usize },
    #[error("domain assignment {name}={value} has no matching constant expression")]
    MissingDomainValue { name: String, value: u64 },
    #[error("entry path substitution failed: {0}")]
    Expression(String),
    #[error("entry successor dispatch failed: {0}")]
    Dispatch(String),
    #[error("entry successor profile exploration failed: {0}")]
    Profile(String),
    #[error("VM-exit state recovery failed: {0}")]
    NativeExit(String),
    #[error("path reached VM exit without a retained terminal symbolic state/layout")]
    MissingNativeExitEvidence,
}

impl From<EntryProfileRecoveryError> for EntryPathRecoveryError {
    fn from(error: EntryProfileRecoveryError) -> Self {
        Self::Entry(error.to_string())
    }
}

impl From<SymbolicDomainError> for EntryPathRecoveryError {
    fn from(error: SymbolicDomainError) -> Self {
        Self::Domain(error.to_string())
    }
}

impl From<ExprError> for EntryPathRecoveryError {
    fn from(error: ExprError) -> Self {
        Self::Expression(error.to_string())
    }
}

impl From<crate::BytecodeDecodeError> for EntryPathRecoveryError {
    fn from(error: crate::BytecodeDecodeError) -> Self {
        Self::Dispatch(error.to_string())
    }
}

impl From<VmProfileExplorerError> for EntryPathRecoveryError {
    fn from(error: VmProfileExplorerError) -> Self {
        Self::Profile(error.to_string())
    }
}

impl From<NativeExitStateError> for EntryPathRecoveryError {
    fn from(error: NativeExitStateError) -> Self {
        Self::NativeExit(error.to_string())
    }
}

fn replacements_for(
    domains: &BTreeMap<String, Vec<Expr>>,
    assignments: &BTreeMap<String, u64>,
) -> Result<BTreeMap<String, Expr>, EntryPathRecoveryError> {
    assignments
        .iter()
        .map(|(name, value)| {
            let expression = domains
                .get(name)
                .and_then(|domain| {
                    domain.iter().find(|candidate| {
                        matches!(candidate.kind, ExprKind::Const { value: candidate } if candidate == *value)
                    })
                })
                .cloned()
                .ok_or_else(|| EntryPathRecoveryError::MissingDomainValue {
                    name: name.clone(),
                    value: *value,
                })?;
            Ok((name.clone(), expression))
        })
        .collect()
}

fn extract_native_exit(
    chain: &VmProfileChainReport,
) -> Result<Option<NativeExitState>, EntryPathRecoveryError> {
    if chain.stop_reason != VmProfileChainStop::VmExit {
        return Ok(None);
    }
    let terminal = chain
        .segments
        .last()
        .ok_or(EntryPathRecoveryError::MissingNativeExitEvidence)?;
    let symbolic = terminal
        .symbolic
        .as_ref()
        .ok_or(EntryPathRecoveryError::MissingNativeExitEvidence)?;
    let exit = terminal
        .decode
        .handler_recoveries
        .get(&terminal.decode.decode.final_state.handler)
        .and_then(|handler| handler.vm_exit.as_ref())
        .ok_or(EntryPathRecoveryError::MissingNativeExitEvidence)?;
    Ok(Some(recover_native_exit_state(symbolic, exit)?))
}

/// The unique dispatch candidate and handler anchor that decode a branch target.
fn branch_dispatch(
    entry_profile: &RecoveredEntryProfile,
) -> Result<(&DispatchExtractionCandidate, Va), EntryPathRecoveryError> {
    let boundary = entry_profile
        .decode
        .handler_recoveries
        .get(&entry_profile.decode.decode.final_state.handler)
        .ok_or(EntryPathRecoveryError::AmbiguousBranchDispatch { count: 0 })?;
    let [dispatch] = boundary.dispatch_analysis.candidates.as_slice() else {
        return Err(EntryPathRecoveryError::AmbiguousBranchDispatch {
            count: boundary.dispatch_analysis.candidates.len(),
        });
    };
    let handler_anchor = dispatch
        .handler_rebase
        .ok_or(EntryPathRecoveryError::MissingHandlerAnchor)?;
    Ok((dispatch, handler_anchor))
}

/// Decodes the bytecode at `value` and explores the chain from `symbolic`.
fn explore_branch(
    pe: &PeImage<'_>,
    entry_profile: &RecoveredEntryProfile,
    dispatch: &DispatchExtractionCandidate,
    handler_anchor: Va,
    value: u64,
    symbolic: crate::SymbolicVmInitialState,
    limits: VmProfileExplorerLimits,
) -> Result<(VmProfileChainReport, Option<NativeExitState>), EntryPathRecoveryError> {
    let first = dispatch.codec.decode(
        pe,
        crate::RelativeDispatchState {
            vip: Va(value),
            rolling_key: value,
            handler: handler_anchor,
        },
    )?;
    let chain = explore_vm_profile_chain(
        pe,
        VmProfileExecutionState {
            decoder: first.state_after,
            profile: entry_profile.profile,
            symbolic,
        },
        limits,
    )?;
    let native_exit = extract_native_exit(&chain)?;
    Ok((chain, native_exit))
}

/// Recovers every explicitly bounded entry path starting at the first virtual
/// branch. Each assignment is independently specialized before decoding and
/// retains its full profile chain and exact native exit state.
///
/// # Errors
///
/// Fails closed on incomplete domains, ambiguous branch dispatch, resource
/// limits, decode failures, or a VM-exit path lacking exact terminal evidence.
pub fn recover_entry_paths(
    pe: &PeImage<'_>,
    entry: &VmEntryCandidate,
    domains: &BTreeMap<String, Vec<Expr>>,
    limits: EntryPathRecoveryLimits,
) -> Result<RecoveredEntryPathFamily, EntryPathRecoveryError> {
    let entry_profile = recover_entry_profile(pe, entry, limits.entry)?;
    // An entry whose first profile ends in a virtual branch is specialized per
    // branch target. An entry that instead ends in a cross-profile transition
    // has no branch target at all: its successor state comes from the
    // transition, which the chain explorer recovers directly.
    let Some(branch_target) = entry_profile.symbolic.branch_target.as_ref() else {
        let symbolic = entry_profile.symbolic.continuation_state();
        let first = entry_profile.decode.decode.final_state.clone();
        let chain = explore_vm_profile_chain(
            pe,
            VmProfileExecutionState {
                decoder: first,
                profile: entry_profile.profile,
                symbolic,
            },
            limits.chain,
        )?;
        let native_exit = extract_native_exit(&chain)?;
        let assignments = BTreeMap::new();
        let path = RecoveredEntryPath {
            assignments: assignments.clone(),
            chain,
            native_exit,
        };
        return Ok(RecoveredEntryPathFamily {
            entry_profile,
            branch_enumeration: crate::SymbolicDomainReport {
                schema_version: 1,
                dependencies: Vec::new(),
                unresolved_dependencies: Vec::new(),
                combinations_evaluated: 1,
                unique_values: Vec::new(),
                alternatives: Vec::new(),
                complete: true,
            },
            domain_complete: true,
            all_paths_reach_vm_exit: path.native_exit.is_some(),
            paths: vec![path],
        });
    };
    let branch_enumeration = enumerate_symbolic_value(branch_target, domains, limits.domain)?;
    if branch_enumeration.alternatives.len() > limits.max_paths {
        return Err(EntryPathRecoveryError::PathLimit {
            count: branch_enumeration.alternatives.len(),
            limit: limits.max_paths,
        });
    }
    let (dispatch, handler_anchor) = branch_dispatch(&entry_profile)?;

    let mut paths = Vec::with_capacity(branch_enumeration.alternatives.len());
    for alternative in &branch_enumeration.alternatives {
        let replacements = replacements_for(domains, &alternative.assignments)?;
        let symbolic = entry_profile
            .symbolic
            .continuation_state()
            .substitute(&replacements)?;
        let (chain, native_exit) = explore_branch(
            pe,
            &entry_profile,
            dispatch,
            handler_anchor,
            alternative.value,
            symbolic,
            limits.chain,
        )?;
        paths.push(RecoveredEntryPath {
            assignments: alternative.assignments.clone(),
            chain,
            native_exit,
        });
    }
    let all_paths_reach_vm_exit =
        !paths.is_empty() && paths.iter().all(|path| path.native_exit.is_some());
    Ok(RecoveredEntryPathFamily {
        domain_complete: branch_enumeration.complete,
        all_paths_reach_vm_exit,
        entry_profile,
        branch_enumeration,
        paths,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BranchingRecoveryLimits {
    pub paths: EntryPathRecoveryLimits,
    pub input_domain: InputDomainLimits,
    /// Distinct branch-target values that may be explored.
    pub max_targets: usize,
}

impl Default for BranchingRecoveryLimits {
    fn default() -> Self {
        Self {
            paths: EntryPathRecoveryLimits::default(),
            input_domain: InputDomainLimits::default(),
            max_targets: 16,
        }
    }
}

/// Paths of a function whose first virtual branch depends on its live-in
/// registers, with a proven-exhaustive inferred input domain.
///
/// `family.paths[i]` is taken exactly when `branch_target == targets[i]`; the
/// targets are the complete image of `branch_target` over all register values.
/// Paths are explored from the unspecialized entry state (their `assignments`
/// are empty), so each exit stays an expression over the live-in registers and
/// holds for every input in its class, not just a witness.
#[derive(Clone, Debug)]
pub struct GuardedEntryPathFamily {
    pub family: RecoveredEntryPathFamily,
    pub branch_target: Expr,
    pub targets: Vec<u64>,
    pub certificates: BTreeMap<String, DomainCertificate>,
}

#[derive(Clone, Debug, Eq, Error, PartialEq, Serialize)]
pub enum BranchingRecoveryError {
    #[error(transparent)]
    Path(#[from] EntryPathRecoveryError),
    #[error("input domain is not provably finite: {0}")]
    Undecidable(#[from] InputDomainRefusal),
    #[error("branch has {count} distinct targets, exceeding limit {limit}")]
    TargetLimit { count: usize, limit: usize },
}

/// Recovers a function whose first virtual branch depends on its inputs, but
/// only when a finite domain for those inputs can be proven exhaustive.
///
/// A branch deeper than the first one stays symbolic, so its path never reaches
/// a VM exit and `all_paths_reach_vm_exit` is false.
///
/// # Errors
///
/// Fails closed on anything undecidable or beyond the limits.
pub fn recover_branching_entry_paths(
    pe: &PeImage<'_>,
    entry: &VmEntryCandidate,
    limits: BranchingRecoveryLimits,
) -> Result<GuardedEntryPathFamily, BranchingRecoveryError> {
    let entry_profile = recover_entry_profile(pe, entry, limits.paths.entry)
        .map_err(EntryPathRecoveryError::from)?;
    let branch_target = entry_profile
        .symbolic
        .branch_target
        .clone()
        .ok_or(EntryPathRecoveryError::MissingBranchTarget)?;
    let inferred = infer_branch_input_domains(&branch_target, limits.input_domain)?;
    let domain_limits = SymbolicDomainLimits {
        max_values_per_variable: limits
            .paths
            .domain
            .max_values_per_variable
            .max(limits.input_domain.max_values_per_variable),
        ..limits.paths.domain
    };
    let branch_enumeration =
        enumerate_symbolic_value(&branch_target, &inferred.domains, domain_limits)
            .map_err(EntryPathRecoveryError::from)?;
    let targets = branch_enumeration.unique_values.clone();
    if targets.len() > limits.max_targets {
        return Err(BranchingRecoveryError::TargetLimit {
            count: targets.len(),
            limit: limits.max_targets,
        });
    }
    let (dispatch, handler_anchor) = branch_dispatch(&entry_profile)?;
    let mut paths = Vec::with_capacity(targets.len());
    for target in &targets {
        let (chain, native_exit) = explore_branch(
            pe,
            &entry_profile,
            dispatch,
            handler_anchor,
            *target,
            entry_profile.symbolic.continuation_state(),
            limits.paths.chain,
        )?;
        paths.push(RecoveredEntryPath {
            assignments: BTreeMap::new(),
            chain,
            native_exit,
        });
    }
    let all_paths_reach_vm_exit =
        !paths.is_empty() && paths.iter().all(|path| path.native_exit.is_some());
    Ok(GuardedEntryPathFamily {
        family: RecoveredEntryPathFamily {
            domain_complete: branch_enumeration.complete,
            all_paths_reach_vm_exit,
            entry_profile,
            branch_enumeration,
            paths,
        },
        branch_target,
        targets,
        certificates: inferred.certificates,
    })
}
