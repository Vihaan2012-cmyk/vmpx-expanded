use std::collections::{BTreeMap, BTreeSet, VecDeque};

use serde::Serialize;
use thiserror::Error;
use vmpx_pe::PeImage;

use crate::{
    Expr, ExprError, ExprKind, SymbolicDomainError, SymbolicDomainLimits, SymbolicDomainReport,
    VmProfileChainReport, VmProfileChainStop, VmProfileExecutionState, VmProfileExplorerError,
    VmProfileExplorerLimits, enumerate_symbolic_value, explore_vm_profile_chain,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VmProfilePathLimits {
    pub max_paths: usize,
    pub domain: SymbolicDomainLimits,
    pub chain: VmProfileExplorerLimits,
}

impl Default for VmProfilePathLimits {
    fn default() -> Self {
        Self {
            max_paths: 4_096,
            domain: SymbolicDomainLimits::default(),
            chain: VmProfileExplorerLimits::default(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct VmProfilePathReport {
    pub assignments: BTreeMap<String, u64>,
    pub chain: VmProfileChainReport,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct VmProfilePathFamilyReport {
    pub schema_version: u32,
    pub root_chain: VmProfileChainReport,
    pub paths: Vec<VmProfilePathReport>,
    pub enumerations: Vec<SymbolicDomainReport>,
    pub domain_complete: bool,
    pub all_paths_reach_vm_exit: bool,
    pub unresolved_dependencies: Vec<String>,
    pub total_explored_instructions: usize,
    pub diagnostics: Vec<String>,
}

#[derive(Debug, Error)]
pub enum VmProfilePathExplorerError {
    #[error("VM profile exploration failed: {0}")]
    Profile(#[from] VmProfileExplorerError),

    #[error("finite symbolic-domain analysis failed: {0}")]
    Domain(#[from] SymbolicDomainError),

    #[error("symbolic path specialization failed: {0}")]
    Expression(#[from] ExprError),

    #[error("symbolic transition stop has no retained target expression")]
    MissingTransitionTarget,

    #[error("domain assignment for {name}={value} has no matching constant expression")]
    MissingDomainValue { name: String, value: u64 },

    #[error("finite path exploration exceeded path limit {limit}")]
    PathLimit { limit: usize },

    #[error("finite path exploration did not add a new constraint")]
    NonProgress,
}

fn transition_target(chain: &VmProfileChainReport) -> Option<&Expr> {
    chain
        .segments
        .last()
        .and_then(|segment| segment.transition_target.as_ref())
}

fn assignment_replacements(
    domains: &BTreeMap<String, Vec<Expr>>,
    assignments: &BTreeMap<String, u64>,
) -> Result<BTreeMap<String, Expr>, VmProfilePathExplorerError> {
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
                .ok_or_else(|| VmProfilePathExplorerError::MissingDomainValue {
                    name: name.clone(),
                    value: *value,
                })?;
            Ok((name.clone(), expression))
        })
        .collect()
}

fn merge_assignments(
    current: &BTreeMap<String, u64>,
    additional: &BTreeMap<String, u64>,
) -> Option<BTreeMap<String, u64>> {
    let mut merged = current.clone();
    for (name, value) in additional {
        if merged.get(name).is_some_and(|existing| existing != value) {
            return None;
        }
        merged.insert(name.clone(), *value);
    }
    Some(merged)
}

/// Explores every profile chain made concrete by explicitly supplied finite
/// symbolic-input domains. Each leaf retains its exact assignments and a full
/// independently re-executed chain; paths are never merged by discarding VM
/// decoder or symbolic state.
///
/// Missing domains remain an incomplete structured result. Resource limits,
/// malformed domains, and non-progressing specializations are errors.
///
/// # Errors
///
/// Returns a structured error when native/profile analysis fails, a finite
/// domain is malformed or exceeds its bounds, specialization violates an
/// expression width, or path enumeration cannot make bounded progress.
pub fn explore_vm_profile_paths(
    pe: &PeImage<'_>,
    initial: &VmProfileExecutionState,
    domains: &BTreeMap<String, Vec<Expr>>,
    limits: VmProfilePathLimits,
) -> Result<VmProfilePathFamilyReport, VmProfilePathExplorerError> {
    let root_chain = explore_vm_profile_chain(pe, initial.clone(), limits.chain)?;
    let mut total_explored_instructions = root_chain.total_instructions;
    let mut paths = Vec::new();
    let mut enumerations = Vec::new();
    let mut unresolved_dependencies = BTreeSet::new();
    let mut diagnostics = Vec::new();
    let mut pending = VecDeque::new();
    let mut seen = BTreeSet::new();

    if root_chain.stop_reason == VmProfileChainStop::SymbolicTransitionTarget {
        pending.push_back(BTreeMap::new());
        seen.insert(BTreeMap::new());
    } else {
        paths.push(VmProfilePathReport {
            assignments: BTreeMap::new(),
            chain: root_chain.clone(),
        });
    }

    while let Some(assignments) = pending.pop_front() {
        if paths.len().saturating_add(pending.len()) >= limits.max_paths {
            return Err(VmProfilePathExplorerError::PathLimit {
                limit: limits.max_paths,
            });
        }
        let replacements = assignment_replacements(domains, &assignments)?;
        let specialized = VmProfileExecutionState {
            decoder: initial.decoder,
            profile: initial.profile,
            symbolic: initial.symbolic.substitute(&replacements)?,
        };
        let chain = if assignments.is_empty() {
            root_chain.clone()
        } else {
            let chain = explore_vm_profile_chain(pe, specialized, limits.chain)?;
            total_explored_instructions =
                total_explored_instructions.saturating_add(chain.total_instructions);
            chain
        };

        if chain.stop_reason != VmProfileChainStop::SymbolicTransitionTarget {
            paths.push(VmProfilePathReport { assignments, chain });
            continue;
        }

        let target =
            transition_target(&chain).ok_or(VmProfilePathExplorerError::MissingTransitionTarget)?;
        let enumeration = enumerate_symbolic_value(target, domains, limits.domain)?;
        if !enumeration.complete {
            unresolved_dependencies.extend(enumeration.unresolved_dependencies.iter().cloned());
            diagnostics.push(format!(
                "path {assignments:?} lacks finite domains for {:?}",
                enumeration.unresolved_dependencies
            ));
            enumerations.push(enumeration);
            paths.push(VmProfilePathReport { assignments, chain });
            continue;
        }
        let alternatives = enumeration.alternatives.clone();
        enumerations.push(enumeration);
        let mut progressed = false;
        for alternative in alternatives {
            let Some(merged) = merge_assignments(&assignments, &alternative.assignments) else {
                continue;
            };
            if merged == assignments {
                continue;
            }
            progressed = true;
            if seen.insert(merged.clone()) {
                pending.push_back(merged);
            }
        }
        if !progressed {
            return Err(VmProfilePathExplorerError::NonProgress);
        }
    }

    paths.sort_by(|left, right| left.assignments.cmp(&right.assignments));
    let domain_complete = unresolved_dependencies.is_empty()
        && paths
            .iter()
            .all(|path| path.chain.stop_reason != VmProfileChainStop::SymbolicTransitionTarget);
    let all_paths_reach_vm_exit = !paths.is_empty()
        && paths
            .iter()
            .all(|path| path.chain.stop_reason == VmProfileChainStop::VmExit);
    Ok(VmProfilePathFamilyReport {
        schema_version: 1,
        root_chain,
        paths,
        enumerations,
        domain_complete,
        all_paths_reach_vm_exit,
        unresolved_dependencies: unresolved_dependencies.into_iter().collect(),
        total_explored_instructions,
        diagnostics,
    })
}

#[cfg(test)]
mod tests {
    use crate::BitWidth;

    use super::*;

    #[test]
    fn assignment_replacements_preserve_domain_widths() {
        let domains = BTreeMap::from([(
            "selector".to_owned(),
            vec![
                Expr::constant(BitWidth::W16, 0),
                Expr::constant(BitWidth::W16, 2),
            ],
        )]);
        let replacements =
            assignment_replacements(&domains, &BTreeMap::from([("selector".to_owned(), 2)]))
                .expect("domain value exists");
        assert_eq!(replacements["selector"].width, BitWidth::W16);
    }

    #[test]
    fn conflicting_assignments_are_rejected_without_guessing() {
        let left = BTreeMap::from([("selector".to_owned(), 1)]);
        let right = BTreeMap::from([("selector".to_owned(), 2)]);
        assert!(merge_assignments(&left, &right).is_none());
    }
}
