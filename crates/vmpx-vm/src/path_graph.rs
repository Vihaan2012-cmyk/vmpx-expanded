use serde::Serialize;
use thiserror::Error;

use crate::{
    LinearDecodeStop, RelativeDispatchState, SymbolicVmInitialState, VmHandlerSemantic,
    VmProfileChainSegment, VmProfileChainStop, VmProfilePathFamilyReport, VmProfileReport,
};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
pub struct VmPathGraphNodeId(pub u32);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VmPathGraphEdgeKind {
    LinearContinuation,
    VirtualBranch,
    NativeCallReturn,
    NativeProfileTransition,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct VmPathGraphOccurrence {
    pub path_index: usize,
    pub segment_index: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct VmPathGraphNode {
    pub id: VmPathGraphNodeId,
    pub initial_decoder_state: RelativeDispatchState,
    pub profile: VmProfileReport,
    pub symbolic_initial: SymbolicVmInitialState,
    pub decoded_instructions: usize,
    pub occurrences: Vec<VmPathGraphOccurrence>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct VmPathGraphEdge {
    pub source: VmPathGraphNodeId,
    pub target: VmPathGraphNodeId,
    pub kind: VmPathGraphEdgeKind,
    pub path_indices: Vec<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct VmPathGraphTerminal {
    pub path_index: usize,
    pub source: Option<VmPathGraphNodeId>,
    pub stop_reason: VmProfileChainStop,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct VmPathGraphReport {
    pub schema_version: u32,
    pub nodes: Vec<VmPathGraphNode>,
    pub edges: Vec<VmPathGraphEdge>,
    pub entry_nodes: Vec<VmPathGraphNodeId>,
    pub terminals: Vec<VmPathGraphTerminal>,
    pub path_count: usize,
    pub segment_occurrences: usize,
    pub exactly_merged_occurrences: usize,
    pub domain_complete: bool,
    pub all_paths_reach_vm_exit: bool,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum VmPathGraphError {
    #[error("path graph node count exceeds u32")]
    NodeCountOverflow,
}

fn boundary_semantic(segment: &VmProfileChainSegment) -> Option<VmHandlerSemantic> {
    (segment.decode.decode.stop_reason == LinearDecodeStop::ControlFlowBoundary)
        .then(|| {
            segment
                .decode
                .handler_recoveries
                .get(&segment.decode.decode.final_state.handler)
        })
        .flatten()
        .and_then(|recovery| recovery.handler_analysis.as_ref())
        .map(|analysis| analysis.semantic)
}

fn edge_kind(segment: &VmProfileChainSegment) -> VmPathGraphEdgeKind {
    if segment.boundary_decode.is_some() {
        VmPathGraphEdgeKind::NativeCallReturn
    } else if segment.transition.is_some() {
        VmPathGraphEdgeKind::NativeProfileTransition
    } else if boundary_semantic(segment).is_some_and(|semantic| {
        matches!(
            semantic,
            VmHandlerSemantic::ConditionalBranch | VmHandlerSemantic::UnconditionalBranch
        )
    }) {
        VmPathGraphEdgeKind::VirtualBranch
    } else {
        VmPathGraphEdgeKind::LinearContinuation
    }
}

fn same_exact_entry_state(node: &VmPathGraphNode, segment: &VmProfileChainSegment) -> bool {
    node.initial_decoder_state == segment.initial_decoder_state
        && node.profile == segment.profile
        && node.symbolic_initial == segment.symbolic_initial
        && node.decoded_instructions == segment.decode.decode.instructions.len()
}

fn node_for_segment(
    nodes: &mut Vec<VmPathGraphNode>,
    path_index: usize,
    segment: &VmProfileChainSegment,
) -> Result<VmPathGraphNodeId, VmPathGraphError> {
    if let Some(node) = nodes
        .iter_mut()
        .find(|node| same_exact_entry_state(node, segment))
    {
        node.occurrences.push(VmPathGraphOccurrence {
            path_index,
            segment_index: segment.index,
        });
        return Ok(node.id);
    }
    let id = VmPathGraphNodeId(
        u32::try_from(nodes.len()).map_err(|_| VmPathGraphError::NodeCountOverflow)?,
    );
    nodes.push(VmPathGraphNode {
        id,
        initial_decoder_state: segment.initial_decoder_state,
        profile: segment.profile.clone(),
        symbolic_initial: segment.symbolic_initial.clone(),
        decoded_instructions: segment.decode.decode.instructions.len(),
        occurrences: vec![VmPathGraphOccurrence {
            path_index,
            segment_index: segment.index,
        }],
    });
    Ok(id)
}

fn add_edge(
    edges: &mut Vec<VmPathGraphEdge>,
    source: VmPathGraphNodeId,
    target: VmPathGraphNodeId,
    kind: VmPathGraphEdgeKind,
    path_index: usize,
) {
    if let Some(edge) = edges
        .iter_mut()
        .find(|edge| edge.source == source && edge.target == target && edge.kind == kind)
    {
        if !edge.path_indices.contains(&path_index) {
            edge.path_indices.push(path_index);
        }
    } else {
        edges.push(VmPathGraphEdge {
            source,
            target,
            kind,
            path_indices: vec![path_index],
        });
    }
}

/// Assembles an exact-state structural CFG from a finite profile-path family.
/// Nodes are shared only when decoder state, profile roles/direction, complete
/// symbolic entry state, and decoded block length are equal. Structurally equal
/// bytecode reached with different symbolic state remains separate.
///
/// # Errors
///
/// Returns an error if the bounded in-memory node set cannot be represented by
/// the report's 32-bit node identifiers.
pub fn assemble_profile_path_graph(
    family: &VmProfilePathFamilyReport,
) -> Result<VmPathGraphReport, VmPathGraphError> {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let mut entry_nodes = Vec::new();
    let mut terminals = Vec::new();
    let mut segment_occurrences = 0_usize;

    for (path_index, path) in family.paths.iter().enumerate() {
        let mut previous = None;
        for segment in &path.chain.segments {
            segment_occurrences = segment_occurrences.saturating_add(1);
            let current = node_for_segment(&mut nodes, path_index, segment)?;
            if let Some((previous_id, previous_segment)) = previous {
                add_edge(
                    &mut edges,
                    previous_id,
                    current,
                    edge_kind(previous_segment),
                    path_index,
                );
            } else if !entry_nodes.contains(&current) {
                entry_nodes.push(current);
            }
            previous = Some((current, segment));
        }
        terminals.push(VmPathGraphTerminal {
            path_index,
            source: previous.map(|(node, _)| node),
            stop_reason: path.chain.stop_reason,
        });
    }

    for edge in &mut edges {
        edge.path_indices.sort_unstable();
    }
    entry_nodes.sort();
    terminals.sort_by_key(|terminal| terminal.path_index);
    let exactly_merged_occurrences = segment_occurrences.saturating_sub(nodes.len());
    Ok(VmPathGraphReport {
        schema_version: 1,
        nodes,
        edges,
        entry_nodes,
        terminals,
        path_count: family.paths.len(),
        segment_occurrences,
        exactly_merged_occurrences,
        domain_complete: family.domain_complete,
        all_paths_reach_vm_exit: family.all_paths_reach_vm_exit,
    })
}
