mod bootstrap;
mod call_effects;
mod checksum;
mod checksum_handler;
mod codec_recovery;
mod decoder;
mod dispatch_extract;
mod effect_extract;
mod entry_paths;
mod entry_profile;
mod exception_flow;
mod expr;
mod handler_slice;
mod image_concretize;
mod model;
mod native_api;
mod native_call;
mod native_call_verify;
mod native_context;
mod native_continuation;
mod native_exit_state;
mod native_vm_chain;
mod oep;
mod operand_extract;
mod path_graph;
mod profile_explore;
mod profile_paths;
mod semantic_exec;
mod symbolic_domain;
mod transform;
mod transition_extract;
mod value_flow;
mod vm_exit;

pub use bootstrap::{
    BootstrapDispatchCandidateReport, BootstrapDispatchRecovery, BootstrapDispatchReport,
    BootstrapRecoveryError, recover_bootstrap_dispatch,
};
pub use call_effects::{
    AppliedNativeCallEffect, FRESH_ALLOCATION_RETURN_INTRINSIC_PREFIX, NativeCallArgumentAlias,
    NativeCallArgumentDescriptor, NativeCallArgumentLocation, NativeCallEffectError,
    NativeCallEffectSummary, NativeCallEpoch, NativeCallEpochAllocator, NativeCallImportDescriptor,
    NativeCallImportMatchKind, NativeCallImportResolution, NativeCallInvocationDescriptor,
    NativeCallMemoryDecision, NativeCallMemoryEffect, NativeCallResultEffect,
    NativeCallTargetDescriptor, NativeCallValueSource, OPAQUE_CALL_RETURN_INTRINSIC_PREFIX,
    apply_native_call_effect, fresh_allocation_return_intrinsic_name,
    opaque_call_return_intrinsic_name, resolve_native_call_import,
};
pub use checksum::{
    ChecksumAccumulatorSource, ChecksumCombine, ChecksumShift, ChecksumTableIndex,
    TableChecksumMetadata, TableChecksumMetadataError, VmSemanticMetadata,
};
pub use checksum_handler::{
    RecoveredTableChecksumHandler, TableChecksumHandlerAnalysis, TableChecksumHandlerReport,
    analyze_table_checksum_handler,
};
pub use codec_recovery::{
    HandlerCodecRecoveryConfig, HandlerCodecRecoveryReport, LinearHandlerRecoveryError,
    LinearHandlerRecoveryReport, ResolvedVmProfileRoles, VmProfileRoleCandidateReport,
    VmProfileRoleResolution, VmProfileRoleResolutionConfig, VmProfileRoleResolutionReport,
    recover_and_decode_linear_vm_stream, recover_handler_codecs, resolve_vm_profile_roles,
};
pub use decoder::{
    BytecodeDecodeError, DecodedDispatch, DecodedOperandDispatch, DecodedVmInstruction,
    KeyOperation, KeyedDecodeResult, KeyedTransformCodec, LinearDecodeStop, LinearVmDecodeReport,
    RelativeDispatchCodec, RelativeDispatchState, RelativeThreadedCodec, VmHandlerDecodeSpec,
    decode_linear_vm_stream, decode_linear_vm_stream_direction,
};
pub use dispatch_extract::{
    DispatchExtractionCandidate, DispatchExtractionRejection, DispatchExtractionReport,
    HandlerRegisterRoles, extract_relative_dispatch_codecs,
    extract_relative_dispatch_codecs_with_postlude,
    extract_relative_dispatch_codecs_with_postlude_direction,
};
pub use effect_extract::{
    HandlerEffectRoles, HandlerEffectSummary, HandlerMemoryEffect, HandlerMemoryRegion,
    HandlerResizeEffect, HandlerVspAdjustment, classify_handler_effects, extract_handler_effects,
};
pub use entry_paths::{
    EntryPathRecoveryError, EntryPathRecoveryLimits, RecoveredEntryPath, RecoveredEntryPathFamily,
    recover_entry_paths,
};
pub use entry_profile::{
    EntryProfileRecoveryError, EntryProfileRecoveryLimits, RecoveredEntryProfile,
    recover_entry_profile,
};
pub use exception_flow::{
    ExceptionContinuationResolution, ExceptionContinuationSource, ExceptionStackSledReport,
    VmFaultUnwindCoverage, VmMemoryExceptionCoverageReport, analyze_exception_stack_sled,
    analyze_vm_memory_exception_coverage, analyze_vm_memory_exception_coverage_with_state,
};
pub use expr::{BinaryOp, Expr, ExprError, ExprKind, ResizeOp, UnaryOp};
pub use handler_slice::{
    HandlerSliceConfig, NativeBackwardSlice, NativeInstructionDisposition, NativeSliceInstruction,
    NativeSliceReason, backward_slice_handler_trace,
};
pub use image_concretize::{
    ImageConcretizationError, ImageConcretizationLimits, ImageConcretizationReport,
    ModuloImageTargetError, ModuloImageTargetLimits, ModuloImageTargetReport, RecoveredImageTarget,
    concretize_with_immutable_image, recover_modulo_image_targets,
};
pub use model::{
    BytecodeDirection, EffectKnowledge, RecoveryStatus, StateValue, VmContextLayout, VmDecodeState,
    VmHandlerAnalysis, VmHandlerSemantic, VmInstruction,
};
pub use native_api::summarize_known_import;
pub use native_call::{
    DecodedNativeCallBoundary, NativeCallAnalysis, NativeCallAnalysisReport,
    NativeCallBoundaryDecodeError, NativeCallBoundarySpec, RecoveredNativeCall,
    analyze_native_call_handler,
};
pub use native_call_verify::NativeCallSelectorOutcome;
pub use native_context::{
    NativeContextBinding, NativeContextMappingReport, NativeFrameSlot, NativeFrameSource,
    X64NativeFrameReport, X64NativeRegister, analyze_x64_native_frame, map_native_context_scatter,
};
pub use native_continuation::{
    NativeContinuationLimits, NativeContinuationReport, NativeObservedCString, NativeObservedCall,
    NativeObservedValue, analyze_native_continuation,
};
pub use native_exit_state::{
    NativeExitState, NativeExitStateError, recover_native_exit_state,
    recover_profile_chain_native_exit_state,
};
pub use native_vm_chain::{
    NativeVmEntryContinuation, NativeVmEntryContinuationError, continue_through_native_vm_entry,
};
pub use oep::{OepDecision, decide_oep};
pub use operand_extract::{
    KeyedOperandCodecCandidate, KeyedOperandCodecRejection, KeyedOperandCodecReport,
    extract_keyed_operand_codecs, extract_keyed_operand_codecs_direction,
};
pub use path_graph::{
    VmPathGraphEdge, VmPathGraphEdgeKind, VmPathGraphError, VmPathGraphNode, VmPathGraphNodeId,
    VmPathGraphOccurrence, VmPathGraphReport, VmPathGraphTerminal, assemble_profile_path_graph,
};
pub use profile_explore::{
    VmProfile, VmProfileChainReport, VmProfileChainSegment, VmProfileChainStop,
    VmProfileExecutionState, VmProfileExplorerError, VmProfileExplorerLimits, VmProfileReport,
    explore_vm_profile_chain,
};
pub use profile_paths::{
    VmProfilePathExplorerError, VmProfilePathFamilyReport, VmProfilePathLimits,
    VmProfilePathReport, explore_vm_profile_paths,
};
pub use semantic_exec::{
    SymbolicExecutionError, SymbolicExecutionLimits, SymbolicExecutionStop,
    SymbolicFreshAllocation, SymbolicMemoryHavoc, SymbolicMemoryReadError, SymbolicMemorySource,
    SymbolicMemoryWrite, SymbolicNativeCall, SymbolicVmExecutionReport, SymbolicVmInitialState,
    canonicalize_symbolic_pointer, execute_symbolic_linear_block,
    execute_symbolic_linear_block_with_initial_state,
    execute_symbolic_native_call_boundary_with_initial_state, symbolic_pointer_offset,
};
pub use symbolic_domain::{
    SymbolicConcreteAlternative, SymbolicDomainError, SymbolicDomainLimits, SymbolicDomainReport,
    enumerate_symbolic_value,
};
pub use transform::{BitWidth, Transform, TransformChain, TransformError, TransformOp};
pub use transition_extract::{
    RecoveredVmTransition, VmTransitionAnalysis, VmTransitionReport, analyze_vm_transition,
};
pub use value_flow::{HandlerValueFlowProof, prove_handler_value_flow};
pub use vm_exit::{VmExitReport, VmExitRestoreSlot, analyze_vm_exit};

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};

use iced_x86::{FlowControl, Mnemonic, OpKind, Register};
use serde::Serialize;
use thiserror::Error;
use vmpx_disasm::{DecodedInstruction, decode_one, is_executable_rva};
use vmpx_pe::{PeImage, Rva, Va};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VmEntrySource {
    PeEntryPoint,
    TlsCallback,
    ExecutableScan,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VmEntryReachability {
    LoaderReachable,
    UnprovenExecutableCandidate,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct VmEntryEvidence {
    pub weight: u8,
    pub statement: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct VmEntryCandidate {
    pub rva: Rva,
    pub va: Va,
    pub source: VmEntrySource,
    pub reachability: VmEntryReachability,
    pub confidence: u8,
    pub seed: Option<u64>,
    pub call_target: Option<Rva>,
    pub bootstrap_rva: Option<Rva>,
    pub evidence: Vec<VmEntryEvidence>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct VmInstance {
    pub id: usize,
    pub bootstrap_rva: Rva,
    pub entry_rvas: Vec<Rva>,
    pub distinct_seeds: Vec<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct VmDiscoveryReport {
    pub schema_version: u32,
    pub scanned_stub_candidates: usize,
    pub resource_limited: bool,
    pub entries: Vec<VmEntryCandidate>,
    pub instances: Vec<VmInstance>,
    pub rejected_roots: Vec<VmEntryCandidate>,
    pub diagnostics: Vec<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VmDiscoveryLimits {
    pub scan_executable_stubs: bool,
    pub max_stub_candidates: usize,
    pub max_chain_fragments: usize,
    pub max_chain_instructions: usize,
    pub minimum_confidence: u8,
}

impl Default for VmDiscoveryLimits {
    fn default() -> Self {
        Self {
            scan_executable_stubs: true,
            max_stub_candidates: 25_000,
            max_chain_fragments: 512,
            max_chain_instructions: 20_000,
            minimum_confidence: 60,
        }
    }
}

#[derive(Clone, Debug, Error)]
pub enum VmDiscoveryError {
    #[error("VM discovery address error: {0}")]
    Address(String),
}

#[derive(Clone, Debug, Default)]
#[allow(clippy::struct_excessive_bools)]
struct ChainFeatures {
    pushed_registers: BTreeSet<String>,
    pushes_flags: bool,
    stack_allocation: Option<(Rva, u64)>,
    stack_alignment: bool,
    high_base_immediate: bool,
    memory_fetches: usize,
    indirect_dispatch: bool,
    decoded_instructions: usize,
    resource_limited: bool,
}

fn immediate(instruction: &DecodedInstruction) -> Option<u64> {
    match instruction.instruction.op0_kind() {
        OpKind::Immediate8 => Some(u64::from(instruction.instruction.immediate8())),
        OpKind::Immediate8to16 => Some(u64::from(
            instruction.instruction.immediate8to16().cast_unsigned(),
        )),
        OpKind::Immediate8to32 => Some(u64::from(
            instruction.instruction.immediate8to32().cast_unsigned(),
        )),
        OpKind::Immediate8to64 => Some(instruction.instruction.immediate8to64().cast_unsigned()),
        OpKind::Immediate16 => Some(u64::from(instruction.instruction.immediate16())),
        OpKind::Immediate32 => Some(u64::from(instruction.instruction.immediate32())),
        OpKind::Immediate32to64 => Some(instruction.instruction.immediate32to64().cast_unsigned()),
        OpKind::Immediate64 => Some(instruction.instruction.immediate64()),
        _ => None,
    }
}

fn register_is_stack_pointer(register: Register) -> bool {
    matches!(register, Register::RSP | Register::ESP | Register::SP)
}

#[allow(clippy::too_many_lines)]
fn analyze_chain(pe: &PeImage<'_>, target: Rva, limits: VmDiscoveryLimits) -> ChainFeatures {
    let mut features = ChainFeatures::default();
    let mut queue = VecDeque::from([target]);
    let mut starts = BTreeSet::from([target]);
    let mut decoded_addresses = BTreeSet::new();

    while let Some(start) = queue.pop_front() {
        if starts.len() > limits.max_chain_fragments
            || features.decoded_instructions >= limits.max_chain_instructions
        {
            features.resource_limited = true;
            break;
        }
        let mut current = start;
        loop {
            if features.decoded_instructions >= limits.max_chain_instructions {
                features.resource_limited = true;
                break;
            }
            if !decoded_addresses.insert(current) {
                break;
            }
            let Ok(decoded) = decode_one(pe, current) else {
                break;
            };
            features.decoded_instructions += 1;
            let instruction = &decoded.instruction;
            if instruction.mnemonic() == Mnemonic::Push
                && instruction.op0_kind() == OpKind::Register
            {
                features
                    .pushed_registers
                    .insert(format!("{:?}", instruction.op0_register()));
            }
            if instruction.mnemonic() == Mnemonic::Pushfq {
                features.pushes_flags = true;
            }
            if instruction.mnemonic() == Mnemonic::Sub
                && register_is_stack_pointer(instruction.op0_register())
                && let Some(amount) = immediate_for_operand(instruction, 1)
                && (0x80..=0x1000).contains(&amount)
            {
                features.stack_allocation.get_or_insert((current, amount));
            }
            if instruction.mnemonic() == Mnemonic::And
                && register_is_stack_pointer(instruction.op0_register())
                && immediate_for_operand(instruction, 1)
                    .is_some_and(|mask| mask.trailing_zeros() >= 4)
            {
                features.stack_alignment = true;
            }
            if instruction.mnemonic() == Mnemonic::Mov
                && instruction.op0_kind() == OpKind::Register
                && immediate_for_operand(instruction, 1)
                    .is_some_and(|value| value != 0 && value & u64::from(u32::MAX) == 0)
            {
                features.high_base_immediate = true;
            }
            if instruction.memory_base() != Register::None {
                features.memory_fetches = features.memory_fetches.saturating_add(1);
            }
            let flow = instruction.flow_control();
            let next = decoded.next_rva();
            let target_rva = decoded.direct_target_rva(pe);
            match flow {
                FlowControl::ConditionalBranch => {
                    if let Some(branch) = target_rva.filter(|rva| is_executable_rva(pe, *rva))
                        && starts.insert(branch)
                    {
                        queue.push_back(branch);
                    }
                    if let Some(fallthrough) = next.filter(|rva| is_executable_rva(pe, *rva))
                        && starts.insert(fallthrough)
                    {
                        queue.push_back(fallthrough);
                    }
                    break;
                }
                FlowControl::UnconditionalBranch => {
                    if let Some(branch) = target_rva.filter(|rva| is_executable_rva(pe, *rva))
                        && starts.insert(branch)
                    {
                        queue.push_back(branch);
                    } else if target_rva.is_none() {
                        features.indirect_dispatch = true;
                    }
                    break;
                }
                FlowControl::IndirectBranch => {
                    features.indirect_dispatch = true;
                    break;
                }
                FlowControl::Return
                | FlowControl::Interrupt
                | FlowControl::Exception
                | FlowControl::XbeginXabortXend => break,
                FlowControl::Call | FlowControl::IndirectCall | FlowControl::Next => {}
            }
            let Some(fallthrough) = next.filter(|rva| is_executable_rva(pe, *rva)) else {
                break;
            };
            current = fallthrough;
        }
    }
    features
}

fn immediate_for_operand(instruction: &iced_x86::Instruction, operand: u32) -> Option<u64> {
    match instruction.op_kind(operand) {
        OpKind::Immediate8 => Some(u64::from(instruction.immediate8())),
        OpKind::Immediate8to16 => Some(u64::from(instruction.immediate8to16().cast_unsigned())),
        OpKind::Immediate8to32 => Some(u64::from(instruction.immediate8to32().cast_unsigned())),
        OpKind::Immediate8to64 => Some(instruction.immediate8to64().cast_unsigned()),
        OpKind::Immediate16 => Some(u64::from(instruction.immediate16())),
        OpKind::Immediate32 => Some(u64::from(instruction.immediate32())),
        OpKind::Immediate32to64 => Some(instruction.immediate32to64().cast_unsigned()),
        OpKind::Immediate64 => Some(instruction.immediate64()),
        _ => None,
    }
}

fn add_evidence(
    candidate: &mut VmEntryCandidate,
    score: &mut u16,
    weight: u8,
    statement: impl Into<String>,
) {
    *score = score.saturating_add(u16::from(weight));
    candidate.evidence.push(VmEntryEvidence {
        weight,
        statement: statement.into(),
    });
}

#[allow(clippy::too_many_lines)]
fn analyze_stub(
    pe: &PeImage<'_>,
    rva: Rva,
    source: VmEntrySource,
    limits: VmDiscoveryLimits,
    chain_cache: &mut HashMap<Rva, ChainFeatures>,
) -> Option<VmEntryCandidate> {
    let first = decode_one(pe, rva).ok()?;
    let va = first.va;
    let mut candidate = VmEntryCandidate {
        rva,
        va,
        source,
        reachability: if matches!(
            source,
            VmEntrySource::PeEntryPoint | VmEntrySource::TlsCallback
        ) {
            VmEntryReachability::LoaderReachable
        } else {
            VmEntryReachability::UnprovenExecutableCandidate
        },
        confidence: 0,
        seed: None,
        call_target: None,
        bootstrap_rva: None,
        evidence: Vec::new(),
    };
    let mut score = 0_u16;
    if matches!(
        source,
        VmEntrySource::PeEntryPoint | VmEntrySource::TlsCallback
    ) {
        add_evidence(&mut candidate, &mut score, 10, "loader-reachable root");
    }
    if first.instruction.mnemonic() != Mnemonic::Push {
        candidate.confidence = u8::try_from(score).unwrap_or(u8::MAX);
        return Some(candidate);
    }
    candidate.seed = immediate(&first);
    if candidate.seed.is_some() {
        add_evidence(
            &mut candidate,
            &mut score,
            15,
            "entry begins by pushing an immediate seed",
        );
    }
    let second_rva = first.next_rva()?;
    let second = decode_one(pe, second_rva).ok()?;
    if second.instruction.flow_control() != FlowControl::Call {
        candidate.confidence = u8::try_from(score).unwrap_or(u8::MAX);
        return Some(candidate);
    }
    let target = second.direct_target_rva(pe)?;
    candidate.call_target = Some(target);
    add_evidence(
        &mut candidate,
        &mut score,
        15,
        format!("seed is followed by a direct call to executable RVA {target}"),
    );
    let features = chain_cache
        .entry(target)
        .or_insert_with(|| analyze_chain(pe, target, limits));
    if features.pushed_registers.len() >= 8 {
        add_evidence(
            &mut candidate,
            &mut score,
            15,
            format!(
                "downstream chain saves {} distinct registers",
                features.pushed_registers.len()
            ),
        );
    }
    if features.pushes_flags {
        add_evidence(
            &mut candidate,
            &mut score,
            5,
            "downstream chain saves native flags",
        );
    }
    if let Some((allocation_rva, amount)) = features.stack_allocation {
        candidate.bootstrap_rva = Some(allocation_rva);
        add_evidence(
            &mut candidate,
            &mut score,
            20,
            format!("downstream bootstrap reserves {amount:#X} bytes at {allocation_rva}"),
        );
    }
    if features.stack_alignment {
        add_evidence(
            &mut candidate,
            &mut score,
            5,
            "downstream bootstrap explicitly aligns the stack",
        );
    }
    if features.high_base_immediate {
        add_evidence(
            &mut candidate,
            &mut score,
            5,
            "downstream chain composes a pointer from a high aligned base",
        );
    }
    if features.memory_fetches >= 2 {
        add_evidence(
            &mut candidate,
            &mut score,
            5,
            "downstream chain performs state-dependent memory fetches",
        );
    }
    if features.indirect_dispatch {
        add_evidence(
            &mut candidate,
            &mut score,
            10,
            "downstream chain terminates in indirect dispatch",
        );
    }
    if features.resource_limited {
        candidate.evidence.push(VmEntryEvidence {
            weight: 0,
            statement: "downstream chain analysis reached a resource limit".to_owned(),
        });
    }
    candidate.confidence = u8::try_from(score.min(100)).unwrap_or(100);
    Some(candidate)
}

/// Analyzes one explicitly supplied executable RVA as a possible VM entry.
/// This is used for continuation targets recovered from semantic execution;
/// unlike the broad executable scan it does not search or infer reachability.
#[must_use]
pub fn analyze_vm_entry_at(
    pe: &PeImage<'_>,
    rva: Rva,
    limits: VmDiscoveryLimits,
) -> Option<VmEntryCandidate> {
    analyze_stub(
        pe,
        rva,
        VmEntrySource::ExecutableScan,
        limits,
        &mut HashMap::new(),
    )
}

fn executable_scan_roots(pe: &PeImage<'_>, limits: VmDiscoveryLimits) -> (Vec<Rva>, usize, bool) {
    let mut roots = Vec::new();
    let mut examined = 0_usize;
    let mut limited = false;
    let entry_section = pe
        .sections
        .iter()
        .find(|section| section.contains_mapped_rva(pe.optional.address_of_entry_point))
        .map(|section| section.index);
    for section in pe.sections.iter().filter(|section| {
        if section.characteristics & 0x2000_0000 == 0 {
            return false;
        }
        Some(section.index) == entry_section || section_entropy(pe, section.index) >= 7.0
    }) {
        if section.pointer_to_raw_data.0 == 0 || section.size_of_raw_data < 2 {
            continue;
        }
        let Some(start) = usize::try_from(section.pointer_to_raw_data.0).ok() else {
            continue;
        };
        let Some(size) = usize::try_from(section.size_of_raw_data).ok() else {
            continue;
        };
        let Some(end) = start
            .checked_add(size)
            .filter(|end| *end <= pe.data().len())
        else {
            continue;
        };
        let bytes = &pe.data()[start..end];
        for (offset, byte) in bytes.iter().copied().enumerate() {
            // Opcode prefilter is candidate generation only. Every survivor is
            // decoded and validated by downstream stack/context/dispatch effects.
            if !matches!(byte, 0x68 | 0x6A) {
                continue;
            }
            let Some(delta) = u32::try_from(offset).ok() else {
                continue;
            };
            let Some(rva) = section.virtual_address.checked_add(delta) else {
                continue;
            };
            let Ok(first) = decode_one(pe, rva) else {
                continue;
            };
            if first.instruction.mnemonic() != Mnemonic::Push || immediate(&first).is_none() {
                continue;
            }
            let Some(second_rva) = first.next_rva() else {
                continue;
            };
            let Ok(second) = decode_one(pe, second_rva) else {
                continue;
            };
            if second.instruction.flow_control() == FlowControl::Call
                && second.direct_target_rva(pe).is_some()
            {
                examined = examined.saturating_add(1);
                if examined > limits.max_stub_candidates {
                    limited = true;
                    break;
                }
                roots.push(rva);
            }
        }
        if limited {
            break;
        }
    }
    roots.sort_unstable();
    roots.dedup();
    (roots, examined, limited)
}

#[allow(clippy::cast_precision_loss)]
fn section_entropy(pe: &PeImage<'_>, section_index: usize) -> f64 {
    let Some(section) = pe.sections.get(section_index) else {
        return 0.0;
    };
    let Some(start) = usize::try_from(section.pointer_to_raw_data.0).ok() else {
        return 0.0;
    };
    let Some(size) = usize::try_from(section.size_of_raw_data).ok() else {
        return 0.0;
    };
    let Some(end) = start.checked_add(size) else {
        return 0.0;
    };
    let Some(bytes) = pe.data().get(start..end) else {
        return 0.0;
    };
    if bytes.is_empty() {
        return 0.0;
    }
    let mut counts = [0_u64; 256];
    for byte in bytes {
        counts[usize::from(*byte)] += 1;
    }
    let length = bytes.len() as f64;
    counts
        .into_iter()
        .filter(|count| *count != 0)
        .map(|count| {
            let probability = count as f64 / length;
            -probability * probability.log2()
        })
        .sum()
}

/// Discovers VM-entry candidates from loader roots and semantically validated
/// executable stub candidates.
///
/// # Errors
///
/// Returns an error only for irrecoverable root-address conversion failures;
/// individual malformed candidates are rejected and reported conservatively.
#[allow(clippy::too_many_lines)]
pub fn discover_vm_entries(
    pe: &PeImage<'_>,
    limits: VmDiscoveryLimits,
) -> Result<VmDiscoveryReport, VmDiscoveryError> {
    let mut roots = BTreeMap::new();
    roots.insert(
        pe.optional.address_of_entry_point,
        VmEntrySource::PeEntryPoint,
    );
    let pe_report = pe.report();
    if let Some(tls) = pe_report.tls {
        for callback in tls.callbacks {
            let callback_rva =
                pe.optional
                    .image_base
                    .checked_rva_of(callback)
                    .ok_or_else(|| {
                        VmDiscoveryError::Address(format!(
                            "TLS callback {callback} is outside the image"
                        ))
                    })?;
            roots
                .entry(callback_rva)
                .or_insert(VmEntrySource::TlsCallback);
        }
    }
    let (scan_roots, scanned_stub_candidates, resource_limited) = if limits.scan_executable_stubs {
        executable_scan_roots(pe, limits)
    } else {
        (Vec::new(), 0, false)
    };
    for root in scan_roots {
        roots.entry(root).or_insert(VmEntrySource::ExecutableScan);
    }

    let mut chain_cache = HashMap::new();
    let mut entries = Vec::new();
    let mut rejected_roots = Vec::new();
    for (root, source) in roots {
        let Some(candidate) = analyze_stub(pe, root, source, limits, &mut chain_cache) else {
            continue;
        };
        if candidate.confidence >= limits.minimum_confidence {
            entries.push(candidate);
        } else if !matches!(source, VmEntrySource::ExecutableScan) {
            rejected_roots.push(candidate);
        }
    }
    entries.sort_by_key(|candidate| candidate.rva);

    let mut grouped: BTreeMap<Rva, (Vec<Rva>, BTreeSet<u64>)> = BTreeMap::new();
    for entry in &entries {
        if let Some(bootstrap) = entry.bootstrap_rva {
            let group = grouped.entry(bootstrap).or_default();
            group.0.push(entry.rva);
            if let Some(seed) = entry.seed {
                group.1.insert(seed);
            }
        }
    }
    let instances = grouped
        .into_iter()
        .enumerate()
        .map(|(id, (bootstrap_rva, (mut entry_rvas, seeds)))| {
            entry_rvas.sort_unstable();
            entry_rvas.dedup();
            VmInstance {
                id,
                bootstrap_rva,
                entry_rvas,
                distinct_seeds: seeds.into_iter().collect(),
            }
        })
        .collect();
    let mut diagnostics = Vec::new();
    if resource_limited {
        diagnostics.push(format!(
            "executable stub scan reached limit {}",
            limits.max_stub_candidates
        ));
    }
    Ok(VmDiscoveryReport {
        schema_version: 1,
        scanned_stub_candidates,
        resource_limited,
        entries,
        instances,
        rejected_roots,
        diagnostics,
    })
}
