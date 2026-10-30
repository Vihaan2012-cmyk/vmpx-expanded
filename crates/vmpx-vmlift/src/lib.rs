//! Semantic-lifting readiness analysis.
//!
//! This crate is the hard gate between recovered VM instructions and native
//! code generation. It inventories every required lowering capability and
//! refuses to describe a routine as codegen-ready while any semantic, state,
//! ABI, or control-flow dependency is missing.

mod cfg_lift;
mod constant_dispatch;
mod decoded_lift;
mod effect_liveness;
mod entry_frame;
mod equivalence;
mod expr_codegen;
mod live_in;
mod nested_entry;
mod passthrough;
mod replacement;

pub use passthrough::{is_environment_boundary, is_native_passthrough};

pub use cfg_lift::{
    CfgProbe, DecodedCfgInput, LiftedCfgFunction, lift_decoded_cfg, lift_decoded_cfg_with_nested,
    probe_decoded_cfg,
};
pub use nested_entry::{NestedEntry, NestedEntryResolver, PeNestedEntries};
pub use constant_dispatch::{
    ConstantDispatchConfig, ConstantDispatchError, compile_constant_dispatch,
};
pub use decoded_lift::{
    DecodedLiftError, DecodedLinearInput, LiftAssumption, LiftedLinearFunction, SemanticLowering,
    first_unliftable, lift_decoded_linear, semantic_lowering,
};
pub use entry_frame::{frame_initial_state, recover_entry_frame};
pub use effect_liveness::{
    CollectedCall, CollectedEffects, CollectedStore, EntryEffectLivenessReport,
    PathEffectLivenessReport, audit_entry_effect_liveness, collect_path_effects,
};
pub use equivalence::{
    Assignment, AssignmentDivergence, Divergence, ExitSemantics, Side, Verdict, VerificationReport,
    compare_exits, verify_over_domain,
};
pub use expr_codegen::ImportCall;
pub use expr_codegen::{
    ExpressionCodegenConfig, ExpressionCodegenError, ExpressionInput, LoweredExpression,
    compile_expression, lower_expression,
};
pub use live_in::{
    AccessKind, EntryCallingConvention, RegisterAccess, calling_convention, infer_live_in,
};
pub use replacement::{
    EffectCall, EffectStore, Replacement, ReplacementConfig, ReplacementError, build_replacement,
    build_replacement_popping,
};

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;
use thiserror::Error;
use vmpx_vm::{
    BitWidth, DecodedVmInstruction, Expr, ResizeOp, SymbolicFreshAllocation, SymbolicNativeCall,
    SymbolicVmExecutionReport, VmHandlerSemantic, VmProfileChainReport, VmProfileChainStop,
    VmProfilePathFamilyReport,
};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LoweringCapability {
    WidthAwareIntegerBackend,
    UnaryBackend,
    ShiftBackend,
    ResizeBackend,
    MemoryBackend,
    BranchBackend,
    StackSsaLowering,
    NativeContextMapping,
    DynamicStackPointer,
    FlagsLivenessAndMaterialization,
    TimestampCounter,
    WideUnsignedMultiply,
    UnsignedDivide,
    TableChecksum,
    NativeCallLowering,
    ImportFixups,
    VirtualCfg,
    NativeExitBridge,
    WideSignedMultiply,
    DoubleShiftBackend,
    CpuidIntrinsic,
    NativeFlagsRestore,
    SegmentRelativeLoad,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct InstructionShapeCount {
    pub semantic: VmHandlerSemantic,
    pub operation_width: Option<BitWidth>,
    pub input_width: Option<BitWidth>,
    pub resize: Option<ResizeOp>,
    pub count: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct LiftabilityIssue {
    pub instruction_index: Option<usize>,
    pub semantic: Option<VmHandlerSemantic>,
    pub message: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct LiftabilityReport {
    pub schema_version: u32,
    pub total_instructions: usize,
    pub native_call_boundaries: usize,
    pub complete_vm_exit: bool,
    pub instruction_shapes: Vec<InstructionShapeCount>,
    pub required_capabilities: Vec<LoweringCapability>,
    pub implemented_capabilities: Vec<LoweringCapability>,
    pub missing_capabilities: Vec<LoweringCapability>,
    pub issues: Vec<LiftabilityIssue>,
    pub ready_for_codegen: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LiftedEffect {
    MemoryWrite {
        instruction_index: usize,
        address: Expr,
        value: Expr,
    },
    MemoryHavoc {
        instruction_index: usize,
        cause: String,
    },
    NativeCall {
        call: Box<SymbolicNativeCall>,
    },
}

impl LiftedEffect {
    fn order_key(&self) -> (usize, u8) {
        match self {
            Self::NativeCall { call } => (call.instruction_index, 0),
            Self::MemoryHavoc {
                instruction_index, ..
            } => (*instruction_index, 1),
            Self::MemoryWrite {
                instruction_index, ..
            } => (*instruction_index, 2),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct LiftedSemanticBlock {
    pub segment_index: usize,
    pub semantic_instruction_count: usize,
    pub effects: Vec<LiftedEffect>,
    pub context_outputs: BTreeMap<i64, Expr>,
    pub final_flags: Expr,
    pub branch_target: Option<Expr>,
    pub live_allocations: Vec<SymbolicFreshAllocation>,
    pub memory_epoch: u64,
    pub memory_tainted: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct LiftedSemanticRoutine {
    pub schema_version: u32,
    pub blocks: Vec<LiftedSemanticBlock>,
    pub semantic_instruction_count: usize,
    pub complete_vm_exit: bool,
    pub contains_memory_havoc: bool,
    pub path_specific: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct LiftedSemanticPath {
    pub assignments: BTreeMap<String, u64>,
    pub routine: LiftedSemanticRoutine,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct LiftedSemanticPathFamily {
    pub schema_version: u32,
    pub paths: Vec<LiftedSemanticPath>,
    pub domain_complete: bool,
    pub all_complete_vm_exit: bool,
    pub contains_memory_havoc: bool,
    pub semantic_instruction_count: usize,
}

#[derive(Clone, Debug, Eq, Error, PartialEq, Serialize)]
pub enum LiftError {
    #[error("profile-chain segment {segment_index} has no successful symbolic state")]
    MissingSymbolicState { segment_index: usize },
}

fn append_report_effects(output: &mut Vec<LiftedEffect>, report: &SymbolicVmExecutionReport) {
    output.extend(
        report
            .external_writes
            .iter()
            .cloned()
            .map(|write| LiftedEffect::MemoryWrite {
                instruction_index: write.instruction_index,
                address: write.address,
                value: write.value,
            }),
    );
    output.extend(
        report
            .memory_havocs
            .iter()
            .cloned()
            .map(|havoc| LiftedEffect::MemoryHavoc {
                instruction_index: havoc.instruction_index,
                cause: havoc.cause,
            }),
    );
    output.extend(
        report
            .native_calls
            .iter()
            .cloned()
            .map(|call| LiftedEffect::NativeCall {
                call: Box::new(call),
            }),
    );
}

fn lift_phases(
    segment_index: usize,
    semantic_instruction_count: usize,
    phases: &[&SymbolicVmExecutionReport],
) -> LiftedSemanticBlock {
    let final_state = *phases
        .last()
        .expect("caller supplies at least one symbolic phase");
    let mut effects = Vec::new();
    for phase in phases.iter().copied() {
        let mut phase_effects = Vec::new();
        append_report_effects(&mut phase_effects, phase);
        phase_effects.sort_by_key(LiftedEffect::order_key);
        effects.extend(phase_effects);
    }
    LiftedSemanticBlock {
        segment_index,
        semantic_instruction_count,
        effects,
        context_outputs: final_state.context_cells.clone(),
        final_flags: final_state.final_flags.clone(),
        branch_target: final_state.branch_target.clone(),
        live_allocations: final_state.fresh_allocations.clone(),
        memory_epoch: final_state.memory_epoch,
        memory_tainted: final_state.memory_tainted,
    }
}

/// Lifts one symbolic block after VM-stack execution has eliminated handler
/// mechanics. This is also the unit boundary used by graph assembly.
#[must_use]
pub fn lift_symbolic_block(
    block_index: usize,
    semantic_instruction_count: usize,
    report: &SymbolicVmExecutionReport,
) -> LiftedSemanticBlock {
    lift_phases(block_index, semantic_instruction_count, &[report])
}

fn lift_segment(
    segment: &vmpx_vm::VmProfileChainSegment,
) -> Result<LiftedSemanticBlock, LiftError> {
    let Some(base) = segment.symbolic.as_ref() else {
        return Err(LiftError::MissingSymbolicState {
            segment_index: segment.index,
        });
    };
    let mut phases = vec![base];
    if let Some(boundary) = segment.boundary_symbolic.as_ref() {
        phases.push(boundary);
    }
    if let Some(transition) = segment.transition_pop.as_ref() {
        phases.push(transition);
    }
    Ok(lift_phases(
        segment.index,
        segment.decode.decode.instructions.len() + usize::from(segment.boundary_decode.is_some()),
        &phases,
    ))
}

/// Converts a successfully symbolically executed profile chain into a compact
/// program-semantic routine. VM bytecode, handler addresses, rolling keys, and
/// dispatch mechanics are absent; only expressions, ordered external effects,
/// context outputs, and control targets remain.
///
/// The current explorer follows one concrete path at a time, so `path_specific`
/// remains true until virtual-CFG blocks are merged by the graph lifter.
///
/// # Errors
///
/// Rejects a segment without a successful symbolic state instead of silently
/// dropping its effects.
pub fn lift_profile_chain(
    chain: &VmProfileChainReport,
) -> Result<LiftedSemanticRoutine, LiftError> {
    let blocks = chain
        .segments
        .iter()
        .map(lift_segment)
        .collect::<Result<Vec<_>, _>>()?;
    let semantic_instruction_count = blocks
        .iter()
        .map(|block| block.semantic_instruction_count)
        .sum();
    let contains_memory_havoc = blocks.iter().any(|block| {
        block.memory_tainted
            || block
                .effects
                .iter()
                .any(|effect| matches!(effect, LiftedEffect::MemoryHavoc { .. }))
    });
    Ok(LiftedSemanticRoutine {
        schema_version: 1,
        blocks,
        semantic_instruction_count,
        complete_vm_exit: chain.stop_reason == VmProfileChainStop::VmExit,
        contains_memory_havoc,
        path_specific: true,
    })
}

/// Lifts every exact finite-domain profile path without conflating states from
/// different assignments. This is the semantic input to later CFG merging;
/// each constituent routine remains explicitly path-specific until that merge
/// proves compatible block-boundary state.
///
/// # Errors
///
/// Rejects any path containing a profile segment without a successful symbolic
/// execution report.
pub fn lift_profile_path_family(
    family: &VmProfilePathFamilyReport,
) -> Result<LiftedSemanticPathFamily, LiftError> {
    let paths = family
        .paths
        .iter()
        .map(|path| {
            Ok(LiftedSemanticPath {
                assignments: path.assignments.clone(),
                routine: lift_profile_chain(&path.chain)?,
            })
        })
        .collect::<Result<Vec<_>, LiftError>>()?;
    let contains_memory_havoc = paths.iter().any(|path| path.routine.contains_memory_havoc);
    let semantic_instruction_count = paths
        .iter()
        .map(|path| path.routine.semantic_instruction_count)
        .sum();
    Ok(LiftedSemanticPathFamily {
        schema_version: 1,
        paths,
        domain_complete: family.domain_complete,
        all_complete_vm_exit: family.all_paths_reach_vm_exit,
        contains_memory_havoc,
        semantic_instruction_count,
    })
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct ShapeKey {
    semantic: u8,
    width: u8,
    input_width: u8,
    resize: u8,
}

#[derive(Clone, Copy)]
struct ShapeValue {
    semantic: VmHandlerSemantic,
    operation_width: Option<BitWidth>,
    input_width: Option<BitWidth>,
    resize: Option<ResizeOp>,
}

fn semantic_order(semantic: VmHandlerSemantic) -> u8 {
    semantic as u8
}

fn width_order(width: Option<BitWidth>) -> u8 {
    match width {
        None => 0,
        Some(BitWidth::W8) => 1,
        Some(BitWidth::W16) => 2,
        Some(BitWidth::W32) => 3,
        Some(BitWidth::W64) => 4,
    }
}

fn resize_order(resize: Option<ResizeOp>) -> u8 {
    match resize {
        None => 0,
        Some(ResizeOp::Truncate) => 1,
        Some(ResizeOp::ZeroExtend) => 2,
        Some(ResizeOp::SignExtend) => 3,
    }
}

fn capabilities_for_semantic(semantic: VmHandlerSemantic) -> &'static [LoweringCapability] {
    use LoweringCapability as Capability;
    use VmHandlerSemantic as Semantic;
    match semantic {
        Semantic::Dispatch
        | Semantic::NativeTransition
        | Semantic::ContextInitialize
        | Semantic::Unknown => &[],
        Semantic::ConstantPush => &[
            Capability::StackSsaLowering,
            Capability::WidthAwareIntegerBackend,
        ],
        Semantic::VirtualRegisterRead | Semantic::VirtualRegisterWrite => &[
            Capability::StackSsaLowering,
            Capability::NativeContextMapping,
            Capability::WidthAwareIntegerBackend,
        ],
        Semantic::MemoryLoad | Semantic::MemoryStore => &[
            Capability::StackSsaLowering,
            Capability::MemoryBackend,
            Capability::WidthAwareIntegerBackend,
        ],
        Semantic::Add
        | Semantic::Sub
        | Semantic::Xor
        | Semantic::And
        | Semantic::Or
        | Semantic::Compare
        | Semantic::Flags => &[
            Capability::StackSsaLowering,
            Capability::WidthAwareIntegerBackend,
            Capability::FlagsLivenessAndMaterialization,
        ],
        Semantic::Nor | Semantic::Nand | Semantic::Not | Semantic::Neg => &[
            Capability::StackSsaLowering,
            Capability::WidthAwareIntegerBackend,
            Capability::UnaryBackend,
            Capability::FlagsLivenessAndMaterialization,
        ],
        Semantic::ShiftLeft
        | Semantic::ShiftRightLogical
        | Semantic::ShiftRightArithmetic
        | Semantic::RotateLeft
        | Semantic::RotateRight
        | Semantic::Shift
        | Semantic::Rotate => &[
            Capability::StackSsaLowering,
            Capability::ShiftBackend,
            Capability::FlagsLivenessAndMaterialization,
        ],
        Semantic::ReadTimestampCounter => {
            &[Capability::StackSsaLowering, Capability::TimestampCounter]
        }
        Semantic::Mul => &[
            Capability::StackSsaLowering,
            Capability::WideUnsignedMultiply,
            Capability::FlagsLivenessAndMaterialization,
        ],
        Semantic::SignedMultiply => &[
            Capability::StackSsaLowering,
            Capability::WideSignedMultiply,
            Capability::FlagsLivenessAndMaterialization,
        ],
        Semantic::ShiftLeftDouble | Semantic::ShiftRightDouble => &[
            Capability::StackSsaLowering,
            Capability::DoubleShiftBackend,
            Capability::FlagsLivenessAndMaterialization,
        ],
        Semantic::Cpuid => &[Capability::StackSsaLowering, Capability::CpuidIntrinsic],
        Semantic::PopFlags => &[
            Capability::StackSsaLowering,
            Capability::NativeFlagsRestore,
            Capability::FlagsLivenessAndMaterialization,
        ],
        Semantic::GsRelativeLoad => &[
            Capability::StackSsaLowering,
            Capability::MemoryBackend,
            Capability::SegmentRelativeLoad,
        ],
        Semantic::UnsignedDivide => &[
            Capability::StackSsaLowering,
            Capability::UnsignedDivide,
            Capability::FlagsLivenessAndMaterialization,
        ],
        Semantic::TableChecksum => &[
            Capability::StackSsaLowering,
            Capability::MemoryBackend,
            Capability::TableChecksum,
        ],
        Semantic::Push | Semantic::Pop | Semantic::StackAdjust => &[
            Capability::StackSsaLowering,
            Capability::DynamicStackPointer,
        ],
        Semantic::ConditionalBranch | Semantic::UnconditionalBranch => &[
            Capability::VirtualCfg,
            Capability::BranchBackend,
            Capability::FlagsLivenessAndMaterialization,
        ],
        Semantic::NativeCall => &[Capability::NativeCallLowering, Capability::ImportFixups],
        Semantic::VmExit => &[Capability::NativeExitBridge],
        Semantic::SignExtend | Semantic::ZeroExtend => {
            &[Capability::StackSsaLowering, Capability::ResizeBackend]
        }
    }
}

fn current_implemented_capabilities() -> BTreeSet<LoweringCapability> {
    [
        LoweringCapability::WidthAwareIntegerBackend,
        LoweringCapability::UnaryBackend,
        LoweringCapability::ShiftBackend,
        LoweringCapability::ResizeBackend,
        LoweringCapability::MemoryBackend,
        LoweringCapability::BranchBackend,
        LoweringCapability::StackSsaLowering,
        LoweringCapability::DoubleShiftBackend,
        LoweringCapability::CpuidIntrinsic,
        LoweringCapability::NativeFlagsRestore,
        LoweringCapability::SegmentRelativeLoad,
        LoweringCapability::WideUnsignedMultiply,
        LoweringCapability::WideSignedMultiply,
    ]
    .into_iter()
    .collect()
}

fn semantic_requires_width(semantic: VmHandlerSemantic) -> bool {
    !matches!(
        semantic,
        VmHandlerSemantic::Dispatch
            | VmHandlerSemantic::NativeCall
            | VmHandlerSemantic::NativeTransition
            | VmHandlerSemantic::ConditionalBranch
            | VmHandlerSemantic::UnconditionalBranch
            | VmHandlerSemantic::VmExit
            | VmHandlerSemantic::Unknown
    )
}

fn audit<'a>(
    instructions: impl IntoIterator<Item = &'a DecodedVmInstruction>,
    native_call_boundaries: usize,
    complete_vm_exit: bool,
    requires_virtual_cfg: bool,
) -> LiftabilityReport {
    let implemented = current_implemented_capabilities();
    let mut required = BTreeSet::new();
    let mut shapes: BTreeMap<ShapeKey, (ShapeValue, usize)> = BTreeMap::new();
    let mut issues = Vec::new();
    let mut total_instructions = 0_usize;

    for (instruction_index, instruction) in instructions.into_iter().enumerate() {
        total_instructions = total_instructions.saturating_add(1);
        required.extend(capabilities_for_semantic(instruction.semantic));
        if instruction.resize.is_some() {
            required.insert(LoweringCapability::ResizeBackend);
        }
        if instruction.semantic == VmHandlerSemantic::Unknown {
            issues.push(LiftabilityIssue {
                instruction_index: Some(instruction_index),
                semantic: Some(instruction.semantic),
                message: "unknown VM semantic cannot be lowered".to_owned(),
            });
        }
        if semantic_requires_width(instruction.semantic) && instruction.operation_width.is_none() {
            issues.push(LiftabilityIssue {
                instruction_index: Some(instruction_index),
                semantic: Some(instruction.semantic),
                message: "semantic operation width is missing".to_owned(),
            });
        }
        let key = ShapeKey {
            semantic: semantic_order(instruction.semantic),
            width: width_order(instruction.operation_width),
            input_width: width_order(instruction.input_width),
            resize: resize_order(instruction.resize),
        };
        let value = ShapeValue {
            semantic: instruction.semantic,
            operation_width: instruction.operation_width,
            input_width: instruction.input_width,
            resize: instruction.resize,
        };
        shapes
            .entry(key)
            .and_modify(|(_, count)| *count = count.saturating_add(1))
            .or_insert((value, 1));
    }

    if native_call_boundaries != 0 {
        required.extend([
            LoweringCapability::NativeCallLowering,
            LoweringCapability::ImportFixups,
        ]);
    }
    if complete_vm_exit {
        required.insert(LoweringCapability::NativeExitBridge);
    } else {
        issues.push(LiftabilityIssue {
            instruction_index: None,
            semantic: None,
            message: "routine does not terminate at a proven VM exit".to_owned(),
        });
    }
    if requires_virtual_cfg {
        required.insert(LoweringCapability::VirtualCfg);
    }

    let missing: Vec<_> = required.difference(&implemented).copied().collect();
    let ready_for_codegen = missing.is_empty() && issues.is_empty();
    LiftabilityReport {
        schema_version: 1,
        total_instructions,
        native_call_boundaries,
        complete_vm_exit,
        instruction_shapes: shapes
            .into_values()
            .map(|(shape, count)| InstructionShapeCount {
                semantic: shape.semantic,
                operation_width: shape.operation_width,
                input_width: shape.input_width,
                resize: shape.resize,
                count,
            })
            .collect(),
        required_capabilities: required.into_iter().collect(),
        implemented_capabilities: implemented.into_iter().collect(),
        missing_capabilities: missing,
        issues,
        ready_for_codegen,
    }
}

/// Audits an instruction sequence without assuming a completed control-flow
/// path. This is useful while a virtual CFG is still being recovered.
#[must_use]
pub fn audit_instructions(instructions: &[DecodedVmInstruction]) -> LiftabilityReport {
    audit(instructions, 0, false, false)
}

/// Audits a recovered cross-profile chain and treats only a proven VM exit as
/// a complete native-routine boundary.
#[must_use]
pub fn audit_profile_chain(chain: &VmProfileChainReport) -> LiftabilityReport {
    let native_call_boundaries = chain
        .segments
        .iter()
        .filter(|segment| segment.boundary_decode.is_some())
        .count();
    let instructions = chain
        .segments
        .iter()
        .flat_map(|segment| segment.decode.decode.instructions.iter());
    audit(
        instructions,
        native_call_boundaries,
        chain.stop_reason == VmProfileChainStop::VmExit,
        true,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use vmpx_pe::Va;
    use vmpx_vm::{
        EffectKnowledge, RelativeDispatchState, SymbolicExecutionStop, SymbolicMemoryHavoc,
        SymbolicMemoryWrite,
    };

    fn instruction(
        semantic: VmHandlerSemantic,
        operation_width: Option<BitWidth>,
    ) -> DecodedVmInstruction {
        let state = RelativeDispatchState {
            vip: Va(0x1000),
            rolling_key: 0,
            handler: Va(0x2000),
        };
        DecodedVmInstruction {
            state_before: state,
            encoded_bytes: Vec::new(),
            semantic,
            semantic_metadata: None,
            input_width: operation_width,
            operation_width,
            resize: None,
            operand: None,
            vsp_delta: Some(0),
            flags_effect: EffectKnowledge::Unknown,
            confidence: 100,
            state_after: state,
        }
    }

    #[test]
    fn reports_exact_shapes_and_missing_pipeline_capabilities() {
        let instructions = vec![
            instruction(VmHandlerSemantic::ConstantPush, Some(BitWidth::W64)),
            instruction(VmHandlerSemantic::Add, Some(BitWidth::W32)),
            instruction(VmHandlerSemantic::ReadTimestampCounter, Some(BitWidth::W64)),
        ];
        let report = audit_instructions(&instructions);
        assert_eq!(report.total_instructions, 3);
        assert_eq!(report.instruction_shapes.len(), 3);
        assert!(
            report
                .implemented_capabilities
                .contains(&LoweringCapability::StackSsaLowering)
        );
        assert!(
            report
                .missing_capabilities
                .contains(&LoweringCapability::FlagsLivenessAndMaterialization)
        );
        assert!(
            report
                .missing_capabilities
                .contains(&LoweringCapability::TimestampCounter)
        );
        assert!(!report.ready_for_codegen);
    }

    #[test]
    fn unknown_and_missing_width_are_explicit_issues() {
        let instructions = vec![
            instruction(VmHandlerSemantic::Unknown, None),
            instruction(VmHandlerSemantic::MemoryLoad, None),
        ];
        let report = audit_instructions(&instructions);
        assert_eq!(report.issues.len(), 3);
        assert!(
            serde_json::to_string(&report)
                .expect("serialize report")
                .contains("unknown VM semantic")
        );
    }

    #[test]
    fn lifted_block_discards_vm_mechanics_and_orders_effects() {
        let address = Expr::register(BitWidth::W64, "output_pointer");
        let value = Expr::constant(BitWidth::W32, 7);
        let report = SymbolicVmExecutionReport {
            schema_version: 1,
            executed_instructions: 3,
            stop_reason: SymbolicExecutionStop::EndOfStream,
            final_vsp: Expr::register(BitWidth::W64, "vsp_initial"),
            final_vsp_offset: Some(0),
            final_flags: Expr::constant(BitWidth::W64, 0x202),
            branch_target: None,
            branch_target_dependencies: Vec::new(),
            context_cells: BTreeMap::from([(0, value.clone())]),
            memory_values: Vec::new(),
            memory_epoch: 1,
            memory_tainted: true,
            image_reads_suppressed: false,
            assumed_external_disjoint_from_vm_stack: false,
            external_writes: vec![SymbolicMemoryWrite {
                instruction_index: 2,
                address: address.clone(),
                value: value.clone(),
                epoch: 1,
            }],
            memory_havocs: vec![SymbolicMemoryHavoc {
                instruction_index: 1,
                cause: "opaque call".to_owned(),
            }],
            native_calls: Vec::new(),
            fresh_allocations: Vec::new(),
            next_native_call_epoch: 0,
            peak_expression_nodes: 3,
            tracked_memory_bytes: 0,
            resolved_image_bytes: 0,
        };
        let lifted = lift_symbolic_block(4, 3, &report);
        assert_eq!(lifted.segment_index, 4);
        assert!(matches!(
            lifted.effects.as_slice(),
            [
                LiftedEffect::MemoryHavoc { .. },
                LiftedEffect::MemoryWrite { .. }
            ]
        ));
        let json = serde_json::to_string(&lifted).expect("serialize lifted block");
        assert!(!json.contains("rolling_key"));
        assert!(!json.contains("handler"));
        assert!(!json.contains("encoded_bytes"));
    }
}
