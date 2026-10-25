use std::collections::{BTreeMap, BTreeSet};

use iced_x86::Register;
use serde::Serialize;
use vmpx_disasm::{
    CfgLimits, DirectJumpTraceLimits, DirectJumpTraceStop, DirectJumpTraceStopKind, DisasmError,
    RegisterDispatchProofLimits, trace_direct_jump_fragments,
};
use vmpx_pe::{PeImage, Rva, Va};

use crate::{
    BitWidth, BytecodeDecodeError, BytecodeDirection, DecodedVmInstruction,
    DispatchExtractionReport, EffectKnowledge, HandlerEffectRoles, HandlerEffectSummary,
    HandlerMemoryRegion, HandlerRegisterRoles, HandlerSliceConfig, HandlerValueFlowProof,
    KeyedOperandCodecReport, KeyedTransformCodec, LinearDecodeStop, LinearVmDecodeReport,
    NativeBackwardSlice, NativeCallAnalysisReport, NativeCallBoundarySpec, RecoveryStatus,
    RelativeDispatchCodec, RelativeDispatchState, RelativeThreadedCodec,
    TableChecksumHandlerReport, VmExitReport, VmHandlerAnalysis, VmHandlerDecodeSpec,
    VmHandlerSemantic, VmSemanticMetadata, analyze_native_call_handler,
    analyze_table_checksum_handler, analyze_vm_exit, backward_slice_handler_trace,
    classify_handler_effects, extract_handler_effects, extract_keyed_operand_codecs_direction,
    extract_relative_dispatch_codecs_with_postlude_direction, prove_handler_value_flow,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HandlerCodecRecoveryConfig {
    pub roles: HandlerRegisterRoles,
    pub bytecode_direction: BytecodeDirection,
    pub effect_roles: Option<HandlerEffectRoles>,
    pub live_out_registers: Vec<Register>,
    pub live_out_rflags: u32,
    pub trace_limits: DirectJumpTraceLimits,
    pub proof_limits: RegisterDispatchProofLimits,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VmProfileRoleResolutionConfig {
    pub roles: HandlerRegisterRoles,
    pub bytecode_direction: BytecodeDirection,
    pub vsp_candidates: Vec<Register>,
    pub context_base: Register,
    pub trace_limits: DirectJumpTraceLimits,
    pub proof_limits: RegisterDispatchProofLimits,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct VmProfileRoleCandidateReport {
    pub vsp_register: String,
    pub viable: bool,
    pub recovery: HandlerCodecRecoveryReport,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct VmProfileRoleResolutionReport {
    pub schema_version: u32,
    pub first_handler: Rva,
    pub status: RecoveryStatus,
    pub selected_vsp_register: Option<String>,
    pub candidates: Vec<VmProfileRoleCandidateReport>,
    pub diagnostics: Vec<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResolvedVmProfileRoles {
    pub handler_roles: HandlerRegisterRoles,
    pub effect_roles: HandlerEffectRoles,
    pub bytecode_direction: BytecodeDirection,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VmProfileRoleResolution {
    pub report: VmProfileRoleResolutionReport,
    pub recovered: Option<ResolvedVmProfileRoles>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct HandlerCodecRecoveryReport {
    pub schema_version: u32,
    pub entry: Rva,
    pub status: RecoveryStatus,
    pub trace_fragments: usize,
    pub decoded_instructions: usize,
    pub trace_stop: DirectJumpTraceStop,
    pub slice: NativeBackwardSlice,
    pub dispatch_analysis: DispatchExtractionReport,
    pub operand_analysis: KeyedOperandCodecReport,
    pub dispatch: Option<RelativeDispatchCodec>,
    pub operand: Option<KeyedTransformCodec>,
    pub effects: Option<HandlerEffectSummary>,
    pub handler_analysis: Option<VmHandlerAnalysis>,
    pub value_flow: Option<HandlerValueFlowProof>,
    pub native_call: Option<NativeCallAnalysisReport>,
    pub table_checksum: Option<TableChecksumHandlerReport>,
    pub vm_exit: Option<VmExitReport>,
    pub semantic_metadata: Option<VmSemanticMetadata>,
    pub boundary_spec: Option<NativeCallBoundarySpec>,
    pub decode_spec: Option<VmHandlerDecodeSpec>,
    pub confidence: u8,
    pub diagnostics: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct LinearHandlerRecoveryReport {
    pub schema_version: u32,
    pub decode: LinearVmDecodeReport,
    pub handler_recoveries: BTreeMap<Va, HandlerCodecRecoveryReport>,
}

#[derive(Debug, thiserror::Error)]
pub enum LinearHandlerRecoveryError {
    #[error("handler VA {handler} is outside the input image")]
    HandlerOutsideImage {
        handler: Va,
        partial: Box<LinearHandlerRecoveryReport>,
    },

    #[error("native handler recovery failed at {handler}: {source}")]
    NativeHandler {
        handler: Va,
        #[source]
        source: DisasmError,
        partial: Box<LinearHandlerRecoveryReport>,
    },

    #[error("bytecode recovery failed at handler {handler}: {source}")]
    Bytecode {
        handler: Va,
        #[source]
        source: BytecodeDecodeError,
        partial: Box<LinearHandlerRecoveryReport>,
    },
}

impl LinearHandlerRecoveryError {
    #[must_use]
    pub fn partial_report(&self) -> &LinearHandlerRecoveryReport {
        match self {
            Self::HandlerOutsideImage { partial, .. }
            | Self::NativeHandler { partial, .. }
            | Self::Bytecode { partial, .. } => partial,
        }
    }
}

fn is_linear_decodable_semantic(semantic: VmHandlerSemantic) -> bool {
    semantic != VmHandlerSemantic::Unknown && !semantic.ends_linear_block()
}

fn codec_recovery_status(
    dispatch_candidates: usize,
    operand_candidates: usize,
    rejected_operands: usize,
) -> RecoveryStatus {
    if dispatch_candidates == 0 {
        RecoveryStatus::Failed
    } else if dispatch_candidates > 1 || operand_candidates > 1 {
        RecoveryStatus::Ambiguous
    } else if rejected_operands > 0 {
        RecoveryStatus::PartiallyRecovered
    } else {
        RecoveryStatus::Recovered
    }
}

fn is_dispatch_only_effect(
    effects: &HandlerEffectSummary,
    operand_candidates: usize,
    handler_rebase: Option<Va>,
) -> bool {
    handler_rebase.is_some()
        && operand_candidates == 0
        && effects.vsp_delta == Some(0)
        && effects.vsp_adjustments.is_empty()
        && effects.memory.iter().all(|effect| {
            matches!(
                effect.region,
                HandlerMemoryRegion::Bytecode | HandlerMemoryRegion::NativeStackScratch
            )
        })
        && !effects.captures_native_flags_to_vsp
        && !effects.pushes_pre_adjustment_vsp
        && !effects.loads_vsp_from_top
        && !effects.loads_vip_from_vsp
        && !effects.resets_key_from_vip
        && effects.arithmetic_operations.is_empty()
        && effects.diagnostics.is_empty()
}

/// Runs the bounded native handler pipeline and selects codecs only when each
/// recovered role is unambiguous.
///
/// # Errors
///
/// Propagates invalid entry, decode, or postlude-proof setup errors. Ordinary
/// extraction failures remain diagnostics with a non-recovered status.
#[allow(clippy::too_many_lines)]
pub fn recover_handler_codecs(
    pe: &PeImage<'_>,
    entry: Rva,
    config: &HandlerCodecRecoveryConfig,
) -> Result<HandlerCodecRecoveryReport, DisasmError> {
    let trace = trace_direct_jump_fragments(pe, entry, config.trace_limits)?;
    let slice = backward_slice_handler_trace(
        &trace,
        &HandlerSliceConfig {
            live_out_registers: config.live_out_registers.clone(),
            live_out_rflags: config.live_out_rflags,
        },
    );
    let mut dispatch_analysis = extract_relative_dispatch_codecs_with_postlude_direction(
        pe,
        &trace,
        &slice,
        config.roles,
        config.bytecode_direction,
        config.proof_limits,
    )?;
    let dispatch_sources: BTreeSet<_> = dispatch_analysis
        .candidates
        .iter()
        .map(|candidate| candidate.encoded_load)
        .collect();
    let operand_analysis = extract_keyed_operand_codecs_direction(
        &trace,
        &slice,
        config.roles,
        &dispatch_sources,
        config.bytecode_direction,
    );

    let mut table_checksum = None;
    let mut checksum_semantic = false;
    let mut semantic_metadata = None;
    if dispatch_analysis.candidates.is_empty()
        && operand_analysis.candidates.is_empty()
        && trace.stop.kind == DirectJumpTraceStopKind::ConditionalBranch
        && let Some(effect_roles) = config.effect_roles
    {
        let analysis = analyze_table_checksum_handler(
            pe,
            entry,
            config.roles,
            effect_roles.vsp,
            config.bytecode_direction,
            &config.live_out_registers,
            config.trace_limits,
            config.proof_limits,
        )?;
        if let Some(recovered) = analysis.recovered {
            let Some(recovered_dispatch) = recovered.report.dispatch_analysis.clone() else {
                unreachable!("recovered checksum handler must retain dispatch evidence")
            };
            dispatch_analysis = recovered_dispatch;
            semantic_metadata = Some(VmSemanticMetadata::TableChecksum(
                recovered.metadata.clone(),
            ));
            checksum_semantic = true;
            table_checksum = Some(recovered.report);
        } else {
            table_checksum = Some(analysis.report);
        }
    }

    let mut vm_exit = None;
    let mut vm_exit_semantic = false;
    if dispatch_analysis.candidates.is_empty()
        && trace.stop.kind == DirectJumpTraceStopKind::Return
        && let Some(effect_roles) = config.effect_roles
    {
        let report = analyze_vm_exit(pe, entry, effect_roles.vsp, config.trace_limits)?;
        vm_exit_semantic = report.status == RecoveryStatus::Recovered;
        vm_exit = Some(report);
    }

    let mut native_call = None;
    let mut native_call_semantic = false;
    let mut native_call_resume_diagnostic = None;
    if dispatch_analysis.candidates.is_empty()
        && trace.stop.kind == DirectJumpTraceStopKind::ConditionalBranch
        && let [selector] = operand_analysis.candidates.as_slice()
        && selector.codec.width == BitWidth::W8
        && selector.full_register_value_proven
        && let Some(effect_roles) = config.effect_roles
    {
        let analysis = analyze_native_call_handler(
            pe,
            entry,
            config.roles,
            effect_roles.vsp,
            selector.decoded_register_id,
            selector.key_update,
            CfgLimits {
                max_blocks: config.trace_limits.max_fragments,
                max_instructions: config.trace_limits.max_instructions,
                max_instructions_per_block: config.trace_limits.max_instructions_per_fragment,
            },
        )?;
        if let Some(recovered) = analysis.recovered {
            if let Some(dispatch_source) = recovered.report.dispatch_source {
                let resume_trace =
                    trace_direct_jump_fragments(pe, dispatch_source, config.trace_limits)?;
                let resume_slice = backward_slice_handler_trace(
                    &resume_trace,
                    &HandlerSliceConfig {
                        live_out_registers: config.live_out_registers.clone(),
                        live_out_rflags: config.live_out_rflags,
                    },
                );
                let resume_dispatch = extract_relative_dispatch_codecs_with_postlude_direction(
                    pe,
                    &resume_trace,
                    &resume_slice,
                    config.roles,
                    config.bytecode_direction,
                    config.proof_limits,
                )?;
                if resume_dispatch.candidates.len() == 1 {
                    dispatch_analysis = resume_dispatch;
                    native_call_semantic = true;
                } else {
                    native_call_resume_diagnostic = Some(format!(
                        "native-call return path recovered {} dispatch codecs",
                        resume_dispatch.candidates.len()
                    ));
                }
            } else {
                native_call_resume_diagnostic =
                    Some("recovered native-call proof has no dispatch source".to_owned());
            }
            native_call = Some(recovered.report);
        } else {
            native_call = Some(analysis.report);
        }
    }

    let mut diagnostics = Vec::new();
    if let Some(diagnostic) = native_call_resume_diagnostic {
        diagnostics.push(diagnostic);
    }
    if dispatch_analysis.candidates.len() != 1 && !vm_exit_semantic {
        diagnostics.push(format!(
            "expected one dispatch codec, recovered {}",
            dispatch_analysis.candidates.len()
        ));
    }
    if operand_analysis.candidates.len() > 1 {
        diagnostics.push(format!(
            "multiple operand codecs remain: {}",
            operand_analysis.candidates.len()
        ));
    }
    diagnostics.extend(
        operand_analysis
            .rejected_sources
            .iter()
            .map(|rejection| format!("operand {}: {}", rejection.encoded_load, rejection.reason)),
    );
    let dispatch = (dispatch_analysis.candidates.len() == 1)
        .then(|| dispatch_analysis.candidates[0].codec.clone());
    let dispatch_rebases_vip = dispatch_analysis
        .candidates
        .first()
        .is_some_and(|candidate| candidate.vip_rebased_before_source);
    let operand = match operand_analysis.candidates.as_slice() {
        [candidate] => Some(candidate.codec.clone()),
        _ => None,
    };
    let codec_status = codec_recovery_status(
        dispatch_analysis.candidates.len(),
        operand_analysis.candidates.len(),
        operand_analysis.rejected_sources.len(),
    );
    let (effects, handler_analysis, value_flow) = if vm_exit_semantic {
        match vm_exit.as_ref() {
            Some(report) => {
                let mut evidence = report.evidence.clone();
                evidence.push(
                    "VM machinery terminates after restoring native architectural state".to_owned(),
                );
                (
                    None,
                    Some(VmHandlerAnalysis {
                        entry,
                        semantic: VmHandlerSemantic::VmExit,
                        operand_width: None,
                        input_width: None,
                        operation_width: None,
                        resize: None,
                        stack_delta: None,
                        vip_delta: None,
                        rolling_key_effect: None,
                        flags_effect: EffectKnowledge::Modified,
                        confidence: report.confidence,
                        evidence: evidence.clone(),
                    }),
                    Some(HandlerValueFlowProof {
                        schema_version: 1,
                        semantic: VmHandlerSemantic::VmExit,
                        proven: true,
                        producer: report.stack_switch,
                        consumer: report.native_return,
                        evidence,
                    }),
                )
            }
            None => (None, None, None),
        }
    } else if checksum_semantic {
        match (
            table_checksum.as_ref(),
            dispatch_analysis.candidates.as_slice(),
            semantic_metadata.as_ref(),
        ) {
            (
                Some(report),
                [dispatch_candidate],
                Some(VmSemanticMetadata::TableChecksum(metadata)),
            ) => {
                let vip_delta = i32::try_from(dispatch_candidate.vip_delta).ok();
                let mut evidence = report.evidence.clone();
                evidence.push(
                    "native counted loop is represented as one semantic checksum operation"
                        .to_owned(),
                );
                (
                    None,
                    Some(VmHandlerAnalysis {
                        entry,
                        semantic: VmHandlerSemantic::TableChecksum,
                        operand_width: None,
                        input_width: None,
                        operation_width: Some(metadata.checksum_width),
                        resize: None,
                        stack_delta: Some(metadata.vsp_delta),
                        vip_delta,
                        rolling_key_effect: None,
                        flags_effect: EffectKnowledge::Preserved,
                        confidence: report.confidence,
                        evidence: evidence.clone(),
                    }),
                    Some(HandlerValueFlowProof {
                        schema_version: 1,
                        semantic: VmHandlerSemantic::TableChecksum,
                        proven: true,
                        producer: report.pointer_load,
                        consumer: report.output_store,
                        evidence,
                    }),
                )
            }
            _ => (None, None, None),
        }
    } else if native_call_semantic {
        match (
            native_call.as_ref(),
            dispatch_analysis.candidates.as_slice(),
        ) {
            (Some(report), [dispatch_candidate]) => {
                let total_vip_delta = operand_analysis
                    .candidates
                    .first()
                    .and_then(|operand| operand.vip_delta.checked_add(dispatch_candidate.vip_delta))
                    .and_then(|delta| i32::try_from(delta).ok());
                let mut evidence = report.evidence.clone();
                evidence.push("decoded W8 operand selects the Win64 argument count".to_owned());
                (
                    None,
                    Some(VmHandlerAnalysis {
                        entry,
                        semantic: VmHandlerSemantic::NativeCall,
                        operand_width: Some(BitWidth::W8),
                        input_width: None,
                        operation_width: None,
                        resize: None,
                        stack_delta: None,
                        vip_delta: total_vip_delta,
                        rolling_key_effect: None,
                        flags_effect: EffectKnowledge::Unknown,
                        confidence: report.confidence,
                        evidence: evidence.clone(),
                    }),
                    Some(HandlerValueFlowProof {
                        schema_version: 1,
                        semantic: VmHandlerSemantic::NativeCall,
                        proven: true,
                        producer: report.target_load,
                        consumer: report.call_site,
                        evidence,
                    }),
                )
            }
            _ => (None, None, None),
        }
    } else {
        match (config.effect_roles, dispatch_analysis.candidates.as_slice()) {
            (Some(roles), [dispatch_candidate]) => {
                let effects = extract_handler_effects(
                    &trace,
                    &slice,
                    roles,
                    config.roles,
                    dispatch_candidate.encoded_load,
                );
                let operand_width = operand.as_ref().map(|codec| codec.width);
                let operand_vip_delta = match operand_analysis.candidates.as_slice() {
                    [] => Some(0_i64),
                    [candidate] => Some(candidate.vip_delta),
                    _ => None,
                };
                let total_vip_delta = operand_vip_delta.and_then(|operand_delta| {
                    operand_delta
                        .checked_add(dispatch_candidate.vip_delta)
                        .and_then(|delta| i32::try_from(delta).ok())
                });
                let mut analysis =
                    classify_handler_effects(entry, &effects, operand_width, total_vip_delta);
                let dispatch_only = analysis.semantic == VmHandlerSemantic::Unknown
                    && is_dispatch_only_effect(
                        &effects,
                        operand_analysis.candidates.len(),
                        dispatch_candidate.handler_rebase,
                    );
                if dispatch_only {
                    analysis.semantic = VmHandlerSemantic::Dispatch;
                    analysis.operation_width = None;
                    analysis.stack_delta = Some(0);
                    analysis.flags_effect = EffectKnowledge::Preserved;
                    analysis.confidence = 100;
                    analysis.evidence.push(
                        "native handler only establishes a static handler anchor and performs threaded dispatch"
                            .to_owned(),
                    );
                }
                let (operand_consumer, operand_register, operand_full_register_value_proven) =
                    match operand_analysis.candidates.as_slice() {
                        [candidate] => (
                            candidate.first_consumer,
                            Some(candidate.decoded_register_id),
                            candidate.full_register_value_proven,
                        ),
                        _ => (None, None, false),
                    };
                let proof = if dispatch_only {
                    HandlerValueFlowProof {
                        schema_version: 1,
                        semantic: VmHandlerSemantic::Dispatch,
                        proven: true,
                        producer: dispatch_candidate.handler_rebase_rva,
                        consumer: Some(dispatch_candidate.handler_update),
                        evidence: vec![
                            "static handler anchor reaches the relative dispatch update without program-level effects"
                                .to_owned(),
                        ],
                    }
                } else {
                    prove_handler_value_flow(
                        &trace,
                        &slice,
                        &effects,
                        &analysis,
                        operand_consumer,
                        operand_register,
                        operand_full_register_value_proven,
                    )
                };
                if analysis.semantic != VmHandlerSemantic::Unknown && !proof.proven {
                    analysis.evidence.extend(proof.evidence.iter().cloned());
                    analysis.semantic = VmHandlerSemantic::Unknown;
                    analysis.operation_width = None;
                    analysis.confidence = 0;
                }
                (Some(effects), Some(analysis), Some(proof))
            }
            _ => (None, None, None),
        }
    };
    let mut decode_spec = match (&dispatch, &handler_analysis) {
        (Some(dispatch), Some(analysis))
            if codec_status == RecoveryStatus::Recovered
                && !dispatch_rebases_vip
                && is_linear_decodable_semantic(analysis.semantic) =>
        {
            Some(VmHandlerDecodeSpec {
                semantic: analysis.semantic,
                semantic_metadata: semantic_metadata.clone(),
                input_width: analysis.input_width,
                operation_width: analysis.operation_width,
                resize: analysis.resize,
                operand: operand.clone(),
                dispatch: dispatch.clone(),
                handler_base: dispatch_analysis.candidates[0].handler_rebase,
                vsp_delta: analysis.stack_delta,
                flags_effect: analysis.flags_effect,
                confidence: analysis.confidence,
            })
        }
        _ => None,
    };
    let boundary_spec = if native_call_semantic {
        match (&operand, &dispatch, &native_call) {
            (Some(operand), Some(dispatch), Some(report))
                if report.status == RecoveryStatus::Recovered
                    && report.selector_outcomes.len() == 256 =>
            {
                Some(NativeCallBoundarySpec {
                    codec: RelativeThreadedCodec {
                        operand: operand.clone(),
                        dispatch: dispatch.clone(),
                    },
                    outcomes: report.selector_outcomes.clone(),
                })
            }
            _ => None,
        }
    } else {
        None
    };
    let mut status = if vm_exit_semantic {
        RecoveryStatus::Recovered
    } else {
        codec_status
    };
    if config.effect_roles.is_some() && decode_spec.is_none() && status == RecoveryStatus::Recovered
    {
        if handler_analysis.as_ref().is_some_and(|analysis| {
            analysis.semantic != VmHandlerSemantic::Unknown
                && !is_linear_decodable_semantic(analysis.semantic)
        }) {
            diagnostics.push(
                "control-flow semantic recovered; successor state requires CFG exploration"
                    .to_owned(),
            );
        } else {
            diagnostics
                .push("handler codecs recovered but semantic effects remain unknown".to_owned());
        }
        status = RecoveryStatus::PartiallyRecovered;
    }
    let confidence = match status {
        RecoveryStatus::Recovered => 100,
        RecoveryStatus::PartiallyRecovered => 70,
        _ => 0,
    };
    if let Some(spec) = &mut decode_spec {
        spec.confidence = spec.confidence.min(confidence);
    }
    Ok(HandlerCodecRecoveryReport {
        schema_version: 12,
        entry,
        status,
        trace_fragments: trace.fragments.len(),
        decoded_instructions: trace.decoded_instructions,
        trace_stop: trace.stop,
        slice,
        dispatch_analysis,
        operand_analysis,
        dispatch,
        operand,
        effects,
        handler_analysis,
        value_flow,
        native_call,
        table_checksum,
        vm_exit,
        semantic_metadata,
        boundary_spec,
        decode_spec,
        confidence,
        diagnostics,
    })
}

fn profile_role_candidate_is_viable(recovery: &HandlerCodecRecoveryReport) -> bool {
    let codecs_complete = codec_recovery_status(
        recovery.dispatch_analysis.candidates.len(),
        recovery.operand_analysis.candidates.len(),
        recovery.operand_analysis.rejected_sources.len(),
    ) == RecoveryStatus::Recovered;
    codecs_complete
        && recovery
            .handler_analysis
            .as_ref()
            .is_some_and(|analysis| analysis.semantic != VmHandlerSemantic::Unknown)
        && recovery
            .value_flow
            .as_ref()
            .is_some_and(|proof| proof.proven)
}

/// Resolves an ambiguous VSP register by demanding that the first handler of
/// the new VM profile has proven semantic effects under exactly one candidate.
/// No candidate is selected merely because it permits bytecode decoding.
///
/// # Errors
///
/// Propagates native trace decoding errors from the first handler.
pub fn resolve_vm_profile_roles(
    pe: &PeImage<'_>,
    first_handler: Rva,
    config: &VmProfileRoleResolutionConfig,
) -> Result<VmProfileRoleResolution, DisasmError> {
    let mut reports = Vec::new();
    let mut viable = Vec::new();
    let mut examined = BTreeSet::new();
    for vsp in config.vsp_candidates.iter().copied() {
        if !examined.insert(vsp) {
            continue;
        }
        let recovery = recover_handler_codecs(
            pe,
            first_handler,
            &HandlerCodecRecoveryConfig {
                roles: config.roles,
                bytecode_direction: config.bytecode_direction,
                effect_roles: Some(HandlerEffectRoles {
                    vsp,
                    context_base: config.context_base,
                }),
                live_out_registers: vec![
                    vsp,
                    config.roles.handler,
                    config.roles.vip,
                    config.roles.rolling_key,
                ],
                live_out_rflags: 0,
                trace_limits: config.trace_limits,
                proof_limits: config.proof_limits,
            },
        )?;
        let is_viable = profile_role_candidate_is_viable(&recovery);
        if is_viable {
            viable.push(vsp);
        }
        reports.push(VmProfileRoleCandidateReport {
            vsp_register: format!("{vsp:?}"),
            viable: is_viable,
            recovery,
        });
    }
    let (status, selected, diagnostics) = match viable.as_slice() {
        [vsp] => (RecoveryStatus::Recovered, Some(*vsp), Vec::new()),
        [] => (
            RecoveryStatus::Failed,
            None,
            vec!["no VSP candidate produced proven first-handler semantics".to_owned()],
        ),
        _ => (
            RecoveryStatus::Ambiguous,
            None,
            vec![format!(
                "{} VSP candidates produced proven first-handler semantics",
                viable.len()
            )],
        ),
    };
    let report = VmProfileRoleResolutionReport {
        schema_version: 1,
        first_handler,
        status,
        selected_vsp_register: selected.map(|vsp| format!("{vsp:?}")),
        candidates: reports,
        diagnostics,
    };
    let recovered = selected.map(|vsp| ResolvedVmProfileRoles {
        handler_roles: config.roles,
        effect_roles: HandlerEffectRoles {
            vsp,
            context_base: config.context_base,
        },
        bytecode_direction: config.bytecode_direction,
    });
    Ok(VmProfileRoleResolution { report, recovered })
}

fn linear_report(
    instructions: Vec<DecodedVmInstruction>,
    final_state: RelativeDispatchState,
    cumulative_vsp_delta: Option<i64>,
    stop_reason: LinearDecodeStop,
    unresolved_handler: Option<Va>,
    handler_recoveries: BTreeMap<Va, HandlerCodecRecoveryReport>,
) -> LinearHandlerRecoveryReport {
    LinearHandlerRecoveryReport {
        schema_version: 1,
        decode: LinearVmDecodeReport {
            schema_version: 2,
            instructions,
            final_state,
            cumulative_vsp_delta,
            stop_reason,
            unresolved_handler,
        },
        handler_recoveries,
    }
}

/// Recovers each native handler on demand and decodes a bounded straight-line
/// VM stream without an externally authored handler table.
///
/// Repeated complete decoder states terminate exploration. A handler whose
/// codecs or semantic effects remain unresolved is retained in the report and
/// produces `MissingHandlerSpec`; the decoder never invents an operation.
///
/// # Errors
///
/// Returns structured native-disassembly, address, or bytecode errors.
#[allow(clippy::too_many_lines)]
pub fn recover_and_decode_linear_vm_stream(
    pe: &PeImage<'_>,
    initial_state: RelativeDispatchState,
    config: &HandlerCodecRecoveryConfig,
    max_instructions: usize,
) -> Result<LinearHandlerRecoveryReport, LinearHandlerRecoveryError> {
    let mut instructions = Vec::new();
    let mut seen = BTreeSet::new();
    let mut state = initial_state;
    let mut cumulative_vsp_delta = Some(0_i64);
    let mut specs: BTreeMap<Va, VmHandlerDecodeSpec> = BTreeMap::new();
    let mut handler_recoveries = BTreeMap::new();

    for _ in 0..max_instructions {
        if !seen.insert(state) {
            return Ok(linear_report(
                instructions,
                state,
                cumulative_vsp_delta,
                LinearDecodeStop::RepeatedState,
                None,
                handler_recoveries,
            ));
        }
        let spec = if let Some(spec) = specs.get(&state.handler) {
            spec.clone()
        } else {
            let Some(handler_rva) = pe.optional.image_base.checked_rva_of(state.handler) else {
                let partial = linear_report(
                    instructions,
                    state,
                    cumulative_vsp_delta,
                    LinearDecodeStop::InvalidHandlerAddress,
                    Some(state.handler),
                    handler_recoveries,
                );
                return Err(LinearHandlerRecoveryError::HandlerOutsideImage {
                    handler: state.handler,
                    partial: Box::new(partial),
                });
            };
            let recovery = match recover_handler_codecs(pe, handler_rva, config) {
                Ok(recovery) => recovery,
                Err(source) => {
                    let partial = linear_report(
                        instructions,
                        state,
                        cumulative_vsp_delta,
                        LinearDecodeStop::NativeHandlerError,
                        Some(state.handler),
                        handler_recoveries,
                    );
                    return Err(LinearHandlerRecoveryError::NativeHandler {
                        handler: state.handler,
                        source,
                        partial: Box::new(partial),
                    });
                }
            };
            let spec = recovery.decode_spec.clone();
            let codecs_complete = codec_recovery_status(
                recovery.dispatch_analysis.candidates.len(),
                recovery.operand_analysis.candidates.len(),
                recovery.operand_analysis.rejected_sources.len(),
            ) == RecoveryStatus::Recovered;
            let semantic = recovery
                .handler_analysis
                .as_ref()
                .map(|analysis| analysis.semantic);
            let boundary_transport_proven = match semantic {
                Some(VmHandlerSemantic::VmExit) => recovery
                    .vm_exit
                    .as_ref()
                    .is_some_and(|report| report.status == RecoveryStatus::Recovered),
                Some(VmHandlerSemantic::NativeCall) => {
                    codecs_complete && recovery.boundary_spec.is_some()
                }
                Some(_) => codecs_complete,
                None => false,
            };
            let control_boundary = boundary_transport_proven
                && recovery
                    .value_flow
                    .as_ref()
                    .is_some_and(|proof| proof.proven)
                && recovery
                    .handler_analysis
                    .as_ref()
                    .is_some_and(|analysis| analysis.semantic.ends_linear_block());
            handler_recoveries.insert(state.handler, recovery);
            let Some(spec) = spec else {
                return Ok(linear_report(
                    instructions,
                    state,
                    cumulative_vsp_delta,
                    if control_boundary {
                        LinearDecodeStop::ControlFlowBoundary
                    } else {
                        LinearDecodeStop::MissingHandlerSpec
                    },
                    (!control_boundary).then_some(state.handler),
                    handler_recoveries,
                ));
            };
            specs.insert(state.handler, spec.clone());
            spec
        };
        let instruction = match spec.decode_direction(pe, state, config.bytecode_direction) {
            Ok(instruction) => instruction,
            Err(source) => {
                let partial = linear_report(
                    instructions,
                    state,
                    cumulative_vsp_delta,
                    LinearDecodeStop::BytecodeError,
                    Some(state.handler),
                    handler_recoveries,
                );
                return Err(LinearHandlerRecoveryError::Bytecode {
                    handler: state.handler,
                    source,
                    partial: Box::new(partial),
                });
            }
        };
        cumulative_vsp_delta = match (cumulative_vsp_delta, instruction.vsp_delta) {
            (Some(current), Some(delta)) => {
                if let Some(next) = current.checked_add(i64::from(delta)) {
                    Some(next)
                } else {
                    let source = BytecodeDecodeError::Address {
                        va: state.vip,
                        message: "cumulative VSP delta overflows".to_owned(),
                    };
                    let partial = linear_report(
                        instructions,
                        state,
                        cumulative_vsp_delta,
                        LinearDecodeStop::BytecodeError,
                        Some(state.handler),
                        handler_recoveries,
                    );
                    return Err(LinearHandlerRecoveryError::Bytecode {
                        handler: state.handler,
                        source,
                        partial: Box::new(partial),
                    });
                }
            }
            _ => None,
        };
        state = instruction.state_after;
        instructions.push(instruction);
    }

    Ok(linear_report(
        instructions,
        state,
        cumulative_vsp_delta,
        LinearDecodeStop::InstructionLimit,
        None,
        handler_recoveries,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ambiguous_or_rejected_operands_are_not_executable_statuses() {
        assert_eq!(codec_recovery_status(1, 2, 0), RecoveryStatus::Ambiguous);
        assert_eq!(
            codec_recovery_status(1, 1, 1),
            RecoveryStatus::PartiallyRecovered
        );
        assert_eq!(codec_recovery_status(1, 0, 0), RecoveryStatus::Recovered);
        assert_eq!(codec_recovery_status(0, 0, 0), RecoveryStatus::Failed);
        assert_eq!(codec_recovery_status(2, 0, 0), RecoveryStatus::Ambiguous);
    }
}
