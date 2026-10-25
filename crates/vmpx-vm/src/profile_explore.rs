use serde::Serialize;
use thiserror::Error;
use vmpx_disasm::{DirectJumpTraceLimits, DisasmError, RegisterDispatchProofLimits};
use vmpx_pe::{PeImage, Rva, Va};

use crate::{
    BytecodeDecodeError, BytecodeDirection, DecodedNativeCallBoundary, Expr, ExprKind,
    HandlerCodecRecoveryConfig, HandlerEffectRoles, HandlerRegisterRoles, LinearDecodeStop,
    LinearHandlerRecoveryError, LinearHandlerRecoveryReport, NativeCallBoundaryDecodeError,
    RecoveryStatus, RelativeDispatchState, SymbolicExecutionError, SymbolicExecutionLimits,
    SymbolicVmExecutionReport, SymbolicVmInitialState, VmHandlerSemantic,
    VmProfileRoleResolutionConfig, VmProfileRoleResolutionReport, VmTransitionReport,
    analyze_vm_transition, execute_symbolic_linear_block_with_initial_state,
    execute_symbolic_native_call_boundary_with_initial_state, recover_and_decode_linear_vm_stream,
    resolve_vm_profile_roles,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VmProfile {
    pub handler_roles: HandlerRegisterRoles,
    pub effect_roles: HandlerEffectRoles,
    pub bytecode_direction: BytecodeDirection,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VmProfileExecutionState {
    pub decoder: RelativeDispatchState,
    pub profile: VmProfile,
    pub symbolic: SymbolicVmInitialState,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VmProfileExplorerLimits {
    pub max_segments: usize,
    pub max_instructions_per_segment: usize,
    pub max_total_instructions: usize,
    pub trace: DirectJumpTraceLimits,
    pub dispatch_proof: RegisterDispatchProofLimits,
    pub symbolic: SymbolicExecutionLimits,
}

impl Default for VmProfileExplorerLimits {
    fn default() -> Self {
        Self {
            max_segments: 64,
            max_instructions_per_segment: 4_096,
            max_total_instructions: 65_536,
            trace: DirectJumpTraceLimits::default(),
            dispatch_proof: RegisterDispatchProofLimits::default(),
            symbolic: SymbolicExecutionLimits::default(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct VmProfileReport {
    pub vip_register: String,
    pub rolling_key_register: String,
    pub handler_register: String,
    pub vsp_register: String,
    pub context_base_register: String,
    pub bytecode_direction: BytecodeDirection,
}

impl From<VmProfile> for VmProfileReport {
    fn from(profile: VmProfile) -> Self {
        Self {
            vip_register: format!("{:?}", profile.handler_roles.vip),
            rolling_key_register: format!("{:?}", profile.handler_roles.rolling_key),
            handler_register: format!("{:?}", profile.handler_roles.handler),
            vsp_register: format!("{:?}", profile.effect_roles.vsp),
            context_base_register: format!("{:?}", profile.effect_roles.context_base),
            bytecode_direction: profile.bytecode_direction,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct VmProfileChainSegment {
    pub index: usize,
    pub initial_decoder_state: RelativeDispatchState,
    pub profile: VmProfileReport,
    pub symbolic_initial: SymbolicVmInitialState,
    pub decode: LinearHandlerRecoveryReport,
    pub symbolic: Option<SymbolicVmExecutionReport>,
    pub symbolic_error: Option<SymbolicExecutionError>,
    pub boundary_decode: Option<DecodedNativeCallBoundary>,
    pub boundary_symbolic: Option<SymbolicVmExecutionReport>,
    pub transition: Option<VmTransitionReport>,
    pub transition_pop: Option<SymbolicVmExecutionReport>,
    pub transition_target: Option<Expr>,
    pub next_profile_roles: Option<VmProfileRoleResolutionReport>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VmProfileChainStop {
    SegmentLimit,
    TotalInstructionLimit,
    DecoderBoundary,
    ControlFlowBoundary,
    UnsupportedTransition,
    SymbolicTransitionTarget,
    AmbiguousProfileRoles,
    SymbolicExecutionFailure,
    VmExit,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct VmProfileChainReport {
    pub schema_version: u32,
    pub segments: Vec<VmProfileChainSegment>,
    pub total_instructions: usize,
    pub stop_reason: VmProfileChainStop,
    pub next_decoder_state: Option<RelativeDispatchState>,
    pub diagnostics: Vec<String>,
}

#[derive(Debug, Error)]
pub enum VmProfileExplorerError {
    #[error("linear handler recovery failed: {0}")]
    Linear(#[from] LinearHandlerRecoveryError),

    #[error("native profile analysis failed: {0}")]
    Native(#[from] DisasmError),

    #[error("transition dispatch decode failed: {0}")]
    Bytecode(#[from] BytecodeDecodeError),

    #[error("native-call boundary decode failed: {0}")]
    NativeCallBoundary(#[from] NativeCallBoundaryDecodeError),

    #[error("symbolic profile execution failed: {0}")]
    Symbolic(#[from] SymbolicExecutionError),

    #[error("handler {handler} is outside the input image")]
    HandlerOutsideImage { handler: Va },
}

fn recovery_config(
    profile: VmProfile,
    limits: VmProfileExplorerLimits,
) -> HandlerCodecRecoveryConfig {
    HandlerCodecRecoveryConfig {
        roles: profile.handler_roles,
        bytecode_direction: profile.bytecode_direction,
        effect_roles: Some(profile.effect_roles),
        live_out_registers: vec![
            profile.effect_roles.vsp,
            profile.handler_roles.handler,
            profile.handler_roles.vip,
            profile.handler_roles.rolling_key,
        ],
        live_out_rflags: 0,
        trace_limits: limits.trace,
        proof_limits: limits.dispatch_proof,
    }
}

fn boundary_semantic(report: &LinearHandlerRecoveryReport) -> Option<VmHandlerSemantic> {
    (report.decode.stop_reason == LinearDecodeStop::ControlFlowBoundary)
        .then(|| {
            report
                .handler_recoveries
                .get(&report.decode.final_state.handler)
        })
        .flatten()
        .and_then(|recovery| recovery.handler_analysis.as_ref())
        .map(|analysis| analysis.semantic)
}

fn handler_rva(pe: &PeImage<'_>, handler: Va) -> Result<Rva, VmProfileExplorerError> {
    pe.optional
        .image_base
        .checked_rva_of(handler)
        .ok_or(VmProfileExplorerError::HandlerOutsideImage { handler })
}

fn stopped_report(
    segments: Vec<VmProfileChainSegment>,
    total_instructions: usize,
    stop_reason: VmProfileChainStop,
    next_decoder_state: Option<RelativeDispatchState>,
    diagnostics: Vec<String>,
) -> VmProfileChainReport {
    VmProfileChainReport {
        schema_version: 3,
        segments,
        total_instructions,
        stop_reason,
        next_decoder_state,
        diagnostics,
    }
}

/// Follows proven same-profile virtual branches and native profile changes
/// while carrying the complete symbolic VM stack/context state. A native
/// transition is continued only after its relevant memory effects, native RSP
/// balance, and context-base preservation are proven; this routine never
/// guesses constraints or merges decoder states.
///
/// # Errors
///
/// Returns structured native, decoder, or symbolic-execution failures. Normal
/// ambiguity and resource limits are represented in the returned report.
#[allow(clippy::too_many_lines)]
pub fn explore_vm_profile_chain(
    pe: &PeImage<'_>,
    initial: VmProfileExecutionState,
    limits: VmProfileExplorerLimits,
) -> Result<VmProfileChainReport, VmProfileExplorerError> {
    let mut current = initial;
    let mut segments = Vec::new();
    let mut total_instructions = 0_usize;

    for index in 0..limits.max_segments {
        let remaining_total = limits
            .max_total_instructions
            .saturating_sub(total_instructions);
        if remaining_total == 0 {
            return Ok(stopped_report(
                segments,
                total_instructions,
                VmProfileChainStop::TotalInstructionLimit,
                Some(current.decoder),
                vec![format!(
                    "reached total instruction limit {}",
                    limits.max_total_instructions
                )],
            ));
        }
        let segment_instruction_limit = limits.max_instructions_per_segment.min(remaining_total);
        let decode = recover_and_decode_linear_vm_stream(
            pe,
            current.decoder,
            &recovery_config(current.profile, limits),
            segment_instruction_limit,
        )?;
        let instruction_count = decode.decode.instructions.len();
        let Some(next_total) = total_instructions.checked_add(instruction_count) else {
            return Ok(stopped_report(
                segments,
                total_instructions,
                VmProfileChainStop::TotalInstructionLimit,
                Some(current.decoder),
                vec!["total decoded instruction count overflowed".to_owned()],
            ));
        };
        if next_total > limits.max_total_instructions {
            return Ok(stopped_report(
                segments,
                total_instructions,
                VmProfileChainStop::TotalInstructionLimit,
                Some(current.decoder),
                vec![format!(
                    "next segment would exceed total instruction limit {}",
                    limits.max_total_instructions
                )],
            ));
        }
        total_instructions = next_total;
        let boundary = boundary_semantic(&decode);
        let symbolic_boundary =
            boundary.filter(|semantic| *semantic != VmHandlerSemantic::NativeCall);
        let symbolic = execute_symbolic_linear_block_with_initial_state(
            &decode.decode.instructions,
            symbolic_boundary,
            limits.symbolic,
            &current.symbolic,
            Some(pe),
        );
        let (symbolic, symbolic_error) = match symbolic {
            Ok(report) => (Some(report), None),
            Err(error) => {
                // Preserve the exact state immediately before the failing
                // instruction. Re-executing the already proven prefix keeps
                // the existing Result API and never advances through the
                // failed semantic operation.
                let prefix_end = error
                    .instruction_index
                    .min(decode.decode.instructions.len());
                let partial = execute_symbolic_linear_block_with_initial_state(
                    &decode.decode.instructions[..prefix_end],
                    None,
                    limits.symbolic,
                    &current.symbolic,
                    Some(pe),
                )
                .ok();
                (partial, Some(error))
            }
        };
        let mut segment = VmProfileChainSegment {
            index,
            initial_decoder_state: current.decoder,
            profile: current.profile.into(),
            symbolic_initial: current.symbolic.clone(),
            decode,
            symbolic,
            symbolic_error,
            boundary_decode: None,
            boundary_symbolic: None,
            transition: None,
            transition_pop: None,
            transition_target: None,
            next_profile_roles: None,
        };

        if segment_instruction_limit == remaining_total
            && segment.decode.decode.stop_reason == LinearDecodeStop::InstructionLimit
        {
            let next = segment.decode.decode.final_state;
            segments.push(segment);
            return Ok(stopped_report(
                segments,
                total_instructions,
                VmProfileChainStop::TotalInstructionLimit,
                Some(next),
                vec![format!(
                    "reached total instruction limit {}",
                    limits.max_total_instructions
                )],
            ));
        }

        if let Some(error) = segment.symbolic_error.clone() {
            let next = segment.decode.decode.final_state;
            let diagnostic = error.to_string();
            segments.push(segment);
            return Ok(stopped_report(
                segments,
                total_instructions,
                VmProfileChainStop::SymbolicExecutionFailure,
                Some(next),
                vec![diagnostic],
            ));
        }

        if segment.decode.decode.stop_reason == LinearDecodeStop::ControlFlowBoundary {
            let next = segment.decode.decode.final_state;
            let Some(symbolic) = segment.symbolic.as_ref() else {
                segments.push(segment);
                return Ok(stopped_report(
                    segments,
                    total_instructions,
                    VmProfileChainStop::SymbolicExecutionFailure,
                    Some(next),
                    vec!["control boundary has no symbolic predecessor state".to_owned()],
                ));
            };
            if boundary == Some(VmHandlerSemantic::VmExit) {
                segments.push(segment);
                return Ok(stopped_report(
                    segments,
                    total_instructions,
                    VmProfileChainStop::VmExit,
                    None,
                    vec![
                        "proven VM exit restores native registers, flags, stack, and RET continuation"
                            .to_owned(),
                    ],
                ));
            }
            if boundary == Some(VmHandlerSemantic::NativeCall) {
                if total_instructions >= limits.max_total_instructions {
                    segments.push(segment);
                    return Ok(stopped_report(
                        segments,
                        total_instructions,
                        VmProfileChainStop::TotalInstructionLimit,
                        Some(next),
                        vec!["native-call boundary exceeds the total instruction limit".to_owned()],
                    ));
                }
                let spec = segment
                    .decode
                    .handler_recoveries
                    .get(&next.handler)
                    .and_then(|recovery| recovery.boundary_spec.as_ref());
                let Some(spec) = spec else {
                    segments.push(segment);
                    return Ok(stopped_report(
                        segments,
                        total_instructions,
                        VmProfileChainStop::UnsupportedTransition,
                        Some(next),
                        vec!["native-call boundary has no proven decode specification".to_owned()],
                    ));
                };
                let decoded =
                    spec.decode_direction(pe, next, current.profile.bytecode_direction)?;
                let call_symbolic = execute_symbolic_native_call_boundary_with_initial_state(
                    decoded.state_before.vip,
                    &decoded.outcome,
                    limits.symbolic,
                    &symbolic.continuation_state(),
                    Some(pe),
                )?;
                total_instructions = total_instructions.saturating_add(1);
                current = VmProfileExecutionState {
                    decoder: decoded.state_after,
                    profile: current.profile,
                    symbolic: call_symbolic.continuation_state(),
                };
                segment.boundary_decode = Some(decoded);
                segment.boundary_symbolic = Some(call_symbolic);
                segments.push(segment);
                continue;
            }
            segment
                .transition_target
                .clone_from(&symbolic.branch_target);
            if let Some(Expr {
                kind: ExprKind::Const { value: target },
                ..
            }) = &symbolic.branch_target
                && !symbolic.memory_tainted
                && let Some(recovery) = segment.decode.handler_recoveries.get(&next.handler)
                && recovery.handler_analysis.as_ref().is_some_and(|analysis| {
                    analysis.semantic == VmHandlerSemantic::UnconditionalBranch
                })
                && recovery.effects.as_ref().is_some_and(|effects| {
                    effects.loads_vip_from_vsp
                        && effects.resets_key_from_vip
                        && effects.reinitializes_handler_base
                })
                && let [dispatch] = recovery.dispatch_analysis.candidates.as_slice()
                && dispatch.vip_rebased_before_source
                && let Some(handler_anchor) = dispatch.handler_rebase
            {
                let first_dispatch = dispatch.codec.decode_direction(
                    pe,
                    RelativeDispatchState {
                        vip: Va(*target),
                        rolling_key: *target,
                        handler: handler_anchor,
                    },
                    current.profile.bytecode_direction,
                )?;
                current = VmProfileExecutionState {
                    decoder: first_dispatch.state_after,
                    profile: current.profile,
                    symbolic: symbolic.continuation_state(),
                };
                segments.push(segment);
                continue;
            }
            let diagnostic = if symbolic.memory_tainted {
                "virtual control-flow target follows an imprecise external-memory store; successor remains symbolic"
            } else {
                "virtual control-flow successor requires graph exploration"
            };
            segments.push(segment);
            return Ok(stopped_report(
                segments,
                total_instructions,
                VmProfileChainStop::ControlFlowBoundary,
                Some(next),
                vec![diagnostic.to_owned()],
            ));
        }
        let Some(unresolved_handler) = segment.decode.decode.unresolved_handler else {
            let next = segment.decode.decode.final_state;
            let decoder_stop = segment.decode.decode.stop_reason;
            segments.push(segment);
            return Ok(stopped_report(
                segments,
                total_instructions,
                VmProfileChainStop::DecoderBoundary,
                Some(next),
                vec![format!("linear decoder stopped with {decoder_stop:?}")],
            ));
        };
        let transition = analyze_vm_transition(
            pe,
            handler_rva(pe, unresolved_handler)?,
            current.profile.effect_roles.vsp,
            current.profile.effect_roles.context_base,
            limits.trace,
        )?;
        segment.transition = Some(transition.report.clone());
        let Some(recovered_transition) = transition.recovered else {
            let next = segment.decode.decode.final_state;
            let diagnostics = transition.report.diagnostics;
            segments.push(segment);
            return Ok(stopped_report(
                segments,
                total_instructions,
                VmProfileChainStop::UnsupportedTransition,
                Some(next),
                diagnostics,
            ));
        };
        debug_assert!(recovered_transition.report.state_effects_proven);
        let Some(symbolic) = segment.symbolic.as_ref() else {
            let next = segment.decode.decode.final_state;
            segments.push(segment);
            return Ok(stopped_report(
                segments,
                total_instructions,
                VmProfileChainStop::SymbolicExecutionFailure,
                Some(next),
                vec![
                    "transition analysis requires the preceding symbolic execution report"
                        .to_owned(),
                ],
            ));
        };
        let pop = execute_symbolic_linear_block_with_initial_state(
            &[],
            Some(VmHandlerSemantic::UnconditionalBranch),
            limits.symbolic,
            &symbolic.continuation_state(),
            Some(pe),
        )?;
        segment.transition_target.clone_from(&pop.branch_target);
        segment.transition_pop = Some(pop.clone());
        let Some(Expr {
            kind: ExprKind::Const { value: target },
            ..
        }) = pop.branch_target
        else {
            let next = segment.decode.decode.final_state;
            segments.push(segment);
            return Ok(stopped_report(
                segments,
                total_instructions,
                VmProfileChainStop::SymbolicTransitionTarget,
                Some(next),
                vec![
                    "transition target is not concrete under the current symbolic state".to_owned(),
                ],
            ));
        };
        let first_dispatch = recovered_transition.dispatch.decode_direction(
            pe,
            RelativeDispatchState {
                vip: Va(target),
                rolling_key: target,
                handler: recovered_transition.handler_anchor,
            },
            recovered_transition.report.bytecode_direction,
        )?;
        let roles = resolve_vm_profile_roles(
            pe,
            handler_rva(pe, first_dispatch.state_after.handler)?,
            &VmProfileRoleResolutionConfig {
                roles: recovered_transition.roles,
                bytecode_direction: recovered_transition.report.bytecode_direction,
                vsp_candidates: recovered_transition.vsp_candidates,
                context_base: current.profile.effect_roles.context_base,
                trace_limits: limits.trace,
                proof_limits: limits.dispatch_proof,
            },
        )?;
        segment.next_profile_roles = Some(roles.report.clone());
        let Some(resolved) = roles.recovered else {
            let next = first_dispatch.state_after;
            let diagnostics = roles.report.diagnostics;
            segments.push(segment);
            return Ok(stopped_report(
                segments,
                total_instructions,
                VmProfileChainStop::AmbiguousProfileRoles,
                Some(next),
                diagnostics,
            ));
        };
        debug_assert_eq!(roles.report.status, RecoveryStatus::Recovered);
        current = VmProfileExecutionState {
            decoder: first_dispatch.state_after,
            profile: VmProfile {
                handler_roles: resolved.handler_roles,
                effect_roles: resolved.effect_roles,
                bytecode_direction: resolved.bytecode_direction,
            },
            symbolic: pop.continuation_state(),
        };
        segments.push(segment);
    }

    Ok(stopped_report(
        segments,
        total_instructions,
        VmProfileChainStop::SegmentLimit,
        Some(current.decoder),
        vec![format!(
            "reached profile segment limit {}",
            limits.max_segments
        )],
    ))
}

#[cfg(test)]
mod tests {
    use iced_x86::Register;

    use super::*;

    #[test]
    fn zero_segment_limit_stops_without_work() {
        let report = stopped_report(
            Vec::new(),
            0,
            VmProfileChainStop::SegmentLimit,
            None,
            Vec::new(),
        );
        assert_eq!(report.stop_reason, VmProfileChainStop::SegmentLimit);
        assert!(report.segments.is_empty());
    }

    #[test]
    fn profile_report_preserves_exact_register_roles() {
        let report: VmProfileReport = VmProfile {
            handler_roles: HandlerRegisterRoles {
                vip: Register::R11,
                rolling_key: Register::R13,
                handler: Register::RBP,
            },
            effect_roles: HandlerEffectRoles {
                vsp: Register::R8,
                context_base: Register::RSP,
            },
            bytecode_direction: BytecodeDirection::Descending,
        }
        .into();
        assert_eq!(report.vip_register, "R11");
        assert_eq!(report.vsp_register, "R8");
        assert_eq!(report.bytecode_direction, BytecodeDirection::Descending);
    }
}
