use serde::Serialize;
use thiserror::Error;
use vmpx_disasm::{DirectJumpTraceLimits, DisasmError, RegisterDispatchProofLimits};
use vmpx_pe::PeImage;

use crate::{
    BootstrapDispatchRecovery, BootstrapDispatchReport, BootstrapRecoveryError,
    BytecodeDecodeError, HandlerCodecRecoveryConfig, HandlerEffectRoles,
    LinearHandlerRecoveryError, LinearHandlerRecoveryReport, NativeContextMappingReport,
    RecoveryStatus, SymbolicExecutionError, SymbolicExecutionLimits, SymbolicVmExecutionReport,
    VmEntryCandidate, VmProfile, X64NativeFrameReport, analyze_x64_native_frame,
    execute_symbolic_linear_block_with_initial_state, map_native_context_scatter,
    recover_and_decode_linear_vm_stream, recover_bootstrap_dispatch,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EntryProfileRecoveryLimits {
    pub trace: DirectJumpTraceLimits,
    pub dispatch_proof: RegisterDispatchProofLimits,
    pub instruction_limit: usize,
    pub symbolic: SymbolicExecutionLimits,
}

impl Default for EntryProfileRecoveryLimits {
    fn default() -> Self {
        Self {
            trace: DirectJumpTraceLimits::default(),
            dispatch_proof: RegisterDispatchProofLimits::default(),
            instruction_limit: 4_096,
            symbolic: SymbolicExecutionLimits::default(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct RecoveredEntryProfile {
    pub bootstrap_report: BootstrapDispatchReport,
    pub bootstrap: BootstrapDispatchRecovery,
    pub native_frame: X64NativeFrameReport,
    pub context_mapping: NativeContextMappingReport,
    pub profile: VmProfile,
    pub decode: LinearHandlerRecoveryReport,
    pub operational_instruction_offset: usize,
    pub symbolic: SymbolicVmExecutionReport,
}

#[derive(Clone, Debug, Eq, Error, PartialEq, Serialize)]
pub enum EntryProfileRecoveryError {
    #[error("bootstrap recovery failed: {0}")]
    Bootstrap(String),
    #[error("bootstrap dispatch remains ambiguous: {message}")]
    AmbiguousBootstrap { message: String },
    #[error("native frame analysis failed: {0}")]
    NativeFrame(String),
    #[error("native frame is not fully recovered: {message}")]
    IncompleteNativeFrame { message: String },
    #[error("native frame has no architectural cursor register")]
    MissingCursor,
    #[error("initial bootstrap dispatch failed: {0}")]
    InitialDispatch(String),
    #[error("entry profile decode failed: {0}")]
    Decode(String),
    #[error("context scatter is incomplete: {message}")]
    IncompleteContextScatter { message: String },
    #[error("entry profile has no classified control-flow boundary")]
    MissingBoundarySemantic,
    #[error("entry profile symbolic execution failed: {0}")]
    Symbolic(String),
}

impl From<BootstrapRecoveryError> for EntryProfileRecoveryError {
    fn from(error: BootstrapRecoveryError) -> Self {
        Self::Bootstrap(error.to_string())
    }
}

impl From<DisasmError> for EntryProfileRecoveryError {
    fn from(error: DisasmError) -> Self {
        Self::NativeFrame(error.to_string())
    }
}

impl From<BytecodeDecodeError> for EntryProfileRecoveryError {
    fn from(error: BytecodeDecodeError) -> Self {
        Self::InitialDispatch(error.to_string())
    }
}

impl From<LinearHandlerRecoveryError> for EntryProfileRecoveryError {
    fn from(error: LinearHandlerRecoveryError) -> Self {
        Self::Decode(error.to_string())
    }
}

impl From<SymbolicExecutionError> for EntryProfileRecoveryError {
    fn from(error: SymbolicExecutionError) -> Self {
        Self::Symbolic(error.to_string())
    }
}

/// Recovers a loader-visible VM entry through its first virtual control-flow
/// boundary using only native bootstrap data flow and handler behavior.
///
/// The routine derives the mutated register roles and first dispatch codec,
/// evaluates the immediate-seed bootstrap, maps the complete saved-native frame
/// through the decoded context scatter, then symbolically executes the
/// operational prefix. Fixture addresses, handler numbers, and authored
/// transform chains are not inputs.
///
/// # Errors
///
/// Fails closed on ambiguous bootstrap roles, an incomplete native frame or
/// context scatter, an unclassified boundary, or any decode/symbolic error.
pub fn recover_entry_profile(
    pe: &PeImage<'_>,
    entry: &VmEntryCandidate,
    limits: EntryProfileRecoveryLimits,
) -> Result<RecoveredEntryProfile, EntryProfileRecoveryError> {
    let (bootstrap_report, bootstrap) = recover_bootstrap_dispatch(pe, entry, limits.trace)?;
    let bootstrap = bootstrap.ok_or_else(|| EntryProfileRecoveryError::AmbiguousBootstrap {
        message: bootstrap_report.diagnostics.join("; "),
    })?;
    let native_frame = analyze_x64_native_frame(pe, entry.rva, limits.trace)?;
    if native_frame.status != RecoveryStatus::Recovered {
        return Err(EntryProfileRecoveryError::IncompleteNativeFrame {
            message: native_frame.diagnostics.join("; "),
        });
    }
    let vsp = native_frame
        .cursor_architectural_register
        .ok_or(EntryProfileRecoveryError::MissingCursor)?
        .to_iced();
    let first = bootstrap.codec.decode_direction(
        pe,
        bootstrap.initial_state,
        bootstrap.bytecode_direction,
    )?;
    let effect_roles = HandlerEffectRoles {
        vsp,
        context_base: iced_x86::Register::RSP,
    };
    let profile = VmProfile {
        handler_roles: bootstrap.roles,
        effect_roles,
        bytecode_direction: bootstrap.bytecode_direction,
    };
    let decode = recover_and_decode_linear_vm_stream(
        pe,
        first.state_after,
        &HandlerCodecRecoveryConfig {
            roles: profile.handler_roles,
            bytecode_direction: profile.bytecode_direction,
            effect_roles: Some(profile.effect_roles),
            live_out_registers: vec![
                vsp,
                profile.handler_roles.handler,
                profile.handler_roles.vip,
                profile.handler_roles.rolling_key,
            ],
            live_out_rflags: 0,
            trace_limits: limits.trace,
            proof_limits: limits.dispatch_proof,
        },
        limits.instruction_limit,
    )?;
    let operational_instruction_offset = native_frame.slots.len();
    let scatter = decode
        .decode
        .instructions
        .get(..operational_instruction_offset)
        .ok_or_else(|| EntryProfileRecoveryError::IncompleteContextScatter {
            message: format!(
                "decoded {} instructions for a {}-slot native frame",
                decode.decode.instructions.len(),
                operational_instruction_offset
            ),
        })?;
    let context_mapping = map_native_context_scatter(&native_frame, scatter);
    let symbolic_initial = context_mapping.symbolic_initial_state().ok_or_else(|| {
        EntryProfileRecoveryError::IncompleteContextScatter {
            message: context_mapping.diagnostics.join("; "),
        }
    })?;
    // A profile either ends at a linear block boundary (branch, native call,
    // exit) or at a cross-profile transition handler. Transition handlers have
    // no program-level semantic of their own, so requiring one would reject a
    // valid entry whose first profile hands control to another profile. Both
    // shapes are accepted here; the chain explorer proves which applies.
    let boundary_semantic = decode
        .handler_recoveries
        .get(&decode.decode.final_state.handler)
        .and_then(|handler| handler.handler_analysis.as_ref())
        .map(|analysis| analysis.semantic)
        .filter(|semantic| semantic.ends_linear_block());
    if boundary_semantic.is_none()
        && decode.decode.stop_reason != crate::LinearDecodeStop::MissingHandlerSpec
    {
        return Err(EntryProfileRecoveryError::MissingBoundarySemantic);
    }
    let operational = &decode.decode.instructions[operational_instruction_offset..];
    let symbolic = execute_symbolic_linear_block_with_initial_state(
        operational,
        boundary_semantic,
        limits.symbolic,
        &symbolic_initial,
        Some(pe),
    )?;
    Ok(RecoveredEntryProfile {
        bootstrap_report,
        bootstrap,
        native_frame,
        context_mapping,
        profile,
        decode,
        operational_instruction_offset,
        symbolic,
    })
}
