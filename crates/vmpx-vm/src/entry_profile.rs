use serde::Serialize;
use thiserror::Error;
use vmpx_disasm::{
    DirectJumpTraceLimits, DisasmError, RegisterDispatchProofLimits, trace_direct_jump_fragments,
};
use vmpx_pe::{PeImage, Rva};

use crate::{
    BootstrapDispatchRecovery, BootstrapDispatchReport, BootstrapRecoveryError,
    BytecodeDecodeError, GlobalTableBase, HandlerCodecRecoveryConfig, HandlerEffectRoles,
    LinearHandlerRecoveryError, LinearHandlerRecoveryReport, NativeContextMappingReport,
    RecoveryStatus, RelativeDispatchState, SymbolicExecutionError, SymbolicExecutionLimits,
    SymbolicVmExecutionReport, VmEntryCandidate, VmProfile, X64NativeFrameReport,
    analyze_x64_native_frame, execute_symbolic_linear_block_with_initial_state,
    map_native_context_scatter, recover_and_decode_linear_vm_stream_with_table,
    recover_bootstrap_dispatch,
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

pub(crate) fn bootstrap_global_table(
    bootstrap: &BootstrapDispatchRecovery,
) -> Option<GlobalTableBase> {
    bootstrap
        .codec
        .table
        .and_then(|table| table.table_base)
        .zip(bootstrap.table_register)
        .map(|(base, register)| GlobalTableBase { register, base })
}

pub(crate) fn entry_stream_config(
    profile: &VmProfile,
    limits: EntryProfileRecoveryLimits,
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

/// Walks the linear VM stream from `state`, threading the bootstrap-proven table
/// anchor (when the dispatch is table-indexed) into every handler.
fn decode_entry_stream(
    pe: &PeImage<'_>,
    bootstrap: &BootstrapDispatchRecovery,
    state: RelativeDispatchState,
    profile: &VmProfile,
    limits: EntryProfileRecoveryLimits,
) -> Result<LinearHandlerRecoveryReport, LinearHandlerRecoveryError> {
    recover_and_decode_linear_vm_stream_with_table(
        pe,
        state,
        &entry_stream_config(profile, limits),
        limits.instruction_limit,
        bootstrap_global_table(bootstrap),
    )
}

/// The register the VM entry captures native RSP into, taken from the first
/// full-width `mov reg, rsp` on the call-target trace before any nested call.
fn last_rsp_capture(
    pe: &PeImage<'_>,
    call_target: Rva,
    limits: DirectJumpTraceLimits,
) -> Result<Option<iced_x86::Register>, DisasmError> {
    use iced_x86::{FlowControl, Mnemonic, OpKind, Register};
    let trace = trace_direct_jump_fragments(pe, call_target, limits)?;
    // The VM operand-stack pointer is the stack-pointer snapshot that is still
    // live when the first handler dispatches. A heavily mutated bootstrap may
    // take several `mov reg,rsp` / `lea reg,[rsp+disp]` snapshots before
    // dispatch, with the earlier destinations repurposed afterwards (e.g.
    // `mov r11,rsp` ... `lea r11,[rbp-200h]`), so the *last* snapshot before
    // any call is the surviving cursor; the earlier ones are mutation decoys.
    // A wrong choice makes handler shapes fail closed (the VM stalls) rather
    // than mis-decode, so taking the final snapshot never trades soundness.
    let mut cursor = None;
    for decoded in trace.instructions() {
        let instruction = &decoded.instruction;
        if instruction.flow_control() == FlowControl::Call {
            break;
        }
        if instruction.mnemonic() == Mnemonic::Mov
            && instruction.op0_kind() == OpKind::Register
            && instruction.op1_kind() == OpKind::Register
            && instruction.op0_register().size() == 8
            && instruction.op1_register() == Register::RSP
            && instruction.op0_register() != Register::RSP
        {
            cursor = Some(instruction.op0_register());
        }
        // Mutated bootstraps capture the stack pointer with `lea reg,[rsp+disp]`
        // instead of `mov reg,rsp` (e.g. `lea rbp,[rsp-28h]`). Base must be RSP
        // with no index; the frame layout recovers the displacement separately.
        if instruction.mnemonic() == Mnemonic::Lea
            && instruction.op0_kind() == OpKind::Register
            && instruction.op0_register().size() == 8
            && instruction.op0_register() != Register::RSP
            && instruction.memory_base() == Register::RSP
            && instruction.memory_index() == Register::None
        {
            cursor = Some(instruction.op0_register());
        }
    }
    Ok(cursor)
}

/// The linear VM stream of an entry, decoded without requiring a fully proven
/// saved-native frame.
#[derive(Clone, Debug)]
pub struct RecoveredEntryStream {
    pub bootstrap_report: BootstrapDispatchReport,
    pub bootstrap: BootstrapDispatchRecovery,
    pub native_frame: X64NativeFrameReport,
    pub profile: VmProfile,
    pub decode: LinearHandlerRecoveryReport,
}

/// Decodes an entry's linear VM stream from the bootstrap-proven roles, initial
/// state and table anchor. The VSP register comes from the proven native frame
/// when available, otherwise from the last stack-pointer snapshot the bootstrap
/// takes before dispatch (`mov reg, rsp` / `lea reg,[rsp+disp]`); a wrong VSP
/// makes handler semantics fail closed rather than mis-decode.
///
/// # Errors
///
/// Fails on an ambiguous bootstrap, a missing VSP capture, or decode errors.
pub fn recover_entry_stream(
    pe: &PeImage<'_>,
    entry: &VmEntryCandidate,
    limits: EntryProfileRecoveryLimits,
) -> Result<RecoveredEntryStream, EntryProfileRecoveryError> {
    let setup = prepare_entry_stream(pe, entry, limits)?;
    let decode = decode_entry_stream(
        pe,
        &setup.bootstrap,
        setup.first_state,
        &setup.profile,
        limits,
    )?;
    Ok(RecoveredEntryStream {
        bootstrap_report: setup.bootstrap_report,
        bootstrap: setup.bootstrap,
        native_frame: setup.native_frame,
        profile: setup.profile,
        decode,
    })
}

/// Bootstrap-derived roles and the first dispatched decoder state of an entry.
pub(crate) struct EntryStreamSetup {
    pub bootstrap_report: BootstrapDispatchReport,
    pub bootstrap: BootstrapDispatchRecovery,
    pub native_frame: X64NativeFrameReport,
    pub profile: VmProfile,
    pub first_state: RelativeDispatchState,
}

pub(crate) fn prepare_entry_stream(
    pe: &PeImage<'_>,
    entry: &VmEntryCandidate,
    limits: EntryProfileRecoveryLimits,
) -> Result<EntryStreamSetup, EntryProfileRecoveryError> {
    let (bootstrap_report, bootstrap) = recover_bootstrap_dispatch(pe, entry, limits.trace)?;
    let bootstrap = bootstrap.ok_or_else(|| EntryProfileRecoveryError::AmbiguousBootstrap {
        message: bootstrap_report.diagnostics.join("; "),
    })?;
    let native_frame = analyze_x64_native_frame(pe, entry.rva, limits.trace)?;
    let vsp = match native_frame.cursor_architectural_register {
        Some(register) if native_frame.status == RecoveryStatus::Recovered => register.to_iced(),
        _ => entry
            .call_target
            .map(|target| last_rsp_capture(pe, target, limits.trace))
            .transpose()?
            .flatten()
            .ok_or(EntryProfileRecoveryError::MissingCursor)?,
    };
    let first = bootstrap.codec.decode_direction(
        pe,
        bootstrap.initial_state,
        bootstrap.bytecode_direction,
    )?;
    let profile = VmProfile {
        handler_roles: bootstrap.roles,
        effect_roles: HandlerEffectRoles {
            vsp,
            context_base: iced_x86::Register::RSP,
        },
        bytecode_direction: bootstrap.bytecode_direction,
    };
    Ok(EntryStreamSetup {
        bootstrap_report,
        bootstrap,
        native_frame,
        profile,
        first_state: first.state_after,
    })
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
    let decode = decode_entry_stream(pe, &bootstrap, first.state_after, &profile, limits)?;
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

#[cfg(test)]
mod last_rsp_capture_tests {
    use iced_x86::code_asm::*;
    use vmpx_disasm::DirectJumpTraceLimits;
    use vmpx_pe::{PeImage, Rva};

    use super::last_rsp_capture;

    // Minimal 64-bit PE with a single `.text` section holding `code` at RVA 0x1000.
    fn executable_pe64(code: &[u8]) -> Vec<u8> {
        let mut data = vec![0_u8; 0x800];
        data[0..2].copy_from_slice(&0x5A4D_u16.to_le_bytes());
        data[0x3C..0x40].copy_from_slice(&0x80_u32.to_le_bytes());
        data[0x80..0x84].copy_from_slice(&0x0000_4550_u32.to_le_bytes());
        let coff = 0x84;
        data[coff..coff + 2].copy_from_slice(&0x8664_u16.to_le_bytes());
        data[coff + 2..coff + 4].copy_from_slice(&1_u16.to_le_bytes());
        data[coff + 16..coff + 18].copy_from_slice(&0xF0_u16.to_le_bytes());
        data[coff + 18..coff + 20].copy_from_slice(&0x2022_u16.to_le_bytes());
        let optional = coff + 20;
        data[optional..optional + 2].copy_from_slice(&0x20B_u16.to_le_bytes());
        data[optional + 16..optional + 20].copy_from_slice(&0x1000_u32.to_le_bytes());
        data[optional + 20..optional + 24].copy_from_slice(&0x1000_u32.to_le_bytes());
        data[optional + 24..optional + 32].copy_from_slice(&0x1_4000_0000_u64.to_le_bytes());
        data[optional + 32..optional + 36].copy_from_slice(&0x1000_u32.to_le_bytes());
        data[optional + 36..optional + 40].copy_from_slice(&0x200_u32.to_le_bytes());
        data[optional + 56..optional + 60].copy_from_slice(&0x2000_u32.to_le_bytes());
        data[optional + 60..optional + 64].copy_from_slice(&0x200_u32.to_le_bytes());
        data[optional + 68..optional + 70].copy_from_slice(&3_u16.to_le_bytes());
        data[optional + 108..optional + 112].copy_from_slice(&16_u32.to_le_bytes());
        let section = optional + 0xF0;
        data[section..section + 5].copy_from_slice(b".text");
        data[section + 8..section + 12].copy_from_slice(&0x600_u32.to_le_bytes());
        data[section + 12..section + 16].copy_from_slice(&0x1000_u32.to_le_bytes());
        data[section + 16..section + 20].copy_from_slice(&0x600_u32.to_le_bytes());
        data[section + 20..section + 24].copy_from_slice(&0x200_u32.to_le_bytes());
        data[section + 36..section + 40].copy_from_slice(&0x6000_0020_u32.to_le_bytes());
        data[0x200..0x200 + code.len()].copy_from_slice(code);
        data
    }

    fn capture_of(code: &[u8]) -> Option<iced_x86::Register> {
        let bytes = executable_pe64(code);
        let pe = PeImage::parse(&bytes).expect("valid synthetic PE");
        last_rsp_capture(&pe, Rva(0x1000), DirectJumpTraceLimits::default())
            .expect("trace decodes")
    }

    // A mutated bootstrap snapshots RSP into a decoy register that is then
    // repurposed, before taking the real snapshot. The surviving (last) one wins.
    #[test]
    fn prefers_last_snapshot_over_repurposed_decoy() {
        let mut a = CodeAssembler::new(64).expect("asm");
        a.mov(r11, rsp).expect("decoy capture");
        a.lea(r11, qword_ptr(rbp - 0x200)).expect("repurpose decoy");
        a.mov(rbp, rsp).expect("real capture");
        a.jmp(r14).expect("dispatch");
        let code = a.assemble(0x1_4000_1000).expect("assemble");
        assert_eq!(capture_of(&code), Some(iced_x86::Register::RBP));
    }

    // A clean bootstrap with a single snapshot is unchanged.
    #[test]
    fn single_snapshot_is_the_cursor() {
        let mut a = CodeAssembler::new(64).expect("asm");
        a.lea(rbp, qword_ptr(rsp - 0x28)).expect("lea capture");
        a.jmp(r14).expect("dispatch");
        let code = a.assemble(0x1_4000_1000).expect("assemble");
        assert_eq!(capture_of(&code), Some(iced_x86::Register::RBP));
    }

    // Scanning stops at the first call: a snapshot after the call is ignored.
    #[test]
    fn ignores_snapshots_after_a_call() {
        let mut a = CodeAssembler::new(64).expect("asm");
        a.mov(rbp, rsp).expect("pre-call capture");
        a.call(0x1_4000_2000_u64).expect("call");
        a.mov(r12, rsp).expect("post-call capture");
        a.jmp(r14).expect("dispatch");
        let code = a.assemble(0x1_4000_1000).expect("assemble");
        assert_eq!(capture_of(&code), Some(iced_x86::Register::RBP));
    }
}
