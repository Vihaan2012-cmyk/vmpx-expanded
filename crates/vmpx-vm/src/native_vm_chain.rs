use std::collections::{BTreeMap, HashMap};

use thiserror::Error;
use vmpx_pe::{PeImage, Rva};

use crate::{
    BinaryOp, BitWidth, EntryProfileRecoveryError, EntryProfileRecoveryLimits, Expr, ExprError,
    NativeExitState, NativeFrameSource, RecoveredEntryProfile, SymbolicVmExecutionReport,
    SymbolicVmInitialState, VmEntryCandidate, VmProfileChainReport, VmProfileExecutionState,
    VmProfileExplorerError, VmProfileExplorerLimits, X64NativeRegister,
    canonicalize_symbolic_pointer, explore_vm_profile_chain, recover_entry_profile,
};

#[derive(Clone, Debug)]
pub struct NativeVmEntryContinuation {
    pub entry_profile: RecoveredEntryProfile,
    pub composed_initial: SymbolicVmInitialState,
    pub chain: VmProfileChainReport,
}

#[derive(Clone, Debug, Error, PartialEq)]
pub enum NativeVmEntryContinuationError {
    #[error("next VM entry profile recovery failed: {0}")]
    Entry(String),
    #[error("next VM entry has no complete native context map")]
    IncompleteContext,
    #[error("next VM entry frame has no direct-call return continuation")]
    MissingCallContinuation,
    #[error("next VM entry saved-frame size exceeds the supported range")]
    FrameSize,
    #[error("native-to-VM stack composition overflows")]
    StackComposition,
    #[error("native-to-VM expression composition failed: {0}")]
    Expression(String),
    #[error("next VM entry bootstrap dispatch failed: {0}")]
    Dispatch(String),
    #[error("continued VM profile exploration failed: {0}")]
    Explore(String),
}

impl From<EntryProfileRecoveryError> for NativeVmEntryContinuationError {
    fn from(error: EntryProfileRecoveryError) -> Self {
        Self::Entry(error.to_string())
    }
}

impl From<ExprError> for NativeVmEntryContinuationError {
    fn from(error: ExprError) -> Self {
        Self::Expression(error.to_string())
    }
}

impl From<VmProfileExplorerError> for NativeVmEntryContinuationError {
    fn from(error: VmProfileExplorerError) -> Self {
        Self::Explore(error.to_string())
    }
}

fn add_signed(expression: Expr, delta: i64) -> Result<Expr, NativeVmEntryContinuationError> {
    Expr::binary(
        BinaryOp::Add,
        expression,
        Expr::constant(BitWidth::W64, u64::from_ne_bytes(delta.to_ne_bytes())),
    )
    .map_err(Into::into)
}

fn byte_address(base: Expr, byte: u32) -> Result<Expr, NativeVmEntryContinuationError> {
    Expr::binary(
        BinaryOp::Add,
        base,
        Expr::constant(BitWidth::W64, u64::from(byte)),
    )
    .map_err(Into::into)
}

fn overlay_prior_memory(
    initial: &mut SymbolicVmInitialState,
    prior: &SymbolicVmExecutionReport,
) -> Result<(), NativeVmEntryContinuationError> {
    let next_frame = std::mem::take(&mut initial.memory_values);
    let mut byte_values = Vec::<(Expr, Expr)>::new();
    let mut indices = HashMap::<Expr, usize>::new();
    for (address, value) in prior.memory_values.iter().chain(&next_frame) {
        for byte in 0..value.width.bytes() {
            let address = canonicalize_symbolic_pointer(&byte_address(address.clone(), byte)?);
            let value = Expr::extract(BitWidth::W8, byte * 8, value.clone())?;
            if let Some(index) = indices.get(&address).copied() {
                byte_values[index].1 = value;
            } else {
                indices.insert(address.clone(), byte_values.len());
                byte_values.push((address, value));
            }
        }
    }
    initial.memory_values = byte_values;
    initial.memory_epoch = prior.memory_epoch;
    initial.memory_tainted = prior.memory_tainted;
    initial.next_native_call_epoch = prior.next_native_call_epoch;
    initial
        .fresh_allocations
        .clone_from(&prior.fresh_allocations);
    Ok(())
}

fn compose_replacements(
    pe: &PeImage<'_>,
    prior_symbolic: &SymbolicVmExecutionReport,
    prior_exit: &NativeExitState,
    next_profile: &RecoveredEntryProfile,
) -> Result<BTreeMap<String, Expr>, NativeVmEntryContinuationError> {
    let continuation = next_profile
        .native_frame
        .slots
        .iter()
        .find_map(|slot| match slot.source {
            NativeFrameSource::ReturnAddress { continuation, .. } => Some(continuation),
            _ => None,
        })
        .ok_or(NativeVmEntryContinuationError::MissingCallContinuation)?;
    frame_replacements(
        pe,
        prior_symbolic,
        prior_exit,
        next_profile.native_frame.slots.len(),
        Some(continuation),
    )
}

fn frame_replacements(
    pe: &PeImage<'_>,
    prior_symbolic: &SymbolicVmExecutionReport,
    prior_exit: &NativeExitState,
    next_slots: usize,
    continuation: Option<Rva>,
) -> Result<BTreeMap<String, Expr>, NativeVmEntryContinuationError> {
    let next_frame_size = next_slots
        .checked_mul(8)
        .and_then(|value| i64::try_from(value).ok())
        .ok_or(NativeVmEntryContinuationError::FrameSize)?;
    let prior_frame_size = i64::from(prior_exit.frame_size);
    let next_vsp = add_signed(
        prior_symbolic.final_vsp.clone(),
        prior_frame_size
            .checked_sub(next_frame_size)
            .ok_or(NativeVmEntryContinuationError::StackComposition)?,
    )?;
    let mut replacements = BTreeMap::from([
        ("vsp_initial".to_owned(), next_vsp),
        (
            "native_rflags_initial".to_owned(),
            prior_exit.rflags.clone(),
        ),
    ]);
    if let Some(continuation) = continuation {
        let continuation = pe
            .optional
            .image_base
            .checked_add_rva(continuation)
            .ok_or(NativeVmEntryContinuationError::MissingCallContinuation)?;
        replacements.insert(
            "native_return_address_initial".to_owned(),
            Expr::constant(BitWidth::W64, continuation.get()),
        );
    }
    for register in X64NativeRegister::ALL {
        let value = prior_exit
            .registers
            .get(&register)
            .cloned()
            .ok_or(NativeVmEntryContinuationError::IncompleteContext)?;
        replacements.insert(register.symbolic_name().to_owned(), value);
    }
    Ok(replacements)
}

/// Re-bases a nested entry's recovered native frame (`frame`, with symbolic
/// stack base, register, flags and return-address inputs) onto the exact state
/// a proven VM exit left behind.
///
/// # Errors
///
/// Fails closed on a missing exit register, stack-size overflow, or a
/// substitution/overlay width error.
pub(crate) fn compose_native_frame(
    pe: &PeImage<'_>,
    prior_symbolic: &SymbolicVmExecutionReport,
    prior_exit: &NativeExitState,
    mut frame: SymbolicVmInitialState,
    next_slots: usize,
    continuation: Option<Rva>,
) -> Result<SymbolicVmInitialState, NativeVmEntryContinuationError> {
    let replacements =
        frame_replacements(pe, prior_symbolic, prior_exit, next_slots, continuation)?;
    frame = frame.substitute(&replacements)?;
    overlay_prior_memory(&mut frame, prior_symbolic)?;
    Ok(frame)
}

/// Continues semantic VM exploration across a proven native VM-exit `RET`
/// whose target is another recovered VM-entry stub. The next entry's native
/// frame, call continuation, flags, registers, and saved-stack cursor are
/// composed from the exact prior exit state before decoding proceeds.
///
/// # Errors
///
/// Fails closed on incomplete frames, expression/stack overflow, ambiguous
/// bootstrap recovery, dispatch failure, or bounded profile-exploration error.
pub fn continue_through_native_vm_entry(
    pe: &PeImage<'_>,
    prior_symbolic: &SymbolicVmExecutionReport,
    prior_exit: &NativeExitState,
    next_entry: &VmEntryCandidate,
    entry_limits: EntryProfileRecoveryLimits,
    chain_limits: VmProfileExplorerLimits,
) -> Result<NativeVmEntryContinuation, NativeVmEntryContinuationError> {
    let entry_profile = recover_entry_profile(pe, next_entry, entry_limits)?;
    let replacements = compose_replacements(pe, prior_symbolic, prior_exit, &entry_profile)?;
    let mut composed_initial = entry_profile
        .context_mapping
        .symbolic_initial_state()
        .ok_or(NativeVmEntryContinuationError::IncompleteContext)?
        .substitute(&replacements)?;
    overlay_prior_memory(&mut composed_initial, prior_symbolic)?;
    let first = entry_profile
        .bootstrap
        .codec
        .decode_direction(
            pe,
            entry_profile.bootstrap.initial_state,
            entry_profile.bootstrap.bytecode_direction,
        )
        .map_err(|error| NativeVmEntryContinuationError::Dispatch(error.to_string()))?;
    let chain = explore_vm_profile_chain(
        pe,
        VmProfileExecutionState {
            decoder: first.state_after,
            profile: entry_profile.profile,
            symbolic: composed_initial.clone(),
        },
        chain_limits,
    )?;
    Ok(NativeVmEntryContinuation {
        entry_profile,
        composed_initial,
        chain,
    })
}
