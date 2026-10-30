use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;
use thiserror::Error;

use crate::{
    BinaryOp, BitWidth, Expr, RecoveryStatus, SymbolicMemoryReadError, SymbolicVmExecutionReport,
    VmExitReport, VmProfileChainReport, VmProfileChainStop, X64NativeRegister,
};

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct NativeExitState {
    pub schema_version: u32,
    pub registers: BTreeMap<X64NativeRegister, Expr>,
    pub rflags: Expr,
    pub return_address: Expr,
    pub frame_size: u32,
}

#[derive(Clone, Debug, Eq, Error, PartialEq, Serialize)]
pub enum NativeExitStateError {
    #[error("native exit is not proven recovered")]
    ExitNotRecovered,
    #[error("native exit restore layout is incomplete: {message}")]
    IncompleteLayout { message: String },
    #[error("native exit restore frame cannot be read: {0}")]
    Memory(#[from] SymbolicMemoryReadError),
    #[error("native exit restore-frame address arithmetic failed: {message}")]
    Address { message: String },
    #[error("profile chain reached VM exit without retained terminal symbolic/handler evidence")]
    MissingTerminalEvidence,
}

fn frame_address(base: &Expr, offset: u32) -> Result<Expr, NativeExitStateError> {
    Expr::binary(
        BinaryOp::Add,
        base.clone(),
        Expr::constant(BitWidth::W64, u64::from(offset)),
    )
    .map_err(|error| NativeExitStateError::Address {
        message: error.to_string(),
    })
}

/// Recovers the exact architectural state consumed by a proven VM-exit frame
/// from the terminal symbolic memory snapshot.
///
/// # Errors
///
/// Rejects unproven or incomplete exit layouts, duplicate/missing native
/// registers, address failures, and any restore-frame byte absent from the
/// symbolic memory snapshot. No unknown value is invented.
pub fn recover_native_exit_state(
    symbolic: &SymbolicVmExecutionReport,
    exit: &VmExitReport,
) -> Result<NativeExitState, NativeExitStateError> {
    if exit.status != RecoveryStatus::Recovered {
        return Err(NativeExitStateError::ExitNotRecovered);
    }
    let expected = X64NativeRegister::ALL.into_iter().collect::<BTreeSet<_>>();
    let actual = exit
        .restore_slots
        .iter()
        .map(|slot| slot.register)
        .collect::<BTreeSet<_>>();
    if exit.restore_slots.len() != expected.len() || actual != expected {
        return Err(NativeExitStateError::IncompleteLayout {
            message: "restore slots do not contain every non-RSP GPR exactly once".to_owned(),
        });
    }
    let flags_offset =
        exit.flags_stack_offset
            .ok_or_else(|| NativeExitStateError::IncompleteLayout {
                message: "flags stack offset is absent".to_owned(),
            })?;
    let return_offset =
        exit.return_stack_offset
            .ok_or_else(|| NativeExitStateError::IncompleteLayout {
                message: "return-address stack offset is absent".to_owned(),
            })?;
    let frame_size = exit
        .frame_size
        .ok_or_else(|| NativeExitStateError::IncompleteLayout {
            message: "restore-frame size is absent".to_owned(),
        })?;
    if return_offset.checked_add(8) != Some(frame_size) {
        return Err(NativeExitStateError::IncompleteLayout {
            message: "frame size does not end immediately after the native continuation".to_owned(),
        });
    }

    let mut registers = BTreeMap::new();
    for slot in &exit.restore_slots {
        let address = frame_address(&symbolic.final_vsp, slot.stack_offset)?;
        let value = symbolic.read_tracked_memory(address, BitWidth::W64)?;
        if registers.insert(slot.register, value).is_some() {
            return Err(NativeExitStateError::IncompleteLayout {
                message: format!("duplicate restore slot for {:?}", slot.register),
            });
        }
    }
    let rflags = symbolic.read_tracked_memory(
        frame_address(&symbolic.final_vsp, flags_offset)?,
        BitWidth::W64,
    )?;
    let return_address = symbolic.read_tracked_memory(
        frame_address(&symbolic.final_vsp, return_offset)?,
        BitWidth::W64,
    )?;
    Ok(NativeExitState {
        schema_version: 1,
        registers,
        rflags,
        return_address,
        frame_size,
    })
}

/// Extracts an exact native exit from a completed profile chain. Non-exit
/// chains return `Ok(None)`; a chain claiming `VmExit` without its terminal
/// symbolic state or handler proof is rejected.
///
/// # Errors
///
/// Propagates exact restore-frame recovery errors and rejects missing terminal
/// evidence instead of inventing a continuation state.
pub fn recover_profile_chain_native_exit_state(
    chain: &VmProfileChainReport,
) -> Result<Option<NativeExitState>, NativeExitStateError> {
    if chain.stop_reason != VmProfileChainStop::VmExit {
        return Ok(None);
    }
    let terminal = chain
        .segments
        .last()
        .ok_or(NativeExitStateError::MissingTerminalEvidence)?;
    let symbolic = terminal
        .symbolic
        .as_ref()
        .ok_or(NativeExitStateError::MissingTerminalEvidence)?;
    let exit = terminal
        .decode
        .handler_recoveries
        .get(&terminal.decode.decode.final_state.handler)
        .and_then(|handler| handler.vm_exit.as_ref())
        .ok_or(NativeExitStateError::MissingTerminalEvidence)?;
    recover_native_exit_state(symbolic, exit).map(Some)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{SymbolicExecutionStop, VmExitRestoreSlot};
    use vmpx_pe::Rva;

    fn memory_qword(output: &mut Vec<(Expr, Expr)>, base: &Expr, offset: u32, value: u64) {
        for byte in 0_u32..8 {
            let address = frame_address(base, offset + byte).expect("test frame address");
            output.push((
                address,
                Expr::constant(BitWidth::W8, (value >> (byte * 8)) & 0xFF),
            ));
        }
    }

    fn symbolic(memory_values: Vec<(Expr, Expr)>, vsp: Expr) -> SymbolicVmExecutionReport {
        SymbolicVmExecutionReport {
            schema_version: 4,
            executed_instructions: 0,
            stop_reason: SymbolicExecutionStop::ControlFlowBoundary,
            final_vsp: vsp,
            final_vsp_offset: Some(0),
            final_flags: Expr::constant(BitWidth::W64, 0),
            branch_target: None,
            branch_target_dependencies: Vec::new(),
            context_cells: BTreeMap::new(),
            memory_values,
            memory_epoch: 0,
            memory_tainted: false,
            image_reads_suppressed: false,
            assumed_external_disjoint_from_vm_stack: false,
            external_writes: Vec::new(),
            memory_havocs: Vec::new(),
            native_calls: Vec::new(),
            fresh_allocations: Vec::new(),
            next_native_call_epoch: 0,
            peak_expression_nodes: 1,
            tracked_memory_bytes: 136,
            resolved_image_bytes: 0,
        }
    }

    fn exit_report() -> VmExitReport {
        VmExitReport {
            schema_version: 2,
            entry: Rva(0x1000),
            status: RecoveryStatus::Recovered,
            stack_switch: Some(Rva(0x1000)),
            restored_registers: X64NativeRegister::ALL
                .into_iter()
                .map(|register| format!("{register:?}"))
                .collect(),
            restore_slots: X64NativeRegister::ALL
                .into_iter()
                .enumerate()
                .map(|(index, register)| VmExitRestoreSlot {
                    register,
                    stack_offset: u32::try_from(index).expect("small index") * 8,
                    pop: Rva(0x1001 + u32::try_from(index).expect("small index")),
                })
                .collect(),
            flags_restore: Some(Rva(0x1010)),
            flags_stack_offset: Some(120),
            native_return: Some(Rva(0x1011)),
            return_stack_offset: Some(128),
            frame_size: Some(136),
            confidence: 100,
            evidence: Vec::new(),
            diagnostics: Vec::new(),
        }
    }

    #[test]
    fn extracts_every_native_restore_value_from_exact_symbolic_bytes() {
        let vsp = Expr::register(BitWidth::W64, "exit_vsp");
        let mut memory = Vec::new();
        for (index, _) in X64NativeRegister::ALL.into_iter().enumerate() {
            memory_qword(
                &mut memory,
                &vsp,
                u32::try_from(index).expect("small index") * 8,
                0x1000 + u64::try_from(index).expect("small index"),
            );
        }
        memory_qword(&mut memory, &vsp, 120, 0x202);
        memory_qword(&mut memory, &vsp, 128, 0x1_4000_1234);
        let state = recover_native_exit_state(&symbolic(memory, vsp), &exit_report())
            .expect("recover exact exit state");
        assert_eq!(
            state.registers[&X64NativeRegister::Rax],
            Expr::constant(BitWidth::W64, 0x1000)
        );
        assert_eq!(state.rflags, Expr::constant(BitWidth::W64, 0x202));
        assert_eq!(
            state.return_address,
            Expr::constant(BitWidth::W64, 0x1_4000_1234)
        );
    }

    #[test]
    fn refuses_incomplete_restore_memory() {
        let vsp = Expr::register(BitWidth::W64, "exit_vsp");
        assert!(matches!(
            recover_native_exit_state(&symbolic(Vec::new(), vsp), &exit_report()),
            Err(NativeExitStateError::Memory(
                SymbolicMemoryReadError::MissingByte { .. }
            ))
        ));
    }
}
