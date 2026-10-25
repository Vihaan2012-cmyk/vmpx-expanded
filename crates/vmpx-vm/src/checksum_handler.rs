use iced_x86::{Instruction, InstructionInfoFactory, Mnemonic, OpAccess, OpKind, Register};
use serde::Serialize;
use vmpx_disasm::{
    DirectJumpTraceLimits, DirectJumpTraceStopKind, DisasmError, RegisterDispatchProofLimits,
    trace_direct_jump_fragments,
};
use vmpx_pe::{PeImage, Rva, Va};

use crate::{
    BitWidth, BytecodeDirection, ChecksumAccumulatorSource, ChecksumCombine, ChecksumShift,
    ChecksumTableIndex, DispatchExtractionReport, HandlerRegisterRoles, HandlerSliceConfig,
    RecoveryStatus, RelativeDispatchCodec, TableChecksumMetadata, backward_slice_handler_trace,
    extract_relative_dispatch_codecs_with_postlude_direction,
};

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct TableChecksumHandlerReport {
    pub schema_version: u32,
    pub entry: Rva,
    pub status: RecoveryStatus,
    pub pointer_load: Option<Rva>,
    pub length_load: Option<Rva>,
    pub loop_branch: Option<Rva>,
    pub output_store: Option<Rva>,
    pub common_tail: Option<Rva>,
    pub dispatch_analysis: Option<DispatchExtractionReport>,
    pub metadata: Option<TableChecksumMetadata>,
    pub confidence: u8,
    pub evidence: Vec<String>,
    pub diagnostics: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RecoveredTableChecksumHandler {
    pub report: TableChecksumHandlerReport,
    pub metadata: TableChecksumMetadata,
    pub dispatch: RelativeDispatchCodec,
    pub dispatch_vip_delta: i32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TableChecksumHandlerAnalysis {
    pub report: TableChecksumHandlerReport,
    pub recovered: Option<RecoveredTableChecksumHandler>,
}

fn canonical(register: Register) -> Register {
    register.full_register()
}

fn register_operand(instruction: &Instruction, operand: u32) -> Option<Register> {
    (instruction.op_kind(operand) == OpKind::Register).then(|| instruction.op_register(operand))
}

fn immediate(instruction: &Instruction, operand: u32) -> Option<u64> {
    instruction.try_immediate(operand).ok()
}

fn signed_immediate(instruction: &Instruction, operand: u32) -> Option<i64> {
    immediate(instruction, operand).map(|value| i64::from_ne_bytes(value.to_ne_bytes()))
}

fn memory_displacement(instruction: &Instruction) -> i64 {
    i64::from_ne_bytes(instruction.memory_displacement64().to_ne_bytes())
}

fn memory_width(instruction: &Instruction) -> usize {
    instruction.memory_size().size()
}

fn is_memory_operand(kind: OpKind) -> bool {
    matches!(
        kind,
        OpKind::Memory
            | OpKind::MemorySegSI
            | OpKind::MemorySegESI
            | OpKind::MemorySegRSI
            | OpKind::MemorySegDI
            | OpKind::MemorySegEDI
            | OpKind::MemorySegRDI
    )
}

fn direct_load_from(
    instruction: &Instruction,
    base: Register,
    displacement: i64,
    width: usize,
) -> Option<Register> {
    (instruction.mnemonic() == Mnemonic::Mov
        && register_operand(instruction, 0).is_some()
        && is_memory_operand(instruction.op1_kind())
        && canonical(instruction.memory_base()) == canonical(base)
        && instruction.memory_index() == Register::None
        && memory_displacement(instruction) == displacement
        && memory_width(instruction) == width)
        .then(|| register_operand(instruction, 0))
        .flatten()
}

fn writes_register(
    factory: &mut InstructionInfoFactory,
    instruction: &Instruction,
    register: Register,
) -> bool {
    factory
        .info(instruction)
        .used_registers()
        .iter()
        .any(|used| {
            canonical(used.register()) == canonical(register)
                && matches!(
                    used.access(),
                    OpAccess::Write
                        | OpAccess::CondWrite
                        | OpAccess::ReadWrite
                        | OpAccess::ReadCondWrite
                        | OpAccess::NoMemAccess
                )
        })
}

fn flags_defined(instruction: &Instruction) -> bool {
    instruction.rflags_written()
        | instruction.rflags_cleared()
        | instruction.rflags_set()
        | instruction.rflags_undefined()
        != 0
}

fn same_register_binary(instruction: &Instruction, mnemonic: Mnemonic, register: Register) -> bool {
    instruction.mnemonic() == mnemonic
        && register_operand(instruction, 0)
            .zip(register_operand(instruction, 1))
            .is_some_and(|(left, right)| {
                canonical(left) == canonical(register) && canonical(right) == canonical(register)
            })
}

fn static_lea_address(pe: &PeImage<'_>, instruction: &Instruction) -> Option<(Register, Va)> {
    if instruction.mnemonic() == Mnemonic::Mov
        && instruction.op0_kind() == OpKind::Register
        && !is_memory_operand(instruction.op1_kind())
        && let Some(value) = immediate(instruction, 1)
    {
        let address = Va(value);
        pe.optional.image_base.checked_rva_of(address)?;
        return Some((instruction.op0_register(), address));
    }
    if instruction.mnemonic() != Mnemonic::Lea || !is_memory_operand(instruction.op1_kind()) {
        return None;
    }
    let destination = register_operand(instruction, 0)?;
    let address = if instruction.is_ip_rel_memory_operand() {
        Va(instruction.ip_rel_memory_address())
    } else if instruction.memory_base() == Register::None
        && instruction.memory_index() == Register::None
    {
        Va(instruction.memory_displacement64())
    } else {
        return None;
    };
    pe.optional.image_base.checked_rva_of(address)?;
    Some((destination, address))
}

fn table_is_file_backed(pe: &PeImage<'_>, address: Va) -> bool {
    let Some(start) = pe.optional.image_base.checked_rva_of(address) else {
        return false;
    };
    let Some(end) = start.checked_add(1023) else {
        return false;
    };
    pe.rva_to_file_offset(start).is_ok() && pe.rva_to_file_offset(end).is_ok()
}

fn report_failure(
    entry: Rva,
    pointer_load: Option<Rva>,
    length_load: Option<Rva>,
    diagnostics: Vec<String>,
) -> TableChecksumHandlerAnalysis {
    TableChecksumHandlerAnalysis {
        report: TableChecksumHandlerReport {
            schema_version: 1,
            entry,
            status: RecoveryStatus::Failed,
            pointer_load,
            length_load,
            loop_branch: None,
            output_store: None,
            common_tail: None,
            dispatch_analysis: None,
            metadata: None,
            confidence: 0,
            evidence: Vec::new(),
            diagnostics,
        },
        recovered: None,
    }
}

/// Recognizes a counted W32 byte-table checksum loop and proves that both the
/// zero-length and loop-exit paths converge on the same ordinary threaded
/// dispatch tail. Constants and register choices are recovered from data flow;
/// no handler addresses, table contents, or mutation bytes are signatured.
///
/// # Errors
///
/// Native decode/mapping failures are returned. Pattern ambiguity and semantic
/// rejection remain structured diagnostics in the report.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn analyze_table_checksum_handler(
    pe: &PeImage<'_>,
    entry: Rva,
    roles: HandlerRegisterRoles,
    vsp: Register,
    bytecode_direction: BytecodeDirection,
    live_out_registers: &[Register],
    trace_limits: DirectJumpTraceLimits,
    proof_limits: RegisterDispatchProofLimits,
) -> Result<TableChecksumHandlerAnalysis, DisasmError> {
    let prefix = trace_direct_jump_fragments(pe, entry, trace_limits)?;
    if prefix.stop.kind != DirectJumpTraceStopKind::ConditionalBranch {
        return Ok(report_failure(
            entry,
            None,
            None,
            vec!["handler prefix does not end in a conditional zero-length split".to_owned()],
        ));
    }
    let prefix_instructions = prefix.instructions().collect::<Vec<_>>();
    let mut pointer = None;
    let mut length = None;
    for decoded in &prefix_instructions {
        if let Some(register) = direct_load_from(&decoded.instruction, vsp, 0, 8) {
            pointer = Some((decoded.rva, register));
        }
        if let Some(register) = direct_load_from(&decoded.instruction, vsp, 8, 8) {
            length = Some((decoded.rva, register));
        }
    }
    let Some((pointer_load, pointer_register)) = pointer else {
        return Ok(report_failure(
            entry,
            None,
            length.map(|(rva, _)| rva),
            vec!["no unique W64 pointer load from VSP+0 was recovered".to_owned()],
        ));
    };
    let Some((length_load, length_register)) = length else {
        return Ok(report_failure(
            entry,
            Some(pointer_load),
            None,
            vec!["no unique W64 length load from VSP+8 was recovered".to_owned()],
        ));
    };

    let vsp_adjustments = prefix_instructions
        .iter()
        .filter_map(|decoded| {
            let instruction = &decoded.instruction;
            (instruction.mnemonic() == Mnemonic::Add
                && register_operand(instruction, 0)
                    .is_some_and(|register| canonical(register) == canonical(vsp)))
            .then(|| signed_immediate(instruction, 1))
            .flatten()
        })
        .collect::<Vec<_>>();
    if vsp_adjustments.as_slice() != [12] {
        return Ok(report_failure(
            entry,
            Some(pointer_load),
            Some(length_load),
            vec![format!(
                "expected one packed pointer/length VSP adjustment of 12, recovered {vsp_adjustments:?}"
            )],
        ));
    }

    let Some(branch_instruction) = prefix_instructions.last() else {
        return Ok(report_failure(
            entry,
            Some(pointer_load),
            Some(length_load),
            vec!["conditional prefix contains no instructions".to_owned()],
        ));
    };
    let zero_successor = match branch_instruction.instruction.mnemonic() {
        Mnemonic::Je => prefix.stop.direct_target,
        Mnemonic::Jne => prefix.stop.fallthrough,
        mnemonic => {
            return Ok(report_failure(
                entry,
                Some(pointer_load),
                Some(length_load),
                vec![format!(
                    "length split uses {mnemonic:?}, expected equality/inequality"
                )],
            ));
        }
    };
    let nonzero_successor = match branch_instruction.instruction.mnemonic() {
        Mnemonic::Je => prefix.stop.fallthrough,
        Mnemonic::Jne => prefix.stop.direct_target,
        _ => None,
    };
    let (Some(zero_successor), Some(nonzero_successor)) = (zero_successor, nonzero_successor)
    else {
        return Ok(report_failure(
            entry,
            Some(pointer_load),
            Some(length_load),
            vec!["length split does not have two executable successors".to_owned()],
        ));
    };

    let test_index = prefix_instructions.iter().rposition(|decoded| {
        same_register_binary(&decoded.instruction, Mnemonic::Test, length_register)
    });
    let Some(test_index) = test_index else {
        return Ok(report_failure(
            entry,
            Some(pointer_load),
            Some(length_load),
            vec!["conditional split is not driven by TEST length,length".to_owned()],
        ));
    };
    if prefix_instructions[test_index + 1..prefix_instructions.len().saturating_sub(1)]
        .iter()
        .any(|decoded| flags_defined(&decoded.instruction))
    {
        return Ok(report_failure(
            entry,
            Some(pointer_load),
            Some(length_load),
            vec!["length flags are clobbered before the conditional split".to_owned()],
        ));
    }

    let zero_trace = trace_direct_jump_fragments(pe, zero_successor, trace_limits)?;
    let zero_instructions = zero_trace.instructions().collect::<Vec<_>>();
    let output = zero_instructions
        .iter()
        .enumerate()
        .find_map(|(not_index, decoded)| {
            let instruction = &decoded.instruction;
            if instruction.mnemonic() != Mnemonic::Not {
                return None;
            }
            let accumulator = register_operand(instruction, 0)?;
            if accumulator.size() != 4 {
                return None;
            }
            zero_instructions[not_index + 1..]
                .iter()
                .find(|candidate| {
                    let store = &candidate.instruction;
                    store.mnemonic() == Mnemonic::Mov
                        && is_memory_operand(store.op0_kind())
                        && canonical(store.memory_base()) == canonical(vsp)
                        && store.memory_index() == Register::None
                        && memory_displacement(store) == 0
                        && memory_width(store) == 4
                        && register_operand(store, 1)
                            .is_some_and(|source| canonical(source) == canonical(accumulator))
                })
                .map(|store| (accumulator, decoded.rva, store.rva))
        });
    let Some((accumulator, _final_not, output_store)) = output else {
        return Ok(report_failure(
            entry,
            Some(pointer_load),
            Some(length_load),
            vec!["zero-length tail does not finalize and store a W32 accumulator".to_owned()],
        ));
    };
    if !prefix_instructions.iter().any(|decoded| {
        matches!(
            decoded.instruction.mnemonic(),
            Mnemonic::Xor | Mnemonic::Sub
        ) && same_register_binary(
            &decoded.instruction,
            decoded.instruction.mnemonic(),
            accumulator,
        ) && register_operand(&decoded.instruction, 0).is_some_and(|register| register.size() >= 4)
    }) {
        return Ok(report_failure(
            entry,
            Some(pointer_load),
            Some(length_load),
            vec!["checksum accumulator is not initialized to zero before the split".to_owned()],
        ));
    }

    let zero_slice = backward_slice_handler_trace(
        &zero_trace,
        &HandlerSliceConfig {
            live_out_registers: live_out_registers.to_vec(),
            live_out_rflags: 0,
        },
    );
    let dispatch_analysis = extract_relative_dispatch_codecs_with_postlude_direction(
        pe,
        &zero_trace,
        &zero_slice,
        roles,
        bytecode_direction,
        proof_limits,
    )?;
    let [dispatch_candidate] = dispatch_analysis.candidates.as_slice() else {
        return Ok(TableChecksumHandlerAnalysis {
            report: TableChecksumHandlerReport {
                schema_version: 1,
                entry,
                status: RecoveryStatus::Failed,
                pointer_load: Some(pointer_load),
                length_load: Some(length_load),
                loop_branch: None,
                output_store: Some(output_store),
                common_tail: Some(zero_successor),
                dispatch_analysis: Some(dispatch_analysis),
                metadata: None,
                confidence: 0,
                evidence: Vec::new(),
                diagnostics: vec![
                    "zero-length common tail did not yield exactly one threaded dispatch codec"
                        .to_owned(),
                ],
            },
            recovered: None,
        });
    };
    let dispatch_candidate = dispatch_candidate.clone();

    let loop_trace = trace_direct_jump_fragments(pe, nonzero_successor, trace_limits)?;
    if loop_trace.stop.kind != DirectJumpTraceStopKind::ConditionalBranch {
        return Ok(report_failure(
            entry,
            Some(pointer_load),
            Some(length_load),
            vec!["nonzero path does not end in a counted loop branch".to_owned()],
        ));
    }
    let loop_instructions = loop_trace.instructions().collect::<Vec<_>>();
    let loop_rvas = loop_instructions
        .iter()
        .map(|decoded| decoded.rva)
        .collect::<std::collections::BTreeSet<_>>();
    let successors = [loop_trace.stop.direct_target, loop_trace.stop.fallthrough];
    let mut loop_entry = None;
    let mut loop_exit = None;
    for successor in successors.into_iter().flatten() {
        if loop_rvas.contains(&successor) {
            loop_entry = Some(successor);
        } else {
            loop_exit = Some(successor);
        }
    }
    let (Some(loop_entry), Some(loop_exit)) = (loop_entry, loop_exit) else {
        return Ok(report_failure(
            entry,
            Some(pointer_load),
            Some(length_load),
            vec!["counted loop does not have one back-edge and one exit".to_owned()],
        ));
    };
    let Some(loop_branch) = loop_instructions.last() else {
        return Ok(report_failure(
            entry,
            Some(pointer_load),
            Some(length_load),
            vec!["counted loop is empty".to_owned()],
        ));
    };
    if loop_branch.instruction.mnemonic() != Mnemonic::Jne {
        return Ok(report_failure(
            entry,
            Some(pointer_load),
            Some(length_load),
            vec!["counted loop back-edge is not JNE after decrement".to_owned()],
        ));
    }
    let dec_index = loop_instructions.iter().rposition(|decoded| {
        decoded.instruction.mnemonic() == Mnemonic::Dec
            && register_operand(&decoded.instruction, 0)
                .is_some_and(|register| canonical(register) == canonical(length_register))
    });
    let Some(dec_index) = dec_index else {
        return Ok(report_failure(
            entry,
            Some(pointer_load),
            Some(length_load),
            vec!["counted loop does not decrement the recovered length register".to_owned()],
        ));
    };
    if loop_instructions[dec_index + 1..loop_instructions.len().saturating_sub(1)]
        .iter()
        .any(|decoded| flags_defined(&decoded.instruction))
    {
        return Ok(report_failure(
            entry,
            Some(pointer_load),
            Some(length_load),
            vec!["loop-count flags are clobbered before JNE".to_owned()],
        ));
    }

    let lea_candidates = loop_instructions
        .iter()
        .filter_map(|decoded| {
            static_lea_address(pe, &decoded.instruction)
                .map(|(register, address)| (decoded.rva, register, address))
        })
        .collect::<Vec<_>>();
    let mut recurrence = None;
    for (lea_rva, table_register, table_address) in lea_candidates {
        if !table_is_file_backed(pe, table_address) {
            continue;
        }
        let byte_load = loop_instructions.iter().enumerate().find(|(_, decoded)| {
            let instruction = &decoded.instruction;
            instruction.mnemonic() == Mnemonic::Movzx
                && register_operand(instruction, 0).is_some()
                && is_memory_operand(instruction.op1_kind())
                && canonical(instruction.memory_base()) == canonical(pointer_register)
                && instruction.memory_index() == Register::None
                && memory_width(instruction) == 1
                && decoded.rva.0 >= loop_entry.0
        });
        let Some((byte_index, byte_load)) = byte_load else {
            continue;
        };
        let Some(index_register) = register_operand(&byte_load.instruction, 0) else {
            continue;
        };
        let xor_index = loop_instructions[byte_index + 1..]
            .iter()
            .position(|decoded| {
                decoded.instruction.mnemonic() == Mnemonic::Xor
                    && register_operand(&decoded.instruction, 0)
                        .is_some_and(|register| canonical(register) == canonical(index_register))
                    && register_operand(&decoded.instruction, 1)
                        .is_some_and(|register| canonical(register) == canonical(accumulator))
            })
            .map(|index| index + byte_index + 1);
        let Some(xor_index) = xor_index else {
            continue;
        };
        let mask_index = loop_instructions[xor_index + 1..]
            .iter()
            .position(|decoded| {
                decoded.instruction.mnemonic() == Mnemonic::And
                    && register_operand(&decoded.instruction, 0)
                        .is_some_and(|register| canonical(register) == canonical(index_register))
                    && immediate(&decoded.instruction, 1) == Some(0xff)
            })
            .map(|index| index + xor_index + 1);
        let Some(mask_index) = mask_index else {
            continue;
        };
        let table_index = loop_instructions[mask_index + 1..]
            .iter()
            .position(|decoded| {
                let instruction = &decoded.instruction;
                instruction.mnemonic() == Mnemonic::Mov
                    && register_operand(instruction, 0)
                        .is_some_and(|register| canonical(register) == canonical(index_register))
                    && is_memory_operand(instruction.op1_kind())
                    && canonical(instruction.memory_base()) == canonical(table_register)
                    && canonical(instruction.memory_index()) == canonical(index_register)
                    && instruction.memory_index_scale() == 4
                    && memory_width(instruction) == 4
            })
            .map(|index| index + mask_index + 1);
        let Some(table_index) = table_index else {
            continue;
        };
        let shift_index = loop_instructions[table_index + 1..]
            .iter()
            .position(|decoded| {
                decoded.instruction.mnemonic() == Mnemonic::Shr
                    && register_operand(&decoded.instruction, 0)
                        .is_some_and(|register| canonical(register) == canonical(accumulator))
                    && immediate(&decoded.instruction, 1) == Some(8)
            })
            .map(|index| index + table_index + 1);
        let Some(shift_index) = shift_index else {
            continue;
        };
        let combine_index = loop_instructions[shift_index + 1..]
            .iter()
            .position(|decoded| {
                decoded.instruction.mnemonic() == Mnemonic::Xor
                    && register_operand(&decoded.instruction, 0)
                        .is_some_and(|register| canonical(register) == canonical(accumulator))
                    && register_operand(&decoded.instruction, 1)
                        .is_some_and(|register| canonical(register) == canonical(index_register))
            })
            .map(|index| index + shift_index + 1);
        let Some(combine_index) = combine_index else {
            continue;
        };
        let iteration_xors = loop_instructions[combine_index + 1..dec_index]
            .iter()
            .filter_map(|decoded| {
                let instruction = &decoded.instruction;
                (instruction.mnemonic() == Mnemonic::Xor
                    && register_operand(instruction, 0)
                        .is_some_and(|register| canonical(register) == canonical(accumulator)))
                .then(|| immediate(instruction, 1))
                .flatten()
                .and_then(|value| u32::try_from(value).ok())
            })
            .collect::<Vec<_>>();
        let iteration_xor = match iteration_xors.as_slice() {
            [] => 0,
            [value] => *value,
            _ => continue,
        };
        let pointer_advances = loop_instructions[byte_index + 1..dec_index]
            .iter()
            .filter(|decoded| {
                let instruction = &decoded.instruction;
                (instruction.mnemonic() == Mnemonic::Inc
                    && register_operand(instruction, 0)
                        .is_some_and(|register| canonical(register) == canonical(pointer_register)))
                    || (instruction.mnemonic() == Mnemonic::Add
                        && register_operand(instruction, 0).is_some_and(|register| {
                            canonical(register) == canonical(pointer_register)
                        })
                        && signed_immediate(instruction, 1) == Some(1))
            })
            .count();
        if pointer_advances != 1 {
            continue;
        }
        recurrence = Some((
            lea_rva,
            table_address,
            byte_load.rva,
            iteration_xor,
            index_register,
        ));
        break;
    }
    let Some((lea_rva, table_address, byte_load, iteration_xor, index_register)) = recurrence
    else {
        return Ok(report_failure(
            entry,
            Some(pointer_load),
            Some(length_load),
            vec!["nonzero loop does not prove the supported W32 byte-table recurrence".to_owned()],
        ));
    };

    let exit_trace = trace_direct_jump_fragments(pe, loop_exit, trace_limits)?;
    let exit_instructions = exit_trace.instructions().collect::<Vec<_>>();
    if !exit_instructions
        .iter()
        .any(|decoded| decoded.rva == zero_successor)
    {
        return Ok(report_failure(
            entry,
            Some(pointer_load),
            Some(length_load),
            vec!["counted loop exit does not converge on the zero-length tail".to_owned()],
        ));
    }
    if canonical(index_register) == canonical(roles.vip) {
        let saves_vip = loop_instructions.iter().any(|decoded| {
            decoded.instruction.mnemonic() == Mnemonic::Push
                && register_operand(&decoded.instruction, 0)
                    .is_some_and(|register| canonical(register) == canonical(roles.vip))
        });
        let restores_vip = exit_instructions.iter().any(|decoded| {
            decoded.instruction.mnemonic() == Mnemonic::Pop
                && register_operand(&decoded.instruction, 0)
                    .is_some_and(|register| canonical(register) == canonical(roles.vip))
        });
        if !saves_vip || !restores_vip {
            return Ok(report_failure(
                entry,
                Some(pointer_load),
                Some(length_load),
                vec![
                    "loop aliases the VIP register without a proven balanced save/restore"
                        .to_owned(),
                ],
            ));
        }
    } else {
        let mut factory = InstructionInfoFactory::new();
        if loop_instructions
            .iter()
            .any(|decoded| writes_register(&mut factory, &decoded.instruction, roles.vip))
        {
            return Ok(report_failure(
                entry,
                Some(pointer_load),
                Some(length_load),
                vec![
                    "loop clobbers VIP without using it as the proven saved index register"
                        .to_owned(),
                ],
            ));
        }
    }

    let metadata = TableChecksumMetadata {
        checksum_width: BitWidth::W32,
        accumulator_source: ChecksumAccumulatorSource::Constant { value: 0 },
        initial_xor: 0,
        pointer_stack_offset: 0,
        length_stack_offset: 8,
        length_width: BitWidth::W64,
        pointer_stride: 1,
        table_address,
        table_entries: 256,
        table_entry_width: BitWidth::W32,
        table_index: ChecksumTableIndex::AccumulatorLowByteXorInputByte,
        shift: ChecksumShift::LogicalRight,
        shift_bits: 8,
        combine: ChecksumCombine::ShiftedAccumulatorXorTableEntry,
        iteration_xor,
        final_not: true,
        final_xor: 0,
        result_stack_offset: 0,
        vsp_delta: 12,
    };
    debug_assert!(metadata.validate().is_ok());
    let dispatch_vip_delta = i32::try_from(dispatch_candidate.vip_delta).unwrap_or(i32::MAX);
    if dispatch_vip_delta == i32::MAX {
        return Ok(report_failure(
            entry,
            Some(pointer_load),
            Some(length_load),
            vec!["threaded dispatch VIP delta exceeds i32".to_owned()],
        ));
    }
    let evidence = vec![
        "loads a W64 pointer and packed zero-extended length from VSP+0/VSP+8".to_owned(),
        "zero length and counted-loop exit converge on one W32 finalization/store tail".to_owned(),
        format!(
            "byte loop at {byte_load} indexes a file-backed 256xW32 table established at {lea_rva}"
        ),
        "each iteration combines SHR32(accumulator, 8) with the selected table entry".to_owned(),
        format!("each iteration advances the pointer and XORs {iteration_xor:#010X}"),
        "loop count is decremented to zero and any VIP alias is restored before dispatch"
            .to_owned(),
        "common tail yields exactly one ordinary keyed relative-dispatch codec".to_owned(),
    ];
    let report = TableChecksumHandlerReport {
        schema_version: 1,
        entry,
        status: RecoveryStatus::Recovered,
        pointer_load: Some(pointer_load),
        length_load: Some(length_load),
        loop_branch: Some(loop_branch.rva),
        output_store: Some(output_store),
        common_tail: Some(zero_successor),
        dispatch_analysis: Some(dispatch_analysis),
        metadata: Some(metadata.clone()),
        confidence: 100,
        evidence,
        diagnostics: Vec::new(),
    };
    Ok(TableChecksumHandlerAnalysis {
        report: report.clone(),
        recovered: Some(RecoveredTableChecksumHandler {
            report,
            metadata,
            dispatch: dispatch_candidate.codec,
            dispatch_vip_delta,
        }),
    })
}

#[cfg(test)]
mod tests {
    use iced_x86::code_asm::*;

    use super::*;

    fn executable_pe64(code: &[u8]) -> Vec<u8> {
        let mut data = vec![0_u8; 0x1600];
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
        data[optional + 56..optional + 60].copy_from_slice(&0x3000_u32.to_le_bytes());
        data[optional + 60..optional + 64].copy_from_slice(&0x200_u32.to_le_bytes());
        data[optional + 68..optional + 70].copy_from_slice(&3_u16.to_le_bytes());
        data[optional + 108..optional + 112].copy_from_slice(&16_u32.to_le_bytes());
        let section = optional + 0xF0;
        data[section..section + 5].copy_from_slice(b".text");
        data[section + 8..section + 12].copy_from_slice(&0x1400_u32.to_le_bytes());
        data[section + 12..section + 16].copy_from_slice(&0x1000_u32.to_le_bytes());
        data[section + 16..section + 20].copy_from_slice(&0x1400_u32.to_le_bytes());
        data[section + 20..section + 24].copy_from_slice(&0x200_u32.to_le_bytes());
        data[section + 36..section + 40].copy_from_slice(&0x6000_0020_u32.to_le_bytes());
        data[0x200..0x200 + code.len()].copy_from_slice(code);
        for index in 0..256_u32 {
            let offset = 0x1200 + usize::try_from(index * 4).expect("small table offset");
            data[offset..offset + 4].copy_from_slice(&(index ^ 0xA5A5_5A5A).to_le_bytes());
        }
        data
    }

    fn checksum_code(valid_combine: bool, subtract_to_zero: bool) -> Vec<u8> {
        let mut assembler = CodeAssembler::new(64).expect("x64 assembler");
        let mut loop_body = assembler.create_label();
        let mut common_tail = assembler.create_label();
        assembler.mov(r12, qword_ptr(rbp)).expect("pointer load");
        assembler.mov(r10, qword_ptr(rbp + 8)).expect("length load");
        assembler.add(rbp, 12).expect("consume packed inputs");
        if subtract_to_zero {
            assembler.sub(r9d, r9d).expect("zero accumulator");
        } else {
            assembler.xor(r9d, r9d).expect("zero accumulator");
        }
        assembler.test(r10, r10).expect("test length");
        assembler.jz(common_tail).expect("zero-length tail");
        assembler
            .mov(rdx, 0x1_4000_2000_u64)
            .expect("table address");
        assembler.push(rsi).expect("save aliased VIP");
        assembler.set_label(&mut loop_body).expect("loop label");
        assembler.movzx(esi, byte_ptr(r12)).expect("input byte");
        assembler.xor(esi, r9d).expect("form table index");
        assembler.and(esi, 0xff).expect("mask table index");
        assembler
            .mov(esi, dword_ptr(rdx + rsi * 4))
            .expect("table lookup");
        assembler.shr(r9d, 8).expect("shift accumulator");
        if valid_combine {
            assembler.xor(r9d, esi).expect("combine table entry");
        } else {
            assembler.add(r9d, esi).expect("invalid table combine");
        }
        assembler.inc(r12).expect("advance pointer");
        assembler.xor(r9d, 0x1234_5678_i32).expect("iteration xor");
        assembler.dec(r10).expect("decrement length");
        assembler.jne(loop_body).expect("loop back-edge");
        assembler.pop(rsi).expect("restore VIP");
        assembler.set_label(&mut common_tail).expect("tail label");
        assembler.not(r9d).expect("finalize checksum");
        assembler.mov(dword_ptr(rbp), r9d).expect("store checksum");
        assembler.mov(eax, dword_ptr(rsi)).expect("dispatch token");
        assembler.add(rsi, 4).expect("advance VIP");
        assembler.xor(eax, r8d).expect("decode dispatch");
        assembler.push(r8).expect("save rolling key");
        assembler
            .xor(dword_ptr(rsp), eax)
            .expect("update low rolling key");
        assembler.pop(r8).expect("restore updated key");
        assembler.movsxd(rax, eax).expect("extend displacement");
        assembler.add(rdi, rax).expect("update handler");
        assembler.jmp(rdi).expect("threaded dispatch");
        assembler.assemble(0x1_4000_1000).expect("assemble handler")
    }

    fn analyze(code: &[u8]) -> TableChecksumHandlerAnalysis {
        let bytes = executable_pe64(code);
        let pe = PeImage::parse(&bytes).expect("valid synthetic PE");
        analyze_table_checksum_handler(
            &pe,
            Rva(0x1000),
            HandlerRegisterRoles {
                vip: Register::RSI,
                rolling_key: Register::R8,
                handler: Register::RDI,
            },
            Register::RBP,
            BytecodeDirection::Ascending,
            &[Register::RBP, Register::RSI, Register::RDI, Register::R8],
            DirectJumpTraceLimits::default(),
            RegisterDispatchProofLimits::default(),
        )
        .expect("synthetic handler must decode")
    }

    #[test]
    fn recognizes_mutation_independent_table_checksum_contract() {
        let analysis = analyze(&checksum_code(true, false));
        let recovered = analysis
            .recovered
            .unwrap_or_else(|| panic!("checksum must recover: {:#?}", analysis.report));
        assert_eq!(recovered.metadata.iteration_xor, 0x1234_5678);
        assert_eq!(recovered.metadata.table_address, Va(0x1_4000_2000));
        assert_eq!(recovered.metadata.vsp_delta, 12);
        assert_eq!(recovered.dispatch_vip_delta, 4);
    }

    #[test]
    fn recognizes_subtract_to_zero_accumulator_initialization() {
        let analysis = analyze(&checksum_code(true, true));
        assert!(analysis.recovered.is_some(), "{:#?}", analysis.report);
    }

    #[test]
    fn rejects_non_xor_table_combination() {
        let analysis = analyze(&checksum_code(false, false));
        assert!(analysis.recovered.is_none());
        assert!(
            analysis
                .report
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.contains("recurrence"))
        );
    }
}
