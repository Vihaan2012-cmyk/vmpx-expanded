use iced_x86::{Instruction, InstructionInfoFactory, Mnemonic, OpKind, Register};
use serde::Serialize;
use vmpx_disasm::{DecodedInstruction, DirectJumpTrace};
use vmpx_pe::Rva;

use crate::{
    BitWidth, HandlerEffectSummary, HandlerMemoryRegion, NativeBackwardSlice, VmHandlerAnalysis,
    VmHandlerSemantic,
    dispatch_extract::{
        canonical, has_flat_segment_address, instruction_writes_view, is_active,
        same_register_value, signed_immediate, writes_register,
    },
};

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct HandlerValueFlowProof {
    pub schema_version: u32,
    pub semantic: VmHandlerSemantic,
    pub proven: bool,
    pub producer: Option<Rva>,
    pub consumer: Option<Rva>,
    pub evidence: Vec<String>,
}

fn active_prefix<'a>(
    trace: &'a DirectJumpTrace,
    slice: &NativeBackwardSlice,
    semantic_end: Rva,
) -> Vec<&'a DecodedInstruction> {
    let dispositions = slice
        .instructions
        .iter()
        .map(|instruction| (instruction.rva, instruction.disposition))
        .collect::<std::collections::BTreeMap<_, _>>();
    trace
        .instructions()
        .take_while(|instruction| instruction.rva != semantic_end)
        .filter(|instruction| {
            dispositions
                .get(&instruction.rva)
                .copied()
                .is_some_and(is_active)
        })
        .collect()
}

fn unique_effect_rva(
    summary: &HandlerEffectSummary,
    region: HandlerMemoryRegion,
    displacement: i64,
    width: BitWidth,
    reads: bool,
    writes: bool,
) -> Option<Rva> {
    let matches = summary
        .memory
        .iter()
        .filter(|effect| {
            effect.region == region
                && effect.displacement == displacement
                && effect.width_bytes == width.bytes()
                && effect.reads == reads
                && effect.writes == writes
                && match region {
                    HandlerMemoryRegion::NativeContext => effect.index.is_some(),
                    HandlerMemoryRegion::VmStack
                    | HandlerMemoryRegion::NativeStackScratch
                    | HandlerMemoryRegion::Bytecode
                    | HandlerMemoryRegion::Other => effect.index.is_none(),
                }
        })
        .map(|effect| effect.rva)
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [rva] => Some(*rva),
        _ => None,
    }
}

fn effect_has_entry_displacement(summary: &HandlerEffectSummary, rva: Rva, expected: i64) -> bool {
    summary.memory.iter().any(|effect| {
        effect.rva == rva
            && effect.region == HandlerMemoryRegion::VmStack
            && effect.entry_displacement == Some(expected)
    })
}

fn instruction_index(active: &[&DecodedInstruction], rva: Rva) -> Option<usize> {
    active.iter().position(|instruction| instruction.rva == rva)
}

fn latest_definition(
    active: &[&DecodedInstruction],
    before: usize,
    register: Register,
) -> Option<usize> {
    let mut factory = InstructionInfoFactory::new();
    (0..before)
        .rev()
        .find(|index| instruction_writes_view(&mut factory, &active[*index].instruction, register))
}

fn load_destination(instruction: &Instruction) -> Option<Register> {
    (matches!(
        instruction.mnemonic(),
        Mnemonic::Mov | Mnemonic::Movsx | Mnemonic::Movsxd | Mnemonic::Movzx
    ) && instruction.op0_kind() == OpKind::Register
        && instruction.op1_kind() == OpKind::Memory)
        .then(|| instruction.op0_register())
}

fn store_source(instruction: &Instruction) -> Option<Register> {
    (instruction.mnemonic() == Mnemonic::Mov
        && instruction.op0_kind() == OpKind::Memory
        && instruction.op1_kind() == OpKind::Register)
        .then(|| instruction.op1_register())
}

fn register_has_width(register: Register, width: BitWidth) -> bool {
    u32::try_from(register.size()).ok() == Some(width.bytes())
}

fn has_plain_memory_address(instruction: &Instruction) -> bool {
    instruction.memory_base() != Register::None
        && instruction.memory_base().size() == 8
        && instruction.memory_index() == Register::None
        && has_flat_segment_address(instruction)
}

fn context_base_is_stable(active: &[&DecodedInstruction], before: usize, base: Register) -> bool {
    let mut delta = 0_i64;
    let mut factory = InstructionInfoFactory::new();
    for decoded in active.iter().take(before) {
        let instruction = &decoded.instruction;
        if !writes_register(&mut factory, instruction, canonical(base)) {
            continue;
        }
        let adjustment = match instruction.mnemonic() {
            Mnemonic::Push | Mnemonic::Pushfq => Some(-8),
            Mnemonic::Pop
                if !(instruction.op0_kind() == OpKind::Register
                    && canonical(instruction.op0_register()) == canonical(base)) =>
            {
                Some(8)
            }
            Mnemonic::Popfq => Some(8),
            Mnemonic::Inc
                if instruction.op0_kind() == OpKind::Register
                    && instruction.op0_register().size() == 8
                    && canonical(instruction.op0_register()) == canonical(base) =>
            {
                Some(1)
            }
            Mnemonic::Dec
                if instruction.op0_kind() == OpKind::Register
                    && instruction.op0_register().size() == 8
                    && canonical(instruction.op0_register()) == canonical(base) =>
            {
                Some(-1)
            }
            Mnemonic::Add | Mnemonic::Sub
                if instruction.op0_kind() == OpKind::Register
                    && instruction.op0_register().size() == 8
                    && canonical(instruction.op0_register()) == canonical(base) =>
            {
                signed_immediate(instruction, 1).map(|value| {
                    if instruction.mnemonic() == Mnemonic::Sub {
                        value.saturating_neg()
                    } else {
                        value
                    }
                })
            }
            Mnemonic::Lea
                if instruction.op0_kind() == OpKind::Register
                    && instruction.op0_register().size() == 8
                    && canonical(instruction.op0_register()) == canonical(base)
                    && instruction.op1_kind() == OpKind::Memory
                    && instruction.memory_base().size() == 8
                    && canonical(instruction.memory_base()) == canonical(base)
                    && instruction.memory_index() == Register::None =>
            {
                Some(i64::from_ne_bytes(
                    instruction.memory_displacement64().to_ne_bytes(),
                ))
            }
            _ => None,
        };
        let Some(adjustment) = adjustment else {
            return false;
        };
        let Some(next) = delta.checked_add(adjustment) else {
            return false;
        };
        delta = next;
    }
    delta == 0
}

fn context_address_is_proven(
    active: &[&DecodedInstruction],
    index: usize,
    operand_register: Register,
    operand_width: BitWidth,
) -> bool {
    let instruction = &active[index].instruction;
    instruction.memory_base() != Register::None
        && instruction.memory_base().size() == 8
        && instruction.memory_index() != Register::None
        && instruction.memory_index().size() == 8
        && instruction.memory_index_scale() == 1
        && has_flat_segment_address(instruction)
        && same_register_value(operand_register, instruction.memory_index(), operand_width)
        && context_base_is_stable(active, index, instruction.memory_base())
}

fn definition_is_load(
    active: &[&DecodedInstruction],
    summary: &HandlerEffectSummary,
    index: usize,
    expected_register: Register,
    region: HandlerMemoryRegion,
    displacement: i64,
    width: BitWidth,
) -> bool {
    let instruction = &active[index].instruction;
    load_destination(instruction)
        .is_some_and(|destination| same_register_value(destination, expected_register, width))
        && match region {
            HandlerMemoryRegion::VmStack | HandlerMemoryRegion::Other => {
                has_plain_memory_address(instruction)
            }
            HandlerMemoryRegion::NativeContext => {
                instruction.memory_base().size() == 8
                    && instruction.memory_index().size() == 8
                    && has_flat_segment_address(instruction)
            }
            HandlerMemoryRegion::NativeStackScratch | HandlerMemoryRegion::Bytecode => true,
        }
        && unique_effect_rva(summary, region, displacement, width, true, false)
            == Some(active[index].rva)
        && (region != HandlerMemoryRegion::VmStack
            || effect_has_entry_displacement(summary, active[index].rva, displacement))
}

fn source_reaches_store_from_load(
    active: &[&DecodedInstruction],
    summary: &HandlerEffectSummary,
    store_rva: Rva,
    load_rva: Rva,
    load_region: HandlerMemoryRegion,
    load_displacement: i64,
    width: BitWidth,
) -> bool {
    let Some(store_index) = instruction_index(active, store_rva) else {
        return false;
    };
    let Some(source) = store_source(&active[store_index].instruction) else {
        return false;
    };
    register_has_width(source, width)
        && unique_effect_rva(summary, HandlerMemoryRegion::VmStack, 0, width, false, true)
            == Some(store_rva)
        && latest_definition(active, store_index, source).is_some_and(|definition| {
            active[definition].rva == load_rva
                && definition_is_load(
                    active,
                    summary,
                    definition,
                    source,
                    load_region,
                    load_displacement,
                    width,
                )
        })
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn operand_flow(
    semantic: VmHandlerSemantic,
    width: BitWidth,
    input_width: Option<BitWidth>,
    resize: Option<crate::ResizeOp>,
    active: &[&DecodedInstruction],
    summary: &HandlerEffectSummary,
    operand_consumer: Option<Rva>,
    operand_register: Option<Register>,
    operand_full_register_value_proven: bool,
    operand_width: Option<BitWidth>,
) -> Option<(Option<Rva>, Rva, String)> {
    match semantic {
        VmHandlerSemantic::ConstantPush => {
            let store =
                unique_effect_rva(summary, HandlerMemoryRegion::VmStack, 0, width, false, true)?;
            let store_index = instruction_index(active, store)?;
            let source = store_source(&active[store_index].instruction)?;
            let operand = operand_register?;
            let width_relation_proven = match (input_width, resize) {
                (Some(input_width), None) if input_width == width => {
                    same_register_value(operand, source, width)
                }
                (Some(input_width), Some(resize)) if input_width.bits() < width.bits() => {
                    let matching_resizes = summary
                        .resize_operations
                        .iter()
                        .filter(|effect| {
                            effect.operation == resize
                                && effect.input_width == input_width
                                && effect.output_width.bits() >= width.bits()
                                && summary.memory.iter().any(|memory| {
                                    memory.rva == effect.rva
                                        && memory.region == HandlerMemoryRegion::Bytecode
                                        && memory.width_bytes == input_width.bytes()
                                        && memory.reads
                                        && !memory.writes
                                })
                        })
                        .collect::<Vec<_>>();
                    let [resize_effect] = matching_resizes.as_slice() else {
                        return None;
                    };
                    let resize_index = instruction_index(active, resize_effect.rva)?;
                    let instruction = &active[resize_index].instruction;
                    let expected_mnemonic = match resize {
                        crate::ResizeOp::ZeroExtend => Mnemonic::Movzx,
                        crate::ResizeOp::SignExtend => {
                            if input_width == BitWidth::W32 && width == BitWidth::W64 {
                                Mnemonic::Movsxd
                            } else {
                                Mnemonic::Movsx
                            }
                        }
                        crate::ResizeOp::Truncate => return None,
                    };
                    instruction.mnemonic() == expected_mnemonic
                        && instruction.op0_kind() == OpKind::Register
                        && instruction.op0_register().size()
                            >= usize::try_from(width.bytes()).unwrap_or(usize::MAX)
                        && same_register_value(instruction.op0_register(), operand, input_width)
                        && same_register_value(operand, source, width)
                        && operand_full_register_value_proven
                }
                _ => false,
            };
            (operand_consumer == Some(store)
                && width_relation_proven
                && register_has_width(source, width)
                && summary.vsp_delta.is_some_and(|delta| {
                    effect_has_entry_displacement(summary, store, i64::from(delta))
                }))
            .then(|| {
                (
                    operand_consumer,
                    store,
                    if resize.is_some() {
                        "decoded operand's proven resize reaches the wider VM-stack store"
                            .to_owned()
                    } else {
                        "decoded operand's first consumer is the VM-stack store".to_owned()
                    },
                )
            })
        }
        VmHandlerSemantic::VirtualRegisterRead => {
            let input_width = input_width.unwrap_or(width);
            let context_load = unique_effect_rva(
                summary,
                HandlerMemoryRegion::NativeContext,
                0,
                input_width,
                true,
                false,
            )?;
            let stack_store =
                unique_effect_rva(summary, HandlerMemoryRegion::VmStack, 0, width, false, true)?;
            let context_index = instruction_index(active, context_load)?;
            let context_destination = load_destination(&active[context_index].instruction)?;
            let stack_index = instruction_index(active, stack_store)?;
            let stack_source = store_source(&active[stack_index].instruction)?;
            let width_relation_proven = match resize {
                None if input_width == width => {
                    definition_is_load(
                        active,
                        summary,
                        context_index,
                        context_destination,
                        HandlerMemoryRegion::NativeContext,
                        0,
                        width,
                    ) && register_has_width(context_destination, width)
                }
                Some(resize) if input_width.bits() < width.bits() => {
                    let expected_mnemonic = match resize {
                        crate::ResizeOp::ZeroExtend => Mnemonic::Movzx,
                        crate::ResizeOp::SignExtend => {
                            if input_width == BitWidth::W32 && width == BitWidth::W64 {
                                Mnemonic::Movsxd
                            } else {
                                Mnemonic::Movsx
                            }
                        }
                        crate::ResizeOp::Truncate => return None,
                    };
                    summary.resize_operations.iter().any(|effect| {
                        effect.rva == context_load
                            && effect.operation == resize
                            && effect.input_width == input_width
                            && effect.output_width.bits() >= width.bits()
                    }) && active[context_index].instruction.mnemonic() == expected_mnemonic
                        && register_has_width(context_destination, width)
                }
                _ => false,
            };
            (operand_consumer == Some(context_load)
                && operand_full_register_value_proven
                && operand_register.zip(operand_width).is_some_and(|(operand, operand_width)| {
                    context_address_is_proven(active, context_index, operand, operand_width)
                })
                && width_relation_proven
                && same_register_value(context_destination, stack_source, width)
                && register_has_width(stack_source, width)
                && latest_definition(active, stack_index, stack_source) == Some(context_index)
                && summary.vsp_delta.is_some_and(|delta| {
                    effect_has_entry_displacement(summary, stack_store, i64::from(delta))
                }))
            .then(|| {
                (
                    Some(context_load),
                    stack_store,
                    if resize.is_some() {
                        "decoded slot selects a context load whose proven resize reaches the wider VM-stack store"
                            .to_owned()
                    } else {
                        "decoded slot selects the context load whose value reaches the VM stack"
                            .to_owned()
                    },
                )
            })
        }
        VmHandlerSemantic::VirtualRegisterWrite => {
            let input_width = input_width.unwrap_or(width);
            let stack_load = unique_effect_rva(
                summary,
                HandlerMemoryRegion::VmStack,
                0,
                input_width,
                true,
                false,
            )?;
            let context_store = unique_effect_rva(
                summary,
                HandlerMemoryRegion::NativeContext,
                0,
                width,
                false,
                true,
            )?;
            let context_index = instruction_index(active, context_store)?;
            let context_instruction = &active[context_index].instruction;
            let context_source = store_source(context_instruction)?;
            let stack_load_index = instruction_index(active, stack_load)?;
            let stack_destination = load_destination(&active[stack_load_index].instruction)?;
            let width_relation_proven = match resize {
                None if input_width == width => {
                    same_register_value(stack_destination, context_source, width)
                        && register_has_width(stack_destination, width)
                }
                Some(crate::ResizeOp::Truncate) if input_width.bits() > width.bits() => {
                    register_has_width(stack_destination, input_width)
                        && same_register_value(stack_destination, context_source, width)
                }
                _ => false,
            };
            (operand_consumer == Some(context_store)
                && operand_full_register_value_proven
                && operand_register
                    .zip(operand_width)
                    .is_some_and(|(operand, operand_width)| {
                        context_address_is_proven(active, context_index, operand, operand_width)
                    })
                && register_has_width(context_source, width)
                && width_relation_proven
                && latest_definition(active, context_index, context_source)
                    == Some(stack_load_index)
                && definition_is_load(
                    active,
                    summary,
                    stack_load_index,
                    stack_destination,
                    HandlerMemoryRegion::VmStack,
                    0,
                    input_width,
                )
                && effect_has_entry_displacement(summary, stack_load, 0))
            .then(|| {
                (
                    Some(stack_load),
                    context_store,
                    "VM-stack value reaches the context store selected by the decoded slot"
                        .to_owned(),
                )
            })
        }
        _ => None,
    }
}

fn binary_input_definition(
    active: &[&DecodedInstruction],
    summary: &HandlerEffectSummary,
    operation_index: usize,
    operand: u32,
    displacement: i64,
    width: BitWidth,
) -> bool {
    let instruction = &active[operation_index].instruction;
    match instruction.op_kind(operand) {
        OpKind::Memory => {
            has_plain_memory_address(instruction)
                && unique_effect_rva(
                    summary,
                    HandlerMemoryRegion::VmStack,
                    displacement,
                    width,
                    true,
                    false,
                ) == Some(active[operation_index].rva)
                && effect_has_entry_displacement(summary, active[operation_index].rva, displacement)
        }
        OpKind::Register => {
            latest_definition(active, operation_index, instruction.op_register(operand))
                .is_some_and(|definition| {
                    definition_is_load(
                        active,
                        summary,
                        definition,
                        instruction.op_register(operand),
                        HandlerMemoryRegion::VmStack,
                        displacement,
                        width,
                    )
                })
        }
        _ => false,
    }
}

fn not_input_definition(
    active: &[&DecodedInstruction],
    summary: &HandlerEffectSummary,
    operation_index: usize,
    register: Register,
    displacement: i64,
    width: BitWidth,
) -> bool {
    let Some(not_index) = latest_definition(active, operation_index, register) else {
        return false;
    };
    let not = &active[not_index].instruction;
    not.mnemonic() == Mnemonic::Not
        && not.op0_kind() == OpKind::Register
        && same_register_value(not.op0_register(), register, width)
        && register_has_width(not.op0_register(), width)
        && latest_definition(active, not_index, register).is_some_and(|definition| {
            definition_is_load(
                active,
                summary,
                definition,
                register,
                HandlerMemoryRegion::VmStack,
                displacement,
                width,
            )
        })
}

#[allow(clippy::too_many_lines)]
fn arithmetic_flow(
    semantic: VmHandlerSemantic,
    width: BitWidth,
    active: &[&DecodedInstruction],
    summary: &HandlerEffectSummary,
) -> Option<(Rva, Rva, String)> {
    let result_store =
        unique_effect_rva(summary, HandlerMemoryRegion::VmStack, 8, width, false, true)?;
    let expected_result_entry = match semantic {
        VmHandlerSemantic::ShiftLeft
        | VmHandlerSemantic::ShiftRightLogical
        | VmHandlerSemantic::ShiftRightArithmetic
        | VmHandlerSemantic::RotateLeft
        | VmHandlerSemantic::RotateRight => i64::from(summary.vsp_delta?).checked_add(8)?,
        _ => i64::from(width.bytes()),
    };
    if !effect_has_entry_displacement(summary, result_store, expected_result_entry) {
        return None;
    }
    let store_index = instruction_index(active, result_store)?;
    let result_register = store_source(&active[store_index].instruction)?;
    if !register_has_width(result_register, width) {
        return None;
    }
    let operation_index = latest_definition(active, store_index, result_register)?;
    let operation = &active[operation_index].instruction;
    if operation.op0_kind() != OpKind::Register
        || !same_register_value(operation.op0_register(), result_register, width)
        || !register_has_width(operation.op0_register(), width)
    {
        return None;
    }
    let width_displacement = i64::from(width.bytes());
    let input_flow = match semantic {
        VmHandlerSemantic::Add | VmHandlerSemantic::Sub
            if operation.mnemonic()
                == if semantic == VmHandlerSemantic::Add {
                    Mnemonic::Add
                } else {
                    Mnemonic::Sub
                } =>
        {
            latest_definition(active, operation_index, result_register).is_some_and(|definition| {
                definition_is_load(
                    active,
                    summary,
                    definition,
                    result_register,
                    HandlerMemoryRegion::VmStack,
                    0,
                    width,
                )
            }) && (operation.op1_kind() != OpKind::Register
                || register_has_width(operation.op1_register(), width))
                && binary_input_definition(
                    active,
                    summary,
                    operation_index,
                    1,
                    width_displacement,
                    width,
                )
        }
        VmHandlerSemantic::Nor if operation.mnemonic() == Mnemonic::And => {
            not_input_definition(active, summary, operation_index, result_register, 0, width)
                && operation.op1_kind() == OpKind::Register
                && register_has_width(operation.op1_register(), width)
                && not_input_definition(
                    active,
                    summary,
                    operation_index,
                    operation.op1_register(),
                    width_displacement,
                    width,
                )
        }
        VmHandlerSemantic::Nand if operation.mnemonic() == Mnemonic::Or => {
            not_input_definition(active, summary, operation_index, result_register, 0, width)
                && operation.op1_kind() == OpKind::Register
                && register_has_width(operation.op1_register(), width)
                && not_input_definition(
                    active,
                    summary,
                    operation_index,
                    operation.op1_register(),
                    width_displacement,
                    width,
                )
        }
        VmHandlerSemantic::ShiftLeft
        | VmHandlerSemantic::ShiftRightLogical
        | VmHandlerSemantic::ShiftRightArithmetic
        | VmHandlerSemantic::RotateLeft
        | VmHandlerSemantic::RotateRight => {
            let expected = match semantic {
                VmHandlerSemantic::ShiftLeft => Mnemonic::Shl,
                VmHandlerSemantic::ShiftRightLogical => Mnemonic::Shr,
                VmHandlerSemantic::ShiftRightArithmetic => Mnemonic::Sar,
                VmHandlerSemantic::RotateLeft => Mnemonic::Rol,
                VmHandlerSemantic::RotateRight => Mnemonic::Ror,
                _ => unreachable!(),
            };
            latest_definition(active, operation_index, result_register).is_some_and(|definition| {
                definition_is_load(
                    active,
                    summary,
                    definition,
                    result_register,
                    HandlerMemoryRegion::VmStack,
                    0,
                    width,
                )
            }) && operation.mnemonic() == expected
                && operation.op1_kind() == OpKind::Register
                && latest_definition(active, operation_index, operation.op1_register()).is_some_and(
                    |definition| {
                        let count_slot_bytes = summary
                            .vsp_delta
                            .and_then(|delta| delta.checked_add(8))
                            .and_then(|bytes| u32::try_from(bytes).ok());
                        let instruction = &active[definition].instruction;
                        let destination = load_destination(instruction);
                        count_slot_bytes.is_some_and(|slot_width| {
                            slot_width > 0
                                && summary.memory.iter().any(|effect| {
                                    effect.rva == active[definition].rva
                                        && effect.region == HandlerMemoryRegion::VmStack
                                        && effect.displacement == width_displacement
                                        && effect.entry_displacement == Some(width_displacement)
                                        && effect.reads
                                        && !effect.writes
                                        && effect.index.is_none()
                                        && effect.width_bytes > 0
                                        && effect.width_bytes <= slot_width
                                })
                        }) && destination.is_some_and(|destination| {
                            same_register_value(destination, operation.op1_register(), BitWidth::W8)
                        })
                    },
                )
        }
        _ => false,
    };
    let flags_store = unique_effect_rva(
        summary,
        HandlerMemoryRegion::VmStack,
        0,
        BitWidth::W64,
        false,
        true,
    );
    let flags_position = flags_store.is_some_and(|store| {
        summary
            .vsp_delta
            .is_some_and(|delta| effect_has_entry_displacement(summary, store, i64::from(delta)))
    });
    (input_flow
        && flags_position
        && summary.captured_flags_source == Some(active[operation_index].rva))
    .then(|| {
        (
            active[operation_index].rva,
            result_store,
            "stack operands reach the native operation and its result/flags reach the VM stack"
                .to_owned(),
        )
    })
}

#[allow(clippy::too_many_lines)]
fn resized_binary_flow(
    semantic: VmHandlerSemantic,
    input_width: BitWidth,
    output_width: BitWidth,
    resize: crate::ResizeOp,
    active: &[&DecodedInstruction],
    summary: &HandlerEffectSummary,
) -> Option<(Rva, Rva, String)> {
    if resize != crate::ResizeOp::ZeroExtend || input_width.bits() >= output_width.bits() {
        return None;
    }
    let result_store = unique_effect_rva(
        summary,
        HandlerMemoryRegion::VmStack,
        8,
        output_width,
        false,
        true,
    )?;
    if !effect_has_entry_displacement(summary, result_store, i64::from(output_width.bytes())) {
        return None;
    }
    let store_index = instruction_index(active, result_store)?;
    let result_register = store_source(&active[store_index].instruction)?;
    if !register_has_width(result_register, output_width) {
        return None;
    }
    let operation_rva = summary.captured_flags_source?;
    let operation_index = instruction_index(active, operation_rva)?;
    let operation = &active[operation_index].instruction;
    let expected_mnemonic = match semantic {
        VmHandlerSemantic::Add => Mnemonic::Add,
        VmHandlerSemantic::Sub => Mnemonic::Sub,
        VmHandlerSemantic::Nor => Mnemonic::And,
        VmHandlerSemantic::Nand => Mnemonic::Or,
        _ => return None,
    };
    if operation.mnemonic() != expected_mnemonic
        || operation.op0_kind() != OpKind::Register
        || operation.op1_kind() != OpKind::Register
        || !register_has_width(operation.op0_register(), input_width)
        || !register_has_width(operation.op1_register(), input_width)
        || !same_register_value(operation.op0_register(), result_register, input_width)
    {
        return None;
    }
    let first_definition = latest_definition(active, operation_index, operation.op0_register())?;
    let first_load = if matches!(semantic, VmHandlerSemantic::Add | VmHandlerSemantic::Sub) {
        first_definition
    } else {
        let not = &active[first_definition].instruction;
        if not.mnemonic() != Mnemonic::Not
            || not.op0_kind() != OpKind::Register
            || !same_register_value(not.op0_register(), operation.op0_register(), input_width)
            || !register_has_width(not.op0_register(), input_width)
        {
            return None;
        }
        latest_definition(active, first_definition, operation.op0_register())?
    };
    let first_instruction = &active[first_load].instruction;
    let first_input_proven = first_instruction.mnemonic() == Mnemonic::Movzx
        && first_instruction.op0_kind() == OpKind::Register
        && first_instruction.op0_register().size()
            >= usize::try_from(output_width.bytes()).unwrap_or(usize::MAX)
        && definition_is_load(
            active,
            summary,
            first_load,
            operation.op0_register(),
            HandlerMemoryRegion::VmStack,
            0,
            input_width,
        )
        && summary.resize_operations.iter().any(|effect| {
            effect.rva == active[first_load].rva
                && effect.operation == resize
                && effect.input_width == input_width
                && effect.output_width.bits() >= output_width.bits()
        });
    let second_input_proven = if matches!(semantic, VmHandlerSemantic::Add | VmHandlerSemantic::Sub)
    {
        binary_input_definition(
            active,
            summary,
            operation_index,
            1,
            i64::from(output_width.bytes()),
            input_width,
        )
    } else {
        not_input_definition(
            active,
            summary,
            operation_index,
            operation.op1_register(),
            i64::from(output_width.bytes()),
            input_width,
        )
    };
    let result_flow_proven =
        latest_definition(active, store_index, result_register) == Some(operation_index);
    let flags_store = unique_effect_rva(
        summary,
        HandlerMemoryRegion::VmStack,
        0,
        BitWidth::W64,
        false,
        true,
    )?;
    let flags_position = summary
        .vsp_delta
        .is_some_and(|delta| effect_has_entry_displacement(summary, flags_store, i64::from(delta)));
    (first_input_proven && second_input_proven && result_flow_proven && flags_position).then(|| {
        (
            operation_rva,
            result_store,
            "narrow stack operands reach the native operation; its zero-extended result and flags reach the VM stack"
                .to_owned(),
        )
    })
}

fn timestamp_counter_flow(
    active: &[&DecodedInstruction],
    summary: &HandlerEffectSummary,
) -> Option<(Rva, Rva, String)> {
    let [source] = summary.timestamp_counter_reads.as_slice() else {
        return None;
    };
    let source_index = instruction_index(active, *source)?;
    if active[source_index].instruction.mnemonic() != Mnemonic::Rdtsc {
        return None;
    }
    let high_store = unique_effect_rva(
        summary,
        HandlerMemoryRegion::VmStack,
        0,
        BitWidth::W32,
        false,
        true,
    )?;
    let low_store = unique_effect_rva(
        summary,
        HandlerMemoryRegion::VmStack,
        4,
        BitWidth::W32,
        false,
        true,
    )?;
    let high_index = instruction_index(active, high_store)?;
    let low_index = instruction_index(active, low_store)?;
    let high_source = store_source(&active[high_index].instruction)?;
    let low_source = store_source(&active[low_index].instruction)?;
    let high_flow = same_register_value(high_source, Register::EDX, BitWidth::W32)
        && register_has_width(high_source, BitWidth::W32)
        && latest_definition(active, high_index, high_source) == Some(source_index)
        && effect_has_entry_displacement(summary, high_store, -8);
    let low_flow = same_register_value(low_source, Register::EAX, BitWidth::W32)
        && register_has_width(low_source, BitWidth::W32)
        && latest_definition(active, low_index, low_source) == Some(source_index)
        && effect_has_entry_displacement(summary, low_store, -4);
    (high_flow && low_flow).then(|| {
        (
            *source,
            low_store,
            "RDTSC's EDX:EAX outputs reach two ordered dword VM-stack stores".to_owned(),
        )
    })
}

fn unsigned_divide_registers(width: BitWidth) -> Option<(Register, Register, Register, Register)> {
    match width {
        BitWidth::W16 => Some((Register::DX, Register::AX, Register::DX, Register::AX)),
        BitWidth::W32 => Some((Register::EDX, Register::EAX, Register::EDX, Register::EAX)),
        BitWidth::W8 | BitWidth::W64 => None,
    }
}

#[allow(clippy::too_many_lines)]
fn unsigned_divide_flow(
    width: BitWidth,
    active: &[&DecodedInstruction],
    summary: &HandlerEffectSummary,
) -> Option<(Rva, Rva, String)> {
    let (high_input, low_input, remainder, quotient) = unsigned_divide_registers(width)?;
    let operation_rva = summary.captured_flags_source?;
    let operation_index = instruction_index(active, operation_rva)?;
    let operation = &active[operation_index].instruction;
    if operation.mnemonic() != Mnemonic::Div {
        return None;
    }
    let bytes = width.bytes();
    let high_definition = latest_definition(active, operation_index, high_input)?;
    let low_definition = latest_definition(active, operation_index, low_input)?;
    let high_input_proven = definition_is_load(
        active,
        summary,
        high_definition,
        high_input,
        HandlerMemoryRegion::VmStack,
        0,
        width,
    );
    let low_input_proven = definition_is_load(
        active,
        summary,
        low_definition,
        low_input,
        HandlerMemoryRegion::VmStack,
        i64::from(bytes),
        width,
    );
    let divisor_proven = binary_input_definition(
        active,
        summary,
        operation_index,
        0,
        i64::from(bytes.saturating_mul(2)),
        width,
    );
    let remainder_store =
        unique_effect_rva(summary, HandlerMemoryRegion::VmStack, 8, width, false, true)?;
    let quotient_store = unique_effect_rva(
        summary,
        HandlerMemoryRegion::VmStack,
        i64::from(8_u32.saturating_add(bytes)),
        width,
        false,
        true,
    )?;
    let remainder_index = instruction_index(active, remainder_store)?;
    let quotient_index = instruction_index(active, quotient_store)?;
    let remainder_source = store_source(&active[remainder_index].instruction)?;
    let quotient_source = store_source(&active[quotient_index].instruction)?;
    let result_flow = same_register_value(remainder_source, remainder, width)
        && same_register_value(quotient_source, quotient, width)
        && register_has_width(remainder_source, width)
        && register_has_width(quotient_source, width)
        && latest_definition(active, remainder_index, remainder_source) == Some(operation_index)
        && latest_definition(active, quotient_index, quotient_source) == Some(operation_index)
        && effect_has_entry_displacement(summary, remainder_store, i64::from(bytes))
        && effect_has_entry_displacement(
            summary,
            quotient_store,
            i64::from(bytes.saturating_mul(2)),
        );
    let flags_store = unique_effect_rva(
        summary,
        HandlerMemoryRegion::VmStack,
        0,
        BitWidth::W64,
        false,
        true,
    )?;
    let flags_flow = summary
        .vsp_delta
        .is_some_and(|delta| effect_has_entry_displacement(summary, flags_store, i64::from(delta)));
    (high_input_proven && low_input_proven && divisor_proven && result_flow && flags_flow).then(
        || {
            (
                operation_rva,
                quotient_store,
                "three stack operands reach DIV; quotient, remainder, and captured flags reach ordered VM-stack cells"
                    .to_owned(),
            )
        },
    )
}

#[allow(clippy::too_many_lines)]
fn unsigned_multiply_flow(
    width: BitWidth,
    active: &[&DecodedInstruction],
    summary: &HandlerEffectSummary,
) -> Option<(Rva, Rva, String)> {
    let (multiplier, accumulator, high_result, low_result) = unsigned_divide_registers(width)?;
    let operation_rva = summary.captured_flags_source?;
    let operation_index = instruction_index(active, operation_rva)?;
    let operation = &active[operation_index].instruction;
    if operation.mnemonic() != Mnemonic::Mul
        || operation.op0_kind() != OpKind::Register
        || !same_register_value(operation.op0_register(), multiplier, width)
        || !register_has_width(operation.op0_register(), width)
    {
        return None;
    }
    let bytes = width.bytes();
    let multiplier_definition = latest_definition(active, operation_index, multiplier)?;
    let accumulator_definition = latest_definition(active, operation_index, accumulator)?;
    let inputs_proven = definition_is_load(
        active,
        summary,
        multiplier_definition,
        multiplier,
        HandlerMemoryRegion::VmStack,
        0,
        width,
    ) && definition_is_load(
        active,
        summary,
        accumulator_definition,
        accumulator,
        HandlerMemoryRegion::VmStack,
        i64::from(bytes),
        width,
    );
    let high_store =
        unique_effect_rva(summary, HandlerMemoryRegion::VmStack, 8, width, false, true)?;
    let low_store = unique_effect_rva(
        summary,
        HandlerMemoryRegion::VmStack,
        i64::from(8_u32.saturating_add(bytes)),
        width,
        false,
        true,
    )?;
    let high_index = instruction_index(active, high_store)?;
    let low_index = instruction_index(active, low_store)?;
    let high_source = store_source(&active[high_index].instruction)?;
    let low_source = store_source(&active[low_index].instruction)?;
    let results_proven = same_register_value(high_source, high_result, width)
        && same_register_value(low_source, low_result, width)
        && register_has_width(high_source, width)
        && register_has_width(low_source, width)
        && latest_definition(active, high_index, high_source) == Some(operation_index)
        && latest_definition(active, low_index, low_source) == Some(operation_index)
        && effect_has_entry_displacement(summary, high_store, 0)
        && effect_has_entry_displacement(summary, low_store, i64::from(bytes));
    let flags_store = unique_effect_rva(
        summary,
        HandlerMemoryRegion::VmStack,
        0,
        BitWidth::W64,
        false,
        true,
    )?;
    let flags_proven = effect_has_entry_displacement(summary, flags_store, -8);
    (inputs_proven && results_proven && flags_proven).then(|| {
        (
            operation_rva,
            low_store,
            "two stack operands reach MUL; high/low product and captured flags reach ordered VM-stack cells"
                .to_owned(),
        )
    })
}

fn memory_load_flow(
    width: BitWidth,
    active: &[&DecodedInstruction],
    summary: &HandlerEffectSummary,
) -> Option<(Rva, Rva, String)> {
    let external_load =
        unique_effect_rva(summary, HandlerMemoryRegion::Other, 0, width, true, false)?;
    let result_store =
        unique_effect_rva(summary, HandlerMemoryRegion::VmStack, 0, width, false, true)?;
    if !source_reaches_store_from_load(
        active,
        summary,
        result_store,
        external_load,
        HandlerMemoryRegion::Other,
        0,
        width,
    ) {
        return None;
    }
    if !summary
        .vsp_delta
        .is_some_and(|delta| effect_has_entry_displacement(summary, result_store, i64::from(delta)))
    {
        return None;
    }
    let load_index = instruction_index(active, external_load)?;
    let load_instruction = &active[load_index].instruction;
    if !has_plain_memory_address(load_instruction) {
        return None;
    }
    let base = load_instruction.memory_base();
    let pointer_definition = latest_definition(active, load_index, base)?;
    definition_is_load(
        active,
        summary,
        pointer_definition,
        base,
        HandlerMemoryRegion::VmStack,
        0,
        BitWidth::W64,
    )
    .then(|| {
        (
            external_load,
            result_store,
            "VM-stack pointer reaches the external load and its value reaches the VM stack"
                .to_owned(),
        )
    })
}

fn resized_memory_load_flow(
    input_width: BitWidth,
    output_width: BitWidth,
    resize: crate::ResizeOp,
    active: &[&DecodedInstruction],
    summary: &HandlerEffectSummary,
) -> Option<(Rva, Rva, String)> {
    let resize_effects = summary
        .resize_operations
        .iter()
        .filter(|effect| {
            effect.operation == resize
                && effect.input_width == input_width
                && effect.output_width == output_width
        })
        .collect::<Vec<_>>();
    let [resize_effect] = resize_effects.as_slice() else {
        return None;
    };
    let external_load = unique_effect_rva(
        summary,
        HandlerMemoryRegion::Other,
        0,
        input_width,
        true,
        false,
    )?;
    if external_load != resize_effect.rva {
        return None;
    }
    let result_store = unique_effect_rva(
        summary,
        HandlerMemoryRegion::VmStack,
        0,
        output_width,
        false,
        true,
    )?;
    let load_index = instruction_index(active, external_load)?;
    let store_index = instruction_index(active, result_store)?;
    let load = &active[load_index].instruction;
    let store = &active[store_index].instruction;
    let destination = load_destination(load)?;
    let source = store_source(store)?;
    let expected_mnemonic = match resize {
        crate::ResizeOp::ZeroExtend => load.mnemonic() == Mnemonic::Movzx,
        crate::ResizeOp::SignExtend => {
            matches!(load.mnemonic(), Mnemonic::Movsx | Mnemonic::Movsxd)
        }
        crate::ResizeOp::Truncate => false,
    };
    if !expected_mnemonic
        || !has_plain_memory_address(load)
        || !register_has_width(destination, output_width)
        || !same_register_value(destination, source, output_width)
        || latest_definition(active, store_index, source) != Some(load_index)
        || !summary.vsp_delta.is_some_and(|delta| {
            effect_has_entry_displacement(summary, result_store, i64::from(delta))
        })
    {
        return None;
    }
    let pointer_definition = latest_definition(active, load_index, load.memory_base())?;
    definition_is_load(
        active,
        summary,
        pointer_definition,
        load.memory_base(),
        HandlerMemoryRegion::VmStack,
        0,
        BitWidth::W64,
    )
    .then(|| {
        (
            external_load,
            result_store,
            format!(
                "VM-stack pointer reaches an external {input_width:?} load whose value is {resize:?} to {output_width:?} and reaches the VM stack"
            ),
        )
    })
}

fn memory_store_flow(
    width: BitWidth,
    active: &[&DecodedInstruction],
    summary: &HandlerEffectSummary,
) -> Option<(Rva, Rva, String)> {
    let external_store =
        unique_effect_rva(summary, HandlerMemoryRegion::Other, 0, width, false, true)?;
    let store_index = instruction_index(active, external_store)?;
    let instruction = &active[store_index].instruction;
    let source = store_source(instruction)?;
    if !has_plain_memory_address(instruction) || !register_has_width(source, width) {
        return None;
    }
    let pointer_definition = latest_definition(active, store_index, instruction.memory_base())?;
    let value_definition = latest_definition(active, store_index, source)?;
    (definition_is_load(
        active,
        summary,
        pointer_definition,
        instruction.memory_base(),
        HandlerMemoryRegion::VmStack,
        0,
        BitWidth::W64,
    ) && definition_is_load(
        active,
        summary,
        value_definition,
        source,
        HandlerMemoryRegion::VmStack,
        8,
        width,
    ))
    .then(|| {
        (
            active[pointer_definition].rva,
            external_store,
            "VM-stack pointer and value reach the external store".to_owned(),
        )
    })
}

/// Proves the minimal register def-use links required by a structural handler
/// classification. Shape agreement alone is not sufficient for execution.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn prove_handler_value_flow(
    trace: &DirectJumpTrace,
    slice: &NativeBackwardSlice,
    summary: &HandlerEffectSummary,
    analysis: &VmHandlerAnalysis,
    operand_consumer: Option<Rva>,
    operand_register: Option<Register>,
    operand_full_register_value_proven: bool,
) -> HandlerValueFlowProof {
    let active = active_prefix(trace, slice, summary.semantic_end);
    let flow = analysis
        .operation_width
        .and_then(|width| match analysis.semantic {
            VmHandlerSemantic::ConstantPush
            | VmHandlerSemantic::VirtualRegisterRead
            | VmHandlerSemantic::VirtualRegisterWrite => operand_flow(
                analysis.semantic,
                width,
                analysis.input_width,
                analysis.resize,
                &active,
                summary,
                operand_consumer,
                operand_register,
                operand_full_register_value_proven,
                analysis.operand_width,
            ),
            VmHandlerSemantic::Add
            | VmHandlerSemantic::Sub
            | VmHandlerSemantic::Nor
            | VmHandlerSemantic::Nand
            | VmHandlerSemantic::ShiftLeft
            | VmHandlerSemantic::ShiftRightLogical
            | VmHandlerSemantic::ShiftRightArithmetic
            | VmHandlerSemantic::RotateLeft
            | VmHandlerSemantic::RotateRight => {
                let flow = match (analysis.semantic, analysis.input_width, analysis.resize) {
                    (
                        VmHandlerSemantic::Add
                        | VmHandlerSemantic::Sub
                        | VmHandlerSemantic::Nor
                        | VmHandlerSemantic::Nand,
                        Some(input_width),
                        Some(resize),
                    ) => resized_binary_flow(
                        analysis.semantic,
                        input_width,
                        width,
                        resize,
                        &active,
                        summary,
                    ),
                    (_, None, None) => arithmetic_flow(analysis.semantic, width, &active, summary),
                    _ => None,
                };
                flow.map(|(producer, consumer, evidence)| (Some(producer), consumer, evidence))
            }
            VmHandlerSemantic::MemoryLoad => match (analysis.input_width, analysis.resize) {
                (Some(input_width), Some(resize)) => {
                    resized_memory_load_flow(input_width, width, resize, &active, summary)
                }
                (Some(input_width), None) if input_width == width => {
                    memory_load_flow(width, &active, summary)
                }
                _ => None,
            }
            .map(|(producer, consumer, evidence)| (Some(producer), consumer, evidence)),
            VmHandlerSemantic::MemoryStore => memory_store_flow(width, &active, summary)
                .map(|(producer, consumer, evidence)| (Some(producer), consumer, evidence)),
            VmHandlerSemantic::ReadTimestampCounter => timestamp_counter_flow(&active, summary)
                .map(|(producer, consumer, evidence)| (Some(producer), consumer, evidence)),
            VmHandlerSemantic::UnsignedDivide => unsigned_divide_flow(width, &active, summary)
                .map(|(producer, consumer, evidence)| (Some(producer), consumer, evidence)),
            VmHandlerSemantic::Mul => unsigned_multiply_flow(width, &active, summary)
                .map(|(producer, consumer, evidence)| (Some(producer), consumer, evidence)),
            VmHandlerSemantic::Push
                if summary.pushes_pre_adjustment_vsp && summary.vsp_delta == Some(-8) =>
            {
                unique_effect_rva(
                    summary,
                    HandlerMemoryRegion::VmStack,
                    0,
                    BitWidth::W64,
                    false,
                    true,
                )
                .filter(|consumer| effect_has_entry_displacement(summary, *consumer, -8))
                .map(|consumer| {
                    (
                        None,
                        consumer,
                        "pre-adjustment VSP reaches its unique stack store".to_owned(),
                    )
                })
            }
            VmHandlerSemantic::Pop if summary.loads_vsp_from_top => unique_effect_rva(
                summary,
                HandlerMemoryRegion::VmStack,
                0,
                BitWidth::W64,
                true,
                false,
            )
            .filter(|consumer| effect_has_entry_displacement(summary, *consumer, 0))
            .map(|consumer| {
                (
                    None,
                    consumer,
                    "VSP is defined by the unique qword load from its current top".to_owned(),
                )
            }),
            VmHandlerSemantic::UnconditionalBranch
                if summary.loads_vip_from_vsp
                    && summary.resets_key_from_vip
                    && summary.reinitializes_handler_base =>
            {
                unique_effect_rva(
                    summary,
                    HandlerMemoryRegion::VmStack,
                    0,
                    BitWidth::W64,
                    true,
                    false,
                )
                .filter(|consumer| effect_has_entry_displacement(summary, *consumer, 0))
                .map(|consumer| {
                    (
                        None,
                        consumer,
                        "final VIP and key alias the popped target and the handler base is static"
                            .to_owned(),
                    )
                })
            }
            _ => None,
        });
    let proven = flow.is_some();
    let (producer, consumer, evidence) = flow.map_or_else(
        || {
            (
                None,
                None,
                vec!["required register def-use links were not proven".to_owned()],
            )
        },
        |(producer, consumer, evidence)| (producer, Some(consumer), vec![evidence]),
    );
    HandlerValueFlowProof {
        schema_version: 1,
        semantic: analysis.semantic,
        proven,
        producer,
        consumer,
        evidence,
    }
}

#[cfg(test)]
mod tests {
    use iced_x86::{Code, Decoder, DecoderOptions, code_asm::*};
    use vmpx_disasm::DecodedInstruction;
    use vmpx_pe::{Rva, Va};

    use super::*;

    fn decode(mut assembler: CodeAssembler) -> Vec<DecodedInstruction> {
        let bytes = assembler.assemble(0x1_4000_1000).expect("assemble");
        let mut decoder = Decoder::with_ip(64, &bytes, 0x1_4000_1000, DecoderOptions::NONE);
        let mut instructions = Vec::new();
        while decoder.can_decode() {
            let instruction = decoder.decode();
            assert_ne!(instruction.code(), Code::INVALID);
            let offset = usize::try_from(instruction.ip() - 0x1_4000_1000).expect("small offset");
            let length = instruction.len();
            instructions.push(DecodedInstruction {
                rva: Rva(0x1000 + u32::try_from(offset).expect("small RVA")),
                va: Va(instruction.ip()),
                bytes: bytes[offset..offset + length].to_vec(),
                instruction,
            });
        }
        instructions
    }

    #[test]
    fn distinguishes_high_and_low_byte_register_lanes() {
        assert!(!same_register_value(
            Register::AH,
            Register::AL,
            BitWidth::W8
        ));
        assert!(!same_register_value(
            Register::CH,
            Register::CL,
            BitWidth::W8
        ));
        assert!(same_register_value(
            Register::EAX,
            Register::AL,
            BitWidth::W8
        ));

        let mut assembler = CodeAssembler::new(64).expect("x64 assembler");
        assembler.mov(ch, byte_ptr(rbx)).expect("high-byte load");
        assembler.shr(al, cl).expect("low-byte shift count");
        let decoded = decode(assembler);
        let active = decoded.iter().collect::<Vec<_>>();
        assert_eq!(latest_definition(&active, 1, Register::CL), None);
    }

    #[test]
    fn rejects_truncated_or_fs_gs_external_addresses() {
        let mut assembler = CodeAssembler::new(64).expect("x64 assembler");
        assembler.mov(rax, qword_ptr(eax)).expect("32-bit address");
        assembler
            .mov(rbx, qword_ptr(rcx).fs())
            .expect("FS-relative address");
        assembler
            .mov(rcx, qword_ptr(rdx).ss())
            .expect("flat SS override");
        let decoded = decode(assembler);

        assert!(!has_plain_memory_address(&decoded[0].instruction));
        assert!(!has_plain_memory_address(&decoded[1].instruction));
        assert!(has_plain_memory_address(&decoded[2].instruction));
    }

    #[test]
    fn rejects_partial_or_value_replacing_context_base_updates() {
        let mut partial = CodeAssembler::new(64).expect("x64 assembler");
        partial.add(esp, 8).expect("partial adjustment");
        partial.sub(esp, 8).expect("partial inverse adjustment");
        let partial = decode(partial);
        let partial_active = partial.iter().collect::<Vec<_>>();
        assert!(!context_base_is_stable(
            &partial_active,
            partial_active.len(),
            Register::RSP
        ));

        let mut replacement = CodeAssembler::new(64).expect("x64 assembler");
        replacement.pop(rsp).expect("replace stack pointer");
        let replacement = decode(replacement);
        let replacement_active = replacement.iter().collect::<Vec<_>>();
        assert!(!context_base_is_stable(
            &replacement_active,
            replacement_active.len(),
            Register::RSP
        ));
    }
}
