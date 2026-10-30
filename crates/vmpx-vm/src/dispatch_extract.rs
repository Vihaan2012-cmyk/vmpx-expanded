use std::collections::{BTreeMap, BTreeSet};

use iced_x86::{
    FlowControl, Instruction, InstructionInfoFactory, Mnemonic, OpAccess, OpKind, Register,
};
use serde::Serialize;
use vmpx_disasm::{
    DecodedInstruction, DirectJumpTrace, DirectJumpTraceLimits, DirectJumpTraceStopKind,
    DisasmError, RegisterDispatchProof, RegisterDispatchProofLimits, decode_one, is_executable_rva,
    prove_register_dispatch_postlude, trace_direct_jump_fragments,
};
use vmpx_pe::{PeImage, Rva, Va};

use crate::{
    BitWidth, BytecodeDirection, HandlerSliceConfig, KeyOperation, NativeBackwardSlice,
    NativeInstructionDisposition, RelativeDispatchCodec, Transform, TransformChain, TransformOp,
    backward_slice_handler_trace,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HandlerRegisterRoles {
    pub vip: Register,
    pub rolling_key: Register,
    pub handler: Register,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct DispatchExtractionCandidate {
    pub encoded_load: Rva,
    pub decoded_register: String,
    /// The handler establishes a new full-width VIP before this local codec.
    /// Such a codec is usable only after the corresponding control-flow
    /// semantic has supplied that new VIP; it is not a linear state update.
    pub vip_rebased_before_source: bool,
    /// Static handler anchor established before the dispatch load, if any.
    pub handler_rebase: Option<Va>,
    pub handler_rebase_rva: Option<Rva>,
    /// Register holding the table base of a table-indexed dispatch.
    #[serde(skip)]
    pub table_register: Option<Register>,
    pub vip_delta: i64,
    pub handler_update: Rva,
    pub terminal_dispatch: Rva,
    pub codec: RelativeDispatchCodec,
    pub confidence: u8,
    pub evidence: Vec<String>,
    /// Set when the dispatch lives in the shared head reached through the
    /// handler's stack-check postlude rather than inside the handler trace.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub postlude: Option<PostludeDispatch>,
}

/// Registers of the shared stack-check postlude `lea tmp,[ctx+disp];
/// cmp vsp,tmp; ja head`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StackCheckRoles {
    pub vsp: Register,
    pub context_base: Register,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct PostludeDispatch {
    /// First instruction of the stack-check (the `lea`); handler effects end here.
    pub stack_check_start: Rva,
    pub branch: Rva,
    pub head: Rva,
}

impl DispatchExtractionCandidate {
    /// Where the handler's own semantic effects end.
    #[must_use]
    pub fn semantic_end(&self) -> Rva {
        self.postlude
            .map_or(self.encoded_load, |postlude| postlude.stack_check_start)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct DispatchExtractionRejection {
    pub encoded_load: Rva,
    pub reason: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct DispatchExtractionReport {
    pub schema_version: u32,
    pub candidates: Vec<DispatchExtractionCandidate>,
    pub rejected_sources: Vec<DispatchExtractionRejection>,
    pub postlude_proof: Option<RegisterDispatchProof>,
}

pub(crate) fn canonical(register: Register) -> Register {
    register.full_register()
}

fn register_bit_mask(register: Register) -> u64 {
    let bit_offset = if matches!(
        register,
        Register::AH | Register::BH | Register::CH | Register::DH
    ) {
        8
    } else {
        0
    };
    let bits = register.size().saturating_mul(8);
    if bits >= u64::BITS as usize {
        u64::MAX
    } else {
        ((1_u64 << bits) - 1) << bit_offset
    }
}

fn architectural_write_mask(register: Register) -> u64 {
    let full = canonical(register);
    if register == full
        || (register.size() == 4 && full.size() == 8 && register.full_register32() == register)
    {
        u64::MAX
    } else {
        register_bit_mask(register)
    }
}

pub(crate) fn register_value_mask(register: Register, width: BitWidth) -> Option<u64> {
    let width_bits = usize::try_from(width.bits()).ok()?;
    if register.size().saturating_mul(8) < width_bits {
        return None;
    }
    let bit_offset = if matches!(
        register,
        Register::AH | Register::BH | Register::CH | Register::DH
    ) {
        8
    } else {
        0
    };
    Some(if width == BitWidth::W64 {
        u64::MAX
    } else {
        ((1_u64 << width.bits()) - 1) << bit_offset
    })
}

pub(crate) fn same_register_value(left: Register, right: Register, width: BitWidth) -> bool {
    canonical(left) == canonical(right)
        && register_value_mask(left, width)
            .is_some_and(|left_mask| register_value_mask(right, width) == Some(left_mask))
}

pub(crate) fn instruction_writes_view(
    factory: &mut InstructionInfoFactory,
    instruction: &Instruction,
    target: Register,
) -> bool {
    let target_mask = register_bit_mask(target);
    factory
        .info(instruction)
        .used_registers()
        .iter()
        .any(|used| {
            matches!(
                used.access(),
                OpAccess::Write
                    | OpAccess::CondWrite
                    | OpAccess::ReadWrite
                    | OpAccess::ReadCondWrite
            ) && canonical(used.register()) == canonical(target)
                && architectural_write_mask(used.register()) & target_mask != 0
        })
}

pub(crate) fn instruction_writes_value(
    factory: &mut InstructionInfoFactory,
    instruction: &Instruction,
    target: Register,
    width: BitWidth,
) -> bool {
    let Some(target_mask) = register_value_mask(target, width) else {
        return false;
    };
    factory
        .info(instruction)
        .used_registers()
        .iter()
        .any(|used| {
            matches!(
                used.access(),
                OpAccess::Write
                    | OpAccess::CondWrite
                    | OpAccess::ReadWrite
                    | OpAccess::ReadCondWrite
            ) && canonical(used.register()) == canonical(target)
                && architectural_write_mask(used.register()) & target_mask != 0
        })
}

pub(crate) fn has_flat_segment_address(instruction: &Instruction) -> bool {
    matches!(
        instruction.segment_prefix(),
        Register::None | Register::ES | Register::CS | Register::SS | Register::DS
    )
}

pub(crate) fn source_load_width(instruction: &Instruction) -> Option<BitWidth> {
    match instruction.memory_size().size() {
        1 => Some(BitWidth::W8),
        2 => Some(BitWidth::W16),
        4 => Some(BitWidth::W32),
        8 => Some(BitWidth::W64),
        _ => None,
    }
}

fn immediate(instruction: &Instruction, operand: u32) -> Option<u64> {
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

pub(crate) fn signed_immediate(instruction: &Instruction, operand: u32) -> Option<i64> {
    match instruction.op_kind(operand) {
        OpKind::Immediate8 => Some(i64::from(instruction.immediate8())),
        OpKind::Immediate8to16 => Some(i64::from(instruction.immediate8to16())),
        OpKind::Immediate8to32 => Some(i64::from(instruction.immediate8to32())),
        OpKind::Immediate8to64 => Some(instruction.immediate8to64()),
        OpKind::Immediate16 => Some(i64::from(instruction.immediate16())),
        OpKind::Immediate32 => Some(i64::from(instruction.immediate32())),
        OpKind::Immediate32to64 => Some(instruction.immediate32to64()),
        OpKind::Immediate64 => i64::try_from(instruction.immediate64()).ok(),
        _ => None,
    }
}

pub(crate) fn key_operation(mnemonic: Mnemonic) -> Option<KeyOperation> {
    match mnemonic {
        Mnemonic::Add => Some(KeyOperation::Add),
        Mnemonic::Sub => Some(KeyOperation::Sub),
        Mnemonic::Xor => Some(KeyOperation::Xor),
        _ => None,
    }
}

pub(crate) fn immediate_transform(instruction: &Instruction, width: BitWidth) -> Option<Transform> {
    if instruction.op0_kind() != OpKind::Register
        || instruction.op0_register().size() != usize::try_from(width.bytes()).ok()?
    {
        return None;
    }
    let (op, operand) = match instruction.mnemonic() {
        Mnemonic::Add => (TransformOp::Add, immediate(instruction, 1)?),
        Mnemonic::Sub => (TransformOp::Sub, immediate(instruction, 1)?),
        Mnemonic::Xor => (TransformOp::Xor, immediate(instruction, 1)?),
        Mnemonic::Rol => (TransformOp::Rol, immediate(instruction, 1)?),
        Mnemonic::Ror => (TransformOp::Ror, immediate(instruction, 1)?),
        Mnemonic::Inc => (TransformOp::Add, 1),
        Mnemonic::Dec => (TransformOp::Sub, 1),
        Mnemonic::Not => (TransformOp::Not, 0),
        Mnemonic::Neg => (TransformOp::Neg, 0),
        Mnemonic::Bswap => (TransformOp::Bswap, 0),
        _ => return None,
    };
    Some(Transform { op, width, operand })
}

pub(crate) fn writes_register(
    factory: &mut InstructionInfoFactory,
    instruction: &Instruction,
    register: Register,
) -> bool {
    factory
        .info(instruction)
        .used_registers()
        .iter()
        .any(|used| {
            canonical(used.register()) == register
                && matches!(
                    used.access(),
                    OpAccess::Write
                        | OpAccess::CondWrite
                        | OpAccess::ReadWrite
                        | OpAccess::ReadCondWrite
                )
        })
}

pub(crate) fn is_active(disposition: NativeInstructionDisposition) -> bool {
    !matches!(
        disposition,
        NativeInstructionDisposition::Dead | NativeInstructionDisposition::StructuralDirectJump
    )
}

pub(crate) fn is_source_load(instruction: &Instruction, vip: Register) -> bool {
    matches!(
        instruction.mnemonic(),
        Mnemonic::Mov | Mnemonic::Movsx | Mnemonic::Movsxd | Mnemonic::Movzx
    ) && instruction.op0_kind() == OpKind::Register
        && instruction.op1_kind() == OpKind::Memory
        && canonical(instruction.memory_base()) == canonical(vip)
        && instruction.memory_base().size() == 8
        && instruction.memory_index() == Register::None
        && instruction.memory_displacement64() == 0
        && has_flat_segment_address(instruction)
}

fn is_sign_extension(instruction: &Instruction, target: Register, width: BitWidth) -> bool {
    matches!(instruction.mnemonic(), Mnemonic::Movsx | Mnemonic::Movsxd)
        && instruction.op0_kind() == OpKind::Register
        && instruction.op1_kind() == OpKind::Register
        && instruction.op0_register().size() == 8
        && u32::try_from(instruction.op1_register().size()).ok() == Some(width.bytes())
        && canonical(instruction.op0_register()) == canonical(target)
        && same_register_value(instruction.op1_register(), target, width)
}

fn handler_adds_target(instruction: &Instruction, handler: Register, target: Register) -> bool {
    instruction.mnemonic() == Mnemonic::Add
        && instruction.op0_kind() == OpKind::Register
        && instruction.op1_kind() == OpKind::Register
        && instruction.op0_register().size() == 8
        && instruction.op1_register().size() == 8
        && canonical(instruction.op0_register()) == canonical(handler)
        && canonical(instruction.op1_register()) == target
}

fn terminal_dispatches_through(trace: &DirectJumpTrace, handler: Register) -> Option<Rva> {
    let instructions: Vec<_> = trace.instructions().collect();
    let terminal = instructions.last()?;
    if terminal.rva != trace.stop.at {
        return None;
    }
    let direct_indirect = terminal.instruction.op0_kind() == OpKind::Register
        && canonical(terminal.instruction.op0_register()) == canonical(handler);
    let push_return = terminal.instruction.mnemonic() == Mnemonic::Ret
        && instructions
            .len()
            .checked_sub(2)
            .and_then(|index| instructions.get(index))
            .is_some_and(|predecessor| {
                predecessor.instruction.mnemonic() == Mnemonic::Push
                    && predecessor.instruction.op0_kind() == OpKind::Register
                    && canonical(predecessor.instruction.op0_register()) == canonical(handler)
            });
    (direct_indirect || push_return).then_some(terminal.rva)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RecoveredKeyUpdate {
    pub operation: KeyOperation,
    pub evidence_rva: Rva,
    pub completion_rva: Rva,
}

fn direct_key_updates(
    instructions: &[&DecodedInstruction],
    key: Register,
    target: Register,
    width: BitWidth,
) -> Vec<RecoveredKeyUpdate> {
    instructions
        .iter()
        .filter_map(|decoded| {
            let instruction = &decoded.instruction;
            if instruction.op0_kind() != OpKind::Register
                || instruction.op1_kind() != OpKind::Register
                || !same_register_value(instruction.op0_register(), key, width)
                || !same_register_value(instruction.op1_register(), target, width)
            {
                return None;
            }
            let destination = instruction.op0_register();
            let preserves_high = width == BitWidth::W64
                || destination.size() < 4
                || canonical(destination).size() == 4;
            preserves_high
                .then(|| key_operation(instruction.mnemonic()))
                .flatten()
                .map(|operation| RecoveredKeyUpdate {
                    operation,
                    evidence_rva: decoded.rva,
                    completion_rva: decoded.rva,
                })
        })
        .collect()
}

fn stack_key_updates(
    instructions: &[&DecodedInstruction],
    key: Register,
    target: Register,
    width: BitWidth,
) -> Vec<RecoveredKeyUpdate> {
    instructions
        .windows(3)
        .filter_map(|window| {
            let push = &window[0].instruction;
            let update = &window[1].instruction;
            let pop = &window[2].instruction;
            let matches = push.mnemonic() == Mnemonic::Push
                && push.op0_kind() == OpKind::Register
                && canonical(push.op0_register()) == canonical(key)
                && update.op0_kind() == OpKind::Memory
                && canonical(update.memory_base()) == Register::RSP
                && update.memory_base().size() == 8
                && update.memory_index() == Register::None
                && update.memory_displacement64() == 0
                && u32::try_from(update.memory_size().size()).ok() == Some(width.bytes())
                && has_flat_segment_address(update)
                && update.op1_kind() == OpKind::Register
                && same_register_value(update.op1_register(), target, width)
                && pop.mnemonic() == Mnemonic::Pop
                && pop.op0_kind() == OpKind::Register
                && pop.op0_register() == canonical(key);
            matches
                .then(|| key_operation(update.mnemonic()))
                .flatten()
                .map(|operation| RecoveredKeyUpdate {
                    operation,
                    evidence_rva: window[1].rva,
                    completion_rva: window[2].rva,
                })
        })
        .collect()
}

pub(crate) fn recover_key_updates(
    instructions: &[&DecodedInstruction],
    key: Register,
    target: Register,
    width: BitWidth,
) -> Vec<RecoveredKeyUpdate> {
    let mut updates = direct_key_updates(instructions, key, target, width);
    updates.extend(stack_key_updates(instructions, key, target, width));
    updates
}

fn static_handler_rebase(instruction: &Instruction, handler: Register) -> Option<Va> {
    (instruction.mnemonic() == Mnemonic::Lea
        && instruction.op0_kind() == OpKind::Register
        && instruction.op0_register() == canonical(handler)
        && instruction.op1_kind() == OpKind::Memory
        && matches!(
            instruction.memory_base(),
            Register::None | Register::RIP | Register::EIP
        )
        && instruction.memory_index() == Register::None
        && has_flat_segment_address(instruction))
    .then(|| Va(instruction.memory_displacement64()))
}

fn is_exact_key_rebase(instruction: &Instruction, key: Register, vip: Register) -> bool {
    instruction.mnemonic() == Mnemonic::Mov
        && instruction.op0_kind() == OpKind::Register
        && instruction.op0_register() == canonical(key)
        && instruction.op1_kind() == OpKind::Register
        && instruction.op1_register() == canonical(vip)
}

fn assigns_full_zero(instruction: &Instruction) -> Option<Register> {
    (instruction.op0_kind() == OpKind::Register
        && instruction.op0_register().size() >= 4
        && ((instruction.mnemonic() == Mnemonic::Mov
            && signed_immediate(instruction, 1) == Some(0))
            || (matches!(instruction.mnemonic(), Mnemonic::Xor | Mnemonic::Sub)
                && instruction.op1_kind() == OpKind::Register
                && instruction.op0_register() == instruction.op1_register())))
    .then(|| canonical(instruction.op0_register()))
}

fn prefix_key_state_is_valid(
    decoded: &[&DecodedInstruction],
    dispositions: &BTreeMap<Rva, NativeInstructionDisposition>,
    roles: HandlerRegisterRoles,
) -> bool {
    let mut factory = InstructionInfoFactory::new();
    let mut zero_registers = BTreeSet::new();
    let mut vip_established = false;
    let mut key_from_vip = false;
    for decoded in decoded {
        if !dispositions
            .get(&decoded.rva)
            .copied()
            .is_some_and(is_active)
        {
            continue;
        }
        let instruction = &decoded.instruction;
        let written_zeroes = zero_registers
            .iter()
            .copied()
            .filter(|register| writes_register(&mut factory, instruction, *register))
            .collect::<Vec<_>>();
        for register in written_zeroes {
            zero_registers.remove(&register);
        }
        if let Some(register) = assigns_full_zero(instruction) {
            zero_registers.insert(register);
        }
        if writes_register(&mut factory, instruction, canonical(roles.vip)) {
            vip_established = instruction.mnemonic() == Mnemonic::Mov
                && instruction.op0_kind() == OpKind::Register
                && instruction.op0_register() == canonical(roles.vip)
                && instruction.op1_kind() == OpKind::Register
                && instruction.op1_register().size() == 8;
        }
        if !writes_register(&mut factory, instruction, canonical(roles.rolling_key)) {
            continue;
        }
        if vip_established && is_exact_key_rebase(instruction, roles.rolling_key, roles.vip) {
            key_from_vip = true;
            continue;
        }
        let preserves = key_from_vip
            && instruction.op0_kind() == OpKind::Register
            && instruction.op0_register() == canonical(roles.rolling_key)
            && matches!(instruction.mnemonic(), Mnemonic::Add | Mnemonic::Sub)
            && (signed_immediate(instruction, 1) == Some(0)
                || (instruction.op1_kind() == OpKind::Register
                    && instruction.op1_register().size() == 8
                    && zero_registers.contains(&canonical(instruction.op1_register()))));
        if !preserves {
            return false;
        }
    }
    true
}

#[allow(clippy::too_many_lines)]
#[allow(clippy::too_many_arguments)]
fn attempt_candidate(
    decoded: &[&DecodedInstruction],
    dispositions: &BTreeMap<Rva, NativeInstructionDisposition>,
    source_index: usize,
    previous_source_index: Option<usize>,
    end_index: usize,
    direction: BytecodeDirection,
    roles: HandlerRegisterRoles,
    terminal_dispatch: Rva,
    terminal_evidence: &str,
) -> Result<DispatchExtractionCandidate, String> {
    let source = decoded[source_index];
    let destination = source.instruction.op0_register();
    let target = canonical(destination);
    let width = source_load_width(&source.instruction)
        .ok_or_else(|| "unsupported decoded width".to_owned())?;
    let mut input_key_operation = None;
    let mut transforms = Vec::new();
    let mut vip_delta = 0_i64;
    let mut sign_extended = width == BitWidth::W64;
    let mut handler_update = None;
    let mut handler_update_index = None;
    let mut input_key_index = None;
    let mut last_transform_index = None;
    let mut factory = InstructionInfoFactory::new();

    let has_prior_source = previous_source_index.is_some();
    let local_prefix_start = previous_source_index.map_or(0, |index| index + 1);
    let mut vip_rebased_before_source = false;
    let mut pre_source_handler_writes = Vec::new();
    let mut pre_source_key_writes = Vec::new();
    for (index, decoded) in decoded.iter().take(source_index).enumerate() {
        let disposition = dispositions
            .get(&decoded.rva)
            .copied()
            .unwrap_or(NativeInstructionDisposition::Dead);
        if is_active(disposition)
            && writes_register(&mut factory, &decoded.instruction, canonical(roles.handler))
        {
            pre_source_handler_writes.push(decoded);
        }
        if is_active(disposition)
            && writes_register(
                &mut factory,
                &decoded.instruction,
                canonical(roles.rolling_key),
            )
        {
            pre_source_key_writes.push(decoded);
        }
        if is_active(disposition)
            && writes_register(&mut factory, &decoded.instruction, canonical(roles.vip))
        {
            let instruction = &decoded.instruction;
            let exact_full_width_rebase = !vip_rebased_before_source
                && !has_prior_source
                && instruction.mnemonic() == Mnemonic::Mov
                && instruction.op0_kind() == OpKind::Register
                && instruction.op1_kind() == OpKind::Register
                && instruction.op0_register() == canonical(roles.vip)
                && instruction.op1_register().size() == 8;
            if exact_full_width_rebase {
                vip_rebased_before_source = true;
                continue;
            }
            if direction == BytecodeDirection::Descending && index >= local_prefix_start {
                if instruction.op0_kind() != OpKind::Register
                    || instruction.op0_register().size() != 8
                    || !matches!(instruction.mnemonic(), Mnemonic::Add | Mnemonic::Sub)
                {
                    return Err(format!(
                        "descending VIP has an unmodelled pre-source write at {}",
                        decoded.rva
                    ));
                }
                let delta = signed_immediate(instruction, 1)
                    .ok_or_else(|| "descending VIP adjustment is not immediate".to_owned())?;
                vip_delta = match instruction.mnemonic() {
                    Mnemonic::Add => vip_delta.checked_add(delta),
                    Mnemonic::Sub => vip_delta.checked_sub(delta),
                    _ => unreachable!(),
                }
                .ok_or_else(|| "descending VIP adjustment overflows".to_owned())?;
                continue;
            }
            if !has_prior_source {
                return Err(format!(
                    "VIP is modified before the encoded load at {}",
                    decoded.rva
                ));
            }
        }
    }
    let (handler_rebase, handler_rebase_rva) = match pre_source_handler_writes.as_slice() {
        [] => (None, None),
        [decoded] => {
            let Some(base) = static_handler_rebase(&decoded.instruction, roles.handler) else {
                return Err("handler register is modified before the encoded load".to_owned());
            };
            (Some(base), Some(decoded.rva))
        }
        _ => {
            return Err("handler register is modified before the encoded load".to_owned());
        }
    };
    if handler_rebase.is_some() && has_prior_source {
        return Err("handler register is modified before the encoded load".to_owned());
    }
    let key_prefix_is_valid = has_prior_source
        || pre_source_key_writes.is_empty()
        || (vip_rebased_before_source
            && prefix_key_state_is_valid(&decoded[..source_index], dispositions, roles));
    if !key_prefix_is_valid {
        return Err("rolling-key register is modified before the encoded load".to_owned());
    }

    for (index, decoded) in decoded
        .iter()
        .enumerate()
        .take(end_index)
        .skip(source_index + 1)
    {
        let disposition = dispositions
            .get(&decoded.rva)
            .copied()
            .unwrap_or(NativeInstructionDisposition::Dead);
        if !is_active(disposition) {
            continue;
        }
        let instruction = &decoded.instruction;
        if handler_adds_target(instruction, roles.handler, target) {
            if handler_update.is_some() || !sign_extended {
                return Err("handler update lacks a final exact sign extension".to_owned());
            }
            handler_update = Some(decoded.rva);
            handler_update_index = Some(index);
            continue;
        }
        if writes_register(&mut factory, instruction, canonical(roles.handler)) {
            return Err(format!(
                "handler register has an unmodelled write at {}",
                decoded.rva
            ));
        }
        if direction == BytecodeDirection::Ascending
            && instruction.op0_kind() == OpKind::Register
            && canonical(instruction.op0_register()) == canonical(roles.vip)
            && matches!(instruction.mnemonic(), Mnemonic::Add | Mnemonic::Sub)
        {
            if instruction.op0_register().size() != 8 {
                return Err("VIP adjustment is not a full-width write".to_owned());
            }
            let delta = signed_immediate(instruction, 1)
                .ok_or_else(|| "VIP adjustment is not an immediate".to_owned())?;
            vip_delta = match instruction.mnemonic() {
                Mnemonic::Add => vip_delta.checked_add(delta),
                Mnemonic::Sub => vip_delta.checked_sub(delta),
                _ => unreachable!(),
            }
            .ok_or_else(|| "VIP adjustment overflows".to_owned())?;
            continue;
        }
        if writes_register(&mut factory, instruction, canonical(roles.vip)) {
            return Err(format!("VIP has an unmodelled write at {}", decoded.rva));
        }
        if handler_update.is_some() && writes_register(&mut factory, instruction, target) {
            continue;
        }
        if is_sign_extension(instruction, destination, width) {
            sign_extended = true;
            continue;
        }
        if width != BitWidth::W64
            && sign_extended
            && handler_update.is_none()
            && writes_register(&mut factory, instruction, target)
        {
            return Err(format!(
                "sign-extended displacement is modified before handler update at {}",
                decoded.rva
            ));
        }
        if instruction_writes_value(&mut factory, instruction, destination, width) {
            sign_extended = width == BitWidth::W64;
            let mixes_key = instruction.op0_kind() == OpKind::Register
                && same_register_value(instruction.op0_register(), destination, width)
                && instruction.op1_kind() == OpKind::Register
                && same_register_value(instruction.op1_register(), roles.rolling_key, width);
            if mixes_key {
                if input_key_operation.is_some() || !transforms.is_empty() {
                    return Err("ambiguous or late input-key operation".to_owned());
                }
                input_key_operation = key_operation(instruction.mnemonic());
                if input_key_operation.is_none() {
                    return Err("unsupported input-key operation".to_owned());
                }
                input_key_index = Some(index);
                continue;
            }
            let transform = immediate_transform(instruction, width)
                .ok_or_else(|| format!("unsupported decoded-register write at {}", decoded.rva))?;
            transforms.push(transform);
            last_transform_index = Some(index);
        }
    }

    let handler_update =
        handler_update.ok_or_else(|| "decoded value does not update the handler".to_owned())?;
    let handler_update_index = handler_update_index
        .ok_or_else(|| "handler update position was not retained".to_owned())?;
    if width != BitWidth::W64 && !sign_extended {
        return Err("relative displacement is not sign-extended".to_owned());
    }
    let active: Vec<_> = decoded[source_index..end_index]
        .iter()
        .copied()
        .filter(|decoded| {
            dispositions
                .get(&decoded.rva)
                .copied()
                .is_some_and(is_active)
        })
        .collect();
    let updates = recover_key_updates(&active, roles.rolling_key, target, width);
    let [update] = updates.as_slice() else {
        return Err(format!(
            "expected one rolling-key update, recovered {}",
            updates.len()
        ));
    };
    let update_key_operation = update.operation;
    let key_update_rva = update.evidence_rva;
    let allowed_key_writes = BTreeSet::from([update.completion_rva]);
    if active.iter().any(|decoded| {
        writes_register(
            &mut factory,
            &decoded.instruction,
            canonical(roles.rolling_key),
        ) && !allowed_key_writes.contains(&decoded.rva)
    }) {
        return Err("rolling-key register has an unmodelled write".to_owned());
    }
    let key_update_index = decoded
        .iter()
        .position(|decoded| decoded.rva == key_update_rva)
        .ok_or_else(|| "rolling-key update position was not retained".to_owned())?;
    let key_completion_index = decoded
        .iter()
        .position(|decoded| decoded.rva == update.completion_rva)
        .ok_or_else(|| "rolling-key completion position was not retained".to_owned())?;
    if input_key_index.is_none_or(|mix| key_update_index <= mix)
        || last_transform_index.is_some_and(|last| key_update_index <= last)
        || key_completion_index >= handler_update_index
    {
        return Err("rolling-key update is out of semantic order".to_owned());
    }
    let input_key_operation =
        input_key_operation.ok_or_else(|| "input-key operation was not recovered".to_owned())?;
    let expected_vip_delta = match direction {
        BytecodeDirection::Ascending => i64::from(width.bytes()),
        BytecodeDirection::Descending => -i64::from(width.bytes()),
        BytecodeDirection::StateDependent | BytecodeDirection::Unknown => {
            return Err("bytecode direction is not concrete".to_owned());
        }
    };
    if vip_delta != expected_vip_delta {
        return Err(format!(
            "VIP delta {vip_delta} does not match encoded width {}",
            width.bytes()
        ));
    }

    let mut evidence = vec![
        format!("encoded {:?} load from the recovered VIP register", width),
        format!("input is mixed with the recovered key using {input_key_operation:?}"),
        format!("{} reversible post-key transforms", transforms.len()),
        format!("VIP advances by {vip_delta} bytes"),
        format!("rolling key updates using {update_key_operation:?} at {key_update_rva}"),
        "decoded displacement is sign-extended before handler update".to_owned(),
        "handler register receives the decoded relative displacement".to_owned(),
        terminal_evidence.to_owned(),
    ];
    if transforms.is_empty() {
        evidence.push("no post-key transform was present".to_owned());
    }
    if vip_rebased_before_source {
        evidence.push(
            "handler establishes a full-width local VIP before dispatch; codec is non-linear"
                .to_owned(),
        );
    }
    if let Some(base) = handler_rebase {
        evidence.push(format!(
            "handler establishes static dispatch anchor {base} before the encoded load"
        ));
    }
    Ok(DispatchExtractionCandidate {
        encoded_load: source.rva,
        decoded_register: format!("{destination:?}"),
        vip_rebased_before_source,
        handler_rebase,
        handler_rebase_rva,
        table_register: None,
        vip_delta,
        handler_update,
        terminal_dispatch,
        codec: RelativeDispatchCodec {
            displacement: crate::KeyedTransformCodec {
                width,
                input_key_operation: Some(input_key_operation),
                input_key_on_left: false,
                transforms: TransformChain { transforms },
                update_key_operation: Some(update_key_operation),
            },
            table: None,
        },
        confidence: 100,
        evidence,
        postlude: None,
    })
}

/// Recognizes a `movsxd handler, [table + index*stride]` table-entry load, the
/// sign-extended relative dispatch of a table-indexed VM. Returns the table
/// base register, the byte stride, and the signed entry width.
fn table_entry_load(
    instruction: &Instruction,
    handler: Register,
    index: Register,
) -> Option<(Register, u32, BitWidth)> {
    if !matches!(instruction.mnemonic(), Mnemonic::Movsxd | Mnemonic::Movsx)
        || instruction.op0_kind() != OpKind::Register
        || canonical(instruction.op0_register()) != canonical(handler)
        || instruction.op0_register().size() != 8
        || instruction.op1_kind() != OpKind::Memory
        || instruction.memory_displacement64() != 0
        || !has_flat_segment_address(instruction)
    {
        return None;
    }
    let table = instruction.memory_base();
    if table == Register::None || table.size() != 8 || canonical(table) != table {
        return None;
    }
    if canonical(instruction.memory_index()) != canonical(index)
        || instruction.memory_index().size() != 8
    {
        return None;
    }
    let entry_width = source_load_width(instruction)?;
    let stride = instruction.memory_index_scale();
    // A dense table of signed entries sign-extended to the 64-bit handler delta.
    (stride == entry_width.bytes()).then_some((table, stride, entry_width))
}

fn handler_adds_table(instruction: &Instruction, handler: Register, table: Register) -> bool {
    instruction.mnemonic() == Mnemonic::Add
        && instruction.op0_kind() == OpKind::Register
        && instruction.op1_kind() == OpKind::Register
        && instruction.op0_register().size() == 8
        && instruction.op1_register().size() == 8
        && canonical(instruction.op0_register()) == canonical(handler)
        && canonical(instruction.op1_register()) == canonical(table)
}

/// Recovers a table-indexed keyless dispatch (VMProtect 3.x "Compact").
///
/// The opcode is read raw from `[vip]` (no rolling key) and used to index a
/// static table of signed relative entries:
/// `handler = table_base + sext(table[opcode])`. The table base is held in a
/// register established by a `lea table, [abs]` static anchor when present.
#[allow(clippy::too_many_arguments)]
fn attempt_table_indexed_candidate(
    decoded: &[&DecodedInstruction],
    dispositions: &BTreeMap<Rva, NativeInstructionDisposition>,
    source_index: usize,
    end_index: usize,
    direction: BytecodeDirection,
    roles: HandlerRegisterRoles,
    terminal_dispatch: Rva,
    terminal_evidence: &str,
) -> Result<DispatchExtractionCandidate, String> {
    let source = decoded[source_index];
    // The opcode must be zero-extended so the index register holds exactly the
    // unsigned table index.
    if source.instruction.mnemonic() != Mnemonic::Movzx {
        return Err("table-indexed opcode load is not zero-extended".to_owned());
    }
    let opcode_width = source_load_width(&source.instruction)
        .ok_or_else(|| "unsupported opcode width".to_owned())?;
    let index = canonical(source.instruction.op0_register());
    if index == canonical(roles.vip)
        || index == canonical(roles.handler)
        || index == canonical(roles.rolling_key)
    {
        return Err("table-indexed opcode register aliases a dispatch role".to_owned());
    }
    let mut factory = InstructionInfoFactory::new();
    let mut vip_delta = 0_i64;
    let mut table_load: Option<(Register, u32, BitWidth, usize)> = None;
    let mut handler_add_index = None;
    for (position, decoded) in decoded
        .iter()
        .enumerate()
        .take(end_index)
        .skip(source_index + 1)
    {
        if !dispositions
            .get(&decoded.rva)
            .copied()
            .is_some_and(is_active)
        {
            continue;
        }
        let instruction = &decoded.instruction;
        if instruction.op0_kind() == OpKind::Register
            && canonical(instruction.op0_register()) == canonical(roles.vip)
            && matches!(instruction.mnemonic(), Mnemonic::Add | Mnemonic::Sub)
        {
            if instruction.op0_register().size() != 8 {
                return Err("VIP adjustment is not a full-width write".to_owned());
            }
            let delta = signed_immediate(instruction, 1)
                .ok_or_else(|| "VIP adjustment is not an immediate".to_owned())?;
            vip_delta = match instruction.mnemonic() {
                Mnemonic::Add => vip_delta.checked_add(delta),
                Mnemonic::Sub => vip_delta.checked_sub(delta),
                _ => unreachable!(),
            }
            .ok_or_else(|| "VIP adjustment overflows".to_owned())?;
            continue;
        }
        if writes_register(&mut factory, instruction, canonical(roles.vip)) {
            return Err(format!("VIP has an unmodelled write at {}", decoded.rva));
        }
        if let Some((table, stride, entry_width)) =
            table_entry_load(instruction, roles.handler, index)
        {
            if table_load.is_some() {
                return Err("duplicate table-indexed dispatch load".to_owned());
            }
            table_load = Some((table, stride, entry_width, position));
            continue;
        }
        if table_load.is_none() && writes_register(&mut factory, instruction, index) {
            return Err(format!(
                "table index register is clobbered before the dispatch load at {}",
                decoded.rva
            ));
        }
        if let Some((table, _, _, _)) = table_load {
            if handler_adds_table(instruction, roles.handler, table) {
                if handler_add_index.is_some() {
                    return Err("duplicate handler base addition".to_owned());
                }
                handler_add_index = Some(position);
                continue;
            }
        }
        if writes_register(&mut factory, instruction, canonical(roles.handler)) {
            return Err(format!(
                "handler register has an unmodelled write at {}",
                decoded.rva
            ));
        }
    }
    let (table, stride, entry_width, table_load_index) =
        table_load.ok_or_else(|| "no table-indexed dispatch load".to_owned())?;
    let handler_add_index = handler_add_index
        .ok_or_else(|| "table-indexed handler base addition is absent".to_owned())?;
    if table_load_index >= handler_add_index {
        return Err("table load does not precede the handler base addition".to_owned());
    }
    let expected_vip_delta = match direction {
        BytecodeDirection::Ascending => i64::from(opcode_width.bytes()),
        BytecodeDirection::Descending => -i64::from(opcode_width.bytes()),
        BytecodeDirection::StateDependent | BytecodeDirection::Unknown => {
            return Err("bytecode direction is not concrete".to_owned());
        }
    };
    if vip_delta != expected_vip_delta {
        return Err(format!(
            "VIP delta {vip_delta} does not match opcode width {}",
            opcode_width.bytes()
        ));
    }
    // Recover the static table anchor from a pre-source `lea table, [abs]` that
    // is not later clobbered.
    let mut table_anchor: Option<(Va, Rva)> = None;
    for decoded in decoded.iter().take(source_index) {
        if !dispositions
            .get(&decoded.rva)
            .copied()
            .is_some_and(is_active)
        {
            continue;
        }
        if let Some(base) = static_handler_rebase(&decoded.instruction, table) {
            table_anchor = Some((base, decoded.rva));
        } else if writes_register(&mut factory, &decoded.instruction, table) {
            table_anchor = None;
        }
    }
    let handler_rebase = table_anchor.map(|(base, _)| base);
    let handler_rebase_rva = table_anchor.map(|(_, rva)| rva);
    let mut evidence = vec![
        format!("table-indexed opcode {opcode_width:?} load from the recovered VIP register"),
        format!("opcode indexes a dense {entry_width:?} signed dispatch table at stride {stride}"),
        "handler is the table base plus the sign-extended table entry".to_owned(),
        format!("VIP advances by {vip_delta} bytes"),
        "no rolling-key transform participates in dispatch".to_owned(),
        terminal_evidence.to_owned(),
    ];
    if let Some((base, _)) = table_anchor {
        evidence.push(format!("table base established by static anchor {base}"));
    }
    Ok(DispatchExtractionCandidate {
        encoded_load: source.rva,
        decoded_register: format!("{:?}", source.instruction.op0_register()),
        vip_rebased_before_source: false,
        handler_rebase,
        handler_rebase_rva,
        table_register: Some(table),
        vip_delta,
        handler_update: decoded[handler_add_index].rva,
        terminal_dispatch,
        codec: RelativeDispatchCodec {
            displacement: crate::KeyedTransformCodec {
                width: entry_width,
                input_key_operation: None,
                input_key_on_left: false,
                transforms: TransformChain {
                    transforms: Vec::new(),
                },
                update_key_operation: None,
            },
            table: Some(crate::TableIndexedDispatch {
                opcode_width,
                entry_width,
                stride,
                table_base: handler_rebase,
            }),
        },
        confidence: 100,
        evidence,
        postlude: None,
    })
}

/// Extracts keyed relative-dispatch codecs from behavior retained by a native
/// handler slice. Every VIP-relative load is considered independently and
/// rejected sources keep a diagnostic instead of being guessed into a codec.
#[must_use]
pub fn extract_relative_dispatch_codecs(
    trace: &DirectJumpTrace,
    slice: &NativeBackwardSlice,
    roles: HandlerRegisterRoles,
) -> DispatchExtractionReport {
    extract_relative_dispatch_codecs_with_terminal(
        trace,
        slice,
        roles,
        BytecodeDirection::Ascending,
        terminal_dispatches_through(trace, roles.handler),
        "trace terminates in a dispatch through the handler register",
        None,
    )
}

fn extract_relative_dispatch_codecs_with_terminal(
    trace: &DirectJumpTrace,
    slice: &NativeBackwardSlice,
    roles: HandlerRegisterRoles,
    direction: BytecodeDirection,
    terminal_dispatch: Option<Rva>,
    terminal_evidence: &str,
    postlude_proof: Option<RegisterDispatchProof>,
) -> DispatchExtractionReport {
    let dispositions: BTreeMap<_, _> = slice
        .instructions
        .iter()
        .map(|instruction| (instruction.rva, instruction.disposition))
        .collect();
    let decoded: Vec<_> = trace.instructions().collect();
    let source_indices: Vec<_> = decoded
        .iter()
        .enumerate()
        .filter(|(_, decoded)| is_source_load(&decoded.instruction, roles.vip))
        .map(|(index, _)| index)
        .collect();
    let mut candidates = Vec::new();
    let mut rejected_sources = Vec::new();

    for (position, index) in source_indices.iter().copied().enumerate() {
        let decoded_instruction = decoded[index];
        let next_source_index = source_indices
            .get(position + 1)
            .copied()
            .unwrap_or(decoded.len());
        let end_index = if direction == BytecodeDirection::Descending {
            decoded[index + 1..next_source_index]
                .iter()
                .position(|decoded| {
                    dispositions
                        .get(&decoded.rva)
                        .copied()
                        .is_some_and(is_active)
                        && writes_register(
                            &mut InstructionInfoFactory::new(),
                            &decoded.instruction,
                            canonical(roles.vip),
                        )
                })
                .map_or(next_source_index, |offset| index + 1 + offset)
        } else {
            next_source_index
        };
        let result = terminal_dispatch.ok_or_else(|| {
            "trace does not terminate through the recovered handler register".to_owned()
        });
        match result.and_then(|terminal| {
            // Table-indexed keyless dispatch and keyed threaded dispatch are
            // structurally disjoint; try the table form first and fall back to
            // the threaded recovery, keeping both diagnostics on failure.
            attempt_table_indexed_candidate(
                &decoded,
                &dispositions,
                index,
                end_index,
                direction,
                roles,
                terminal,
                terminal_evidence,
            )
            .or_else(|table_reason| {
                attempt_candidate(
                    &decoded,
                    &dispositions,
                    index,
                    position
                        .checked_sub(1)
                        .and_then(|previous| source_indices.get(previous).copied()),
                    end_index,
                    direction,
                    roles,
                    terminal,
                    terminal_evidence,
                )
                .map_err(|reason| format!("{reason}; table-indexed: {table_reason}"))
            })
        }) {
            Ok(candidate) => candidates.push(candidate),
            Err(reason) => rejected_sources.push(DispatchExtractionRejection {
                encoded_load: decoded_instruction.rva,
                reason,
            }),
        }
    }
    DispatchExtractionReport {
        schema_version: 2,
        candidates,
        rejected_sources,
        postlude_proof,
    }
}

/// Extracts relative-dispatch codecs and, when the linear trace ends at a
/// conditional, proves that every direct-control-flow successor reaches a
/// dispatch through the recovered handler register without clobbering it.
///
/// # Errors
///
/// Propagates invalid postlude entry or decoding errors from proof setup. Later
/// proof failures remain structured in the report and do not create codecs.
pub fn extract_relative_dispatch_codecs_with_postlude(
    pe: &PeImage<'_>,
    trace: &DirectJumpTrace,
    slice: &NativeBackwardSlice,
    roles: HandlerRegisterRoles,
    proof_limits: RegisterDispatchProofLimits,
) -> Result<DispatchExtractionReport, DisasmError> {
    extract_relative_dispatch_codecs_with_postlude_direction(
        pe,
        trace,
        slice,
        roles,
        BytecodeDirection::Ascending,
        proof_limits,
    )
}

/// Direction-aware relative-dispatch recovery for a proven VM profile.
///
/// # Errors
///
/// Propagates invalid postlude entry or native decoding errors.
pub fn extract_relative_dispatch_codecs_with_postlude_direction(
    pe: &PeImage<'_>,
    trace: &DirectJumpTrace,
    slice: &NativeBackwardSlice,
    roles: HandlerRegisterRoles,
    direction: BytecodeDirection,
    proof_limits: RegisterDispatchProofLimits,
) -> Result<DispatchExtractionReport, DisasmError> {
    extract_relative_dispatch_codecs_with_stack_check(
        pe,
        trace,
        slice,
        roles,
        direction,
        proof_limits,
        None,
    )
}

/// As [`extract_relative_dispatch_codecs_with_postlude_direction`], and when
/// `stack_check` is given also recognizes a handler that ends in the shared
/// stack-check postlude whose `ja` targets a proven table-indexed dispatch head.
///
/// # Errors
///
/// Propagates invalid postlude entry or native decoding errors.
pub fn extract_relative_dispatch_codecs_with_stack_check(
    pe: &PeImage<'_>,
    trace: &DirectJumpTrace,
    slice: &NativeBackwardSlice,
    roles: HandlerRegisterRoles,
    direction: BytecodeDirection,
    proof_limits: RegisterDispatchProofLimits,
    stack_check: Option<StackCheckRoles>,
) -> Result<DispatchExtractionReport, DisasmError> {
    if let Some(terminal) = terminal_dispatches_through(trace, roles.handler) {
        return Ok(extract_relative_dispatch_codecs_with_terminal(
            trace,
            slice,
            roles,
            direction,
            Some(terminal),
            "trace terminates in a dispatch through the handler register",
            None,
        ));
    }
    if trace.stop.kind != DirectJumpTraceStopKind::ConditionalBranch {
        return Ok(extract_relative_dispatch_codecs_with_terminal(
            trace,
            slice,
            roles,
            direction,
            None,
            "terminal dispatch was not proven",
            None,
        ));
    }
    let entry_points: Vec<_> = [trace.stop.direct_target, trace.stop.fallthrough]
        .into_iter()
        .flatten()
        .collect();
    let proof = prove_register_dispatch_postlude(pe, &entry_points, roles.handler, proof_limits)?;
    let terminal = proof.all_paths_dispatch.then(|| {
        proof
            .dispatch_sites
            .first()
            .or_else(|| proof.push_return_sites.first())
            .copied()
    });
    let mut stack_check_rejection = None;
    if terminal.flatten().is_none()
        && let Some(check) = stack_check
    {
        match recognize_postlude_dispatch(pe, trace, roles, check, direction) {
            Ok(Some(candidate)) => {
                return Ok(DispatchExtractionReport {
                    schema_version: 2,
                    candidates: vec![candidate],
                    rejected_sources: Vec::new(),
                    postlude_proof: Some(proof),
                });
            }
            Ok(None) => {}
            Err(reason) => {
                stack_check_rejection = Some(DispatchExtractionRejection {
                    encoded_load: trace.stop.at,
                    reason: format!("stack-check postlude: {reason}"),
                });
            }
        }
    }
    let mut report = extract_relative_dispatch_codecs_with_terminal(
        trace,
        slice,
        roles,
        direction,
        terminal.flatten(),
        "an acyclic all-path postlude proof reaches the handler-register dispatch",
        Some(proof),
    );
    report.rejected_sources.extend(stack_check_rejection);
    Ok(report)
}

const MAX_ANCHOR_LOOKBACK: u32 = 48;
const MAX_POSTLUDE_INSTRUCTIONS: usize = 512;

fn is_flag_junk(instruction: &Instruction) -> bool {
    matches!(
        instruction.mnemonic(),
        Mnemonic::Stc | Mnemonic::Clc | Mnemonic::Cmc
    )
}

fn is_direct_jump(instruction: &Instruction) -> bool {
    instruction.mnemonic() == Mnemonic::Jmp && instruction.op0_kind() == OpKind::NearBranch64
}

/// A dead-flag producer such as a junk `cmp`/`test`: it writes status flags but
/// no register or memory, so it cannot disturb the stack-check operands and its
/// flags are overwritten by the real `cmp vsp,tmp` before the `ja`.
fn writes_only_flags(factory: &mut InstructionInfoFactory, instruction: &Instruction) -> bool {
    let info = factory.info(instruction);
    info.used_memory().is_empty()
        && info.used_registers().iter().all(|used| {
            matches!(
                used.access(),
                OpAccess::Read | OpAccess::CondRead | OpAccess::NoMemAccess
            )
        })
}

struct StackCheckShape {
    start: Rva,
    branch: Rva,
    head: Rva,
    fallthrough: Rva,
}

/// Matches the exact tail `lea tmp,[ctx+disp]; cmp vsp,tmp; ja head`. Only
/// direct jumps may separate `cmp` from `ja` (they leave flags intact) and only
/// direct jumps or carry-flag junk may precede `cmp` (it rewrites every flag).
fn match_stack_check(
    trace: &DirectJumpTrace,
    roles: HandlerRegisterRoles,
    check: StackCheckRoles,
) -> Option<StackCheckShape> {
    if trace.stop.kind != DirectJumpTraceStopKind::ConditionalBranch {
        return None;
    }
    let instructions: Vec<_> = trace.instructions().collect();
    let branch = instructions.last()?;
    if branch.rva != trace.stop.at || branch.instruction.mnemonic() != Mnemonic::Ja {
        return None;
    }
    let head = trace.stop.direct_target?;
    let fallthrough = trace.stop.fallthrough?;
    let mut index = instructions.len() - 1;
    let cmp_index = loop {
        index = index.checked_sub(1)?;
        if !is_direct_jump(&instructions[index].instruction) {
            break index;
        }
    };
    let cmp = &instructions[cmp_index].instruction;
    if cmp.mnemonic() != Mnemonic::Cmp
        || cmp.op0_kind() != OpKind::Register
        || cmp.op1_kind() != OpKind::Register
        || cmp.op0_register() != canonical(check.vsp)
        || cmp.op0_register().size() != 8
        || cmp.op1_register().size() != 8
    {
        return None;
    }
    // `tmp` is scratch for the guard comparison. It must not alias VIP or VSP:
    // both are live VM state that has to survive the postlude into the dispatch
    // head. It MAY alias the handler register: the handler is dead across the
    // postlude because the proven table-indexed head re-establishes it (and the
    // fall-through relocation path is likewise allowed to clobber it); a
    // protector that reuses the handler register as the guard scratch is sound.
    let tmp = cmp.op1_register();
    if tmp != canonical(tmp)
        || [check.vsp, roles.vip]
            .into_iter()
            .any(|register| canonical(register) == tmp)
    {
        return None;
    }
    index = cmp_index;
    let mut factory = InstructionInfoFactory::new();
    let lea_index = loop {
        index = index.checked_sub(1)?;
        let instruction = &instructions[index].instruction;
        if !(is_direct_jump(instruction)
            || is_flag_junk(instruction)
            || writes_only_flags(&mut factory, instruction))
        {
            break index;
        }
    };
    let lea = &instructions[lea_index].instruction;
    let exact_lea = lea.mnemonic() == Mnemonic::Lea
        && lea.op0_kind() == OpKind::Register
        && lea.op0_register() == tmp
        && lea.op1_kind() == OpKind::Memory
        && lea.memory_base() == canonical(check.context_base)
        && lea.memory_base().size() == 8
        && lea.memory_index() == Register::None
        && has_flat_segment_address(lea);
    exact_lea.then_some(StackCheckShape {
        start: instructions[lea_index].rva,
        branch: branch.rva,
        head,
        fallthrough,
    })
}

/// Straight-line decodes that land exactly on `head` and end in the unique
/// static `lea table,[abs]` anchor. Ambiguous or absent anchors are rejected.
fn table_anchor_before(pe: &PeImage<'_>, head: Rva, table: Register) -> Result<(Va, Rva), String> {
    let mut factory = InstructionInfoFactory::new();
    let mut found: Option<(Va, Rva)> = None;
    for back in 1..=MAX_ANCHOR_LOOKBACK {
        let Some(start) = head.0.checked_sub(back) else {
            break;
        };
        let mut cursor = Rva(start);
        let mut anchor = None;
        let mut synchronized = false;
        for _ in 0..MAX_ANCHOR_LOOKBACK {
            if cursor == head {
                synchronized = true;
                break;
            }
            let Ok(decoded) = decode_one(pe, cursor) else {
                break;
            };
            if decoded.instruction.flow_control() != FlowControl::Next {
                break;
            }
            if let Some(base) = static_handler_rebase(&decoded.instruction, table) {
                anchor = Some((base, decoded.rva));
            } else if writes_register(&mut factory, &decoded.instruction, table) {
                anchor = None;
            }
            let Some(next) = decoded.next_rva() else {
                break;
            };
            cursor = next;
        }
        if let (true, Some(anchor)) = (synchronized, anchor) {
            match found {
                None => found = Some(anchor),
                Some(previous) if previous != anchor => {
                    return Err("table anchor before the dispatch head is ambiguous".to_owned());
                }
                Some(_) => {}
            }
        }
    }
    found.ok_or_else(|| "no static table anchor falls through into the dispatch head".to_owned())
}

/// Every direct-flow path from `from` must reach `head` without calls, returns,
/// or indirect transfers, and without writing any `protected` register.
fn region_reaches_head(
    pe: &PeImage<'_>,
    from: Rva,
    head: Rva,
    protected: &[Register],
) -> Result<(), String> {
    let mut factory = InstructionInfoFactory::new();
    let mut visited = BTreeSet::new();
    let mut queue = vec![from];
    while let Some(rva) = queue.pop() {
        if rva == head || !visited.insert(rva) {
            continue;
        }
        if visited.len() > MAX_POSTLUDE_INSTRUCTIONS {
            return Err("fall-through postlude exceeds the instruction limit".to_owned());
        }
        if !is_executable_rva(pe, rva) {
            return Err(format!(
                "fall-through postlude leaves executable code at {rva}"
            ));
        }
        let decoded = decode_one(pe, rva).map_err(|error| error.to_string())?;
        if protected
            .iter()
            .any(|register| writes_register(&mut factory, &decoded.instruction, *register))
        {
            return Err(format!(
                "fall-through postlude writes a protected VM register at {rva}"
            ));
        }
        match decoded.instruction.flow_control() {
            FlowControl::Next => queue.extend(decoded.next_rva()),
            FlowControl::UnconditionalBranch => {
                queue.push(
                    decoded
                        .direct_target_rva(pe)
                        .ok_or_else(|| format!("unresolved jump at {rva}"))?,
                );
            }
            FlowControl::ConditionalBranch => {
                queue.push(
                    decoded
                        .direct_target_rva(pe)
                        .ok_or_else(|| format!("unresolved branch at {rva}"))?,
                );
                queue.extend(decoded.next_rva());
            }
            other => {
                return Err(format!(
                    "fall-through postlude reaches {other:?} at {rva} instead of the dispatch head"
                ));
            }
        }
    }
    Ok(())
}

/// Recognizes a handler whose trace ends in the exact stack-check postlude
/// `lea tmp,[ctx+disp]; cmp vsp,tmp; ja head`, where `head` is a proven keyless
/// table-indexed dispatch and the fall-through path also reaches `head`.
///
/// `Ok(None)` means the trace does not have this shape; `Err` means it does but
/// a soundness condition failed.
fn recognize_postlude_dispatch(
    pe: &PeImage<'_>,
    trace: &DirectJumpTrace,
    roles: HandlerRegisterRoles,
    check: StackCheckRoles,
    direction: BytecodeDirection,
) -> Result<Option<DispatchExtractionCandidate>, String> {
    let Some(shape) = match_stack_check(trace, roles, check) else {
        return Ok(None);
    };
    let head_trace = trace_direct_jump_fragments(pe, shape.head, DirectJumpTraceLimits::default())
        .map_err(|error| error.to_string())?;
    let terminal = terminal_dispatches_through(&head_trace, roles.handler).ok_or_else(|| {
        "branch target does not terminate in a handler-register dispatch".to_owned()
    })?;
    let head_slice = backward_slice_handler_trace(
        &head_trace,
        &HandlerSliceConfig {
            live_out_registers: vec![roles.vip, roles.handler, roles.rolling_key],
            live_out_rflags: 0,
        },
    );
    let report = extract_relative_dispatch_codecs_with_terminal(
        &head_trace,
        &head_slice,
        roles,
        direction,
        Some(terminal),
        "stack-check branch target is a proven table-indexed dispatch head",
        None,
    );
    let [head_candidate] = report.candidates.as_slice() else {
        return Err(format!(
            "branch target is not a unique table-indexed dispatch head ({} codecs, {} rejected)",
            report.candidates.len(),
            report.rejected_sources.len()
        ));
    };
    let mut candidate = head_candidate.clone();
    let Some(entry_table) = candidate.codec.table else {
        return Err("branch target dispatch is not table-indexed".to_owned());
    };
    if !report.rejected_sources.is_empty()
        || candidate.encoded_load != shape.head
        || candidate.handler_rebase.is_some()
    {
        return Err("branch target is not a bare dispatch head".to_owned());
    }
    let index = head_trace
        .instructions()
        .find(|decoded| decoded.rva == candidate.encoded_load)
        .map(|decoded| canonical(decoded.instruction.op0_register()))
        .ok_or_else(|| "dispatch head load was not retained".to_owned())?;
    let table = head_trace
        .instructions()
        .find_map(|decoded| {
            table_entry_load(&decoded.instruction, roles.handler, index).map(|(table, _, _)| table)
        })
        .ok_or_else(|| "dispatch head table load was not retained".to_owned())?;
    let (anchor, anchor_rva) = table_anchor_before(pe, shape.head, table)?;

    let mut factory = InstructionInfoFactory::new();
    if trace
        .instructions()
        .any(|decoded| writes_register(&mut factory, &decoded.instruction, table))
    {
        return Err("handler writes the dispatch table register".to_owned());
    }
    region_reaches_head(
        pe,
        shape.fallthrough,
        shape.head,
        &[roles.vip, check.vsp, table],
    )?;

    candidate.handler_rebase = Some(anchor);
    candidate.handler_rebase_rva = Some(anchor_rva);
    candidate.codec.table = Some(crate::TableIndexedDispatch {
        table_base: Some(anchor),
        ..entry_table
    });
    candidate.postlude = Some(PostludeDispatch {
        stack_check_start: shape.start,
        branch: shape.branch,
        head: shape.head,
    });
    candidate.evidence.push(
        "handler ends in the shared stack-check postlude; ja targets the proven dispatch head"
            .to_owned(),
    );
    candidate.evidence.push(format!(
        "fall-through relocation path reaches the head without writing VIP, VSP, or the table register; table base {anchor} falls into the head"
    ));
    Ok(Some(candidate))
}

#[cfg(test)]
mod tests {
    use iced_x86::{Code, Decoder, DecoderOptions, code_asm::*};
    use vmpx_disasm::{DirectJumpFragment, DirectJumpTraceStop, DirectJumpTraceStopKind};
    use vmpx_pe::Va;

    use super::*;
    use crate::{HandlerSliceConfig, backward_slice_handler_trace};

    fn trace_from_bytes(bytes: &[u8]) -> DirectJumpTrace {
        let mut decoder = Decoder::with_ip(64, bytes, 0x1_4000_1000, DecoderOptions::NONE);
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
        let terminal = instructions.last().expect("at least one instruction");
        let terminal_rva = terminal.rva;
        let end = terminal.next_rva().expect("small end RVA");
        DirectJumpTrace {
            entry: Rva(0x1000),
            decoded_instructions: instructions.len(),
            fragments: vec![DirectJumpFragment {
                start: Rva(0x1000),
                end,
                instructions,
                next_fragment: None,
            }],
            stop: DirectJumpTraceStop {
                kind: DirectJumpTraceStopKind::IndirectBranch,
                at: terminal_rva,
                direct_target: None,
                fallthrough: None,
                detail: None,
            },
        }
    }

    fn report_from_assembler(mut assembler: CodeAssembler) -> DispatchExtractionReport {
        let bytes = assembler.assemble(0x1_4000_1000).expect("assemble handler");
        let trace = trace_from_bytes(&bytes);
        let slice = backward_slice_handler_trace(
            &trace,
            &HandlerSliceConfig {
                live_out_registers: vec![
                    Register::R11,
                    Register::R14,
                    Register::R15,
                    Register::RBP,
                ],
                live_out_rflags: 0,
            },
        );
        extract_relative_dispatch_codecs(
            &trace,
            &slice,
            HandlerRegisterRoles {
                vip: Register::R15,
                rolling_key: Register::RBP,
                handler: Register::R14,
            },
        )
    }

    #[test]
    fn extracts_relative_dispatch_codec_from_data_flow() {
        let mut assembler = CodeAssembler::new(64).expect("x64 assembler");
        assembler.mov(r10d, dword_ptr(r15)).expect("encode load");
        assembler.add(r15, 4).expect("encode VIP advance");
        assembler.xor(r10d, ebp).expect("encode key mix");
        assembler.ror(r10d, 1).expect("encode rotate");
        assembler.neg(r10d).expect("encode negate");
        assembler
            .add(r10d, 0x7638_338D)
            .expect("encode constant add");
        assembler
            .xor(r10d, 0x76DF_F93A)
            .expect("encode constant xor");
        assembler.push(rbp).expect("encode key save");
        assembler
            .xor(dword_ptr(rsp), r10d)
            .expect("encode partial key update");
        assembler.pop(rbp).expect("encode key restore");
        assembler
            .movsxd(r10, r10d)
            .expect("encode displacement extension");
        assembler
            .add(r14, r10)
            .expect("encode relative handler update");
        assembler.jmp(r14).expect("encode indirect dispatch");
        let bytes = assembler.assemble(0x1_4000_1000).expect("assemble handler");
        let trace = trace_from_bytes(&bytes);
        let slice = backward_slice_handler_trace(
            &trace,
            &HandlerSliceConfig {
                live_out_registers: vec![
                    Register::R11,
                    Register::R14,
                    Register::R15,
                    Register::RBP,
                ],
                live_out_rflags: 0,
            },
        );
        let report = extract_relative_dispatch_codecs(
            &trace,
            &slice,
            HandlerRegisterRoles {
                vip: Register::R15,
                rolling_key: Register::RBP,
                handler: Register::R14,
            },
        );

        assert!(report.rejected_sources.is_empty());
        let candidate = report.candidates.first().expect("one dispatch candidate");
        assert_eq!(report.candidates.len(), 1);
        assert_eq!(candidate.vip_delta, 4);
        assert_eq!(candidate.confidence, 100);
        assert_eq!(
            candidate.codec.displacement.input_key_operation,
            Some(KeyOperation::Xor)
        );
        assert_eq!(
            candidate.codec.displacement.update_key_operation,
            Some(KeyOperation::Xor)
        );
        assert_eq!(
            candidate.codec.displacement.transforms.transforms,
            vec![
                Transform {
                    op: TransformOp::Ror,
                    width: BitWidth::W32,
                    operand: 1,
                },
                Transform {
                    op: TransformOp::Neg,
                    width: BitWidth::W32,
                    operand: 0,
                },
                Transform {
                    op: TransformOp::Add,
                    width: BitWidth::W32,
                    operand: 0x7638_338D,
                },
                Transform {
                    op: TransformOp::Xor,
                    width: BitWidth::W32,
                    operand: 0x76DF_F93A,
                },
            ]
        );
    }

    fn conditional_trace(bytes: &[u8]) -> DirectJumpTrace {
        let mut decoder = Decoder::with_ip(64, bytes, 0x1_4000_1000, DecoderOptions::NONE);
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
        let terminal = instructions.last().expect("at least one instruction");
        let terminal_rva = terminal.rva;
        let fallthrough = terminal.next_rva().expect("small end RVA");
        DirectJumpTrace {
            entry: Rva(0x1000),
            decoded_instructions: instructions.len(),
            fragments: vec![DirectJumpFragment {
                start: Rva(0x1000),
                end: fallthrough,
                instructions,
                next_fragment: None,
            }],
            stop: DirectJumpTraceStop {
                kind: DirectJumpTraceStopKind::ConditionalBranch,
                at: terminal_rva,
                direct_target: Some(Rva(0x2000)),
                fallthrough: Some(fallthrough),
                detail: None,
            },
        }
    }

    const STACK_CHECK_ROLES: HandlerRegisterRoles = HandlerRegisterRoles {
        vip: Register::R15,
        rolling_key: Register::RCX,
        handler: Register::R14,
    };
    const STACK_CHECK: StackCheckRoles = StackCheckRoles {
        vsp: Register::RBP,
        context_base: Register::RSP,
    };

    #[test]
    fn stack_check_tolerates_dead_flag_junk_between_lea_and_cmp() {
        // `lea tmp,[ctx+disp]; cmp vip,rsp (dead junk); clc; cmp vsp,tmp; ja head`
        let mut a = CodeAssembler::new(64).expect("x64 assembler");
        a.lea(rax, qword_ptr(rsp + 0x180)).expect("stack-guard lea");
        a.cmp(r15, rsp).expect("dead flag-only junk cmp");
        a.clc().expect("flag junk");
        a.cmp(rbp, rax).expect("real vsp vs guard compare");
        a.ja(0x1_4000_2000_u64).expect("stack-check branch");
        let bytes = a.assemble(0x1_4000_1000).expect("assemble");
        let shape = match_stack_check(&conditional_trace(&bytes), STACK_CHECK_ROLES, STACK_CHECK)
            .expect("stack-check shape recovered despite the junk cmp");
        assert_eq!(shape.head, Rva(0x2000));
    }

    #[test]
    fn stack_check_rejects_register_write_between_lea_and_cmp() {
        // A real register write (not flag-only) must still break the match.
        let mut a = CodeAssembler::new(64).expect("x64 assembler");
        a.lea(rax, qword_ptr(rsp + 0x180)).expect("stack-guard lea");
        a.add(rbx, 1).expect("register write clobbers the tolerance");
        a.cmp(rbp, rax).expect("real compare");
        a.ja(0x1_4000_2000_u64).expect("stack-check branch");
        let bytes = a.assemble(0x1_4000_1000).expect("assemble");
        assert!(match_stack_check(&conditional_trace(&bytes), STACK_CHECK_ROLES, STACK_CHECK).is_none());
    }

    #[test]
    fn stack_check_allows_handler_register_as_guard_scratch() {
        // VMProtect may reuse the handler register as the guard scratch: the
        // handler is dead across the postlude (the table-indexed head reloads
        // it), so `lea handler,[ctx+disp]; cmp vsp,handler; ja head` is sound.
        let mut a = CodeAssembler::new(64).expect("x64 assembler");
        a.lea(r14, qword_ptr(rsp + 0x180)).expect("guard lea into handler");
        a.cmp(rbp, r14).expect("vsp vs guard compare");
        a.ja(0x1_4000_2000_u64).expect("stack-check branch");
        let bytes = a.assemble(0x1_4000_1000).expect("assemble");
        let shape = match_stack_check(&conditional_trace(&bytes), STACK_CHECK_ROLES, STACK_CHECK)
            .expect("handler-register guard scratch is accepted");
        assert_eq!(shape.head, Rva(0x2000));
    }

    #[test]
    fn stack_check_rejects_vip_or_vsp_as_guard_scratch() {
        // The guard scratch must never alias VIP or VSP: both are live VM state
        // that has to survive the postlude into the dispatch head.
        for scratch in [r15, rbp] {
            let mut a = CodeAssembler::new(64).expect("x64 assembler");
            a.lea(scratch, qword_ptr(rsp + 0x180)).expect("guard lea");
            a.cmp(rbp, scratch).expect("vsp vs guard compare");
            a.ja(0x1_4000_2000_u64).expect("stack-check branch");
            let bytes = a.assemble(0x1_4000_1000).expect("assemble");
            assert!(
                match_stack_check(&conditional_trace(&bytes), STACK_CHECK_ROLES, STACK_CHECK)
                    .is_none(),
                "guard scratch aliasing a live VM register must fail closed"
            );
        }
    }

    #[test]
    fn recovers_static_handler_anchor_before_dispatch() {
        let mut assembler = CodeAssembler::new(64).expect("x64 assembler");
        assembler
            .lea(r14, qword_ptr(0x1800_u64))
            .expect("static handler anchor");
        assembler.mov(r10d, dword_ptr(r15)).expect("encoded load");
        assembler.add(r15, 4).expect("VIP advance");
        assembler.xor(r10d, ebp).expect("input-key mix");
        assembler.push(rbp).expect("save full key");
        assembler
            .xor(dword_ptr(rsp), r10d)
            .expect("partial key update");
        assembler.pop(rbp).expect("restore full key");
        assembler.movsxd(r10, r10d).expect("sign extension");
        assembler.add(r14, r10).expect("handler update");
        assembler
            .xor(r10d, r10d)
            .expect("post-handler displacement clobber");
        assembler
            .mov(byte_ptr(rsp + r10), 0)
            .expect("retain the post-handler clobber");
        assembler.jmp(r14).expect("dispatch");

        let report = report_from_assembler(assembler);
        let [candidate] = report.candidates.as_slice() else {
            panic!("expected one candidate: {report:?}");
        };
        assert_eq!(candidate.handler_rebase, Some(Va(0x1800)));
        assert!(candidate.handler_rebase_rva.is_some());
    }

    #[test]
    fn rejects_displaced_segmented_and_address_truncated_sources() {
        let mut assembler = CodeAssembler::new(64).expect("x64 assembler");
        assembler
            .mov(r10d, dword_ptr(r15 + 4))
            .expect("displaced source");
        assembler
            .mov(r10d, dword_ptr(r15).fs())
            .expect("segmented source");
        assembler
            .mov(r10d, dword_ptr(r15d))
            .expect("address-truncated source");
        let bytes = assembler.assemble(0x1_4000_1000).expect("assemble sources");
        let trace = trace_from_bytes(&bytes);
        let instructions = trace.instructions().collect::<Vec<_>>();

        assert_eq!(instructions.len(), 3);
        assert!(
            instructions
                .iter()
                .all(|decoded| !is_source_load(&decoded.instruction, Register::R15))
        );
    }

    #[test]
    fn rejects_partial_vip_adjustment() {
        let mut assembler = CodeAssembler::new(64).expect("x64 assembler");
        assembler.mov(r10d, dword_ptr(r15)).expect("source");
        assembler.add(r15d, 4).expect("truncated VIP advance");
        assembler.xor(r10d, ebp).expect("key mix");
        assembler.neg(r10d).expect("transform");
        assembler.push(rbp).expect("key save");
        assembler.xor(dword_ptr(rsp), r10d).expect("key update");
        assembler.pop(rbp).expect("key restore");
        assembler.movsxd(r10, r10d).expect("sign extension");
        assembler.add(r14, r10).expect("handler update");
        assembler.jmp(r14).expect("dispatch");
        let bytes = assembler.assemble(0x1_4000_1000).expect("assemble handler");
        let trace = trace_from_bytes(&bytes);
        let slice = backward_slice_handler_trace(
            &trace,
            &HandlerSliceConfig {
                live_out_registers: vec![
                    Register::R11,
                    Register::R14,
                    Register::R15,
                    Register::RBP,
                ],
                live_out_rflags: 0,
            },
        );
        let report = extract_relative_dispatch_codecs(
            &trace,
            &slice,
            HandlerRegisterRoles {
                vip: Register::R15,
                rolling_key: Register::RBP,
                handler: Register::R14,
            },
        );

        assert!(report.candidates.is_empty());
        assert!(
            report
                .rejected_sources
                .iter()
                .any(|rejection| rejection.reason.contains("full-width"))
        );
    }

    #[test]
    fn rejects_key_update_before_mix_and_duplicate_updates() {
        for duplicate in [false, true] {
            let mut assembler = CodeAssembler::new(64).expect("x64 assembler");
            assembler.mov(r10, qword_ptr(r15)).expect("source");
            assembler.add(r15, 8).expect("VIP advance");
            if duplicate {
                assembler.xor(r10, rbp).expect("input-key mix");
                assembler.xor(rbp, r10).expect("first key update");
                assembler.add(rbp, r10).expect("second key update");
            } else {
                assembler.xor(rbp, r10).expect("early key update");
                assembler.xor(r10, rbp).expect("late input-key mix");
            }
            assembler.add(r14, r10).expect("handler update");
            assembler.jmp(r14).expect("dispatch");

            let report = report_from_assembler(assembler);
            assert!(report.candidates.is_empty());
        }
    }

    #[test]
    fn rejects_wide_sign_extension_and_pre_source_handler_write() {
        let mut wide_extension = CodeAssembler::new(64).expect("x64 assembler");
        wide_extension.mov(r10w, word_ptr(r15)).expect("source");
        wide_extension.add(r15, 2).expect("VIP advance");
        wide_extension.xor(r10w, bp).expect("input-key mix");
        wide_extension.push(rbp).expect("key save");
        wide_extension.xor(word_ptr(rsp), r10w).expect("key update");
        wide_extension.pop(rbp).expect("key restore");
        wide_extension
            .movsxd(r10, r10d)
            .expect("incorrect wide sign extension");
        wide_extension.add(r14, r10).expect("handler update");
        wide_extension.jmp(r14).expect("dispatch");
        assert!(report_from_assembler(wide_extension).candidates.is_empty());

        let mut handler_write = CodeAssembler::new(64).expect("x64 assembler");
        handler_write.inc(r14).expect("pre-source handler write");
        handler_write.mov(r10, qword_ptr(r15)).expect("source");
        handler_write.add(r15, 8).expect("VIP advance");
        handler_write.xor(r10, rbp).expect("input-key mix");
        handler_write.xor(rbp, r10).expect("key update");
        handler_write.add(r14, r10).expect("handler update");
        handler_write.jmp(r14).expect("dispatch");
        assert!(report_from_assembler(handler_write).candidates.is_empty());
    }

    #[test]
    fn recognizes_table_indexed_keyless_dispatch() {
        // movzx op,[vip]; vip+=1; movsxd handler,[table+op*4]; handler+=table.
        let mut assembler = CodeAssembler::new(64).expect("x64 assembler");
        assembler
            .lea(r11, qword_ptr(0x1800_u64))
            .expect("static table anchor");
        assembler
            .movzx(eax, byte_ptr(r15))
            .expect("zero-extended opcode load");
        assembler.add(r15, 1).expect("VIP advance");
        assembler
            .movsxd(r14, dword_ptr(r11 + rax * 4))
            .expect("sign-extended table entry load");
        assembler.add(r14, r11).expect("handler base addition");
        assembler.jmp(r14).expect("indirect dispatch");

        let report = report_from_assembler(assembler);
        let [candidate] = report.candidates.as_slice() else {
            panic!("expected one table-indexed candidate: {report:?}");
        };
        let table = candidate.codec.table.expect("table-indexed dispatch");
        assert_eq!(table.opcode_width, BitWidth::W8);
        assert_eq!(table.entry_width, BitWidth::W32);
        assert_eq!(table.stride, 4);
        assert_eq!(table.table_base, Some(Va(0x1800)));
        assert_eq!(candidate.vip_delta, 1);
        assert_eq!(candidate.handler_rebase, Some(Va(0x1800)));
        assert!(candidate.codec.displacement.input_key_operation.is_none());
        assert!(candidate.codec.displacement.update_key_operation.is_none());
        assert!(
            candidate
                .codec
                .displacement
                .transforms
                .transforms
                .is_empty()
        );
    }

    #[test]
    fn rejects_unsound_table_indexed_dispatch_shapes() {
        // A sign-extended opcode would corrupt the unsigned table index.
        let mut signed_opcode = CodeAssembler::new(64).expect("x64 assembler");
        signed_opcode
            .lea(r11, qword_ptr(0x1800_u64))
            .expect("anchor");
        signed_opcode
            .movsx(eax, byte_ptr(r15))
            .expect("sign-extended opcode");
        signed_opcode.add(r15, 1).expect("VIP advance");
        signed_opcode
            .movsxd(r14, dword_ptr(r11 + rax * 4))
            .expect("table entry load");
        signed_opcode.add(r14, r11).expect("handler base addition");
        signed_opcode.jmp(r14).expect("dispatch");
        assert!(report_from_assembler(signed_opcode).candidates.is_empty());

        // The table load must be indexed by the decoded opcode register.
        let mut wrong_index = CodeAssembler::new(64).expect("x64 assembler");
        wrong_index.lea(r11, qword_ptr(0x1800_u64)).expect("anchor");
        wrong_index.movzx(eax, byte_ptr(r15)).expect("opcode load");
        wrong_index.add(r15, 1).expect("VIP advance");
        wrong_index
            .movsxd(r14, dword_ptr(r11 + rcx * 4))
            .expect("mis-indexed table load");
        wrong_index.add(r14, r11).expect("handler base addition");
        wrong_index.jmp(r14).expect("dispatch");
        assert!(report_from_assembler(wrong_index).candidates.is_empty());

        // A dense signed table requires the stride to equal the entry width.
        let mut wrong_stride = CodeAssembler::new(64).expect("x64 assembler");
        wrong_stride
            .lea(r11, qword_ptr(0x1800_u64))
            .expect("anchor");
        wrong_stride.movzx(eax, byte_ptr(r15)).expect("opcode load");
        wrong_stride.add(r15, 1).expect("VIP advance");
        wrong_stride
            .movsxd(r14, dword_ptr(r11 + rax * 2))
            .expect("mismatched-stride table load");
        wrong_stride.add(r14, r11).expect("handler base addition");
        wrong_stride.jmp(r14).expect("dispatch");
        assert!(report_from_assembler(wrong_stride).candidates.is_empty());
    }

    #[test]
    fn rejects_pre_source_key_write_and_post_extension_high_byte_write() {
        let mut key_write = CodeAssembler::new(64).expect("x64 assembler");
        key_write.not(rbp).expect("pre-source key mutation");
        key_write.mov(r10, qword_ptr(r15)).expect("source");
        key_write.add(r15, 8).expect("VIP advance");
        key_write.xor(r10, rbp).expect("input-key mix");
        key_write.xor(rbp, r10).expect("key update");
        key_write.add(r14, r10).expect("handler update");
        key_write.jmp(r14).expect("dispatch");
        assert!(report_from_assembler(key_write).candidates.is_empty());

        let mut high_byte = CodeAssembler::new(64).expect("x64 assembler");
        high_byte.movzx(eax, byte_ptr(r15)).expect("byte source");
        high_byte.inc(r15).expect("VIP advance");
        high_byte.xor(al, bpl).expect("input-key mix");
        high_byte.xor(bpl, al).expect("key update");
        high_byte.movsx(rax, al).expect("exact sign extension");
        high_byte.mov(ah, 1).expect("high-byte clobber");
        high_byte.add(r14, rax).expect("handler update");
        high_byte.jmp(r14).expect("dispatch");
        assert!(report_from_assembler(high_byte).candidates.is_empty());
    }
}
