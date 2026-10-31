use std::collections::{BTreeMap, BTreeSet};

use iced_x86::{Instruction, InstructionInfoFactory, Mnemonic, OpAccess, OpKind, Register};
use serde::Serialize;
use vmpx_disasm::DirectJumpTrace;
use vmpx_pe::Rva;

use crate::{
    BitWidth, EffectKnowledge, NativeBackwardSlice, NativeInstructionDisposition, ResizeOp,
    VmHandlerAnalysis, VmHandlerSemantic,
    dispatch_extract::{canonical, is_active, signed_immediate, writes_register},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HandlerEffectRoles {
    pub vsp: Register,
    pub context_base: Register,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HandlerMemoryRegion {
    VmStack,
    NativeContext,
    NativeStackScratch,
    Bytecode,
    Other,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct HandlerMemoryEffect {
    pub rva: Rva,
    pub region: HandlerMemoryRegion,
    pub base: String,
    pub index: Option<String>,
    pub scale: u32,
    pub displacement: i64,
    pub entry_displacement: Option<i64>,
    pub width_bytes: u32,
    pub reads: bool,
    pub writes: bool,
    /// Set for FS/GS-relative accesses, which are never flat addresses.
    pub segment_override: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct HandlerVspAdjustment {
    pub rva: Rva,
    pub delta: i32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct HandlerNarrowVspAccess {
    pub rva: Rva,
    pub width_bytes: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct HandlerResizeEffect {
    pub rva: Rva,
    pub operation: ResizeOp,
    pub input_width: BitWidth,
    pub output_width: BitWidth,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[allow(clippy::struct_excessive_bools)]
pub struct HandlerEffectSummary {
    pub schema_version: u32,
    pub semantic_end: Rva,
    pub vsp_delta: Option<i32>,
    pub vsp_adjustments: Vec<HandlerVspAdjustment>,
    pub memory: Vec<HandlerMemoryEffect>,
    pub captures_native_flags_to_vsp: bool,
    pub captured_flags_source: Option<Rva>,
    pub pushes_pre_adjustment_vsp: bool,
    pub loads_vsp_from_top: bool,
    pub loads_vip_from_vsp: bool,
    pub resets_key_from_vip: bool,
    pub reinitializes_handler_base: bool,
    pub arithmetic_operations: Vec<String>,
    pub resize_operations: Vec<HandlerResizeEffect>,
    pub timestamp_counter_reads: Vec<Rva>,
    /// Stores of the low bytes of the pre-adjustment VSP to the VM stack top.
    pub narrow_vsp_pushes: Vec<HandlerNarrowVspAccess>,
    /// Partial-width loads into the VSP register from the VM stack top.
    pub narrow_vsp_loads: Vec<HandlerNarrowVspAccess>,
    pub cpuid_reads: Vec<Rva>,
    pub native_flag_restores: Vec<Rva>,
    pub diagnostics: Vec<String>,
}

fn access_reads(access: OpAccess) -> bool {
    matches!(
        access,
        OpAccess::Read | OpAccess::CondRead | OpAccess::ReadWrite | OpAccess::ReadCondWrite
    )
}

fn access_writes(access: OpAccess) -> bool {
    matches!(
        access,
        OpAccess::Write | OpAccess::CondWrite | OpAccess::ReadWrite | OpAccess::ReadCondWrite
    )
}

fn signed_displacement(displacement: u64) -> i64 {
    i64::from_ne_bytes(displacement.to_ne_bytes())
}

fn vsp_adjustment(instruction: &Instruction, vsp: Register) -> Result<Option<i32>, String> {
    let target = canonical(vsp);
    if instruction.op0_kind() == OpKind::Register && canonical(instruction.op0_register()) == target
    {
        if instruction.op0_register().size() != 8 {
            return Err("VSP write does not target the full 64-bit register".to_owned());
        }
        let delta = match instruction.mnemonic() {
            Mnemonic::Add | Mnemonic::Sub => {
                let immediate = signed_immediate(instruction, 1)
                    .ok_or_else(|| "VSP arithmetic does not use an immediate".to_owned())?;
                let immediate = i32::try_from(immediate)
                    .map_err(|_| "VSP adjustment exceeds i32".to_owned())?;
                if instruction.mnemonic() == Mnemonic::Sub {
                    immediate
                        .checked_neg()
                        .ok_or_else(|| "VSP subtraction overflows".to_owned())?
                } else {
                    immediate
                }
            }
            Mnemonic::Inc => 1,
            Mnemonic::Dec => -1,
            Mnemonic::Lea
                if instruction.op1_kind() == OpKind::Memory
                    && canonical(instruction.memory_base()) == target
                    && instruction.memory_base().size() == 8
                    && instruction.memory_index() == Register::None =>
            {
                i32::try_from(signed_displacement(instruction.memory_displacement64()))
                    .map_err(|_| "VSP LEA displacement exceeds i32".to_owned())?
            }
            _ => return Err("unmodelled write to the VSP register".to_owned()),
        };
        return Ok(Some(delta));
    }
    Ok(None)
}

fn is_semantic_arithmetic(instruction: &Instruction, vsp: Register) -> bool {
    matches!(
        instruction.mnemonic(),
        Mnemonic::Add
            | Mnemonic::Sub
            | Mnemonic::Imul
            | Mnemonic::Mul
            | Mnemonic::Div
            | Mnemonic::Xor
            | Mnemonic::And
            | Mnemonic::Or
            | Mnemonic::Not
            | Mnemonic::Neg
            | Mnemonic::Shl
            | Mnemonic::Shr
            | Mnemonic::Sar
            | Mnemonic::Rol
            | Mnemonic::Ror
            | Mnemonic::Shld
            | Mnemonic::Shrd
    ) && !(instruction.op0_kind() == OpKind::Register
        && canonical(instruction.op0_register()) == canonical(vsp))
}

fn stores_pre_adjustment_vsp(
    instruction: &Instruction,
    vsp: Register,
    aliases: &BTreeSet<Register>,
) -> bool {
    instruction.mnemonic() == Mnemonic::Mov
        && instruction.op0_kind() == OpKind::Memory
        && canonical(instruction.memory_base()) == canonical(vsp)
        && instruction.memory_index() == Register::None
        && signed_displacement(instruction.memory_displacement64()) == 0
        && instruction.op1_kind() == OpKind::Register
        && instruction.op1_register().size() == 8
        && aliases.contains(&canonical(instruction.op1_register()))
}

fn narrow_vsp_push_width(
    instruction: &Instruction,
    vsp: Register,
    aliases: &BTreeSet<Register>,
) -> Option<u32> {
    let width = u32::try_from(instruction.memory_size().size()).ok()?;
    (instruction.mnemonic() == Mnemonic::Mov
        && instruction.op0_kind() == OpKind::Memory
        && canonical(instruction.memory_base()) == canonical(vsp)
        && instruction.memory_base().size() == 8
        && instruction.memory_index() == Register::None
        && signed_displacement(instruction.memory_displacement64()) == 0
        && matches!(width, 2 | 4)
        && instruction.op1_kind() == OpKind::Register
        && u32::try_from(instruction.op1_register().size()).ok() == Some(width)
        && aliases.contains(&canonical(instruction.op1_register())))
    .then_some(width)
}

fn narrow_vsp_load_width(instruction: &Instruction, vsp: Register) -> Option<u32> {
    let width = u32::try_from(instruction.memory_size().size()).ok()?;
    (instruction.mnemonic() == Mnemonic::Mov
        && instruction.op0_kind() == OpKind::Register
        && canonical(instruction.op0_register()) == canonical(vsp)
        && matches!(width, 2 | 4)
        && u32::try_from(instruction.op0_register().size()).ok() == Some(width)
        && instruction.op1_kind() == OpKind::Memory
        && canonical(instruction.memory_base()) == canonical(vsp)
        && instruction.memory_base().size() == 8
        && instruction.memory_index() == Register::None
        && signed_displacement(instruction.memory_displacement64()) == 0
        && crate::dispatch_extract::has_flat_segment_address(instruction))
    .then_some(width)
}

fn instruction_loads_vsp_from_top(instruction: &Instruction, vsp: Register) -> bool {
    instruction.mnemonic() == Mnemonic::Mov
        && instruction.op0_kind() == OpKind::Register
        && canonical(instruction.op0_register()) == canonical(vsp)
        && instruction.op0_register().size() == 8
        && instruction.op1_kind() == OpKind::Memory
        && canonical(instruction.memory_base()) == canonical(vsp)
        && instruction.memory_index() == Register::None
        && signed_displacement(instruction.memory_displacement64()) == 0
        && instruction.memory_size().size() == 8
}

fn update_pre_adjustment_vsp_aliases(
    factory: &mut InstructionInfoFactory,
    instruction: &Instruction,
    aliases: &mut BTreeSet<Register>,
) {
    let copied_alias = instruction.mnemonic() == Mnemonic::Mov
        && instruction.op0_kind() == OpKind::Register
        && instruction.op0_register().size() == 8
        && instruction.op1_kind() == OpKind::Register
        && instruction.op1_register().size() == 8
        && aliases.contains(&canonical(instruction.op1_register()));
    let destination =
        (instruction.op0_kind() == OpKind::Register).then(|| canonical(instruction.op0_register()));
    if let Some(destination) = destination
        && writes_register(factory, instruction, destination)
    {
        aliases.remove(&destination);
        if copied_alias {
            aliases.insert(destination);
        }
    }
}

fn register_copy(instruction: &Instruction) -> Option<(Register, Register)> {
    (instruction.mnemonic() == Mnemonic::Mov
        && instruction.op0_kind() == OpKind::Register
        && instruction.op0_register().size() == 8
        && instruction.op1_kind() == OpKind::Register
        && instruction.op1_register().size() == 8)
        .then(|| {
            (
                canonical(instruction.op0_register()),
                canonical(instruction.op1_register()),
            )
        })
}

fn loads_qword_from_vsp_top(instruction: &Instruction, vsp: Register) -> Option<Register> {
    (instruction.mnemonic() == Mnemonic::Mov
        && instruction.op0_kind() == OpKind::Register
        && instruction.op0_register().size() == 8
        && instruction.op1_kind() == OpKind::Memory
        && canonical(instruction.memory_base()) == canonical(vsp)
        && instruction.memory_index() == Register::None
        && signed_displacement(instruction.memory_displacement64()) == 0
        && instruction.memory_size().size() == 8)
        .then(|| canonical(instruction.op0_register()))
}

fn pops_native_flags_to_vsp(instruction: &Instruction, vsp: Register) -> bool {
    instruction.mnemonic() == Mnemonic::Pop
        && instruction.op0_kind() == OpKind::Memory
        && canonical(instruction.memory_base()) == canonical(vsp)
        && instruction.memory_index() == Register::None
        && signed_displacement(instruction.memory_displacement64()) == 0
        && instruction.memory_size().size() == 8
}

fn pops_native_flags_to_register(instruction: &Instruction, vsp: Register) -> Option<Register> {
    (instruction.mnemonic() == Mnemonic::Pop
        && instruction.op0_kind() == OpKind::Register
        && instruction.op0_register().size() == 8
        && canonical(instruction.op0_register()) != canonical(vsp)
        && canonical(instruction.op0_register()) != Register::RSP)
        .then(|| canonical(instruction.op0_register()))
}

fn stores_qword_register_to_vsp_top(
    instruction: &Instruction,
    vsp: Register,
    register: Register,
) -> bool {
    instruction.mnemonic() == Mnemonic::Mov
        && instruction.op0_kind() == OpKind::Memory
        && canonical(instruction.memory_base()) == canonical(vsp)
        && instruction.memory_base().size() == 8
        && instruction.memory_index() == Register::None
        && signed_displacement(instruction.memory_displacement64()) == 0
        && instruction.memory_size().size() == 8
        && instruction.op1_kind() == OpKind::Register
        && instruction.op1_register().size() == 8
        && canonical(instruction.op1_register()) == register
}

fn writes_static_handler_base(instruction: &Instruction, handler: Register) -> bool {
    instruction.mnemonic() == Mnemonic::Lea
        && instruction.op0_kind() == OpKind::Register
        && canonical(instruction.op0_register()) == canonical(handler)
        && instruction.op1_kind() == OpKind::Memory
        && matches!(
            instruction.memory_base(),
            Register::None | Register::RIP | Register::EIP
        )
        && instruction.memory_index() == Register::None
}

fn explicit_destination(instruction: &Instruction) -> Option<Register> {
    (instruction.op0_kind() == OpKind::Register).then(|| canonical(instruction.op0_register()))
}

fn assigns_zero(instruction: &Instruction) -> bool {
    let zero_extends_full_register =
        instruction.op0_kind() == OpKind::Register && instruction.op0_register().size() >= 4;
    zero_extends_full_register
        && ((instruction.mnemonic() == Mnemonic::Mov
            && signed_immediate(instruction, 1) == Some(0))
            || (matches!(instruction.mnemonic(), Mnemonic::Xor | Mnemonic::Sub)
                && instruction.op1_kind() == OpKind::Register
                && instruction.op0_register() == instruction.op1_register()))
}

/// A register's own constant self-offset (`reg += imm` / `reg -= imm`,
/// `inc`/`dec reg`, `lea reg,[reg+imm]`): the result stays a constant offset of
/// the register's prior value, so a value provenance (e.g. "derived from the
/// popped VM-stack qword") survives the write. VMProtect's descending bytecode
/// predecrements the VIP (`sub vip,width`) between loading a branch target and
/// the dispatch read; without this, that predecrement would drop the VIP's
/// popped-value provenance before the handler's effects are classified.
fn is_constant_self_offset(instruction: &Instruction, register: Register) -> bool {
    if instruction.op0_kind() != OpKind::Register
        || canonical(instruction.op0_register()) != canonical(register)
        || instruction.op0_register().size() != 8
    {
        return false;
    }
    match instruction.mnemonic() {
        Mnemonic::Add | Mnemonic::Sub => signed_immediate(instruction, 1).is_some(),
        Mnemonic::Inc | Mnemonic::Dec => true,
        Mnemonic::Lea => {
            instruction.op1_kind() == OpKind::Memory
                && canonical(instruction.memory_base()) == canonical(register)
                && instruction.memory_index() == Register::None
        }
        _ => false,
    }
}

fn preserves_destination_by_zero_arithmetic(
    instruction: &Instruction,
    zero_registers: &BTreeSet<Register>,
) -> bool {
    if !matches!(instruction.mnemonic(), Mnemonic::Add | Mnemonic::Sub)
        || instruction.op0_kind() != OpKind::Register
        || instruction.op0_register().size() != 8
    {
        return false;
    }
    signed_immediate(instruction, 1) == Some(0)
        || (instruction.op1_kind() == OpKind::Register
            && instruction.op1_register().size() == 8
            && zero_registers.contains(&canonical(instruction.op1_register())))
}

/// Extracts native memory, VSP, arithmetic, and VM-visible flag effects from
/// the semantic prefix of a handler. `semantic_end` must be the proven encoded
/// dispatch load, so dispatch decoding and shared postludes are excluded.
#[must_use]
pub fn extract_handler_effects(
    trace: &DirectJumpTrace,
    slice: &NativeBackwardSlice,
    roles: HandlerEffectRoles,
    dispatch_roles: crate::HandlerRegisterRoles,
    semantic_end: Rva,
) -> HandlerEffectSummary {
    extract_handler_effects_keyed(trace, slice, roles, dispatch_roles, semantic_end, true)
}

/// As [`extract_handler_effects`]. In a keyless VM the rolling-key and handler
/// registers carry no VM state across handlers, so arithmetic that targets them
/// is ordinary scratch arithmetic and still counts as a semantic operation.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn extract_handler_effects_keyed(
    trace: &DirectJumpTrace,
    slice: &NativeBackwardSlice,
    roles: HandlerEffectRoles,
    dispatch_roles: crate::HandlerRegisterRoles,
    semantic_end: Rva,
    keyed: bool,
) -> HandlerEffectSummary {
    let dispositions: BTreeMap<_, _> = slice
        .instructions
        .iter()
        .map(|instruction| (instruction.rva, instruction.disposition))
        .collect();
    let mut factory = InstructionInfoFactory::new();
    let mut memory = Vec::new();
    let mut vsp_adjustments = Vec::new();
    let mut vsp_delta = Some(0_i32);
    let mut diagnostics = Vec::new();
    let mut last_native_flags_writer = None;
    let mut pending_native_flags_source = None;
    let mut flags_in_register: Option<(Register, Option<Rva>)> = None;
    let mut captures_native_flags_to_vsp = false;
    let mut captured_flags_source = None;
    let mut pushes_pre_adjustment_vsp = false;
    let mut pre_adjustment_vsp_aliases = BTreeSet::from([canonical(roles.vsp)]);
    let mut vsp_top_value_aliases = BTreeSet::new();
    let mut zero_registers = BTreeSet::new();
    let mut final_handler_base_is_static = false;
    let mut arithmetic_operations = Vec::new();
    let mut resize_operations = Vec::new();
    let mut timestamp_counter_reads = Vec::new();
    let mut narrow_vsp_pushes = Vec::new();
    let mut narrow_vsp_loads = Vec::new();
    let mut cpuid_reads = Vec::new();
    let mut native_flag_restores = Vec::new();
    let mut reached_end = false;

    for decoded in trace.instructions() {
        if decoded.rva == semantic_end {
            reached_end = true;
            break;
        }
        let disposition = dispositions
            .get(&decoded.rva)
            .copied()
            .unwrap_or(NativeInstructionDisposition::Dead);
        if !is_active(disposition) {
            continue;
        }
        let instruction = &decoded.instruction;
        if instruction.mnemonic() == Mnemonic::Rdtsc {
            timestamp_counter_reads.push(decoded.rva);
        }
        if instruction.mnemonic() == Mnemonic::Cpuid {
            cpuid_reads.push(decoded.rva);
        }
        if instruction.mnemonic() == Mnemonic::Popfq {
            native_flag_restores.push(decoded.rva);
        }
        let narrow_vsp_load = narrow_vsp_load_width(instruction, roles.vsp);
        if let Some(width_bytes) = narrow_vsp_load {
            narrow_vsp_loads.push(HandlerNarrowVspAccess {
                rva: decoded.rva,
                width_bytes,
            });
        }
        if let Some(width_bytes) =
            narrow_vsp_push_width(instruction, roles.vsp, &pre_adjustment_vsp_aliases)
        {
            narrow_vsp_pushes.push(HandlerNarrowVspAccess {
                rva: decoded.rva,
                width_bytes,
            });
        }
        let vsp_delta_before_instruction = vsp_delta;
        let pops_flags = pending_native_flags_source.is_some()
            && pops_native_flags_to_vsp(instruction, roles.vsp);
        if let Some((register, source)) = flags_in_register {
            if stores_qword_register_to_vsp_top(instruction, roles.vsp, register) {
                captures_native_flags_to_vsp = true;
                captured_flags_source = source;
                flags_in_register = None;
            } else if writes_register(&mut factory, instruction, register) {
                flags_in_register = None;
            }
        }
        if pops_flags {
            captures_native_flags_to_vsp = true;
            captured_flags_source = pending_native_flags_source.take();
        } else if let Some(register) =
            pending_native_flags_source.and(pops_native_flags_to_register(instruction, roles.vsp))
        {
            flags_in_register = Some((register, pending_native_flags_source.take()));
        } else if pending_native_flags_source.is_some()
            && (matches!(
                instruction.mnemonic(),
                Mnemonic::Push | Mnemonic::Pushfq | Mnemonic::Pop | Mnemonic::Call | Mnemonic::Ret
            ) || writes_register(&mut factory, instruction, Register::RSP))
        {
            pending_native_flags_source = None;
        }
        if instruction.mnemonic() == Mnemonic::Pushfq {
            pending_native_flags_source = last_native_flags_writer;
        }
        if stores_pre_adjustment_vsp(instruction, roles.vsp, &pre_adjustment_vsp_aliases) {
            pushes_pre_adjustment_vsp = true;
        }
        if instruction_loads_vsp_from_top(instruction, roles.vsp) || narrow_vsp_load.is_some() {
            vsp_delta = None;
        }
        let loaded_vsp_value = loads_qword_from_vsp_top(instruction, roles.vsp);
        let copied_registers = register_copy(instruction);
        let copied_vsp_value =
            copied_registers.is_some_and(|(_, source)| vsp_top_value_aliases.contains(&source));
        let copied_zero =
            copied_registers.is_some_and(|(_, source)| zero_registers.contains(&source));
        let destination = explicit_destination(instruction);
        // A VIP/key role register that is a popped-value alias keeps that
        // provenance across its own constant self-offset (the descending
        // bytecode predecrement and equivalent bookkeeping). Scoped to these two
        // roles so ordinary scratch/VSP arithmetic is unaffected.
        let preserves_role_offset = destination.is_some_and(|destination| {
            vsp_top_value_aliases.contains(&destination)
                && [dispatch_roles.vip, dispatch_roles.rolling_key]
                    .iter()
                    .any(|role| canonical(*role) == destination)
                && is_constant_self_offset(instruction, destination)
        });
        let preserves_vsp_value = preserves_role_offset
            || destination.is_some_and(|destination| {
                vsp_top_value_aliases.contains(&destination)
                    && preserves_destination_by_zero_arithmetic(instruction, &zero_registers)
            });
        let preserves_zero = destination.is_some_and(|destination| {
            zero_registers.contains(&destination)
                && preserves_destination_by_zero_arithmetic(instruction, &zero_registers)
        });
        for alias in vsp_top_value_aliases.clone() {
            if writes_register(&mut factory, instruction, alias) {
                vsp_top_value_aliases.remove(&alias);
            }
        }
        for register in zero_registers.clone() {
            if writes_register(&mut factory, instruction, register) {
                zero_registers.remove(&register);
            }
        }
        if let Some(destination) = loaded_vsp_value {
            vsp_top_value_aliases.insert(destination);
        }
        if (copied_vsp_value || preserves_vsp_value)
            && let Some(destination) = destination
        {
            vsp_top_value_aliases.insert(destination);
        }
        if (copied_zero || preserves_zero || assigns_zero(instruction))
            && let Some(destination) = destination
        {
            zero_registers.insert(destination);
        }
        if writes_register(&mut factory, instruction, canonical(dispatch_roles.handler)) {
            final_handler_base_is_static =
                writes_static_handler_base(instruction, dispatch_roles.handler);
        }

        match (!instruction_loads_vsp_from_top(instruction, roles.vsp) && narrow_vsp_load.is_none())
            .then(|| vsp_adjustment(instruction, roles.vsp))
            .transpose()
        {
            Ok(None) => {}
            Ok(Some(Some(delta))) => {
                vsp_adjustments.push(HandlerVspAdjustment {
                    rva: decoded.rva,
                    delta,
                });
                vsp_delta = vsp_delta.and_then(|current| current.checked_add(delta));
                if vsp_delta.is_none() {
                    diagnostics.push("cumulative VSP delta overflows i32".to_owned());
                }
            }
            Ok(Some(None)) => {
                if writes_register(&mut factory, instruction, canonical(roles.vsp)) {
                    diagnostics.push(format!(
                        "unmodelled VSP write at {}: {:?}",
                        decoded.rva,
                        instruction.mnemonic()
                    ));
                    vsp_delta = None;
                }
            }
            Err(reason) => {
                diagnostics.push(format!("VSP effect at {}: {reason}", decoded.rva));
                vsp_delta = None;
            }
        }

        if is_semantic_arithmetic(instruction, roles.vsp)
            && !(instruction.op0_kind() == OpKind::Register
                && [
                    Some(dispatch_roles.vip),
                    keyed.then_some(dispatch_roles.rolling_key),
                    keyed.then_some(dispatch_roles.handler),
                ]
                .into_iter()
                .flatten()
                .map(canonical)
                .any(|role| role == canonical(instruction.op0_register())))
        {
            arithmetic_operations.push(format!("{:?}", instruction.mnemonic()));
        }
        let resize_operation = match instruction.mnemonic() {
            Mnemonic::Movzx => Some(ResizeOp::ZeroExtend),
            Mnemonic::Movsx | Mnemonic::Movsxd => Some(ResizeOp::SignExtend),
            _ => None,
        };
        if let Some(operation) = resize_operation
            && instruction.op0_kind() == OpKind::Register
            && instruction.op1_kind() == OpKind::Memory
            && let (Some(input_width), Some(output_width)) = (
                BitWidth::from_bytes(u32::try_from(instruction.memory_size().size()).unwrap_or(0)),
                BitWidth::from_bytes(u32::try_from(instruction.op0_register().size()).unwrap_or(0)),
            )
            && input_width.bits() < output_width.bits()
        {
            resize_operations.push(HandlerResizeEffect {
                rva: decoded.rva,
                operation,
                input_width,
                output_width,
            });
        }

        let info = factory.info(instruction);
        for used in info.used_memory() {
            let base = canonical(used.base());
            let index = canonical(used.index());
            let flat_address = crate::dispatch_extract::has_flat_segment_address(instruction);
            let full_base = used.base().size() == 8;
            let region = if base == canonical(roles.vsp)
                && full_base
                && flat_address
                && index == Register::None
            {
                HandlerMemoryRegion::VmStack
            } else if base == canonical(roles.context_base)
                && full_base
                && flat_address
                && index != Register::None
                && used.index().size() == 8
            {
                HandlerMemoryRegion::NativeContext
            } else if base == canonical(roles.context_base) && full_base && flat_address {
                HandlerMemoryRegion::NativeStackScratch
            } else if base == canonical(dispatch_roles.vip) && full_base && flat_address {
                HandlerMemoryRegion::Bytecode
            } else {
                HandlerMemoryRegion::Other
            };
            let displacement = signed_displacement(used.displacement());
            let effect = HandlerMemoryEffect {
                rva: decoded.rva,
                region,
                base: format!("{:?}", used.base()),
                index: (used.index() != Register::None).then(|| format!("{:?}", used.index())),
                scale: used.scale(),
                displacement,
                entry_displacement: (region == HandlerMemoryRegion::VmStack)
                    .then(|| {
                        vsp_delta_before_instruction
                            .and_then(|delta| i64::from(delta).checked_add(displacement))
                    })
                    .flatten(),
                width_bytes: u32::try_from(used.memory_size().size()).unwrap_or(0),
                reads: access_reads(used.access()),
                writes: access_writes(used.access()),
                segment_override: (!flat_address)
                    .then(|| format!("{:?}", instruction.segment_prefix())),
            };
            memory.push(effect);
        }
        update_pre_adjustment_vsp_aliases(
            &mut factory,
            instruction,
            &mut pre_adjustment_vsp_aliases,
        );
        if instruction.rflags_written()
            | instruction.rflags_cleared()
            | instruction.rflags_set()
            | instruction.rflags_undefined()
            != 0
        {
            last_native_flags_writer = Some(decoded.rva);
        }
    }
    if !reached_end {
        diagnostics.push(format!(
            "semantic boundary {semantic_end} is outside the traced handler"
        ));
        vsp_delta = None;
    }
    let loads_vsp_from_top = vsp_top_value_aliases.contains(&canonical(roles.vsp));
    let loads_vip_from_vsp = vsp_top_value_aliases.contains(&canonical(dispatch_roles.vip));
    let resets_key_from_vip = loads_vip_from_vsp
        && vsp_top_value_aliases.contains(&canonical(dispatch_roles.rolling_key));
    HandlerEffectSummary {
        schema_version: 8,
        semantic_end,
        vsp_delta,
        vsp_adjustments,
        memory,
        captures_native_flags_to_vsp,
        captured_flags_source,
        pushes_pre_adjustment_vsp,
        loads_vsp_from_top,
        loads_vip_from_vsp,
        resets_key_from_vip,
        reinitializes_handler_base: final_handler_base_is_static,
        arithmetic_operations,
        resize_operations,
        timestamp_counter_reads,
        narrow_vsp_pushes,
        narrow_vsp_loads,
        cpuid_reads,
        native_flag_restores,
        diagnostics,
    }
}

fn has_memory(
    summary: &HandlerEffectSummary,
    region: HandlerMemoryRegion,
    displacement: Option<i64>,
    reads: bool,
    writes: bool,
) -> bool {
    summary.memory.iter().any(|effect| {
        effect.region == region
            && displacement.is_none_or(|expected| effect.displacement == expected)
            && (!reads || effect.reads)
            && (!writes || effect.writes)
    })
}

fn has_memory_width(
    summary: &HandlerEffectSummary,
    region: HandlerMemoryRegion,
    displacement: i64,
    width_bytes: u32,
    reads: bool,
    writes: bool,
) -> bool {
    summary.memory.iter().any(|effect| {
        effect.region == region
            && effect.displacement == displacement
            && effect.width_bytes == width_bytes
            && (!reads || effect.reads)
            && (!writes || effect.writes)
    })
}

fn unique_width(candidates: &[BitWidth]) -> Option<BitWidth> {
    match candidates {
        [width] => Some(*width),
        _ => None,
    }
}

/// Classifies only handler families whose normalized effect shapes are
/// currently unambiguous. Anything else remains `Unknown` with its evidence.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn classify_handler_effects(
    entry: Rva,
    summary: &HandlerEffectSummary,
    operand_width: Option<BitWidth>,
    vip_delta: Option<i32>,
) -> VmHandlerAnalysis {
    let stack_read_0 = has_memory(summary, HandlerMemoryRegion::VmStack, Some(0), true, false);
    let stack_write_0 = has_memory(summary, HandlerMemoryRegion::VmStack, Some(0), false, true);
    let context_read = has_memory(
        summary,
        HandlerMemoryRegion::NativeContext,
        None,
        true,
        false,
    );
    let context_write = has_memory(
        summary,
        HandlerMemoryRegion::NativeContext,
        None,
        false,
        true,
    );
    let bytecode_read = has_memory(summary, HandlerMemoryRegion::Bytecode, None, true, false);
    let bytecode_write = has_memory(summary, HandlerMemoryRegion::Bytecode, None, false, true);
    let other_read = has_memory(summary, HandlerMemoryRegion::Other, None, true, false);
    let other_write = has_memory(summary, HandlerMemoryRegion::Other, None, false, true);
    let stack_effect_count = summary
        .memory
        .iter()
        .filter(|effect| effect.region == HandlerMemoryRegion::VmStack)
        .count();
    let context_effect_count = summary
        .memory
        .iter()
        .filter(|effect| effect.region == HandlerMemoryRegion::NativeContext)
        .count();
    let bytecode_effect_count = summary
        .memory
        .iter()
        .filter(|effect| effect.region == HandlerMemoryRegion::Bytecode)
        .count();
    let other_effect_count = summary
        .memory
        .iter()
        .filter(|effect| effect.region == HandlerMemoryRegion::Other)
        .count();
    let is_add = summary.arithmetic_operations == ["Add"];
    let is_sub = summary.arithmetic_operations == ["Sub"];
    let is_nor = summary.arithmetic_operations == ["Not", "Not", "And"];
    let is_nand = summary.arithmetic_operations == ["Not", "Not", "Or"];
    let is_unsigned_divide = summary.arithmetic_operations == ["Div"];
    let is_unsigned_multiply = summary.arithmetic_operations == ["Mul"];
    let is_signed_multiply = summary.arithmetic_operations == ["Imul"];
    let is_double_shift_left = summary.arithmetic_operations == ["Shld"];
    let is_double_shift_right = summary.arithmetic_operations == ["Shrd"];
    let context_read_widths = [BitWidth::W8, BitWidth::W16, BitWidth::W32, BitWidth::W64]
        .into_iter()
        .filter(|width| {
            has_memory_width(
                summary,
                HandlerMemoryRegion::NativeContext,
                0,
                width.bytes(),
                true,
                false,
            ) && has_memory_width(
                summary,
                HandlerMemoryRegion::VmStack,
                0,
                width.bytes(),
                false,
                true,
            ) && summary.vsp_delta == i32::try_from(width.bytes()).ok().and_then(i32::checked_neg)
        })
        .collect::<Vec<_>>();
    let context_read_width = unique_width(&context_read_widths);
    let resized_context_reads = [BitWidth::W8, BitWidth::W16, BitWidth::W32]
        .into_iter()
        .flat_map(|input_width| {
            [BitWidth::W16, BitWidth::W32, BitWidth::W64]
                .into_iter()
                .filter(move |output_width| input_width.bits() < output_width.bits())
                .flat_map(move |output_width| {
                    summary.resize_operations.iter().filter_map(move |resize| {
                        (resize.input_width == input_width
                            && resize.output_width.bits() >= output_width.bits()
                            && summary.memory.iter().any(|effect| {
                                effect.rva == resize.rva
                                    && effect.region == HandlerMemoryRegion::NativeContext
                                    && effect.width_bytes == input_width.bytes()
                                    && effect.reads
                                    && !effect.writes
                            })
                            && has_memory_width(
                                summary,
                                HandlerMemoryRegion::VmStack,
                                0,
                                output_width.bytes(),
                                false,
                                true,
                            )
                            && summary.vsp_delta
                                == i32::try_from(output_width.bytes())
                                    .ok()
                                    .and_then(i32::checked_neg))
                        .then_some((input_width, output_width, resize.operation))
                    })
                })
        })
        .collect::<Vec<_>>();
    let context_write_widths = [BitWidth::W8, BitWidth::W16, BitWidth::W32, BitWidth::W64]
        .into_iter()
        .filter(|width| {
            has_memory_width(
                summary,
                HandlerMemoryRegion::VmStack,
                0,
                width.bytes(),
                true,
                false,
            ) && has_memory_width(
                summary,
                HandlerMemoryRegion::NativeContext,
                0,
                width.bytes(),
                false,
                true,
            ) && summary.vsp_delta == i32::try_from(width.bytes()).ok()
        })
        .collect::<Vec<_>>();
    let context_write_width = unique_width(&context_write_widths);
    let resized_context_writes = [BitWidth::W16, BitWidth::W32, BitWidth::W64]
        .into_iter()
        .flat_map(|input_width| {
            [BitWidth::W8, BitWidth::W16, BitWidth::W32]
                .into_iter()
                .filter(move |output_width| output_width.bits() < input_width.bits())
                .filter_map(move |output_width| {
                    (has_memory_width(
                        summary,
                        HandlerMemoryRegion::VmStack,
                        0,
                        input_width.bytes(),
                        true,
                        false,
                    ) && has_memory_width(
                        summary,
                        HandlerMemoryRegion::NativeContext,
                        0,
                        output_width.bytes(),
                        false,
                        true,
                    ) && summary.vsp_delta == i32::try_from(input_width.bytes()).ok())
                    .then_some((input_width, output_width))
                })
        })
        .collect::<Vec<_>>();
    let resized_constant_pushes = operand_width.map_or_else(Vec::new, |input_width| {
        [BitWidth::W8, BitWidth::W16, BitWidth::W32, BitWidth::W64]
            .into_iter()
            .filter(|output_width| input_width.bits() < output_width.bits())
            .flat_map(|output_width| {
                summary.resize_operations.iter().filter_map(move |resize| {
                    (resize.input_width == input_width
                        && resize.output_width.bits() >= output_width.bits()
                        && summary.memory.iter().any(|effect| {
                            effect.rva == resize.rva
                                && effect.region == HandlerMemoryRegion::Bytecode
                                && effect.width_bytes == input_width.bytes()
                                && effect.reads
                                && !effect.writes
                        })
                        && has_memory_width(
                            summary,
                            HandlerMemoryRegion::VmStack,
                            0,
                            output_width.bytes(),
                            false,
                            true,
                        )
                        && summary.vsp_delta
                            == i32::try_from(output_width.bytes())
                                .ok()
                                .and_then(i32::checked_neg))
                    .then_some((input_width, output_width, resize.operation))
                })
            })
            .collect::<Vec<_>>()
    });
    let binary_operation_widths = [BitWidth::W8, BitWidth::W16, BitWidth::W32, BitWidth::W64]
        .into_iter()
        .filter(|width| {
            let width_bytes = width.bytes();
            has_memory_width(
                summary,
                HandlerMemoryRegion::VmStack,
                0,
                width_bytes,
                true,
                false,
            ) && has_memory_width(
                summary,
                HandlerMemoryRegion::VmStack,
                i64::from(width_bytes),
                width_bytes,
                true,
                false,
            ) && has_memory_width(
                summary,
                HandlerMemoryRegion::VmStack,
                8,
                width_bytes,
                false,
                true,
            ) && has_memory_width(
                summary,
                HandlerMemoryRegion::VmStack,
                0,
                BitWidth::W64.bytes(),
                false,
                true,
            ) && summary.vsp_delta
                == i32::try_from(width_bytes)
                    .ok()
                    .and_then(|width| width.checked_sub(8))
        })
        .collect::<Vec<_>>();
    let binary_operation_width = unique_width(&binary_operation_widths);
    let division_operation_widths = [BitWidth::W16, BitWidth::W32]
        .into_iter()
        .filter(|width| {
            let bytes = width.bytes();
            has_memory_width(summary, HandlerMemoryRegion::VmStack, 0, bytes, true, false)
                && has_memory_width(
                    summary,
                    HandlerMemoryRegion::VmStack,
                    i64::from(bytes),
                    bytes,
                    true,
                    false,
                )
                && has_memory_width(
                    summary,
                    HandlerMemoryRegion::VmStack,
                    i64::from(bytes.saturating_mul(2)),
                    bytes,
                    true,
                    false,
                )
                && has_memory_width(summary, HandlerMemoryRegion::VmStack, 8, bytes, false, true)
                && has_memory_width(
                    summary,
                    HandlerMemoryRegion::VmStack,
                    i64::from(8_u32.saturating_add(bytes)),
                    bytes,
                    false,
                    true,
                )
                && has_memory_width(
                    summary,
                    HandlerMemoryRegion::VmStack,
                    0,
                    BitWidth::W64.bytes(),
                    false,
                    true,
                )
                && summary.vsp_delta
                    == i32::try_from(bytes)
                        .ok()
                        .and_then(|bytes| bytes.checked_sub(8))
        })
        .collect::<Vec<_>>();
    let division_operation_width = unique_width(&division_operation_widths);
    let multiplication_operation_widths = [BitWidth::W16, BitWidth::W32, BitWidth::W64]
        .into_iter()
        .filter(|width| {
            let bytes = width.bytes();
            has_memory_width(summary, HandlerMemoryRegion::VmStack, 0, bytes, true, false)
                && has_memory_width(
                    summary,
                    HandlerMemoryRegion::VmStack,
                    i64::from(bytes),
                    bytes,
                    true,
                    false,
                )
                && has_memory_width(summary, HandlerMemoryRegion::VmStack, 8, bytes, false, true)
                && has_memory_width(
                    summary,
                    HandlerMemoryRegion::VmStack,
                    i64::from(8_u32.saturating_add(bytes)),
                    bytes,
                    false,
                    true,
                )
                && has_memory_width(
                    summary,
                    HandlerMemoryRegion::VmStack,
                    0,
                    BitWidth::W64.bytes(),
                    false,
                    true,
                )
                && summary.vsp_delta == Some(-8)
        })
        .collect::<Vec<_>>();
    let multiplication_operation_width = unique_width(&multiplication_operation_widths);
    // A divide handler that does not push native flags onto the VM stack: three
    // same-width operands (high dividend @0, low dividend @bytes, divisor
    // @2*bytes) reach DIV, and its two results are stored at @0 and @bytes while
    // the stack nets one freed slot (vsp_delta == +bytes). The flag of which
    // result is quotient vs remainder is proven at the register level by
    // value-flow analysis, so the width-level shape alone never commits a lift.
    // VMProtect emits this shape at 16/32/64 bits; the layout scales by width.
    let unflagged_divide_widths = [BitWidth::W16, BitWidth::W32, BitWidth::W64]
        .into_iter()
        .filter(|width| {
            let bytes = width.bytes();
            let b = i64::from(bytes);
            [(0, true, false), (b, true, false), (b.saturating_mul(2), true, false)]
                .into_iter()
                .chain([(0, false, true), (b, false, true)])
                .all(|(displacement, reads, writes)| {
                    has_memory_width(
                        summary,
                        HandlerMemoryRegion::VmStack,
                        displacement,
                        bytes,
                        reads,
                        writes,
                    )
                })
                && summary.vsp_delta == i32::try_from(bytes).ok()
                && !summary.captures_native_flags_to_vsp
        })
        .collect::<Vec<_>>();
    let unflagged_divide_width = unique_width(&unflagged_divide_widths);
    let double_shift_widths = [BitWidth::W16, BitWidth::W32, BitWidth::W64]
        .into_iter()
        .filter(|width| {
            let bytes = width.bytes();
            has_memory_width(summary, HandlerMemoryRegion::VmStack, 0, bytes, true, false)
                && has_memory_width(
                    summary,
                    HandlerMemoryRegion::VmStack,
                    i64::from(bytes),
                    bytes,
                    true,
                    false,
                )
                && has_memory_width(
                    summary,
                    HandlerMemoryRegion::VmStack,
                    i64::from(bytes.saturating_mul(2)),
                    1,
                    true,
                    false,
                )
                && has_memory_width(summary, HandlerMemoryRegion::VmStack, 8, bytes, false, true)
                && has_memory_width(
                    summary,
                    HandlerMemoryRegion::VmStack,
                    0,
                    BitWidth::W64.bytes(),
                    false,
                    true,
                )
                && summary.vsp_delta
                    == i32::try_from(bytes)
                        .ok()
                        .and_then(|bytes| bytes.checked_sub(6))
        })
        .collect::<Vec<_>>();
    let double_shift_width = unique_width(&double_shift_widths);
    let narrow_vsp_push_width = match summary.narrow_vsp_pushes.as_slice() {
        [push] => BitWidth::from_bytes(push.width_bytes).filter(|width| {
            has_memory_width(
                summary,
                HandlerMemoryRegion::VmStack,
                0,
                width.bytes(),
                false,
                true,
            ) && summary.vsp_delta == i32::try_from(width.bytes()).ok().and_then(i32::checked_neg)
        }),
        _ => None,
    };
    let narrow_vsp_load_width = match summary.narrow_vsp_loads.as_slice() {
        [load] => BitWidth::from_bytes(load.width_bytes).filter(|width| {
            has_memory_width(
                summary,
                HandlerMemoryRegion::VmStack,
                0,
                width.bytes(),
                true,
                false,
            )
        }),
        _ => None,
    };
    let is_cpuid = summary.cpuid_reads.len() == 1
        && summary.vsp_delta == Some(-8)
        && [(0, true, false), (4, true, false)]
            .into_iter()
            .chain((0..4).map(|slot| (slot * 4, false, true)))
            .all(|(displacement, reads, writes)| {
                has_memory_width(
                    summary,
                    HandlerMemoryRegion::VmStack,
                    displacement,
                    BitWidth::W32.bytes(),
                    reads,
                    writes,
                )
            });
    // Leaf-only CPUID variant: a single leaf dword is read from the stack top
    // (no stack subleaf operand; CPUID's ECX subleaf is whatever the native ECX
    // holds on entry, an environment value), and EAX/EBX/ECX/EDX are written to
    // four dword cells, net VSP delta -12. Distinct from `is_cpuid`'s two-operand
    // (-8) form that reads both a leaf and a subleaf from the stack.
    let is_cpuid_leaf_only = summary.cpuid_reads.len() == 1
        && summary.vsp_delta == Some(-12)
        && has_memory_width(
            summary,
            HandlerMemoryRegion::VmStack,
            0,
            BitWidth::W32.bytes(),
            true,
            false,
        )
        && (0..4).all(|slot| {
            has_memory_width(
                summary,
                HandlerMemoryRegion::VmStack,
                slot * 4,
                BitWidth::W32.bytes(),
                false,
                true,
            )
        });
    let is_pop_flags = summary.native_flag_restores.len() == 1
        && summary.vsp_delta == Some(8)
        && has_memory_width(
            summary,
            HandlerMemoryRegion::VmStack,
            0,
            BitWidth::W64.bytes(),
            true,
            false,
        );
    let resized_binary_operations = summary
        .resize_operations
        .iter()
        .filter(|resize| {
            resize.operation == ResizeOp::ZeroExtend
                && resize.input_width.bits() < resize.output_width.bits()
                && summary.memory.iter().any(|effect| {
                    effect.rva == resize.rva
                        && effect.region == HandlerMemoryRegion::VmStack
                        && effect.displacement == 0
                        && effect.width_bytes == resize.input_width.bytes()
                        && effect.reads
                        && !effect.writes
                })
                && has_memory_width(
                    summary,
                    HandlerMemoryRegion::VmStack,
                    i64::from(resize.output_width.bytes()),
                    resize.input_width.bytes(),
                    true,
                    false,
                )
                && has_memory_width(
                    summary,
                    HandlerMemoryRegion::VmStack,
                    8,
                    resize.output_width.bytes(),
                    false,
                    true,
                )
                && has_memory_width(
                    summary,
                    HandlerMemoryRegion::VmStack,
                    0,
                    BitWidth::W64.bytes(),
                    false,
                    true,
                )
                && summary.vsp_delta
                    == i32::try_from(resize.output_width.bytes())
                        .ok()
                        .and_then(|width| width.checked_sub(8))
        })
        .copied()
        .collect::<Vec<_>>();
    let shift_operation_widths = [BitWidth::W8, BitWidth::W16, BitWidth::W32, BitWidth::W64]
        .into_iter()
        .filter(|width| {
            let width_bytes = width.bytes();
            let Some(stack_delta) = summary.vsp_delta else {
                return false;
            };
            let Ok(count_slot_bytes) = u32::try_from(stack_delta.saturating_add(8)) else {
                return false;
            };
            BitWidth::from_bytes(count_slot_bytes).is_some()
                && has_memory_width(
                    summary,
                    HandlerMemoryRegion::VmStack,
                    0,
                    width_bytes,
                    true,
                    false,
                )
                && summary.memory.iter().any(|effect| {
                    effect.region == HandlerMemoryRegion::VmStack
                        && effect.displacement == i64::from(width_bytes)
                        && effect.reads
                        && effect.width_bytes <= count_slot_bytes
                })
                && has_memory_width(
                    summary,
                    HandlerMemoryRegion::VmStack,
                    8,
                    width_bytes,
                    false,
                    true,
                )
                && has_memory_width(
                    summary,
                    HandlerMemoryRegion::VmStack,
                    0,
                    BitWidth::W64.bytes(),
                    false,
                    true,
                )
        })
        .collect::<Vec<_>>();
    let shift_operation_width = unique_width(&shift_operation_widths);
    let shift_semantic = match summary.arithmetic_operations.as_slice() {
        [operation] if operation == "Shl" => Some(VmHandlerSemantic::ShiftLeft),
        [operation] if operation == "Shr" => Some(VmHandlerSemantic::ShiftRightLogical),
        [operation] if operation == "Sar" => Some(VmHandlerSemantic::ShiftRightArithmetic),
        [operation] if operation == "Rol" => Some(VmHandlerSemantic::RotateLeft),
        [operation] if operation == "Ror" => Some(VmHandlerSemantic::RotateRight),
        _ => None,
    };
    let memory_load_widths = [BitWidth::W8, BitWidth::W16, BitWidth::W32, BitWidth::W64]
        .into_iter()
        .filter(|width| {
            stack_read_0
                && has_memory_width(
                    summary,
                    HandlerMemoryRegion::Other,
                    0,
                    width.bytes(),
                    true,
                    false,
                )
                && has_memory_width(
                    summary,
                    HandlerMemoryRegion::VmStack,
                    0,
                    width.bytes(),
                    false,
                    true,
                )
                && summary.vsp_delta == i32::try_from(8_u32.saturating_sub(width.bytes())).ok()
        })
        .collect::<Vec<_>>();
    let memory_load_width = unique_width(&memory_load_widths);
    let segment_load = summary.memory.iter().any(|effect| {
        effect.region == HandlerMemoryRegion::Other && effect.segment_override.is_some()
    });
    let gs_load = segment_load
        && summary.memory.iter().all(|effect| {
            effect.region != HandlerMemoryRegion::Other
                || effect.segment_override.as_deref() == Some("GS")
        });
    let resized_memory_loads = summary
        .resize_operations
        .iter()
        .filter(|resize| {
            stack_read_0
                && has_memory_width(
                    summary,
                    HandlerMemoryRegion::Other,
                    0,
                    resize.input_width.bytes(),
                    true,
                    false,
                )
                && summary.memory.iter().any(|effect| {
                    effect.rva == resize.rva
                        && effect.region == HandlerMemoryRegion::Other
                        && effect.width_bytes == resize.input_width.bytes()
                        && effect.reads
                        && !effect.writes
                })
                && has_memory_width(
                    summary,
                    HandlerMemoryRegion::VmStack,
                    0,
                    resize.output_width.bytes(),
                    false,
                    true,
                )
                && summary.vsp_delta
                    == i32::try_from(8_u32.saturating_sub(resize.output_width.bytes())).ok()
        })
        .copied()
        .collect::<Vec<_>>();
    let memory_store_widths = [BitWidth::W8, BitWidth::W16, BitWidth::W32, BitWidth::W64]
        .into_iter()
        .filter(|width| {
            has_memory_width(
                summary,
                HandlerMemoryRegion::VmStack,
                0,
                BitWidth::W64.bytes(),
                true,
                false,
            ) && has_memory_width(
                summary,
                HandlerMemoryRegion::VmStack,
                8,
                width.bytes(),
                true,
                false,
            ) && has_memory_width(
                summary,
                HandlerMemoryRegion::Other,
                0,
                width.bytes(),
                false,
                true,
            ) && (summary.vsp_delta == i32::try_from(BitWidth::W64.bytes() + width.bytes()).ok()
                || (*width == BitWidth::W8 && summary.vsp_delta == Some(10)))
        })
        .collect::<Vec<_>>();
    let memory_store_width = unique_width(&memory_store_widths);

    let no_context_effects = !context_read && !context_write;
    let no_external_effects = !other_read && !other_write && !bytecode_write;
    let mut candidates = Vec::new();
    let mut add_candidate = |semantic: VmHandlerSemantic, width: Option<BitWidth>| {
        let candidate = (semantic, width);
        if !candidates.contains(&candidate) {
            candidates.push(candidate);
        }
    };
    if operand_width.is_none()
        && summary.loads_vip_from_vsp
        && summary.resets_key_from_vip
        && summary.reinitializes_handler_base
        && summary.vsp_delta == Some(8)
        && no_context_effects
        && no_external_effects
        && !bytecode_read
        && stack_effect_count == 1
        && context_effect_count == 0
        && bytecode_effect_count == 0
        && other_effect_count == 0
        && has_memory_width(
            summary,
            HandlerMemoryRegion::VmStack,
            0,
            BitWidth::W64.bytes(),
            true,
            false,
        )
    {
        add_candidate(VmHandlerSemantic::UnconditionalBranch, Some(BitWidth::W64));
    }
    if operand_width.is_none()
        && summary.loads_vsp_from_top
        && stack_read_0
        && no_context_effects
        && no_external_effects
        && !bytecode_read
        && stack_effect_count == 1
        && context_effect_count == 0
        && bytecode_effect_count == 0
        && other_effect_count == 0
    {
        add_candidate(VmHandlerSemantic::Pop, Some(BitWidth::W64));
    }
    if operand_width.is_none()
        && summary.timestamp_counter_reads.len() == 1
        && summary.vsp_delta == Some(-8)
        && has_memory_width(
            summary,
            HandlerMemoryRegion::VmStack,
            0,
            BitWidth::W32.bytes(),
            false,
            true,
        )
        && has_memory_width(
            summary,
            HandlerMemoryRegion::VmStack,
            4,
            BitWidth::W32.bytes(),
            false,
            true,
        )
        && no_context_effects
        && no_external_effects
        && !bytecode_read
        && stack_effect_count == 2
        && context_effect_count == 0
        && bytecode_effect_count == 0
        && other_effect_count == 0
    {
        add_candidate(VmHandlerSemantic::ReadTimestampCounter, Some(BitWidth::W64));
    }
    if let (Some(width), Some(delta)) = (operand_width, summary.vsp_delta)
        && delta == -i32::try_from(width.bytes()).unwrap_or(i32::MAX)
        && has_memory_width(
            summary,
            HandlerMemoryRegion::VmStack,
            0,
            width.bytes(),
            false,
            true,
        )
        && no_context_effects
        && no_external_effects
        && stack_effect_count == 1
        && context_effect_count == 0
        && bytecode_effect_count == 1
        && other_effect_count == 0
    {
        add_candidate(VmHandlerSemantic::ConstantPush, Some(width));
    }
    if operand_width.is_some()
        && resized_constant_pushes.len() == 1
        && no_context_effects
        && no_external_effects
        && stack_effect_count == 1
        && context_effect_count == 0
        && bytecode_effect_count == 1
        && other_effect_count == 0
    {
        add_candidate(
            VmHandlerSemantic::ConstantPush,
            Some(resized_constant_pushes[0].1),
        );
    }
    if matches!(operand_width, Some(BitWidth::W8 | BitWidth::W16))
        && context_read_width.is_some()
        && !context_write
        && !other_read
        && !other_write
        && !bytecode_write
        && stack_effect_count == 1
        && context_effect_count == 1
        && bytecode_effect_count == 1
        && other_effect_count == 0
    {
        add_candidate(VmHandlerSemantic::VirtualRegisterRead, context_read_width);
    }
    if matches!(operand_width, Some(BitWidth::W8 | BitWidth::W16))
        && resized_context_reads.len() == 1
        && !context_write
        && !other_read
        && !other_write
        && !bytecode_write
        && stack_effect_count == 1
        && context_effect_count == 1
        && bytecode_effect_count == 1
        && other_effect_count == 0
    {
        add_candidate(
            VmHandlerSemantic::VirtualRegisterRead,
            Some(resized_context_reads[0].1),
        );
    }
    if matches!(operand_width, Some(BitWidth::W8 | BitWidth::W16))
        && context_write_width.is_some()
        && !context_read
        && !other_read
        && !other_write
        && !bytecode_write
        && stack_effect_count == 1
        && context_effect_count == 1
        && bytecode_effect_count == 1
        && other_effect_count == 0
    {
        add_candidate(VmHandlerSemantic::VirtualRegisterWrite, context_write_width);
    }
    if matches!(operand_width, Some(BitWidth::W8 | BitWidth::W16))
        && resized_context_writes.len() == 1
        && !context_read
        && !other_read
        && !other_write
        && !bytecode_write
        && stack_effect_count == 1
        && context_effect_count == 1
        && bytecode_effect_count == 1
        && other_effect_count == 0
    {
        add_candidate(
            VmHandlerSemantic::VirtualRegisterWrite,
            Some(resized_context_writes[0].1),
        );
    }
    if operand_width.is_none()
        && summary.vsp_delta == Some(-8)
        && stack_write_0
        && summary.pushes_pre_adjustment_vsp
        && no_context_effects
        && no_external_effects
        && !bytecode_read
        && stack_effect_count == 1
        && context_effect_count == 0
        && bytecode_effect_count == 0
        && other_effect_count == 0
    {
        add_candidate(VmHandlerSemantic::Push, Some(BitWidth::W64));
    }
    let ordinary_operandless =
        operand_width.is_none() && no_context_effects && !bytecode_read && !bytecode_write;
    if ordinary_operandless
        && !other_read
        && !other_write
        && summary.captures_native_flags_to_vsp
        && binary_operation_width.is_some()
        && is_add
        && stack_effect_count == 4
        && other_effect_count == 0
    {
        add_candidate(VmHandlerSemantic::Add, binary_operation_width);
    }
    if ordinary_operandless
        && !other_read
        && !other_write
        && summary.captures_native_flags_to_vsp
        && binary_operation_width.is_some()
        && is_sub
        && stack_effect_count == 4
        && other_effect_count == 0
    {
        add_candidate(VmHandlerSemantic::Sub, binary_operation_width);
    }
    if ordinary_operandless
        && !other_read
        && !other_write
        && summary.captures_native_flags_to_vsp
        && division_operation_width.is_some()
        && is_unsigned_divide
        && stack_effect_count == 6
        && other_effect_count == 0
    {
        add_candidate(VmHandlerSemantic::UnsignedDivide, division_operation_width);
    }
    if ordinary_operandless
        && !other_read
        && !other_write
        && summary.captures_native_flags_to_vsp
        && multiplication_operation_width.is_some()
        && is_unsigned_multiply
        && stack_effect_count == 5
        && other_effect_count == 0
    {
        add_candidate(VmHandlerSemantic::Mul, multiplication_operation_width);
    }
    if ordinary_operandless
        && !other_read
        && !other_write
        && summary.captures_native_flags_to_vsp
        && multiplication_operation_width.is_some()
        && is_signed_multiply
        && stack_effect_count == 5
        && other_effect_count == 0
    {
        add_candidate(
            VmHandlerSemantic::SignedMultiply,
            multiplication_operation_width,
        );
    }
    if ordinary_operandless
        && !other_read
        && !other_write
        && unflagged_divide_width.is_some()
        && is_unsigned_divide
        && stack_effect_count == 5
        && other_effect_count == 0
    {
        add_candidate(VmHandlerSemantic::UnsignedDivide, unflagged_divide_width);
    }
    if ordinary_operandless
        && !other_read
        && !other_write
        && summary.captures_native_flags_to_vsp
        && double_shift_width.is_some()
        && (is_double_shift_left || is_double_shift_right)
        && stack_effect_count == 5
        && other_effect_count == 0
    {
        add_candidate(
            if is_double_shift_left {
                VmHandlerSemantic::ShiftLeftDouble
            } else {
                VmHandlerSemantic::ShiftRightDouble
            },
            double_shift_width,
        );
    }
    if ordinary_operandless
        && !other_read
        && !other_write
        && !summary.captures_native_flags_to_vsp
        && summary.arithmetic_operations.is_empty()
        && is_cpuid
        && stack_effect_count == 6
        && other_effect_count == 0
    {
        add_candidate(VmHandlerSemantic::Cpuid, Some(BitWidth::W32));
    }
    if ordinary_operandless
        && !other_read
        && !other_write
        && !summary.captures_native_flags_to_vsp
        && summary.arithmetic_operations.is_empty()
        && is_cpuid_leaf_only
        && stack_effect_count == 5
        && other_effect_count == 0
    {
        add_candidate(VmHandlerSemantic::Cpuid, Some(BitWidth::W32));
    }
    if ordinary_operandless
        && !other_read
        && !other_write
        && !summary.captures_native_flags_to_vsp
        && summary.arithmetic_operations.is_empty()
        && is_pop_flags
        && stack_effect_count == 1
        && other_effect_count == 0
    {
        add_candidate(VmHandlerSemantic::PopFlags, Some(BitWidth::W64));
    }
    if ordinary_operandless
        && !other_read
        && !other_write
        && !summary.captures_native_flags_to_vsp
        && summary.arithmetic_operations.is_empty()
        && narrow_vsp_push_width.is_some()
        && stack_effect_count == 1
        && other_effect_count == 0
    {
        add_candidate(VmHandlerSemantic::Push, narrow_vsp_push_width);
    }
    if ordinary_operandless
        && !other_read
        && !other_write
        && !summary.captures_native_flags_to_vsp
        && summary.arithmetic_operations.is_empty()
        && narrow_vsp_load_width.is_some()
        && summary.vsp_delta.is_none()
        && stack_effect_count == 1
        && other_effect_count == 0
    {
        add_candidate(VmHandlerSemantic::Pop, narrow_vsp_load_width);
    }
    let resized_binary_semantic = if is_add {
        Some(VmHandlerSemantic::Add)
    } else if is_sub {
        Some(VmHandlerSemantic::Sub)
    } else if is_nor {
        Some(VmHandlerSemantic::Nor)
    } else if is_nand {
        Some(VmHandlerSemantic::Nand)
    } else {
        shift_semantic
    };
    if ordinary_operandless
        && !other_read
        && !other_write
        && summary.captures_native_flags_to_vsp
        && resized_binary_operations.len() == 1
        && stack_effect_count == 4
        && other_effect_count == 0
        && let Some(semantic) = resized_binary_semantic
    {
        add_candidate(semantic, Some(resized_binary_operations[0].output_width));
    }
    if ordinary_operandless
        && !other_read
        && !other_write
        && summary.captures_native_flags_to_vsp
        && binary_operation_width.is_some()
        && is_nor
        && stack_effect_count == 4
        && other_effect_count == 0
    {
        add_candidate(VmHandlerSemantic::Nor, binary_operation_width);
    }
    if ordinary_operandless
        && !other_read
        && !other_write
        && summary.captures_native_flags_to_vsp
        && binary_operation_width.is_some()
        && is_nand
        && stack_effect_count == 4
        && other_effect_count == 0
    {
        add_candidate(VmHandlerSemantic::Nand, binary_operation_width);
    }
    if ordinary_operandless
        && !other_read
        && !other_write
        && summary.captures_native_flags_to_vsp
        && shift_operation_width.is_some()
        && stack_effect_count == 4
        && other_effect_count == 0
        && let Some(semantic) = shift_semantic
    {
        add_candidate(semantic, shift_operation_width);
    }
    if ordinary_operandless
        && gs_load
        && memory_load_width == Some(BitWidth::W64)
        && !other_write
        && other_read
        && stack_effect_count == 2
        && other_effect_count == 1
    {
        add_candidate(VmHandlerSemantic::GsRelativeLoad, Some(BitWidth::W64));
    }
    if ordinary_operandless
        && !segment_load
        && memory_load_width.is_some()
        && !other_write
        && other_read
        && stack_effect_count == 2
        && other_effect_count == 1
    {
        add_candidate(VmHandlerSemantic::MemoryLoad, memory_load_width);
    }
    if ordinary_operandless
        && !segment_load
        && resized_memory_loads.len() == 1
        && !other_write
        && other_read
        && stack_effect_count == 2
        && other_effect_count == 1
    {
        add_candidate(
            VmHandlerSemantic::MemoryLoad,
            Some(resized_memory_loads[0].output_width),
        );
    }
    if ordinary_operandless
        && !segment_load
        && memory_store_width.is_some()
        && !other_read
        && other_write
        && stack_effect_count == 2
        && other_effect_count == 1
    {
        add_candidate(VmHandlerSemantic::MemoryStore, memory_store_width);
    }
    let (semantic, operation_width) = match candidates.as_slice() {
        [(semantic, width)] => (*semantic, *width),
        _ => (VmHandlerSemantic::Unknown, None),
    };
    let (input_width, resize) = match semantic {
        VmHandlerSemantic::MemoryLoad => match resized_memory_loads.as_slice() {
            [resized] if Some(resized.output_width) == operation_width => {
                (Some(resized.input_width), Some(resized.operation))
            }
            [] => (operation_width, None),
            _ => (None, None),
        },
        VmHandlerSemantic::ConstantPush => match resized_constant_pushes.as_slice() {
            [(input, output, resize)] if Some(*output) == operation_width => {
                (Some(*input), Some(*resize))
            }
            [] => (operation_width, None),
            _ => (None, None),
        },
        VmHandlerSemantic::Add
        | VmHandlerSemantic::Sub
        | VmHandlerSemantic::Nor
        | VmHandlerSemantic::Nand => match resized_binary_operations.as_slice() {
            [resized] if Some(resized.output_width) == operation_width => {
                (Some(resized.input_width), Some(resized.operation))
            }
            _ => (None, None),
        },
        VmHandlerSemantic::ShiftLeft
        | VmHandlerSemantic::ShiftRightLogical
        | VmHandlerSemantic::ShiftRightArithmetic
        | VmHandlerSemantic::RotateLeft
        | VmHandlerSemantic::RotateRight => match resized_binary_operations.as_slice() {
            [resized] if Some(resized.output_width) == operation_width => {
                (Some(resized.input_width), Some(resized.operation))
            }
            _ => (None, None),
        },
        VmHandlerSemantic::VirtualRegisterRead => match resized_context_reads.as_slice() {
            [(input, output, resize)] if Some(*output) == operation_width => {
                (Some(*input), Some(*resize))
            }
            [] => (operation_width, None),
            _ => (None, None),
        },
        VmHandlerSemantic::VirtualRegisterWrite => match resized_context_writes.as_slice() {
            [(input, output)] if Some(*output) == operation_width => {
                (Some(*input), Some(ResizeOp::Truncate))
            }
            [] => (operation_width, None),
            _ => (None, None),
        },
        _ => (None, None),
    };
    let flags_effect = if summary.captures_native_flags_to_vsp
        || matches!(
            semantic,
            VmHandlerSemantic::PopFlags | VmHandlerSemantic::UnsignedDivide
        ) {
        EffectKnowledge::Modified
    } else if semantic == VmHandlerSemantic::Unknown {
        EffectKnowledge::Unknown
    } else {
        EffectKnowledge::Preserved
    };
    let confidence = if semantic == VmHandlerSemantic::Unknown {
        0
    } else {
        95
    };
    let mut evidence = vec![format!("normalized VSP delta is {:?}", summary.vsp_delta)];
    if context_read {
        evidence.push("reads an indexed native-context slot".to_owned());
    }
    if context_write {
        evidence.push("writes an indexed native-context slot".to_owned());
    }
    if summary.captures_native_flags_to_vsp {
        evidence.push("captures native arithmetic flags into the VM stack".to_owned());
    }
    if summary.pushes_pre_adjustment_vsp {
        evidence.push("pushes the pre-adjustment VSP value onto the VM stack".to_owned());
    }
    if summary.loads_vsp_from_top {
        evidence.push("restores VSP from the qword at the top of the VM stack".to_owned());
    }
    if summary.loads_vip_from_vsp {
        evidence.push("loads VIP from the qword popped from the VM stack".to_owned());
    }
    if summary.resets_key_from_vip {
        evidence.push("resets the rolling key from the new VIP".to_owned());
    }
    if summary.reinitializes_handler_base {
        evidence.push("reinitializes the handler base before dispatch".to_owned());
    }
    if other_read {
        evidence.push("dereferences a non-VM-stack pointer".to_owned());
    }
    if other_write {
        evidence.push("writes through a non-VM-stack pointer".to_owned());
    }
    if context_read_widths.len() > 1
        || resized_context_reads.len() > 1
        || context_write_widths.len() > 1
        || binary_operation_widths.len() > 1
        || division_operation_widths.len() > 1
        || unflagged_divide_widths.len() > 1
        || multiplication_operation_widths.len() > 1
        || shift_operation_widths.len() > 1
        || memory_load_widths.len() > 1
        || resized_memory_loads.len() > 1
        || resized_constant_pushes.len() > 1
        || resized_binary_operations.len() > 1
        || resized_context_writes.len() > 1
        || memory_store_widths.len() > 1
        || double_shift_widths.len() > 1
    {
        evidence.push("multiple operation widths match the normalized effects".to_owned());
    }
    if candidates.len() > 1 {
        evidence.push(format!(
            "multiple semantic candidates match the normalized effects: {candidates:?}"
        ));
    }
    evidence.extend(summary.diagnostics.iter().cloned());
    VmHandlerAnalysis {
        entry,
        semantic,
        operand_width,
        input_width,
        operation_width,
        resize,
        stack_delta: summary.vsp_delta,
        vip_delta,
        rolling_key_effect: None,
        flags_effect,
        confidence,
        evidence,
    }
}

#[cfg(test)]
mod tests {
    use iced_x86::{Code, Decoder, DecoderOptions, code_asm::*};
    use vmpx_disasm::{
        DecodedInstruction, DirectJumpFragment, DirectJumpTraceStop, DirectJumpTraceStopKind,
    };
    use vmpx_pe::Va;

    use super::*;
    use crate::{HandlerSliceConfig, backward_slice_handler_trace};

    fn trace_from_assembler(mut assembler: CodeAssembler) -> DirectJumpTrace {
        let bytes = assembler.assemble(0x1_4000_1000).expect("assemble trace");
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
        let end = instructions
            .last()
            .and_then(DecodedInstruction::next_rva)
            .expect("trace end");
        let stop_at = instructions.last().expect("terminal").rva;
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
                at: stop_at,
                direct_target: None,
                fallthrough: None,
                detail: None,
            },
        }
    }

    fn effects(trace: &DirectJumpTrace, semantic_end: Rva) -> HandlerEffectSummary {
        let slice = backward_slice_handler_trace(
            trace,
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
        extract_handler_effects(
            trace,
            &slice,
            HandlerEffectRoles {
                vsp: Register::R11,
                context_base: Register::RSP,
            },
            crate::HandlerRegisterRoles {
                vip: Register::R15,
                rolling_key: Register::RBP,
                handler: Register::R14,
            },
            semantic_end,
        )
    }

    #[test]
    fn extracts_constant_push_effects_before_dispatch() {
        let mut assembler = CodeAssembler::new(64).expect("x64 assembler");
        assembler.mov(rax, qword_ptr(r15)).expect("operand load");
        assembler.sub(r11, 8).expect("VSP reserve");
        assembler.mov(qword_ptr(r11), rax).expect("VM stack store");
        assembler.mov(ecx, dword_ptr(r15)).expect("dispatch load");
        assembler.jmp(r14).expect("dispatch");
        let trace = trace_from_assembler(assembler);
        let boundary = trace.instructions().nth(3).expect("dispatch load").rva;
        let summary = effects(&trace, boundary);
        let analysis =
            classify_handler_effects(trace.entry, &summary, Some(BitWidth::W64), Some(4));

        assert_eq!(summary.vsp_delta, Some(-8));
        assert!(has_memory(
            &summary,
            HandlerMemoryRegion::VmStack,
            Some(0),
            false,
            true
        ));
        assert_eq!(analysis.semantic, VmHandlerSemantic::ConstantPush);
        assert_eq!(analysis.flags_effect, EffectKnowledge::Preserved);
    }

    #[test]
    fn recognizes_word_constant_push_by_operand_width() {
        let mut assembler = CodeAssembler::new(64).expect("x64 assembler");
        assembler.mov(ax, word_ptr(r15)).expect("operand load");
        assembler.sub(r11, 2).expect("VSP reserve");
        assembler.mov(word_ptr(r11), ax).expect("VM stack store");
        assembler.mov(ecx, dword_ptr(r15)).expect("dispatch load");
        assembler.jmp(r14).expect("dispatch");
        let trace = trace_from_assembler(assembler);
        let boundary = trace.instructions().nth(3).expect("dispatch load").rva;
        let summary = effects(&trace, boundary);
        let analysis =
            classify_handler_effects(trace.entry, &summary, Some(BitWidth::W16), Some(4));

        assert_eq!(summary.vsp_delta, Some(-2));
        assert_eq!(analysis.semantic, VmHandlerSemantic::ConstantPush);
    }

    #[test]
    fn recognizes_add_with_vm_flags_effect() {
        let mut assembler = CodeAssembler::new(64).expect("x64 assembler");
        assembler.mov(rax, qword_ptr(r11)).expect("left input");
        assembler.add(rax, qword_ptr(r11 + 8)).expect("add input");
        assembler.mov(qword_ptr(r11 + 8), rax).expect("result");
        assembler.pushfq().expect("capture flags");
        assembler.pop(qword_ptr(r11)).expect("VM flags");
        assembler.mov(ecx, dword_ptr(r15)).expect("dispatch load");
        assembler.jmp(r14).expect("dispatch");
        let trace = trace_from_assembler(assembler);
        let boundary = trace.instructions().nth(5).expect("dispatch load").rva;
        let summary = effects(&trace, boundary);
        let analysis = classify_handler_effects(trace.entry, &summary, None, Some(4));

        assert_eq!(summary.vsp_delta, Some(0));
        assert!(summary.captures_native_flags_to_vsp);
        assert_eq!(
            summary.captured_flags_source,
            trace
                .instructions()
                .nth(1)
                .map(|instruction| instruction.rva)
        );
        assert_eq!(analysis.semantic, VmHandlerSemantic::Add);
        assert_eq!(analysis.flags_effect, EffectKnowledge::Modified);
        assert_eq!(analysis.operation_width, Some(BitWidth::W64));
    }

    #[test]
    fn recognizes_dword_add_stack_layout() {
        let mut assembler = CodeAssembler::new(64).expect("x64 assembler");
        assembler.mov(r10d, dword_ptr(r11)).expect("left input");
        assembler.mov(r8d, dword_ptr(r11 + 4)).expect("right input");
        assembler.sub(r11, 4).expect("make flags cell");
        assembler.add(r10d, r8d).expect("add input");
        assembler.mov(dword_ptr(r11 + 8), r10d).expect("result");
        assembler.pushfq().expect("capture flags");
        assembler.pop(qword_ptr(r11)).expect("VM flags");
        assembler.mov(edx, dword_ptr(r15)).expect("dispatch load");
        assembler.jmp(r14).expect("dispatch");
        let trace = trace_from_assembler(assembler);
        let boundary = trace.instructions().nth(7).expect("dispatch load").rva;
        let summary = effects(&trace, boundary);
        let analysis = classify_handler_effects(trace.entry, &summary, None, Some(4));

        assert_eq!(summary.vsp_delta, Some(-4));
        assert_eq!(analysis.semantic, VmHandlerSemantic::Add);
        assert_eq!(analysis.operation_width, Some(BitWidth::W32));
        assert_eq!(analysis.flags_effect, EffectKnowledge::Modified);
    }

    #[test]
    fn recognizes_byte_add_with_zero_extended_word_stack_slots() {
        let mut assembler = CodeAssembler::new(64).expect("x64 assembler");
        assembler
            .movzx(r9w, byte_ptr(r11))
            .expect("zero-extended left input");
        assembler.mov(r10b, byte_ptr(r11 + 2)).expect("right input");
        assembler.sub(r11, 6).expect("make flags cell");
        assembler.add(r9b, r10b).expect("byte add");
        assembler
            .mov(word_ptr(r11 + 8), r9w)
            .expect("zero-extended result");
        assembler.pushfq().expect("capture flags");
        assembler.pop(qword_ptr(r11)).expect("VM flags");
        assembler.mov(edx, dword_ptr(r15)).expect("dispatch load");
        assembler.jmp(r14).expect("dispatch");
        let trace = trace_from_assembler(assembler);
        let boundary = trace.instructions().nth(7).expect("dispatch load").rva;
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
        let summary = extract_handler_effects(
            &trace,
            &slice,
            HandlerEffectRoles {
                vsp: Register::R11,
                context_base: Register::RSP,
            },
            crate::HandlerRegisterRoles {
                vip: Register::R15,
                rolling_key: Register::RBP,
                handler: Register::R14,
            },
            boundary,
        );
        let analysis = classify_handler_effects(trace.entry, &summary, None, Some(4));
        let proof =
            crate::prove_handler_value_flow(&trace, &slice, &summary, &analysis, None, None, false);

        assert_eq!(summary.vsp_delta, Some(-6));
        assert_eq!(analysis.semantic, VmHandlerSemantic::Add);
        assert_eq!(analysis.input_width, Some(BitWidth::W8));
        assert_eq!(analysis.operation_width, Some(BitWidth::W16));
        assert_eq!(analysis.resize, Some(ResizeOp::ZeroExtend));
        assert_eq!(analysis.flags_effect, EffectKnowledge::Modified);
        assert!(proof.proven, "{:?}", proof.evidence);
    }

    #[test]
    fn recognizes_byte_nand_with_zero_extended_word_stack_slots() {
        let mut assembler = CodeAssembler::new(64).expect("x64 assembler");
        assembler
            .movzx(r9w, byte_ptr(r11))
            .expect("zero-extended left input");
        assembler.mov(r10b, byte_ptr(r11 + 2)).expect("right input");
        assembler.sub(r11, 6).expect("make flags cell");
        assembler.not(r9b).expect("invert left input");
        assembler.not(r10b).expect("invert right input");
        assembler.or(r9b, r10b).expect("byte NAND");
        assembler
            .mov(word_ptr(r11 + 8), r9w)
            .expect("zero-extended result");
        assembler.pushfq().expect("capture flags");
        assembler.pop(qword_ptr(r11)).expect("VM flags");
        assembler.mov(edx, dword_ptr(r15)).expect("dispatch load");
        assembler.jmp(r14).expect("dispatch");
        let trace = trace_from_assembler(assembler);
        let boundary = trace.instructions().nth(9).expect("dispatch load").rva;
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
        let summary = extract_handler_effects(
            &trace,
            &slice,
            HandlerEffectRoles {
                vsp: Register::R11,
                context_base: Register::RSP,
            },
            crate::HandlerRegisterRoles {
                vip: Register::R15,
                rolling_key: Register::RBP,
                handler: Register::R14,
            },
            boundary,
        );
        let analysis = classify_handler_effects(trace.entry, &summary, None, Some(4));
        let proof =
            crate::prove_handler_value_flow(&trace, &slice, &summary, &analysis, None, None, false);

        assert_eq!(summary.vsp_delta, Some(-6));
        assert_eq!(analysis.semantic, VmHandlerSemantic::Nand);
        assert_eq!(analysis.input_width, Some(BitWidth::W8));
        assert_eq!(analysis.operation_width, Some(BitWidth::W16));
        assert_eq!(analysis.resize, Some(ResizeOp::ZeroExtend));
        assert_eq!(analysis.flags_effect, EffectKnowledge::Modified);
        assert!(proof.proven, "{:?}", proof.evidence);
    }

    #[test]
    fn recognizes_timestamp_counter_split_dword_push() {
        let mut assembler = CodeAssembler::new(64).expect("x64 assembler");
        assembler.rdtsc().expect("read timestamp counter");
        assembler.sub(r11, 8).expect("reserve two dwords");
        assembler
            .mov(dword_ptr(r11), edx)
            .expect("store high dword first");
        assembler
            .mov(dword_ptr(r11 + 4), eax)
            .expect("store low dword second");
        assembler.mov(ecx, dword_ptr(r15)).expect("dispatch load");
        assembler.jmp(r14).expect("dispatch");
        let trace = trace_from_assembler(assembler);
        let boundary = trace.instructions().nth(4).expect("dispatch load").rva;
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
        let summary = extract_handler_effects(
            &trace,
            &slice,
            HandlerEffectRoles {
                vsp: Register::R11,
                context_base: Register::RSP,
            },
            crate::HandlerRegisterRoles {
                vip: Register::R15,
                rolling_key: Register::RBP,
                handler: Register::R14,
            },
            boundary,
        );
        let analysis = classify_handler_effects(trace.entry, &summary, None, Some(4));
        let proof =
            crate::prove_handler_value_flow(&trace, &slice, &summary, &analysis, None, None, false);

        assert_eq!(summary.timestamp_counter_reads.len(), 1);
        assert_eq!(analysis.semantic, VmHandlerSemantic::ReadTimestampCounter);
        assert_eq!(analysis.operation_width, Some(BitWidth::W64));
        assert_eq!(analysis.stack_delta, Some(-8));
        assert!(proof.proven, "{:?}", proof.evidence);
    }

    #[test]
    fn recognizes_unsigned_dword_divide_stack_layout() {
        let mut assembler = CodeAssembler::new(64).expect("x64 assembler");
        assembler.mov(edx, dword_ptr(r11)).expect("dividend high");
        assembler
            .mov(eax, dword_ptr(r11 + 4))
            .expect("dividend low");
        assembler.mov(ecx, dword_ptr(r11 + 8)).expect("divisor");
        assembler.sub(r11, 4).expect("make flags cell");
        assembler.div(ecx).expect("unsigned divide");
        assembler.mov(dword_ptr(r11 + 8), edx).expect("remainder");
        assembler.mov(dword_ptr(r11 + 12), eax).expect("quotient");
        assembler.pushfq().expect("capture native flags");
        assembler.pop(qword_ptr(r11)).expect("VM flags");
        assembler.mov(r10d, dword_ptr(r15)).expect("dispatch load");
        assembler.jmp(r14).expect("dispatch");
        let trace = trace_from_assembler(assembler);
        let boundary = trace.instructions().nth(9).expect("dispatch load").rva;
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
        let summary = extract_handler_effects(
            &trace,
            &slice,
            HandlerEffectRoles {
                vsp: Register::R11,
                context_base: Register::RSP,
            },
            crate::HandlerRegisterRoles {
                vip: Register::R15,
                rolling_key: Register::RBP,
                handler: Register::R14,
            },
            boundary,
        );
        let analysis = classify_handler_effects(trace.entry, &summary, None, Some(4));
        let proof =
            crate::prove_handler_value_flow(&trace, &slice, &summary, &analysis, None, None, false);

        assert_eq!(summary.arithmetic_operations, ["Div"]);
        assert_eq!(analysis.semantic, VmHandlerSemantic::UnsignedDivide);
        assert_eq!(analysis.operation_width, Some(BitWidth::W32));
        assert_eq!(analysis.stack_delta, Some(-4));
        assert_eq!(analysis.flags_effect, EffectKnowledge::Modified);
        assert!(proof.proven, "{:?}", proof.evidence);
    }

    /// VMProtect also emits an unsigned divide that does not push native flags:
    /// the two results occupy the lower stack cells and the stack nets one freed
    /// slot (vsp_delta == +bytes). value-flow still proves quotient<-EAX and
    /// remainder<-EDX from the DIV, so the lift is committed only when provably
    /// correct. This is the shape that stalls `rmath_gcd`/`gsieve_perfect_scan`.
    #[test]
    fn recognizes_unflagged_dword_divide_stack_layout() {
        let mut assembler = CodeAssembler::new(64).expect("x64 assembler");
        assembler.mov(edx, dword_ptr(r11)).expect("dividend high");
        assembler.mov(eax, dword_ptr(r11 + 4)).expect("dividend low");
        assembler.mov(ecx, dword_ptr(r11 + 8)).expect("divisor");
        assembler.add(r11, 4).expect("free one slot");
        assembler.div(ecx).expect("unsigned divide");
        assembler.mov(dword_ptr(r11), edx).expect("remainder");
        assembler.mov(dword_ptr(r11 + 4), eax).expect("quotient");
        assembler.mov(r10d, dword_ptr(r15)).expect("dispatch load");
        assembler.jmp(r14).expect("dispatch");
        let trace = trace_from_assembler(assembler);
        let boundary = trace.instructions().nth(7).expect("dispatch load").rva;
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
        let summary = extract_handler_effects(
            &trace,
            &slice,
            HandlerEffectRoles {
                vsp: Register::R11,
                context_base: Register::RSP,
            },
            crate::HandlerRegisterRoles {
                vip: Register::R15,
                rolling_key: Register::RBP,
                handler: Register::R14,
            },
            boundary,
        );
        let analysis = classify_handler_effects(trace.entry, &summary, None, Some(4));
        let proof =
            crate::prove_handler_value_flow(&trace, &slice, &summary, &analysis, None, None, false);

        assert_eq!(summary.arithmetic_operations, ["Div"]);
        assert!(!summary.captures_native_flags_to_vsp);
        assert_eq!(analysis.semantic, VmHandlerSemantic::UnsignedDivide);
        assert_eq!(analysis.operation_width, Some(BitWidth::W32));
        assert_eq!(analysis.stack_delta, Some(4));
        assert_eq!(analysis.flags_effect, EffectKnowledge::Modified);
        assert!(proof.proven, "{:?}", proof.evidence);
    }

    /// Same divide shape, but the handler stores the quotient and remainder in
    /// the opposite cells. The width-level shape still looks like a divide, so
    /// value-flow is what must reject it: it proves the result-register mapping
    /// and finds EAX/EDX in the wrong slots, so the flow is not proven and the
    /// pipeline keeps the handler Unknown (fail closed) rather than lift a swap.
    #[test]
    fn rejects_unflagged_divide_with_swapped_result_cells() {
        let mut assembler = CodeAssembler::new(64).expect("x64 assembler");
        assembler.mov(edx, dword_ptr(r11)).expect("dividend high");
        assembler.mov(eax, dword_ptr(r11 + 4)).expect("dividend low");
        assembler.mov(ecx, dword_ptr(r11 + 8)).expect("divisor");
        assembler.add(r11, 4).expect("free one slot");
        assembler.div(ecx).expect("unsigned divide");
        assembler.mov(dword_ptr(r11), eax).expect("quotient in remainder cell");
        assembler.mov(dword_ptr(r11 + 4), edx).expect("remainder in quotient cell");
        assembler.mov(r10d, dword_ptr(r15)).expect("dispatch load");
        assembler.jmp(r14).expect("dispatch");
        let trace = trace_from_assembler(assembler);
        let boundary = trace.instructions().nth(7).expect("dispatch load").rva;
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
        let summary = extract_handler_effects(
            &trace,
            &slice,
            HandlerEffectRoles {
                vsp: Register::R11,
                context_base: Register::RSP,
            },
            crate::HandlerRegisterRoles {
                vip: Register::R15,
                rolling_key: Register::RBP,
                handler: Register::R14,
            },
            boundary,
        );
        let analysis = classify_handler_effects(trace.entry, &summary, None, Some(4));
        let proof =
            crate::prove_handler_value_flow(&trace, &slice, &summary, &analysis, None, None, false);

        assert_eq!(analysis.semantic, VmHandlerSemantic::UnsignedDivide);
        assert!(
            !proof.proven,
            "swapped quotient/remainder must not prove sound"
        );
    }

    #[test]
    fn recognizes_unsigned_dword_multiply_stack_layout() {
        let mut assembler = CodeAssembler::new(64).expect("x64 assembler");
        assembler.mov(edx, dword_ptr(r11)).expect("multiplier");
        assembler.mov(eax, dword_ptr(r11 + 4)).expect("accumulator");
        assembler.sub(r11, 8).expect("make result and flags cells");
        assembler.mul(edx).expect("unsigned multiply");
        assembler
            .mov(dword_ptr(r11 + 8), edx)
            .expect("high product");
        assembler
            .mov(dword_ptr(r11 + 12), eax)
            .expect("low product");
        assembler.pushfq().expect("capture native flags");
        assembler.pop(qword_ptr(r11)).expect("VM flags");
        assembler.mov(r10d, dword_ptr(r15)).expect("dispatch load");
        assembler.jmp(r14).expect("dispatch");
        let trace = trace_from_assembler(assembler);
        let boundary = trace.instructions().nth(8).expect("dispatch load").rva;
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
        let summary = extract_handler_effects(
            &trace,
            &slice,
            HandlerEffectRoles {
                vsp: Register::R11,
                context_base: Register::RSP,
            },
            crate::HandlerRegisterRoles {
                vip: Register::R15,
                rolling_key: Register::RBP,
                handler: Register::R14,
            },
            boundary,
        );
        let analysis = classify_handler_effects(trace.entry, &summary, None, Some(4));
        let proof =
            crate::prove_handler_value_flow(&trace, &slice, &summary, &analysis, None, None, false);

        assert_eq!(summary.arithmetic_operations, ["Mul"]);
        assert_eq!(analysis.semantic, VmHandlerSemantic::Mul);
        assert_eq!(analysis.operation_width, Some(BitWidth::W32));
        assert_eq!(analysis.stack_delta, Some(-8));
        assert_eq!(analysis.flags_effect, EffectKnowledge::Modified);
        assert!(proof.proven, "{:?}", proof.evidence);
    }

    #[test]
    fn recognizes_push_of_pre_adjustment_vsp() {
        let mut assembler = CodeAssembler::new(64).expect("x64 assembler");
        assembler.mov(r10, r11).expect("copy VSP");
        assembler.sub(r11, 8).expect("VSP reserve");
        assembler.mov(qword_ptr(r11), r10).expect("push old VSP");
        assembler.mov(ecx, dword_ptr(r15)).expect("dispatch load");
        assembler.jmp(r14).expect("dispatch");
        let trace = trace_from_assembler(assembler);
        let boundary = trace.instructions().nth(3).expect("dispatch load").rva;
        let summary = effects(&trace, boundary);
        let analysis = classify_handler_effects(trace.entry, &summary, None, Some(4));

        assert_eq!(summary.vsp_delta, Some(-8));
        assert!(summary.pushes_pre_adjustment_vsp);
        assert_eq!(analysis.semantic, VmHandlerSemantic::Push);
        assert_eq!(analysis.flags_effect, EffectKnowledge::Preserved);
    }

    #[test]
    fn recognizes_qword_memory_store() {
        let mut assembler = CodeAssembler::new(64).expect("x64 assembler");
        assembler.mov(r12, qword_ptr(r11)).expect("destination");
        assembler.mov(rax, qword_ptr(r11 + 8)).expect("value");
        assembler.add(r11, 16).expect("pop inputs");
        assembler.mov(qword_ptr(r12), rax).expect("store value");
        assembler.mov(ecx, dword_ptr(r15)).expect("dispatch load");
        assembler.jmp(r14).expect("dispatch");
        let trace = trace_from_assembler(assembler);
        let boundary = trace.instructions().nth(4).expect("dispatch load").rva;
        let summary = effects(&trace, boundary);
        let analysis = classify_handler_effects(trace.entry, &summary, None, Some(4));

        assert_eq!(summary.vsp_delta, Some(16));
        assert_eq!(analysis.semantic, VmHandlerSemantic::MemoryStore);
        assert_eq!(analysis.flags_effect, EffectKnowledge::Preserved);
    }

    #[test]
    fn recognizes_nor_with_vm_flags_effect() {
        let mut assembler = CodeAssembler::new(64).expect("x64 assembler");
        assembler.mov(rax, qword_ptr(r11)).expect("left input");
        assembler.mov(rcx, qword_ptr(r11 + 8)).expect("right input");
        assembler.not(rax).expect("invert left");
        assembler.not(rcx).expect("invert right");
        assembler.and(rax, rcx).expect("NOR result");
        assembler
            .mov(qword_ptr(r11 + 8), rax)
            .expect("store result");
        assembler.pushfq().expect("capture flags");
        assembler.pop(qword_ptr(r11)).expect("VM flags");
        assembler.mov(edx, dword_ptr(r15)).expect("dispatch load");
        assembler.jmp(r14).expect("dispatch");
        let trace = trace_from_assembler(assembler);
        let boundary = trace.instructions().nth(8).expect("dispatch load").rva;
        let summary = effects(&trace, boundary);
        let analysis = classify_handler_effects(trace.entry, &summary, None, Some(4));

        assert_eq!(summary.vsp_delta, Some(0));
        assert_eq!(summary.arithmetic_operations, ["Not", "Not", "And"]);
        assert_eq!(analysis.semantic, VmHandlerSemantic::Nor);
        assert_eq!(analysis.flags_effect, EffectKnowledge::Modified);
    }

    #[test]
    fn recognizes_restore_vsp_from_top() {
        let mut assembler = CodeAssembler::new(64).expect("x64 assembler");
        assembler.mov(r11, qword_ptr(r11)).expect("restore VSP");
        assembler.mov(ecx, dword_ptr(r15)).expect("dispatch load");
        assembler.jmp(r14).expect("dispatch");
        let trace = trace_from_assembler(assembler);
        let boundary = trace.instructions().nth(1).expect("dispatch load").rva;
        let summary = effects(&trace, boundary);
        let analysis = classify_handler_effects(trace.entry, &summary, None, Some(4));

        assert_eq!(summary.vsp_delta, None);
        assert!(summary.loads_vsp_from_top);
        assert_eq!(analysis.semantic, VmHandlerSemantic::Pop);
        assert_eq!(analysis.flags_effect, EffectKnowledge::Preserved);
    }

    #[test]
    fn recognizes_dword_nand_with_vm_flags_effect() {
        let mut assembler = CodeAssembler::new(64).expect("x64 assembler");
        assembler.mov(r8d, dword_ptr(r11)).expect("left input");
        assembler.mov(esi, dword_ptr(r11 + 4)).expect("right input");
        assembler.sub(r11, 4).expect("make flags cell");
        assembler.not(r8d).expect("invert left");
        assembler.not(esi).expect("invert right");
        assembler.or(r8d, esi).expect("NAND result");
        assembler.mov(dword_ptr(r11 + 8), r8d).expect("result");
        assembler.pushfq().expect("capture flags");
        assembler.pop(qword_ptr(r11)).expect("VM flags");
        assembler.mov(edx, dword_ptr(r15)).expect("dispatch load");
        assembler.jmp(r14).expect("dispatch");
        let trace = trace_from_assembler(assembler);
        let boundary = trace.instructions().nth(9).expect("dispatch load").rva;
        let summary = effects(&trace, boundary);
        let analysis = classify_handler_effects(trace.entry, &summary, None, Some(4));

        assert_eq!(analysis.semantic, VmHandlerSemantic::Nand);
        assert_eq!(analysis.operation_width, Some(BitWidth::W32));
        assert_eq!(analysis.flags_effect, EffectKnowledge::Modified);
    }

    #[test]
    fn recognizes_dword_virtual_register_write() {
        let mut assembler = CodeAssembler::new(64).expect("x64 assembler");
        assembler.mov(r12b, byte_ptr(r15)).expect("slot operand");
        assembler.mov(edi, dword_ptr(r11)).expect("pop value");
        assembler.add(r11, 4).expect("advance VSP");
        assembler
            .mov(dword_ptr(rsp + r12), edi)
            .expect("context write");
        assembler.mov(edx, dword_ptr(r15)).expect("dispatch load");
        assembler.jmp(r14).expect("dispatch");
        let trace = trace_from_assembler(assembler);
        let boundary = trace.instructions().nth(4).expect("dispatch load").rva;
        let summary = effects(&trace, boundary);
        let analysis = classify_handler_effects(trace.entry, &summary, Some(BitWidth::W8), Some(4));

        assert_eq!(analysis.semantic, VmHandlerSemantic::VirtualRegisterWrite);
        assert_eq!(analysis.operation_width, Some(BitWidth::W32));
        assert_eq!(analysis.stack_delta, Some(4));
    }

    #[test]
    fn recognizes_truncated_word_slot_virtual_register_write() {
        let mut assembler = CodeAssembler::new(64).expect("x64 assembler");
        assembler.mov(cx, word_ptr(r11)).expect("pop word slot");
        assembler.add(r11, 2).expect("advance VSP");
        assembler.movzx(eax, byte_ptr(r15)).expect("slot operand");
        assembler
            .mov(byte_ptr(rsp + rax), cl)
            .expect("byte context write");
        assembler.mov(edx, dword_ptr(r15)).expect("dispatch load");
        assembler.jmp(r14).expect("dispatch");
        let trace = trace_from_assembler(assembler);
        let context_store = trace.instructions().nth(3).expect("context store").rva;
        let boundary = trace.instructions().nth(4).expect("dispatch load").rva;
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
        let summary = extract_handler_effects(
            &trace,
            &slice,
            HandlerEffectRoles {
                vsp: Register::R11,
                context_base: Register::RSP,
            },
            crate::HandlerRegisterRoles {
                vip: Register::R15,
                rolling_key: Register::RBP,
                handler: Register::R14,
            },
            boundary,
        );
        let analysis = classify_handler_effects(trace.entry, &summary, Some(BitWidth::W8), Some(5));
        let proof = crate::prove_handler_value_flow(
            &trace,
            &slice,
            &summary,
            &analysis,
            Some(context_store),
            Some(Register::EAX),
            true,
        );

        assert_eq!(analysis.semantic, VmHandlerSemantic::VirtualRegisterWrite);
        assert_eq!(analysis.input_width, Some(BitWidth::W16));
        assert_eq!(analysis.operation_width, Some(BitWidth::W8));
        assert_eq!(analysis.resize, Some(ResizeOp::Truncate));
        assert_eq!(analysis.stack_delta, Some(2));
        assert!(proof.proven, "{:?}", proof.evidence);
    }

    #[test]
    fn recognizes_qword_logical_right_shift() {
        let mut assembler = CodeAssembler::new(64).expect("x64 assembler");
        assembler.mov(rsi, qword_ptr(r11)).expect("value");
        assembler.mov(cl, byte_ptr(r11 + 8)).expect("count");
        assembler.sub(r11, 6).expect("make result and flags cells");
        assembler.shr(rsi, cl).expect("logical shift");
        assembler.mov(qword_ptr(r11 + 8), rsi).expect("result");
        assembler.pushfq().expect("capture flags");
        assembler.pop(qword_ptr(r11)).expect("VM flags");
        assembler.mov(edx, dword_ptr(r15)).expect("dispatch load");
        assembler.jmp(r14).expect("dispatch");
        let trace = trace_from_assembler(assembler);
        let boundary = trace.instructions().nth(7).expect("dispatch load").rva;
        let summary = effects(&trace, boundary);
        let analysis = classify_handler_effects(trace.entry, &summary, None, Some(4));

        assert_eq!(analysis.semantic, VmHandlerSemantic::ShiftRightLogical);
        assert_eq!(analysis.operation_width, Some(BitWidth::W64));
        assert_eq!(analysis.stack_delta, Some(-6));
        assert_eq!(analysis.flags_effect, EffectKnowledge::Modified);
    }

    #[test]
    fn recognizes_virtual_unconditional_branch_state_reset() {
        let mut assembler = CodeAssembler::new(64).expect("x64 assembler");
        assembler.mov(r10, qword_ptr(r11)).expect("pop target VIP");
        assembler.add(r11, 8).expect("advance VSP");
        assembler.mov(r15, r10).expect("set VIP");
        assembler.mov(rbp, r15).expect("reset key");
        assembler
            .lea(r14, qword_ptr(0x1234_u64))
            .expect("reset handler base");
        assembler.mov(edx, dword_ptr(r15)).expect("dispatch load");
        assembler.jmp(r14).expect("dispatch");
        let trace = trace_from_assembler(assembler);
        let boundary = trace.instructions().nth(5).expect("dispatch load").rva;
        let summary = effects(&trace, boundary);
        let analysis = classify_handler_effects(trace.entry, &summary, None, Some(4));

        assert!(summary.loads_vip_from_vsp);
        assert!(summary.resets_key_from_vip);
        assert!(summary.reinitializes_handler_base);
        assert_eq!(analysis.semantic, VmHandlerSemantic::UnconditionalBranch);
        assert_eq!(analysis.operation_width, Some(BitWidth::W64));
        assert_eq!(analysis.stack_delta, Some(8));
    }

    #[test]
    fn rejects_branch_when_vip_is_overwritten_after_pop() {
        let mut assembler = CodeAssembler::new(64).expect("x64 assembler");
        assembler.mov(r10, qword_ptr(r11)).expect("pop target VIP");
        assembler.add(r11, 8).expect("advance VSP");
        assembler.mov(r15, r10).expect("set VIP");
        assembler.mov(rbp, r15).expect("reset key");
        assembler.xor(r15, r15).expect("overwrite VIP");
        assembler
            .lea(r14, qword_ptr(0x1234_u64))
            .expect("reset handler base");
        assembler.mov(edx, dword_ptr(r15)).expect("dispatch load");
        assembler.jmp(r14).expect("dispatch");
        let trace = trace_from_assembler(assembler);
        let boundary = trace.instructions().nth(6).expect("dispatch load").rva;
        let summary = effects(&trace, boundary);
        let analysis = classify_handler_effects(trace.entry, &summary, None, Some(4));

        assert!(!summary.loads_vip_from_vsp);
        assert_eq!(analysis.semantic, VmHandlerSemantic::Unknown);
    }

    #[test]
    fn recognizes_descending_branch_with_vip_predecrement() {
        // VMProtect's descending bytecode predecrements the VIP (`sub vip,4`)
        // between popping the branch target and the dispatch read. The popped
        // value's provenance must survive that constant self-offset so the
        // handler still classifies as an unconditional branch.
        let mut assembler = CodeAssembler::new(64).expect("x64 assembler");
        assembler.mov(r10, qword_ptr(r11)).expect("pop target VIP");
        assembler.add(r11, 8).expect("advance VSP");
        assembler.mov(r15, r10).expect("set VIP");
        assembler.mov(rbp, r15).expect("reset key");
        assembler.sub(r15, 4).expect("descending VIP predecrement");
        assembler
            .lea(r14, qword_ptr(0x1234_u64))
            .expect("reset handler base");
        assembler.mov(edx, dword_ptr(r15)).expect("dispatch load");
        assembler.jmp(r14).expect("dispatch");
        let trace = trace_from_assembler(assembler);
        let boundary = trace.instructions().nth(6).expect("dispatch load").rva;
        let summary = effects(&trace, boundary);
        let analysis = classify_handler_effects(trace.entry, &summary, None, Some(4));

        assert!(summary.loads_vip_from_vsp);
        assert!(summary.resets_key_from_vip);
        assert!(summary.reinitializes_handler_base);
        assert_eq!(analysis.semantic, VmHandlerSemantic::UnconditionalBranch);
        assert_eq!(analysis.stack_delta, Some(8));
    }

    #[test]
    fn descending_offset_preservation_does_not_survive_variable_write() {
        // The provenance is kept only across a *constant* self-offset. A write
        // that mixes in another register's value (not a zero register) drops it,
        // so the handler fails closed to Unknown rather than a guessed branch.
        let mut assembler = CodeAssembler::new(64).expect("x64 assembler");
        assembler.mov(r10, qword_ptr(r11)).expect("pop target VIP");
        assembler.add(r11, 8).expect("advance VSP");
        assembler.mov(r15, r10).expect("set VIP");
        assembler.mov(rbp, r15).expect("reset key");
        assembler.add(r15, rax).expect("variable write to VIP");
        assembler
            .lea(r14, qword_ptr(0x1234_u64))
            .expect("reset handler base");
        assembler.mov(edx, dword_ptr(r15)).expect("dispatch load");
        assembler.jmp(r14).expect("dispatch");
        let trace = trace_from_assembler(assembler);
        let boundary = trace.instructions().nth(6).expect("dispatch load").rva;
        let summary = effects(&trace, boundary);
        let analysis = classify_handler_effects(trace.entry, &summary, None, Some(4));

        assert!(!summary.loads_vip_from_vsp);
        assert_eq!(analysis.semantic, VmHandlerSemantic::Unknown);
    }

    #[test]
    fn rejects_partial_vsp_segmented_stack_and_stale_pop_alias() {
        let mut assembler = CodeAssembler::new(64).expect("x64 assembler");
        assembler.add(r11d, 8).expect("partial VSP write");
        assembler
            .mov(qword_ptr(r11).fs(), rax)
            .expect("segmented stack-like store");
        assembler.mov(r11, qword_ptr(r11)).expect("load VSP top");
        assembler.xor(r11, r11).expect("overwrite loaded value");
        assembler.mov(edx, dword_ptr(r15)).expect("dispatch load");
        assembler.jmp(r14).expect("dispatch");
        let trace = trace_from_assembler(assembler);
        let boundary = trace.instructions().nth(4).expect("dispatch load").rva;
        let segmented_store = trace.instructions().nth(1).expect("segmented store").rva;
        let summary = effects(&trace, boundary);

        assert_eq!(summary.vsp_delta, None);
        assert!(
            summary
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.contains("full 64-bit"))
        );
        assert!(summary.memory.iter().any(|effect| {
            effect.rva == segmented_store && effect.region == HandlerMemoryRegion::Other
        }));
        assert!(!summary.loads_vsp_from_top);
    }

    #[test]
    fn rejects_constant_store_before_vsp_reservation() {
        let mut assembler = CodeAssembler::new(64).expect("x64 assembler");
        assembler.mov(rax, qword_ptr(r15)).expect("operand load");
        assembler
            .mov(qword_ptr(r11), rax)
            .expect("premature stack store");
        assembler.sub(r11, 8).expect("late VSP reservation");
        assembler.mov(ecx, dword_ptr(r15)).expect("dispatch load");
        assembler.jmp(r14).expect("dispatch");
        let trace = trace_from_assembler(assembler);
        let boundary = trace.instructions().nth(3).expect("dispatch load").rva;
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
        let summary = extract_handler_effects(
            &trace,
            &slice,
            HandlerEffectRoles {
                vsp: Register::R11,
                context_base: Register::RSP,
            },
            crate::HandlerRegisterRoles {
                vip: Register::R15,
                rolling_key: Register::RBP,
                handler: Register::R14,
            },
            boundary,
        );
        let analysis =
            classify_handler_effects(trace.entry, &summary, Some(BitWidth::W64), Some(4));
        let store = trace.instructions().nth(1).expect("stack store").rva;
        let proof = crate::prove_handler_value_flow(
            &trace,
            &slice,
            &summary,
            &analysis,
            Some(store),
            Some(Register::RAX),
            true,
        );

        assert_eq!(analysis.semantic, VmHandlerSemantic::ConstantPush);
        assert!(!proof.proven);
    }

    #[test]
    fn rejects_dynamic_lea_as_handler_base_reset() {
        let mut assembler = CodeAssembler::new(64).expect("x64 assembler");
        assembler.mov(r10, qword_ptr(r11)).expect("pop target VIP");
        assembler.add(r11, 8).expect("advance VSP");
        assembler.mov(r15, r10).expect("set VIP");
        assembler.mov(rbp, r15).expect("reset key");
        assembler
            .lea(r14, qword_ptr(rsp))
            .expect("dynamic handler value");
        assembler.mov(edx, dword_ptr(r15)).expect("dispatch load");
        assembler.jmp(r14).expect("dispatch");
        let trace = trace_from_assembler(assembler);
        let boundary = trace.instructions().nth(5).expect("dispatch load").rva;
        let summary = effects(&trace, boundary);
        let analysis = classify_handler_effects(trace.entry, &summary, None, Some(4));

        assert!(!summary.reinitializes_handler_base);
        assert_eq!(analysis.semantic, VmHandlerSemantic::Unknown);
    }

    #[test]
    fn unrelated_pushfq_does_not_prove_vm_flag_capture() {
        let mut assembler = CodeAssembler::new(64).expect("x64 assembler");
        assembler.pushfq().expect("save unrelated flags");
        assembler.pop(rax).expect("consume flags elsewhere");
        assembler
            .mov(qword_ptr(r11), rbx)
            .expect("unrelated VM write");
        assembler.mov(edx, dword_ptr(r15)).expect("dispatch load");
        assembler.jmp(r14).expect("dispatch");
        let trace = trace_from_assembler(assembler);
        let boundary = trace.instructions().nth(3).expect("dispatch load").rva;
        let summary = effects(&trace, boundary);

        assert!(!summary.captures_native_flags_to_vsp);
        assert_eq!(summary.captured_flags_source, None);
    }

    #[test]
    fn rejects_flags_captured_before_the_classified_arithmetic() {
        let mut assembler = CodeAssembler::new(64).expect("x64 assembler");
        assembler.mov(rax, qword_ptr(r11)).expect("left input");
        assembler.cmp(rcx, rdx).expect("unrelated flag producer");
        assembler.pushfq().expect("capture pre-add flags");
        assembler.add(rax, qword_ptr(r11 + 8)).expect("add input");
        assembler.mov(qword_ptr(r11 + 8), rax).expect("result");
        assembler.pop(qword_ptr(r11)).expect("stale VM flags");
        assembler.mov(edx, dword_ptr(r15)).expect("dispatch load");
        assembler.jmp(r14).expect("dispatch");
        let trace = trace_from_assembler(assembler);
        let boundary = trace.instructions().nth(6).expect("dispatch load").rva;
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
        let summary = extract_handler_effects(
            &trace,
            &slice,
            HandlerEffectRoles {
                vsp: Register::R11,
                context_base: Register::RSP,
            },
            crate::HandlerRegisterRoles {
                vip: Register::R15,
                rolling_key: Register::RBP,
                handler: Register::R14,
            },
            boundary,
        );
        let analysis = classify_handler_effects(trace.entry, &summary, None, Some(4));
        let proof =
            crate::prove_handler_value_flow(&trace, &slice, &summary, &analysis, None, None, false);

        assert_eq!(analysis.semantic, VmHandlerSemantic::Add);
        assert!(!proof.proven);
    }

    #[test]
    fn rejects_extra_active_arithmetic_as_add() {
        let mut assembler = CodeAssembler::new(64).expect("x64 assembler");
        assembler.mov(rax, qword_ptr(r11)).expect("left input");
        assembler.add(rax, qword_ptr(r11 + 8)).expect("add input");
        assembler.xor(rax, rdx).expect("extra active operation");
        assembler.mov(qword_ptr(r11 + 8), rax).expect("result");
        assembler.pushfq().expect("capture flags");
        assembler.pop(qword_ptr(r11)).expect("VM flags");
        assembler.mov(ecx, dword_ptr(r15)).expect("dispatch load");
        assembler.jmp(r14).expect("dispatch");
        let trace = trace_from_assembler(assembler);
        let boundary = trace.instructions().nth(6).expect("dispatch load").rva;
        let summary = effects(&trace, boundary);
        let analysis = classify_handler_effects(trace.entry, &summary, None, Some(4));

        assert_eq!(summary.arithmetic_operations, ["Add", "Xor"]);
        assert_eq!(analysis.semantic, VmHandlerSemantic::Unknown);
    }

    #[test]
    fn rejects_native_operation_width_mismatch_in_value_flow() {
        let mut assembler = CodeAssembler::new(64).expect("x64 assembler");
        assembler.mov(eax, dword_ptr(r11)).expect("left input");
        assembler.mov(ebx, dword_ptr(r11 + 4)).expect("right input");
        assembler.sub(r11, 4).expect("make flags cell");
        assembler.add(rax, rbx).expect("wrong-width add");
        assembler.mov(dword_ptr(r11 + 8), eax).expect("result");
        assembler.pushfq().expect("capture flags");
        assembler.pop(qword_ptr(r11)).expect("VM flags");
        assembler.mov(ecx, dword_ptr(r15)).expect("dispatch load");
        assembler.jmp(r14).expect("dispatch");
        let trace = trace_from_assembler(assembler);
        let boundary = trace.instructions().nth(7).expect("dispatch load").rva;
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
        let summary = extract_handler_effects(
            &trace,
            &slice,
            HandlerEffectRoles {
                vsp: Register::R11,
                context_base: Register::RSP,
            },
            crate::HandlerRegisterRoles {
                vip: Register::R15,
                rolling_key: Register::RBP,
                handler: Register::R14,
            },
            boundary,
        );
        let analysis = classify_handler_effects(trace.entry, &summary, None, Some(4));
        let proof =
            crate::prove_handler_value_flow(&trace, &slice, &summary, &analysis, None, None, false);

        assert_eq!(analysis.semantic, VmHandlerSemantic::Add);
        assert_eq!(analysis.operation_width, Some(BitWidth::W32));
        assert!(!proof.proven);
    }

    #[test]
    fn rejects_entry_shifted_direct_arithmetic_input() {
        let mut assembler = CodeAssembler::new(64).expect("x64 assembler");
        assembler.mov(rax, qword_ptr(r11)).expect("left input");
        assembler
            .lea(r11, qword_ptr(r11 + 8))
            .expect("temporarily move VSP");
        assembler
            .add(rax, qword_ptr(r11 + 8))
            .expect("wrong entry-relative right input");
        assembler.lea(r11, qword_ptr(r11 - 8)).expect("restore VSP");
        assembler.mov(qword_ptr(r11 + 8), rax).expect("result");
        assembler.pushfq().expect("capture flags");
        assembler.pop(qword_ptr(r11)).expect("VM flags");
        assembler.mov(ecx, dword_ptr(r15)).expect("dispatch load");
        assembler.jmp(r14).expect("dispatch");
        let trace = trace_from_assembler(assembler);
        let boundary = trace.instructions().nth(7).expect("dispatch load").rva;
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
        let summary = extract_handler_effects(
            &trace,
            &slice,
            HandlerEffectRoles {
                vsp: Register::R11,
                context_base: Register::RSP,
            },
            crate::HandlerRegisterRoles {
                vip: Register::R15,
                rolling_key: Register::RBP,
                handler: Register::R14,
            },
            boundary,
        );
        let analysis = classify_handler_effects(trace.entry, &summary, None, Some(4));
        let proof =
            crate::prove_handler_value_flow(&trace, &slice, &summary, &analysis, None, None, false);

        assert_eq!(analysis.semantic, VmHandlerSemantic::Add);
        assert!(!proof.proven);
    }

    #[test]
    fn rejects_entry_shifted_shift_count() {
        let mut assembler = CodeAssembler::new(64).expect("x64 assembler");
        assembler.mov(rsi, qword_ptr(r11)).expect("value");
        assembler.add(r11, 8).expect("temporarily move VSP");
        assembler
            .mov(cl, byte_ptr(r11 + 8))
            .expect("wrong entry-relative count");
        assembler.sub(r11, 8).expect("restore VSP");
        assembler.sub(r11, 6).expect("make result and flags cells");
        assembler.shr(rsi, cl).expect("logical shift");
        assembler.mov(qword_ptr(r11 + 8), rsi).expect("result");
        assembler.pushfq().expect("capture flags");
        assembler.pop(qword_ptr(r11)).expect("VM flags");
        assembler.mov(edx, dword_ptr(r15)).expect("dispatch load");
        assembler.jmp(r14).expect("dispatch");
        let trace = trace_from_assembler(assembler);
        let boundary = trace.instructions().nth(9).expect("dispatch load").rva;
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
        let summary = extract_handler_effects(
            &trace,
            &slice,
            HandlerEffectRoles {
                vsp: Register::R11,
                context_base: Register::RSP,
            },
            crate::HandlerRegisterRoles {
                vip: Register::R15,
                rolling_key: Register::RBP,
                handler: Register::R14,
            },
            boundary,
        );
        let analysis = classify_handler_effects(trace.entry, &summary, None, Some(4));
        let proof =
            crate::prove_handler_value_flow(&trace, &slice, &summary, &analysis, None, None, false);

        assert_eq!(analysis.semantic, VmHandlerSemantic::ShiftRightLogical);
        assert!(!proof.proven);
    }

    #[test]
    fn partial_zero_does_not_preserve_popped_vsp_alias() {
        let mut assembler = CodeAssembler::new(64).expect("x64 assembler");
        assembler.mov(rbx, qword_ptr(r11)).expect("load VSP top");
        assembler.mov(al, 0).expect("partial zero");
        assembler
            .add(rbx, rax)
            .expect("unknown full-width adjustment");
        assembler.mov(r11, rbx).expect("replace VSP");
        assembler.mov(edx, dword_ptr(r15)).expect("dispatch load");
        assembler.jmp(r14).expect("dispatch");
        let trace = trace_from_assembler(assembler);
        let boundary = trace.instructions().nth(4).expect("dispatch load").rva;
        let summary = effects(&trace, boundary);
        let analysis = classify_handler_effects(trace.entry, &summary, None, Some(4));

        assert!(!summary.loads_vsp_from_top);
        assert_eq!(analysis.semantic, VmHandlerSemantic::Unknown);
    }

    #[test]
    fn recognizes_zero_extending_byte_memory_load_to_word_stack_value() {
        let mut assembler = CodeAssembler::new(64).expect("x64 assembler");
        assembler.mov(rax, qword_ptr(r11)).expect("pointer input");
        assembler
            .movzx(dx, byte_ptr(rax))
            .expect("zero-extending external load");
        assembler.add(r11, 6).expect("replace pointer with word");
        assembler.mov(word_ptr(r11), dx).expect("word result");
        assembler.mov(ecx, dword_ptr(r15)).expect("dispatch load");
        assembler.jmp(r14).expect("dispatch");
        let trace = trace_from_assembler(assembler);
        let boundary = trace.instructions().nth(4).expect("dispatch load").rva;
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
        let summary = extract_handler_effects(
            &trace,
            &slice,
            HandlerEffectRoles {
                vsp: Register::R11,
                context_base: Register::RSP,
            },
            crate::HandlerRegisterRoles {
                vip: Register::R15,
                rolling_key: Register::RBP,
                handler: Register::R14,
            },
            boundary,
        );
        let analysis = classify_handler_effects(trace.entry, &summary, None, Some(4));
        let proof =
            crate::prove_handler_value_flow(&trace, &slice, &summary, &analysis, None, None, false);

        assert_eq!(analysis.semantic, VmHandlerSemantic::MemoryLoad);
        assert_eq!(analysis.input_width, Some(BitWidth::W8));
        assert_eq!(analysis.operation_width, Some(BitWidth::W16));
        assert_eq!(analysis.resize, Some(ResizeOp::ZeroExtend));
        assert!(proof.proven, "{:?}", proof.evidence);
    }

    #[test]
    fn recognizes_zero_extending_byte_context_read_to_word_stack_value() {
        let mut assembler = CodeAssembler::new(64).expect("x64 assembler");
        assembler
            .movzx(ecx, byte_ptr(r15))
            .expect("decoded context slot");
        assembler
            .movzx(dx, byte_ptr(rsp + rcx))
            .expect("zero-extended context read");
        assembler.sub(r11, 2).expect("reserve word stack slot");
        assembler.mov(word_ptr(r11), dx).expect("store word result");
        assembler.mov(edx, dword_ptr(r15)).expect("dispatch load");
        assembler.jmp(r14).expect("dispatch");
        let trace = trace_from_assembler(assembler);
        let boundary = trace.instructions().nth(4).expect("dispatch load").rva;
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
        let summary = extract_handler_effects(
            &trace,
            &slice,
            HandlerEffectRoles {
                vsp: Register::R11,
                context_base: Register::RSP,
            },
            crate::HandlerRegisterRoles {
                vip: Register::R15,
                rolling_key: Register::RBP,
                handler: Register::R14,
            },
            boundary,
        );
        let analysis = classify_handler_effects(trace.entry, &summary, Some(BitWidth::W8), Some(1));
        let context_load = trace
            .instructions()
            .nth(1)
            .map(|instruction| instruction.rva);
        let proof = crate::prove_handler_value_flow(
            &trace,
            &slice,
            &summary,
            &analysis,
            context_load,
            Some(Register::ECX),
            true,
        );

        assert_eq!(analysis.semantic, VmHandlerSemantic::VirtualRegisterRead);
        assert_eq!(analysis.input_width, Some(BitWidth::W8));
        assert_eq!(analysis.operation_width, Some(BitWidth::W16));
        assert_eq!(analysis.resize, Some(ResizeOp::ZeroExtend));
        assert_eq!(analysis.stack_delta, Some(-2));
        assert!(proof.proven, "{:?}", proof.evidence);
    }

    #[test]
    fn recognizes_zero_extending_byte_constant_push_to_word_stack_value() {
        let mut assembler = CodeAssembler::new(64).expect("x64 assembler");
        assembler
            .movzx(eax, byte_ptr(r15))
            .expect("zero-extending bytecode load");
        assembler.sub(r11, 2).expect("reserve word stack slot");
        assembler.mov(word_ptr(r11), ax).expect("word result");
        assembler.mov(ecx, dword_ptr(r15)).expect("dispatch load");
        assembler.jmp(r14).expect("dispatch");
        let trace = trace_from_assembler(assembler);
        let store = trace.instructions().nth(2).expect("stack store").rva;
        let boundary = trace.instructions().nth(3).expect("dispatch load").rva;
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
        let summary = extract_handler_effects(
            &trace,
            &slice,
            HandlerEffectRoles {
                vsp: Register::R11,
                context_base: Register::RSP,
            },
            crate::HandlerRegisterRoles {
                vip: Register::R15,
                rolling_key: Register::RBP,
                handler: Register::R14,
            },
            boundary,
        );
        let analysis = classify_handler_effects(trace.entry, &summary, Some(BitWidth::W8), Some(5));
        let proof = crate::prove_handler_value_flow(
            &trace,
            &slice,
            &summary,
            &analysis,
            Some(store),
            Some(Register::EAX),
            true,
        );

        assert_eq!(analysis.semantic, VmHandlerSemantic::ConstantPush);
        assert_eq!(analysis.input_width, Some(BitWidth::W8));
        assert_eq!(analysis.operation_width, Some(BitWidth::W16));
        assert_eq!(analysis.resize, Some(ResizeOp::ZeroExtend));
        assert!(proof.proven, "{:?}", proof.evidence);
    }

    #[test]
    fn rejects_indexed_external_address_in_memory_load_flow() {
        let mut assembler = CodeAssembler::new(64).expect("x64 assembler");
        assembler.mov(rax, qword_ptr(r11)).expect("pointer input");
        assembler.mov(ecx, 1).expect("unproven address index");
        assembler
            .mov(rdx, qword_ptr(rax + rcx))
            .expect("indexed external load");
        assembler.mov(qword_ptr(r11), rdx).expect("result");
        assembler.mov(ecx, dword_ptr(r15)).expect("dispatch load");
        assembler.jmp(r14).expect("dispatch");
        let trace = trace_from_assembler(assembler);
        let boundary = trace.instructions().nth(4).expect("dispatch load").rva;
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
        let summary = extract_handler_effects(
            &trace,
            &slice,
            HandlerEffectRoles {
                vsp: Register::R11,
                context_base: Register::RSP,
            },
            crate::HandlerRegisterRoles {
                vip: Register::R15,
                rolling_key: Register::RBP,
                handler: Register::R14,
            },
            boundary,
        );
        let analysis = classify_handler_effects(trace.entry, &summary, None, Some(4));
        let proof =
            crate::prove_handler_value_flow(&trace, &slice, &summary, &analysis, None, None, false);

        assert_eq!(analysis.semantic, VmHandlerSemantic::MemoryLoad);
        assert!(!proof.proven);
    }

    #[test]
    fn rejects_conditional_move_as_pure_external_load() {
        let mut assembler = CodeAssembler::new(64).expect("x64 assembler");
        assembler.mov(rax, qword_ptr(r11)).expect("pointer input");
        assembler
            .cmovz(rdx, qword_ptr(rax))
            .expect("conditional external load");
        assembler.mov(qword_ptr(r11), rdx).expect("result");
        assembler.mov(ecx, dword_ptr(r15)).expect("dispatch load");
        assembler.jmp(r14).expect("dispatch");
        let trace = trace_from_assembler(assembler);
        let boundary = trace.instructions().nth(3).expect("dispatch load").rva;
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
        let summary = extract_handler_effects(
            &trace,
            &slice,
            HandlerEffectRoles {
                vsp: Register::R11,
                context_base: Register::RSP,
            },
            crate::HandlerRegisterRoles {
                vip: Register::R15,
                rolling_key: Register::RBP,
                handler: Register::R14,
            },
            boundary,
        );
        let analysis = classify_handler_effects(trace.entry, &summary, None, Some(4));
        let proof =
            crate::prove_handler_value_flow(&trace, &slice, &summary, &analysis, None, None, false);

        assert_eq!(analysis.semantic, VmHandlerSemantic::MemoryLoad);
        assert!(!proof.proven);
    }

    #[test]
    fn rejects_modified_native_context_base() {
        let mut assembler = CodeAssembler::new(64).expect("x64 assembler");
        assembler
            .movzx(ecx, byte_ptr(r15))
            .expect("decoded context slot");
        assembler
            .lea(rsp, qword_ptr(rsp + 8))
            .expect("move context base");
        assembler
            .mov(rax, qword_ptr(rsp + rcx))
            .expect("context read through moved base");
        assembler.sub(r11, 8).expect("reserve VM stack");
        assembler.mov(qword_ptr(r11), rax).expect("stack result");
        assembler.mov(edx, dword_ptr(r15)).expect("dispatch load");
        assembler.jmp(r14).expect("dispatch");
        let trace = trace_from_assembler(assembler);
        let boundary = trace.instructions().nth(5).expect("dispatch load").rva;
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
        let summary = extract_handler_effects(
            &trace,
            &slice,
            HandlerEffectRoles {
                vsp: Register::R11,
                context_base: Register::RSP,
            },
            crate::HandlerRegisterRoles {
                vip: Register::R15,
                rolling_key: Register::RBP,
                handler: Register::R14,
            },
            boundary,
        );
        let analysis = classify_handler_effects(trace.entry, &summary, Some(BitWidth::W8), Some(5));
        let consumer = trace
            .instructions()
            .nth(2)
            .map(|instruction| instruction.rva);
        let proof = crate::prove_handler_value_flow(
            &trace,
            &slice,
            &summary,
            &analysis,
            consumer,
            Some(Register::ECX),
            true,
        );

        assert_eq!(analysis.semantic, VmHandlerSemantic::VirtualRegisterRead);
        assert!(!proof.proven);
    }

    fn keyless_alu(
        assembler: CodeAssembler,
        boundary_index: usize,
        keyed: bool,
    ) -> (HandlerEffectSummary, VmHandlerAnalysis, bool) {
        let trace = trace_from_assembler(assembler);
        let boundary = trace
            .instructions()
            .nth(boundary_index)
            .expect("dispatch load")
            .rva;
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
        let summary = extract_handler_effects_keyed(
            &trace,
            &slice,
            HandlerEffectRoles {
                vsp: Register::R11,
                context_base: Register::RSP,
            },
            crate::HandlerRegisterRoles {
                vip: Register::R15,
                rolling_key: Register::RBP,
                handler: Register::R14,
            },
            boundary,
            keyed,
        );
        let analysis = classify_handler_effects(trace.entry, &summary, None, Some(1));
        let proof =
            crate::prove_handler_value_flow(&trace, &slice, &summary, &analysis, None, None, false);
        (summary, analysis, proof.proven)
    }

    fn keyless_qword_add(flags_to_stack: impl FnOnce(&mut CodeAssembler)) -> CodeAssembler {
        let mut assembler = CodeAssembler::new(64).expect("x64 assembler");
        assembler.mov(rax, qword_ptr(r11)).expect("left input");
        assembler.mov(rdx, qword_ptr(r11 + 8)).expect("right input");
        assembler.add(rax, rdx).expect("add");
        assembler.pushfq().expect("capture flags");
        assembler.pop(rdx).expect("flags into a register");
        assembler.mov(qword_ptr(r11 + 8), rax).expect("result");
        flags_to_stack(&mut assembler);
        assembler.mov(ecx, dword_ptr(r15)).expect("dispatch load");
        assembler.jmp(r14).expect("dispatch");
        assembler
    }

    #[test]
    fn recognizes_keyless_add_with_flags_routed_through_a_register() {
        let assembler = keyless_qword_add(|assembler| {
            assembler.mov(qword_ptr(r11), rdx).expect("VM flags");
        });
        let (summary, analysis, proven) = keyless_alu(assembler, 7, false);

        assert!(summary.captures_native_flags_to_vsp);
        assert_eq!(analysis.semantic, VmHandlerSemantic::Add);
        assert_eq!(analysis.operation_width, Some(BitWidth::W64));
        assert_eq!(analysis.flags_effect, EffectKnowledge::Modified);
        assert!(proven);
    }

    #[test]
    fn rejects_register_routed_flags_that_are_clobbered_or_misplaced() {
        let clobbered = keyless_qword_add(|assembler| {
            assembler.mov(rdx, rax).expect("clobber the flags copy");
            assembler.mov(qword_ptr(r11), rdx).expect("VM flags");
        });
        let (summary, analysis, _) = keyless_alu(clobbered, 8, false);
        assert!(!summary.captures_native_flags_to_vsp);
        assert_eq!(analysis.semantic, VmHandlerSemantic::Unknown);

        let narrow = keyless_qword_add(|assembler| {
            assembler.mov(dword_ptr(r11), edx).expect("narrow store");
        });
        let (summary, analysis, _) = keyless_alu(narrow, 7, false);
        assert!(!summary.captures_native_flags_to_vsp);
        assert_eq!(analysis.semantic, VmHandlerSemantic::Unknown);

        let misplaced = keyless_qword_add(|assembler| {
            assembler.mov(qword_ptr(r11 + 16), rdx).expect("wrong slot");
        });
        let (summary, analysis, _) = keyless_alu(misplaced, 7, false);
        assert!(!summary.captures_native_flags_to_vsp);
        assert_eq!(analysis.semantic, VmHandlerSemantic::Unknown);
    }

    fn keyless_shift() -> CodeAssembler {
        let mut assembler = CodeAssembler::new(64).expect("x64 assembler");
        assembler.mov(r14, qword_ptr(r11)).expect("value");
        assembler.mov(cl, byte_ptr(r11 + 8)).expect("count");
        assembler.sub(r11, 6).expect("reserve flags cell");
        assembler.shr(r14, cl).expect("shift");
        assembler.pushfq().expect("capture flags");
        assembler.pop(rax).expect("flags into a register");
        assembler.mov(qword_ptr(r11 + 8), r14).expect("result");
        assembler.mov(qword_ptr(r11), rax).expect("VM flags");
        assembler.mov(edx, dword_ptr(r15)).expect("dispatch load");
        assembler.jmp(r14).expect("dispatch");
        assembler
    }

    #[test]
    fn keyless_shift_counts_scratch_arithmetic_on_the_handler_register() {
        let (_, keyless, proven) = keyless_alu(keyless_shift(), 8, false);
        assert_eq!(keyless.semantic, VmHandlerSemantic::ShiftRightLogical);
        assert_eq!(keyless.operation_width, Some(BitWidth::W64));
        assert!(proven);

        let (_, keyed, _) = keyless_alu(keyless_shift(), 8, true);
        assert_eq!(keyed.semantic, VmHandlerSemantic::Unknown);
    }

    fn byte_sub(
        build: impl FnOnce(&mut CodeAssembler),
        boundary_index: usize,
    ) -> (VmHandlerAnalysis, bool) {
        let mut assembler = CodeAssembler::new(64).expect("x64 assembler");
        assembler.movzx(r9w, byte_ptr(r11)).expect("left input");
        assembler.mov(r10b, byte_ptr(r11 + 2)).expect("right input");
        assembler.sub(r11, 6).expect("reserve flags cell");
        build(&mut assembler);
        assembler.pushfq().expect("capture flags");
        assembler.pop(rdi).expect("flags into a register");
        assembler.mov(word_ptr(r11 + 8), r9w).expect("result");
        assembler.mov(qword_ptr(r11), rdi).expect("VM flags");
        assembler.mov(edx, dword_ptr(r15)).expect("dispatch load");
        assembler.jmp(r14).expect("dispatch");
        let (_, analysis, proven) = keyless_alu(assembler, boundary_index, false);
        (analysis, proven)
    }

    #[test]
    fn recognizes_byte_sub_and_rejects_non_matching_shapes() {
        let (analysis, proven) = byte_sub(
            |assembler| {
                assembler.sub(r9b, r10b).expect("byte sub");
            },
            8,
        );
        assert_eq!(analysis.semantic, VmHandlerSemantic::Sub);
        assert_eq!(analysis.input_width, Some(BitWidth::W8));
        assert_eq!(analysis.operation_width, Some(BitWidth::W16));
        assert!(proven);

        let (analysis, _) = byte_sub(
            |assembler| {
                assembler.sub(r9b, r10b).expect("byte sub");
                assembler.and(r9b, 1).expect("extra live operation");
            },
            9,
        );
        assert_eq!(analysis.semantic, VmHandlerSemantic::Unknown);

        let (analysis, proven) = byte_sub(
            |assembler| {
                assembler.sub(r10b, r9b).expect("reversed sub");
                assembler
                    .mov(r9b, r10b)
                    .expect("result from the wrong operand");
            },
            9,
        );
        assert!(analysis.semantic == VmHandlerSemantic::Unknown || !proven);
    }

    /// Builds a handler body, appends the dispatch tail, and returns the
    /// classification with the value-flow demotion applied as in recovery.
    fn classify_body(build: impl FnOnce(&mut CodeAssembler)) -> (VmHandlerAnalysis, bool) {
        let mut assembler = CodeAssembler::new(64).expect("x64 assembler");
        build(&mut assembler);
        assembler.mov(edx, dword_ptr(r15)).expect("dispatch load");
        assembler.jmp(r14).expect("dispatch");
        let trace = trace_from_assembler(assembler);
        let count = trace.instructions().count();
        let boundary = trace
            .instructions()
            .nth(count - 2)
            .expect("dispatch load")
            .rva;
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
        let summary = extract_handler_effects(
            &trace,
            &slice,
            HandlerEffectRoles {
                vsp: Register::R11,
                context_base: Register::RSP,
            },
            crate::HandlerRegisterRoles {
                vip: Register::R15,
                rolling_key: Register::RBP,
                handler: Register::R14,
            },
            boundary,
        );
        let mut analysis = classify_handler_effects(trace.entry, &summary, None, Some(4));
        let proof =
            crate::prove_handler_value_flow(&trace, &slice, &summary, &analysis, None, None, false);
        if !proof.proven {
            analysis.semantic = VmHandlerSemantic::Unknown;
        }
        (analysis, proof.proven)
    }

    fn assert_class(
        build: impl FnOnce(&mut CodeAssembler),
        semantic: VmHandlerSemantic,
        width: BitWidth,
    ) -> VmHandlerAnalysis {
        let (analysis, proven) = classify_body(build);
        assert_eq!(analysis.semantic, semantic);
        assert_eq!(analysis.operation_width, Some(width));
        assert!(proven);
        analysis
    }

    fn assert_unknown(build: impl FnOnce(&mut CodeAssembler)) {
        let (analysis, _) = classify_body(build);
        assert_eq!(analysis.semantic, VmHandlerSemantic::Unknown);
    }

    #[test]
    fn classifies_narrow_vsp_push_and_rejects_clobbered_copy() {
        let push = |width16: bool, clobber: bool| {
            move |a: &mut CodeAssembler| {
                a.mov(r10, r11).expect("copy VSP");
                if clobber {
                    a.mov(r10w, cx).expect("clobber the low word");
                }
                if width16 {
                    a.sub(r11, 2).expect("reserve");
                    a.mov(word_ptr(r11), r10w).expect("push low word");
                } else {
                    a.sub(r11, 4).expect("reserve");
                    a.mov(dword_ptr(r11), r10d).expect("push low dword");
                }
            }
        };
        assert_class(push(true, false), VmHandlerSemantic::Push, BitWidth::W16);
        assert_class(push(false, false), VmHandlerSemantic::Push, BitWidth::W32);
        assert_unknown(push(true, true));
    }

    #[test]
    fn classifies_narrow_vsp_reload_only_from_the_stack_top() {
        assert_class(
            |a| a.mov(r11d, dword_ptr(r11)).expect("reload"),
            VmHandlerSemantic::Pop,
            BitWidth::W32,
        );
        assert_class(
            |a| a.mov(r11w, word_ptr(r11)).expect("reload"),
            VmHandlerSemantic::Pop,
            BitWidth::W16,
        );
        assert_unknown(|a| a.mov(r11d, dword_ptr(r11 + 4)).expect("offset reload"));
        assert_unknown(|a| a.mov(r11d, dword_ptr(r10)).expect("foreign pointer"));
    }

    fn double_shift(right: bool, swap_operands: bool) -> impl FnOnce(&mut CodeAssembler) {
        move |a| {
            a.mov(r9d, dword_ptr(r11)).expect("destination");
            a.mov(r8d, dword_ptr(r11 + 4)).expect("source");
            a.mov(cl, byte_ptr(r11 + 8)).expect("count");
            a.sub(r11, 2).expect("reserve flags cell");
            match (right, swap_operands) {
                (false, false) => a.shld(r9d, r8d, cl),
                (true, false) => a.shrd(r9d, r8d, cl),
                (false, true) => a.shld(r8d, r9d, cl),
                (true, true) => a.shrd(r8d, r9d, cl),
            }
            .expect("double shift");
            a.pushfq().expect("capture flags");
            a.pop(rdi).expect("flags into a register");
            a.mov(dword_ptr(r11 + 8), r9d).expect("result");
            a.mov(qword_ptr(r11), rdi).expect("VM flags");
        }
    }

    #[test]
    fn classifies_double_shifts_and_rejects_swapped_operands() {
        let analysis = assert_class(
            double_shift(false, false),
            VmHandlerSemantic::ShiftLeftDouble,
            BitWidth::W32,
        );
        assert_eq!(analysis.stack_delta, Some(-2));
        assert_class(
            double_shift(true, false),
            VmHandlerSemantic::ShiftRightDouble,
            BitWidth::W32,
        );
        assert_unknown(double_shift(false, true));
        assert_unknown(double_shift(true, true));
    }

    fn byte_shift(clobber_high_byte: bool, count_slot: i32) -> impl FnOnce(&mut CodeAssembler) {
        move |a| {
            a.movzx(ax, byte_ptr(r11)).expect("value");
            a.mov(cl, byte_ptr(r11 + count_slot)).expect("count");
            if clobber_high_byte {
                a.mov(ah, bl).expect("pollute the high byte");
            }
            a.sub(r11, 6).expect("reserve flags cell");
            a.shr(al, cl).expect("byte shift");
            a.pushfq().expect("capture flags");
            a.pop(rdi).expect("flags into a register");
            a.mov(word_ptr(r11 + 8), ax).expect("result");
            a.mov(qword_ptr(r11), rdi).expect("VM flags");
        }
    }

    #[test]
    fn classifies_byte_shift_in_word_slots_and_rejects_dirty_high_byte() {
        let analysis = assert_class(
            byte_shift(false, 2),
            VmHandlerSemantic::ShiftRightLogical,
            BitWidth::W16,
        );
        assert_eq!(analysis.input_width, Some(BitWidth::W8));
        assert_eq!(analysis.resize, Some(ResizeOp::ZeroExtend));
        assert_eq!(analysis.stack_delta, Some(-6));
        assert_unknown(byte_shift(true, 2));
        assert_unknown(byte_shift(false, 3));
    }

    fn wide_multiply(
        signed: bool,
        width64: bool,
        swapped_inputs: bool,
    ) -> impl FnOnce(&mut CodeAssembler) {
        move |a| {
            let (first, second) = if swapped_inputs { (4, 0) } else { (0, 4) };
            if width64 {
                a.mov(rbx, qword_ptr(r11 + first * 2)).expect("multiplier");
                a.mov(rax, qword_ptr(r11 + second * 2))
                    .expect("accumulator");
            } else {
                a.mov(ebx, dword_ptr(r11 + first)).expect("multiplier");
                a.mov(eax, dword_ptr(r11 + second)).expect("accumulator");
            }
            a.sub(r11, 8).expect("reserve result and flags cells");
            match (signed, width64) {
                (false, false) => a.mul(ebx),
                (false, true) => a.mul(rbx),
                (true, false) => a.imul(ebx),
                (true, true) => a.imul(rbx),
            }
            .expect("multiply");
            a.pushfq().expect("capture flags");
            a.pop(rdi).expect("flags into a register");
            if width64 {
                a.mov(qword_ptr(r11 + 8), rdx).expect("high");
                a.mov(qword_ptr(r11 + 16), rax).expect("low");
            } else {
                a.mov(dword_ptr(r11 + 8), edx).expect("high");
                a.mov(dword_ptr(r11 + 12), eax).expect("low");
            }
            a.mov(qword_ptr(r11), rdi).expect("VM flags");
        }
    }

    #[test]
    fn classifies_multiplies_with_any_multiplier_register() {
        assert_class(
            wide_multiply(false, false, false),
            VmHandlerSemantic::Mul,
            BitWidth::W32,
        );
        assert_class(
            wide_multiply(false, true, false),
            VmHandlerSemantic::Mul,
            BitWidth::W64,
        );
        assert_class(
            wide_multiply(true, false, false),
            VmHandlerSemantic::SignedMultiply,
            BitWidth::W32,
        );
        assert_class(
            wide_multiply(true, true, false),
            VmHandlerSemantic::SignedMultiply,
            BitWidth::W64,
        );
        assert_unknown(wide_multiply(false, false, true));
        assert_unknown(wide_multiply(true, true, true));
    }

    #[test]
    fn rejects_two_operand_imul_as_a_wide_multiply() {
        assert_unknown(|a| {
            a.mov(ebx, dword_ptr(r11)).expect("multiplier");
            a.mov(eax, dword_ptr(r11 + 4)).expect("accumulator");
            a.sub(r11, 8).expect("reserve");
            a.imul_2(eax, ebx).expect("two-operand imul");
            a.pushfq().expect("capture flags");
            a.pop(rdi).expect("flags");
            a.mov(dword_ptr(r11 + 8), edx).expect("high");
            a.mov(dword_ptr(r11 + 12), eax).expect("low");
            a.mov(qword_ptr(r11), rdi).expect("VM flags");
        });
    }

    fn qword_divide(swap_results: bool) -> impl FnOnce(&mut CodeAssembler) {
        move |a| {
            a.mov(rdx, qword_ptr(r11)).expect("high dividend");
            a.mov(rax, qword_ptr(r11 + 8)).expect("low dividend");
            a.mov(rbx, qword_ptr(r11 + 16)).expect("divisor");
            a.add(r11, 8).expect("collapse three operands into two");
            a.div(rbx).expect("divide");
            if swap_results {
                a.mov(qword_ptr(r11), rax).expect("quotient");
                a.mov(qword_ptr(r11 + 8), rdx).expect("remainder");
            } else {
                a.mov(qword_ptr(r11), rdx).expect("remainder");
                a.mov(qword_ptr(r11 + 8), rax).expect("quotient");
            }
        }
    }

    #[test]
    fn classifies_unflagged_qword_divide() {
        let analysis = assert_class(
            qword_divide(false),
            VmHandlerSemantic::UnsignedDivide,
            BitWidth::W64,
        );
        assert_eq!(analysis.stack_delta, Some(8));
        assert_unknown(qword_divide(true));
    }

    fn cpuid_body(swap_outputs: bool) -> impl FnOnce(&mut CodeAssembler) {
        move |a| {
            a.mov(eax, dword_ptr(r11)).expect("leaf");
            a.mov(ecx, dword_ptr(r11 + 4)).expect("sub-leaf");
            a.sub(r11, 8).expect("reserve");
            a.cpuid().expect("cpuid");
            a.mov(dword_ptr(r11 + 12), eax).expect("eax");
            if swap_outputs {
                a.mov(dword_ptr(r11 + 8), ecx).expect("ecx in the ebx cell");
                a.mov(dword_ptr(r11 + 4), ebx).expect("ebx in the ecx cell");
            } else {
                a.mov(dword_ptr(r11 + 8), ebx).expect("ebx");
                a.mov(dword_ptr(r11 + 4), ecx).expect("ecx");
            }
            a.mov(dword_ptr(r11), edx).expect("edx");
        }
    }

    #[test]
    fn classifies_cpuid_and_rejects_swapped_output_cells() {
        assert_class(cpuid_body(false), VmHandlerSemantic::Cpuid, BitWidth::W32);
        assert_unknown(cpuid_body(true));
    }

    fn cpuid_leaf_only_body(swap_outputs: bool) -> impl FnOnce(&mut CodeAssembler) {
        move |a| {
            a.mov(eax, dword_ptr(r11)).expect("leaf");
            a.cpuid().expect("cpuid");
            a.sub(r11, 12).expect("reserve");
            a.mov(dword_ptr(r11 + 12), eax).expect("eax");
            if swap_outputs {
                a.mov(dword_ptr(r11 + 8), ecx).expect("ecx in the ebx cell");
                a.mov(dword_ptr(r11 + 4), ebx).expect("ebx in the ecx cell");
            } else {
                a.mov(dword_ptr(r11 + 8), ebx).expect("ebx");
                a.mov(dword_ptr(r11 + 4), ecx).expect("ecx");
            }
            a.mov(dword_ptr(r11), edx).expect("edx");
        }
    }

    #[test]
    fn classifies_leaf_only_cpuid_and_rejects_swapped_output_cells() {
        // A leaf-only CPUID (no stack subleaf, net VSP delta -12) classifies and
        // proves; the same shape with transposed EBX/ECX output cells stays
        // Unknown, so a near-miss cannot be mistaken for the real handler.
        let analysis =
            assert_class(cpuid_leaf_only_body(false), VmHandlerSemantic::Cpuid, BitWidth::W32);
        assert_eq!(analysis.stack_delta, Some(-12));
        assert_unknown(cpuid_leaf_only_body(true));
    }

    fn pop_flags(pushed: AsmRegister64) -> impl FnOnce(&mut CodeAssembler) {
        move |a| {
            a.mov(rdx, qword_ptr(r11)).expect("flags image");
            a.add(r11, 8).expect("pop");
            a.push(pushed).expect("native push");
            a.popfq().expect("restore flags");
        }
    }

    #[test]
    fn classifies_popfq_only_for_the_popped_value() {
        let analysis = assert_class(pop_flags(rdx), VmHandlerSemantic::PopFlags, BitWidth::W64);
        assert_eq!(analysis.flags_effect, EffectKnowledge::Modified);
        assert_unknown(pop_flags(rcx));
    }

    #[test]
    fn classifies_only_gs_relative_pointer_loads() {
        let load = |use_gs: bool| {
            move |a: &mut CodeAssembler| {
                a.mov(rcx, qword_ptr(r11)).expect("pointer");
                if use_gs {
                    a.mov(rdi, qword_ptr(rcx).gs()).expect("gs load");
                } else {
                    a.mov(rdi, qword_ptr(rcx).fs()).expect("fs load");
                }
                a.mov(qword_ptr(r11), rdi).expect("result");
            }
        };
        assert_class(load(true), VmHandlerSemantic::GsRelativeLoad, BitWidth::W64);
        assert_unknown(load(false));
    }

    #[test]
    fn classifies_byte_store_through_a_word_value_slot() {
        let store = |advance: i32| {
            move |a: &mut CodeAssembler| {
                a.mov(rdi, qword_ptr(r11)).expect("pointer");
                a.mov(al, byte_ptr(r11 + 8)).expect("value");
                a.add(r11, advance).expect("pop pointer and value slot");
                a.mov(byte_ptr(rdi), al).expect("store");
            }
        };
        assert_class(store(10), VmHandlerSemantic::MemoryStore, BitWidth::W8);
        assert_unknown(store(11));
    }
}
