use std::collections::{BTreeMap, BTreeSet};

use iced_x86::{Instruction, InstructionInfoFactory, Mnemonic, OpAccess, OpKind, Register};
use serde::Serialize;
use vmpx_disasm::{
    DirectJumpTrace, DirectJumpTraceLimits, DisasmError, trace_direct_jump_fragments,
};
use vmpx_pe::{PeImage, Rva, Va};

use crate::{
    BitWidth, BytecodeDirection, HandlerRegisterRoles, HandlerSliceConfig, KeyedTransformCodec,
    RecoveryStatus, RelativeDispatchCodec, TransformChain, backward_slice_handler_trace,
    dispatch_extract::{
        canonical, has_flat_segment_address, immediate_transform, instruction_writes_value,
        is_active, key_operation, recover_key_updates, same_register_value, signed_immediate,
        writes_register,
    },
};

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct VmTransitionReport {
    pub schema_version: u32,
    pub entry: Rva,
    pub status: RecoveryStatus,
    pub target_load: Option<Rva>,
    pub target_register: Option<String>,
    pub old_vsp_register: String,
    pub old_vsp_delta: Option<i32>,
    pub new_vip_register: Option<String>,
    pub new_rolling_key_register: Option<String>,
    pub new_handler_register: Option<String>,
    pub new_vsp_candidates: Vec<String>,
    pub bytecode_direction: BytecodeDirection,
    pub handler_anchor: Option<Va>,
    pub dispatch: Option<RelativeDispatchCodec>,
    pub context_base_register: String,
    pub native_rsp_delta: Option<i64>,
    pub state_effects_proven: bool,
    pub confidence: u8,
    pub evidence: Vec<String>,
    pub diagnostics: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RecoveredVmTransition {
    pub report: VmTransitionReport,
    pub roles: HandlerRegisterRoles,
    pub vsp_candidates: Vec<Register>,
    pub dispatch: RelativeDispatchCodec,
    pub handler_anchor: Va,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VmTransitionAnalysis {
    pub report: VmTransitionReport,
    pub recovered: Option<RecoveredVmTransition>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Origin {
    PoppedTarget,
    PostPopVsp,
    Zero,
    StaticHandler(Va),
}

fn exact_qword_load_from(instruction: &Instruction, base: Register) -> Option<Register> {
    (instruction.mnemonic() == Mnemonic::Mov
        && instruction.op0_kind() == OpKind::Register
        && instruction.op0_register().size() == 8
        && instruction.op1_kind() == OpKind::Memory
        && instruction.memory_base() == canonical(base)
        && instruction.memory_index() == Register::None
        && instruction.memory_displacement64() == 0
        && instruction.memory_size().size() == 8
        && has_flat_segment_address(instruction))
    .then(|| canonical(instruction.op0_register()))
}

fn exact_old_vsp_pop(instruction: &Instruction, old_vsp: Register) -> bool {
    instruction.mnemonic() == Mnemonic::Add
        && instruction.op0_kind() == OpKind::Register
        && instruction.op0_register() == canonical(old_vsp)
        && signed_immediate(instruction, 1) == Some(8)
}

fn exact_static_lea(instruction: &Instruction) -> Option<(Register, Va)> {
    (instruction.mnemonic() == Mnemonic::Lea
        && instruction.op0_kind() == OpKind::Register
        && instruction.op0_register().size() == 8
        && instruction.op1_kind() == OpKind::Memory
        && matches!(
            instruction.memory_base(),
            Register::None | Register::RIP | Register::EIP
        )
        && instruction.memory_index() == Register::None)
        .then(|| {
            (
                canonical(instruction.op0_register()),
                Va(instruction.memory_displacement64()),
            )
        })
}

fn exact_full_copy(instruction: &Instruction) -> Option<(Register, Register)> {
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

fn exact_full_exchange(instruction: &Instruction) -> Option<(Register, Register)> {
    (instruction.mnemonic() == Mnemonic::Xchg
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

fn preserves_origin_by_zero(
    instruction: &Instruction,
    origins: &BTreeMap<Register, Origin>,
) -> bool {
    instruction.op0_kind() == OpKind::Register
        && instruction.op0_register().size() == 8
        && matches!(instruction.mnemonic(), Mnemonic::Add | Mnemonic::Sub)
        && (signed_immediate(instruction, 1) == Some(0)
            || (instruction.op1_kind() == OpKind::Register
                && instruction.op1_register().size() == 8
                && origins.get(&canonical(instruction.op1_register())) == Some(&Origin::Zero)))
}

fn exact_dword_source(instruction: &Instruction) -> Option<(Register, Register)> {
    (instruction.mnemonic() == Mnemonic::Mov
        && instruction.op0_kind() == OpKind::Register
        && instruction.op0_register().size() == 4
        && instruction.op1_kind() == OpKind::Memory
        && instruction.memory_base().size() == 8
        && instruction.memory_index() == Register::None
        && instruction.memory_displacement64() == 0
        && instruction.memory_size().size() == 4
        && has_flat_segment_address(instruction))
    .then(|| {
        (
            instruction.op0_register(),
            canonical(instruction.memory_base()),
        )
    })
}

fn terminal_register(trace: &DirectJumpTrace) -> Option<Register> {
    let instructions = trace.instructions().collect::<Vec<_>>();
    let terminal = instructions.last()?;
    if terminal.rva != trace.stop.at {
        return None;
    }
    if terminal.instruction.mnemonic() == Mnemonic::Ret {
        return instructions
            .get(instructions.len().checked_sub(2)?)
            .filter(|predecessor| {
                predecessor.instruction.mnemonic() == Mnemonic::Push
                    && predecessor.instruction.op0_kind() == OpKind::Register
            })
            .map(|predecessor| canonical(predecessor.instruction.op0_register()));
    }
    (terminal.instruction.op0_kind() == OpKind::Register)
        .then(|| canonical(terminal.instruction.op0_register()))
}

#[derive(Clone)]
struct TransitionShape {
    target_load: Rva,
    target_register: Register,
    source: Rva,
    decoded_register: Register,
    vip: Register,
    key: Register,
    handler: Register,
    handler_anchor: Va,
    vsp_candidates: Vec<Register>,
    direction: BytecodeDirection,
    cursor_adjustment: Rva,
}

fn access_writes(access: OpAccess) -> bool {
    matches!(
        access,
        OpAccess::Write | OpAccess::CondWrite | OpAccess::ReadWrite | OpAccess::ReadCondWrite
    )
}

fn signed_address_displacement(value: u64) -> i64 {
    i64::from_ne_bytes(value.to_ne_bytes())
}

fn proves_native_stack_scratch_write(
    instruction: &Instruction,
    base: Register,
    index: Register,
    displacement: u64,
    size: usize,
    rsp_delta: i64,
) -> bool {
    if matches!(
        instruction.mnemonic(),
        Mnemonic::Push | Mnemonic::Pushf | Mnemonic::Pushfd | Mnemonic::Pushfq
    ) {
        return true;
    }
    if base != Register::RSP
        || index != Register::None
        || instruction.segment_prefix() != Register::None
    {
        return false;
    }
    let Some(start) = rsp_delta.checked_add(signed_address_displacement(displacement)) else {
        return false;
    };
    let Some(end) = i64::try_from(size)
        .ok()
        .and_then(|size| start.checked_add(size))
    else {
        return false;
    };
    start < 0 && end <= 0
}

fn prove_transition_state_effects(
    trace: &DirectJumpTrace,
    context_base: Register,
) -> Result<i64, String> {
    let context_base = canonical(context_base);
    let mut factory = InstructionInfoFactory::new();
    let mut rsp_delta = 0_i64;
    for decoded in trace.instructions() {
        let instruction = &decoded.instruction;
        let info = factory.info(instruction);
        for used in info
            .used_memory()
            .iter()
            .filter(|used| access_writes(used.access()))
        {
            if !proves_native_stack_scratch_write(
                instruction,
                used.base(),
                used.index(),
                used.displacement(),
                used.memory_size().size(),
                rsp_delta,
            ) {
                return Err(format!(
                    "transition has an unproven memory write at {}",
                    decoded.rva
                ));
            }
        }
        let stack_delta = i64::from(instruction.stack_pointer_increment());
        if writes_register(&mut factory, instruction, context_base)
            && (context_base != Register::RSP || stack_delta == 0)
        {
            return Err(format!(
                "transition clobbers context base {context_base:?} at {}",
                decoded.rva
            ));
        }
        rsp_delta = rsp_delta
            .checked_add(stack_delta)
            .ok_or_else(|| "transition native stack delta overflows".to_owned())?;
    }
    if rsp_delta != 0 {
        return Err(format!(
            "transition does not restore native RSP (delta {rsp_delta})"
        ));
    }
    Ok(rsp_delta)
}

fn exact_cursor_adjustment(instruction: &Instruction, vip: Register) -> Option<BytecodeDirection> {
    (instruction.op0_kind() == OpKind::Register
        && instruction.op0_register() == canonical(vip)
        && instruction.op0_register().size() == 8
        && signed_immediate(instruction, 1) == Some(4))
    .then(|| match instruction.mnemonic() {
        Mnemonic::Add => Some(BytecodeDirection::Ascending),
        Mnemonic::Sub => Some(BytecodeDirection::Descending),
        _ => None,
    })
    .flatten()
}

fn first_post_source_cursor_adjustment(
    trace: &DirectJumpTrace,
    source: Rva,
    vip: Register,
) -> Option<(Rva, BytecodeDirection)> {
    let mut factory = InstructionInfoFactory::new();
    trace
        .instructions()
        .skip_while(|decoded| decoded.rva != source)
        .skip(1)
        .find_map(|decoded| {
            writes_register(&mut factory, &decoded.instruction, canonical(vip)).then(|| {
                exact_cursor_adjustment(&decoded.instruction, vip)
                    .map(|direction| (decoded.rva, direction))
            })
        })
        .flatten()
}

#[allow(clippy::too_many_lines)]
fn discover_shape(trace: &DirectJumpTrace, old_vsp: Register) -> Result<TransitionShape, String> {
    let handler = terminal_register(trace)
        .ok_or_else(|| "transition does not terminate through a register".to_owned())?;
    let mut factory = InstructionInfoFactory::new();
    let mut origins = BTreeMap::new();
    let mut target_load = None;
    let mut target_register = None;
    let mut saw_old_vsp_pop = false;
    let mut descending_vips = BTreeMap::new();
    let mut candidates = Vec::new();

    for decoded in trace.instructions() {
        let instruction = &decoded.instruction;
        if target_load.is_none()
            && let Some(target) = exact_qword_load_from(instruction, old_vsp)
        {
            target_load = Some(decoded.rva);
            target_register = Some(target);
            origins.insert(target, Origin::PoppedTarget);
            continue;
        }
        if target_load.is_some() && !saw_old_vsp_pop && exact_old_vsp_pop(instruction, old_vsp) {
            origins.insert(canonical(old_vsp), Origin::PostPopVsp);
            saw_old_vsp_pop = true;
            continue;
        }
        if let Some((left, right)) = exact_full_exchange(instruction) {
            let left_origin = origins.remove(&left);
            let right_origin = origins.remove(&right);
            if let Some(origin) = right_origin {
                origins.insert(left, origin);
            }
            if let Some(origin) = left_origin {
                origins.insert(right, origin);
            }
            continue;
        }
        if let Some((destination, source)) = exact_full_copy(instruction) {
            let copied = origins.get(&source).copied();
            origins.remove(&destination);
            if let Some(origin) = copied {
                origins.insert(destination, origin);
            }
            continue;
        }
        if let Some((destination, anchor)) = exact_static_lea(instruction) {
            origins.insert(destination, Origin::StaticHandler(anchor));
            continue;
        }
        if let Some(destination) = assigns_full_zero(instruction) {
            origins.insert(destination, Origin::Zero);
            continue;
        }
        if exact_cursor_adjustment(instruction, instruction.op0_register())
            == Some(BytecodeDirection::Descending)
            && origins.get(&canonical(instruction.op0_register())) == Some(&Origin::PoppedTarget)
        {
            descending_vips.insert(canonical(instruction.op0_register()), decoded.rva);
            continue;
        }
        if preserves_origin_by_zero(instruction, &origins) {
            continue;
        }
        if let Some((decoded_register, vip)) = exact_dword_source(instruction)
            && origins.get(&vip) == Some(&Origin::PoppedTarget)
            && let Some(Origin::StaticHandler(anchor)) = origins.get(&handler).copied()
        {
            let cursor = descending_vips
                .get(&vip)
                .copied()
                .map(|rva| (rva, BytecodeDirection::Descending))
                .or_else(|| first_post_source_cursor_adjustment(trace, decoded.rva, vip));
            let Some((cursor_adjustment, direction)) = cursor else {
                continue;
            };
            let popped_registers = origins
                .iter()
                .filter_map(|(register, origin)| {
                    (*origin == Origin::PoppedTarget).then_some(*register)
                })
                .collect::<BTreeSet<_>>();
            let key_candidates = trace
                .instructions()
                .skip_while(|instruction| instruction.rva != decoded.rva)
                .filter_map(|instruction| {
                    let native = &instruction.instruction;
                    (native.op0_kind() == OpKind::Register
                        && same_register_value(
                            native.op0_register(),
                            decoded_register,
                            BitWidth::W32,
                        )
                        && native.op1_kind() == OpKind::Register
                        && key_operation(native.mnemonic()).is_some())
                    .then(|| canonical(native.op1_register()))
                })
                .filter(|register| popped_registers.contains(register) && *register != vip)
                .collect::<BTreeSet<_>>();
            if key_candidates.len() == 1 {
                let vsp_candidates = origins
                    .iter()
                    .filter_map(|(register, origin)| {
                        (*origin == Origin::PostPopVsp).then_some(*register)
                    })
                    .collect();
                candidates.push(TransitionShape {
                    target_load: target_load.expect("checked above"),
                    target_register: target_register.expect("checked above"),
                    source: decoded.rva,
                    decoded_register,
                    vip,
                    key: *key_candidates.first().expect("one key"),
                    handler,
                    handler_anchor: anchor,
                    vsp_candidates,
                    direction,
                    cursor_adjustment,
                });
            }
        }
        let written = origins
            .keys()
            .copied()
            .filter(|register| writes_register(&mut factory, instruction, *register))
            .collect::<Vec<_>>();
        for register in written {
            origins.remove(&register);
            descending_vips.remove(&register);
        }
    }
    if !saw_old_vsp_pop {
        return Err("target load is not followed by an exact 8-byte VSP pop".to_owned());
    }
    let [candidate] = candidates.as_slice() else {
        return Err(format!(
            "expected one VM-profile transition shape, recovered {}",
            candidates.len()
        ));
    };
    Ok(candidate.clone())
}

fn exact_sign_extension(instruction: &Instruction, decoded_register: Register) -> bool {
    instruction.mnemonic() == Mnemonic::Movsxd
        && instruction.op0_kind() == OpKind::Register
        && instruction.op0_register().size() == 8
        && instruction.op1_kind() == OpKind::Register
        && instruction.op1_register().size() == 4
        && same_register_value(instruction.op1_register(), decoded_register, BitWidth::W32)
        && canonical(instruction.op0_register()) == canonical(decoded_register)
}

#[allow(clippy::too_many_lines)]
fn recover_dispatch(
    trace: &DirectJumpTrace,
    shape: &TransitionShape,
) -> Result<RelativeDispatchCodec, String> {
    let slice = backward_slice_handler_trace(
        trace,
        &HandlerSliceConfig {
            live_out_registers: vec![shape.vip, shape.key, shape.handler],
            live_out_rflags: 0,
        },
    );
    let dispositions = slice
        .instructions
        .iter()
        .map(|instruction| (instruction.rva, instruction.disposition))
        .collect::<BTreeMap<_, _>>();
    let active = trace
        .instructions()
        .filter(|decoded| {
            dispositions
                .get(&decoded.rva)
                .copied()
                .is_some_and(is_active)
        })
        .collect::<Vec<_>>();
    let source_index = active
        .iter()
        .position(|decoded| decoded.rva == shape.source)
        .ok_or_else(|| "dispatch source is not retained by the native slice".to_owned())?;
    let mut factory = InstructionInfoFactory::new();
    let mut input_key = None;
    let mut input_index = None;
    let mut transforms = Vec::new();
    let mut last_transform_index = None;
    let mut sign_extension_index = None;
    let mut handler_update_index = None;

    for (index, decoded) in active.iter().enumerate().skip(source_index + 1) {
        let instruction = &decoded.instruction;
        if instruction.mnemonic() == Mnemonic::Add
            && instruction.op0_kind() == OpKind::Register
            && instruction.op0_register() == shape.handler
            && instruction.op1_kind() == OpKind::Register
            && instruction.op1_register().size() == 8
            && canonical(instruction.op1_register()) == canonical(shape.decoded_register)
        {
            handler_update_index = Some(index);
            continue;
        }
        if writes_register(&mut factory, instruction, shape.handler) {
            return Err(format!(
                "new handler register has an unmodelled write at {}",
                decoded.rva
            ));
        }
        if writes_register(&mut factory, instruction, shape.vip)
            && decoded.rva != shape.cursor_adjustment
        {
            return Err(format!(
                "new VIP is modified after its dispatch source at {}",
                decoded.rva
            ));
        }
        if exact_sign_extension(instruction, shape.decoded_register) {
            sign_extension_index = Some(index);
            continue;
        }
        if instruction_writes_value(
            &mut factory,
            instruction,
            shape.decoded_register,
            BitWidth::W32,
        ) {
            let mixes_key = instruction.op0_kind() == OpKind::Register
                && same_register_value(
                    instruction.op0_register(),
                    shape.decoded_register,
                    BitWidth::W32,
                )
                && instruction.op1_kind() == OpKind::Register
                && same_register_value(instruction.op1_register(), shape.key, BitWidth::W32);
            if mixes_key {
                if input_key.is_some() || !transforms.is_empty() {
                    return Err("ambiguous or late transition input-key mix".to_owned());
                }
                input_key = key_operation(instruction.mnemonic());
                input_index = Some(index);
                continue;
            }
            let transform = immediate_transform(instruction, BitWidth::W32)
                .ok_or_else(|| format!("unsupported transition transform at {}", decoded.rva))?;
            transforms.push(transform);
            last_transform_index = Some(index);
        }
    }
    let updates = recover_key_updates(
        &active[source_index..],
        shape.key,
        shape.decoded_register,
        BitWidth::W32,
    );
    let [update] = updates.as_slice() else {
        return Err(format!(
            "expected one transition key update, recovered {}",
            updates.len()
        ));
    };
    let update_index = active
        .iter()
        .position(|decoded| decoded.rva == update.evidence_rva)
        .ok_or_else(|| "transition key update position is missing".to_owned())?;
    let completion_index = active
        .iter()
        .position(|decoded| decoded.rva == update.completion_rva)
        .ok_or_else(|| "transition key completion position is missing".to_owned())?;
    let handler_update_index =
        handler_update_index.ok_or_else(|| "new handler is not updated".to_owned())?;
    if input_index.is_none_or(|mix| update_index <= mix)
        || last_transform_index.is_some_and(|transform| update_index <= transform)
        || completion_index >= handler_update_index
        || sign_extension_index.is_none_or(|extension| {
            extension <= completion_index || extension >= handler_update_index
        })
    {
        return Err("transition dispatch operations are out of semantic order".to_owned());
    }
    let allowed_key_write = update.completion_rva;
    if active[source_index..].iter().any(|decoded| {
        writes_register(&mut factory, &decoded.instruction, shape.key)
            && decoded.rva != allowed_key_write
    }) {
        return Err("new rolling-key register has an unmodelled write".to_owned());
    }
    Ok(RelativeDispatchCodec {
        displacement: KeyedTransformCodec {
            width: BitWidth::W32,
            input_key_operation: input_key,
            input_key_on_left: false,
            transforms: TransformChain { transforms },
            update_key_operation: Some(update.operation),
        },
    })
}

/// Recovers a VM-profile transition from native data flow. This recognizes the
/// invariant state transfer and cursor-relative dispatch behavior; it does not use
/// handler addresses, byte signatures, or fixture-specific constants.
///
/// # Errors
///
/// Propagates bounded native trace decoding errors. Unsupported or ambiguous
/// transition shapes are retained as a failed structured report.
#[allow(clippy::too_many_lines)]
pub fn analyze_vm_transition(
    pe: &PeImage<'_>,
    entry: Rva,
    old_vsp: Register,
    context_base: Register,
    limits: DirectJumpTraceLimits,
) -> Result<VmTransitionAnalysis, DisasmError> {
    let trace = trace_direct_jump_fragments(pe, entry, limits)?;
    let analysis = discover_shape(&trace, old_vsp)
        .and_then(|shape| recover_dispatch(&trace, &shape).map(|dispatch| (shape, dispatch)))
        .and_then(|(shape, dispatch)| {
            prove_transition_state_effects(&trace, context_base)
                .map(|rsp_delta| (shape, dispatch, rsp_delta))
        });
    let old_vsp_name = format!("{:?}", canonical(old_vsp));
    match analysis {
        Ok((shape, dispatch, rsp_delta)) => {
            let roles = HandlerRegisterRoles {
                vip: shape.vip,
                rolling_key: shape.key,
                handler: shape.handler,
            };
            let evidence = vec![
                "target VIP is loaded as a qword from the old VSP".to_owned(),
                "old VSP advances by exactly 8 bytes".to_owned(),
                "new VIP and rolling key derive from the popped target".to_owned(),
                match shape.direction {
                    BytecodeDirection::Ascending => {
                        "new VIP advances by 4 after the dispatch read".to_owned()
                    }
                    BytecodeDirection::Descending => {
                        "new VIP pre-decrements by 4 before the dispatch read".to_owned()
                    }
                    BytecodeDirection::StateDependent | BytecodeDirection::Unknown => {
                        "new VIP cursor direction is unresolved".to_owned()
                    }
                },
                "new handler begins at a statically established anchor".to_owned(),
                "decoded rel32 updates the rolling key and handler before tail dispatch".to_owned(),
                "native RSP is balanced and all writes are proven native-stack scratch".to_owned(),
                format!(
                    "context-base register {:?} is preserved",
                    canonical(context_base)
                ),
            ];
            let report = VmTransitionReport {
                schema_version: 1,
                entry,
                status: RecoveryStatus::Recovered,
                target_load: Some(shape.target_load),
                target_register: Some(format!("{:?}", shape.target_register)),
                old_vsp_register: old_vsp_name,
                old_vsp_delta: Some(8),
                new_vip_register: Some(format!("{:?}", shape.vip)),
                new_rolling_key_register: Some(format!("{:?}", shape.key)),
                new_handler_register: Some(format!("{:?}", shape.handler)),
                new_vsp_candidates: shape
                    .vsp_candidates
                    .iter()
                    .map(|register| format!("{register:?}"))
                    .collect(),
                bytecode_direction: shape.direction,
                handler_anchor: Some(shape.handler_anchor),
                dispatch: Some(dispatch.clone()),
                context_base_register: format!("{:?}", canonical(context_base)),
                native_rsp_delta: Some(rsp_delta),
                state_effects_proven: true,
                confidence: 100,
                evidence,
                diagnostics: Vec::new(),
            };
            let recovered = RecoveredVmTransition {
                report: report.clone(),
                roles,
                vsp_candidates: shape.vsp_candidates,
                dispatch,
                handler_anchor: shape.handler_anchor,
            };
            Ok(VmTransitionAnalysis {
                report,
                recovered: Some(recovered),
            })
        }
        Err(reason) => Ok(VmTransitionAnalysis {
            report: VmTransitionReport {
                schema_version: 1,
                entry,
                status: RecoveryStatus::Failed,
                target_load: None,
                target_register: None,
                old_vsp_register: old_vsp_name,
                old_vsp_delta: None,
                new_vip_register: None,
                new_rolling_key_register: None,
                new_handler_register: None,
                new_vsp_candidates: Vec::new(),
                bytecode_direction: BytecodeDirection::Unknown,
                handler_anchor: None,
                dispatch: None,
                context_base_register: format!("{:?}", canonical(context_base)),
                native_rsp_delta: None,
                state_effects_proven: false,
                confidence: 0,
                evidence: Vec::new(),
                diagnostics: vec![reason],
            },
            recovered: None,
        }),
    }
}

#[cfg(test)]
mod tests {
    use iced_x86::{Code, Decoder, DecoderOptions, code_asm::*};
    use vmpx_disasm::{
        DecodedInstruction, DirectJumpFragment, DirectJumpTraceStop, DirectJumpTraceStopKind,
    };

    use super::*;

    fn transition_trace(direction: BytecodeDirection, push_return: bool) -> DirectJumpTrace {
        let mut assembler = CodeAssembler::new(64).expect("x64 assembler");
        assembler
            .mov(rax, qword_ptr(r14))
            .expect("load transition target");
        assembler.add(r14, 8).expect("pop old VSP");
        assembler.mov(rsi, rax).expect("copy new VIP");
        assembler.mov(r12, rax).expect("copy new key");
        assembler
            .lea(r11, qword_ptr(0x1234_u64))
            .expect("establish handler anchor");
        if direction == BytecodeDirection::Descending {
            assembler.sub(rsi, 4).expect("pre-decrement VIP");
        }
        assembler
            .mov(edx, dword_ptr(rsi))
            .expect("load encoded dispatch");
        if direction == BytecodeDirection::Ascending {
            assembler.add(rsi, 4).expect("post-increment VIP");
        }
        assembler.xor(edx, r12d).expect("mix key");
        assembler.xor(r12d, edx).expect("update key");
        assembler.movsxd(rdx, edx).expect("extend displacement");
        assembler.add(r11, rdx).expect("update handler");
        if push_return {
            assembler.push(r11).expect("push dispatch target");
            assembler.ret().expect("return-style tail dispatch");
        } else {
            assembler.jmp(r11).expect("tail dispatch");
        }
        let bytes = assembler
            .assemble(0x1_4000_1000)
            .expect("assemble transition");
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
        let terminal = instructions.last().expect("terminal instruction");
        let terminal_rva = terminal.rva;
        let trace_end = terminal.next_rva().expect("small trace end");
        DirectJumpTrace {
            entry: Rva(0x1000),
            decoded_instructions: instructions.len(),
            fragments: vec![DirectJumpFragment {
                start: Rva(0x1000),
                end: trace_end,
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

    #[test]
    fn discovers_transition_cursor_direction_from_timing() {
        for direction in [BytecodeDirection::Ascending, BytecodeDirection::Descending] {
            for push_return in [false, true] {
                let shape = discover_shape(
                    &transition_trace(direction, push_return),
                    Register::R14,
                )
                .unwrap_or_else(|error| {
                    panic!(
                        "behavioral transition shape {direction:?}/{push_return} failed: {error}"
                    )
                });
                assert_eq!(shape.direction, direction);
                assert_eq!(shape.vip, Register::RSI);
                assert_eq!(shape.key, Register::R12);
                assert_eq!(shape.handler, Register::R11);
                assert_eq!(shape.vsp_candidates, vec![Register::R14]);
            }
        }
    }

    #[test]
    fn rejects_ascending_increment_before_source() {
        let mut trace = transition_trace(BytecodeDirection::Ascending, false);
        let instructions = &mut trace.fragments[0].instructions;
        let adjustment = instructions
            .iter()
            .position(|decoded| {
                decoded.instruction.mnemonic() == Mnemonic::Add
                    && decoded.instruction.op0_register() == Register::RSI
            })
            .expect("ascending adjustment");
        instructions.swap(adjustment - 1, adjustment);
        assert!(discover_shape(&trace, Register::R14).is_err());
    }
}
