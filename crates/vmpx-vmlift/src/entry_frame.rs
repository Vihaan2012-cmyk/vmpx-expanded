//! Initial VM-stack contents of a virtualized entry, recovered by stepping the
//! native entry stub and its call target up to the VSP capture.
//!
//! This mirrors the CFG walker's private frame recovery so the lift adapter can
//! seed symbolic execution with a proven frame. It fails closed on anything it
//! cannot model.

use std::collections::BTreeMap;

use iced_x86::{
    FlowControl, Instruction, InstructionInfoFactory, Mnemonic, OpAccess, OpKind, Register,
};
use vmpx_disasm::{DecodedInstruction, DirectJumpTraceLimits, trace_direct_jump_fragments};
use vmpx_pe::{PeImage, Rva};
use vmpx_vm::{
    BinaryOp, BitWidth, Expr, NativeFrameSource, SymbolicVmInitialState, X64NativeRegister,
};

#[derive(Clone, Debug, Eq, PartialEq)]
enum Provenance {
    Native(X64NativeRegister),
    Constant(u64),
    Unknown,
}

fn canonical(register: Register) -> Register {
    register.full_register()
}

fn immediate(instruction: &Instruction, operand: u32) -> Option<u64> {
    match instruction.try_op_kind(operand).ok()? {
        OpKind::Immediate8
        | OpKind::Immediate8_2nd
        | OpKind::Immediate16
        | OpKind::Immediate32
        | OpKind::Immediate64
        | OpKind::Immediate8to16
        | OpKind::Immediate8to32
        | OpKind::Immediate8to64
        | OpKind::Immediate32to64 => Some(instruction.immediate(operand)),
        _ => None,
    }
}

fn defines_flags(instruction: &Instruction) -> bool {
    instruction.rflags_written()
        | instruction.rflags_cleared()
        | instruction.rflags_set()
        | instruction.rflags_undefined()
        != 0
}

fn writes_register(
    factory: &mut InstructionInfoFactory,
    instruction: &Instruction,
    register: Register,
) -> bool {
    factory.info(instruction).used_registers().iter().any(|used| {
        canonical(used.register()) == register
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

fn full_pair(instruction: &Instruction, mnemonic: Mnemonic) -> Option<(Register, Register)> {
    (instruction.mnemonic() == mnemonic
        && instruction.op0_kind() == OpKind::Register
        && instruction.op1_kind() == OpKind::Register
        && instruction.op0_register().size() == 8
        && instruction.op1_register().size() == 8)
        .then(|| {
            (
                canonical(instruction.op0_register()),
                canonical(instruction.op1_register()),
            )
        })
}

fn exact_zero(instruction: &Instruction) -> Option<Register> {
    (instruction.op0_kind() == OpKind::Register
        && instruction.op0_register().size() >= 4
        && ((instruction.mnemonic() == Mnemonic::Mov && immediate(instruction, 1) == Some(0))
            || (matches!(instruction.mnemonic(), Mnemonic::Xor | Mnemonic::Sub)
                && instruction.op1_kind() == OpKind::Register
                && instruction.op0_register() == instruction.op1_register())))
    .then(|| canonical(instruction.op0_register()))
}

fn update_provenance(
    factory: &mut InstructionInfoFactory,
    instruction: &Instruction,
    provenance: &mut BTreeMap<Register, Provenance>,
) {
    let get = |map: &BTreeMap<Register, Provenance>, register| {
        map.get(&register).cloned().unwrap_or(Provenance::Unknown)
    };
    if let Some((left, right)) = full_pair(instruction, Mnemonic::Xchg) {
        let (left_value, right_value) = (get(provenance, left), get(provenance, right));
        provenance.insert(left, right_value);
        provenance.insert(right, left_value);
        return;
    }
    if let Some((destination, source)) = full_pair(instruction, Mnemonic::Mov) {
        let value = get(provenance, source);
        provenance.insert(destination, value);
        return;
    }
    if let Some(destination) = exact_zero(instruction) {
        provenance.insert(destination, Provenance::Constant(0));
        return;
    }
    if instruction.mnemonic() == Mnemonic::Mov
        && instruction.op0_kind() == OpKind::Register
        && matches!(instruction.op0_register().size(), 4 | 8)
        && let Some(value) = immediate(instruction, 1)
    {
        let value = if instruction.op0_register().size() == 4 {
            value & 0xFFFF_FFFF
        } else {
            value
        };
        provenance.insert(
            canonical(instruction.op0_register()),
            Provenance::Constant(value),
        );
        return;
    }
    for register in X64NativeRegister::ALL {
        if writes_register(factory, instruction, register.to_iced()) {
            provenance.insert(register.to_iced(), Provenance::Unknown);
        }
    }
}

fn pushed_source(
    decoded: &DecodedInstruction,
    provenance: &BTreeMap<Register, Provenance>,
    original_flags: bool,
) -> Option<NativeFrameSource> {
    let instruction = &decoded.instruction;
    Some(match instruction.mnemonic() {
        Mnemonic::Push if instruction.op0_kind() == OpKind::Register => {
            match provenance.get(&canonical(instruction.op0_register())) {
                Some(Provenance::Native(register)) => NativeFrameSource::Register {
                    register: *register,
                },
                Some(Provenance::Constant(value)) => NativeFrameSource::Constant { value: *value },
                _ => NativeFrameSource::Unknown {
                    description: format!("unproven push at {}", decoded.rva),
                },
            }
        }
        Mnemonic::Push => NativeFrameSource::Constant {
            value: immediate(instruction, 0)?,
        },
        Mnemonic::Pushfq if original_flags => NativeFrameSource::Rflags,
        Mnemonic::Pushfq => NativeFrameSource::Unknown {
            description: "RFLAGS modified before PUSHFQ".to_owned(),
        },
        _ => return None,
    })
}

fn stored_source(
    instruction: &Instruction,
    provenance: &BTreeMap<Register, Provenance>,
) -> Option<NativeFrameSource> {
    if instruction.op1_kind() == OpKind::Register && instruction.op1_register().size() == 8 {
        return Some(match provenance.get(&canonical(instruction.op1_register()))? {
            Provenance::Native(register) => NativeFrameSource::Register {
                register: *register,
            },
            Provenance::Constant(value) => NativeFrameSource::Constant { value: *value },
            Provenance::Unknown => NativeFrameSource::Unknown {
                description: "stored register of unproven value".to_owned(),
            },
        });
    }
    (instruction.memory_size() == iced_x86::MemorySize::UInt64)
        .then(|| immediate(instruction, 1))
        .flatten()
        .map(|value| NativeFrameSource::Constant { value })
}

/// Proven contents of the VM stack at VM start, top first.
///
/// # Errors
///
/// Fails on nested calls, unknown stack writes, stack-pointer arithmetic, or a
/// trace that never reaches the VSP capture.
pub fn recover_entry_frame(
    pe: &PeImage<'_>,
    entry: Rva,
    vsp: Register,
    limits: DirectJumpTraceLimits,
) -> Result<Vec<NativeFrameSource>, String> {
    let caller = trace_direct_jump_fragments(pe, entry, limits).map_err(|e| e.to_string())?;
    let caller_instructions = caller.instructions().collect::<Vec<_>>();
    let (call_index, call) = caller_instructions
        .iter()
        .enumerate()
        .find(|(_, decoded)| decoded.instruction.flow_control() == FlowControl::Call)
        .ok_or("entry trace has no direct call")?;
    let call_target = call
        .direct_target_rva(pe)
        .ok_or("entry call has no direct target")?;
    let mut stack: Vec<NativeFrameSource> = Vec::new();
    for decoded in caller_instructions.iter().copied().take(call_index) {
        let instruction = &decoded.instruction;
        if instruction.mnemonic() == Mnemonic::Push {
            let ordinal = u32::try_from(stack.len()).unwrap_or(u32::MAX);
            stack.push(match immediate(instruction, 0) {
                Some(value) => NativeFrameSource::EntryImmediate { ordinal, value },
                None => NativeFrameSource::Unknown {
                    description: format!("unclassified pre-call stack value at {}", decoded.rva),
                },
            });
        } else if instruction.stack_pointer_increment() != 0 {
            return Err(format!(
                "unproven pre-call stack mutation at {}",
                decoded.rva
            ));
        }
    }
    stack.push(NativeFrameSource::ReturnAddress {
        call_site: call.rva,
        continuation: call.next_rva().ok_or("call continuation overflows")?,
    });

    let target = trace_direct_jump_fragments(pe, call_target, limits).map_err(|e| e.to_string())?;
    let mut provenance: BTreeMap<Register, Provenance> = X64NativeRegister::ALL
        .into_iter()
        .map(|register| (register.to_iced(), Provenance::Native(register)))
        .collect();
    let mut original_flags = true;
    let mut factory = InstructionInfoFactory::new();
    for decoded in target.instructions() {
        let instruction = &decoded.instruction;
        if instruction.mnemonic() == Mnemonic::Mov
            && instruction.op0_kind() == OpKind::Register
            && instruction.op1_kind() == OpKind::Register
            && instruction.op0_register().size() == 8
            && canonical(instruction.op0_register()) == canonical(vsp)
            && canonical(instruction.op1_register()) == Register::RSP
        {
            stack.reverse();
            return Ok(stack);
        }
        match instruction.flow_control() {
            FlowControl::Next | FlowControl::UnconditionalBranch => {}
            _ => return Err(format!("unsupported control flow at {}", decoded.rva)),
        }
        if matches!(instruction.mnemonic(), Mnemonic::Push | Mnemonic::Pushfq) {
            let source = pushed_source(decoded, &provenance, original_flags)
                .ok_or_else(|| format!("unsupported push at {}", decoded.rva))?;
            if instruction.stack_pointer_increment() != -8 {
                return Err(format!("non-qword push at {}", decoded.rva));
            }
            stack.push(source);
        } else {
            if instruction.stack_pointer_increment() != 0 {
                return Err(format!("unproven stack mutation at {}", decoded.rva));
            }
            let writes: Vec<_> = factory
                .info(instruction)
                .used_memory()
                .iter()
                .filter(|memory| {
                    matches!(
                        memory.access(),
                        OpAccess::Write
                            | OpAccess::CondWrite
                            | OpAccess::ReadWrite
                            | OpAccess::ReadCondWrite
                    )
                })
                .map(|memory| (memory.base(), memory.index(), memory.displacement()))
                .collect();
            if let Some(&(base, index, displacement)) = writes.first() {
                let stored = (writes.len() == 1
                    && base == Register::RSP
                    && index == Register::None
                    && instruction.mnemonic() == Mnemonic::Mov
                    && instruction.op0_kind() == OpKind::Memory
                    && displacement % 8 == 0)
                    .then(|| stored_source(instruction, &provenance))
                    .flatten()
                    .ok_or_else(|| format!("unproven memory write at {}", decoded.rva))?;
                let from_top = usize::try_from(displacement / 8)
                    .map_err(|_| format!("stack write offset at {}", decoded.rva))?;
                let position = stack
                    .len()
                    .checked_sub(1)
                    .and_then(|top| top.checked_sub(from_top))
                    .ok_or_else(|| format!("stack write outside the frame at {}", decoded.rva))?;
                stack[position] = stored;
            }
        }
        update_provenance(&mut factory, instruction, &mut provenance);
        if defines_flags(instruction) {
            original_flags = false;
        }
    }
    Err("no VSP capture reached".to_owned())
}

fn symbolic_value(source: &NativeFrameSource) -> Option<Expr> {
    match source {
        NativeFrameSource::Register { register } => {
            Some(Expr::register(BitWidth::W64, register.symbolic_name()))
        }
        NativeFrameSource::Rflags => Some(Expr::register(BitWidth::W64, "native_rflags_initial")),
        NativeFrameSource::Constant { value } | NativeFrameSource::EntryImmediate { value, .. } => {
            Some(Expr::constant(BitWidth::W64, *value))
        }
        NativeFrameSource::ReturnAddress { .. } => Some(Expr::register(
            BitWidth::W64,
            "native_return_address_initial",
        )),
        NativeFrameSource::Unknown { .. } => None,
    }
}

/// Initial symbolic state for a frame; `usize` counts the cells with a proven
/// value (an unknown cell stays absent so a read of it fails closed).
#[must_use]
pub fn frame_initial_state(cells: &[NativeFrameSource]) -> (SymbolicVmInitialState, usize) {
    let mut initial = SymbolicVmInitialState::default();
    let base = Expr::register(BitWidth::W64, "vsp_initial");
    let mut known = 0;
    for (index, cell) in cells.iter().enumerate() {
        let Some(value) = symbolic_value(cell) else {
            continue;
        };
        let Ok(address) = Expr::binary(
            BinaryOp::Add,
            base.clone(),
            Expr::constant(BitWidth::W64, (index as u64) * 8),
        ) else {
            continue;
        };
        initial.memory_values.push((address, value));
        known += 1;
    }
    (initial, known)
}
