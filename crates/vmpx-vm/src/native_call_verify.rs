use std::collections::BTreeMap;

use iced_x86::{
    FlowControl, Instruction, InstructionInfoFactory, Mnemonic, OpAccess, OpKind, Register,
};
use serde::Serialize;
use vmpx_disasm::DecodedInstruction;
use vmpx_pe::{PeImage, Rva};

use crate::dispatch_extract::{canonical, has_flat_segment_address, signed_immediate};

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct NativeCallSelectorOutcome {
    pub selector: u8,
    pub vsp_delta: i32,
    pub target_stack_offset: i32,
    pub register_argument_offsets: Vec<i32>,
    pub stack_argument_offsets: Vec<i32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Value {
    Unknown,
    Const(u64),
    EntryVsp(i64),
    EntryRsp(i64),
    VmStack(i64),
    RoleState(Register),
    CallResult,
}

#[allow(clippy::struct_excessive_bools)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Flags {
    zero: bool,
    carry: bool,
    sign: bool,
    overflow: bool,
}

#[derive(Clone, Debug)]
struct Machine {
    registers: BTreeMap<Register, Value>,
    memory: BTreeMap<(u8, i64), Value>,
    flags: Option<Flags>,
    selector_seeded: bool,
    call_count: usize,
    stack_arguments: Vec<i32>,
}

fn memory_key(value: &Value) -> Option<(u8, i64)> {
    match value {
        Value::EntryVsp(offset) => Some((0, *offset)),
        Value::EntryRsp(offset) => Some((1, *offset)),
        _ => None,
    }
}

fn register_mask(register: Register) -> Option<(u32, u32)> {
    let bits = u32::try_from(register.size()).ok()?.checked_mul(8)?;
    let shift = match register {
        Register::AH | Register::BH | Register::CH | Register::DH => 8,
        _ => 0,
    };
    Some((bits, shift))
}

fn bit_mask(bits: u32) -> u64 {
    if bits == 64 {
        u64::MAX
    } else {
        (1_u64 << bits) - 1
    }
}

impl Machine {
    fn new(vsp: Register, preserved: &[Register]) -> Self {
        let mut registers = BTreeMap::new();
        registers.insert(canonical(vsp), Value::EntryVsp(0));
        registers.insert(Register::RSP, Value::EntryRsp(0));
        for role in preserved {
            registers.insert(canonical(*role), Value::RoleState(canonical(*role)));
        }
        Self {
            registers,
            memory: BTreeMap::new(),
            flags: None,
            selector_seeded: false,
            call_count: 0,
            stack_arguments: Vec::new(),
        }
    }

    fn read_register(&self, register: Register) -> Value {
        let canonical = canonical(register);
        let value = self
            .registers
            .get(&canonical)
            .copied()
            .unwrap_or(Value::Unknown);
        let Some((bits, shift)) = register_mask(register) else {
            return Value::Unknown;
        };
        if bits == 64 {
            return value;
        }
        match value {
            Value::Const(value) => Value::Const((value >> shift) & bit_mask(bits)),
            _ => Value::Unknown,
        }
    }

    fn write_register(&mut self, register: Register, value: Value) {
        let canonical = canonical(register);
        let Some((bits, shift)) = register_mask(register) else {
            self.registers.insert(canonical, Value::Unknown);
            return;
        };
        let next = if bits == 64 {
            value
        } else if bits == 32 && shift == 0 {
            match value {
                Value::Const(value) => Value::Const(value & bit_mask(32)),
                _ => Value::Unknown,
            }
        } else {
            match (self.registers.get(&canonical), value) {
                (Some(Value::Const(previous)), Value::Const(value)) => {
                    let mask = bit_mask(bits) << shift;
                    Value::Const((previous & !mask) | ((value << shift) & mask))
                }
                _ => Value::Unknown,
            }
        };
        self.registers.insert(canonical, next);
    }

    fn evaluate_address(&self, instruction: &Instruction) -> Value {
        if !has_flat_segment_address(instruction) {
            return Value::Unknown;
        }
        let displacement = i64::from_ne_bytes(instruction.memory_displacement64().to_ne_bytes());
        let base = if instruction.memory_base() == Register::None {
            Value::Const(0)
        } else {
            self.read_register(instruction.memory_base())
        };
        let index = if instruction.memory_index() == Register::None {
            Value::Const(0)
        } else {
            self.read_register(instruction.memory_index())
        };
        let scaled_index = match index {
            Value::Const(value) => value.wrapping_mul(u64::from(instruction.memory_index_scale())),
            _ => return Value::Unknown,
        };
        add_value(
            add_value(base, Value::Const(scaled_index)),
            Value::Const(u64::from_ne_bytes(displacement.to_ne_bytes())),
        )
    }

    fn read_memory(&self, address: Value) -> Value {
        if let Some(key) = memory_key(&address)
            && let Some(value) = self.memory.get(&key)
        {
            return *value;
        }
        match address {
            Value::EntryVsp(offset) => Value::VmStack(offset),
            _ => Value::Unknown,
        }
    }

    fn write_memory(&mut self, address: Value, value: Value) -> Result<(), String> {
        let Some(key) = memory_key(&address) else {
            return Err("native-call proof encountered an unbounded memory write".to_owned());
        };
        self.memory.insert(key, value);
        Ok(())
    }

    fn push(&mut self, value: Value) -> Result<(), String> {
        if let Value::VmStack(offset) = value {
            self.stack_arguments.push(
                i32::try_from(offset)
                    .map_err(|_| "stack-argument source offset exceeds i32".to_owned())?,
            );
        }
        let rsp = add_value(
            self.read_register(Register::RSP),
            Value::Const(u64::MAX - 7),
        );
        self.write_register(Register::RSP, rsp);
        if memory_key(&rsp).is_some() {
            self.write_memory(rsp, value)?;
        }
        Ok(())
    }

    fn pop(&mut self, destination: Register) {
        let rsp = self.read_register(Register::RSP);
        let value = self.read_memory(rsp);
        self.write_register(destination, value);
        self.write_register(Register::RSP, add_value(rsp, Value::Const(8)));
    }
}

fn add_value(left: Value, right: Value) -> Value {
    match (left, right) {
        (Value::Const(left), Value::Const(right)) => Value::Const(left.wrapping_add(right)),
        (Value::EntryVsp(offset), Value::Const(value))
        | (Value::Const(value), Value::EntryVsp(offset)) => {
            Value::EntryVsp(offset.wrapping_add(i64::from_ne_bytes(value.to_ne_bytes())))
        }
        (Value::EntryRsp(offset), Value::Const(value))
        | (Value::Const(value), Value::EntryRsp(offset)) => {
            Value::EntryRsp(offset.wrapping_add(i64::from_ne_bytes(value.to_ne_bytes())))
        }
        _ => Value::Unknown,
    }
}

fn subtract_value(left: Value, right: Value) -> Value {
    match right {
        Value::Const(value) => add_value(left, Value::Const(value.wrapping_neg())),
        _ => Value::Unknown,
    }
}

fn immediate(instruction: &Instruction, operand: u32) -> Option<u64> {
    signed_immediate(instruction, operand).map(|value| u64::from_ne_bytes(value.to_ne_bytes()))
}

fn operand_value(machine: &Machine, instruction: &Instruction, operand: u32) -> Value {
    match instruction.op_kind(operand) {
        OpKind::Register => machine.read_register(instruction.op_register(operand)),
        OpKind::Memory => machine.read_memory(machine.evaluate_address(instruction)),
        _ => immediate(instruction, operand).map_or(Value::Unknown, Value::Const),
    }
}

fn known_flags(left: u64, right: u64, bits: u32, is_test: bool) -> Flags {
    let mask = bit_mask(bits);
    let sign_mask = 1_u64 << (bits - 1);
    let left = left & mask;
    let right = right & mask;
    let result = if is_test {
        left & right
    } else {
        left.wrapping_sub(right) & mask
    };
    Flags {
        zero: result == 0,
        carry: !is_test && left < right,
        sign: result & sign_mask != 0,
        overflow: !is_test && ((left ^ right) & (left ^ result) & sign_mask != 0),
    }
}

fn evaluate_branch(mnemonic: Mnemonic, flags: Option<Flags>) -> Option<bool> {
    let flags = flags?;
    match mnemonic {
        Mnemonic::Je => Some(flags.zero),
        Mnemonic::Jne => Some(!flags.zero),
        Mnemonic::Jb => Some(flags.carry),
        Mnemonic::Jae => Some(!flags.carry),
        Mnemonic::Jbe => Some(flags.carry || flags.zero),
        Mnemonic::Ja => Some(!flags.carry && !flags.zero),
        Mnemonic::Jl => Some(flags.sign != flags.overflow),
        Mnemonic::Jge => Some(flags.sign == flags.overflow),
        Mnemonic::Jle => Some(flags.zero || flags.sign != flags.overflow),
        Mnemonic::Jg => Some(!flags.zero && flags.sign == flags.overflow),
        _ => None,
    }
}

fn invalidate_unmodeled_writes(machine: &mut Machine, instruction: &Instruction) {
    let mut factory = InstructionInfoFactory::new();
    for used in factory.info(instruction).used_registers() {
        if matches!(
            used.access(),
            OpAccess::Write | OpAccess::CondWrite | OpAccess::ReadWrite | OpAccess::ReadCondWrite
        ) && canonical(used.register()) != Register::RIP
        {
            machine.write_register(used.register(), Value::Unknown);
        }
    }
    if instruction.rflags_written() | instruction.rflags_cleared() | instruction.rflags_set() != 0 {
        machine.flags = None;
    }
}

#[allow(clippy::too_many_lines)]
fn execute_instruction(machine: &mut Machine, instruction: &Instruction) -> Result<(), String> {
    match instruction.mnemonic() {
        Mnemonic::Mov if instruction.op0_kind() == OpKind::Register => {
            let value = operand_value(machine, instruction, 1);
            machine.write_register(instruction.op0_register(), value);
        }
        Mnemonic::Mov if instruction.op0_kind() == OpKind::Memory => {
            let address = machine.evaluate_address(instruction);
            let value = operand_value(machine, instruction, 1);
            machine.write_memory(address, value)?;
        }
        Mnemonic::Movzx | Mnemonic::Movsx | Mnemonic::Movsxd
            if instruction.op0_kind() == OpKind::Register =>
        {
            let source = operand_value(machine, instruction, 1);
            let value = match source {
                Value::Const(value) => {
                    let source_bits = if instruction.op1_kind() == OpKind::Register {
                        u32::try_from(instruction.op1_register().size())
                            .ok()
                            .and_then(|bytes| bytes.checked_mul(8))
                    } else {
                        u32::try_from(instruction.memory_size().size())
                            .ok()
                            .and_then(|bytes| bytes.checked_mul(8))
                    }
                    .unwrap_or(64);
                    let value = value & bit_mask(source_bits);
                    if instruction.mnemonic() == Mnemonic::Movzx || source_bits == 64 {
                        Value::Const(value)
                    } else {
                        let shift = 64 - source_bits;
                        Value::Const(((value << shift).cast_signed() >> shift).cast_unsigned())
                    }
                }
                _ => Value::Unknown,
            };
            machine.write_register(instruction.op0_register(), value);
        }
        Mnemonic::Lea if instruction.op0_kind() == OpKind::Register => {
            let address = machine.evaluate_address(instruction);
            machine.write_register(instruction.op0_register(), address);
        }
        Mnemonic::Add | Mnemonic::Sub | Mnemonic::Xor | Mnemonic::And | Mnemonic::Or
            if instruction.op0_kind() == OpKind::Register =>
        {
            let destination = instruction.op0_register();
            let left = machine.read_register(destination);
            let right = operand_value(machine, instruction, 1);
            let value = match instruction.mnemonic() {
                Mnemonic::Add => add_value(left, right),
                Mnemonic::Sub => subtract_value(left, right),
                Mnemonic::Xor if left == right => Value::Const(0),
                Mnemonic::Xor | Mnemonic::And | Mnemonic::Or => match (&left, &right) {
                    (Value::Const(left), Value::Const(right)) => {
                        Value::Const(match instruction.mnemonic() {
                            Mnemonic::Xor => left ^ right,
                            Mnemonic::And => left & right,
                            Mnemonic::Or => left | right,
                            _ => unreachable!(),
                        })
                    }
                    _ => Value::Unknown,
                },
                _ => unreachable!(),
            };
            let bits = u32::try_from(destination.size())
                .ok()
                .and_then(|bytes| bytes.checked_mul(8))
                .unwrap_or(64);
            if let (Value::Const(left), Value::Const(right)) = (&left, &right) {
                machine.flags = Some(known_flags(
                    *left,
                    *right,
                    bits,
                    instruction.mnemonic() == Mnemonic::And,
                ));
            } else {
                machine.flags = None;
            }
            machine.write_register(destination, value);
        }
        Mnemonic::Shl if instruction.op0_kind() == OpKind::Register => {
            let destination = instruction.op0_register();
            let left = machine.read_register(destination);
            let right = operand_value(machine, instruction, 1);
            let value = match (left, right) {
                (Value::Const(left), Value::Const(right)) => {
                    Value::Const(left.wrapping_shl(u32::try_from(right & 63).unwrap_or(0)))
                }
                _ => Value::Unknown,
            };
            machine.write_register(destination, value);
            machine.flags = None;
        }
        Mnemonic::Inc | Mnemonic::Dec if instruction.op0_kind() == OpKind::Register => {
            let destination = instruction.op0_register();
            let left = machine.read_register(destination);
            let right = Value::Const(1);
            let value = if instruction.mnemonic() == Mnemonic::Inc {
                add_value(left, right)
            } else {
                subtract_value(left, right)
            };
            let bits = u32::try_from(destination.size())
                .ok()
                .and_then(|bytes| bytes.checked_mul(8))
                .unwrap_or(64);
            if let (Value::Const(left), Value::Const(right)) = (&left, &right) {
                machine.flags = Some(known_flags(*left, *right, bits, false));
            } else {
                machine.flags = None;
            }
            machine.write_register(destination, value);
        }
        Mnemonic::Cmp | Mnemonic::Test => {
            let left = operand_value(machine, instruction, 0);
            let right = operand_value(machine, instruction, 1);
            let bits = match instruction.op0_kind() {
                OpKind::Register => u32::try_from(instruction.op0_register().size())
                    .ok()
                    .and_then(|bytes| bytes.checked_mul(8)),
                OpKind::Memory => u32::try_from(instruction.memory_size().size())
                    .ok()
                    .and_then(|bytes| bytes.checked_mul(8)),
                _ => None,
            }
            .unwrap_or(64);
            machine.flags = match (left, right) {
                (Value::Const(left), Value::Const(right)) => Some(known_flags(
                    left,
                    right,
                    bits,
                    instruction.mnemonic() == Mnemonic::Test,
                )),
                _ => None,
            };
        }
        Mnemonic::Push => machine.push(operand_value(machine, instruction, 0))?,
        Mnemonic::Pop if instruction.op0_kind() == OpKind::Register => {
            machine.pop(instruction.op0_register());
        }
        Mnemonic::Jmp
        | Mnemonic::Je
        | Mnemonic::Jne
        | Mnemonic::Jb
        | Mnemonic::Jae
        | Mnemonic::Jbe
        | Mnemonic::Ja
        | Mnemonic::Jl
        | Mnemonic::Jge
        | Mnemonic::Jle
        | Mnemonic::Jg
        | Mnemonic::Call => {}
        _ => invalidate_unmodeled_writes(machine, instruction),
    }
    Ok(())
}

fn apply_call(machine: &mut Machine) {
    for register in [
        Register::RAX,
        Register::RCX,
        Register::RDX,
        Register::R8,
        Register::R9,
        Register::R10,
        Register::R11,
    ] {
        machine.write_register(register, Value::Unknown);
    }
    machine.write_register(Register::RAX, Value::CallResult);
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn verify_selector(
    pe: &PeImage<'_>,
    instructions: &BTreeMap<Rva, DecodedInstruction>,
    entry: Rva,
    call_site: Rva,
    dispatch_source: Rva,
    terminal_dispatch: Rva,
    selector_completion: Rva,
    selector_register: Register,
    vsp: Register,
    preserved: &[Register],
    selector: u8,
    max_steps: usize,
) -> Result<NativeCallSelectorOutcome, String> {
    let mut machine = Machine::new(vsp, preserved);
    let mut pc = entry;
    for _ in 0..max_steps {
        if pc == dispatch_source {
            if machine.call_count != 1 {
                return Err(format!(
                    "selector {selector} reaches dispatch decode after {} calls",
                    machine.call_count
                ));
            }
            for role in preserved.iter().copied() {
                if machine.read_register(role) != Value::RoleState(canonical(role)) {
                    return Err(format!(
                        "selector {selector} does not restore role {role:?} before dispatch"
                    ));
                }
            }
            if machine.read_register(Register::RSP) != Value::EntryRsp(0) {
                return Err(format!(
                    "selector {selector} does not restore native RSP exactly"
                ));
            }
            let Value::EntryVsp(vsp_delta) = machine.read_register(vsp) else {
                return Err(format!(
                    "selector {selector} does not restore a proven entry-VSP-relative pointer"
                ));
            };
            if machine.read_memory(Value::EntryVsp(vsp_delta)) != Value::CallResult {
                return Err(format!(
                    "selector {selector} does not place the native result at final VSP"
                ));
            }
            let register_count = usize::from(selector.min(4));
            let argument_registers = [Register::RCX, Register::RDX, Register::R8, Register::R9];
            let mut register_argument_offsets = Vec::new();
            for (index, register) in argument_registers
                .into_iter()
                .take(register_count)
                .enumerate()
            {
                let expected = i64::try_from((index + 1) * 8)
                    .map_err(|_| "register-argument offset overflows".to_owned())?;
                if machine.read_register(register) != Value::Unknown {
                    // Volatile argument registers are intentionally clobbered by the call.
                }
                register_argument_offsets.push(
                    i32::try_from(expected)
                        .map_err(|_| "register-argument offset exceeds i32".to_owned())?,
                );
            }
            if machine.stack_arguments.len() != usize::from(selector.saturating_sub(4)) {
                return Err(format!(
                    "selector {selector} marshals {} stack arguments, expected {}",
                    machine.stack_arguments.len(),
                    selector.saturating_sub(4)
                ));
            }
            return Ok(NativeCallSelectorOutcome {
                selector,
                vsp_delta: i32::try_from(vsp_delta)
                    .map_err(|_| "native-call VSP delta exceeds i32".to_owned())?,
                target_stack_offset: 0,
                register_argument_offsets,
                stack_argument_offsets: machine.stack_arguments,
            });
        }

        let decoded = instructions
            .get(&pc)
            .ok_or_else(|| format!("selector {selector} leaves the recovered CFG at {pc}"))?;
        let instruction = &decoded.instruction;
        execute_instruction(&mut machine, instruction)?;
        if pc == selector_completion {
            machine.selector_seeded = true;
            machine.write_register(selector_register, Value::Const(u64::from(selector)));
            for role in preserved.iter().copied() {
                machine.write_register(role, Value::RoleState(canonical(role)));
            }
        }

        pc = match instruction.flow_control() {
            FlowControl::Next => decoded
                .next_rva()
                .ok_or_else(|| "native-call instruction has no fallthrough".to_owned())?,
            FlowControl::UnconditionalBranch => decoded
                .direct_target_rva(pe)
                .ok_or_else(|| "native-call proof encountered an indirect early jump".to_owned())?,
            FlowControl::ConditionalBranch => {
                let taken =
                    evaluate_branch(instruction.mnemonic(), machine.flags).ok_or_else(|| {
                        format!(
                            "selector {selector} reaches an unresolved conditional branch at {pc}"
                        )
                    })?;
                if taken {
                    decoded.direct_target_rva(pe).ok_or_else(|| {
                        "native-call conditional branch has no direct target".to_owned()
                    })?
                } else {
                    decoded.next_rva().ok_or_else(|| {
                        "native-call conditional branch has no fallthrough".to_owned()
                    })?
                }
            }
            FlowControl::IndirectCall => {
                if pc != call_site || machine.call_count != 0 || !machine.selector_seeded {
                    return Err(format!(
                        "selector {selector} reaches an unexpected native call"
                    ));
                }
                if operand_value(&machine, instruction, 0) != Value::VmStack(0) {
                    return Err(format!(
                        "selector {selector} call target is not loaded from VM-stack offset zero"
                    ));
                }
                let register_count = usize::from(selector.min(4));
                for (index, register) in [Register::RCX, Register::RDX, Register::R8, Register::R9]
                    .into_iter()
                    .take(register_count)
                    .enumerate()
                {
                    let expected = i64::try_from((index + 1) * 8)
                        .map_err(|_| "register-argument offset overflows".to_owned())?;
                    if machine.read_register(register) != Value::VmStack(expected) {
                        return Err(format!(
                            "selector {selector} does not marshal argument {} from VM-stack offset {expected}",
                            index + 1
                        ));
                    }
                }
                machine.call_count += 1;
                apply_call(&mut machine);
                decoded
                    .next_rva()
                    .ok_or_else(|| "native call has no resume instruction".to_owned())?
            }
            FlowControl::IndirectBranch => {
                if pc == terminal_dispatch {
                    return Err(format!(
                        "selector {selector} reaches threaded dispatch before its decoded source"
                    ));
                }
                return Err(format!(
                    "selector {selector} reaches an unexpected indirect branch at {pc}"
                ));
            }
            other => {
                return Err(format!(
                    "selector {selector} reaches unsupported control flow {other:?} at {pc}"
                ));
            }
        };
    }
    Err(format!(
        "selector {selector} exceeds the native-call proof step limit {max_steps}"
    ))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn verify_native_call_outcomes(
    pe: &PeImage<'_>,
    instructions: &BTreeMap<Rva, DecodedInstruction>,
    entry: Rva,
    call_site: Rva,
    dispatch_source: Rva,
    terminal_dispatch: Rva,
    selector_completion: Rva,
    selector_register: Register,
    vsp: Register,
    preserved: &[Register],
    max_steps_per_selector: usize,
) -> Result<Vec<NativeCallSelectorOutcome>, String> {
    if preserved
        .iter()
        .copied()
        .chain([vsp])
        .any(|register| canonical(register) == canonical(selector_register))
    {
        return Err("selector register overlaps a live VM role".to_owned());
    }
    let mut outcomes = Vec::with_capacity(256);
    for selector in 0_u8..=u8::MAX {
        outcomes.push(verify_selector(
            pe,
            instructions,
            entry,
            call_site,
            dispatch_source,
            terminal_dispatch,
            selector_completion,
            selector_register,
            vsp,
            preserved,
            selector,
            max_steps_per_selector,
        )?);
    }
    Ok(outcomes)
}
