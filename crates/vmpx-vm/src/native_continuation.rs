use std::collections::BTreeMap;

use iced_x86::{
    FlowControl, Instruction, InstructionInfoFactory, Mnemonic, OpAccess, OpKind, Register,
};
use serde::Serialize;
use vmpx_disasm::{DisasmError, decode_one};
use vmpx_pe::{PeImage, Rva, Va};

use crate::{
    ExprKind, NativeCallImportResolution, NativeExitState, RecoveryStatus, X64NativeRegister,
    dispatch_extract::canonical, resolve_native_call_import,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeContinuationLimits {
    pub max_instructions: usize,
    pub max_stack_bytes: usize,
    pub max_c_string_bytes: usize,
}

impl Default for NativeContinuationLimits {
    fn default() -> Self {
        Self {
            max_instructions: 100_000,
            max_stack_bytes: 64 * 1024,
            max_c_string_bytes: 4 * 1024,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NativeObservedValue {
    Unknown,
    Concrete { value: u64 },
    StackRelative { offset: i64 },
    StackAffine { coefficient: i8, constant: u64 },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct NativeObservedCString {
    pub argument_register: String,
    pub stack_offset: i64,
    pub bytes: Vec<u8>,
    pub utf8: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct NativeObservedCall {
    pub call_site: Rva,
    pub target: NativeObservedValue,
    pub import: Option<NativeCallImportResolution>,
    pub arguments: BTreeMap<String, NativeObservedValue>,
    pub c_strings: Vec<NativeObservedCString>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct NativeContinuationReport {
    pub schema_version: u32,
    pub entry: Rva,
    pub status: RecoveryStatus,
    pub executed_instructions: usize,
    pub calls: Vec<NativeObservedCall>,
    /// Native RSP relative to its value on entry immediately before RET.
    pub return_stack_offset: Option<i64>,
    pub returns_to_unmodeled_caller: bool,
    pub evidence: Vec<String>,
    pub diagnostics: Vec<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct AffineValue {
    stack_coefficient: i8,
    constant: u64,
}

impl AffineValue {
    const fn concrete(value: u64) -> Self {
        Self {
            stack_coefficient: 0,
            constant: value,
        }
    }

    const fn stack(offset: u64) -> Self {
        Self {
            stack_coefficient: 1,
            constant: offset,
        }
    }

    fn add(self, right: Self) -> Option<Self> {
        let coefficient = self
            .stack_coefficient
            .checked_add(right.stack_coefficient)?;
        (-1..=1).contains(&coefficient).then_some(Self {
            stack_coefficient: coefficient,
            constant: self.constant.wrapping_add(right.constant),
        })
    }

    fn sub(self, right: Self) -> Option<Self> {
        let coefficient = self
            .stack_coefficient
            .checked_sub(right.stack_coefficient)?;
        (-1..=1).contains(&coefficient).then_some(Self {
            stack_coefficient: coefficient,
            constant: self.constant.wrapping_sub(right.constant),
        })
    }

    fn multiply(self, factor: u64) -> Option<Self> {
        if factor == 0 {
            return Some(Self::concrete(0));
        }
        (factor == 1).then_some(self)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AbstractValue {
    Unknown,
    Affine(AffineValue),
}

impl AbstractValue {
    const fn concrete(value: u64) -> Self {
        Self::Affine(AffineValue::concrete(value))
    }

    const fn stack(offset: u64) -> Self {
        Self::Affine(AffineValue::stack(offset))
    }

    fn as_concrete(self) -> Option<u64> {
        match self {
            Self::Affine(value) if value.stack_coefficient == 0 => Some(value.constant),
            _ => None,
        }
    }

    fn affine(self) -> Option<AffineValue> {
        match self {
            Self::Unknown => None,
            Self::Affine(value) => Some(value),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct FlagState {
    carry: Option<bool>,
    zero: Option<bool>,
}

struct ConcreteState {
    registers: BTreeMap<Register, AbstractValue>,
    stack: BTreeMap<u64, u8>,
    flags: FlagState,
}

impl ConcreteState {
    fn from_exit(exit: &NativeExitState) -> Self {
        let mut registers = BTreeMap::new();
        registers.insert(Register::RSP, AbstractValue::stack(0));
        for register in X64NativeRegister::ALL {
            let value = exit
                .registers
                .get(&register)
                .and_then(|expression| match expression.kind {
                    ExprKind::Const { value } => Some(AbstractValue::concrete(value)),
                    _ => None,
                })
                .unwrap_or(AbstractValue::Unknown);
            registers.insert(register.to_iced(), value);
        }
        Self {
            registers,
            stack: BTreeMap::new(),
            flags: FlagState::default(),
        }
    }

    fn register(&self, register: Register) -> AbstractValue {
        let full = canonical(register);
        let value = self
            .registers
            .get(&full)
            .copied()
            .unwrap_or(AbstractValue::Unknown);
        if register == full {
            return value;
        }
        let Some(concrete) = value.as_concrete() else {
            return AbstractValue::Unknown;
        };
        let shift = if matches!(
            register,
            Register::AH | Register::BH | Register::CH | Register::DH
        ) {
            8
        } else {
            0
        };
        let bits = register.size().saturating_mul(8);
        let mask = if bits >= 64 {
            u64::MAX
        } else {
            (1_u64 << bits) - 1
        };
        AbstractValue::concrete((concrete >> shift) & mask)
    }

    fn write_register(&mut self, register: Register, value: AbstractValue) {
        let full = canonical(register);
        if register == full {
            self.registers.insert(full, value);
            return;
        }
        if register.size() == 4 && full.size() == 8 {
            self.registers.insert(
                full,
                value.as_concrete().map_or(AbstractValue::Unknown, |value| {
                    AbstractValue::concrete(value & u64::from(u32::MAX))
                }),
            );
            return;
        }
        let Some(old) = self.register(full).as_concrete() else {
            self.registers.insert(full, AbstractValue::Unknown);
            return;
        };
        let Some(new) = value.as_concrete() else {
            self.registers.insert(full, AbstractValue::Unknown);
            return;
        };
        let shift = if matches!(
            register,
            Register::AH | Register::BH | Register::CH | Register::DH
        ) {
            8
        } else {
            0
        };
        let bits = register.size().saturating_mul(8);
        let low_mask = if bits >= 64 {
            u64::MAX
        } else {
            (1_u64 << bits) - 1
        };
        let mask = low_mask << shift;
        self.registers.insert(
            full,
            AbstractValue::concrete((old & !mask) | ((new << shift) & mask)),
        );
    }
}

fn signed_offset(value: u64) -> i64 {
    i64::from_ne_bytes(value.to_ne_bytes())
}

fn observed(value: AbstractValue) -> NativeObservedValue {
    match value {
        AbstractValue::Unknown => NativeObservedValue::Unknown,
        AbstractValue::Affine(value) if value.stack_coefficient == 0 => {
            NativeObservedValue::Concrete {
                value: value.constant,
            }
        }
        AbstractValue::Affine(value) if value.stack_coefficient == 1 => {
            NativeObservedValue::StackRelative {
                offset: signed_offset(value.constant),
            }
        }
        AbstractValue::Affine(value) => NativeObservedValue::StackAffine {
            coefficient: value.stack_coefficient,
            constant: value.constant,
        },
    }
}

fn operand_width(instruction: &Instruction, operand: u32) -> usize {
    match instruction.op_kind(operand) {
        OpKind::Register => instruction.op_register(operand).size(),
        OpKind::Memory => instruction.memory_size().size(),
        _ => 8,
    }
}

fn mask_for_width(width: usize) -> u64 {
    if width >= 8 {
        u64::MAX
    } else {
        (1_u64 << (width * 8)) - 1
    }
}

fn effective_address(state: &ConcreteState, instruction: &Instruction) -> AbstractValue {
    if instruction.is_ip_rel_memory_operand() {
        return AbstractValue::concrete(instruction.ip_rel_memory_address());
    }
    let mut address = AffineValue::concrete(instruction.memory_displacement64());
    if instruction.memory_base() != Register::None {
        let Some(base) = state.register(instruction.memory_base()).affine() else {
            return AbstractValue::Unknown;
        };
        let Some(sum) = address.add(base) else {
            return AbstractValue::Unknown;
        };
        address = sum;
    }
    if instruction.memory_index() != Register::None {
        let Some(index) = state.register(instruction.memory_index()).affine() else {
            return AbstractValue::Unknown;
        };
        let Some(index) = index.multiply(u64::from(instruction.memory_index_scale())) else {
            return AbstractValue::Unknown;
        };
        let Some(sum) = address.add(index) else {
            return AbstractValue::Unknown;
        };
        address = sum;
    }
    AbstractValue::Affine(address)
}

fn read_image(pe: &PeImage<'_>, va: u64, width: usize) -> Option<u64> {
    if !(1..=8).contains(&width) {
        return None;
    }
    let rva = pe.optional.image_base.checked_rva_of(Va(va))?;
    let offset = usize::try_from(pe.rva_to_file_offset(rva).ok()?.get()).ok()?;
    let bytes = pe.data().get(offset..offset.checked_add(width)?)?;
    Some(
        bytes
            .iter()
            .enumerate()
            .fold(0_u64, |value, (index, byte)| {
                value | (u64::from(*byte) << (index * 8))
            }),
    )
}

fn read_memory(
    pe: &PeImage<'_>,
    state: &ConcreteState,
    address: AbstractValue,
    width: usize,
) -> AbstractValue {
    let Some(address) = address.affine() else {
        return AbstractValue::Unknown;
    };
    if address.stack_coefficient == 0 {
        return read_image(pe, address.constant, width)
            .map_or(AbstractValue::Unknown, AbstractValue::concrete);
    }
    if address.stack_coefficient != 1 || !(1..=8).contains(&width) {
        return AbstractValue::Unknown;
    }
    let mut value = 0_u64;
    for index in 0..width {
        let Some(byte) = state
            .stack
            .get(&address.constant.wrapping_add(index as u64))
        else {
            return AbstractValue::Unknown;
        };
        value |= u64::from(*byte) << (index * 8);
    }
    AbstractValue::concrete(value)
}

fn write_memory(
    state: &mut ConcreteState,
    address: AbstractValue,
    width: usize,
    value: AbstractValue,
    limits: NativeContinuationLimits,
) -> bool {
    let (Some(address), Some(value)) = (address.affine(), value.as_concrete()) else {
        return false;
    };
    if address.stack_coefficient != 1 || !(1..=8).contains(&width) {
        return false;
    }
    let offset = signed_offset(address.constant).unsigned_abs();
    if offset > u64::try_from(limits.max_stack_bytes).unwrap_or(u64::MAX) {
        return false;
    }
    for index in 0..width {
        state.stack.insert(
            address.constant.wrapping_add(index as u64),
            ((value >> (index * 8)) & 0xFF) as u8,
        );
    }
    true
}

fn operand_value(
    pe: &PeImage<'_>,
    state: &ConcreteState,
    instruction: &Instruction,
    operand: u32,
) -> AbstractValue {
    match instruction.op_kind(operand) {
        OpKind::Register => state.register(instruction.op_register(operand)),
        OpKind::Memory => read_memory(
            pe,
            state,
            effective_address(state, instruction),
            instruction.memory_size().size(),
        ),
        OpKind::Immediate8
        | OpKind::Immediate8_2nd
        | OpKind::Immediate16
        | OpKind::Immediate32
        | OpKind::Immediate64
        | OpKind::Immediate8to16
        | OpKind::Immediate8to32
        | OpKind::Immediate8to64
        | OpKind::Immediate32to64 => instruction
            .try_immediate(operand)
            .ok()
            .map_or(AbstractValue::Unknown, AbstractValue::concrete),
        _ => AbstractValue::Unknown,
    }
}

fn write_operand(
    state: &mut ConcreteState,
    instruction: &Instruction,
    operand: u32,
    value: AbstractValue,
    limits: NativeContinuationLimits,
) -> bool {
    match instruction.op_kind(operand) {
        OpKind::Register => {
            state.write_register(instruction.op_register(operand), value);
            true
        }
        OpKind::Memory => write_memory(
            state,
            effective_address(state, instruction),
            instruction.memory_size().size(),
            value,
            limits,
        ),
        _ => false,
    }
}

fn binary_affine(left: AbstractValue, right: AbstractValue, operation: Mnemonic) -> AbstractValue {
    let (Some(left), Some(right)) = (left.affine(), right.affine()) else {
        return AbstractValue::Unknown;
    };
    let value = match operation {
        Mnemonic::Add => left.add(right),
        Mnemonic::Sub => left.sub(right),
        _ => None,
    };
    value.map_or(AbstractValue::Unknown, AbstractValue::Affine)
}

fn c_string_at_stack(
    state: &ConcreteState,
    register: Register,
    limits: NativeContinuationLimits,
) -> Option<NativeObservedCString> {
    let address = state.register(register).affine()?;
    if address.stack_coefficient != 1 {
        return None;
    }
    let mut bytes = Vec::new();
    for index in 0..limits.max_c_string_bytes {
        let byte = *state
            .stack
            .get(&address.constant.wrapping_add(index as u64))?;
        if byte == 0 {
            return Some(NativeObservedCString {
                argument_register: format!("{register:?}"),
                stack_offset: signed_offset(address.constant),
                utf8: String::from_utf8(bytes.clone()).ok(),
                bytes,
            });
        }
        bytes.push(byte);
    }
    None
}

fn access_writes(access: OpAccess) -> bool {
    matches!(
        access,
        OpAccess::Write | OpAccess::CondWrite | OpAccess::ReadWrite | OpAccess::ReadCondWrite
    )
}

fn invalidate_unsupported(
    state: &mut ConcreteState,
    instruction: &Instruction,
    factory: &mut InstructionInfoFactory,
) -> Result<(), String> {
    let info = factory.info(instruction);
    if !info.used_memory().is_empty() {
        return Err(format!(
            "unsupported instruction has an unmodeled memory access: {:?}",
            instruction.mnemonic()
        ));
    }
    for register in info.used_registers() {
        if !access_writes(register.access()) {
            continue;
        }
        let full = canonical(register.register());
        if full == Register::RSP {
            return Err(format!(
                "unsupported instruction changes native RSP: {:?}",
                instruction.mnemonic()
            ));
        }
        if !matches!(full, Register::RIP | Register::EIP | Register::None) {
            state.registers.insert(full, AbstractValue::Unknown);
        }
    }
    if instruction.rflags_written()
        | instruction.rflags_cleared()
        | instruction.rflags_set()
        | instruction.rflags_undefined()
        != 0
    {
        state.flags = FlagState::default();
    }
    Ok(())
}

fn failed(
    entry: Rva,
    executed_instructions: usize,
    calls: Vec<NativeObservedCall>,
    diagnostic: String,
) -> NativeContinuationReport {
    NativeContinuationReport {
        schema_version: 1,
        entry,
        status: RecoveryStatus::Failed,
        executed_instructions,
        calls,
        return_stack_offset: None,
        returns_to_unmodeled_caller: false,
        evidence: Vec::new(),
        diagnostics: vec![diagnostic],
    }
}

/// Bounded concrete/affine execution for a native continuation reached with an
/// exact VM-exit register state. The stack origin stays symbolic, allowing
/// stack-local buffers to be interpreted without inventing a process address.
/// Unsupported instructions invalidate only their proven register/flag writes;
/// unmodeled memory writes, unknown branches, and resource exhaustion fail
/// closed.
///
/// # Errors
///
/// Returns a disassembly error when the entry instruction cannot be decoded.
/// Later unsupported or malformed behavior is retained as a failed report.
#[allow(clippy::too_many_lines)]
pub fn analyze_native_continuation(
    pe: &PeImage<'_>,
    entry: Rva,
    initial: &NativeExitState,
    limits: NativeContinuationLimits,
) -> Result<NativeContinuationReport, DisasmError> {
    decode_one(pe, entry)?;
    let mut state = ConcreteState::from_exit(initial);
    let mut current = entry;
    let mut executed = 0_usize;
    let mut calls = Vec::new();
    let mut factory = InstructionInfoFactory::new();

    while executed < limits.max_instructions {
        let decoded = match decode_one(pe, current) {
            Ok(decoded) => decoded,
            Err(error) => {
                return Ok(failed(entry, executed, calls, error.to_string()));
            }
        };
        executed += 1;
        let instruction = &decoded.instruction;
        let next = decoded.next_rva();
        match instruction.mnemonic() {
            Mnemonic::Mov => {
                let value = operand_value(pe, &state, instruction, 1);
                if !write_operand(&mut state, instruction, 0, value, limits) {
                    return Ok(failed(
                        entry,
                        executed,
                        calls,
                        format!(
                            "unmodeled MOV destination at {}: destination {:?}, value {:?}",
                            decoded.rva,
                            (instruction.op0_kind() == OpKind::Memory)
                                .then(|| effective_address(&state, instruction)),
                            value
                        ),
                    ));
                }
            }
            Mnemonic::Lea => {
                let value = effective_address(&state, instruction);
                if instruction.op0_kind() != OpKind::Register {
                    return Ok(failed(
                        entry,
                        executed,
                        calls,
                        format!("unmodeled LEA destination at {}", decoded.rva),
                    ));
                }
                state.write_register(instruction.op0_register(), value);
            }
            Mnemonic::Add | Mnemonic::Sub => {
                let width = operand_width(instruction, 0);
                let left = operand_value(pe, &state, instruction, 0);
                let right = operand_value(pe, &state, instruction, 1);
                let result =
                    if let (Some(left), Some(right)) = (left.as_concrete(), right.as_concrete()) {
                        let mask = mask_for_width(width);
                        let raw = if instruction.mnemonic() == Mnemonic::Add {
                            left.wrapping_add(right)
                        } else {
                            left.wrapping_sub(right)
                        };
                        let result = raw & mask;
                        state.flags.zero = Some(result == 0);
                        state.flags.carry = Some(if instruction.mnemonic() == Mnemonic::Add {
                            result < (left & mask)
                        } else {
                            (left & mask) < (right & mask)
                        });
                        AbstractValue::concrete(result)
                    } else {
                        state.flags = FlagState::default();
                        binary_affine(left, right, instruction.mnemonic())
                    };
                if !write_operand(&mut state, instruction, 0, result, limits) {
                    return Ok(failed(
                        entry,
                        executed,
                        calls,
                        format!("unmodeled arithmetic destination at {}", decoded.rva),
                    ));
                }
            }
            Mnemonic::Xor => {
                let width = operand_width(instruction, 0);
                let left = operand_value(pe, &state, instruction, 0).as_concrete();
                let right = operand_value(pe, &state, instruction, 1).as_concrete();
                let same_register = instruction.op0_kind() == OpKind::Register
                    && instruction.op1_kind() == OpKind::Register
                    && instruction.op0_register() == instruction.op1_register();
                let value = if same_register {
                    Some(AbstractValue::concrete(0))
                } else {
                    left.zip(right).map(|(left, right)| {
                        AbstractValue::concrete((left ^ right) & mask_for_width(width))
                    })
                };
                state.flags.carry = Some(false);
                state.flags.zero = value
                    .and_then(AbstractValue::as_concrete)
                    .map(|value| value == 0);
                if !write_operand(
                    &mut state,
                    instruction,
                    0,
                    value.unwrap_or(AbstractValue::Unknown),
                    limits,
                ) {
                    return Ok(failed(
                        entry,
                        executed,
                        calls,
                        format!("unmodeled XOR destination at {}", decoded.rva),
                    ));
                }
            }
            Mnemonic::Rol => {
                let width = operand_width(instruction, 0);
                let bits = u32::try_from(width.saturating_mul(8)).unwrap_or(64);
                let value = operand_value(pe, &state, instruction, 0).as_concrete();
                let count = operand_value(pe, &state, instruction, 1)
                    .as_concrete()
                    .and_then(|value| u32::try_from(value & 0xFF).ok());
                let result = (bits != 0)
                    .then_some(())
                    .and_then(|()| value.zip(count))
                    .map(|(value, count)| {
                        let mask = mask_for_width(width);
                        let count = count % bits;
                        let value = value & mask;
                        let rotated = if count == 0 {
                            value
                        } else {
                            ((value << count) | (value >> (bits - count))) & mask
                        };
                        AbstractValue::concrete(rotated)
                    });
                if !write_operand(
                    &mut state,
                    instruction,
                    0,
                    result.unwrap_or(AbstractValue::Unknown),
                    limits,
                ) {
                    return Ok(failed(
                        entry,
                        executed,
                        calls,
                        format!("unmodeled ROL destination at {}", decoded.rva),
                    ));
                }
            }
            Mnemonic::Inc => {
                let width = operand_width(instruction, 0);
                let result = operand_value(pe, &state, instruction, 0)
                    .as_concrete()
                    .map_or(AbstractValue::Unknown, |value| {
                        AbstractValue::concrete(value.wrapping_add(1) & mask_for_width(width))
                    });
                state.flags.zero = result.as_concrete().map(|value| value == 0);
                if !write_operand(&mut state, instruction, 0, result, limits) {
                    return Ok(failed(
                        entry,
                        executed,
                        calls,
                        format!("unmodeled INC destination at {}", decoded.rva),
                    ));
                }
            }
            Mnemonic::Cmp => {
                let width = operand_width(instruction, 0);
                let left = operand_value(pe, &state, instruction, 0).as_concrete();
                let right = operand_value(pe, &state, instruction, 1).as_concrete();
                state.flags = left
                    .zip(right)
                    .map_or_else(FlagState::default, |(left, right)| {
                        let mask = mask_for_width(width);
                        let left = left & mask;
                        let right = right & mask;
                        FlagState {
                            carry: Some(left < right),
                            zero: Some(left == right),
                        }
                    });
            }
            Mnemonic::Stc => state.flags.carry = Some(true),
            Mnemonic::Clc => state.flags.carry = Some(false),
            Mnemonic::Nop => {}
            Mnemonic::Jmp => {
                let Some(target) = decoded.direct_target_rva(pe) else {
                    return Ok(failed(
                        entry,
                        executed,
                        calls,
                        format!("indirect JMP at {}", decoded.rva),
                    ));
                };
                current = target;
                continue;
            }
            Mnemonic::Je | Mnemonic::Jne | Mnemonic::Jb | Mnemonic::Jae => {
                let condition = match instruction.mnemonic() {
                    Mnemonic::Je => state.flags.zero,
                    Mnemonic::Jne => state.flags.zero.map(|value| !value),
                    Mnemonic::Jb => state.flags.carry,
                    Mnemonic::Jae => state.flags.carry.map(|value| !value),
                    _ => None,
                };
                let Some(taken) = condition else {
                    return Ok(failed(
                        entry,
                        executed,
                        calls,
                        format!("conditional branch at {} has unknown flags", decoded.rva),
                    ));
                };
                if taken {
                    let Some(target) = decoded.direct_target_rva(pe) else {
                        return Ok(failed(
                            entry,
                            executed,
                            calls,
                            format!("conditional branch at {} has no direct target", decoded.rva),
                        ));
                    };
                    current = target;
                    continue;
                }
            }
            Mnemonic::Call => {
                let target = operand_value(pe, &state, instruction, 0);
                let mut arguments = BTreeMap::new();
                let mut c_strings = Vec::new();
                for register in [Register::RCX, Register::RDX, Register::R8, Register::R9] {
                    arguments.insert(format!("{register:?}"), observed(state.register(register)));
                    if let Some(string) = c_string_at_stack(&state, register, limits) {
                        c_strings.push(string);
                    }
                }
                let import = target.as_concrete().map(|target| {
                    resolve_native_call_import(&pe.report().imports, pe.optional.image_base, target)
                });
                calls.push(NativeObservedCall {
                    call_site: decoded.rva,
                    target: observed(target),
                    import,
                    arguments,
                    c_strings,
                });
                for register in [
                    Register::RAX,
                    Register::RCX,
                    Register::RDX,
                    Register::R8,
                    Register::R9,
                    Register::R10,
                    Register::R11,
                ] {
                    state.registers.insert(register, AbstractValue::Unknown);
                }
                state.flags = FlagState::default();
            }
            Mnemonic::Ret => {
                let return_stack_offset = state
                    .register(Register::RSP)
                    .affine()
                    .filter(|value| value.stack_coefficient == 1)
                    .map(|value| signed_offset(value.constant));
                let status = if return_stack_offset.is_some() {
                    RecoveryStatus::Recovered
                } else {
                    RecoveryStatus::Failed
                };
                return Ok(NativeContinuationReport {
                    schema_version: 1,
                    entry,
                    status,
                    executed_instructions: executed,
                    calls,
                    return_stack_offset,
                    returns_to_unmodeled_caller: return_stack_offset.is_some(),
                    evidence: vec![
                        "bounded native execution reached RET without an unknown branch".to_owned(),
                        "native stack addresses remained affine to the entry stack".to_owned(),
                    ],
                    diagnostics: if return_stack_offset == Some(0) {
                        Vec::new()
                    } else {
                        vec![
                            "native continuation does not restore its entry RSP before RET"
                                .to_owned(),
                        ]
                    },
                });
            }
            _ => {
                if matches!(instruction.flow_control(), FlowControl::ConditionalBranch) {
                    return Ok(failed(
                        entry,
                        executed,
                        calls,
                        format!("unsupported conditional branch at {}", decoded.rva),
                    ));
                }
                if let Err(diagnostic) =
                    invalidate_unsupported(&mut state, instruction, &mut factory)
                {
                    return Ok(failed(entry, executed, calls, diagnostic));
                }
            }
        }
        let Some(next) = next else {
            return Ok(failed(
                entry,
                executed,
                calls,
                format!(
                    "instruction at {} has no representable fallthrough",
                    decoded.rva
                ),
            ));
        };
        current = next;
    }
    Ok(failed(
        entry,
        executed,
        calls,
        format!(
            "native continuation exceeded the {}-instruction limit",
            limits.max_instructions
        ),
    ))
}

#[cfg(test)]
mod tests {
    use iced_x86::code_asm::*;

    use super::*;
    use crate::{BitWidth, Expr};

    const IMAGE_BASE: u64 = 0x1_4000_0000;
    const CODE_RVA: u32 = 0x1000;
    const SOURCE_RVA: u32 = 0x1800;

    fn executable_pe64(code: &[u8], source: &[u8]) -> Vec<u8> {
        let mut data = vec![0_u8; 0x1200];
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
        data[optional + 16..optional + 20].copy_from_slice(&CODE_RVA.to_le_bytes());
        data[optional + 20..optional + 24].copy_from_slice(&CODE_RVA.to_le_bytes());
        data[optional + 24..optional + 32].copy_from_slice(&IMAGE_BASE.to_le_bytes());
        data[optional + 32..optional + 36].copy_from_slice(&0x1000_u32.to_le_bytes());
        data[optional + 36..optional + 40].copy_from_slice(&0x200_u32.to_le_bytes());
        data[optional + 56..optional + 60].copy_from_slice(&0x3000_u32.to_le_bytes());
        data[optional + 60..optional + 64].copy_from_slice(&0x200_u32.to_le_bytes());
        data[optional + 68..optional + 70].copy_from_slice(&3_u16.to_le_bytes());
        data[optional + 108..optional + 112].copy_from_slice(&16_u32.to_le_bytes());
        let section = optional + 0xF0;
        data[section..section + 5].copy_from_slice(b".text");
        data[section + 8..section + 12].copy_from_slice(&0x1000_u32.to_le_bytes());
        data[section + 12..section + 16].copy_from_slice(&CODE_RVA.to_le_bytes());
        data[section + 16..section + 20].copy_from_slice(&0x1000_u32.to_le_bytes());
        data[section + 20..section + 24].copy_from_slice(&0x200_u32.to_le_bytes());
        data[section + 36..section + 40].copy_from_slice(&0x6000_0020_u32.to_le_bytes());
        data[0x200..0x200 + code.len()].copy_from_slice(code);
        let source_offset = 0x200 + usize::try_from(SOURCE_RVA - CODE_RVA).expect("offset");
        data[source_offset..source_offset + source.len()].copy_from_slice(source);
        data
    }

    fn encoded_source(seed: u32, plaintext: &[u8]) -> Vec<u8> {
        plaintext
            .iter()
            .chain(std::iter::once(&0))
            .enumerate()
            .map(|(index, byte)| {
                let rotation = u32::try_from(index & 31).expect("rotation");
                let key = seed.rotate_left(rotation).to_le_bytes()[0]
                    .wrapping_add(index.to_le_bytes()[0]);
                key ^ byte
            })
            .collect()
    }

    fn decoder_code(seed: u32, maximum: u64, call_target: u64) -> Vec<u8> {
        let mut assembler = CodeAssembler::new(64).expect("x64 assembler");
        let mut loop_label = assembler.create_label();
        let mut done = assembler.create_label();
        assembler.sub(rsp, 0x40).expect("stack frame");
        assembler
            .lea(rax, qword_ptr(rsp + 0x20))
            .expect("destination");
        assembler.mov(r9, rcx).expect("source");
        assembler.sub(r9, rax).expect("source delta");
        assembler.xor(r8d, r8d).expect("index");
        assembler.set_label(&mut loop_label).expect("loop label");
        assembler
            .lea(rdx, qword_ptr(rsp + r8 + 0x20))
            .expect("destination byte");
        assembler.mov(ecx, r8d).expect("rotation count");
        assembler.mov(eax, seed).expect("seed");
        assembler.rol(eax, cl).expect("rotate");
        assembler.add(al, r8b).expect("key index");
        assembler.xor(al, byte_ptr(r9 + rdx)).expect("decode byte");
        assembler.mov(byte_ptr(rdx), al).expect("store byte");
        assembler.je(done).expect("zero terminator");
        assembler.inc(r8).expect("advance");
        assembler
            .cmp(r8, i32::try_from(maximum).expect("test bound fits i32"))
            .expect("bound");
        assembler.jb(loop_label).expect("loop");
        assembler.set_label(&mut done).expect("done label");
        assembler
            .lea(rcx, qword_ptr(rsp + 0x20))
            .expect("call argument");
        assembler.mov(rax, call_target).expect("call target");
        assembler.call(rax).expect("indirect call");
        assembler.add(rsp, 0x40).expect("restore stack");
        assembler.ret().expect("return");
        assembler
            .assemble(IMAGE_BASE + u64::from(CODE_RVA))
            .expect("assemble decoder")
    }

    #[test]
    fn executes_mutated_static_string_decoder_and_call_without_a_signature() {
        let seed = 0xA1B2_C3D4;
        let plaintext = b"generic.dll";
        let encoded = encoded_source(seed, plaintext);
        let code = decoder_code(seed, 0x80, 0x1234_5678_9ABC_DEF0);
        let bytes = executable_pe64(&code, &encoded);
        let pe = PeImage::parse(&bytes).expect("valid synthetic PE");
        let mut registers = BTreeMap::new();
        registers.insert(
            X64NativeRegister::Rcx,
            Expr::constant(BitWidth::W64, IMAGE_BASE + u64::from(SOURCE_RVA)),
        );
        let initial = NativeExitState {
            schema_version: 1,
            registers,
            rflags: Expr::constant(BitWidth::W64, 0),
            return_address: Expr::constant(BitWidth::W64, IMAGE_BASE + u64::from(CODE_RVA)),
            frame_size: 0,
        };
        let report = analyze_native_continuation(
            &pe,
            Rva(CODE_RVA),
            &initial,
            NativeContinuationLimits::default(),
        )
        .expect("entry must decode");
        assert_eq!(report.status, RecoveryStatus::Recovered, "{report:#?}");
        assert_eq!(report.return_stack_offset, Some(0));
        let [call] = report.calls.as_slice() else {
            panic!("one call expected: {report:#?}")
        };
        assert_eq!(
            call.target,
            NativeObservedValue::Concrete {
                value: 0x1234_5678_9ABC_DEF0
            }
        );
        let string = call
            .c_strings
            .iter()
            .find(|string| string.argument_register == "RCX")
            .unwrap_or_else(|| panic!("RCX C string expected: {call:#?}"));
        assert_eq!(string.bytes, plaintext);
        assert_eq!(string.utf8.as_deref(), Some("generic.dll"));
    }

    #[test]
    fn unknown_branch_fails_closed() {
        let mut assembler = CodeAssembler::new(64).expect("x64 assembler");
        let mut target = assembler.create_label();
        assembler.je(target).expect("unknown conditional");
        assembler.ret().expect("fallthrough");
        assembler.set_label(&mut target).expect("target");
        assembler.ret().expect("target return");
        let code = assembler
            .assemble(IMAGE_BASE + u64::from(CODE_RVA))
            .expect("assemble branch");
        let bytes = executable_pe64(&code, &[]);
        let pe = PeImage::parse(&bytes).expect("valid synthetic PE");
        let initial = NativeExitState {
            schema_version: 1,
            registers: BTreeMap::new(),
            rflags: Expr::register(BitWidth::W64, "unknown_flags"),
            return_address: Expr::register(BitWidth::W64, "unknown_return"),
            frame_size: 0,
        };
        let report = analyze_native_continuation(
            &pe,
            Rva(CODE_RVA),
            &initial,
            NativeContinuationLimits::default(),
        )
        .expect("entry must decode");
        assert_eq!(report.status, RecoveryStatus::Failed);
        assert!(report.diagnostics[0].contains("unknown flags"));
    }
}
