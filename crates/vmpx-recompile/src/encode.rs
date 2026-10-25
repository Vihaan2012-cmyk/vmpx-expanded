use crate::abi::{ValidatedFunction, spill_displacement, validate};
use crate::{
    BinaryOp, BinarySource, CallTarget, CompileError, Condition, Function, Label, NativeAddress,
    Operation, PhysicalRegister, ResizeOp, ShiftCount, ShiftOp, UnaryOp, VirtualRegister, Width,
};
use iced_x86::code_asm::{
    AsmRegister8, AsmRegister16, AsmRegister32, AsmRegister64, CodeAssembler, byte_ptr, dword_ptr,
    qword_ptr,
    registers::gpr8::{
        al, bl, bpl, cl, dil, dl, r8b, r9b, r10b, r11b, r12b, r13b, r14b, r15b, sil,
    },
    registers::gpr16::{ax, bp, bx, cx, di, dx, r8w, r9w, r10w, r11w, r12w, r13w, r14w, r15w, si},
    registers::gpr32::{
        eax, ebp, ebx, ecx, edi, edx, esi, r8d, r9d, r10d, r11d, r12d, r13d, r14d, r15d,
    },
    registers::gpr64::{
        r8, r9, r10, r11, r12, r13, r14, r15, rax, rbp, rbx, rcx, rdi, rdx, rsi, rsp,
    },
    word_ptr,
};
use std::collections::BTreeMap;

/// Native address assigned to the first output byte.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CodeAddress(pub u64);

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompiledFunction {
    pub address: CodeAddress,
    pub bytes: Vec<u8>,
    pub label_offsets: BTreeMap<Label, u32>,
    pub stack_frame_size: u32,
}

#[derive(Clone, Copy, Debug)]
struct Fixup {
    displacement_offset: usize,
    instruction_end: usize,
    target: Label,
}

fn machine_register(register: PhysicalRegister) -> AsmRegister64 {
    match register {
        PhysicalRegister::Rax => rax,
        PhysicalRegister::Rcx => rcx,
        PhysicalRegister::Rdx => rdx,
        PhysicalRegister::Rbx => rbx,
        PhysicalRegister::Rbp => rbp,
        PhysicalRegister::Rsi => rsi,
        PhysicalRegister::Rdi => rdi,
        PhysicalRegister::R8 => r8,
        PhysicalRegister::R9 => r9,
        PhysicalRegister::R10 => r10,
        PhysicalRegister::R11 => r11,
        PhysicalRegister::R12 => r12,
        PhysicalRegister::R13 => r13,
        PhysicalRegister::R14 => r14,
        PhysicalRegister::R15 => r15,
    }
}

#[derive(Clone, Copy)]
enum MachineRegister {
    W8(AsmRegister8),
    W16(AsmRegister16),
    W32(AsmRegister32),
    W64(AsmRegister64),
}

fn machine_register_width(register: PhysicalRegister, width: Width) -> MachineRegister {
    match (register, width) {
        (PhysicalRegister::Rax, Width::W8) => MachineRegister::W8(al),
        (PhysicalRegister::Rcx, Width::W8) => MachineRegister::W8(cl),
        (PhysicalRegister::Rdx, Width::W8) => MachineRegister::W8(dl),
        (PhysicalRegister::Rbx, Width::W8) => MachineRegister::W8(bl),
        (PhysicalRegister::Rbp, Width::W8) => MachineRegister::W8(bpl),
        (PhysicalRegister::Rsi, Width::W8) => MachineRegister::W8(sil),
        (PhysicalRegister::Rdi, Width::W8) => MachineRegister::W8(dil),
        (PhysicalRegister::R8, Width::W8) => MachineRegister::W8(r8b),
        (PhysicalRegister::R9, Width::W8) => MachineRegister::W8(r9b),
        (PhysicalRegister::R10, Width::W8) => MachineRegister::W8(r10b),
        (PhysicalRegister::R11, Width::W8) => MachineRegister::W8(r11b),
        (PhysicalRegister::R12, Width::W8) => MachineRegister::W8(r12b),
        (PhysicalRegister::R13, Width::W8) => MachineRegister::W8(r13b),
        (PhysicalRegister::R14, Width::W8) => MachineRegister::W8(r14b),
        (PhysicalRegister::R15, Width::W8) => MachineRegister::W8(r15b),
        (PhysicalRegister::Rax, Width::W16) => MachineRegister::W16(ax),
        (PhysicalRegister::Rcx, Width::W16) => MachineRegister::W16(cx),
        (PhysicalRegister::Rdx, Width::W16) => MachineRegister::W16(dx),
        (PhysicalRegister::Rbx, Width::W16) => MachineRegister::W16(bx),
        (PhysicalRegister::Rbp, Width::W16) => MachineRegister::W16(bp),
        (PhysicalRegister::Rsi, Width::W16) => MachineRegister::W16(si),
        (PhysicalRegister::Rdi, Width::W16) => MachineRegister::W16(di),
        (PhysicalRegister::R8, Width::W16) => MachineRegister::W16(r8w),
        (PhysicalRegister::R9, Width::W16) => MachineRegister::W16(r9w),
        (PhysicalRegister::R10, Width::W16) => MachineRegister::W16(r10w),
        (PhysicalRegister::R11, Width::W16) => MachineRegister::W16(r11w),
        (PhysicalRegister::R12, Width::W16) => MachineRegister::W16(r12w),
        (PhysicalRegister::R13, Width::W16) => MachineRegister::W16(r13w),
        (PhysicalRegister::R14, Width::W16) => MachineRegister::W16(r14w),
        (PhysicalRegister::R15, Width::W16) => MachineRegister::W16(r15w),
        (PhysicalRegister::Rax, Width::W32) => MachineRegister::W32(eax),
        (PhysicalRegister::Rcx, Width::W32) => MachineRegister::W32(ecx),
        (PhysicalRegister::Rdx, Width::W32) => MachineRegister::W32(edx),
        (PhysicalRegister::Rbx, Width::W32) => MachineRegister::W32(ebx),
        (PhysicalRegister::Rbp, Width::W32) => MachineRegister::W32(ebp),
        (PhysicalRegister::Rsi, Width::W32) => MachineRegister::W32(esi),
        (PhysicalRegister::Rdi, Width::W32) => MachineRegister::W32(edi),
        (PhysicalRegister::R8, Width::W32) => MachineRegister::W32(r8d),
        (PhysicalRegister::R9, Width::W32) => MachineRegister::W32(r9d),
        (PhysicalRegister::R10, Width::W32) => MachineRegister::W32(r10d),
        (PhysicalRegister::R11, Width::W32) => MachineRegister::W32(r11d),
        (PhysicalRegister::R12, Width::W32) => MachineRegister::W32(r12d),
        (PhysicalRegister::R13, Width::W32) => MachineRegister::W32(r13d),
        (PhysicalRegister::R14, Width::W32) => MachineRegister::W32(r14d),
        (PhysicalRegister::R15, Width::W32) => MachineRegister::W32(r15d),
        (register, Width::W64) => MachineRegister::W64(machine_register(register)),
    }
}

fn assigned(
    validated: &ValidatedFunction,
    register: VirtualRegister,
) -> Result<MachineRegister, CompileError> {
    validated
        .registers
        .get(&register.id)
        .copied()
        .map(|physical| machine_register_width(physical, register.width))
        .ok_or(CompileError::UnassignedRegister { register })
}

fn assigned_qword(
    validated: &ValidatedFunction,
    register: VirtualRegister,
) -> Result<AsmRegister64, CompileError> {
    match assigned(validated, register)? {
        MachineRegister::W64(register) => Ok(register),
        MachineRegister::W8(_) | MachineRegister::W16(_) | MachineRegister::W32(_) => {
            Err(CompileError::AddressBaseNotQword { register })
        }
    }
}

fn assigned_view(
    validated: &ValidatedFunction,
    register: VirtualRegister,
    width: Width,
) -> Result<MachineRegister, CompileError> {
    validated
        .registers
        .get(&register.id)
        .copied()
        .map(|physical| machine_register_width(physical, width))
        .ok_or(CompileError::UnassignedRegister { register })
}

fn append_assembled(
    output: &mut Vec<u8>,
    address: CodeAddress,
    operation_index: usize,
    build: impl FnOnce(&mut CodeAssembler) -> Result<(), iced_x86::IcedError>,
) -> Result<(), CompileError> {
    let mut assembler = CodeAssembler::new(64).map_err(|error| CompileError::Encoding {
        operation_index,
        message: error.to_string(),
    })?;
    build(&mut assembler).map_err(|error| CompileError::Encoding {
        operation_index,
        message: error.to_string(),
    })?;
    let offset = u64::try_from(output.len()).map_err(|_| CompileError::CodeSizeOverflow)?;
    let instruction_address = address
        .0
        .checked_add(offset)
        .ok_or(CompileError::CodeSizeOverflow)?;
    let bytes =
        assembler
            .assemble(instruction_address)
            .map_err(|error| CompileError::Encoding {
                operation_index,
                message: error.to_string(),
            })?;
    output
        .len()
        .checked_add(bytes.len())
        .ok_or(CompileError::CodeSizeOverflow)?;
    output.extend_from_slice(&bytes);
    Ok(())
}

#[derive(Clone, Copy)]
enum ResolvedBinarySource {
    Register(MachineRegister),
    Immediate(i32),
}

fn resolve_source(
    source: BinarySource,
    validated: &ValidatedFunction,
) -> Result<ResolvedBinarySource, CompileError> {
    match source {
        BinarySource::Register(register) => {
            assigned(validated, register).map(ResolvedBinarySource::Register)
        }
        BinarySource::Immediate(value) => Ok(ResolvedBinarySource::Immediate(value)),
    }
}

macro_rules! emit_binary_register {
    ($assembler:expr, $operation:expr, $destination:expr, $source:expr) => {{
        match $operation {
            BinaryOp::Add => $assembler.add($destination, $source),
            BinaryOp::Subtract => $assembler.sub($destination, $source),
            BinaryOp::Multiply => $assembler.imul_2($destination, $source),
            BinaryOp::Xor => $assembler.xor($destination, $source),
            BinaryOp::And => $assembler.and($destination, $source),
            BinaryOp::Or => $assembler.or($destination, $source),
        }
    }};
}

macro_rules! emit_binary_immediate {
    ($assembler:expr, $operation:expr, $destination:expr, $source:expr) => {{
        match $operation {
            BinaryOp::Add => $assembler.add($destination, $source),
            BinaryOp::Subtract => $assembler.sub($destination, $source),
            BinaryOp::Multiply => $assembler.imul_3($destination, $destination, $source),
            BinaryOp::Xor => $assembler.xor($destination, $source),
            BinaryOp::And => $assembler.and($destination, $source),
            BinaryOp::Or => $assembler.or($destination, $source),
        }
    }};
}

macro_rules! emit_binary_byte_register {
    ($assembler:expr, $operation:expr, $destination:expr, $source:expr) => {{
        match $operation {
            BinaryOp::Add => $assembler.add($destination, $source),
            BinaryOp::Subtract => $assembler.sub($destination, $source),
            BinaryOp::Multiply => unreachable!("byte multiply rejected during validation"),
            BinaryOp::Xor => $assembler.xor($destination, $source),
            BinaryOp::And => $assembler.and($destination, $source),
            BinaryOp::Or => $assembler.or($destination, $source),
        }
    }};
}

macro_rules! emit_binary_byte_immediate {
    ($assembler:expr, $operation:expr, $destination:expr, $source:expr) => {{
        match $operation {
            BinaryOp::Add => $assembler.add($destination, $source),
            BinaryOp::Subtract => $assembler.sub($destination, $source),
            BinaryOp::Multiply => unreachable!("byte multiply rejected during validation"),
            BinaryOp::Xor => $assembler.xor($destination, $source),
            BinaryOp::And => $assembler.and($destination, $source),
            BinaryOp::Or => $assembler.or($destination, $source),
        }
    }};
}

fn emit_source_binary(
    assembler: &mut CodeAssembler,
    operation: BinaryOp,
    destination: MachineRegister,
    source: ResolvedBinarySource,
) -> Result<(), iced_x86::IcedError> {
    match (destination, source) {
        (
            MachineRegister::W8(destination),
            ResolvedBinarySource::Register(MachineRegister::W8(source)),
        ) => {
            emit_binary_byte_register!(assembler, operation, destination, source)
        }
        (
            MachineRegister::W16(destination),
            ResolvedBinarySource::Register(MachineRegister::W16(source)),
        ) => {
            emit_binary_register!(assembler, operation, destination, source)
        }
        (
            MachineRegister::W32(destination),
            ResolvedBinarySource::Register(MachineRegister::W32(source)),
        ) => {
            emit_binary_register!(assembler, operation, destination, source)
        }
        (
            MachineRegister::W64(destination),
            ResolvedBinarySource::Register(MachineRegister::W64(source)),
        ) => {
            emit_binary_register!(assembler, operation, destination, source)
        }
        (MachineRegister::W8(destination), ResolvedBinarySource::Immediate(source)) => {
            emit_binary_byte_immediate!(assembler, operation, destination, source)
        }
        (MachineRegister::W16(destination), ResolvedBinarySource::Immediate(source)) => {
            emit_binary_immediate!(assembler, operation, destination, source)
        }
        (MachineRegister::W32(destination), ResolvedBinarySource::Immediate(source)) => {
            emit_binary_immediate!(assembler, operation, destination, source)
        }
        (MachineRegister::W64(destination), ResolvedBinarySource::Immediate(source)) => {
            emit_binary_immediate!(assembler, operation, destination, source)
        }
        (_, ResolvedBinarySource::Register(_)) => unreachable!("validated widths differ"),
    }
}

fn emit_move_immediate(
    assembler: &mut CodeAssembler,
    destination: MachineRegister,
    value: u64,
) -> Result<(), iced_x86::IcedError> {
    let bytes = value.to_le_bytes();
    match destination {
        MachineRegister::W8(destination) => assembler.mov(destination, u32::from(bytes[0])),
        MachineRegister::W16(destination) => assembler.mov(
            destination,
            u32::from(u16::from_le_bytes([bytes[0], bytes[1]])),
        ),
        MachineRegister::W32(destination) => assembler.mov(
            destination,
            u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        ),
        MachineRegister::W64(destination) => assembler.mov(destination, value),
    }
}

fn emit_move(
    assembler: &mut CodeAssembler,
    destination: MachineRegister,
    source: MachineRegister,
) -> Result<(), iced_x86::IcedError> {
    match (destination, source) {
        (MachineRegister::W8(destination), MachineRegister::W8(source)) => {
            assembler.mov(destination, source)
        }
        (MachineRegister::W16(destination), MachineRegister::W16(source)) => {
            assembler.mov(destination, source)
        }
        (MachineRegister::W32(destination), MachineRegister::W32(source)) => {
            assembler.mov(destination, source)
        }
        (MachineRegister::W64(destination), MachineRegister::W64(source)) => {
            assembler.mov(destination, source)
        }
        _ => unreachable!("validated move widths differ"),
    }
}

fn emit_load(
    assembler: &mut CodeAssembler,
    destination: MachineRegister,
    base: AsmRegister64,
    displacement: i32,
) -> Result<(), iced_x86::IcedError> {
    match destination {
        MachineRegister::W8(destination) => {
            assembler.mov(destination, byte_ptr(base + displacement))
        }
        MachineRegister::W16(destination) => {
            assembler.mov(destination, word_ptr(base + displacement))
        }
        MachineRegister::W32(destination) => {
            assembler.mov(destination, dword_ptr(base + displacement))
        }
        MachineRegister::W64(destination) => {
            assembler.mov(destination, qword_ptr(base + displacement))
        }
    }
}

fn emit_store(
    assembler: &mut CodeAssembler,
    base: AsmRegister64,
    displacement: i32,
    source: MachineRegister,
) -> Result<(), iced_x86::IcedError> {
    match source {
        MachineRegister::W8(source) => assembler.mov(byte_ptr(base + displacement), source),
        MachineRegister::W16(source) => assembler.mov(word_ptr(base + displacement), source),
        MachineRegister::W32(source) => assembler.mov(dword_ptr(base + displacement), source),
        MachineRegister::W64(source) => assembler.mov(qword_ptr(base + displacement), source),
    }
}

fn emit_compare(
    assembler: &mut CodeAssembler,
    left: MachineRegister,
    right: ResolvedBinarySource,
) -> Result<(), iced_x86::IcedError> {
    match (left, right) {
        (MachineRegister::W8(left), ResolvedBinarySource::Register(MachineRegister::W8(right))) => {
            assembler.cmp(left, right)
        }
        (
            MachineRegister::W16(left),
            ResolvedBinarySource::Register(MachineRegister::W16(right)),
        ) => assembler.cmp(left, right),
        (
            MachineRegister::W32(left),
            ResolvedBinarySource::Register(MachineRegister::W32(right)),
        ) => assembler.cmp(left, right),
        (
            MachineRegister::W64(left),
            ResolvedBinarySource::Register(MachineRegister::W64(right)),
        ) => assembler.cmp(left, right),
        (MachineRegister::W8(left), ResolvedBinarySource::Immediate(right)) => {
            assembler.cmp(left, right)
        }
        (MachineRegister::W16(left), ResolvedBinarySource::Immediate(right)) => {
            assembler.cmp(left, right)
        }
        (MachineRegister::W32(left), ResolvedBinarySource::Immediate(right)) => {
            assembler.cmp(left, right)
        }
        (MachineRegister::W64(left), ResolvedBinarySource::Immediate(right)) => {
            assembler.cmp(left, right)
        }
        (_, ResolvedBinarySource::Register(_)) => unreachable!("validated compare widths differ"),
    }
}

fn emit_unary(
    assembler: &mut CodeAssembler,
    operation: UnaryOp,
    destination: MachineRegister,
) -> Result<(), iced_x86::IcedError> {
    macro_rules! emit {
        ($register:expr) => {
            match operation {
                UnaryOp::Not => assembler.not($register),
                UnaryOp::Negate => assembler.neg($register),
            }
        };
    }
    match destination {
        MachineRegister::W8(register) => emit!(register),
        MachineRegister::W16(register) => emit!(register),
        MachineRegister::W32(register) => emit!(register),
        MachineRegister::W64(register) => emit!(register),
    }
}

#[derive(Clone, Copy)]
enum ResolvedShiftCount {
    Immediate(u32),
    Cl,
}

fn emit_shift(
    assembler: &mut CodeAssembler,
    operation: ShiftOp,
    destination: MachineRegister,
    count: ResolvedShiftCount,
) -> Result<(), iced_x86::IcedError> {
    macro_rules! emit {
        ($register:expr, $count:expr) => {
            match operation {
                ShiftOp::Left => assembler.shl($register, $count),
                ShiftOp::RightLogical => assembler.shr($register, $count),
                ShiftOp::RightArithmetic => assembler.sar($register, $count),
                ShiftOp::RotateLeft => assembler.rol($register, $count),
                ShiftOp::RotateRight => assembler.ror($register, $count),
            }
        };
    }
    match (destination, count) {
        (MachineRegister::W8(register), ResolvedShiftCount::Immediate(count)) => {
            emit!(register, count)
        }
        (MachineRegister::W16(register), ResolvedShiftCount::Immediate(count)) => {
            emit!(register, count)
        }
        (MachineRegister::W32(register), ResolvedShiftCount::Immediate(count)) => {
            emit!(register, count)
        }
        (MachineRegister::W64(register), ResolvedShiftCount::Immediate(count)) => {
            emit!(register, count)
        }
        (MachineRegister::W8(register), ResolvedShiftCount::Cl) => emit!(register, cl),
        (MachineRegister::W16(register), ResolvedShiftCount::Cl) => emit!(register, cl),
        (MachineRegister::W32(register), ResolvedShiftCount::Cl) => emit!(register, cl),
        (MachineRegister::W64(register), ResolvedShiftCount::Cl) => emit!(register, cl),
    }
}

fn emit_resize(
    assembler: &mut CodeAssembler,
    operation: ResizeOp,
    destination: MachineRegister,
    source: MachineRegister,
    source_as_destination_width: MachineRegister,
    destination_as_source_width: MachineRegister,
) -> Result<(), iced_x86::IcedError> {
    match operation {
        ResizeOp::Truncate => emit_move(assembler, destination, source_as_destination_width),
        ResizeOp::ZeroExtend => match (destination, source) {
            (MachineRegister::W16(destination), MachineRegister::W8(source)) => {
                assembler.movzx(destination, source)
            }
            (MachineRegister::W32(destination), MachineRegister::W8(source)) => {
                assembler.movzx(destination, source)
            }
            (MachineRegister::W64(destination), MachineRegister::W8(source)) => {
                assembler.movzx(destination, source)
            }
            (MachineRegister::W32(destination), MachineRegister::W16(source)) => {
                assembler.movzx(destination, source)
            }
            (MachineRegister::W64(destination), MachineRegister::W16(source)) => {
                assembler.movzx(destination, source)
            }
            (MachineRegister::W64(_), MachineRegister::W32(source)) => emit_move(
                assembler,
                destination_as_source_width,
                MachineRegister::W32(source),
            ),
            _ => unreachable!("invalid zero extension passed validation"),
        },
        ResizeOp::SignExtend => match (destination, source) {
            (MachineRegister::W16(destination), MachineRegister::W8(source)) => {
                assembler.movsx(destination, source)
            }
            (MachineRegister::W32(destination), MachineRegister::W8(source)) => {
                assembler.movsx(destination, source)
            }
            (MachineRegister::W64(destination), MachineRegister::W8(source)) => {
                assembler.movsx(destination, source)
            }
            (MachineRegister::W32(destination), MachineRegister::W16(source)) => {
                assembler.movsx(destination, source)
            }
            (MachineRegister::W64(destination), MachineRegister::W16(source)) => {
                assembler.movsx(destination, source)
            }
            (MachineRegister::W64(destination), MachineRegister::W32(source)) => {
                assembler.movsxd(destination, source)
            }
            _ => unreachable!("invalid sign extension passed validation"),
        },
    }
}

fn conditional_opcode(condition: Condition) -> u8 {
    match condition {
        Condition::Overflow => 0x80,
        Condition::NotOverflow => 0x81,
        Condition::Below => 0x82,
        Condition::AboveOrEqual => 0x83,
        Condition::Equal => 0x84,
        Condition::NotEqual => 0x85,
        Condition::BelowOrEqual => 0x86,
        Condition::Above => 0x87,
        Condition::Sign => 0x88,
        Condition::NotSign => 0x89,
        Condition::Parity => 0x8A,
        Condition::NotParity => 0x8B,
        Condition::Less => 0x8C,
        Condition::GreaterOrEqual => 0x8D,
        Condition::LessOrEqual => 0x8E,
        Condition::Greater => 0x8F,
    }
}

fn emit_jump(output: &mut Vec<u8>, target: Label, fixups: &mut Vec<Fixup>) {
    output.push(0xE9);
    let displacement_offset = output.len();
    output.extend_from_slice(&0_i32.to_le_bytes());
    fixups.push(Fixup {
        displacement_offset,
        instruction_end: output.len(),
        target,
    });
}

fn emit_conditional_jump(
    output: &mut Vec<u8>,
    condition: Condition,
    target: Label,
    fixups: &mut Vec<Fixup>,
) {
    output.extend_from_slice(&[0x0F, conditional_opcode(condition)]);
    let displacement_offset = output.len();
    output.extend_from_slice(&0_i32.to_le_bytes());
    fixups.push(Fixup {
        displacement_offset,
        instruction_end: output.len(),
        target,
    });
}

fn relative_call_displacement(
    output: &[u8],
    address: CodeAddress,
    instruction_size: u64,
    target: NativeAddress,
    operation_index: usize,
) -> Result<i32, CompileError> {
    let offset = u64::try_from(output.len()).map_err(|_| CompileError::CodeSizeOverflow)?;
    let instruction_end = address
        .0
        .checked_add(offset)
        .and_then(|value| value.checked_add(instruction_size))
        .ok_or(CompileError::CodeSizeOverflow)?;
    let relative = i128::from(target.0) - i128::from(instruction_end);
    i32::try_from(relative).map_err(|_| CompileError::CallOutOfRange {
        operation_index,
        target,
    })
}

fn emit_direct_call(
    output: &mut Vec<u8>,
    address: CodeAddress,
    target: NativeAddress,
    operation_index: usize,
) -> Result<(), CompileError> {
    let displacement = relative_call_displacement(output, address, 5, target, operation_index)?;
    output.push(0xE8);
    output.extend_from_slice(&displacement.to_le_bytes());
    Ok(())
}

fn emit_iat_call(
    output: &mut Vec<u8>,
    address: CodeAddress,
    slot: NativeAddress,
    operation_index: usize,
) -> Result<(), CompileError> {
    let displacement = relative_call_displacement(output, address, 6, slot, operation_index)?;
    output.extend_from_slice(&[0xFF, 0x15]);
    output.extend_from_slice(&displacement.to_le_bytes());
    Ok(())
}

fn emit_epilogue(
    output: &mut Vec<u8>,
    function: &Function,
    address: CodeAddress,
    operation_index: usize,
) -> Result<(), CompileError> {
    if function.frame.stack_allocation != 0 {
        let stack_allocation = i32::try_from(function.frame.stack_allocation).map_err(|_| {
            CompileError::StackAllocationTooLarge {
                bytes: function.frame.stack_allocation,
            }
        })?;
        append_assembled(output, address, operation_index, |assembler| {
            assembler.add(rsp, stack_allocation)
        })?;
    }
    for register in function.frame.preserve.iter().rev().copied() {
        append_assembled(output, address, operation_index, |assembler| {
            assembler.pop(machine_register(register))
        })?;
    }
    append_assembled(output, address, operation_index, CodeAssembler::ret)
}

/// Compiles a validated lowered function and resolves all branches in a second
/// pass. Every jump uses a fixed-width near encoding, so patching cannot change
/// any previously measured label offset.
///
/// # Errors
///
/// Returns a structured error when IR, allocation, ABI, label, address, branch,
/// or iced-x86 encoding invariants fail.
#[allow(clippy::too_many_lines)]
pub fn compile(
    function: &Function,
    address: CodeAddress,
) -> Result<CompiledFunction, CompileError> {
    let validated = validate(function)?;
    let mut output = Vec::new();
    let mut label_offsets = BTreeMap::new();
    let mut fixups = Vec::new();

    for register in function.frame.preserve.iter().copied() {
        append_assembled(&mut output, address, 0, |assembler| {
            assembler.push(machine_register(register))
        })?;
    }
    if function.frame.stack_allocation != 0 {
        let stack_allocation = i32::try_from(function.frame.stack_allocation).map_err(|_| {
            CompileError::StackAllocationTooLarge {
                bytes: function.frame.stack_allocation,
            }
        })?;
        append_assembled(&mut output, address, 0, |assembler| {
            assembler.sub(rsp, stack_allocation)
        })?;
    }

    for (operation_index, operation) in function.operations.iter().enumerate() {
        match operation {
            Operation::Label(label) => {
                let offset =
                    u32::try_from(output.len()).map_err(|_| CompileError::CodeSizeOverflow)?;
                label_offsets.insert(*label, offset);
            }
            Operation::MoveImmediate { destination, value } => {
                let destination = assigned(&validated, *destination)?;
                append_assembled(&mut output, address, operation_index, |assembler| {
                    emit_move_immediate(assembler, destination, *value)
                })?;
            }
            Operation::Move {
                destination,
                source,
            } => {
                let destination = assigned(&validated, *destination)?;
                let source = assigned(&validated, *source)?;
                append_assembled(&mut output, address, operation_index, |assembler| {
                    emit_move(assembler, destination, source)
                })?;
            }
            Operation::Binary {
                operation,
                destination,
                source,
            } => {
                let destination = assigned(&validated, *destination)?;
                let source = resolve_source(*source, &validated)?;
                append_assembled(&mut output, address, operation_index, |assembler| {
                    emit_source_binary(assembler, *operation, destination, source)
                })?;
            }
            Operation::Unary {
                operation,
                destination,
            } => {
                let destination = assigned(&validated, *destination)?;
                append_assembled(&mut output, address, operation_index, |assembler| {
                    emit_unary(assembler, *operation, destination)
                })?;
            }
            Operation::Shift {
                operation,
                destination,
                count,
            } => {
                let destination = assigned(&validated, *destination)?;
                let count = match count {
                    ShiftCount::Immediate(count) => {
                        ResolvedShiftCount::Immediate(u32::from(*count))
                    }
                    ShiftCount::Register(register) => {
                        let _ = assigned(&validated, *register)?;
                        ResolvedShiftCount::Cl
                    }
                };
                append_assembled(&mut output, address, operation_index, |assembler| {
                    emit_shift(assembler, *operation, destination, count)
                })?;
            }
            Operation::Resize {
                operation,
                destination,
                source,
            } => {
                let destination_register = assigned(&validated, *destination)?;
                let source_register = assigned(&validated, *source)?;
                let source_as_destination_width =
                    assigned_view(&validated, *source, destination.width)?;
                let destination_as_source_width =
                    assigned_view(&validated, *destination, source.width)?;
                append_assembled(&mut output, address, operation_index, |assembler| {
                    emit_resize(
                        assembler,
                        *operation,
                        destination_register,
                        source_register,
                        source_as_destination_width,
                        destination_as_source_width,
                    )
                })?;
            }
            Operation::Load {
                destination,
                address: memory,
            } => {
                let destination = assigned(&validated, *destination)?;
                let base = assigned_qword(&validated, memory.base)?;
                append_assembled(&mut output, address, operation_index, |assembler| {
                    emit_load(assembler, destination, base, memory.displacement)
                })?;
            }
            Operation::Store {
                address: memory,
                source,
            } => {
                let base = assigned_qword(&validated, memory.base)?;
                let source = assigned(&validated, *source)?;
                append_assembled(&mut output, address, operation_index, |assembler| {
                    emit_store(assembler, base, memory.displacement, source)
                })?;
            }
            Operation::SpillLoad { destination, slot } => {
                let destination = assigned(&validated, *destination)?;
                let displacement = spill_displacement(&function.frame, *slot)?;
                append_assembled(&mut output, address, operation_index, |assembler| {
                    emit_load(assembler, destination, rsp, displacement)
                })?;
            }
            Operation::SpillStore { slot, source } => {
                let source = assigned(&validated, *source)?;
                let displacement = spill_displacement(&function.frame, *slot)?;
                append_assembled(&mut output, address, operation_index, |assembler| {
                    emit_store(assembler, rsp, displacement, source)
                })?;
            }
            Operation::Call { target, .. } => match target {
                CallTarget::Direct(target) => {
                    emit_direct_call(&mut output, address, *target, operation_index)?;
                }
                CallTarget::ImportAddressTable(slot) => {
                    emit_iat_call(&mut output, address, *slot, operation_index)?;
                }
                CallTarget::Register(target) => {
                    let target = assigned_qword(&validated, *target)?;
                    append_assembled(&mut output, address, operation_index, |assembler| {
                        assembler.call(target)
                    })?;
                }
            },
            Operation::Compare { left, right } => {
                let left = assigned(&validated, *left)?;
                let right = resolve_source(*right, &validated)?;
                append_assembled(&mut output, address, operation_index, |assembler| {
                    emit_compare(assembler, left, right)
                })?;
            }
            Operation::Jump(target) => emit_jump(&mut output, *target, &mut fixups),
            Operation::BranchIf { condition, target } => {
                emit_conditional_jump(&mut output, *condition, *target, &mut fixups);
            }
            Operation::Return => {
                emit_epilogue(&mut output, function, address, operation_index)?;
            }
        }
    }

    for fixup in fixups {
        let target =
            label_offsets
                .get(&fixup.target)
                .copied()
                .ok_or(CompileError::UndefinedLabel {
                    label: fixup.target,
                })?;
        let source_end =
            i64::try_from(fixup.instruction_end).map_err(|_| CompileError::CodeSizeOverflow)?;
        let relative = i64::from(target) - source_end;
        let relative = i32::try_from(relative).map_err(|_| CompileError::BranchOutOfRange {
            source_offset: u32::try_from(fixup.displacement_offset).unwrap_or(u32::MAX),
            target: fixup.target,
        })?;
        let end = fixup
            .displacement_offset
            .checked_add(4)
            .ok_or(CompileError::CodeSizeOverflow)?;
        let destination = output
            .get_mut(fixup.displacement_offset..end)
            .ok_or(CompileError::CodeSizeOverflow)?;
        destination.copy_from_slice(&relative.to_le_bytes());
    }

    Ok(CompiledFunction {
        address,
        bytes: output,
        label_offsets,
        stack_frame_size: function.frame.stack_allocation,
    })
}
