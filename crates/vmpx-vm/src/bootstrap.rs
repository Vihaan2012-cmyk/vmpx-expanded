use std::collections::BTreeMap;

use iced_x86::{
    FlowControl, Instruction, InstructionInfoFactory, Mnemonic, OpAccess, OpKind, Register,
};
use serde::Serialize;
use thiserror::Error;
use vmpx_disasm::{
    DirectJumpTraceLimits, DisasmError, RegisterDispatchProofLimits, trace_direct_jump_fragments,
};
use vmpx_pe::{PeImage, Rva, Va};

use crate::{
    BytecodeDirection, HandlerRegisterRoles, HandlerSliceConfig, NativeBackwardSlice,
    RelativeDispatchCodec, RelativeDispatchState, VmEntryCandidate, backward_slice_handler_trace,
    extract_relative_dispatch_codecs_with_postlude_direction,
};

const X64_GPRS: [Register; 15] = [
    Register::RAX,
    Register::RCX,
    Register::RDX,
    Register::RBX,
    Register::RBP,
    Register::RSI,
    Register::RDI,
    Register::R8,
    Register::R9,
    Register::R10,
    Register::R11,
    Register::R12,
    Register::R13,
    Register::R14,
    Register::R15,
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BootstrapDispatchRecovery {
    pub roles: HandlerRegisterRoles,
    pub bytecode_direction: BytecodeDirection,
    pub codec: RelativeDispatchCodec,
    pub slice: NativeBackwardSlice,
    pub encoded_load: Rva,
    pub initial_state: RelativeDispatchState,
    pub confidence: u8,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct BootstrapDispatchCandidateReport {
    pub vip_register: String,
    pub rolling_key_register: String,
    pub handler_register: String,
    pub bytecode_direction: BytecodeDirection,
    pub encoded_load: Rva,
    pub initial_vip: Va,
    pub initial_rolling_key: u64,
    pub handler_anchor: Va,
    pub confidence: u8,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct BootstrapDispatchReport {
    pub schema_version: u32,
    pub entry: Rva,
    pub call_target: Option<Rva>,
    pub dispatch_trace_start: Rva,
    pub candidates: Vec<BootstrapDispatchCandidateReport>,
    pub rejected_role_assignments: usize,
    pub rejection_samples: Vec<String>,
    pub diagnostics: Vec<String>,
}

#[derive(Debug, Error)]
pub enum BootstrapRecoveryError {
    #[error("VM entry has no direct call target")]
    MissingCallTarget,
    #[error("native bootstrap trace failed: {0}")]
    Disasm(#[from] DisasmError),
    #[error("VM entry has no concrete immediate seed")]
    MissingSeed,
    #[error("bootstrap initial state could not be evaluated: {message}")]
    InitialState { message: String },
}

#[derive(Default)]
struct ConcreteBootstrapMachine {
    registers: BTreeMap<Register, u64>,
    stack: BTreeMap<i64, u64>,
    rsp_offset: i64,
}

fn bit_mask(bits: u32) -> u64 {
    if bits >= 64 {
        u64::MAX
    } else {
        (1_u64 << bits) - 1
    }
}

fn register_shift(register: Register) -> u32 {
    if matches!(
        register,
        Register::AH | Register::BH | Register::CH | Register::DH
    ) {
        8
    } else {
        0
    }
}

impl ConcreteBootstrapMachine {
    fn new(seed: u64) -> Self {
        let mut machine = Self::default();
        // At the direct call target, [RSP] is the call continuation and the
        // entry stub's immediately preceding PUSH is the qword at [RSP+8].
        machine.stack.insert(8, seed);
        machine
    }

    fn read_register(&self, register: Register) -> Option<u64> {
        if register.full_register() == Register::RSP {
            return Some(u64::from_ne_bytes(self.rsp_offset.to_ne_bytes()));
        }
        let value = *self.registers.get(&register.full_register())?;
        let bits = u32::try_from(register.size().checked_mul(8)?).ok()?;
        Some((value >> register_shift(register)) & bit_mask(bits))
    }

    fn write_register(&mut self, register: Register, value: u64) {
        let full = register.full_register();
        if full == Register::RSP {
            self.rsp_offset = i64::from_ne_bytes(value.to_ne_bytes());
            return;
        }
        let bits = u32::try_from(register.size().saturating_mul(8)).unwrap_or(64);
        if register == full {
            self.registers.insert(full, value);
        } else if bits == 32 && full.size() == 8 {
            self.registers.insert(full, value & bit_mask(32));
        } else if let Some(previous) = self.registers.get(&full).copied() {
            let shift = register_shift(register);
            let lane = bit_mask(bits) << shift;
            self.registers
                .insert(full, (previous & !lane) | ((value << shift) & lane));
        } else {
            self.registers.remove(&full);
        }
    }

    fn invalidate_register(&mut self, register: Register) {
        let full = register.full_register();
        if full == Register::RSP {
            self.stack.clear();
        } else {
            self.registers.remove(&full);
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

    fn stack_address(&self, instruction: &Instruction) -> Option<i64> {
        if instruction.memory_base().full_register() != Register::RSP
            || instruction.memory_index() != Register::None
        {
            return None;
        }
        let displacement = i64::from_ne_bytes(instruction.memory_displacement64().to_ne_bytes());
        self.rsp_offset.checked_add(displacement)
    }

    fn operand(&self, instruction: &Instruction, operand: u32) -> Option<u64> {
        match instruction.op_kind(operand) {
            OpKind::Register => self.read_register(instruction.op_register(operand)),
            OpKind::Memory if instruction.memory_size().size() == 8 => self
                .stack_address(instruction)
                .and_then(|address| self.stack.get(&address).copied()),
            _ => Self::immediate(instruction, operand),
        }
    }

    fn effective_address(&self, instruction: &Instruction) -> Option<u64> {
        if instruction.is_ip_rel_memory_operand() {
            return Some(instruction.ip_rel_memory_address());
        }
        let base = match instruction.memory_base() {
            Register::None => 0,
            register => self.read_register(register)?,
        };
        let index = match instruction.memory_index() {
            Register::None => 0,
            register => self
                .read_register(register)?
                .checked_mul(u64::from(instruction.memory_index_scale()))?,
        };
        base.checked_add(index)?
            .checked_add(instruction.memory_displacement64())
    }

    fn execute_binary(&mut self, instruction: &Instruction) {
        let destination = instruction.op0_register();
        let value = self
            .read_register(destination)
            .zip(self.operand(instruction, 1))
            .map(|(left, right)| match instruction.mnemonic() {
                Mnemonic::Add => left.wrapping_add(right),
                Mnemonic::Sub => left.wrapping_sub(right),
                Mnemonic::Xor => left ^ right,
                Mnemonic::And => left & right,
                Mnemonic::Or => left | right,
                _ => unreachable!(),
            });
        if let Some(value) = value {
            self.write_register(destination, value);
        } else {
            self.invalidate_register(destination);
        }
    }

    fn execute_unary(&mut self, instruction: &Instruction) {
        let destination = instruction.op0_register();
        if let Some(value) = self.read_register(destination) {
            let value = match instruction.mnemonic() {
                Mnemonic::Neg => value.wrapping_neg(),
                Mnemonic::Not => !value,
                Mnemonic::Inc => value.wrapping_add(1),
                Mnemonic::Dec => value.wrapping_sub(1),
                _ => unreachable!(),
            };
            self.write_register(destination, value);
        } else {
            self.invalidate_register(destination);
        }
    }

    fn execute_rotate(&mut self, instruction: &Instruction) {
        let destination = instruction.op0_register();
        let bits = u32::try_from(destination.size().saturating_mul(8)).unwrap_or(64);
        let value = self
            .read_register(destination)
            .zip(self.operand(instruction, 1));
        if let Some((value, count)) = value {
            let count_mask = if bits == 64 { 63 } else { 31 };
            let count = u32::try_from(count & count_mask).unwrap_or(0);
            let truncated = value & bit_mask(bits);
            let value = match instruction.mnemonic() {
                Mnemonic::Rol | Mnemonic::Ror if count == 0 => truncated,
                Mnemonic::Rol => {
                    ((truncated << count) | (truncated >> (bits - count))) & bit_mask(bits)
                }
                Mnemonic::Ror => {
                    ((truncated >> count) | (truncated << (bits - count))) & bit_mask(bits)
                }
                _ => unreachable!(),
            };
            self.write_register(destination, value);
        } else {
            self.invalidate_register(destination);
        }
    }

    fn execute_supported(&mut self, instruction: &Instruction) -> bool {
        match instruction.mnemonic() {
            Mnemonic::Pushfq => {
                self.rsp_offset = self.rsp_offset.saturating_sub(8);
                self.stack.remove(&self.rsp_offset);
                true
            }
            Mnemonic::Push => {
                self.rsp_offset = self.rsp_offset.saturating_sub(8);
                if let Some(value) = self.operand(instruction, 0) {
                    self.stack.insert(self.rsp_offset, value);
                } else {
                    self.stack.remove(&self.rsp_offset);
                }
                true
            }
            Mnemonic::Pop if instruction.op0_kind() == OpKind::Register => {
                let destination = instruction.op0_register();
                if let Some(value) = self.stack.get(&self.rsp_offset).copied() {
                    self.write_register(destination, value);
                } else {
                    self.invalidate_register(destination);
                }
                self.rsp_offset = self.rsp_offset.saturating_add(8);
                true
            }
            Mnemonic::Mov if instruction.op0_kind() == OpKind::Register => {
                let destination = instruction.op0_register();
                if let Some(value) = self.operand(instruction, 1) {
                    self.write_register(destination, value);
                } else {
                    self.invalidate_register(destination);
                }
                true
            }
            Mnemonic::Lea if instruction.op0_kind() == OpKind::Register => {
                let destination = instruction.op0_register();
                if let Some(value) = self.effective_address(instruction) {
                    self.write_register(destination, value);
                } else {
                    self.invalidate_register(destination);
                }
                true
            }
            Mnemonic::Add | Mnemonic::Sub | Mnemonic::Xor | Mnemonic::And | Mnemonic::Or
                if instruction.op0_kind() == OpKind::Register =>
            {
                self.execute_binary(instruction);
                true
            }
            Mnemonic::Neg | Mnemonic::Not | Mnemonic::Inc | Mnemonic::Dec
                if instruction.op0_kind() == OpKind::Register =>
            {
                self.execute_unary(instruction);
                true
            }
            Mnemonic::Rol | Mnemonic::Ror if instruction.op0_kind() == OpKind::Register => {
                self.execute_rotate(instruction);
                true
            }
            Mnemonic::Bswap if instruction.op0_kind() == OpKind::Register => {
                let destination = instruction.op0_register();
                if let Some(value) = self.read_register(destination) {
                    let value = match destination.size() {
                        4 => u64::from(
                            u32::try_from(value & u64::from(u32::MAX))
                                .unwrap_or(0)
                                .swap_bytes(),
                        ),
                        8 => value.swap_bytes(),
                        _ => {
                            self.invalidate_register(destination);
                            return true;
                        }
                    };
                    self.write_register(destination, value);
                } else {
                    self.invalidate_register(destination);
                }
                true
            }
            _ => false,
        }
    }

    fn execute(&mut self, instruction: &Instruction) {
        if self.execute_supported(instruction) {
            return;
        }
        let mut factory = InstructionInfoFactory::new();
        for used in factory.info(instruction).used_registers() {
            if matches!(
                used.access(),
                OpAccess::Write
                    | OpAccess::CondWrite
                    | OpAccess::ReadWrite
                    | OpAccess::ReadCondWrite
            ) {
                self.invalidate_register(used.register());
            }
        }
    }
}

fn terminal_handler_register(trace: &vmpx_disasm::DirectJumpTrace) -> Option<Register> {
    let terminal = trace.instructions().last()?.instruction;
    (terminal.flow_control() == FlowControl::IndirectBranch
        && terminal.op0_kind() == OpKind::Register)
        .then(|| terminal.op0_register().full_register())
}

fn writes_full_register(
    factory: &mut InstructionInfoFactory,
    instruction: &iced_x86::Instruction,
    target: Register,
) -> bool {
    factory
        .info(instruction)
        .used_registers()
        .iter()
        .any(|used| {
            matches!(used.access(), OpAccess::Write | OpAccess::ReadWrite)
                && used.register().full_register() == target
                && (used.register() == target
                    || (target.size() == 8
                        && used.register().size() == 4
                        && used.register().full_register32() == used.register()))
        })
}

fn role_trace_start(
    trace: &vmpx_disasm::DirectJumpTrace,
    vip: Register,
    rolling_key: Register,
    direction: BytecodeDirection,
) -> Rva {
    let decoded = trace.instructions().collect::<Vec<_>>();
    let source_index = decoded.iter().position(|decoded| {
        decoded.instruction.memory_base().full_register() == vip
            && decoded.instruction.op_count() != 0
    });
    let Some(source_index) = source_index else {
        return trace.entry;
    };
    let mut factory = InstructionInfoFactory::new();
    let writes = decoded[..source_index]
        .iter()
        .enumerate()
        .filter(|(_, decoded)| {
            writes_full_register(&mut factory, &decoded.instruction, vip)
                || writes_full_register(&mut factory, &decoded.instruction, rolling_key)
        })
        .collect::<Vec<_>>();
    let start_after = if direction == BytecodeDirection::Descending
        && let Some((last_index, last)) = writes.last().copied()
        && last.instruction.op0_kind() == OpKind::Register
        && last.instruction.op0_register().full_register() == vip
        && last.instruction.op0_register().size() == 8
        && matches!(last.instruction.mnemonic(), Mnemonic::Add | Mnemonic::Sub)
        && ConcreteBootstrapMachine::immediate(&last.instruction, 1).is_some()
    {
        writes
            .iter()
            .rev()
            .find(|(index, _)| *index < last_index)
            .map(|(_, decoded)| *decoded)
    } else {
        writes.last().map(|(_, decoded)| *decoded)
    };
    start_after
        .and_then(|decoded| decoded.next_rva())
        .unwrap_or(trace.entry)
}

fn evaluate_initial_state(
    pe: &PeImage<'_>,
    entry: &VmEntryCandidate,
    roles: HandlerRegisterRoles,
    encoded_load: Rva,
    limits: DirectJumpTraceLimits,
) -> Result<RelativeDispatchState, BootstrapRecoveryError> {
    let seed = entry.seed.ok_or(BootstrapRecoveryError::MissingSeed)?;
    let call_target = entry
        .call_target
        .ok_or(BootstrapRecoveryError::MissingCallTarget)?;
    let trace = trace_direct_jump_fragments(pe, call_target, limits)?;
    let mut machine = ConcreteBootstrapMachine::new(seed);
    let mut reached_load = false;
    for decoded in trace.instructions() {
        if decoded.rva == encoded_load {
            reached_load = true;
            break;
        }
        machine.execute(&decoded.instruction);
    }
    if !reached_load {
        return Err(BootstrapRecoveryError::InitialState {
            message: format!("encoded dispatch load {encoded_load} is absent from entry trace"),
        });
    }
    let value = |register, description| {
        machine
            .read_register(register)
            .ok_or_else(|| BootstrapRecoveryError::InitialState {
                message: format!("{description} register {register:?} is not concrete"),
            })
    };
    Ok(RelativeDispatchState {
        vip: Va(value(roles.vip, "VIP")?),
        rolling_key: value(roles.rolling_key, "rolling key")?,
        handler: Va(value(roles.handler, "handler anchor")?),
    })
}

/// Recovers a VM entry bootstrap's dispatch register roles and keyed relative
/// transform by enumerating architectural role assignments and accepting only
/// behaviorally valid dispatch slices. No handler bytes, addresses, or fixed
/// register allocation are assumed.
///
/// # Errors
///
/// Returns a structured error only when the entry lacks a call target or its
/// native direct-jump trace cannot be decoded. Ambiguity is retained in the
/// report and yields no recovered candidate.
#[allow(clippy::too_many_lines)]
pub fn recover_bootstrap_dispatch(
    pe: &PeImage<'_>,
    entry: &VmEntryCandidate,
    limits: DirectJumpTraceLimits,
) -> Result<(BootstrapDispatchReport, Option<BootstrapDispatchRecovery>), BootstrapRecoveryError> {
    let call_target = entry
        .call_target
        .ok_or(BootstrapRecoveryError::MissingCallTarget)?;
    let dispatch_trace_start = entry.bootstrap_rva.unwrap_or(call_target);
    let trace = trace_direct_jump_fragments(pe, dispatch_trace_start, limits)?;
    let Some(handler) = terminal_handler_register(&trace) else {
        return Ok((
            BootstrapDispatchReport {
                schema_version: 1,
                entry: entry.rva,
                call_target: Some(call_target),
                dispatch_trace_start,
                candidates: Vec::new(),
                rejected_role_assignments: 0,
                rejection_samples: Vec::new(),
                diagnostics: vec![
                    "bootstrap trace does not terminate in a register-indirect branch".to_owned(),
                ],
            },
            None,
        ));
    };

    let mut recovered = Vec::new();
    let mut rejected = 0_usize;
    let mut rejection_samples = Vec::new();
    for vip in X64_GPRS.into_iter().filter(|register| *register != handler) {
        for rolling_key in X64_GPRS
            .into_iter()
            .filter(|register| *register != handler && *register != vip)
        {
            let roles = HandlerRegisterRoles {
                vip,
                rolling_key,
                handler,
            };
            for bytecode_direction in [BytecodeDirection::Ascending, BytecodeDirection::Descending]
            {
                let candidate_start =
                    role_trace_start(&trace, vip, rolling_key, bytecode_direction);
                let candidate_trace = trace_direct_jump_fragments(pe, candidate_start, limits)?;
                let slice = backward_slice_handler_trace(
                    &candidate_trace,
                    &HandlerSliceConfig {
                        live_out_registers: vec![vip, rolling_key, handler],
                        live_out_rflags: 0,
                    },
                );
                let extraction = extract_relative_dispatch_codecs_with_postlude_direction(
                    pe,
                    &candidate_trace,
                    &slice,
                    roles,
                    bytecode_direction,
                    RegisterDispatchProofLimits::default(),
                )?;
                for candidate in extraction.candidates {
                    let mut initial_state = match evaluate_initial_state(
                        pe,
                        entry,
                        roles,
                        candidate.encoded_load,
                        limits,
                    ) {
                        Ok(state) => state,
                        Err(error) => {
                            rejected = rejected.saturating_add(1);
                            if rejection_samples.len() < 64 {
                                rejection_samples.push(format!(
                                    "VIP {vip:?}, key {rolling_key:?}, handler {handler:?}, direction {bytecode_direction:?}: {error}"
                                ));
                            }
                            continue;
                        }
                    };
                    if bytecode_direction == BytecodeDirection::Descending {
                        initial_state.vip = initial_state
                            .vip
                            .checked_add(u64::from(candidate.codec.displacement.width.bytes()))
                            .ok_or_else(|| BootstrapRecoveryError::InitialState {
                                message: "descending pre-fetch VIP adjustment overflows".to_owned(),
                            })?;
                    }
                    if !recovered
                        .iter()
                        .any(|existing: &BootstrapDispatchRecovery| {
                            existing.roles == roles
                                && existing.bytecode_direction == bytecode_direction
                                && existing.encoded_load == candidate.encoded_load
                                && existing.codec == candidate.codec
                        })
                    {
                        recovered.push(BootstrapDispatchRecovery {
                            roles,
                            bytecode_direction,
                            codec: candidate.codec,
                            slice: slice.clone(),
                            encoded_load: candidate.encoded_load,
                            initial_state,
                            confidence: candidate.confidence,
                        });
                    }
                }
                rejected = rejected.saturating_add(extraction.rejected_sources.len());
                for rejection in extraction.rejected_sources {
                    if rejection_samples.len() < 64 {
                        rejection_samples.push(format!(
                            "VIP {vip:?}, key {rolling_key:?}, handler {handler:?}, direction {bytecode_direction:?}, load {}: {}",
                            rejection.encoded_load, rejection.reason
                        ));
                    }
                }
            }
        }
    }
    recovered.sort_by_key(|candidate| {
        (
            format!("{:?}", candidate.roles.vip),
            format!("{:?}", candidate.roles.rolling_key),
            format!("{:?}", candidate.bytecode_direction),
            candidate.encoded_load,
        )
    });
    let report_candidates = recovered
        .iter()
        .map(|candidate| BootstrapDispatchCandidateReport {
            vip_register: format!("{:?}", candidate.roles.vip),
            rolling_key_register: format!("{:?}", candidate.roles.rolling_key),
            handler_register: format!("{:?}", candidate.roles.handler),
            bytecode_direction: candidate.bytecode_direction,
            encoded_load: candidate.encoded_load,
            initial_vip: candidate.initial_state.vip,
            initial_rolling_key: candidate.initial_state.rolling_key,
            handler_anchor: candidate.initial_state.handler,
            confidence: candidate.confidence,
        })
        .collect::<Vec<_>>();
    let diagnostics = match recovered.len() {
        0 => vec!["no role assignment produced a valid bootstrap dispatch".to_owned()],
        1 => Vec::new(),
        count => vec![format!(
            "{count} role assignments remain behaviorally valid; bootstrap is ambiguous"
        )],
    };
    let selected = (recovered.len() == 1).then(|| recovered.remove(0));
    Ok((
        BootstrapDispatchReport {
            schema_version: 1,
            entry: entry.rva,
            call_target: Some(call_target),
            dispatch_trace_start,
            candidates: report_candidates,
            rejected_role_assignments: rejected,
            rejection_samples,
            diagnostics,
        },
        selected,
    ))
}
