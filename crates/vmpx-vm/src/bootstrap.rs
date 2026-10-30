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
    /// Register that held the statically anchored table base at dispatch.
    pub table_register: Option<Register>,
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
            OpKind::Memory => {
                // Stack slots are stored as aligned qwords; a narrower read
                // (e.g. the VIP setup `mov esi,[rsp+80h]`) takes its low bytes.
                let size = u32::try_from(instruction.memory_size().size()).ok()?;
                if size == 0 || size > 8 {
                    return None;
                }
                let address = self.stack_address(instruction)?;
                let value = self.stack.get(&address).copied()?;
                Some(value & bit_mask(size.saturating_mul(8)))
            }
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
            // x86 masks the count to 5 bits for 8/16/32-bit operands, so a narrow
            // rotate can carry a count wider than the operand. Rather than underflow
            // `bits - count` and fabricate a value, refuse: drop the destination so
            // recovery fails closed instead of guessing.
            if count > bits {
                self.invalidate_register(destination);
                return;
            }
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

/// Recognizes the terminal dispatch of a bootstrap trace.
///
/// Protector bootstraps dispatch either with an indirect branch through a
/// register or with the equivalent `push reg; ret` tail, which transfers to the
/// same register value. Both forms must be accepted, because which one a given
/// VM instance uses is a mutation, not a semantic difference.
fn terminal_handler_register(trace: &vmpx_disasm::DirectJumpTrace) -> Option<Register> {
    let decoded = trace.instructions().collect::<Vec<_>>();
    let terminal = decoded.last()?.instruction;
    if terminal.flow_control() == FlowControl::IndirectBranch
        && terminal.op0_kind() == OpKind::Register
    {
        return Some(terminal.op0_register().full_register());
    }
    // `push reg; ret` is an indirect jump through that register.
    if terminal.mnemonic() == Mnemonic::Ret && terminal.op_count() == 0 {
        let previous = decoded.get(decoded.len().checked_sub(2)?)?.instruction;
        if previous.mnemonic() == Mnemonic::Push && previous.op0_kind() == OpKind::Register {
            return Some(previous.op0_register().full_register());
        }
    }
    None
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
    // The dispatch window begins after VIP setup: the VIP-relative token load is
    // the real dispatch source, not an earlier `lea vip,[vip + const]` that
    // establishes the full-width VIP (its address base is also the VIP).
    let source_index = decoded
        .iter()
        .position(|decoded| crate::dispatch_extract::is_source_load(&decoded.instruction, vip));
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
    handler_override: Option<Va>,
    keyless: bool,
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
    // A keyless VM never consults the rolling key, so a non-concrete key
    // register is immaterial; a table-indexed dispatch anchors on the recovered
    // table base rather than a rolling handler register.
    let rolling_key = if keyless {
        machine.read_register(roles.rolling_key).unwrap_or(0)
    } else {
        value(roles.rolling_key, "rolling key")?
    };
    let handler = match handler_override {
        Some(base) => base,
        None => Va(value(roles.handler, "handler anchor")?),
    };
    Ok(RelativeDispatchState {
        vip: Va(value(roles.vip, "VIP")?),
        rolling_key,
        handler,
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
                    let table = candidate.codec.table;
                    let handler_override = table.and_then(|table| table.table_base);
                    let mut initial_state = match evaluate_initial_state(
                        pe,
                        entry,
                        roles,
                        candidate.encoded_load,
                        limits,
                        handler_override,
                        table.is_some(),
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
                    // A keyless (table-indexed) dispatch does not consult the
                    // rolling-key register, so every rolling-key role assignment
                    // yields the same VM; collapse them to one candidate by
                    // ignoring the rolling-key role in the identity.
                    if !recovered
                        .iter()
                        .any(|existing: &BootstrapDispatchRecovery| {
                            existing.roles.vip == roles.vip
                                && existing.roles.handler == roles.handler
                                && (table.is_some()
                                    || existing.roles.rolling_key == roles.rolling_key)
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
                            table_register: candidate.table_register,
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

#[cfg(test)]
mod tests {
    use super::*;
    use iced_x86::Decoder;

    fn decode(bytes: &[u8]) -> Instruction {
        Decoder::new(64, bytes, 0).decode()
    }

    #[test]
    fn narrow_rotate_count_wider_than_operand_fails_closed() {
        // `rol ax, 0x14`: 16-bit operand, count masked to 5 bits is 20 > 16. This
        // previously underflowed `bits - count` and panicked; now the destination
        // is dropped so recovery refuses the entry rather than fabricating a value.
        let instruction = decode(&[0x66, 0xC1, 0xC0, 0x14]);
        assert_eq!(instruction.mnemonic(), Mnemonic::Rol);
        let mut machine = ConcreteBootstrapMachine::default();
        machine.write_register(Register::RAX, 0x1234_5678_9abc_def0);
        machine.execute(&instruction);
        assert_eq!(machine.read_register(Register::RAX), None);
    }

    #[test]
    fn narrow_rotate_count_within_operand_is_unchanged() {
        // `rol ax, 4`: count 4 <= 16 bits stays on the concrete path (byte-identical).
        let instruction = decode(&[0x66, 0xC1, 0xC0, 0x04]);
        assert_eq!(instruction.mnemonic(), Mnemonic::Rol);
        let mut machine = ConcreteBootstrapMachine::default();
        machine.write_register(Register::RAX, 0x0000_0000_0000_1234);
        machine.execute(&instruction);
        assert_eq!(machine.read_register(Register::AX), Some(0x2341));
    }
}
