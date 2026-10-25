use std::collections::{BTreeMap, BTreeSet, VecDeque};

use iced_x86::{FlowControl, InstructionInfoFactory, Mnemonic, OpAccess, OpKind, Register};
use serde::Serialize;
use vmpx_pe::{PeImage, Rva};

use crate::{DisasmError, decode_one, is_executable_rva};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegisterDispatchProofLimits {
    pub max_states: usize,
    pub max_stack_depth: usize,
}

impl Default for RegisterDispatchProofLimits {
    fn default() -> Self {
        Self {
            max_states: 50_000,
            max_stack_depth: 256,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RegisterDispatchProof {
    pub entry_points: Vec<Rva>,
    pub dispatch_register: String,
    pub all_paths_dispatch: bool,
    pub acyclic: bool,
    pub resource_limited: bool,
    pub explored_states: usize,
    pub dispatch_sites: Vec<Rva>,
    pub push_return_sites: Vec<Rva>,
    pub violations: Vec<String>,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct ProofState {
    rva: Rva,
    machine: ProofMachineState,
}

#[derive(Clone, Debug)]
struct ProofNode {
    successors: Vec<ProofState>,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum AbstractValue {
    Unknown,
    Constant(u64),
    EntryRegister(u8),
    Offset(Box<Self>, i64),
    Masked(Box<Self>, u64),
}

impl AbstractValue {
    fn offset(self, delta: i64) -> Self {
        if delta == 0 {
            return self;
        }
        match self {
            Self::Unknown => Self::Unknown,
            Self::Constant(value) => Self::Constant(value.wrapping_add_signed(delta)),
            Self::Offset(base, previous) => previous
                .checked_add(delta)
                .map_or(Self::Unknown, |combined| Self::Offset(base, combined)),
            value => Self::Offset(Box::new(value), delta),
        }
    }

    fn masked(self, mask: u64) -> Self {
        match self {
            Self::Unknown => Self::Unknown,
            Self::Constant(value) => Self::Constant(value & mask),
            value => Self::Masked(Box::new(value), mask),
        }
    }

    fn base_and_offset(&self) -> Option<(&Self, i64)> {
        match self {
            Self::Unknown => None,
            Self::Offset(base, offset) => Some((base, *offset)),
            value => Some((value, 0)),
        }
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct RegisterFile {
    values: [AbstractValue; 16],
}

impl RegisterFile {
    fn entry() -> Self {
        Self {
            values: std::array::from_fn(|index| {
                AbstractValue::EntryRegister(u8::try_from(index).expect("GPR index fits in u8"))
            }),
        }
    }

    fn get(&self, register: Register) -> AbstractValue {
        gpr_index(register).map_or(AbstractValue::Unknown, |index| self.values[index].clone())
    }

    fn set(&mut self, register: Register, value: AbstractValue) {
        if let Some(index) = gpr_index(register) {
            self.values[index] = value;
        }
    }

    fn original_register(register: Register) -> AbstractValue {
        gpr_index(register).map_or(AbstractValue::Unknown, |index| {
            AbstractValue::EntryRegister(u8::try_from(index).expect("GPR index fits in u8"))
        })
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct StackSlot {
    address: AbstractValue,
    value: AbstractValue,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum DirectionFlag {
    Clear,
    Set,
    Unknown,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct ProofMachineState {
    registers: RegisterFile,
    stack: Vec<StackSlot>,
    direction: DirectionFlag,
}

impl ProofMachineState {
    fn entry() -> Self {
        Self {
            registers: RegisterFile::entry(),
            stack: Vec::new(),
            direction: DirectionFlag::Unknown,
        }
    }

    fn has_original_dispatch(&self, dispatch_register: Register) -> bool {
        self.registers.get(dispatch_register) == RegisterFile::original_register(dispatch_register)
    }
}

fn gpr_index(register: Register) -> Option<usize> {
    Some(match canonical(register) {
        Register::RAX => 0,
        Register::RCX => 1,
        Register::RDX => 2,
        Register::RBX => 3,
        Register::RSP => 4,
        Register::RBP => 5,
        Register::RSI => 6,
        Register::RDI => 7,
        Register::R8 => 8,
        Register::R9 => 9,
        Register::R10 => 10,
        Register::R11 => 11,
        Register::R12 => 12,
        Register::R13 => 13,
        Register::R14 => 14,
        Register::R15 => 15,
        _ => return None,
    })
}

fn canonical(register: Register) -> Register {
    register.full_register()
}

fn is_indirect_register_dispatch(instruction: &iced_x86::Instruction, register: Register) -> bool {
    instruction.flow_control() == FlowControl::IndirectBranch
        && instruction.op0_kind() == OpKind::Register
        && canonical(instruction.op0_register()) == canonical(register)
}

fn access_writes(access: OpAccess) -> bool {
    matches!(
        access,
        OpAccess::Write | OpAccess::CondWrite | OpAccess::ReadWrite | OpAccess::ReadCondWrite
    )
}

fn is_full_architectural_write(register: Register) -> bool {
    let full = canonical(register);
    register == full
        || (register.size() == 4 && full.size() == 8 && register.full_register32() == register)
}

fn immediate_value(instruction: &iced_x86::Instruction, operand: u32) -> Option<u64> {
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

fn signed_displacement(value: u64) -> i64 {
    i64::from_ne_bytes(value.to_ne_bytes())
}

fn operand_value(
    instruction: &iced_x86::Instruction,
    operand: u32,
    registers: &RegisterFile,
) -> AbstractValue {
    match instruction.op_kind(operand) {
        OpKind::Register => registers.get(instruction.op_register(operand)),
        _ => immediate_value(instruction, operand)
            .map_or(AbstractValue::Unknown, AbstractValue::Constant),
    }
}

fn memory_address(
    base: Register,
    index: Register,
    displacement: u64,
    registers: &RegisterFile,
) -> AbstractValue {
    if index != Register::None {
        return AbstractValue::Unknown;
    }
    let base = if base == Register::None {
        AbstractValue::Constant(0)
    } else {
        registers.get(base)
    };
    base.offset(signed_displacement(displacement))
}

fn written_value(
    instruction: &iced_x86::Instruction,
    target: Register,
    registers: &RegisterFile,
) -> AbstractValue {
    if instruction.op0_kind() != OpKind::Register
        || canonical(instruction.op0_register()) != canonical(target)
        || !is_full_architectural_write(instruction.op0_register())
    {
        return AbstractValue::Unknown;
    }

    match instruction.mnemonic() {
        Mnemonic::Mov => operand_value(instruction, 1, registers),
        Mnemonic::Lea
            if instruction.memory_index() == Register::None
                && !matches!(instruction.memory_segment(), Register::FS | Register::GS) =>
        {
            memory_address(
                instruction.memory_base(),
                instruction.memory_index(),
                instruction.memory_displacement64(),
                registers,
            )
        }
        Mnemonic::Add | Mnemonic::Sub => {
            let right = operand_value(instruction, 1, registers);
            let AbstractValue::Constant(right) = right else {
                return AbstractValue::Unknown;
            };
            let delta = signed_displacement(right);
            let delta = if instruction.mnemonic() == Mnemonic::Sub {
                let Some(delta) = delta.checked_neg() else {
                    return AbstractValue::Unknown;
                };
                delta
            } else {
                delta
            };
            registers.get(target).offset(delta)
        }
        Mnemonic::Inc => registers.get(target).offset(1),
        Mnemonic::Dec => registers.get(target).offset(-1),
        Mnemonic::And => {
            let Some(mask) = immediate_value(instruction, 1) else {
                return AbstractValue::Unknown;
            };
            registers.get(target).masked(mask)
        }
        _ => AbstractValue::Unknown,
    }
}

fn values_proven_equal(left: &AbstractValue, right: &AbstractValue) -> bool {
    !matches!(left, AbstractValue::Unknown) && left == right
}

fn relative_distance(from: &AbstractValue, to: &AbstractValue) -> Option<u64> {
    if let (AbstractValue::Constant(from), AbstractValue::Constant(to)) = (from, to) {
        Some(to.wrapping_sub(*from))
    } else {
        let (from_base, from_offset) = from.base_and_offset()?;
        let (to_base, to_offset) = to.base_and_offset()?;
        (from_base == to_base)
            .then(|| u64::from_ne_bytes(to_offset.wrapping_sub(from_offset).to_ne_bytes()))
    }
}

fn ranges_proven_disjoint(
    protected_start: &AbstractValue,
    protected_size: u64,
    write_start: &AbstractValue,
    write_size: u64,
) -> bool {
    if protected_size == 0 || write_size == 0 {
        return false;
    }
    let Some(distance) = relative_distance(protected_start, write_start) else {
        return false;
    };
    let modulus = u128::from(u64::MAX) + 1;
    let forward = u128::from(distance);
    forward >= u128::from(protected_size) && modulus - forward >= u128::from(write_size)
}

fn is_push(instruction: &iced_x86::Instruction) -> bool {
    matches!(
        instruction.mnemonic(),
        Mnemonic::Push | Mnemonic::Pushf | Mnemonic::Pushfd | Mnemonic::Pushfq
    )
}

fn is_pop(instruction: &iced_x86::Instruction) -> bool {
    matches!(
        instruction.mnemonic(),
        Mnemonic::Pop | Mnemonic::Popf | Mnemonic::Popfd | Mnemonic::Popfq
    )
}

fn protected_slots(
    machine: &ProofMachineState,
    dispatch_register: Register,
) -> impl Iterator<Item = &StackSlot> {
    let original = RegisterFile::original_register(dispatch_register);
    machine
        .stack
        .iter()
        .filter(move |slot| slot.value == original)
}

fn prove_memory_writes_disjoint(
    factory: &mut InstructionInfoFactory,
    instruction: &iced_x86::Instruction,
    machine: &ProofMachineState,
    dispatch_register: Register,
) -> Result<(), String> {
    let protected: Vec<_> = protected_slots(machine, dispatch_register).collect();
    if protected.is_empty() {
        return Ok(());
    }

    if instruction.has_rep_prefix() {
        if instruction.mnemonic() != Mnemonic::Movsb || machine.direction != DirectionFlag::Clear {
            return Err(
                "repeated memory write cannot prove saved-dispatch-slot integrity".to_owned(),
            );
        }
        let AbstractValue::Constant(count) = machine.registers.get(Register::RCX) else {
            return Err("REP MOVSB count is not constant while dispatch is saved".to_owned());
        };
        let start = machine.registers.get(Register::RDI);
        if protected
            .iter()
            .any(|slot| !ranges_proven_disjoint(&slot.address, 8, &start, count))
        {
            return Err("REP MOVSB may overwrite a saved dispatch register".to_owned());
        }
        return Ok(());
    }

    let info = factory.info(instruction);
    for used in info
        .used_memory()
        .iter()
        .filter(|used| access_writes(used.access()))
    {
        if matches!(used.segment(), Register::FS | Register::GS) {
            return Err("segmented memory write may alias a saved dispatch register".to_owned());
        }
        let start = memory_address(
            used.base(),
            used.index(),
            used.displacement(),
            &machine.registers,
        );
        let Ok(size) = u64::try_from(used.memory_size().size()) else {
            return Err("memory-write width does not fit the proof model".to_owned());
        };
        if protected
            .iter()
            .any(|slot| !ranges_proven_disjoint(&slot.address, 8, &start, size))
        {
            return Err("memory write may overwrite a saved dispatch register".to_owned());
        }
    }
    Ok(())
}

fn update_written_registers(
    factory: &mut InstructionInfoFactory,
    instruction: &iced_x86::Instruction,
    machine: &mut ProofMachineState,
) {
    let info = factory.info(instruction);
    let mut writes = BTreeSet::new();
    for used in info
        .used_registers()
        .iter()
        .filter(|used| access_writes(used.access()))
    {
        let register = canonical(used.register());
        if gpr_index(register).is_some() {
            writes.insert(register);
        }
    }
    let updates: Vec<_> = writes
        .into_iter()
        .map(|register| {
            (
                register,
                written_value(instruction, register, &machine.registers),
            )
        })
        .collect();
    for (register, value) in updates {
        machine.registers.set(register, value);
    }
}

fn step_machine(
    factory: &mut InstructionInfoFactory,
    instruction: &iced_x86::Instruction,
    machine: &mut ProofMachineState,
    dispatch_register: Register,
    limits: RegisterDispatchProofLimits,
) -> Result<(), String> {
    if is_push(instruction) {
        if machine.stack.len() >= limits.max_stack_depth {
            return Err("dispatch proof reached its saved-stack depth limit".to_owned());
        }
        let rsp = machine.registers.get(Register::RSP).offset(-8);
        let value = if instruction.mnemonic() == Mnemonic::Push
            && instruction.op0_kind() == OpKind::Register
        {
            machine.registers.get(instruction.op0_register())
        } else {
            AbstractValue::Unknown
        };
        machine.stack.push(StackSlot {
            address: rsp.clone(),
            value,
        });
        machine.registers.set(Register::RSP, rsp);
        return Ok(());
    }

    if is_pop(instruction) {
        let rsp = machine.registers.get(Register::RSP);
        let value = if let Some(slot) = machine.stack.pop() {
            if !values_proven_equal(&rsp, &slot.address) {
                return Err("stack pointer no longer addresses the modeled saved value".to_owned());
            }
            slot.value
        } else {
            AbstractValue::Unknown
        };
        if instruction.mnemonic() == Mnemonic::Pop && instruction.op0_kind() == OpKind::Register {
            machine.registers.set(instruction.op0_register(), value);
        }
        machine.registers.set(Register::RSP, rsp.offset(8));
        if matches!(
            instruction.mnemonic(),
            Mnemonic::Popf | Mnemonic::Popfd | Mnemonic::Popfq
        ) {
            machine.direction = DirectionFlag::Unknown;
        }
        return Ok(());
    }

    prove_memory_writes_disjoint(factory, instruction, machine, dispatch_register)?;
    match instruction.mnemonic() {
        Mnemonic::Cld => machine.direction = DirectionFlag::Clear,
        Mnemonic::Std => machine.direction = DirectionFlag::Set,
        _ => {}
    }
    update_written_registers(factory, instruction, machine);
    Ok(())
}

fn checked_successor(
    pe: &PeImage<'_>,
    from: Rva,
    target: Option<Rva>,
    machine: &ProofMachineState,
    violations: &mut BTreeSet<String>,
) -> Option<ProofState> {
    let Some(target) = target else {
        violations.insert(format!("control transfer at {from} has no direct target"));
        return None;
    };
    if !is_executable_rva(pe, target) {
        violations.insert(format!(
            "control transfer at {from} targets non-executable {target}"
        ));
        return None;
    }
    Some(ProofState {
        rva: target,
        machine: machine.clone(),
    })
}

fn graph_is_acyclic(nodes: &BTreeMap<ProofState, ProofNode>) -> bool {
    let mut indegree: BTreeMap<_, usize> = nodes.keys().map(|state| (state.clone(), 0)).collect();
    for node in nodes.values() {
        for successor in &node.successors {
            if let Some(value) = indegree.get_mut(successor) {
                *value = value.saturating_add(1);
            }
        }
    }
    let mut queue: VecDeque<_> = indegree
        .iter()
        .filter_map(|(state, degree)| (*degree == 0).then_some(state.clone()))
        .collect();
    let mut removed = 0_usize;
    while let Some(state) = queue.pop_front() {
        removed = removed.saturating_add(1);
        if let Some(node) = nodes.get(&state) {
            for successor in &node.successors {
                if let Some(degree) = indegree.get_mut(successor) {
                    *degree = degree.saturating_sub(1);
                    if *degree == 0 {
                        queue.push_back(successor.clone());
                    }
                }
            }
        }
    }
    removed == nodes.len()
}

/// Proves that every direct-control-flow path beginning at `entry_points`
/// reaches an indirect transfer through `dispatch_register` or the equivalent
/// adjacent `push dispatch_register; ret` sequence.
///
/// The proof is deliberately strict. A dispatch-register clobber is accepted
/// only while the original value is held in a modeled stack slot whose memory
/// integrity is proven until an exact restoration. Calls, unresolved memory
/// aliases, malformed successors, other terminal transfers, graph cycles, and
/// resource exhaustion all prevent success.
///
/// # Errors
///
/// Returns an error if no entry point was supplied or an entry is not valid
/// executable code. Failures after entry validation are retained as proof
/// violations.
#[allow(clippy::too_many_lines)]
pub fn prove_register_dispatch_postlude(
    pe: &PeImage<'_>,
    entry_points: &[Rva],
    dispatch_register: Register,
    limits: RegisterDispatchProofLimits,
) -> Result<RegisterDispatchProof, DisasmError> {
    if entry_points.is_empty() {
        return Err(DisasmError::Unmapped {
            rva: Rva(0),
            message: "dispatch proof requires at least one entry".to_owned(),
        });
    }
    for entry in entry_points {
        if !is_executable_rva(pe, *entry) {
            return Err(DisasmError::Unmapped {
                rva: *entry,
                message: "dispatch proof entry is not executable".to_owned(),
            });
        }
        decode_one(pe, *entry)?;
    }

    let mut queue: VecDeque<_> = entry_points
        .iter()
        .copied()
        .map(|rva| ProofState {
            rva,
            machine: ProofMachineState::entry(),
        })
        .collect();
    let mut queued: BTreeSet<_> = queue.iter().cloned().collect();
    let mut nodes = BTreeMap::new();
    let mut dispatch_sites = BTreeSet::new();
    let mut push_return_sites = BTreeSet::new();
    let mut violations = BTreeSet::new();
    let mut resource_limited = false;
    let mut factory = InstructionInfoFactory::new();

    while let Some(state) = queue.pop_front() {
        if nodes.contains_key(&state) {
            continue;
        }
        if nodes.len() >= limits.max_states {
            resource_limited = true;
            violations.insert(format!(
                "dispatch proof reached its state limit at {}",
                state.rva
            ));
            break;
        }
        let decoded = match decode_one(pe, state.rva) {
            Ok(decoded) => decoded,
            Err(error) => {
                violations.insert(error.to_string());
                nodes.insert(
                    state,
                    ProofNode {
                        successors: Vec::new(),
                    },
                );
                continue;
            }
        };
        let instruction = &decoded.instruction;
        let flow = instruction.flow_control();
        if is_indirect_register_dispatch(instruction, dispatch_register) {
            if state.machine.has_original_dispatch(dispatch_register)
                && state.machine.stack.is_empty()
            {
                dispatch_sites.insert(state.rva);
            } else {
                violations.insert(format!(
                    "dispatch at {} does not use the proven original {:?} value with a balanced save stack",
                    state.rva,
                    canonical(dispatch_register)
                ));
            }
            nodes.insert(
                state,
                ProofNode {
                    successors: Vec::new(),
                },
            );
            continue;
        }
        if flow == FlowControl::Return {
            let rsp = state.machine.registers.get(Register::RSP);
            let pushed_original = state.machine.stack.last().is_some_and(|slot| {
                values_proven_equal(&rsp, &slot.address)
                    && slot.value == RegisterFile::original_register(dispatch_register)
            });
            if pushed_original {
                push_return_sites.insert(state.rva);
            } else {
                violations.insert(format!(
                    "return at {} is not reached with the original dispatch value at the stack top",
                    state.rva
                ));
            }
            nodes.insert(
                state,
                ProofNode {
                    successors: Vec::new(),
                },
            );
            continue;
        }
        if matches!(flow, FlowControl::Call | FlowControl::IndirectCall) {
            violations.insert(format!("unmodeled call at {}", state.rva));
            nodes.insert(
                state,
                ProofNode {
                    successors: Vec::new(),
                },
            );
            continue;
        }
        if flow == FlowControl::IndirectBranch {
            violations.insert(format!(
                "indirect branch at {} does not use {:?}",
                state.rva,
                canonical(dispatch_register)
            ));
            nodes.insert(
                state,
                ProofNode {
                    successors: Vec::new(),
                },
            );
            continue;
        }
        if matches!(
            flow,
            FlowControl::Interrupt | FlowControl::Exception | FlowControl::XbeginXabortXend
        ) {
            violations.insert(format!("unsupported terminal {flow:?} at {}", state.rva));
            nodes.insert(
                state,
                ProofNode {
                    successors: Vec::new(),
                },
            );
            continue;
        }

        let mut next_machine = state.machine.clone();
        if let Err(reason) = step_machine(
            &mut factory,
            instruction,
            &mut next_machine,
            dispatch_register,
            limits,
        ) {
            violations.insert(format!("instruction at {}: {reason}", state.rva));
            nodes.insert(
                state,
                ProofNode {
                    successors: Vec::new(),
                },
            );
            continue;
        }

        let mut successors = Vec::new();
        match flow {
            FlowControl::Next => {
                if let Some(successor) = checked_successor(
                    pe,
                    state.rva,
                    decoded.next_rva(),
                    &next_machine,
                    &mut violations,
                ) {
                    successors.push(successor);
                }
            }
            FlowControl::UnconditionalBranch => {
                if let Some(successor) = checked_successor(
                    pe,
                    state.rva,
                    decoded.direct_target_rva(pe),
                    &next_machine,
                    &mut violations,
                ) {
                    successors.push(successor);
                }
            }
            FlowControl::ConditionalBranch => {
                for target in [decoded.direct_target_rva(pe), decoded.next_rva()] {
                    if let Some(successor) =
                        checked_successor(pe, state.rva, target, &next_machine, &mut violations)
                    {
                        successors.push(successor);
                    }
                }
            }
            FlowControl::Call
            | FlowControl::IndirectCall
            | FlowControl::IndirectBranch
            | FlowControl::Return
            | FlowControl::Interrupt
            | FlowControl::Exception
            | FlowControl::XbeginXabortXend => unreachable!("terminal flow handled above"),
        }
        for successor in &successors {
            if queued.insert(successor.clone()) {
                queue.push_back(successor.clone());
            }
        }
        nodes.insert(state, ProofNode { successors });
    }

    let acyclic = !resource_limited && graph_is_acyclic(&nodes);
    if !acyclic && !resource_limited {
        violations.insert("postlude control-flow graph contains a cycle".to_owned());
    }
    let explored_states = nodes.len();
    let has_dispatch = !dispatch_sites.is_empty() || !push_return_sites.is_empty();
    let all_paths_dispatch = !resource_limited && acyclic && violations.is_empty() && has_dispatch;
    Ok(RegisterDispatchProof {
        entry_points: entry_points.to_vec(),
        dispatch_register: format!("{:?}", canonical(dispatch_register)),
        all_paths_dispatch,
        acyclic,
        resource_limited,
        explored_states,
        dispatch_sites: dispatch_sites.into_iter().collect(),
        push_return_sites: push_return_sites.into_iter().collect(),
        violations: violations.into_iter().collect(),
    })
}

#[cfg(test)]
mod tests {
    use iced_x86::code_asm::*;

    use super::*;

    fn executable_pe64(code: &[u8]) -> Vec<u8> {
        let mut data = vec![0_u8; 0x400];
        data[0..2].copy_from_slice(&0x5A4D_u16.to_le_bytes());
        data[0x3C..0x40].copy_from_slice(&(0x80_u32).to_le_bytes());
        data[0x80..0x84].copy_from_slice(&0x0000_4550_u32.to_le_bytes());
        let coff = 0x84;
        data[coff..coff + 2].copy_from_slice(&(0x8664_u16).to_le_bytes());
        data[coff + 2..coff + 4].copy_from_slice(&(1_u16).to_le_bytes());
        data[coff + 16..coff + 18].copy_from_slice(&(0xF0_u16).to_le_bytes());
        data[coff + 18..coff + 20].copy_from_slice(&(0x2022_u16).to_le_bytes());
        let optional = coff + 20;
        data[optional..optional + 2].copy_from_slice(&0x20B_u16.to_le_bytes());
        data[optional + 16..optional + 20].copy_from_slice(&(0x1000_u32).to_le_bytes());
        data[optional + 20..optional + 24].copy_from_slice(&(0x1000_u32).to_le_bytes());
        data[optional + 24..optional + 32].copy_from_slice(&(0x1_4000_0000_u64).to_le_bytes());
        data[optional + 32..optional + 36].copy_from_slice(&(0x1000_u32).to_le_bytes());
        data[optional + 36..optional + 40].copy_from_slice(&(0x200_u32).to_le_bytes());
        data[optional + 56..optional + 60].copy_from_slice(&(0x2000_u32).to_le_bytes());
        data[optional + 60..optional + 64].copy_from_slice(&(0x200_u32).to_le_bytes());
        data[optional + 68..optional + 70].copy_from_slice(&(3_u16).to_le_bytes());
        data[optional + 108..optional + 112].copy_from_slice(&(16_u32).to_le_bytes());
        let section = optional + 0xF0;
        data[section..section + 5].copy_from_slice(b".text");
        data[section + 8..section + 12].copy_from_slice(&(0x200_u32).to_le_bytes());
        data[section + 12..section + 16].copy_from_slice(&(0x1000_u32).to_le_bytes());
        data[section + 16..section + 20].copy_from_slice(&(0x200_u32).to_le_bytes());
        data[section + 20..section + 24].copy_from_slice(&(0x200_u32).to_le_bytes());
        data[section + 36..section + 40].copy_from_slice(&(0x6000_0020_u32).to_le_bytes());
        data[0x200..0x200 + code.len()].copy_from_slice(code);
        data
    }

    fn proof_from_assembler(
        mut assembler: CodeAssembler,
        limits: RegisterDispatchProofLimits,
    ) -> RegisterDispatchProof {
        let code = assembler
            .assemble(0x1_4000_1000)
            .expect("assemble proof fixture");
        let data = executable_pe64(&code);
        let pe = PeImage::parse(&data).expect("valid synthetic PE");
        prove_register_dispatch_postlude(&pe, &[Rva(0x1000)], Register::R14, limits)
            .expect("proof graph must decode")
    }

    #[test]
    fn proves_both_conditional_paths_dispatch() {
        // jz +4; nop; jmp r14; push r14; ret
        let data = executable_pe64(&[0x74, 0x04, 0x90, 0x41, 0xFF, 0xE6, 0x41, 0x56, 0xC3]);
        let pe = PeImage::parse(&data).expect("valid synthetic PE");
        let proof = prove_register_dispatch_postlude(
            &pe,
            &[Rva(0x1000)],
            Register::R14,
            RegisterDispatchProofLimits::default(),
        )
        .expect("proof graph must decode");

        assert!(proof.all_paths_dispatch, "{:?}", proof.violations);
        assert!(proof.acyclic);
        assert_eq!(proof.dispatch_sites, vec![Rva(0x1003)]);
        assert_eq!(proof.push_return_sites, vec![Rva(0x1008)]);
    }

    #[test]
    fn rejects_dispatch_register_clobber() {
        // xor r14,r14; jmp r14
        let data = executable_pe64(&[0x4D, 0x31, 0xF6, 0x41, 0xFF, 0xE6]);
        let pe = PeImage::parse(&data).expect("valid synthetic PE");
        let proof = prove_register_dispatch_postlude(
            &pe,
            &[Rva(0x1000)],
            Register::R14,
            RegisterDispatchProofLimits::default(),
        )
        .expect("proof graph must decode");

        assert!(!proof.all_paths_dispatch);
        assert!(
            proof
                .violations
                .iter()
                .any(|value| value.contains("proven original"))
        );
    }

    #[test]
    fn proves_saved_restored_dispatch_with_disjoint_rep_copy() {
        let mut assembler = CodeAssembler::new(64).expect("x64 assembler");
        assembler.mov(r15, rsp).expect("capture replacement stack");
        assembler
            .sub(r15, 0x100)
            .expect("reserve replacement stack");
        assembler.mov(rsp, r15).expect("switch stack");
        assembler.push(r14).expect("save dispatch");
        assembler.push(rsi).expect("nested save");
        assembler.pushfq().expect("save flags");
        assembler.not(r14).expect("temporary dispatch clobber");
        assembler.mov(rdi, r15).expect("copy destination");
        assembler.mov(rcx, 0x80_u64).expect("bounded copy size");
        assembler.cld().expect("forward string direction");
        assembler.rep().movsb().expect("copy old stack window");
        assembler.popfq().expect("restore flags");
        assembler.pop(rsi).expect("restore nested save");
        assembler.pop(r14).expect("restore dispatch");
        assembler.jmp(r14).expect("dispatch");

        let proof = proof_from_assembler(assembler, RegisterDispatchProofLimits::default());
        assert!(proof.all_paths_dispatch, "{:?}", proof.violations);
        assert!(proof.acyclic);
    }

    #[test]
    fn rejects_saved_dispatch_slot_overwrites_and_wrong_restore_order() {
        let mut direct_overwrite = CodeAssembler::new(64).expect("x64 assembler");
        direct_overwrite.push(r14).expect("save dispatch");
        direct_overwrite.not(r14).expect("clobber dispatch");
        direct_overwrite
            .mov(qword_ptr(rsp), 0)
            .expect("overwrite saved dispatch");
        direct_overwrite.pop(r14).expect("invalid restore");
        direct_overwrite.jmp(r14).expect("dispatch");
        let proof = proof_from_assembler(direct_overwrite, RegisterDispatchProofLimits::default());
        assert!(!proof.all_paths_dispatch);
        assert!(
            proof
                .violations
                .iter()
                .any(|value| value.contains("overwrite"))
        );

        let mut alias_overwrite = CodeAssembler::new(64).expect("x64 assembler");
        alias_overwrite.push(r14).expect("save dispatch");
        alias_overwrite.mov(rax, rsp).expect("alias saved slot");
        alias_overwrite
            .mov(qword_ptr(rax), 0)
            .expect("overwrite through alias");
        alias_overwrite.pop(r14).expect("invalid restore");
        alias_overwrite.jmp(r14).expect("dispatch");
        assert!(
            !proof_from_assembler(alias_overwrite, RegisterDispatchProofLimits::default())
                .all_paths_dispatch
        );

        let mut wrong_order = CodeAssembler::new(64).expect("x64 assembler");
        wrong_order.push(r14).expect("save dispatch");
        wrong_order.push(rax).expect("nested save");
        wrong_order.not(r14).expect("clobber dispatch");
        wrong_order.pop(r14).expect("pop wrong value");
        wrong_order.jmp(r14).expect("invalid dispatch");
        assert!(
            !proof_from_assembler(wrong_order, RegisterDispatchProofLimits::default())
                .all_paths_dispatch
        );
    }

    #[test]
    fn enforces_saved_stack_depth_limit() {
        let mut assembler = CodeAssembler::new(64).expect("x64 assembler");
        assembler.push(r14).expect("first save");
        assembler.push(rax).expect("excess nested save");
        assembler.pop(rax).expect("restore nested save");
        assembler.pop(r14).expect("restore dispatch");
        assembler.jmp(r14).expect("dispatch");
        let proof = proof_from_assembler(
            assembler,
            RegisterDispatchProofLimits {
                max_stack_depth: 1,
                ..RegisterDispatchProofLimits::default()
            },
        );
        assert!(!proof.all_paths_dispatch);
        assert!(
            proof
                .violations
                .iter()
                .any(|value| value.contains("depth limit"))
        );
    }
}
