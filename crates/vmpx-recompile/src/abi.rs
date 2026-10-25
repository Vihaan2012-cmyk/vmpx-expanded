use crate::{
    CallTarget, CompileError, Function, Operation, PhysicalRegister, SpillSlot, VirtualRegister,
    Width,
};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

pub(crate) const MAX_SPILL_SLOTS: u16 = 4_096;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegisterAssignment {
    pub virtual_register: VirtualRegister,
    pub physical_register: PhysicalRegister,
}

impl RegisterAssignment {
    #[must_use]
    pub const fn new(
        virtual_register: VirtualRegister,
        physical_register: PhysicalRegister,
    ) -> Self {
        Self {
            virtual_register,
            physical_register,
        }
    }
}

/// Win64 frame facts. Preservation is emitted automatically in listed order.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct StackFrame {
    pub stack_allocation: u32,
    pub preserve: Vec<PhysicalRegister>,
    /// Require the body stack state to satisfy Win64 call-site alignment and
    /// shadow-space requirements.
    pub call_ready: bool,
    /// Slots the caller has already populated before the body runs. A composed
    /// routine uses these for its shared input frame, so a load from one is
    /// defined even though no store appears in the emitted body.
    pub preinitialized_slots: Vec<SpillSlot>,
}

#[derive(Clone, Debug)]
pub(crate) struct ValidatedFunction {
    pub registers: BTreeMap<u32, PhysicalRegister>,
}

fn operation_register_accesses(
    operation: &Operation,
    read: &mut BTreeSet<VirtualRegister>,
    written: &mut BTreeSet<VirtualRegister>,
) {
    fn add_source(source: crate::BinarySource, read: &mut BTreeSet<VirtualRegister>) {
        if let crate::BinarySource::Register(register) = source {
            read.insert(register);
        }
    }
    match operation {
        Operation::Label(_)
        | Operation::Jump(_)
        | Operation::BranchIf { .. }
        | Operation::Return => {}
        Operation::MoveImmediate { destination, .. } | Operation::SpillLoad { destination, .. } => {
            written.insert(*destination);
        }
        Operation::Unary { destination, .. } => {
            read.insert(*destination);
            written.insert(*destination);
        }
        Operation::Move {
            destination,
            source,
        }
        | Operation::Resize {
            destination,
            source,
            ..
        } => {
            read.insert(*source);
            written.insert(*destination);
        }
        Operation::Binary {
            destination,
            source,
            ..
        } => {
            read.insert(*destination);
            written.insert(*destination);
            add_source(*source, read);
        }
        Operation::Shift {
            destination, count, ..
        } => {
            read.insert(*destination);
            written.insert(*destination);
            if let crate::ShiftCount::Register(register) = count {
                read.insert(*register);
            }
        }
        Operation::Load {
            destination,
            address,
        } => {
            read.insert(address.base);
            written.insert(*destination);
        }
        Operation::Store { address, source } => {
            read.insert(address.base);
            read.insert(*source);
        }
        Operation::SpillStore { source, .. } => {
            read.insert(*source);
        }
        Operation::Call {
            target,
            arguments,
            result,
        } => {
            if let CallTarget::Register(register) = target {
                read.insert(*register);
            }
            read.extend(arguments.iter().copied());
            written.extend(result.iter().copied());
        }
        Operation::Compare { left, right } => {
            read.insert(*left);
            add_source(*right, read);
        }
    }
}

fn require_same_width(left: VirtualRegister, right: VirtualRegister) -> Result<(), CompileError> {
    if left.width == right.width {
        Ok(())
    } else {
        Err(CompileError::RegisterWidthMismatch {
            expected: left.width,
            actual: right.width,
        })
    }
}

fn validate_operation_widths(operation: &Operation) -> Result<(), CompileError> {
    fn validate_source(
        destination: VirtualRegister,
        source: crate::BinarySource,
    ) -> Result<(), CompileError> {
        if let crate::BinarySource::Register(source) = source {
            require_same_width(destination, source)?;
        }
        Ok(())
    }

    match operation {
        Operation::Move {
            destination,
            source,
        } => require_same_width(*destination, *source),
        Operation::Binary {
            operation,
            destination,
            source,
        } => {
            validate_source(*destination, *source)?;
            if *operation == crate::BinaryOp::Multiply && destination.width == Width::W8 {
                return Err(CompileError::UnsupportedBinaryWidth {
                    operation: *operation,
                    width: destination.width,
                });
            }
            Ok(())
        }
        Operation::Compare { left, right } => validate_source(*left, *right),
        Operation::Resize {
            operation,
            destination,
            source,
        } => {
            let source_bits = source.width.bits();
            let destination_bits = destination.width.bits();
            let valid = match operation {
                crate::ResizeOp::Truncate => destination_bits < source_bits,
                crate::ResizeOp::ZeroExtend | crate::ResizeOp::SignExtend => {
                    destination_bits > source_bits
                }
            };
            if valid {
                Ok(())
            } else {
                Err(CompileError::InvalidResize {
                    operation: *operation,
                    source_width: source.width,
                    destination_width: destination.width,
                })
            }
        }
        Operation::Shift { count, .. } => {
            if let crate::ShiftCount::Register(register) = count
                && register.width != Width::W8
            {
                return Err(CompileError::ShiftCountNotByte {
                    register: *register,
                });
            }
            Ok(())
        }
        Operation::Load {
            address,
            destination,
        } => {
            if address.base.width != Width::W64 {
                return Err(CompileError::AddressBaseNotQword {
                    register: address.base,
                });
            }
            let _ = destination;
            Ok(())
        }
        Operation::Store { address, source } => {
            if address.base.width != Width::W64 {
                return Err(CompileError::AddressBaseNotQword {
                    register: address.base,
                });
            }
            let _ = source;
            Ok(())
        }
        Operation::SpillLoad { .. }
        | Operation::SpillStore { .. }
        | Operation::Call { .. }
        | Operation::Label(_)
        | Operation::MoveImmediate { .. }
        | Operation::Unary { .. }
        | Operation::Jump(_)
        | Operation::BranchIf { .. }
        | Operation::Return => Ok(()),
    }
}

fn operation_successors(
    function: &Function,
    labels: &BTreeMap<crate::Label, usize>,
    index: usize,
) -> Vec<usize> {
    let mut successors = Vec::with_capacity(2);
    match &function.operations[index] {
        Operation::Return => {}
        Operation::Jump(label) => {
            if let Some(target) = labels.get(label) {
                successors.push(*target);
            }
        }
        Operation::BranchIf { target, .. } => {
            if let Some(target) = labels.get(target) {
                successors.push(*target);
            }
            if index + 1 < function.operations.len() {
                successors.push(index + 1);
            }
        }
        _ if index + 1 < function.operations.len() => successors.push(index + 1),
        _ => {}
    }
    successors
}

fn spill_base(frame: &StackFrame) -> u32 {
    if frame.call_ready { 32 } else { 0 }
}

pub(crate) fn spill_displacement(frame: &StackFrame, slot: SpillSlot) -> Result<i32, CompileError> {
    if slot.0 >= MAX_SPILL_SLOTS {
        return Err(CompileError::SpillSlotLimit {
            slot,
            maximum: MAX_SPILL_SLOTS,
        });
    }
    let offset = u32::from(slot.0)
        .checked_mul(8)
        .and_then(|offset| offset.checked_add(spill_base(frame)))
        .ok_or(CompileError::CodeSizeOverflow)?;
    let required_end = offset
        .checked_add(8)
        .ok_or(CompileError::CodeSizeOverflow)?;
    if required_end > frame.stack_allocation {
        return Err(CompileError::SpillOutsideFrame {
            slot,
            required_end,
            stack_allocation: frame.stack_allocation,
        });
    }
    i32::try_from(offset).map_err(|_| CompileError::CodeSizeOverflow)
}

fn validate_spill_initialization(function: &Function) -> Result<(), CompileError> {
    let mut labels = BTreeMap::new();
    for (index, operation) in function.operations.iter().enumerate() {
        if let Operation::Label(label) = operation {
            labels.insert(*label, index);
        }
    }
    let mut entry_states: Vec<Option<BTreeMap<SpillSlot, u8>>> =
        vec![None; function.operations.len()];
    let mut initial: BTreeMap<SpillSlot, u8> = BTreeMap::new();
    for slot in &function.frame.preinitialized_slots {
        initial.insert(*slot, 8);
    }
    entry_states[0] = Some(initial);
    let mut pending = VecDeque::from([0_usize]);
    while let Some(index) = pending.pop_front() {
        let Some(mut state) = entry_states[index].clone() else {
            continue;
        };
        match &function.operations[index] {
            Operation::SpillLoad { destination, slot } => {
                let required_bytes = u8::try_from(destination.width.bits() / 8).map_err(|_| {
                    CompileError::UnsupportedWidth {
                        width: destination.width,
                    }
                })?;
                let initialized_bytes = state.get(slot).copied().unwrap_or(0);
                if initialized_bytes < required_bytes {
                    return Err(CompileError::UninitializedSpill {
                        operation_index: index,
                        slot: *slot,
                        initialized_bytes,
                        required_bytes,
                    });
                }
            }
            Operation::SpillStore { slot, source } => {
                let stored_bytes = u8::try_from(source.width.bits() / 8).map_err(|_| {
                    CompileError::UnsupportedWidth {
                        width: source.width,
                    }
                })?;
                state.insert(*slot, stored_bytes);
            }
            _ => {}
        }

        for successor in operation_successors(function, &labels, index) {
            let changed = if let Some(existing) = &mut entry_states[successor] {
                let mut intersection = BTreeMap::new();
                for (slot, existing_bytes) in existing.iter() {
                    let common_bytes = (*existing_bytes).min(state.get(slot).copied().unwrap_or(0));
                    if common_bytes != 0 {
                        intersection.insert(*slot, common_bytes);
                    }
                }
                if *existing == intersection {
                    false
                } else {
                    *existing = intersection;
                    true
                }
            } else {
                entry_states[successor] = Some(state.clone());
                true
            };
            if changed {
                pending.push_back(successor);
            }
        }
    }
    Ok(())
}

fn validate_call_abi(
    function: &Function,
    registers: &BTreeMap<u32, PhysicalRegister>,
) -> Result<(), CompileError> {
    const ARGUMENT_REGISTERS: [PhysicalRegister; 4] = [
        PhysicalRegister::Rcx,
        PhysicalRegister::Rdx,
        PhysicalRegister::R8,
        PhysicalRegister::R9,
    ];
    for (operation_index, operation) in function.operations.iter().enumerate() {
        let Operation::Call {
            target,
            arguments,
            result,
        } = operation
        else {
            continue;
        };
        if !function.frame.call_ready {
            return Err(CompileError::CallFrameNotReady { operation_index });
        }
        if arguments.len() > ARGUMENT_REGISTERS.len() {
            return Err(CompileError::TooManyCallArguments {
                operation_index,
                actual: arguments.len(),
            });
        }
        for (ordinal, (argument, expected)) in arguments
            .iter()
            .zip(ARGUMENT_REGISTERS.iter().copied())
            .enumerate()
        {
            let actual =
                registers
                    .get(&argument.id)
                    .copied()
                    .ok_or(CompileError::UnassignedRegister {
                        register: *argument,
                    })?;
            if actual != expected {
                return Err(CompileError::CallArgumentRegister {
                    ordinal,
                    expected,
                    actual,
                });
            }
        }
        if let Some(result) = result {
            let actual = registers
                .get(&result.id)
                .copied()
                .ok_or(CompileError::UnassignedRegister { register: *result })?;
            if actual != PhysicalRegister::Rax {
                return Err(CompileError::CallResultRegister { actual });
            }
        }
        if let CallTarget::Register(register) = target
            && register.width != Width::W64
        {
            return Err(CompileError::CallTargetNotQword {
                register: *register,
            });
        }
    }
    Ok(())
}

#[derive(Clone, Eq, PartialEq)]
struct CallState {
    registers: BTreeSet<u32>,
    flags: bool,
}

fn validate_call_clobbers(
    function: &Function,
    registers: &BTreeMap<u32, PhysicalRegister>,
) -> Result<(), CompileError> {
    let labels = function
        .operations
        .iter()
        .enumerate()
        .filter_map(|(index, operation)| match operation {
            Operation::Label(label) => Some((*label, index)),
            _ => None,
        })
        .collect::<BTreeMap<_, _>>();
    let mut entry_states = vec![None; function.operations.len()];
    entry_states[0] = Some(CallState {
        registers: registers.keys().copied().collect(),
        flags: true,
    });
    let mut pending = VecDeque::from([0_usize]);
    while let Some(index) = pending.pop_front() {
        let Some(mut state) = entry_states[index].clone() else {
            continue;
        };
        let operation = &function.operations[index];
        let mut read = BTreeSet::new();
        let mut written = BTreeSet::new();
        operation_register_accesses(operation, &mut read, &mut written);
        for register in read {
            let physical = registers
                .get(&register.id)
                .copied()
                .ok_or(CompileError::UnassignedRegister { register })?;
            if !physical.is_win64_nonvolatile() && !state.registers.contains(&register.id) {
                return Err(CompileError::CallClobberedRegisterRead {
                    operation_index: index,
                    register,
                    physical,
                });
            }
        }
        if matches!(operation, Operation::BranchIf { .. }) && !state.flags {
            return Err(CompileError::CallClobberedFlagsRead {
                operation_index: index,
            });
        }
        if matches!(operation, Operation::Call { .. }) {
            state.flags = false;
            state.registers.retain(|id| {
                registers
                    .get(id)
                    .is_some_and(|register| register.is_win64_nonvolatile())
            });
        } else if matches!(operation, Operation::Compare { .. }) {
            // CMP defines every status flag consumed by the backend's current
            // conditional-branch vocabulary. Other arithmetic operations have
            // width/count-specific undefined flags and are not used as a
            // blanket post-call revalidation point.
            state.flags = true;
        }
        state
            .registers
            .extend(written.into_iter().map(|register| register.id));

        for successor in operation_successors(function, &labels, index) {
            let changed = if let Some(existing) = &mut entry_states[successor] {
                let intersection = existing
                    .registers
                    .intersection(&state.registers)
                    .copied()
                    .collect();
                let merged = CallState {
                    registers: intersection,
                    flags: existing.flags && state.flags,
                };
                if *existing == merged {
                    false
                } else {
                    *existing = merged;
                    true
                }
            } else {
                entry_states[successor] = Some(state.clone());
                true
            };
            if changed {
                pending.push_back(successor);
            }
        }
    }
    Ok(())
}

/// Validates register allocation, control-flow shape, and Win64 frame facts.
///
/// # Errors
///
/// Returns a structured error for unsupported register widths, ambiguous or
/// missing assignments, ABI violations, malformed labels, unreachable
/// operations, or a final fallthrough path.
pub fn validate_function(function: &Function) -> Result<(), CompileError> {
    validate(function).map(|_| ())
}

#[allow(clippy::too_many_lines)]
pub(crate) fn validate(function: &Function) -> Result<ValidatedFunction, CompileError> {
    if function.operations.is_empty() {
        return Err(CompileError::EmptyFunction);
    }

    let mut registers = BTreeMap::new();
    let mut assigned_views = BTreeSet::new();
    let mut physical = BTreeMap::new();
    for assignment in &function.assignments {
        if !assigned_views.insert(assignment.virtual_register) {
            return Err(CompileError::DuplicateVirtualAssignment {
                register: assignment.virtual_register,
            });
        }
        let id = assignment.virtual_register.id;
        if let Some(expected) = registers.insert(id, assignment.physical_register)
            && expected != assignment.physical_register
        {
            return Err(CompileError::RegisterFamilyAssignmentMismatch {
                id,
                expected,
                actual: assignment.physical_register,
            });
        }
        if let Some(existing_id) = physical.insert(assignment.physical_register, id)
            && existing_id != id
        {
            return Err(CompileError::DuplicatePhysicalAssignment {
                register: assignment.physical_register,
            });
        }
    }

    let mut used = BTreeSet::new();
    let mut written = BTreeSet::new();
    for operation in &function.operations {
        validate_operation_widths(operation)?;
        match operation {
            Operation::SpillLoad { slot, .. } | Operation::SpillStore { slot, .. } => {
                let _ = spill_displacement(&function.frame, *slot)?;
            }
            _ => {}
        }
        let mut read = BTreeSet::new();
        let mut operation_written = BTreeSet::new();
        operation_register_accesses(operation, &mut read, &mut operation_written);
        used.extend(read);
        used.extend(operation_written.iter().copied());
        written.extend(operation_written);
    }
    for register in used {
        let _ = registers
            .get(&register.id)
            .ok_or(CompileError::UnassignedRegister { register })?;
    }
    for register in written {
        let physical_register = registers
            .get(&register.id)
            .copied()
            .ok_or(CompileError::UnassignedRegister { register })?;
        if physical_register.is_win64_nonvolatile()
            && !function.frame.preserve.contains(&physical_register)
        {
            return Err(CompileError::UnpreservedNonvolatile {
                register: physical_register,
            });
        }
    }
    validate_call_abi(function, &registers)?;
    for operation in &function.operations {
        if let Operation::Shift {
            count: crate::ShiftCount::Register(register),
            ..
        } = operation
        {
            let actual =
                registers
                    .get(&register.id)
                    .copied()
                    .ok_or(CompileError::UnassignedRegister {
                        register: *register,
                    })?;
            if actual != PhysicalRegister::Rcx {
                return Err(CompileError::ShiftCountNotCl {
                    register: *register,
                    actual,
                });
            }
        }
    }

    let mut preserved = BTreeSet::new();
    for register in &function.frame.preserve {
        if !register.is_win64_nonvolatile() {
            return Err(CompileError::PreservesVolatile {
                register: *register,
            });
        }
        if !preserved.insert(*register) {
            return Err(CompileError::DuplicatePreservedRegister {
                register: *register,
            });
        }
    }
    if !function.frame.stack_allocation.is_multiple_of(8) {
        return Err(CompileError::InvalidStackAllocation {
            bytes: function.frame.stack_allocation,
        });
    }
    if i32::try_from(function.frame.stack_allocation).is_err() {
        return Err(CompileError::StackAllocationTooLarge {
            bytes: function.frame.stack_allocation,
        });
    }
    if function.frame.call_ready {
        if function.frame.stack_allocation < 32 {
            return Err(CompileError::MissingShadowSpace {
                bytes: function.frame.stack_allocation,
            });
        }
        let pushes = u32::try_from(function.frame.preserve.len())
            .map_err(|_| CompileError::CodeSizeOverflow)?;
        let pushed_bytes = pushes
            .checked_mul(8)
            .ok_or(CompileError::CodeSizeOverflow)?;
        let total = pushed_bytes
            .checked_add(function.frame.stack_allocation)
            .ok_or(CompileError::CodeSizeOverflow)?;
        let actual = u8::try_from((8_u32.wrapping_sub(total)) & 0xF)
            .map_err(|_| CompileError::CodeSizeOverflow)?;
        if actual != 0 {
            return Err(CompileError::MisalignedCallFrame { actual });
        }
    }

    let mut labels = BTreeSet::new();
    let mut referenced = BTreeSet::new();
    let mut pending_labels = Vec::new();
    let mut terminated = false;
    let mut saw_executable = false;
    for (operation_index, operation) in function.operations.iter().enumerate() {
        if let Operation::Label(label) = operation {
            if !labels.insert(*label) {
                return Err(CompileError::DuplicateLabel { label: *label });
            }
            pending_labels.push(*label);
            terminated = false;
        } else {
            if terminated {
                return Err(CompileError::OperationAfterTerminator { operation_index });
            }
            pending_labels.clear();
            saw_executable = true;
            match operation {
                Operation::Jump(label) => {
                    referenced.insert(*label);
                    terminated = true;
                }
                Operation::BranchIf { target, .. } => {
                    referenced.insert(*target);
                }
                Operation::Return => terminated = true,
                _ => {}
            }
        }
    }
    if let Some(label) = pending_labels.first() {
        return Err(CompileError::EmptyLabel { label: *label });
    }
    if !saw_executable {
        return Err(CompileError::EmptyFunction);
    }
    if !terminated {
        return Err(CompileError::FallthroughPastEnd);
    }
    if let Some(label) = referenced.difference(&labels).next() {
        return Err(CompileError::UndefinedLabel { label: *label });
    }
    validate_spill_initialization(function)?;
    validate_call_clobbers(function, &registers)?;

    Ok(ValidatedFunction { registers })
}
