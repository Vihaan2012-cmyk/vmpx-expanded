//! Composition of several lowered functions into one.
//!
//! Each `lower_expression` call allocates virtual register IDs, spill slots, and
//! labels starting from zero, so two lowered bodies cannot be concatenated
//! directly. This module renames those three namespaces into disjoint ranges
//! before splicing, which is what a whole-routine replacement needs.

use std::collections::{BTreeMap, BTreeSet};

use crate::{
    Address, BinarySource, CallTarget, Function, Label, Operation, RegisterAssignment, ShiftCount,
    SpillSlot, StackFrame, VirtualRegister, Width,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Fragment {
    pub function: Function,
    /// Widths of the virtual registers this fragment defines.
    pub widths: BTreeMap<u32, Width>,
    /// Registers whose value is dead when the fragment ends. Two fragments may
    /// name the same physical register here, and compose will alias them so the
    /// backend's one-virtual-register-per-physical-register rule still holds.
    pub scratch_dead: BTreeSet<crate::PhysicalRegister>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Composition {
    pub function: Function,
    /// Index of the first operation contributed by each fragment, in order.
    pub fragment_starts: Vec<usize>,
    /// For each fragment, its original spill slots mapped to composed slots.
    pub slot_maps: Vec<BTreeMap<u16, u16>>,
}

/// Concatenates lowered fragments into a single function with disjoint
/// register, spill-slot, and label namespaces.
///
/// Every fragment keeps its own operations verbatim except for the renaming;
/// no instruction is reordered or dropped. The composed frame reserves the
/// largest stack allocation any fragment needs and preserves the union of
/// their preserved registers.
///
/// # Errors
///
/// Returns an error when a fragment's spill slots exceed the `u16` slot space
/// or when the combined slot count would overflow the frame size.
pub fn compose(fragments: &[Fragment]) -> Result<Composition, ComposeError> {
    let mut operations = Vec::new();
    let mut assignments: Vec<RegisterAssignment> = Vec::new();
    let mut fragment_starts = Vec::with_capacity(fragments.len());
    let mut slot_maps = Vec::with_capacity(fragments.len());
    // Composed virtual-register ids that some fragment declared dead on exit.
    let mut dead_ids: BTreeSet<u32> = BTreeSet::new();
    let mut next_register = 0_u32;
    let mut next_slot = 0_u16;
    let mut next_label = 0_u32;
    let mut preserve: Vec<crate::PhysicalRegister> = Vec::new();
    let mut slot_bytes: u32 = 0;

    for fragment in fragments {
        fragment_starts.push(operations.len());
        let mut renamer = Renamer {
            registers: BTreeMap::new(),
            slots: BTreeMap::new(),
            labels: BTreeMap::new(),
            next_register,
            next_slot,
            next_label,
        };

        // Assignments first so every fragment register has a home.
        //
        // Every fragment register keeps a distinct virtual register. The
        // backend rejects two virtual registers sharing one physical register,
        // so a caller that reuses a physical register across fragments must
        // resolve that conflict before composing; compose does not silently
        // alias two values together.
        for assignment in &fragment.function.assignments {
            let renamed = renamer.register(assignment.virtual_register)?;
            if !assignments
                .iter()
                .any(|existing| existing.virtual_register.id == renamed.id)
            {
                assignments.push(RegisterAssignment::new(
                    renamed,
                    assignment.physical_register,
                ));
            }
        }
        for operation in &fragment.function.operations {
            let renamed = renamer.operation(operation)?;
            operations.push(renamed);
        }
        if std::env::var_os("VMPX_COMPOSE_DEBUG").is_some() {
            let dups: Vec<_> = {
                let mut seen = BTreeMap::new();
                let mut d = Vec::new();
                for a in &assignments {
                    if let Some(prev) = seen.insert(a.physical_register, a.virtual_register.id) {
                        d.push((a.physical_register, prev, a.virtual_register.id));
                    }
                }
                d
            };
            eprintln!("after fragment: dups={dups:?}");
        }
        if std::env::var_os("VMPX_COMPOSE_DEBUG").is_some() {
            eprintln!(
                "frag start={} assigns={} next_reg={}",
                fragment_starts.last().copied().unwrap_or(0),
                fragment.function.assignments.len(),
                renamer.next_register
            );
        }

        for assignment in &fragment.function.assignments {
            if fragment
                .scratch_dead
                .contains(&assignment.physical_register)
                && let Some(composed) = renamer.registers.get(&assignment.virtual_register.id)
            {
                dead_ids.insert(composed.id);
            }
        }
        next_register = renamer.next_register;
        next_slot = renamer.next_slot;
        slot_maps.push(
            renamer
                .slots
                .iter()
                .map(|(original, composed)| (*original, composed.0))
                .collect(),
        );
        next_label = renamer.next_label;

        for register in &fragment.function.frame.preserve {
            if !preserve.contains(register) {
                preserve.push(*register);
            }
        }
        // Spill slots are renumbered into one shared range, so the composed
        // frame must cover the total slot demand rather than the largest single
        // fragment's allocation.
        slot_bytes = slot_bytes.saturating_add(fragment.function.frame.stack_allocation);
    }

    // Slots are numbered from zero, so `next_slot` is exactly the count needed.
    //
    // The backend allows one virtual register per physical register, but compose
    // itself must never merge two values: two fragments that both compute into
    // `RAX` hold different values and need distinct storage. Renaming is
    // therefore unconditional, and aliasing happens only where a fragment
    // explicitly declared a register dead on exit.
    //
    // A dead register may only be aliased onto a *live* home for the same
    // physical register. If no live assignment exists, the dead register keeps
    // its own storage, so the backend still rejects genuinely ambiguous input
    // instead of silently merging values.
    let mut live_home: BTreeMap<crate::PhysicalRegister, VirtualRegister> = BTreeMap::new();
    for assignment in assignments
        .iter()
        .filter(|assignment| !dead_ids.contains(&assignment.virtual_register.id))
    {
        live_home
            .entry(assignment.physical_register)
            .or_insert(assignment.virtual_register);
    }
    let mut aliases: BTreeMap<u32, VirtualRegister> = BTreeMap::new();
    let mut deduped: Vec<RegisterAssignment> = Vec::new();
    for assignment in &assignments {
        let id = assignment.virtual_register.id;
        match live_home.get(&assignment.physical_register) {
            Some(home) if dead_ids.contains(&id) && home.id != id => {
                aliases.insert(id, *home);
            }
            _ => deduped.push(*assignment),
        }
    }
    if !aliases.is_empty() {
        let mut renamer = Renamer {
            registers: aliases,
            slots: BTreeMap::new(),
            labels: BTreeMap::new(),
            next_register,
            next_slot,
            next_label,
        };
        operations = operations
            .iter()
            .map(|operation| renamer.operation(operation))
            .collect::<Result<Vec<_>, _>>()?;
    }
    let assignments = deduped;
    let frame = StackFrame {
        stack_allocation: u32::from(next_slot).saturating_mul(8),
        preserve,
        preinitialized_slots: Vec::new(),
        call_ready: false,
    };
    Ok(Composition {
        function: Function {
            operations,
            assignments,
            frame,
        },
        fragment_starts,
        slot_maps,
    })
}

#[derive(Debug, thiserror::Error, Eq, PartialEq)]
pub enum ComposeError {
    #[error("composed virtual register count exceeds the u32 id space")]
    RegisterOverflow,
    #[error("composed spill slot count exceeds the u16 slot space")]
    SlotOverflow,
    #[error("composed label count exceeds the u32 label space")]
    LabelOverflow,
}

struct Renamer {
    registers: BTreeMap<u32, VirtualRegister>,
    slots: BTreeMap<u16, SpillSlot>,
    labels: BTreeMap<Label, Label>,
    next_register: u32,
    next_slot: u16,
    next_label: u32,
}

impl Renamer {
    fn register(&mut self, value: VirtualRegister) -> Result<VirtualRegister, ComposeError> {
        if let Some(existing) = self.registers.get(&value.id) {
            return Ok(*existing);
        }
        let renamed = VirtualRegister::new(self.next_register, value.width);
        self.next_register = self
            .next_register
            .checked_add(1)
            .ok_or(ComposeError::RegisterOverflow)?;
        self.registers.insert(value.id, renamed);
        Ok(renamed)
    }

    fn address(&mut self, value: Address) -> Result<Address, ComposeError> {
        Ok(Address::new(self.register(value.base)?, value.displacement))
    }

    fn source(&mut self, value: BinarySource) -> Result<BinarySource, ComposeError> {
        Ok(match value {
            BinarySource::Register(v) => BinarySource::Register(self.register(v)?),
            BinarySource::Immediate(v) => BinarySource::Immediate(v),
        })
    }

    fn slot(&mut self, value: SpillSlot) -> Result<SpillSlot, ComposeError> {
        if let Some(existing) = self.slots.get(&value.0) {
            return Ok(*existing);
        }
        let renamed = SpillSlot(self.next_slot);
        self.next_slot = self
            .next_slot
            .checked_add(1)
            .ok_or(ComposeError::SlotOverflow)?;
        self.slots.insert(value.0, renamed);
        Ok(renamed)
    }

    fn label(&mut self, value: Label) -> Result<Label, ComposeError> {
        if let Some(existing) = self.labels.get(&value) {
            return Ok(*existing);
        }
        let renamed = Label(self.next_label);
        self.next_label = self
            .next_label
            .checked_add(1)
            .ok_or(ComposeError::LabelOverflow)?;
        self.labels.insert(value, renamed);
        Ok(renamed)
    }

    fn operation(&mut self, operation: &Operation) -> Result<Operation, ComposeError> {
        Ok(match *operation {
            Operation::Label(v) => Operation::Label(self.label(v)?),
            Operation::MoveImmediate { destination, value } => Operation::MoveImmediate {
                destination: self.register(destination)?,
                value,
            },
            Operation::Move {
                destination,
                source: from,
            } => Operation::Move {
                destination: self.register(destination)?,
                source: self.register(from)?,
            },
            Operation::Binary {
                operation,
                destination,
                source: from,
            } => Operation::Binary {
                operation,
                destination: self.register(destination)?,
                source: self.source(from)?,
            },
            Operation::Unary {
                operation,
                destination,
            } => Operation::Unary {
                operation,
                destination: self.register(destination)?,
            },
            Operation::Shift {
                operation,
                destination,
                count,
            } => Operation::Shift {
                operation,
                destination: self.register(destination)?,
                count: match count {
                    ShiftCount::Immediate(v) => ShiftCount::Immediate(v),
                    ShiftCount::Register(v) => ShiftCount::Register(self.register(v)?),
                },
            },
            Operation::Resize {
                operation,
                destination,
                source: from,
            } => Operation::Resize {
                operation,
                destination: self.register(destination)?,
                source: self.register(from)?,
            },
            Operation::Load {
                destination,
                address: at,
            } => Operation::Load {
                destination: self.register(destination)?,
                address: self.address(at)?,
            },
            Operation::Store {
                address: at,
                source: from,
            } => Operation::Store {
                address: self.address(at)?,
                source: self.register(from)?,
            },
            Operation::SpillLoad {
                destination,
                slot: from,
            } => Operation::SpillLoad {
                destination: self.register(destination)?,
                slot: self.slot(from)?,
            },
            Operation::SpillStore {
                slot: to,
                source: from,
            } => Operation::SpillStore {
                slot: self.slot(to)?,
                source: self.register(from)?,
            },
            Operation::Call {
                target,
                ref arguments,
                result,
            } => Operation::Call {
                target: match target {
                    CallTarget::Direct(v) => CallTarget::Direct(v),
                    CallTarget::ImportAddressTable(v) => CallTarget::ImportAddressTable(v),
                    CallTarget::Register(v) => CallTarget::Register(self.register(v)?),
                },
                arguments: arguments
                    .iter()
                    .map(|argument| self.register(*argument))
                    .collect::<Result<Vec<_>, _>>()?,
                result: result.map(|v| self.register(v)).transpose()?,
            },
            Operation::Compare {
                left,
                right: against,
            } => Operation::Compare {
                left: self.register(left)?,
                right: self.source(against)?,
            },
            Operation::Jump(v) => Operation::Jump(self.label(v)?),
            Operation::BranchIf { condition, target } => Operation::BranchIf {
                condition,
                target: self.label(target)?,
            },
            Operation::Return => Operation::Return,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PhysicalRegister, UnaryOp};

    fn fragment(value: u64) -> Fragment {
        let register = VirtualRegister::qword(0);
        Fragment {
            function: Function {
                operations: vec![
                    Operation::MoveImmediate {
                        destination: register,
                        value,
                    },
                    Operation::Unary {
                        operation: UnaryOp::Not,
                        destination: register,
                    },
                    Operation::SpillStore {
                        slot: SpillSlot(0),
                        source: register,
                    },
                    Operation::SpillLoad {
                        destination: register,
                        slot: SpillSlot(0),
                    },
                    Operation::Return,
                ],
                assignments: vec![RegisterAssignment::new(register, PhysicalRegister::Rax)],
                frame: StackFrame {
                    stack_allocation: 8,
                    preserve: Vec::new(),
                    preinitialized_slots: Vec::new(),
                    call_ready: false,
                },
            },
            widths: BTreeMap::from([(0, Width::W64)]),
            scratch_dead: BTreeSet::new(),
        }
    }

    #[test]
    fn compose_gives_each_fragment_disjoint_namespaces() {
        let composition = compose(&[fragment(1), fragment(2)]).expect("compose");
        assert_eq!(composition.fragment_starts, vec![0, 5]);
        assert_eq!(composition.function.operations.len(), 10);
        // Two fragments each used register id 0 and slot 0; both must be renamed.
        let registers: std::collections::BTreeSet<u32> = composition
            .function
            .assignments
            .iter()
            .map(|assignment| assignment.virtual_register.id)
            .collect();
        assert_eq!(registers.len(), 2);
        let slots: std::collections::BTreeSet<u16> = composition
            .function
            .operations
            .iter()
            .filter_map(|operation| match operation {
                Operation::SpillStore { slot, .. } | Operation::SpillLoad { slot, .. } => {
                    Some(slot.0)
                }
                _ => None,
            })
            .collect();
        assert_eq!(slots.len(), 2);
        // Two fragments each use one slot; both slots are live in the composed
        // frame, so the allocation covers both.
        assert_eq!(composition.function.frame.stack_allocation, 16);
    }
}
