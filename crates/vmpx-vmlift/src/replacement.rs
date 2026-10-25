//! Whole-routine native replacement for a recovered VM exit state.
//!
//! A recovered entry path ends with an exact symbolic value for every native
//! register. This module lowers each of those values into the project-owned x64
//! IR and joins them into one function that leaves exactly that register state
//! behind, so a VM entry can be redirected to it.
//!
//! The lowering deliberately does not compute values in place. The expression
//! backend needs a result register, a temporary, and `RCX` as the variable-shift
//! count, and those overlap with the architectural destinations. Instead every
//! value is computed through scratch registers and spilled to its own slot; a
//! single epilogue then loads each slot into the register that value belongs in.
//! Loading in one pass is safe because every load reads a slot and writes a
//! register, and no slot address depends on a register.

use std::collections::{BTreeMap, BTreeSet};

use thiserror::Error;
use vmpx_vm::{Expr, X64NativeRegister};

use vmpx_recompile::{
    Address, BinarySource, CallTarget, CodeAddress, CompileError, CompiledFunction, Function,
    Operation, PhysicalRegister, RegisterAssignment, ShiftCount, SpillSlot, StackFrame,
    VirtualRegister, compile,
};

use crate::{ExpressionCodegenConfig, ExpressionInput, lower_expression};

/// Registers the expression backend may use as scratch. `RCX` is excluded
/// because it is the mandatory variable-shift count register.
const SCRATCH: [PhysicalRegister; 6] = [
    PhysicalRegister::R10,
    PhysicalRegister::R11,
    PhysicalRegister::R8,
    PhysicalRegister::R9,
    PhysicalRegister::Rdx,
    PhysicalRegister::Rax,
];

#[derive(Clone, Debug)]
pub struct ReplacementConfig {
    /// Symbolic input name to the physical register that carries it on entry.
    /// These are captured once into the shared input frame, so no per-value
    /// fragment needs an input register of its own.
    pub inputs: Vec<(String, PhysicalRegister)>,
    pub max_expression_nodes: usize,
    pub max_spill_slots: usize,
}

#[derive(Debug, Error)]
pub enum ReplacementError {
    #[error("register {register:?} could not be lowered: {message}")]
    Lowering {
        register: X64NativeRegister,
        message: String,
    },

    #[error("composition failed: {0}")]
    Compose(#[from] vmpx_recompile::ComposeError),

    #[error("native encoding failed: {0}")]
    Compile(#[from] CompileError),

    #[error("no scratch register is available for {register:?}")]
    NoScratch { register: X64NativeRegister },

    #[error("more than 65535 symbolic inputs were supplied")]
    TooManyInputs,
}

#[derive(Debug)]
pub struct Replacement {
    pub function: Function,
    pub compiled: CompiledFunction,
    /// Registers whose value is emitted by the epilogue.
    pub emitted: BTreeSet<X64NativeRegister>,
}

/// Lowers every register of one native exit state and compiles the result.
///
/// # Errors
///
/// Returns the first register whose expression cannot be lowered, or any
/// composition or encoding failure. A partial routine is never returned as a
/// compiled function.
pub fn build_replacement(
    registers: &BTreeMap<X64NativeRegister, Expr>,
    config: &ReplacementConfig,
    address: CodeAddress,
) -> Result<Replacement, ReplacementError> {
    let mut ordered: Vec<(&X64NativeRegister, &Expr)> = registers.iter().collect();
    ordered.sort_by_key(|(register, _)| format!("{register:?}"));

    // Capture every symbolic input once into the shared input frame. Each value
    // then reads its inputs from those slots, so the per-value fragments consume
    // no input registers and cannot collide with an architectural destination.
    let mut capture: Vec<Operation> = Vec::new();
    let mut capture_assignments: Vec<RegisterAssignment> = Vec::new();
    let mut capture_ids: BTreeMap<PhysicalRegister, VirtualRegister> = BTreeMap::new();
    let mut inputs: Vec<ExpressionInput> = Vec::new();
    for (index, (name, register)) in config.inputs.iter().enumerate() {
        let slot = SpillSlot(u16::try_from(index).map_err(|_| ReplacementError::TooManyInputs)?);
        let id = *capture_ids
            .entry(*register)
            .or_insert_with(|| VirtualRegister::qword(40_000 + u32::try_from(index).unwrap_or(0)));
        if !capture_assignments
            .iter()
            .any(|assignment| assignment.virtual_register.id == id.id)
        {
            capture_assignments.push(RegisterAssignment::new(id, *register));
            capture.push(Operation::SpillStore { slot, source: id });
        }
        inputs.push(ExpressionInput::slot(
            name.clone(),
            vmpx_vm::BitWidth::W64,
            slot,
        ));
    }
    let input_slot_count =
        u16::try_from(config.inputs.len()).map_err(|_| ReplacementError::TooManyInputs)?;

    // Assemble the routine directly rather than through `compose`. Compose
    // renumbers spill slots per fragment into one shared range, which would give
    // each fragment a different index for the same shared input slot. Here the
    // slot layout is fixed up front:
    //
    //   [0, input_slot_count)              shared input frame
    //   [result_base + i)                  result slot for value i
    //
    // Registers are canonicalised to one virtual register per physical register,
    // which is what the backend requires.
    let result_base = input_slot_count;
    let mut operations: Vec<Operation> = Vec::new();
    let mut assignments: Vec<RegisterAssignment> = Vec::new();
    let mut canonical: BTreeMap<PhysicalRegister, VirtualRegister> = BTreeMap::new();
    let mut next_register = 0_u32;
    let mut preserve: Vec<PhysicalRegister> = Vec::new();

    // 1. Capture block: fill the shared input frame from the entry registers.
    let mut capture_ops: Vec<Operation> = Vec::new();
    for (index, (_, register)) in config.inputs.iter().enumerate() {
        let slot = SpillSlot(u16::try_from(index).map_err(|_| ReplacementError::TooManyInputs)?);
        let source = canonical_register(
            *register,
            &mut canonical,
            &mut assignments,
            &mut next_register,
        );
        capture_ops.push(Operation::SpillStore { slot, source });
    }
    operations.extend(capture_ops);

    // 2. One body per exit register. Each body reads its inputs from the shared
    // frame and stores its result into its own slot.
    for (index, (register, expression)) in ordered.iter().enumerate() {
        let destination = physical_for(register);
        let mut candidates = SCRATCH
            .into_iter()
            .filter(|candidate| *candidate != destination)
            .collect::<Vec<_>>();
        if candidates.len() < 2 {
            return Err(ReplacementError::NoScratch {
                register: **register,
            });
        }
        let rotation = index % candidates.len();
        candidates.rotate_left(rotation);
        let result_scratch = candidates[0];
        let temporary_scratch = candidates
            .iter()
            .copied()
            .find(|candidate| *candidate != result_scratch)
            .ok_or(ReplacementError::NoScratch {
                register: **register,
            })?;

        let per_value = ExpressionCodegenConfig {
            inputs: inputs.clone(),
            result_register: result_scratch,
            temporary_register: temporary_scratch,
            shift_count_register: PhysicalRegister::Rcx,
            max_expression_nodes: config.max_expression_nodes,
            max_spill_slots: config.max_spill_slots,
        };
        let lowered = lower_expression(expression, &per_value).map_err(|error| {
            ReplacementError::Lowering {
                register: **register,
                message: error.to_string(),
            }
        })?;
        let mut function = lowered.function;
        let result = function
            .assignments
            .iter()
            .find(|assignment| assignment.physical_register == result_scratch)
            .map(|assignment| assignment.virtual_register)
            .ok_or(ReplacementError::NoScratch {
                register: **register,
            })?;
        function
            .operations
            .retain(|operation| !matches!(operation, Operation::Return));
        // Every slot the lowering allocated for its own temporaries sits at or
        // above the input frame; move those above the result slots so no value's
        // temporaries collide with another's results.
        let local_base = result_base
            .checked_add(u16::try_from(ordered.len()).map_err(|_| ReplacementError::TooManyInputs)?)
            .ok_or(ReplacementError::TooManyInputs)?;
        for operation in &mut function.operations {
            match operation {
                Operation::SpillLoad { slot, .. } | Operation::SpillStore { slot, .. } => {
                    if slot.0 >= result_base {
                        slot.0 = slot
                            .0
                            .checked_sub(result_base)
                            .and_then(|offset| local_base.checked_add(offset))
                            .ok_or(ReplacementError::TooManyInputs)?;
                    }
                }
                _ => {}
            }
        }

        // Rewrite this body's registers onto the canonical per-register ids.
        let mut body_ops: Vec<Operation> = Vec::new();
        let mut body_map: BTreeMap<u32, VirtualRegister> = BTreeMap::new();
        for assignment in &function.assignments {
            let canonical_id = canonical_register(
                assignment.physical_register,
                &mut canonical,
                &mut assignments,
                &mut next_register,
            );
            body_map.insert(assignment.virtual_register.id, canonical_id);
        }
        for operation in &function.operations {
            body_ops.push(rename_registers(operation, &body_map));
        }
        operations.extend(body_ops);

        let result_slot = result_base
            .checked_add(u16::try_from(index).map_err(|_| ReplacementError::TooManyInputs)?)
            .ok_or(ReplacementError::TooManyInputs)?;
        let canonical_result = body_map[&result.id];
        operations.push(Operation::SpillStore {
            slot: SpillSlot(result_slot),
            source: canonical_result,
        });

        for register in &function.frame.preserve {
            if !preserve.contains(register) {
                preserve.push(*register);
            }
        }
    }

    // 3. Epilogue: load each result slot into its architectural register.
    for (index, (register, _)) in ordered.iter().enumerate() {
        let physical = physical_for(register);
        let destination = canonical_register(
            physical,
            &mut canonical,
            &mut assignments,
            &mut next_register,
        );
        let result_slot = result_base
            .checked_add(u16::try_from(index).map_err(|_| ReplacementError::TooManyInputs)?)
            .ok_or(ReplacementError::TooManyInputs)?;
        operations.push(Operation::SpillLoad {
            destination,
            slot: SpillSlot(result_slot),
        });
    }
    operations.push(Operation::Return);

    // Every nonvolatile register the body writes must be preserved.
    for assignment in &assignments {
        let register = assignment.physical_register;
        if register.is_win64_nonvolatile() && !preserve.contains(&register) {
            preserve.push(register);
        }
    }

    let highest = operations
        .iter()
        .filter_map(|operation| match operation {
            Operation::SpillLoad { slot, .. } | Operation::SpillStore { slot, .. } => Some(slot.0),
            _ => None,
        })
        .max()
        .unwrap_or(0);
    let function = Function {
        operations,
        assignments,
        frame: StackFrame {
            stack_allocation: u32::from(highest.saturating_add(1)).saturating_mul(8),
            preserve,
            call_ready: false,
            preinitialized_slots: (0..input_slot_count).map(SpillSlot).collect(),
        },
    };

    let compiled = compile(&function, address)?;
    Ok(Replacement {
        function,
        compiled,
        emitted: ordered.iter().map(|(register, _)| **register).collect(),
    })
}

/// Returns the canonical virtual register for a physical register, allocating
/// one on first use. The backend allows exactly one virtual register per
/// physical register, so every fragment naming the same register shares it.
fn canonical_register(
    physical: PhysicalRegister,
    canonical: &mut BTreeMap<PhysicalRegister, VirtualRegister>,
    assignments: &mut Vec<RegisterAssignment>,
    next_register: &mut u32,
) -> VirtualRegister {
    if let Some(existing) = canonical.get(&physical) {
        return *existing;
    }
    let created = VirtualRegister::qword(*next_register);
    *next_register = next_register.saturating_add(1);
    canonical.insert(physical, created);
    assignments.push(RegisterAssignment::new(created, physical));
    created
}

/// Rewrites the virtual registers of one operation through `map`.
fn rename_registers(operation: &Operation, map: &BTreeMap<u32, VirtualRegister>) -> Operation {
    let register = |value: VirtualRegister| -> VirtualRegister {
        map.get(&value.id).copied().unwrap_or(value)
    };
    match *operation {
        Operation::Label(label) => Operation::Label(label),
        Operation::MoveImmediate { destination, value } => Operation::MoveImmediate {
            destination: register(destination),
            value,
        },
        Operation::Move {
            destination,
            source,
        } => Operation::Move {
            destination: register(destination),
            source: register(source),
        },
        Operation::Binary {
            operation,
            destination,
            source,
        } => Operation::Binary {
            operation,
            destination: register(destination),
            source: match source {
                BinarySource::Register(value) => BinarySource::Register(register(value)),
                BinarySource::Immediate(value) => BinarySource::Immediate(value),
            },
        },
        Operation::Unary {
            operation,
            destination,
        } => Operation::Unary {
            operation,
            destination: register(destination),
        },
        Operation::Shift {
            operation,
            destination,
            count,
        } => Operation::Shift {
            operation,
            destination: register(destination),
            count: match count {
                ShiftCount::Register(value) => ShiftCount::Register(register(value)),
                ShiftCount::Immediate(value) => ShiftCount::Immediate(value),
            },
        },
        Operation::Resize {
            operation,
            destination,
            source,
        } => Operation::Resize {
            operation,
            destination: register(destination),
            source: register(source),
        },
        Operation::Load {
            destination,
            address,
        } => Operation::Load {
            destination: register(destination),
            address: Address::new(register(address.base), address.displacement),
        },
        Operation::Store { address, source } => Operation::Store {
            address: Address::new(register(address.base), address.displacement),
            source: register(source),
        },
        Operation::SpillLoad { destination, slot } => Operation::SpillLoad {
            destination: register(destination),
            slot,
        },
        Operation::SpillStore { slot, source } => Operation::SpillStore {
            slot,
            source: register(source),
        },
        Operation::Call {
            target,
            ref arguments,
            result,
        } => Operation::Call {
            target: match target {
                CallTarget::Register(value) => CallTarget::Register(register(value)),
                other => other,
            },
            arguments: arguments.iter().map(|value| register(*value)).collect(),
            result: result.map(register),
        },
        Operation::Compare { left, right } => Operation::Compare {
            left: register(left),
            right: match right {
                BinarySource::Register(value) => BinarySource::Register(register(value)),
                BinarySource::Immediate(value) => BinarySource::Immediate(value),
            },
        },
        Operation::Jump(label) => Operation::Jump(label),
        Operation::BranchIf { condition, target } => Operation::BranchIf { condition, target },
        Operation::Return => Operation::Return,
    }
}

fn physical_for(register: &X64NativeRegister) -> PhysicalRegister {
    match register {
        X64NativeRegister::Rax => PhysicalRegister::Rax,
        X64NativeRegister::Rbx => PhysicalRegister::Rbx,
        X64NativeRegister::Rcx => PhysicalRegister::Rcx,
        X64NativeRegister::Rdx => PhysicalRegister::Rdx,
        X64NativeRegister::Rbp => PhysicalRegister::Rbp,
        X64NativeRegister::Rsi => PhysicalRegister::Rsi,
        X64NativeRegister::Rdi => PhysicalRegister::Rdi,
        X64NativeRegister::R8 => PhysicalRegister::R8,
        X64NativeRegister::R9 => PhysicalRegister::R9,
        X64NativeRegister::R10 => PhysicalRegister::R10,
        X64NativeRegister::R11 => PhysicalRegister::R11,
        X64NativeRegister::R12 => PhysicalRegister::R12,
        X64NativeRegister::R13 => PhysicalRegister::R13,
        X64NativeRegister::R14 => PhysicalRegister::R14,
        X64NativeRegister::R15 => PhysicalRegister::R15,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vmpx_vm::BitWidth;

    fn inputs() -> Vec<(String, PhysicalRegister)> {
        vec![
            ("native_rax_initial".to_owned(), PhysicalRegister::Rbx),
            ("native_rcx_initial".to_owned(), PhysicalRegister::Rsi),
            ("vsp_initial".to_owned(), PhysicalRegister::Rdi),
        ]
    }

    #[test]
    fn builds_a_complete_function_for_a_constant_exit_state() {
        let registers = BTreeMap::from([
            (X64NativeRegister::Rax, Expr::constant(BitWidth::W64, 7)),
            (X64NativeRegister::Rbx, Expr::constant(BitWidth::W64, 9)),
        ]);
        let config = ReplacementConfig {
            inputs: inputs(),
            max_expression_nodes: 10_000,
            max_spill_slots: 32,
        };
        let replacement = build_replacement(&registers, &config, CodeAddress(0x1_4000_1000))
            .expect("replacement compiles");
        assert_eq!(replacement.emitted.len(), 2);
        assert!(replacement.compiled.bytes.len() > 16);
        // The emitted body must end in RET.
        assert_eq!(replacement.compiled.bytes.last().copied(), Some(0xC3));
    }

    #[test]
    fn preserves_every_nonvolatile_register_it_writes() {
        let registers = BTreeMap::from([(
            X64NativeRegister::R12,
            Expr::constant(BitWidth::W64, 0x1234),
        )]);
        let config = ReplacementConfig {
            inputs: inputs(),
            max_expression_nodes: 10_000,
            max_spill_slots: 32,
        };
        let replacement = build_replacement(&registers, &config, CodeAddress(0x1_4000_1000))
            .expect("replacement compiles");
        assert!(
            replacement
                .function
                .frame
                .preserve
                .contains(&PhysicalRegister::R12)
        );
        // The prologue must push R12 (41 54) before writing it.
        assert!(
            replacement
                .compiled
                .bytes
                .windows(2)
                .any(|w| w == [0x41, 0x54])
        );
    }

    #[test]
    fn refuses_an_expression_it_cannot_lower() {
        let registers = BTreeMap::from([(
            X64NativeRegister::Rax,
            Expr::intrinsic(BitWidth::W64, "fresh_allocation_return_epoch_0", vec![]),
        )]);
        let config = ReplacementConfig {
            inputs: inputs(),
            max_expression_nodes: 10_000,
            max_spill_slots: 32,
        };
        let error = build_replacement(&registers, &config, CodeAddress(0x1_4000_1000))
            .expect_err("intrinsic must not be emitted");
        assert!(matches!(error, ReplacementError::Lowering { .. }));
    }
}
