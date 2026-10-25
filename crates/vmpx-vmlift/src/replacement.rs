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
    CodeAddress, CompileError, CompiledFunction, Fragment, Function, Operation, PhysicalRegister,
    RegisterAssignment, SpillSlot, VirtualRegister, compile, compose,
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

    let mut fragments: Vec<Fragment> = Vec::new();
    for (index, (register, expression)) in ordered.iter().enumerate() {
        // Pick a scratch result and a distinct temporary that is not the shift
        // count register and not this value's own destination.
        let destination = physical_for(register);
        let mut candidates = SCRATCH
            .into_iter()
            // A scratch may coincide with another value's destination: that
            // value's own fragment has already finished by then, and compose
            // aliases the dead scratch onto the existing assignment.
            .filter(|candidate| *candidate != destination)
            .collect::<Vec<_>>();
        if candidates.len() < 2 {
            return Err(ReplacementError::NoScratch {
                register: **register,
            });
        }
        // Rotate the choice so consecutive values do not all pin one register.
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
        // Each fragment writes its own value; the slot is resolved after
        // composition, where the numbering is known.
        function.operations.push(Operation::SpillStore {
            slot: SpillSlot(0),
            source: result,
        });
        function.operations.push(Operation::Return);
        let widths = function
            .assignments
            .iter()
            .map(|assignment| {
                (
                    assignment.virtual_register.id,
                    assignment.virtual_register.width,
                )
            })
            .collect();
        // Every scratch this value used is dead once it has been spilled. The
        // value's own destination is not written by this fragment, so it is not
        // listed.
        let own_destination = physical_for(register);
        if std::env::var_os("VMPX_REP_DEBUG").is_some() {
            eprintln!(
                "value {index} {register:?} dest={destination:?} result={result_scratch:?} temp={temporary_scratch:?} assigns={:?}",
                function
                    .assignments
                    .iter()
                    .map(|a| a.physical_register)
                    .collect::<Vec<_>>()
            );
        }
        // A fragment only reads inputs, computes through scratch, and stores its
        // result to a slot; nothing it assigns is live when it ends. Every
        // register it names is therefore dead and may share storage with a later
        // fragment.
        // The fragment only spills; the epilogue is the sole writer of
        // architectural state. Nothing this fragment assigns is live when it
        // ends, including a register that happens to be a later destination.
        let _ = own_destination;
        let scratch_dead: BTreeSet<PhysicalRegister> = function
            .assignments
            .iter()
            .map(|assignment| assignment.physical_register)
            .collect();
        fragments.push(Fragment {
            function,
            widths,
            scratch_dead,
        });
    }

    let composition = compose(&fragments)?;
    let composed_stack = composition.function.frame.stack_allocation;
    let mut function = composition.function;

    // Every slot index below `input_slot_count` belongs to the shared input
    // frame. Shift all composed slots above it so the two frames never overlap.
    if input_slot_count != 0 {
        for operation in &mut function.operations {
            match operation {
                Operation::SpillLoad { slot, .. } | Operation::SpillStore { slot, .. } => {
                    slot.0 = slot
                        .0
                        .checked_add(input_slot_count)
                        .ok_or(ReplacementError::TooManyInputs)?;
                }
                _ => {}
            }
        }
    }
    function
        .operations
        .retain(|operation| !matches!(operation, Operation::Return));

    // One epilogue load per register, reading the composed slot the fragment
    // stored into. The epilogue reuses an existing assignment when the physical
    // register already has one, because the backend permits a single virtual
    // register per physical register.
    let mut epilogue: Vec<Operation> = Vec::new();
    let mut assignments = function.assignments.clone();
    let mut emitted = BTreeSet::new();
    for (index, (register, _)) in ordered.iter().enumerate() {
        let physical = physical_for(register);
        let slot = composition
            .slot_maps
            .get(index)
            .and_then(|map| map.get(&0).copied())
            .and_then(|slot| slot.checked_add(input_slot_count))
            .ok_or(ReplacementError::TooManyInputs)?;
        // The epilogue is the only live writer of architectural state. Give it
        // one fresh virtual register per destination and drop every composed
        // assignment for that physical register, since all of them belonged to
        // fragments that have already finished.
        // Reuse the capture register's storage when the input already occupies
        // this architectural register; it has been read into a slot by then, so
        // overwriting it is safe. Otherwise allocate a fresh home.
        let destination = capture_assignments
            .iter()
            .chain(assignments.iter())
            .find(|assignment| assignment.physical_register == physical)
            .map(|assignment| assignment.virtual_register)
            .unwrap_or_else(|| {
                // No earlier assignment owns this register; the composed ones
                // belonged to finished fragments, so drop them and take it over.
                assignments.retain(|assignment| assignment.physical_register != physical);
                let created = VirtualRegister::qword(100_000 + u32::try_from(index).unwrap_or(0));
                assignments.push(RegisterAssignment::new(created, physical));
                created
            });
        epilogue.push(Operation::SpillLoad {
            destination,
            slot: SpillSlot(slot),
        });
        emitted.insert(**register);
    }
    epilogue.push(Operation::Return);
    // The capture block runs first, then every value, then the epilogue.
    let mut operations = capture;
    operations.append(&mut function.operations);
    operations.extend(epilogue);
    function.operations = operations;
    let mut all_assignments = capture_assignments;
    all_assignments.extend(assignments);
    function.assignments = all_assignments;
    // Size the frame from the highest slot any operation actually touches.
    // Compose sizes it from the slots it renumbered, but the epilogue reads
    // those slots through the composed numbering, so measure the emitted body.
    let highest = function
        .operations
        .iter()
        .filter_map(|operation| match operation {
            Operation::SpillLoad { slot, .. } | Operation::SpillStore { slot, .. } => Some(slot.0),
            _ => None,
        })
        .max()
        .unwrap_or(0);
    function.frame.stack_allocation = u32::from(highest.saturating_add(1)).saturating_mul(8);
    let _ = composed_stack;

    let compiled = compile(&function, address)?;
    Ok(Replacement {
        function,
        compiled,
        emitted,
    })
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
