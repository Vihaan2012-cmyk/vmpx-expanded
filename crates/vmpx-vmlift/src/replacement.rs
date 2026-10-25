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
    /// Symbolic input name to physical register, for every value the exit state
    /// can reference.
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

    let inputs = config
        .inputs
        .iter()
        .map(|(name, register)| ExpressionInput {
            name: name.clone(),
            width: vmpx_vm::BitWidth::W64,
            physical_register: *register,
        })
        .collect::<Vec<_>>();

    // A scratch register must not hold a live symbolic input: the backend would
    // then have two virtual registers on one physical register, one of which is
    // read at function entry.
    let input_registers: BTreeSet<PhysicalRegister> = config
        .inputs
        .iter()
        .map(|(_, register)| *register)
        .collect();

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
            .filter(|candidate| *candidate != destination && !input_registers.contains(candidate))
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
        let scratch_dead: BTreeSet<PhysicalRegister> = function
            .assignments
            .iter()
            .map(|assignment| assignment.physical_register)
            .filter(|register| *register != own_destination)
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
            .unwrap_or(u16::MAX);
        // The input binding for this physical register is only read at entry and
        // is dead once the epilogue writes it, so the epilogue reuses it.
        let destination = assignments
            .iter()
            .find(|assignment| assignment.physical_register == physical)
            .map(|assignment| assignment.virtual_register)
            .unwrap_or_else(|| {
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
    function.operations.extend(epilogue);
    function.assignments = assignments;
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
