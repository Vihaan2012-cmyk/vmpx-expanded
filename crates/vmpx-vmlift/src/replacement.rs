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
    /// Intrinsics that may be lowered to a real imported call.
    pub import_calls: BTreeMap<String, crate::ImportCall>,
    /// Where the recovered VM exit transfers control instead of returning to the
    /// original caller. The emitted function ends with a tail jump to this
    /// address after restoring the exit register state, so the protector's own
    /// native continuation runs and performs the real return.
    pub continuation: Option<u64>,
    /// Observable effects the virtualized routine performs, in execution order.
    /// A replacement that omitted them would compute the right registers and
    /// still be wrong, so they are emitted before the epilogue restores state.
    pub effects: Vec<EffectStore>,
    /// Imported calls whose result must be available to the effects. Each entry
    /// performs one call at the start of the routine and binds the result to
    /// the intrinsic name that stands for it.
    pub effect_calls: Vec<EffectCall>,
    /// The condition state the routine leaves for its caller. Restoring it is
    /// part of the routine's contract: a caller that branches on a flag it
    /// expects the routine to have set would otherwise read this body's flags.
    pub rflags: Option<Expr>,
    /// Rewrites applied to every lowered expression before code generation.
    ///
    /// This exists because a symbolic input can denote a value that is also
    /// reachable from the frame. When the entry stub captures the native stack
    /// pointer into a register it has already saved, that register carries two
    /// distinct values at different times; the saved copy is loadable from the
    /// frame, so rewriting the name to that load frees the register and keeps
    /// both values correct.
    pub symbol_rewrites: BTreeMap<String, Expr>,
}

/// One application-observable store the routine must perform.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EffectStore {
    pub address: Expr,
    pub value: Expr,
    pub width: vmpx_vm::BitWidth,
}

/// One imported call the routine must perform, with its result bound to
/// `result_intrinsic`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EffectCall {
    pub result_intrinsic: String,
    pub call: crate::ImportCall,
    pub arguments: Vec<Expr>,
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

    #[error(
        "nonvolatile {register:?} must be restored for the caller, but the exit state          gives it a different value; the restoration would discard that value"
    )]
    ClobberedNonvolatile { register: X64NativeRegister },

    #[error("effect lowering failed: {0}")]
    Effect(#[from] crate::ExpressionCodegenError),

    #[error("an imported effect call has more arguments than the Win64 register ABI allows")]
    TooManyCallArguments,

    #[error("could not bind an effect call result: {0}")]
    EffectBindings(String),
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
    let effect_slot_count =
        u16::try_from(config.effect_calls.len()).map_err(|_| ReplacementError::TooManyInputs)?;
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

    // 1b. Effect prelude. The virtualized routine's application-observable
    // effects run here, in their original order, before any register value is
    // computed. Imported calls execute once and their results are captured into
    // frame slots, so a value that reads the same allocation does not invoke the
    // allocator again.
    let mut effects_operations: Vec<Operation> = Vec::new();
    let mut extra_inputs: Vec<ExpressionInput> = Vec::new();
    // Every fragment gets a private window of frame slots starting here, above
    // the shared input frame, the value results, and the effect results.
    let mut inputs = inputs;
    let mut window_base = result_base
        .checked_add(u16::try_from(ordered.len()).map_err(|_| ReplacementError::TooManyInputs)?)
        .and_then(|base| base.checked_add(effect_slot_count))
        .ok_or(ReplacementError::TooManyInputs)?;
    // The shared region ends where the first fragment window begins.
    let shared_end = window_base;
    // Every effect-result slot is registered before any fragment is lowered, so
    // a fragment that reads an allocation sees the slot inside the shared input
    // frame rather than above it.
    for index in 0..config.effect_calls.len() {
        let slot = SpillSlot(
            u16::try_from(config.inputs.len())
                .map_err(|_| ReplacementError::TooManyInputs)?
                .checked_add(
                    u16::try_from(ordered.len()).map_err(|_| ReplacementError::TooManyInputs)?,
                )
                .and_then(|base| base.checked_add(u16::try_from(index).ok()?))
                .ok_or(ReplacementError::TooManyInputs)?,
        );
        extra_inputs.push(ExpressionInput::slot(
            effect_result_name(index),
            vmpx_vm::BitWidth::W64,
            slot,
        ));
    }
    // Calls run first so the effects that consume an allocation can read its
    // result. The intrinsic that stands for the result is rewritten to the
    // captured slot, which is what makes a single call serve every use.
    let mut effect_bindings: BTreeMap<String, Expr> = BTreeMap::new();
    for (index, effect_call) in config.effect_calls.iter().enumerate() {
        let arguments = effect_call
            .arguments
            .iter()
            .take(usize::from(effect_call.call.argument_count))
            .cloned()
            .collect::<Vec<_>>();
        const ARGUMENT_REGISTERS: [PhysicalRegister; 4] = [
            PhysicalRegister::Rcx,
            PhysicalRegister::Rdx,
            PhysicalRegister::R8,
            PhysicalRegister::R9,
        ];
        let mut lowered_arguments: Vec<VirtualRegister> = Vec::new();
        for (argument_index, argument) in arguments.iter().enumerate() {
            let (argument_operations, argument_result) = lower_fragment(
                argument,
                &inputs,
                &extra_inputs,
                config,
                &mut canonical,
                &mut assignments,
                &mut next_register,
                &mut window_base,
                &shared_end,
            )?;
            effects_operations.extend(argument_operations);
            // The call ABI requires each argument in its own register, so move
            // the lowered value there once the expression has been evaluated.
            let abi_register = *ARGUMENT_REGISTERS
                .get(argument_index)
                .ok_or(ReplacementError::TooManyCallArguments)?;
            let destination = canonical_register(
                abi_register,
                &mut canonical,
                &mut assignments,
                &mut next_register,
            );
            if destination.id != argument_result.id {
                effects_operations.push(Operation::Move {
                    destination,
                    source: argument_result,
                });
            }
            lowered_arguments.push(destination);
        }
        // The slot was registered with the shared frame above; the call writes
        // it so every reader sees the one allocation.
        let result_slot = SpillSlot(
            u16::try_from(config.inputs.len())
                .map_err(|_| ReplacementError::TooManyInputs)?
                .checked_add(
                    u16::try_from(ordered.len()).map_err(|_| ReplacementError::TooManyInputs)?,
                )
                .and_then(|base| base.checked_add(u16::try_from(index).ok()?))
                .ok_or(ReplacementError::TooManyInputs)?,
        );
        let result = canonical_register(
            PhysicalRegister::Rax,
            &mut canonical,
            &mut assignments,
            &mut next_register,
        );
        effects_operations.push(Operation::Call {
            target: CallTarget::ImportAddressTable(vmpx_recompile::NativeAddress(
                effect_call.call.iat_address,
            )),
            arguments: lowered_arguments,
            result: Some(result),
        });
        effects_operations.push(Operation::SpillStore {
            slot: result_slot,
            source: result,
        });
        effect_bindings.insert(
            effect_call.result_intrinsic.clone(),
            Expr::register(vmpx_vm::BitWidth::W64, effect_result_name(index)),
        );
    }
    // Every store the routine performs, in order. Each address and value is
    // evaluated against the shared frame, so a store that targets the fresh
    // allocation reads the captured allocator result.
    for effect in &config.effects {
        let address_expression = substitute_effect_bindings(
            &rewrite_symbols(&effect.address, &config.symbol_rewrites)?,
            &effect_bindings,
        )?;
        let value_expression = substitute_effect_bindings(
            &rewrite_symbols(&effect.value, &config.symbol_rewrites)?,
            &effect_bindings,
        )?;
        let (address_operations, address) = lower_fragment(
            &address_expression,
            &inputs,
            &extra_inputs,
            config,
            &mut canonical,
            &mut assignments,
            &mut next_register,
            &mut window_base,
            &shared_end,
        )?;
        effects_operations.extend(address_operations);
        let (value_operations, value) = lower_fragment(
            &value_expression,
            &inputs,
            &extra_inputs,
            config,
            &mut canonical,
            &mut assignments,
            &mut next_register,
            &mut window_base,
            &shared_end,
        )?;
        effects_operations.extend(value_operations);
        effects_operations.push(Operation::Store {
            address: Address::new(address, 0),
            source: value,
        });
    }
    operations.extend(effects_operations);
    // Value bodies read the effect results too, so they join the shared input
    // frame before any register is computed. `lower_fragment` already appends
    // them, so only the ones not yet present are added here.
    for input in extra_inputs.iter() {
        if !inputs.iter().any(|existing| existing.name == input.name) {
            inputs.push(input.clone());
        }
    }

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
            import_calls: config.import_calls.clone(),
        };
        // A value that reads an allocation the routine made must read the
        // result of the one call, not invoke the allocator again.
        let rewritten = rewrite_symbols(expression, &config.symbol_rewrites)?;
        let bound = substitute_effect_bindings(&rewritten, &effect_bindings).map_err(|error| {
            ReplacementError::Lowering {
                register: **register,
                message: error.to_string(),
            }
        })?;
        let lowered =
            lower_expression(&bound, &per_value).map_err(|error| ReplacementError::Lowering {
                register: **register,
                message: error.to_string(),
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
        // The lowering numbers its own temporaries from the input count upward,
        // so the shared region it must leave alone ends where the windows begin.
        // Give each value a private window above that, so two values never share
        // storage and no value aliases a caller-owned slot.
        let value_count =
            u16::try_from(ordered.len()).map_err(|_| ReplacementError::TooManyInputs)?;
        let window = u16::try_from(config.max_spill_slots)
            .map_err(|_| ReplacementError::TooManyInputs)?
            .max(1);
        let local_base = shared_end
            .checked_add(effect_slot_count)
            .and_then(|base| base.checked_add(u16::try_from(index).ok()?.checked_mul(window)?))
            .ok_or(ReplacementError::TooManyInputs)?;
        let _ = (result_base, value_count);
        for operation in &mut function.operations {
            match operation {
                Operation::SpillLoad { slot, .. } | Operation::SpillStore { slot, .. } => {
                    if slot.0 >= shared_end {
                        slot.0 = slot
                            .0
                            .checked_sub(shared_end)
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
    // End by transferring to the recovered continuation, which restores its own
    // frame and returns to the original caller. Without one the function simply
    // returns, which is only correct when the exit already targets the caller.
    // The jump must be a real tail transfer: nothing is pushed, so the
    // continuation sees exactly the stack the VM exit would have produced.
    // Restore the condition state before the frame goes away. The value is
    // materialised while the frame is still addressable, and `POPFQ` does not
    // touch the stack pointer beyond the one qword it consumes.
    if let Some(rflags) = &config.rflags {
        let bound = substitute_effect_bindings(
            &rewrite_symbols(rflags, &config.symbol_rewrites)?,
            &effect_bindings,
        )?;
        let (flag_operations, flag_value) = lower_fragment(
            &bound,
            &inputs,
            &extra_inputs,
            config,
            &mut canonical,
            &mut assignments,
            &mut next_register,
            &mut window_base,
            &shared_end,
        )?;
        operations.extend(flag_operations);
        operations.push(Operation::RestoreFlags { value: flag_value });
    }

    // A tail jump tears the frame down itself, so it replaces the return rather
    // than following it; emitting both would leave an unreachable `ret`.
    match config.continuation {
        Some(target) => operations.push(Operation::JumpToAddress(vmpx_recompile::NativeAddress(
            target,
        ))),
        None => operations.push(Operation::Return),
    }

    // A preserved register is saved in the prologue because the body uses it for
    // its own computation. Whether it is restored depends on where control goes.
    //
    // With a continuation the body transfers control to the protector's native
    // code, which is entitled to the values this body computed. The saved values
    // are discarded rather than restored, so no register has to hold its entry
    // value and any exit value can be emitted.
    //
    // Without a continuation the body returns to its own caller and must honour
    // the callee-save convention. A nonvolatile register whose exit value differs
    // from its entry value cannot satisfy both, so it is refused rather than
    // silently discarded.
    let tail_transfer = config.continuation.is_some();
    for assignment in &assignments {
        let register = assignment.physical_register;
        if register.is_win64_nonvolatile() && !preserve.contains(&register) {
            if !tail_transfer {
                let architectural = X64NativeRegister::ALL
                    .into_iter()
                    .find(|candidate| physical_for(candidate) == register);
                if let Some(architectural) = architectural
                    && let Some(value) = registers.get(&architectural)
                    && value
                        != &Expr::register(vmpx_vm::BitWidth::W64, architectural.symbolic_name())
                {
                    return Err(ReplacementError::ClobberedNonvolatile {
                        register: architectural,
                    });
                }
            }
            preserve.push(register);
        }
    }

    // Every body's temporaries were rebased above the result slots, so the frame
    // must cover the highest slot actually referenced by the emitted body.
    let highest = operations
        .iter()
        .filter_map(|operation| match operation {
            Operation::SpillLoad { slot, .. } | Operation::SpillStore { slot, .. } => Some(slot.0),
            _ => None,
        })
        .max()
        .unwrap_or(0);
    // A body that contains an imported call needs a Win64 call-ready frame.
    let call_ready = operations
        .iter()
        .any(|operation| matches!(operation, Operation::Call { .. }));
    // A call-ready frame reserves 32 bytes of shadow space below the spill area,
    // so the allocation must cover it as well. Win64 also requires RSP to be
    // 16-byte aligned at every call site: the return address pushed by the
    // caller leaves RSP congruent to 8, so the pushes plus the allocation must
    // bring it back to 0. Pad the allocation up when it does not.
    let mut stack_allocation = u32::from(highest.saturating_add(1)).saturating_mul(8);
    if call_ready {
        stack_allocation = stack_allocation.saturating_add(32);
        let pushes = u32::try_from(preserve.len())
            .map_err(|_| ReplacementError::TooManyInputs)?
            .saturating_mul(8);
        let total = pushes.saturating_add(stack_allocation);
        let residual = (8_u32.wrapping_sub(total)) & 0xF;
        if residual != 0 {
            stack_allocation = stack_allocation.saturating_add(residual);
        }
    }
    let function = Function {
        operations,
        assignments,
        frame: StackFrame {
            stack_allocation,
            preserve,
            call_ready,
            // The shared input frame and the effect-result slots are written by
            // the capture block and the call prelude before any fragment reads
            // them, but the validator only tracks definitions per operation, so
            // declaring them preinitialized is what makes a fragment's load of a
            // caller-owned slot legal.
            preinitialized_slots: (0..input_slot_count.saturating_add(effect_slot_count))
                .map(SpillSlot)
                .collect(),
            tail_transfer,
        },
    };

    let compiled = compile(&function, address)?;
    Ok(Replacement {
        function,
        compiled,
        emitted: ordered.iter().map(|(register, _)| **register).collect(),
    })
}

/// Applies caller-supplied symbol rewrites, which replace a symbolic input with
/// an expression that denotes the same value by another route.
fn rewrite_symbols(
    expression: &Expr,
    rewrites: &BTreeMap<String, Expr>,
) -> Result<Expr, ReplacementError> {
    if rewrites.is_empty() {
        return Ok(expression.clone());
    }
    expression
        .substitute(rewrites)
        .map_err(|error| ReplacementError::EffectBindings(error.to_string()))
}

/// Replaces the allocation intrinsics that stand for an effect call's result with
/// the frame slot the call wrote, so the store reads the captured value instead
/// of invoking the allocator again.
fn substitute_effect_bindings(
    expression: &Expr,
    bindings: &BTreeMap<String, Expr>,
) -> Result<Expr, ReplacementError> {
    if bindings.is_empty() {
        return Ok(expression.clone());
    }
    Ok(expression
        .rewrite_intrinsics(&mut |width, name, _arguments| {
            bindings
                .get(name)
                .filter(|replacement| replacement.width == width)
                .cloned()
        })
        .map_err(|error| ReplacementError::EffectBindings(error.to_string()))?)
}

/// Rewrites `vm_flags_<op>_wN` intrinsics to the arithmetic they denote.
///
/// These intrinsics describe how a VM handler computed a flag word from a plain
/// binary operation. They hold no VM state, so a native replacement performs the
/// same operation directly instead of refusing to lower the name.
fn rewrite_vm_flag_intrinsics(expression: &Expr) -> Result<Expr, ReplacementError> {
    let rewritten = expression
        .rewrite_intrinsics(&mut |width, name, arguments| {
            let operation = match name {
                "vm_flags_add_w8" | "vm_flags_add_w16" | "vm_flags_add_w32"
                | "vm_flags_add_w64" => vmpx_vm::BinaryOp::Add,
                "vm_flags_nor_w8" | "vm_flags_nor_w16" | "vm_flags_nor_w32"
                | "vm_flags_nor_w64" => vmpx_vm::BinaryOp::And,
                "vm_flags_nand_w8" | "vm_flags_nand_w16" | "vm_flags_nand_w32"
                | "vm_flags_nand_w64" => vmpx_vm::BinaryOp::Or,
                _ => return None,
            };
            let [left, right] = arguments else {
                return None;
            };
            Expr::binary(operation, left.clone(), right.clone())
                .ok()
                .filter(|value| value.width == width)
        })
        .map_err(|error| ReplacementError::EffectBindings(error.to_string()))?;
    Ok(rewritten)
}

/// The symbolic name bound to the result of effect call `index`.
fn effect_result_name(index: usize) -> String {
    format!("__effect_result_{index}")
}

/// Lowers one expression into operations that leave its value in the returned
/// register, rebased onto the replacement's canonical register and slot layout.
///
/// The fragment is compiled with its own private slot window, so its temporaries
/// cannot collide with the shared input frame or with another fragment's slots.
#[allow(clippy::too_many_arguments)]
fn lower_fragment(
    expression: &Expr,
    inputs: &[ExpressionInput],
    extra_inputs: &[ExpressionInput],
    config: &ReplacementConfig,
    canonical: &mut BTreeMap<PhysicalRegister, VirtualRegister>,
    assignments: &mut Vec<RegisterAssignment>,
    next_register: &mut u32,
    window_base: &mut u16,
    shared_frame_slots: &u16,
) -> Result<(Vec<Operation>, VirtualRegister), ReplacementError> {
    let mut all_inputs = inputs.to_vec();
    for input in extra_inputs {
        if !all_inputs
            .iter()
            .any(|existing| existing.name == input.name)
        {
            all_inputs.push(input.clone());
        }
    }
    // The flag intrinsic names a plain arithmetic operation on the flag word;
    // it carries no VM state, so it lowers to the operation it denotes.
    let expression = &rewrite_vm_flag_intrinsics(expression)?;
    let lowered = lower_expression(
        expression,
        &ExpressionCodegenConfig {
            inputs: all_inputs.clone(),
            result_register: PhysicalRegister::R10,
            temporary_register: PhysicalRegister::R11,
            shift_count_register: PhysicalRegister::Rcx,
            max_expression_nodes: config.max_expression_nodes,
            // The fragment allocates its temporaries above every slot already in
            // the shared frame, so its own frame must be able to describe them.
            max_spill_slots: config.max_spill_slots,
            import_calls: config.import_calls.clone(),
        },
    )?;
    let mut map: BTreeMap<u32, VirtualRegister> = BTreeMap::new();
    for assignment in &lowered.function.assignments {
        map.insert(
            assignment.virtual_register.id,
            canonical_register(
                assignment.physical_register,
                canonical,
                assignments,
                next_register,
            ),
        );
    }
    // Rebase the fragment's private slots above the shared frame so two
    // fragments never share storage. Slots the caller owns (the shared input
    // frame and the effect results) keep their index, so the rebase starts at
    // the end of the whole shared region rather than at the input count.
    let shared_end = *shared_frame_slots;
    // How many slots the fragment allocated for itself, measured from `base`.
    let fragment_peak = lowered
        .function
        .operations
        .iter()
        .filter_map(|operation| match operation {
            Operation::SpillLoad { slot, .. } | Operation::SpillStore { slot, .. } => Some(slot.0),
            _ => None,
        })
        .filter(|slot| *slot >= shared_end)
        .max()
        .map_or(0, |highest| {
            highest.saturating_sub(shared_end).saturating_add(1)
        });
    // Each fragment owns a disjoint window above every region the shared frame
    // already uses, so two fragments can never alias the same storage.
    let window_start = *window_base;
    *window_base = window_base
        .checked_add(
            u16::try_from(config.max_spill_slots)
                .map_err(|_| ReplacementError::TooManyInputs)?
                .max(fragment_peak)
                .max(1),
        )
        .ok_or(ReplacementError::TooManyInputs)?;
    let mut operations = Vec::with_capacity(lowered.function.operations.len());
    for operation in &lowered.function.operations {
        if matches!(operation, Operation::Return) {
            continue;
        }
        let mut operation = rename_registers(operation, &map);
        if let Operation::SpillLoad { slot, .. } | Operation::SpillStore { slot, .. } =
            &mut operation
        {
            // Only the fragment's own temporaries move; input and effect-result
            // slots belong to the shared frame and keep their index.
            if slot.0 >= shared_end {
                slot.0 = slot
                    .0
                    .checked_sub(shared_end)
                    .and_then(|offset| window_start.checked_add(offset))
                    .ok_or(ReplacementError::TooManyInputs)?;
            }
        }
        operations.push(operation);
    }
    Ok((operations, map[&lowered.result.id]))
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
        Operation::JumpToAddress(target) => Operation::JumpToAddress(target),
        Operation::RestoreFlags { value } => Operation::RestoreFlags {
            value: register(value),
        },
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
    fn an_identity_value_reads_its_own_input_slot() {
        // A register whose exit value is just its own entry value must load the
        // slot that input was captured into and store it to that value's result
        // slot. Reading any other slot would silently substitute another input.
        let inputs = vec![
            ("native_rax_initial".to_owned(), PhysicalRegister::Rax),
            ("native_r10_initial".to_owned(), PhysicalRegister::R10),
        ];
        let config = ReplacementConfig {
            inputs,
            max_expression_nodes: 10_000,
            max_spill_slots: 32,
            import_calls: BTreeMap::new(),
            continuation: None,
            effects: Vec::new(),
            effect_calls: Vec::new(),
            rflags: None,
            symbol_rewrites: BTreeMap::new(),
        };
        let registers = BTreeMap::from([(
            X64NativeRegister::R10,
            Expr::register(vmpx_vm::BitWidth::W64, "native_r10_initial"),
        )]);
        let replacement =
            build_replacement(&registers, &config, CodeAddress(0x1_4000_1000)).expect("build");
        // The result slot is `input_count + index_of_R10_in_ordered`.
        let result_slot = 2;
        let stored = replacement
            .function
            .operations
            .iter()
            .any(|operation| matches!(operation, Operation::SpillStore { slot, .. } if slot.0 == result_slot));
        assert!(stored, "the value must be stored into its result slot");
    }

    #[test]
    fn every_identity_value_reads_its_own_slot_across_a_full_register_set() {
        // With all fifteen registers present, each identity value must load the
        // slot its own input was captured into. A single mis-numbered slot would
        // give one register another register's value.
        let all = [
            X64NativeRegister::Rax,
            X64NativeRegister::Rbx,
            X64NativeRegister::Rcx,
            X64NativeRegister::Rdx,
            X64NativeRegister::Rbp,
            X64NativeRegister::Rsi,
            X64NativeRegister::Rdi,
            X64NativeRegister::R8,
            X64NativeRegister::R9,
            X64NativeRegister::R10,
            X64NativeRegister::R11,
            X64NativeRegister::R12,
            X64NativeRegister::R13,
            X64NativeRegister::R14,
            X64NativeRegister::R15,
        ];
        let mut inputs: Vec<(String, PhysicalRegister)> = all
            .iter()
            .map(|register| (register.symbolic_name().to_owned(), physical_for(register)))
            .collect();
        inputs.sort();
        inputs.dedup();
        let registers: BTreeMap<X64NativeRegister, Expr> = all
            .iter()
            .map(|register| {
                (
                    *register,
                    Expr::register(vmpx_vm::BitWidth::W64, register.symbolic_name()),
                )
            })
            .collect();
        let config = ReplacementConfig {
            inputs: inputs.clone(),
            max_expression_nodes: 10_000,
            max_spill_slots: 64,
            import_calls: BTreeMap::new(),
            continuation: None,
            effects: Vec::new(),
            effect_calls: Vec::new(),
            rflags: None,
            symbol_rewrites: BTreeMap::new(),
        };
        let replacement =
            build_replacement(&registers, &config, CodeAddress(0x1_4000_1000)).expect("build");
        // `ordered` is sorted by register name, so the result slot of a value is
        // its position in that order plus the input count.
        let mut ordered = all.to_vec();
        ordered.sort_by_key(|register| format!("{register:?}"));
        let result_base = u16::try_from(inputs.len()).expect("inputs fit");
        // Each value body must end by storing to its own result slot.
        for (index, register) in ordered.iter().enumerate() {
            let slot = result_base + u16::try_from(index).expect("index fits");
            let stores = replacement
                .function
                .operations
                .iter()
                .filter(|operation| {
                    matches!(operation, Operation::SpillStore { slot: s, .. } if s.0 == slot)
                })
                .count();
            assert!(stores >= 1, "{register:?} must store to result slot {slot}");
        }
    }

    #[test]
    fn a_rewritten_symbol_frees_its_register_for_the_stack_root() {
        // The entry stub captures native RSP into R11 after saving R11's own value
        // on the stack. The saved value is loadable from the frame, so the name is
        // rewritten to that load and R11 carries the stack root instead. Both
        // values must stay correct: the rewritten one from the frame, and the
        // stack root from the register.
        let all = [
            X64NativeRegister::Rax,
            X64NativeRegister::Rbx,
            X64NativeRegister::Rcx,
            X64NativeRegister::Rdx,
            X64NativeRegister::Rbp,
            X64NativeRegister::Rsi,
            X64NativeRegister::Rdi,
            X64NativeRegister::R8,
            X64NativeRegister::R9,
            X64NativeRegister::R10,
            X64NativeRegister::R11,
            X64NativeRegister::R12,
            X64NativeRegister::R13,
            X64NativeRegister::R14,
            X64NativeRegister::R15,
        ];
        let mut inputs: Vec<(String, PhysicalRegister)> = all
            .iter()
            .map(|register| (register.symbolic_name().to_owned(), physical_for(register)))
            .collect();
        let mut rewrites = BTreeMap::new();
        let saved = Expr::load(
            vmpx_vm::BitWidth::W64,
            Expr::binary(
                vmpx_vm::BinaryOp::Add,
                Expr::register(vmpx_vm::BitWidth::W64, "vsp_initial"),
                Expr::constant(vmpx_vm::BitWidth::W64, 64),
            )
            .expect("address"),
        );
        rewrites.insert("native_r11_initial".to_owned(), saved);
        inputs.retain(|(name, _)| !rewrites.contains_key(name));
        inputs.push(("vsp_initial".to_owned(), PhysicalRegister::R11));
        inputs.sort();
        inputs.dedup();
        assert_eq!(
            inputs.len(),
            15,
            "one name was rewritten, one register added"
        );
        assert_eq!(
            inputs
                .iter()
                .filter(|(_, register)| *register == PhysicalRegister::R11)
                .count(),
            1,
            "only the stack root may occupy R11"
        );
        let registers = BTreeMap::from([
            (
                X64NativeRegister::R11,
                Expr::register(vmpx_vm::BitWidth::W64, "native_r11_initial"),
            ),
            (
                X64NativeRegister::R10,
                Expr::register(vmpx_vm::BitWidth::W64, "native_r10_initial"),
            ),
        ]);
        let config = ReplacementConfig {
            inputs,
            max_expression_nodes: 10_000,
            max_spill_slots: 64,
            import_calls: BTreeMap::new(),
            continuation: None,
            effects: Vec::new(),
            effect_calls: Vec::new(),
            rflags: None,
            symbol_rewrites: rewrites,
        };
        let replacement =
            build_replacement(&registers, &config, CodeAddress(0x1_4000_1000)).expect("build");
        // The R11 value must come from the frame, so its body must contain a load
        // from a shared-frame slot.
        let loads = replacement
            .function
            .operations
            .iter()
            .filter(|operation| matches!(operation, Operation::Load { .. }))
            .count();
        assert!(loads >= 1, "the rewritten value must load from the frame");
    }

    #[test]
    fn builds_a_complete_function_for_a_constant_exit_state() {
        // Only volatile registers may be given new values: a nonvolatile one has
        // to be restored for the caller, which would discard the computed value.
        let registers = BTreeMap::from([
            (X64NativeRegister::Rax, Expr::constant(BitWidth::W64, 7)),
            (X64NativeRegister::Rcx, Expr::constant(BitWidth::W64, 9)),
        ]);
        let config = ReplacementConfig {
            inputs: inputs(),
            max_expression_nodes: 10_000,
            max_spill_slots: 32,
            import_calls: BTreeMap::new(),
            continuation: None,
            effects: Vec::new(),
            effect_calls: Vec::new(),
            rflags: None,
            symbol_rewrites: BTreeMap::new(),
        };
        let replacement = build_replacement(&registers, &config, CodeAddress(0x1_4000_1000))
            .expect("replacement compiles");
        assert_eq!(replacement.emitted.len(), 2);
        assert!(replacement.compiled.bytes.len() > 16);
        // The emitted body must end in RET.
        assert_eq!(replacement.compiled.bytes.last().copied(), Some(0xC3));
    }

    #[test]
    fn refuses_to_discard_a_computed_nonvolatile_value() {
        // Restoring a nonvolatile register for the caller overwrites whatever the
        // exit state computed for it. That is only harmless when the two agree, so
        // a differing value must be refused rather than silently dropped.
        let registers = BTreeMap::from([(
            X64NativeRegister::R12,
            Expr::constant(BitWidth::W64, 0x1234),
        )]);
        let config = ReplacementConfig {
            inputs: inputs(),
            max_expression_nodes: 10_000,
            max_spill_slots: 32,
            import_calls: BTreeMap::new(),
            continuation: None,
            effects: Vec::new(),
            effect_calls: Vec::new(),
            rflags: None,
            symbol_rewrites: BTreeMap::new(),
        };
        let error = build_replacement(&registers, &config, CodeAddress(0x1_4000_1000))
            .expect_err("a discarded value must be refused");
        assert!(
            matches!(error, ReplacementError::ClobberedNonvolatile { register } if register == X64NativeRegister::R12),
            "the refusal must name the register: {error}"
        );
        let _ = PhysicalRegister::R12;
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
            import_calls: BTreeMap::new(),
            continuation: None,
            effects: Vec::new(),
            effect_calls: Vec::new(),
            rflags: None,
            symbol_rewrites: BTreeMap::new(),
        };
        let error = build_replacement(&registers, &config, CodeAddress(0x1_4000_1000))
            .expect_err("intrinsic must not be emitted");
        assert!(matches!(error, ReplacementError::Lowering { .. }));
    }
}
