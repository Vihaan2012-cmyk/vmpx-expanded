//! Bridge from a decoded linear VM stream to a native replacement.
//!
//! The decoder yields [`DecodedVmInstruction`]s; the replacement backend wants
//! an exact native exit state. This module symbolically executes the decoded
//! stream from a proven entry frame, recovers the exit state from the VM-exit
//! handler's restore layout, checks that the exit really is a callee-style
//! return into the original continuation, and compiles the result.
//!
//! Every VM instruction is first gated by [`semantic_lowering`]. A semantic with
//! no verified lowering stops the lift with its index and reason; nothing is
//! approximated.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;
use thiserror::Error;
use vmpx_recompile::{CodeAddress, PhysicalRegister};
use vmpx_vm::{
    BinaryOp, BitWidth, DecodedVmInstruction, Expr, ExprKind, NativeFrameSource,
    SymbolicExecutionLimits, SymbolicMemorySource, VmExitReport, VmHandlerSemantic,
    X64NativeRegister, execute_symbolic_linear_block_with_initial_state,
    symbolic_pointer_offset,
};

use crate::{
    EffectStore, Replacement, ReplacementConfig, ReplacementError, build_replacement_popping,
    entry_frame::frame_initial_state,
};

/// How one VM semantic reaches native code.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SemanticLowering {
    /// Symbolically executed to a pure expression the backend computes.
    Computed,
    /// Re-emitted as the identical native instruction. `boundary` marks a
    /// machine-state read that static equivalence must trust.
    Passthrough { boundary: bool },
    /// Terminates a linear block; handled by the exit bridge.
    BlockExit,
    /// No verified lowering. The lift refuses.
    Refused(&'static str),
}

#[must_use]
pub fn semantic_lowering(semantic: VmHandlerSemantic) -> SemanticLowering {
    use SemanticLowering as L;
    use VmHandlerSemantic as S;
    match semantic {
        S::Dispatch
        | S::ConstantPush
        | S::VirtualRegisterRead
        | S::VirtualRegisterWrite
        | S::Push
        | S::Pop
        | S::Add
        | S::Sub
        | S::Nor
        | S::Nand
        | S::ShiftLeft
        | S::ShiftRightLogical
        | S::ShiftRightArithmetic
        | S::RotateLeft
        | S::RotateRight
        | S::MemoryLoad
        | S::MemoryStore => L::Computed,
        S::Cpuid | S::GsRelativeLoad => L::Passthrough { boundary: true },
        S::ShiftLeftDouble | S::ShiftRightDouble | S::PopFlags => {
            L::Passthrough { boundary: false }
        }
        S::VmExit => L::BlockExit,
        S::ReadTimestampCounter => L::Refused("rdtsc has no lowering yet"),
        S::Mul | S::SignedMultiply | S::UnsignedDivide => {
            L::Refused("wide multiply/divide has no lowering yet")
        }
        S::TableChecksum => L::Refused("table checksum has no lowering"),
        S::ConditionalBranch | S::UnconditionalBranch | S::NativeCall => {
            L::Refused("control transfer needs the CFG lift")
        }
        S::NativeTransition | S::ContextInitialize => L::Refused("profile transition"),
        S::Xor
        | S::And
        | S::Or
        | S::Not
        | S::Neg
        | S::Compare
        | S::Flags
        | S::StackAdjust
        | S::Shift
        | S::Rotate
        | S::SignExtend
        | S::ZeroExtend => L::Refused("not modelled by the symbolic executor"),
        S::Unknown => L::Refused("unknown semantic"),
    }
}

/// A condition the replacement relies on that the lift does not itself prove.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LiftAssumption {
    /// The VM restores the caller's flags, the replacement leaves them
    /// clobbered. Valid because flags are volatile across a Win64 call.
    FlagsVolatileAtReturn,
    /// A native call whose target has no memory summary is assumed not to write
    /// the VM stack or context. Applied only when neither the target nor any
    /// argument can carry a pointer into the VM stack.
    VmStateSurvivesNativeCall,
    /// A branch was resolved by the walk under its preferred-base frame: a
    /// register used only to relocate in-image addresses was bound to 0 (the
    /// relocation delta, 0 at the preferred base). Carried over from the CFG's
    /// `VmCfgAssumption::PreferredImageBase` when a guarded branch is reused.
    PreferredImageBase,
}

#[derive(Debug, Error)]
pub enum DecodedLiftError {
    #[error("instruction {index} ({semantic:?}) cannot be lowered: {reason}")]
    Unliftable {
        index: usize,
        semantic: VmHandlerSemantic,
        reason: &'static str,
    },
    #[error("symbolic execution failed: {0}")]
    Symbolic(String),
    #[error("native exit recovery failed: {0}")]
    NativeExit(String),
    #[error("the exit is not a clean callee return: {0}")]
    NotCalleeReturn(String),
    #[error("{0}")]
    Unsupported(String),
    #[error("CFG lift refused at block {block}: {reason}")]
    Cfg { block: vmpx_pe::Va, reason: String },
    #[error("replacement build failed: {0}")]
    Replacement(#[from] ReplacementError),
}

pub struct DecodedLinearInput<'a> {
    pub instructions: &'a [DecodedVmInstruction],
    /// Initial VM stack, top first.
    pub frame: &'a [NativeFrameSource],
    /// VSP-to-top-cell byte offset from [`crate::recover_entry_frame`].
    pub top_offset: u64,
    pub exit: &'a VmExitReport,
    pub memory: Option<&'a dyn SymbolicMemorySource>,
    pub limits: SymbolicExecutionLimits,
}

#[derive(Debug)]
pub struct LiftedLinearFunction {
    pub replacement: Replacement,
    pub exit_state: StubExit,
    pub instructions_lifted: usize,
    pub semantic_counts: BTreeMap<String, usize>,
    pub assumptions: Vec<LiftAssumption>,
}

fn physical_for(register: X64NativeRegister) -> PhysicalRegister {
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

/// Native state at a VM exit that returns to the original caller, expressed
/// relative to the stack pointer at the entry stub.
#[derive(Debug)]
pub struct StubExit {
    pub registers: BTreeMap<X64NativeRegister, Expr>,
    pub rflags: Expr,
    /// Bytes of caller stack discarded before the `ret`; zero is a plain `ret`.
    pub return_pop: u32,
}

/// An exit that transfers to a computed target instead of returning to the
/// caller: the `ret` pops a word the entry frame held, so the continuation runs
/// on the stack the VM left inside that frame.
#[derive(Debug)]
pub struct RegionExit {
    pub registers: BTreeMap<X64NativeRegister, Expr>,
    pub rflags: Expr,
    /// Words from the `ret` slot up to the entry stack pointer, lowest first.
    /// Word zero is the computed target; the rest are the continuation's stack.
    pub staged: Vec<Expr>,
    /// Offset of the `ret` slot from the initial VM stack pointer.
    pub slot_offset: u32,
}

#[derive(Debug)]
pub enum RecoveredExit {
    Callee(StubExit),
    Region(RegionExit),
}

fn add_offset(base: &Expr, offset: u32) -> Result<Expr, DecodedLiftError> {
    Expr::binary(BinaryOp::Add, base.clone(), Expr::constant(BitWidth::W64, u64::from(offset)))
        .map_err(|error| DecodedLiftError::NativeExit(error.to_string()))
}

/// Recovers a clean callee-return exit; a region exit is refused.
///
/// # Errors
///
/// See [`recover_exit`]; additionally refuses a region exit.
pub fn recover_stub_exit(
    symbolic: &vmpx_vm::SymbolicVmExecutionReport,
    exit: &VmExitReport,
    frame_cells: usize,
) -> Result<StubExit, DecodedLiftError> {
    match recover_exit(symbolic, exit, frame_cells)? {
        RecoveredExit::Callee(exit) => Ok(exit),
        RecoveredExit::Region(_) => Err(DecodedLiftError::NotCalleeReturn(
            "the return address was written by the VM, not read from caller stack".to_owned(),
        )),
    }
}

/// Reads the exit frame from the symbolic VM stack. The saved registers and
/// flags must be tracked values. A return address the VM never wrote (untouched
/// caller stack) at or above the entry stack pointer is a callee return, so a
/// native `ret` after `add rsp, pop` reads exactly the same word. A return
/// address held in a tracked cell of the entry frame is a region exit.
///
/// # Errors
///
/// Refuses a return through a VM-written slot outside the entry frame, an
/// untracked staged word, or an inconsistent layout.
pub fn recover_exit(
    symbolic: &vmpx_vm::SymbolicVmExecutionReport,
    exit: &VmExitReport,
    frame_cells: usize,
) -> Result<RecoveredExit, DecodedLiftError> {
    let fail = |message: String| DecodedLiftError::NativeExit(message);
    if exit.status != vmpx_vm::RecoveryStatus::Recovered {
        return Err(fail("VM exit layout is not proven".to_owned()));
    }
    let (Some(flags_offset), Some(return_offset), Some(frame_size)) =
        (exit.flags_stack_offset, exit.return_stack_offset, exit.frame_size)
    else {
        return Err(fail("VM exit layout is incomplete".to_owned()));
    };
    if return_offset.checked_add(8) != Some(frame_size) {
        return Err(fail("exit frame does not end at the return address".to_owned()));
    }
    let slots: BTreeSet<X64NativeRegister> =
        exit.restore_slots.iter().map(|slot| slot.register).collect();
    if exit.restore_slots.len() != X64NativeRegister::ALL.len()
        || slots.len() != X64NativeRegister::ALL.len()
    {
        return Err(fail("exit does not restore every register exactly once".to_owned()));
    }
    let final_offset = symbolic_pointer_offset(&symbolic.final_vsp, "vsp_initial")
        .ok_or_else(|| fail("final VSP is not an affine offset of the entry frame".to_owned()))?;
    let read = |offset: u32| -> Result<Expr, DecodedLiftError> {
        symbolic
            .read_tracked_memory(add_offset(&symbolic.final_vsp, offset)?, BitWidth::W64)
            .map_err(|error| fail(error.to_string()))
    };
    let mut registers = BTreeMap::new();
    for slot in &exit.restore_slots {
        registers.insert(slot.register, read(slot.stack_offset)?);
    }
    let rflags = read(flags_offset)?;

    let entry_sp = i64::try_from(frame_cells * 8).unwrap_or(i64::MAX);
    let return_slot = final_offset + i64::from(return_offset);
    match read(return_offset) {
        Err(DecodedLiftError::NativeExit(message)) if message.contains("no byte at address") => {}
        Err(error) => return Err(error),
        Ok(_) if (0..entry_sp).contains(&return_slot) && return_slot % 8 == 0 => {
            let mut staged = Vec::new();
            for slot in (return_slot..entry_sp).step_by(8) {
                let address = Expr::binary(
                    BinaryOp::Add,
                    symbolic.final_vsp.clone(),
                    Expr::constant(BitWidth::W64, (slot - final_offset).cast_unsigned()),
                )
                .map_err(|error| fail(error.to_string()))?;
                staged.push(
                    symbolic
                        .read_tracked_memory(address, BitWidth::W64)
                        .map_err(|error| fail(error.to_string()))?,
                );
            }
            return Ok(RecoveredExit::Region(RegionExit {
                registers,
                rflags,
                staged,
                slot_offset: u32::try_from(return_slot).map_err(|error| fail(error.to_string()))?,
            }));
        }
        Ok(_) => {
            return Err(DecodedLiftError::NotCalleeReturn(
                "the return address was written by the VM, not read from caller stack".to_owned(),
            ));
        }
    }
    if return_slot < entry_sp {
        return Err(DecodedLiftError::NotCalleeReturn(
            "the exit returns into the entry frame (the protector's continuation)".to_owned(),
        ));
    }
    let return_pop = u32::try_from(return_slot - entry_sp)
        .ok()
        .filter(|pop| pop % 8 == 0)
        .ok_or_else(|| DecodedLiftError::NotCalleeReturn("unaligned stack unwind".to_owned()))?;
    Ok(RecoveredExit::Callee(StubExit {
        registers,
        rflags,
        return_pop,
    }))
}

/// First instruction with no verified lowering, if any.
#[must_use]
pub fn first_unliftable(
    instructions: &[DecodedVmInstruction],
) -> Option<(usize, VmHandlerSemantic, &'static str)> {
    instructions
        .iter()
        .enumerate()
        .find_map(|(index, instruction)| match semantic_lowering(instruction.semantic) {
            SemanticLowering::Refused(reason) => Some((index, instruction.semantic, reason)),
            _ => None,
        })
}

fn only_entry_registers(expression: &Expr, what: &str, uses_root: &mut bool) -> Result<(), DecodedLiftError> {
    let allowed: BTreeSet<&str> = X64NativeRegister::ALL
        .iter()
        .map(|register| register.symbolic_name())
        .collect();
    let foreign: Vec<String> = expression
        .dependencies()
        .into_iter()
        .filter(|name| !allowed.contains(name.as_str()))
        .filter(|name| {
            let root = name == "vsp_initial";
            *uses_root |= root;
            !root
        })
        .collect();
    if foreign.is_empty() {
        Ok(())
    } else {
        Err(DecodedLiftError::Unsupported(format!(
            "{what} depends on {foreign:?}, which a callee-style replacement cannot read"
        )))
    }
}

/// The executor names a read of memory no tracked store covers
/// `memory_load_epoch_N(address)`. With no external store, native call or havoc
/// anywhere in the routine nothing can change that memory, so the read denotes a
/// plain load, which the replacement performs against the same bytes.
fn reads_of_untouched_memory(expression: &Expr) -> Result<Expr, DecodedLiftError> {
    expression
        .rewrite_intrinsics(&mut |width, name, arguments| match arguments {
            [address]
                if name.starts_with("memory_load_epoch_") && address.width == BitWidth::W64 =>
            {
                Some(Expr::load(width, address.clone()))
            }
            _ => None,
        })
        .map_err(|error| DecodedLiftError::Unsupported(error.to_string()))
}

fn is_register(expression: &Expr, name: &str) -> bool {
    matches!(&expression.simplify().kind, ExprKind::Register { name: found } if found == name)
}

/// Lifts a decoded linear function that ends in a proven VM exit.
///
/// # Errors
///
/// Refuses at the first semantic without a verified lowering, on any symbolic,
/// exit-recovery or layout failure, on memory havoc, native calls or fresh
/// allocations (not bridged yet), and on any exit value the replacement cannot
/// compute from the entry registers.
pub fn lift_decoded_linear(
    input: &DecodedLinearInput<'_>,
    address: CodeAddress,
) -> Result<LiftedLinearFunction, DecodedLiftError> {
    if let Some((index, semantic, reason)) = first_unliftable(input.instructions) {
        return Err(DecodedLiftError::Unliftable {
            index,
            semantic,
            reason,
        });
    }
    let (initial, _) = frame_initial_state(input.frame, input.top_offset);
    let symbolic = execute_symbolic_linear_block_with_initial_state(
        input.instructions,
        Some(VmHandlerSemantic::VmExit),
        input.limits,
        &initial,
        input.memory,
    )
    .map_err(|error| DecodedLiftError::Symbolic(error.to_string()))?;
    if !symbolic.memory_havocs.is_empty()
        || !symbolic.native_calls.is_empty()
        || !symbolic.fresh_allocations.is_empty()
    {
        return Err(DecodedLiftError::Unsupported(
            "memory havoc, native calls and allocations are not bridged yet".to_owned(),
        ));
    }
    let mut exit_state = recover_stub_exit(&symbolic, input.exit, input.frame.len())?;
    if symbolic.external_writes.is_empty() {
        for value in exit_state.registers.values_mut() {
            *value = reads_of_untouched_memory(value)?;
        }
        exit_state.rflags = reads_of_untouched_memory(&exit_state.rflags)?;
    }
    let mut uses_root = false;
    for (register, value) in &exit_state.registers {
        only_entry_registers(value, &format!("{register:?}"), &mut uses_root)?;
    }

    let mut assumptions = Vec::new();
    let rflags = if is_register(&exit_state.rflags, "native_rflags_initial") {
        assumptions.push(LiftAssumption::FlagsVolatileAtReturn);
        None
    } else {
        only_entry_registers(&exit_state.rflags, "exit rflags", &mut uses_root)?;
        Some(exit_state.rflags.clone())
    };
    let mut effects = Vec::new();
    for write in &symbolic.external_writes {
        only_entry_registers(&write.address, "store address", &mut uses_root)?;
        only_entry_registers(&write.value, "store value", &mut uses_root)?;
        effects.push(EffectStore {
            address: write.address.clone(),
            value: write.value.clone(),
            width: write.value.width,
        });
    }

    let config = ReplacementConfig {
        inputs: X64NativeRegister::ALL
            .iter()
            .map(|register| (register.symbolic_name().to_owned(), physical_for(*register)))
            .collect(),
        max_expression_nodes: 200_000,
        max_spill_slots: 64,
        import_calls: BTreeMap::new(),
        continuation: None,
        effects,
        effect_calls: Vec::new(),
        rflags,
        symbol_rewrites: BTreeMap::new(),
    };
    let replacement = build_replacement_popping(
        &exit_state.registers,
        &config,
        address,
        Some(exit_state.return_pop),
        uses_root.then(|| u32::try_from(input.frame.len() * 8).unwrap_or(u32::MAX)),
    )?;

    let mut semantic_counts = BTreeMap::new();
    for instruction in input.instructions {
        *semantic_counts
            .entry(format!("{:?}", instruction.semantic))
            .or_insert(0) += 1;
    }
    Ok(LiftedLinearFunction {
        replacement,
        exit_state,
        instructions_lifted: input.instructions.len(),
        semantic_counts,
        assumptions,
    })
}
