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

/// Upper bound on the words a region exit may stage as its continuation stack.
/// A below-VSP exit frame stages the whole span up to the entry stack pointer;
/// an implausibly large span is refused rather than walked.
const MAX_STAGED_CONTINUATION_WORDS: i64 = 512;

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
        // Wide multiply and unsigned divide are symbolically executed to packed
        // double-width results and their flag words; `expr_codegen` lowers each by
        // re-emitting the real `mul`/`imul`/`div` (see those lowerings).
        S::Mul | S::SignedMultiply | S::UnsignedDivide => L::Computed,
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
    /// An external store whose address could not be placed in a tracked region
    /// was narrowed under the assumption that it does not alias the VM's private
    /// operand stack (caller/parameter pointers do not point into it). The store
    /// is emitted; only the non-vsp-rooted cells it could alias were discarded.
    /// Mirrors the CFG's `VmCfgAssumption::AssumesExternalStoresDisjointFromVmStack`.
    ExternalStoresDisjointFromVmStack,
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
/// caller: the `ret` pops a word the VM wrote below the entry stack pointer, so
/// the continuation runs on the stack the VM left. The slot sits inside the
/// captured entry frame, or below the captured VSP when the VM built a fresh
/// exit frame there (a negative offset).
#[derive(Debug)]
pub struct RegionExit {
    pub registers: BTreeMap<X64NativeRegister, Expr>,
    pub rflags: Expr,
    /// Words from the `ret` slot up to the entry stack pointer, lowest first.
    /// Word zero is the computed target; the rest are the continuation's stack.
    pub staged: Vec<Expr>,
    /// Signed byte offset of the `ret` slot from the initial VM stack pointer;
    /// negative when the VM's exit frame sits below the captured VSP.
    pub slot_offset: i64,
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
    top_offset: u64,
) -> Result<StubExit, DecodedLiftError> {
    match recover_exit(symbolic, exit, frame_cells, top_offset)? {
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
/// `top_offset` is the VSP-to-top-cell byte offset of the entry frame (see
/// [`crate::recover_entry_frame`]); the entry stack pointer sits one word above
/// the highest captured cell, at `top_offset + frame_cells * 8`. A capture that
/// ignored it placed the entry stack pointer too low and mistook an in-frame
/// continuation slot (region exit) for a VM-written caller slot.
///
/// A VM-written slot exactly at the entry stack pointer whose value is provably
/// the caller's own pushed return address (threaded unchanged through the VM) is
/// a clean callee return: popping it is byte-identical to popping the untouched
/// slot. Any other VM-written slot above the frame is refused.
///
/// # Errors
///
/// Refuses a return through a VM-written slot outside the entry frame that is
/// not the proven caller return, an untracked staged word, or an inconsistent
/// layout.
pub fn recover_exit(
    symbolic: &vmpx_vm::SymbolicVmExecutionReport,
    exit: &VmExitReport,
    frame_cells: usize,
    top_offset: u64,
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

    let entry_sp = i64::try_from(frame_cells * 8)
        .unwrap_or(i64::MAX)
        .saturating_add(top_offset.cast_signed());
    let return_slot = final_offset + i64::from(return_offset);
    match read(return_offset) {
        Err(DecodedLiftError::NativeExit(message)) if message.contains("no byte at address") => {}
        Err(error) => return Err(error),
        // The VM wrote the return slot below the entry stack pointer: the target
        // is a computed continuation the exit transfers through, with the words
        // up to the entry stack pointer its stack (region exit). The slot may sit
        // inside the captured entry frame or below the captured VSP (the VM built
        // a fresh exit frame there); either way every staged word must be a
        // tracked value or the lift fails closed.
        Ok(_) if return_slot < entry_sp && return_slot % 8 == 0 => {
            let staged_words = entry_sp.checked_sub(return_slot).map_or(0, |span| span / 8);
            if staged_words > MAX_STAGED_CONTINUATION_WORDS {
                return Err(fail(format!(
                    "region continuation stages {staged_words} words, over the {MAX_STAGED_CONTINUATION_WORDS} limit"
                )));
            }
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
                slot_offset: return_slot,
            }));
        }
        // The VM wrote the slot exactly at the entry stack pointer with the
        // caller's own pushed return address, threaded unchanged through the VM
        // stack. Popping it is byte-identical to popping the untouched caller
        // slot, so this is a clean callee return.
        Ok(value) if return_slot == entry_sp && is_caller_return_address(&value) => {}
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

/// Whether `expression` is provably the caller's pushed return address: the
/// opaque root the entry frame threads for its `ReturnAddress` cell. Only this
/// structural identity proves a VM-written return slot equals the standard
/// callee return; a reconstructed address (e.g. relocation base plus a constant)
/// is not accepted, so the lift fails closed on it.
fn is_caller_return_address(expression: &Expr) -> bool {
    is_register(expression, "native_return_address_initial")
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
    let mut exit_state =
        recover_stub_exit(&symbolic, input.exit, input.frame.len(), input.top_offset)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use vmpx_pe::Rva;
    use vmpx_vm::{
        RecoveryStatus, SymbolicExecutionStop, SymbolicVmExecutionReport, VmExitRestoreSlot,
    };

    fn store_qword(memory: &mut Vec<(Expr, Expr)>, offset: i64, value: &Expr) {
        let base = Expr::register(BitWidth::W64, "vsp_initial");
        for byte in 0..8 {
            let address = Expr::binary(
                BinaryOp::Add,
                base.clone(),
                Expr::constant(BitWidth::W64, offset.wrapping_add(i64::from(byte)).cast_unsigned()),
            )
            .expect("byte address");
            memory.push((address, Expr::extract(BitWidth::W8, byte * 8, value.clone()).unwrap()));
        }
    }

    /// Standard 17-qword exit frame (15 registers, flags, return) whose return
    /// slot at `return_offset` holds `return_value`. `final_vsp == vsp_initial`.
    fn report(return_value: &Expr) -> (SymbolicVmExecutionReport, VmExitReport) {
        let mut memory = Vec::new();
        for (index, register) in X64NativeRegister::ALL.iter().enumerate() {
            let offset = i64::try_from(index * 8).unwrap();
            store_qword(&mut memory, offset, &Expr::register(BitWidth::W64, register.symbolic_name()));
        }
        store_qword(&mut memory, 120, &Expr::register(BitWidth::W64, "native_rflags_initial"));
        store_qword(&mut memory, 128, return_value);
        let symbolic = SymbolicVmExecutionReport {
            schema_version: 4,
            executed_instructions: 0,
            stop_reason: SymbolicExecutionStop::ControlFlowBoundary,
            final_vsp: Expr::register(BitWidth::W64, "vsp_initial"),
            final_vsp_offset: Some(0),
            final_flags: Expr::constant(BitWidth::W64, 0),
            branch_target: None,
            branch_target_dependencies: Vec::new(),
            context_cells: BTreeMap::new(),
            memory_values: memory,
            memory_epoch: 0,
            memory_tainted: false,
            image_reads_suppressed: false,
            assumed_external_disjoint_from_vm_stack: false,
            external_writes: Vec::new(),
            memory_havocs: Vec::new(),
            native_calls: Vec::new(),
            fresh_allocations: Vec::new(),
            next_native_call_epoch: 0,
            peak_expression_nodes: 1,
            tracked_memory_bytes: 136,
            resolved_image_bytes: 0,
        };
        let exit = VmExitReport {
            schema_version: 2,
            entry: Rva(0x1000),
            status: RecoveryStatus::Recovered,
            stack_switch: None,
            restored_registers: Vec::new(),
            restore_slots: X64NativeRegister::ALL
                .iter()
                .enumerate()
                .map(|(index, register)| VmExitRestoreSlot {
                    register: *register,
                    stack_offset: u32::try_from(index * 8).unwrap(),
                    pop: Rva(0),
                })
                .collect(),
            flags_restore: None,
            flags_stack_offset: Some(120),
            native_return: None,
            return_stack_offset: Some(128),
            frame_size: Some(136),
            confidence: 100,
            evidence: Vec::new(),
            diagnostics: Vec::new(),
        };
        (symbolic, exit)
    }

    // The return slot (offset 128) lies one qword above the top captured cell of
    // a `top_offset`-shifted 12-cell frame (top cell at 40 + 11*8 = 128), so the
    // VM-written target is a continuation inside the frame: a region exit. The
    // same layout read with the un-shifted entry stack pointer wrongly placed the
    // slot above the frame and refused it.
    #[test]
    fn a_top_offset_shifted_in_frame_written_slot_is_a_region_exit() {
        let target = Expr::binary(
            BinaryOp::Add,
            Expr::register(BitWidth::W64, "native_r13_initial"),
            Expr::constant(BitWidth::W64, 0x1_8001_D7DE),
        )
        .unwrap();
        let (symbolic, exit) = report(&target);
        // 12 cells, top_offset 0x28: entry stack pointer at 0x28 + 96 = 136 > 128.
        assert!(matches!(
            recover_exit(&symbolic, &exit, 12, 0x28),
            Ok(RecoveredExit::Region(_))
        ));
        // Ignoring top_offset put the entry stack pointer at 96 < 128 and refused.
        assert!(matches!(
            recover_exit(&symbolic, &exit, 12, 0),
            Err(DecodedLiftError::NotCalleeReturn(_))
        ));
    }

    // A VM-written return slot sitting exactly at the entry stack pointer whose
    // value is provably the caller's own pushed return address is a clean callee
    // return (pop 0): popping it is byte-identical to the untouched slot.
    #[test]
    fn a_written_slot_proven_to_be_the_caller_return_is_a_clean_return() {
        let return_address = Expr::register(BitWidth::W64, "native_return_address_initial");
        let (symbolic, exit) = report(&return_address);
        // 16 cells, top_offset 0: entry stack pointer at 128 == return slot.
        match recover_exit(&symbolic, &exit, 16, 0) {
            Ok(RecoveredExit::Callee(stub)) => assert_eq!(stub.return_pop, 0),
            other => panic!("expected a clean callee return, got {other:?}"),
        }
    }

    // The same position, but the VM computed some other target: not provably the
    // caller return, so it is refused (fail closed) rather than lifted as `ret`.
    #[test]
    fn a_written_slot_with_a_computed_non_return_target_is_refused() {
        let computed = Expr::constant(BitWidth::W64, 0x1_8002_A700);
        let (symbolic, exit) = report(&computed);
        assert!(matches!(
            recover_exit(&symbolic, &exit, 16, 0),
            Err(DecodedLiftError::NotCalleeReturn(_))
        ));
    }

    // The VM built its exit frame below the captured VSP (`final_vsp` is negative
    // of `vsp_initial`), so the return slot and the whole staged continuation sit
    // below the entry frame. When every staged word up to the entry stack pointer
    // is tracked, this is a region exit with a negative `slot_offset`.
    #[test]
    fn a_below_vsp_written_slot_is_a_region_exit_with_a_negative_offset() {
        // final_vsp = vsp_initial - 232; return slot at final_vsp + 128 = -104.
        let final_offset: i64 = -232;
        let mut memory = Vec::new();
        // Track every qword from the exit frame bottom up to the entry stack
        // pointer (frame_cells = 17, top_offset = 0 => entry_sp = 136).
        for slot in (final_offset..136).step_by(8) {
            store_qword(&mut memory, slot, &Expr::constant(BitWidth::W64, 0x1111_u64.wrapping_add(slot.cast_unsigned())));
        }
        let final_vsp = Expr::binary(
            BinaryOp::Add,
            Expr::register(BitWidth::W64, "vsp_initial"),
            Expr::constant(BitWidth::W64, final_offset.cast_unsigned()),
        )
        .unwrap();
        let symbolic = SymbolicVmExecutionReport {
            schema_version: 4,
            executed_instructions: 0,
            stop_reason: SymbolicExecutionStop::ControlFlowBoundary,
            final_vsp,
            final_vsp_offset: Some(final_offset),
            final_flags: Expr::constant(BitWidth::W64, 0),
            branch_target: None,
            branch_target_dependencies: Vec::new(),
            context_cells: BTreeMap::new(),
            memory_values: memory,
            memory_epoch: 0,
            memory_tainted: false,
            image_reads_suppressed: false,
            assumed_external_disjoint_from_vm_stack: false,
            external_writes: Vec::new(),
            memory_havocs: Vec::new(),
            native_calls: Vec::new(),
            fresh_allocations: Vec::new(),
            next_native_call_epoch: 0,
            peak_expression_nodes: 1,
            tracked_memory_bytes: 0,
            resolved_image_bytes: 0,
        };
        let (_, exit) = report(&Expr::constant(BitWidth::W64, 0));
        match recover_exit(&symbolic, &exit, 17, 0) {
            Ok(RecoveredExit::Region(region)) => {
                assert_eq!(region.slot_offset, -104);
                // staged spans the return slot up to the entry stack pointer.
                assert_eq!(region.staged.len(), usize::try_from((136 - -104) / 8).unwrap());
            }
            other => panic!("expected a below-VSP region exit, got {other:?}"),
        }
    }
}
