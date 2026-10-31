//! Shared facts about operations: barriers, register accesses, flag effects.

use super::Instr;
use crate::abi::operation_register_accesses;
use crate::{Operation, UnaryOp, VirtualRegister, Width};
use std::collections::BTreeSet;

/// Ends a block or has effects the passes do not model. Every fact is dropped
/// at a barrier, and a scan that reaches one treats state as live.
pub(crate) fn is_barrier(operation: &Operation) -> bool {
    matches!(
        operation,
        Operation::Label(_)
            | Operation::Jump(_)
            | Operation::JumpToAddress(_)
            | Operation::BranchIf { .. }
            | Operation::Return
            | Operation::ReturnPopping { .. }
            | Operation::ReturnRetaining { .. }
            | Operation::Call { .. }
            | Operation::RestoreFlags { .. }
            | Operation::Cpuid { .. }
            | Operation::MultiplyFlags { .. }
            | Operation::UnsignedDivide { .. }
    )
}

/// Register-only operations: no memory, frame, flag-input or control effect
/// beyond writing their register destination (and possibly flags).
pub(crate) fn is_pure_register_op(operation: &Operation) -> bool {
    matches!(
        operation,
        Operation::MoveImmediate { .. }
            | Operation::Move { .. }
            | Operation::Binary { .. }
            | Operation::Unary { .. }
            | Operation::Shift { .. }
            | Operation::Resize { .. }
            | Operation::Compare { .. }
            | Operation::DoubleShift { .. }
            | Operation::ReadFlags { .. }
            | Operation::EntryStackPointer { .. }
    )
}

pub(crate) struct Accesses {
    pub reads: BTreeSet<VirtualRegister>,
    pub writes: BTreeSet<VirtualRegister>,
}

pub(crate) fn accesses(operation: &Operation) -> Accesses {
    let mut reads = BTreeSet::new();
    let mut writes = BTreeSet::new();
    operation_register_accesses(operation, &mut reads, &mut writes);
    Accesses { reads, writes }
}

#[derive(Clone, Copy)]
pub(crate) struct FlagEffect {
    /// Earlier flag definitions stay observable through this operation.
    pub reads: bool,
    /// Fully redefines the arithmetic flags.
    pub defines: bool,
}

/// Conservative: only operations that define every arithmetic flag count as
/// definitions; logical, multiply and shift forms leave some flag undefined or
/// unchanged, so they are treated as keeping earlier flags observable.
pub(crate) fn flag_effect(operation: &Operation) -> FlagEffect {
    use crate::BinaryOp;
    let (reads, defines) = match operation {
        Operation::Binary {
            operation: BinaryOp::Add | BinaryOp::Subtract,
            ..
        }
        | Operation::Unary {
            operation: UnaryOp::Negate,
            ..
        }
        | Operation::Compare { .. } => (false, true),
        Operation::Binary { .. }
        | Operation::Shift { .. }
        | Operation::DoubleShift { .. }
        | Operation::BranchIf { .. }
        | Operation::ReadFlags { .. } => (true, false),
        _ => (false, false),
    };
    FlagEffect { reads, defines }
}

/// True when the flags written by `program[index]` are provably redefined
/// before any possible reader. Block ends and barriers count as live.
pub(crate) fn flags_dead_after(program: &[Instr], index: usize) -> bool {
    for instr in &program[index + 1..] {
        if is_barrier(&instr.op) {
            return false;
        }
        let effect = flag_effect(&instr.op);
        if effect.reads {
            return false;
        }
        if effect.defines {
            return true;
        }
    }
    false
}

pub(crate) const fn view_mask(width: Width) -> u64 {
    super::expr::mask(width.bits())
}

/// Sign-extends the low `width` bits of `value` to 64 bits.
pub(crate) fn sign_extend(value: u64, width: Width) -> u64 {
    let bits = width.bits();
    if bits == 64 {
        value
    } else {
        let shift = 64 - bits;
        (((value << shift) as i64) >> shift) as u64
    }
}
