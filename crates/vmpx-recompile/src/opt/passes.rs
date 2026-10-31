//! The certified passes. Every pass is block-local: facts are dropped at any
//! barrier, and nothing marked volatile is ever edited or removed.

use super::analysis::{
    accesses, flags_dead_after, is_barrier, is_pure_register_op, sign_extend, view_mask,
};
use super::expr::{self, Expr, Rule};
use super::{Certificate, Instr, OptimizeOptions, Pass, Rejection, certify_rules};
use crate::{
    Address, BinaryOp, BinarySource, Operation, ResizeOp, ShiftCount, ShiftOp, UnaryOp,
    VirtualRegister, Width,
};
use std::collections::BTreeMap;

// ---------------------------------------------------------------- folding

fn expr_op(operation: BinaryOp) -> expr::Op {
    match operation {
        BinaryOp::Add => expr::Op::Add,
        BinaryOp::Subtract => expr::Op::Sub,
        BinaryOp::Multiply => expr::Op::Mul,
        BinaryOp::Xor => expr::Op::Xor,
        BinaryOp::And => expr::Op::And,
        BinaryOp::Or => expr::Op::Or,
    }
}

/// Value of an `Immediate(i32)` operand at `width`, or `None` when the
/// immediate is not representable at that width (left alone, never guessed).
fn immediate_value(value: i32, width: Width) -> Option<u64> {
    let signed = i64::from(value);
    let fits = match width {
        Width::W8 => i8::try_from(signed).is_ok(),
        Width::W16 => i16::try_from(signed).is_ok(),
        Width::W32 | Width::W64 => true,
    };
    fits.then(|| (signed as u64) & view_mask(width))
}

/// Inverse of [`immediate_value`].
fn to_immediate(value: u64, width: Width) -> Option<i32> {
    i32::try_from(sign_extend(value & view_mask(width), width) as i64).ok()
}

fn eval_binary(operation: BinaryOp, left: u64, right: u64, width: Width) -> u64 {
    expr::bin(expr_op(operation), expr::var(0), expr::var(1)).eval(&[left, right], width.bits())
}

pub(crate) fn eval_shift(operation: ShiftOp, value: u64, count: u8, width: Width) -> u64 {
    let bits = width.bits();
    let k = u32::from(count) & if bits == 64 { 63 } else { 31 };
    let m = view_mask(width);
    let v = value & m;
    match operation {
        ShiftOp::Left => {
            if k >= bits {
                0
            } else {
                (v << k) & m
            }
        }
        ShiftOp::RightLogical => {
            if k >= bits {
                0
            } else {
                v >> k
            }
        }
        ShiftOp::RightArithmetic => {
            (((sign_extend(v, width) as i64) >> k.min(bits - 1)) as u64) & m
        }
        ShiftOp::RotateLeft => {
            let r = k % bits;
            if r == 0 {
                v
            } else {
                ((v << r) | (v >> (bits - r))) & m
            }
        }
        ShiftOp::RotateRight => {
            let r = k % bits;
            if r == 0 {
                v
            } else {
                ((v >> r) | (v << (bits - r))) & m
            }
        }
    }
}

#[derive(Default)]
struct Known(BTreeMap<u32, u64>);

impl Known {
    fn view(&self, register: VirtualRegister) -> Option<u64> {
        self.0
            .get(&register.id)
            .map(|value| value & view_mask(register.width))
    }

    /// Mirrors the AMD64 write rules: 64 bits replace, 32 bits zero-extend,
    /// 8/16 bits merge into the previous value (unknown if it was unknown).
    fn write(&mut self, register: VirtualRegister, value: Option<u64>) {
        let Some(value) = value else {
            self.0.remove(&register.id);
            return;
        };
        let merged = match register.width {
            Width::W64 => Some(value),
            Width::W32 => Some(value & 0xffff_ffff),
            narrow => {
                let m = view_mask(narrow);
                self.0.get(&register.id).map(|old| (old & !m) | (value & m))
            }
        };
        match merged {
            Some(v) => {
                self.0.insert(register.id, v);
            }
            None => {
                self.0.remove(&register.id);
            }
        }
    }
}

pub struct ConstantFold;

impl ConstantFold {
    fn fold(
        operation: &Operation,
        known: &Known,
        flags_dead: impl Fn() -> bool,
    ) -> Option<Operation> {
        match operation {
            Operation::Move {
                destination,
                source,
            } => Some(Operation::MoveImmediate {
                destination: *destination,
                value: known.view(*source)?,
            }),
            Operation::Resize {
                operation,
                destination,
                source,
            } => {
                let value = known.view(*source)?;
                let value = match operation {
                    ResizeOp::Truncate | ResizeOp::ZeroExtend => value,
                    ResizeOp::SignExtend => sign_extend(value, source.width),
                } & view_mask(destination.width);
                Some(Operation::MoveImmediate {
                    destination: *destination,
                    value,
                })
            }
            Operation::Unary {
                operation,
                destination,
            } => {
                let value = known.view(*destination)?;
                let folded = match operation {
                    UnaryOp::Not => !value,
                    UnaryOp::Negate => {
                        if !flags_dead() {
                            return None;
                        }
                        value.wrapping_neg()
                    }
                } & view_mask(destination.width);
                Some(Operation::MoveImmediate {
                    destination: *destination,
                    value: folded,
                })
            }
            Operation::Shift {
                operation,
                destination,
                count: ShiftCount::Immediate(count),
            } => {
                let masked = u32::from(*count)
                    & if destination.width == Width::W64 {
                        63
                    } else {
                        31
                    };
                if masked == 0 || !flags_dead() {
                    return None;
                }
                let value = known.view(*destination)?;
                Some(Operation::MoveImmediate {
                    destination: *destination,
                    value: eval_shift(*operation, value, *count, destination.width),
                })
            }
            Operation::Binary {
                operation,
                destination,
                source,
            } => {
                let width = destination.width;
                let right = match source {
                    BinarySource::Immediate(value) => Some(immediate_value(*value, width)?),
                    BinarySource::Register(register) => known.view(*register),
                };
                match (known.view(*destination), right) {
                    (Some(left), Some(right)) if flags_dead() => Some(Operation::MoveImmediate {
                        destination: *destination,
                        value: eval_binary(*operation, left, right, width),
                    }),
                    (_, Some(right)) if matches!(source, BinarySource::Register(_)) => {
                        Some(Operation::Binary {
                            operation: *operation,
                            destination: *destination,
                            source: BinarySource::Immediate(to_immediate(right, width)?),
                        })
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }
}

impl Pass for ConstantFold {
    fn name(&self) -> &'static str {
        "constant-fold"
    }

    fn certify(&self) -> Result<Vec<Certificate>, Rejection> {
        Ok(vec![
            Certificate::Structural {
                invariant: "values come only from tracked MoveImmediate facts, evaluated by the \
                            single bitvector model (expr::eval / eval_shift) at the operand width",
            },
            Certificate::Structural {
                invariant: "an operation that writes flags is replaced by a flag-free move only \
                            when a later full flag definition precedes any flag reader in the block",
            },
            Certificate::Structural {
                invariant: "Register->Immediate rewrites are width-checked and keep the flag-writing \
                            instruction, so flag state is unchanged",
            },
        ])
    }

    fn run(&self, program: &mut Vec<Instr>, _options: &OptimizeOptions) -> usize {
        let mut known = Known::default();
        let mut rewrites = 0;
        for index in 0..program.len() {
            if is_barrier(&program[index].op) {
                known = Known::default();
                continue;
            }
            if !program[index].volatile
                && let Some(new_op) = Self::fold(&program[index].op, &known, || {
                    flags_dead_after(program, index)
                })
            {
                program[index].op = new_op;
                rewrites += 1;
            }
            match &program[index].op {
                Operation::MoveImmediate { destination, value } => {
                    known.write(*destination, Some(*value & view_mask(destination.width)));
                }
                other => {
                    for written in accesses(other).writes {
                        known.write(written, None);
                    }
                }
            }
        }
        rewrites
    }
}

// ---------------------------------------------------------------- rules

/// Applies the verified rewrite rules to single `Binary` instructions.
pub struct AlgebraicRules {
    rules: Vec<Rule>,
}

impl AlgebraicRules {
    #[must_use]
    pub fn standard() -> Self {
        Self::with_rules(expr::standard_rules())
    }

    /// A custom rule set. `certify` rejects the pass if any rule is not proven.
    #[must_use]
    pub fn with_rules(rules: Vec<Rule>) -> Self {
        Self { rules }
    }
}

impl Pass for AlgebraicRules {
    fn name(&self) -> &'static str {
        "algebraic-rules"
    }

    fn certify(&self) -> Result<Vec<Certificate>, Rejection> {
        certify_rules(&self.rules)
    }

    fn run(&self, program: &mut Vec<Instr>, _options: &OptimizeOptions) -> usize {
        let mut rewrites = 0;
        let mut index = 0;
        while index < program.len() {
            let instr = &program[index];
            if instr.volatile {
                index += 1;
                continue;
            }
            let Operation::Binary {
                operation,
                destination,
                source,
            } = instr.op
            else {
                index += 1;
                continue;
            };
            let width = destination.width;
            let right = match source {
                BinarySource::Immediate(value) => {
                    let Some(value) = immediate_value(value, width) else {
                        index += 1;
                        continue;
                    };
                    // Canonical sign-extended form so -1 is the all-ones constant.
                    Expr::Const(sign_extend(value, width))
                }
                BinarySource::Register(register) if register.id == destination.id => expr::var(0),
                BinarySource::Register(_) => expr::var(1),
            };
            let pattern = expr::bin(expr_op(operation), expr::var(0), right);
            let replacement = expr::apply_rules(&self.rules, &pattern).map(|(_, result)| result);
            let new_op = match replacement {
                // A 32-bit write also clears the upper half, so an identity at
                // W32 must stay a (flag-free) self move, not vanish.
                Some(Expr::Var(0)) if width == Width::W32 => Some(Some(Operation::Move {
                    destination,
                    source: destination,
                })),
                Some(Expr::Var(0)) => Some(None),
                Some(Expr::Const(value)) => Some(Some(Operation::MoveImmediate {
                    destination,
                    value: value & view_mask(width),
                })),
                _ => None,
            };
            match new_op {
                Some(replacement) if flags_dead_after(program, index) => {
                    rewrites += 1;
                    if let Some(op) = replacement {
                        program[index].op = op;
                        index += 1;
                    } else {
                        program.remove(index);
                    }
                }
                _ => index += 1,
            }
        }
        rewrites
    }
}

// ---------------------------------------------------------------- copy propagation

pub struct CopyPropagation;

fn substitute(copies: &BTreeMap<u32, VirtualRegister>, register: &mut VirtualRegister) -> bool {
    match copies.get(&register.id) {
        Some(source) if register.width.bits() <= source.width.bits() => {
            *register = VirtualRegister::new(source.id, register.width);
            true
        }
        _ => false,
    }
}

impl CopyPropagation {
    /// Rewrites only pure read operands; read-modify-write destinations, call
    /// arguments and shift counts are never substituted.
    fn rewrite_reads(operation: &mut Operation, copies: &BTreeMap<u32, VirtualRegister>) -> usize {
        let mut changed = 0;
        let mut sub = |register: &mut VirtualRegister| {
            if substitute(copies, register) {
                changed += 1;
            }
        };
        match operation {
            Operation::Move { source, .. } | Operation::Resize { source, .. } => sub(source),
            Operation::Binary {
                source: BinarySource::Register(register),
                ..
            } => sub(register),
            Operation::Compare { left, right } => {
                sub(left);
                if let BinarySource::Register(register) = right {
                    sub(register);
                }
            }
            Operation::Store { address, source } => {
                sub(&mut address.base);
                sub(source);
            }
            Operation::Load { address, .. } | Operation::GsLoad { address, .. } => {
                sub(&mut address.base);
            }
            Operation::SpillStore { source, .. } => sub(source),
            _ => {}
        }
        changed
    }
}

impl Pass for CopyPropagation {
    fn name(&self) -> &'static str {
        "copy-propagation"
    }

    fn certify(&self) -> Result<Vec<Certificate>, Rejection> {
        Ok(vec![
            Certificate::Structural {
                invariant: "a copy fact dst<-src is created by a same-width Move and dropped when \
                            either register family is written or at any barrier",
            },
            Certificate::Structural {
                invariant: "a read is redirected only if its width does not exceed the copy width, \
                            so zero-extension/merge bits of the destination are never observed",
            },
        ])
    }

    fn run(&self, program: &mut Vec<Instr>, _options: &OptimizeOptions) -> usize {
        let mut copies: BTreeMap<u32, VirtualRegister> = BTreeMap::new();
        let mut rewrites = 0;
        for instr in program.iter_mut() {
            if is_barrier(&instr.op) {
                copies.clear();
                continue;
            }
            if !instr.volatile {
                rewrites += Self::rewrite_reads(&mut instr.op, &copies);
            }
            for written in accesses(&instr.op).writes {
                copies.remove(&written.id);
                copies.retain(|_, source| source.id != written.id);
            }
            if let Operation::Move {
                destination,
                source,
            } = instr.op
                && destination.id != source.id
            {
                copies.insert(destination.id, source);
            }
        }
        rewrites
    }
}

// ---------------------------------------------------------------- dead definitions

/// Removes a flag-free register definition that is fully overwritten later in
/// the same block before any read.
pub struct DeadDefinitionElimination;

fn definition_is_dead(program: &[Instr], index: usize, target: VirtualRegister) -> bool {
    for instr in &program[index + 1..] {
        if is_barrier(&instr.op) {
            return false;
        }
        let access = accesses(&instr.op);
        if access.reads.iter().any(|r| r.id == target.id) {
            return false;
        }
        if access
            .writes
            .iter()
            .any(|w| w.id == target.id && matches!(w.width, Width::W32 | Width::W64))
        {
            return true;
        }
    }
    false
}

impl Pass for DeadDefinitionElimination {
    fn name(&self) -> &'static str {
        "dead-definition"
    }

    fn certify(&self) -> Result<Vec<Certificate>, Rejection> {
        Ok(vec![
            Certificate::Structural {
                invariant: "only Move/MoveImmediate/Resize (no flag or memory effect) are removable",
            },
            Certificate::Structural {
                invariant: "removed only if a 32/64-bit write to the same family occurs before any \
                            read of it, with no barrier between; barriers and block ends are live",
            },
        ])
    }

    fn run(&self, program: &mut Vec<Instr>, _options: &OptimizeOptions) -> usize {
        let mut rewrites = 0;
        let mut index = 0;
        while index < program.len() {
            let instr = &program[index];
            let target = match instr.op {
                Operation::Move {
                    destination,
                    source,
                } if destination.id != source.id => Some(destination),
                Operation::MoveImmediate { destination, .. }
                | Operation::Resize { destination, .. } => Some(destination),
                _ => None,
            };
            match target {
                Some(target) if !instr.volatile && definition_is_dead(program, index, target) => {
                    program.remove(index);
                    rewrites += 1;
                }
                _ => index += 1,
            }
        }
        rewrites
    }
}

// ---------------------------------------------------------------- dead stores

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Alias {
    /// Same bytes.
    Must,
    /// Provably non-overlapping.
    Disjoint,
    /// Cannot be decided; treated as possibly overlapping.
    May,
}

fn bytes(width: Width) -> i64 {
    i64::from(width.bits() / 8)
}

/// Aliasing between two accesses. Only a shared base family with constant
/// displacements is decidable; everything else is `May`.
#[must_use]
pub fn alias(a: Address, a_width: Width, b: Address, b_width: Width) -> Alias {
    if a.base.id != b.base.id {
        return Alias::May;
    }
    let (a0, b0) = (i64::from(a.displacement), i64::from(b.displacement));
    let (a1, b1) = (a0 + bytes(a_width), b0 + bytes(b_width));
    if a0 == b0 && a1 == b1 {
        Alias::Must
    } else if a1 <= b0 || b1 <= a0 {
        Alias::Disjoint
    } else {
        Alias::May
    }
}

/// Removes a non-volatile store overwritten by an identical-range store before
/// anything can observe it.
pub struct DeadStoreElimination;

fn memory_store_is_dead(
    program: &[Instr],
    index: usize,
    address: Address,
    width: Width,
    allow_disjoint: bool,
) -> bool {
    for instr in &program[index + 1..] {
        if instr.volatile || is_barrier(&instr.op) {
            return false;
        }
        match &instr.op {
            Operation::Store {
                address: other,
                source,
            } => match alias(address, width, *other, source.width) {
                Alias::Must => return true,
                Alias::Disjoint if allow_disjoint => {}
                _ => return false,
            },
            Operation::Load {
                address: other,
                destination,
            } => match alias(address, width, *other, destination.width) {
                Alias::Disjoint if allow_disjoint => {}
                _ => return false,
            },
            other if is_pure_register_op(other) => {}
            _ => return false,
        }
        if accesses(&instr.op)
            .writes
            .iter()
            .any(|w| w.id == address.base.id)
        {
            return false;
        }
    }
    false
}

fn spill_store_is_dead(
    program: &[Instr],
    index: usize,
    slot: crate::SpillSlot,
    width: Width,
) -> bool {
    for instr in &program[index + 1..] {
        if instr.volatile || is_barrier(&instr.op) {
            return false;
        }
        match &instr.op {
            Operation::SpillStore {
                slot: other,
                source,
            } if *other == slot => return source.width.bits() >= width.bits(),
            Operation::SpillLoad { slot: other, .. } if *other == slot => return false,
            Operation::SpillStore { .. } | Operation::SpillLoad { .. } => {}
            other if is_pure_register_op(other) => {}
            _ => return false,
        }
    }
    false
}

impl Pass for DeadStoreElimination {
    fn name(&self) -> &'static str {
        "dead-store"
    }

    fn certify(&self) -> Result<Vec<Certificate>, Rejection> {
        Ok(vec![
            Certificate::Structural {
                invariant: "a memory store is removed only when a same-base, same-displacement, \
                            same-width store follows with no intervening memory/frame/volatile/barrier \
                            operation and the base family unwritten; any May alias keeps the store",
            },
            Certificate::Structural {
                invariant: "a spill store is removed only when the same slot is rewritten at >= width \
                            before any read of the slot, with no address-based memory access between \
                            (frame slots may alias entry-stack-derived pointers)",
            },
            Certificate::Structural {
                invariant: "volatile operations are neither removed nor stepped over",
            },
        ])
    }

    fn run(&self, program: &mut Vec<Instr>, options: &OptimizeOptions) -> usize {
        let mut rewrites = 0;
        let mut index = 0;
        while index < program.len() {
            let instr = &program[index];
            let dead = !instr.volatile
                && match instr.op {
                    Operation::Store { address, source } => memory_store_is_dead(
                        program,
                        index,
                        address,
                        source.width,
                        options.assume_fault_free_disjoint_access,
                    ),
                    Operation::SpillStore { slot, source } => {
                        spill_store_is_dead(program, index, slot, source.width)
                    }
                    _ => false,
                };
            if dead {
                program.remove(index);
                rewrites += 1;
            } else {
                index += 1;
            }
        }
        rewrites
    }
}
