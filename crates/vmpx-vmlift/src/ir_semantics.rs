//! Reads a lowered (optionally optimized) straight-line recompile IR function
//! back into [`ExitSemantics`], so the recompiled output can be checked against
//! the recovered VM semantics by something other than the pipeline that built it.
//!
//! This is an evaluator for the IR the replacement lowering emits, not a VM
//! engine. Anything it cannot model exactly (branches, calls, machine-state
//! passthroughs, wide reads of narrow slots, loads that may alias an earlier
//! store) is an error or a recorded gap; it never guesses.

use std::collections::BTreeMap;

use vmpx_recompile::opt::{OptimizeOptions, Optimizer};
use vmpx_recompile::{
    BinaryOp as IrBinary, BinarySource, Condition, Function, Label, Operation, PhysicalRegister,
    ResizeOp as IrResize, ShiftCount, ShiftOp, UnaryOp as IrUnary, VirtualRegister, Width,
};
use vmpx_vm::{BinaryOp, BitWidth, Expr, ExprKind, NativeExitState, ResizeOp, UnaryOp};

use crate::replacement::{physical_for, rewrite_symbols};
use crate::{
    CollectedEffects, CollectedStore, ExitSemantics, LiftedCfgFunction, LoopInfo, PathTerminal,
    RecoveredPath, Replacement, ReplacementConfig, SegmentKey, build_replacement,
};

const ENTRY_RSP: &str = "__entry_rsp";
const ENTRY_FLAGS: &str = "native_rflags_initial";

fn bw(width: Width) -> BitWidth {
    match width {
        Width::W8 => BitWidth::W8,
        Width::W16 => BitWidth::W16,
        Width::W32 => BitWidth::W32,
        Width::W64 => BitWidth::W64,
    }
}

fn err(error: impl std::fmt::Display) -> String {
    error.to_string()
}

/// What the hardware flags hold, as far as the IR determines it.
#[derive(Clone)]
enum FlagState {
    /// A known flag word (the entry flags, or one just restored).
    Known(Expr),
    /// Defined by the last flag-setting instruction: the intrinsic that names
    /// its hardware flags, over its operands.
    Defined { name: String, args: Vec<Expr> },
    Unknown,
}

const STATUS_FLAG_MASK: u64 = 0x8D5;
const STATUS_FLAG_FIXED: u64 = 0x202;

/// A spill slot: the value stored at its store width, plus the narrower value
/// the source register last held when the store was of the full register.
#[derive(Clone)]
struct Slot {
    width: BitWidth,
    value: Expr,
    narrow: Option<(BitWidth, Expr)>,
}

#[derive(Clone)]
struct Reg {
    full: Expr,
    /// Value of the low `width` bits as last written, when narrower than full.
    narrow: Option<(BitWidth, Expr)>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Terminator {
    Return,
    ReturnPopping(u32),
    ReturnRetaining(u32),
    JumpTo(u64),
    /// Control reached a loop header: the carried values' seeds.
    LoopEntry { ordinal: usize, seeds: Vec<Expr> },
    /// The loop jumped back: the carried values for the next iteration.
    BackEdge { ordinal: usize, next: Vec<Expr> },
}

/// One compare-and-branch on an executed path: `left == right` held (or not).
#[derive(Clone, Debug)]
pub struct IrCondition {
    pub left: Expr,
    pub right: Expr,
    pub equal: bool,
}

/// One entry-to-terminator path through a (possibly branching) function.
#[derive(Clone, Debug)]
pub struct IrPath {
    pub conditions: Vec<IrCondition>,
    pub exit: IrExit,
}

/// What a straight-line IR function leaves behind.
#[derive(Clone, Debug)]
pub struct IrExit {
    pub registers: BTreeMap<PhysicalRegister, Expr>,
    pub rflags: Option<Expr>,
    pub stores: Vec<CollectedStore>,
    pub gaps: Vec<String>,
    /// A load follows a store it may alias: the loaded value is taken to be
    /// what the lifter's read-stability analysis intended.
    pub load_after_store: bool,
    pub terminator: Terminator,
}

#[derive(Clone)]
struct Machine<'a> {
    function: &'a Function,
    phys: BTreeMap<u32, PhysicalRegister>,
    regs: BTreeMap<PhysicalRegister, Reg>,
    slots: BTreeMap<u16, Slot>,
    stores: Vec<CollectedStore>,
    /// Stores before this index belong to an enclosing segment.
    store_base: usize,
    /// A load was issued after a store it is not provably disjoint from.
    load_after_store: bool,
    gaps: Vec<String>,
    rflags: Option<Expr>,
    flags: FlagState,
    compare: Option<(Expr, Expr)>,
    conditions: Vec<IrCondition>,
}

fn same_base_offset(expr: &Expr) -> (Expr, u64) {
    if let ExprKind::Binary {
        op: BinaryOp::Add,
        left,
        right,
    } = &expr.kind
        && let ExprKind::Const { value } = right.kind
    {
        return (Expr::clone(left), value);
    }
    (expr.clone(), 0)
}

fn provably_disjoint(a: &Expr, a_bytes: u64, b: &Expr, b_bytes: u64) -> bool {
    let ((base_a, off_a), (base_b, off_b)) = (same_base_offset(a), same_base_offset(b));
    if base_a != base_b {
        return false;
    }
    off_a.wrapping_add(a_bytes) <= off_b || off_b.wrapping_add(b_bytes) <= off_a
}

impl Machine<'_> {
    fn physical(&self, register: VirtualRegister) -> Result<PhysicalRegister, String> {
        self.phys
            .get(&register.id)
            .copied()
            .ok_or_else(|| format!("virtual register {} has no assignment", register.id))
    }

    fn read_as(&self, register: VirtualRegister, width: BitWidth) -> Result<Expr, String> {
        let p = self.physical(register)?;
        let reg = self
            .regs
            .get(&p)
            .ok_or_else(|| format!("{p:?} read before it holds a value"))?;
        if width == BitWidth::W64 {
            return Ok(reg.full.clone());
        }
        if let Some((nw, value)) = &reg.narrow {
            if *nw == width {
                return Ok(value.clone());
            }
            if nw.bits() > width.bits() {
                return Expr::resize(ResizeOp::Truncate, width, value.clone()).map_err(err);
            }
        }
        Expr::resize(ResizeOp::Truncate, width, reg.full.clone()).map_err(err)
    }

    fn read(&self, register: VirtualRegister) -> Result<Expr, String> {
        self.read_as(register, bw(register.width))
    }

    fn write(&mut self, register: VirtualRegister, value: Expr) -> Result<(), String> {
        let p = self.physical(register)?;
        let width = bw(register.width);
        if value.width != width {
            return Err(format!(
                "write of {:?} value into {:?} view",
                value.width, width
            ));
        }
        let state = match width {
            BitWidth::W64 => Reg {
                full: value,
                narrow: None,
            },
            BitWidth::W32 => Reg {
                full: Expr::resize(ResizeOp::ZeroExtend, BitWidth::W64, value.clone())
                    .map_err(err)?,
                narrow: Some((width, value)),
            },
            BitWidth::W16 | BitWidth::W8 => {
                let old = self
                    .regs
                    .get(&p)
                    .ok_or_else(|| format!("{p:?} partially written before it holds a value"))?
                    .full
                    .clone();
                let keep = Expr::binary(
                    BinaryOp::And,
                    old,
                    Expr::constant(BitWidth::W64, !(u64::MAX >> (64 - width.bits()))),
                )
                .map_err(err)?;
                let low = Expr::resize(ResizeOp::ZeroExtend, BitWidth::W64, value.clone())
                    .map_err(err)?;
                Reg {
                    full: Expr::binary(BinaryOp::Or, keep, low).map_err(err)?,
                    narrow: Some((width, value)),
                }
            }
        };
        self.regs.insert(p, state);
        Ok(())
    }

    fn source_value(
        &self,
        source: BinarySource,
        width: BitWidth,
    ) -> Result<Expr, String> {
        match source {
            BinarySource::Register(r) => self.read(r),
            BinarySource::Immediate(v) => Ok(Expr::constant(width, i64::from(v) as u64)),
        }
    }

    fn step(&mut self, operation: &Operation) -> Result<Option<Terminator>, String> {
        match operation {
            Operation::MoveImmediate { destination, value } => {
                self.write(*destination, Expr::constant(bw(destination.width), *value))?;
            }
            Operation::Move {
                destination,
                source,
            } => {
                let width = bw(destination.width);
                let v = self.read_as(*source, width)?;
                let narrow = if width == BitWidth::W64 && source.width == Width::W64 {
                    self.regs
                        .get(&self.physical(*source)?)
                        .and_then(|r| r.narrow.clone())
                } else {
                    None
                };
                self.write(*destination, v)?;
                self.keep_narrow(*destination, narrow)?;
            }
            Operation::Binary {
                operation,
                destination,
                source,
            } => {
                let w = bw(destination.width);
                let left = self.read(*destination)?;
                let right = self.source_value(*source, w)?;
                let kind = match operation {
                    IrBinary::Add => Some("add"),
                    IrBinary::And => Some("and"),
                    IrBinary::Or => Some("or"),
                    _ => None,
                };
                self.flags = kind.map_or(FlagState::Unknown, |kind| FlagState::Defined {
                    name: format!("hw_flags_{kind}_w{}", w.bits()),
                    args: vec![left.clone(), right.clone()],
                });
                let op = match operation {
                    IrBinary::Add => BinaryOp::Add,
                    IrBinary::Subtract => BinaryOp::Sub,
                    IrBinary::Multiply => BinaryOp::Mul,
                    IrBinary::Xor => BinaryOp::Xor,
                    IrBinary::And => BinaryOp::And,
                    IrBinary::Or => BinaryOp::Or,
                };
                self.write(*destination, Expr::binary(op, left, right).map_err(err)?)?;
            }
            Operation::Unary {
                operation,
                destination,
            } => {
                if *operation == IrUnary::Negate {
                    self.flags = FlagState::Unknown;
                }
                let v = self.read(*destination)?;
                let op = match operation {
                    IrUnary::Not => UnaryOp::Not,
                    IrUnary::Negate => UnaryOp::Neg,
                };
                self.write(*destination, Expr::unary(op, v))?;
            }
            Operation::Shift {
                operation,
                destination,
                count,
            } => {
                self.flags = FlagState::Unknown;
                let w = bw(destination.width);
                let value = self.read(*destination)?;
                let amount = match count {
                    ShiftCount::Immediate(n) => Expr::constant(w, u64::from(*n)),
                    ShiftCount::Register(r) => {
                        let c = self.read_as(*r, BitWidth::W8)?;
                        if w == BitWidth::W8 {
                            c
                        } else {
                            Expr::resize(ResizeOp::ZeroExtend, w, c).map_err(err)?
                        }
                    }
                };
                let op = match operation {
                    ShiftOp::Left => BinaryOp::Shl,
                    ShiftOp::RightLogical => BinaryOp::Shr,
                    ShiftOp::RightArithmetic => BinaryOp::Sar,
                    ShiftOp::RotateLeft => BinaryOp::Rol,
                    ShiftOp::RotateRight => BinaryOp::Ror,
                };
                self.write(*destination, Expr::binary(op, value, amount).map_err(err)?)?;
            }
            Operation::Resize {
                operation,
                destination,
                source,
            } => {
                let dw = bw(destination.width);
                let (op, sw) = match operation {
                    IrResize::Truncate => (ResizeOp::Truncate, dw),
                    IrResize::ZeroExtend => (ResizeOp::ZeroExtend, bw(source.width)),
                    IrResize::SignExtend => (ResizeOp::SignExtend, bw(source.width)),
                };
                let v = self.read_as(*source, sw)?;
                self.write(*destination, Expr::resize(op, dw, v).map_err(err)?)?;
            }
            Operation::Load {
                destination,
                address,
            } => {
                let a = self.address(address.base, address.displacement)?;
                let width = bw(destination.width);
                for store in &self.stores {
                    if !provably_disjoint(
                        &store.address,
                        u64::from(store.width.bytes()),
                        &a,
                        u64::from(width.bytes()),
                    ) {
                        self.load_after_store = true;
                        break;
                    }
                }
                self.write(*destination, Expr::load(width, a))?;
            }
            Operation::Store { address, source } => {
                let a = self.address(address.base, address.displacement)?;
                let width = bw(source.width);
                let value = self.read(*source)?;
                self.stores.push(CollectedStore {
                    address: a,
                    value,
                    width,
                });
            }
            Operation::SpillLoad { destination, slot } => {
                let width = bw(destination.width);
                let v = self.slot_value(*slot, width)?;
                let narrow = if width == BitWidth::W64 {
                    self.slots.get(&slot.0).and_then(|s| s.narrow.clone())
                } else {
                    None
                };
                self.write(*destination, v)?;
                self.keep_narrow(*destination, narrow)?;
            }
            Operation::SpillStore { slot, source } => {
                let v = self.read(*source)?;
                let narrow = if source.width == Width::W64 {
                    self.regs
                        .get(&self.physical(*source)?)
                        .and_then(|r| r.narrow.clone())
                } else {
                    None
                };
                self.slots.insert(
                    slot.0,
                    Slot {
                        width: v.width,
                        value: v,
                        narrow,
                    },
                );
            }
            Operation::EntryStackPointer { destination } => {
                self.write(*destination, Expr::register(BitWidth::W64, ENTRY_RSP))?;
            }
            Operation::ReadFlags { destination } => {
                let flags = match self.flags.clone() {
                    FlagState::Known(e) => e,
                    FlagState::Defined { name, args } => {
                        Expr::intrinsic(BitWidth::W64, name, args)
                    }
                    FlagState::Unknown => return Err("flags read after an unmodeled change".to_owned()),
                };
                self.write(*destination, flags)?;
            }
            Operation::RestoreFlags { value } => {
                let v = self.read_as(*value, BitWidth::W64)?;
                self.flags = FlagState::Known(v.clone());
                self.rflags = Some(v);
            }
            Operation::Cpuid {
                leaf,
                subleaf,
                out_eax,
                out_ebx,
                out_ecx,
                out_edx,
            } => {
                let args = vec![
                    self.slot_value(*leaf, BitWidth::W32)?,
                    self.slot_value(*subleaf, BitWidth::W32)?,
                ];
                for (slot, register) in [
                    (out_eax, "eax"),
                    (out_ebx, "ebx"),
                    (out_ecx, "ecx"),
                    (out_edx, "edx"),
                ] {
                    let value = Expr::intrinsic(
                        BitWidth::W32,
                        format!("x86_cpuid_{register}"),
                        args.clone(),
                    );
                    self.slots.insert(
                        slot.0,
                        Slot {
                            width: BitWidth::W32,
                            value,
                            narrow: None,
                        },
                    );
                }
            }
            Operation::GsLoad {
                destination,
                address,
            } => {
                let a = self.address(address.base, address.displacement)?;
                let value = Expr::intrinsic(bw(destination.width), "x86_gs_load", vec![a]);
                self.write(*destination, value)?;
            }
            Operation::Compare { left, right } => {
                let l = self.read(*left)?;
                let r = self.source_value(*right, l.width)?;
                self.flags = FlagState::Defined {
                    name: format!("hw_flags_sub_w{}", l.width.bits()),
                    args: vec![l.clone(), r.clone()],
                };
                self.compare = Some((l, r));
            }
            Operation::DoubleShift {
                direction,
                destination,
                source,
                count,
            } => {
                let w = bw(destination.width);
                let dest = self.read(*destination)?;
                let src = self.read(*source)?;
                let ShiftCount::Register(count) = count else {
                    return Err("immediate double-shift count not modeled".to_owned());
                };
                let amount = self.read_as(*count, BitWidth::W8)?;
                let side = match direction {
                    vmpx_recompile::DoubleShiftDirection::Left => "left",
                    vmpx_recompile::DoubleShiftDirection::Right => "right",
                };
                let args = vec![dest, src, amount];
                self.flags = FlagState::Defined {
                    name: format!("vm_flags_double_shift_{side}_w{}", w.bits()),
                    args: args.clone(),
                };
                let value =
                    Expr::intrinsic(w, format!("x86_double_shift_{side}_w{}", w.bits()), args);
                self.write(*destination, value)?;
            }
            Operation::Label(_) => {}
            Operation::Return => return Ok(Some(Terminator::Return)),
            Operation::ReturnPopping { bytes } => {
                return Ok(Some(Terminator::ReturnPopping(*bytes)));
            }
            Operation::ReturnRetaining { bytes } => {
                return Ok(Some(Terminator::ReturnRetaining(*bytes)));
            }
            Operation::JumpToAddress(target) => return Ok(Some(Terminator::JumpTo(target.0))),
            other => return Err(format!("operation not modeled: {other:?}")),
        }
        Ok(None)
    }

    /// A full-register copy keeps what the source knew about its low bits.
    fn keep_narrow(
        &mut self,
        destination: VirtualRegister,
        narrow: Option<(BitWidth, Expr)>,
    ) -> Result<(), String> {
        if let Some(narrow) = narrow {
            let p = self.physical(destination)?;
            if let Some(reg) = self.regs.get_mut(&p) {
                reg.narrow = Some(narrow);
            }
        }
        Ok(())
    }

    fn slot_value(
        &self,
        slot: vmpx_recompile::SpillSlot,
        width: BitWidth,
    ) -> Result<Expr, String> {
        let slot_state = self
            .slots
            .get(&slot.0)
            .cloned()
            .ok_or_else(|| format!("slot {} read before written", slot.0))?;
        if let Some((nw, ne)) = &slot_state.narrow
            && nw.bits() >= width.bits()
            && width != slot_state.width
        {
            return if *nw == width {
                Ok(ne.clone())
            } else {
                Expr::resize(ResizeOp::Truncate, width, ne.clone()).map_err(err)
            };
        }
        let (stored, value) = (slot_state.width, slot_state.value);
        if stored == width {
            Ok(value)
        } else if stored.bits() > width.bits() {
            Expr::resize(ResizeOp::Truncate, width, value).map_err(err)
        } else {
            Err(format!("slot {} read wider than it was written", slot.0))
        }
    }

    fn address(&self, base: VirtualRegister, displacement: i32) -> Result<Expr, String> {
        let b = self.read_as(base, BitWidth::W64)?;
        if displacement == 0 {
            return Ok(b);
        }
        Expr::binary(
            BinaryOp::Add,
            b,
            Expr::constant(BitWidth::W64, i64::from(displacement) as u64),
        )
        .map_err(err)
    }
}

const MAX_PATHS: usize = 4_096;

fn label_positions(function: &Function) -> BTreeMap<Label, usize> {
    function
        .operations
        .iter()
        .enumerate()
        .filter_map(|(i, op)| match op {
            Operation::Label(l) => Some((*l, i)),
            _ => None,
        })
        .collect()
}

struct Walk<'a> {
    labels: BTreeMap<Label, usize>,
    /// Loop header label to loop ordinal.
    headers: BTreeMap<Label, usize>,
    loops: &'a [LoopInfo],
    initial: &'a BTreeMap<PhysicalRegister, Expr>,
    out: BTreeMap<SegmentKey, Vec<IrPath>>,
    pending: Vec<(SegmentKey, Machine<'a>, usize)>,
    started: std::collections::BTreeSet<usize>,
    total: usize,
}

impl<'a> Walk<'a> {
    fn jump(&self, from: usize, label: &Label) -> Result<usize, String> {
        let to = *self.labels.get(label).ok_or("branch to an unknown label")?;
        if to <= from {
            return Err("backward branch that is not a recognised loop back-edge".to_owned());
        }
        Ok(to)
    }

    fn phi_values(&self, machine: &Machine<'_>, ordinal: usize) -> Result<Vec<Expr>, String> {
        self.loops[ordinal]
            .phis
            .iter()
            .map(|phi| {
                machine.slot_value(phi.slot, phi.width).map_err(|e| {
                    format!("loop {ordinal} carried slot {}: {e}", phi.slot.0)
                })
            })
            .collect()
    }

    fn push(
        &mut self,
        segment: SegmentKey,
        machine: Machine<'a>,
        terminator: Terminator,
    ) -> Result<(), String> {
        self.total += 1;
        if self.total > MAX_PATHS {
            return Err("too many paths".to_owned());
        }
        let path = finish(machine, terminator, self.initial);
        self.out.entry(segment).or_default().push(path);
        Ok(())
    }

    fn run(
        &mut self,
        segment: SegmentKey,
        mut machine: Machine<'a>,
        mut pc: usize,
    ) -> Result<(), String> {
        let function = machine.function;
        while let Some(operation) = function.operations.get(pc) {
            match operation {
                Operation::Label(label) => {
                    if let Some(&ordinal) = self.headers.get(label) {
                        // First arrival at a loop: end this segment with the
                        // seeds, and analyse one iteration from the header with
                        // the carried slots as free symbols.
                        let seeds = self.phi_values(&machine, ordinal)?;
                        let mut body = machine.clone();
                        for phi in &self.loops[ordinal].phis {
                            body.slots.insert(
                                phi.slot.0,
                                Slot {
                                    width: phi.width,
                                    value: Expr::register(phi.width, phi.name.clone()),
                                    narrow: None,
                                },
                            );
                        }
                        body.conditions.clear();
                        body.store_base = body.stores.len();
                        if !self.started.insert(ordinal) {
                            return Err(format!("loop {ordinal} is entered twice"));
                        }
                        self.pending.push((SegmentKey::Loop(ordinal), body, pc + 1));
                        return self.push(segment, machine, Terminator::LoopEntry { ordinal, seeds });
                    }
                }
                Operation::BranchIf { condition, target } => {
                    let taken_means_equal = match condition {
                        Condition::Equal => true,
                        Condition::NotEqual => false,
                        other => return Err(format!("branch condition {other:?} not modeled")),
                    };
                    let (left, right) = machine
                        .compare
                        .clone()
                        .ok_or("branch without a preceding compare")?;
                    let mut taken = machine.clone();
                    taken.conditions.push(IrCondition {
                        left: left.clone(),
                        right: right.clone(),
                        equal: taken_means_equal,
                    });
                    let to = self.jump(pc, target)?;
                    self.run(segment, taken, to)?;
                    machine.conditions.push(IrCondition {
                        left,
                        right,
                        equal: !taken_means_equal,
                    });
                }
                Operation::Jump(target) => {
                    if let Some(&ordinal) = self.headers.get(target) {
                        let next = self.phi_values(&machine, ordinal)?;
                        return self.push(segment, machine, Terminator::BackEdge { ordinal, next });
                    }
                    pc = self.jump(pc, target)?;
                    continue;
                }
                other => {
                    if let Some(terminator) = machine.step(other)? {
                        return self.push(segment, machine, terminator);
                    }
                }
            }
            pc += 1;
        }
        Err("function falls off the end without a terminator".to_owned())
    }
}

fn finish(
    machine: Machine<'_>,
    terminator: Terminator,
    initial: &BTreeMap<PhysicalRegister, Expr>,
) -> IrPath {
    let mut registers: BTreeMap<PhysicalRegister, Expr> = machine
        .regs
        .iter()
        .map(|(p, r)| (*p, r.full.clone()))
        .collect();
    if terminator == Terminator::Return && !machine.function.frame.tail_transfer {
        for p in &machine.function.frame.preserve {
            if let Some(entry) = initial.get(p) {
                registers.insert(*p, entry.clone());
            }
        }
    }
    IrPath {
        conditions: machine.conditions,
        exit: IrExit {
            registers,
            rflags: machine.rflags,
            stores: machine.stores[machine.store_base..].to_vec(),
            gaps: machine.gaps,
            load_after_store: machine.load_after_store,
            terminator,
        },
    }
}

/// Evaluates every path of an IR function from the given physical-register
/// entry values, segment by segment: the code from the entry, then one
/// iteration of each loop from its header with the carried slots as free
/// symbols. Branches fork taken-first, so paths come out in the order a
/// taken-first walk of the code visits them.
///
/// # Errors
/// Any operation outside the modeled subset (other branch conditions, calls,
/// machine-state passthroughs), a read of an unwritten slot or register, or a
/// body that never terminates.
pub fn evaluate_ir_segments(
    function: &Function,
    initial: &BTreeMap<PhysicalRegister, Expr>,
    loops: &[LoopInfo],
) -> Result<BTreeMap<SegmentKey, Vec<IrPath>>, String> {
    let phys = function
        .assignments
        .iter()
        .map(|a| (a.virtual_register.id, a.physical_register))
        .collect();
    let regs = initial
        .iter()
        .map(|(p, e)| {
            (
                *p,
                Reg {
                    full: e.clone(),
                    narrow: None,
                },
            )
        })
        .collect();
    let machine = Machine {
        function,
        phys,
        regs,
        slots: BTreeMap::new(),
        stores: Vec::new(),
        store_base: 0,
        load_after_store: false,
        gaps: Vec::new(),
        rflags: None,
        flags: FlagState::Known(Expr::register(BitWidth::W64, ENTRY_FLAGS)),
        compare: None,
        conditions: Vec::new(),
    };
    let mut walk = Walk {
        labels: label_positions(function),
        headers: loops
            .iter()
            .enumerate()
            .map(|(i, l)| (l.header, i))
            .collect(),
        loops,
        initial,
        out: BTreeMap::new(),
        pending: Vec::new(),
        started: std::collections::BTreeSet::new(),
        total: 0,
    };
    walk.run(SegmentKey::Root, machine, 0)?;
    while let Some((segment, machine, pc)) = walk.pending.pop() {
        walk.run(segment, machine, pc)?;
    }
    Ok(walk.out)
}

/// Evaluates a loop-free function.
///
/// # Errors
/// As [`evaluate_ir_segments`].
pub fn evaluate_ir_paths(
    function: &Function,
    initial: &BTreeMap<PhysicalRegister, Expr>,
) -> Result<Vec<IrPath>, String> {
    let mut segments = evaluate_ir_segments(function, initial, &[])?;
    Ok(segments.remove(&SegmentKey::Root).unwrap_or_default())
}

/// Evaluates a function that must have exactly one path.
///
/// # Errors
/// As [`evaluate_ir_paths`], or if the function branches.
pub fn evaluate_ir(
    function: &Function,
    initial: &BTreeMap<PhysicalRegister, Expr>,
) -> Result<IrExit, String> {
    let mut paths = evaluate_ir_paths(function, initial)?;
    if paths.len() != 1 {
        return Err(format!("{} paths, exactly one required", paths.len()));
    }
    Ok(paths.remove(0).exit)
}

/// Operations whose effects are externally visible: the optimizer never edits,
/// removes, or steps over them.
fn volatile_indices(function: &Function) -> std::collections::BTreeSet<usize> {
    function
        .operations
        .iter()
        .enumerate()
        .filter(|(_, op)| {
            matches!(
                op,
                Operation::Store { .. }
                    | Operation::Load { .. }
                    | Operation::GsLoad { .. }
                    | Operation::Call { .. }
                    | Operation::RestoreFlags { .. }
                    | Operation::Cpuid { .. }
                    | Operation::MultiplyFlags { .. }
                    | Operation::UnsignedDivide { .. }
            )
        })
        .map(|(i, _)| i)
        .collect()
}

/// One path's recovered exit and the same exit read back from the recompiled
/// output, in one symbol universe, ready for `Obligation::from_exits`.
#[derive(Debug)]
pub struct RecompiledPair {
    pub vm: ExitSemantics,
    pub recompiled: ExitSemantics,
    /// The recompiled output derives the return/continuation address itself
    /// (a tail jump); a return leaves it unattested.
    pub return_address_attested: bool,
    /// The recovered path restores flags, so the flags cell is comparable.
    pub rflags_attested: bool,
    /// The pair relies on loads after stores denoting the lifter's intended
    /// memory values (its read-stability analysis), which is not re-proven.
    pub memory_order_assumed: bool,
    /// Extra equalities the pairing relies on (branch selectors, branch
    /// constants, popped bytes): `(name, recovered side, recompiled side)`.
    pub extra_cells: Vec<(String, Expr, Expr)>,
}

/// Size and rewrite statistics for one recompiled function.
#[derive(Debug, Default)]
pub struct RecompileStats {
    pub operations_before: usize,
    pub operations_after: usize,
    pub rewrites: usize,
}

#[derive(Debug)]
pub struct Readback {
    pub pairs: Vec<RecompiledPair>,
    pub stats: RecompileStats,
}

/// The recovered side numbers each machine-state read (`x86_cpuid_eax_12`);
/// the recompiled IR has no such serial, and equal arguments read equal state,
/// so the serial is dropped on the recovered side.
fn recovered_expr(e: &Expr, rewrites: &BTreeMap<String, Expr>) -> Result<Expr, String> {
    let rewritten = rewrite_symbols(e, rewrites).map_err(err)?;
    rewritten
        .rewrite_intrinsics(&mut |width, name, arguments| {
            if let Some(word) = vm_flag_word(name, arguments) {
                return Some(word);
            }
            if let Some(product) = packed_multiply(name, width, arguments) {
                return Some(product);
            }
            let (stem, serial) = name.rsplit_once('_')?;
            let machine_read = stem.starts_with("x86_cpuid_")
                || stem == "x86_gs_load"
                || stem.starts_with("x86_double_shift_")
                || stem.starts_with("vm_flags_double_shift_");
            (machine_read && !serial.is_empty() && serial.bytes().all(|b| b.is_ascii_digit()))
                .then(|| Expr::intrinsic(width, stem, arguments.to_vec()))
        })
        .map_err(err)
}

type Memo = std::collections::HashMap<Expr, Expr>;

/// `Resize` of a `Resize`: collapses chains of the same direction and a
/// truncation of an extension, all identities.
fn collapse_resize(op: ResizeOp, width: BitWidth, inner: Expr) -> Result<Expr, String> {
    if let ExprKind::Resize {
        op: inner_op,
        value: x,
    } = &inner.kind
    {
        let x: &Expr = x;
        let same_direction = matches!(
            (op, *inner_op),
            (ResizeOp::ZeroExtend, ResizeOp::ZeroExtend)
                | (ResizeOp::SignExtend, ResizeOp::SignExtend)
                | (ResizeOp::Truncate, ResizeOp::Truncate)
        );
        if same_direction {
            return collapse_resize(op, width, x.clone());
        }
        let extension = matches!(*inner_op, ResizeOp::ZeroExtend | ResizeOp::SignExtend);
        if op == ResizeOp::Truncate && extension {
            if width.bits() <= x.width.bits() {
                return collapse_resize(ResizeOp::Truncate, width, x.clone());
            }
            return collapse_resize(*inner_op, width, x.clone());
        }
        if op == ResizeOp::ZeroExtend && *inner_op == ResizeOp::SignExtend {
            // not an identity; leave as is
        }
    }
    if inner.width == width {
        return Ok(inner);
    }
    Expr::resize(op, width, inner).map_err(err)
}

/// Rewrites width-slicing forms to one normal form so the two sides compare
/// structurally: `Concat(0, x)` is a zero extension, `Extract(x, 0)` a
/// truncation, `Extract(x, k)` a truncated logical shift, and a general
/// `Concat(h, l)` the or of `l` and `h` shifted up. All are identities.
fn normalize_with(e: &Expr, memo: &mut Memo) -> Result<Expr, String> {
    if let Some(done) = memo.get(e) {
        return Ok(done.clone());
    }
    let mut n = |x: &Expr| normalize_with(x, memo);
    let out = match &e.kind {
        ExprKind::Const { .. } | ExprKind::Register { .. } => e.clone(),
        ExprKind::Unary { op, value } => Expr::unary(*op, n(value)?),
        ExprKind::Binary { op, left, right } => {
            Expr::binary(*op, n(left)?, n(right)?).map_err(err)?
        }
        ExprKind::Load { address } => Expr::load(e.width, n(address)?),
        ExprKind::Intrinsic { name, arguments } => Expr::intrinsic(
            e.width,
            name.clone(),
            arguments.iter().map(&mut n).collect::<Result<_, _>>()?,
        ),
        ExprKind::Resize { op, value } => collapse_resize(*op, e.width, n(value)?)?,
        ExprKind::Extract { value, lsb } => {
            let v = n(value)?;
            let shifted = if *lsb == 0 {
                v
            } else {
                let count = Expr::constant(v.width, u64::from(*lsb));
                Expr::binary(BinaryOp::Shr, v, count).map_err(err)?
            };
            collapse_resize(ResizeOp::Truncate, e.width, shifted)?
        }
        ExprKind::Concat { high, low } => {
            let (h, l) = (n(high)?, n(low)?);
            let low_wide = collapse_resize(ResizeOp::ZeroExtend, e.width, l.clone())?;
            if matches!(h.kind, ExprKind::Const { value: 0 }) {
                low_wide
            } else {
                let high_wide = collapse_resize(ResizeOp::ZeroExtend, e.width, h)?;
                let up = Expr::binary(
                    BinaryOp::Shl,
                    high_wide,
                    Expr::constant(e.width, u64::from(l.width.bits())),
                )
                .map_err(err)?;
                Expr::binary(BinaryOp::Or, low_wide, up).map_err(err)?
            }
        }
    };
    memo.insert(e.clone(), out.clone());
    Ok(out)
}

/// Public so callers can state the normal form they compare in.
///
/// # Errors
/// A rebuilt node has an invalid width (cannot happen for well-formed input).
pub fn normalize_expr(e: &Expr) -> Result<Expr, String> {
    normalize_with(e, &mut Memo::new())
}

fn normalize_semantics(s: &mut ExitSemantics, memo: &mut Memo) -> Result<(), String> {
    for v in s.exit.registers.values_mut() {
        *v = normalize_with(v, memo)?;
    }
    s.exit.rflags = normalize_with(&s.exit.rflags, memo)?;
    for store in &mut s.effects.stores {
        store.address = normalize_with(&store.address, memo)?;
        store.value = normalize_with(&store.value, memo)?;
    }
    Ok(())
}

fn normalize_pair(pair: &mut RecompiledPair) -> Result<(), String> {
    let mut memo = Memo::new();
    normalize_semantics(&mut pair.vm, &mut memo)?;
    normalize_semantics(&mut pair.recompiled, &mut memo)?;
    for (_, a, b) in &mut pair.extra_cells {
        *a = normalize_with(a, &mut memo)?;
        *b = normalize_with(b, &mut memo)?;
    }
    Ok(())
}

/// `x86_{signed,unsigned}_multiply_wN_K(a, b)` for N <= 32 is the packed
/// double-width product in a qword: both operands widened to 64 bits (sign- or
/// zero-extended) and multiplied, which is how the lowering defines it.
fn packed_multiply(name: &str, width: BitWidth, arguments: &[Expr]) -> Option<Expr> {
    let (signed, rest) = if let Some(rest) = name.strip_prefix("x86_signed_multiply_w") {
        (true, rest)
    } else {
        (false, name.strip_prefix("x86_unsigned_multiply_w")?)
    };
    let bits: u32 = rest.split('_').next()?.parse().ok()?;
    let [a, b] = arguments else {
        return None;
    };
    if bits > 32 || width != BitWidth::W64 || a.width.bits() != bits || b.width.bits() != bits {
        return None;
    }
    let op = if signed {
        ResizeOp::SignExtend
    } else {
        ResizeOp::ZeroExtend
    };
    let widen = |x: &Expr| Expr::resize(op, BitWidth::W64, x.clone()).ok();
    Expr::binary(BinaryOp::Mul, widen(a)?, widen(b)?).ok()
}

/// A VM status-flag word as the native lowering defines it: the hardware flags
/// of the flag-defining instruction, masked and with the fixed bits set.
fn vm_flag_word(name: &str, arguments: &[Expr]) -> Option<Expr> {
    let (family, bits) = name.strip_prefix("vm_flags_")?.rsplit_once("_w")?;
    let bits: u32 = bits.parse().ok()?;
    let [a, b] = arguments else {
        return None;
    };
    let (kind, left, right) = match family {
        "add" => ("add", a.clone(), b.clone()),
        "sub" => ("sub", a.clone(), b.clone()),
        "nor" => ("and", Expr::unary(UnaryOp::Not, a.clone()), Expr::unary(UnaryOp::Not, b.clone())),
        "nand" => ("or", Expr::unary(UnaryOp::Not, a.clone()), Expr::unary(UnaryOp::Not, b.clone())),
        _ => return None,
    };
    let raw = Expr::intrinsic(
        BitWidth::W64,
        format!("hw_flags_{kind}_w{bits}"),
        vec![left, right],
    );
    let masked = Expr::binary(BinaryOp::And, raw, Expr::constant(BitWidth::W64, STATUS_FLAG_MASK)).ok()?;
    Expr::binary(BinaryOp::Or, masked, Expr::constant(BitWidth::W64, STATUS_FLAG_FIXED)).ok()
}

fn rewrite_exit(
    semantics: &ExitSemantics,
    rewrites: &BTreeMap<String, Expr>,
) -> Result<ExitSemantics, String> {
    let r = |e: &Expr| recovered_expr(e, rewrites);
    Ok(ExitSemantics {
        exit: NativeExitState {
            xmm: std::collections::BTreeMap::new(),
            schema_version: semantics.exit.schema_version,
            registers: semantics
                .exit
                .registers
                .iter()
                .map(|(k, v)| Ok((*k, r(v)?)))
                .collect::<Result<_, String>>()?,
            rflags: r(&semantics.exit.rflags)?,
            return_address: semantics.exit.return_address.clone(),
            frame_size: semantics.exit.frame_size,
        },
        effects: CollectedEffects {
            stores: semantics
                .effects
                .stores
                .iter()
                .map(|s| {
                    Ok(CollectedStore {
                        address: r(&s.address)?,
                        value: r(&s.value)?,
                        width: s.width,
                    })
                })
                .collect::<Result<_, String>>()?,
            calls: semantics.effects.calls.clone(),
            unsupported: semantics.effects.unsupported.clone(),
        },
    })
}

/// Optionally runs the standard optimizer, then reads every segment back.
fn recompile_segments(
    function: &Function,
    inputs: &[(String, PhysicalRegister)],
    loops: &[LoopInfo],
    optimize: bool,
) -> Result<(BTreeMap<SegmentKey, Vec<IrPath>>, RecompileStats), String> {
    let before = function.operations.len();
    let (function, rewrites) = if optimize {
        let options = OptimizeOptions {
            volatile: volatile_indices(function),
            ..OptimizeOptions::default()
        };
        let (optimized, report) = Optimizer::standard().run(function, &options);
        if let Some(reason) = report.input_error {
            return Err(reason);
        }
        (optimized, report.total_rewrites())
    } else {
        (function.clone(), 0)
    };
    let mut initial: BTreeMap<PhysicalRegister, Expr> = PHYSICAL
        .iter()
        .map(|p| (*p, Expr::register(BitWidth::W64, format!("__init_{p:?}"))))
        .collect();
    let mut seen = std::collections::BTreeSet::new();
    for (name, p) in inputs {
        if seen.insert(*p) {
            initial.insert(*p, Expr::register(BitWidth::W64, name.clone()));
        }
    }
    let stats = RecompileStats {
        operations_before: before,
        operations_after: function.operations.len(),
        rewrites,
    };
    Ok((evaluate_ir_segments(&function, &initial, loops)?, stats))
}

fn recompiled_semantics(vm: &ExitSemantics, exit: &IrExit) -> Result<ExitSemantics, String> {
    let mut registers = BTreeMap::new();
    for register in vm.exit.registers.keys() {
        let value = exit
            .registers
            .get(&physical_for(register))
            .ok_or_else(|| format!("{register:?} missing from recompiled exit"))?;
        registers.insert(*register, value.clone());
    }
    let return_address = match exit.terminator {
        Terminator::JumpTo(target) => Expr::constant(BitWidth::W64, target),
        _ => vm.exit.return_address.clone(),
    };
    Ok(ExitSemantics {
        exit: NativeExitState {
            xmm: std::collections::BTreeMap::new(),
            schema_version: vm.exit.schema_version,
            registers,
            rflags: exit
                .rflags
                .clone()
                .unwrap_or_else(|| vm.exit.rflags.clone()),
            return_address,
            frame_size: vm.exit.frame_size,
        },
        effects: CollectedEffects {
            stores: exit.stores.clone(),
            calls: Vec::new(),
            unsupported: exit.gaps.clone(),
        },
    })
}

/// Builds the replacement for `vm` with `config`, optionally optimizes it, and
/// reads it back as semantics. `config` must already carry `vm`'s stores in
/// `effects` and its flags in `rflags`, exactly as the production build does.
///
/// # Errors
/// The replacement cannot be built, or its IR is outside the modeled subset.
pub fn recompiled_pair(
    vm: &ExitSemantics,
    config: &ReplacementConfig,
    optimize: bool,
) -> Result<Readback, String> {
    let replacement: Replacement = build_replacement(
        &vm.exit.registers,
        config,
        vmpx_recompile::CodeAddress(0x1_0000_0000),
    )
    .map_err(err)?;
    let (mut segments, stats) =
        recompile_segments(&replacement.function, &config.inputs, &[], optimize)?;
    let mut paths = segments.remove(&SegmentKey::Root).unwrap_or_default();
    if paths.len() != 1 {
        return Err(format!("{} paths, exactly one required", paths.len()));
    }
    let exit = paths.remove(0).exit;
    let recompiled = recompiled_semantics(vm, &exit)?;
    let mut pair = RecompiledPair {
        vm: rewrite_exit(vm, &config.symbol_rewrites)?,
        recompiled,
        return_address_attested: matches!(exit.terminator, Terminator::JumpTo(_)),
        rflags_attested: exit.rflags.is_some(),
        memory_order_assumed: exit.load_after_store,
        extra_cells: Vec::new(),
    };
    normalize_pair(&mut pair)?;
    Ok(Readback {
        pairs: vec![pair],
        stats,
    })
}

fn effects_only(stores: Vec<CollectedStore>, gaps: Vec<String>) -> ExitSemantics {
    ExitSemantics {
        exit: NativeExitState {
            xmm: std::collections::BTreeMap::new(),
            schema_version: 1,
            registers: BTreeMap::new(),
            rflags: Expr::constant(BitWidth::W64, 0),
            return_address: Expr::constant(BitWidth::W64, 0),
            frame_size: 0,
        },
        effects: CollectedEffects {
            stores,
            calls: Vec::new(),
            unsupported: gaps,
        },
    }
}

fn pair_path(
    label: &str,
    tree: &RecoveredPath,
    ir: &IrPath,
    rewrites: &BTreeMap<String, Expr>,
) -> Result<RecompiledPair, String> {
    if tree.conditions.len() != ir.conditions.len()
        || tree
            .conditions
            .iter()
            .zip(&ir.conditions)
            .any(|(a, b)| a.equal != b.equal)
    {
        return Err(format!("{label}: branch structure differs"));
    }
    let r = |e: &Expr| recovered_expr(e, rewrites);
    let mut extra = Vec::new();
    for (k, (a, b)) in tree.conditions.iter().zip(&ir.conditions).enumerate() {
        extra.push((
            format!("{label} selector {k}"),
            r(&a.selector)?,
            b.left.clone(),
        ));
        extra.push((
            format!("{label} branch value {k}"),
            Expr::constant(BitWidth::W64, a.value),
            b.right.clone(),
        ));
    }
    let constant = |v: u64| Expr::constant(BitWidth::W64, v);
    match (&tree.terminal, &ir.exit.terminator) {
        (
            PathTerminal::Exit {
                semantics,
                rflags_clobbered,
                return_pop,
                retains,
            },
            Terminator::ReturnPopping(bytes) | Terminator::ReturnRetaining(bytes),
        ) if *retains == matches!(ir.exit.terminator, Terminator::ReturnRetaining(_)) => {
            extra.push((
                format!("{label} popped bytes"),
                constant(u64::from(*return_pop)),
                constant(u64::from(*bytes)),
            ));
            Ok(RecompiledPair {
                vm: rewrite_exit(semantics, rewrites)?,
                recompiled: recompiled_semantics(semantics, &ir.exit)?,
                return_address_attested: false,
                rflags_attested: !rflags_clobbered && ir.exit.rflags.is_some(),
                memory_order_assumed: ir.exit.load_after_store,
                extra_cells: extra,
            })
        }
        (
            PathTerminal::LoopEntry { ordinal, seeds },
            Terminator::LoopEntry {
                ordinal: ir_ordinal,
                seeds: ir_seeds,
            },
        )
        | (
            PathTerminal::BackEdge {
                ordinal,
                next: seeds,
            },
            Terminator::BackEdge {
                ordinal: ir_ordinal,
                next: ir_seeds,
            },
        ) => {
            if ordinal != ir_ordinal || seeds.len() != ir_seeds.len() {
                return Err(format!("{label}: loop carried values differ in shape"));
            }
            let kind = if matches!(tree.terminal, PathTerminal::LoopEntry { .. }) {
                "seed"
            } else {
                "next"
            };
            for (k, (a, b)) in seeds.iter().zip(ir_seeds).enumerate() {
                extra.push((format!("{label} loop {ordinal} {kind} {k}"), r(a)?, b.clone()));
            }
            Ok(RecompiledPair {
                vm: rewrite_exit(&effects_only(tree.stores.clone(), Vec::new()), rewrites)?,
                recompiled: effects_only(ir.exit.stores.clone(), ir.exit.gaps.clone()),
                return_address_attested: false,
                rflags_attested: false,
                memory_order_assumed: ir.exit.load_after_store,
                extra_cells: extra,
            })
        }
        _ => Err(format!("{label}: terminators differ in kind")),
    }
}

/// Reads a lifted CFG function's emitted IR back, segment by segment, against
/// the recovered decision tree. Paths are paired by emission order and checked:
/// the branch selectors, branch constants and popped bytes of every pair are
/// cells of the obligation, so a mis-paired or restructured path cannot pass.
/// A loop is checked inductively: its seeds, and one iteration from the header
/// with the carried values as free symbols (exit conditions, exit states,
/// stores, and the next carried values).
///
/// # Errors
/// The tree could not be enumerated, the IR is outside the modeled subset, or
/// the IR's path structure does not match the tree's.
pub fn recompiled_cfg_pairs(
    lifted: &LiftedCfgFunction,
    optimize: bool,
    strict_memory: bool,
) -> Result<Readback, String> {
    let recovered = lifted.recovered_paths.as_ref().map_err(Clone::clone)?;
    let (segments, stats) = recompile_segments(
        &lifted.replacement.function,
        &lifted.config.inputs,
        &lifted.emitted_loops,
        optimize,
    )?;
    let mut tree: BTreeMap<SegmentKey, Vec<&RecoveredPath>> = BTreeMap::new();
    for path in recovered {
        tree.entry(path.segment).or_default().push(path);
    }
    if tree.keys().ne(segments.keys()) {
        return Err(format!(
            "tree segments {:?} differ from recompiled segments {:?}",
            tree.keys().collect::<Vec<_>>(),
            segments.keys().collect::<Vec<_>>()
        ));
    }
    let rewrites = &lifted.config.symbol_rewrites;
    let mut pairs = Vec::new();
    for (key, tree_paths) in &tree {
        let ir_paths = &segments[key];
        if tree_paths.len() != ir_paths.len() {
            return Err(format!(
                "{key:?}: tree has {} paths but the recompiled IR has {}",
                tree_paths.len(),
                ir_paths.len()
            ));
        }
        for (index, (t, i)) in tree_paths.iter().zip(ir_paths).enumerate() {
            let mut pair = pair_path(&format!("{key:?} path {index}"), t, i, rewrites)?;
            normalize_pair(&mut pair)?;
            if strict_memory && pair.memory_order_assumed {
                pair.recompiled
                    .effects
                    .unsupported
                    .push("a load follows a store it may alias".to_owned());
            }
            pairs.push(pair);
        }
    }
    Ok(Readback { pairs, stats })
}

const PHYSICAL: [PhysicalRegister; 15] = [
    PhysicalRegister::Rax,
    PhysicalRegister::Rcx,
    PhysicalRegister::Rdx,
    PhysicalRegister::Rbx,
    PhysicalRegister::Rbp,
    PhysicalRegister::Rsi,
    PhysicalRegister::Rdi,
    PhysicalRegister::R8,
    PhysicalRegister::R9,
    PhysicalRegister::R10,
    PhysicalRegister::R11,
    PhysicalRegister::R12,
    PhysicalRegister::R13,
    PhysicalRegister::R14,
    PhysicalRegister::R15,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slicing_forms_normalize_to_one_shape() {
        let load = Expr::load(BitWidth::W16, Expr::register(BitWidth::W64, "p"));
        let wide = Expr::resize(ResizeOp::ZeroExtend, BitWidth::W64, load.clone()).unwrap();
        let narrow = Expr::resize(ResizeOp::Truncate, BitWidth::W16, wide).unwrap();
        assert_eq!(normalize_expr(&narrow).unwrap(), load);
        let sliced = Expr::extract(BitWidth::W16, 0, Expr::resize(
            ResizeOp::ZeroExtend, BitWidth::W64, load.clone()).unwrap()).unwrap();
        let once = normalize_expr(&sliced).unwrap();
        assert_eq!(once, load);
        assert_eq!(normalize_expr(&once).unwrap(), once);

        let x = Expr::register(BitWidth::W64, "x");
        let low = Expr::extract(BitWidth::W32, 0, x.clone()).unwrap();
        let zero = Expr::constant(BitWidth::W32, 0);
        let concat = Expr::concat(zero, low).unwrap();
        let direct = Expr::resize(
            ResizeOp::ZeroExtend,
            BitWidth::W64,
            Expr::resize(ResizeOp::Truncate, BitWidth::W32, x).unwrap(),
        )
        .unwrap();
        assert_eq!(
            normalize_expr(&concat).unwrap(),
            normalize_expr(&direct).unwrap()
        );
    }
}
