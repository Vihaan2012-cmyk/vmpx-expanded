//! Lift of a branching virtualized function from its recovered CFG.
//!
//! The CFG is expanded into a tree of paths: each block is symbolically
//! executed from the exact state its predecessor left, so a join is handled by
//! re-expanding the successor under each incoming state rather than by merging
//! states. A conditional VM branch becomes a native compare-and-jump on the
//! symbolic branch target; a block ending in a native call becomes a real call
//! through the symbolic `[vsp]` target. Everything else must be proven or the
//! lift refuses.
//!
//! A branch is lowered only when its targets resolve over a proven domain and
//! every target is a recovered CFG edge. A back edge, an unproven edge, an
//! unlowerable condition or a missing exit report refuses the whole function.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use vmpx_pe::Va;
use vmpx_recompile::{
    Address, BinarySource, CallTarget, CodeAddress, Condition, Function, Label, Operation,
    PhysicalRegister, RegisterAssignment, SpillSlot, StackFrame, VirtualRegister, Width, compile,
};
use vmpx_vm::{
    BinaryOp, BitWidth, BlockExit, CfgEdgeKind, Expr, ExprKind, NativeCallMemoryDecision, ResizeOp,
    NativeCallSelectorOutcome, NativeExitState, NativeFrameSource, SymbolicExecutionLimits,
    SymbolicMemorySource, SymbolicVmExecutionReport, SymbolicVmInitialState, VmCfg, VmCfgAssumption,
    VmCfgLimits, VmExitReport, VmHandlerSemantic, X64NativeRegister, canonicalize_symbolic_pointer,
    execute_symbolic_linear_block_with_initial_state,
    execute_symbolic_native_call_boundary_with_initial_state, recover_native_exit_state,
    symbolic_pointer_offset,
};

use crate::decoded_lift::{
    DecodedLiftError, LiftAssumption, RecoveredExit, first_unliftable, recover_exit,
};
use crate::entry_frame::frame_initial_state;
use crate::nested_entry::{NestedEntry, NestedEntryResolver};
use crate::replacement::{
    ENTRY_STACK_POINTER, Trust, canonical_register, lower_fragment_with, operation_slots,
    physical_for, rewrite_symbols,
};
use crate::{EffectStore, ExpressionInput, Replacement, ReplacementConfig, ReplacementError};

#[cfg(test)]
mod tests;

const MAX_CALL_ARGUMENTS: usize = 4;
const CALL_SCRATCH_SLOTS: u16 = 5;

pub struct DecodedCfgInput<'a> {
    pub cfg: &'a VmCfg,
    /// Initial VM stack, top first.
    pub frame: &'a [NativeFrameSource],
    /// VSP-to-top-cell byte offset from [`crate::recover_entry_frame`].
    pub top_offset: u64,
    /// Recovered VM-exit layouts, keyed by the exit handler's address.
    pub exits: &'a BTreeMap<Va, VmExitReport>,
    /// Proven call frames, keyed by the key of the block that ends in the call.
    pub native_calls: &'a BTreeMap<Va, NativeCallSelectorOutcome>,
    pub memory: Option<&'a dyn SymbolicMemorySource>,
    pub limits: SymbolicExecutionLimits,
    /// Limits for resolving a branch target over its input domain.
    pub target_limits: VmCfgLimits,
    /// Block expansions across all paths before the lift gives up.
    pub max_expanded_blocks: usize,
}

#[derive(Debug)]
pub struct LiftedCfgFunction {
    pub replacement: Replacement,
    /// Distinct CFG blocks that took part in the lift.
    pub blocks: BTreeSet<Va>,
    /// Block expansions, counting a joined block once per incoming path.
    pub expanded_blocks: usize,
    pub exit_paths: usize,
    pub conditional_branches: usize,
    /// Natural loops emitted as native back-edges.
    pub loops: usize,
    pub native_calls: usize,
    /// Nested VM entries inlined into the function.
    pub nested_entries: usize,
    pub assumptions: Vec<LiftAssumption>,
}

struct CallSite {
    index: usize,
    target: Expr,
    arguments: Vec<Expr>,
}

/// A memory read captured at its native-order position because it could not be
/// safely re-issued where its value is used (a store or native call that may
/// affect it runs in between). The load is emitted here and spilled; later uses
/// read the spill slot. See [`Builder::capture_read`].
struct LoadRead {
    index: usize,
    address: Expr,
    width: BitWidth,
}

enum StepKind {
    Store(EffectStore),
    Call(CallSite),
    Load(LoadRead),
}

/// A side effect emitted at a point on the path, tagged with the memory epoch at
/// which it commits so reads can be ordered against it.
struct Step {
    epoch: u64,
    kind: StepKind,
}

struct ExitLeaf {
    registers: BTreeMap<X64NativeRegister, Expr>,
    /// Flags carried out to the caller; `None` leaves them clobbered (entry
    /// flags, volatile across a Win64 return).
    rflags: Option<Expr>,
    return_pop: u32,
}

/// An exit that returns through a word of the entry frame: control transfers to
/// `staged[0]` with the remaining words left as the continuation's stack.
struct RegionLeaf {
    registers: BTreeMap<X64NativeRegister, Expr>,
    /// Flags the continuation starts with; `None` is the entry flags.
    rflags: Option<Expr>,
    staged: Vec<Expr>,
    /// Offset of `staged[0]` from the initial VM stack pointer.
    slot_offset: u32,
}

/// One loop-carried cell, emitted as a frame slot (`phi`) live across the
/// back-edge. `init` is the pre-loop value over entry inputs; `next` is the
/// per-iteration value over the current `phi`s (and entry inputs).
struct LoopPhi {
    id: u16,
    init: Expr,
    next: Expr,
}

/// A natural (reducible) loop: a header with `phi` loop variables, one or more
/// exit branches along the body, and each exit's continuation once the loop is
/// left. Besides the pure state transform in `phis`, the body may carry external
/// stores (in `steps`) that are reproduced in order each iteration, exactly as
/// the original ran them.
struct LoopLeaf {
    phis: Vec<LoopPhi>,
    /// Side effects (external stores, and their captured reads) the body runs,
    /// in native (epoch) order. Their address/value expressions are over the
    /// current `phi`s and entry inputs, so each iteration reproduces the exact
    /// per-iteration store. A step committing at or before an exit's epoch runs
    /// before that exit's branch (so on the iteration that takes it); a step
    /// past the last exit runs only when the back-edge is taken.
    steps: Vec<Step>,
    /// The loop's exits in body (epoch-ascending) order; at least one. Each is a
    /// conditional break reproduced in turn; the back-edge is taken when none
    /// fires.
    exits: Vec<LoopExit>,
}

/// One exit of a reducible loop: a conditional branch in the body that leaves the
/// loop under its own condition, with the continuation that runs when it is taken.
struct LoopExit {
    /// Memory epoch at which this exit's branch is evaluated: the split in the
    /// body's steps between those emitted before this branch and those after.
    epoch: u64,
    /// The exit branch selector over the current `phi`s.
    condition: Expr,
    /// Recovered address of the in-loop (continue) successor: the branch stays in
    /// the loop when `condition` dispatches to it, and leaves otherwise.
    continue_value: u64,
    /// Where control goes when this exit is taken, with the `phi`s left live.
    continuation: Box<Node>,
}

enum End {
    Exit(ExitLeaf),
    Region(RegionLeaf),
    /// Arms are ordered by target value; the last is the fall-through.
    Branch {
        condition: Expr,
        arms: Vec<(u64, Node)>,
    },
    Loop(LoopLeaf),
}

struct Node {
    steps: Vec<Step>,
    end: End,
}

/// Name of the frame-slot input backing loop `phi` variable `id`.
fn loop_phi_name(id: u16) -> String {
    format!("__loop_phi_{id}")
}

/// A loop-carried VM-state cell, addressed in the symbolic state.
#[derive(Clone)]
enum Cell {
    Flags,
    Context(i64),
    Memory(Expr),
}

/// Stable string key for a cell, comparing structurally between states.
fn cell_key(cell: &Cell) -> String {
    match cell {
        Cell::Flags => "flags".to_owned(),
        Cell::Context(offset) => format!("c{offset}"),
        Cell::Memory(address) => format!("m{address:?}"),
    }
}

/// Every live cell of `state` keyed stably, so two states' live cells compare.
/// A VM-stack cell below the live top (already popped) is dead: subsequent code
/// cannot read it without first overwriting it, so it is not loop-carried state.
fn cell_map(state: &SymbolicVmInitialState) -> BTreeMap<String, (Cell, Expr)> {
    let top = symbolic_pointer_offset(&state.vsp, "vsp_initial");
    let mut map = BTreeMap::new();
    map.insert("flags".to_owned(), (Cell::Flags, state.flags.clone()));
    for (offset, value) in &state.context_values {
        map.insert(
            format!("c{offset}"),
            (Cell::Context(*offset), value.clone()),
        );
    }
    for (address, value) in &state.memory_values {
        if let (Some(top), Some(cell)) = (top, symbolic_pointer_offset(address, "vsp_initial"))
            && cell < top
        {
            continue;
        }
        map.insert(
            format!("m{address:?}"),
            (Cell::Memory(address.clone()), value.clone()),
        );
    }
    map
}

fn get_cell<'a>(state: &'a SymbolicVmInitialState, cell: &Cell) -> Option<&'a Expr> {
    match cell {
        Cell::Flags => Some(&state.flags),
        Cell::Context(offset) => state.context_values.get(offset),
        Cell::Memory(address) => state
            .memory_values
            .iter()
            .find(|(candidate, _)| candidate == address)
            .map(|(_, value)| value),
    }
}

fn set_cell(state: &mut SymbolicVmInitialState, cell: &Cell, value: Expr) {
    match cell {
        Cell::Flags => state.flags = value,
        Cell::Context(offset) => {
            state.context_values.insert(*offset, value);
        }
        Cell::Memory(address) => {
            if let Some(entry) = state
                .memory_values
                .iter_mut()
                .find(|(candidate, _)| candidate == address)
            {
                entry.1 = value;
            } else {
                state.memory_values.push((address.clone(), value));
            }
        }
    }
}

#[derive(Clone)]
struct PathStore {
    /// Upper bound on the memory epoch right after this store.
    end_epoch: u64,
    address: Expr,
    bytes: u32,
}

#[derive(Clone, Default)]
struct Path {
    trail: Vec<Va>,
    /// Memory epoch at the latest external store or call on this path. A load
    /// from an earlier epoch cannot be re-read natively after that barrier.
    barrier: u64,
    /// Memory epoch at the latest native call on this path; no read crosses it.
    call_barrier: u64,
    /// External stores on this path, in native order.
    stores: Vec<PathStore>,
    /// A store on this path wrote a value derived from the VM stack pointer.
    root_escaped: bool,
    /// Native-call result intrinsics bound to their frame-slot input names.
    results: BTreeMap<String, String>,
}

struct Builder<'a> {
    input: &'a DecodedCfgInput<'a>,
    nested: Option<&'a dyn NestedEntryResolver>,
    nested_entries: usize,
    expanded: usize,
    blocks: BTreeSet<Va>,
    exit_paths: usize,
    region_exits: usize,
    /// Largest staged continuation stack over all region exits, in bytes.
    retained_bytes: u32,
    branches: usize,
    loops: usize,
    call_sites: usize,
    /// In-place captured reads allocated so far; also the next slot id.
    read_sites: usize,
    uses_root: bool,
    assumptions: Vec<LiftAssumption>,
    /// Natural-loop bodies keyed by header, recovered up front.
    loop_bodies: BTreeMap<Va, BTreeSet<Va>>,
    /// Loop `phi` slots allocated so far; also the next slot id.
    loop_phis: u16,
    /// Active `phi` input names a prepared expression may legitimately read.
    active_phis: BTreeSet<String>,
    /// Memory epoch at which the current node's steps began; stores/calls below
    /// it live in ancestor nodes. Set at the top of [`Builder::walk`].
    node_base_epoch: u64,
    /// Native-call barrier inherited by the current node (ancestor calls only).
    node_call_barrier: u64,
    /// Branch nodes currently open on the path from the root to the node being
    /// built, outermost first. A read in a descendant whose ordered position
    /// lies in an ancestor's epoch range is eagerly captured into that
    /// ancestor's `steps` and threaded forward through its spill slot. See
    /// [`cross_node_capture`].
    ancestors: Vec<OpenNode>,
}

/// A branch node still being built: its steps buffer stays reachable while its
/// arms are walked so a descendant read can be captured at the right epoch
/// position here. `base_epoch` is the memory epoch this node's steps began at;
/// it owns the epoch range `[base_epoch, child_base)`, so the owner of a read's
/// epoch is the deepest open node whose `base_epoch` does not exceed it.
struct OpenNode {
    base_epoch: u64,
    /// Call-result spill-slot names produced at or before this node, the only
    /// call results a capture placed here may reference.
    available_calls: BTreeSet<String>,
    steps: Vec<Step>,
}

/// Forward (non-back-edge) predecessors of each block. A `(from, to)` pair that
/// carries a `LoopBack` edge is a back-edge and is excluded whatever other kind
/// also joins it, since the walk records the back-edge's `Branch` twin too.
fn forward_predecessors(cfg: &VmCfg) -> BTreeMap<Va, Vec<Va>> {
    let back: BTreeSet<(Va, Va)> = cfg
        .edges
        .iter()
        .filter(|edge| edge.kind == CfgEdgeKind::LoopBack)
        .map(|edge| (edge.from, edge.to))
        .collect();
    let mut preds: BTreeMap<Va, Vec<Va>> = BTreeMap::new();
    for edge in &cfg.edges {
        if back.contains(&(edge.from, edge.to)) {
            continue;
        }
        preds.entry(edge.to).or_default().push(edge.from);
    }
    preds
}

/// Dominator sets over the forward CFG from the entry (iterative fixpoint).
/// Unreachable blocks keep the full node set and never found a loop.
fn dominators(cfg: &VmCfg, preds: &BTreeMap<Va, Vec<Va>>) -> BTreeMap<Va, BTreeSet<Va>> {
    let all: BTreeSet<Va> = cfg.blocks.keys().copied().collect();
    let mut dom: BTreeMap<Va, BTreeSet<Va>> = all
        .iter()
        .map(|&node| {
            let set = if node == cfg.entry {
                BTreeSet::from([node])
            } else {
                all.clone()
            };
            (node, set)
        })
        .collect();
    let mut changed = true;
    while changed {
        changed = false;
        for &node in &all {
            if node == cfg.entry {
                continue;
            }
            let incoming = preds.get(&node).map(Vec::as_slice).unwrap_or_default();
            let mut new: Option<BTreeSet<Va>> = None;
            for pred in incoming {
                let pred_dom = &dom[pred];
                new = Some(match new {
                    None => pred_dom.clone(),
                    Some(acc) => acc.intersection(pred_dom).copied().collect(),
                });
            }
            let mut new = new.unwrap_or_default();
            new.insert(node);
            if new != dom[&node] {
                dom.insert(node, new);
                changed = true;
            }
        }
    }
    dom
}

/// Nodes of the natural loop of back-edge `source` -> `header`: the header plus
/// every node that reaches `source` without passing through the header.
fn natural_loop_body(
    header: Va,
    source: Va,
    preds: &BTreeMap<Va, Vec<Va>>,
) -> BTreeSet<Va> {
    let mut body = BTreeSet::from([header]);
    if source != header {
        body.insert(source);
        let mut worklist = vec![source];
        while let Some(node) = worklist.pop() {
            for &pred in preds.get(&node).map(Vec::as_slice).unwrap_or_default() {
                if body.insert(pred) {
                    worklist.push(pred);
                }
            }
        }
    }
    body
}

/// Reducible natural loops keyed by header. A header reached by more than one
/// back-edge, or whose back-edge source it does not dominate, is dropped so the
/// walk fails closed on it.
fn recover_natural_loops(cfg: &VmCfg) -> BTreeMap<Va, BTreeSet<Va>> {
    let preds = forward_predecessors(cfg);
    let dom = dominators(cfg, &preds);
    let mut map: BTreeMap<Va, BTreeSet<Va>> = BTreeMap::new();
    let mut invalid: BTreeSet<Va> = BTreeSet::new();
    for edge in &cfg.edges {
        if edge.kind != CfgEdgeKind::LoopBack {
            continue;
        }
        let (source, header) = (edge.from, edge.to);
        if invalid.contains(&header) {
            continue;
        }
        let reducible = dom.get(&source).is_some_and(|set| set.contains(&header));
        if map.contains_key(&header) || !reducible {
            // A second back-edge to this header or an irreducible one: refuse it.
            map.remove(&header);
            invalid.insert(header);
            continue;
        }
        let body = natural_loop_body(header, source, &preds);
        if body
            .iter()
            .all(|node| dom.get(node).is_some_and(|set| set.contains(&header)))
        {
            map.insert(header, body);
        } else {
            invalid.insert(header);
        }
    }
    map
}

/// The outcome of one symbolic iteration of a loop body.
struct BodyRun {
    /// State flowing back into the header along the back-edge.
    reentry: SymbolicVmInitialState,
    /// The loop's exits in body (epoch-ascending) order; at least one.
    exits: Vec<PendingExit>,
    /// Prepared body side effects (stores and their captured reads), in native
    /// (epoch) order. Populated only on the collecting iteration; empty otherwise.
    body_steps: Vec<Step>,
    /// The path after the whole iteration (back-edge taken): carries every body
    /// store, so the per-iteration `next` phi values order their reads correctly.
    body_path: Path,
}

/// One exit of the loop, recovered at its exit-branch block. A loop may have
/// several; each carries the state and effects as of its own branch.
struct PendingExit {
    condition: Expr,
    continue_value: u64,
    exit_key: Va,
    /// State the exit branch leaves for the continuation.
    exit_state: SymbolicVmInitialState,
    guarded: bool,
    /// Memory epoch at which this exit's branch is evaluated.
    epoch: u64,
    /// The path as of this exit branch: its `stores`/`barrier` carry the effects
    /// that precede the exit, so its continuation orders its reads against
    /// exactly what ran before the loop was left here.
    path: Path,
}

/// The loop-carried cells: those whose value differs between the pre-loop state
/// `pre` and the back-edge re-entry state `reentry`. The stack pointer, fresh
/// allocations, memory precision and live-cell set must be loop-invariant.
fn loop_cells(
    header: Va,
    pre: &SymbolicVmInitialState,
    reentry: &SymbolicVmInitialState,
) -> Result<Vec<(Cell, BitWidth)>, DecodedLiftError> {
    if pre.memory_tainted || reentry.memory_tainted {
        return refuse(header, "loop follows an imprecise external-memory store");
    }
    if pre.vsp != reentry.vsp {
        return refuse(header, "loop changes the VM stack pointer");
    }
    if pre.fresh_allocations != reentry.fresh_allocations {
        return refuse(header, "loop changes the fresh allocations");
    }
    let (pre_cells, reentry_cells) = (cell_map(pre), cell_map(reentry));
    if pre_cells.keys().ne(reentry_cells.keys()) {
        return refuse(header, "loop changes its set of live cells");
    }
    let mut cells = Vec::new();
    for (key, (cell, pre_value)) in &pre_cells {
        let (_, reentry_value) = &reentry_cells[key];
        if pre_value != reentry_value {
            if pre_value.width != reentry_value.width {
                return refuse(header, "a loop-carried cell changes width");
            }
            // The phi lives in a 64-bit frame slot; a narrower carried cell would
            // leave the slot's upper bytes undefined. Fail closed on it.
            if pre_value.width != BitWidth::W64 {
                return refuse(header, "a loop-carried cell is not 64 bits wide");
            }
            cells.push((cell.clone(), pre_value.width));
        }
    }
    Ok(cells)
}

/// Proves the phi iteration closes: it changes exactly the recognized loop-
/// carried cells and leaves every other cell at its invariant value. A cell that
/// moves but was not recognized means a loop-carried value went uncaptured.
fn verify_closure(
    header: Va,
    phi_state: &SymbolicVmInitialState,
    reentry: &SymbolicVmInitialState,
    specs: &[(Cell, u16, BitWidth)],
) -> Result<(), DecodedLiftError> {
    if phi_state.vsp != reentry.vsp || phi_state.fresh_allocations != reentry.fresh_allocations {
        return refuse(header, "loop changes the stack pointer or allocations");
    }
    let (before, after) = (cell_map(phi_state), cell_map(reentry));
    if before.keys().ne(after.keys()) {
        return refuse(header, "loop changes its set of live cells under iteration");
    }
    let carried: BTreeSet<String> = specs.iter().map(|(cell, _, _)| cell_key(cell)).collect();
    for (key, (_, before_value)) in &before {
        if carried.contains(key) {
            continue;
        }
        if before_value != &after[key].1 {
            return refuse(header, "loop carries a cell not recognized as loop-carried");
        }
    }
    Ok(())
}

fn refuse<T>(block: Va, reason: impl Into<String>) -> Result<T, DecodedLiftError> {
    Err(DecodedLiftError::Cfg {
        block,
        reason: reason.into(),
    })
}

fn call_result_name(index: usize) -> String {
    format!("__call_result_{index}")
}

const READ_RESULT_PREFIX: &str = "__read_result_";

fn read_result_name(index: usize) -> String {
    format!("{READ_RESULT_PREFIX}{index}")
}

/// Reference to a captured read's spill slot (held 64-bit), narrowed to the
/// read's own width.
fn read_result_ref(width: BitWidth, index: usize) -> Option<Expr> {
    let register = Expr::register(BitWidth::W64, read_result_name(index));
    if width == BitWidth::W64 {
        Some(register)
    } else {
        Expr::extract(width, 0, register).ok()
    }
}

/// Both addresses are fixed offsets from the entry stack pointer and their byte
/// ranges do not overlap.
fn provably_disjoint(a: &Expr, a_bytes: u32, b: &Expr, b_bytes: u32) -> bool {
    let (Some(a_start), Some(b_start)) = (
        symbolic_pointer_offset(a, "vsp_initial"),
        symbolic_pointer_offset(b, "vsp_initial"),
    ) else {
        return false;
    };
    let (Some(a_end), Some(b_end)) = (
        a_start.checked_add(i64::from(a_bytes)),
        b_start.checked_add(i64::from(b_bytes)),
    ) else {
        return false;
    };
    a_end <= b_start || b_end <= a_start
}

/// An earlier read can be re-issued natively at a later point when no native
/// call intervenes and every store since it is provably disjoint from it.
fn read_is_stable(path: &Path, epoch: u64, address: &Expr, bytes: u32) -> bool {
    epoch >= path.call_barrier
        && path
            .stores
            .iter()
            .filter(|store| store.end_epoch > epoch)
            .all(|store| provably_disjoint(address, bytes, &store.address, store.bytes))
}

/// Whether a read that cannot be re-issued at the use site can instead be kept
/// in place: emitted at the start of the current node (just after the state it
/// inherits) and spilled, with later uses reading the spill.
///
/// This is sound only when every store or native call that could affect the
/// read and runs *before* the current node already preserves the read's value
/// up to the node's entry — those steps are emitted in ancestor nodes and cannot
/// be reordered against. A store or call that runs *within* this node (epoch past
/// `node_base`) is handled by positioning the emitted load ahead of it in
/// [`Builder::capture_read`], so it is not an obstacle here.
///
/// Concretely the read must not have been crossed by an ancestor native call
/// (`epoch >= node_call_barrier`) and every ancestor store committed after the
/// read (`node_base >= store.end_epoch > epoch`) must be provably disjoint.
fn read_fits_in_node(
    path: &Path,
    epoch: u64,
    address: &Expr,
    bytes: u32,
    node_base: u64,
    node_call_barrier: u64,
) -> bool {
    epoch >= node_call_barrier
        && path
            .stores
            .iter()
            .filter(|store| store.end_epoch > epoch && store.end_epoch <= node_base)
            .all(|store| provably_disjoint(address, bytes, &store.address, store.bytes))
}

/// Whether an expression embeds a raw memory load. Such a load would re-read
/// memory at whatever point it is emitted, so an address carrying one cannot be
/// hoisted into an ancestor (it could observe different memory there).
fn expr_has_load(expression: &Expr) -> bool {
    match &expression.kind {
        ExprKind::Const { .. } | ExprKind::Register { .. } => false,
        ExprKind::Load { .. } => true,
        ExprKind::Unary { value, .. } | ExprKind::Extract { value, .. } | ExprKind::Resize { value, .. } => {
            expr_has_load(value)
        }
        ExprKind::Binary { left, right, .. } => expr_has_load(left) || expr_has_load(right),
        ExprKind::Concat { high, low } => expr_has_load(high) || expr_has_load(low),
        ExprKind::Intrinsic { arguments, .. } => arguments.iter().any(expr_has_load),
    }
}

/// Eagerly captures a memory read whose ordered position lies in an ancestor
/// node (its required store/call was emitted there and cannot be reordered
/// against in the current node). The read is captured at its epoch-ordered
/// position in the owner ancestor, spilled to a result slot, and referenced
/// here through that slot — the slot's value is threaded forward unchanged
/// because the ancestor's steps are emitted, unconditionally, before any of its
/// arms.
///
/// Returns the slot reference, or `None` to fail closed when the capture cannot
/// be placed soundly: no owning ancestor; an address that embeds a raw load (it
/// could read different memory in the ancestor); an address that still carries
/// an unresolved earlier read; or an address depending on a value not available
/// at the owner (a later call result, a loop phi, or anything foreign).
fn cross_node_capture(
    ancestors: &mut [OpenNode],
    read_sites: &mut usize,
    uses_root: &mut bool,
    allowed_registers: &BTreeSet<&str>,
    epoch: u64,
    address: &Expr,
    width: BitWidth,
) -> Option<Expr> {
    // The address is evaluated at the capture point, so it must mean the same
    // there: no embedded load, and no unresolved earlier read left behind.
    if expr_has_load(address)
        || address
            .intrinsic_dependencies()
            .iter()
            .any(|name| name.starts_with("memory_load_epoch_"))
    {
        return None;
    }
    // Owner = deepest open node whose epoch range covers the read.
    let owner = ancestors.iter_mut().rev().find(|node| node.base_epoch <= epoch)?;
    // Every address dependency must be nameable at the owner. Entry registers,
    // the stack root and already-captured read slots are stable everywhere; a
    // call result must have been produced at or before the owner; anything else
    // (a loop phi, a foreign value) fails closed.
    for dependency in address.dependencies() {
        if dependency == "vsp_initial" {
            *uses_root = true;
        } else if dependency.starts_with(READ_RESULT_PREFIX)
            || allowed_registers.contains(dependency.as_str())
        {
            // Stable across node boundaries.
        } else if dependency.starts_with("__call_result_") {
            if !owner.available_calls.contains(&dependency) {
                return None;
            }
        } else {
            return None;
        }
    }
    let index = *read_sites;
    let reference = read_result_ref(width, index)?;
    *read_sites = index + 1;
    Builder::place_captured_read(&mut owner.steps, index, epoch, address.clone(), width);
    Some(reference)
}

fn is_native_rflags_initial(expression: &Expr) -> bool {
    matches!(
        &expression.simplify().kind,
        ExprKind::Register { name } if name == "native_rflags_initial"
    )
}

impl Builder<'_> {
    /// Rewrites an expression into what the native function can evaluate at the
    /// current point, or refuses.
    ///
    /// A memory read is re-issued as a native load inline when it is provably
    /// unaffected by anything between it and this point (`epoch >= barrier` or
    /// [`read_is_stable`]). Otherwise, rather than refusing, the read is kept in
    /// its original program order: captured at its native-order position in
    /// `steps` and referenced here through its spill slot (see
    /// [`Builder::capture_read`]). Only a read that cannot even be ordered
    /// correctly within this node still refuses.
    fn prepare(
        &mut self,
        block: Va,
        expression: &Expr,
        barrier: u64,
        path: &Path,
        steps: &mut Vec<Step>,
    ) -> Result<Expr, DecodedLiftError> {
        let results = &path.results;
        let node_base = self.node_base_epoch;
        let node_call_barrier = self.node_call_barrier;
        // Disjoint field borrows the capture closure needs alongside `read_sites`.
        let read_sites = &mut self.read_sites;
        let ancestors = self.ancestors.as_mut_slice();
        let mut captured_uses_root = false;
        let uses_root_flag = &mut captured_uses_root;
        let allowed_registers: BTreeSet<&str> = X64NativeRegister::ALL
            .iter()
            .map(|register| register.symbolic_name())
            .collect();
        let rewritten = expression
            .rewrite_intrinsics(&mut |width, name, arguments| {
                if let Some(register) = results.get(name) {
                    return (width == BitWidth::W64)
                        .then(|| Expr::register(width, register.clone()));
                }
                let epoch = name
                    .strip_prefix("memory_load_epoch_")?
                    .parse::<u64>()
                    .ok()?;
                let [address] = arguments else { return None };
                if address.width != BitWidth::W64 {
                    return None;
                }
                if epoch >= barrier || read_is_stable(path, epoch, address, width.bytes()) {
                    return Some(Expr::load(width, address.clone()));
                }
                // Cannot re-issue the read here. Keep it in place if ordering
                // permits within this node; otherwise try capturing it in the
                // ancestor node whose epoch range owns the read (cross-node eager
                // capture); otherwise leave it for the refusal below.
                if !read_fits_in_node(path, epoch, address, width.bytes(), node_base, node_call_barrier)
                {
                    return cross_node_capture(
                        ancestors,
                        read_sites,
                        uses_root_flag,
                        &allowed_registers,
                        epoch,
                        address,
                        width,
                    );
                }
                let index = *read_sites;
                let reference = read_result_ref(width, index)?;
                *read_sites = index + 1;
                Self::place_captured_read(steps, index, epoch, address.clone(), width);
                Some(reference)
            })
            .map_err(|error| DecodedLiftError::Cfg {
                block,
                reason: error.to_string(),
            })?;
        if rewritten
            .intrinsic_dependencies()
            .iter()
            .any(|name| name.starts_with("memory_load_epoch_"))
        {
            return refuse(
                block,
                "a memory read predates a store or native call that precedes its use",
            );
        }
        let allowed: BTreeSet<&str> = X64NativeRegister::ALL
            .iter()
            .map(|register| register.symbolic_name())
            .chain(std::iter::once("vsp_initial"))
            .chain(results.values().map(String::as_str))
            .chain(self.active_phis.iter().map(String::as_str))
            .collect();
        let dependencies = rewritten.dependencies();
        let foreign: Vec<&String> = dependencies
            .iter()
            .filter(|name| {
                !allowed.contains(name.as_str()) && !name.starts_with(READ_RESULT_PREFIX)
            })
            .collect();
        if !foreign.is_empty() {
            return refuse(
                block,
                format!(
                    "value depends on {foreign:?}, which a callee-style replacement cannot read"
                ),
            );
        }
        if dependencies.contains("vsp_initial") || captured_uses_root {
            self.uses_root = true;
        }
        Ok(rewritten)
    }

    /// Inserts a captured read's load into `steps` at its native-order position:
    /// after every step that commits at or before the read's epoch and before the
    /// first step that commits after it. Steps stay ordered by epoch, so the load
    /// runs after any store/call the read must observe and before any that must
    /// not reach it.
    fn place_captured_read(
        steps: &mut Vec<Step>,
        index: usize,
        epoch: u64,
        address: Expr,
        width: BitWidth,
    ) {
        let position = steps
            .iter()
            .position(|step| step.epoch > epoch)
            .unwrap_or(steps.len());
        steps.insert(
            position,
            Step {
                epoch,
                kind: StepKind::Load(LoadRead {
                    index,
                    address,
                    width,
                }),
            },
        );
    }

    /// The branch's proven successors, taken straight from the walk's CFG edges
    /// rather than re-derived: each `Branch`/`GuardedBranch` edge from this
    /// block is a successor the walk already resolved, and the successor block's
    /// recorded address is the value the native branch dispatches on. `guarded`
    /// is true when any such edge was forked over a proven input domain or
    /// boolean flag. Returned ascending by value, matching the emitter's arm
    /// order. Fails closed on an edge whose target is not a recovered block or
    /// carries no address; `LoopBack` edges (handled by the walk's repeat guard)
    /// and non-branch edges are ignored.
    fn branch_successors(&self, block: Va) -> Result<(Vec<(u64, Va)>, bool), DecodedLiftError> {
        let cfg = self.input.cfg;
        let mut successors: Vec<(u64, Va)> = Vec::new();
        let mut guarded = false;
        for edge in &cfg.edges {
            if edge.from != block {
                continue;
            }
            match edge.kind {
                CfgEdgeKind::Branch => {}
                CfgEdgeKind::GuardedBranch => guarded = true,
                _ => continue,
            }
            let Some(successor) = cfg.blocks.get(&edge.to) else {
                return refuse(
                    block,
                    format!("branch edge target {} is not a recovered block", edge.to),
                );
            };
            let Some(address) = successor.address else {
                return refuse(
                    block,
                    format!("branch successor {} has no recorded target address", edge.to),
                );
            };
            if !successors.iter().any(|(_, key)| *key == edge.to) {
                successors.push((address.0, edge.to));
            }
        }
        if successors.is_empty() {
            return refuse(block, "branch has no proven CFG edge from this block");
        }
        successors.sort_by_key(|(value, _)| *value);
        Ok((successors, guarded))
    }

    fn call_return_block(&self, block: Va) -> Result<Va, DecodedLiftError> {
        let mut returns = self
            .input
            .cfg
            .edges
            .iter()
            .filter(|edge| edge.from == block && edge.kind == CfgEdgeKind::NativeCallReturn);
        match (returns.next(), returns.next()) {
            (Some(edge), None) => Ok(edge.to),
            _ => refuse(block, "native call has no single proven return edge"),
        }
    }

    #[allow(clippy::too_many_lines)]
    fn walk(
        &mut self,
        mut key: Va,
        mut state: SymbolicVmInitialState,
        mut path: Path,
        mut steps: Vec<Step>,
    ) -> Result<Node, DecodedLiftError> {
        let input = self.input;
        // This node's steps begin from the state it inherits: any store or call
        // below these epochs was emitted in an ancestor node and cannot be
        // reordered against. Captured reads must respect that (see
        // `read_fits_in_node`).
        self.node_base_epoch = state.memory_epoch;
        self.node_call_barrier = path.call_barrier;
        loop {
            if self.expanded >= input.max_expanded_blocks {
                return refuse(key, "block expansion budget exhausted");
            }
            self.expanded += 1;
            if path.trail.contains(&key) {
                return refuse(key, "loop: a block repeats on one path");
            }
            path.trail.push(key);
            self.blocks.insert(key);
            if let Some(body) = self.loop_bodies.get(&key).cloned() {
                return self.lift_loop(key, state, path, steps, &body);
            }
            let Some(block) = input.cfg.blocks.get(&key) else {
                return refuse(key, "block is not in the recovered CFG");
            };
            if let Some((index, semantic, reason)) = first_unliftable(&block.instructions) {
                return refuse(key, format!("instruction {index} ({semantic:?}): {reason}"));
            }
            let boundary = match block.exit {
                BlockExit::VmExit => Some(VmHandlerSemantic::VmExit),
                BlockExit::Branch => Some(VmHandlerSemantic::UnconditionalBranch),
                BlockExit::NativeCall => None,
                BlockExit::Stalled => return refuse(key, "block did not reach a control boundary"),
            };
            let report = execute_symbolic_linear_block_with_initial_state(
                &block.instructions,
                boundary,
                input.limits,
                &state,
                input.memory,
            )
            .map_err(|error| DecodedLiftError::Cfg {
                block: key,
                reason: format!("symbolic execution failed: {error}"),
            })?;
            if !report.memory_havocs.is_empty()
                || !report.native_calls.is_empty()
                || !report.fresh_allocations.is_empty()
                || report.memory_tainted
            {
                return refuse(key, "block body havocs memory, calls natively or allocates");
            }

            let entry_barrier = path.barrier;
            for (index, write) in report.external_writes.iter().enumerate() {
                let barrier = if index == 0 {
                    entry_barrier
                } else {
                    report.memory_epoch
                };
                if write.value.dependencies().contains("vsp_initial") {
                    path.root_escaped = true;
                }
                let address = self.prepare(key, &write.address, barrier, &path, &mut steps)?;
                let value = self.prepare(key, &write.value, barrier, &path, &mut steps)?;
                let width = value.width;
                path.stores.push(PathStore {
                    end_epoch: write.epoch,
                    address: write.address.clone(),
                    bytes: write.value.width.bytes(),
                });
                steps.push(Step {
                    epoch: write.epoch,
                    kind: StepKind::Store(EffectStore {
                        address,
                        value,
                        width,
                    }),
                });
            }
            if !report.external_writes.is_empty() {
                path.barrier = path.barrier.max(report.memory_epoch);
            }

            match block.exit {
                BlockExit::VmExit => {
                    let handler = block
                        .instructions
                        .last()
                        .map_or(block.start.handler, |last| last.state_after.handler);
                    let Some(layout) = block.vm_exit.as_ref().or_else(|| input.exits.get(&handler))
                    else {
                        return refuse(
                            key,
                            format!("no recovered VM-exit layout for handler {handler}"),
                        );
                    };
                    if let Some(resolver) = self.nested
                        && let Some(to) = input.cfg.edges.iter().find_map(|edge| {
                            (edge.from == key && edge.kind == CfgEdgeKind::NestedEntry)
                                .then_some(edge.to)
                        })
                        && let Some(composed) =
                            self.compose_nested(key, to, &report, layout, resolver)?
                    {
                        self.nested_entries += 1;
                        key = to;
                        state = composed;
                        continue;
                    }
                    let recovered = recover_exit(
                        &report,
                        layout,
                        input.frame.len(),
                        input.top_offset,
                    )
                    .map_err(|error| DecodedLiftError::Cfg {
                        block: key,
                        reason: error.to_string(),
                    })?;
                    let exit = match recovered {
                        RecoveredExit::Callee(exit) => exit,
                        RecoveredExit::Region(exit) => {
                            let rflags = if is_native_rflags_initial(&exit.rflags) {
                                None
                            } else {
                                if symbolic_pointer_offset(&exit.rflags, "vsp_initial").is_some() {
                                    return refuse(
                                        key,
                                        "the region exit flags are a stack pointer",
                                    );
                                }
                                Some(self.prepare(key, &exit.rflags, path.barrier, &path, &mut steps)?)
                            };
                            let mut registers = BTreeMap::new();
                            for (register, value) in &exit.registers {
                                registers.insert(
                                    *register,
                                    self.prepare(key, value, path.barrier, &path, &mut steps)?,
                                );
                            }
                            let mut staged = Vec::with_capacity(exit.staged.len());
                            for word in &exit.staged {
                                staged.push(self.prepare(key, word, path.barrier, &path, &mut steps)?);
                            }
                            let bytes = u32::try_from(staged.len() * 8).unwrap_or(u32::MAX);
                            self.retained_bytes = self.retained_bytes.max(bytes);
                            self.region_exits += 1;
                            self.exit_paths += 1;
                            self.uses_root = true;
                            return Ok(Node {
                                steps,
                                end: End::Region(RegionLeaf {
                                    registers,
                                    rflags,
                                    staged,
                                    slot_offset: exit.slot_offset,
                                }),
                            });
                        }
                    };
                    // A popping return lowers to a flag-preserving `lea; ret`, so
                    // flags the exit carries out reach the caller intact. When they
                    // differ from the entry flags they are the real result: capture
                    // and restore them. Entry flags stay volatile per Win64.
                    let rflags = if is_native_rflags_initial(&exit.rflags) {
                        self.assume(LiftAssumption::FlagsVolatileAtReturn);
                        None
                    } else {
                        if symbolic_pointer_offset(&exit.rflags, "vsp_initial").is_some() {
                            return refuse(key, "the exit flags are a stack pointer");
                        }
                        Some(self.prepare(key, &exit.rflags, path.barrier, &path, &mut steps)?)
                    };
                    let mut registers = BTreeMap::new();
                    for (register, value) in &exit.registers {
                        registers.insert(*register, self.prepare(key, value, path.barrier, &path, &mut steps)?);
                    }
                    self.exit_paths += 1;
                    return Ok(Node {
                        steps,
                        end: End::Exit(ExitLeaf {
                            registers,
                            rflags,
                            return_pop: exit.return_pop,
                        }),
                    });
                }
                BlockExit::Branch => {
                    let Some(target) = report.branch_target.as_ref() else {
                        return refuse(key, "branch has no symbolic target");
                    };
                    // Reuse the walk's already-resolved edges: the branch targets
                    // are exactly the recovered CFG successors, so the lift never
                    // re-derives them (and so inherits the walk's flag-fork and
                    // preferred-base resolutions instead of refusing without them).
                    let (successors, guarded) = self.branch_successors(key)?;
                    let next = report.continuation_state();
                    if let [(_, only)] = successors[..] {
                        if guarded {
                            return refuse(key, "guarded branch resolved to a single target");
                        }
                        key = only;
                        state = next;
                        continue;
                    }
                    // The runtime branch expression is dispatched against each
                    // resolved value; a value proven only under the preferred-base
                    // frame makes the lifted function carry that assumption too.
                    if guarded && input.cfg.assumptions.contains(&VmCfgAssumption::PreferredImageBase)
                    {
                        self.assume(LiftAssumption::PreferredImageBase);
                    }
                    let condition = self.prepare(key, target, path.barrier, &path, &mut steps)?;
                    self.branches += 1;
                    // Keep this node's steps reachable while its arms are walked,
                    // so a descendant read whose ordered position lies in this
                    // node's epoch range is captured here (cross-node eager
                    // capture) rather than refused. The node is this walk's steps
                    // begun at `node_base_epoch`.
                    self.ancestors.push(OpenNode {
                        base_epoch: self.node_base_epoch,
                        available_calls: path.results.values().cloned().collect(),
                        steps: std::mem::take(&mut steps),
                    });
                    let mut arms = Vec::with_capacity(successors.len());
                    let mut result = Ok(());
                    for (value, successor) in successors {
                        match self.walk(successor, next.clone(), path.clone(), Vec::new()) {
                            Ok(arm) => arms.push((value, arm)),
                            Err(error) => {
                                result = Err(error);
                                break;
                            }
                        }
                    }
                    // Reclaim the (now capture-augmented) steps before returning.
                    let steps = self.ancestors.pop().expect("ancestors balanced").steps;
                    result?;
                    return Ok(Node {
                        steps,
                        end: End::Branch { condition, arms },
                    });
                }
                BlockExit::NativeCall => {
                    let Some(outcome) = block
                        .native_call
                        .as_ref()
                        .or_else(|| input.native_calls.get(&key))
                    else {
                        return refuse(key, "no proven call frame for this native call");
                    };
                    if !outcome.stack_argument_offsets.is_empty() {
                        return refuse(key, "stack-passed call arguments are not bridged");
                    }
                    let pre = report.continuation_state();
                    let vip = block
                        .instructions
                        .last()
                        .map_or(block.start.vip, |last| last.state_after.vip);
                    let call = execute_symbolic_native_call_boundary_with_initial_state(
                        vip,
                        outcome,
                        input.limits,
                        &pre,
                        input.memory,
                    )
                    .map_err(|error| DecodedLiftError::Cfg {
                        block: key,
                        reason: format!("native-call effect failed: {error}"),
                    })?;
                    let [symbolic_call] = &call.native_calls[..] else {
                        return refuse(key, "native-call boundary did not yield exactly one call");
                    };
                    if call.fresh_allocations.len() != pre.fresh_allocations.len()
                        || call.memory_tainted
                    {
                        return refuse(key, "native call allocates or taints memory");
                    }
                    let invocation = &symbolic_call.invocation;
                    if invocation.arguments.len() > MAX_CALL_ARGUMENTS {
                        return refuse(key, "native call passes more than four register arguments");
                    }
                    let ExprKind::Intrinsic {
                        name: result_name, ..
                    } = &symbolic_call.applied_effect.result.kind
                    else {
                        return refuse(key, "native-call result is not an opaque value");
                    };
                    if symbolic_call.applied_effect.result.width != BitWidth::W64 {
                        return refuse(key, "native-call result is not 64 bits");
                    }
                    let target =
                        self.prepare(key, &invocation.target.value, path.barrier, &path, &mut steps)?;
                    let mut arguments = Vec::with_capacity(invocation.arguments.len());
                    for argument in &invocation.arguments {
                        arguments.push(self.prepare(key, &argument.value, path.barrier, &path, &mut steps)?);
                    }
                    let post = match &symbolic_call.applied_effect.memory {
                        NativeCallMemoryDecision::PreserveTracked => call.continuation_state(),
                        NativeCallMemoryDecision::PreserveTrackedAndCreateFreshAllocation {
                            ..
                        } => return refuse(key, "allocator calls are not bridged in the CFG lift"),
                        NativeCallMemoryDecision::HavocTracked { .. } => {
                            if path.root_escaped
                                || invocation
                                    .arguments
                                    .iter()
                                    .map(|argument| &argument.value)
                                    .chain(std::iter::once(&invocation.target.value))
                                    .any(|value| value.dependencies().contains("vsp_initial"))
                            {
                                return refuse(
                                    key,
                                    "a pointer into the VM stack may reach the callee, so the VM state cannot be assumed to survive it",
                                );
                            }
                            self.assume(LiftAssumption::VmStateSurvivesNativeCall);
                            restore_vm_state(&pre, &call)
                        }
                    };
                    let index = self.call_sites;
                    self.call_sites += 1;
                    steps.push(Step {
                        epoch: call.memory_epoch,
                        kind: StepKind::Call(CallSite {
                            index,
                            target,
                            arguments,
                        }),
                    });
                    path.results
                        .insert(result_name.clone(), call_result_name(index));
                    path.barrier = path.barrier.max(call.memory_epoch);
                    path.call_barrier = path.barrier;
                    key = self.call_return_block(key)?;
                    state = post;
                }
                BlockExit::Stalled => unreachable!("stalled blocks are refused above"),
            }
        }
    }

    /// Lifts a reducible natural loop reached at its header with the exact
    /// pre-loop state. One exact iteration identifies the loop-carried cells; a
    /// second iteration over fresh `phi` inputs recovers their closed per-
    /// iteration update and the exit condition, proving the body closes over the
    /// `phi`s alone. Anything not provably a clean single-exit side-effect-free
    /// natural loop fails closed.
    fn lift_loop(
        &mut self,
        header: Va,
        pre_state: SymbolicVmInitialState,
        path: Path,
        mut steps: Vec<Step>,
        body_set: &BTreeSet<Va>,
    ) -> Result<Node, DecodedLiftError> {
        for &node in body_set {
            self.blocks.insert(node);
        }
        // Iteration one, exact: the cells whose value differs between the loop
        // entry and the back-edge re-entry are the loop-carried variables.
        let exact = self.run_loop_body(header, body_set, pre_state.clone(), &path, false)?;
        let cells = loop_cells(header, &pre_state, &exact.reentry)?;

        // Iteration two, over a fresh phi per loop-carried cell.
        let mut phi_state = pre_state.clone();
        let mut specs: Vec<(Cell, u16, BitWidth)> = Vec::new();
        for (cell, width) in cells {
            let id = self.loop_phis;
            self.loop_phis = self
                .loop_phis
                .checked_add(1)
                .ok_or_else(|| DecodedLiftError::Cfg {
                    block: header,
                    reason: "too many loop-carried cells".to_owned(),
                })?;
            let name = loop_phi_name(id);
            self.active_phis.insert(name.clone());
            set_cell(&mut phi_state, &cell, Expr::register(width, name));
            specs.push((cell, id, width));
        }

        // Phi inits run before the loop, in the pre-header node this `steps` feeds
        // and over the pre-loop path, so they are recovered before the body's node
        // context is established below.
        let mut inits = Vec::with_capacity(specs.len());
        for (cell, _, _) in &specs {
            let init_value = get_cell(&pre_state, cell).ok_or_else(|| DecodedLiftError::Cfg {
                block: header,
                reason: "loop-carried cell has no pre-loop value".to_owned(),
            })?;
            inits.push(self.prepare(header, init_value, path.barrier, &path, &mut steps)?);
        }

        // Iteration two also prepares the body's stores into `body_steps`, over
        // the fresh phi inputs, and threads them through `body_path`.
        let run = self.run_loop_body(header, body_set, phi_state.clone(), &path, true)?;

        // The body must change exactly the recognized loop-carried cells and
        // leave every other cell at its invariant value, or a carried value went
        // unrecognized: fail closed.
        verify_closure(header, &phi_state, &run.reentry, &specs)?;

        if run.exits.iter().any(|exit| exit.guarded)
            && self
                .input
                .cfg
                .assumptions
                .contains(&VmCfgAssumption::PreferredImageBase)
        {
            self.assume(LiftAssumption::PreferredImageBase);
        }

        // The per-iteration `next` phi values and every exit `condition` are
        // recovered in the body node's context (its entry epoch and the stores
        // threaded through the body) so a read a body store precedes is captured
        // in order rather than re-issued unsoundly. `next` is read at the end of
        // the iteration (`body_path`, every store); each `condition` at its exit
        // branch (its own `path`, only the stores that precede it). All of this
        // happens before any exit continuation is walked below, since walking
        // resets the node context the prepares depend on.
        let body_path = run.body_path;
        let mut body_steps = run.body_steps;
        self.node_base_epoch = phi_state.memory_epoch;
        self.node_call_barrier = path.call_barrier;
        let mut phis = Vec::with_capacity(specs.len());
        for ((cell, id, width), init) in specs.iter().zip(inits) {
            let next_value = get_cell(&run.reentry, cell).ok_or_else(|| DecodedLiftError::Cfg {
                block: header,
                reason: "loop-carried cell has no re-entry value".to_owned(),
            })?;
            if next_value.width != *width {
                return refuse(header, "a loop-carried cell changes width");
            }
            let next =
                self.prepare(header, next_value, body_path.barrier, &body_path, &mut body_steps)?;
            phis.push(LoopPhi {
                id: *id,
                init,
                next,
            });
        }
        let mut conditions = Vec::with_capacity(run.exits.len());
        for pending in &run.exits {
            conditions.push(self.prepare(
                header,
                &pending.condition,
                pending.path.barrier,
                &pending.path,
                &mut body_steps,
            )?);
            self.branches += 1;
        }
        self.loops += 1;
        // Each exit continuation runs when the loop is left there, so it inherits
        // only the stores that precede its branch (via its own `path`), not those
        // on the body's tail past it.
        let mut exits = Vec::with_capacity(run.exits.len());
        for (pending, condition) in run.exits.into_iter().zip(conditions) {
            let continuation = self.walk(pending.exit_key, pending.exit_state, pending.path, Vec::new())?;
            exits.push(LoopExit {
                epoch: pending.epoch,
                condition,
                continue_value: pending.continue_value,
                continuation: Box::new(continuation),
            });
        }
        Ok(Node {
            steps,
            end: End::Loop(LoopLeaf {
                phis,
                steps: body_steps,
                exits,
            }),
        })
    }

    /// Runs one iteration of a loop body from `start`, following single in-loop
    /// successors and the loop's exit branches, back to the header. The body is a
    /// single cycle: each block has exactly one in-loop successor it follows, but
    /// may also branch out of the loop (an exit). Every exit is collected in body
    /// order. Refuses any side effect, non-branch block, in-loop forking, or
    /// branching the structured form forbids.
    ///
    /// When `collect` is set this is the emitting iteration: the body's external
    /// stores are prepared (address and value over the current `phi`s and entry
    /// inputs) into `body_steps` in native order and threaded through `work_path`,
    /// so the returned [`BodyRun`] carries exactly what each iteration must
    /// reproduce, tagged by epoch so the emitter can tell the stores that run
    /// every iteration from those taken only on the back-edge. Any native call,
    /// allocation, or store that could alias tracked VM memory (havoc/taint) is
    /// not cleanly re-expressible per iteration and fails closed.
    fn run_loop_body(
        &mut self,
        header: Va,
        body_set: &BTreeSet<Va>,
        start: SymbolicVmInitialState,
        path: &Path,
        collect: bool,
    ) -> Result<BodyRun, DecodedLiftError> {
        let input = self.input;
        let mut key = header;
        if collect {
            // The whole body is one emitted node: its reads order against the
            // state it enters with and the stores it makes, never reordered past
            // an ancestor call.
            self.node_base_epoch = start.memory_epoch;
            self.node_call_barrier = path.call_barrier;
        }
        let mut work_path = path.clone();
        let mut body_steps: Vec<Step> = Vec::new();
        let mut state = start;
        let mut exits: Vec<PendingExit> = Vec::new();
        let mut budget = body_set.len() + 1;
        loop {
            budget = budget.checked_sub(1).ok_or_else(|| DecodedLiftError::Cfg {
                block: key,
                reason: "loop body is not a simple cycle".to_owned(),
            })?;
            let Some(block) = input.cfg.blocks.get(&key) else {
                return refuse(key, "loop body block is not in the recovered CFG");
            };
            if block.exit != BlockExit::Branch {
                return refuse(key, "loop body block does not end in a branch");
            }
            if let Some((index, semantic, reason)) = first_unliftable(&block.instructions) {
                return refuse(key, format!("instruction {index} ({semantic:?}): {reason}"));
            }
            let report = execute_symbolic_linear_block_with_initial_state(
                &block.instructions,
                Some(VmHandlerSemantic::UnconditionalBranch),
                input.limits,
                &state,
                input.memory,
            )
            .map_err(|error| DecodedLiftError::Cfg {
                block: key,
                reason: format!("symbolic execution failed: {error}"),
            })?;
            // A native call, allocation, or any store that could alias tracked VM
            // memory (havoc/taint) is not re-expressible as a clean per-iteration
            // effect: fail closed. Precise external stores are emitted instead.
            if !report.memory_havocs.is_empty()
                || !report.native_calls.is_empty()
                || !report.fresh_allocations.is_empty()
                || report.memory_tainted
            {
                return refuse(key, "loop body calls natively, allocates, or havocs memory");
            }
            if collect && !report.external_writes.is_empty() {
                let entry_barrier = work_path.barrier;
                for (index, write) in report.external_writes.iter().enumerate() {
                    let barrier = if index == 0 {
                        entry_barrier
                    } else {
                        report.memory_epoch
                    };
                    if write.value.dependencies().contains("vsp_initial") {
                        work_path.root_escaped = true;
                    }
                    let address =
                        self.prepare(key, &write.address, barrier, &work_path, &mut body_steps)?;
                    let value =
                        self.prepare(key, &write.value, barrier, &work_path, &mut body_steps)?;
                    let width = value.width;
                    work_path.stores.push(PathStore {
                        end_epoch: write.epoch,
                        address: write.address.clone(),
                        bytes: write.value.width.bytes(),
                    });
                    body_steps.push(Step {
                        epoch: write.epoch,
                        kind: StepKind::Store(EffectStore {
                            address,
                            value,
                            width,
                        }),
                    });
                }
                work_path.barrier = work_path.barrier.max(report.memory_epoch);
            }
            self.expanded += 1;
            let next = report.continuation_state();
            let (successors, guarded) = self.branch_successors(key)?;
            let (inside, outside): (Vec<_>, Vec<_>) = successors
                .iter()
                .partition(|(_, successor)| body_set.contains(successor));
            match (inside.as_slice(), outside.as_slice()) {
                ([(_, continue_key)], []) => {
                    if *continue_key == header {
                        if exits.is_empty() {
                            return refuse(key, "loop has no exit branch");
                        }
                        return Ok(BodyRun {
                            reentry: next,
                            exits,
                            body_steps,
                            body_path: work_path,
                        });
                    }
                    key = *continue_key;
                    state = next;
                }
                ([(continue_value, continue_key)], [(_, exit_key)]) => {
                    if successors.len() != 2 {
                        return refuse(key, "loop exit branch is not two-way");
                    }
                    let Some(target) = report.branch_target.as_ref() else {
                        return refuse(key, "loop exit branch has no symbolic target");
                    };
                    // Each exit is a conditional break reproduced in turn; the
                    // body stays a single cycle, so epochs rise in body order.
                    exits.push(PendingExit {
                        condition: target.clone(),
                        continue_value: *continue_value,
                        exit_key: *exit_key,
                        exit_state: next.clone(),
                        guarded,
                        epoch: report.memory_epoch,
                        path: work_path.clone(),
                    });
                    if *continue_key == header {
                        return Ok(BodyRun {
                            reentry: next,
                            exits,
                            body_steps,
                            body_path: work_path,
                        });
                    }
                    key = *continue_key;
                    state = next;
                }
                _ => return refuse(key, "loop body has unsupported branching"),
            }
        }
    }

    /// The state the nested entry starts from after the exit at `key` returns
    /// into it, or `None` when the resolver cannot prove the entry (the exit is
    /// then lowered as an ordinary exit).
    fn compose_nested(
        &self,
        key: Va,
        to: Va,
        report: &SymbolicVmExecutionReport,
        layout: &VmExitReport,
        resolver: &dyn NestedEntryResolver,
    ) -> Result<Option<SymbolicVmInitialState>, DecodedLiftError> {
        let fail = |reason: String| DecodedLiftError::Cfg { block: key, reason };
        let exit = recover_native_exit_state(report, layout)
            .map_err(|error| fail(format!("nested exit state unrecovered: {error}")))?;
        let Some(entry) = resolver.resolve(&exit.return_address) else {
            return Ok(None);
        };
        if self
            .input
            .cfg
            .blocks
            .get(&to)
            .and_then(|nested| nested.address)
            != Some(entry.bytecode_start)
        {
            return refuse(
                key,
                "the resolved nested entry is not the CFG's nested block",
            );
        }
        compose_nested_state(report, &exit, &entry)
            .map(Some)
            .map_err(|error| fail(format!("nested entry composition failed: {error}")))
    }

    fn assume(&mut self, assumption: LiftAssumption) {
        if !self.assumptions.contains(&assumption) {
            self.assumptions.push(assumption);
        }
    }
}

/// Re-bases the nested entry's frame onto the exact state the exit left: its
/// stack sits where the exit's `ret` leaves the stack pointer, its registers and
/// flags are what the exit restored, and the outer memory carries over.
fn compose_nested_state(
    prior: &SymbolicVmExecutionReport,
    exit: &NativeExitState,
    entry: &NestedEntry,
) -> Result<SymbolicVmInitialState, String> {
    let error = |error: vmpx_vm::ExprError| error.to_string();
    let frame_bytes = i64::try_from(entry.frame.len() * 8).map_err(|e| e.to_string())?;
    let delta = i64::from(exit.frame_size) - frame_bytes;
    let vsp = Expr::binary(
        BinaryOp::Add,
        prior.final_vsp.clone(),
        Expr::constant(BitWidth::W64, delta.cast_unsigned()),
    )
    .map_err(error)?;
    let mut replacements = BTreeMap::from([
        ("vsp_initial".to_owned(), vsp),
        ("native_rflags_initial".to_owned(), exit.rflags.clone()),
    ]);
    if let Some(continuation) = entry.continuation {
        replacements.insert(
            "native_return_address_initial".to_owned(),
            Expr::constant(BitWidth::W64, continuation.0),
        );
    }
    for register in X64NativeRegister::ALL {
        let value = exit
            .registers
            .get(&register)
            .ok_or_else(|| format!("exit restores no {register:?}"))?;
        replacements.insert(register.symbolic_name().to_owned(), value.clone());
    }
    let (frame, _) = frame_initial_state(&entry.frame, entry.top_offset);
    let mut initial = frame.substitute(&replacements).map_err(error)?;

    let mut bytes: Vec<(Expr, Expr)> = Vec::new();
    let mut index = HashMap::<Expr, usize>::new();
    for (address, value) in prior.memory_values.iter().chain(&initial.memory_values) {
        for byte in 0..value.width.bytes() {
            let offset = Expr::binary(
                BinaryOp::Add,
                address.clone(),
                Expr::constant(BitWidth::W64, u64::from(byte)),
            )
            .map_err(error)?;
            let address = canonicalize_symbolic_pointer(&offset);
            let value = Expr::extract(BitWidth::W8, byte * 8, value.clone()).map_err(error)?;
            if let Some(&slot) = index.get(&address) {
                bytes[slot].1 = value;
            } else {
                index.insert(address.clone(), bytes.len());
                bytes.push((address, value));
            }
        }
    }
    initial.memory_values = bytes;
    initial.memory_epoch = prior.memory_epoch;
    initial.memory_tainted = prior.memory_tainted;
    initial.next_native_call_epoch = prior.next_native_call_epoch;
    initial
        .fresh_allocations
        .clone_from(&prior.fresh_allocations);
    Ok(initial)
}

/// State after an opaque native call that cannot reach the VM stack: the
/// call's own result cell replaces what it covers, every other VM-stack cell and
/// every context cell keeps its pre-call value, and memory outside the VM stack
/// stays forgotten.
fn restore_vm_state(
    pre: &SymbolicVmInitialState,
    call: &SymbolicVmExecutionReport,
) -> SymbolicVmInitialState {
    let mut cells: HashMap<Expr, Expr> = pre
        .memory_values
        .iter()
        .filter(|(address, _)| symbolic_pointer_offset(address, "vsp_initial").is_some())
        .cloned()
        .collect();
    cells.extend(call.memory_values.iter().cloned());
    let mut post = call.continuation_state();
    post.memory_values = cells.into_iter().collect();
    post.memory_values
        .sort_by_cached_key(|(address, _)| format!("{address:?}"));
    post.context_values = pre.context_values.clone();
    post
}

/// How far the path expansion gets on a possibly incomplete CFG. Nothing is
/// emitted; this is a diagnostic of where a real lift would stop.
#[derive(Debug)]
pub struct CfgProbe {
    pub blocks: BTreeSet<Va>,
    pub expanded_blocks: usize,
    pub exit_paths: usize,
    pub conditional_branches: usize,
    pub native_calls: usize,
    pub stop: Option<DecodedLiftError>,
}

/// Expands the CFG into paths without requiring it to be complete and without
/// emitting code.
#[must_use]
pub fn probe_decoded_cfg(input: &DecodedCfgInput<'_>) -> CfgProbe {
    let (initial, _) = frame_initial_state(input.frame, input.top_offset);
    let mut builder = Builder {
        input,
        nested: None,
        nested_entries: 0,
        expanded: 0,
        blocks: BTreeSet::new(),
        exit_paths: 0,
        region_exits: 0,
        retained_bytes: 0,
        branches: 0,
        loops: 0,
        call_sites: 0,
        read_sites: 0,
        uses_root: false,
        assumptions: Vec::new(),
        loop_bodies: recover_natural_loops(input.cfg),
        loop_phis: 0,
        active_phis: BTreeSet::new(),
        node_base_epoch: 0,
        node_call_barrier: 0,
        ancestors: Vec::new(),
    };
    let stop = builder
        .walk(input.cfg.entry, initial, Path::default(), Vec::new())
        .err();
    CfgProbe {
        blocks: builder.blocks,
        expanded_blocks: builder.expanded,
        exit_paths: builder.exit_paths,
        conditional_branches: builder.branches,
        native_calls: builder.call_sites,
        stop,
    }
}

/// Lifts a fully recovered branching function to one native replacement.
///
/// # Errors
///
/// Refuses an incomplete CFG, a loop, any branch whose targets are not proven
/// recovered edges, any condition, value or native call the backend cannot
/// lower, and any block without a verified lowering. No partial function is
/// ever returned.
pub fn lift_decoded_cfg(
    input: &DecodedCfgInput<'_>,
    address: CodeAddress,
) -> Result<LiftedCfgFunction, DecodedLiftError> {
    lift_with_nested(input, None, address)
}

/// [`lift_decoded_cfg`] that also follows `NestedEntry` edges the resolver
/// proves, lowering the nested region inline instead of returning into its
/// entry stub. An edge the resolver cannot prove keeps the ordinary exit.
///
/// # Errors
///
/// As [`lift_decoded_cfg`]; additionally refuses a resolved entry that is not
/// the CFG's nested block or whose state cannot be composed.
pub fn lift_decoded_cfg_with_nested(
    input: &DecodedCfgInput<'_>,
    resolver: &dyn NestedEntryResolver,
    address: CodeAddress,
) -> Result<LiftedCfgFunction, DecodedLiftError> {
    lift_with_nested(input, Some(resolver), address)
}

fn lift_with_nested(
    input: &DecodedCfgInput<'_>,
    nested: Option<&dyn NestedEntryResolver>,
    address: CodeAddress,
) -> Result<LiftedCfgFunction, DecodedLiftError> {
    let cfg = input.cfg;
    if !cfg.is_complete() {
        return refuse(
            cfg.entry,
            format!(
                "the CFG is incomplete ({} stalls, limit hit: {:?})",
                cfg.stalls.len(),
                cfg.limit_hit
            ),
        );
    }
    let (initial, _) = frame_initial_state(input.frame, input.top_offset);
    let mut builder = Builder {
        input,
        nested,
        nested_entries: 0,
        expanded: 0,
        blocks: BTreeSet::new(),
        exit_paths: 0,
        region_exits: 0,
        retained_bytes: 0,
        branches: 0,
        loops: 0,
        call_sites: 0,
        read_sites: 0,
        uses_root: false,
        assumptions: Vec::new(),
        loop_bodies: recover_natural_loops(cfg),
        loop_phis: 0,
        active_phis: BTreeSet::new(),
        node_base_epoch: 0,
        node_call_barrier: 0,
        ancestors: Vec::new(),
    };
    let root = builder.walk(cfg.entry, initial, Path::default(), Vec::new())?;

    let config = ReplacementConfig {
        inputs: X64NativeRegister::ALL
            .iter()
            .map(|register| (register.symbolic_name().to_owned(), physical_for(register)))
            .collect(),
        max_expression_nodes: 200_000,
        max_spill_slots: 64,
        import_calls: BTreeMap::new(),
        continuation: None,
        effects: Vec::new(),
        effect_calls: Vec::new(),
        rflags: None,
        symbol_rewrites: BTreeMap::new(),
    };
    let frame_bytes = builder
        .uses_root
        .then(|| u32::try_from(input.frame.len() * 8).unwrap_or(u32::MAX));
    let region = (builder.region_exits > 0).then_some(builder.retained_bytes);
    let replacement = build_tree_replacement(
        &root,
        builder.call_sites,
        builder.read_sites,
        builder.loop_phis,
        frame_bytes,
        region,
        &config,
        address,
    )?;
    Ok(LiftedCfgFunction {
        replacement,
        blocks: builder.blocks,
        expanded_blocks: builder.expanded,
        exit_paths: builder.exit_paths,
        conditional_branches: builder.branches,
        loops: builder.loops,
        native_calls: builder.call_sites,
        nested_entries: builder.nested_entries,
        assumptions: builder.assumptions,
    })
}

struct Emitter<'c> {
    config: &'c ReplacementConfig,
    inputs: Vec<ExpressionInput>,
    canonical: BTreeMap<PhysicalRegister, VirtualRegister>,
    assignments: Vec<RegisterAssignment>,
    next_register: u32,
    window_base: u16,
    shared_end: u16,
    trust: Trust,
    operations: Vec<Operation>,
    next_label: u32,
    call_result_base: u16,
    /// Base slot of the in-place captured-read results.
    read_result_base: u16,
    result_base: u16,
    scratch_base: u16,
    /// Spill slot holding the flags the function was entered with; present
    /// only when a region exit must hand them to its continuation.
    flags_slot: Option<SpillSlot>,
    /// Base slot of the loop `phi` variables (live across back-edges).
    phi_base: u16,
    /// Base slot of the per-iteration staging slots for loop `phi` updates.
    phi_temp_base: u16,
}

impl Emitter<'_> {
    fn register(&mut self, physical: PhysicalRegister) -> VirtualRegister {
        canonical_register(
            physical,
            &mut self.canonical,
            &mut self.assignments,
            &mut self.next_register,
        )
    }

    fn lower(&mut self, expression: &Expr) -> Result<VirtualRegister, ReplacementError> {
        let bound = rewrite_symbols(expression, &self.config.symbol_rewrites)?;
        self.window_base = self.shared_end;
        let (operations, result) = lower_fragment_with(
            &bound,
            &self.inputs,
            &[],
            self.config,
            &mut self.canonical,
            &mut self.assignments,
            &mut self.next_register,
            &mut self.window_base,
            &self.shared_end,
            &mut self.trust,
            true,
        )?;
        self.operations.extend(operations);
        Ok(result)
    }

    fn scratch(&self, index: u16) -> SpillSlot {
        SpillSlot(self.scratch_base + index)
    }

    fn emit_store(&mut self, store: &EffectStore) -> Result<(), ReplacementError> {
        let address = self.lower(&store.address)?;
        let held = self.scratch(0);
        self.operations.push(Operation::SpillStore {
            slot: held,
            source: address,
        });
        let value = self.lower(&store.value)?;
        let base = self.register(PhysicalRegister::R11);
        self.operations.push(Operation::SpillLoad {
            destination: base,
            slot: held,
        });
        let width = match store.width {
            BitWidth::W8 => Width::W8,
            BitWidth::W16 => Width::W16,
            BitWidth::W32 => Width::W32,
            BitWidth::W64 => Width::W64,
        };
        self.operations.push(Operation::Store {
            address: Address::new(base, 0),
            source: VirtualRegister::new(value.id, width),
        });
        Ok(())
    }

    /// Emits a captured read: loads memory at its native-order position and
    /// spills the (zero-extended) value into the read's result slot for a later
    /// use to read back. See [`Builder::capture_read`].
    fn emit_load(&mut self, read: &LoadRead) -> Result<(), ReplacementError> {
        let load = Expr::load(read.width, read.address.clone());
        let widened = if read.width == BitWidth::W64 {
            load
        } else {
            Expr::resize(ResizeOp::ZeroExtend, BitWidth::W64, load)
                .map_err(|error| ReplacementError::EffectBindings(error.to_string()))?
        };
        let value = self.lower(&widened)?;
        let index = u16::try_from(read.index).map_err(|_| ReplacementError::TooManyInputs)?;
        self.operations.push(Operation::SpillStore {
            slot: SpillSlot(self.read_result_base + index),
            source: value,
        });
        Ok(())
    }

    fn emit_call(&mut self, call: &CallSite) -> Result<(), ReplacementError> {
        const ARGUMENT_REGISTERS: [PhysicalRegister; MAX_CALL_ARGUMENTS] = [
            PhysicalRegister::Rcx,
            PhysicalRegister::Rdx,
            PhysicalRegister::R8,
            PhysicalRegister::R9,
        ];
        for (index, argument) in call.arguments.iter().enumerate() {
            let value = self.lower(argument)?;
            let slot = self.scratch(1 + u16::try_from(index).unwrap_or(u16::MAX));
            self.operations.push(Operation::SpillStore {
                slot,
                source: value,
            });
        }
        let target = self.lower(&call.target)?;
        let target_slot = self.scratch(0);
        self.operations.push(Operation::SpillStore {
            slot: target_slot,
            source: target,
        });
        let mut registers = Vec::with_capacity(call.arguments.len());
        for (index, physical) in ARGUMENT_REGISTERS
            .iter()
            .take(call.arguments.len())
            .enumerate()
        {
            let destination = self.register(*physical);
            let slot = self.scratch(1 + u16::try_from(index).unwrap_or(u16::MAX));
            self.operations
                .push(Operation::SpillLoad { destination, slot });
            registers.push(destination);
        }
        let callee = self.register(PhysicalRegister::R10);
        self.operations.push(Operation::SpillLoad {
            destination: callee,
            slot: target_slot,
        });
        let result = self.register(PhysicalRegister::Rax);
        self.operations.push(Operation::Call {
            target: CallTarget::Register(callee),
            arguments: registers,
            result: Some(result),
        });
        let index = u16::try_from(call.index).map_err(|_| ReplacementError::TooManyInputs)?;
        self.operations.push(Operation::SpillStore {
            slot: SpillSlot(self.call_result_base + index),
            source: result,
        });
        Ok(())
    }

    fn emit_exit(&mut self, leaf: &ExitLeaf) -> Result<(), ReplacementError> {
        self.stash_exit_registers(&leaf.registers)?;
        // Materialise the carried flags after the register stash clobbers them and
        // before the flag-preserving load and popping return.
        if let Some(rflags) = &leaf.rflags {
            let value = self.lower(rflags)?;
            self.operations.push(Operation::RestoreFlags { value });
        }
        self.load_exit_registers();
        self.operations.push(Operation::ReturnPopping {
            bytes: leaf.return_pop,
        });
        Ok(())
    }

    /// Stages the continuation's stack in the top of the frame, restores the
    /// entry flags, loads the exit registers and returns through the staged
    /// target. The frame keeps those top bytes free of spill slots.
    fn emit_region(&mut self, leaf: &RegionLeaf) -> Result<(), ReplacementError> {
        self.stash_exit_registers(&leaf.registers)?;
        for (index, word) in leaf.staged.iter().enumerate() {
            let offset = u64::from(leaf.slot_offset) + 8 * index as u64;
            let address = Expr::binary(
                BinaryOp::Add,
                Expr::register(BitWidth::W64, "vsp_initial"),
                Expr::constant(BitWidth::W64, offset),
            )
            .map_err(|error| ReplacementError::EffectBindings(error.to_string()))?;
            self.emit_store(&EffectStore {
                address,
                value: word.clone(),
                width: BitWidth::W64,
            })?;
        }
        let value = if let Some(rflags) = &leaf.rflags {
            self.lower(rflags)?
        } else {
            let slot = self.flags_slot.ok_or_else(|| {
                ReplacementError::EffectBindings(
                    "region exit without a captured flag state".to_owned(),
                )
            })?;
            let value = self.register(PhysicalRegister::R10);
            self.operations.push(Operation::SpillLoad {
                destination: value,
                slot,
            });
            value
        };
        self.operations.push(Operation::RestoreFlags { value });
        self.load_exit_registers();
        let bytes = u32::try_from(leaf.staged.len() * 8).unwrap_or(u32::MAX);
        self.operations.push(Operation::ReturnRetaining { bytes });
        Ok(())
    }

    fn stash_exit_registers(
        &mut self,
        registers: &BTreeMap<X64NativeRegister, Expr>,
    ) -> Result<(), ReplacementError> {
        for (index, register) in X64NativeRegister::ALL.iter().enumerate() {
            let expression = registers.get(register).ok_or(ReplacementError::Lowering {
                register: *register,
                message: "the exit state has no value for this register".to_owned(),
            })?;
            let value = self.lower(expression).map_err(|error| match error {
                ReplacementError::Effect(inner) => ReplacementError::Lowering {
                    register: *register,
                    message: inner.to_string(),
                },
                other => other,
            })?;
            let slot = SpillSlot(self.result_base + u16::try_from(index).unwrap_or(u16::MAX));
            self.operations.push(Operation::SpillStore {
                slot,
                source: value,
            });
        }
        Ok(())
    }

    fn load_exit_registers(&mut self) {
        for (index, register) in X64NativeRegister::ALL.iter().enumerate() {
            let destination = self.register(physical_for(register));
            let slot = SpillSlot(self.result_base + u16::try_from(index).unwrap_or(u16::MAX));
            self.operations
                .push(Operation::SpillLoad { destination, slot });
        }
    }

    fn emit_step(&mut self, step: &Step) -> Result<(), ReplacementError> {
        match &step.kind {
            StepKind::Store(store) => self.emit_store(store),
            StepKind::Call(call) => self.emit_call(call),
            StepKind::Load(read) => self.emit_load(read),
        }
    }

    fn emit_steps(&mut self, steps: &[Step]) -> Result<(), ReplacementError> {
        for step in steps {
            self.emit_step(step)?;
        }
        Ok(())
    }

    fn emit_node(&mut self, node: &Node) -> Result<(), ReplacementError> {
        self.emit_steps(&node.steps)?;
        match &node.end {
            End::Exit(leaf) => self.emit_exit(leaf),
            End::Region(leaf) => self.emit_region(leaf),
            End::Branch { condition, arms } => {
                let Some((last, taken)) = arms.split_last() else {
                    return Err(ReplacementError::EffectBindings(
                        "branch has no arms".to_owned(),
                    ));
                };
                let selector = self.lower(condition)?;
                let held = self.scratch(0);
                self.operations.push(Operation::SpillStore {
                    slot: held,
                    source: selector,
                });
                let mut labels = Vec::with_capacity(taken.len());
                for (value, _) in taken {
                    let selector = self.register(PhysicalRegister::R10);
                    let constant = self.register(PhysicalRegister::R11);
                    self.operations.push(Operation::SpillLoad {
                        destination: selector,
                        slot: held,
                    });
                    self.operations.push(Operation::MoveImmediate {
                        destination: constant,
                        value: *value,
                    });
                    self.operations.push(Operation::Compare {
                        left: selector,
                        right: BinarySource::Register(constant),
                    });
                    let label = Label(self.next_label);
                    self.next_label += 1;
                    self.operations.push(Operation::BranchIf {
                        condition: Condition::Equal,
                        target: label,
                    });
                    labels.push(label);
                }
                self.emit_node(&last.1)?;
                for (label, (_, arm)) in labels.into_iter().zip(taken) {
                    self.operations.push(Operation::Label(label));
                    self.emit_node(arm)?;
                }
                Ok(())
            }
            End::Loop(leaf) => self.emit_loop(leaf),
        }
    }

    /// Emits a natural loop. The `phi` slots are seeded before the header label;
    /// each iteration runs the body interleaved with its exit branches, in epoch
    /// order. An exit branch stays in the loop (falls through to the next segment)
    /// when the selector picks the continue address and leaves to that exit's
    /// continuation otherwise. When no exit fires, the `phi` slots are updated
    /// from their per-iteration values and control jumps back to the header. Each
    /// exit's continuation is emitted past the back-edge and reached only by its
    /// own branch (it ends in a transfer, so control never falls between them).
    fn emit_loop(&mut self, leaf: &LoopLeaf) -> Result<(), ReplacementError> {
        for phi in &leaf.phis {
            let value = self.lower(&phi.init)?;
            self.operations.push(Operation::SpillStore {
                slot: SpillSlot(self.phi_base + phi.id),
                source: value,
            });
        }
        let header = Label(self.next_label);
        self.next_label += 1;
        self.operations.push(Operation::Label(header));
        // One forward label per exit; its continuation is emitted past the
        // back-edge below.
        let mut exit_labels = Vec::with_capacity(leaf.exits.len());
        for _ in &leaf.exits {
            let label = Label(self.next_label);
            self.next_label += 1;
            exit_labels.push(label);
        }
        // Body steps interleave with the exit branches by epoch: before each
        // exit's branch, emit every not-yet-emitted step committing at or before
        // its epoch (so it runs on the iteration that takes that exit); steps past
        // the last exit run only when the back-edge is taken.
        let mut emitted = 0;
        for (exit, &label) in leaf.exits.iter().zip(&exit_labels) {
            while emitted < leaf.steps.len() && leaf.steps[emitted].epoch <= exit.epoch {
                self.emit_step(&leaf.steps[emitted])?;
                emitted += 1;
            }
            let selector = self.lower(&exit.condition)?;
            let held = self.scratch(0);
            self.operations.push(Operation::SpillStore {
                slot: held,
                source: selector,
            });
            let selector = self.register(PhysicalRegister::R10);
            let constant = self.register(PhysicalRegister::R11);
            self.operations.push(Operation::SpillLoad {
                destination: selector,
                slot: held,
            });
            self.operations.push(Operation::MoveImmediate {
                destination: constant,
                value: exit.continue_value,
            });
            self.operations.push(Operation::Compare {
                left: selector,
                right: BinarySource::Register(constant),
            });
            // Leave through this exit when the selector is not the continue
            // address; otherwise fall through to the next segment.
            self.operations.push(Operation::BranchIf {
                condition: Condition::NotEqual,
                target: label,
            });
        }
        // Back-edge taken: run the body's continue-only (post-last-exit) steps,
        // then update phis and jump back.
        while emitted < leaf.steps.len() {
            self.emit_step(&leaf.steps[emitted])?;
            emitted += 1;
        }
        // Stage every next value from the current phis before overwriting any.
        for phi in &leaf.phis {
            let value = self.lower(&phi.next)?;
            self.operations.push(Operation::SpillStore {
                slot: SpillSlot(self.phi_temp_base + phi.id),
                source: value,
            });
        }
        for phi in &leaf.phis {
            let moved = self.register(PhysicalRegister::R10);
            self.operations.push(Operation::SpillLoad {
                destination: moved,
                slot: SpillSlot(self.phi_temp_base + phi.id),
            });
            self.operations.push(Operation::SpillStore {
                slot: SpillSlot(self.phi_base + phi.id),
                source: moved,
            });
        }
        self.operations.push(Operation::Jump(header));
        // Each exit's continuation, reached only by its branch above.
        for (exit, label) in leaf.exits.iter().zip(exit_labels) {
            self.operations.push(Operation::Label(label));
            self.emit_node(&exit.continuation)?;
        }
        Ok(())
    }
}

/// Emits the path tree as one function: capture the entry registers, then walk
/// the tree, jumping to a labelled arm on a matching branch target and falling
/// through into the last. Every leaf restores the exit registers and returns.
fn build_tree_replacement(
    root: &Node,
    call_count: usize,
    read_count: usize,
    loop_phis: u16,
    stack_root_bytes: Option<u32>,
    region_retained: Option<u32>,
    config: &ReplacementConfig,
    address: CodeAddress,
) -> Result<Replacement, ReplacementError> {
    let extended;
    let config = if let Some(bytes) = stack_root_bytes {
        let entry = Expr::register(BitWidth::W64, ENTRY_STACK_POINTER);
        let root_pointer = Expr::binary(
            BinaryOp::Sub,
            entry,
            Expr::constant(BitWidth::W64, u64::from(bytes)),
        )
        .map_err(|error| ReplacementError::EffectBindings(error.to_string()))?;
        let mut owned = config.clone();
        owned
            .inputs
            .push((ENTRY_STACK_POINTER.to_owned(), PhysicalRegister::Rax));
        owned
            .symbol_rewrites
            .insert("vsp_initial".to_owned(), root_pointer);
        extended = owned;
        &extended
    } else {
        config
    };

    let input_slots =
        u16::try_from(config.inputs.len()).map_err(|_| ReplacementError::TooManyInputs)?;
    let call_slots = u16::try_from(call_count).map_err(|_| ReplacementError::TooManyInputs)?;
    let call_result_base = input_slots;
    let result_base = input_slots
        .checked_add(call_slots)
        .ok_or(ReplacementError::TooManyInputs)?;
    let register_count = u16::try_from(X64NativeRegister::ALL.len()).unwrap_or(u16::MAX);
    let scratch_base = result_base + register_count;
    let flags_slot = region_retained.map(|_| SpillSlot(scratch_base + CALL_SCRATCH_SLOTS));
    let phi_base = scratch_base + CALL_SCRATCH_SLOTS + u16::from(flags_slot.is_some());
    let phi_temp_base = phi_base
        .checked_add(loop_phis)
        .ok_or(ReplacementError::TooManyInputs)?;
    let read_result_base = phi_temp_base
        .checked_add(loop_phis)
        .ok_or(ReplacementError::TooManyInputs)?;
    let read_slots = u16::try_from(read_count).map_err(|_| ReplacementError::TooManyInputs)?;
    let shared_end = read_result_base
        .checked_add(read_slots)
        .ok_or(ReplacementError::TooManyInputs)?;

    let mut inputs =
        Vec::with_capacity(config.inputs.len() + call_count + usize::from(loop_phis) + read_count);
    for (index, (name, _)) in config.inputs.iter().enumerate() {
        inputs.push(ExpressionInput::slot(
            name.clone(),
            BitWidth::W64,
            SpillSlot(u16::try_from(index).map_err(|_| ReplacementError::TooManyInputs)?),
        ));
    }
    for index in 0..call_count {
        inputs.push(ExpressionInput::slot(
            call_result_name(index),
            BitWidth::W64,
            SpillSlot(call_result_base + u16::try_from(index).unwrap_or(u16::MAX)),
        ));
    }
    for id in 0..loop_phis {
        inputs.push(ExpressionInput::slot(
            loop_phi_name(id),
            BitWidth::W64,
            SpillSlot(phi_base + id),
        ));
    }
    for index in 0..read_count {
        inputs.push(ExpressionInput::slot(
            read_result_name(index),
            BitWidth::W64,
            SpillSlot(read_result_base + u16::try_from(index).unwrap_or(u16::MAX)),
        ));
    }

    let mut emitter = Emitter {
        config,
        inputs,
        canonical: BTreeMap::new(),
        assignments: Vec::new(),
        next_register: 0,
        window_base: shared_end,
        shared_end,
        trust: Trust::default(),
        operations: Vec::new(),
        next_label: 0,
        call_result_base,
        read_result_base,
        result_base,
        scratch_base,
        flags_slot,
        phi_base,
        phi_temp_base,
    };
    for (index, (name, physical)) in config.inputs.iter().enumerate() {
        let source = emitter.register(*physical);
        if name == ENTRY_STACK_POINTER {
            emitter.operations.push(Operation::EntryStackPointer {
                destination: source,
            });
        }
        emitter.operations.push(Operation::SpillStore {
            slot: SpillSlot(u16::try_from(index).map_err(|_| ReplacementError::TooManyInputs)?),
            source,
        });
    }
    if let Some(slot) = flags_slot {
        let destination = emitter.register(PhysicalRegister::R10);
        emitter
            .operations
            .push(Operation::ReadFlags { destination });
        emitter.operations.push(Operation::SpillStore {
            slot,
            source: destination,
        });
    }
    emitter.emit_node(root)?;

    let Emitter {
        operations,
        assignments,
        trust,
        ..
    } = emitter;
    let mut preserve: Vec<PhysicalRegister> = Vec::new();
    for assignment in &assignments {
        if assignment.physical_register.is_win64_nonvolatile()
            && !preserve.contains(&assignment.physical_register)
        {
            preserve.push(assignment.physical_register);
        }
    }
    let highest = operations
        .iter()
        .flat_map(operation_slots)
        .max()
        .unwrap_or(0);
    let call_ready = operations
        .iter()
        .any(|operation| matches!(operation, Operation::Call { .. }));
    let mut stack_allocation = u32::from(highest.saturating_add(1)).saturating_mul(8);
    let pushes = u32::try_from(preserve.len())
        .map_err(|_| ReplacementError::TooManyInputs)?
        .saturating_mul(8);
    // The staged continuation stack overwrites the frame top, which must hold
    // no spill slot: the saved registers there are discarded by a tail transfer.
    if let Some(retained) = region_retained {
        stack_allocation = stack_allocation.saturating_add(retained.saturating_sub(pushes));
    }
    if call_ready {
        stack_allocation = stack_allocation.saturating_add(32);
        let residual = (8_u32.wrapping_sub(pushes.saturating_add(stack_allocation))) & 0xF;
        stack_allocation = stack_allocation.saturating_add(residual);
    }
    let function = Function {
        operations,
        assignments,
        frame: StackFrame {
            stack_allocation,
            preserve,
            call_ready,
            preinitialized_slots: (0..input_slots).map(SpillSlot).collect(),
            tail_transfer: true,
        },
    };
    let compiled = compile(&function, address)?;
    Ok(Replacement {
        function,
        compiled,
        emitted: X64NativeRegister::ALL.into_iter().collect(),
        passthroughs: trust.passthroughs,
        environment_boundaries: trust.environment_boundaries,
    })
}
