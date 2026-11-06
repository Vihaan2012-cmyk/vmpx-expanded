//! Cut-point segmentation of a recovered VM CFG with input-dependent loops.
//!
//! `unroll.rs` forks at every input-dependent branch, so a loop whose exit depends
//! on the inputs never terminates it. Here the CFG is cut at its loop headers
//! (every cycle passes through one) and the function becomes a family of
//! *segments*: the entry segment, and one segment per cut that starts from the
//! cut's block with the machine state replaced by fresh `vmphi` symbols. Inside a
//! segment the code is acyclic, so its path family is finite, exclusive and
//! exhaustive exactly as the unroll builds it. A path ends in a VM exit or at a
//! cut, carrying the values the destination's phis take next.
//!
//! The phis of a cut are the VM state cells present at *every* arrival (flags, the
//! context cells, and the live VM-stack cells grouped into aligned words). Cells
//! missing from some arrival are dropped from the cut: after it they read as
//! unknown, which only loses precision. Nothing is guessed. The stack pointer, the
//! allocations and the native-call epoch must be identical at every arrival, and a
//! segment stores only into the VM's own dead frame (which the phis carry), so a
//! segment that ends at a cut changes no memory the function can observe.
//!
//! Whether the two machines run in lockstep is the loop prover's job (see
//! `vmpx_verify::loops`); this module only produces the representation, and
//! refuses (`Err`) anything outside it.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use vmpx_pe::Va;
use vmpx_vm::{
    BinaryOp, BitWidth, BlockExit, CfgEdgeKind, Expr, ExprKind, NativeExitState,
    SymbolicExecutionLimits, SymbolicVmInitialState, VmHandlerSemantic,
    execute_symbolic_linear_block_with_initial_state, recover_native_exit_state, symbolic_pointer_offset,
};

use crate::decoded_lift::{RecoveredExit, StubExit, first_unliftable, recover_exit};
use crate::cfg_lift::compose_nested_state;
use crate::entry_frame::frame_initial_state;
use crate::nested_entry::NestedEntryResolver;
use crate::rolled::loop_headers;
use crate::unroll::{RegionTarget, UnrolledPath, is_environment_probe};
use crate::{CollectedEffects, CollectedStore, DecodedCfgInput, ExitSemantics, PathCondition};

/// Most restarts while the cut cell sets shrink to a fixed point.
const MAX_ATTEMPTS: usize = 200;

/// One phi of a cut.
#[derive(Clone, Debug)]
pub struct SegPhi {
    pub name: String,
    pub width: BitWidth,
}

/// How a segment path ends.
#[derive(Clone, Debug)]
pub enum SegEnd {
    Exit(Box<UnrolledPath>),
    /// A cut, with the values its phis take, in `Segment::phis` order.
    Cut { header: Va, next: Vec<Expr> },
}

/// One path of a segment, with its branch conditions (`selector == value`, or `!=`).
#[derive(Clone, Debug)]
pub struct SegPath {
    pub conditions: Vec<PathCondition>,
    pub end: SegEnd,
}

/// The segment that starts at one cut.
#[derive(Clone, Debug)]
pub struct Segment {
    pub phis: Vec<SegPhi>,
    pub paths: Vec<SegPath>,
}

/// A recovered VM function as entry segment plus one segment per cut.
#[derive(Clone, Debug)]
pub struct SegmentedFunction {
    pub entry: Vec<SegPath>,
    pub loops: BTreeMap<Va, Segment>,
    /// Byte offset of the entry stack pointer from `vsp_initial`.
    pub entry_sp_offset: i64,
    pub forks: usize,
    pub branches_decided: usize,
    pub blocks: usize,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum CellId {
    Flags,
    Ctx(i64),
    /// `bytes` bytes of the VM stack at this offset from `vsp_initial`.
    Mem(i64, u32),
}

fn root() -> Expr {
    Expr::register(BitWidth::W64, "vsp_initial")
}

fn at_offset(offset: i64) -> Result<Expr, String> {
    if offset == 0 {
        return Ok(root());
    }
    Expr::binary(
        BinaryOp::Add,
        root(),
        Expr::constant(BitWidth::W64, offset.cast_unsigned()),
    )
    .map_err(|e| e.to_string())
}

/// What a state holds at a cut.
struct Snap {
    flags: Expr,
    ctx: BTreeMap<i64, Expr>,
    /// Live VM-stack bytes (at or above the stack pointer) by offset.
    bytes: BTreeMap<i64, Expr>,
}

fn snapshot(state: &SymbolicVmInitialState) -> Result<Snap, String> {
    let top = symbolic_pointer_offset(&state.vsp, "vsp_initial")
        .ok_or("the VM stack pointer is not vsp_initial plus a constant")?;
    let mut bytes = BTreeMap::new();
    for (address, value) in &state.memory_values {
        let offset = symbolic_pointer_offset(address, "vsp_initial")
            .ok_or("tracked memory outside the VM stack at a loop cut")?;
        let n = value.width.bytes();
        for i in 0..n {
            let at = offset
                .checked_add(i64::from(i))
                .ok_or("tracked memory offset overflows")?;
            if at < top {
                continue;
            }
            let byte = if n == 1 {
                value.clone()
            } else {
                Expr::extract(BitWidth::W8, i * 8, value.clone()).map_err(|e| e.to_string())?
            };
            bytes.insert(at, byte);
        }
    }
    Ok(Snap {
        flags: state.flags.clone(),
        ctx: state.context_values.clone(),
        bytes,
    })
}

/// `V` when bytes `offset..offset+n` are exactly the byte slices of one `8n`-bit `V`.
fn whole_value(bytes: &BTreeMap<i64, Expr>, offset: i64, n: u32) -> Option<Expr> {
    let mut whole: Option<&Expr> = None;
    for k in 0..n {
        let byte = bytes.get(&(offset + i64::from(k)))?;
        let ExprKind::Extract { value, lsb } = &byte.kind else {
            return None;
        };
        if *lsb != k * 8 || value.width.bytes() != n {
            return None;
        }
        match whole {
            None => whole = Some(&**value),
            Some(first) if first == &**value => {}
            Some(_) => return None,
        }
    }
    whole.cloned()
}

fn combine(bytes: &[Expr]) -> Result<Expr, String> {
    if let [one] = bytes {
        return Ok(one.clone());
    }
    let mid = bytes.len() / 2;
    let low = combine(&bytes[..mid])?;
    let high = combine(&bytes[mid..])?;
    Expr::concat(high, low).map_err(|e| e.to_string())
}

fn read_mem(bytes: &BTreeMap<i64, Expr>, offset: i64, n: u32) -> Option<Expr> {
    if n == 1 {
        return bytes.get(&offset).cloned();
    }
    if let Some(whole) = whole_value(bytes, offset, n) {
        return Some(whole);
    }
    let parts: Option<Vec<Expr>> = (0..n).map(|k| bytes.get(&(offset + i64::from(k))).cloned()).collect();
    combine(&parts?).ok()
}

/// Greedy aligned grouping of live bytes into the widest whole-value cells.
fn group(bytes: &BTreeMap<i64, Expr>) -> Vec<(i64, u32)> {
    let offsets: Vec<i64> = bytes.keys().copied().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < offsets.len() {
        let offset = offsets[i];
        let mut take = 1_u32;
        for n in [8_u32, 4, 2] {
            let end = i + n as usize;
            let contiguous = end <= offsets.len() && (0..n).all(|k| offsets[i + k as usize] == offset + i64::from(k));
            let aligned_word = n == 8 && offset.rem_euclid(8) == 0;
            if contiguous && (aligned_word || whole_value(bytes, offset, n).is_some()) {
                take = n;
                break;
            }
        }
        out.push((offset, take));
        i += take as usize;
    }
    out
}

fn read_cell(snap: &Snap, cell: CellId) -> Option<Expr> {
    match cell {
        CellId::Flags => Some(snap.flags.clone()),
        CellId::Ctx(k) => snap.ctx.get(&k).cloned(),
        CellId::Mem(offset, n) => read_mem(&snap.bytes, offset, n),
    }
}

struct Template {
    /// The carried cells (each a phi), in `phis` order.
    cells: Vec<CellId>,
    /// Cells assumed to hold the same value at every arrival (the value of the
    /// first arrival); verified by the arrival check, dropped when any disagrees.
    invariants: Vec<(CellId, Expr)>,
    /// The state a segment from this cut starts in.
    base: SymbolicVmInitialState,
    phis: Vec<SegPhi>,
}

enum Fail {
    /// The cell set of a cut shrank: rerun every segment with it.
    Restart,
    Err(String),
}

impl From<String> for Fail {
    fn from(s: String) -> Self {
        Self::Err(s)
    }
}

impl From<&str> for Fail {
    fn from(s: &str) -> Self {
        Self::Err(s.to_owned())
    }
}

#[derive(Clone)]
struct Walk {
    state: SymbolicVmInitialState,
    stores: Vec<(CollectedStore, u64)>,
    conditions: Vec<PathCondition>,
    executed: usize,
    env_trail: Vec<Va>,
}

struct Walker<'a, 'b> {
    input: &'a DecodedCfgInput<'b>,
    nested: Option<&'a dyn NestedEntryResolver>,
    cuts: &'a BTreeSet<Va>,
    allowed: &'a mut HashMap<Va, HashSet<CellId>>,
    /// Cells found not invariant at a cut (they become phis).
    varying: &'a mut HashMap<Va, HashSet<CellId>>,
    broken: Vec<(Va, CellId)>,
    templates: HashMap<Va, Template>,
    queue: Vec<Va>,
    max_blocks: usize,
    max_paths: usize,
    blocks: usize,
    paths: usize,
    forks: usize,
    decided: usize,
    entry_sp: i64,
}

/// True when every cycle of the branch graph passes through a cut.
fn cuts_break_every_cycle(input: &DecodedCfgInput<'_>, cuts: &BTreeSet<Va>) -> bool {
    let cfg = input.cfg;
    let mut indegree: BTreeMap<Va, usize> = cfg
        .blocks
        .keys()
        .filter(|k| !cuts.contains(*k))
        .map(|k| (*k, 0))
        .collect();
    let live = |k: &Va| indegree.contains_key(k);
    let edges: Vec<(Va, Va)> = cfg
        .edges
        .iter()
        .filter(|e| matches!(e.kind, CfgEdgeKind::Branch | CfgEdgeKind::GuardedBranch))
        .filter(|e| live(&e.from) && live(&e.to))
        .map(|e| (e.from, e.to))
        .collect();
    for (_, to) in &edges {
        *indegree.get_mut(to).expect("live") += 1;
    }
    let mut ready: Vec<Va> = indegree.iter().filter(|(_, d)| **d == 0).map(|(k, _)| *k).collect();
    let mut removed = 0;
    while let Some(k) = ready.pop() {
        removed += 1;
        for (from, to) in &edges {
            if *from == k {
                let d = indegree.get_mut(to).expect("live");
                *d -= 1;
                if *d == 0 {
                    ready.push(*to);
                }
            }
        }
    }
    removed == indegree.len()
}

/// Cuts the CFG at its loop headers and executes every segment.
///
/// # Errors
/// The CFG is incomplete, a cycle avoids every cut, a block is unliftable or
/// does something the segments do not model (native call, havoc, allocation,
/// nested entry, a store outside the VM's dead frame), the stack pointer or
/// allocations differ between arrivals at a cut, or a budget is hit.
pub fn segment_decoded_cfg(
    input: &DecodedCfgInput<'_>,
    nested: Option<&dyn NestedEntryResolver>,
    max_blocks: usize,
    max_paths: usize,
) -> Result<SegmentedFunction, String> {
    if !input.cfg.is_complete() {
        let why = input
            .cfg
            .stalls
            .first()
            .map_or_else(|| format!("{:?}", input.cfg.limit_hit), |stall| format!("{stall:?}"));
        return Err(format!("the CFG is incomplete: {why}"));
    }
    let cuts = loop_headers(input);
    if cuts.is_empty() {
        return Err("the CFG has no loop".to_owned());
    }
    if !cuts_break_every_cycle(input, &cuts) {
        return Err("a cycle of the CFG avoids every loop header (irreducible)".to_owned());
    }
    let entry_sp = i64::try_from(input.top_offset)
        .unwrap_or(i64::MAX)
        .saturating_add(i64::try_from(input.frame.len() * 8).unwrap_or(i64::MAX));
    let mut allowed: HashMap<Va, HashSet<CellId>> = HashMap::new();
    let mut varying: HashMap<Va, HashSet<CellId>> = HashMap::new();
    for _ in 0..MAX_ATTEMPTS {
        let mut walker = Walker {
            input,
            nested,
            cuts: &cuts,
            allowed: &mut allowed,
            varying: &mut varying,
            broken: Vec::new(),
            templates: HashMap::new(),
            queue: Vec::new(),
            max_blocks,
            max_paths,
            blocks: 0,
            paths: 0,
            forks: 0,
            decided: 0,
            entry_sp,
        };
        match walker.run() {
            Ok(f) => return Ok(f),
            Err(Fail::Restart) => {}
            Err(Fail::Err(e)) => return Err(e),
        }
    }
    Err("the cut cell sets did not converge".to_owned())
}

impl Walker<'_, '_> {
    fn run(&mut self) -> Result<SegmentedFunction, Fail> {
        let input = self.input;
        let (state, _) = frame_initial_state(input.frame, input.top_offset);
        let mut entry = Vec::new();
        self.walk(input.cfg.entry, fresh(state), true, &mut entry)?;
        let mut loops = BTreeMap::new();
        while let Some(header) = self.queue.pop() {
            let (base, phis) = {
                let t = &self.templates[&header];
                (t.base.clone(), t.phis.clone())
            };
            let mut paths = Vec::new();
            self.walk(header, fresh(base), true, &mut paths)?;
            loops.insert(header, Segment { phis, paths });
        }
        if !self.broken.is_empty() {
            for (header, cell) in std::mem::take(&mut self.broken) {
                self.varying.entry(header).or_default().insert(cell);
            }
            return Err(Fail::Restart);
        }
        Ok(SegmentedFunction {
            entry,
            loops,
            entry_sp_offset: self.entry_sp,
            forks: self.forks,
            branches_decided: self.decided,
            blocks: self.blocks,
        })
    }

    fn successors(&self, key: Va) -> Vec<(u64, Va)> {
        let cfg = self.input.cfg;
        let mut out: Vec<(u64, Va)> = Vec::new();
        for edge in &cfg.edges {
            if edge.from != key || !matches!(edge.kind, CfgEdgeKind::Branch | CfgEdgeKind::GuardedBranch) {
                continue;
            }
            if let Some(address) = cfg.blocks.get(&edge.to).and_then(|b| b.address)
                && !out.iter().any(|(_, k)| *k == edge.to)
            {
                out.push((address.0, edge.to));
            }
        }
        out.sort_by_key(|(value, _)| *value);
        out
    }

    /// Records an arrival at cut `header`; the carried values in phi order.
    fn arrive(&mut self, header: Va, state: &SymbolicVmInitialState) -> Result<Vec<Expr>, Fail> {
        if state.memory_tainted {
            return Err("an arrival at a loop cut follows an imprecise external-memory store".into());
        }
        let snap = snapshot(state)?;
        let Some(template) = self.templates.get(&header) else {
            let mut cells = vec![CellId::Flags];
            cells.extend(snap.ctx.keys().map(|k| CellId::Ctx(*k)));
            cells.extend(group(&snap.bytes).into_iter().map(|(o, n)| CellId::Mem(o, n)));
            if let Some(allowed) = self.allowed.get(&header) {
                cells.retain(|c| allowed.contains(c));
            }
            cells.retain(|c| read_cell(&snap, *c).is_some());
            let mut base = state.clone();
            base.context_values.clear();
            base.memory_values.clear();
            let varying = self.varying.get(&header);
            let mut phi_cells = Vec::new();
            let mut invariants = Vec::new();
            let mut phis = Vec::new();
            let mut values = Vec::new();
            for cell in &cells {
                let value = read_cell(&snap, *cell).expect("retained readable");
                let is_phi = varying.is_some_and(|v| v.contains(cell));
                let symbol = if is_phi {
                    let name = format!("vmphi{:x}_{}", header.0, phis.len());
                    let phi = Expr::register(value.width, name.clone());
                    phis.push(SegPhi { name, width: value.width });
                    phi_cells.push(*cell);
                    values.push(value.clone());
                    phi
                } else {
                    invariants.push((*cell, value.clone()));
                    value
                };
                match *cell {
                    CellId::Flags => base.flags = symbol,
                    CellId::Ctx(k) => {
                        base.context_values.insert(k, symbol);
                    }
                    CellId::Mem(offset, _) => base.memory_values.push((at_offset(offset)?, symbol)),
                }
            }
            self.templates.insert(header, Template { cells: phi_cells, invariants, base, phis });
            self.queue.push(header);
            return Ok(values);
        };
        let base = &template.base;
        if base.vsp != state.vsp
            || base.fresh_allocations != state.fresh_allocations
            || base.next_native_call_epoch != state.next_native_call_epoch
            || base.image_reads_suppressed != state.image_reads_suppressed
            || base.protection_may_have_changed != state.protection_may_have_changed
        {
            return Err(
                "the stack pointer, allocations, native calls or memory protection differ between arrivals at a loop cut"
                    .into(),
            );
        }
        let mut values = Vec::with_capacity(template.cells.len());
        let mut missing: Vec<CellId> = Vec::new();
        for (cell, phi) in template.cells.iter().zip(&template.phis) {
            match read_cell(&snap, *cell) {
                Some(v) if v.width == phi.width => values.push(v),
                Some(_) => return Err("a loop-carried cell changes width".into()),
                None => missing.push(*cell),
            }
        }
        let mut broken: Vec<(Va, CellId)> = Vec::new();
        for (cell, expected) in &template.invariants {
            match read_cell(&snap, *cell) {
                Some(v) if v == *expected || v.simplify() == expected.simplify() => {}
                Some(_) => broken.push((header, *cell)),
                None => missing.push(*cell),
            }
        }
        if !missing.is_empty() {
            let keep: HashSet<CellId> = template
                .cells
                .iter()
                .copied()
                .chain(template.invariants.iter().map(|(c, _)| *c))
                .filter(|c| !missing.contains(c))
                .collect();
            self.allowed.insert(header, keep);
            return Err(Fail::Restart);
        }
        self.broken.extend(broken);
        Ok(values)
    }

    #[allow(clippy::too_many_lines)]
    fn walk(&mut self, mut key: Va, mut walk: Walk, mut first: bool, out: &mut Vec<SegPath>) -> Result<(), Fail> {
        let input = self.input;
        let cfg = input.cfg;
        loop {
            if !first && self.cuts.contains(&key) {
                let next = self.arrive(key, &walk.state)?;
                self.push_path(out, SegPath { conditions: walk.conditions, end: SegEnd::Cut { header: key, next } })?;
                return Ok(());
            }
            first = false;
            self.blocks += 1;
            walk.executed += 1;
            if self.blocks > self.max_blocks {
                return Err(format!("block budget {} exhausted", self.max_blocks).into());
            }
            let block = cfg
                .blocks
                .get(&key)
                .ok_or_else(|| format!("block {key} is not in the recovered CFG"))?;
            if let Some((index, semantic, reason)) = first_unliftable(&block.instructions) {
                return Err(format!("block {key} instruction {index} ({semantic:?}): {reason}").into());
            }
            let boundary = match block.exit {
                BlockExit::VmExit => VmHandlerSemantic::VmExit,
                BlockExit::Branch => VmHandlerSemantic::UnconditionalBranch,
                BlockExit::NativeCall => return Err(format!("block {key} makes a native call").into()),
                BlockExit::Stalled => return Err(format!("block {key} did not reach a boundary").into()),
            };
            let report = execute_symbolic_linear_block_with_initial_state(
                &block.instructions,
                Some(boundary),
                SymbolicExecutionLimits::default(),
                &walk.state,
                input.memory,
            )
            .map_err(|e| format!("block {key} symbolic execution failed: {e}"))?;
            if !report.native_calls.is_empty()
                || !report.fresh_allocations.is_empty()
                || report.memory_tainted
                || !report.memory_havocs.is_empty()
            {
                return Err(format!("block {key} havocs memory, calls natively or allocates").into());
            }
            for write in &report.external_writes {
                let inside = symbolic_pointer_offset(&write.address, "vsp_initial")
                    .and_then(|off| off.checked_add(i64::from(write.value.width.bytes())))
                    .is_some_and(|end| end <= self.entry_sp);
                if !inside {
                    return Err(format!("block {key} stores outside the VM's dead frame").into());
                }
                walk.stores.push((
                    CollectedStore { address: write.address.clone(), value: write.value.clone(), width: write.value.width, epoch: write.epoch },
                    write.epoch,
                ));
            }
            match block.exit {
                BlockExit::VmExit => {
                    let handler = block.instructions.last().map_or(block.start.handler, |last| last.state_after.handler);
                    let layout = block
                        .vm_exit
                        .as_ref()
                        .or_else(|| input.exits.get(&handler))
                        .ok_or_else(|| format!("no recovered VM-exit layout for handler {handler}"))?;
                    let nested_to = cfg
                        .edges
                        .iter()
                        .find_map(|e| (e.from == key && e.kind == CfgEdgeKind::NestedEntry).then_some(e.to));
                    if let Some(to) = nested_to {
                        let resolver = self.nested.ok_or("exit re-enters a nested VM entry")?;
                        let exit = recover_native_exit_state(&report, layout)
                            .map_err(|e| format!("nested exit state unrecovered: {e}"))?;
                        let entry = resolver
                            .resolve(&exit.return_address)
                            .ok_or("the nested VM entry is not provably resolved")?;
                        if cfg.blocks.get(&to).and_then(|b| b.address) != Some(entry.bytecode_start) {
                            return Err("the resolved nested entry is not the CFG's nested block".into());
                        }
                        walk.state = compose_nested_state(&report, &exit, &entry)
                            .map_err(|e| format!("nested entry composition failed: {e}"))?;
                        key = to;
                        continue;
                    }
                    let recovered = recover_exit(&report, layout, input.frame.len(), input.top_offset)
                        .map_err(|e| e.to_string())?;
                    let mut region_target = None;
                    let (exit, region_exit) = match recovered {
                        RecoveredExit::Callee(exit) => (exit, false),
                        RecoveredExit::Trampoline(_) => {
                            return Err("the exit transfers to a computed continuation above the entry frame".into());
                        }
                        RecoveredExit::Region(region) => {
                            region_target = region.staged.first().map(|target| RegionTarget {
                                target: target.simplify(),
                                staged_words: region.staged.len(),
                            });
                            (StubExit { registers: region.registers, rflags: region.rflags, return_pop: 0 }, true)
                        }
                    };
                    let rflags_is_entry_flags = matches!(
                        &exit.rflags.simplify().kind,
                        ExprKind::Register { name } if name == "native_rflags_initial"
                    );
                    let (collected, store_epochs): (Vec<_>, Vec<_>) = walk.stores.into_iter().unzip();
                    let path = UnrolledPath {
                        conditions: walk.conditions.clone(),
                        semantics: ExitSemantics {
                            exit: NativeExitState {
                                xmm: BTreeMap::new(),
                                schema_version: 1,
                                registers: exit.registers,
                                rflags: exit.rflags,
                                return_address: Expr::constant(BitWidth::W64, 0),
                                frame_size: 0,
                            },
                            effects: CollectedEffects { stores: collected, calls: Vec::new(), unsupported: Vec::new() },
                        },
                        store_epochs,
                        rflags_is_entry_flags,
                        return_pop: exit.return_pop,
                        blocks_executed: walk.executed,
                        region_exit,
                        region: region_target,
                        env_forks: walk.env_trail.len(),
                    };
                    self.push_path(
                        out,
                        SegPath { conditions: walk.conditions, end: SegEnd::Exit(Box::new(path)) },
                    )?;
                    return Ok(());
                }
                BlockExit::Branch => {
                    let target = report
                        .branch_target
                        .as_ref()
                        .ok_or_else(|| format!("block {key} branch has no symbolic target"))?
                        .simplify();
                    let successors = self.successors(key);
                    walk.state = report.continuation_state();
                    if let ExprKind::Const { value } = target.kind {
                        key = successors
                            .iter()
                            .find(|(address, _)| *address == value)
                            .map(|(_, k)| *k)
                            .ok_or_else(|| format!("block {key} branch target {value:#x} is not a recovered successor"))?;
                        self.decided += 1;
                        continue;
                    }
                    if successors.len() < 2 {
                        if std::env::var_os("VMPX_SEG_TRACE").is_some() {
                            eprintln!(
                                "[seg] block {key} target {:?}; edges {:?}",
                                target,
                                cfg.edges.iter().filter(|e| e.from == key).collect::<Vec<_>>()
                            );
                        }
                        return Err(format!(
                            "block {key} branch target depends on inputs but has {} recovered successor(s)",
                            successors.len()
                        )
                        .into());
                    }
                    self.forks += 1;
                    let env = is_environment_probe(&target);
                    if env && walk.env_trail.contains(&key) {
                        return Err(format!("environment gate at {key} repeats inside one segment").into());
                    }
                    let mut prefix = walk;
                    if env {
                        prefix.env_trail.push(key);
                    }
                    let last = successors.len() - 1;
                    for (index, (value, successor)) in successors.iter().enumerate() {
                        let mut arm = prefix.clone();
                        if index != last {
                            arm.conditions.push(PathCondition { selector: target.clone(), value: *value, equal: true });
                        }
                        self.walk(*successor, arm, false, out)?;
                        prefix.conditions.push(PathCondition { selector: target.clone(), value: *value, equal: false });
                    }
                    return Ok(());
                }
                BlockExit::NativeCall | BlockExit::Stalled => unreachable!("rejected above"),
            }
        }
    }

    fn push_path(&mut self, out: &mut Vec<SegPath>, path: SegPath) -> Result<(), Fail> {
        self.paths += 1;
        if self.paths > self.max_paths {
            return Err(format!("path budget {} exhausted", self.max_paths).into());
        }
        out.push(path);
        Ok(())
    }
}

fn fresh(state: SymbolicVmInitialState) -> Walk {
    Walk { state, stores: Vec::new(), conditions: Vec::new(), executed: 0, env_trail: Vec::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn byte(name: &str, k: u32) -> Expr {
        Expr::extract(BitWidth::W8, k * 8, Expr::register(BitWidth::W64, name)).unwrap()
    }

    #[test]
    fn bytes_of_one_word_group_into_that_word() {
        let bytes: BTreeMap<i64, Expr> = (0..8).map(|k| (16 + i64::from(k), byte("v", k))).collect();
        assert_eq!(group(&bytes), vec![(16, 8)]);
        assert_eq!(read_mem(&bytes, 16, 8), Some(Expr::register(BitWidth::W64, "v")));
    }

    #[test]
    fn an_aligned_word_of_unrelated_bytes_is_still_one_cell() {
        let bytes: BTreeMap<i64, Expr> =
            (0..8).map(|k| (i64::from(k), Expr::register(BitWidth::W8, format!("b{k}")))).collect();
        assert_eq!(group(&bytes), vec![(0, 8)]);
        assert_eq!(read_mem(&bytes, 0, 8).expect("all bytes live").width, BitWidth::W64);
    }

    #[test]
    fn misaligned_unrelated_bytes_stay_bytes_and_a_gap_breaks_a_cell() {
        let bytes: BTreeMap<i64, Expr> =
            (0..8).map(|k| (4 + i64::from(k), Expr::register(BitWidth::W8, format!("b{k}")))).collect();
        assert!(group(&bytes).iter().all(|(_, n)| *n == 1));
        let gap: BTreeMap<i64, Expr> =
            (0..7).map(|k| (i64::from(k), Expr::register(BitWidth::W8, format!("c{k}")))).collect();
        assert_eq!(read_mem(&gap, 0, 8), None);
    }
}
