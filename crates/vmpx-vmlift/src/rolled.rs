//! SPIKE: ROLLED recovery of a VM CFG (for the decompiler).
//!
//! `unroll.rs` executes a counted loop to its end, which proves equivalence but
//! yields a flat trace. Here a loop header reached for the first time is
//! summarized instead: one exact trial iteration finds the loop-carried cells
//! (state at the header on entry vs. on re-entry), those become fresh `phi`
//! symbols, and the body is re-walked once over them. The result is a tree:
//! forks, loops (phis + one-iteration body) and exits, with exit values as
//! expressions over the phis and the entry registers.
//!
//! Fail-closed: anything outside the supported shape (input-dependent control in
//! the trial iteration, stores outside the VM's dead frame, native calls, a
//! carried cell the body does not close over) is an `Err` for the whole function.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use vmpx_pe::Va;
use vmpx_vm::{
    BinaryOp, BitWidth, BlockExit, CfgEdgeKind, Expr, ExprKind, SymbolicExecutionLimits,
    SymbolicVmInitialState, VmHandlerSemantic, X64NativeRegister,
    execute_symbolic_linear_block_with_initial_state,
};

use crate::decoded_lift::{RecoveredExit, first_unliftable, recover_exit};
use crate::entry_frame::frame_initial_state;
use crate::DecodedCfgInput;

#[derive(Clone, Debug)]
pub struct RPhi {
    pub name: String,
    pub width: BitWidth,
    /// Value on loop entry (over entry registers and enclosing phis).
    pub init: Expr,
}

#[derive(Clone, Debug)]
pub struct RExit {
    pub registers: BTreeMap<X64NativeRegister, Expr>,
    /// Returns through staged stack (a protector gate failure), not the caller.
    pub region: bool,
}

#[derive(Clone, Debug)]
pub enum RNode {
    Exit(RExit),
    /// `selector == value` picks the arm; arms are in ascending value order.
    Fork {
        selector: Expr,
        env: bool,
        block: Va,
        arms: Vec<(u64, RNode)>,
    },
    Loop {
        header: Va,
        phis: Vec<RPhi>,
        body: Box<RNode>,
    },
    /// Next-iteration value of every live phi of the innermost loop, in `phis` order.
    BackEdge { next: Vec<Expr> },
    /// An arm with no reachable function exit (spin on a failed probe, repeat).
    Dead(String),
}

#[derive(Clone, Debug)]
pub struct RolledFunction {
    pub root: RNode,
    pub loops: usize,
    pub forks: usize,
    pub straight_blocks: usize,
}

struct LoopCtx {
    header: Va,
    /// Live phi names, in `RNode::Loop::phis` order.
    names: Vec<String>,
    cells: Vec<Cell>,
    /// The state the body is walked from (carried cells hold their phi).
    base: SymbolicVmInitialState,
    /// Cells that moved on a back edge but are not yet carried: the loop is not
    /// closed over `cells` until this is empty.
    pending: Vec<Cell>,
}

#[derive(Clone, Debug, PartialEq)]
enum Cell {
    Flags,
    Ctx(i64),
    Mem(Expr),
}

struct Walker<'a, 'b> {
    input: &'a DecodedCfgInput<'b>,
    headers: BTreeSet<Va>,
    budget: usize,
    blocks: usize,
    forks: usize,
    loops: usize,
    phi_id: usize,
    loop_stack: Vec<LoopCtx>,
}

/// Rolled walk of `input`'s CFG.
///
/// # Errors
/// An incomplete CFG or any shape listed in the module docs.
pub fn roll_decoded_cfg(input: &DecodedCfgInput<'_>, max_blocks: usize) -> Result<RolledFunction, String> {
    if !input.cfg.is_complete() {
        return Err("the CFG is incomplete".to_owned());
    }
    let (state, _) = frame_initial_state(input.frame, input.top_offset);
    let mut w = Walker {
        input,
        headers: loop_headers(input),
        budget: max_blocks,
        blocks: 0,
        forks: 0,
        loops: 0,
        phi_id: 0,
        loop_stack: Vec::new(),
    };
    let root = w.walk(input.cfg.entry, state, Vec::new(), false)?;
    Ok(RolledFunction { root, loops: w.loops, forks: w.forks, straight_blocks: w.blocks })
}

fn loop_headers(input: &DecodedCfgInput<'_>) -> BTreeSet<Va> {
    let cfg = input.cfg;
    let mut preds: BTreeMap<Va, Vec<Va>> = cfg.blocks.keys().map(|k| (*k, vec![])).collect();
    for e in &cfg.edges {
        if matches!(e.kind, CfgEdgeKind::Branch | CfgEdgeKind::GuardedBranch) {
            preds.entry(e.to).or_default().push(e.from);
        }
    }
    let all: BTreeSet<Va> = cfg.blocks.keys().copied().collect();
    let mut dom: BTreeMap<Va, BTreeSet<Va>> = cfg.blocks.keys().map(|k| (*k, all.clone())).collect();
    dom.insert(cfg.entry, BTreeSet::from([cfg.entry]));
    loop {
        let mut changed = false;
        for k in cfg.blocks.keys() {
            if *k == cfg.entry {
                continue;
            }
            let mut it = preds[k].iter().map(|p| dom[p].clone());
            let mut new = it.next().unwrap_or_default();
            for s in it {
                new = new.intersection(&s).copied().collect();
            }
            new.insert(*k);
            if new != dom[k] {
                dom.insert(*k, new);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    let mut headers = BTreeSet::new();
    for e in &cfg.edges {
        if matches!(e.kind, CfgEdgeKind::Branch | CfgEdgeKind::GuardedBranch)
            && dom.get(&e.from).is_some_and(|d| d.contains(&e.to))
        {
            headers.insert(e.to);
        }
    }
    headers
}

fn is_environment_probe(selector: &Expr) -> bool {
    let intrinsics = selector.intrinsic_dependencies();
    let probes = intrinsics.iter().any(|n| {
        n.starts_with("x86_cpuid") || n.starts_with("x86_gs_load") || n.starts_with("x86_fs_load")
    });
    probes
        || (selector.dependencies().is_empty()
            && intrinsics
                .iter()
                .all(|n| n.starts_with("memory_load_epoch_") || n.starts_with("vm_flags_")))
}

/// Offset of `address` from `vsp_initial` when it has the form `vsp_initial + c`.
fn vsp_offset(address: &Expr) -> Option<i64> {
    let a = address.simplify();
    match &a.kind {
        ExprKind::Register { name } if name == "vsp_initial" => Some(0),
        ExprKind::Binary { op: BinaryOp::Add, left, right } => {
            match (&left.kind, &right.kind) {
                (ExprKind::Register { name }, ExprKind::Const { value }) if name == "vsp_initial" => {
                    Some(value.cast_signed())
                }
                (ExprKind::Const { value }, ExprKind::Register { name }) if name == "vsp_initial" => {
                    Some(value.cast_signed())
                }
                _ => None,
            }
        }
        _ => None,
    }
}

fn mem_map(s: &SymbolicVmInitialState) -> HashMap<Expr, Expr> {
    s.memory_values.iter().cloned().collect()
}

fn get_cell(s: &SymbolicVmInitialState, c: &Cell) -> Option<Expr> {
    match c {
        Cell::Flags => Some(s.flags.clone()),
        Cell::Ctx(k) => s.context_values.get(k).cloned(),
        Cell::Mem(a) => s.memory_values.iter().find(|(x, _)| x == a).map(|(_, v)| v.clone()),
    }
}

fn set_cell(s: &mut SymbolicVmInitialState, c: &Cell, v: Expr) {
    match c {
        Cell::Flags => s.flags = v,
        Cell::Ctx(k) => {
            s.context_values.insert(*k, v);
        }
        Cell::Mem(a) => {
            if let Some(slot) = s.memory_values.iter_mut().find(|(x, _)| x == a) {
                slot.1 = v;
            } else {
                s.memory_values.push((a.clone(), v));
            }
        }
    }
}

/// Cells whose value differs between `a` and `b`.
fn differing_cells(a: &SymbolicVmInitialState, b: &SymbolicVmInitialState) -> Result<Vec<Cell>, String> {
    if a.vsp != b.vsp {
        return Err("the loop is not stack balanced (vsp differs on re-entry)".to_owned());
    }
    if a.fresh_allocations != b.fresh_allocations || a.next_native_call_epoch != b.next_native_call_epoch {
        return Err("the loop allocates or calls".to_owned());
    }
    let mut out = Vec::new();
    if a.flags != b.flags {
        out.push(Cell::Flags);
    }
    // Only cells present on both sides can be carried; a cell that appears or
    // vanishes is a stack temporary (written before read / dead on re-entry).
    for (k, v) in &a.context_values {
        if b.context_values.get(k).is_some_and(|w| w != v) {
            out.push(Cell::Ctx(*k));
        }
    }
    let (ma, mb) = (mem_map(a), mem_map(b));
    let mut addrs: Vec<&Expr> = ma.keys().collect();
    addrs.sort_by_key(|e| vsp_offset(e).unwrap_or(i64::MAX));
    for addr in addrs {
        if mb.get(addr).is_some_and(|w| w != &ma[addr]) {
            out.push(Cell::Mem(addr.clone()));
        }
    }
    Ok(out)
}

impl Walker<'_, '_> {
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

    /// Executes one block; returns the report and its boundary kind.
    fn exec(
        &mut self,
        key: Va,
        state: &SymbolicVmInitialState,
    ) -> Result<(vmpx_vm::SymbolicVmExecutionReport, BlockExit), String> {
        self.blocks += 1;
        if self.blocks > self.budget {
            return Err(format!("block budget {} exhausted", self.budget));
        }
        let cfg = self.input.cfg;
        let block = cfg.blocks.get(&key).ok_or_else(|| format!("block {key} is not in the recovered CFG"))?;
        if let Some((index, semantic, reason)) = first_unliftable(&block.instructions) {
            return Err(format!("block {key} instruction {index} ({semantic:?}): {reason}"));
        }
        let boundary = match block.exit {
            BlockExit::VmExit => VmHandlerSemantic::VmExit,
            BlockExit::Branch => VmHandlerSemantic::UnconditionalBranch,
            BlockExit::NativeCall => return Err(format!("block {key} makes a native call")),
            BlockExit::Stalled => return Err(format!("block {key} did not reach a boundary")),
        };
        let report = execute_symbolic_linear_block_with_initial_state(
            &block.instructions,
            Some(boundary),
            SymbolicExecutionLimits::default(),
            state,
            self.input.memory,
        )
        .map_err(|e| format!("block {key} symbolic execution failed: {e}"))?;
        if !report.native_calls.is_empty()
            || !report.fresh_allocations.is_empty()
            || report.memory_tainted
            || !report.memory_havocs.is_empty()
        {
            return Err(format!("block {key} havocs memory, calls natively or allocates"));
        }
        let entry_sp = i64::try_from(self.input.top_offset)
            .unwrap_or(i64::MAX)
            .saturating_add(i64::try_from(self.input.frame.len() * 8).unwrap_or(i64::MAX));
        for w in &report.external_writes {
            match vsp_offset(&w.address) {
                Some(off) if off + i64::from(w.value.width.bytes()) <= entry_sp => {}
                _ => return Err(format!("block {key} stores outside the VM's dead frame")),
            }
        }
        Ok((report, block.exit))
    }

    /// One exact iteration from `header` over `state`; the state on re-entry.
    fn trial(&mut self, header: Va, state: &SymbolicVmInitialState) -> Result<Option<SymbolicVmInitialState>, String> {
        let mut key = header;
        let mut state = state.clone();
        for _ in 0..64 {
            let (report, exit) = self.exec(key, &state)?;
            if exit == BlockExit::VmExit {
                return Ok(None);
            }
            let target = report
                .branch_target
                .as_ref()
                .ok_or_else(|| format!("block {key} branch has no symbolic target"))?
                .simplify();
            state = report.continuation_state();
            let ExprKind::Const { value } = target.kind else {
                return Err("input-dependent control flow inside a counted-loop trial iteration".to_owned());
            };
            let next = self
                .successors(key)
                .into_iter()
                .find(|(a, _)| *a == value)
                .map(|(_, k)| k)
                .ok_or_else(|| format!("block {key} branch target {value:#x} is not a recovered successor"))?;
            if next == header {
                return Ok(Some(state));
            }
            key = next;
        }
        Err("trial iteration did not return to the header".to_owned())
    }

    fn live_phis(
        cells: &[Cell],
        names: &[String],
        next: &[Vec<Expr>],
        roots: &[&Expr],
    ) -> Vec<usize> {
        let mut live: BTreeSet<usize> = BTreeSet::new();
        let reads = |e: &Expr, live: &mut BTreeSet<usize>| -> bool {
            let deps = e.dependencies();
            let mut grew = false;
            for (i, n) in names.iter().enumerate() {
                if deps.contains(n) && live.insert(i) {
                    grew = true;
                }
            }
            grew
        };
        for r in roots {
            reads(r, &mut live);
        }
        loop {
            let mut grew = false;
            for i in live.clone() {
                for n in next { grew |= reads(&n[i], &mut live); }
            }
            if !grew {
                break;
            }
        }
        let _ = cells;
        live.into_iter().collect()
    }

    #[allow(clippy::too_many_lines)]
    fn walk(
        &mut self,
        mut key: Va,
        mut state: SymbolicVmInitialState,
        mut trail: Vec<Va>,
        mut entering: bool,
    ) -> Result<RNode, String> {
        loop {
            if !entering {
                if self.loop_stack.last().is_some_and(|c| c.header == key) {
                    // back at the header of the loop being summarized: the iteration closes
                    let moved = {
                        let ctx = self.loop_stack.last().expect("ctx");
                        differing_cells(&ctx.base, &state)?
                    };
                    let ctx = self.loop_stack.last_mut().expect("ctx");
                    for m in moved {
                        if !ctx.cells.contains(&m) && !ctx.pending.contains(&m) {
                            ctx.pending.push(m);
                        }
                    }
                    let next = ctx
                        .cells
                        .iter()
                        .map(|c| get_cell(&state, c).ok_or_else(|| "a carried cell vanished".to_owned()))
                        .collect::<Result<Vec<_>, _>>()?;
                    return Ok(RNode::BackEdge { next });
                }
                if trail.contains(&key) {
                    return Ok(RNode::Dead(format!("block {key} repeats on one path (spin)")));
                }
                if self.headers.contains(&key) {
                    match self.trial(key, &state) {
                        Ok(Some(reentry)) => return self.summarize(key, state, reentry, trail),
                        Ok(None) => {}
                        Err(_) => {}
                    }
                }
            }
            entering = false;
            trail.push(key);
            let (report, exit) = self.exec(key, &state)?;
            match exit {
                BlockExit::VmExit => {
                    let cfg = self.input.cfg;
                    let block = &cfg.blocks[&key];
                    if cfg.edges.iter().any(|e| e.from == key && e.kind == CfgEdgeKind::NestedEntry) {
                        return Err("exit re-enters a nested VM entry".to_owned());
                    }
                    let handler = block
                        .instructions
                        .last()
                        .map_or(block.start.handler, |last| last.state_after.handler);
                    let layout = block
                        .vm_exit
                        .as_ref()
                        .or_else(|| self.input.exits.get(&handler))
                        .ok_or_else(|| format!("no recovered VM-exit layout for handler {handler}"))?;
                    let recovered = recover_exit(&report, layout, self.input.frame.len(), self.input.top_offset)
                        .map_err(|e| e.to_string())?;
                    let (registers, region) = match recovered {
                        RecoveredExit::Callee(e) => (e.registers, false),
                        RecoveredExit::Region(r) => (r.registers, true),
                    };
                    return Ok(RNode::Exit(RExit { registers, region }));
                }
                BlockExit::Branch => {
                    let target = report
                        .branch_target
                        .as_ref()
                        .ok_or_else(|| format!("block {key} branch has no symbolic target"))?
                        .simplify();
                    let successors = self.successors(key);
                    state = report.continuation_state();
                    if let ExprKind::Const { value } = target.kind {
                        key = successors
                            .iter()
                            .find(|(a, _)| *a == value)
                            .map(|(_, k)| *k)
                            .ok_or_else(|| format!("block {key} branch target {value:#x} is not a recovered successor"))?;
                        continue;
                    }
                    if successors.len() < 2 {
                        return Err(format!(
                            "block {key} branch target depends on inputs but has {} recovered successor(s)",
                            successors.len()
                        ));
                    }
                    self.forks += 1;
                    let env = is_environment_probe(&target);
                    let mut arms = Vec::new();
                    for (value, succ) in &successors {
                        let node = self.walk(*succ, state.clone(), trail.clone(), false)?;
                        arms.push((*value, node));
                    }
                    return Ok(RNode::Fork { selector: target, env, block: key, arms });
                }
                BlockExit::NativeCall | BlockExit::Stalled => unreachable!("rejected in exec"),
            }
        }
    }

    fn summarize(
        &mut self,
        header: Va,
        pre: SymbolicVmInitialState,
        reentry: SymbolicVmInitialState,
        trail: Vec<Va>,
    ) -> Result<RNode, String> {
        let mut cells = differing_cells(&pre, &reentry)?;
        if cells.is_empty() {
            return Ok(RNode::Dead(format!("loop at {header} changes no state (spin)")));
        }
        let (mut body, ctx, names, inits) = 'close: loop {
            // one phi per carried cell
            let mut phi_state = pre.clone();
            let mut names = Vec::new();
            let mut inits = Vec::new();
            for c in &cells {
                let init = get_cell(&pre, c).ok_or_else(|| "a carried cell has no pre-loop value".to_owned())?;
                let name = format!("__phi_{}", self.phi_id);
                self.phi_id += 1;
                set_cell(&mut phi_state, c, Expr::register(init.width, name.clone()));
                names.push(name);
                inits.push(init);
            }
            // walk the body once over the phis
            self.loops += 1;
            self.loop_stack.push(LoopCtx {
                header,
                names: names.clone(),
                cells: cells.clone(),
                base: phi_state.clone(),
                pending: Vec::new(),
            });
            let body = self.walk(header, phi_state, trail.clone(), true);
            let ctx = self.loop_stack.pop().expect("loop ctx");
            self.loops -= 1;
            let body = body?;
            if ctx.pending.is_empty() {
                self.loops += 1;
                break 'close (body, ctx, names, inits);
            }
            // cells that moved on a back edge but were not carried: carry them too
            cells.extend(ctx.pending.iter().cloned());
            if cells.len() > 512 {
                return Err("too many loop-carried cells".to_owned());
            }
        };
        // prune phis that nothing observable depends on
        let mut roots: Vec<Expr> = Vec::new();
        collect_roots(&body, &mut roots);
        let mut all_next: Vec<Vec<Expr>> = Vec::new();
        all_back_edges(&body, &mut all_next);
        if all_next.is_empty() { return Err("the loop has no back edge".to_owned()); }
        let root_refs: Vec<&Expr> = roots.iter().collect();
        let live = Self::live_phis(&ctx.cells, &names, &all_next, &root_refs);
        prune(&mut body, &live);
        let phis = live
            .iter()
            .map(|&i| RPhi { name: names[i].clone(), width: inits[i].width, init: inits[i].clone() })
            .collect();
        Ok(RNode::Loop { header, phis, body: Box::new(body) })
    }
}

fn all_back_edges(n: &RNode, out: &mut Vec<Vec<Expr>>) {
    match n {
        RNode::BackEdge { next } => out.push(next.clone()),
        RNode::Fork { arms, .. } => arms.iter().for_each(|(_, a)| all_back_edges(a, out)),
        RNode::Loop { .. } | RNode::Exit(_) | RNode::Dead(_) => {}
    }
}

fn collect_roots(n: &RNode, out: &mut Vec<Expr>) {
    match n {
        RNode::Exit(e) => out.extend(e.registers.values().cloned()),
        RNode::Fork { selector, arms, .. } => {
            out.push(selector.clone());
            for (_, a) in arms {
                collect_roots(a, out);
            }
        }
        RNode::Loop { body, phis, .. } => {
            out.extend(phis.iter().map(|p| p.init.clone()));
            collect_roots(body, out);
        }
        RNode::BackEdge { .. } | RNode::Dead(_) => {}
    }
}

fn prune(n: &mut RNode, live: &[usize]) {
    match n {
        RNode::BackEdge { next } => {
            *next = live.iter().filter_map(|&i| next.get(i).cloned()).collect();
        }
        RNode::Fork { arms, .. } => arms.iter_mut().for_each(|(_, a)| prune(a, live)),
        RNode::Loop { .. } | RNode::Exit(_) | RNode::Dead(_) => {}
    }
}
