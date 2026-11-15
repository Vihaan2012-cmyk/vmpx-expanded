//! Cut-point execution of a native function with input-dependent loops.
//!
//! A loop whose branches the flags cannot decide never terminates the forking
//! executor: each revisit forks again with a new literal. Here the CFG is cut at a
//! feedback vertex set (every cycle contains a cut), so between cuts the code is
//! acyclic and finite. The function is then a family of *segments*:
//!
//! * the entry segment, from the function entry to the cuts and exits it reaches;
//! * one segment per cut, from the cut block to the next cuts and exits, started
//!   from a state in which every register and XMM lane is a fresh symbol (a phi).
//!
//! Each segment is a finite, exclusive and exhaustive path family over its start
//! state, exactly as [`super::execute_native_paths`] forks. A path ends in an exit
//! or at a cut, carrying the values the phis take next. Nothing is guessed: a loop
//! is represented, not unrolled, and the result is only evidence once a prover has
//! discharged the induction (see `crate::loops`).
//!
//! Restrictions, each failing closed:
//! * flags are not carried; a cut block that reads a flag it did not define is
//!   never chosen as a cut (and a cycle without a clean block is refused);
//! * `rsp`, the stores and the calls made before a cut are not phis: they must be
//!   identical on every arrival at the cut, and a segment ending at a cut may add
//!   no store and no call (so memory is the same in every iteration);
//! * no phi value may mention the frame address, so a phi can never name the
//!   function's own frame (what lets a phi be an "entry-time pointer").

use std::collections::{BTreeMap, BTreeSet};

use vmpx_disasm::{Flag, FlagState, Gpr};
use vmpx_mutation::cfg_recovery::{NativeBlock, NativeCfg, Terminator};
use vmpx_vm::{BitWidth, Expr};
use vmpx_vmlift::CollectedCall;

use super::{
    CutState, ForkLimits, NativeEnv, SegmentEnd, SegmentPath, Store, entry_run, mentions_frame, run_segments,
};

/// Symbol-name prefix of a loop phi; `differential::is_entry_pointer` accepts it.
pub const PHI_PREFIX: &str = "native_phi";

/// One loop-carried machine component.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Component {
    Gpr(Gpr),
    /// One 64-bit XMM lane.
    Xmm(u8, bool),
}

/// The carried components in a fixed order: every GPR but `rsp`, then the XMM lanes.
#[must_use]
pub fn components() -> Vec<Component> {
    let mut v: Vec<Component> = Gpr::ALL.iter().filter(|g| **g != Gpr::Rsp).map(|g| Component::Gpr(*g)).collect();
    for n in 0..16u8 {
        v.push(Component::Xmm(n, false));
        v.push(Component::Xmm(n, true));
    }
    v
}

fn phi_symbol(header: u64, c: Component) -> Expr {
    let tail = match c {
        Component::Gpr(g) => g.name().to_owned(),
        Component::Xmm(n, hi) => format!("xmm{n}{}", if hi { "hi" } else { "lo" }),
    };
    Expr::register(BitWidth::W64, format!("{PHI_PREFIX}{header:x}_{tail}"))
}

/// A cut's value of each carried component, in [`components`] order.
fn component_values(state: &CutState) -> Vec<Expr> {
    components()
        .into_iter()
        .map(|c| match c {
            Component::Gpr(g) => state.regs[&g].clone(),
            Component::Xmm(n, hi) => {
                let (lo, hi_lane) = &state.xmm[&n];
                if hi { hi_lane.clone() } else { lo.clone() }
            }
        })
        .collect()
}

/// One cut's segment family.
#[derive(Clone, Debug)]
pub struct LoopSegment {
    /// The phi symbols, in [`components`] order.
    pub phis: Vec<Expr>,
    pub paths: Vec<SegmentPath>,
}

/// A native function as entry segment plus one segment per cut.
#[derive(Clone, Debug)]
pub struct NativeLoops {
    pub cuts: BTreeSet<u64>,
    pub entry: Vec<SegmentPath>,
    pub loops: BTreeMap<u64, LoopSegment>,
    /// Labeled modeling assumptions any segment depends on.
    pub assumptions: Vec<String>,
}

/// The flags a condition code reads.
fn cc_flags(cc: &str) -> &'static [Flag] {
    match cc {
        "o" | "no" => &[Flag::Of],
        "b" | "ae" => &[Flag::Cf],
        "e" | "ne" => &[Flag::Zf],
        "be" | "a" => &[Flag::Cf, Flag::Zf],
        "s" | "ns" => &[Flag::Sf],
        "p" | "np" => &[Flag::Pf],
        "l" | "ge" => &[Flag::Sf, Flag::Of],
        "le" | "g" => &[Flag::Zf, Flag::Sf, Flag::Of],
        _ => &[Flag::Cf, Flag::Pf, Flag::Af, Flag::Zf, Flag::Sf, Flag::Of],
    }
}

const ALL_FLAGS: [Flag; 6] = [Flag::Cf, Flag::Pf, Flag::Af, Flag::Zf, Flag::Sf, Flag::Of];

/// True when the block reads a flag value it did not itself define (or cannot be
/// analysed): such a block cannot start a segment, whose flags are undefined.
fn reads_incoming_flags(block: &NativeBlock) -> bool {
    let Some(sem) = &block.semantics else {
        return true;
    };
    let mut roots: Vec<&Expr> = sem.regs.values().collect();
    for st in sem.flags.values() {
        if let FlagState::Value(e) = st {
            roots.push(e);
        }
    }
    for w in &sem.writes {
        roots.push(&w.address);
        roots.push(&w.value);
    }
    for (lo, hi) in sem.xmm.values() {
        roots.push(lo);
        roots.push(hi);
    }
    for d in &sem.deferred_alias {
        roots.push(&d.load_address);
        roots.push(&d.store_address);
    }
    let names: Vec<String> = ALL_FLAGS.iter().map(|f| format!("{}_in", f.name())).collect();
    if roots.iter().any(|e| e.dependencies().iter().any(|d| names.contains(d))) {
        return true;
    }
    if let Terminator::CondJump { cc, .. } = &block.terminator {
        return cc_flags(cc)
            .iter()
            .any(|f| !matches!(sem.flags.get(f), Some(FlagState::Value(_))));
    }
    false
}

fn successors(t: &Terminator) -> Vec<u64> {
    match t {
        Terminator::FallThrough(a) | Terminator::Jump(a) => vec![*a],
        Terminator::CondJump { taken, not_taken, .. } => vec![*taken, *not_taken],
        Terminator::Call { next, .. } => vec![*next],
        Terminator::Return | Terminator::TailCall { .. } | Terminator::Unsupported(_) => Vec::new(),
    }
}

struct Tarjan<'a> {
    succ: &'a BTreeMap<u64, Vec<u64>>,
    cuts: &'a BTreeSet<u64>,
    index: BTreeMap<u64, usize>,
    low: BTreeMap<u64, usize>,
    on_stack: BTreeSet<u64>,
    stack: Vec<u64>,
    next: usize,
    out: Vec<Vec<u64>>,
}

impl Tarjan<'_> {
    fn visit(&mut self, v: u64) {
        self.index.insert(v, self.next);
        self.low.insert(v, self.next);
        self.next += 1;
        self.stack.push(v);
        self.on_stack.insert(v);
        for &w in &self.succ[&v] {
            if self.cuts.contains(&w) || !self.succ.contains_key(&w) {
                continue;
            }
            if !self.index.contains_key(&w) {
                self.visit(w);
                let lw = self.low[&w];
                let lv = self.low[&v];
                self.low.insert(v, lv.min(lw));
            } else if self.on_stack.contains(&w) {
                let iw = self.index[&w];
                let lv = self.low[&v];
                self.low.insert(v, lv.min(iw));
            }
        }
        if self.low[&v] == self.index[&v] {
            let mut comp = Vec::new();
            while let Some(w) = self.stack.pop() {
                self.on_stack.remove(&w);
                comp.push(w);
                if w == v {
                    break;
                }
            }
            self.out.push(comp);
        }
    }
}

/// A feedback vertex set of the reachable CFG whose members all start with a
/// defined flag state: removing it leaves the graph acyclic, so every cycle passes
/// through a cut.
///
/// # Errors
/// A cycle in which every block reads a flag it did not define.
pub fn loop_cuts(cfg: &NativeCfg) -> Result<BTreeSet<u64>, String> {
    let mut succ: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
    let mut work = vec![cfg.entry];
    while let Some(b) = work.pop() {
        if succ.contains_key(&b) {
            continue;
        }
        let Some(block) = cfg.blocks.get(&b) else { continue };
        let next: Vec<u64> = successors(&block.terminator).into_iter().filter(|t| cfg.blocks.contains_key(t)).collect();
        work.extend(next.iter().copied());
        succ.insert(b, next);
    }
    let mut cuts: BTreeSet<u64> = BTreeSet::new();
    loop {
        let mut t = Tarjan {
            succ: &succ,
            cuts: &cuts,
            index: BTreeMap::new(),
            low: BTreeMap::new(),
            on_stack: BTreeSet::new(),
            stack: Vec::new(),
            next: 0,
            out: Vec::new(),
        };
        for &v in succ.keys() {
            if !cuts.contains(&v) && !t.index.contains_key(&v) {
                t.visit(v);
            }
        }
        let mut added = false;
        for comp in t.out {
            let cyclic = comp.len() > 1 || succ[&comp[0]].contains(&comp[0]);
            if !cyclic {
                continue;
            }
            let members: BTreeSet<u64> = comp.iter().copied().collect();
            let in_degree = |n: u64| members.iter().filter(|m| succ[*m].contains(&n)).count();
            let pick = comp
                .iter()
                .copied()
                .filter(|n| cfg.blocks.get(n).is_some_and(|b| !reads_incoming_flags(b)))
                .max_by_key(|n| (in_degree(*n), std::cmp::Reverse(*n)));
            let Some(pick) = pick else {
                let at = comp.iter().min().copied().unwrap_or(0);
                return Err(format!(
                    "the loop containing block {at:#x} has no block that starts with a defined flag state to cut at"
                ));
            };
            cuts.insert(pick);
            added = true;
        }
        if !added {
            return Ok(cuts);
        }
    }
}

/// The memory/effect context every arrival at a cut must agree on.
struct CutInfo {
    rsp: Expr,
    stores: Vec<Store>,
    calls: Vec<CollectedCall>,
    frame_escaped: bool,
}

fn same_stores(a: &[Store], b: &[Store]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| {
            x.2 == y.2 && x.0.simplify() == y.0.simplify() && x.1.simplify() == y.1.simplify()
        })
}

fn same_calls(a: &[CollectedCall], b: &[CollectedCall]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| {
            x.iat_rva == y.iat_rva
                && x.result_intrinsic == y.result_intrinsic
                && x.arguments.len() == y.arguments.len()
                && x.arguments.iter().zip(&y.arguments).all(|(p, q)| p.simplify() == q.simplify())
        })
}

/// Records an arrival at a cut, enforcing the restrictions in the module docs.
/// `origin` is the cut the arriving segment started at (`None`: the entry).
fn register(
    infos: &mut BTreeMap<u64, CutInfo>,
    queue: &mut Vec<u64>,
    assumptions: &mut BTreeSet<String>,
    origin: Option<u64>,
    header: u64,
    state: &CutState,
) -> Result<(), String> {
    if let Some(from) = origin {
        let src = &infos[&from];
        if !same_stores(&src.stores, &state.stores) || !same_calls(&src.calls, &state.calls) {
            return Err(format!(
                "the loop segment from {from:#x} to {header:#x} stores or calls; memory would differ between iterations"
            ));
        }
    }
    if component_values(state).iter().any(mentions_frame) {
        return Err(format!("a value carried to the loop at {header:#x} mentions the frame address"));
    }
    assumptions.extend(state.assumptions.iter().cloned());
    let rsp = state.regs[&Gpr::Rsp].simplify();
    match infos.get(&header) {
        None => {
            infos.insert(
                header,
                CutInfo {
                    rsp,
                    stores: state.stores.clone(),
                    calls: state.calls.clone(),
                    frame_escaped: state.frame_escaped,
                },
            );
            queue.push(header);
        }
        Some(info) => {
            if info.rsp != rsp
                || info.frame_escaped != state.frame_escaped
                || !same_stores(&info.stores, &state.stores)
                || !same_calls(&info.calls, &state.calls)
            {
                return Err(format!(
                    "arrivals at the loop block {header:#x} disagree on the stack pointer, stores, calls or frame escape"
                ));
            }
        }
    }
    Ok(())
}

/// Executes `cfg` as an entry segment plus one segment per loop cut.
///
/// # Errors
/// As [`super::execute_native_paths`] for any segment, plus every restriction in the
/// module docs.
pub fn execute_native_loops(
    cfg: &NativeCfg,
    max_blocks: usize,
    native_env: &NativeEnv,
    limits: ForkLimits,
) -> Result<NativeLoops, String> {
    let cuts = loop_cuts(cfg)?;
    let mut total = 0_usize;
    let entry = run_segments(cfg, max_blocks, native_env, limits, &cuts, entry_run(cfg), &mut total)?;
    let mut infos: BTreeMap<u64, CutInfo> = BTreeMap::new();
    let mut queue: Vec<u64> = Vec::new();
    let mut assumptions: BTreeSet<String> = BTreeSet::new();
    for leaf in &entry {
        match &leaf.end {
            SegmentEnd::Cut { header, state } => register(&mut infos, &mut queue, &mut assumptions, None, *header, state)?,
            SegmentEnd::Exit(f) => assumptions.extend(f.assumptions.iter().cloned()),
        }
    }
    let comps = components();
    let mut loops: BTreeMap<u64, LoopSegment> = BTreeMap::new();
    while let Some(header) = queue.pop() {
        let info = &infos[&header];
        let mut first = entry_run(cfg);
        first.ip = header;
        first.executed = 0;
        for g in Gpr::ALL {
            let v = if g == Gpr::Rsp { info.rsp.clone() } else { phi_symbol(header, Component::Gpr(g)) };
            first.regs.insert(g, v);
        }
        for n in 0..16u8 {
            first.xmm.insert(n, (phi_symbol(header, Component::Xmm(n, false)), phi_symbol(header, Component::Xmm(n, true))));
        }
        first.stores = info.stores.clone();
        first.calls = info.calls.clone();
        first.frame_escaped = info.frame_escaped;
        let paths = run_segments(cfg, max_blocks, native_env, limits, &cuts, first, &mut total)?;
        for leaf in &paths {
            match &leaf.end {
                SegmentEnd::Cut { header: next, state } => {
                    register(&mut infos, &mut queue, &mut assumptions, Some(header), *next, state)?;
                }
                SegmentEnd::Exit(f) => assumptions.extend(f.assumptions.iter().cloned()),
            }
        }
        loops.insert(header, LoopSegment { phis: comps.iter().map(|c| phi_symbol(header, *c)).collect(), paths });
    }
    Ok(NativeLoops { cuts, entry, loops, assumptions: assumptions.into_iter().collect() })
}

/// Values a path carries to its destination cut, in [`components`] order.
#[must_use]
pub fn carried(state: &CutState) -> Vec<Expr> {
    component_values(state)
}
