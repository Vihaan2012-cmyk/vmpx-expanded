//! The concolic loop: observe -> find an unobserved feasible CFG branch -> path condition ->
//! z3 -> new input -> re-run -> confirm. OBSERVED TIER ONLY: nothing here is a proof.

use std::collections::{BTreeMap, BTreeSet, HashSet, VecDeque};

use vmpx_emulate::VmInstanceSpec;
use vmpx_pe::{PeImage, Rva, Va};
use vmpx_vm::{
    EntryProfileRecoveryLimits, ObservedBranchTargets, ObservedLoopDepths, ObservedMemory, ObservedRun,
    VmCfgLimits, VmDiscoveryLimits, analyze_vm_entry_at, recover_vm_cfg_with_observed,
};

use crate::harness::{Pokes, discover_specs, observe};
use crate::lower::{LowerInfo, lower};
use crate::recover::{Recovered, recover};
use crate::solve::{Lit, Solved, backend, solve};
use crate::symbolic::{successors, walk_expected};

/// Taken VM branches have |delta| above this (same threshold the static bridge uses).
const FAR: u64 = 0x100;
/// Block visits analysed per run (loop unrolling cap: the path-explosion guard).
pub const MAX_VISITS: usize = 10;

/// A concrete input: the first integer argument and overrides of context-relative memory cells.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Input {
    pub arg: u64,
    /// model address -> (value, bytes); emulated at `ctx + address`.
    pub cells: BTreeMap<u64, (u64, u32)>,
}

impl Input {
    #[must_use]
    pub fn describe(&self) -> String {
        let cells: Vec<String> = self.cells.iter().map(|(a, (v, b))| format!("cell[{a:#x}:{b}]={v:#x}")).collect();
        format!("arg={} {}", self.arg as i64, cells.join(" "))
    }
}

#[derive(Debug)]
pub struct Run {
    pub stop: String,
    pub steps: u64,
    pub dispatches: usize,
    pub visits: Vec<Va>,
    pub edges: BTreeSet<(u64, u64)>,
    pub branch_targets: Vec<(u64, u64)>,
}

#[derive(Clone, Debug)]
pub enum Class {
    /// The flipped selector reads machine state (cpuid / gs): no input can steer it.
    EnvGated(Vec<String>),
    /// The selector kept leaves that cannot be lowered (symbolic-address loads, unknown intrinsics).
    Unlowerable(Vec<String>),
    Unsat,
    Solved(Input),
    SolverGaveUp(String),
}

#[derive(Clone, Debug)]
pub struct Attempt {
    pub iter: usize,
    pub edge: (u64, u64),
    pub class: Class,
    /// Set once the solved input was re-run: did the edge actually appear?
    pub confirmed: Option<bool>,
}

#[derive(Debug, Default)]
pub struct IterRecord {
    pub iter: usize,
    pub input: String,
    pub stop: String,
    pub steps: u64,
    pub visits: Vec<u64>,
    pub new_edges: Vec<(u64, u64)>,
    pub covered: usize,
    pub total: usize,
    pub note: String,
}

#[derive(Debug, Default)]
pub struct Report {
    pub solver: String,
    pub iters: Vec<IterRecord>,
    pub attempts: Vec<Attempt>,
    pub observed: BTreeSet<(u64, u64)>,
    pub static_edges: BTreeSet<(u64, u64)>,
    pub branch_targets: BTreeSet<(u64, u64)>,
}

pub struct Engine<'a> {
    pe: &'a PeImage<'a>,
    pub rec: Recovered,
    export: String,
    stub_rva: u32,
    specs: Vec<VmInstanceSpec>,
    base: u64,
    slot_rva: u64,
    ctx: u64,
    budget: u64,
    keys: HashSet<u64>,
    pub static_edges: BTreeSet<(u64, u64)>,
}

impl<'a> Engine<'a> {
    /// # Errors
    /// Recovery or dispatch-discovery failure.
    pub fn new(
        pe: &'a PeImage<'a>,
        export: &str,
        stub_rva: u32,
        slot_rva: u64,
        ctx: u64,
        budget: u64,
    ) -> Result<Self, String> {
        let rec = recover(pe, stub_rva, 100_000)?;
        let base = pe.optional.image_base.0;
        let size = u64::from(pe.optional.size_of_image);
        let base_pokes: Pokes = vec![(base + slot_rva, ctx.to_le_bytes().to_vec())];
        let specs = discover_specs(pe, export, 5, (0, size), &base_pokes, 20_000_000)?;
        let keys = rec.cfg.blocks.keys().map(|k| k.0).collect();
        let mut e = Self {
            pe,
            rec,
            export: export.to_owned(),
            stub_rva,
            specs,
            base,
            slot_rva,
            ctx,
            budget,
            keys,
            static_edges: BTreeSet::new(),
        };
        let input = e.rec.input(e.pe);
        let mut edges = BTreeSet::new();
        for k in e.rec.cfg.blocks.keys() {
            for (_, to) in successors(&input, *k) {
                edges.insert((k.0, to.0));
            }
        }
        e.static_edges = edges;
        Ok(e)
    }

    fn pokes(&self, input: &Input) -> Pokes {
        let mut p: Pokes = vec![(self.base + self.slot_rva, self.ctx.to_le_bytes().to_vec())];
        for (a, (v, b)) in &input.cells {
            p.push((self.ctx + a, v.to_le_bytes()[..*b as usize].to_vec()));
        }
        p
    }

    /// One contained, step-bounded emulator run of the export under `input`.
    ///
    /// # Errors
    /// Session setup failure.
    pub fn run(&self, input: &Input) -> Result<Run, String> {
        let o = observe(self.pe, &self.export, input.arg, &self.specs, &self.pokes(input), self.budget)?;
        let entry = self.rec.cfg.entry.0;
        let mut visits = Vec::new();
        if o.seq.first() == Some(&entry) {
            visits.push(Va(entry));
        }
        for w in o.seq.windows(2) {
            if w[0].abs_diff(w[1]) > FAR && self.keys.contains(&w[1]) {
                visits.push(Va(w[1]));
            }
        }
        let edges = visits.windows(2).map(|w| (w[0].0, w[1].0)).collect();
        Ok(Run {
            stop: format!("{:?}", o.call.stop),
            steps: o.call.steps,
            dispatches: o.seq.len(),
            visits,
            edges,
            branch_targets: o.obs.branch_targets.clone(),
        })
    }

    /// Effective value of each wanted cell under `input` (override, else what the emulator holds).
    fn effective(&self, input: &Input, wanted: &BTreeMap<u64, u32>) -> Result<BTreeMap<u64, u64>, String> {
        let mut s = crate::harness::Session::new(self.pe, 50_000_000)?;
        for (a, b) in self.pokes(input) {
            s.poke(a, &b)?;
        }
        let mut out = BTreeMap::new();
        for (a, bytes) in wanted {
            let v = input.cells.get(a).map_or_else(
                || s.emu.mem.read_uint(self.ctx + a, *bytes as usize).unwrap_or(0),
                |(v, _)| *v,
            );
            out.insert(*a, v);
        }
        Ok(out)
    }

    /// Drive the loop from `i0`.
    ///
    /// # Errors
    /// Emulator session failures.
    #[allow(clippy::too_many_lines)]
    pub fn explore(&self, i0: Input, max_iters: usize) -> Result<Report, String> {
        let (be, be_name) = backend();
        let mut rep = Report {
            solver: be_name.to_owned(),
            static_edges: self.static_edges.clone(),
            ..Report::default()
        };
        let total = self.static_edges.len();
        let mut tried: BTreeSet<(u64, u64)> = BTreeSet::new();
        let mut queue: VecDeque<(Input, (u64, u64))> = VecDeque::new();
        let mut cur = i0;
        let mut pending: Option<(u64, u64)> = None;
        for iter in 0..max_iters {
            let run = self.run(&cur)?;
            let new: Vec<(u64, u64)> = run.edges.iter().filter(|e| !rep.observed.contains(e)).copied().collect();
            rep.observed.extend(run.edges.iter().copied());
            rep.branch_targets.extend(run.branch_targets.iter().copied());
            let mut rec = IterRecord {
                iter,
                input: cur.describe(),
                stop: run.stop.clone(),
                steps: run.steps,
                visits: run.visits.iter().take(MAX_VISITS).map(|v| v.0).collect(),
                new_edges: new,
                covered: rep.observed.len(),
                total,
                note: String::new(),
            };
            if let Some(edge) = pending.take() {
                let ok = rep.observed.contains(&edge);
                if let Some(a) = rep.attempts.iter_mut().rev().find(|a| a.edge == edge && a.confirmed.is_none()) {
                    a.confirmed = Some(ok);
                }
                rec.note = format!("target edge {:#x}->{:#x} {}", edge.0, edge.1, if ok { "CONFIRMED observed" } else { "MISSED (model/approximation wrong)" });
            }

            let visits: Vec<Va> = run.visits.iter().copied().take(MAX_VISITS).collect();
            let input = self.rec.input(self.pe);
            match walk_expected(&input, &visits, 64) {
                Err(e) => {
                    rec.note.push_str(&format!(" | symbolic walk: {e}"));
                }
                Ok(path) => {
                    let lowered: Vec<Result<(vmpx_vm::Expr, LowerInfo), String>> =
                        path.forks.iter().map(|f| lower(&f.selector)).collect();
                    // effective values of every cell the path mentions
                    let mut wanted: BTreeMap<u64, u32> = BTreeMap::new();
                    for (e, i) in lowered.iter().flatten().map(|(e, i)| (e, i)) {
                        let _ = e;
                        for (a, b) in i.cells.values() {
                            wanted.insert(*a, *b);
                        }
                    }
                    let eff = self.effective(&cur, &wanted)?;
                    for (fi, f) in path.forks.iter().enumerate() {
                        for (j, (addr, key)) in f.succ.iter().enumerate() {
                            let edge = (f.block.0, key.0);
                            if j == f.chosen || rep.observed.contains(&edge) || !tried.insert(edge) {
                                continue;
                            }
                            let class = self.classify(&path.forks, &lowered, &eff, fi, *addr, &cur, be.as_ref());
                            if let Class::Solved(inp) = &class {
                                queue.push_back((inp.clone(), edge));
                            }
                            rep.attempts.push(Attempt { iter, edge, class, confirmed: None });
                        }
                    }
                }
            }
            rep.iters.push(rec);
            match queue.pop_front() {
                None => break,
                Some((inp, edge)) => {
                    cur = inp;
                    pending = Some(edge);
                    // mark attempt as run (confirmation happens next iteration)
                    let _ = edge;
                }
            }
        }
        Ok(rep)
    }

    #[allow(clippy::too_many_arguments)]
    fn classify(
        &self,
        forks: &[crate::symbolic::Fork],
        lowered: &[Result<(vmpx_vm::Expr, LowerInfo), String>],
        eff: &BTreeMap<u64, u64>,
        fi: usize,
        target_addr: u64,
        cur: &Input,
        be: &dyn vmpx_verify::equiv::SmtBackend,
    ) -> Class {
        let Ok((te, ti)) = &lowered[fi] else {
            return Class::Unlowerable(vec!["lowering failed".into()]);
        };
        if !ti.stuck.is_empty() {
            return Class::Unlowerable(ti.stuck.clone());
        }
        if !ti.env.is_empty() {
            return Class::EnvGated(ti.env.iter().cloned().collect());
        }
        let mut pins: BTreeMap<String, vmpx_vm::Expr> = BTreeMap::new();
        let mut lits: Vec<Lit> = Vec::new();
        for (k, fk) in forks.iter().enumerate().take(fi) {
            let Ok((e, i)) = &lowered[k] else { continue };
            if i.stuck.is_empty() && i.env.is_empty() {
                lits.push(Lit { sel: e.clone(), val: fk.succ[fk.chosen].0, eq: true });
            } else {
                // env-involved prefix: assume the machine state is unchanged on re-run (concretise)
                // and hold its memory cells at their current values.
                for (name, (a, b)) in &i.cells {
                    if !ti.cells.contains_key(name) {
                        let w = if *b == 8 { vmpx_vm::BitWidth::W64 } else { vmpx_vm::BitWidth::W32 };
                        pins.insert(name.clone(), vmpx_vm::Expr::constant(w, eff.get(a).copied().unwrap_or(0)));
                    }
                }
            }
        }
        lits.push(Lit { sel: te.clone(), val: target_addr, eq: true });
        for l in &mut lits {
            if let Ok(s) = l.sel.substitute(&pins) {
                l.sel = s;
            }
        }
        match solve(be, &lits, 20_000) {
            Solved::Unsat => Class::Unsat,
            Solved::Uninterpreted(w) | Solved::Other(w) => Class::SolverGaveUp(w),
            Solved::Sat(model) => {
                let mut next = cur.clone();
                let mut all_cells: BTreeMap<String, (u64, u32)> = ti.cells.clone();
                for (e, i) in lowered.iter().flatten() {
                    let _ = e;
                    all_cells.extend(i.cells.clone());
                }
                for (name, v) in model {
                    if let Some((a, b)) = all_cells.get(&name) {
                        let v = if *b == 8 { v } else { v & 0xffff_ffff };
                        if eff.get(a).copied() != Some(v) {
                            next.cells.insert(*a, (v, *b));
                        }
                    } else if name == "native_rcx_initial" {
                        next.arg = v;
                    }
                }
                if next == *cur {
                    Class::SolverGaveUp("model equals the current input (condition holds already?)".into())
                } else {
                    Class::Solved(next)
                }
            }
        }
    }

    /// Feed the observed edges through the static observed-branch-target bridge and report what
    /// the static side did with them.
    ///
    /// # Errors
    /// Static recovery failure.
    pub fn bridge(&self, rep: &Report) -> Result<String, String> {
        let bt = ObservedBranchTargets::from_vip_edges(rep.branch_targets.iter().copied());
        let candidate = analyze_vm_entry_at(self.pe, Rva(self.stub_rva), VmDiscoveryLimits::default())
            .ok_or("no entry candidate")?;
        let limits = VmCfgLimits {
            entry: EntryProfileRecoveryLimits { instruction_limit: 100_000, ..EntryProfileRecoveryLimits::default() },
            ..VmCfgLimits::default()
        };
        let mut run = ObservedRun::new(ObservedMemory::from_slots(Vec::new()), ObservedLoopDepths::new());
        run.branch_targets = bt.clone();
        let cfg = recover_vm_cfg_with_observed(self.pe, &candidate, limits, &run).map_err(|e| e.to_string())?;
        let mut facts = vmpx_emulate::ObservedFacts::new();
        for &(s, t) in &rep.branch_targets {
            facts.record_branch_target(s, t);
        }
        Ok(format!(
            "ObservedBranchTargets::from_vip_edges: {} far-edge branch site(s) from {} raw (site,target) pairs; ObservedFacts.branch_targets={} (Observed-stamped); \
             recover_vm_cfg_with_observed -> blocks={} edges={} complete={} assumptions={:?} (baseline blocks={} edges={})",
            bt.site_count(),
            rep.branch_targets.len(),
            facts.branch_targets.len(),
            cfg.blocks.len(),
            cfg.edges.len(),
            cfg.is_complete(),
            cfg.assumptions,
            self.rec.cfg.blocks.len(),
            self.rec.cfg.edges.len(),
        ))
    }
}
