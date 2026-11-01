//! Layer 5: a [`Hook`] that observes a VM run — the sequence of handler entries
//! the dispatchers transfer to, VIP-keyed edges per interpreter instance, and
//! the concrete recursion depth of a data-dependent loop/recursion under ONE
//! concrete input.
//!
//! # Containment
//!
//! Pure observation: every callback only reads `cpu`/`mem` and appends to an
//! in-model record; it never redirects control, touches the host, or guesses.
//! The run is driven entirely by the software interpreter.
//!
//! # Soundness / provenance
//!
//! The handler trace, VIP edges and recursion depth reflect ONE concrete
//! execution path, not a proof. [`VmObservations::seed_loop_depth`] and
//! [`VmObservations::seed_branch_targets`] stamp them into [`ObservedFacts`] as
//! [`crate::Observed`] facts so the static side can SEED its lifts while
//! keeping them on the quarantined tier — never sound output. When several
//! instances share a dispatch slice and the live VIP matches none of their
//! proven bytecode regions, the dispatch is counted in
//! [`VmObservations::unattributed`] and no edge is recorded.
//!
//! # Install
//!
//! ```ignore
//! // one dispatcher at an absolute VA (VIP in RSI):
//! let hook = VmObserveHook::new(Some(dispatcher_va), Some(loop_head_va));
//! // many instances: dispatchers keyed by RVA, resolved through the load base
//! let hook = VmObserveHook::with_instances(base, specs, None);
//! let obs = hook.observations();         // shared handle, clone before install
//! let mut emu = ...;
//! emu.hook = Some(Box::new(hook));
//! emu.run_until(sentinel, MAX);
//! let o = obs.borrow();
//! let handlers = &o.handlers;            // dispatched handler entries, in order
//! o.seed_branch_targets(&mut facts);     // observed VIP edges -> ObservedFacts
//! ```

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::rc::Rc;

use iced_x86::Register;

use crate::cpu::Cpu;
use crate::emu::{FlowEvent, FlowKind, Hook, HookAction, Interest};
use crate::memory::Memory;
use crate::observe::ObservedFacts;

/// Where an instance keeps its VIP at dispatch time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VipLoc {
    Reg(Register),
    /// Qword at `base + disp`: a VIP spilled to memory (e.g. a stack slot).
    Mem { base: Register, disp: i64 },
}

impl From<Register> for VipLoc {
    fn from(r: Register) -> Self {
        VipLoc::Reg(r)
    }
}

impl VipLoc {
    /// The VIP, or `None` if a register/slot is unreadable. `rsp_bias` is
    /// added to an RSP base (a transfer sited after the stub's `push`).
    fn read(self, cpu: &Cpu, mem: &Memory, rsp_bias: u64) -> Option<u64> {
        match self {
            VipLoc::Reg(r) => cpu.reg(r),
            VipLoc::Mem { base, disp } => {
                let b = cpu.reg(base)?;
                let b = if base == Register::RSP { b.wrapping_add(rsp_bias) } else { b };
                mem.read_uint(b.wrapping_add(disp as u64), 8).ok()
            }
        }
    }
}

/// One interpreter instance behind a dispatcher.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VmInstance {
    /// Specs sharing an id are one instance (one walk) across their dispatchers.
    pub id: u32,
    /// Where this instance's VIP lives at dispatcher entry.
    pub vip: VipLoc,
    /// Proven bytecode region as a half-open RVA span. `None` = unconstrained,
    /// accepted only when the instance is alone on its dispatch slice.
    pub bytecode_region: Option<(u64, u64)>,
}

/// How a spec's `dispatcher_rva` is recognised at run time.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DispatchAt {
    /// Control arrives at the dispatcher; its first `jmp`/`ret` out is the
    /// handler transfer and the VIP is read on arrival.
    #[default]
    Arrival,
    /// The address IS the dispatching `jmp reg` / `ret` itself, seen by its
    /// source (so a stub entered by fall-through is not missed). The VIP is
    /// read at the transfer; `rsp_bias` is added to an RSP-based VIP slot
    /// (8 when the transfer is the `ret` of a `push reg; ret` stub).
    Transfer { rsp_bias: u64 },
}

/// A statically discovered dispatcher (by RVA) and the instance it serves.
/// Several specs may share one `dispatcher_rva` (shared native dispatch slice).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VmInstanceSpec {
    pub dispatcher_rva: u64,
    pub at: DispatchAt,
    pub instance: VmInstance,
}

/// A VM edge observed in one concrete run: the VIPs at consecutive dispatches
/// of `instance` and the handler executed between them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ObservedVmEdge {
    pub instance: u32,
    pub from_vip: u64,
    pub to_vip: u64,
    /// Handler entered from the `from_vip` dispatch (0 if none was seen).
    pub handler_va: u64,
    /// Times this exact edge was observed.
    pub dispatch_count: u64,
}

/// Shared recording filled during a run. Clone the [`Rc`] before installing the
/// hook, then read it afterwards.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VmObservations {
    /// Lowest dispatcher entry VA the run was watched against, if any.
    pub dispatcher: Option<u64>,
    /// Loop-head / recursion-entry VA whose back-edge visits are counted.
    pub loop_head: Option<u64>,
    /// Handler entries the dispatchers transferred to, in dispatch order.
    pub handlers: Vec<u64>,
    /// Count of control-flow transfers that landed on `loop_head` — the
    /// observed recursion depth for this concrete input.
    pub back_edges: u64,
    /// Observed VM VIP edges `(site_vip, target_vip)` across all instances:
    /// at each dispatcher re-entry the VIP register holds the bytecode VIP
    /// about to be decoded, so consecutive dispatch VIPs form the edge from one
    /// bytecode site to the next the VM reached. De-duped. The branch SITEs the
    /// static `cfg` stalls on are the subset it queries; straight-line edges
    /// are inert (the static side keys by site and independently re-validates
    /// every target as real in-image bytecode).
    pub branch_targets: Vec<(u64, u64)>,
    /// The same edges keyed per instance, with the handler between them and a
    /// hit count; de-duped on `(instance, from_vip, to_vip, handler_va)`.
    pub edges: Vec<ObservedVmEdge>,
    /// Dispatcher entries whose live VIP matched no candidate instance.
    pub unattributed: u64,
    /// New edges not recorded because the edge table hit its cap.
    pub edges_dropped: u64,
    /// Bumped by [`VmObservations::break_chain`]; the hook then forgets every instance's
    /// last VIP so no edge is recorded across two independent guest calls.
    pub chain_breaks: u64,
}

impl VmObservations {
    /// Mark an independent guest-call boundary (e.g. between lifecycle entry points).
    pub fn break_chain(&mut self) {
        self.chain_breaks += 1;
    }

    /// Flush the observed concrete recursion depth into `facts` (one
    /// [`crate::Observed`]-stamped entry). No-op if no loop-head was configured.
    pub fn seed_loop_depth(&self, facts: &mut ObservedFacts) {
        if let Some(h) = self.loop_head {
            facts.record_loop_depth(h, self.back_edges);
        }
    }

    /// Flush the observed VM VIP edges into `facts` as [`crate::Observed`]-
    /// stamped branch targets. The static side groups them by site.
    pub fn seed_branch_targets(&self, facts: &mut ObservedFacts) {
        for &(site, target) in &self.branch_targets {
            facts.record_branch_target(site, target);
        }
    }

    /// Edges recorded for one instance.
    pub fn instance_edges(&self, instance: u32) -> impl Iterator<Item = &ObservedVmEdge> {
        self.edges.iter().filter(move |e| e.instance == instance)
    }
}

/// Cap on recorded handler entries / distinct edges (memory backstop).
const MAX_RECORDS: usize = 1 << 21;

/// An instance with its region resolved to absolute VAs, plus run state.
struct Live {
    id: u32,
    vip: VipLoc,
    region: Option<(u64, u64)>,
    rsp_bias: u64,
}

/// Per-instance walk state, shared by every dispatcher serving the instance.
#[derive(Default)]
struct Walk {
    last_vip: Option<u64>,
    last_handler: u64,
}

/// Observes VM-dispatcher handler traces, per-instance VIP edges and concolic
/// recursion depth.
///
/// Handler detection ("to/through the dispatcher"): a transfer whose target is
/// a dispatcher marks that the run is inside it; the next `jmp` out of the
/// dispatcher is its indirect dispatch, and its target is recorded as the
/// entered handler. This assumes the dispatcher reaches a handler via its first
/// `jmp` after entry — the `VMProtect` central-dispatch shape.
pub struct VmObserveHook {
    obs: Rc<RefCell<VmObservations>>,
    /// Arrival-dispatcher VA -> candidate indices into `live`.
    dispatchers: BTreeMap<u64, Vec<usize>>,
    /// Transfer-site VA (a `jmp`/`ret` source) -> candidate indices into `live`.
    tails: HashMap<u64, Vec<usize>>,
    live: Vec<Live>,
    walks: HashMap<u32, Walk>,
    /// Instance entered at the last dispatcher arrival, awaiting its jmp-out.
    armed: Option<usize>,
    /// `(instance, from, to, handler)` -> index into `obs.edges`.
    edge_ix: HashMap<(u32, u64, u64, u64), usize>,
    /// De-dupe set mirroring `obs.branch_targets`.
    bt_seen: HashSet<(u64, u64)>,
    /// Flow spans covering every dispatcher and the loop head.
    spans: [(u64, u64); 2],
    seen_breaks: u64,
}

/// Cover the points with at most two half-open spans, splitting at the largest
/// gap.
fn cover(mut pts: Vec<u64>) -> [(u64, u64); 2] {
    pts.sort_unstable();
    pts.dedup();
    let one = |a: &[u64]| match (a.first(), a.last()) {
        (Some(&lo), Some(&hi)) => (lo, hi.saturating_add(1)),
        _ => (0, 0),
    };
    if pts.len() < 2 {
        return [one(&pts), (0, 0)];
    }
    let k = (1..pts.len()).max_by_key(|&i| pts[i] - pts[i - 1]).unwrap_or(1);
    [one(&pts[..k]), one(&pts[k..])]
}

impl VmObserveHook {
    /// Single dispatcher at absolute `dispatcher` (VIP in RSI, unconstrained
    /// region, instance id 0); `loop_head` is an absolute VA.
    #[must_use]
    pub fn new(dispatcher: Option<u64>, loop_head: Option<u64>) -> Self {
        let spec = dispatcher.map(|d| VmInstanceSpec {
            dispatcher_rva: d,
            at: DispatchAt::Arrival,
            instance: VmInstance { id: 0, vip: Register::RSI.into(), bytecode_region: None },
        });
        Self::with_instances(0, spec, loop_head)
    }

    /// Multi-instance observer. Dispatcher and bytecode-region RVAs are
    /// resolved through `base` (the loaded image base), so a dumped image at a
    /// different base works unchanged. Specs sharing a `dispatcher_rva` are
    /// disambiguated at run time by the live VIP against their regions.
    /// `loop_head` is an absolute VA.
    #[must_use]
    pub fn with_instances(
        base: u64,
        specs: impl IntoIterator<Item = VmInstanceSpec>,
        loop_head: Option<u64>,
    ) -> Self {
        let mut dispatchers: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
        let mut tails: HashMap<u64, Vec<usize>> = HashMap::new();
        let mut live = Vec::new();
        for s in specs {
            let va = base.wrapping_add(s.dispatcher_rva);
            let rsp_bias = match s.at {
                DispatchAt::Arrival => {
                    dispatchers.entry(va).or_default().push(live.len());
                    0
                }
                DispatchAt::Transfer { rsp_bias } => {
                    tails.entry(va).or_default().push(live.len());
                    rsp_bias
                }
            };
            live.push(Live {
                rsp_bias,
                id: s.instance.id,
                vip: s.instance.vip,
                region: s
                    .instance
                    .bytecode_region
                    .map(|(lo, hi)| (base.wrapping_add(lo), base.wrapping_add(hi))),
            });
        }
        let mut pts: Vec<u64> = dispatchers.keys().copied().collect();
        pts.extend(loop_head);
        Self {
            obs: Rc::new(RefCell::new(VmObservations {
                dispatcher: dispatchers.keys().next().copied(),
                loop_head,
                ..VmObservations::default()
            })),
            dispatchers,
            tails,
            live,
            walks: HashMap::new(),
            armed: None,
            edge_ix: HashMap::new(),
            bt_seen: HashSet::new(),
            spans: cover(pts),
            seen_breaks: 0,
        }
    }

    /// Shared handle to the observations collected during the run.
    #[must_use]
    pub fn observations(&self) -> Rc<RefCell<VmObservations>> {
        Rc::clone(&self.obs)
    }

    /// Instance among `cands` and its live VIP, chosen by matching the VIP
    /// against each candidate's bytecode region.
    fn resolve(&self, cands: &[usize], cpu: &Cpu, mem: &Memory) -> Option<(usize, u64)> {
        cands.iter().find_map(|&i| {
            let l = &self.live[i];
            let vip = l.vip.read(cpu, mem, l.rsp_bias)?;
            let ok = match l.region {
                Some((lo, hi)) => (lo..hi).contains(&vip),
                None => cands.len() == 1,
            };
            ok.then_some((i, vip))
        })
    }
}

impl VmObserveHook {
    /// Record the edge from the instance's previous dispatch VIP to `vip`.
    fn note_dispatch(&mut self, o: &mut VmObservations, i: usize, vip: u64) {
        let id = self.live[i].id;
        let w = self.walks.entry(id).or_default();
        let handler = w.last_handler;
        let Some(site) = w.last_vip.replace(vip) else { return };
        if o.branch_targets.len() < MAX_RECORDS && self.bt_seen.insert((site, vip)) {
            o.branch_targets.push((site, vip));
        }
        let edges = &mut o.edges;
        let key = (id, site, vip, handler);
        let ix = match self.edge_ix.get(&key) {
            Some(&ix) => Some(ix),
            None if edges.len() < MAX_RECORDS => {
                edges.push(ObservedVmEdge {
                    instance: id,
                    from_vip: site,
                    to_vip: vip,
                    handler_va: handler,
                    dispatch_count: 0,
                });
                self.edge_ix.insert(key, edges.len() - 1);
                Some(edges.len() - 1)
            }
            None => None,
        };
        match ix {
            Some(ix) => edges[ix].dispatch_count += 1,
            None => o.edges_dropped += 1,
        }
    }
}

impl Hook for VmObserveHook {
    fn interest(&self) -> Interest {
        // Transfer sites are matched by source, which spans cannot select.
        // Once inside a dispatcher the next jmp (target unknown) is the handler.
        if self.armed.is_some() || !self.tails.is_empty() {
            return Interest { flow_all: true, ..Interest::NONE };
        }
        Interest { flow: self.spans, ..Interest::NONE }
    }

    fn on_flow(&mut self, ev: &FlowEvent, cpu: &mut Cpu, mem: &mut Memory) -> HookAction {
        let obs = Rc::clone(&self.obs);
        let mut o = obs.borrow_mut();

        if o.chain_breaks != self.seen_breaks {
            self.seen_breaks = o.chain_breaks;
            self.armed = None;
            self.walks.clear();
        }

        // Concolic depth: every arrival at the loop head is a back-edge visit.
        if o.loop_head == Some(ev.to) {
            o.back_edges += 1;
        }

        // Handler trace: record the dispatcher's first jmp/ret-out (`jmp reg`
        // or the threaded `push reg; ret`) as a handler entry, then re-arm when
        // control returns to a dispatcher.
        if let (Some(i), FlowKind::Jmp | FlowKind::Ret) = (self.armed, ev.kind) {
            if o.handlers.len() < MAX_RECORDS {
                o.handlers.push(ev.to);
            }
            self.walks.entry(self.live[i].id).or_default().last_handler = ev.to;
            self.armed = None;
        }
        if let Some(cands) = self.dispatchers.get(&ev.to) {
            let Some((i, vip)) = self.resolve(cands, cpu, mem) else {
                o.unattributed += 1;
                return HookAction::Continue;
            };
            self.armed = Some(i);
            // At dispatcher entry the VIP is the bytecode site about to be
            // decoded; the previous dispatch VIP -> this one is the edge.
            self.note_dispatch(&mut o, i, vip);
        }

        // Transfer-sited dispatch: VIP and handler are both known right here.
        if matches!(ev.kind, FlowKind::Jmp | FlowKind::Ret) {
            if let Some(cands) = self.tails.get(&ev.from) {
                let Some((i, vip)) = self.resolve(cands, cpu, mem) else {
                    o.unattributed += 1;
                    return HookAction::Continue;
                };
                self.note_dispatch(&mut o, i, vip);
                if o.handlers.len() < MAX_RECORDS {
                    o.handlers.push(ev.to);
                }
                self.walks.entry(self.live[i].id).or_default().last_handler = ev.to;
            }
        }

        HookAction::Continue
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DISPATCHER: u64 = 0x1_4000_5000;

    /// Drive one dispatcher re-entry with `vip` in RSI (the VMProtect 3.11
    /// Compact VIP register).
    fn dispatch(hook: &mut VmObserveHook, cpu: &mut Cpu, mem: &mut Memory, vip: u64) {
        cpu.set_reg(Register::RSI, vip);
        let ev = FlowEvent { kind: FlowKind::Jmp, from: 0, to: DISPATCHER, next: 0 };
        hook.on_flow(&ev, cpu, mem);
    }

    #[test]
    fn records_vm_branch_target_edge_at_dispatch() {
        let mut hook = VmObserveHook::new(Some(DISPATCHER), None);
        let obs = hook.observations();
        let mut cpu = Cpu::new(0);
        let mut mem = Memory::new();

        // VIP walks two straight-line sites, then branches to 0x2000; the loop
        // back-edge re-takes the same branch so the edge is seen twice.
        for vip in [0x1000_u64, 0x1004, 0x2000, 0x1004, 0x2000] {
            dispatch(&mut hook, &mut cpu, &mut mem, vip);
        }

        let edges = &obs.borrow().branch_targets;
        assert!(edges.contains(&(0x1004, 0x2000)), "observed the branch edge: {edges:?}");
        // De-duped: the re-taken branch edge appears once.
        assert_eq!(edges.iter().filter(|&&e| e == (0x1004, 0x2000)).count(), 1);

        // Seeds into ObservedFacts as Observed-stamped branch targets.
        let mut facts = ObservedFacts::new();
        obs.borrow().seed_branch_targets(&mut facts);
        assert!(
            facts.branch_targets.iter().any(|b| b.site_vip == 0x1004 && b.target_vip == 0x2000),
            "seeded branch target recorded",
        );
    }

    const BASE: u64 = 0x1_4000_0000;

    fn inst(id: u32, vip_reg: Register, region: Option<(u64, u64)>) -> VmInstance {
        VmInstance { id, vip: vip_reg.into(), bytecode_region: region }
    }

    fn spec(dispatcher_rva: u64, instance: VmInstance) -> VmInstanceSpec {
        VmInstanceSpec { dispatcher_rva, at: DispatchAt::Arrival, instance }
    }

    /// Enter the dispatcher at `d` with `vip` in `reg`, then jmp out to `handler`.
    fn step(h: &mut VmObserveHook, c: &mut Cpu, m: &mut Memory, d: u64, reg: Register, vip: u64, handler: u64) {
        c.set_reg(reg, vip);
        h.on_flow(&FlowEvent { kind: FlowKind::Jmp, from: 0, to: d, next: 0 }, c, m);
        h.on_flow(&FlowEvent { kind: FlowKind::Jmp, from: d, to: handler, next: 0 }, c, m);
    }

    fn edge(instance: u32, from_vip: u64, to_vip: u64, handler_va: u64, n: u64) -> ObservedVmEdge {
        ObservedVmEdge { instance, from_vip, to_vip, handler_va, dispatch_count: n }
    }

    fn two_dispatcher_specs() -> Vec<VmInstanceSpec> {
        vec![
            spec(0x5000, inst(1, Register::RSI, None)),
            spec(0x6000, inst(2, Register::RDI, None)),
        ]
    }

    fn run_two(base: u64) -> Vec<ObservedVmEdge> {
        let mut h = VmObserveHook::with_instances(base, two_dispatcher_specs(), None);
        let obs = h.observations();
        let (mut c, mut m) = (Cpu::new(0), Memory::new());
        let (d1, d2) = (base + 0x5000, base + 0x6000);
        let (ha, hb) = (base + 0x9000, base + 0x9100);
        // Interleaved: instance 1 on RSI VIPs, instance 2 on RDI VIPs.
        step(&mut h, &mut c, &mut m, d1, Register::RSI, 0x1000, ha);
        step(&mut h, &mut c, &mut m, d2, Register::RDI, 0x8000, hb);
        step(&mut h, &mut c, &mut m, d1, Register::RSI, 0x1004, ha);
        step(&mut h, &mut c, &mut m, d2, Register::RDI, 0x8010, hb);
        step(&mut h, &mut c, &mut m, d1, Register::RSI, 0x2000, ha);
        let o = obs.borrow();
        assert_eq!(o.unattributed, 0);
        o.edges.clone()
    }

    #[test]
    fn multi_instance_edges_land_on_the_right_instance() {
        let edges = run_two(BASE);
        let (ha, hb) = (BASE + 0x9000, BASE + 0x9100);
        assert_eq!(
            edges,
            vec![
                edge(1, 0x1000, 0x1004, ha, 1),
                edge(2, 0x8000, 0x8010, hb, 1),
                edge(1, 0x1004, 0x2000, ha, 1),
            ]
        );
    }

    #[test]
    fn dispatchers_resolve_through_the_load_base() {
        // Same RVA specs on a different base: same edges, base-relative handlers.
        let other = 0x7ff6_0000_0000;
        let edges = run_two(other);
        assert_eq!(edges.len(), 3);
        assert_eq!(edges[0], edge(1, 0x1000, 0x1004, other + 0x9000, 1));
        // A dispatcher at the OLD base's address is not watched on the new one.
        let mut h = VmObserveHook::with_instances(other, two_dispatcher_specs(), None);
        let (mut c, mut m) = (Cpu::new(0), Memory::new());
        step(&mut h, &mut c, &mut m, BASE + 0x5000, Register::RSI, 0x1, 0x2);
        step(&mut h, &mut c, &mut m, BASE + 0x5000, Register::RSI, 0x3, 0x2);
        assert!(h.observations().borrow().edges.is_empty());
    }

    #[test]
    fn shared_dispatch_slice_is_resolved_by_live_vip_region() {
        let a = (0x1000, 0x2000);
        let b = (0x3000, 0x4000);
        let specs = [
            spec(0x5000, inst(1, Register::RSI, Some(a))),
            spec(0x5000, inst(2, Register::RSI, Some(b))),
        ];
        let mut h = VmObserveHook::with_instances(BASE, specs, None);
        let obs = h.observations();
        let (mut c, mut m) = (Cpu::new(0), Memory::new());
        let d = BASE + 0x5000;
        let hd = BASE + 0x9000;
        // Instance 1's VIPs are VAs inside A; instance 2's inside B.
        for vip in [BASE + 0x1000, BASE + 0x3000, BASE + 0x1008, BASE + 0x3010] {
            step(&mut h, &mut c, &mut m, d, Register::RSI, vip, hd);
        }
        // A VIP in neither region is never attributed.
        step(&mut h, &mut c, &mut m, d, Register::RSI, BASE + 0x7000, hd);
        let o = obs.borrow();
        assert_eq!(o.unattributed, 1);
        assert_eq!(
            o.edges,
            vec![
                edge(1, BASE + 0x1000, BASE + 0x1008, hd, 1),
                edge(2, BASE + 0x3000, BASE + 0x3010, hd, 1),
            ]
        );
        assert_eq!(o.instance_edges(2).count(), 1);
    }

    #[test]
    fn edge_carries_handler_and_dispatch_count() {
        let mut h = VmObserveHook::with_instances(
            BASE,
            [spec(0x5000, inst(7, Register::RSI, None))],
            None,
        );
        let obs = h.observations();
        let (mut c, mut m) = (Cpu::new(0), Memory::new());
        let d = BASE + 0x5000;
        // Same VIP pair seen via handler A0 repeatedly and once via B0.
        for (vip, hv) in [(0x10, 0xA0), (0x20, 0xA0), (0x10, 0xB0), (0x20, 0xA0), (0x10, 0xA0)] {
            step(&mut h, &mut c, &mut m, d, Register::RSI, vip, hv);
        }
        let o = obs.borrow();
        assert_eq!(
            o.edges,
            vec![
                edge(7, 0x10, 0x20, 0xA0, 1),
                edge(7, 0x20, 0x10, 0xA0, 2),
                edge(7, 0x10, 0x20, 0xB0, 1),
            ]
        );
        assert_eq!(o.handlers, vec![0xA0, 0xA0, 0xB0, 0xA0, 0xA0]);

        // Repeating a full cycle bumps the count instead of adding a row.
        drop(o);
        for (vip, hv) in [(0x20, 0xA0), (0x10, 0xA0)] {
            step(&mut h, &mut c, &mut m, d, Register::RSI, vip, hv);
        }
        let o = obs.borrow();
        assert_eq!(o.edges.len(), 3);
        assert_eq!(o.edges[0], edge(7, 0x10, 0x20, 0xA0, 2));
        assert_eq!(o.edges[1], edge(7, 0x20, 0x10, 0xA0, 3));
    }

    fn delivered(i: &Interest, to: u64) -> bool {
        i.flow_all || i.flow.iter().any(|&(lo, hi)| (lo..hi).contains(&to))
    }

    #[test]
    fn interest_skips_non_dispatcher_flows() {
        let loop_head = BASE + 0x100;
        let mut h = VmObserveHook::with_instances(BASE, two_dispatcher_specs(), Some(loop_head));
        let i = h.interest();
        assert!(!i.flow_all && !i.read_all && !i.write_all);
        for to in [BASE + 0x5000, BASE + 0x6000, loop_head] {
            assert!(delivered(&i, to), "{to:#x} must be delivered");
        }
        for to in [BASE + 0x4fff, BASE + 0x6001, BASE + 0x9000, 0x1234, BASE + 0x60_0000] {
            assert!(!delivered(&i, to), "{to:#x} must be skipped");
        }

        // Entering a dispatcher widens interest to catch the handler jmp-out,
        // which then narrows again.
        let (mut c, mut m) = (Cpu::new(0), Memory::new());
        c.set_reg(Register::RSI, 0x1000);
        h.on_flow(&FlowEvent { kind: FlowKind::Jmp, from: 0, to: BASE + 0x5000, next: 0 }, &mut c, &mut m);
        assert!(h.interest().flow_all);
        h.on_flow(&FlowEvent { kind: FlowKind::Jmp, from: 0, to: BASE + 0x9000, next: 0 }, &mut c, &mut m);
        assert!(!h.interest().flow_all);
        assert_eq!(h.observations().borrow().handlers, vec![BASE + 0x9000]);
    }

    /// Arrive at a threaded `push reg; ret` dispatcher, then `ret` to a handler.
    fn ret_step(h: &mut VmObserveHook, c: &mut Cpu, m: &mut Memory, d: u64, handler: u64) {
        h.on_flow(&FlowEvent { kind: FlowKind::Jmp, from: 0, to: d, next: 0 }, c, m);
        h.on_flow(&FlowEvent { kind: FlowKind::Ret, from: d + 1, to: handler, next: 0 }, c, m);
    }

    #[test]
    fn one_instance_spanning_several_dispatch_stubs_walks_as_one() {
        // Threaded VMs replicate the dispatch stub: the instance id, not the
        // stub, owns the walk, so edges chain across stubs.
        let specs = [0x5000, 0x6000, 0x7000].map(|d| spec(d, inst(9, Register::RSI, None)));
        let mut h = VmObserveHook::with_instances(BASE, specs, None);
        let obs = h.observations();
        let (mut c, mut m) = (Cpu::new(0), Memory::new());
        let hd = BASE + 0x9000;
        for (d, vip) in [(0x5000, 0x100), (0x6000, 0x104), (0x7000, 0x108), (0x5000, 0x10c)] {
            step(&mut h, &mut c, &mut m, BASE + d, Register::RSI, vip, hd);
        }
        let o = obs.borrow();
        let chain: Vec<_> = o.edges.iter().map(|e| (e.from_vip, e.to_vip)).collect();
        assert_eq!(chain, vec![(0x100, 0x104), (0x104, 0x108), (0x108, 0x10c)]);
    }

    #[test]
    fn memory_resident_vip_is_read_at_each_dispatch() {
        // VIP spilled to a stack slot [rsp+0x20], threaded `push rbp; ret` dispatch.
        const STACK: u64 = 0x20_0000;
        let vip = VipLoc::Mem { base: Register::RSP, disp: 0x20 };
        let mut h = VmObserveHook::with_instances(
            BASE,
            [spec(0x5000, VmInstance { id: 3, vip, bytecode_region: None })],
            None,
        );
        let obs = h.observations();
        let (mut c, mut m) = (Cpu::new(0), Memory::new());
        m.map(STACK, 0x1000, crate::memory::Perm::RW).unwrap();
        c.set_reg(Register::RSP, STACK);
        let (d, hd) = (BASE + 0x5000, BASE + 0x9000);
        // The slot walks down the bytecode; a register-only observer sees nothing.
        for v in [0x4000_u64, 0x3ffc, 0x3ff4, 0x3000] {
            m.write_uint(STACK + 0x20, 8, v).unwrap();
            ret_step(&mut h, &mut c, &mut m, d, hd);
        }
        let o = obs.borrow();
        assert_eq!(o.unattributed, 0);
        assert_eq!(o.handlers, vec![hd; 4], "ret-out recorded as the handler");
        assert_eq!(
            o.edges,
            vec![
                edge(3, 0x4000, 0x3ffc, hd, 1),
                edge(3, 0x3ffc, 0x3ff4, hd, 1),
                edge(3, 0x3ff4, 0x3000, hd, 1),
            ]
        );
    }

    #[test]
    fn unreadable_memory_vip_is_unattributed_not_a_junk_edge() {
        let vip = VipLoc::Mem { base: Register::RSP, disp: 0 };
        let mut h = VmObserveHook::with_instances(
            BASE,
            [spec(0x5000, VmInstance { id: 1, vip, bytecode_region: None })],
            None,
        );
        let (mut c, mut m) = (Cpu::new(0), Memory::new());
        c.set_reg(Register::RSP, 0xdead_0000);
        ret_step(&mut h, &mut c, &mut m, BASE + 0x5000, BASE + 0x9000);
        let obs = h.observations();
        let o = obs.borrow();
        assert_eq!((o.unattributed, o.edges.len()), (1, 0));
    }

    #[test]
    fn many_dispatchers_stay_covered_by_two_spans() {
        let specs: Vec<_> = (0..20)
            .map(|k| {
                let rva = if k < 10 { 0x1000 + k * 0x40 } else { 0x40_0000 + k * 0x40 };
                spec(rva, inst(k as u32, Register::RSI, None))
            })
            .collect();
        let rvas: Vec<u64> = specs.iter().map(|s| s.dispatcher_rva).collect();
        let h = VmObserveHook::with_instances(BASE, specs, None);
        let i = h.interest();
        assert!(rvas.iter().all(|r| delivered(&i, BASE + r)));
        // The two clusters are not merged into one giant span.
        assert!(!delivered(&i, BASE + 0x20_0000));
        assert!(!delivered(&i, BASE + 0x500));
    }

    #[test]
    fn no_dispatcher_means_no_interest() {
        let h = VmObserveHook::new(None, None);
        let i = h.interest();
        assert!(!i.flow_all && !delivered(&i, 0) && !delivered(&i, DISPATCHER));
    }
}
