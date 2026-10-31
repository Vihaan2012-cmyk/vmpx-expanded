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
use std::collections::{BTreeMap, HashMap};
use std::rc::Rc;

use iced_x86::Register;

use crate::cpu::Cpu;
use crate::emu::{FlowEvent, FlowKind, Hook, HookAction, Interest};
use crate::memory::Memory;
use crate::observe::ObservedFacts;

/// One interpreter instance behind a dispatcher.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VmInstance {
    pub id: u32,
    /// Register holding this instance's VIP at dispatcher entry.
    pub vip_reg: Register,
    /// Proven bytecode region as a half-open RVA span. `None` = unconstrained,
    /// accepted only when the instance is alone on its dispatch slice.
    pub bytecode_region: Option<(u64, u64)>,
}

/// A statically discovered dispatcher (by RVA) and the instance it serves.
/// Several specs may share one `dispatcher_rva` (shared native dispatch slice).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VmInstanceSpec {
    pub dispatcher_rva: u64,
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
}

impl VmObservations {
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

/// An instance with its region resolved to absolute VAs, plus run state.
struct Live {
    id: u32,
    vip_reg: Register,
    region: Option<(u64, u64)>,
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
    /// Dispatcher VA -> candidate indices into `live`.
    dispatchers: BTreeMap<u64, Vec<usize>>,
    live: Vec<Live>,
    /// Instance entered at the last dispatcher arrival, awaiting its jmp-out.
    armed: Option<usize>,
    /// `(instance, from, to, handler)` -> index into `obs.edges`.
    edge_ix: HashMap<(u32, u64, u64, u64), usize>,
    /// Flow spans covering every dispatcher and the loop head.
    spans: [(u64, u64); 2],
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
            instance: VmInstance { id: 0, vip_reg: Register::RSI, bytecode_region: None },
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
        let mut live = Vec::new();
        for s in specs {
            dispatchers.entry(base.wrapping_add(s.dispatcher_rva)).or_default().push(live.len());
            live.push(Live {
                id: s.instance.id,
                vip_reg: s.instance.vip_reg,
                region: s
                    .instance
                    .bytecode_region
                    .map(|(lo, hi)| (base.wrapping_add(lo), base.wrapping_add(hi))),
                last_vip: None,
                last_handler: 0,
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
            live,
            armed: None,
            edge_ix: HashMap::new(),
            spans: cover(pts),
        }
    }

    /// Shared handle to the observations collected during the run.
    #[must_use]
    pub fn observations(&self) -> Rc<RefCell<VmObservations>> {
        Rc::clone(&self.obs)
    }

    /// Instance served by the dispatcher at `d` and its live VIP, chosen by
    /// matching the VIP against each candidate's bytecode region.
    fn resolve(&self, d: u64, cpu: &Cpu) -> Option<(usize, u64)> {
        let cands = self.dispatchers.get(&d)?;
        cands.iter().find_map(|&i| {
            let l = &self.live[i];
            let vip = cpu.reg(l.vip_reg).unwrap_or_default();
            let ok = match l.region {
                Some((lo, hi)) => (lo..hi).contains(&vip),
                None => cands.len() == 1,
            };
            ok.then_some((i, vip))
        })
    }
}

impl Hook for VmObserveHook {
    fn interest(&self) -> Interest {
        // Once inside a dispatcher the next jmp (target unknown) is the handler.
        if self.armed.is_some() {
            return Interest { flow_all: true, ..Interest::NONE };
        }
        Interest { flow: self.spans, ..Interest::NONE }
    }

    fn on_flow(&mut self, ev: &FlowEvent, cpu: &mut Cpu, _mem: &mut Memory) -> HookAction {
        let mut o = self.obs.borrow_mut();

        // Concolic depth: every arrival at the loop head is a back-edge visit.
        if o.loop_head == Some(ev.to) {
            o.back_edges += 1;
        }

        // Handler trace: record the dispatcher's first jmp-out as a handler
        // entry, then re-arm when control returns to a dispatcher.
        if let (Some(i), FlowKind::Jmp) = (self.armed, ev.kind) {
            o.handlers.push(ev.to);
            self.live[i].last_handler = ev.to;
            self.armed = None;
        }
        if self.dispatchers.contains_key(&ev.to) {
            let Some((i, vip)) = self.resolve(ev.to, cpu) else {
                o.unattributed += 1;
                return HookAction::Continue;
            };
            self.armed = Some(i);

            // At dispatcher entry the VIP is the bytecode site about to be
            // decoded; the previous dispatch VIP -> this one is the edge.
            let l = &mut self.live[i];
            let (id, handler) = (l.id, l.last_handler);
            if let Some(site) = l.last_vip.replace(vip) {
                let o = &mut *o;
                if !o.branch_targets.contains(&(site, vip)) {
                    o.branch_targets.push((site, vip));
                }
                let edges = &mut o.edges;
                let ix = *self.edge_ix.entry((id, site, vip, handler)).or_insert_with(|| {
                    edges.push(ObservedVmEdge {
                        instance: id,
                        from_vip: site,
                        to_vip: vip,
                        handler_va: handler,
                        dispatch_count: 0,
                    });
                    edges.len() - 1
                });
                edges[ix].dispatch_count += 1;
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
        VmInstance { id, vip_reg, bytecode_region: region }
    }

    fn spec(dispatcher_rva: u64, instance: VmInstance) -> VmInstanceSpec {
        VmInstanceSpec { dispatcher_rva, instance }
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
