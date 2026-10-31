//! Layer 5: a [`Hook`] that observes a VM run — the sequence of handler entries
//! the dispatcher transfers to, and the concrete recursion depth of a
//! data-dependent loop/recursion under ONE concrete input.
//!
//! # Containment
//!
//! Pure observation: every callback only reads `cpu`/`mem` and appends to an
//! in-model record; it never redirects control, touches the host, or guesses.
//! The run is driven entirely by the software interpreter.
//!
//! # Soundness / provenance
//!
//! The handler trace and the recursion depth reflect ONE concrete execution
//! path, not a proof. [`VmObservations::seed_loop_depth`] stamps the depth into
//! [`ObservedFacts`] as an [`crate::Observed`] fact so the static side can SEED
//! its push-down-stack recursion lift (unbounded in pure symbolic analysis)
//! while keeping it on the quarantined tier — never sound output. The depth is
//! valid only for the concrete input that produced it.
//!
//! # Install
//!
//! ```ignore
//! let hook = VmObserveHook::new(Some(dispatcher_va), Some(loop_head_rva));
//! let obs = hook.observations();         // shared handle, clone before install
//! let mut emu = ...;
//! emu.hook = Some(Box::new(hook));
//! emu.run_until(sentinel, MAX);
//! let o = obs.borrow();
//! let handlers = &o.handlers;            // dispatched handler entries, in order
//! o.seed_loop_depth(&mut facts);         // observed depth -> ObservedFacts
//! ```

use std::cell::RefCell;
use std::rc::Rc;

use iced_x86::Register;

use crate::cpu::Cpu;
use crate::emu::{FlowEvent, FlowKind, Hook, HookAction};
use crate::memory::Memory;
use crate::observe::ObservedFacts;

/// Shared recording filled during a run. Clone the [`Rc`] before installing the
/// hook, then read it afterwards.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VmObservations {
    /// Dispatcher entry VA the run was watched against (the bootstrap target the
    /// static side discovers), if any.
    pub dispatcher: Option<u64>,
    /// Loop-head / recursion-entry RVA whose back-edge visits are counted.
    pub loop_head: Option<u64>,
    /// Handler entries the dispatcher transferred to, in dispatch order.
    pub handlers: Vec<u64>,
    /// Count of control-flow transfers that landed on `loop_head` — the
    /// observed recursion depth for this concrete input.
    pub back_edges: u64,
    /// Observed VM VIP edges `(site_vip, target_vip)`: at each dispatcher
    /// re-entry the VIP register (RSI, VMProtect 3.11 Compact) holds the
    /// bytecode VIP about to be decoded, so consecutive dispatch VIPs form the
    /// edge from one bytecode site to the next the VM reached. De-duped. The
    /// branch SITEs the static `cfg` stalls on are the subset it queries;
    /// straight-line edges are inert (the static side keys by site and
    /// independently re-validates every target as real in-image bytecode).
    pub branch_targets: Vec<(u64, u64)>,
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
}

/// Observes VM-dispatcher handler traces and concolic recursion depth.
///
/// Handler detection ("to/through the dispatcher"): a transfer whose target is
/// the dispatcher marks that the run is inside it; the next `jmp` out of the
/// dispatcher is its indirect dispatch, and its target is recorded as the
/// entered handler. This assumes the dispatcher reaches a handler via its first
/// `jmp` after entry — the `VMProtect` central-dispatch shape.
pub struct VmObserveHook {
    obs: Rc<RefCell<VmObservations>>,
    in_dispatcher: bool,
    /// VIP (RSI) seen at the previous dispatcher re-entry — the site whose
    /// bytecode led to the next dispatch VIP.
    last_vip: Option<u64>,
}

impl VmObserveHook {
    #[must_use]
    pub fn new(dispatcher: Option<u64>, loop_head: Option<u64>) -> Self {
        Self {
            obs: Rc::new(RefCell::new(VmObservations {
                dispatcher,
                loop_head,
                handlers: Vec::new(),
                back_edges: 0,
                branch_targets: Vec::new(),
            })),
            in_dispatcher: false,
            last_vip: None,
        }
    }

    /// Shared handle to the observations collected during the run.
    #[must_use]
    pub fn observations(&self) -> Rc<RefCell<VmObservations>> {
        Rc::clone(&self.obs)
    }
}

impl Hook for VmObserveHook {
    fn on_flow(&mut self, ev: &FlowEvent, cpu: &mut Cpu, _mem: &mut Memory) -> HookAction {
        let mut o = self.obs.borrow_mut();

        // Concolic depth: every arrival at the loop head is a back-edge visit.
        if o.loop_head == Some(ev.to) {
            o.back_edges += 1;
        }

        // Handler trace: record the dispatcher's first jmp-out as a handler
        // entry, then re-arm when control returns to the dispatcher.
        if let Some(d) = o.dispatcher {
            if self.in_dispatcher && matches!(ev.kind, FlowKind::Jmp) {
                o.handlers.push(ev.to);
                self.in_dispatcher = false;
            }
            if ev.to == d {
                self.in_dispatcher = true;

                // Branch targets: at dispatcher re-entry the VIP (RSI) is the
                // bytecode site about to be decoded. The edge from the previous
                // dispatch VIP to this one is the (site, target) the branch
                // bridge consumes. De-dupe to the observed VIP-edge set.
                let vip = cpu.reg(Register::RSI).unwrap_or_default();
                if let Some(site) = self.last_vip {
                    let edge = (site, vip);
                    if !o.branch_targets.contains(&edge) {
                        o.branch_targets.push(edge);
                    }
                }
                self.last_vip = Some(vip);
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
}
