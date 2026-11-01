//! Layer 5b: structural discovery of a VM's dispatch sites and VIP location.
//!
//! A threaded `VMProtect` handler ends by computing `next = base + decrypt(off)`
//! into a handler-pointer register and transferring there with `jmp reg` or
//! `push reg; ret`, advancing the VIP by a few bytes per handler. Nothing here
//! is hand-picked: a [`SurveyHook`] watches every such indirect transfer during
//! a contained run, groups the sites by handler-pointer register, and picks the
//! VIP as the register or stack slot that holds an in-image address stepping
//! by small, one-directional deltas between consecutive dispatches. A VM may
//! switch role registers between profiles, so the pick is made per window of
//! consecutive dispatches and each stub votes for the location of the windows
//! it appears in. The result becomes [`VmInstanceSpec`]s for
//! [`crate::VmObserveHook`].
//!
//! Pure observation: callbacks only read `cpu`/`mem`; memory is bounded by
//! `MAX_SITES` (fail-closed `Stop`) and `SAMPLE_CAP`.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use iced_x86::Register;

use crate::cpu::Cpu;
use crate::emu::{FlowEvent, FlowKind, Hook, HookAction, Interest};
use crate::memory::Memory;
use crate::vm_observe::{DispatchAt, VipLoc, VmInstance, VmInstanceSpec};

const GPRS: [Register; 16] = [
    Register::RAX,
    Register::RCX,
    Register::RDX,
    Register::RBX,
    Register::RSP,
    Register::RBP,
    Register::RSI,
    Register::RDI,
    Register::R8,
    Register::R9,
    Register::R10,
    Register::R11,
    Register::R12,
    Register::R13,
    Register::R14,
    Register::R15,
];
const RSP: usize = 4;
/// Stack-relative qword slots tried as a spilled VIP: `[rsp, rsp + 8*SLOTS)`.
const SLOTS: usize = 32;
const SAMPLE_CAP: usize = 32_768;
/// Consecutive dispatches scored together; sampling is dense for the first
/// `DENSE` events of a group, then one window every `BURST_EVERY` events.
const WINDOW: u64 = 256;
const DENSE: u64 = 4_096;
const BURST_EVERY: u64 = 4_096;
const MAX_SITES: usize = 1 << 16;
/// Dispatch events a register group needs to count as a VM, and samples to score.
const MIN_EVENTS: u64 = 1_000;
const MIN_SAMPLES: usize = 64;
/// A VIP step between consecutive dispatches is at most this many bytes.
const MAX_STEP: u64 = 0x100;
const MAX_ENTRIES: usize = 4_096;

#[derive(Clone, Copy)]
struct Sample {
    regs: [u64; 16],
    slots: [u64; SLOTS],
    site: u64,
    win: u64,
}

#[derive(Clone, Copy)]
struct Site {
    reg: u8,
    push_ret: bool,
    count: u64,
}

/// One dispatching transfer: the `jmp reg` or the `ret` of a `push reg; ret`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DispatchSite {
    pub va: u64,
    pub hits: u64,
    pub push_ret: bool,
}

/// Raw survey tallies; read through [`Survey::discover`].
pub struct Survey {
    sites: HashMap<u64, Site>,
    totals: [u64; 16],
    samples: Vec<Vec<Sample>>,
    overflow: bool,
}

/// One VM found behind a handler-pointer register.
#[derive(Clone, Debug, PartialEq)]
pub struct DiscoveredVm {
    /// Register the threaded dispatch transfers through.
    pub dispatch_reg: Register,
    /// Dispatch events seen for this register.
    pub events: u64,
    /// The dispatching transfers of this VM, hottest first.
    pub entries: Vec<DispatchSite>,
    /// Where the VIP lives, if a candidate scored.
    pub vip: Option<VipChoice>,
}

/// The chosen VIP location and how well it scored.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VipChoice {
    pub loc: VipLoc,
    /// Fraction of consecutive dispatch pairs where it stepped by a small delta.
    pub step_fraction: f64,
}

impl DiscoveredVm {
    /// Observer specs (instance `id`, dispatchers as RVAs off `base`) whose VIP
    /// must fall in `region` (RVA span; a junk read is then unattributed, not an
    /// edge); empty when no VIP location was found.
    #[must_use]
    pub fn specs(&self, base: u64, id: u32, region: (u64, u64)) -> Vec<VmInstanceSpec> {
        let Some(v) = self.vip else { return Vec::new() };
        self.entries
            .iter()
            .map(|d| VmInstanceSpec {
                dispatcher_rva: d.va.wrapping_sub(base),
                at: DispatchAt::Transfer { rsp_bias: if d.push_ret { 8 } else { 0 } },
                instance: VmInstance { id, vip: v.loc, bytecode_region: Some(region) },
            })
            .collect()
    }
}

impl Survey {
    fn new() -> Self {
        Self {
            sites: HashMap::new(),
            totals: [0; 16],
            samples: vec![Vec::new(); 16],
            overflow: false,
        }
    }

    /// `true` if the site table hit its cap and the run was stopped.
    #[must_use]
    pub fn overflowed(&self) -> bool {
        self.overflow
    }

    /// The VMs found, hottest first: one per (handler-pointer register, VIP
    /// location). `image` is the half-open VA range of the loaded image (the
    /// VIP must point into it). A register group with no scoring window is
    /// reported once with `vip: None`.
    #[must_use]
    pub fn discover(&self, image: (u64, u64)) -> Vec<DiscoveredVm> {
        self.discover_min(image, MIN_EVENTS)
    }

    /// [`Survey::discover`] with a caller-chosen dispatch-event floor (a short run of a small
    /// function has fewer than the default 1000 dispatches).
    #[must_use]
    pub fn discover_min(&self, image: (u64, u64), min_events: u64) -> Vec<DiscoveredVm> {
        let mut out = Vec::new();
        for r in 0..16 {
            let events = self.totals[r];
            if events < min_events {
                continue;
            }
            let mut sites: Vec<DispatchSite> = self
                .sites
                .iter()
                .filter(|(_, s)| usize::from(s.reg) == r)
                .map(|(&va, s)| DispatchSite { va, hits: s.count, push_ret: s.push_ret })
                .collect();
            sites.sort_by_key(|d| (std::cmp::Reverse(d.hits), d.va));
            sites.truncate(MAX_ENTRIES);

            // Score each window; tally per-location and per-site votes.
            let mut locs: Vec<(VipLoc, usize, f64)> = Vec::new();
            let mut site_votes: HashMap<u64, Vec<(VipLoc, usize)>> = HashMap::new();
            for w in self.samples[r].chunk_by(|a, b| a.win == b.win) {
                let Some(c) = pick_vip(w, r, image) else { continue };
                match locs.iter_mut().find(|l| l.0 == c.loc) {
                    Some(l) => {
                        l.1 += 1;
                        l.2 += c.step_fraction;
                    }
                    None => locs.push((c.loc, 1, c.step_fraction)),
                }
                for site in w.iter().map(|x| x.site).collect::<std::collections::BTreeSet<_>>() {
                    let v = site_votes.entry(site).or_default();
                    match v.iter_mut().find(|e| e.0 == c.loc) {
                        Some(e) => e.1 += 1,
                        None => v.push((c.loc, 1)),
                    }
                }
            }
            let Some(majority) = locs.iter().max_by_key(|l| l.1).map(|l| l.0) else {
                out.push(DiscoveredVm { dispatch_reg: GPRS[r], events, entries: sites, vip: None });
                continue;
            };
            // A stub never seen in a scoring window joins the majority profile.
            for &(loc, n, frac) in &locs {
                let entries: Vec<DispatchSite> = sites
                    .iter()
                    .filter(|d| {
                        let home = site_votes
                            .get(&d.va)
                            .and_then(|v| v.iter().max_by_key(|e| e.1))
                            .map_or(majority, |e| e.0);
                        home == loc
                    })
                    .copied()
                    .collect();
                if !entries.is_empty() {
                    out.push(DiscoveredVm {
                        dispatch_reg: GPRS[r],
                        events,
                        entries,
                        vip: Some(VipChoice { loc, step_fraction: frac / n as f64 }),
                    });
                }
            }
        }
        out.sort_by_key(|d| std::cmp::Reverse(d.events));
        out
    }
}

/// Score every register and stack slot as the VIP over consecutive dispatch
/// samples: in-image, stepping by small deltas, mostly one direction.
fn pick_vip(samples: &[Sample], dispatch_reg: usize, image: (u64, u64)) -> Option<VipChoice> {
    if samples.len() < MIN_SAMPLES {
        return None;
    }
    let in_image = |v: u64| (image.0..image.1).contains(&v);
    let cands = (0..16)
        .filter(|&r| r != RSP && r != dispatch_reg)
        .map(|r| (VipLoc::Reg(GPRS[r]), r))
        .chain((0..SLOTS).map(|i| (VipLoc::Mem { base: Register::RSP, disp: 8 * i as i64 }, 16 + i)));
    let mut best: Option<VipChoice> = None;
    for (loc, k) in cands {
        let at = |s: &Sample| if k < 16 { s.regs[k] } else { s.slots[k - 16] };
        if samples.iter().filter(|s| in_image(at(s))).count() * 100 < samples.len() * 95 {
            continue;
        }
        let (mut stepped, mut down, mut up) = (0usize, 0usize, 0usize);
        for w in samples.windows(2) {
            let (a, b) = (at(&w[0]), at(&w[1]));
            let d = b.wrapping_sub(a);
            if d != 0 && (d <= MAX_STEP || d.wrapping_neg() <= MAX_STEP) {
                stepped += 1;
                if d <= MAX_STEP { up += 1 } else { down += 1 }
            }
        }
        let frac = stepped as f64 / (samples.len() - 1) as f64;
        if frac >= 0.7 && up.max(down) * 10 >= stepped * 8 && best.is_none_or(|b| frac > b.step_fraction) {
            best = Some(VipChoice { loc, step_fraction: frac });
        }
    }
    best
}

/// Classify the indirect transfer at `from`: `(handler-pointer reg, is the
/// `ret` of a `push reg; ret`)` for `jmp reg` (Jmp) or `push reg; ret` (Ret).
/// Anything else is not a threaded dispatch.
fn classify(kind: FlowKind, from: u64, mem: &Memory) -> Option<(usize, bool)> {
    match kind {
        FlowKind::Jmp => {
            let mut b = [0u8; 3];
            mem.read(from, &mut b).ok()?;
            let (rex, m) = if b[0] == 0x41 { (8, &b[1..]) } else { (0, &b[..2]) };
            (m[0] == 0xFF && m[1] & 0xF8 == 0xE0).then(|| (rex + usize::from(m[1] & 7), false))
        }
        FlowKind::Ret => {
            let mut b = [0u8; 2];
            mem.read(from.checked_sub(2)?, &mut b).ok()?;
            if !(0x50..=0x57).contains(&b[1]) {
                return None;
            }
            let rex = if b[0] == 0x41 { 8 } else { 0 };
            Some((rex + usize::from(b[1] & 7), true))
        }
        _ => None,
    }
}

/// Watches every indirect transfer of the `jmp reg` / `push reg; ret` shape,
/// keyed by the transfer instruction itself (not the stub entry).
pub struct SurveyHook {
    s: Rc<RefCell<Survey>>,
}

impl Default for SurveyHook {
    fn default() -> Self {
        Self::new()
    }
}

impl SurveyHook {
    #[must_use]
    pub fn new() -> Self {
        Self { s: Rc::new(RefCell::new(Survey::new())) }
    }

    /// Shared handle to the tallies; clone before installing the hook.
    #[must_use]
    pub fn survey(&self) -> Rc<RefCell<Survey>> {
        Rc::clone(&self.s)
    }
}

impl Hook for SurveyHook {
    fn interest(&self) -> Interest {
        Interest { flow_all: true, ..Interest::NONE }
    }

    fn on_flow(&mut self, ev: &FlowEvent, cpu: &mut Cpu, mem: &mut Memory) -> HookAction {
        let Some((r, push_ret)) = classify(ev.kind, ev.from, mem) else {
            return HookAction::Continue;
        };
        let s = &mut *self.s.borrow_mut();
        let entry = ev.from;
        if s.sites.len() >= MAX_SITES && !s.sites.contains_key(&entry) {
            s.overflow = true;
            return HookAction::Stop;
        }
        s.sites.entry(entry).or_insert(Site { reg: r as u8, push_ret, count: 0 }).count += 1;
        let n = s.totals[r];
        s.totals[r] += 1;
        let take = n < DENSE || n % BURST_EVERY < WINDOW;
        if take && s.samples[r].len() < SAMPLE_CAP {
            // At the `ret` the stub's push has run: restore the pre-push rsp.
            let sp = cpu.regs[RSP].wrapping_add(if push_ret { 8 } else { 0 });
            let mut regs = cpu.regs;
            regs[RSP] = sp;
            let mut slots = [0u64; SLOTS];
            for (i, v) in slots.iter_mut().enumerate() {
                *v = mem.read_uint(sp.wrapping_add(8 * i as u64), 8).unwrap_or(0);
            }
            s.samples[r].push(Sample { regs, slots, site: entry, win: n / WINDOW });
        }
        HookAction::Continue
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::Perm;

    const CODE: u64 = 0x1000;
    const STACK: u64 = 0x20_0000;
    const IMG: (u64, u64) = (0x1_4000_0000, 0x1_5000_0000);

    /// Code page: `jmp rbp` at +0x100, `push rbp; ret` at +0x200/+0x201, a
    /// plain `ret` at +0x300 and a direct `jmp` at +0x400.
    fn memory() -> Memory {
        let mut m = Memory::new();
        m.map(CODE, 0x1000, Perm::RX).unwrap();
        m.map(STACK, 0x1000, Perm::RW).unwrap();
        m.poke(CODE + 0x100, &[0xFF, 0xE5]).unwrap();
        m.poke(CODE + 0x200, &[0x55, 0xC3]).unwrap();
        m.poke(CODE + 0x2FF, &[0x90, 0xC3]).unwrap();
        m.poke(CODE + 0x400, &[0xE9, 0, 0, 0, 0]).unwrap();
        m
    }

    fn flow(kind: FlowKind, from: u64) -> FlowEvent {
        FlowEvent { kind, from, to: 0, next: 0 }
    }

    /// Drive `n` threaded dispatches alternating over the two stub shapes.
    /// `vip(i)` is the VIP at dispatch `i`; `place` puts it in cpu/memory.
    fn drive(n: u64, place: impl Fn(&mut Cpu, &mut Memory, u64)) -> Survey {
        let mut hook = SurveyHook::new();
        let s = hook.survey();
        let mut m = memory();
        let mut c = Cpu::new(0);
        c.set_reg(Register::RSP, STACK + 0x800);
        c.set_reg(Register::R8, STACK + 0x400);
        for i in 0..n {
            let vip = IMG.0 + 0x9000 - 4 * i;
            place(&mut c, &mut m, vip);
            c.set_reg(Register::RBP, IMG.0 + 0x100 * (i * 7919 % 4096)); // handler ptr: jumps
            c.set_reg(Register::R13, i.wrapping_mul(0x9E37_79B9_7F4A_7C15)); // rolling key
            // VM stack pointer: pushes/pops, no net direction.
            c.set_reg(Register::R8, STACK + 0x400 + 8 * (i % 3));
            if i % 2 == 0 {
                hook.on_flow(&flow(FlowKind::Jmp, CODE + 0x100), &mut c, &mut m);
            } else {
                // The stub's push has run: rsp is one slot lower.
                let sp = c.reg(Register::RSP).unwrap();
                c.set_reg(Register::RSP, sp - 8);
                hook.on_flow(&flow(FlowKind::Ret, CODE + 0x201), &mut c, &mut m);
                c.set_reg(Register::RSP, sp);
            }
            // Noise the classifier must ignore.
            hook.on_flow(&flow(FlowKind::Ret, CODE + 0x300), &mut c, &mut m);
            hook.on_flow(&flow(FlowKind::Jmp, CODE + 0x400), &mut c, &mut m);
        }
        drop(hook);
        Rc::try_unwrap(s).ok().unwrap().into_inner()
    }

    #[test]
    fn discovers_dispatch_stubs_and_register_vip() {
        let s = drive(2_000, |c, _, vip| {
            c.set_reg(Register::R11, vip);
        });
        let vms = s.discover(IMG);
        assert_eq!(vms.len(), 1, "noise sites form no VM: {vms:?}");
        let vm = &vms[0];
        assert_eq!(vm.dispatch_reg, Register::RBP);
        assert_eq!(vm.events, 2_000);
        let mut entries: Vec<u64> = vm.entries.iter().map(|e| e.va).collect();
        entries.sort_unstable();
        assert_eq!(entries, vec![CODE + 0x100, CODE + 0x201], "the jmp, and the ret of push;ret");
        assert!(vm.entries.iter().all(|e| e.push_ret == (e.va == CODE + 0x201)));
        let vip = vm.vip.expect("vip found");
        assert_eq!(vip.loc, VipLoc::Reg(Register::R11));
        assert!(vip.step_fraction > 0.99);
        // Specs are RVAs off the base and carry the discovered VIP location.
        let specs = vm.specs(0x1000, 5, (0, 0x1_0000_0000));
        assert_eq!(specs.len(), 2);
        assert!(specs.iter().all(|s| s.instance.id == 5 && s.instance.vip == vip.loc));
        assert!(specs.iter().any(|s| s.dispatcher_rva == 0x100 && s.at == DispatchAt::Transfer { rsp_bias: 0 }));
        assert!(specs.iter().any(|s| s.dispatcher_rva == 0x201 && s.at == DispatchAt::Transfer { rsp_bias: 8 }));
    }

    #[test]
    fn discovers_a_memory_resident_vip() {
        // The VIP lives only in a stack slot [rsp+0x28]; every register is junk.
        let s = drive(2_000, |c, m, vip| {
            let sp = c.reg(Register::RSP).unwrap();
            m.write_uint(sp + 0x28, 8, vip).unwrap();
        });
        let vm = &s.discover(IMG)[0];
        assert_eq!(vm.vip.map(|v| v.loc), Some(VipLoc::Mem { base: Register::RSP, disp: 0x28 }));
    }

    #[test]
    fn no_vip_when_nothing_steps_through_the_image() {
        let s = drive(2_000, |c, _, _| {
            c.set_reg(Register::R11, 0x7000_0000); // outside the image, constant
        });
        let vm = &s.discover(IMG)[0];
        assert_eq!(vm.vip, None);
        assert!(vm.specs(0, 0, (0, 1)).is_empty(), "no VIP, no observer spec");
    }

    #[test]
    fn profile_switch_splits_one_register_group_by_stub() {
        // One handler-pointer register (rbp), two stub copies: the VIP is r11
        // behind the first and r12 behind the second.
        let mut hook = SurveyHook::new();
        let s = hook.survey();
        let mut m = memory();
        m.poke(CODE + 0x500, &[0xFF, 0xE5]).unwrap();
        let mut c = Cpu::new(0);
        c.set_reg(Register::RSP, STACK + 0x800);
        for i in 0..4_000u64 {
            let (stub, vip_reg) = if i < 2_000 { (0x100, Register::R11) } else { (0x500, Register::R12) };
            c.set_reg(vip_reg, IMG.0 + 0x9000 - 4 * i);
            c.set_reg(Register::RBP, IMG.0 + 0x100 * (i * 7919 % 4096));
            hook.on_flow(&flow(FlowKind::Jmp, CODE + stub), &mut c, &mut m);
        }
        drop(hook);
        let vms = s.borrow().discover(IMG);
        assert_eq!(vms.len(), 2, "{vms:?}");
        let loc_of = |stub: u64| {
            let vm = vms.iter().find(|v| v.entries.iter().any(|e| e.va == CODE + stub));
            vm.and_then(|v| v.vip).map(|v| v.loc)
        };
        assert_eq!(loc_of(0x100), Some(VipLoc::Reg(Register::R11)));
        assert_eq!(loc_of(0x500), Some(VipLoc::Reg(Register::R12)));
    }

    #[test]
    fn rare_indirect_jumps_are_not_a_vm() {
        let s = drive(100, |c, _, vip| {
            c.set_reg(Register::R11, vip);
        });
        assert!(s.discover(IMG).is_empty(), "below the event floor");
    }
}
