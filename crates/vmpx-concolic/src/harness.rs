//! Contained-emulator harness: run a protected DLL's DllMain once, then call an
//! export with a chosen concrete input while observing VM dispatch.
//! Everything runs in the software interpreter; nothing executes natively.

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::rc::Rc;

use iced_x86::Register;
use vmpx_emulate::{
    Cpu, DispatchAt, Emulator, FlowEvent, FlowKind, Hook, HookAction, Interest, LoadOptions,
    MemEvent, Memory, Stop, SurveyHook, VipLoc, VmInstanceSpec, VmObservations, VmObserveHook,
    WinApiHook, load,
};
use vmpx_pe::PeImage;

const RAX: usize = 0;
const RSP: usize = 4;
const WIN64_ARGS: [usize; 4] = [1, 2, 8, 9];

/// Runs several hooks back to back (the emulator has a single hook slot).
pub struct Tee(pub Vec<Box<dyn Hook>>);

impl Hook for Tee {
    fn interest(&self) -> Interest {
        self.0.iter().fold(Interest::NONE, |a, h| a.union(h.interest()))
    }
    fn on_mem(&mut self, ev: &MemEvent, cpu: &mut Cpu, mem: &mut Memory) -> HookAction {
        let mut acc = HookAction::Continue;
        for h in &mut self.0 {
            match h.on_mem(ev, cpu, mem) {
                HookAction::Stop => return HookAction::Stop,
                HookAction::Redirect => acc = HookAction::Redirect,
                HookAction::Continue => {}
            }
        }
        acc
    }
    fn on_flow(&mut self, ev: &FlowEvent, cpu: &mut Cpu, mem: &mut Memory) -> HookAction {
        let mut acc = HookAction::Continue;
        for h in &mut self.0 {
            match h.on_flow(ev, cpu, mem) {
                HookAction::Stop => return HookAction::Stop,
                HookAction::Redirect => acc = HookAction::Redirect,
                HookAction::Continue => {}
            }
        }
        acc
    }
    fn on_syscall(&mut self, cpu: &mut Cpu, mem: &mut Memory) -> Option<HookAction> {
        self.0.iter_mut().find_map(|h| h.on_syscall(cpu, mem))
    }
}

/// Records the ORDERED sequence of VIPs at the VM's dispatching transfers
/// (`VmObserveHook` keeps only de-duplicated edges, which loses iteration order).
pub struct SeqHook {
    sites: HashMap<u64, u64>,
    vip: VipLoc,
    pub seq: Rc<RefCell<Vec<u64>>>,
    cap: usize,
}

impl SeqHook {
    #[must_use]
    pub fn new(base: u64, specs: &[VmInstanceSpec], cap: usize) -> Option<Self> {
        let first = specs.first()?;
        let mut sites = HashMap::new();
        for s in specs {
            let bias = match s.at {
                DispatchAt::Transfer { rsp_bias } => rsp_bias,
                DispatchAt::Arrival => 0,
            };
            sites.insert(base.wrapping_add(s.dispatcher_rva), bias);
        }
        Some(Self { sites, vip: first.instance.vip, seq: Rc::new(RefCell::new(Vec::new())), cap })
    }
}

impl Hook for SeqHook {
    fn interest(&self) -> Interest {
        Interest { flow_all: true, ..Interest::NONE }
    }
    fn on_flow(&mut self, ev: &FlowEvent, cpu: &mut Cpu, mem: &mut Memory) -> HookAction {
        if matches!(ev.kind, FlowKind::Jmp | FlowKind::Ret) {
            if let Some(&bias) = self.sites.get(&ev.from) {
                let v = match self.vip {
                    VipLoc::Reg(r) => cpu.reg(r),
                    VipLoc::Mem { base, disp } => cpu.reg(base).and_then(|b| {
                        let b = if base == Register::RSP { b.wrapping_add(bias) } else { b };
                        mem.read_uint(b.wrapping_add(disp as u64), 8).ok()
                    }),
                };
                let mut s = self.seq.borrow_mut();
                if let Some(v) = v {
                    if s.len() < self.cap {
                        s.push(v);
                    }
                }
            }
        }
        HookAction::Continue
    }
}

/// A freshly loaded image with DllMain already run; consumed by one export call.
pub struct Session {
    pub emu: Emulator,
    pub base: u64,
    pub size: u64,
    sentinel: u64,
    entry_rsp: u64,
    pub exports: BTreeMap<String, u64>,
    pub init_steps: u64,
    pub api_audit: Rc<RefCell<BTreeMap<String, u64>>>,
    pub api_facts: Rc<RefCell<vmpx_emulate::ObservedFacts>>,
    api: Box<dyn Hook>,
}

/// What one export call produced.
#[derive(Debug)]
pub struct CallResult {
    pub stop: Stop,
    pub rax: u64,
    pub steps: u64,
}

impl Session {
    /// Load `pe` and run DllMain(PROCESS_ATTACH) to its clean return.
    ///
    /// # Errors
    /// Load failure, or DllMain did not return cleanly within `init_budget`.
    pub fn new(pe: &PeImage<'_>, init_budget: u64) -> Result<Self, String> {
        Self::new_with(pe, init_budget, Vec::new())
    }

    /// As [`Session::new`], with extra observers present during init (they stay installed).
    ///
    /// # Errors
    /// As [`Session::new`].
    pub fn new_with(pe: &PeImage<'_>, init_budget: u64, init_extra: Vec<Box<dyn Hook>>) -> Result<Self, String> {
        let mut img = load(pe, &LoadOptions::default()).map_err(|e| format!("load: {e:?}"))?;
        let entry = img.entry.ok_or("no entry point")?;
        let base = img.image_base;
        let api = WinApiHook::from_image(&img);
        let api_audit = api.audit();
        let api_facts = api.facts();
        let sentinel = img.return_sentinel();
        let entry_rsp = img.initial_rsp;
        let exports = img.exports.clone();
        let mut emu = img.prepare(entry, &[base, 1, 0]).map_err(|e| format!("prepare: {e:?}"))?;
        emu.record_trace = false;
        emu.resolve_undefined_flags = true;
        let mut init_hooks: Vec<Box<dyn Hook>> = vec![Box::new(api)];
        init_hooks.extend(init_extra);
        emu.hook = Some(Box::new(Tee(init_hooks)));
        // The OS loader runs TLS callbacks (PROCESS_ATTACH) before the entry point; the
        // protector's runtime initialises its VM context there.
        let tls: Vec<u64> = pe.report().tls.as_ref().map(|t| t.callbacks.iter().map(|v| v.0).collect()).unwrap_or_default();
        for cb in tls {
            let sp = entry_rsp - 8;
            emu.mem.poke(sp, &sentinel.to_le_bytes()).map_err(|e| format!("poke: {e:?}"))?;
            emu.cpu.regs[RSP] = sp;
            emu.cpu.rflags = 2;
            emu.cpu.undef_flags = 0;
            emu.cpu.regs[WIN64_ARGS[0]] = base;
            emu.cpu.regs[WIN64_ARGS[1]] = 1;
            emu.cpu.regs[WIN64_ARGS[2]] = 0;
            emu.cpu.rip = cb;
            let stop = emu.run_until(sentinel, init_budget);
            if stop != Stop::Reached {
                return Err(format!("TLS callback {cb:#x} did not return: {stop:?}"));
            }
        }
        let sp = entry_rsp - 8;
        emu.mem.poke(sp, &sentinel.to_le_bytes()).map_err(|e| format!("poke: {e:?}"))?;
        emu.cpu.regs[RSP] = sp;
        emu.cpu.rflags = 2;
        emu.cpu.undef_flags = 0;
        emu.cpu.regs[WIN64_ARGS[0]] = base;
        emu.cpu.regs[WIN64_ARGS[1]] = 1;
        emu.cpu.regs[WIN64_ARGS[2]] = 0;
        emu.cpu.rip = entry;
        let stop = emu.run_until(sentinel, init_budget);
        if stop != Stop::Reached {
            return Err(format!("DllMain did not return: {stop:?}"));
        }
        let init_steps = emu.steps();
        let api = emu.hook.take().ok_or("hook lost")?;
        Ok(Self {
            emu,
            base,
            size: u64::from(pe.optional.size_of_image),
            sentinel,
            entry_rsp,
            exports,
            init_steps,
            api_audit,
            api_facts,
            api,
        })
    }

    /// Write guest memory (spike environment seeding; labelled in reports).
    ///
    /// # Errors
    /// The address is unmapped.
    pub fn poke(&mut self, addr: u64, bytes: &[u8]) -> Result<(), String> {
        self.emu.mem.poke(addr, bytes).map_err(|e| format!("poke {addr:#x}: {e:?}"))
    }

    /// Call `va` with Win64 integer `args` under `extra` observers.
    #[must_use]
    pub fn call(mut self, va: u64, args: &[u64], extra: Vec<Box<dyn Hook>>, budget: u64) -> CallResult {
        let sp = self.entry_rsp - 8;
        let _ = self.emu.mem.poke(sp, &self.sentinel.to_le_bytes());
        self.emu.cpu.regs[RSP] = sp;
        self.emu.cpu.rflags = 2;
        self.emu.cpu.undef_flags = 0;
        for (i, &r) in WIN64_ARGS.iter().enumerate() {
            self.emu.cpu.regs[r] = args.get(i).copied().unwrap_or(0);
        }
        self.emu.cpu.rip = va;
        let mut hooks = vec![self.api];
        hooks.extend(extra);
        self.emu.hook = Some(Box::new(Tee(hooks)));
        let before = self.emu.steps();
        let stop = self.emu.run_until(self.sentinel, budget);
        CallResult { stop, rax: self.emu.cpu.regs[RAX], steps: self.emu.steps() - before }
    }
}

/// Guest writes to apply after init and before the call: `(address, bytes)`.
pub type Pokes = Vec<(u64, Vec<u8>)>;

fn seeded(pe: &PeImage<'_>, pokes: &Pokes) -> Result<Session, String> {
    let mut s = Session::new(pe, 50_000_000)?;
    for (a, b) in pokes {
        s.poke(*a, b)?;
    }
    Ok(s)
}

/// Structural dispatch discovery on a (long enough) survey call; returns observer specs.
///
/// # Errors
/// No VM with a VIP location was found.
pub fn discover_specs(
    pe: &PeImage<'_>,
    export: &str,
    survey_arg: u64,
    region_rva: (u64, u64),
    pokes: &Pokes,
    budget: u64,
) -> Result<Vec<VmInstanceSpec>, String> {
    let s = seeded(pe, pokes)?;
    let va = *s.exports.get(export).ok_or("export not found")?;
    let (base, size) = (s.base, s.size);
    let hook = SurveyHook::new();
    let survey = hook.survey();
    let res = s.call(va, &[survey_arg], vec![Box::new(hook)], budget);
    let found = survey.borrow().discover_min((base, base + size), 100);
    let best = found.iter().find(|d| d.vip.is_some()).ok_or_else(|| {
        format!(
            "no VM with a VIP location found (stop={:?}, steps={}, groups={})",
            res.stop,
            res.steps,
            found.len()
        )
    })?;
    Ok(best.specs(base, 0, region_rva))
}

/// Observed run of `export(arg)`: de-duplicated edges (the bridge source) + ordered VIP sequence.
pub struct Observed {
    pub call: CallResult,
    pub obs: VmObservations,
    pub seq: Vec<u64>,
}

/// One observed call on a fresh session.
///
/// # Errors
/// Session setup failure.
pub fn observe(
    pe: &PeImage<'_>,
    export: &str,
    arg: u64,
    specs: &[VmInstanceSpec],
    pokes: &Pokes,
    budget: u64,
) -> Result<Observed, String> {
    let s = seeded(pe, pokes)?;
    let va = *s.exports.get(export).ok_or("export not found")?;
    let base = s.base;
    let vm = VmObserveHook::with_instances(base, specs.iter().copied(), None);
    let handle = vm.observations();
    let seqh = SeqHook::new(base, specs, 1 << 20).ok_or("no specs")?;
    let seq = Rc::clone(&seqh.seq);
    let call = s.call(va, &[arg], vec![Box::new(vm), Box::new(seqh)], budget);
    let obs = handle.borrow().clone();
    let seq = seq.borrow().clone();
    Ok(Observed { call, obs, seq })
}
