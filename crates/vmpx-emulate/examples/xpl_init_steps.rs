//! Measure emulator steps to reach stable post-init for a mapped (unpacked) `.xpl`.
//! usage: xpl_init_steps <mapped.bin> <max_steps> [csv_out]
//! Runs TLS callbacks then DllMain(PROCESS_ATTACH) in 1M-step slices (no trace; memory stays flat),
//! logging per-slice milestones: rip, .idata writes, first-seen Win32/XPLM imports.
//! Env: XPL_PHASES=tls0,DllMain selects phases; XPL_NOREF=1 drops the export reference;
//! XPL_STACK_KB overrides the guest stack (default 8 MiB).
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::io::Write;
use std::rc::Rc;
use std::time::Instant;

use vmpx_emulate::{
    Access, Cpu, FlowEvent, FlowKind, Hook, HookAction, Interest, LoadOptions, MemEvent, Memory,
    Stop, WinApiHook, Xplm, load,
};
use vmpx_pe::PeImage;

const SLICE: u64 = 1_000_000;

/// Sampling profiler over guest `rip`.
///
/// There was no way to see WHERE the init spends its steps, only how many it
/// spends, so every speed decision was a guess -- and the first guess (the
/// protector-loop folds) measured 90M steps/s against a 116M/s baseline with no
/// step reduction at all, because those folds never matched this code.
///
/// `XPL_PROFILE=<slice>` runs in slices of that many steps and records `rip`
/// between them. Reading a register between slices costs nothing per step, so
/// the profile is free; at a 20k slice a 39B-step init yields ~2M samples.
#[derive(Default)]
struct Profile {
    slice: u64,
    hits: BTreeMap<u64, u64>,
}

impl Profile {
    fn from_env() -> Self {
        let slice = std::env::var("XPL_PROFILE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        Self { slice, hits: BTreeMap::new() }
    }

    fn on(&self) -> bool {
        self.slice > 0
    }

    fn sample(&mut self, rip: u64) {
        *self.hits.entry(rip).or_default() += 1;
    }

    fn report(&self, total_steps: u64) {
        if !self.on() || self.hits.is_empty() {
            return;
        }
        let samples: u64 = self.hits.values().sum();
        let mut top: Vec<(u64, u64)> = self.hits.iter().map(|(k, v)| (*k, *v)).collect();
        top.sort_by_key(|(_, hits)| std::cmp::Reverse(*hits));
        println!(
            "profile: {samples} samples over {total_steps} steps, {} distinct rip",
            self.hits.len()
        );
        for (rip, hits) in top.iter().take(25) {
            let share = (*hits as f64) * 100.0 / (samples as f64);
            let steps = (*hits as f64) * (self.slice as f64);
            println!("  {rip:#x}  {share:6.2}%  ~{:.2}e9 steps", steps / 1e9);
        }
    }
}

#[derive(Default)]
struct Shared {
    base_steps: u64,
    idata_writes: u64,
    idata_first: Option<u64>,
    idata_last: u64,
    first_seen: Vec<(u64, String)>,
    calls: BTreeMap<String, u64>,
}

struct Watch {
    inner: WinApiHook,
    stubs: BTreeMap<u64, (String, String)>,
    sh: Rc<RefCell<Shared>>,
    idata: (u64, u64),
}

impl Hook for Watch {
    fn interest(&self) -> Interest {
        self.inner.interest().union(Interest { write: [self.idata, (0, 0)], ..Interest::NONE })
    }
    fn on_mem(&mut self, ev: &MemEvent, cpu: &mut Cpu, mem: &mut Memory) -> HookAction {
        if ev.access == Access::Write && ev.addr >= self.idata.0 && ev.addr < self.idata.1 {
            let mut s = self.sh.borrow_mut();
            s.idata_writes += 1;
            let b = s.base_steps;
            s.idata_first.get_or_insert(b);
            s.idata_last = b;
        }
        self.inner.on_mem(ev, cpu, mem)
    }
    fn on_flow(&mut self, ev: &FlowEvent, cpu: &mut Cpu, mem: &mut Memory) -> HookAction {
        if matches!(ev.kind, FlowKind::Call | FlowKind::Jmp | FlowKind::Ret) {
            if let Some((d, f)) = self.stubs.get(&ev.to) {
                let k = format!("{d}!{f}");
                let mut s = self.sh.borrow_mut();
                let n = s.calls.entry(k.clone()).or_default();
                *n += 1;
                if *n == 1 {
                    let b = s.base_steps;
                    s.first_seen.push((b, k));
                }
            }
        }
        self.inner.on_flow(ev, cpu, mem)
    }
    fn on_syscall(&mut self, cpu: &mut Cpu, mem: &mut Memory) -> Option<HookAction> {
        self.inner.on_syscall(cpu, mem)
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mapped = std::fs::read(&a[1]).expect("mapped image");
    let max_total: u64 = a[2].parse().expect("max_steps");
    let mut csv = a.get(3).map(|p| std::fs::File::create(p).expect("csv"));
    let pe = PeImage::parse(&mapped).unwrap();
    let rep = pe.report();
    let mut img = load(&pe, &LoadOptions { stack_size: std::env::var("XPL_STACK_KB").ok().and_then(|v| v.parse::<u64>().ok()).map_or(8 << 20, |kb| kb << 10) }).unwrap();
    let base = img.image_base;
    let sentinel = img.return_sentinel();
    let (initial_rsp, gs, entry) = (img.initial_rsp, img.gs_base, img.entry.unwrap());
    let img_stack = (img.stack_base, img.stack_top);
    let tls: Vec<u64> = rep.tls.as_ref().map(|t| t.callbacks.iter().map(|v| v.0).collect()).unwrap_or_default();
    let idata = rep
        .sections
        .iter()
        .find(|s| s.name.eq_ignore_ascii_case(".idata"))
        .map_or((0, 0), |s| (base + u64::from(s.virtual_address.0), base + u64::from(s.virtual_address.0) + u64::from(s.mapped_span())));

    let mut reference = vmpx_emulate::export_reference_from_imports(&pe);
    let db = vmpx_emulate::SysDllDb::new();
    let mut mods: std::collections::BTreeSet<String> =
        vmpx_emulate::DEFAULT_SYSTEM_MODULES.iter().map(|m| (*m).to_owned()).collect();
    mods.extend(reference.keys().cloned());
    for m in mods {
        if let Some(n) = db.exports(&m) {
            reference.insert(m, n);
        }
    }
    if std::env::var_os("XPL_NOREF").is_some() {
        reference.clear();
    }
    let mut winapi = WinApiHook::from_image(&img).with_export_reference(reference);
    if let Some(n) = db.syscall_numbers("ntdll.dll") {
        winapi = winapi.with_syscall_numbers(n);
    }
    winapi = winapi.with_xplm(Rc::new(RefCell::new(Xplm::new())));
    let audit = winapi.audit();
    let sh = Rc::new(RefCell::new(Shared::default()));
    let watch = Watch { inner: winapi, stubs: img.import_stubs.clone(), sh: Rc::clone(&sh), idata };
    println!("entry={entry:#x} tls={tls:x?} idata={idata:x?} max_steps={max_total}");
    let mut emu = img.prepare(entry, &[]).unwrap();
    // Match the detonation path, which sets `accelerate_protector_loops: true` by
    // default. This example drives the emulator directly instead of going through
    // `detonate`, so it inherited `Emulator`'s `false` and interpreted every
    // VMProtect integrity and copy loop byte by byte. Those folds replace N
    // iterations with one closed-form step, and the init is tens of BILLIONS of
    // steps, so this is a step-COUNT lever, not a step-rate one.
    //
    // `XPL_NOACCEL=1` restores byte-by-byte interpretation for an A/B.
    emu.accelerate_checksums = std::env::var_os("XPL_NOACCEL").is_none();
    // Load-bearing, not an observer: `WinApiHook` IMPLEMENTS the Win32/XPLM
    // emulation, so dropping it to save per-write dispatch stops the run at the
    // first API call (measured: 122 steps).
    emu.hook = Some(Box::new(watch));
    emu.resolve_undefined_flags = true;
    emu.record_trace = false;

    if let Ok(inp) = std::env::var("XPL_SNAPSHOT_IN") {
        let (c, m) = vmpx_emulate::snapshot::load(std::path::Path::new(&inp)).expect("snapshot load");
        println!("restored from {inp}: rip={:#x}", c.rip);
        emu.cpu = c;
        emu.mem = m;
        let t = Instant::now();
        let before = emu.steps();
        let s = emu.run_until(sentinel, max_total);
        let ran = emu.steps() - before;
        let b = sh.borrow();
        println!(
            "resumed: ran {ran} steps stop={s:?} rip={:#x} rax={:#x} api_calls={} wall={:.1}s",
            emu.cpu.rip, emu.cpu.regs[0], b.calls.values().sum::<u64>(), t.elapsed().as_secs_f64()
        );
        // The resume path used to end here, so a continued init could not be
        // saved and every attempt had to restart from zero. The phase budget is
        // the binding cost -- 120B steps did not finish even the first TLS
        // callback -- so continuing has to be able to checkpoint.
        if let Ok(path) = std::env::var("XPL_SNAPSHOT_OUT") {
            let p = std::path::Path::new(&path);
            vmpx_emulate::snapshot::save(&emu.cpu, &emu.mem, p).expect("snapshot save");
            let size = std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
            println!(
                "snapshot saved -> {path} ({size} bytes) rip={:#x} ran={ran}",
                emu.cpu.rip
            );
        }
        return;
    }

    let mut phases: Vec<(String, u64, Vec<u64>)> = Vec::new();
    for (i, cb) in tls.iter().enumerate() {
        phases.push((format!("tls{i}"), *cb, vec![base, 1, 0]));
    }
    phases.push(("DllMain".into(), entry, vec![base, 1, 0]));
    if let Ok(want) = std::env::var("XPL_PHASES") {
        phases.retain(|(l, _, _)| want.split(',').any(|w| w == l));
    }

    let mut prof = Profile::from_env();
    let slice = if prof.on() { prof.slice } else { SLICE };
    let t0 = Instant::now();
    let mut total = 0u64;
    'outer: for (label, f, args) in &phases {
        emu.cpu = Cpu::new(*f);
        emu.cpu.gs_base = gs;
        emu.cpu.regs[4] = initial_rsp - 8;
        let _ = emu.mem.poke(initial_rsp - 8, &sentinel.to_le_bytes());
        for (i, v) in args.iter().enumerate() {
            emu.cpu.regs[[1usize, 2, 8, 9][i]] = *v;
        }
        let start = total;
        let stop = loop {
            sh.borrow_mut().base_steps = total;
            let budget = slice.min(max_total.saturating_sub(total));
            if budget == 0 {
                break Stop::StepLimit;
            }
            let before = emu.steps();
            let s = emu.run_until(sentinel, budget);
            total += emu.steps() - before;
            if prof.on() {
                prof.sample(emu.cpu.rip);
            }
            {
                let b = sh.borrow();
                if let Some(c) = csv.as_mut() {
                    let _ = writeln!(c, "{label},{total},{:#x},{},{},{:#x}", emu.cpu.rip, b.idata_writes, b.calls.values().sum::<u64>(), emu.cpu.regs[4]);
                }
                if (total / slice) % (100 * SLICE / slice).max(1) == 0 {
                    println!(
                        "[{label}] steps={total} rip={:#x} idata_w={} api_calls={} distinct_api={} wall={:.0}s",
                        emu.cpu.rip, b.idata_writes, b.calls.values().sum::<u64>(), b.calls.len(), t0.elapsed().as_secs_f64()
                    );
                    let _ = std::io::stdout().flush();
                }
            }
            if !matches!(s, Stop::StepLimit) {
                break s;
            }
            if total >= max_total {
                break Stop::StepLimit;
            }
        };
        println!("[{label}] done steps_in_phase={} total={total} stop={stop:?} rip={:#x} rax={:#x} wall={:.0}s", total - start, emu.cpu.rip, emu.cpu.regs[0], t0.elapsed().as_secs_f64());
        if !matches!(stop, Stop::Reached) {
            if let Stop::Fault { addr, .. } = &stop {
                println!("   fault addr {addr:#x} perm={:?}", emu.mem.perm_at(*addr));
            }
            println!("   rsp={:#x} stack=[{:#x},{:#x})", emu.cpu.regs[4], img_stack.0, img_stack.1);
            break 'outer;
        }
    }
    if let Ok(path) = std::env::var("XPL_SNAPSHOT_OUT") {
        let p = std::path::Path::new(&path);
        vmpx_emulate::snapshot::save(&emu.cpu, &emu.mem, p).expect("snapshot save");
        let sz = std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
        println!("snapshot saved -> {path} ({sz} bytes) rip={:#x} total_steps={total} wall={:.0}s", emu.cpu.rip, t0.elapsed().as_secs_f64());
    }
    prof.report(total);
    let s = sh.borrow();
    println!("idata writes={} first_at={:?} last_at={}", s.idata_writes, s.idata_first, s.idata_last);
    println!("first-seen imports (steps, name):");
    for (st, n) in &s.first_seen {
        println!("  {st:>12} {n}");
    }
    let mut top: Vec<_> = s.calls.iter().collect();
    top.sort_by_key(|x| std::cmp::Reverse(*x.1));
    println!("top calls: {:?}", top.iter().take(15).collect::<Vec<_>>());
    println!("audit: {:?}", audit.borrow());
}
