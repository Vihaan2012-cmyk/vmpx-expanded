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
    Perm, Stop, WinApiHook, Xplm, load,
};
use vmpx_pe::PeImage;

const SLICE: u64 = 1_000_000;

/// What the protector-loop folds actually did.
///
/// Without this, a fold that never matches looks exactly like a fold that
/// matches and does not help -- which is how the first attempt at speeding the
/// init up went unexplained for a session.
fn folds(prof: &vmpx_emulate::Prof) -> String {
    format!(
        "folds: checksum={} iters={} copies={} bytes={} literals={} bits={} declined={}",
        prof.folded,
        prof.folded_iterations,
        prof.copies_folded,
        prof.copied_bytes,
        prof.literals_folded,
        prof.literal_bits,
        prof.fold_declined
    ) + &format!(
        " | probes={} walks={} ({:.1}%) stale={}",
        prof.shape_probes,
        prof.shape_walks,
        if prof.shape_probes == 0 {
            0.0
        } else {
            100.0 * prof.shape_walks as f64 / prof.shape_probes as f64
        },
        prof.shape_stale
    )
}

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
        // How much of the run is CONTAINED, which is the only number that caps
        // what any fold of a region can be worth. A flat per-address profile
        // says nothing on its own: 1229 addresses each under 0.5% is what both
        // "one routine, fully unrolled" and "sweeping the whole image" look
        // like, and those need opposite work. The window shares separate them.
        let hottest = top[0].0;
        for window in [0x800u64, 0x2000, 0x8000, 0x4_0000, 0x100_0000] {
            let inside: u64 = self
                .hits
                .iter()
                .filter(|(rip, _)| rip.abs_diff(hottest) <= window)
                .map(|(_, hits)| *hits)
                .sum();
            let share = (inside as f64) * 100.0 / (samples as f64);
            let distinct = self.hits.keys().filter(|rip| rip.abs_diff(hottest) <= window).count();
            println!(
                "  within {window:#010x} of {hottest:#x}: {share:6.2}%  ({distinct} addresses)"
            );
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
    flow_calls: u64,
    flow_rets: u64,
    gcp: u64,
    start_rsp: u64,
    min_rsp: u64,
    gcp_log: Vec<(u64, u64, u64)>,
    wpm_addrs: Vec<u64>,
    gcp_regs: Vec<[u64; 16]>,
    // 48 qwords of stack [rsp-0x80, rsp+0x100) captured at each logged gcp call
    gcp_stack: Vec<Vec<u64>>,
    // (rip, addr, size, is_write, value) captured while gcp in [memlog_lo, memlog_hi]
    memlog: Vec<(u64, u64, u32, bool, u64)>,
    // (from, to, next, taken) for Jcc while gcp in [jcc_lo, jcc_hi]
    jcclog: Vec<(u64, u64, u64, bool)>,
    // (from, regs, rflags) captured when a flow source is in the watch set
    watch_regs: Vec<(u64, [u64; 16], u64)>,
    // recursion onset: first gcp call spaced exactly 0xC0 below the previous one
    prev_gcp_rsp: u64,
    onset: Option<(u64, u64)>, // (rsp, rip) at first 0xC0-spaced gcp
}

struct Watch {
    inner: WinApiHook,
    stubs: BTreeMap<u64, (String, String)>,
    sh: Rc<RefCell<Shared>>,
    idata: (u64, u64),
    diag: bool,
    memlog: Option<(u64, u64)>, // capture mem accesses while gcp in [lo, hi]
    jcclog: Option<(u64, u64)>, // capture Jcc branches while gcp in [lo, hi]
    watch: Vec<u64>,            // flow sources whose register state to capture
}

impl Hook for Watch {
    fn interest(&self) -> Interest {
        let mut base = self.inner.interest().union(Interest { write: [self.idata, (0, 0)], ..Interest::NONE });
        if self.memlog.is_some() {
            base = base.union(Interest { read_all: true, ..Interest::NONE });
        }
        if self.jcclog.is_some() || !self.watch.is_empty() {
            base = base.union(Interest { flow_all: true, ..Interest::NONE });
        }
        base
    }
    fn on_mem(&mut self, ev: &MemEvent, cpu: &mut Cpu, mem: &mut Memory) -> HookAction {
        if ev.access == Access::Write && ev.addr >= self.idata.0 && ev.addr < self.idata.1 {
            let mut s = self.sh.borrow_mut();
            s.idata_writes += 1;
            let b = s.base_steps;
            s.idata_first.get_or_insert(b);
            s.idata_last = b;
        }
        if let Some((lo, hi)) = self.memlog {
            let mut s = self.sh.borrow_mut();
            if s.gcp >= lo && s.gcp < hi && s.memlog.len() < 20000 {
                let is_write = ev.access == Access::Write;
                let v = ev.value.unwrap_or_else(|| {
                    mem.read_uint(ev.addr, (ev.size as usize).min(8)).unwrap_or(0)
                });
                s.memlog.push((cpu.rip, ev.addr, ev.size, is_write, v));
            }
        }
        self.inner.on_mem(ev, cpu, mem)
    }
    fn on_flow(&mut self, ev: &FlowEvent, cpu: &mut Cpu, mem: &mut Memory) -> HookAction {
        if !self.diag {
            return self.inner.on_flow(ev, cpu, mem);
        }
        if let (Some((lo, hi)), FlowKind::Jcc { taken }) = (self.jcclog, ev.kind) {
            let mut s = self.sh.borrow_mut();
            if s.gcp >= lo && s.gcp < hi && s.jcclog.len() < 200000 {
                s.jcclog.push((ev.from, ev.to, ev.next, taken));
            }
        }
        if !self.watch.is_empty() && self.watch.contains(&ev.from) {
            let mut s = self.sh.borrow_mut();
            if s.watch_regs.len() < 400 {
                s.watch_regs.push((ev.from, cpu.regs, cpu.rflags));
            }
        }
        let rsp = cpu.regs[4];
        {
            let mut s = self.sh.borrow_mut();
            if s.start_rsp == 0 {
                s.start_rsp = rsp;
                s.min_rsp = rsp;
            }
            s.min_rsp = s.min_rsp.min(rsp);
            match ev.kind {
                FlowKind::Call => s.flow_calls += 1,
                FlowKind::Ret => s.flow_rets += 1,
                _ => {}
            }
        }
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
                if f == "GetCurrentProcess" {
                    s.gcp += 1;
                    if s.onset.is_none() && s.prev_gcp_rsp != 0 && s.prev_gcp_rsp.wrapping_sub(rsp) == 0xC0 {
                        s.onset = Some((rsp, cpu.rip));
                    }
                    s.prev_gcp_rsp = rsp;
                    if s.gcp_log.len() < 12 {
                        let (g, start) = (s.gcp, s.start_rsp);
                        s.gcp_log.push((g, rsp, start.wrapping_sub(rsp)));
                        s.gcp_regs.push(cpu.regs);
                        let win: Vec<u64> = (0..48)
                            .map(|i| mem.read_uint(rsp.wrapping_sub(0x80).wrapping_add(i * 8), 8).unwrap_or(0))
                            .collect();
                        s.gcp_stack.push(win);
                    }
                }
                if f == "WriteProcessMemory" && s.wpm_addrs.len() < 60 {
                    s.wpm_addrs.push(cpu.regs[2]);
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
    let memlog = std::env::var("XPL_MEMLOG").ok().and_then(|v| {
        let (lo, hi) = v.split_once(':')?;
        Some((lo.parse::<u64>().ok()?, hi.parse::<u64>().ok()?))
    });
    let jcclog = std::env::var("XPL_JCC").ok().and_then(|v| {
        let (lo, hi) = v.split_once(':')?;
        Some((lo.parse::<u64>().ok()?, hi.parse::<u64>().ok()?))
    });
    let watchset: Vec<u64> = std::env::var("XPL_WATCH").ok().map(|v| v.split(',').filter_map(|s| u64::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok()).collect()).unwrap_or_default();
    let watch = Watch { inner: winapi, stubs: img.import_stubs.clone(), sh: Rc::clone(&sh), idata, diag: std::env::var_os("XPL_DIAG").is_some(), memlog, jcclog, watch: watchset };
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
    emu.accelerate_literals = std::env::var_os("XPL_NOLZMA").is_none();
    // Load-bearing, not an observer: `WinApiHook` IMPLEMENTS the Win32/XPLM
    // emulation, so dropping it to save per-write dispatch stops the run at the
    // first API call (measured: 122 steps).
    emu.hook = Some(Box::new(watch));
    emu.resolve_undefined_flags = std::env::var_os("XPL_NO_RESOLVE").is_none();
    // NOTE: pre-existing WIP — `Emulator::preserve_undef_flags` was never added to
    // the struct; re-enable once that field exists.
    // emu.preserve_undef_flags = std::env::var_os("XPL_OLD_FLAGS").is_none();
    emu.record_trace = false;

    if let Ok(inp) = std::env::var("XPL_SNAPSHOT_IN") {
        let (c, m) = vmpx_emulate::snapshot::load(std::path::Path::new(&inp)).expect("snapshot load");
        println!("restored from {inp}: rip={:#x}", c.rip);
        emu.cpu = c;
        emu.mem = m;
        // TEB.DeallocationStack (x64 offset 0x1478) — the loader never sets it (=0).
        // If the recursion's base case compares rsp against it, 0 makes it never stop.
        if let Some(v) = std::env::var("XPL_SET_DEALLOC").ok().and_then(|s| s.parse::<u64>().ok().or_else(|| u64::from_str_radix(s.trim_start_matches("0x"), 16).ok())) {
            let dealloc = if v == 1 { img_stack.0 } else { v };
            let _ = emu.mem.poke(gs + 0x1478, &dealloc.to_le_bytes());
            println!("set TEB.DeallocationStack (gs+0x1478) = {dealloc:#x}");
        }
        if let Some(mb) = std::env::var("XPL_EXTRA_STACK_MB").ok().and_then(|v| v.parse::<u64>().ok()) {
            let lo = (img_stack.0 - (mb << 20)) & !0xfff;
            match emu.mem.map(lo, img_stack.0 - lo, Perm::RW) {
                Ok(()) => println!("extended stack: mapped [{lo:#x},{:#x}) = {mb} MB below stack_base", img_stack.0),
                Err(e) => println!("stack extend failed: {e:?}"),
            }
        }
        if std::env::var_os("XPL_TRACE_OUT").is_some() {
            emu.record_trace = true;
        }
        let t = Instant::now();
        let before = emu.steps();
        let s = emu.run_until(sentinel, max_total);
        let ran = emu.steps() - before;
        let b = sh.borrow();
        println!(
            "resumed: ran {ran} steps stop={s:?} rip={:#x} rax={:#x} api_calls={} wall={:.1}s",
            emu.cpu.rip, emu.cpu.regs[0], b.calls.values().sum::<u64>(), t.elapsed().as_secs_f64()
        );
        println!(
            "flow: calls={} rets={} imbalance(calls-rets)={} | rsp start={:#x} min={:#x} now={:#x} consumed={}",
            b.flow_calls, b.flow_rets,
            b.flow_calls as i64 - b.flow_rets as i64,
            b.start_rsp, b.min_rsp, emu.cpu.regs[4],
            b.start_rsp.wrapping_sub(b.min_rsp)
        );
        if let Some((rsp, rip)) = b.onset {
            println!("RECURSION ONSET: first 0xC0-spaced gcp at rip={rip:#x} rsp={rsp:#x} (consumed-so-far {:#x})", b.start_rsp.wrapping_sub(rsp));
        }
        println!("GetCurrentProcess calls={}:", b.gcp);
        for (g, rsp, depth) in &b.gcp_log {
            println!("  gcp#{g} rsp={rsp:#x} stack_consumed_since_start={depth}");
        }
        println!("WriteProcessMemory targets ({} logged) — advancing=bounded sweep, repeating=divergence:", b.wpm_addrs.len());
        for (i, a) in b.wpm_addrs.iter().enumerate() {
            let delta = if i > 0 { *a as i64 - b.wpm_addrs[i - 1] as i64 } else { 0 };
            println!("  wpm[{i:>3}] = {a:#x} (delta {delta:+#x})");
        }
        let names = ["rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"];
        if let Some(r0) = b.gcp_regs.first() {
            println!("GetCurrentProcess level deltas (loop var = constant delta); level0 regs:");
            for (r, n) in names.iter().enumerate() {
                print!(" {n}={:#x}", r0[r]);
            }
            println!();
        }
        for i in 1..b.gcp_regs.len() {
            print!("  level {i} delta:");
            for r in 0..16 {
                let d = b.gcp_regs[i][r] as i64 - b.gcp_regs[i - 1][r] as i64;
                if d != 0 {
                    print!(" {}={:+#x}", names[r], d);
                }
            }
            println!();
        }
        if b.gcp_stack.len() >= 2 {
            println!("=== per-level stack window [rsp-0x80, rsp+0x100), relative-slot diffs ===");
            for lvl in 1..b.gcp_stack.len() {
                let (prev, cur) = (&b.gcp_stack[lvl - 1], &b.gcp_stack[lvl]);
                let diffs: Vec<String> = (0..cur.len())
                    .filter(|&i| cur[i] != prev[i])
                    .map(|i| format!("slot{}[rsp{:+#x}]: {:#x}->{:#x}", i, (i as i64 - 16) * 8, prev[i], cur[i]))
                    .collect();
                println!("level {lvl}: {} slots changed vs prev (same relative pos): {}", diffs.len(),
                    if diffs.is_empty() { "<<INVARIANT>>".into() } else { diffs.join("  ") });
            }
        }
        if !b.watch_regs.is_empty() {
            let names = ["rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"];
            println!("=== watched branch register state (ZF=1 would take the exit) ===");
            let froms: std::collections::BTreeSet<u64> = b.watch_regs.iter().map(|(f, ..)| *f).collect();
            for f in froms {
                println!("-- branch 0x{f:x} --");
                let caps: Vec<_> = b.watch_regs.iter().filter(|(ff, ..)| *ff == f).collect();
                let mut prev: Option<[u64; 16]> = None;
                for (i, (_, regs, rflags)) in caps.iter().take(6).enumerate() {
                    let zf = (*rflags >> 6) & 1;
                    let diff: Vec<String> = (0..16)
                        .filter(|&r| prev.map_or(true, |p| p[r] != regs[r]))
                        .map(|r| format!("{}={:#x}", names[r], regs[r]))
                        .collect();
                    println!("  lvl{i} ZF={zf}: {}", diff.join(" "));
                    prev = Some(*regs);
                }
            }
        }
        if !b.jcclog.is_empty() {
            use std::collections::BTreeMap as BM;
            // Loop body = every rip that was executed as a branch source.
            let body: std::collections::BTreeSet<u64> = b.jcclog.iter().map(|(f, ..)| *f).collect();
            // Per-branch: taken/not-taken counts and the followed/unfollowed targets.
            let mut per: BM<u64, (u64, u64, u64, u64)> = BM::new(); // from -> (ntaken, nnot, to, next)
            for &(f, to, next, taken) in &b.jcclog {
                let e = per.entry(f).or_insert((0, 0, to, next));
                if taken { e.0 += 1 } else { e.1 += 1 }
            }
            println!("=== Jcc analysis: {} branches captured, {} unique sources ===", b.jcclog.len(), per.len());
            // Exit branch candidate: always-one-way, whose UNFOLLOWED target is outside the loop body.
            println!("always-one-way branches whose road-not-taken leaves the loop body (base-case candidates):");
            for (f, (nt, nn, to, next)) in &per {
                let always_taken = *nn == 0 && *nt > 0;
                let always_not = *nt == 0 && *nn > 0;
                let unfollowed = if always_taken { *next } else if always_not { *to } else { continue };
                if !body.contains(&unfollowed) {
                    println!("  from={f:#x} taken={nt} not={nn} followed={:#x} EXIT_TO={unfollowed:#x}",
                        if always_taken { *to } else { *next });
                }
            }
        }
        if !b.memlog.is_empty() {
            println!("=== memlog: {} accesses captured in window ===", b.memlog.len());
            // Reads of a FIXED address across the window = the env cell the scan re-reads.
            let mut by_addr: BTreeMap<u64, (u64, u32, bool, u64)> = BTreeMap::new();
            for &(rip, addr, size, w, v) in &b.memlog {
                let e = by_addr.entry(addr).or_insert((0, size, w, v));
                e.0 += 1;
                e.3 = v;
            }
            let mut rep: Vec<_> = by_addr.iter().collect();
            rep.sort_by_key(|(_, (n, ..))| std::cmp::Reverse(*n));
            println!("most-read addresses (addr: count size write last_val):");
            for (addr, (n, size, w, v)) in rep.iter().take(20) {
                println!("  {addr:#x}: n={n} size={size} write={w} val={v:#x}");
            }
            println!("first 40 accesses (rip addr size write val):");
            for &(rip, addr, size, w, v) in b.memlog.iter().take(40) {
                println!("  rip={rip:#x} addr={addr:#x} size={size} {} val={v:#x}", if w { "W" } else { "R" });
            }
        }
        let mut top: Vec<_> = b.calls.iter().collect();
        top.sort_by_key(|x| std::cmp::Reverse(*x.1));
        println!("top calls: {:?}", top.iter().take(12).collect::<Vec<_>>());
        drop(b);
        if let Ok(out) = std::env::var("XPL_TRACE_OUT") {
            let mut f = std::fs::File::create(&out).unwrap();
            for te in &emu.trace {
                let _ = writeln!(f, "{:#x}\t{:?}\tea={:?}\tval={:?}\tvsp={:?}", te.rip, te.mnemonic, te.ea, te.value, te.vsp);
            }
            println!("trace {} entries -> {out}", emu.trace.len());
        }
        println!("{}", folds(&emu.prof));
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
                    println!("   {}", folds(&emu.prof));
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
    println!("{}", folds(&emu.prof));
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
