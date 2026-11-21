//! Drive the self-protected VMProtect binary past its self-integrity "guard"
//! branches (anti-debug / "File corrupted" tamper checks) by redirecting the
//! guard's conditional jump away from the hard-error path, then report how far it
//! gets. VMProtect's self-checks are guards (branch-to-error on failure), so
//! forcing the good branch lets the real work (its protection engine) proceed —
//! the same "neutralize, don't devirtualize" tactic used for native guard entries.
//!
//! Guards are given as `VMPX_GUARDS=<from>:<badto>:<goodto>,...` (hex VAs): when
//! flow goes `from -> badto`, it is redirected to `goodto`. A default guard (the
//! observed "File corrupted" branch) is applied when the env is unset.
//! usage: vmp_neutralize <exe-or-rvamap> [max_steps]
use std::cell::RefCell;
use std::rc::Rc;

use vmpx_emulate::{
    detonate_with, export_reference_from_imports, Cpu, DetonateOptions, FlowEvent, Hook,
    HookAction, Interest, Memory, SysDllDb, DEFAULT_SYSTEM_MODULES,
};
use vmpx_pe::PeImage;

// The "File corrupted" verdict reaches the raiser via the SHARED helper
// `0x1427cad3c: ja 0x1427cadd4` -> `jmp rsi`, where rsi is the hash-selected target.
// At the real integrity failure rsi == the error handler (0x14289e9ff), so gating
// the redirect on that value targets ONLY the integrity check (the helper is reused
// all over; redirecting an unrelated use corrupts control flow). VMPX_GATE_RSI
// overrides the gate value.
struct Guards {
    rules: Vec<((u64, u64), u64)>,
    gate_rsi: u64,
    dbg: Rc<RefCell<u32>>,
    hits: Rc<RefCell<Vec<(u64, u64, u64)>>>,
}

impl Hook for Guards {
    fn interest(&self) -> Interest {
        Interest { flow_all: true, ..Interest::NONE }
    }
    fn on_flow(&mut self, ev: &FlowEvent, cpu: &mut Cpu, _mem: &mut Memory) -> HookAction {
        let rsi = cpu.reg(iced_x86::Register::RSI).unwrap_or(0);
        // Debug: log any dispatch whose target rsi lands in the error-handler page
        // (0x14289e000..0x1428a0000), to catch the "File corrupted" dispatch itself.
        if std::env::var_os("VMPX_GLOG").is_some()
            && (0x1_4289_e000..0x1_428a_0000).contains(&rsi)
        {
            eprintln!("[glog-err] {:#x} -> {:#x}  rsi={rsi:#x}", ev.from, ev.to);
        }
        if self.gate_rsi != 0 && rsi != self.gate_rsi {
            return HookAction::Continue;
        }
        for &((from, bad), good) in &self.rules {
            if ev.from == from && ev.to == bad {
                cpu.rip = good;
                self.hits.borrow_mut().push((from, bad, good));
                return HookAction::Redirect;
            }
        }
        HookAction::Continue
    }
}

fn parse_guards() -> Vec<((u64, u64), u64)> {
    let h = |s: &str| u64::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok();
    match std::env::var("VMPX_GUARDS") {
        Ok(v) => v
            .split(',')
            .filter_map(|g| {
                let p: Vec<&str> = g.split(':').collect();
                match p.as_slice() {
                    [a, b, c] => Some(((h(a)?, h(b)?), h(c)?)),
                    _ => None,
                }
            })
            .collect(),
        // Default: the observed "File corrupted" guard branch.
        Err(_) => vec![((0x1_427c_ad3c, 0x1_427c_add4), 0x1_427c_ad42)],
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let bytes = std::fs::read(&a[1]).expect("image");
    let max_steps: u64 = a.get(2).and_then(|v| v.parse().ok()).unwrap_or(8_000_000_000);
    let pe = PeImage::parse(&bytes).unwrap();

    let mut export_reference = export_reference_from_imports(&pe);
    let db = SysDllDb::new();
    for m in DEFAULT_SYSTEM_MODULES {
        if let Some(n) = db.exports(m) {
            export_reference.insert((*m).to_owned(), n);
        }
    }
    let discover = std::env::var_os("VMPX_NO_DISCOVER").is_none();
    let opts = DetonateOptions {
        max_steps,
        vm_discover: discover,
        export_reference,
        dump: false,
        ..Default::default()
    };

    let gate_rsi = std::env::var("VMPX_GATE_RSI")
        .ok()
        .and_then(|s| u64::from_str_radix(s.trim_start_matches("0x"), 16).ok())
        .unwrap_or(0x1_4289_e9ff);
    let hits = Rc::new(RefCell::new(Vec::new()));
    let hook = Guards { rules: parse_guards(), gate_rsi, dbg: Rc::new(RefCell::new(0)), hits: Rc::clone(&hits) };
    println!("guards: {} rule(s), gate rsi={:#x}", hook.rules.len(), gate_rsi);

    match detonate_with(&pe, &opts, Some(Box::new(hook))) {
        Ok(r) => {
            println!("stop={:?} steps={} rip={:#x}", r.stop, r.steps, r.rip);
            println!("oep={:?}", r.observed.oep);
            let h = hits.borrow();
            println!("guards fired: {}", h.len());
            for (from, bad, good) in h.iter().take(20) {
                println!("  {from:#x}: {bad:#x} -> {good:#x}");
            }
            if !r.audit.is_empty() {
                println!("audit:");
                for (k, v) in r.audit.iter().take(20) {
                    println!("  {k}: {v}");
                }
            }
        }
        Err(e) => println!("detonate error: {e:?}"),
    }
}
