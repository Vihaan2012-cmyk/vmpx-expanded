//! Diagnose the anti-emulation bail: record a ring buffer of the last control-flow
//! transfers (with RAX, since the exit gate is `and al,1; jne`) and dump them when
//! the run stops at the `ExitProcess` call. The last conditional branch before the
//! exit is the bail decision; its `from` address + the RAX feeding it reveal the
//! check. usage: vmp_bail <exe-or-rvamap> [max_steps]
use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

use vmpx_emulate::{
    detonate_with, export_reference_from_imports, Access, Cpu, DetonateOptions, FlowEvent, FlowKind,
    Hook, HookAction, Interest, MemEvent, Memory, SysDllDb, DEFAULT_SYSTEM_MODULES,
};
use vmpx_pe::PeImage;

const CAP: usize = 120;
const RCAP: usize = 4000; // recent-read ring buffer depth

struct FlowTrace {
    buf: Rc<RefCell<VecDeque<(FlowKind, u64, u64, u64)>>>,
    reads: Rc<RefCell<VecDeque<(u64, u64)>>>, // (addr, size) of ALL recent reads
    writes: Rc<RefCell<u64>>,                 // writes into the checksummed regions
    awrites: Rc<RefCell<Vec<(u64,u64,u64)>>>, // (addr,size,value) writes into the result array
    envreg: Rc<RefCell<std::collections::BTreeMap<u64,u64>>>,   // non-image read regions (>>16)
    envexact: Rc<RefCell<std::collections::BTreeMap<u64,u64>>>, // exact KUSD/TEB reads
    trig: Rc<Option<u64>>,                    // stop when flow reaches this address
    dump: Rc<Option<u64>>,                    // dump GPRs when flow reaches this address
    dumped: Rc<RefCell<bool>>,
}

impl Hook for FlowTrace {
    fn interest(&self) -> Interest {
        Interest { flow_all: true, read: [(0, u64::MAX), (0, 0)], ..Interest::NONE }
    }
    fn on_flow(&mut self, ev: &FlowEvent, cpu: &mut Cpu, mem: &mut Memory) -> HookAction {
        use iced_x86::Register as R;
        let rax = cpu.reg(R::RAX).unwrap_or(0);
        // VMPX_BAIL_DUMP=<hex addr>: dump all GPRs the first time flow reaches it.
        if let Some(d) = *self.dump {
            if (ev.to == d || ev.from == d) && !*self.dumped.borrow() {
                *self.dumped.borrow_mut() = true;
                let regs = [R::RAX, R::RBX, R::RCX, R::RDX, R::RSI, R::RDI, R::RBP, R::RSP,
                            R::R8, R::R9, R::R10, R::R11, R::R12, R::R13, R::R14, R::R15];
                let names = ["rax","rbx","rcx","rdx","rsi","rdi","rbp","rsp","r8","r9","r10","r11","r12","r13","r14","r15"];
                eprintln!("[dump @ {:#x}]", d);
                for (r, n) in regs.iter().zip(names) {
                    eprintln!("  {n}={:#x}", cpu.reg(*r).unwrap_or(0));
                }
                // Dump the runtime bytes of the hashed region at rbp, to compare
                // against the on-disk image (detects runtime in-place decryption).
                let rbp = cpu.reg(R::RBP).unwrap_or(0);
                eprintln!("  --- runtime bytes [rbp-0x20 .. rbp+0x20] (rbp={rbp:#x}) ---");
                for base in [rbp.wrapping_sub(0x20), rbp, rbp.wrapping_add(0x10)] {
                    let mut s = String::new();
                    for j in 0..16 {
                        s.push_str(&format!("{:02x} ", mem.read_uint(base + j, 1).unwrap_or(0)));
                    }
                    eprintln!("    [{base:#x}] {s}");
                }
            }
        }
        // VMPX_BAIL_TRIG=<hex addr>: when flow first reaches that address, stop so the
        // ring buffer holds the path INTO it (e.g. the hard-error caller 0x14289ead9).
        if let Some(t) = *self.trig {
            if ev.to == t || ev.from == t {
                return HookAction::Stop;
            }
        }
        let mut b = self.buf.borrow_mut();
        if b.len() >= CAP {
            b.pop_front();
        }
        b.push_back((ev.kind, ev.from, ev.to, rax));
        HookAction::Continue
    }
    fn on_mem(&mut self, ev: &MemEvent, _cpu: &mut Cpu, _mem: &mut Memory) -> HookAction {
        if ev.access == Access::Read {
            let mut r = self.reads.borrow_mut();
            if r.len() >= RCAP {
                r.pop_front();
            }
            r.push_back((ev.addr, ev.size as u64));
            // Histogram reads of NON-image, NON-stack memory (env inputs that could
            // seed the VM rolling key): bucket by 64KiB, plus exact KUSD/TEB/PEB.
            let a = ev.addr;
            let in_image = (0x1_4000_0000..0x1_4532_1000).contains(&a);
            let in_stack = (0x10_0000_0000..0x10_0004_0000).contains(&a);
            if !in_image && !in_stack {
                *self.envreg.borrow_mut().entry(a >> 16).or_insert(0) += 1;
                // KUSD (0x7ffe0000..) and TEB/PEB (0x30_0000_0000..) exact.
                if (0x7ffe_0000..0x7fff_0000).contains(&a)
                    || (0x30_0000_0000..0x30_0010_0000).contains(&a)
                {
                    *self.envexact.borrow_mut().entry(a).or_insert(0) += 1;
                }
            }
        }
        // VMPX_WWATCH=<hex>: stop at the first WRITE into [addr, addr+0x40) and
        // report the writing instruction + value (locating the decrypt routine).
        if ev.access == Access::Write {
            let lo = std::env::var("VMPX_WWATCH")
                .ok()
                .and_then(|s| u64::from_str_radix(s.trim_start_matches("0x"), 16).ok())
                .unwrap_or(0);
            if lo != 0 && (lo..lo + 0x40).contains(&ev.addr) {
                eprintln!(
                    "[WWATCH] write {:#x} sz={} val={:#x} at rip={:#x}",
                    ev.addr, ev.size, ev.value.unwrap_or(0), _cpu.rip
                );
                return HookAction::Stop;
            }
        }
        HookAction::Continue
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let bytes = std::fs::read(&a[1]).expect("image");
    let max_steps: u64 = a.get(2).and_then(|v| v.parse().ok()).unwrap_or(200_000_000);
    let pe = PeImage::parse(&bytes).unwrap();

    let mut export_reference = export_reference_from_imports(&pe);
    let db = SysDllDb::new();
    for m in DEFAULT_SYSTEM_MODULES {
        if let Some(n) = db.exports(m) {
            export_reference.insert((*m).to_owned(), n);
        }
    }
    // VMPX_NO_ACCEL=1 disables protector-loop folding so per-byte writes (e.g. the
    // section decrypt) surface as individual on_mem events.
    let accel = std::env::var_os("VMPX_NO_ACCEL").is_none();
    let opts = DetonateOptions {
        max_steps,
        export_reference,
        accelerate_protector_loops: accel,
        ..Default::default()
    };

    let trig = Rc::new(
        std::env::var("VMPX_BAIL_TRIG")
            .ok()
            .and_then(|s| u64::from_str_radix(s.trim_start_matches("0x"), 16).ok()),
    );
    let buf = Rc::new(RefCell::new(VecDeque::new()));
    let reads = Rc::new(RefCell::new(VecDeque::new()));
    let writes = Rc::new(RefCell::new(0u64));
    let awrites = Rc::new(RefCell::new(Vec::new()));
    let envreg = Rc::new(RefCell::new(std::collections::BTreeMap::new()));
    let envexact = Rc::new(RefCell::new(std::collections::BTreeMap::new()));
    let dump = Rc::new(std::env::var("VMPX_BAIL_DUMP").ok().and_then(|s| u64::from_str_radix(s.trim_start_matches("0x"),16).ok()));
    let dumped = Rc::new(RefCell::new(false));
    let hook = FlowTrace { buf: Rc::clone(&buf), reads: Rc::clone(&reads), writes: Rc::clone(&writes), awrites: Rc::clone(&awrites), envreg: Rc::clone(&envreg), envexact: Rc::clone(&envexact), trig: Rc::clone(&trig), dump: Rc::clone(&dump), dumped: Rc::clone(&dumped) };

    match detonate_with(&pe, &opts, Some(Box::new(hook))) {
        Ok(r) => {
            println!("stop={:?} steps={} rip={:#x}", r.stop, r.steps, r.rip);
            {
                let er=envreg.borrow(); let mut v:Vec<_>=er.iter().collect(); v.sort_by(|a,b|b.1.cmp(a.1));
                println!("=== non-image/non-stack read regions (64KiB, top 15) ===");
                for (k,c) in v.iter().take(15){ println!("  {:#013x}  {} reads", (*k)<<16, c); }
                let ee=envexact.borrow(); let mut w:Vec<_>=ee.iter().collect(); w.sort_by(|a,b|b.1.cmp(a.1));
                println!("=== exact KUSD/TEB-PEB reads (top 25) ===");
                for (k,c) in w.iter().take(25){ println!("  {:#013x}  {} reads", k, c); }
            }
            // Histogram the recent reads by 64KiB region, to reveal what memory the
            // checksum/comparison right before the stop was sweeping.
            let rd = reads.borrow();
            let mut regions: std::collections::BTreeMap<u64, (usize, u64, u64)> =
                std::collections::BTreeMap::new();
            for (addr, _sz) in rd.iter() {
                let e = regions.entry(addr >> 16).or_insert((0, u64::MAX, 0));
                e.0 += 1;
                e.1 = e.1.min(*addr);
                e.2 = e.2.max(*addr);
            }
            println!("checksum-region writes during run: {}", writes.borrow());
            let aw=awrites.borrow();
            println!("result-array writes: {}", aw.len());
            for (ad,sz,v) in aw.iter() { println!("  [{ad:#x}] sz={sz} = {v:#x}"); }
            println!("=== recent-read regions (last {} reads, by 64KiB) ===", rd.len());
            let mut rv: Vec<_> = regions.iter().collect();
            rv.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));
            for (_k, (cnt, lo, hi)) in rv.iter().take(20) {
                println!("  {cnt:6} reads  [{lo:#013x}..{hi:#013x}]");
            }
            println!("=== last {} control-flow transfers before the stop (kind from -> to | rax) ===", CAP);
            for (k, from, to, rax) in buf.borrow().iter() {
                let tag = match k {
                    FlowKind::Jcc { taken } => format!("Jcc[{}]", if *taken { "taken" } else { "fall" }),
                    other => format!("{other:?}"),
                };
                println!("  {tag:12} {from:#011x} -> {to:#011x}   rax={rax:#x} (al={:#04x})", rax & 0xff);
            }
        }
        Err(e) => println!("detonate error: {e:?}"),
    }
}
