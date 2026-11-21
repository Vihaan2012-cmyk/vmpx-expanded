//! Test the frozen-clock anti-emulation hypothesis: a hook advances the
//! KUSER_SHARED_DATA time fields (InterruptTime @0x008, SystemTime @0x014,
//! TickCount @0x320) as the run progresses, so a VMProtect timing check that
//! reads the clock twice sees it move. If the run gets PAST the known bail
//! (`call rax` -> ExitProcess @0x1429368e7), the frozen clock was the tell.
//! usage: vmp_clockfix <exe> [max_steps]
use vmpx_emulate::{
    detonate_with, export_reference_from_imports, Cpu, DetonateOptions, FlowEvent, FlowKind, Hook,
    HookAction, Interest, Memory, SysDllDb, DEFAULT_SYSTEM_MODULES,
};
use vmpx_pe::PeImage;

const KUSD: u64 = 0x7FFE_0000;
const FIXED_FILETIME: u64 = 0x01D0_0000_0000_0000;
// Uptime base = 4h 38m 16s = 16,696,000 ms (and advancing from there).
const UPTIME_MS: u64 = 16_696_000;
const UPTIME_100NS: u64 = UPTIME_MS * 10_000;
// KUSD.TickCountMultiplier (0x004); ElapsedMs = (TickCountQuad * MULT) >> 24.
const TICK_MULT: u128 = 0x0FA0_0000;

struct ClockFix {
    t: u64,     // elapsed 100ns ticks since start
    n: u64,     // flow-event counter (advance every STRIDE)
    exited: bool,
}

impl ClockFix {
    fn write_ksystime(mem: &mut Memory, off: u64, v: u64) {
        let _ = mem.poke(KUSD + off, &(v as u32).to_le_bytes());
        let _ = mem.poke(KUSD + off + 4, &((v >> 32) as u32).to_le_bytes());
        let _ = mem.poke(KUSD + off + 8, &((v >> 32) as u32).to_le_bytes());
    }
}

impl Hook for ClockFix {
    fn interest(&self) -> Interest {
        Interest { flow_all: true, ..Interest::NONE }
    }
    fn on_flow(&mut self, ev: &FlowEvent, _cpu: &mut Cpu, mem: &mut Memory) -> HookAction {
        if matches!(ev.kind, FlowKind::Call) && ev.to == 0x1_4293_68e7 {
            self.exited = true;
        }
        self.n = self.n.wrapping_add(1);
        if self.n % 16 == 0 {
            self.t += 10_000; // ~1ms per 16 control-flow transfers (100ns units)
            let up_100ns = UPTIME_100NS + self.t; // uptime, advancing
            let up_ms = up_100ns / 10_000;
            // InterruptTime (uptime, 100ns) and SystemTime (wall clock, advancing).
            Self::write_ksystime(mem, 0x008, up_100ns);
            Self::write_ksystime(mem, 0x014, FIXED_FILETIME + self.t);
            // TickCountQuad so that (quad * MULT) >> 24 == up_ms.
            let quad = ((up_ms as u128) << 24) / TICK_MULT;
            Self::write_ksystime(mem, 0x320, quad as u64);
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
    let opts = DetonateOptions { max_steps, export_reference, ..Default::default() };
    let hook = Box::new(ClockFix { t: 0, n: 0, exited: false });
    match detonate_with(&pe, &opts, Some(hook)) {
        Ok(r) => {
            println!("stop={:?} steps={} rip={:#x}", r.stop, r.steps, r.rip);
            println!("oep={:?}", r.observed.oep);
            let past_bail = r.rip != 0x1_4293_68e7 && r.steps > 77_000_000;
            println!(
                "reached ExitProcess@0x1429368e7 as before? {}",
                if r.rip == 0x1_4293_68e7 { "YES (still bails)" } else { "NO — ran past it!" }
            );
            let _ = past_bail;
        }
        Err(e) => println!("detonate error: {e:?}"),
    }
}
