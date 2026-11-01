//! SPIKE driver: concolic loop over one recovered VM export. OBSERVED TIER ONLY.
//!
//! usage: concolic_loop IMAGE EXPORT STUB_RVA SLOT_RVA [ARG] [BUDGET_STEPS] [MAX_ITERS]
//!   IMAGE      an unpacked protected DLL (e.g. prog3_vm_interp_dll.virt.dll)
//!   EXPORT     export name, e.g. Vm_Factorial
//!   STUB_RVA   the `push seed; call wrapper` VM entry stub the export jumps to (0x22810)
//!   SLOT_RVA   image global the VM loads its runtime context pointer from (0x58900 for prog3 virt)
use vmpx_concolic::ctx::detect_ctx_page;
use vmpx_concolic::engine::{Class, Engine, Input};
use vmpx_concolic::pretty::show;
use vmpx_pe::PeImage;

fn rva(s: &str) -> u64 {
    u64::from_str_radix(s.trim_start_matches("0x"), 16).expect("hex")
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 5 {
        eprintln!("usage: concolic_loop IMAGE EXPORT STUB_RVA SLOT_RVA [ARG] [BUDGET_STEPS] [MAX_ITERS]");
        std::process::exit(2);
    }
    let bytes = std::fs::read(&a[1]).expect("read image");
    let pe = PeImage::parse(&bytes).expect("parse");
    let arg: u64 = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(5);
    let budget: u64 = a.get(6).and_then(|s| s.parse().ok()).unwrap_or(3_000_000);
    let max_iters: usize = a.get(7).and_then(|s| s.parse().ok()).unwrap_or(8);

    println!("== SPIKE (observed tier, contained emulator, z3 path conditions) ==");
    let ctx = detect_ctx_page(&pe).expect("init").expect("init allocated no context page");
    println!("[env assumption] runtime VM-context block = init-allocated page {ctx:#x}; image global {:#x} seeded with it", rva(&a[4]));
    let t = std::time::Instant::now();
    let eng = Engine::new(&pe, &a[2], rva(&a[3]) as u32, rva(&a[4]), ctx, budget).expect("engine");
    println!(
        "recovered CFG: {} blocks, {} branch edges, complete={}; setup {:?}",
        eng.rec.cfg.blocks.len(),
        eng.static_edges.len(),
        eng.rec.cfg.is_complete(),
        t.elapsed()
    );
    for (f, to) in &eng.static_edges {
        println!("   static edge {f:#x} -> {to:#x}");
    }
    let t = std::time::Instant::now();
    let rep = eng.explore(Input { arg, cells: Default::default() }, max_iters).expect("explore");
    println!("\nsolver: {}   total {:?}", rep.solver, t.elapsed());
    for r in &rep.iters {
        println!("\niter {}: input {{{}}}", r.iter, r.input);
        println!("   stop={} steps={}  visits={:x?}", r.stop, r.steps, r.visits);
        for (f, to) in &r.new_edges {
            println!("   NEW branch edge {f:#x} -> {to:#x}");
        }
        println!("   coverage {}/{} static branch edges   {}", r.covered, r.total, r.note);
    }
    println!("\nattempts:");
    for at in &rep.attempts {
        let what = match &at.class {
            Class::EnvGated(v) => format!("ENV-GATED (selector reads machine state: {} leaf(s); not input-steerable)", v.len()),
            Class::Unlowerable(v) => format!("UNLOWERABLE {v:?}"),
            Class::Unsat => "UNSAT with the observed prefix".to_owned(),
            Class::Solved(i) => format!("z3 SAT -> input {{{}}}  re-run confirmed={:?}", i.describe(), at.confirmed),
            Class::SolverGaveUp(w) => format!("solver gave up: {w}"),
        };
        println!("   iter {} edge {:#x}->{:#x}: {what}", at.iter, at.edge.0, at.edge.1);
    }
    let cov: Vec<_> = rep.observed.iter().collect();
    println!("\nfinal observed-tier coverage: {}/{} static branch edges", cov.len(), rep.static_edges.len());
    let missing: Vec<_> = rep.static_edges.difference(&rep.observed).collect();
    println!("never observed: {:x?}", missing);
    let extra: Vec<_> = rep.observed.difference(&rep.static_edges).collect();
    println!("observed but NOT in recovered CFG (static/dynamic divergence): {:x?}", extra);
    match eng.bridge(&rep) {
        Ok(s) => println!("\nbridge: {s}"),
        Err(e) => println!("\nbridge error: {e}"),
    }
    let _ = show;
}
