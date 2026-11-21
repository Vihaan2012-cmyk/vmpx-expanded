//! Route (b): detonate the self-protected VMProtect binary in the contained
//! emulator with structural VM discovery + observation, and report the concrete
//! dispatcher/handler VAs the run saw. One such concrete handler VA seeds the
//! static Stage-2 handler-body lift (handlerbody.rs).
//! usage: vmp_observe <exe-or-rvamap> [max_steps]
use std::collections::BTreeSet;

use vmpx_emulate::{
    detonate, export_reference_from_imports, DetonateOptions, SysDllDb, DEFAULT_SYSTEM_MODULES,
};
use vmpx_pe::PeImage;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let bytes = std::fs::read(&a[1]).expect("image");
    let max_steps: u64 = a.get(2).and_then(|v| v.parse().ok()).unwrap_or(500_000_000);
    let pe = PeImage::parse(&bytes).unwrap();

    // Let a protector that walks export tables find real names.
    let mut export_reference = export_reference_from_imports(&pe);
    let db = SysDllDb::new();
    let mut mods: BTreeSet<String> =
        DEFAULT_SYSTEM_MODULES.iter().map(|m| (*m).to_owned()).collect();
    mods.extend(export_reference.keys().cloned());
    for m in mods {
        if let Some(n) = db.exports(&m) {
            export_reference.insert(m, n);
        }
    }

    // VMPX_NO_DISCOVER=1 drops the per-flow VM-observation hook for a fast drive
    // (find/fix blockers quickly); re-enable discovery once it runs to observe.
    let discover = std::env::var_os("VMPX_NO_DISCOVER").is_none();
    let opts = DetonateOptions {
        max_steps,
        vm_discover: discover,
        export_reference,
        dump: false,
        ..Default::default()
    };

    match detonate(&pe, &opts) {
        Ok(r) => {
            println!("stop={:?} steps={} rip={:#x}", r.stop, r.steps, r.rip);
            println!("oep={:?}", r.observed.oep);
            println!("discovered VMs: {}", r.discovered.len());
            for d in r.discovered.iter().take(20) {
                println!("  {d:?}");
            }
            if let Some(vm) = &r.vm {
                println!("dispatcher={:?}", vm.dispatcher);
                println!("handlers observed (dispatch order): {}", vm.handlers.len());
                let mut uniq: BTreeSet<u64> = BTreeSet::new();
                for h in &vm.handlers {
                    uniq.insert(*h);
                }
                println!("unique handler VAs: {}", uniq.len());
                println!("observed VM edges (instance/from_vip/to_vip/handler/count): {}", vm.edges.len());
                // Optional export dir (arg 3): full data for offline logic reconstruction.
                if let Some(dir) = a.get(3) {
                    let base = PeImage::parse(&bytes).unwrap().optional.image_base;
                    let _ = base;
                    let mut hs = String::new();
                    for h in &uniq {
                        hs.push_str(&format!("{h:#x}\n"));
                    }
                    std::fs::write(format!("{dir}/handlers.txt"), hs).unwrap();
                    let mut es = String::new();
                    for e in &vm.edges {
                        es.push_str(&format!(
                            "{:#x} {:#x} {:#x} {:#x} {:#x} {}\n",
                            e.instance, e.from_vip, e.to_vip, e.handler_va, e.rolling_key, e.dispatch_count
                        ));
                    }
                    std::fs::write(format!("{dir}/edges.txt"), es).unwrap();
                    // Full per-edge decode-instant state: instance from_vip handler_va
                    // tail_va then 16 GPRs (RAX..R15). Seeds codec recovery + key-reg ID.
                    let mut ws = String::new();
                    for e in &vm.edges {
                        ws.push_str(&format!(
                            "{:#x} {:#x} {:#x} {:#x}",
                            e.instance, e.from_vip, e.handler_va, e.tail_va
                        ));
                        for r in &e.regs {
                            ws.push_str(&format!(" {r:#x}"));
                        }
                        ws.push('\n');
                    }
                    std::fs::write(format!("{dir}/walk.txt"), ws).unwrap();
                    let mut tr = String::new();
                    for h in &vm.handlers {
                        tr.push_str(&format!("{h:#x}\n"));
                    }
                    std::fs::write(format!("{dir}/trace.txt"), tr).unwrap();
                    println!("exported handlers.txt / edges.txt / trace.txt to {dir}");
                }
            } else {
                println!("no VM observations");
            }
            if !r.audit.is_empty() {
                println!("audit (anti-analysis probes/stops):");
                for (k, v) in &r.audit {
                    println!("  {k}: {v}");
                }
            }
        }
        Err(e) => println!("detonate error: {e:?}"),
    }
}
