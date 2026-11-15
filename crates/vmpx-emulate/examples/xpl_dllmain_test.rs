//! Bare, un-instrumented init-completion test: does DllMain return with a big
//! stack? Lean (WinApiHook only) so the emulator runs at full speed.
//! usage: xpl_dllmain_test <mapped.bin> [max_steps] [stack_kb]
use std::collections::BTreeSet;
use std::time::Instant;

use vmpx_emulate::{
    DEFAULT_SYSTEM_MODULES, DetonateOptions, SysDllDb, detonate, export_reference_from_imports,
};
use vmpx_pe::PeImage;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mapped = std::fs::read(&a[1]).expect("image");
    let max: u64 = a.get(2).and_then(|v| v.parse().ok()).unwrap_or(120_000_000_000);
    let stack_kb: u64 = a.get(3).and_then(|v| v.parse().ok()).unwrap_or(262_144);
    let pe = PeImage::parse(&mapped).unwrap();

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

    let opts = DetonateOptions {
        max_steps: max,
        stack_size: stack_kb << 10,
        export_reference,
        dump: false,
        ..Default::default()
    };
    println!("running DllMain, max_steps={max} stack={stack_kb}KiB ...");
    let t = Instant::now();
    match detonate(&pe, &opts) {
        Ok(r) => {
            let secs = t.elapsed().as_secs_f64();
            println!(
                "stop={:?} steps={} rip={:#x} wall={:.0}s rate={:.0}M/s",
                r.stop,
                r.steps,
                r.rip,
                secs,
                r.steps as f64 / secs / 1e6
            );
            if r.stop == vmpx_emulate::Stop::Reached {
                println!(">>> DllMain RETURNED: init is a BOUNDED sweep, not a divergence.");
            }
        }
        Err(e) => println!("error {e:?}"),
    }
}
