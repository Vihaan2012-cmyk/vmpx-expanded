//! Restore a post-init `.xpl` snapshot, drive the XPLM plugin lifecycle
//! (XPluginStart/Enable + registered callbacks) in the contained emulator, and
//! report the VM branch edges the observer saw.
//! usage: xpl_dynamic_observe <mapped.bin> <snapshot.bin> [lifecycle_step_budget]
use std::collections::BTreeSet;

use vmpx_emulate::{
    DEFAULT_SYSTEM_MODULES, DetonateOptions, LifecycleOptions, SysDllDb, detonate,
    export_reference_from_imports,
};
use vmpx_pe::PeImage;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mapped = std::fs::read(&a[1]).expect("mapped image");
    let snap = std::path::PathBuf::from(&a[2]);
    let budget: u64 = a.get(3).and_then(|v| v.parse().ok()).unwrap_or(5_000_000_000);
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
        start_from_snapshot: Some(snap),
        lifecycle: Some(LifecycleOptions { max_steps: budget, ..Default::default() }),
        xplm: true,
        vm_discover: true,
        export_reference,
        dump: false,
        ..Default::default()
    };

    match detonate(&pe, &opts) {
        Ok(r) => {
            println!("restored + lifecycle: stop={:?} total_steps={}", r.stop, r.steps);
            println!("discovered VMs: {}", r.discovered.len());
            if let Some(lc) = &r.lifecycle {
                println!(
                    "lifecycle: name={:?} sig={:?} skipped={:?} calls={}",
                    lc.plugin_name,
                    lc.plugin_sig,
                    lc.skipped,
                    lc.calls.len()
                );
                for c in &lc.calls {
                    println!(
                        "  {:>18} stop={:?} steps={} ret={:#x} vm_after={:?}",
                        c.label, c.stop, c.steps, c.ret, c.vm_after
                    );
                }
            }
            if let Some(vm) = &r.vm {
                println!("observed VM branch edges: {}", vm.branch_targets.len());
                for (from, to) in vm.branch_targets.iter().take(40) {
                    println!("  {from:#x} -> {to:#x}");
                }
            } else {
                println!("no VM observations");
            }
        }
        Err(e) => println!("detonate error: {e:?}"),
    }
}
