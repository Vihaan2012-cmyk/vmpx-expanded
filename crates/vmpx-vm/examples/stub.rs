use std::{collections::BTreeMap, fs};
use vmpx_pe::{PeImage, Rva};
use vmpx_vm::{
    EntryPathRecoveryLimits, Expr, VmDiscoveryLimits, VmProfileExplorerLimits, analyze_vm_entry_at,
    recover_entry_paths,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bytes = fs::read(&args[1]).unwrap();
    let pe = PeImage::parse(&bytes).unwrap();
    for target in [0xc45fcc2u32, 0xc4ea18d, 0xc718c1e] {
        println!("\n=== stub {target:#x}");
        let rva = Rva(target);
        let Some(candidate) = analyze_vm_entry_at(&pe, rva, VmDiscoveryLimits::default()) else {
            println!("  no VM entry candidate");
            continue;
        };
        println!(
            "  confidence={} seed={:?} call_target={:?} bootstrap={:?}",
            candidate.confidence, candidate.seed, candidate.call_target, candidate.bootstrap_rva
        );
        // No DLL-reason domain here: a native call enters the VM with all registers symbolic.
        let domains: BTreeMap<String, Vec<Expr>> = BTreeMap::from([(
            "native_rdx_initial".to_owned(),
            vec![Expr::register(vmpx_vm::BitWidth::W64, "native_rdx_initial")],
        )]);
        let limits = EntryPathRecoveryLimits {
            chain: VmProfileExplorerLimits {
                max_segments: 32,
                max_instructions_per_segment: 2048,
                max_total_instructions: 8192,
                ..Default::default()
            },
            ..Default::default()
        };
        match recover_entry_paths(&pe, &candidate, &domains, limits) {
            Ok(family) => {
                println!(
                    "  domain_complete={} all_exit={} paths={}",
                    family.domain_complete,
                    family.all_paths_reach_vm_exit,
                    family.paths.len()
                );
                for path in &family.paths {
                    println!(
                        "    stop={:?} segs={} ops={} native_exit={}",
                        path.chain.stop_reason,
                        path.chain.segments.len(),
                        path.chain.total_instructions,
                        path.native_exit.is_some()
                    );
                    for d in path.chain.diagnostics.iter().take(3) {
                        println!("       dg {d}");
                    }
                }
            }
            Err(e) => println!("  recovery failed: {e}"),
        }
    }
}
