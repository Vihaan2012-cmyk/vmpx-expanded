use std::{collections::BTreeMap, fs};
use vmpx_pe::PeImage;
use vmpx_vm::{
    BitWidth, EntryPathRecoveryLimits, Expr, VmDiscoveryLimits, VmEntrySource,
    VmProfileExplorerLimits, analyze_vm_entry_at, discover_vm_entries, recover_entry_paths,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bytes = fs::read(&args[1]).unwrap();
    let pe = PeImage::parse(&bytes).unwrap();
    let discovery = discover_vm_entries(
        &pe,
        VmDiscoveryLimits {
            scan_executable_stubs: false,
            ..Default::default()
        },
    )
    .unwrap();
    let entry = discovery
        .entries
        .iter()
        .find(|e| e.source == VmEntrySource::PeEntryPoint)
        .unwrap();
    let domains = BTreeMap::from([(
        "native_rdx_initial".to_owned(),
        (0..=3).map(|r| Expr::constant(BitWidth::W64, r)).collect(),
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
    let family = recover_entry_paths(&pe, entry, &domains, limits).unwrap();
    let path = family
        .paths
        .iter()
        .find(|p| p.assignments.get("native_rdx_initial") == Some(&0))
        .unwrap();
    let exit = path.native_exit.as_ref().unwrap();
    let cont = vmpx_vm::recover_modulo_image_targets(
        &pe,
        &exit.return_address,
        vmpx_vm::ModuloImageTargetLimits::default(),
    )
    .unwrap();
    let target = cont.targets[0].rva.unwrap();
    println!("reason 0 continuation rva {target:?}");
    match analyze_vm_entry_at(&pe, target, VmDiscoveryLimits::default()) {
        Some(c) => {
            println!(
                "  recognized as VM entry: confidence={} seed={:?} bootstrap={:?}",
                c.confidence, c.seed, c.bootstrap_rva
            );
            // Try recovering it as an entry with the same reason domain.
            let nested_domains = BTreeMap::from([(
                "native_rdx_initial".to_owned(),
                (0..=3).map(|r| Expr::constant(BitWidth::W64, r)).collect(),
            )]);
            match recover_entry_paths(&pe, &c, &nested_domains, limits) {
                Ok(f) => {
                    println!(
                        "  nested domain_complete={} all_exit={} paths={}",
                        f.domain_complete,
                        f.all_paths_reach_vm_exit,
                        f.paths.len()
                    );
                    for p in &f.paths {
                        println!(
                            "    reason {:?} stop={:?} ops={} exit={}",
                            p.assignments.get("native_rdx_initial"),
                            p.chain.stop_reason,
                            p.chain.total_instructions,
                            p.native_exit.is_some()
                        );
                    }
                }
                Err(e) => println!("  nested recovery failed: {e}"),
            }
        }
        None => println!("  not a VM entry"),
    }
}
