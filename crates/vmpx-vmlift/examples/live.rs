use std::{collections::BTreeMap, fs};
use vmpx_pe::PeImage;
use vmpx_vm::{
    BitWidth, EntryPathRecoveryLimits, Expr, VmDiscoveryLimits, VmProfileExplorerLimits,
    discover_vm_entries, recover_entry_paths,
};
use vmpx_vmlift::audit_entry_effect_liveness;

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    let bytes = fs::read(&args[1]).expect("read");
    let pe = PeImage::parse(&bytes).expect("parse");
    let discovery = discover_vm_entries(
        &pe,
        VmDiscoveryLimits {
            scan_executable_stubs: false,
            ..Default::default()
        },
    )
    .expect("d");
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
    for entry in &discovery.entries {
        println!("\n### {:?} {:?}", entry.source, entry.rva);
        let family = match recover_entry_paths(&pe, entry, &domains, limits) {
            Ok(f) => f,
            Err(e) => {
                println!("  recovery failed: {e}");
                continue;
            }
        };
        let audit = audit_entry_effect_liveness(&family);
        println!(
            "  ready={} domain_complete={} all_exit={}",
            audit.ready_for_native_replacement,
            audit.domain_complete,
            audit.all_paths_reach_vm_exit
        );
        for p in &audit.paths {
            println!(
                "  path {:?} ready={} writes={} calls={} havoc={} nonvol_ok={} ret_caller={}",
                p.assignments,
                p.ready_for_native_replacement,
                p.observable_writes,
                p.observable_calls,
                p.memory_havocs,
                p.nonvolatile_registers_preserved,
                p.returns_to_original_caller
            );
            for i in &p.issues {
                println!("      - {i}");
            }
        }
    }
}
