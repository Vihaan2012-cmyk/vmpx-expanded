use std::{collections::BTreeMap, fs};
use vmpx_pe::PeImage;
use vmpx_vm::{
    BitWidth, EntryPathRecoveryLimits, Expr, VmDiscoveryLimits, VmEntrySource,
    VmProfileExplorerLimits, discover_vm_entries, recover_entry_paths,
};
use vmpx_vmlift::{audit_profile_chain, lift_profile_chain};

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
    for path in &family.paths {
        let reason = path.assignments.get("native_rdx_initial").copied();
        println!("\n=== reason {reason:?} stop={:?}", path.chain.stop_reason);
        let audit = audit_profile_chain(&path.chain);
        println!("  missing capabilities: {:?}", audit.missing_capabilities);
        match lift_profile_chain(&path.chain) {
            Ok(routine) => {
                println!(
                    "  blocks={} ops={} complete_vm_exit={} havoc={}",
                    routine.blocks.len(),
                    routine.semantic_instruction_count,
                    routine.complete_vm_exit,
                    routine.contains_memory_havoc
                );
                let effects: usize = routine.blocks.iter().map(|b| b.effects.len()).sum();
                let ctx: usize = routine.blocks.iter().map(|b| b.context_outputs.len()).sum();
                println!("  total effects={effects} context_outputs={ctx}");
                for (i, b) in routine.blocks.iter().enumerate().take(3) {
                    println!(
                        "    block {i}: ops={} effects={} ctx={} branch={:?}",
                        b.semantic_instruction_count,
                        b.effects.len(),
                        b.context_outputs.len(),
                        b.branch_target
                            .as_ref()
                            .map(|e| format!("{:?}", e.kind).chars().take(60).collect::<String>())
                    );
                }
            }
            Err(e) => println!("  lift failed: {e}"),
        }
    }
}
