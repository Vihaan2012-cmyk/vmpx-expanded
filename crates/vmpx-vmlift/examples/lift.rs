use std::{collections::BTreeMap, fs};
use vmpx_pe::PeImage;
use vmpx_vm::{
    BitWidth, EntryPathRecoveryLimits, Expr, VmDiscoveryLimits,
    VmProfileExplorerLimits, discover_vm_entries, recover_entry_paths,
};
use vmpx_vmlift::audit_profile_chain;

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    let bytes = fs::read(&args[1]).expect("read");
    let pe = PeImage::parse(&bytes).expect("parse");
    let discovery = discover_vm_entries(&pe,
        VmDiscoveryLimits { scan_executable_stubs: false, ..Default::default() }).expect("d");
    let domains = BTreeMap::from([(
        "native_rdx_initial".to_owned(),
        (0..=3).map(|r| Expr::constant(BitWidth::W64, r)).collect(),
    )]);
    let limits = EntryPathRecoveryLimits {
        chain: VmProfileExplorerLimits {
            max_segments: 32, max_instructions_per_segment: 2048, max_total_instructions: 8192,
            ..Default::default() },
        ..Default::default()
    };
    for entry in &discovery.entries {
        let Ok(family) = recover_entry_paths(&pe, entry, &domains, limits) else { continue };
        for path in &family.paths {
            let report = audit_profile_chain(&path.chain);
            println!("\n{:?} {:?} ready={} ops={}", entry.source, path.assignments,
                report.ready_for_codegen, report.total_instructions);
            println!("  missing: {:?}", report.missing_capabilities);
            for shape in &report.instruction_shapes {
                println!("   shape {:?} {:?}/{:?} resize={:?} x{}", shape.semantic,
                    shape.operation_width, shape.input_width, shape.resize, shape.count);
            }
            for issue in report.issues.iter().take(6) {
                println!("   issue {:?}", issue.message);
            }
        }
    }
}
