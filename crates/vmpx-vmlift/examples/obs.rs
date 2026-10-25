use std::{collections::BTreeMap, fs};
use vmpx_pe::PeImage;
use vmpx_vm::{
    BitWidth, EntryPathRecoveryLimits, Expr, VmDiscoveryLimits, VmEntrySource,
    VmProfileExplorerLimits, discover_vm_entries, recover_entry_paths, symbolic_pointer_offset,
};
use vmpx_vmlift::audit_entry_effect_liveness;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bytes = fs::read(&args[1]).unwrap();
    let pe = PeImage::parse(&bytes).unwrap();
    let discovery = discover_vm_entries(&pe, VmDiscoveryLimits { scan_executable_stubs: false, ..Default::default() }).unwrap();
    let entry = discovery.entries.iter().find(|e| e.source == VmEntrySource::PeEntryPoint).unwrap();
    let domains = BTreeMap::from([(
        "native_rdx_initial".to_owned(),
        (0..=3).map(|r| Expr::constant(BitWidth::W64, r)).collect(),
    )]);
    let limits = EntryPathRecoveryLimits {
        chain: VmProfileExplorerLimits { max_segments: 32, max_instructions_per_segment: 2048,
            max_total_instructions: 8192, ..Default::default() },
        ..Default::default()
    };
    let family = recover_entry_paths(&pe, entry, &domains, limits).unwrap();
    let audit = audit_entry_effect_liveness(&family);
    for (path, report) in family.paths.iter().zip(&audit.paths) {
        if report.observable_writes == 0 { continue }
        println!("reason {:?}: observable_writes={}", path.assignments, report.observable_writes);
        let mut reports: Vec<&vmpx_vm::SymbolicVmExecutionReport> = vec![&family.entry_profile.symbolic];
        for s in &path.chain.segments {
            reports.extend(s.symbolic.iter());
            reports.extend(s.boundary_symbolic.iter());
            reports.extend(s.transition_pop.iter());
        }
        for (i, r) in reports.iter().enumerate() {
            for w in &r.external_writes {
                let is_vsp = symbolic_pointer_offset(&w.address, "vsp_initial").is_some();
                let deps = w.address.intrinsic_dependencies();
                let has_fresh = deps.iter().any(|d| d.starts_with("fresh_allocation"));
                if !is_vsp && !has_fresh {
                    println!("  report {i} OBSERVABLE addr_deps={deps:?} value_deps={:?}",
                        w.value.intrinsic_dependencies());
                } else if has_fresh {
                    println!("  report {i} fresh-addr deps={deps:?} (alloc may escape)");
                }
            }
        }
    }
}
