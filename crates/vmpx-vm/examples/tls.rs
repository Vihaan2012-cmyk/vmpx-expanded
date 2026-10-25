use std::{collections::BTreeMap, fs};
use vmpx_pe::PeImage;
use vmpx_vm::{
    BitWidth, EntryPathRecoveryLimits, Expr, VmDiscoveryLimits, VmEntrySource,
    VmProfileExplorerLimits, discover_vm_entries, recover_entry_paths,
};

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    let bytes = fs::read(&args[1]).expect("read");
    let pe = PeImage::parse(&bytes).expect("parse");
    let discovery = discover_vm_entries(
        &pe,
        VmDiscoveryLimits { scan_executable_stubs: false, ..Default::default() },
    ).expect("discovery");
    let entry = discovery.entries.iter()
        .find(|e| e.source == VmEntrySource::TlsCallback).expect("tls");
    let domains = BTreeMap::from([(
        "native_rdx_initial".to_owned(),
        (0..=3).map(|r| Expr::constant(BitWidth::W64, r)).collect(),
    )]);
    let limits = EntryPathRecoveryLimits {
        chain: VmProfileExplorerLimits {
            max_segments: 32, max_instructions_per_segment: 2048, max_total_instructions: 8192,
            ..Default::default()
        },
        ..Default::default()
    };
    match recover_entry_paths(&pe, entry, &domains, limits) {
        Ok(family) => {
            println!("domain_complete={} all_exit={}", family.domain_complete, family.all_paths_reach_vm_exit);
            for path in &family.paths {
                let r = path.assignments.get("native_rdx_initial").copied().unwrap_or(99);
                println!("reason {r} stop {:?} segs {} ops {}", path.chain.stop_reason,
                    path.chain.segments.len(), path.chain.total_instructions);
                if let Some(e) = &path.native_exit {
                    println!("   RAX={:?}", e.registers.get(&vmpx_vm::X64NativeRegister::Rax).map(|v| format!("{:?}", v.kind)));
                    println!("   ret={:?}", format!("{:?}", e.return_address.kind).chars().take(200).collect::<String>());
                }
                for d in path.chain.diagnostics.iter().take(3) { println!("   dg {d}"); }
            }
        }
        Err(e) => println!("FAILED: {e}"),
    }
}
