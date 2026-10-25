use std::{collections::BTreeMap, fs};
use vmpx_pe::PeImage;
use vmpx_vm::{
    BitWidth, EntryPathRecoveryLimits, Expr, VmDiscoveryLimits, VmEntrySource,
    VmProfileExplorerLimits, X64NativeRegister, discover_vm_entries, recover_entry_paths,
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
    for src in [VmEntrySource::PeEntryPoint, VmEntrySource::TlsCallback] {
        let Some(entry) = discovery.entries.iter().find(|e| e.source == src) else {
            continue;
        };
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
        let Ok(family) = recover_entry_paths(&pe, entry, &domains, limits) else {
            continue;
        };
        for path in &family.paths {
            let Some(exit) = path.native_exit.as_ref() else {
                continue;
            };
            let rax = exit.registers.get(&X64NativeRegister::Rax);
            println!(
                "{src:?} reason={:?} RAX={:?}",
                path.assignments.get("native_rdx_initial"),
                rax.map(|v| format!("{:?}", v.kind)
                    .chars()
                    .take(120)
                    .collect::<String>())
            );
        }
    }
}
