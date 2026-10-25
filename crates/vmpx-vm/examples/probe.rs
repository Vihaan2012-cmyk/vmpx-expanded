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
        VmDiscoveryLimits {
            scan_executable_stubs: false,
            ..Default::default()
        },
    )
    .expect("discovery");
    for e in &discovery.entries {
        println!("ENTRY {:?} rva={:?} conf={}", e.source, e.rva, e.confidence);
    }
    let entry = discovery
        .entries
        .iter()
        .find(|e| e.source == VmEntrySource::PeEntryPoint)
        .expect("entry");
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
    let family = recover_entry_paths(&pe, entry, &domains, limits).expect("paths");
    for path in &family.paths {
        let reason = path.assignments["native_rdx_initial"];
        println!(
            "\n=== reason {reason} stop {:?} segs {}",
            path.chain.stop_reason,
            path.chain.segments.len()
        );
        for (i, seg) in path.chain.segments.iter().enumerate() {
            let Some(sym) = &seg.symbolic else { continue };
            if sym.native_calls.is_empty() && sym.external_writes.is_empty() {
                continue;
            }
            println!(
                " seg {i} calls={} writes={}",
                sym.native_calls.len(),
                sym.external_writes.len()
            );
            for c in &sym.native_calls {
                println!(
                    "   call target={:?} res={:?}",
                    c.invocation.target.value.kind,
                    c.import_resolution
                        .as_ref()
                        .map(|r| format!("{r:?}"))
                        .map(|s| s.chars().take(160).collect::<String>())
                );
            }
            for w in &sym.external_writes {
                println!("   write addr={:?}", w.address.kind);
            }
        }
        if let Some(exit) = &path.native_exit {
            println!(" exit ret={:?}", exit.return_address.kind);
            for (r, v) in &exit.registers {
                if format!("{r:?}") == "Rax" {
                    println!("   RAX={:?}", v.kind);
                }
            }
        }
    }
}
