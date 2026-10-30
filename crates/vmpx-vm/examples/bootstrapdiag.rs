// Diagnose bootstrap-dispatch recovery on a real sample.
// usage: bootstrapdiag <file> <entry-rva-hex>
use std::fs;
use vmpx_pe::{PeImage, Rva};
use vmpx_disasm::DirectJumpTraceLimits;
use vmpx_vm::{VmDiscoveryLimits, analyze_vm_entry_at, recover_bootstrap_dispatch};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bytes = fs::read(&args[1]).unwrap();
    let pe = PeImage::parse(&bytes).unwrap();
    let rva = Rva(u32::from_str_radix(args[2].trim_start_matches("0x"), 16).unwrap());
    let entry = analyze_vm_entry_at(&pe, rva, VmDiscoveryLimits::default())
        .expect("no VM entry candidate at rva");
    println!(
        "entry rva={:#x} seed={:#x?} call_target={:#x?} bootstrap_rva={:#x?}",
        entry.rva.0, entry.seed, entry.call_target, entry.bootstrap_rva
    );
    match recover_bootstrap_dispatch(&pe, &entry, DirectJumpTraceLimits::default()) {
        Ok((report, recovery)) => {
            println!("candidates={} rejected={}", report.candidates.len(), report.rejected_role_assignments);
            println!("diagnostics: {:?}", report.diagnostics);
            println!("recovered: {}", if recovery.is_some() { "YES" } else { "no" });
            println!("--- rejection samples (first 30) ---");
            for s in report.rejection_samples.iter().take(30) {
                println!("  {s}");
            }
        }
        Err(e) => println!("ERROR: {e}"),
    }
}
