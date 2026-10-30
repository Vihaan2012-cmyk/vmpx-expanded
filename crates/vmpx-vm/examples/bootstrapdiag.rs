use std::fs;
use vmpx_disasm::DirectJumpTraceLimits;
use vmpx_pe::{PeImage, Rva};
use vmpx_vm::{VmDiscoveryLimits, analyze_vm_entry_at, recover_bootstrap_dispatch};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bytes = fs::read(&args[1]).unwrap();
    let pe = PeImage::parse(&bytes).unwrap();
    let target = u32::from_str_radix(args[2].trim_start_matches("0x"), 16).unwrap();
    let candidate = analyze_vm_entry_at(&pe, Rva(target), VmDiscoveryLimits::default()).unwrap();
    let (report, recovered) =
        recover_bootstrap_dispatch(&pe, &candidate, DirectJumpTraceLimits::default()).unwrap();
    println!(
        "candidates={} rejected={}",
        report.candidates.len(),
        report.rejected_role_assignments
    );
    println!("recovered: {}", if recovered.is_some() { "YES" } else { "NO" });
    for c in &report.candidates {
        println!(
            "  VIP={} key={} handler={} dir={:?} anchor={} vip0={} encoded_load={}",
            c.vip_register,
            c.rolling_key_register,
            c.handler_register,
            c.bytecode_direction,
            c.handler_anchor,
            c.initial_vip,
            c.encoded_load,
        );
    }
    for d in &report.diagnostics {
        println!("  diag: {d}");
    }
    for s in report.rejection_samples.iter().take(6) {
        println!("  reject: {s}");
    }
}
