use std::fs;

use vmpx_pe::PeImage;
use vmpx_vm::{VmDiscoveryLimits, VmEntrySource, discover_vm_entries};

#[test]
#[ignore = "large controlled VMProtect analysis fixture"]
fn discovers_loader_reachable_vm_entries_without_sample_offsets() {
    let bytes = fs::read(
        r"D:\vmp\testfilevmp\systems.throughvmpstatic.thatsalsoopeninidaprornthroughmcp.xpl",
    )
    .expect("controlled mapped-image fixture must exist");
    let pe = PeImage::parse(&bytes).expect("controlled fixture must parse");
    let report = discover_vm_entries(
        &pe,
        VmDiscoveryLimits {
            scan_executable_stubs: false,
            ..VmDiscoveryLimits::default()
        },
    )
    .expect("loader-root discovery must succeed");

    assert_eq!(report.entries.len(), 2);
    assert!(report.entries.iter().any(|entry| {
        entry.source == VmEntrySource::PeEntryPoint
            && entry.rva == pe.optional.address_of_entry_point
            && entry.confidence == 100
    }));
    assert!(report.entries.iter().any(|entry| {
        entry.source == VmEntrySource::TlsCallback
            && entry.bootstrap_rva.is_some()
            && entry.confidence == 100
    }));
    assert_eq!(report.instances.len(), 2);
}
