use std::{collections::BTreeSet, fs};

use vmpx_disasm::{
    find_ip_relative_references_resync, find_ip_relative_references_resync_in_ranges,
};
use vmpx_pe::{IMAGE_DIRECTORY_ENTRY_IAT, PeImage, Rva};

#[test]
#[ignore = "large controlled mixed code/data fixture"]
fn resynchronizes_to_ida_verified_iat_loads() {
    let bytes = fs::read(
        r"D:\vmp\testfilevmp\systems.throughvmpstatic.thatsalsoopeninidaprornthroughmcp.xpl",
    )
    .expect("controlled mapped-image fixture must exist");
    let pe = PeImage::parse(&bytes).expect("controlled fixture must parse");
    let report = pe.report();
    let slots: BTreeSet<_> = report
        .imports
        .iter()
        .flat_map(|module| module.entries.iter())
        .map(|entry| entry.iat_rva)
        .collect();
    let iat = pe
        .directory(IMAGE_DIRECTORY_ENTRY_IAT)
        .expect("fixture IAT");
    let excluded_end = iat.address.checked_add(iat.size).expect("bounded IAT");
    let scan = find_ip_relative_references_resync(
        &pe,
        &slots,
        &[(Rva(iat.address), Rva(excluded_end))],
        30_000_000,
    )
    .expect("bounded resynchronizing scan must succeed");

    assert!(!scan.resource_limited);
    assert_eq!(
        scan.references
            .iter()
            .map(|reference| reference.instruction_rva)
            .collect::<BTreeSet<_>>(),
        BTreeSet::from([
            Rva(0xEFD_382A),
            Rva(0xEFD_3D2C),
            Rva(0xEFD_3D48),
            Rva(0xEFD_3DED),
            Rva(0xEFD_3F53),
        ])
    );

    let function_ranges: Vec<_> = report
        .exception_functions
        .iter()
        .filter(|function| function.begin < function.end)
        .map(|function| (function.begin, function.end))
        .collect();
    let function_scan =
        find_ip_relative_references_resync_in_ranges(&pe, &slots, &function_ranges, 30_000_000)
            .expect("bounded overlapping exception ranges must resynchronize");
    assert!(!function_scan.resource_limited);
    let recovered: BTreeSet<_> = function_scan
        .references
        .iter()
        .map(|reference| reference.instruction_rva)
        .collect();
    assert!(recovered.contains(&Rva(0xEFD_382A)));
    assert!(recovered.contains(&Rva(0xEFD_3D48)));
    assert!(recovered.contains(&Rva(0xEFD_3DED)));
}
