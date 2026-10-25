use std::fs;

use vmpx_unpack::{UnpackLimits, unpack_to_mapped_image};

#[test]
#[ignore = "large controlled differential fixture"]
fn reproduces_supplied_vmpstatic_mapped_image_byte_for_byte() {
    let packed = fs::read(r"D:\vmp\testfilevmp\systems.original.xpl")
        .expect("controlled packed fixture must exist");
    let expected = fs::read(
        r"D:\vmp\testfilevmp\systems.throughvmpstatic.thatsalsoopeninidaprornthroughmcp.xpl",
    )
    .expect("controlled mapped-image fixture must exist");
    let actual = unpack_to_mapped_image(&packed, UnpackLimits::default())
        .expect("fixture unpacking must succeed");
    assert_eq!(actual.image, expected);
    assert!(actual.report.in_memory_reparsed);
    assert!(actual.report.warnings.is_empty());
}
