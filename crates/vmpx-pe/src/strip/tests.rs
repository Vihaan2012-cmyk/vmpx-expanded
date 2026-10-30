use super::*;
use crate::{
    FileOffset, IMAGE_DIRECTORY_ENTRY_BASERELOC, IMAGE_DIRECTORY_ENTRY_EXCEPTION,
    IMAGE_DIRECTORY_ENTRY_EXPORT, IMAGE_DIRECTORY_ENTRY_IMPORT, IMAGE_DIRECTORY_ENTRY_SECURITY,
};

const IMAGE_BASE: u64 = 0x1_4000_0000;
const OPTIONAL: usize = 0x98;
const DIRS: usize = OPTIONAL + 112;

fn put(data: &mut [u8], offset: usize, bytes: &[u8]) {
    data[offset..offset + bytes.len()].copy_from_slice(bytes);
}

/// PE32+ with .text 0x1000, .data 0x2000, .vmp0 0x3000, .vmp1 0x4000; each has
/// 0x200 raw bytes filled with a distinct marker.
fn build() -> Vec<u8> {
    let mut data = vec![0_u8; 0x400 + 4 * 0x200];
    put(&mut data, 0, b"MZ");
    put(&mut data, 0x3C, &0x80_u32.to_le_bytes());
    put(&mut data, 0x80, b"PE\0\0");
    put(&mut data, 0x84, &0x8664_u16.to_le_bytes());
    put(&mut data, 0x86, &4_u16.to_le_bytes());
    put(&mut data, 0x94, &0xF0_u16.to_le_bytes());
    put(&mut data, 0x96, &0x2022_u16.to_le_bytes());
    put(&mut data, OPTIONAL, &0x20B_u16.to_le_bytes());
    put(&mut data, OPTIONAL + 4, &0x400_u32.to_le_bytes());
    put(&mut data, OPTIONAL + 8, &0x400_u32.to_le_bytes());
    put(&mut data, OPTIONAL + 16, &0x1000_u32.to_le_bytes());
    put(&mut data, OPTIONAL + 20, &0x1000_u32.to_le_bytes());
    put(&mut data, OPTIONAL + 24, &IMAGE_BASE.to_le_bytes());
    put(&mut data, OPTIONAL + 32, &0x1000_u32.to_le_bytes());
    put(&mut data, OPTIONAL + 36, &0x200_u32.to_le_bytes());
    put(&mut data, OPTIONAL + 56, &0x5000_u32.to_le_bytes());
    put(&mut data, OPTIONAL + 60, &0x400_u32.to_le_bytes());
    put(&mut data, OPTIONAL + 68, &3_u16.to_le_bytes());
    put(&mut data, OPTIONAL + 108, &16_u32.to_le_bytes());
    let sections = [
        (".text", 0x1000_u32, 0x100_u32, 0x6000_0020_u32),
        (".data", 0x2000, 0x300, 0xC000_0040),
        (".vmp0", 0x3000, 0x180, 0x6000_0020),
        (".vmp1", 0x4000, 0x180, 0xC000_0040),
    ];
    for (index, (name, rva, vsize, flags)) in sections.into_iter().enumerate() {
        let header = OPTIONAL + 0xF0 + index * 40;
        let raw = 0x400 + index * 0x200;
        put(&mut data, header, name.as_bytes());
        put(&mut data, header + 8, &vsize.to_le_bytes());
        put(&mut data, header + 12, &rva.to_le_bytes());
        put(&mut data, header + 16, &0x200_u32.to_le_bytes());
        put(
            &mut data,
            header + 20,
            &u32::try_from(raw).unwrap().to_le_bytes(),
        );
        put(&mut data, header + 36, &flags.to_le_bytes());
        data[raw..raw + 0x200].fill(0xA0 + u8::try_from(index).unwrap());
    }
    data
}

fn at(data: &mut [u8], rva: u32, bytes: &[u8]) {
    let offset = PeImage::parse(data)
        .unwrap()
        .rva_to_file_offset(Rva(rva))
        .unwrap()
        .0;
    put(data, usize::try_from(offset).unwrap(), bytes);
}

fn set_directory(data: &mut [u8], index: usize, address: u32, size: u32) {
    put(data, DIRS + index * 8, &address.to_le_bytes());
    put(data, DIRS + index * 8 + 4, &size.to_le_bytes());
}

fn strip(data: &[u8], names: &[&str], drop: &[usize]) -> Result<StrippedPe, StripError> {
    let selectors: Vec<_> = names
        .iter()
        .map(|name| SectionSelector::Name((*name).to_owned()))
        .collect();
    strip_sections(
        data,
        &StripRequest {
            remove: &selectors,
            drop_directories: drop,
        },
    )
}

fn refs(error: StripError) -> Vec<DanglingRef> {
    match error {
        StripError::Dangling(refs) => refs,
        other => panic!("expected dangling refs, got {other}"),
    }
}

#[test]
fn removes_unreferenced_trailing_section() {
    let data = build();
    assert!(
        PeImage::parse(&data)
            .unwrap()
            .report()
            .diagnostics
            .is_empty()
    );
    let result = strip(&data, &[".vmp1"], &[]).unwrap();
    let pe = PeImage::parse(&result.data).unwrap();
    assert_eq!(pe.sections.len(), 3);
    assert_eq!(pe.coff.number_of_sections, 3);
    assert_eq!(pe.optional.size_of_image, 0x4000);
    assert_eq!(result.data.len(), data.len() - 0x200);
    assert_eq!(result.file_bytes_removed, 0x200);
    assert!(pe.report().diagnostics.is_empty());
    let vacated = OPTIONAL + 0xF0 + 3 * 40;
    assert!(result.data[vacated..vacated + 40].iter().all(|b| *b == 0));
    assert_eq!(result.removed[0].name, ".vmp1");
}

#[test]
fn removes_middle_section_keeping_rvas_and_compacting_raw() {
    let data = build();
    let result = strip(&data, &[".vmp0"], &[]).unwrap();
    let pe = PeImage::parse(&result.data).unwrap();
    let names: Vec<_> = pe.sections.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, [".text", ".data", ".vmp1"]);
    assert_eq!(pe.coff.number_of_sections, 3);
    assert_eq!(pe.optional.size_of_image, 0x5000);
    assert_eq!(pe.optional.size_of_code, 0x200);
    assert_eq!(pe.sections[2].virtual_address, Rva(0x4000));
    assert_eq!(pe.sections[2].pointer_to_raw_data, FileOffset(0x800));
    assert_eq!(result.data[0x800..0xA00], [0xA3; 0x200]);
    assert_eq!(result.data.len(), data.len() - 0x200);
    assert!(pe.report().diagnostics.is_empty());
}

#[test]
fn selects_by_index_and_rejects_bad_selection() {
    let data = build();
    let by_index = strip_sections(
        &data,
        &StripRequest {
            remove: &[SectionSelector::Index(3)],
            drop_directories: &[],
        },
    )
    .unwrap();
    assert_eq!(by_index.removed[0].name, ".vmp1");
    assert_eq!(
        strip(&data, &[".nope"], &[]),
        Err(StripError::NoSuchName(".nope".into()))
    );
    assert_eq!(
        strip(&data, &[".text", ".data", ".vmp0", ".vmp1"], &[]),
        Err(StripError::RemovesAllSections)
    );
}

#[test]
fn refuses_entry_point_in_removed_section() {
    let mut data = build();
    put(&mut data, OPTIONAL + 16, &0x3010_u32.to_le_bytes());
    let found = refs(strip(&data, &[".vmp0"], &[]).unwrap_err());
    assert_eq!(found[0].source, "AddressOfEntryPoint");
    assert_eq!(found[0].section, ".vmp0");
    assert_eq!(found[0].target, "RVA 0x3010");
    let message = strip(&data, &[".vmp0"], &[]).unwrap_err().to_string();
    assert!(message.contains("AddressOfEntryPoint"), "{message}");
}

#[test]
fn refuses_import_iat_inside_removed_section_and_accepts_it_when_retained() {
    let mut data = build();
    // descriptor @0x2000, OFT @0x2040, hint/name @0x2080, DLL name @0x20A0.
    let mut descriptor = [0_u8; 20];
    descriptor[0..4].copy_from_slice(&0x2040_u32.to_le_bytes());
    descriptor[12..16].copy_from_slice(&0x20A0_u32.to_le_bytes());
    descriptor[16..20].copy_from_slice(&0x3000_u32.to_le_bytes());
    at(&mut data, 0x2000, &[0; 0x100]);
    at(&mut data, 0x3000, &[0; 0x20]);
    at(&mut data, 0x2000, &descriptor);
    at(&mut data, 0x2040, &0x2080_u64.to_le_bytes());
    at(&mut data, 0x3000, &0x2080_u64.to_le_bytes());
    at(&mut data, 0x2082, b"Sleep\0");
    at(&mut data, 0x20A0, b"kernel32.dll\0");
    set_directory(&mut data, IMAGE_DIRECTORY_ENTRY_IMPORT, 0x2000, 40);
    let found = refs(strip(&data, &[".vmp0"], &[]).unwrap_err());
    assert!(
        found
            .iter()
            .any(|r| r.source == "import kernel32.dll FirstThunk")
    );
    assert!(
        found
            .iter()
            .any(|r| r.source == "import kernel32.dll IAT slot")
    );

    // IAT in .data instead (retained): removing .vmp1 leaves imports intact.
    at(&mut data, 0x3000, &[0; 8]);
    at(&mut data, 0x2010, &0x2060_u32.to_le_bytes());
    at(&mut data, 0x2060, &0x2080_u64.to_le_bytes());
    let ok = strip(&data, &[".vmp1"], &[]).unwrap();
    let pe = PeImage::parse(&ok.data).unwrap();
    assert_eq!(pe.report().imports[0].name, "kernel32.dll");
}

#[test]
fn refuses_absolute_pointer_into_removed_section_and_drops_its_own_fixups() {
    let mut data = build();
    // .data holds two reloc blocks: one fixing .text:0x1008, one fixing .vmp0:0x3000.
    let mut table = Vec::new();
    for (page, entry) in [(0x1000_u32, 0xA008_u16), (0x3000, 0xA000)] {
        table.extend_from_slice(&page.to_le_bytes());
        table.extend_from_slice(&12_u32.to_le_bytes());
        table.extend_from_slice(&entry.to_le_bytes());
        table.extend_from_slice(&0_u16.to_le_bytes());
    }
    at(&mut data, 0x2000, &[0; 0x100]);
    at(&mut data, 0x2000, &table);
    set_directory(&mut data, IMAGE_DIRECTORY_ENTRY_BASERELOC, 0x2000, 24);
    at(&mut data, 0x1008, &(IMAGE_BASE + 0x3010).to_le_bytes());
    let found = refs(strip(&data, &[".vmp0"], &[]).unwrap_err());
    assert_eq!(found[0].source, "relocated pointer at RVA 0x1008");
    assert_eq!(found[0].target, "RVA 0x3010");

    at(&mut data, 0x1008, &(IMAGE_BASE + 0x1020).to_le_bytes());
    let result = strip(&data, &[".vmp0"], &[]).unwrap();
    assert_eq!(result.relocation_entries_dropped, 1);
    let pe = PeImage::parse(&result.data).unwrap();
    let report = pe.report();
    assert_eq!(report.relocations.len(), 1);
    assert_eq!(report.relocations[0].page_rva, Rva(0x1000));
    assert_eq!(
        pe.directory(IMAGE_DIRECTORY_ENTRY_BASERELOC).unwrap().size,
        12
    );
    assert!(report.diagnostics.is_empty());
}

#[test]
fn directory_inside_removed_section_needs_explicit_drop() {
    let mut data = build();
    set_directory(&mut data, IMAGE_DIRECTORY_ENTRY_EXPORT, 0x3000, 0x40);
    assert!(matches!(
        strip(&data, &[".vmp0"], &[]),
        Err(StripError::DirectoryInRemoved { index: 0, .. })
    ));
    assert_eq!(
        strip(&data, &[".vmp0"], &[0, 2]),
        Err(StripError::DropNotInRemoved(2))
    );
    let result = strip(&data, &[".vmp0"], &[0]).unwrap();
    let pe = PeImage::parse(&result.data).unwrap();
    assert!(pe.directory(IMAGE_DIRECTORY_ENTRY_EXPORT).is_none());
    assert_eq!(pe.directories[0].address, 0);

    set_directory(&mut data, IMAGE_DIRECTORY_ENTRY_EXPORT, 0x3170, 0x100);
    assert!(matches!(
        strip(&data, &[".vmp0"], &[0]),
        Err(StripError::DirectoryStraddles { .. })
    ));
}

#[test]
fn drops_runtime_functions_of_removed_code_and_checks_unwind_refs() {
    let mut data = build();
    let mut table = Vec::new();
    for value in [0x1000_u32, 0x1010, 0x2100, 0x3000, 0x3010, 0x3100] {
        table.extend_from_slice(&value.to_le_bytes());
    }
    at(&mut data, 0x2000, &table);
    at(&mut data, 0x2100, &[1, 0, 0, 0]);
    at(&mut data, 0x3100, &[1, 0, 0, 0]);
    set_directory(&mut data, IMAGE_DIRECTORY_ENTRY_EXCEPTION, 0x2000, 24);
    let result = strip(&data, &[".vmp0"], &[]).unwrap();
    assert_eq!(result.runtime_functions_dropped, 1);
    let pe = PeImage::parse(&result.data).unwrap();
    assert_eq!(pe.report().exception_functions.len(), 1);
    assert_eq!(
        pe.directory(IMAGE_DIRECTORY_ENTRY_EXCEPTION).unwrap().size,
        12
    );

    at(&mut data, 0x2008, &0x3100_u32.to_le_bytes());
    let found = refs(strip(&data, &[".vmp0"], &[]).unwrap_err());
    assert_eq!(found[0].source, "runtime function 0x1000 UnwindData");
}

#[test]
fn refuses_signed_images() {
    let mut data = build();
    set_directory(&mut data, IMAGE_DIRECTORY_ENTRY_SECURITY, 0x400, 8);
    assert_eq!(
        strip(&data, &[".vmp1"], &[]),
        Err(StripError::SecurityDirectoryPresent)
    );
}

#[test]
fn recomputes_checksum_when_present() {
    let mut data = build();
    put(&mut data, OPTIONAL + 64, &1_u32.to_le_bytes());
    let result = strip(&data, &[".vmp1"], &[]).unwrap();
    let checksum = u32::from_le_bytes(
        result.data[OPTIONAL + 64..OPTIONAL + 68]
            .try_into()
            .unwrap(),
    );
    assert_eq!(checksum, pe_checksum(&result.data, OPTIONAL + 64));
    assert_ne!(checksum, 1);
}
