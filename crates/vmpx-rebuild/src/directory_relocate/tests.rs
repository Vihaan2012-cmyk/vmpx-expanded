use super::*;

/// Builds a minimal PE32+ DLL whose export directory lives in a section named
/// `.vmp3`, which is the shape this module exists for.
fn exporting_image() -> Vec<u8> {
    const FILE_ALIGNMENT: u32 = 0x200;
    const SECTION_ALIGNMENT: u32 = 0x1000;
    const HEADERS: usize = 0x400;
    const TEXT_RVA: u32 = 0x1000;
    const VMP_RVA: u32 = 0x2000;
    const VMP_RAW: usize = 0x600;

    // Export payload, laid out from VMP_RVA: header, functions, names,
    // ordinals, then the strings.
    let names: [&[u8]; 2] = [b"XPluginStart", b"XPluginStop"];
    let functions: [u32; 2] = [TEXT_RVA, TEXT_RVA + 0x10];
    let functions_rva = VMP_RVA + 40;
    let names_rva = functions_rva + 8;
    let ordinals_rva = names_rva + 8;
    let strings_rva = ordinals_rva + 4;

    let mut strings = Vec::new();
    strings.extend_from_slice(b"probe.dll\0");
    let mut name_rvas = Vec::new();
    for name in names {
        name_rvas.push(strings_rva + u32::try_from(strings.len()).expect("small"));
        strings.extend_from_slice(name);
        strings.push(0);
    }

    let mut payload = Vec::new();
    payload.extend_from_slice(&0u32.to_le_bytes()); // Characteristics
    payload.extend_from_slice(&0x1234_5678u32.to_le_bytes()); // TimeDateStamp
    payload.extend_from_slice(&0u16.to_le_bytes()); // MajorVersion
    payload.extend_from_slice(&0u16.to_le_bytes()); // MinorVersion
    payload.extend_from_slice(&strings_rva.to_le_bytes()); // Name
    payload.extend_from_slice(&1u32.to_le_bytes()); // Base
    payload.extend_from_slice(&2u32.to_le_bytes()); // NumberOfFunctions
    payload.extend_from_slice(&2u32.to_le_bytes()); // NumberOfNames
    payload.extend_from_slice(&functions_rva.to_le_bytes());
    payload.extend_from_slice(&names_rva.to_le_bytes());
    payload.extend_from_slice(&ordinals_rva.to_le_bytes());
    for function in functions {
        payload.extend_from_slice(&function.to_le_bytes());
    }
    for rva in &name_rvas {
        payload.extend_from_slice(&rva.to_le_bytes());
    }
    for ordinal in [0u16, 1u16] {
        payload.extend_from_slice(&ordinal.to_le_bytes());
    }
    payload.extend_from_slice(&strings);
    let export_size = u32::try_from(payload.len()).expect("small export directory");

    let mut image = vec![0u8; HEADERS];
    image[0] = b'M';
    image[1] = b'Z';
    let nt = 0x80usize;
    image[0x3c..0x40].copy_from_slice(&u32::try_from(nt).expect("small").to_le_bytes());
    image[nt..nt + 4].copy_from_slice(b"PE\0\0");
    // COFF: machine, sections, ..., SizeOfOptionalHeader, Characteristics.
    image[nt + 4..nt + 6].copy_from_slice(&0x8664u16.to_le_bytes());
    image[nt + 6..nt + 8].copy_from_slice(&3u16.to_le_bytes());
    image[nt + 20..nt + 22].copy_from_slice(&240u16.to_le_bytes());
    image[nt + 22..nt + 24].copy_from_slice(&0x2022u16.to_le_bytes()); // DLL | EXECUTABLE
    let opt = nt + 24;
    image[opt..opt + 2].copy_from_slice(&0x20bu16.to_le_bytes()); // PE32+
    image[opt + 16..opt + 20].copy_from_slice(&TEXT_RVA.to_le_bytes()); // entry point
    image[opt + 24..opt + 32].copy_from_slice(&0x1_8000_0000u64.to_le_bytes()); // ImageBase
    image[opt + 32..opt + 36].copy_from_slice(&SECTION_ALIGNMENT.to_le_bytes());
    image[opt + 36..opt + 40].copy_from_slice(&FILE_ALIGNMENT.to_le_bytes());
    image[opt + 40..opt + 42].copy_from_slice(&6u16.to_le_bytes()); // MajorOSVersion
    image[opt + 56..opt + 60].copy_from_slice(&0x5000u32.to_le_bytes()); // SizeOfImage
    image[opt + 60..opt + 64]
        .copy_from_slice(&u32::try_from(HEADERS).expect("small").to_le_bytes());
    image[opt + 68..opt + 70].copy_from_slice(&2u16.to_le_bytes()); // Subsystem
    image[opt + 108..opt + 112].copy_from_slice(&16u32.to_le_bytes()); // NumberOfRvaAndSizes
    let dirs = opt + 112;
    image[dirs..dirs + 4].copy_from_slice(&VMP_RVA.to_le_bytes());
    image[dirs + 4..dirs + 8].copy_from_slice(&export_size.to_le_bytes());

    // Section table: `.text`, `.vmp3` (holding the exports), and one free slot
    // so `append_section` has header slack to use.
    let table = dirs + 16 * 8;
    let mut put = |index: usize, name: &[u8], rva: u32, size: u32, raw: u32, chars: u32| {
        let at = table + index * 40;
        image[at..at + name.len()].copy_from_slice(name);
        image[at + 8..at + 12].copy_from_slice(&size.to_le_bytes());
        image[at + 12..at + 16].copy_from_slice(&rva.to_le_bytes());
        image[at + 16..at + 20].copy_from_slice(&size.to_le_bytes());
        image[at + 20..at + 24].copy_from_slice(&raw.to_le_bytes());
        image[at + 36..at + 40].copy_from_slice(&chars.to_le_bytes());
    };
    put(0, b".text", TEXT_RVA, 0x200, 0x400, 0x6000_0020);
    put(
        1,
        b".vmp3",
        VMP_RVA,
        u32::try_from(VMP_RAW).expect("small"),
        0x600,
        0x4000_0040,
    );

    image.resize(HEADERS + 0x200 + VMP_RAW, 0);
    image[HEADERS + 0x200..HEADERS + 0x200 + payload.len()].copy_from_slice(&payload);
    image
}

#[test]
fn an_export_directory_is_rebuilt_outside_the_protector_section() {
    let image = exporting_image();
    let before = PeImage::parse(&image).expect("probe image parses");
    let original = before.directory(EXPORT_DIRECTORY).cloned().expect("exports");

    let moved = rebuild_export_directory(&image, *b".edata2\0").expect("rebuilt");
    let after = PeImage::parse(&moved.bytes).expect("rebuilt image parses");
    let directory = after.directory(EXPORT_DIRECTORY).expect("exports");

    assert_ne!(directory.address, original.address, "the directory moved");
    assert_eq!(directory.address, moved.section.rva.0);
    // Out of `.vmp3` and into the appended section.
    let owner = after
        .sections
        .iter()
        .find(|section| section.contains_mapped_rva(Rva(directory.address)))
        .expect("the new directory lies in a section");
    assert_eq!(owner.name, ".edata2");

    let report = after.report();
    let exports = report.exports.expect("exports are readable");
    let names: Vec<_> = exports
        .symbols
        .iter()
        .filter_map(|symbol| symbol.name.clone())
        .collect();
    assert_eq!(names, vec!["XPluginStart", "XPluginStop"]);
}

/// The function addresses name code, not the directory, so they must come
/// through a rebuild completely unchanged.
#[test]
fn rebuilding_does_not_disturb_the_exported_addresses() {
    let image = exporting_image();
    let before = PeImage::parse(&image).expect("parses");
    let original: Vec<_> = before
        .report()
        .exports
        .expect("exports")
        .symbols
        .iter()
        .map(|symbol| (symbol.name.clone(), symbol.rva))
        .collect();

    let moved = rebuild_export_directory(&image, *b".edata2\0").expect("rebuilt");
    let after = PeImage::parse(&moved.bytes).expect("parses");
    let rebuilt: Vec<_> = after
        .report()
        .exports
        .expect("exports")
        .symbols
        .iter()
        .map(|symbol| (symbol.name.clone(), symbol.rva))
        .collect();
    assert_eq!(original, rebuilt);
}

/// A forwarder's string lives inside the directory and only means anything at
/// its original address, so it must fail closed rather than be relocated.
#[test]
fn a_forwarded_export_is_refused() {
    let mut image = exporting_image();
    let pe = PeImage::parse(&image).expect("parses");
    let directory = pe.directory(EXPORT_DIRECTORY).cloned().expect("exports");
    let functions_rva = Rva(directory.address + 40);
    let offset = pe
        .rva_to_file_offset(functions_rva)
        .expect("mapped")
        .0 as usize;
    // Point the first export back inside the directory: that is a forwarder.
    image[offset..offset + 4].copy_from_slice(&(directory.address + 8).to_le_bytes());
    assert!(matches!(
        rebuild_export_directory(&image, *b".edata2\0"),
        Err(DirectoryRelocateError::Forwarder { .. })
    ));
}

/// Entries for code inside a removed section would dangle, so the rebuild drops
/// exactly those and keeps the rest in order.
#[test]
fn an_exception_table_drops_only_the_removed_sections_entries() {
    let image = exporting_image();
    // Give the probe an exception directory: two `.text` entries around one
    // `.vmp3` entry, so order and selection are both observable.
    let mut image = image;
    let table_rva = 0x2400u32;
    let (offset, dirs) = {
        let pe = PeImage::parse(&image).expect("parses");
        (
            pe.rva_to_file_offset(Rva(table_rva)).expect("inside .vmp3").0 as usize,
            pe.nt_headers_offset.0 as usize + 24 + 112,
        )
    };
    let entries: [[u32; 3]; 3] = [
        [0x1000, 0x1010, 0x1200],
        [0x2100, 0x2110, 0x2200],
        [0x1020, 0x1030, 0x1210],
    ];
    for (index, entry) in entries.iter().enumerate() {
        for (word, value) in entry.iter().enumerate() {
            let at = offset + index * 12 + word * 4;
            image[at..at + 4].copy_from_slice(&value.to_le_bytes());
        }
    }
    image[dirs + 3 * 8..dirs + 3 * 8 + 4].copy_from_slice(&table_rva.to_le_bytes());
    image[dirs + 3 * 8 + 4..dirs + 3 * 8 + 8].copy_from_slice(&36u32.to_le_bytes());

    let (moved, summary) =
        rebuild_exception_directory(&image, *b".pdata2\0", &[".vmp3"]).expect("rebuilt");
    assert_eq!((summary.kept, summary.dropped), (2, 1));

    let after = PeImage::parse(&moved.bytes).expect("parses");
    let directory = after.directory(3).expect("exception directory");
    assert_eq!(directory.address, moved.section.rva.0);
    assert_eq!(directory.size, 24);
    let at = after
        .rva_to_file_offset(Rva(directory.address))
        .expect("mapped")
        .0 as usize;
    let begin = |index: usize| {
        u32::from_le_bytes(
            moved.bytes[at + index * 12..at + index * 12 + 4]
                .try_into()
                .expect("4 bytes"),
        )
    };
    // The surviving `.text` entries, in their original order.
    assert_eq!((begin(0), begin(1)), (0x1000, 0x1020));
}

/// A kept entry whose unwind info is inside a removed section must fail closed:
/// the table would parse and then fault during an unwind.
#[test]
fn an_entry_whose_unwind_info_is_removed_is_refused() {
    let mut image = exporting_image();
    let table_rva = 0x2400u32;
    let (offset, dirs) = {
        let pe = PeImage::parse(&image).expect("parses");
        (
            pe.rva_to_file_offset(Rva(table_rva)).expect("inside .vmp3").0 as usize,
            pe.nt_headers_offset.0 as usize + 24 + 112,
        )
    };
    for (word, value) in [0x1000u32, 0x1010, 0x2300].iter().enumerate() {
        let at = offset + word * 4;
        image[at..at + 4].copy_from_slice(&value.to_le_bytes());
    }
    image[dirs + 3 * 8..dirs + 3 * 8 + 4].copy_from_slice(&table_rva.to_le_bytes());
    image[dirs + 3 * 8 + 4..dirs + 3 * 8 + 8].copy_from_slice(&12u32.to_le_bytes());
    assert!(matches!(
        rebuild_exception_directory(&image, *b".pdata2\0", &[".vmp3"]),
        Err(DirectoryRelocateError::Malformed(_))
    ));
}
