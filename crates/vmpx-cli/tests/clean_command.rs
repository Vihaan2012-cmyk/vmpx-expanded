use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

fn minimal_pe64() -> Vec<u8> {
    let mut data = vec![0_u8; 0x400];
    data[0..2].copy_from_slice(&0x5A4D_u16.to_le_bytes());
    data[0x3C..0x40].copy_from_slice(&0x80_u32.to_le_bytes());
    data[0x80..0x84].copy_from_slice(&0x0000_4550_u32.to_le_bytes());
    let coff = 0x84;
    data[coff..coff + 2].copy_from_slice(&0x8664_u16.to_le_bytes());
    data[coff + 2..coff + 4].copy_from_slice(&1_u16.to_le_bytes());
    data[coff + 16..coff + 18].copy_from_slice(&0xF0_u16.to_le_bytes());
    data[coff + 18..coff + 20].copy_from_slice(&0x2022_u16.to_le_bytes());
    let optional = coff + 20;
    data[optional..optional + 2].copy_from_slice(&0x020B_u16.to_le_bytes());
    data[optional + 16..optional + 20].copy_from_slice(&0x1000_u32.to_le_bytes());
    data[optional + 20..optional + 24].copy_from_slice(&0x1000_u32.to_le_bytes());
    data[optional + 24..optional + 32].copy_from_slice(&0x1_4000_0000_u64.to_le_bytes());
    data[optional + 32..optional + 36].copy_from_slice(&0x1000_u32.to_le_bytes());
    data[optional + 36..optional + 40].copy_from_slice(&0x200_u32.to_le_bytes());
    data[optional + 56..optional + 60].copy_from_slice(&0x2000_u32.to_le_bytes());
    data[optional + 60..optional + 64].copy_from_slice(&0x200_u32.to_le_bytes());
    data[optional + 68..optional + 70].copy_from_slice(&3_u16.to_le_bytes());
    data[optional + 72..optional + 80].copy_from_slice(&0x10_0000_u64.to_le_bytes());
    data[optional + 80..optional + 88].copy_from_slice(&0x1000_u64.to_le_bytes());
    data[optional + 88..optional + 96].copy_from_slice(&0x10_0000_u64.to_le_bytes());
    data[optional + 96..optional + 104].copy_from_slice(&0x1000_u64.to_le_bytes());
    data[optional + 108..optional + 112].copy_from_slice(&16_u32.to_le_bytes());
    let section = optional + 0xF0;
    data[section..section + 5].copy_from_slice(b".text");
    data[section + 8..section + 12].copy_from_slice(&0x180_u32.to_le_bytes());
    data[section + 12..section + 16].copy_from_slice(&0x1000_u32.to_le_bytes());
    data[section + 16..section + 20].copy_from_slice(&0x200_u32.to_le_bytes());
    data[section + 20..section + 24].copy_from_slice(&0x200_u32.to_le_bytes());
    data[section + 36..section + 40].copy_from_slice(&0x6000_0020_u32.to_le_bytes());
    data[0x200] = 0xC3;
    data
}

/// Controlled fixtures are not distributed with the repository. Point
/// `VMPX_FIXTURE_DIR` at a directory containing `systems.original.xpl` and
/// `systems.throughvmpstatic.thatsalsoopeninidaprornthroughmcp.xpl` to run the
/// ignored large-fixture regressions.
fn fixture_directory() -> PathBuf {
    std::env::var_os("VMPX_FIXTURE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("testfilevmp"))
}

fn test_directory(label: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("test clock is after epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("vmpx-{label}-{}-{nonce}", std::process::id()))
}

fn run_clean(input: &Path, output: &Path, report: &Path) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_vmpx"))
        .args(["clean"])
        .arg(input)
        .arg(output)
        .args(["--report"])
        .arg(report)
        .output()
        .expect("vmpx clean process starts")
}

fn run_clean_reconstructed(
    input: &Path,
    output: &Path,
    report: &Path,
    reconstructed: &Path,
) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_vmpx"))
        .args(["clean"])
        .arg(input)
        .arg(output)
        .args(["--report"])
        .arg(report)
        .args(["--reconstructed"])
        .arg(reconstructed)
        .output()
        .expect("vmpx clean process starts")
}

#[test]
fn clean_preserves_verified_unprotected_pe_and_reparses_disk_output() {
    let directory = test_directory("clean-noop");
    fs::create_dir(&directory).expect("create isolated test directory");
    let input = directory.join("input.xpl");
    let output = directory.join("output.xpl");
    let report = directory.join("report.json");
    let bytes = minimal_pe64();
    fs::write(&input, &bytes).expect("write synthetic input");

    let result = run_clean(&input, &output, &report);
    assert!(
        result.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(fs::read(&output).expect("read clean output"), bytes);
    let json: serde_json::Value =
        serde_json::from_slice(&fs::read(&report).expect("read clean report"))
            .expect("valid clean JSON");
    assert_eq!(json["schema_version"], 1);
    assert_eq!(json["ready_to_write"], true);
    assert_eq!(json["validation"]["output_written"], true);
    assert_eq!(json["validation"]["output_reparsed"], true);
    assert_eq!(json["blockers"].as_array().map(Vec::len), Some(0));

    fs::remove_file(input).expect("remove isolated input");
    fs::remove_file(output).expect("remove isolated output");
    fs::remove_file(report).expect("remove isolated report");
    fs::remove_dir(directory).expect("remove isolated test directory");
}

#[test]
#[ignore = "large controlled VMProtect fixture"]
fn clean_protected_fixture_reports_blockers_without_writing_output() {
    let directory = test_directory("clean-protected");
    fs::create_dir(&directory).expect("create isolated test directory");
    let output = directory.join("output.xpl");
    let report = directory.join("report.json");
    let input =
        fixture_directory().join("systems.throughvmpstatic.thatsalsoopeninidaprornthroughmcp.xpl");
    let input = input.as_path();

    let result = run_clean(input, &output, &report);
    assert!(!result.status.success());
    assert!(!output.exists());
    let json: serde_json::Value =
        serde_json::from_slice(&fs::read(&report).expect("read refusal report"))
            .expect("valid refusal JSON");
    assert_eq!(json["ready_to_write"], false);
    assert_eq!(
        json["virtualized_routines"].as_array().map(Vec::len),
        Some(4)
    );
    assert!(
        json["blockers"]
            .as_array()
            .is_some_and(|items| !items.is_empty())
    );
    let native = json["native_continuations"]
        .as_array()
        .and_then(|continuations| {
            continuations
                .iter()
                .find(|continuation| continuation["source_assignments"]["native_rdx_initial"] == 1)
        })
        .and_then(|continuation| continuation["native_analysis"].as_object())
        .expect("reason-1 native continuation analysis");
    assert_eq!(native["status"], "recovered");
    assert_eq!(native["return_stack_offset"], 0);
    assert_eq!(native["calls"][0]["import"]["status"], "unique");
    assert_eq!(
        native["calls"][0]["import"]["import"]["name"],
        "GetModuleHandleA"
    );
    assert!(
        native["calls"][0]["c_strings"]
            .as_array()
            .is_some_and(|strings| strings.iter().any(|string| {
                string["argument_register"] == "RCX" && string["utf8"] == "kernel32.dll"
            }))
    );

    fs::remove_file(report).expect("remove isolated report");
    fs::remove_dir(directory).expect("remove isolated test directory");
}

#[test]
#[ignore = "large controlled VMProtect fixture"]
fn clean_protected_fixture_emits_conventional_layout_image() {
    let directory = test_directory("clean-layout");
    fs::create_dir(&directory).expect("create isolated test directory");
    let output = directory.join("output.xpl");
    let report = directory.join("report.json");
    let reconstructed = directory.join("reconstructed.xpl");
    let input = fixture_directory().join("systems.original.xpl");
    let input = input.as_path();

    let result = run_clean_reconstructed(input, &output, &report, &reconstructed);
    assert!(!result.status.success(), "the proof gate must still refuse");
    assert!(!output.exists(), "the strict output must stay absent");

    let json: serde_json::Value =
        serde_json::from_slice(&fs::read(&report).expect("read refusal report"))
            .expect("valid refusal JSON");
    assert_eq!(json["ready_to_write"], false);
    assert_eq!(json["layout"]["attempted"], true);
    assert_eq!(json["layout"]["sections"], 27);
    assert_eq!(json["layout"]["zero_fill_sections"], 1);
    assert_eq!(json["layout"]["loader_valid"], true);
    assert_eq!(json["validation"]["output_reparsed"], true);

    // The recompilation stage must emit real native code for every path whose
    // exit state is fully lowerable: both viable DLL-reason paths, plus the VM
    // entry that reason 0's continuation re-enters.
    assert_eq!(json["recompilation"]["metrics"]["emitted"], 3);
    assert!(
        json["recompilation"]["metrics"]["bytes"]
            .as_u64()
            .is_some_and(|bytes| bytes > 512),
        "emitted code must be more than a stub"
    );

    let bytes = fs::read(&reconstructed).expect("reconstructed image exists");
    assert_eq!(
        u64::try_from(bytes.len()).expect("image length fits"),
        json["layout"]["file_size"]
            .as_u64()
            .expect("layout file size")
    );
    assert!(
        bytes.len() < 253_702_144,
        "layout must be smaller than the RVA image"
    );

    // The reconstructed image must reparse and keep the protector entry intact.
    let check = Command::new(env!("CARGO_BIN_EXE_vmpx"))
        .args(["verify"])
        .arg(&reconstructed)
        .output()
        .expect("verify starts");
    assert!(
        check.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&check.stderr)
    );

    fs::remove_file(reconstructed).expect("remove reconstructed image");
    fs::remove_file(report).expect("remove isolated report");
    fs::remove_dir(directory).expect("remove isolated test directory");
}

/// The strongest available correctness statement for the layout stage: the
/// conventional image must map into memory byte-for-byte identically to the
/// RVA-layout analysis image produced by unpacking, which is itself verified
/// byte-identical to the external VMPStatic reference. Only the raw-layout
/// section fields may differ.
#[test]
#[ignore = "large controlled VMProtect fixture"]
fn conventional_image_maps_identically_to_the_analysis_image() {
    let directory = test_directory("clean-map");
    fs::create_dir(&directory).expect("create isolated test directory");
    let output = directory.join("output.xpl");
    let report = directory.join("report.json");
    let reconstructed = directory.join("reconstructed.xpl");
    let mapped = directory.join("mapped.xpl");
    let input = fixture_directory().join("systems.original.xpl");
    let input = input.as_path();

    let unpack = Command::new(env!("CARGO_BIN_EXE_vmpx"))
        .args(["unpack"])
        .arg(input)
        .arg(&mapped)
        .output()
        .expect("unpack starts");
    assert!(
        unpack.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&unpack.stderr)
    );

    let result = run_clean_reconstructed(input, &output, &report, &reconstructed);
    assert!(!result.status.success());

    fn loaded_image(bytes: &[u8]) -> Vec<u8> {
        let pe = vmpx_pe::PeImage::parse(bytes).expect("valid PE");
        let mut image = vec![0_u8; pe.optional.size_of_image as usize];
        let headers = pe.optional.size_of_headers as usize;
        image[..headers].copy_from_slice(&bytes[..headers]);
        for section in &pe.sections {
            let span = section.mapped_span() as usize;
            let start = section.virtual_address.0 as usize;
            let mut chunk = vec![0_u8; span];
            if section.size_of_raw_data != 0 {
                let raw = section.pointer_to_raw_data.0 as usize;
                let available = (section.size_of_raw_data as usize).min(bytes.len() - raw);
                chunk[..available].copy_from_slice(&bytes[raw..raw + available]);
            }
            image[start..start + span].copy_from_slice(&chunk);
        }
        // SizeOfRawData and PointerToRawData are loader metadata, not content.
        let table = pe.nt_headers_offset.0 as usize
            + 4
            + 20
            + usize::try_from(pe.coff.size_of_optional_header).expect("optional header");
        for index in 0..pe.sections.len() {
            let base = table + index * 40;
            image[base + 16..base + 24].fill(0);
        }
        image
    }

    let mapped_bytes = fs::read(&mapped).expect("read analysis image");
    let reconstructed_bytes = fs::read(&reconstructed).expect("read conventional image");
    assert_eq!(
        loaded_image(&mapped_bytes),
        loaded_image(&reconstructed_bytes),
        "conventional layout must preserve the loaded memory image exactly"
    );

    fs::remove_file(mapped).expect("remove analysis image");
    fs::remove_file(reconstructed).expect("remove conventional image");
    fs::remove_file(report).expect("remove report");
    fs::remove_dir(directory).expect("remove directory");
}
