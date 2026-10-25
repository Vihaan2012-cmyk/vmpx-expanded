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
    let input = Path::new(
        r"D:\vmp\testfilevmp\systems.throughvmpstatic.thatsalsoopeninidaprornthroughmcp.xpl",
    );

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
