use std::{
    fs,
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

#[test]
fn report_prints_tiers_separately_on_synthetic_pe() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("vmpx-report-{}-{nonce}", std::process::id()));
    fs::create_dir_all(&dir).expect("mkdir");
    let input = dir.join("tiny.dll");
    fs::write(&input, minimal_pe64()).expect("write");

    let out = Command::new(env!("CARGO_BIN_EXE_vmpx"))
        .arg("report")
        .arg(&input)
        .output()
        .expect("run vmpx report");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout);

    let line = |needle: &str| {
        text.lines()
            .find(|l| l.contains(needle))
            .unwrap_or_else(|| panic!("missing line {needle:?} in:\n{text}"))
            .to_owned()
    };
    // Three separate headline numbers, each on its own line.
    assert!(line("N1 CFG-complete").contains("0 / 0"));
    assert!(line("N2 PROVEN").contains("0 / 0"));
    assert!(line("N3 proven-equivalent").contains("not wired"));
    // Experimental tiers are separate and labeled; unrun stages say so.
    assert!(text.contains("EXPERIMENTAL TIERS"));
    assert!(line("serve-RO").contains("not wired"));
    assert!(line("observed-tier").contains("not run"));
    assert!(line("import resolution").contains("not run"));

    let json = Command::new(env!("CARGO_BIN_EXE_vmpx"))
        .args(["report", "--json"])
        .arg(&input)
        .output()
        .expect("run vmpx report --json");
    assert!(json.status.success());
    let value: serde_json::Value = serde_json::from_slice(&json.stdout).expect("json");
    assert_eq!(value["sweep"]["n1_cfg_complete"], 0);
    assert_eq!(value["sweep"]["n2_proven"], 0);
    assert_eq!(value["n3_proven_equivalent"]["status"], "not wired");
}

#[test]
fn report_help_parses() {
    let out = Command::new(env!("CARGO_BIN_EXE_vmpx"))
        .args(["report", "--help"])
        .output()
        .expect("run");
    assert!(out.status.success());
    assert!(String::from_utf8_lossy(&out.stdout).contains("--dynamic"));
}
