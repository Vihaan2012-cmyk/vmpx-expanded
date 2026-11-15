// Which half of `image_cannot_reprotect` gives way? That one predicate gates the
// PROVEN image-read serve, and ~73% of sound-mode incompleteness on systems.xpl
// is an unresolved image read, so the answer decides whether N2 has headroom.
//
// Mirrors the real predicates in immutable_image.rs exactly: the import half,
// and the section half's BYTE patterns (syscall/sysenter/int 2Eh, and a
// segment-prefixed TEB/PEB offset) -- not string needles.
//
// Usage: protcheck IMAGE

use vmpx_pe::PeImage;

const SCN_MEM_EXECUTE: u32 = 0x2000_0000;
const SCN_MEM_WRITE: u32 = 0x8000_0000;
const SCN_CNT_CODE: u32 = 0x0000_0020;

fn import_may_reprotect(module: &str, name: Option<&str>) -> bool {
    let Some(name) = name else { return true };
    if module.to_ascii_lowercase().starts_with("ntdll") {
        return true;
    }
    let native_api = |prefix: &str| {
        name.strip_prefix(prefix)
            .and_then(|rest| rest.chars().next())
            .is_some_and(char::is_uppercase)
    };
    if name.starts_with("Ldr") || native_api("Nt") || native_api("Zw") {
        return true;
    }
    let lower = name.to_ascii_lowercase();
    [
        "protect",
        "writeprocessmemory",
        "writevirtualmemory",
        "mapviewof",
        "unmapviewof",
        "getprocaddress",
        "loadlibrary",
        "getmodulehandle",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

fn main() {
    let bytes = std::fs::read(&std::env::args().nth(1).expect("image")).expect("read");
    let pe = PeImage::parse(&bytes).expect("parse");

    println!("== import half ==");
    let mut flagged: Vec<String> = Vec::new();
    for module in &pe.report().imports {
        for symbol in &module.entries {
            if import_may_reprotect(&module.name, symbol.name.as_deref()) {
                flagged.push(format!("{}!{}", module.name, symbol.name.as_deref().unwrap_or("<ordinal>")));
            }
        }
    }
    println!("  {} reprotect-capable import(s)", flagged.len());
    for entry in flagged.iter().take(12) {
        println!("    {entry}");
    }
    if flagged.len() > 12 {
        println!("    ... and {} more", flagged.len() - 12);
    }

    println!("== section half (byte patterns) ==");
    let data = pe.data();
    for section in &pe.sections {
        let executable = section.characteristics & (SCN_MEM_EXECUTE | SCN_CNT_CODE) != 0;
        let writable = section.characteristics & SCN_MEM_WRITE != 0;
        let name = String::from_utf8_lossy(section.name.as_ref())
            .trim_end_matches('\0')
            .to_owned();
        if !executable {
            println!("  {name:10} not executable -> passes");
            continue;
        }
        if writable {
            println!("  {name:10} FAILS: executable AND writable");
            continue;
        }
        let start = section.pointer_to_raw_data.0 as usize;
        let end = start.saturating_add(section.size_of_raw_data as usize).min(data.len());
        let raw = data.get(start..end).unwrap_or(&[]);
        let syscalls = raw
            .windows(2)
            .filter(|pair| matches!(**pair, [0x0F, 0x05] | [0x0F, 0x34] | [0xCD, 0x2E]))
            .count();
        let teb = raw
            .windows(4)
            .enumerate()
            .filter(|(index, dword)| {
                matches!(**dword, [0x60 | 0x30, 0, 0, 0])
                    && raw[index.saturating_sub(6)..*index]
                        .iter()
                        .any(|byte| matches!(byte, 0x64 | 0x65))
            })
            .count();
        let verdict = if syscalls + teb == 0 { "passes" } else { "FAILS" };
        println!(
            "  {name:10} exec+ro {:>10} bytes: {syscalls} syscall-pair, {teb} seg-prefixed TEB/PEB -> {verdict}",
            raw.len()
        );
        if syscalls + teb > 0 {
            // Over tens of MB of VM bytecode a 2-byte pattern is expected by
            // chance alone: 48 MB / 65536 is about 730 hits of `0F 05`.
            let expected = raw.len() as f64 / 65536.0 * 3.0;
            println!("    (a 2-byte pattern would appear ~{expected:.0}x by chance in this much data)");
        }
    }
}
