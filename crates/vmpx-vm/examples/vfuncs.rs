// Function identification by PROVEN REACHABILITY, not by byte scan.
//
// `discover_vm_entries`' executable scan yields `UnprovenExecutableCandidate`
// hits -- on systems.xpl all 814 of them sit inside `.vmp2`, 789 inside a single
// 4 MB protector RUNTIME_FUNCTION, so nothing downstream can prove any of them
// is a function entry. That is why `clean` refuses native replacement.
//
// A virtualized function leaves its `.text` body as a direct transfer into the
// protector, so this instead:
//   1. scans executable non-protector sections for `call rel32` / `jmp rel32`
//      whose target lands in a protector section,
//   2. scores each distinct target with `analyze_vm_entry_at`,
//   3. maps the transfer SITE back to its enclosing `.pdata` RUNTIME_FUNCTION.
//
// The result carries what replacement needs and the byte scan cannot give: a
// proven entry address, a proven extent (begin+length), and a real call site.
//
// Measured on systems.xpl: 587 call targets are almost all import-protection
// stubs (513 score confidence 0); the JMP targets hold the real functions --
// 13 at confidence >= 80, resolving to 8 distinct proven `.text` functions.
//
// Usage: vfuncs IMAGE [MIN_CONFIDENCE]
use std::collections::{BTreeMap, BTreeSet};
use std::fs;

use vmpx_pe::{PeImage, Rva};
use vmpx_vm::{VmDiscoveryLimits, analyze_vm_entry_at};

const IMAGE_SCN_MEM_EXECUTE: u32 = 0x2000_0000;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let min_confidence: u8 = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(80);
    let bytes = fs::read(&args[1]).expect("image");
    let pe = PeImage::parse(&bytes).expect("parse");

    let protector: Vec<(u32, u32)> = pe
        .sections
        .iter()
        .filter(|s| s.name.starts_with(".vmp"))
        .map(|s| (s.virtual_address.0, s.virtual_address.0 + s.mapped_span()))
        .collect();
    let in_protector = |rva: u32| protector.iter().any(|(a, b)| rva >= *a && rva < *b);

    // Proven function extents, from the image's own unwind table.
    let mut extent: BTreeMap<u32, u32> = BTreeMap::new();
    for f in &pe.report().exception_functions {
        if f.end.0 > f.begin.0 {
            extent.insert(f.begin.0, f.end.0 - f.begin.0);
        }
    }
    let starts: Vec<u32> = extent.keys().copied().collect();
    let enclosing = |rva: u32| {
        let i = starts.partition_point(|s| *s <= rva);
        i.checked_sub(1)
            .map(|i| starts[i])
            .filter(|s| s + extent[s] > rva)
    };

    // Direct transfers out of real code and into the protector.
    let mut sites: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    for section in pe.sections.iter().filter(|s| {
        s.characteristics & IMAGE_SCN_MEM_EXECUTE != 0 && !s.name.starts_with(".vmp")
    }) {
        let lo = section.virtual_address.0;
        let hi = lo.saturating_add(section.mapped_span());
        for rva in lo..hi.saturating_sub(5) {
            let Ok(offset) = pe.rva_to_file_offset(Rva(rva)) else { continue };
            let at = offset.0 as usize;
            if at + 5 > bytes.len() {
                continue;
            }
            if bytes[at] != 0xE9 && bytes[at] != 0xE8 {
                continue;
            }
            let disp = i32::from_le_bytes([bytes[at + 1], bytes[at + 2], bytes[at + 3], bytes[at + 4]]);
            let Some(target) = rva.checked_add(5).and_then(|a| a.checked_add_signed(disp)) else {
                continue;
            };
            if in_protector(target) {
                sites.entry(target).or_default().push(rva);
            }
        }
    }

    let mut inventory = Vec::new();
    let mut scored = 0usize;
    for (target, from) in &sites {
        let Some(candidate) = analyze_vm_entry_at(&pe, Rva(*target), VmDiscoveryLimits::default())
        else {
            continue;
        };
        scored += 1;
        if candidate.confidence < min_confidence {
            continue;
        }
        for site in from {
            inventory.push((*target, *site, enclosing(*site), candidate.confidence, candidate.seed));
        }
    }

    println!(
        "{} distinct protector targets reached by a direct transfer from real code; {scored} scored; {} at confidence >= {min_confidence}",
        sites.len(),
        inventory.len()
    );
    println!("\nvm_entry    site        proven function      confidence  seed");
    for (target, site, func, confidence, seed) in &inventory {
        match func {
            Some(begin) => println!(
                "  {target:<#10x} {site:<#10x}  {begin:#x} len {:<6} {confidence:<10} {seed:?}",
                extent[begin]
            ),
            None => println!("  {target:<#10x} {site:<#10x}  (no .pdata coverage)  {confidence:<10} {seed:?}"),
        }
    }
    let proven: BTreeSet<u32> = inventory.iter().filter_map(|(_, _, f, _, _)| *f).collect();
    println!(
        "\n{} distinct PROVEN virtualized functions (proven entry + extent + call site)",
        proven.len()
    );
}
