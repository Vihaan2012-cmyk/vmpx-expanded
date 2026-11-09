// Audits the executable bytes `.pdata` does NOT describe.
//
// `dangle` decodes only functions the exception directory names, which on
// systems.xpl is 95.3% of `.text` -- leaving ~1.21 MB that no sweep has ever
// looked at. The 4-slot thunk table at RVA 0x1740ab8 lives there, so the gap
// is known to hold real references.
//
// A raw byte scan over that much data produces mostly chance hits (a random
// dword lands in the removed range ~1% of the time), so each candidate is
// VALIDATED: decode at the offset, require the instruction to reference
// unmapped space, and require the next few instructions to decode cleanly.
// Padding-preceded candidates (the thunk-table shape) are reported separately
// because that is positive structural evidence, not a coincidence.
//
// Usage: uncovered IMAGE

use iced_x86::{Decoder, DecoderOptions, FlowControl, Register};
use vmpx_pe::{PeImage, Rva};

const EXEC: u32 = 0x2000_0000;
/// How many instructions after the candidate must decode cleanly.
const LOOKAHEAD: usize = 4;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bytes = std::fs::read(&args[1]).expect("image");
    let pe = PeImage::parse(&bytes).expect("parse");
    let base = pe.optional.image_base.get();
    let size_of_image = pe.optional.size_of_image;
    let mapped = |rva: u32| {
        pe.sections
            .iter()
            .any(|section| section.contains_mapped_rva(Rva(rva)))
    };

    // Merged .pdata coverage.
    let mut covered: Vec<(u32, u32)> = pe
        .report()
        .exception_functions
        .iter()
        .filter(|f| f.end.0 > f.begin.0)
        .map(|f| (f.begin.0, f.end.0))
        .collect();
    // UNCOVERED_ALL=1 drops the .pdata filter so the scan covers EVERY
    // executable byte. That cross-checks the linear per-function decode, which
    // desynchronizes on embedded data and could silently miss sites.
    if std::env::var_os("UNCOVERED_ALL").is_some() {
        covered.clear();
    }
    covered.sort_unstable();
    let mut merged: Vec<(u32, u32)> = Vec::new();
    for (begin, end) in covered {
        match merged.last_mut() {
            Some(last) if begin <= last.1 => last.1 = last.1.max(end),
            _ => merged.push((begin, end)),
        }
    }

    let mut scanned = 0_u64;
    let mut validated = Vec::new();
    let mut padded = Vec::new();
    for section in &pe.sections {
        if section.characteristics & EXEC == 0 {
            continue;
        }
        let start = section.virtual_address.0;
        let size = section.virtual_size.max(section.size_of_raw_data);
        // The uncovered sub-ranges of this section.
        let mut cursor = start;
        let mut gaps: Vec<(u32, u32)> = Vec::new();
        for (begin, end) in &merged {
            if *end <= start || *begin >= start + size {
                continue;
            }
            if *begin > cursor {
                gaps.push((cursor, (*begin).min(start + size)));
            }
            cursor = cursor.max(*end);
        }
        if cursor < start + size {
            gaps.push((cursor, start + size));
        }

        for (gap_begin, gap_end) in gaps {
            let Ok(offset) = pe.rva_to_file_offset(Rva(gap_begin)) else {
                continue;
            };
            let from = offset.0 as usize;
            let to = from
                .saturating_add((gap_end - gap_begin) as usize)
                .min(bytes.len());
            let Some(window) = bytes.get(from..to) else {
                continue;
            };
            scanned += window.len() as u64;
            for at in 0..window.len() {
                let rva = gap_begin + u32::try_from(at).unwrap_or(0);
                let Some(code) = window.get(at..) else { continue };
                let mut decoder =
                    Decoder::with_ip(64, code, base + u64::from(rva), DecoderOptions::NONE);
                let instruction = decoder.decode();
                if instruction.is_invalid() {
                    continue;
                }
                // Does it reference something no section covers?
                let mut target = None;
                if matches!(
                    instruction.flow_control(),
                    FlowControl::Call | FlowControl::UnconditionalBranch
                ) && instruction.near_branch64() != 0
                {
                    target = instruction.near_branch64().checked_sub(base);
                } else if instruction.memory_base() == Register::RIP {
                    target = instruction.ip_rel_memory_address().checked_sub(base);
                }
                let Some(rel) = target.and_then(|v| u32::try_from(v).ok()) else {
                    continue;
                };
                if rel >= size_of_image || mapped(rel) {
                    continue;
                }
                // Validate: the following instructions must decode cleanly.
                let mut ok = true;
                for _ in 0..LOOKAHEAD {
                    if !decoder.can_decode() {
                        break;
                    }
                    if decoder.decode().is_invalid() {
                        ok = false;
                        break;
                    }
                }
                if !ok {
                    continue;
                }
                let entry = format!(
                    "{rva:#x} ({}) {} -> {rel:#x}",
                    section.name.as_str(),
                    instruction
                );
                // Padding before the site is structural evidence of a real
                // thunk slot rather than a byte coincidence.
                let pad = at >= 4
                    && window[at - 4..at]
                        .iter()
                        .all(|b| matches!(b, 0x90 | 0xCC | 0x00));
                if pad {
                    padded.push(entry);
                } else {
                    validated.push(entry);
                }
            }
        }
    }

    println!(
        "{}: scanned {scanned} executable byte(s) that .pdata does not describe",
        args[1]
    );
    println!(
        "candidates referencing unmapped space: {} padding-preceded, {} other",
        padded.len(),
        validated.len()
    );
    println!("\n-- padding-preceded (thunk-slot shape) --");
    for entry in padded.iter().take(40) {
        println!("  {entry}");
    }
    println!("\n-- other (byte coincidence unless proven otherwise) --");
    for entry in validated.iter().take(if std::env::var_os("UNCOVERED_ALL").is_some() { 0 } else { 400 }) {
        println!("  {entry}");
    }
}
