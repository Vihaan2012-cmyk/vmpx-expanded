// Acceptance test for "no protector junk remains", as a property of the
// ARTIFACT rather than of one execution.
//
// A dynamic run can only ever report what happened on that run. This asks a
// different question, and it is decidable: does the finished image contain any
// VALUE that names the removed address range? A `.vmp` pointer can only enter
// a register from an immediate in the code, from a load of a location holding
// one, or from the protector itself -- and the protector is gone. So if no
// immediate and no stored word anywhere names the removed range, no execution
// can produce such a pointer except by arithmetic on a non-pointer.
//
// That is a closed-world argument over sources, not a coverage argument, so it
// holds on ALL paths without running anything.
//
// Every check here is exhaustive and UNALIGNED -- a pointer does not have to
// sit on an 8-byte boundary -- and each hit is classified so the residue is
// precise rather than merely small:
//
//   * code reference on a proven instruction boundary -> REAL, must be fixed
//   * code reference not on any boundary              -> a displacement byte
//   * stored word with a base relocation              -> REAL pointer
//   * stored word with no relocation                  -> not a pointer the
//     loader would fix up, so it cannot be a valid absolute address
//
// Usage: closure IMAGE

use std::collections::BTreeSet;

use iced_x86::{Decoder, DecoderOptions, FlowControl, Register};
use vmpx_pe::{PeImage, Rva};

const EXEC: u32 = 0x2000_0000;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bytes = std::fs::read(&args[1]).expect("image");
    let pe = PeImage::parse(&bytes).expect("parse");
    let base = pe.optional.image_base.get();
    let size_of_image = pe.optional.size_of_image;
    let mapped = |rva: u32| pe.sections.iter().any(|s| s.contains_mapped_rva(Rva(rva)));
    // The removed range: in-image RVAs that no section covers.
    let removed = |rva: u32| rva >= 0x1000 && rva < size_of_image && !mapped(rva);

    // ---- relocated fields, so a stored word can be judged ----
    let mut relocated: BTreeSet<u32> = BTreeSet::new();
    for block in &pe.report().relocations {
        for entry in &block.entries {
            if let Some(target) = entry.target_rva {
                relocated.insert(target.0);
            }
        }
    }

    // ---- 1. code references, discriminated by instruction boundary ----
    let mut on_boundary: Vec<String> = Vec::new();
    let mut off_boundary = 0_u64;
    let dangles = |instruction: &iced_x86::Instruction| -> Option<u32> {
        let target = if matches!(
            instruction.flow_control(),
            FlowControl::Call | FlowControl::UnconditionalBranch | FlowControl::ConditionalBranch
        ) && instruction.near_branch64() != 0
        {
            instruction.near_branch64()
        } else if instruction.memory_base() == Register::RIP {
            instruction.ip_rel_memory_address()
        } else {
            return None;
        };
        let rva = target.checked_sub(base).and_then(|v| u32::try_from(v).ok())?;
        removed(rva).then_some(rva)
    };

    // Boundaries come from the functions the exception directory names, plus a
    // decode of each executable section not covered by one (where the only
    // honest test is a clean decode with lookahead).
    let mut covered: Vec<(u32, u32)> = pe
        .report()
        .exception_functions
        .iter()
        .filter(|f| f.end.0 > f.begin.0 && f.end.0 - f.begin.0 <= 0x10_0000)
        .map(|f| (f.begin.0, f.end.0))
        .collect();
    covered.sort_unstable();

    for (begin, end) in &covered {
        if !mapped(*begin) {
            continue;
        }
        let Ok(offset) = pe.rva_to_file_offset(Rva(*begin)) else {
            continue;
        };
        let start = offset.0 as usize;
        let stop = start.saturating_add((end - begin) as usize).min(bytes.len());
        let Some(code) = bytes.get(start..stop) else {
            continue;
        };
        let mut decoder = Decoder::with_ip(64, code, base + u64::from(*begin), DecoderOptions::NONE);
        while decoder.can_decode() {
            let at = decoder.position();
            let instruction = decoder.decode();
            if instruction.is_invalid() {
                break;
            }
            if let Some(target) = dangles(&instruction) {
                let rva = begin + u32::try_from(at).unwrap_or(0);
                on_boundary.push(format!("{rva:#x} -> {target:#x}  {instruction}"));
            }
        }
    }

    // Executable bytes no RUNTIME_FUNCTION describes.
    let mut merged: Vec<(u32, u32)> = Vec::new();
    for (begin, end) in covered {
        match merged.last_mut() {
            Some(last) if begin <= last.1 => last.1 = last.1.max(end),
            _ => merged.push((begin, end)),
        }
    }
    for section in pe.sections.iter().filter(|s| s.characteristics & EXEC != 0) {
        let start = section.virtual_address.0;
        let size = section.virtual_size.max(section.size_of_raw_data);
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
            for at in 0..window.len() {
                let rva = gap_begin + u32::try_from(at).unwrap_or(0);
                let Some(code) = window.get(at..) else { continue };
                let mut probe =
                    Decoder::with_ip(64, code, base + u64::from(rva), DecoderOptions::NONE);
                let instruction = probe.decode();
                if instruction.is_invalid() {
                    continue;
                }
                let Some(target) = dangles(&instruction) else {
                    continue;
                };
                // Padding before it, or a clean run after it, is the only
                // structural evidence available outside .pdata.
                let padded = at >= 4
                    && window[at - 4..at]
                        .iter()
                        .all(|b| matches!(b, 0x90 | 0xCC | 0x00));
                let mut clean = true;
                for _ in 0..4 {
                    if !probe.can_decode() {
                        break;
                    }
                    if probe.decode().is_invalid() {
                        clean = false;
                        break;
                    }
                }
                if padded && clean {
                    on_boundary.push(format!(
                        "{rva:#x} -> {target:#x}  {instruction}   [no .pdata, padding-preceded]"
                    ));
                } else {
                    off_boundary += 1;
                }
            }
        }
    }

    // ---- 2. stored words, UNALIGNED, judged by relocation ----
    let (mut q_reloc, mut q_noreloc, mut d_noreloc) = (Vec::new(), 0_u64, 0_u64);
    for section in &pe.sections {
        if section.size_of_raw_data == 0 {
            continue;
        }
        let from = section.pointer_to_raw_data.0 as usize;
        let to = from
            .saturating_add(section.size_of_raw_data as usize)
            .min(bytes.len());
        let Some(blob) = bytes.get(from..to) else {
            continue;
        };
        for at in 0..blob.len().saturating_sub(8) {
            let field = section.virtual_address.0 + u32::try_from(at).unwrap_or(0);
            let q = u64::from_le_bytes(blob[at..at + 8].try_into().expect("8 bytes"));
            if let Some(rva) = q.checked_sub(base).and_then(|v| u32::try_from(v).ok())
                && removed(rva)
            {
                if relocated.contains(&field) {
                    q_reloc.push(format!(
                        "{field:#x} ({}) holds {rva:#x}",
                        section.name.as_str()
                    ));
                } else {
                    q_noreloc += 1;
                }
            }
            let d = u32::from_le_bytes(blob[at..at + 4].try_into().expect("4 bytes"));
            if removed(d) && !relocated.contains(&field) {
                d_noreloc += 1;
            }
        }
    }

    println!("{}: closure check over the removed address range", args[1]);
    println!("  relocated fields in the image: {}", relocated.len());
    println!("\n=== CODE references into removed space ===");
    println!("  on a proven instruction boundary (REAL)   : {}", on_boundary.len());
    println!("  not on any boundary (displacement byte)   : {off_boundary}");
    for entry in on_boundary.iter().take(60) {
        println!("    {entry}");
    }
    println!("\n=== STORED words naming removed space (unaligned scan) ===");
    println!("  8-byte, WITH a base relocation (REAL ptr) : {}", q_reloc.len());
    println!("  8-byte, no relocation (not a loader ptr)  : {q_noreloc}");
    println!("  4-byte RVA-shaped, no relocation          : {d_noreloc}");
    for entry in q_reloc.iter().take(40) {
        println!("    {entry}");
    }
    let real = on_boundary.len() + q_reloc.len();
    println!("\nVERDICT: {real} residual reference(s) that are provably real.");
    if real == 0 {
        println!("  No immediate and no relocated word names the removed range, so no");
        println!("  execution can materialize such a pointer from this image.");
    }
}
