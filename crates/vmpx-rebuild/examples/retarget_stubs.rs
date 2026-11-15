// Closes the trivial class of the full-PE gap: dangling RIP-relative loads of
// a protector pointer slot whose target is a single `ret` byte.
//
// On systems.xpl the stripped image has 39 dangling data references, each
// `mov rax,[<removed .vmp address>]` followed by a call or jump. The slot is
// PRE-FILLED in the file, so reading it in the intact image gives the target.
// Five of those targets are one `0xC3` byte -- a do-nothing stub. This appends
// a section holding a replacement `ret` per distinct such target plus a
// pointer table, then repoints each referencing instruction at the new slot.
//
// Usage: retarget_stubs INTACT STRIPPED OUTPUT

use std::collections::BTreeMap;

use iced_x86::{Decoder, DecoderOptions, Register};
use vmpx_pe::{PeImage, Rva};
use vmpx_rebuild::{AppendSection, append_section, retarget_rip_relative, validate_for_loader};

const RX: u32 = 0x6000_0020; // CODE | READ | EXECUTE

/// Every RIP-relative operand in a `.pdata`-described function that addresses
/// something no section of `pe` covers, as (instruction RVA, slot RVA).
fn dangling_rip_operands(bytes: &[u8], pe: &PeImage<'_>) -> Vec<(u32, u32)> {
    let base = pe.optional.image_base.get();
    let mapped = |rva: u32| {
        pe.sections
            .iter()
            .any(|section| section.contains_mapped_rva(Rva(rva)))
    };
    let mut found = Vec::new();
    for function in &pe.report().exception_functions {
        let (begin, end) = (function.begin.0, function.end.0);
        if end <= begin || end - begin > 0x10_0000 || !mapped(begin) {
            continue;
        }
        let Ok(offset) = pe.rva_to_file_offset(Rva(begin)) else {
            continue;
        };
        let start = offset.0 as usize;
        let stop = start.saturating_add((end - begin) as usize).min(bytes.len());
        let Some(code) = bytes.get(start..stop) else {
            continue;
        };
        let mut decoder = Decoder::with_ip(64, code, base + u64::from(begin), DecoderOptions::NONE);
        while decoder.can_decode() {
            let instruction = decoder.decode();
            if instruction.memory_base() != Register::RIP {
                continue;
            }
            let Some(slot) = instruction
                .ip_rel_memory_address()
                .checked_sub(base)
                .and_then(|v| u32::try_from(v).ok())
            else {
                continue;
            };
            if slot < pe.optional.size_of_image && !mapped(slot) {
                let at = u32::try_from(instruction.ip() - base).unwrap_or(0);
                found.push((at, slot));
            }
        }
    }
    found
}

/// The qword stored at `rva` in `pe`, read from file data.
fn qword_at(bytes: &[u8], pe: &PeImage<'_>, rva: u32) -> Option<u64> {
    let offset = pe.rva_to_file_offset(Rva(rva)).ok()?;
    let at = usize::try_from(offset.0).ok()?;
    let slice = bytes.get(at..at.checked_add(8)?)?;
    Some(u64::from_le_bytes(slice.try_into().ok()?))
}

fn byte_at(bytes: &[u8], pe: &PeImage<'_>, rva: u32) -> Option<u8> {
    let offset = pe.rva_to_file_offset(Rva(rva)).ok()?;
    bytes.get(usize::try_from(offset.0).ok()?).copied()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let intact_bytes = std::fs::read(&args[1]).expect("intact image");
    let stripped_bytes = std::fs::read(&args[2]).expect("stripped image");
    let intact = PeImage::parse(&intact_bytes).expect("parse intact");
    let stripped = PeImage::parse(&stripped_bytes).expect("parse stripped");
    let base = stripped.optional.image_base.get();
    assert_eq!(base, intact.optional.image_base.get(), "image bases differ");

    // Group the dangling references by the `ret`-stub target their slot holds.
    let mut by_target: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    let mut other = 0_usize;
    for (site, slot) in dangling_rip_operands(&stripped_bytes, &stripped) {
        let Some(pointer) = qword_at(&intact_bytes, &intact, slot) else {
            other += 1;
            continue;
        };
        let Some(target) = pointer.checked_sub(base).and_then(|v| u32::try_from(v).ok()) else {
            other += 1;
            continue;
        };
        // A do-nothing stub is exactly one `ret`. Anything else needs real
        // recovery and is left alone.
        if byte_at(&intact_bytes, &intact, target) == Some(0xC3) {
            by_target.entry(target).or_default().push(site);
        } else {
            other += 1;
        }
    }
    println!(
        "{} dangling slot reference(s) resolve to a bare `ret`, over {} distinct target(s); \
         {other} reference(s) need real recovery",
        by_target.values().map(Vec::len).sum::<usize>(),
        by_target.len()
    );
    if by_target.is_empty() {
        return;
    }

    // Payload: one `ret` per distinct target, then a qword per target holding
    // that replacement's address. Sites load from the qword, not the code.
    let count = by_target.len();
    let table = (count + 7) & !7;
    let mut payload = vec![0xC3_u8; count];
    payload.resize(table, 0x90);
    let payload_rva = vmpx_rebuild::next_section_rva(&stripped_bytes).expect("next rva").0;
    for index in 0..count {
        let stub = u64::from(payload_rva) + u64::try_from(index).unwrap_or(0);
        payload.extend_from_slice(&(base + stub).to_le_bytes());
    }

    let rebuilt = append_section(
        &stripped_bytes,
        AppendSection {
            name: *b".vmpx1\0\0",
            data: &payload,
            characteristics: RX,
        },
    )
    .expect("append");
    println!(
        "appended {} at {:?}, {} byte(s)",
        String::from_utf8_lossy(&rebuilt.section.name).trim_end_matches('\0'),
        rebuilt.section.rva,
        payload.len()
    );
    assert_eq!(rebuilt.section.rva.0, payload_rva, "section moved");

    let mut image = rebuilt.bytes;
    let mut done = 0_usize;
    for (index, (target, sites)) in by_target.iter().enumerate() {
        let slot = payload_rva
            + u32::try_from(table).unwrap_or(0)
            + u32::try_from(index * 8).unwrap_or(0);
        for site in sites {
            match retarget_rip_relative(&image, Rva(*site), Rva(slot)) {
                Ok(next) => {
                    image = next;
                    done += 1;
                }
                Err(error) => println!("  {site:#x} -> slot {slot:#x} (was {target:#x}): {error}"),
            }
        }
    }
    println!("repointed {done} reference(s)");

    match validate_for_loader(&image) {
        Ok(report) => println!("loader validation passed: {report:?}"),
        Err(error) => println!("LOADER VALIDATION FAILED: {error}"),
    }
    std::fs::write(&args[3], &image).expect("write output");
    println!("wrote {} ({} bytes)", args[3], image.len());
}
