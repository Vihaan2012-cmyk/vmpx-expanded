// Repoints code that still calls through the PRE-REBUILD IAT.
//
// The directory rebuild moves the IAT out of the protector section, but code
// emitted before that -- including our own reintegration stub -- can still hold
// a RIP-relative reference to the old slot. On systems.xpl the stub at
// `.vmpx0+0x9b` calls `[0xd7c87e0]`, a slot inside the removed `.vmp3`.
//
// The fix is a slot-for-slot remap, and it is only valid if the rebuilt IAT
// preserved the ordering. That is CHECKED, not assumed: every non-null slot at
// the same byte offset must name the same import in both images, or the tool
// refuses. On systems.xpl all 1259 agree.
//
// Usage: reiat INTACT STRIPPED OUTPUT

use iced_x86::{Decoder, DecoderOptions, Register};
use vmpx_pe::{PeImage, Rva};
use vmpx_rebuild::{retarget_rip_relative, validate_for_loader};

const EXEC: u32 = 0x2000_0000;
const IAT_DIRECTORY: usize = 12;

/// The import name a thunk slot's value points at, minus the 2-byte hint.
fn import_name(bytes: &[u8], pe: &PeImage<'_>, by_name_rva: u32) -> Option<String> {
    let offset = pe.rva_to_file_offset(Rva(by_name_rva.checked_add(2)?)).ok()?;
    let at = usize::try_from(offset.0).ok()?;
    let tail = bytes.get(at..at.saturating_add(96))?;
    let end = tail.iter().position(|b| *b == 0).unwrap_or(tail.len());
    Some(String::from_utf8_lossy(&tail[..end]).into_owned())
}

fn iat_range(pe: &PeImage<'_>) -> (u32, u32) {
    pe.directory(IAT_DIRECTORY)
        .map_or((0, 0), |entry| (entry.address, entry.size))
}

fn slots(bytes: &[u8], pe: &PeImage<'_>, (rva, size): (u32, u32)) -> Vec<Option<String>> {
    let Ok(offset) = pe.rva_to_file_offset(Rva(rva)) else {
        return Vec::new();
    };
    let at = usize::try_from(offset.0).unwrap_or(usize::MAX);
    let Some(blob) = bytes.get(at..at.saturating_add(size as usize)) else {
        return Vec::new();
    };
    blob.chunks_exact(8)
        .map(|chunk| {
            let value = u64::from_le_bytes(chunk.try_into().expect("8 bytes"));
            if value == 0 {
                return None;
            }
            u32::try_from(value & 0x7fff_ffff)
                .ok()
                .and_then(|name_rva| import_name(bytes, pe, name_rva))
        })
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let intact_bytes = std::fs::read(&args[1]).expect("intact");
    let stripped_bytes = std::fs::read(&args[2]).expect("stripped");
    let intact = PeImage::parse(&intact_bytes).expect("parse intact");
    let stripped = PeImage::parse(&stripped_bytes).expect("parse stripped");
    let base = stripped.optional.image_base.get();

    let old = iat_range(&intact);
    let new = iat_range(&stripped);
    println!("old IAT {:#x} size {:#x}", old.0, old.1);
    println!("new IAT {:#x} size {:#x}", new.0, new.1);
    if old.1 != new.1 {
        println!("REFUSED: the IAT changed size, so slots do not correspond by offset");
        return;
    }

    // Checked precondition: same offset must mean the same import.
    let a = slots(&intact_bytes, &intact, old);
    let b = slots(&stripped_bytes, &stripped, new);
    let mut compared = 0_usize;
    for (index, (left, right)) in a.iter().zip(&b).enumerate() {
        if left.is_none() || right.is_none() {
            continue;
        }
        compared += 1;
        if left != right {
            println!(
                "REFUSED: slot {index} (offset {:#x}) is {:?} in the original and {:?} in the \
                 rebuild; the ordering was not preserved",
                index * 8,
                left,
                right
            );
            return;
        }
    }
    println!("slot correspondence verified: {compared} named slot(s) agree by offset");

    // Find every code reference into the OLD IAT range.
    let mut sites: Vec<(u32, u32)> = Vec::new();
    for section in stripped.sections.iter().filter(|s| s.characteristics & EXEC != 0) {
        let start = section.virtual_address.0;
        let size = section.virtual_size.max(section.size_of_raw_data);
        let Ok(offset) = stripped.rva_to_file_offset(Rva(start)) else {
            continue;
        };
        let from = offset.0 as usize;
        let to = from.saturating_add(size as usize).min(stripped_bytes.len());
        let Some(code) = stripped_bytes.get(from..to) else {
            continue;
        };
        // Only sections the exception directory does NOT describe are decoded
        // linearly from the start; those are the emitted ones, where that is
        // the intended reading. Described sections are swept per function.
        let described = stripped
            .report()
            .exception_functions
            .iter()
            .any(|f| f.end.0 > start && f.begin.0 < start + size);
        if described {
            continue;
        }
        let mut decoder = Decoder::with_ip(64, code, base + u64::from(start), DecoderOptions::NONE);
        while decoder.can_decode() {
            let at = decoder.position();
            let instruction = decoder.decode();
            if instruction.is_invalid() {
                break;
            }
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
            if slot >= old.0 && slot < old.0 + old.1 {
                sites.push((start + u32::try_from(at).unwrap_or(0), slot));
            }
        }
    }
    println!("references into the old IAT: {}", sites.len());

    let mut image = stripped_bytes.clone();
    let mut fixed = 0_usize;
    for (site, slot) in &sites {
        let target = new.0 + (slot - old.0);
        let name = b
            .get(((slot - old.0) / 8) as usize)
            .and_then(Option::as_deref)
            .unwrap_or("?");
        match retarget_rip_relative(&image, Rva(*site), Rva(target)) {
            Ok(next) => {
                image = next;
                fixed += 1;
                println!("  {site:#x}: {slot:#x} -> {target:#x}  ({name})");
            }
            Err(error) => println!("  {site:#x}: {slot:#x} -> {target:#x} FAILED: {error}"),
        }
    }
    println!("repointed {fixed} of {}", sites.len());
    match validate_for_loader(&image) {
        Ok(report) => println!("loader validation passed: {} sections, {} imports", report.sections, report.import_symbols),
        Err(error) => println!("LOADER VALIDATION FAILED: {error}"),
    }
    std::fs::write(&args[3], &image).expect("write");
    println!("wrote {}", args[3]);
}
