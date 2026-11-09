// Repoints code that loads a function pointer from a slot in a removed
// section, once that function has been relocated.
//
// The pattern is `mov rax,[<removed slot>]; call rax`. The slot is pre-filled
// in the intact image, so reading it there says which function the site wanted.
// Given `old=new` RVA pairs for functions that have moved, this emits one
// qword per distinct new target and rewrites each referencing instruction to
// read from it -- the same length-preserving rewrite `retarget_rip_relative`
// performs, so no layout changes.
//
// Sites whose slot resolves to a function NOT in the mapping are left alone and
// reported, so the output says what remains rather than quietly covering it.
//
// Usage: repoint INTACT STRIPPED OUTPUT old=new [old=new ...]   (RVAs, hex)

use std::collections::BTreeMap;

use iced_x86::{Decoder, DecoderOptions, FlowControl, Register};
use vmpx_pe::{PeImage, Rva};
use vmpx_rebuild::{
    AppendSection, append_section, retarget_direct_branch, retarget_rip_relative,
    validate_for_loader,
};

const RDATA: u32 = 0x4000_0040; // INITIALIZED_DATA | READ

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let intact_bytes = std::fs::read(&args[1]).expect("intact");
    let stripped_bytes = std::fs::read(&args[2]).expect("stripped");
    let intact = PeImage::parse(&intact_bytes).expect("parse intact");
    let stripped = PeImage::parse(&stripped_bytes).expect("parse stripped");
    let base = stripped.optional.image_base.get();

    let mapping: BTreeMap<u32, u32> = args[4..]
        .iter()
        .filter_map(|pair| {
            let (old, new) = pair.split_once('=')?;
            Some((
                u32::from_str_radix(old.trim().trim_start_matches("0x"), 16).ok()?,
                u32::from_str_radix(new.trim().trim_start_matches("0x"), 16).ok()?,
            ))
        })
        .collect();
    println!("{} relocation(s) given", mapping.len());
    for (old, new) in &mapping {
        println!("  {old:#x} -> {new:#x}");
    }

    let mapped = |rva: u32| {
        stripped
            .sections
            .iter()
            .any(|s| s.contains_mapped_rva(Rva(rva)))
    };
    let qword_at = |rva: u32| -> Option<u64> {
        let offset = intact.rva_to_file_offset(Rva(rva)).ok()?;
        let at = usize::try_from(offset.0).ok()?;
        intact_bytes
            .get(at..at.checked_add(8)?)
            .and_then(|s| s.try_into().ok())
            .map(u64::from_le_bytes)
    };

    // Every RIP-relative operand in a `.pdata`-described function that reads a
    // slot the strip removed, resolved through the intact image.
    let mut sites: Vec<(u32, u32)> = Vec::new(); // (instruction rva, wanted target)
    let mut unmapped: BTreeMap<u32, usize> = BTreeMap::new();
    let mut direct: Vec<(u32, u32)> = Vec::new();
    for function in &stripped.report().exception_functions {
        let (begin, end) = (function.begin.0, function.end.0);
        if end <= begin || end - begin > 0x10_0000 || !mapped(begin) {
            continue;
        }
        let Ok(offset) = stripped.rva_to_file_offset(Rva(begin)) else {
            continue;
        };
        let from = offset.0 as usize;
        let to = from
            .saturating_add((end - begin) as usize)
            .min(stripped_bytes.len());
        let Some(code) = stripped_bytes.get(from..to) else {
            continue;
        };
        let mut decoder = Decoder::with_ip(64, code, base + u64::from(begin), DecoderOptions::NONE);
        while decoder.can_decode() {
            let at = decoder.position();
            let instruction = decoder.decode();
            if instruction.is_invalid() {
                break;
            }
            // A DIRECT transfer to a relocated function needs its rel32
            // retargeted, not a pointer slot. Same mapping, different rewrite.
            if matches!(
                instruction.flow_control(),
                FlowControl::Call | FlowControl::UnconditionalBranch
            ) && instruction.near_branch64() != 0
            {
                if let Some(target) = instruction
                    .near_branch64()
                    .checked_sub(base)
                    .and_then(|v| u32::try_from(v).ok())
                    && let Some(new) = mapping.get(&target)
                {
                    direct.push((begin + u32::try_from(at).unwrap_or(0), *new));
                }
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
            if slot >= stripped.optional.size_of_image || mapped(slot) {
                continue;
            }
            let target = qword_at(slot)
                .and_then(|v| v.checked_sub(base))
                .and_then(|v| u32::try_from(v).ok());
            match target.and_then(|t| mapping.get(&t)) {
                Some(new) => sites.push((begin + u32::try_from(at).unwrap_or(0), *new)),
                None => *unmapped.entry(target.unwrap_or(0)).or_default() += 1,
            }
        }
    }
    println!(
        "\n{} site(s) load a relocated function; {} site(s) resolve elsewhere",
        sites.len(),
        unmapped.values().sum::<usize>()
    );
    for (target, count) in &unmapped {
        println!("  not relocated: {target:#x} referenced by {count} site(s)");
    }
    println!("{} direct transfer(s) reach a relocated function", direct.len());
    if sites.is_empty() && direct.is_empty() {
        return;
    }

    // One qword per distinct new target.
    let mut order: Vec<u32> = sites.iter().map(|(_, new)| *new).collect();
    order.sort_unstable();
    order.dedup();
    let section_rva = vmpx_rebuild::next_section_rva(&stripped_bytes).expect("rva").0;
    let mut payload = Vec::new();
    let mut slot_of: BTreeMap<u32, u32> = BTreeMap::new();
    for target in &order {
        slot_of.insert(
            *target,
            section_rva + u32::try_from(payload.len()).unwrap_or(0),
        );
        payload.extend_from_slice(&(base + u64::from(*target)).to_le_bytes());
    }
    let rebuilt = append_section(
        &stripped_bytes,
        AppendSection {
            name: *b".vmpxptr",
            data: &payload,
            characteristics: RDATA,
        },
    )
    .expect("append");
    assert_eq!(rebuilt.section.rva.0, section_rva, "section moved");
    println!("\n{} pointer slot(s) at rva {section_rva:#x}", order.len());

    let mut image = rebuilt.bytes;
    let (mut done, mut failed) = (0_usize, 0_usize);
    for (site, new) in &sites {
        let slot = slot_of[new];
        match retarget_rip_relative(&image, Rva(*site), Rva(slot)) {
            Ok(next) => {
                image = next;
                done += 1;
            }
            Err(error) => {
                failed += 1;
                if failed <= 5 {
                    println!("  {site:#x} -> {slot:#x} FAILED: {error}");
                }
            }
        }
    }
    println!("repointed {done} site(s); {failed} failed");
    let (mut redirected, mut direct_failed) = (0_usize, 0_usize);
    for (site, new) in &direct {
        match retarget_direct_branch(&image, Rva(*site), Rva(*new)) {
            Ok(next) => {
                image = next;
                redirected += 1;
            }
            Err(error) => {
                direct_failed += 1;
                println!("  direct {site:#x} -> {new:#x} FAILED: {error}");
            }
        }
    }
    println!("retargeted {redirected} direct transfer(s); {direct_failed} failed");
    match validate_for_loader(&image) {
        Ok(report) => println!(
            "loader validation passed: {} sections, {} imports",
            report.sections, report.import_symbols
        ),
        Err(error) => println!("LOADER VALIDATION FAILED: {error}"),
    }
    std::fs::write(&args[3], &image).expect("write");
    println!("wrote {}", args[3]);
}
