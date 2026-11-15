// Measures what it would take to MOVE the import address table.
//
// The IAT is the one directory inside the protector section that code reaches
// by address: `call [rip+disp32]`. Relocating it means rewriting every such
// displacement, so the question that decides feasibility is how many there are
// and whether they all sit inside code the exception directory already
// describes.
//
// Usage: iatrefs IMAGE

use std::collections::BTreeMap;

use iced_x86::{Decoder, DecoderOptions, Register};
use vmpx_pe::{PeImage, Rva};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bytes = std::fs::read(&args[1]).expect("image");
    let pe = PeImage::parse(&bytes).expect("parse");
    let base = pe.optional.image_base.get();

    let iat = pe.directory(12).expect("IAT directory");
    let (iat_lo, iat_hi) = (iat.address, iat.address + iat.size);
    println!("IAT rva {:#x}..{:#x} ({} bytes)", iat_lo, iat_hi, iat.size);

    let section_of = |rva: u32| {
        pe.sections
            .iter()
            .find(|section| section.contains_mapped_rva(Rva(rva)))
            .map_or("?", |section| section.name.as_str())
    };
    let removed = |rva: u32| section_of(rva).starts_with(".vmp");

    // Every function the exception directory describes, grouped by the section
    // it lives in: that is the code a rewrite could reach systematically.
    let functions = pe.report().exception_functions;
    println!("exception-described functions: {}", functions.len());

    let mut refs_by_section: BTreeMap<&str, usize> = BTreeMap::new();
    let mut vmp_refs_by_section: BTreeMap<&str, usize> = BTreeMap::new();
    let mut covered = 0_usize;
    let mut undecodable = 0_usize;
    let mut samples = Vec::new();

    for function in &functions {
        let begin = function.begin.0;
        let end = function.end.0;
        if end <= begin || end - begin > 0x10_0000 {
            continue;
        }
        let Ok(offset) = pe.rva_to_file_offset(Rva(begin)) else {
            undecodable += 1;
            continue;
        };
        let start = offset.0 as usize;
        let stop = start + (end - begin) as usize;
        let Some(code) = bytes.get(start..stop.min(bytes.len())) else {
            undecodable += 1;
            continue;
        };
        covered += 1;
        let mut decoder = Decoder::with_ip(64, code, base + u64::from(begin), DecoderOptions::NONE);
        while decoder.can_decode() {
            let instruction = decoder.decode();
            if instruction.memory_base() != Register::RIP {
                continue;
            }
            let target = instruction.ip_rel_memory_address();
            let Some(rva) = target.checked_sub(base).and_then(|v| u32::try_from(v).ok()) else {
                continue;
            };
            let home = section_of(begin);
            if rva >= iat_lo && rva < iat_hi {
                *refs_by_section.entry(home).or_default() += 1;
                if samples.len() < 6 {
                    samples.push(format!(
                        "{:#x} ({home}) -> IAT {:#x}  {:?}",
                        instruction.ip() - base,
                        rva,
                        instruction.mnemonic()
                    ));
                }
            } else if removed(rva) {
                *vmp_refs_by_section.entry(home).or_default() += 1;
            }
        }
    }

    println!("functions decoded: {covered} (skipped {undecodable})");
    println!("RIP-relative references INTO THE IAT, by referring section:");
    for (section, count) in &refs_by_section {
        println!("  {section:>8}: {count}");
    }
    println!("RIP-relative references into other .vmp bytes, by referring section:");
    for (section, count) in &vmp_refs_by_section {
        println!("  {section:>8}: {count}");
    }
    for sample in &samples {
        println!("  sample {sample}");
    }
}
