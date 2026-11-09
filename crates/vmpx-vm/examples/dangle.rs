// Counts references from surviving code to addresses no section covers.
//
// "The sections are gone" and "the file works" are different claims. The strip
// checker proves no RETAINED STRUCTURE points into a removed section -- the
// directories, the relocations, the entry point. It says nothing about a `call`
// in `.text` whose target was in `.vmp3`. This decodes every function the
// exception directory describes and reports each reference that now lands
// nowhere.
//
// Usage: dangle IMAGE

use std::collections::BTreeMap;

use iced_x86::{Decoder, DecoderOptions, FlowControl, Register};
use vmpx_pe::{PeImage, Rva};

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
    let home = |rva: u32| {
        pe.sections
            .iter()
            .find(|section| section.contains_mapped_rva(Rva(rva)))
            .map_or("?", |section| section.name.as_str())
    };

    let functions = pe.report().exception_functions;
    let mut control: BTreeMap<&str, usize> = BTreeMap::new();
    let mut data: BTreeMap<&str, usize> = BTreeMap::new();
    let mut decoded = 0_usize;
    let mut samples = Vec::new();

    for function in &functions {
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
        decoded += 1;
        let mut decoder = Decoder::with_ip(64, code, base + u64::from(begin), DecoderOptions::NONE);
        while decoder.can_decode() {
            let instruction = decoder.decode();
            let from = home(begin);
            // A direct call or jump whose target is no longer mapped.
            if matches!(
                instruction.flow_control(),
                FlowControl::Call | FlowControl::UnconditionalBranch | FlowControl::ConditionalBranch
            ) && instruction.near_branch64() != 0
            {
                let target = instruction.near_branch64();
                // "Unmapped" on a stripped image; on an intact one, a transfer
                // from native code INTO a protector section is the same
                // question asked before the strip: which VM entries does
                // anything outside the VM actually reach?
                if let Some(rva) = target.checked_sub(base).and_then(|v| u32::try_from(v).ok())
                    && rva < size_of_image
                    && (!mapped(rva)
                        || (home(rva).starts_with(".vmp") && !from.starts_with(".vmp")))
                {
                    *control.entry(from).or_default() += 1;
                    if samples.len() < 64 {
                        samples.push(format!(
                            "{:#x} ({from}) {:?} -> {:#x} [unmapped]",
                            instruction.ip() - base,
                            instruction.mnemonic(),
                            rva
                        ));
                    }
                }
            }
            // A RIP-relative operand pointing at an address nothing covers.
            if instruction.memory_base() == Register::RIP {
                let target = instruction.ip_rel_memory_address();
                if let Some(rva) = target.checked_sub(base).and_then(|v| u32::try_from(v).ok())
                    && rva < size_of_image
                    && !mapped(rva)
                {
                    *data.entry(from).or_default() += 1;
                }
            }
        }
    }

    println!("{}: {} sections, {decoded} function(s) decoded", args[1], pe.sections.len());
    println!("dangling CONTROL transfers by source section:");
    for (section, count) in &control {
        println!("  {section:>9}: {count}");
    }
    if control.is_empty() {
        println!("  none");
    }
    println!("dangling DATA references by source section:");
    for (section, count) in &data {
        println!("  {section:>9}: {count}");
    }
    if data.is_empty() {
        println!("  none");
    }
    for sample in &samples {
        println!("  sample {sample}");
    }
}
