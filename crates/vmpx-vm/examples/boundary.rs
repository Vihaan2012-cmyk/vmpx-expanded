// Decides between two instruments that disagree.
//
// Linear decode from each `.pdata` function entry finds 48 references into
// removed space. A byte scan that decodes at EVERY offset and keeps whatever
// decodes cleanly finds ~1291. One of them is wrong, and "decodes cleanly" is
// a weak filter -- a displacement byte often decodes as a plausible opcode.
//
// The discriminator: a real instruction lies on the instruction stream a decode
// from the function's entry produces. So for each function, record the set of
// instruction START offsets, then ask of every byte-scan candidate:
//
//   on a boundary AND linear found it  -> agreed, real
//   on a boundary AND linear missed it -> a BUG in the linear sweep
//   not on a boundary                  -> byte-scan false positive
//
// Usage: boundary STRIPPED_IMAGE

use std::collections::BTreeSet;

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
            .any(|s| s.contains_mapped_rva(Rva(rva)))
    };
    // Does this instruction reference an address no section covers?
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
        (rva < size_of_image && !mapped(rva)).then_some(rva)
    };

    let (mut agreed, mut linear_missed, mut off_boundary, mut funcs) = (0_u64, 0_u64, 0_u64, 0_u64);
    let mut missed_samples = Vec::new();

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
        funcs += 1;

        // Pass 1: the linear stream from the function entry.
        let mut boundaries: BTreeSet<u32> = BTreeSet::new();
        let mut linear_hits: BTreeSet<u32> = BTreeSet::new();
        let mut decoder = Decoder::with_ip(64, code, base + u64::from(begin), DecoderOptions::NONE);
        while decoder.can_decode() {
            let at = u32::try_from(decoder.position()).unwrap_or(0);
            let instruction = decoder.decode();
            if instruction.is_invalid() {
                break;
            }
            boundaries.insert(at);
            if dangles(&instruction).is_some() {
                linear_hits.insert(at);
            }
        }

        // Pass 2: decode at every offset, with the same lookahead filter the
        // byte scan uses, and classify against the boundary set.
        for at in 0..code.len() {
            let Some(window) = code.get(at..) else { continue };
            let rva = begin + u32::try_from(at).unwrap_or(0);
            let mut probe = Decoder::with_ip(64, window, base + u64::from(rva), DecoderOptions::NONE);
            let instruction = probe.decode();
            if instruction.is_invalid() || dangles(&instruction).is_none() {
                continue;
            }
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
            if !clean {
                continue;
            }
            let key = u32::try_from(at).unwrap_or(0);
            if boundaries.contains(&key) {
                if linear_hits.contains(&key) {
                    agreed += 1;
                } else {
                    linear_missed += 1;
                    if missed_samples.len() < 20 {
                        missed_samples.push(format!("{rva:#x} in fn {begin:#x}: {instruction}"));
                    }
                }
            } else {
                off_boundary += 1;
            }
        }
    }

    println!("{}: {funcs} function(s) decoded twice", args[1]);
    println!("\nbyte-scan candidates inside .pdata functions, classified:");
    println!("  ON an instruction boundary, linear sweep FOUND it  : {agreed}");
    println!("  ON an instruction boundary, linear sweep MISSED it : {linear_missed}");
    println!("  NOT on any instruction boundary (false positive)   : {off_boundary}");
    for sample in &missed_samples {
        println!("    missed: {sample}");
    }
}
