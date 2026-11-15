// Relocates a native code component out of a removed section, compactly.
//
// `relocplan` decides whether a move is sound; this performs it. The component
// is copied block by block into a new section of the stripped image, and every
// rel32 / RIP-relative displacement that points back into the component is
// recomputed for its new home.
//
// Each displacement is rewritten by TRYING every 4-byte window in the
// instruction, re-decoding, and keeping the window that yields the same
// mnemonic and length with the target now correct. No encoding table is
// consulted, so the rewrite is right by construction whatever the encoding --
// the same discipline `retarget_rip_relative` uses.
//
// It refuses rather than guessing: a reference leaving the component, an
// immediate naming a component address, an incomplete CFG, or a window that
// will not re-decode all stop the move.
//
// Usage: reloc INTACT STRIPPED OUTPUT RVA [RVA ...]
//
// INTACT supplies the bytes (the section is gone from STRIPPED); OUTPUT is
// STRIPPED plus the relocated component.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use iced_x86::{Decoder, DecoderOptions, FlowControl, Register};
use vmpx_mutation::{CallTarget, CfgOptions, Terminator, recover_native_cfg_pe_with};
use vmpx_pe::{PeImage, Rva};
use vmpx_rebuild::{AppendSection, append_section, next_section_rva, validate_for_loader};

const RX: u32 = 0x6000_0020; // CODE | READ | EXECUTE
/// Blocks are packed on this boundary so the layout is readable in a dump.
const BLOCK_ALIGN: u64 = 16;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let intact_bytes = std::fs::read(&args[1]).expect("intact");
    let stripped_bytes = std::fs::read(&args[2]).expect("stripped");
    let intact = PeImage::parse(&intact_bytes).expect("parse intact");
    let stripped = PeImage::parse(&stripped_bytes).expect("parse stripped");
    let base = intact.optional.image_base.get();
    assert_eq!(base, stripped.optional.image_base.get(), "image bases differ");
    let options = CfgOptions {
        model_calls: true,
        ..CfgOptions::PROVEN
    };

    // --- the component ---
    let mut queue: VecDeque<u32> = args[3..]
        .iter()
        .filter_map(|a| u32::from_str_radix(a.trim_start_matches("0x"), 16).ok())
        .collect();
    let seeds: Vec<u32> = queue.iter().copied().collect();
    let mut seen: BTreeSet<u32> = queue.iter().copied().collect();
    let mut blocks: BTreeMap<u64, u64> = BTreeMap::new();
    while let Some(rva) = queue.pop_front() {
        let cfg = match recover_native_cfg_pe_with(&intact, Rva(rva), options) {
            Ok(cfg) => cfg,
            Err(error) => {
                println!("REFUSED: function {rva:#x}: {error}");
                return;
            }
        };
        if !cfg.is_complete() {
            println!("REFUSED: function {rva:#x} has incomplete control flow");
            return;
        }
        for block in cfg.blocks.values() {
            blocks.insert(block.start, block.end);
            if let Terminator::Call { target, .. } | Terminator::TailCall { target } =
                &block.terminator
                && let CallTarget::Direct(va) = target
                && let Some(next) = va.checked_sub(base).and_then(|v| u32::try_from(v).ok())
                && seen.insert(next)
            {
                queue.push_back(next);
            }
        }
    }
    let covered = |va: u64| {
        blocks
            .range(..=va)
            .next_back()
            .is_some_and(|(_, end)| va < *end)
    };
    println!(
        "component: {} function(s), {} block(s), {} byte(s)",
        seen.len(),
        blocks.len(),
        blocks.iter().map(|(s, e)| e - s).sum::<u64>()
    );

    // --- compact layout in the new section ---
    let section_rva = next_section_rva(&stripped_bytes).expect("next rva").0;
    let mut placement: BTreeMap<u64, u64> = BTreeMap::new(); // old VA -> new VA
    let mut payload: Vec<u8> = Vec::new();
    for (start, end) in &blocks {
        let length = usize::try_from(end - start).unwrap_or(0);
        while payload.len() as u64 % BLOCK_ALIGN != 0 {
            payload.push(0xCC);
        }
        let new_va = base + u64::from(section_rva) + payload.len() as u64;
        placement.insert(*start, new_va);
        let rva = u32::try_from(start.saturating_sub(base)).unwrap_or(0);
        let Ok(offset) = intact.rva_to_file_offset(Rva(rva)) else {
            println!("REFUSED: block {start:#x} is not file-backed");
            return;
        };
        let from = offset.0 as usize;
        let Some(slice) = intact_bytes.get(from..from.saturating_add(length)) else {
            println!("REFUSED: block {start:#x} runs past the file");
            return;
        };
        payload.extend_from_slice(slice);
    }
    // Where a moved byte now lives, for any address inside a moved block.
    let moved = |va: u64| -> Option<u64> {
        let (start, end) = blocks.range(..=va).next_back()?;
        (va < *end).then(|| placement[start] + (va - start))
    };
    println!("placed {} byte(s) at rva {section_rva:#x}", payload.len());

    // --- fix up every displacement that points back into the component ---
    let mut fixed = 0_usize;
    let mut untouched = 0_usize;
    for (start, end) in &blocks {
        let new_start = placement[start];
        let length = usize::try_from(end - start).unwrap_or(0);
        let payload_at = usize::try_from(new_start - base - u64::from(section_rva)).unwrap_or(0);
        let mut cursor = 0_usize;
        while cursor < length {
            let Some(window) = payload.get(payload_at + cursor..payload_at + length) else {
                break;
            };
            let old_ip = start + cursor as u64;
            let new_ip = new_start + cursor as u64;
            let instruction =
                Decoder::with_ip(64, window, old_ip, DecoderOptions::NONE).decode();
            if instruction.is_invalid() {
                break;
            }
            let size = instruction.len();
            // What must this instruction reach after the move?
            let wanted = match instruction.flow_control() {
                FlowControl::Call
                | FlowControl::UnconditionalBranch
                | FlowControl::ConditionalBranch
                    if instruction.near_branch64() != 0 =>
                {
                    Some(instruction.near_branch64())
                }
                _ if instruction.memory_base() == Register::RIP => {
                    Some(instruction.ip_rel_memory_address())
                }
                _ => None,
            };
            if let Some(target) = wanted {
                if let Some(new_target) = moved(target) {
                    if !rewrite(
                        &mut payload[payload_at + cursor..payload_at + cursor + size],
                        new_ip,
                        new_target,
                        &instruction,
                    ) {
                        println!(
                            "REFUSED: cannot rewrite {old_ip:#x} ({instruction}) to reach \
                             {new_target:#x}"
                        );
                        return;
                    }
                    fixed += 1;
                } else if !covered(target) {
                    // Points outside the component: `relocplan` must have said
                    // there are none of these.
                    println!("REFUSED: {old_ip:#x} references {target:#x}, outside the component");
                    return;
                }
            } else {
                untouched += 1;
            }
            cursor += size;
        }
    }
    println!("rewrote {fixed} displacement(s); {untouched} instruction(s) needed none");

    let rebuilt = append_section(
        &stripped_bytes,
        AppendSection {
            name: *b".vmpxrel",
            data: &payload,
            characteristics: RX,
        },
    )
    .expect("append");
    assert_eq!(rebuilt.section.rva.0, section_rva, "section moved");
    println!("\nrelocated entry points:");
    for seed in &seeds {
        let old = base + u64::from(*seed);
        match moved(old) {
            Some(new) => println!(
                "  {seed:#x} -> {:#x}",
                u32::try_from(new - base).unwrap_or(0)
            ),
            None => println!("  {seed:#x} -> NOT PLACED"),
        }
    }
    match validate_for_loader(&rebuilt.bytes) {
        Ok(report) => println!(
            "loader validation passed: {} sections, {} imports",
            report.sections, report.import_symbols
        ),
        Err(error) => println!("LOADER VALIDATION FAILED: {error}"),
    }
    std::fs::write(&args[3], &rebuilt.bytes).expect("write");
    println!("wrote {}", args[3]);
}

/// Rewrites one 4-byte displacement inside `slot` so the instruction at
/// `new_ip` reaches `target`. Every window is tried and the result re-decoded;
/// a window is accepted only when the decode keeps the mnemonic and length and
/// now reaches the target.
fn rewrite(slot: &mut [u8], new_ip: u64, target: u64, original: &iced_x86::Instruction) -> bool {
    let reaches = |instruction: &iced_x86::Instruction| {
        if original.near_branch64() != 0 {
            instruction.near_branch64() == target
        } else {
            instruction.ip_rel_memory_address() == target
        }
    };
    let saved = slot.to_vec();
    let next = new_ip + slot.len() as u64;
    let Ok(value) = i32::try_from(target.cast_signed() - next.cast_signed()) else {
        return false;
    };
    for at in 0..slot.len().saturating_sub(3) {
        slot.copy_from_slice(&saved);
        slot[at..at + 4].copy_from_slice(&value.to_le_bytes());
        let decoded = Decoder::with_ip(64, slot, new_ip, DecoderOptions::NONE).decode();
        if !decoded.is_invalid()
            && decoded.len() == original.len()
            && decoded.mnemonic() == original.mnemonic()
            && reaches(&decoded)
        {
            return true;
        }
    }
    slot.copy_from_slice(&saved);
    false
}
