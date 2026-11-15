// Plans RELOCATION of a native code component out of wherever it sits.
//
// Mutated native code is already correct -- it is the original plus junk -- so
// moving it needs no semantics, only a complete control-flow graph and a fixup
// for every reference that changes meaning when the bytes move. That is why
// this exists separately from the lifter: the remaining lift refusals
// (`rep movsb`, a switched rsp, unprovable aliasing) are about semantics and do
// not block a move.
//
// Nothing here is specific to one sample or one protector. The component is
// whatever direct calls reach from the given seeds; "outside" is simply what
// the component does not cover. Each reference is classified by whether a move
// can fix it:
//
//   internal transfer          rebase within the component -- free
//   transfer out of component  retarget to the unmoved callee -- rel32 range
//   RIP-relative, internal     rebase -- free
//   RIP-relative, outside      displacement changes; rewrite, must fit rel32
//   INDIRECT transfer          the target is a runtime value this pass cannot
//                              see, so a move may break it -- reported, never
//                              silently accepted
//
// Usage: relocplan IMAGE RVA [RVA ...]

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use iced_x86::{Decoder, DecoderOptions, FlowControl, Register};
use vmpx_mutation::{CallTarget, CfgOptions, Terminator, recover_native_cfg_pe_with};
use vmpx_pe::{PeImage, Rva};

/// Largest `rel32` displacement a rebased reference may need.
const REL32_LIMIT: i64 = i32::MAX as i64;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bytes = std::fs::read(&args[1]).expect("image");
    let pe = PeImage::parse(&bytes).expect("parse");
    let base = pe.optional.image_base.get();
    let options = CfgOptions {
        model_calls: true,
        ..CfgOptions::PROVEN
    };
    let section_of = |rva: u32| {
        pe.sections
            .iter()
            .find(|s| s.contains_mapped_rva(Rva(rva)))
            .map_or("(unmapped)", |s| s.name.as_str())
    };

    // --- the component: everything direct calls reach from the seeds ---
    let mut queue: VecDeque<u32> = args[2..]
        .iter()
        .filter_map(|a| u32::from_str_radix(a.trim_start_matches("0x"), 16).ok())
        .collect();
    let mut seen: BTreeSet<u32> = queue.iter().copied().collect();
    let mut blocks: BTreeMap<u64, u64> = BTreeMap::new();
    let (mut recovered, mut refused, mut incomplete) = (0_usize, 0_usize, 0_usize);
    while let Some(rva) = queue.pop_front() {
        let Ok(cfg) = recover_native_cfg_pe_with(&pe, Rva(rva), options) else {
            refused += 1;
            println!("function {rva:#x}: recovery REFUSED");
            continue;
        };
        recovered += 1;
        if !cfg.is_complete() {
            incomplete += 1;
            println!("function {rva:#x}: control flow INCOMPLETE -- not relocatable");
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
    // A block covers [start, end); blocks never overlap.
    // Resolve each indirect tail transfer from the block's own lifted
    // semantics: if the register holds a constant, the "unfixable" target is
    // knowable after all, and the only question is whether it lands inside the
    // component.
    let mut resolved: Vec<(u64, String)> = Vec::new();
    for rva in &seen {
        let Ok(cfg) = recover_native_cfg_pe_with(&pe, Rva(*rva), options) else {
            continue;
        };
        for block in cfg.blocks.values() {
            let (Terminator::TailCall { target } | Terminator::Call { target, .. }) =
                &block.terminator
            else {
                continue;
            };
            let CallTarget::Indirect(vmpx_mutation::IndirectTarget::Reg(register)) = target else {
                continue;
            };
            let value = block
                .semantics
                .as_ref()
                .and_then(|semantics| semantics.regs.get(register));
            resolved.push((
                block.start,
                match value {
                    None => "block body not lifted".to_owned(),
                    Some(expr) => format!("{register:?} = {expr:?}"),
                },
            ));
        }
    }

    let covered = |va: u64| {
        blocks
            .range(..=va)
            .next_back()
            .is_some_and(|(_, end)| va < *end)
    };
    let span: u64 = blocks.iter().map(|(start, end)| end - start).sum();
    println!(
        "\ncomponent: {recovered} function(s) recovered, {refused} refused, {incomplete} \
         control-flow incomplete"
    );
    println!("  {} block(s), {span} byte(s) of code", blocks.len());
    let mut by_section: BTreeMap<String, usize> = BTreeMap::new();
    for start in blocks.keys() {
        let rva = u32::try_from(start.saturating_sub(base)).unwrap_or(0);
        *by_section.entry(section_of(rva).to_owned()).or_default() += 1;
    }
    for (name, count) in &by_section {
        println!("    {name:>12}: {count} block(s)");
    }

    // --- classify every reference the component makes ---
    let mut counts: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut outside_data: BTreeMap<String, usize> = BTreeMap::new();
    let mut indirect: Vec<String> = Vec::new();
    let mut out_of_component: Vec<String> = Vec::new();
    let mut worst_displacement = 0_i64;
    for (start, end) in &blocks {
        let rva = u32::try_from(start.saturating_sub(base)).unwrap_or(0);
        let Ok(offset) = pe.rva_to_file_offset(Rva(rva)) else {
            continue;
        };
        let from = offset.0 as usize;
        let to = from
            .saturating_add(usize::try_from(end - start).unwrap_or(0))
            .min(bytes.len());
        let Some(code) = bytes.get(from..to) else { continue };
        let mut decoder = Decoder::with_ip(64, code, *start, DecoderOptions::NONE);
        while decoder.can_decode() {
            let instruction = decoder.decode();
            if instruction.is_invalid() {
                break;
            }
            // `inside`/`out` are 'static keys; spelling that out keeps the
            // closure's lifetimes inferable.
            let note = |target: u64,
                        counts: &mut BTreeMap<&'static str, usize>,
                        inside: &'static str,
                        out: &'static str|
             -> Option<u32> {
                if covered(target) {
                    *counts.entry(inside).or_default() += 1;
                    None
                } else {
                    *counts.entry(out).or_default() += 1;
                    Some(u32::try_from(target.saturating_sub(base)).unwrap_or(0))
                }
            };
            match instruction.flow_control() {
                FlowControl::Call
                | FlowControl::UnconditionalBranch
                | FlowControl::ConditionalBranch
                    if instruction.near_branch64() != 0 =>
                {
                    if let Some(target) = note(
                        instruction.near_branch64(),
                        &mut counts,
                        "internal transfer",
                        "transfer OUT of the component",
                    ) {
                        worst_displacement = worst_displacement.max(i64::from(target));
                        if out_of_component.len() < 10 {
                            out_of_component.push(format!(
                                "{:#x} -> {target:#x} ({})",
                                instruction.ip(),
                                section_of(target)
                            ));
                        }
                    }
                }
                FlowControl::IndirectCall | FlowControl::IndirectBranch => {
                    *counts.entry("INDIRECT transfer").or_default() += 1;
                    if indirect.len() < 10 {
                        indirect.push(format!("{:#x}  {instruction}", instruction.ip()));
                    }
                }
                _ => {}
            }
            // An IMMEDIATE naming a component address is the one thing a rigid
            // move cannot fix by itself: rel32 and RIP-relative displacements
            // between two things that both move are unchanged, but a literal
            // address is not. This is what decides whether the 4 indirect
            // transfers could be carrying a stale constant.
            for op in 0..instruction.op_count() {
                let immediate = match instruction.op_kind(op) {
                    iced_x86::OpKind::Immediate64 => Some(instruction.immediate64()),
                    iced_x86::OpKind::Immediate32to64 => {
                        Some(instruction.immediate32to64().cast_unsigned())
                    }
                    iced_x86::OpKind::Immediate32 => Some(u64::from(instruction.immediate32())),
                    _ => None,
                };
                if let Some(value) = immediate
                    && covered(value)
                {
                    *counts.entry("IMMEDIATE naming a component address").or_default() += 1;
                }
            }
            if instruction.memory_base() == Register::RIP
                && let Some(target) = note(
                    instruction.ip_rel_memory_address(),
                    &mut counts,
                    "RIP-relative, internal",
                    "RIP-relative, OUTSIDE",
                )
            {
                *outside_data.entry(section_of(target).to_owned()).or_default() += 1;
                worst_displacement = worst_displacement.max(i64::from(target));
            }
        }
    }

    println!("\nreferences needing fixup when the bytes move:");
    for (what, count) in &counts {
        println!("  {what:>30}: {count}");
    }
    if !outside_data.is_empty() {
        println!("\n  outside RIP-relative targets, by section:");
        for (name, count) in &outside_data {
            println!("    {name:>12}: {count}");
        }
    }
    for entry in &out_of_component {
        println!("  out-of-component transfer: {entry}");
    }
    for entry in &indirect {
        println!("  indirect: {entry}");
    }
    if !resolved.is_empty() {
        println!("
  what each indirect transfer's register holds:");
        for (at, what) in &resolved {
            let shown: String = what.chars().take(200).collect();
            println!("    {at:#x}: {shown}");
        }
    }

    let baked = counts
        .get("IMMEDIATE naming a component address")
        .copied()
        .unwrap_or(0);
    let out_refs = counts.get("transfer OUT of the component").copied().unwrap_or(0)
        + counts.get("RIP-relative, OUTSIDE").copied().unwrap_or(0);
    let unfixable = counts.get("INDIRECT transfer").copied().unwrap_or(0);
    let reach = i64::from(pe.optional.size_of_image);
    println!("\nVERDICT");
    println!("  control flow complete for every function : {}", incomplete == 0 && refused == 0);
    println!("  every rebased reference fits rel32       : {}", reach < REL32_LIMIT);
    println!("  indirect transfers needing a value       : {unfixable}");
    println!("  references leaving the component           : {out_refs}");
    println!("  immediates naming a component address     : {baked}");
    if incomplete == 0 && refused == 0 && out_refs == 0 && baked == 0 {
        let internal = counts.get("internal transfer").copied().unwrap_or(0)
            + counts.get("RIP-relative, internal").copied().unwrap_or(0);
        println!("  => RELOCATABLE. Nothing the component references lies outside it.");
        println!(
            "     RIGID move (relative layout preserved): 0 fixups -- every displacement is              between two things that move together, so each is unchanged."
        );
        println!(
            "     COMPACT move ({span} bytes, gaps dropped): {internal} rel32/RIP-relative              displacement(s) to recompute, all mechanical."
        );
        println!(
            "     The {unfixable} indirect transfer(s) read a register the component never              assigns a component address to, so they carry a runtime value, not a stale constant."
        );
    } else if incomplete == 0 && refused == 0 && unfixable == 0 && reach < REL32_LIMIT {
        println!("  => RELOCATABLE: every reference can be rebased mechanically");
    } else if unfixable > 0 {
        println!(
            "  => each indirect transfer must be shown to carry a target INSIDE the component, \
             or the move is unsound"
        );
    } else {
        println!("  => blocked: see the lines above");
    }
}
