// Bounds the three things a static dangling-reference sweep cannot see, and
// answers "how do we know so few functions were virtualized".
//
// 1. INDIRECT transfers. `call [rip+slot]` resolves (read the slot). `call reg`
//    / `jmp reg` does not -- so the honest output is a COUNT of unresolvable
//    sites, which bounds the unknown instead of pretending it is zero.
// 2. COMPUTED .vmp addresses. Any instruction that materializes a constant
//    inside a protector section -- `mov r64, imm64`, `lea r64,[rip+disp]`,
//    `mov r32, imm32` -- whether or not it is ever dereferenced.
// 3. The virtualization-STUB census. A virtualized function leaves a short
//    `.pdata`-described stub that transfers into the protector. Counting those
//    over every RUNTIME_FUNCTION bounds how many functions were virtualized,
//    independently of VM-entry discovery.
//
// Usage: audit INTACT_IMAGE
//
// Run against the INTACT image: the protector sections must still be present
// for "points into .vmp" to be answerable.

use std::collections::{BTreeMap, BTreeSet};

use iced_x86::{Decoder, DecoderOptions, FlowControl, Instruction, OpKind, Register};
use vmpx_pe::{PeImage, Rva};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bytes = std::fs::read(&args[1]).expect("image");
    let pe = PeImage::parse(&bytes).expect("parse");
    let base = pe.optional.image_base.get();

    let section_of = |rva: u32| -> Option<&str> {
        pe.sections
            .iter()
            .find(|s| s.contains_mapped_rva(Rva(rva)))
            .map(|s| s.name.as_str())
    };
    let is_protector = |rva: u32| section_of(rva).is_some_and(|n| n.starts_with(".vmp"));
    let qword_at = |rva: u32| -> Option<u64> {
        let offset = pe.rva_to_file_offset(Rva(rva)).ok()?;
        let at = usize::try_from(offset.0).ok()?;
        bytes
            .get(at..at.checked_add(8)?)
            .and_then(|s| s.try_into().ok())
            .map(u64::from_le_bytes)
    };
    let to_rva = |va: u64| -> Option<u32> { va.checked_sub(base).and_then(|v| u32::try_from(v).ok()) };

    // Counters
    let mut reg_indirect = 0_u64;
    let mut mem_indirect_unresolvable = 0_u64;
    let mut mem_indirect_resolved_vmp = BTreeSet::new();
    let mut mem_indirect_resolved_other = 0_u64;
    let mut computed: BTreeMap<u32, u64> = BTreeMap::new();
    let mut stub_candidates: Vec<(u32, u32, u32)> = Vec::new(); // (fn begin, len, target)
    let mut functions = 0_u64;
    let mut size_hist: BTreeMap<&str, u64> = BTreeMap::new();

    let note_computed = |computed: &mut BTreeMap<u32, u64>, value: u64| {
        if let Some(rva) = value.checked_sub(base).and_then(|v| u32::try_from(v).ok())
            && rva < pe.optional.size_of_image
        {
            *computed.entry(rva).or_default() += 1;
        }
    };

    for function in &pe.report().exception_functions {
        let (begin, end) = (function.begin.0, function.end.0);
        if end <= begin || end - begin > 0x10_0000 {
            continue;
        }
        if section_of(begin).is_none_or(|n| n.starts_with(".vmp")) {
            continue; // only audit the application's own code
        }
        let Ok(offset) = pe.rva_to_file_offset(Rva(begin)) else {
            continue;
        };
        let start = offset.0 as usize;
        let stop = start.saturating_add((end - begin) as usize).min(bytes.len());
        let Some(code) = bytes.get(start..stop) else {
            continue;
        };
        functions += 1;
        let length = end - begin;
        let bucket = match length {
            0..=16 => "<=16",
            17..=32 => "17-32",
            33..=64 => "33-64",
            65..=128 => "65-128",
            _ => ">128",
        };
        *size_hist.entry(bucket).or_default() += 1;

        let mut decoder = Decoder::with_ip(64, code, base + u64::from(begin), DecoderOptions::NONE);
        let mut instructions = 0_u32;
        let mut protector_target = None;
        while decoder.can_decode() {
            let instruction = decoder.decode();
            if instruction.is_invalid() {
                break;
            }
            instructions += 1;

            // --- 1. indirect transfers ---
            if matches!(
                instruction.flow_control(),
                FlowControl::IndirectCall | FlowControl::IndirectBranch
            ) {
                match instruction.op0_kind() {
                    OpKind::Register => reg_indirect += 1,
                    OpKind::Memory if instruction.memory_base() == Register::RIP => {
                        let slot = to_rva(instruction.ip_rel_memory_address());
                        match slot.and_then(qword_at).and_then(to_rva) {
                            Some(target) if is_protector(target) => {
                                mem_indirect_resolved_vmp.insert(target);
                            }
                            Some(_) => mem_indirect_resolved_other += 1,
                            None => mem_indirect_unresolvable += 1,
                        }
                    }
                    _ => mem_indirect_unresolvable += 1,
                }
            }

            // --- 2. computed/materialized protector addresses ---
            if instruction.memory_base() == Register::RIP {
                note_computed(&mut computed, instruction.ip_rel_memory_address());
            }
            for op in 0..instruction.op_count() {
                match instruction.op_kind(op) {
                    OpKind::Immediate64 => note_computed(&mut computed, instruction.immediate64()),
                    OpKind::Immediate32to64 => {
                        note_computed(&mut computed, instruction.immediate32to64().cast_unsigned());
                    }
                    OpKind::Immediate32 => {
                        note_computed(&mut computed, u64::from(instruction.immediate32()));
                    }
                    _ => {}
                }
            }

            // --- 3. stub shape: a direct transfer into a protector section ---
            if matches!(
                instruction.flow_control(),
                FlowControl::Call | FlowControl::UnconditionalBranch
            ) && instruction.near_branch64() != 0
                && let Some(target) = to_rva(instruction.near_branch64())
                && is_protector(target)
            {
                protector_target = Some(target);
            }
        }
        // Every application function that transfers into the protector at all,
        // with its instruction count -- so a short STUB (a virtualized body
        // replaced wholesale) is distinguishable from an intact function that
        // merely CALLS a guard.
        if let Some(target) = protector_target {
            stub_candidates.push((begin, instructions, target));
        }
        let _ = Instruction::default();
    }

    println!("audited {functions} application RUNTIME_FUNCTION(s) in {}", args[1]);
    println!("\n=== 1. indirect transfers (what a static sweep cannot resolve) ===");
    println!("  call/jmp through a register (UNRESOLVABLE statically) : {reg_indirect}");
    println!("  call/jmp [rip+slot], slot holds a .vmp address        : {}", mem_indirect_resolved_vmp.len());
    println!("  call/jmp [rip+slot], slot holds a non-.vmp address    : {mem_indirect_resolved_other}");
    println!("  call/jmp through memory, slot not statically readable : {mem_indirect_unresolvable}");
    for target in &mem_indirect_resolved_vmp {
        println!("    -> {target:#x} ({})", section_of(*target).unwrap_or("?"));
    }

    println!("\n=== 2. protector addresses MATERIALIZED as a constant ===");
    let into_vmp: Vec<_> = computed.iter().filter(|(rva, _)| is_protector(**rva)).collect();
    println!(
        "  distinct constants naming a .vmp address : {} (over {} sites)",
        into_vmp.len(),
        into_vmp.iter().map(|(_, n)| **n).sum::<u64>()
    );
    for (rva, count) in into_vmp.iter().take(24) {
        println!("    {rva:#x} ({}) x{count}", section_of(**rva).unwrap_or("?"));
    }

    println!("\n=== 3. virtualization-stub census ===");
    println!("  application functions transferring into a protector section: {}", stub_candidates.len());
    let stubs = stub_candidates.iter().filter(|(_, n, _)| *n <= 24).count();
    println!("  of those, SHORT enough to be a replaced body (<=24 insns): {stubs}");
    let targets: BTreeSet<u32> = stub_candidates.iter().map(|(_, _, t)| *t).collect();
    println!("  distinct protector targets they reach      : {}", targets.len());
    for (begin, insns, target) in stub_candidates.iter().take(40) {
        println!("    fn {begin:#x} {insns} insn(s) -> {target:#x}");
    }
    println!("\n  application function size histogram:");
    for (bucket, count) in &size_hist {
        println!("    {bucket:>7} bytes : {count}");
    }
}
