use iced_x86::{
    Decoder, DecoderOptions, FlowControl, Formatter, Instruction, IntelFormatter, OpKind, Register,
};
use std::{collections::BTreeMap, fs};
use vmpx_pe::PeImage;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bytes = fs::read(&args[1]).unwrap();
    let pe = PeImage::parse(&bytes).unwrap();
    let base = pe.optional.image_base.0;
    let vm: Vec<(String, u64, u64)> = pe
        .sections
        .iter()
        .filter(|s| s.name.starts_with(".vmp"))
        .map(|s| {
            (
                s.name.clone(),
                u64::from(s.virtual_address.0) + base,
                u64::from(s.virtual_address.0) + base + u64::from(s.mapped_span()),
            )
        })
        .collect();
    let in_vm = |a: u64| {
        vm.iter()
            .find(|(_, s, e)| a >= *s && a < *e)
            .map(|(n, _, _)| n.clone())
    };
    let mut counts: BTreeMap<(String, String, String), usize> = BTreeMap::new();
    for sec in pe
        .sections
        .iter()
        .filter(|s| s.characteristics & 0x2000_0000 != 0 && !s.name.starts_with(".vmp"))
    {
        let Ok(start) = pe.rva_to_file_offset(sec.virtual_address) else {
            continue;
        };
        let data = &bytes
            [start.0 as usize..(start.0 as usize + sec.size_of_raw_data as usize).min(bytes.len())];
        let mut dec = Decoder::with_ip(
            64,
            data,
            u64::from(sec.virtual_address.0) + base,
            DecoderOptions::NONE,
        );
        let mut ins = Instruction::default();
        while dec.can_decode() {
            ins = dec.decode();
            if ins.is_invalid() {
                continue;
            }
            let mut targets: Vec<u64> = Vec::new();
            for i in 0..ins.op_count() {
                if ins.op_kind(i) == OpKind::NearBranch64 || ins.op_kind(i) == OpKind::NearBranch32
                {
                    targets.push(ins.near_branch64());
                }
                if ins.op_kind(i) == OpKind::Memory && ins.is_ip_rel_memory_operand() {
                    targets.push(ins.ip_rel_memory_address());
                }
            }
            for t in targets {
                if let Some(n) = in_vm(t) {
                    let kind = match ins.flow_control() {
                        FlowControl::Call => "call",
                        FlowControl::UnconditionalBranch => "jmp",
                        FlowControl::ConditionalBranch => "jcc",
                        _ => "mem",
                    };
                    *counts
                        .entry((sec.name.clone(), n, kind.to_string()))
                        .or_default() += 1;
                    if kind == "call" || kind == "jmp" {
                        println!("  {} at {:#x} -> {:#x}", kind, ins.ip(), t);
                    }
                }
            }
        }
    }
    println!("refs from non-VM executable sections into VM sections:");
    for ((src, sec, kind), c) in &counts {
        println!("  {src} -> {sec} {kind} x{c}");
    }
    if counts.is_empty() {
        println!("  NONE");
    }
}
