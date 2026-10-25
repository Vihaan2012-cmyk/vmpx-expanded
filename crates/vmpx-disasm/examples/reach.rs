use std::{collections::{BTreeMap, BTreeSet, VecDeque}, fs};
use vmpx_disasm::{CfgLimits, recover_cfg};
use vmpx_pe::PeImage;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bytes = fs::read(&args[1]).unwrap();
    let pe = PeImage::parse(&bytes).unwrap();
    let is_vm = |rva: u32| pe.sections.iter().any(|s| s.name.starts_with(".vmp") && s.contains_mapped_rva(vmpx_pe::Rva(rva)));

    // Roots: exports, TLS callbacks, entry point
    let report = pe.report();
    let mut roots: Vec<(String, u32)> = Vec::new();
    if let Some(exports) = &report.exports {
        for symbol in &exports.symbols {
            roots.push((format!("export {}", symbol.name.clone().unwrap_or_default()), symbol.rva.0));
        }
    }
    if let Some(tls) = &report.tls {
        for (i, cb) in tls.callbacks.iter().enumerate() { if let Some(rva) = pe.optional.image_base.checked_rva_of(*cb) { roots.push((format!("tls{i}"), rva.0)); } }
    }
    roots.push(("entry".to_owned(), pe.optional.address_of_entry_point.0));

    let mut seen: BTreeSet<u32> = BTreeSet::new();
    let mut queue: VecDeque<u32> = VecDeque::new();
    let mut origin: BTreeMap<u32, String> = BTreeMap::new();
    for (name, rva) in &roots { queue.push_back(*rva); origin.insert(*rva, name.clone()); }

    let mut vm_hits: Vec<(String, u32, u32)> = Vec::new();
    let mut blocks_total = 0usize;
    let mut calls: BTreeSet<u32> = BTreeSet::new();
    while let Some(rva) = queue.pop_front() {
        if !seen.insert(rva) { continue; }
        if is_vm(rva) { vm_hits.push((origin.get(&rva).cloned().unwrap_or_default(), rva, rva)); continue; }
        let Ok(cfg) = recover_cfg(&pe, vmpx_pe::Rva(rva), CfgLimits { max_blocks: 4096, max_instructions: 200_000, max_instructions_per_block: 8192 }) else { continue };
        for block in &cfg.blocks {
            blocks_total += 1;
            for succ in &block.successors {
                if is_vm(succ.0) { vm_hits.push((origin.get(&rva).cloned().unwrap_or_default(), rva, succ.0)); }
                else if !seen.contains(&succ.0) { let o = origin.get(&rva).cloned().unwrap_or_default(); origin.entry(succ.0).or_insert(o); queue.push_back(succ.0); }
            }
            for call in &block.direct_calls {
                if is_vm(call.0) { vm_hits.push((origin.get(&rva).cloned().unwrap_or_default(), rva, call.0)); }
                else if calls.insert(call.0) { let o = origin.get(&rva).cloned().unwrap_or_default(); origin.entry(call.0).or_insert(o); queue.push_back(call.0); }
            }
        }
    }
    println!("roots: {}", roots.len());
    println!("reachable blocks: {blocks_total}, distinct starts: {}", seen.len());
    println!("direct calls discovered: {}", calls.len());
    println!("references into VM sections: {}", vm_hits.len());
    for (origin_name, from, target) in vm_hits.iter().take(20) {
        println!("  from {origin_name} {from:#x} -> {target:#x}");
    }
    let mut by_origin: BTreeMap<&str, usize> = BTreeMap::new();
    for (o, _, _) in &vm_hits { *by_origin.entry(o.as_str()).or_default() += 1; }
    println!("by origin: {by_origin:?}");
}
