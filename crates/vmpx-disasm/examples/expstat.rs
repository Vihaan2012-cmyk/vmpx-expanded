use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    fs,
};
use vmpx_disasm::{CfgLimits, recover_cfg};
use vmpx_pe::{PeImage, Rva};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bytes = fs::read(&args[1]).unwrap();
    let pe = PeImage::parse(&bytes).unwrap();
    let report = pe.report();
    let is_vm = |rva: Rva| {
        pe.sections
            .iter()
            .any(|s| s.name.starts_with(".vmp") && s.contains_mapped_rva(rva))
    };
    let Some(exports) = &report.exports else {
        return;
    };
    for symbol in &exports.symbols {
        let mut seen = BTreeSet::new();
        let mut queue = VecDeque::from([symbol.rva]);
        let mut vm_refs: BTreeMap<u32, (u32, &'static str)> = BTreeMap::new();
        let mut depth = 0;
        while let Some(start) = queue.pop_front() {
            if depth > 4000 {
                break;
            }
            depth += 1;
            if !seen.insert(start.0) {
                continue;
            }
            if is_vm(start) {
                vm_refs.insert(start.0, (start.0, "entry"));
                continue;
            }
            let Ok(cfg) = recover_cfg(
                &pe,
                start,
                CfgLimits {
                    max_blocks: 2048,
                    max_instructions: 60000,
                    max_instructions_per_block: 8192,
                },
            ) else {
                continue;
            };
            for block in &cfg.blocks {
                for succ in &block.successors {
                    if is_vm(*succ) {
                        vm_refs.insert(succ.0, (block.start.0, "branch"));
                    } else if !seen.contains(&succ.0) {
                        queue.push_back(*succ);
                    }
                }
                for call in &block.direct_calls {
                    if is_vm(*call) {
                        vm_refs.insert(call.0, (block.start.0, "call"));
                    } else if !seen.contains(&call.0) {
                        queue.push_back(*call);
                    }
                }
            }
        }
        println!(
            "{:28} rva={:#08x} blocks={:5} vm_refs={}",
            symbol.name.clone().unwrap_or_default(),
            symbol.rva.0,
            seen.len(),
            vm_refs.len()
        );
        for (target, (from, kind)) in vm_refs.iter().take(4) {
            println!("      {kind} {from:#x} -> {target:#x}");
        }
    }
}
