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
    let starts: BTreeSet<u32> = report
        .exception_functions
        .iter()
        .map(|f| f.begin.0)
        .collect();
    let is_vm = |rva: Rva| {
        pe.sections
            .iter()
            .any(|s| s.name.starts_with(".vmp") && s.contains_mapped_rva(rva))
    };
    let Some(exports) = &report.exports else {
        return;
    };

    let mut all_vm_calls: BTreeMap<u32, BTreeSet<String>> = BTreeMap::new();
    let mut all_vm_branches: BTreeMap<u32, BTreeSet<String>> = BTreeMap::new();
    let mut total_funcs = 0usize;
    for symbol in &exports.symbols {
        let name = symbol.name.clone().unwrap_or_default();
        let mut seen = BTreeSet::new();
        let mut q = VecDeque::from([symbol.rva]);
        while let Some(start) = q.pop_front() {
            if !seen.insert(start.0) {
                continue;
            }
            let Ok(cfg) = recover_cfg(
                &pe,
                start,
                CfgLimits {
                    max_blocks: 4096,
                    max_instructions: 120000,
                    max_instructions_per_block: 8192,
                },
            ) else {
                continue;
            };
            for block in &cfg.blocks {
                for succ in &block.successors {
                    if is_vm(*succ) {
                        all_vm_branches
                            .entry(succ.0)
                            .or_default()
                            .insert(name.clone());
                    } else if starts.contains(&succ.0) && !seen.contains(&succ.0) {
                        q.push_back(*succ);
                    }
                }
                for call in &block.direct_calls {
                    if is_vm(*call) {
                        all_vm_calls.entry(call.0).or_default().insert(name.clone());
                    } else if starts.contains(&call.0) && !seen.contains(&call.0) {
                        q.push_back(*call);
                    }
                }
            }
        }
        total_funcs += seen.len();
    }
    println!("total reachable functions: {total_funcs}");
    println!("VM branch targets: {}", all_vm_branches.len());
    println!("VM call targets: {}", all_vm_calls.len());
    for (t, u) in all_vm_branches.iter() {
        let off = pe.rva_to_file_offset(Rva(*t)).ok().map(|o| o.0 as usize);
        let kind = off
            .map(|o| {
                let b = &bytes[o..(o + 10).min(bytes.len())];
                if b.len() >= 10 && b[0] == 0x68 && b[5] == 0xE8 {
                    "entry_stub"
                } else if b[0] == 0xE9 {
                    "jmp_thunk"
                } else {
                    "other"
                }
            })
            .unwrap_or("?");
        println!("  branch {t:#x} {kind} users={u:?}");
    }
    for (t, u) in all_vm_calls.iter() {
        let off = pe.rva_to_file_offset(Rva(*t)).ok().map(|o| o.0 as usize);
        let kind = off
            .map(|o| {
                let b = &bytes[o..(o + 10).min(bytes.len())];
                if b.len() >= 10 && b[0] == 0x68 && b[5] == 0xE8 {
                    "entry_stub"
                } else {
                    "other"
                }
            })
            .unwrap_or("?");
        println!("  call   {t:#x} {kind} users={u:?}");
    }
}
