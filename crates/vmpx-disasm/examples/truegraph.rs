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

    // Ground truth function starts from .pdata
    let starts: BTreeSet<u32> = report
        .exception_functions
        .iter()
        .map(|f| f.begin.0)
        .collect();
    println!("pdata function starts: {}", starts.len());

    // An entry stub is exactly: push imm32 ; call rel32 where target is in a VM section
    let stub = |rva: Rva| -> Option<u32> {
        let off = pe.rva_to_file_offset(rva).ok()?.0 as usize;
        let b = bytes.get(off..off + 10)?;
        if b[0] != 0x68 || b[5] != 0xE8 {
            return None;
        }
        let rel = i32::from_le_bytes([b[6], b[7], b[8], b[9]]);
        let target = (rva.0 as i64 + 10 + i64::from(rel)) as u32;
        is_vm(Rva(target)).then_some(target)
    };

    let Some(exports) = &report.exports else {
        return;
    };
    let mut all_stubs: BTreeMap<u32, BTreeSet<String>> = BTreeMap::new();
    let mut vm_branches: BTreeMap<u32, BTreeSet<String>> = BTreeMap::new();
    for symbol in &exports.symbols {
        let name = symbol.name.clone().unwrap_or_default();
        let mut seen = BTreeSet::new();
        let mut q = VecDeque::from([symbol.rva]);
        let mut stubs = BTreeSet::new();
        while let Some(start) = q.pop_front() {
            if !seen.insert(start.0) {
                continue;
            }
            if is_vm(start) {
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
                // only follow into blocks that are genuine function starts OR within this function
                for succ in &block.successors {
                    if is_vm(*succ) {
                        vm_branches.entry(succ.0).or_default().insert(name.clone());
                        continue;
                    }
                    if starts.contains(&succ.0) && !seen.contains(&succ.0) {
                        q.push_back(*succ);
                    }
                }
                for call in &block.direct_calls {
                    if let Some(bootstrap) = stub(*call) {
                        stubs.insert(*call);
                        all_stubs.entry(call.0).or_default().insert(name.clone());
                        let _ = bootstrap;
                    } else if starts.contains(&call.0) && !seen.contains(&call.0) {
                        q.push_back(*call);
                    }
                }
            }
        }
        let branches = vm_branches
            .iter()
            .filter(|(_, u)| u.contains(&name))
            .count();
        println!(
            "{:28} blocks={:6} real_vm_stubs={:3} vm_branches={:3}",
            name,
            seen.len(),
            stubs.len(),
            branches
        );
        for s in stubs.iter().take(6) {
            println!("      {:#x} -> {:#x}", s.0, stub(*s).unwrap_or(0));
        }
    }
    println!(
        "\ntotal distinct real VM entry stubs reachable from exports: {}",
        all_stubs.len()
    );
    let mut by_user: BTreeMap<String, usize> = BTreeMap::new();
    for users in all_stubs.values() {
        for u in users {
            *by_user.entry(u.clone()).or_default() += 1;
        }
    }
    println!("{by_user:?}");
    println!("\ndistinct VM branch targets: {}", vm_branches.len());
    for (t, users) in vm_branches.iter().take(15) {
        println!("  {t:#x} {users:?}");
    }
}
