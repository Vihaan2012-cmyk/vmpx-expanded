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

    // Reachable non-VM blocks across all exports, and the VM stubs they enter.
    let mut seen: BTreeSet<u32> = BTreeSet::new();
    let mut queue: VecDeque<Rva> = VecDeque::new();
    let mut stubs: BTreeMap<u32, BTreeSet<String>> = BTreeMap::new();
    for symbol in &exports.symbols {
        queue.push_back(symbol.rva);
        let name = symbol.name.clone().unwrap_or_default();
        // record origin lazily by re-walking per export below
        let _ = name;
    }
    let _ = &mut seen;
    let _ = &mut queue;

    for symbol in &exports.symbols {
        let name = symbol.name.clone().unwrap_or_default();
        let mut local = BTreeSet::new();
        let mut q = VecDeque::from([symbol.rva]);
        while let Some(start) = q.pop_front() {
            if !local.insert(start.0) {
                continue;
            }
            if is_vm(start) {
                stubs.entry(start.0).or_default().insert(name.clone());
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
                for s in &block.successors {
                    if is_vm(*s) {
                        stubs.entry(s.0).or_default().insert(name.clone());
                    } else if !local.contains(&s.0) {
                        q.push_back(*s);
                    }
                }
                for c in &block.direct_calls {
                    if is_vm(*c) {
                        stubs.entry(c.0).or_default().insert(name.clone());
                    } else if !local.contains(&c.0) {
                        q.push_back(*c);
                    }
                }
            }
        }
    }
    println!(
        "distinct VM entry targets reachable from exports: {}",
        stubs.len()
    );
    let mut by_sec: BTreeMap<&str, usize> = BTreeMap::new();
    for target in stubs.keys() {
        let sec = pe
            .sections
            .iter()
            .find(|s| s.contains_mapped_rva(Rva(*target)))
            .map(|s| s.name.as_str())
            .unwrap_or("?");
        *by_sec.entry(sec).or_default() += 1;
    }
    println!("by section: {by_sec:?}");
    let mut shown = 0;
    for (target, users) in &stubs {
        if shown >= 12 {
            break;
        }
        // classify: push imm32; call rel32 => entry stub
        let off = pe
            .rva_to_file_offset(Rva(*target))
            .ok()
            .map(|o| o.0 as usize);
        let kind = off
            .map(|o| {
                let b = &bytes[o..(o + 8).min(bytes.len())];
                if b[0] == 0x68 && b[5] == 0xE8 {
                    "push-imm;call"
                } else if b[0] == 0xE9 {
                    "jmp"
                } else {
                    "other"
                }
            })
            .unwrap_or("unmapped");
        println!("  {target:#x} {kind:14} users={:?}", users);
        shown += 1;
    }
}
