use std::{
    collections::{BTreeSet, VecDeque},
    fs,
};
use vmpx_disasm::{CfgLimits, recover_cfg};
use vmpx_pe::PeImage;

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
    let Some(exports) = &report.exports else {
        return;
    };
    let thunks: BTreeSet<u32> = [0x788fd4u32, 0x78938f, 0x79822a].into_iter().collect();

    let mut reached_thunks: Vec<(String, u32)> = Vec::new();
    for symbol in &exports.symbols {
        let name = symbol.name.clone().unwrap_or_default();
        let mut seen = BTreeSet::new();
        let mut q = VecDeque::from([symbol.rva]);
        while let Some(start) = q.pop_front() {
            if !seen.insert(start.0) {
                continue;
            }
            if thunks.contains(&start.0) {
                reached_thunks.push((name.clone(), start.0));
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
                    if starts.contains(&succ.0) && !seen.contains(&succ.0) {
                        q.push_back(*succ);
                    }
                }
                for call in &block.direct_calls {
                    if starts.contains(&call.0) && !seen.contains(&call.0) {
                        q.push_back(*call);
                    }
                }
            }
        }
        println!(
            "{:28} functions={:6} thunks_reached={}",
            name,
            seen.len(),
            reached_thunks.iter().filter(|(n, _)| *n == name).count()
        );
    }
    println!("\nthunk hits: {reached_thunks:?}");
}
