use std::fs;

use vmpx_pe::{PeImage, Rva};
use vmpx_vm::{
    BlockExit, EntryProfileRecoveryLimits, VmCfgLimits, VmDiscoveryLimits, analyze_vm_entry_at,
    recover_vm_cfg,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bytes = fs::read(&args[1]).unwrap();
    let pe = PeImage::parse(&bytes).unwrap();
    let verbose = args.iter().any(|arg| arg == "-v");
    for target in args[2..].iter().filter(|arg| arg.starts_with("0x")) {
        let target = u32::from_str_radix(target.trim_start_matches("0x"), 16).unwrap();
        println!("=== entry {target:#x}");
        let Some(candidate) = analyze_vm_entry_at(&pe, Rva(target), VmDiscoveryLimits::default())
        else {
            println!("  no VM entry candidate");
            continue;
        };
        let limits = VmCfgLimits {
            entry: EntryProfileRecoveryLimits {
                instruction_limit: 100_000,
                ..EntryProfileRecoveryLimits::default()
            },
            ..VmCfgLimits::default()
        };
        let cfg = match recover_vm_cfg(&pe, &candidate, limits) {
            Ok(cfg) => cfg,
            Err(error) => {
                println!("  cfg recovery failed: {error}");
                continue;
            }
        };
        println!(
            "  blocks={} instructions={} edges={} visits={} frame_cells_known={} frame_note={:?}",
            cfg.blocks.len(),
            cfg.total_instructions(),
            cfg.edges.len(),
            cfg.visits,
            cfg.frame_cells_known,
            cfg.frame_note
        );
        let exits = cfg
            .blocks
            .values()
            .filter(|block| block.exit == BlockExit::VmExit)
            .count();
        println!(
            "  vm_exit_blocks={exits} complete={} all_paths_reach_vm_exit={} limit_hit={:?}",
            cfg.is_complete(),
            cfg.all_paths_reach_vm_exit(),
            cfg.limit_hit
        );
        let carried_calls = cfg.native_calls().len();
        let carried_exits = cfg.exit_layouts().len();
        let call_blocks = cfg
            .blocks
            .values()
            .filter(|block| block.exit == BlockExit::NativeCall)
            .count();
        println!(
            "  native_call_blocks={call_blocks} carrying_outcome={carried_calls} exit_blocks_with_layout={} exit_layout_handlers={carried_exits}",
            cfg.blocks
                .values()
                .filter(|block| block.vm_exit.is_some())
                .count()
        );
        for stall in &cfg.stalls {
            println!(
                "  STALL block={} vip={}: {}",
                stall.block, stall.vip, stall.reason
            );
        }
        if verbose {
            for (key, block) in &cfg.blocks {
                println!(
                    "  block {key} addr={:?} n={} exit={:?}",
                    block.address,
                    block.instructions.len(),
                    block.exit
                );
            }
            for edge in &cfg.edges {
                println!("  edge {} -> {} {:?}", edge.from, edge.to, edge.kind);
            }
        }
    }
}
