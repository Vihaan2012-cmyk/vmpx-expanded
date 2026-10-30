// Recovers a VM entry's CFG and attempts the CFG-level lift, reporting per-block
// lowering status and the first thing that stops the lift.
//
// usage: cfglift <pe> <entry-rva> [<entry-rva>...]
use std::{collections::BTreeMap, fs};

use vmpx_pe::{PeImage, Rva};
use vmpx_recompile::CodeAddress;
use vmpx_vm::{
    BlockExit, EntryProfileRecoveryLimits, RecoveryStatus, SymbolicExecutionLimits, VmCfgLimits,
    VmDiscoveryLimits, analyze_vm_entry_at, analyze_vm_exit, recover_entry_stream, recover_vm_cfg,
};
use vmpx_vmlift::{
    DecodedCfgInput, first_unliftable, lift_decoded_cfg, probe_decoded_cfg, recover_entry_frame,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bytes = fs::read(&args[1]).unwrap();
    let pe = PeImage::parse(&bytes).unwrap();
    let next_rva = vmpx_rebuild::next_section_rva(&bytes).expect("next section rva");
    let address = CodeAddress(pe.optional.image_base.0 + u64::from(next_rva.0));
    for target in args[2..].iter().filter(|arg| arg.starts_with("0x")) {
        let target = u32::from_str_radix(target.trim_start_matches("0x"), 16).unwrap();
        println!("=== entry {target:#x}");
        let Some(candidate) = analyze_vm_entry_at(&pe, Rva(target), VmDiscoveryLimits::default())
        else {
            println!("  no VM entry candidate");
            continue;
        };
        let entry_limits = EntryProfileRecoveryLimits {
            instruction_limit: 100_000,
            ..EntryProfileRecoveryLimits::default()
        };
        let limits = VmCfgLimits {
            entry: entry_limits,
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
            "  blocks={} edges={} complete={} stalls={}",
            cfg.blocks.len(),
            cfg.edges.len(),
            cfg.is_complete(),
            cfg.stalls.len()
        );
        for (key, block) in &cfg.blocks {
            let status = first_unliftable(&block.instructions).map_or_else(
                || "bodies lift".to_owned(),
                |(index, semantic, reason)| format!("instruction {index} {semantic:?}: {reason}"),
            );
            println!(
                "  block {key} n={} exit={:?} -> {status}",
                block.instructions.len(),
                block.exit
            );
        }
        for stall in &cfg.stalls {
            println!("  STALL {}: {}", stall.block, stall.reason);
        }
        let stream = match recover_entry_stream(&pe, &candidate, entry_limits) {
            Ok(stream) => stream,
            Err(error) => {
                println!("  STOP: entry stream recovery failed: {error}");
                continue;
            }
        };
        let frame = match recover_entry_frame(
            &pe,
            candidate.rva,
            stream.profile.effect_roles.vsp,
            entry_limits.trace,
        ) {
            Ok(frame) => frame,
            Err(error) => {
                println!("  STOP: entry frame not recovered: {error}");
                continue;
            }
        };
        let mut exits: BTreeMap<_, _> = stream
            .decode
            .handler_recoveries
            .iter()
            .filter_map(|(handler, recovery)| Some((*handler, recovery.vm_exit.clone()?)))
            .collect();
        for block in cfg
            .blocks
            .values()
            .filter(|block| block.exit == BlockExit::VmExit)
        {
            let handler = block
                .instructions
                .last()
                .map_or(block.start.handler, |last| last.state_after.handler);
            if exits.contains_key(&handler) {
                continue;
            }
            let Some(rva) = pe.optional.image_base.checked_rva_of(handler) else {
                continue;
            };
            if let Ok(report) = analyze_vm_exit(
                &pe,
                rva,
                stream.profile.effect_roles.vsp,
                entry_limits.trace,
            ) && report.status == RecoveryStatus::Recovered
            {
                exits.insert(handler, report);
            }
        }
        let calls = BTreeMap::new();
        let exit_blocks = cfg
            .blocks
            .values()
            .filter(|block| block.exit == BlockExit::VmExit)
            .count();
        println!(
            "  exit layouts known for {} handler(s); {exit_blocks} VM-exit block(s); call frames not exposed by the CFG",
            exits.len()
        );
        let input = DecodedCfgInput {
            cfg: &cfg,
            frame: &frame,
            exits: &exits,
            native_calls: &calls,
            memory: Some(&pe),
            limits: SymbolicExecutionLimits::default(),
            target_limits: limits,
            max_expanded_blocks: 256,
        };
        let probe = probe_decoded_cfg(&input);
        println!(
            "  probe: {} distinct blocks reached ({} expansions), {} exit paths, {} branches, {} calls; stop: {}",
            probe.blocks.len(),
            probe.expanded_blocks,
            probe.exit_paths,
            probe.conditional_branches,
            probe.native_calls,
            probe
                .stop
                .as_ref()
                .map_or_else(|| "none".to_owned(), ToString::to_string)
        );
        match lift_decoded_cfg(&input, address) {
            Ok(lifted) => {
                let replacement = &lifted.replacement;
                println!(
                    "  LIFTED {} blocks ({} expansions), {} exit paths, {} branches, {} calls -> {} IR operations, {} native bytes",
                    lifted.blocks.len(),
                    lifted.expanded_blocks,
                    lifted.exit_paths,
                    lifted.conditional_branches,
                    lifted.native_calls,
                    replacement.function.operations.len(),
                    replacement.compiled.bytes.len()
                );
                println!("  assumptions: {:?}", lifted.assumptions);
            }
            Err(error) => println!("  STOP: {error}"),
        }
    }
}
