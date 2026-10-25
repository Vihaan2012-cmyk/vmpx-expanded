use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    fs,
};
use vmpx_disasm::DirectJumpTraceLimits;
use vmpx_disasm::{CfgLimits, recover_cfg};
use vmpx_pe::{PeImage, Rva};
use vmpx_vm::{
    EntryProfileRecoveryLimits, HandlerCodecRecoveryConfig, VmDiscoveryLimits, analyze_vm_entry_at,
    recover_and_decode_linear_vm_stream, recover_bootstrap_dispatch,
};

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

    let mut targets: BTreeSet<u32> = BTreeSet::new();
    for symbol in &exports.symbols {
        let mut local = BTreeSet::new();
        let mut q = VecDeque::from([symbol.rva]);
        while let Some(start) = q.pop_front() {
            if !local.insert(start.0) {
                continue;
            }
            if is_vm(start) {
                targets.insert(start.0);
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
                        targets.insert(s.0);
                    } else if !local.contains(&s.0) {
                        q.push_back(*s);
                    }
                }
                for c in &block.direct_calls {
                    if is_vm(*c) {
                        targets.insert(c.0);
                    } else if !local.contains(&c.0) {
                        q.push_back(*c);
                    }
                }
            }
        }
    }
    println!("export-reachable VM targets: {}", targets.len());
    let mut ok = 0;
    let mut failed = 0;
    for target in &targets {
        let rva = Rva(*target);
        // only real entry stubs: push imm32; call rel32
        let Ok(off) = pe.rva_to_file_offset(rva) else {
            println!("  {target:#x} unmapped");
            failed += 1;
            continue;
        };
        let b = &bytes[off.0 as usize..(off.0 as usize + 10).min(bytes.len())];
        let is_stub = b.len() >= 10 && b[0] == 0x68 && b[5] == 0xE8;
        if !is_stub {
            println!("  {target:#x} not an entry stub: {}", hex(b));
            failed += 1;
            continue;
        }
        let candidate = analyze_vm_entry_at(&pe, rva, VmDiscoveryLimits::default());
        let Some(candidate) = candidate.filter(|c| c.confidence >= 60) else {
            println!("  {target:#x} no confident VM entry");
            failed += 1;
            continue;
        };
        let bootstrap =
            match recover_bootstrap_dispatch(&pe, &candidate, DirectJumpTraceLimits::default()) {
                Ok((_, Some(recovery))) => recovery,
                Ok((report, None)) => {
                    println!(
                        "  {target:#x} ambiguous bootstrap: {}",
                        report.diagnostics.join("; ")
                    );
                    failed += 1;
                    continue;
                }
                Err(e) => {
                    println!("  {target:#x} bootstrap failed: {e}");
                    failed += 1;
                    continue;
                }
            };
        let vsp = iced_x86::Register::None;
        match recover_and_decode_linear_vm_stream(
            &pe,
            bootstrap.initial_state.clone(),
            &HandlerCodecRecoveryConfig {
                roles: bootstrap.roles,
                bytecode_direction: bootstrap.bytecode_direction,
                effect_roles: None,
                live_out_registers: vec![
                    bootstrap.roles.handler,
                    bootstrap.roles.vip,
                    bootstrap.roles.rolling_key,
                ],
                live_out_rflags: 0,
                trace_limits: DirectJumpTraceLimits::default(),
                proof_limits: vmpx_disasm::RegisterDispatchProofLimits::default(),
            },
            2048,
        ) {
            Ok(stream) => {
                println!(
                    "  {target:#x} seed={:?} conf={} handlers={} ops={} stop={:?}",
                    candidate.seed,
                    candidate.confidence,
                    stream.handler_recoveries.len(),
                    stream.decode.instructions.len(),
                    stream.decode.stop_reason
                );
                ok += 1;
            }
            Err(e) => {
                println!("  {target:#x} decode failed: {e}");
                failed += 1
            }
        }
        let _ = EntryProfileRecoveryLimits::default();
    }
    println!("decoded {ok}, failed {failed}");
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}
