use std::fs;
use vmpx_disasm::DirectJumpTraceLimits;
use vmpx_pe::{PeImage, Rva};
use vmpx_vm::{HandlerCodecRecoveryConfig, HandlerEffectRoles, recover_handler_codecs};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bytes = fs::read(&args[1]).unwrap();
    let pe = PeImage::parse(&bytes).unwrap();
    let rva: u32 = u32::from_str_radix(args[2].trim_start_matches("0x"), 16).unwrap();
    let roles = vmpx_vm::HandlerRegisterRoles {
        vip: iced_x86::Register::RBX,
        rolling_key: iced_x86::Register::R12,
        handler: iced_x86::Register::R14,
    };
    let report = recover_handler_codecs(
        &pe,
        Rva(rva),
        &HandlerCodecRecoveryConfig {
            roles,
            bytecode_direction: vmpx_vm::BytecodeDirection::Descending,
            effect_roles: Some(HandlerEffectRoles {
                vsp: iced_x86::Register::R13,
                context_base: iced_x86::Register::RSP,
            }),
            live_out_registers: vec![
                iced_x86::Register::R13,
                iced_x86::Register::R14,
                iced_x86::Register::RBX,
                iced_x86::Register::R12,
            ],
            live_out_rflags: 0,
            trace_limits: DirectJumpTraceLimits::default(),
            proof_limits: vmpx_disasm::RegisterDispatchProofLimits::default(),
        },
    )
    .unwrap();
    println!("status={:?}", report.status);
    println!(
        "dispatch candidates={}",
        report.dispatch_analysis.candidates.len()
    );
    for c in &report.dispatch_analysis.candidates {
        println!(
            "  load={:?} reg={} vip_delta={} handler_update={:?} rebase={:?} rebase_rva={:?}",
            c.encoded_load,
            c.decoded_register,
            c.vip_delta,
            c.handler_update,
            c.handler_rebase,
            c.handler_rebase_rva
        );
    }
    for r in &report.dispatch_analysis.rejected_sources {
        println!("  rejected {:?}: {}", r.encoded_load, r.reason);
    }
    println!(
        "operand candidates={}",
        report.operand_analysis.candidates.len()
    );
    println!(
        "analysis={:?}",
        report
            .handler_analysis
            .as_ref()
            .map(|a| (a.semantic, a.confidence))
    );
    for d in report.diagnostics.iter().take(6) {
        println!("  dg {d}");
    }
    if let Some(e) = &report.effects {
        println!(
            "vsp_delta={:?} arith={:?} mem={:?}",
            e.vsp_delta,
            e.arithmetic_operations.clone(),
            e.memory
                .iter()
                .map(|m| (format!("{:?}", m.region), m.reads, m.writes))
                .collect::<Vec<_>>()
        );
        println!(
            "loads_vip={} resets_key={} reinit_handler={} captures_flags={}",
            e.loads_vip_from_vsp,
            e.resets_key_from_vip,
            e.reinitializes_handler_base,
            e.captures_native_flags_to_vsp
        );
    }
}
