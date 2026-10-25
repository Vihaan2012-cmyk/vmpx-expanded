use std::fs;
use vmpx_pe::{PeImage, Rva};
use vmpx_vm::{
    EntryProfileRecoveryLimits, HandlerCodecRecoveryConfig, HandlerEffectRoles, VmDiscoveryLimits,
    analyze_vm_entry_at, analyze_x64_native_frame, recover_and_decode_linear_vm_stream,
    recover_bootstrap_dispatch,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bytes = fs::read(&args[1]).unwrap();
    let pe = PeImage::parse(&bytes).unwrap();
    let target: u32 = u32::from_str_radix(args[2].trim_start_matches("0x"), 16).unwrap();
    let limits = EntryProfileRecoveryLimits::default();
    let candidate = analyze_vm_entry_at(&pe, Rva(target), VmDiscoveryLimits::default()).unwrap();
    let (_, bootstrap) = recover_bootstrap_dispatch(&pe, &candidate, limits.trace).unwrap();
    let bootstrap = bootstrap.unwrap();
    let frame = analyze_x64_native_frame(&pe, candidate.rva, limits.trace).unwrap();
    let vsp = frame.cursor_architectural_register.unwrap().to_iced();
    let effect_roles = HandlerEffectRoles {
        vsp,
        context_base: iced_x86::Register::RSP,
    };
    let roles = bootstrap.roles;
    let first = bootstrap
        .codec
        .decode_direction(&pe, bootstrap.initial_state, bootstrap.bytecode_direction)
        .unwrap();
    let decode = recover_and_decode_linear_vm_stream(
        &pe,
        first.state_after,
        &HandlerCodecRecoveryConfig {
            roles,
            bytecode_direction: bootstrap.bytecode_direction,
            effect_roles: Some(effect_roles),
            live_out_registers: vec![vsp, roles.handler, roles.vip, roles.rolling_key],
            live_out_rflags: 0,
            trace_limits: limits.trace,
            proof_limits: limits.dispatch_proof,
        },
        limits.instruction_limit,
    )
    .unwrap();
    println!(
        "stop={:?} unresolved={:?}",
        decode.decode.stop_reason, decode.decode.unresolved_handler
    );
    let un = decode.decode.unresolved_handler.unwrap();
    println!("unresolved rva={:#x}", un.0 - 0x36b9a0000);
    if let Some(rec) = decode.handler_recoveries.get(&un) {
        println!(
            "status={:?} sem={:?} spec={} boundary={}",
            rec.status,
            rec.handler_analysis.as_ref().map(|a| a.semantic),
            rec.decode_spec.is_some(),
            rec.boundary_spec.is_some()
        );
        for d in rec.diagnostics.iter().take(5) {
            println!("  dg {d}");
        }
        println!("dispatch cands={}", rec.dispatch_analysis.candidates.len());
        for c in &rec.dispatch_analysis.candidates {
            println!(
                "   load={:?} vip_delta={} handler_update={:?} rebase={:?} vip_rebased={}",
                c.encoded_load,
                c.vip_delta,
                c.handler_update,
                c.handler_rebase,
                c.vip_rebased_before_source
            );
        }
        for r in &rec.dispatch_analysis.rejected_sources {
            println!("   rejected {:?}: {}", r.encoded_load, r.reason);
        }
    }
}
