use std::fs;
use vmpx_disasm::DirectJumpTraceLimits;
use vmpx_pe::{PeImage, Rva};
use vmpx_vm::{
    HandlerCodecRecoveryConfig, HandlerEffectRoles, RecoveryStatus, VmDiscoveryLimits,
    analyze_vm_entry_at, analyze_x64_native_frame, recover_and_decode_linear_vm_stream,
    recover_bootstrap_dispatch,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bytes = fs::read(&args[1]).unwrap();
    let pe = PeImage::parse(&bytes).unwrap();
    let target: u32 = u32::from_str_radix(args[2].trim_start_matches("0x"), 16).unwrap();
    let candidate = analyze_vm_entry_at(&pe, Rva(target), VmDiscoveryLimits::default()).unwrap();
    let (_, bootstrap) =
        recover_bootstrap_dispatch(&pe, &candidate, DirectJumpTraceLimits::default()).unwrap();
    let bootstrap = bootstrap.unwrap();
    let frame =
        analyze_x64_native_frame(&pe, candidate.rva, DirectJumpTraceLimits::default()).unwrap();
    println!(
        "native frame status={:?} cursor={:?}",
        frame.status, frame.cursor_architectural_register
    );
    let Some(cursor) = frame.cursor_architectural_register else {
        return;
    };
    let vsp = cursor.to_iced();
    println!(
        "roles vip={:?} key={:?} handler={:?} dir={:?}",
        bootstrap.roles.vip,
        bootstrap.roles.rolling_key,
        bootstrap.roles.handler,
        bootstrap.bytecode_direction
    );
    let first = bootstrap
        .codec
        .decode_direction(&pe, bootstrap.initial_state, bootstrap.bytecode_direction)
        .unwrap();
    let stream = recover_and_decode_linear_vm_stream(
        &pe,
        first.state_after,
        &HandlerCodecRecoveryConfig {
            roles: bootstrap.roles,
            bytecode_direction: bootstrap.bytecode_direction,
            effect_roles: Some(HandlerEffectRoles {
                vsp,
                context_base: iced_x86::Register::RSP,
            }),
            live_out_registers: vec![
                vsp,
                bootstrap.roles.handler,
                bootstrap.roles.vip,
                bootstrap.roles.rolling_key,
            ],
            live_out_rflags: 0,
            trace_limits: DirectJumpTraceLimits::default(),
            proof_limits: vmpx_disasm::RegisterDispatchProofLimits::default(),
        },
        2048,
    )
    .unwrap();
    println!(
        "decoded {} stop={:?} status={:?}",
        stream.decode.instructions.len(),
        stream.decode.stop_reason,
        stream.decode.final_state.handler
    );

    let final_handler = stream.decode.final_state.handler;
    println!("final handler {final_handler:?}");
    if let Some(rec) = stream.handler_recoveries.get(&final_handler) {
        println!("  status={:?}", rec.status);
        println!(
            "  semantic={:?}",
            rec.handler_analysis.as_ref().map(|a| a.semantic)
        );
        println!("  decode_spec={}", rec.decode_spec.is_some());
        println!("  boundary_spec={}", rec.boundary_spec.is_some());
        for d in rec.diagnostics.iter().take(4) {
            println!("    dg {d}");
        }
    }
    let mut counts = std::collections::BTreeMap::new();
    for rec in stream.handler_recoveries.values() {
        let key = format!(
            "{:?}/{:?}",
            rec.status,
            rec.handler_analysis.as_ref().map(|a| a.semantic)
        );
        *counts.entry(key).or_insert(0usize) += 1;
    }
    println!("handler status counts: {counts:?}");
    let _ = RecoveryStatus::Recovered;
}
