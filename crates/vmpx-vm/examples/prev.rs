use std::fs;
use vmpx_disasm::DirectJumpTraceLimits;
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
            effect_roles: Some(HandlerEffectRoles {
                vsp,
                context_base: iced_x86::Register::RSP,
            }),
            live_out_registers: vec![vsp, roles.handler, roles.vip, roles.rolling_key],
            live_out_rflags: 0,
            trace_limits: limits.trace,
            proof_limits: limits.dispatch_proof,
        },
        limits.instruction_limit,
    )
    .unwrap();
    let n = decode.decode.instructions.len();
    for ins in decode.decode.instructions.iter().skip(n.saturating_sub(3)) {
        println!(
            "  vip={:?} handler={:?} sem={:?} -> vip={:?} handler={:?}",
            ins.state_before.vip,
            ins.state_before.handler,
            ins.semantic,
            ins.state_after.vip,
            ins.state_after.handler
        );
    }
    println!(
        "unresolved={:?} stop={:?}",
        decode.decode.unresolved_handler, decode.decode.stop_reason
    );
}
