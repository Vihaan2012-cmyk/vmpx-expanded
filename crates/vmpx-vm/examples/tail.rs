use std::fs;
use vmpx_disasm::DirectJumpTraceLimits;
use vmpx_pe::{PeImage, Rva};
use vmpx_vm::{
    HandlerCodecRecoveryConfig, HandlerEffectRoles, VmDiscoveryLimits, analyze_vm_entry_at,
    analyze_x64_native_frame, recover_and_decode_linear_vm_stream, recover_bootstrap_dispatch,
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
    let vsp = frame.cursor_architectural_register.unwrap().to_iced();
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
        "total {} stop={:?}",
        stream.decode.instructions.len(),
        stream.decode.stop_reason
    );
    for ins in stream.decode.instructions.iter().rev().take(6).rev() {
        println!(
            "  vip={:?} handler={:?} semantic={:?}",
            ins.state_before.vip, ins.state_before.handler, ins.semantic
        );
    }
    let fh = stream.decode.final_state.handler;
    println!(
        "final handler {fh:?} rva={:#x}",
        fh.0.wrapping_sub(0x36b9a0000)
    );
}
