use std::fs;
use vmpx_pe::{PeImage, Rva};
use vmpx_vm::{
    EntryProfileRecoveryLimits, HandlerCodecRecoveryConfig, HandlerEffectRoles, RecoveryStatus,
    VmDiscoveryLimits, VmProfileExecutionState, VmProfileExplorerLimits, analyze_vm_entry_at,
    analyze_x64_native_frame, explore_vm_profile_chain, map_native_context_scatter,
    recover_and_decode_linear_vm_stream, recover_bootstrap_dispatch,
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
    println!("frame={:?} slots={}", frame.status, frame.slots.len());
    if frame.status != RecoveryStatus::Recovered {
        println!("frame diagnostics: {:?}", frame.diagnostics);
        return;
    }
    let vsp = frame.cursor_architectural_register.unwrap().to_iced();
    let effect_roles = HandlerEffectRoles {
        vsp,
        context_base: iced_x86::Register::RSP,
    };
    let profile = vmpx_vm::VmProfile {
        handler_roles: bootstrap.roles,
        effect_roles,
        bytecode_direction: bootstrap.bytecode_direction,
    };
    let first = bootstrap
        .codec
        .decode_direction(&pe, bootstrap.initial_state, bootstrap.bytecode_direction)
        .unwrap();
    let decode = recover_and_decode_linear_vm_stream(
        &pe,
        first.state_after,
        &HandlerCodecRecoveryConfig {
            roles: profile.handler_roles,
            bytecode_direction: profile.bytecode_direction,
            effect_roles: Some(profile.effect_roles),
            live_out_registers: vec![
                vsp,
                profile.handler_roles.handler,
                profile.handler_roles.vip,
                profile.handler_roles.rolling_key,
            ],
            live_out_rflags: 0,
            trace_limits: limits.trace,
            proof_limits: limits.dispatch_proof,
        },
        limits.instruction_limit,
    )
    .unwrap();
    let offset = frame.slots.len();
    println!(
        "decoded {} ops, scatter slots {offset}",
        decode.decode.instructions.len()
    );
    let Some(scatter) = decode.decode.instructions.get(..offset) else {
        println!("not enough instructions for scatter");
        return;
    };
    let mapping = map_native_context_scatter(&frame, scatter);
    println!("mapping diagnostics: {:?}", mapping.diagnostics);
    let Some(symbolic) = mapping.symbolic_initial_state() else {
        println!("no symbolic initial state");
        return;
    };
    let boundary = decode
        .handler_recoveries
        .get(&decode.decode.final_state.handler)
        .and_then(|r| r.handler_analysis.as_ref())
        .map(|a| a.semantic);
    println!(
        "boundary semantic: {boundary:?} stop={:?}",
        decode.decode.stop_reason
    );
    let chain = explore_vm_profile_chain(
        &pe,
        VmProfileExecutionState {
            decoder: first.state_after,
            profile,
            symbolic,
        },
        VmProfileExplorerLimits {
            max_segments: 32,
            max_instructions_per_segment: 2048,
            max_total_instructions: 8192,
            ..Default::default()
        },
    )
    .unwrap();
    println!(
        "chain segments={} ops={} stop={:?}",
        chain.segments.len(),
        chain.total_instructions,
        chain.stop_reason
    );
    for d in chain.diagnostics.iter().take(5) {
        println!("  dg {d}");
    }
    for (i, seg) in chain.segments.iter().enumerate() {
        println!(
            "  seg {i} dir={:?} ops={} stop={:?}",
            seg.profile.bytecode_direction,
            seg.decode.decode.instructions.len(),
            seg.decode.decode.stop_reason
        );
    }
}
