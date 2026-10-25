use std::{collections::BTreeSet, fs};
use vmpx_pe::{PeImage, Rva};
use vmpx_vm::{
    EntryProfileRecoveryLimits, HandlerCodecRecoveryConfig, HandlerEffectRoles, VmDiscoveryLimits,
    VmEntrySource, analyze_vm_entry_at, analyze_x64_native_frame, discover_vm_entries,
    recover_and_decode_linear_vm_stream, recover_bootstrap_dispatch,
};

fn handlers_for(pe: &PeImage<'_>, target: u32) -> Option<BTreeSet<u32>> {
    let limits = EntryProfileRecoveryLimits::default();
    let candidate = analyze_vm_entry_at(pe, Rva(target), VmDiscoveryLimits::default())?;
    let (_, bootstrap) = recover_bootstrap_dispatch(pe, &candidate, limits.trace).ok()?;
    let bootstrap = bootstrap?;
    let frame = analyze_x64_native_frame(pe, candidate.rva, limits.trace).ok()?;
    let vsp = frame.cursor_architectural_register?.to_iced();
    let roles = bootstrap.roles;
    let first = bootstrap
        .codec
        .decode_direction(pe, bootstrap.initial_state, bootstrap.bytecode_direction)
        .ok()?;
    let decode = recover_and_decode_linear_vm_stream(
        pe,
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
    .ok()?;
    Some(
        decode
            .decode
            .instructions
            .iter()
            .map(|i| i.state_before.handler.0.wrapping_sub(0x36b9a0000) as u32)
            .collect(),
    )
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bytes = fs::read(&args[1]).unwrap();
    let pe = PeImage::parse(&bytes).unwrap();
    let discovery = discover_vm_entries(
        &pe,
        VmDiscoveryLimits {
            scan_executable_stubs: false,
            ..Default::default()
        },
    )
    .unwrap();
    let entry = discovery
        .entries
        .iter()
        .find(|e| e.source == VmEntrySource::PeEntryPoint)
        .unwrap();
    let known = handlers_for(&pe, entry.rva.0).expect("entry handlers");
    println!("entry-point chain handler set: {}", known.len());
    for stub in [0xc45fcc2u32, 0xc4ea18d, 0xc718c1e] {
        let set = handlers_for(&pe, stub).unwrap_or_default();
        let novel: Vec<u32> = set.difference(&known).copied().collect();
        println!(
            "stub {stub:#x}: {} handlers, {} novel vs entry chain",
            set.len(),
            novel.len()
        );
        for h in novel.iter().take(5) {
            println!("    novel {h:#x}");
        }
    }
    // Is the suspicious handler in the validated set?
    for h in [0xC603B37u32, 0xC773772, 0xC6D79DF] {
        println!("handler {h:#x} in entry chain: {}", known.contains(&h));
    }
}
