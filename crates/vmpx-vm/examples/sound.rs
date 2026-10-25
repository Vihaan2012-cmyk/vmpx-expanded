use std::{collections::BTreeMap, fs};
use vmpx_pe::{PeImage, Rva};
use vmpx_vm::{
    EntryProfileRecoveryLimits, HandlerCodecRecoveryConfig, HandlerEffectRoles, VmDiscoveryLimits,
    analyze_vm_entry_at, analyze_x64_native_frame, discover_vm_entries,
    recover_and_decode_linear_vm_stream, recover_bootstrap_dispatch,
};

fn section_of(pe: &PeImage<'_>, va: u64) -> String {
    let Some(rva) = pe.optional.image_base.checked_rva_of(vmpx_pe::Va(va)) else {
        return "unmapped".to_owned();
    };
    pe.sections
        .iter()
        .find(|s| s.contains_mapped_rva(rva))
        .map_or_else(|| "gap".to_owned(), |s| s.name.clone())
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

    let mut targets: Vec<(String, u32)> = discovery
        .entries
        .iter()
        .map(|e| (format!("{:?}", e.source), e.rva.0))
        .collect();
    targets.push(("stub1".to_owned(), 0xc45fcc2));
    targets.push(("stub2".to_owned(), 0xc4ea18d));
    targets.push(("stub3".to_owned(), 0xc718c1e));

    for (name, target) in targets {
        let limits = EntryProfileRecoveryLimits::default();
        let Some(candidate) = analyze_vm_entry_at(&pe, Rva(target), VmDiscoveryLimits::default())
        else {
            continue;
        };
        let Ok((_, Some(bootstrap))) = recover_bootstrap_dispatch(&pe, &candidate, limits.trace)
        else {
            continue;
        };
        let Ok(frame) = analyze_x64_native_frame(&pe, candidate.rva, limits.trace) else {
            continue;
        };
        let Some(cursor) = frame.cursor_architectural_register else {
            continue;
        };
        let vsp = cursor.to_iced();
        let roles = bootstrap.roles;
        let Ok(first) = bootstrap.codec.decode_direction(
            &pe,
            bootstrap.initial_state,
            bootstrap.bytecode_direction,
        ) else {
            continue;
        };
        let Ok(decode) = recover_and_decode_linear_vm_stream(
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
        ) else {
            continue;
        };

        let mut vip_sections: BTreeMap<String, usize> = BTreeMap::new();
        for ins in &decode.decode.instructions {
            *vip_sections
                .entry(section_of(&pe, ins.state_before.vip.0))
                .or_default() += 1;
        }
        let bad: usize = vip_sections
            .iter()
            .filter(|(s, _)| !s.starts_with(".vmp"))
            .map(|(_, c)| *c)
            .sum();
        println!(
            "{name:16} {target:#010x} ops={:3} vsp_delta={:?} stop={:?}",
            decode.decode.instructions.len(),
            decode.decode.cumulative_vsp_delta,
            decode.decode.stop_reason
        );
        println!("    vip sections: {vip_sections:?}  (outside .vmp: {bad})");
    }
}
