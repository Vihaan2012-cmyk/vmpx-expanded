use std::{collections::BTreeMap, fs};

use vmpx_pe::{PeImage, Rva};
use vmpx_vm::{
    BitWidth, EntryPathRecoveryLimits, EntryProfileRecoveryLimits, Expr, ModuloImageTargetLimits,
    VmDiscoveryLimits, VmEntrySource, VmHandlerSemantic, VmProfileChainStop,
    VmProfileExplorerLimits, analyze_vm_entry_at, analyze_vm_memory_exception_coverage,
    continue_through_native_vm_entry, discover_vm_entries, recover_entry_paths,
    recover_modulo_image_targets,
};
use vmpx_vmlift::audit_entry_effect_liveness;

#[test]
#[ignore = "large controlled mapped-image fixture"]
#[allow(clippy::too_many_lines)]
fn refuses_p06_reintegration_at_proven_exception_transition() {
    let bytes = fs::read(
        r"D:\vmp\testfilevmp\systems.throughvmpstatic.thatsalsoopeninidaprornthroughmcp.xpl",
    )
    .expect("controlled fixture must exist");
    let pe = PeImage::parse(&bytes).expect("controlled fixture must parse");
    let discovery = discover_vm_entries(
        &pe,
        VmDiscoveryLimits {
            scan_executable_stubs: false,
            ..VmDiscoveryLimits::default()
        },
    )
    .expect("loader-root discovery must succeed");
    let entry = discovery
        .entries
        .iter()
        .find(|entry| entry.source == VmEntrySource::PeEntryPoint)
        .expect("protected header entry must be a VM entry");
    let domain = BTreeMap::from([(
        "native_rdx_initial".to_owned(),
        (0_u64..=3)
            .map(|reason| Expr::constant(BitWidth::W64, reason))
            .collect(),
    )]);
    let limits = EntryPathRecoveryLimits {
        chain: VmProfileExplorerLimits {
            max_segments: 32,
            max_instructions_per_segment: 2_048,
            max_total_instructions: 8_192,
            ..VmProfileExplorerLimits::default()
        },
        ..EntryPathRecoveryLimits::default()
    };
    let family = recover_entry_paths(&pe, entry, &domain, limits)
        .expect("bounded DLL-entry paths must recover without guessing");
    let liveness = audit_entry_effect_liveness(&family);
    assert!(!family.all_paths_reach_vm_exit);
    assert!(!liveness.ready_for_native_replacement);

    let mut targets = BTreeMap::new();
    for path in &family.paths {
        let reason = path.assignments["native_rdx_initial"];
        if reason <= 1 {
            let exit = path.native_exit.as_ref().expect("ordinary path has exit");
            let report = recover_modulo_image_targets(
                &pe,
                &exit.return_address,
                ModuloImageTargetLimits::default(),
            )
            .expect("ordinary continuation uses one bounded modulo driver");
            assert!(report.complete);
            assert_eq!(report.targets.len(), 1);
            assert!(report.targets[0].executable);
            targets.insert(reason, report.targets[0].rva.expect("image RVA"));
        } else {
            assert!(path.native_exit.is_none());
            assert_eq!(
                path.chain.stop_reason,
                VmProfileChainStop::SymbolicExecutionFailure
            );
            assert!(
                path.chain
                    .diagnostics
                    .iter()
                    .any(|message| message.contains("requires exception-flow analysis"))
            );
        }
    }

    assert_eq!(targets[&0].get(), 0x0EFD_0F97);
    assert_eq!(targets[&1].get(), 0x0EFD_3EC9);
    let detach_entry = analyze_vm_entry_at(&pe, targets[&0], VmDiscoveryLimits::default())
        .expect("detach continuation must decode as a VM-entry candidate");
    assert!(detach_entry.confidence >= 60);
    let detach_path = family
        .paths
        .iter()
        .find(|path| path.assignments["native_rdx_initial"] == 0)
        .expect("detach path");
    let prior_symbolic = detach_path
        .chain
        .segments
        .last()
        .and_then(|segment| segment.symbolic.as_ref())
        .expect("detach path retains terminal symbolic state");
    let detach_profile = continue_through_native_vm_entry(
        &pe,
        prior_symbolic,
        detach_path
            .native_exit
            .as_ref()
            .expect("detach native exit"),
        &detach_entry,
        EntryProfileRecoveryLimits::default(),
        limits.chain,
    )
    .expect("descending continuation must compose with the prior native exit");

    assert_eq!(
        detach_profile.entry_profile.profile.bytecode_direction,
        vmpx_vm::BytecodeDirection::Descending
    );
    assert_eq!(detach_profile.chain.total_instructions, 390);
    assert_eq!(
        detach_profile.chain.stop_reason,
        VmProfileChainStop::SymbolicExecutionFailure
    );
    let error = detach_profile.chain.segments[0]
        .symbolic_error
        .as_ref()
        .expect("proven null dereference is retained");
    assert_eq!(error.instruction_index, 146);
    assert_eq!(error.semantic, VmHandlerSemantic::MemoryLoad);
    assert!(error.message.contains("requires exception-flow analysis"));
    let terminal = &detach_profile.chain.segments[0];
    let fault_instruction = &terminal.decode.decode.instructions[error.instruction_index];
    let handler = &terminal.decode.handler_recoveries[&fault_instruction.state_before.handler];
    let coverage = analyze_vm_memory_exception_coverage(&pe, handler)
        .expect("faulting handler unwind metadata must parse");
    assert_eq!(coverage.fault_sites, vec![Rva(0x0F0B_9F95)]);
    assert!(coverage.all_fault_sites_have_language_handler);
    assert!(!coverage.continuation_recovered);
    assert_eq!(coverage.coverage.len(), 1);
    assert_eq!(
        coverage.coverage[0].unwind.exception_handler,
        Some(Rva(0x0EE0_F248))
    );
    let cfg = vmpx_disasm::recover_cfg(
        &pe,
        Rva(0x0EE0_F248),
        vmpx_disasm::CfgLimits::default(),
    )
    .expect("exception handler CFG");
    for block in cfg.blocks {
        eprintln!("BLOCK {} -> {:?}", block.start, block.successors);
        for instruction in block.instructions {
            eprintln!("  {} {}", instruction.rva, instruction.text);
        }
    }
}
