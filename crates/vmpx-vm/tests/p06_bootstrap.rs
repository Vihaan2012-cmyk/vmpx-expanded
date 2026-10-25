use std::{collections::BTreeMap, fs};

use iced_x86::Register;
use vmpx_disasm::DirectJumpTraceLimits;
use vmpx_pe::{PeImage, Va};
use vmpx_vm::{
    BitWidth, EntryPathRecoveryLimits, Expr, ExprKind, VmDiscoveryLimits, VmEntrySource,
    VmProfileExplorerLimits, X64NativeRegister, discover_vm_entries, recover_bootstrap_dispatch,
    recover_entry_paths,
};

#[test]
#[ignore = "large controlled mapped-image fixture"]
fn recovers_p06_bootstrap_dispatch_without_fixture_roles() {
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
    .expect("loader-root VM discovery must succeed");
    let entry = discovery
        .entries
        .iter()
        .find(|entry| entry.source == VmEntrySource::PeEntryPoint)
        .expect("header VM entry must be discovered");
    let (report, recovered) =
        recover_bootstrap_dispatch(&pe, entry, DirectJumpTraceLimits::default())
            .expect("bootstrap native trace must decode");
    eprintln!("{report:#?}");
    let recovered = recovered.unwrap_or_else(|| panic!("unique bootstrap roles: {report:#?}"));
    assert_eq!(recovered.roles.vip, Register::R15);
    assert_eq!(recovered.roles.rolling_key, Register::RBP);
    assert_eq!(recovered.roles.handler, Register::R14);
    assert_eq!(recovered.initial_state.vip, Va(0x0003_7A80_C9DE));
    assert_eq!(recovered.initial_state.rolling_key, 0x3_7A80_C9DE);
    assert_eq!(recovered.initial_state.handler, Va(0x0003_7AA6_4271));
    let first = recovered
        .codec
        .decode(&pe, recovered.initial_state)
        .expect("recovered bootstrap codec must decode the first dispatch");
    assert_eq!(first.state_after.vip, Va(0x0003_7A80_C9E2));
    assert_eq!(first.state_after.rolling_key, 0x3_8572_2D91);
    assert_eq!(first.state_after.handler, Va(0x0003_7A99_26C0));
    let domains = BTreeMap::from([(
        "native_rdx_initial".to_owned(),
        (0_u64..=3)
            .map(|reason| Expr::constant(BitWidth::W64, reason))
            .collect(),
    )]);
    let family = recover_entry_paths(
        &pe,
        entry,
        &domains,
        EntryPathRecoveryLimits {
            chain: VmProfileExplorerLimits {
                max_segments: 32,
                max_instructions_per_segment: 2_048,
                max_total_instructions: 8_192,
                ..VmProfileExplorerLimits::default()
            },
            ..EntryPathRecoveryLimits::default()
        },
    )
    .expect("all bounded entry paths must recover without authored VM constants");
    let profile = &family.entry_profile;
    eprintln!(
        "bootstrap-derived decode: {} operations, stop {:?}, unresolved {:?}",
        profile.decode.decode.instructions.len(),
        profile.decode.decode.stop_reason,
        profile.decode.decode.unresolved_handler,
    );
    assert_eq!(profile.operational_instruction_offset, 19);
    assert_eq!(profile.decode.decode.unresolved_handler, None);
    assert_eq!(profile.decode.decode.instructions.len(), 321);
    assert_eq!(
        profile.symbolic.branch_target_dependencies,
        vec!["native_rdx_initial".to_owned()]
    );
    assert!(family.domain_complete);
    assert!(family.all_paths_reach_vm_exit);
    assert_eq!(family.paths.len(), 4);
    let outputs = family
        .paths
        .iter()
        .map(|path| {
            let reason = path.assignments["native_rdx_initial"];
            let rax = &path
                .native_exit
                .as_ref()
                .expect("all paths have an exact native exit")
                .registers[&X64NativeRegister::Rax];
            let ExprKind::Const { value } = rax.kind else {
                panic!("reason {reason} RAX is not constant: {rax:?}");
            };
            (reason, value)
        })
        .collect::<BTreeMap<_, _>>();
    assert_eq!(outputs, BTreeMap::from([(0, 1), (1, 24), (2, 0), (3, 0)]));
}
