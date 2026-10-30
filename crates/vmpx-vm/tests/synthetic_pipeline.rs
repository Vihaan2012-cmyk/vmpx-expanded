//! End-to-end pipeline tests over a SYNTHETIC VM (no protected sample needed).
//!
//! `support/synth_vm.rs` assembles a tiny VMProtect-shaped VM whose handlers
//! are the minimal shapes the current recognizers accept, plus encoded
//! bytecode that exits to a known native RVA. These tests then run the real
//! recovery stack over the resulting PE.
//!
//! Exercised (against the real library code, no mocks):
//! - PE parsing, `discover_vm_entries` (entry-point root and executable stub
//!   scan), `analyze_vm_entry_at`, instance grouping.
//! - `recover_bootstrap_dispatch`: unique VIP/key/handler role resolution,
//!   keyed relative-dispatch codec recovery, concrete bootstrap evaluation.
//! - `analyze_x64_native_frame` (19-slot frame), per-handler codec recovery
//!   and effect classification (context read/write, constant push, virtual
//!   branch, VM exit), linear decode, context-scatter mapping.
//! - Symbolic execution, branch-domain enumeration over `native_rdx_initial`,
//!   multi-path `recover_entry_paths`, and `NativeExitState` recovery: the
//!   exit registers, RFLAGS and the native continuation address (OEP).
//!
//! NOT exercised, and NOT proven by this file:
//! - Fidelity to real `VMProtect`. The generator was written against the
//!   recognizers, so passing is self-consistency, not evidence that real
//!   protected binaries are recognized. Real fixtures remain required.
//! - Arithmetic/memory/flag handlers, conditional branches, native calls,
//!   native-transition (multi-profile) chains, table checksums, descending or
//!   state-dependent bytecode, VSP-role ambiguity, junk/mutation, TLS roots,
//!   multiple VM instances, branch targets that are computed (here the target
//!   is the untouched native RDX).
//! - Everything downstream of `recover_entry_paths` (OEP selection policy,
//!   reintegration, rebuild, CLI `clean`).

#[path = "support/synth_vm.rs"]
mod synth_vm;

use std::collections::BTreeMap;

use vmpx_disasm::{decode_one, is_executable_rva};
use vmpx_pe::{PeImage, Rva, Va};
use vmpx_vm::{
    BitWidth, BlockExit, CfgEdgeKind, EntryPathRecoveryLimits, Expr, ExprKind, NativeExitState,
    RecoveryStatus, VmCfgLimits, VmDiscoveryLimits, VmEntryReachability, VmEntrySource,
    X64NativeRegister, analyze_vm_entry_at, discover_vm_entries, recover_bootstrap_dispatch,
    recover_entry_paths, recover_vm_cfg,
};

use synth_vm::SyntheticVm;

fn parse(vm: &SyntheticVm) -> PeImage<'_> {
    PeImage::parse(&vm.image).expect("synthetic PE parses")
}

fn entry_candidate(pe: &PeImage<'_>) -> vmpx_vm::VmEntryCandidate {
    discover_vm_entries(pe, VmDiscoveryLimits::default())
        .expect("discovery succeeds")
        .entries
        .into_iter()
        .find(|entry| entry.source == VmEntrySource::PeEntryPoint)
        .expect("entry-point VM candidate")
}

fn constant(value: u64) -> Expr {
    Expr::constant(BitWidth::W64, value)
}

fn rva_of(pe: &PeImage<'_>, expression: &Expr) -> Rva {
    let ExprKind::Const { value } = expression.kind else {
        panic!("not a constant: {expression:?}");
    };
    pe.optional
        .image_base
        .checked_rva_of(Va(value))
        .expect("address inside the image")
}

fn assert_native_registers_preserved(exit: &NativeExitState, except: &[X64NativeRegister]) {
    for register in X64NativeRegister::ALL {
        if except.contains(&register) {
            continue;
        }
        assert_eq!(
            exit.registers[&register],
            Expr::register(BitWidth::W64, register.symbolic_name()),
            "{register:?} must survive the VM round trip untouched"
        );
    }
    assert_eq!(
        exit.rflags,
        Expr::register(BitWidth::W64, "native_rflags_initial")
    );
}

fn assert_native_continuation(pe: &PeImage<'_>, exit: &NativeExitState, oep: u32) {
    let target = rva_of(pe, &exit.return_address);
    assert_eq!(target, Rva(oep));
    assert!(is_executable_rva(pe, target));
    decode_one(pe, target).expect("native continuation decodes");
    assert_eq!(exit.frame_size, 136);
}

#[test]
fn discovers_synthetic_vm_entry() {
    let vm = synth_vm::build_linear();
    let pe = parse(&vm);
    let report = discover_vm_entries(&pe, VmDiscoveryLimits::default()).expect("discovery");

    let [entry] = report.entries.as_slice() else {
        panic!("expected exactly one entry: {report:#?}");
    };
    assert_eq!(entry.rva, Rva(vm.entry_rva));
    assert_eq!(entry.source, VmEntrySource::PeEntryPoint);
    assert_eq!(entry.reachability, VmEntryReachability::LoaderReachable);
    assert_eq!(entry.seed, Some(u64::from(vm.seed)));
    assert_eq!(entry.bootstrap_rva, Some(Rva(vm.bootstrap_rva)));
    assert!(entry.confidence >= 90, "{entry:#?}");
    assert!(report.rejected_roots.is_empty());
    assert_eq!(report.instances.len(), 1);
    assert_eq!(report.instances[0].entry_rvas, vec![Rva(vm.entry_rva)]);
    assert_eq!(report.instances[0].bootstrap_rva, Rva(vm.bootstrap_rva));

    let direct = analyze_vm_entry_at(&pe, Rva(vm.entry_rva), VmDiscoveryLimits::default())
        .expect("direct analysis");
    assert_eq!(direct.seed, entry.seed);
    assert_eq!(direct.bootstrap_rva, entry.bootstrap_rva);
    // Direct analysis is never loader-reachable, so it lacks the 10-point root bonus.
    assert_eq!(direct.source, VmEntrySource::ExecutableScan);
    assert_eq!(direct.confidence + 10, entry.confidence);
}

#[test]
fn loader_root_discovery_does_not_need_the_stub_scan() {
    let vm = synth_vm::build_linear();
    let pe = parse(&vm);
    let report = discover_vm_entries(
        &pe,
        VmDiscoveryLimits {
            scan_executable_stubs: false,
            ..VmDiscoveryLimits::default()
        },
    )
    .expect("discovery");
    assert_eq!(report.scanned_stub_candidates, 0);
    assert_eq!(report.entries.len(), 1);
}

#[test]
fn non_vm_entry_point_is_rejected_but_stub_is_still_scanned() {
    let vm = synth_vm::build_linear();
    let mut image = vm.image.clone();
    let at = synth_vm::ENTRY_POINT_FIELD_OFFSET;
    image[at..at + 4].copy_from_slice(&vm.oep_rvas[0].to_le_bytes());
    let pe = PeImage::parse(&image).expect("parse");

    let loader_only = discover_vm_entries(
        &pe,
        VmDiscoveryLimits {
            scan_executable_stubs: false,
            ..VmDiscoveryLimits::default()
        },
    )
    .expect("discovery");
    assert!(loader_only.entries.is_empty(), "{loader_only:#?}");
    assert_eq!(loader_only.rejected_roots.len(), 1);

    let scanned = discover_vm_entries(&pe, VmDiscoveryLimits::default()).expect("discovery");
    let [stub] = scanned.entries.as_slice() else {
        panic!("scan must still find the stub: {scanned:#?}");
    };
    assert_eq!(stub.rva, Rva(vm.entry_rva));
    assert_eq!(
        stub.reachability,
        VmEntryReachability::UnprovenExecutableCandidate
    );
}

#[test]
fn recovers_unique_bootstrap_dispatch() {
    let vm = synth_vm::build_linear();
    let pe = parse(&vm);
    let entry = entry_candidate(&pe);
    let (report, recovered) =
        recover_bootstrap_dispatch(&pe, &entry, vmpx_disasm::DirectJumpTraceLimits::default())
            .expect("bootstrap trace decodes");
    let recovered =
        recovered.unwrap_or_else(|| panic!("bootstrap must be unambiguous: {report:#?}"));
    assert_eq!(report.candidates.len(), 1);
    assert_eq!(recovered.roles.vip, iced_x86::Register::R15);
    assert_eq!(recovered.roles.rolling_key, iced_x86::Register::RBP);
    assert_eq!(recovered.roles.handler, iced_x86::Register::R14);
    assert_eq!(recovered.initial_state.vip, Va(vm.block_vas[0]));
    assert_eq!(recovered.initial_state.rolling_key, vm.block_vas[0]);
    let first = recovered
        .codec
        .decode(&pe, recovered.initial_state)
        .expect("first dispatch decodes");
    assert_eq!(
        first.state_after.handler,
        Va(vm.image_base + u64::from(vm.write_handler_rva))
    );
}

#[test]
fn recovers_linear_entry_path_to_known_oep() {
    let vm = synth_vm::build_linear();
    let pe = parse(&vm);
    let entry = entry_candidate(&pe);
    let family = recover_entry_paths(
        &pe,
        &entry,
        &BTreeMap::new(),
        EntryPathRecoveryLimits::default(),
    )
    .expect("entry paths recover");

    let profile = &family.entry_profile;
    assert_eq!(profile.native_frame.status, RecoveryStatus::Recovered);
    assert_eq!(profile.native_frame.slots.len(), synth_vm::FRAME_SLOTS);
    assert_eq!(profile.context_mapping.status, RecoveryStatus::Recovered);
    assert_eq!(
        profile.operational_instruction_offset,
        synth_vm::FRAME_SLOTS
    );
    assert_eq!(profile.decode.decode.instructions.len(), 36);
    assert!(profile.symbolic.branch_target.is_none());

    assert!(family.domain_complete);
    assert!(family.all_paths_reach_vm_exit);
    let [path] = family.paths.as_slice() else {
        panic!("linear program has exactly one path");
    };
    assert!(path.assignments.is_empty());
    let exit = path.native_exit.as_ref().expect("native exit recovered");
    assert_native_continuation(&pe, exit, vm.oep_rvas[0]);
    assert_eq!(
        exit.registers[&X64NativeRegister::Rax],
        constant(synth_vm::FORCED_RAX)
    );
    assert_eq!(
        exit.registers[&X64NativeRegister::Rcx],
        constant(u64::from(vm.seed))
    );
    assert_native_registers_preserved(exit, &[X64NativeRegister::Rax, X64NativeRegister::Rcx]);
}

#[test]
fn recovers_each_branch_path_to_its_own_oep() {
    let vm = synth_vm::build_branching();
    let pe = parse(&vm);
    let entry = entry_candidate(&pe);
    let domains = BTreeMap::from([(
        "native_rdx_initial".to_owned(),
        vm.block_vas[1..]
            .iter()
            .map(|va| constant(*va))
            .collect::<Vec<_>>(),
    )]);
    let family = recover_entry_paths(&pe, &entry, &domains, EntryPathRecoveryLimits::default())
        .expect("branching entry paths recover");

    assert_eq!(
        family.entry_profile.symbolic.branch_target_dependencies,
        vec!["native_rdx_initial".to_owned()]
    );
    assert!(family.domain_complete);
    assert!(family.all_paths_reach_vm_exit);
    assert_eq!(family.paths.len(), 2);

    let outcomes = family
        .paths
        .iter()
        .map(|path| {
            let rdx = path.assignments["native_rdx_initial"];
            let exit = path.native_exit.as_ref().expect("native exit recovered");
            assert_eq!(exit.registers[&X64NativeRegister::Rdx], constant(rdx));
            assert_native_registers_preserved(
                exit,
                &[
                    X64NativeRegister::Rax,
                    X64NativeRegister::Rcx,
                    X64NativeRegister::Rdx,
                ],
            );
            let ExprKind::Const { value: rax } = exit.registers[&X64NativeRegister::Rax].kind
            else {
                panic!("RAX must be constant");
            };
            (rdx, (rva_of(&pe, &exit.return_address), rax))
        })
        .collect::<BTreeMap<_, _>>();
    assert_eq!(
        outcomes,
        BTreeMap::from([
            (
                vm.block_vas[1],
                (Rva(vm.oep_rvas[0]), synth_vm::BRANCH_RAX[0])
            ),
            (
                vm.block_vas[2],
                (Rva(vm.oep_rvas[1]), synth_vm::BRANCH_RAX[1])
            ),
        ])
    );
}

fn expected_exit_is_reached(
    image: &[u8],
    expected_oep: u32,
    domains: &BTreeMap<String, Vec<Expr>>,
) -> bool {
    let Ok(pe) = PeImage::parse(image) else {
        return false;
    };
    let Some(entry) = discover_vm_entries(&pe, VmDiscoveryLimits::default())
        .ok()
        .and_then(|report| {
            report
                .entries
                .into_iter()
                .find(|e| e.source == VmEntrySource::PeEntryPoint)
        })
    else {
        return false;
    };
    let Ok(family) = recover_entry_paths(&pe, &entry, domains, EntryPathRecoveryLimits::default())
    else {
        return false;
    };
    family.all_paths_reach_vm_exit
        && family.paths.iter().all(|path| {
            path.native_exit.as_ref().is_some_and(|exit| {
                exit.return_address
                    == constant(pe.optional.image_base.get() + u64::from(expected_oep))
            })
        })
}

#[test]
fn control_pristine_image_reaches_expected_exit() {
    let vm = synth_vm::build_linear();
    assert!(expected_exit_is_reached(
        &vm.image,
        vm.oep_rvas[0],
        &BTreeMap::new()
    ));
}

#[test]
fn corrupted_bytecode_does_not_reach_the_expected_exit() {
    let vm = synth_vm::build_linear();
    // First dispatch token, then the middle of the return-address operand.
    for offset in [0_u32, 99 + 2] {
        let mut image = vm.image.clone();
        image[synth_vm::file_offset(vm.bytecode_rva + offset)] ^= 0x5A;
        assert!(
            !expected_exit_is_reached(&image, vm.oep_rvas[0], &BTreeMap::new()),
            "flipping bytecode byte {offset} must break recovery"
        );
    }
}

#[test]
fn unprovable_vm_exit_handler_is_not_reported_as_an_exit() {
    let vm = synth_vm::build_linear();
    let mut image = vm.image.clone();
    let start = synth_vm::file_offset(vm.exit_handler_rva);
    let popfq = start
        + image[start..]
            .iter()
            .position(|byte| *byte == 0x9D)
            .expect("popfq in exit handler");
    assert_eq!(image[popfq + 1], 0xC3);
    image[popfq + 1] = 0xCC;
    assert!(!expected_exit_is_reached(
        &image,
        vm.oep_rvas[0],
        &BTreeMap::new()
    ));
}

fn walk_cfg(vm: &SyntheticVm) -> vmpx_vm::VmCfg {
    let pe = parse(vm);
    let entry = entry_candidate(&pe);
    recover_vm_cfg(&pe, &entry, VmCfgLimits::default()).expect("cfg recovery starts")
}

#[test]
fn cfg_walk_of_a_linear_function_is_one_exit_block() {
    let cfg = walk_cfg(&synth_vm::build_linear());
    assert_eq!(cfg.blocks.len(), 1, "{cfg:#?}");
    assert!(cfg.edges.is_empty());
    assert!(cfg.all_paths_reach_vm_exit(), "{cfg:#?}");
    let block = cfg.blocks.values().next().expect("block");
    assert_eq!(block.exit, BlockExit::VmExit);
    assert_eq!(block.instructions.len(), 36);
    assert!(block.vm_exit.is_some(), "{cfg:#?}");
    assert_eq!(cfg.exit_layouts().len(), 1);
}

#[test]
fn cfg_walk_follows_a_branch_through_a_pushed_constant_to_both_blocks() {
    let vm = synth_vm::build_const_branch(None);
    let cfg = walk_cfg(&vm);
    assert_eq!(cfg.blocks.len(), 2, "{cfg:#?}");
    assert!(cfg.stalls.is_empty(), "{cfg:#?}");
    assert!(cfg.all_paths_reach_vm_exit(), "{cfg:#?}");
    let [edge] = cfg.edges.as_slice() else {
        panic!("one proven edge expected: {cfg:#?}");
    };
    assert_eq!(edge.kind, CfgEdgeKind::Branch);
    assert_eq!(edge.from, cfg.entry);
    let exits: Vec<_> = cfg.blocks.values().map(|block| block.exit).collect();
    assert!(exits.contains(&BlockExit::Branch) && exits.contains(&BlockExit::VmExit));
    let successor = &cfg.blocks[&edge.to];
    assert_eq!(successor.address, Some(Va(vm.block_vas[1])));
    assert_eq!(successor.exit, BlockExit::VmExit);
}

#[test]
fn cfg_walk_stops_at_a_branch_target_it_cannot_prove() {
    // The target is the saved native RDX itself: no finite proven domain.
    let cfg = walk_cfg(&synth_vm::build_branching());
    assert_eq!(cfg.blocks.len(), 1, "{cfg:#?}");
    assert!(cfg.edges.is_empty(), "no edge may be invented: {cfg:#?}");
    let [stall] = cfg.stalls.as_slice() else {
        panic!("one stall expected: {cfg:#?}");
    };
    assert!(
        stall.reason.contains("unresolved branch target"),
        "{}",
        stall.reason
    );
    assert!(!cfg.is_complete() && !cfg.all_paths_reach_vm_exit());
}

#[test]
fn cfg_walk_refuses_a_constant_target_outside_the_bytecode_section() {
    let cfg = walk_cfg(&synth_vm::build_const_branch(Some(0x1234)));
    assert_eq!(cfg.blocks.len(), 1, "{cfg:#?}");
    assert!(cfg.edges.is_empty());
    assert!(
        cfg.stalls
            .iter()
            .any(|stall| stall.reason.contains("outside the bytecode section")),
        "{cfg:#?}"
    );
}
