use super::*;
use iced_x86::{Decoder, DecoderOptions, Mnemonic};
use vmpx_pe::Rva;
use vmpx_vm::{
    CfgEdge, DecodedVmInstruction, EffectKnowledge, RecoveryStatus, RelativeDispatchState,
    VmBasicBlock, VmExitRestoreSlot,
};

const EXIT_HANDLER: Va = Va(0xE0);
const BRANCH_BASE: u64 = 0x1000;
const ZERO_FLAG_STEP: u64 = 0x100;
const ADDRESS: u64 = 0x1_8000_0000;

fn state(vip: u64, handler: Va) -> RelativeDispatchState {
    RelativeDispatchState {
        vip: Va(vip),
        rolling_key: 0,
        handler,
    }
}

fn ins(semantic: VmHandlerSemantic, operand: Option<u64>, vsp_delta: i32) -> DecodedVmInstruction {
    DecodedVmInstruction {
        state_before: state(0x10, Va(0x20)),
        encoded_bytes: Vec::new(),
        semantic,
        semantic_metadata: None,
        input_width: None,
        operation_width: Some(BitWidth::W64),
        resize: None,
        operand,
        vsp_delta: Some(vsp_delta),
        flags_effect: EffectKnowledge::Preserved,
        confidence: 100,
        state_after: state(0x14, Va(0x30)),
    }
}

fn block(
    key: u64,
    address: Option<u64>,
    mut instructions: Vec<DecodedVmInstruction>,
    exit: BlockExit,
) -> (Va, VmBasicBlock) {
    let last_handler = if exit == BlockExit::VmExit {
        EXIT_HANDLER
    } else {
        Va(0x30)
    };
    if let Some(last) = instructions.last_mut() {
        last.state_after = state(0x14, last_handler);
    }
    (
        Va(key),
        VmBasicBlock {
            start: state(key, EXIT_HANDLER),
            address: address.map(Va),
            instructions,
            exit,
        },
    )
}

fn edge(from: u64, to: u64, kind: CfgEdgeKind) -> CfgEdge {
    CfgEdge {
        from: Va(from),
        to: Va(to),
        kind,
    }
}

fn cfg(blocks: Vec<(Va, VmBasicBlock)>, edges: Vec<CfgEdge>, entry: u64) -> VmCfg {
    VmCfg {
        entry: Va(entry),
        blocks: blocks.into_iter().collect(),
        edges,
        stalls: Vec::new(),
        limit_hit: None,
        visits: 0,
        frame_cells_known: 17,
        frame_note: None,
    }
}

/// Two scratch cells on top, then the 14 non-RAX registers and the flags.
fn frame(second: NativeFrameSource) -> Vec<NativeFrameSource> {
    let mut cells = vec![
        NativeFrameSource::Register {
            register: X64NativeRegister::Rdx,
        },
        second,
    ];
    cells.extend(
        X64NativeRegister::ALL[1..]
            .iter()
            .map(|register| NativeFrameSource::Register {
                register: *register,
            }),
    );
    cells.push(NativeFrameSource::Rflags);
    cells
}

/// The exit frame starts at the entry frame's second cell: RAX first.
fn exits() -> BTreeMap<Va, VmExitReport> {
    let report = VmExitReport {
        schema_version: 2,
        entry: Rva(0x1000),
        status: RecoveryStatus::Recovered,
        stack_switch: None,
        restored_registers: Vec::new(),
        restore_slots: X64NativeRegister::ALL
            .into_iter()
            .enumerate()
            .map(|(index, register)| VmExitRestoreSlot {
                register,
                stack_offset: u32::try_from(index).expect("small") * 8,
                pop: Rva(0x1001),
            })
            .collect(),
        flags_restore: Some(Rva(0x1010)),
        flags_stack_offset: Some(120),
        native_return: Some(Rva(0x1011)),
        return_stack_offset: Some(128),
        frame_size: Some(136),
        confidence: 100,
        evidence: Vec::new(),
        diagnostics: Vec::new(),
    };
    BTreeMap::from([(EXIT_HANDLER, report)])
}

/// `target = BRANCH_BASE + ZF(rdx - 1) * ZERO_FLAG_STEP`, left on the VM stack,
/// with the two scratch cells consumed.
fn zero_flag_target_program() -> Vec<DecodedVmInstruction> {
    use VmHandlerSemantic as S;
    let spill = |slot| ins(S::VirtualRegisterWrite, Some(slot), 8);
    let reload = |slot| ins(S::VirtualRegisterRead, Some(slot), -8);
    let push = |value| ins(S::ConstantPush, Some(value), -8);
    vec![
        ins(S::Sub, None, 0),
        spill(0x10),
        spill(0x18),
        push(57),
        reload(0x10),
        ins(S::ShiftLeft, None, 0),
        spill(0x20),
        spill(0x10),
        push(63),
        reload(0x10),
        ins(S::ShiftRightLogical, None, 0),
        spill(0x20),
        spill(0x10),
        push(8),
        reload(0x10),
        ins(S::ShiftLeft, None, 0),
        spill(0x20),
        push(BRANCH_BASE),
        ins(S::Add, None, 0),
        spill(0x20),
    ]
}

fn returns_constant(key: u64, address: u64, value: u64) -> (Va, VmBasicBlock) {
    block(
        key,
        Some(address),
        vec![ins(VmHandlerSemantic::ConstantPush, Some(value), -8)],
        BlockExit::VmExit,
    )
}

fn conditional_cfg() -> VmCfg {
    cfg(
        vec![
            block(
                0xA0,
                Some(0x900),
                zero_flag_target_program(),
                BlockExit::Branch,
            ),
            returns_constant(0xB0, BRANCH_BASE, 111),
            returns_constant(0xC0, BRANCH_BASE + ZERO_FLAG_STEP, 222),
        ],
        vec![
            edge(0xA0, 0xB0, CfgEdgeKind::GuardedBranch),
            edge(0xA0, 0xC0, CfgEdgeKind::GuardedBranch),
        ],
        0xA0,
    )
}

fn lift_with(
    cfg: &VmCfg,
    frame: &[NativeFrameSource],
    exits: &BTreeMap<Va, VmExitReport>,
    calls: &BTreeMap<Va, NativeCallSelectorOutcome>,
) -> Result<LiftedCfgFunction, DecodedLiftError> {
    let input = DecodedCfgInput {
        cfg,
        frame,
        exits,
        native_calls: calls,
        memory: None,
        limits: SymbolicExecutionLimits::default(),
        target_limits: VmCfgLimits::default(),
        max_expanded_blocks: 64,
    };
    lift_decoded_cfg(&input, CodeAddress(ADDRESS))
}

fn lift(
    cfg: &VmCfg,
    frame: &[NativeFrameSource],
    calls: &BTreeMap<Va, NativeCallSelectorOutcome>,
) -> Result<LiftedCfgFunction, DecodedLiftError> {
    lift_with(cfg, frame, &exits(), calls)
}

fn mnemonics(lifted: &LiftedCfgFunction) -> Vec<Mnemonic> {
    Decoder::with_ip(
        64,
        &lifted.replacement.compiled.bytes,
        ADDRESS,
        DecoderOptions::NONE,
    )
    .iter()
    .map(|instruction| instruction.mnemonic())
    .collect()
}

fn count(operations: &[Operation], matcher: impl Fn(&Operation) -> bool) -> usize {
    operations
        .iter()
        .filter(|operation| matcher(operation))
        .count()
}

fn constant_frame() -> Vec<NativeFrameSource> {
    frame(NativeFrameSource::Constant { value: 1 })
}

#[test]
fn a_two_block_conditional_branch_lifts_to_a_native_compare_and_jump() {
    let lifted = lift(&conditional_cfg(), &constant_frame(), &BTreeMap::new())
        .expect("conditional branch lifts");
    assert_eq!(lifted.conditional_branches, 1);
    assert_eq!(lifted.exit_paths, 2);
    assert_eq!(lifted.blocks.len(), 3);
    assert_eq!(
        lifted.assumptions,
        vec![LiftAssumption::FlagsVolatileAtReturn]
    );

    let operations = &lifted.replacement.function.operations;
    assert_eq!(
        count(operations, |o| matches!(
            o,
            Operation::BranchIf {
                condition: Condition::Equal,
                ..
            }
        )),
        1
    );
    assert_eq!(count(operations, |o| matches!(o, Operation::Label(_))), 1);
    assert_eq!(
        count(operations, |o| matches!(
            o,
            Operation::ReturnPopping { bytes: 0 }
        )),
        2
    );
    // The lowest target is compared against; the other arm falls through.
    assert_eq!(
        count(operations, |o| matches!(
            o,
            Operation::MoveImmediate { value, .. } if *value == BRANCH_BASE
        )),
        2
    );
    for leaf_value in [111_u64, 222] {
        assert_eq!(
            count(operations, |o| matches!(
                o,
                Operation::MoveImmediate { value, .. } if *value == leaf_value
            )),
            1,
            "leaf constant {leaf_value}"
        );
    }
    assert!(count(operations, |o| matches!(o, Operation::ReadFlags { .. })) >= 1);
    let decoded = mnemonics(&lifted);
    for expected in [Mnemonic::Pushfq, Mnemonic::Je, Mnemonic::Ret] {
        assert!(
            decoded.contains(&expected),
            "missing {expected:?} in {decoded:?}"
        );
    }
    assert_eq!(decoded.iter().filter(|m| **m == Mnemonic::Ret).count(), 2);
}

fn call_frame_cfg() -> (VmCfg, BTreeMap<Va, NativeCallSelectorOutcome>) {
    let cfg = cfg(
        vec![
            block(0xA0, Some(0x900), Vec::new(), BlockExit::NativeCall),
            block(0xB0, None, Vec::new(), BlockExit::VmExit),
        ],
        vec![edge(0xA0, 0xB0, CfgEdgeKind::NativeCallReturn)],
        0xA0,
    );
    let outcome = NativeCallSelectorOutcome {
        selector: 0,
        vsp_delta: 8,
        target_stack_offset: 0,
        register_argument_offsets: vec![8],
        stack_argument_offsets: Vec::new(),
    };
    (cfg, BTreeMap::from([(Va(0xA0), outcome)]))
}

#[test]
fn a_block_ending_in_a_native_call_lifts_with_the_call_and_the_return_block() {
    let (cfg, calls) = call_frame_cfg();
    let lifted = lift(
        &cfg,
        &frame(NativeFrameSource::Register {
            register: X64NativeRegister::Rcx,
        }),
        &calls,
    )
    .expect("native call lifts");
    assert_eq!(lifted.native_calls, 1);
    assert_eq!(lifted.exit_paths, 1);
    assert!(
        lifted
            .assumptions
            .contains(&LiftAssumption::VmStateSurvivesNativeCall)
    );

    let operations = &lifted.replacement.function.operations;
    let calls: Vec<_> = operations
        .iter()
        .filter_map(|o| match o {
            Operation::Call {
                target,
                arguments,
                result,
            } => Some((target, arguments, result)),
            _ => None,
        })
        .collect();
    let [(CallTarget::Register(_), arguments, Some(_))] = calls[..] else {
        panic!("expected one register-indirect call, got {calls:?}");
    };
    assert_eq!(arguments.len(), 1);
    assert!(lifted.replacement.function.frame.call_ready);
    let decoded = mnemonics(&lifted);
    assert!(decoded.contains(&Mnemonic::Call));
    assert_eq!(decoded.iter().filter(|m| **m == Mnemonic::Ret).count(), 1);
}

#[test]
fn a_native_call_that_may_receive_a_vm_stack_pointer_refuses() {
    let cfg = cfg(
        vec![
            block(
                0xA0,
                Some(0x900),
                vec![ins(VmHandlerSemantic::Push, None, -8)],
                BlockExit::NativeCall,
            ),
            block(0xB0, None, Vec::new(), BlockExit::VmExit),
        ],
        vec![edge(0xA0, 0xB0, CfgEdgeKind::NativeCallReturn)],
        0xA0,
    );
    let outcome = NativeCallSelectorOutcome {
        selector: 0,
        vsp_delta: 8,
        target_stack_offset: 8,
        register_argument_offsets: vec![0],
        stack_argument_offsets: Vec::new(),
    };
    let calls = BTreeMap::from([(Va(0xA0), outcome)]);
    let error = lift(
        &cfg,
        &frame(NativeFrameSource::Register {
            register: X64NativeRegister::Rcx,
        }),
        &calls,
    )
    .expect_err("a stack pointer argument must refuse");
    assert!(error.to_string().contains("VM stack"), "{error}");
}

#[test]
fn a_branch_target_without_a_proven_edge_refuses_cleanly() {
    let mut broken = conditional_cfg();
    broken.edges.retain(|edge| edge.to != Va(0xC0));
    let error = lift(&broken, &constant_frame(), &BTreeMap::new())
        .expect_err("an unproven edge must refuse");
    assert!(error.to_string().contains("no proven CFG edge"), "{error}");
}

#[test]
fn a_target_that_is_not_a_recovered_block_refuses() {
    let mut broken = conditional_cfg();
    broken.blocks.remove(&Va(0xC0));
    let error = lift(&broken, &constant_frame(), &BTreeMap::new())
        .expect_err("a missing block must refuse");
    assert!(
        error.to_string().contains("not a recovered block"),
        "{error}"
    );
}

#[test]
fn an_unproven_branch_condition_refuses() {
    // The target is a raw register, with no finite proven domain.
    let cfg = cfg(
        vec![
            block(0xA0, Some(0x900), Vec::new(), BlockExit::Branch),
            returns_constant(0xB0, BRANCH_BASE, 1),
        ],
        vec![edge(0xA0, 0xB0, CfgEdgeKind::GuardedBranch)],
        0xA0,
    );
    let error = lift(&cfg, &constant_frame(), &BTreeMap::new())
        .expect_err("an unproven condition must refuse");
    assert!(error.to_string().contains("branch target"), "{error}");
}

#[test]
fn a_loop_and_an_incomplete_cfg_refuse() {
    let looping = cfg(
        vec![block(
            0xA0,
            Some(0x900),
            vec![ins(VmHandlerSemantic::ConstantPush, Some(0x900), -8)],
            BlockExit::Branch,
        )],
        vec![edge(0xA0, 0xA0, CfgEdgeKind::Branch)],
        0xA0,
    );
    let error =
        lift(&looping, &constant_frame(), &BTreeMap::new()).expect_err("a loop must refuse");
    assert!(error.to_string().contains("loop"), "{error}");

    let mut incomplete = conditional_cfg();
    incomplete.stalls.push(vmpx_vm::CfgStall {
        block: Va(0xA0),
        vip: Va(0xA0),
        reason: "stalled".to_owned(),
    });
    let error = lift(&incomplete, &constant_frame(), &BTreeMap::new())
        .expect_err("an incomplete CFG must refuse");
    assert!(error.to_string().contains("incomplete"), "{error}");
}

fn identity_exit() -> Node {
    Node {
        steps: Vec::new(),
        end: End::Exit(ExitLeaf {
            registers: X64NativeRegister::ALL
                .iter()
                .map(|register| {
                    (
                        *register,
                        Expr::register(BitWidth::W64, register.symbolic_name()),
                    )
                })
                .collect(),
            return_pop: 8,
        }),
    }
}

fn entry(register: X64NativeRegister) -> Expr {
    Expr::register(BitWidth::W64, register.symbolic_name())
}

fn tree_config() -> ReplacementConfig {
    ReplacementConfig {
        inputs: X64NativeRegister::ALL
            .iter()
            .map(|register| (register.symbolic_name().to_owned(), physical_for(register)))
            .collect(),
        max_expression_nodes: 10_000,
        max_spill_slots: 64,
        import_calls: BTreeMap::new(),
        continuation: None,
        effects: Vec::new(),
        effect_calls: Vec::new(),
        rflags: None,
        symbol_rewrites: BTreeMap::new(),
    }
}

#[test]
fn a_three_way_branch_and_a_narrow_store_keep_program_order() {
    let narrow = Expr::extract(BitWidth::W32, 0, entry(X64NativeRegister::Rdx)).expect("extract");
    let mut first = identity_exit();
    first.steps.push(Step::Store(EffectStore {
        address: entry(X64NativeRegister::Rcx),
        value: narrow,
        width: BitWidth::W32,
    }));
    let root = Node {
        steps: Vec::new(),
        end: End::Branch {
            condition: entry(X64NativeRegister::Rax),
            arms: vec![(3, first), (5, identity_exit()), (9, identity_exit())],
        },
    };
    let replacement =
        build_tree_replacement(&root, 0, None, &tree_config(), CodeAddress(ADDRESS)).expect("emit");
    let operations = &replacement.function.operations;
    assert_eq!(
        count(operations, |o| matches!(o, Operation::BranchIf { .. })),
        2
    );
    assert_eq!(count(operations, |o| matches!(o, Operation::Label(_))), 2);
    assert_eq!(
        count(operations, |o| matches!(
            o,
            Operation::ReturnPopping { bytes: 8 }
        )),
        3
    );
    let stores: Vec<_> = operations
        .iter()
        .filter_map(|o| match o {
            Operation::Store { source, .. } => Some(source.width),
            _ => None,
        })
        .collect();
    assert_eq!(stores, vec![Width::W32]);
    let decoded: Vec<Mnemonic> = Decoder::with_ip(
        64,
        &replacement.compiled.bytes,
        ADDRESS,
        DecoderOptions::NONE,
    )
    .iter()
    .map(|instruction| instruction.mnemonic())
    .collect();
    assert_eq!(decoded.iter().filter(|m| **m == Mnemonic::Je).count(), 2);
    assert_eq!(decoded.iter().filter(|m| **m == Mnemonic::Ret).count(), 3);
}

#[test]
fn a_missing_exit_layout_refuses() {
    let error = lift_with(
        &conditional_cfg(),
        &constant_frame(),
        &BTreeMap::new(),
        &BTreeMap::new(),
    )
    .expect_err("no layout");
    assert!(error.to_string().contains("VM-exit layout"), "{error}");
}
