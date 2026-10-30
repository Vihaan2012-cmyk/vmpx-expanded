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
            boundary_handler: None,
            native_call: None,
            vm_exit: None,
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
        assumptions: std::collections::BTreeSet::new(),
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
        top_offset: 0,
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

/// Like `constant_frame`, but the flags cell carries a computed value, so the
/// VM-exit hands the caller flags that differ from the entry flags.
fn computed_flags_frame() -> Vec<NativeFrameSource> {
    let mut cells = constant_frame();
    *cells.last_mut().expect("flags cell") = NativeFrameSource::Constant { value: 0x44 };
    cells
}

#[test]
fn a_flag_carrying_popping_return_exit_is_accepted_and_restores_the_flags() {
    // The conditional CFG's two VM exits are callee (popping) returns; the
    // computed-flags frame makes them carry flags that differ from entry.
    let lifted = lift(&conditional_cfg(), &computed_flags_frame(), &BTreeMap::new())
        .expect("a flag-carrying popping return lifts");
    assert_eq!(lifted.exit_paths, 2);
    // The carried flags are the real result; they are restored, not abandoned as
    // volatile, because the popping return now preserves them.
    assert!(
        !lifted
            .assumptions
            .contains(&LiftAssumption::FlagsVolatileAtReturn)
    );
    let operations = &lifted.replacement.function.operations;
    assert_eq!(
        count(operations, |o| matches!(o, Operation::RestoreFlags { .. })),
        2
    );
    assert_eq!(
        count(operations, |o| matches!(o, Operation::ReturnPopping { .. })),
        2
    );
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
fn a_branch_without_any_proven_edge_refuses_cleanly() {
    // With the branch targets taken from the CFG edges, a branch block that
    // carries no proven edge has no successors to dispatch on and fails closed.
    let mut broken = conditional_cfg();
    broken.edges.retain(|edge| edge.from != Va(0xA0));
    let error = lift(&broken, &constant_frame(), &BTreeMap::new())
        .expect_err("a branch with no proven edge must refuse");
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
fn a_guarded_branch_with_a_single_proven_edge_refuses() {
    // A forked branch that resolved to only one successor is not exhaustively
    // covered, so the lift refuses rather than dispatch on a lone arm.
    let cfg = cfg(
        vec![
            block(0xA0, Some(0x900), Vec::new(), BlockExit::Branch),
            returns_constant(0xB0, BRANCH_BASE, 1),
        ],
        vec![edge(0xA0, 0xB0, CfgEdgeKind::GuardedBranch)],
        0xA0,
    );
    let error = lift(&cfg, &constant_frame(), &BTreeMap::new())
        .expect_err("a single-target guarded branch must refuse");
    assert!(error.to_string().contains("single target"), "{error}");
}

#[test]
fn a_guarded_branch_lifts_from_its_cfg_edges_and_carries_the_preferred_base_assumption() {
    // The branch dispatches on the successor addresses recovered by the walk,
    // taken straight from the two GuardedBranch edges rather than re-derived.
    // To prove the values come from the edges, the successor blocks are given
    // addresses that differ from what the branch expression enumerates to; the
    // arm compared against is the lower successor address, not the expression's
    // own constant. The CFG resolved the branch under its preferred-base frame,
    // so the lifted function inherits that assumption.
    const LOW_TARGET: u64 = 0x7000;
    const HIGH_TARGET: u64 = 0x7100;
    let mut cfg = cfg(
        vec![
            block(
                0xA0,
                Some(0x900),
                zero_flag_target_program(),
                BlockExit::Branch,
            ),
            returns_constant(0xB0, LOW_TARGET, 111),
            returns_constant(0xC0, HIGH_TARGET, 222),
        ],
        vec![
            edge(0xA0, 0xB0, CfgEdgeKind::GuardedBranch),
            edge(0xA0, 0xC0, CfgEdgeKind::GuardedBranch),
        ],
        0xA0,
    );
    cfg.assumptions
        .insert(vmpx_vm::VmCfgAssumption::PreferredImageBase);
    let lifted = lift(&cfg, &constant_frame(), &BTreeMap::new())
        .expect("a guarded branch lifts from its proven edges");
    assert_eq!(lifted.conditional_branches, 1);
    assert_eq!(lifted.exit_paths, 2);
    assert!(
        lifted
            .assumptions
            .contains(&LiftAssumption::PreferredImageBase)
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
    // The arm dispatches on the recovered successor address, not on a value the
    // branch expression itself resolves to.
    assert!(
        count(operations, |o| matches!(
            o,
            Operation::MoveImmediate { value, .. } if *value == LOW_TARGET
        )) >= 1,
        "dispatches on the recovered successor address"
    );
    for leaf in [111_u64, 222] {
        assert_eq!(
            count(operations, |o| matches!(
                o,
                Operation::MoveImmediate { value, .. } if *value == leaf
            )),
            1,
            "leaf constant {leaf}"
        );
    }
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

/// A block that pushes `target` and branches: with two CFG edges it is a two-way
/// branch, with one it is an unconditional jump. Address `address`, key `key`.
fn branch_block(key: u64, address: u64) -> (Va, VmBasicBlock) {
    block(
        key,
        Some(address),
        vec![ins(VmHandlerSemantic::ConstantPush, Some(0x900), -8)],
        BlockExit::Branch,
    )
}

/// Entry that consumes the two scratch frame cells (so the exit frame lines up),
/// seeds context slot `0x28` to 0, then branches to the header.
fn loop_entry(key: u64, address: u64) -> (Va, VmBasicBlock) {
    use VmHandlerSemantic as S;
    block(
        key,
        Some(address),
        vec![
            ins(S::VirtualRegisterWrite, Some(0x10), 8),
            ins(S::VirtualRegisterWrite, Some(0x18), 8),
            ins(S::ConstantPush, Some(0), -8),
            ins(S::VirtualRegisterWrite, Some(0x28), 8),
            ins(S::ConstantPush, Some(0x900), -8),
        ],
        BlockExit::Branch,
    )
}

/// Loop body that writes a fresh constant into slot `0x28` each iteration, then
/// branches back to the header: a real per-iteration update (init 0, next 0x55).
fn loop_body_const(key: u64, address: u64) -> (Va, VmBasicBlock) {
    use VmHandlerSemantic as S;
    block(
        key,
        Some(address),
        vec![
            ins(S::ConstantPush, Some(0x55), -8),
            ins(S::VirtualRegisterWrite, Some(0x28), 8),
            ins(S::ConstantPush, Some(0x900), -8),
        ],
        BlockExit::Branch,
    )
}

const LOOP_HEADER_ADDR: u64 = 0x910;
const LOOP_BODY_ADDR: u64 = 0x920;

/// A reducible natural loop: entry seeds the counter, the header tests it and
/// either runs the body (back-edge) or leaves to a VM-exit.
fn reducible_loop_cfg() -> VmCfg {
    cfg(
        vec![
            loop_entry(0xA0, 0x900),
            branch_block(0xB0, LOOP_HEADER_ADDR),
            loop_body_const(0xC0, LOOP_BODY_ADDR),
            returns_constant(0xD0, 0x930, 111),
        ],
        vec![
            edge(0xA0, 0xB0, CfgEdgeKind::Branch),
            edge(0xB0, 0xC0, CfgEdgeKind::GuardedBranch),
            edge(0xB0, 0xD0, CfgEdgeKind::GuardedBranch),
            edge(0xC0, 0xB0, CfgEdgeKind::GuardedBranch),
            edge(0xC0, 0xB0, CfgEdgeKind::LoopBack),
        ],
        0xA0,
    )
}

#[test]
fn a_reducible_loop_emits_a_native_back_edge_not_a_refusal() {
    let lifted = lift(&reducible_loop_cfg(), &constant_frame(), &BTreeMap::new())
        .expect("a reducible natural loop lifts");
    assert_eq!(lifted.loops, 1, "one natural loop emitted");
    assert_eq!(lifted.exit_paths, 1);
    let operations = &lifted.replacement.function.operations;
    // A backward jump closes the loop; the acyclic emitter never emits `Jump`.
    assert_eq!(
        count(operations, |o| matches!(o, Operation::Jump(_))),
        1,
        "the loop back-edge is a native jump"
    );
    // The loop-exit branch dispatches on the body (continue) address.
    assert!(
        count(operations, |o| matches!(
            o,
            Operation::MoveImmediate { value, .. } if *value == LOOP_BODY_ADDR
        )) >= 1,
        "dispatches on the recovered continue address"
    );
    // The per-iteration value is the real body constant, not a widened unknown.
    assert!(
        count(operations, |o| matches!(
            o,
            Operation::MoveImmediate { value, .. } if *value == 0x55
        )) >= 1,
        "emits the real per-iteration update"
    );
    let decoded = mnemonics(&lifted);
    assert!(decoded.contains(&Mnemonic::Jmp), "a back-edge jump: {decoded:?}");
}

/// Entry seeding two loop-carried slots to distinct constants, consuming the
/// scratch frame cells, then branching to the header.
fn swap_entry(key: u64, address: u64) -> (Va, VmBasicBlock) {
    use VmHandlerSemantic as S;
    block(
        key,
        Some(address),
        vec![
            ins(S::VirtualRegisterWrite, Some(0x10), 8),
            ins(S::VirtualRegisterWrite, Some(0x18), 8),
            ins(S::ConstantPush, Some(0), -8),
            ins(S::VirtualRegisterWrite, Some(0x28), 8),
            ins(S::ConstantPush, Some(7), -8),
            ins(S::VirtualRegisterWrite, Some(0x30), 8),
            ins(S::ConstantPush, Some(0x900), -8),
        ],
        BlockExit::Branch,
    )
}

/// Body that swaps slots `0x28` and `0x30` each iteration: each slot's next
/// value is the other's current value, so emitting it must read the phi slots.
fn swap_body(key: u64, address: u64) -> (Va, VmBasicBlock) {
    use VmHandlerSemantic as S;
    block(
        key,
        Some(address),
        vec![
            ins(S::VirtualRegisterRead, Some(0x28), -8),
            ins(S::VirtualRegisterRead, Some(0x30), -8),
            ins(S::VirtualRegisterWrite, Some(0x28), 8),
            ins(S::VirtualRegisterWrite, Some(0x30), 8),
            ins(S::ConstantPush, Some(0x900), -8),
        ],
        BlockExit::Branch,
    )
}

#[test]
fn a_loop_with_phi_dependent_updates_emits_both_back_edge_variables() {
    let looping = cfg(
        vec![
            swap_entry(0xA0, 0x900),
            branch_block(0xB0, LOOP_HEADER_ADDR),
            swap_body(0xC0, LOOP_BODY_ADDR),
            returns_constant(0xD0, 0x930, 111),
        ],
        vec![
            edge(0xA0, 0xB0, CfgEdgeKind::Branch),
            edge(0xB0, 0xC0, CfgEdgeKind::GuardedBranch),
            edge(0xB0, 0xD0, CfgEdgeKind::GuardedBranch),
            edge(0xC0, 0xB0, CfgEdgeKind::GuardedBranch),
            edge(0xC0, 0xB0, CfgEdgeKind::LoopBack),
        ],
        0xA0,
    );
    let lifted =
        lift(&looping, &constant_frame(), &BTreeMap::new()).expect("a phi-dependent loop lifts");
    assert_eq!(lifted.loops, 1);
    // Two loop-carried cells, each seeded to its own constant before the loop.
    for seed in [0_u64, 7] {
        assert!(
            count(&lifted.replacement.function.operations, |o| matches!(
                o,
                Operation::MoveImmediate { value, .. } if *value == seed
            )) >= 1,
            "seeds loop variable to {seed}"
        );
    }
    assert_eq!(
        count(&lifted.replacement.function.operations, |o| matches!(
            o,
            Operation::Jump(_)
        )),
        1
    );
}

#[test]
fn a_reducible_two_exit_loop_emits_both_exits() {
    // A `while` test at the header (exit to 0xD0) and a conditional break in the
    // body (exit to 0xE0), both leaving the same reducible loop. The body still
    // carries a real per-iteration update (slot 0x28: 0 -> 0x55), so each exit's
    // loop-carried state is the phi value at its own point.
    let looping = cfg(
        vec![
            loop_entry(0xA0, 0x900),
            branch_block(0xB0, LOOP_HEADER_ADDR),
            loop_body_const(0xC0, LOOP_BODY_ADDR),
            returns_constant(0xD0, 0x930, 111),
            returns_constant(0xE0, 0x940, 222),
        ],
        vec![
            edge(0xA0, 0xB0, CfgEdgeKind::Branch),
            edge(0xB0, 0xC0, CfgEdgeKind::GuardedBranch),
            edge(0xB0, 0xD0, CfgEdgeKind::GuardedBranch),
            edge(0xC0, 0xB0, CfgEdgeKind::GuardedBranch),
            edge(0xC0, 0xB0, CfgEdgeKind::LoopBack),
            edge(0xC0, 0xE0, CfgEdgeKind::GuardedBranch),
        ],
        0xA0,
    );
    let lifted = lift(&looping, &constant_frame(), &BTreeMap::new())
        .expect("a reducible two-exit loop lifts");
    assert_eq!(lifted.loops, 1, "one natural loop emitted");
    assert_eq!(lifted.exit_paths, 2, "both exit continuations are lifted");
    assert_eq!(lifted.conditional_branches, 2, "both exit branches are emitted");
    let operations = &lifted.replacement.function.operations;
    // One back-edge jump closes the loop; each exit is a conditional break.
    assert_eq!(
        count(operations, |o| matches!(o, Operation::Jump(_))),
        1,
        "exactly one back-edge jump"
    );
    assert_eq!(
        count(operations, |o| matches!(
            o,
            Operation::BranchIf {
                condition: Condition::NotEqual,
                ..
            }
        )),
        2,
        "two conditional-break branches"
    );
    // Each exit dispatches on its own in-loop continue address: the body for the
    // header test, the header itself for the body break.
    for continue_addr in [LOOP_BODY_ADDR, LOOP_HEADER_ADDR] {
        assert!(
            count(operations, |o| matches!(
                o,
                Operation::MoveImmediate { value, .. } if *value == continue_addr
            )) >= 1,
            "dispatches on continue address {continue_addr:#x}"
        );
    }
    // The real per-iteration update still runs on the back-edge.
    assert!(
        count(operations, |o| matches!(
            o,
            Operation::MoveImmediate { value, .. } if *value == 0x55
        )) >= 1,
        "emits the per-iteration update"
    );
    // Both continuations return.
    assert_eq!(
        count(operations, |o| matches!(o, Operation::ReturnPopping { .. })),
        2,
        "each exit continuation returns"
    );
}

#[test]
fn an_in_loop_branch_fails_closed() {
    // Reducible (0xB0 dominates the whole body), but the header forks to two
    // in-loop blocks: an internal branch diamond, not the single-cycle body the
    // structured form supports. It must fail closed rather than mis-recover it.
    let looping = cfg(
        vec![
            branch_block(0xA0, 0x900),
            branch_block(0xB0, 0x910),
            branch_block(0xC0, 0x920),
            branch_block(0xD0, 0x930),
            branch_block(0xE0, 0x940),
            returns_constant(0xF0, 0x950, 111),
        ],
        vec![
            edge(0xA0, 0xB0, CfgEdgeKind::Branch),
            edge(0xB0, 0xC0, CfgEdgeKind::GuardedBranch),
            edge(0xB0, 0xD0, CfgEdgeKind::GuardedBranch),
            edge(0xC0, 0xE0, CfgEdgeKind::Branch),
            edge(0xD0, 0xE0, CfgEdgeKind::Branch),
            edge(0xE0, 0xB0, CfgEdgeKind::GuardedBranch),
            edge(0xE0, 0xF0, CfgEdgeKind::GuardedBranch),
            edge(0xE0, 0xB0, CfgEdgeKind::LoopBack),
        ],
        0xA0,
    );
    let error = lift(&looping, &constant_frame(), &BTreeMap::new())
        .expect_err("an in-loop branch must refuse");
    assert!(
        error.to_string().contains("unsupported branching"),
        "{error}"
    );
}

/// A loop body that updates the counter in slot `0x28` (init 0, next 0x55) and
/// also writes a constant through a VM-stack pointer (`Push` materialises the
/// current VSP), a precise external store below the live top. VSP-neutral apart
/// from the shared `-8` branch-target push, so the iteration stays balanced.
fn loop_body_with_store(key: u64, address: u64) -> (Va, VmBasicBlock) {
    use VmHandlerSemantic as S;
    block(
        key,
        Some(address),
        vec![
            ins(S::ConstantPush, Some(0x55), -8),
            ins(S::VirtualRegisterWrite, Some(0x28), 8),
            ins(S::ConstantPush, Some(0xAB), -8),
            ins(S::Push, None, -8),
            ins(S::MemoryStore, None, 16),
            ins(S::ConstantPush, Some(0x900), -8),
        ],
        BlockExit::Branch,
    )
}

#[test]
fn a_single_exit_loop_with_a_body_store_emits_it_each_iteration() {
    let looping = cfg(
        vec![
            loop_entry(0xA0, 0x900),
            branch_block(0xB0, LOOP_HEADER_ADDR),
            loop_body_with_store(0xC0, LOOP_BODY_ADDR),
            returns_constant(0xD0, 0x930, 111),
        ],
        vec![
            edge(0xA0, 0xB0, CfgEdgeKind::Branch),
            edge(0xB0, 0xC0, CfgEdgeKind::GuardedBranch),
            edge(0xB0, 0xD0, CfgEdgeKind::GuardedBranch),
            edge(0xC0, 0xB0, CfgEdgeKind::GuardedBranch),
            edge(0xC0, 0xB0, CfgEdgeKind::LoopBack),
        ],
        0xA0,
    );
    let lifted = lift(&looping, &constant_frame(), &BTreeMap::new())
        .expect("a single-exit loop with a precise body store lifts");
    assert_eq!(lifted.loops, 1, "one natural loop emitted");
    let operations = &lifted.replacement.function.operations;
    // The body's external store is reproduced as a native store.
    let store_at = operations
        .iter()
        .position(|o| matches!(o, Operation::Store { .. }))
        .expect("the per-iteration body store is emitted");
    // It runs inside the loop: after the header label and before the back-edge.
    let first_label = operations
        .iter()
        .position(|o| matches!(o, Operation::Label(_)))
        .expect("the loop has a header label");
    let back_edge = operations
        .iter()
        .position(|o| matches!(o, Operation::Jump(_)))
        .expect("the loop closes with a back-edge jump");
    assert!(
        first_label < store_at && store_at < back_edge,
        "the store is emitted in the loop body each iteration (label {first_label}, store {store_at}, jump {back_edge})"
    );
    assert!(
        mnemonics(&lifted).contains(&Mnemonic::Jmp),
        "a back-edge jump closes the loop"
    );
}

#[test]
fn a_loop_body_store_through_an_opaque_pointer_fails_closed() {
    // The store goes through a pointer loaded from memory (opaque / havocs the
    // tracked model), so it cannot be reproduced as a sound per-iteration effect:
    // the lift must refuse rather than drop or mis-place the store.
    use VmHandlerSemantic as S;
    let body = block(
        0xC0,
        Some(LOOP_BODY_ADDR),
        vec![
            ins(S::ConstantPush, Some(0x55), -8),
            ins(S::VirtualRegisterWrite, Some(0x28), 8),
            ins(S::ConstantPush, Some(0xAB), -8),
            ins(S::ConstantPush, Some(0x4000), -8),
            ins(S::MemoryLoad, None, 0),
            ins(S::MemoryStore, None, 16),
            ins(S::ConstantPush, Some(0x900), -8),
        ],
        BlockExit::Branch,
    );
    let looping = cfg(
        vec![
            loop_entry(0xA0, 0x900),
            branch_block(0xB0, LOOP_HEADER_ADDR),
            body,
            returns_constant(0xD0, 0x930, 111),
        ],
        vec![
            edge(0xA0, 0xB0, CfgEdgeKind::Branch),
            edge(0xB0, 0xC0, CfgEdgeKind::GuardedBranch),
            edge(0xB0, 0xD0, CfgEdgeKind::GuardedBranch),
            edge(0xC0, 0xB0, CfgEdgeKind::GuardedBranch),
            edge(0xC0, 0xB0, CfgEdgeKind::LoopBack),
        ],
        0xA0,
    );
    let error = lift(&looping, &constant_frame(), &BTreeMap::new())
        .expect_err("a loop body store through an opaque pointer must refuse");
    assert!(
        error.to_string().contains("havocs")
            || error.to_string().contains("side effects"),
        "{error}"
    );
}

#[test]
fn a_pure_body_loop_emits_no_store() {
    // Regression: a loop whose body only transforms VM state carries no side
    // effects, so no native store is emitted.
    let lifted = lift(&reducible_loop_cfg(), &constant_frame(), &BTreeMap::new())
        .expect("a pure-body loop lifts");
    assert_eq!(lifted.loops, 1);
    assert_eq!(
        count(&lifted.replacement.function.operations, |o| matches!(
            o,
            Operation::Store { .. }
        )),
        0,
        "a pure loop body emits no store"
    );
}

#[test]
fn an_irreducible_loop_fails_closed() {
    // Two entries into the cycle (entry reaches both A and B), so the back-edge
    // target does not dominate its source: not a natural loop, refused.
    let looping = cfg(
        vec![
            branch_block(0xA0, 0x900),
            branch_block(0xB0, 0x910),
            branch_block(0xC0, 0x920),
            returns_constant(0xD0, 0x930, 111),
        ],
        vec![
            edge(0xA0, 0xB0, CfgEdgeKind::GuardedBranch),
            edge(0xA0, 0xC0, CfgEdgeKind::GuardedBranch),
            edge(0xB0, 0xC0, CfgEdgeKind::Branch),
            edge(0xC0, 0xB0, CfgEdgeKind::Branch),
            edge(0xC0, 0xB0, CfgEdgeKind::LoopBack),
            edge(0xB0, 0xD0, CfgEdgeKind::GuardedBranch),
        ],
        0xA0,
    );
    let error = lift(&looping, &constant_frame(), &BTreeMap::new())
        .expect_err("an irreducible loop must refuse");
    assert!(error.to_string().contains("loop"), "{error}");
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
            rflags: None,
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
    first.steps.push(Step {
        epoch: 1,
        kind: StepKind::Store(EffectStore {
            address: entry(X64NativeRegister::Rcx),
            value: narrow,
            width: BitWidth::W32,
        }),
    });
    let root = Node {
        steps: Vec::new(),
        end: End::Branch {
            condition: entry(X64NativeRegister::Rax),
            arms: vec![(3, first), (5, identity_exit()), (9, identity_exit())],
        },
    };
    let replacement =
        build_tree_replacement(&root, 0, 0, 0, None, None, &tree_config(), CodeAddress(ADDRESS))
            .expect("emit");
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

fn stack_slot(offset: u64) -> Expr {
    Expr::binary(
        BinaryOp::Add,
        Expr::register(BitWidth::W64, "vsp_initial"),
        Expr::constant(BitWidth::W64, offset),
    )
    .expect("slot")
}

/// Prepares a W32 read of `[vsp + offset]` at memory epoch `epoch` under a node
/// whose steps began at `node_base` (ancestor call barrier `node_call_barrier`).
/// `steps` are the current node's already-emitted side effects; a kept-in-place
/// read inserts its load into them.
fn prepare_read_in_node(
    path: &Path,
    epoch: u64,
    offset: u64,
    node_base: u64,
    node_call_barrier: u64,
    steps: &mut Vec<Step>,
) -> Result<Expr, DecodedLiftError> {
    let cfg = conditional_cfg();
    let (frame, exits, calls) = (constant_frame(), exits(), BTreeMap::new());
    let input = DecodedCfgInput {
        cfg: &cfg,
        frame: &frame,
        top_offset: 0,
        exits: &exits,
        native_calls: &calls,
        memory: None,
        limits: SymbolicExecutionLimits::default(),
        target_limits: VmCfgLimits::default(),
        max_expanded_blocks: 64,
    };
    let mut builder = Builder {
        input: &input,
        nested: None,
        nested_entries: 0,
        expanded: 0,
        blocks: BTreeSet::new(),
        exit_paths: 0,
        region_exits: 0,
        retained_bytes: 0,
        branches: 0,
        loops: 0,
        call_sites: 0,
        read_sites: 0,
        uses_root: false,
        assumptions: Vec::new(),
        loop_bodies: BTreeMap::new(),
        loop_phis: 0,
        active_phis: BTreeSet::new(),
        node_base_epoch: node_base,
        node_call_barrier,
        ancestors: Vec::new(),
    };
    let read = Expr::intrinsic(
        BitWidth::W32,
        format!("memory_load_epoch_{epoch}"),
        vec![stack_slot(offset)],
    );
    builder.prepare(Va(1), &read, path.barrier, path, steps)
}

/// Prepares a W64 read of `address` at memory epoch `epoch` under a descendant
/// node whose steps began at `node_base`, with `ancestor` (base epoch
/// `ancestor_base`, call barrier 0) open above it. Returns the prepared
/// expression together with the ancestor's steps after any cross-node capture,
/// so a test can check the load was placed in the ancestor and threaded by slot.
fn prepare_read_with_ancestor(
    path: &Path,
    epoch: u64,
    address: Expr,
    node_base: u64,
    ancestor_base: u64,
    ancestor_steps: Vec<Step>,
    ancestor_calls: BTreeSet<String>,
) -> (Result<Expr, DecodedLiftError>, Vec<Step>) {
    let cfg = conditional_cfg();
    let (frame, exits, calls) = (constant_frame(), exits(), BTreeMap::new());
    let input = DecodedCfgInput {
        cfg: &cfg,
        frame: &frame,
        top_offset: 0,
        exits: &exits,
        native_calls: &calls,
        memory: None,
        limits: SymbolicExecutionLimits::default(),
        target_limits: VmCfgLimits::default(),
        max_expanded_blocks: 64,
    };
    let mut builder = Builder {
        input: &input,
        nested: None,
        nested_entries: 0,
        expanded: 0,
        blocks: BTreeSet::new(),
        exit_paths: 0,
        region_exits: 0,
        retained_bytes: 0,
        branches: 0,
        loops: 0,
        call_sites: 0,
        read_sites: 0,
        uses_root: false,
        assumptions: Vec::new(),
        loop_bodies: BTreeMap::new(),
        loop_phis: 0,
        active_phis: BTreeSet::new(),
        node_base_epoch: node_base,
        node_call_barrier: 0,
        ancestors: vec![OpenNode {
            base_epoch: ancestor_base,
            available_calls: ancestor_calls,
            steps: ancestor_steps,
        }],
    };
    let read = Expr::intrinsic(BitWidth::W64, format!("memory_load_epoch_{epoch}"), vec![address]);
    let mut steps = Vec::new();
    let prepared = builder.prepare(Va(1), &read, path.barrier, path, &mut steps);
    let ancestor = builder.ancestors.pop().expect("ancestor").steps;
    (prepared, ancestor)
}

fn global_slot(offset: u64) -> Expr {
    Expr::binary(
        BinaryOp::Add,
        entry(X64NativeRegister::R13),
        Expr::constant(BitWidth::W64, offset),
    )
    .expect("global")
}

/// A store step at `epoch` over `[vsp + offset]`, used to order a kept-in-place
/// read against.
fn store_step(epoch: u64, offset: u64, width: BitWidth) -> Step {
    Step {
        epoch,
        kind: StepKind::Store(EffectStore {
            address: stack_slot(offset),
            value: Expr::constant(width, 0),
            width,
        }),
    }
}

fn path_with_store(store_offset: u64, store_bytes: u32) -> Path {
    Path {
        barrier: 20,
        stores: vec![PathStore {
            end_epoch: 20,
            address: stack_slot(store_offset),
            bytes: store_bytes,
        }],
        ..Path::default()
    }
}

#[test]
fn a_read_before_a_disjoint_store_is_hoisted_to_a_native_load() {
    // A provably disjoint store does not block re-issuing the read: it stays a
    // direct native load, with nothing captured.
    let mut steps = Vec::new();
    let prepared =
        prepare_read_in_node(&path_with_store(0xc0, 2), 10, 0xb78, 0, 0, &mut steps).expect("disjoint");
    assert!(matches!(prepared.kind, ExprKind::Load { .. }), "{prepared:?}");
    assert!(steps.is_empty(), "no capture needed");
    let mut abutting = Vec::new();
    let prepared =
        prepare_read_in_node(&path_with_store(0xb7c, 8), 10, 0xb78, 0, 0, &mut abutting).expect("abuts");
    assert!(matches!(prepared.kind, ExprKind::Load { .. }), "{prepared:?}");
    assert!(abutting.is_empty());
}

#[test]
fn a_read_blocked_by_an_in_node_store_is_kept_in_place_not_refused() {
    // The store runs in this node (epoch above `node_base`) and may alias the
    // read, so the read cannot be re-issued after it. Rather than refusing, the
    // read is captured at its native-order position and referenced by slot.
    for (store, bytes) in [(0xb78, 4), (0xb76, 4), (0xb7b, 1)] {
        let mut steps = vec![store_step(20, store, BitWidth::W32)];
        let prepared = prepare_read_in_node(&path_with_store(store, bytes), 10, 0xb78, 0, 0, &mut steps)
            .expect("kept in place");
        assert!(
            prepared
                .dependencies()
                .iter()
                .any(|name| name.starts_with(READ_RESULT_PREFIX)),
            "{prepared:?}"
        );
        let load_at = steps
            .iter()
            .position(|step| matches!(step.kind, StepKind::Load(_)))
            .expect("load inserted");
        let store_at = steps
            .iter()
            .position(|step| matches!(step.kind, StepKind::Store(_)))
            .expect("store present");
        assert!(load_at < store_at, "read kept ahead of the store it must not cross");
    }
}

#[test]
fn a_read_blocked_by_an_ancestor_store_refuses() {
    // The store was emitted in an ancestor node (epoch at or below `node_base`)
    // and cannot be reordered against, so a may-alias read fails closed.
    let mut steps = Vec::new();
    let error = prepare_read_in_node(&path_with_store(0xb78, 4), 10, 0xb78, 20, 0, &mut steps)
        .expect_err("ancestor overlap");
    assert!(error.to_string().contains("predates"), "{error}");
    assert!(steps.is_empty(), "nothing captured on a refusal");
    let mut path = path_with_store(0xc0, 2);
    path.stores[0].address = entry(X64NativeRegister::Rcx);
    let mut steps = Vec::new();
    assert!(prepare_read_in_node(&path, 10, 0xb78, 20, 0, &mut steps).is_err());
}

#[test]
fn a_captured_read_sits_after_the_store_it_observes_and_before_the_one_it_must_not() {
    // Two in-node stores to the same cell straddle the read (epoch 15): it must
    // observe the earlier store (epoch 10) and not the later one (epoch 20).
    let path = Path {
        barrier: 20,
        stores: vec![
            PathStore {
                end_epoch: 10,
                address: stack_slot(0xb78),
                bytes: 4,
            },
            PathStore {
                end_epoch: 20,
                address: stack_slot(0xb78),
                bytes: 4,
            },
        ],
        ..Path::default()
    };
    let mut steps = vec![store_step(10, 0xb78, BitWidth::W32), store_step(20, 0xb78, BitWidth::W32)];
    let prepared = prepare_read_in_node(&path, 15, 0xb78, 0, 0, &mut steps).expect("kept in place");
    assert!(
        prepared
            .dependencies()
            .iter()
            .any(|name| name.starts_with(READ_RESULT_PREFIX)),
        "{prepared:?}"
    );
    let epochs: Vec<u64> = steps.iter().map(|step| step.epoch).collect();
    assert_eq!(epochs, vec![10, 15, 20], "load ordered between the two stores");
}

#[test]
fn a_read_that_a_native_call_crossed_refuses_but_a_later_read_does_not() {
    // An ancestor native call at barrier 15 cannot be reordered against: a read
    // from before it fails closed, a read at or after it stays a direct load.
    let mut path = path_with_store(0xc0, 2);
    path.call_barrier = 15;
    let mut steps = Vec::new();
    assert!(prepare_read_in_node(&path, 10, 0xb78, 0, 15, &mut steps).is_err());
    let mut steps = Vec::new();
    assert!(prepare_read_in_node(&path, 15, 0xb78, 0, 15, &mut steps).is_ok());
    // A read at or after the store's epoch observes it and re-issues cleanly.
    let mut steps = Vec::new();
    assert!(prepare_read_in_node(&path_with_store(0xb78, 4), 20, 0xb78, 0, 0, &mut steps).is_ok());
}

#[test]
fn a_read_blocked_in_its_own_node_is_eagerly_captured_in_the_owning_ancestor() {
    // Block B (node_base 100) reads [r13 + const] at epoch 10, which an
    // ancestor store to [vsp + 0xb78] (same base, not provably disjoint from an
    // r13-relative address) has crossed by the time B runs. The read cannot be
    // re-issued in B, so it is captured in the ancestor node that owns epoch 10
    // (base 0), ordered before that store, and B reads the spill slot.
    let path = Path {
        barrier: 20,
        stores: vec![PathStore {
            end_epoch: 20,
            address: stack_slot(0xb78),
            bytes: 4,
        }],
        ..Path::default()
    };
    let ancestor_steps = vec![store_step(20, 0xb78, BitWidth::W64)];
    let (prepared, ancestor) = prepare_read_with_ancestor(
        &path,
        10,
        global_slot(0x200),
        100,
        0,
        ancestor_steps,
        BTreeSet::new(),
    );
    let prepared = prepared.expect("captured across the node boundary");
    assert!(
        prepared
            .dependencies()
            .iter()
            .any(|name| name.starts_with(READ_RESULT_PREFIX)),
        "use site reads the threaded spill slot, not a fresh load: {prepared:?}"
    );
    let epochs: Vec<u64> = ancestor.iter().map(|step| step.epoch).collect();
    assert_eq!(epochs, vec![10, 20], "capture placed before the store it precedes");
    assert!(
        matches!(ancestor[0].kind, StepKind::Load(_)),
        "the ancestor hosts the captured load"
    );
}

#[test]
fn a_cross_node_capture_whose_address_is_unavailable_in_the_ancestor_fails_closed() {
    // The read's address depends on a call result produced only after the owning
    // ancestor (not in its available set), so the capture cannot be placed
    // soundly there: fail closed rather than observe the wrong memory.
    let path = Path {
        barrier: 20,
        stores: vec![PathStore {
            end_epoch: 20,
            address: stack_slot(0xb78),
            bytes: 4,
        }],
        ..Path::default()
    };
    let later_call = Expr::binary(
        BinaryOp::Add,
        Expr::register(BitWidth::W64, call_result_name(0)),
        Expr::constant(BitWidth::W64, 0x10),
    )
    .expect("addr");
    let (prepared, ancestor) = prepare_read_with_ancestor(
        &path,
        10,
        later_call,
        100,
        0,
        vec![store_step(20, 0xb78, BitWidth::W64)],
        BTreeSet::new(), // the ancestor knows of no call result
    );
    assert!(prepared.is_err(), "unavailable address must fail closed");
    assert!(
        !ancestor.iter().any(|step| matches!(step.kind, StepKind::Load(_))),
        "nothing captured on a failed-closed read"
    );
}

#[test]
fn a_cross_node_read_with_no_owning_ancestor_fails_closed() {
    // No open ancestor owns the read's epoch, so there is nowhere sound to host
    // the capture: fail closed (this is the pre-existing ancestor-store refusal).
    let mut steps = Vec::new();
    let error = prepare_read_in_node(&path_with_store(0xb78, 4), 10, 0xb78, 20, 0, &mut steps)
        .expect_err("no ancestor to capture into");
    assert!(error.to_string().contains("predates"), "{error}");
    assert!(steps.is_empty(), "nothing captured on a refusal");
}

const REGION_TARGET: u64 = 0x4000_1234;
const REGION_WORD: u64 = 0xAA;

/// The exit's `ret` slot is the entry frame's second-to-last cell: a computed
/// return into the continuation, with one staged word above it.
fn region_frame(target: NativeFrameSource, staged: NativeFrameSource) -> Vec<NativeFrameSource> {
    let mut cells = constant_frame();
    cells[14] = NativeFrameSource::Rflags;
    cells[15] = target;
    cells[16] = staged;
    cells
}

fn region_cfg() -> VmCfg {
    cfg(vec![returns_constant(0xA0, 0x900, 111)], Vec::new(), 0xA0)
}

#[test]
fn a_proven_computed_return_lifts_as_a_region_transfer_to_the_target() {
    let frame = region_frame(
        NativeFrameSource::Constant {
            value: REGION_TARGET,
        },
        NativeFrameSource::Constant { value: REGION_WORD },
    );
    let lifted = lift(&region_cfg(), &frame, &BTreeMap::new()).expect("region exit lifts");
    assert_eq!(lifted.exit_paths, 1);
    assert!(lifted.assumptions.is_empty(), "{:?}", lifted.assumptions);

    let function = &lifted.replacement.function;
    let operations = &function.operations;
    assert_eq!(
        count(operations, |o| matches!(
            o,
            Operation::ReturnRetaining { bytes: 16 }
        )),
        1
    );
    assert_eq!(
        count(operations, |o| matches!(
            o,
            Operation::ReturnPopping { .. } | Operation::Return
        )),
        0
    );
    for word in [REGION_TARGET, REGION_WORD] {
        assert_eq!(
            count(operations, |o| matches!(
                o,
                Operation::MoveImmediate { value, .. } if *value == word
            )),
            1,
            "staged word {word:#x}"
        );
    }
    assert_eq!(
        count(operations, |o| matches!(o, Operation::Store { .. })),
        2
    );
    assert_eq!(
        count(operations, |o| matches!(o, Operation::ReadFlags { .. })),
        1
    );
    assert_eq!(
        count(operations, |o| matches!(o, Operation::RestoreFlags { .. })),
        1
    );

    let instructions: Vec<_> = Decoder::with_ip(
        64,
        &lifted.replacement.compiled.bytes,
        ADDRESS,
        DecoderOptions::NONE,
    )
    .iter()
    .collect();
    let [.., release, ret] = &instructions[..] else {
        panic!("function too short");
    };
    let frame_total =
        function.frame.preserve.len() as u64 * 8 + u64::from(function.frame.stack_allocation);
    assert!(frame_total >= 16);
    assert_eq!(ret.mnemonic(), Mnemonic::Ret);
    assert_eq!(release.mnemonic(), Mnemonic::Lea);
    assert_eq!(release.op0_register(), iced_x86::Register::RSP);
    assert_eq!(release.memory_displacement64(), frame_total - 16);
}

#[test]
fn an_unproven_computed_return_refuses() {
    let word = NativeFrameSource::Constant { value: REGION_WORD };
    let unknown = || NativeFrameSource::Unknown {
        description: "unproven".to_owned(),
    };
    let cases = [
        // The protector's own return address is not a value the lift can read.
        region_frame(
            NativeFrameSource::ReturnAddress {
                call_site: Rva(0x10),
                continuation: Rva(0x20),
            },
            word.clone(),
        ),
        // A staged continuation word the entry frame never proved.
        region_frame(
            NativeFrameSource::Constant {
                value: REGION_TARGET,
            },
            unknown(),
        ),
        // The ret slot itself is unproven.
        region_frame(unknown(), word),
    ];
    for frame in cases {
        let error = lift(&region_cfg(), &frame, &BTreeMap::new()).expect_err("must refuse");
        assert!(matches!(error, DecodedLiftError::Cfg { .. }), "{error}");
    }
}

#[test]
fn a_native_call_lifts_from_the_outcome_the_cfg_carries() {
    let (mut cfg, calls) = call_frame_cfg();
    let outcome = calls[&Va(0xA0)].clone();
    cfg.blocks
        .get_mut(&Va(0xA0))
        .expect("call block")
        .native_call = Some(outcome);
    let mut layouts = exits();
    cfg.blocks.get_mut(&Va(0xB0)).expect("exit block").vm_exit = layouts.remove(&EXIT_HANDLER);
    let rcx = frame(NativeFrameSource::Register {
        register: X64NativeRegister::Rcx,
    });
    let lifted = lift_with(&cfg, &rcx, &BTreeMap::new(), &BTreeMap::new())
        .expect("carried call frame and exit layout lift");
    assert_eq!(lifted.native_calls, 1);
    assert!(mnemonics(&lifted).contains(&Mnemonic::Call));

    let (bare, _) = call_frame_cfg();
    let error = lift_with(&bare, &rcx, &exits(), &BTreeMap::new())
        .expect_err("a call with no proven frame must refuse");
    assert!(
        error.to_string().contains("no proven call frame"),
        "{error}"
    );
}

const NESTED_STUB: u64 = 0x4000_5678;
const NESTED_CONTINUATION: u64 = 0x4000_9ABC;
const NESTED_START: u64 = 0x950;

/// A resolver that proves exactly one return address.
struct StubResolver {
    bytecode_start: u64,
}

impl NestedEntryResolver for StubResolver {
    fn resolve(&self, return_address: &Expr) -> Option<NestedEntry> {
        let ExprKind::Const { value } = return_address.simplify().kind else {
            return None;
        };
        (value == NESTED_STUB).then(|| {
            let mut cells: Vec<_> = X64NativeRegister::ALL[1..]
                .iter()
                .map(|register| NativeFrameSource::Register {
                    register: *register,
                })
                .collect();
            cells.push(NativeFrameSource::Rflags);
            cells.push(NativeFrameSource::Constant {
                value: NESTED_CONTINUATION,
            });
            NestedEntry {
                bytecode_start: Va(self.bytecode_start),
                frame: cells,
                top_offset: 0,
                continuation: None,
            }
        })
    }
}

/// The outer exit returns into a stub; the nested block exits with its own RAX.
fn nested_cfg() -> VmCfg {
    cfg(
        vec![
            returns_constant(0xA0, 0x900, 111),
            returns_constant(0xB0, NESTED_START, 222),
        ],
        vec![edge(0xA0, 0xB0, CfgEdgeKind::NestedEntry)],
        0xA0,
    )
}

fn stub_frame(target: u64) -> Vec<NativeFrameSource> {
    region_frame(
        NativeFrameSource::Constant { value: target },
        NativeFrameSource::Constant { value: REGION_WORD },
    )
}

fn immediates(lifted: &LiftedCfgFunction, wanted: u64) -> usize {
    count(
        &lifted.replacement.function.operations,
        |o| matches!(o, Operation::MoveImmediate { value, .. } if *value == wanted),
    )
}

fn lift_nested(
    cfg: &VmCfg,
    resolver: &dyn NestedEntryResolver,
    frame: &[NativeFrameSource],
) -> Result<LiftedCfgFunction, DecodedLiftError> {
    let exits = exits();
    let calls = BTreeMap::new();
    let input = DecodedCfgInput {
        cfg,
        frame,
        top_offset: 0,
        exits: &exits,
        native_calls: &calls,
        memory: None,
        limits: SymbolicExecutionLimits::default(),
        target_limits: VmCfgLimits::default(),
        max_expanded_blocks: 64,
    };
    lift_decoded_cfg_with_nested(&input, resolver, CodeAddress(ADDRESS))
}

#[test]
fn a_proven_nested_entry_is_lowered_inline_instead_of_returning_into_its_stub() {
    let resolver = StubResolver {
        bytecode_start: NESTED_START,
    };
    let lifted =
        lift_nested(&nested_cfg(), &resolver, &stub_frame(NESTED_STUB)).expect("nested lifts");
    assert_eq!(lifted.nested_entries, 1);
    assert_eq!(lifted.exit_paths, 1);
    assert_eq!(lifted.blocks, BTreeSet::from([Va(0xA0), Va(0xB0)]));
    // The nested exit's own register and continuation are emitted; the outer
    // exit's register and the stub it would have returned into are not.
    assert_eq!(immediates(&lifted, 222), 1);
    assert_eq!(immediates(&lifted, NESTED_CONTINUATION), 1);
    assert_eq!(immediates(&lifted, REGION_WORD), 1);
    assert_eq!(immediates(&lifted, 111), 0);
    assert_eq!(immediates(&lifted, NESTED_STUB), 0);
    assert_eq!(
        count(&lifted.replacement.function.operations, |o| matches!(
            o,
            Operation::ReturnRetaining { bytes: 16 }
        )),
        1
    );
}

#[test]
fn an_unproven_nested_entry_keeps_the_ordinary_region_exit() {
    let resolver = StubResolver {
        bytecode_start: NESTED_START,
    };
    let lifted = lift_nested(&nested_cfg(), &resolver, &stub_frame(0x4000_0001))
        .expect("an unresolved entry lowers as a plain region exit");
    assert_eq!(lifted.nested_entries, 0);
    assert_eq!(lifted.blocks, BTreeSet::from([Va(0xA0)]));
    assert_eq!(immediates(&lifted, 111), 1);

    let plain = lift(&nested_cfg(), &stub_frame(NESTED_STUB), &BTreeMap::new())
        .expect("without a resolver the exit is a region transfer into the stub");
    assert_eq!(plain.nested_entries, 0);
    assert_eq!(immediates(&plain, NESTED_STUB), 1);
}

#[test]
fn a_resolved_entry_that_is_not_the_cfgs_nested_block_refuses() {
    let resolver = StubResolver {
        bytecode_start: NESTED_START + 8,
    };
    let error = lift_nested(&nested_cfg(), &resolver, &stub_frame(NESTED_STUB))
        .expect_err("mismatched entry must refuse");
    assert!(
        error.to_string().contains("not the CFG's nested block"),
        "{error}"
    );
}
