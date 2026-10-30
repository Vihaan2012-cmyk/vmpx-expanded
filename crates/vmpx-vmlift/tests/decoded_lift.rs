use vmpx_pe::{Rva, Va};
use vmpx_recompile::{CodeAddress, Operation};
use vmpx_vm::{
    BitWidth, DecodedVmInstruction, EffectKnowledge, Expr, NativeFrameSource, RecoveryStatus,
    RelativeDispatchState, SymbolicExecutionLimits, VmExitReport, VmExitRestoreSlot,
    VmHandlerSemantic, X64NativeRegister,
};
use vmpx_vmlift::{
    DecodedLiftError, DecodedLinearInput, LiftAssumption, LiftedLinearFunction, SemanticLowering,
    lift_decoded_linear, semantic_lowering,
};

fn instruction(
    semantic: VmHandlerSemantic,
    operand: Option<u64>,
    vsp_delta: Option<i32>,
) -> DecodedVmInstruction {
    let state = RelativeDispatchState {
        vip: Va(0x1000),
        rolling_key: 0,
        handler: Va(0x2000),
    };
    DecodedVmInstruction {
        state_before: state,
        encoded_bytes: Vec::new(),
        semantic,
        semantic_metadata: None,
        input_width: Some(BitWidth::W64),
        operation_width: Some(BitWidth::W64),
        resize: None,
        operand,
        vsp_delta,
        flags_effect: EffectKnowledge::Unknown,
        confidence: 100,
        state_after: state,
    }
}

fn exit_report() -> VmExitReport {
    VmExitReport {
        schema_version: 2,
        entry: Rva(0x1000),
        status: RecoveryStatus::Recovered,
        stack_switch: None,
        restored_registers: Vec::new(),
        restore_slots: X64NativeRegister::ALL
            .iter()
            .enumerate()
            .map(|(index, register)| VmExitRestoreSlot {
                register: *register,
                stack_offset: u32::try_from(index * 8).expect("small offset"),
                pop: Rva(0),
            })
            .collect(),
        flags_restore: None,
        flags_stack_offset: Some(120),
        native_return: None,
        return_stack_offset: Some(128),
        frame_size: Some(136),
        confidence: 100,
        evidence: Vec::new(),
        diagnostics: Vec::new(),
    }
}

fn register_cell(register: X64NativeRegister) -> NativeFrameSource {
    NativeFrameSource::Register { register }
}

fn lift(
    instructions: &[DecodedVmInstruction],
    frame: &[NativeFrameSource],
) -> Result<LiftedLinearFunction, DecodedLiftError> {
    let exit = exit_report();
    lift_decoded_linear(
        &DecodedLinearInput {
            instructions,
            frame,
            top_offset: 0,
            exit: &exit,
            memory: None,
            limits: SymbolicExecutionLimits::default(),
        },
        CodeAddress(0x1_4000_0000),
    )
}

#[test]
fn a_stack_unwinding_epilogue_lifts_to_register_moves_and_a_popping_return() {
    // `push vsp; push 32; add; discard flags; pop vsp` raises the VM stack by 32
    // bytes, so the exit frame is read 32 bytes up and the return slot lands
    // exactly at the entry stack pointer.
    let mut frame = vec![NativeFrameSource::Constant { value: 0 }; 4];
    frame.extend(X64NativeRegister::ALL.iter().map(|r| register_cell(*r)));
    frame.push(NativeFrameSource::Rflags);
    frame.swap(4, 5);
    let stream = [
        instruction(VmHandlerSemantic::Push, None, Some(-8)),
        instruction(VmHandlerSemantic::ConstantPush, Some(32), Some(-8)),
        instruction(VmHandlerSemantic::Add, None, Some(0)),
        instruction(VmHandlerSemantic::VirtualRegisterWrite, Some(1000), Some(8)),
        instruction(VmHandlerSemantic::Pop, None, None),
    ];
    let lifted = lift(&stream, &frame).expect("liftable stream lifts");
    assert_eq!(lifted.exit_state.return_pop, 0);
    assert_eq!(
        lifted.exit_state.registers[&X64NativeRegister::Rax],
        Expr::register(BitWidth::W64, "native_rbx_initial")
    );
    assert_eq!(
        lifted.exit_state.registers[&X64NativeRegister::Rbx],
        Expr::register(BitWidth::W64, "native_rax_initial")
    );
    assert_eq!(
        lifted.replacement.function.operations.last(),
        Some(&Operation::ReturnPopping { bytes: 0 })
    );
    assert_eq!(
        lifted.assumptions,
        vec![LiftAssumption::FlagsVolatileAtReturn]
    );
    assert_eq!(lifted.semantic_counts["Push"], 1);
}

#[test]
fn a_gs_load_lifts_as_a_passthrough_environment_boundary() {
    let mut frame: Vec<_> = X64NativeRegister::ALL[1..]
        .iter()
        .map(|r| register_cell(*r))
        .collect();
    frame.push(NativeFrameSource::Rflags);
    let stream = [
        instruction(VmHandlerSemantic::ConstantPush, Some(0x60), Some(-8)),
        instruction(VmHandlerSemantic::GsRelativeLoad, None, Some(0)),
    ];
    let lifted = lift(&stream, &frame).expect("gs load lifts by passthrough");
    assert!(
        lifted
            .replacement
            .function
            .operations
            .iter()
            .any(|operation| matches!(operation, Operation::GsLoad { .. }))
    );
    assert!(
        lifted
            .replacement
            .environment_boundaries
            .contains("x86_gs_load_1")
    );
}

#[test]
fn an_unliftable_semantic_stops_the_lift_at_its_index() {
    let frame = [NativeFrameSource::Rflags];
    for semantic in [
        VmHandlerSemantic::Mul,
        VmHandlerSemantic::TableChecksum,
        VmHandlerSemantic::ReadTimestampCounter,
        VmHandlerSemantic::ConditionalBranch,
        VmHandlerSemantic::Xor,
    ] {
        let stream = [
            instruction(VmHandlerSemantic::ConstantPush, Some(1), Some(-8)),
            instruction(semantic, None, Some(0)),
        ];
        match lift(&stream, &frame) {
            Err(DecodedLiftError::Unliftable {
                index,
                semantic: found,
                ..
            }) => assert_eq!((index, found), (1, semantic)),
            other => panic!("{semantic:?} must refuse, got {other:?}"),
        }
    }
}

#[test]
fn an_exit_into_the_protector_continuation_is_refused() {
    // With no stack adjustment the return slot is inside the entry frame.
    let mut frame: Vec<_> = X64NativeRegister::ALL
        .iter()
        .map(|r| register_cell(*r))
        .collect();
    frame.push(NativeFrameSource::Rflags);
    frame.push(NativeFrameSource::ReturnAddress {
        call_site: Rva(0x10),
        continuation: Rva(0x15),
    });
    assert!(matches!(lift(&[], &frame), Err(DecodedLiftError::NotCalleeReturn(_))));
}

#[test]
fn the_five_former_gaps_are_passthroughs() {
    for semantic in [
        VmHandlerSemantic::Cpuid,
        VmHandlerSemantic::GsRelativeLoad,
        VmHandlerSemantic::ShiftLeftDouble,
        VmHandlerSemantic::ShiftRightDouble,
        VmHandlerSemantic::PopFlags,
    ] {
        assert!(matches!(
            semantic_lowering(semantic),
            SemanticLowering::Passthrough { .. }
        ));
    }
}
