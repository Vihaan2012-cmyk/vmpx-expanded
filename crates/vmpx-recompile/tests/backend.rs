use iced_x86::{Code, Decoder, DecoderOptions, FlowControl, Mnemonic, Register};
use vmpx_recompile::{
    Address, BinaryOp, BinarySource, CallTarget, CodeAddress, CompileError, Condition, Function,
    Label, NativeAddress, Operation, PhysicalRegister, RegisterAssignment, ResizeOp, ShiftCount,
    ShiftOp, SpillSlot, StackFrame, UnaryOp, VirtualRegister, Width, compile, validate_function,
};

fn vr(id: u32) -> VirtualRegister {
    VirtualRegister::qword(id)
}

fn assigned(register: VirtualRegister, physical: PhysicalRegister) -> RegisterAssignment {
    RegisterAssignment::new(register, physical)
}

fn decode_all(bytes: &[u8], address: u64) -> Vec<iced_x86::Instruction> {
    let mut decoder = Decoder::with_ip(64, bytes, address, DecoderOptions::NONE);
    let mut instructions = Vec::new();
    while decoder.can_decode() {
        let instruction = decoder.decode();
        assert_ne!(instruction.code(), Code::INVALID);
        instructions.push(instruction);
    }
    instructions
}

#[test]
fn emits_arithmetic_memory_and_return() {
    let base = vr(0);
    let value = vr(1);
    let function = Function {
        assignments: vec![
            assigned(base, PhysicalRegister::Rcx),
            assigned(value, PhysicalRegister::Rax),
        ],
        frame: StackFrame::default(),
        operations: vec![
            Operation::Load {
                destination: value,
                address: Address::new(base, 24),
            },
            Operation::Binary {
                operation: BinaryOp::Add,
                destination: value,
                source: BinarySource::Immediate(7),
            },
            Operation::Store {
                address: Address::new(base, 32),
                source: value,
            },
            Operation::Return,
        ],
    };
    let address = 0x0000_0001_4000_1000;
    let compiled = compile(&function, CodeAddress(address)).expect("compile qword function");
    let instructions = decode_all(&compiled.bytes, address);

    assert_eq!(instructions.len(), 4);
    assert_eq!(instructions[0].mnemonic(), Mnemonic::Mov);
    assert_eq!(instructions[0].memory_base(), Register::RCX);
    assert_eq!(instructions[0].memory_displacement64(), 24);
    assert_eq!(instructions[1].mnemonic(), Mnemonic::Add);
    assert_eq!(instructions[2].memory_displacement64(), 32);
    assert_eq!(instructions[3].mnemonic(), Mnemonic::Ret);
}

#[test]
fn resolves_forward_and_backward_near_branches() {
    let counter = vr(0);
    let loop_label = Label(1);
    let exit_label = Label(2);
    let function = Function {
        assignments: vec![assigned(counter, PhysicalRegister::Rax)],
        frame: StackFrame::default(),
        operations: vec![
            Operation::MoveImmediate {
                destination: counter,
                value: 3,
            },
            Operation::Label(loop_label),
            Operation::Binary {
                operation: BinaryOp::Subtract,
                destination: counter,
                source: BinarySource::Immediate(1),
            },
            Operation::Compare {
                left: counter,
                right: BinarySource::Immediate(0),
            },
            Operation::BranchIf {
                condition: Condition::Equal,
                target: exit_label,
            },
            Operation::Jump(loop_label),
            Operation::Label(exit_label),
            Operation::Return,
        ],
    };
    let address = 0x0000_0001_8000_0000;
    let compiled = compile(&function, CodeAddress(address)).expect("compile branches");
    let instructions = decode_all(&compiled.bytes, address);
    let branches: Vec<_> = instructions
        .iter()
        .filter(|instruction| {
            matches!(
                instruction.flow_control(),
                FlowControl::ConditionalBranch | FlowControl::UnconditionalBranch
            )
        })
        .collect();

    assert_eq!(branches.len(), 2);
    assert_eq!(
        branches[0].near_branch_target(),
        address + u64::from(compiled.label_offsets[&exit_label])
    );
    assert_eq!(
        branches[1].near_branch_target(),
        address + u64::from(compiled.label_offsets[&loop_label])
    );
    assert!(branches[0].near_branch_target() > branches[0].ip());
    assert!(branches[1].near_branch_target() < branches[1].ip());
}

#[test]
fn emits_balanced_nonvolatile_frame() {
    let value = vr(0);
    let function = Function {
        assignments: vec![assigned(value, PhysicalRegister::Rbx)],
        frame: StackFrame {
            stack_allocation: 32,
            preserve: vec![PhysicalRegister::Rbx],
            call_ready: true,
            preinitialized_slots: Vec::new(),
        },
        operations: vec![
            Operation::MoveImmediate {
                destination: value,
                value: 0x1122_3344_5566_7788,
            },
            Operation::Return,
        ],
    };
    let compiled = compile(&function, CodeAddress(0x1000)).expect("compile ABI frame");
    let instructions = decode_all(&compiled.bytes, 0x1000);
    let mnemonics: Vec<_> = instructions
        .iter()
        .map(iced_x86::Instruction::mnemonic)
        .collect();

    assert_eq!(
        mnemonics,
        [
            Mnemonic::Push,
            Mnemonic::Sub,
            Mnemonic::Mov,
            Mnemonic::Add,
            Mnemonic::Pop,
            Mnemonic::Ret
        ]
    );
    assert_eq!(instructions[0].op0_register(), Register::RBX);
    assert_eq!(instructions[4].op0_register(), Register::RBX);
}

#[test]
fn rejects_unpreserved_nonvolatile_and_bad_alignment() {
    let value = vr(0);
    let mut function = Function {
        assignments: vec![assigned(value, PhysicalRegister::R12)],
        frame: StackFrame::default(),
        operations: vec![
            Operation::MoveImmediate {
                destination: value,
                value: 1,
            },
            Operation::Return,
        ],
    };
    assert_eq!(
        validate_function(&function),
        Err(CompileError::UnpreservedNonvolatile {
            register: PhysicalRegister::R12
        })
    );

    function.frame = StackFrame {
        stack_allocation: 40,
        preserve: vec![PhysicalRegister::R12],
        call_ready: true,
        preinitialized_slots: Vec::new(),
    };
    assert_eq!(
        validate_function(&function),
        Err(CompileError::MisalignedCallFrame { actual: 8 })
    );
}

#[test]
fn permits_read_only_nonvolatile_inputs_but_preserves_partial_writes() {
    let base = vr(0);
    let destination = vr(1);
    let read_only = Function {
        assignments: vec![
            assigned(base, PhysicalRegister::Rbx),
            assigned(destination, PhysicalRegister::Rax),
        ],
        frame: StackFrame::default(),
        operations: vec![
            Operation::Load {
                destination,
                address: Address::new(base, 0),
            },
            Operation::Return,
        ],
    };
    let compiled = compile(&read_only, CodeAddress(0)).expect("read-only RBX is ABI-safe");
    let instructions = decode_all(&compiled.bytes, 0);
    assert_eq!(instructions[0].memory_base(), Register::RBX);
    assert_eq!(
        instructions.last().map(iced_x86::Instruction::mnemonic),
        Some(Mnemonic::Ret)
    );

    let byte_view = VirtualRegister::new(0, Width::W8);
    let partial_write = Function {
        assignments: vec![assigned(byte_view, PhysicalRegister::Rbx)],
        frame: StackFrame::default(),
        operations: vec![
            Operation::MoveImmediate {
                destination: byte_view,
                value: 1,
            },
            Operation::Return,
        ],
    };
    assert_eq!(
        validate_function(&partial_write),
        Err(CompileError::UnpreservedNonvolatile {
            register: PhysicalRegister::Rbx,
        })
    );
}

#[test]
fn register_widths_are_views_of_one_allocated_family() {
    let qword_assignment = vr(7);
    let dword_view = VirtualRegister::new(7, Width::W32);
    let function = Function {
        assignments: vec![assigned(qword_assignment, PhysicalRegister::Rax)],
        frame: StackFrame::default(),
        operations: vec![
            Operation::MoveImmediate {
                destination: dword_view,
                value: u64::MAX,
            },
            Operation::Return,
        ],
    };
    let compiled = compile(&function, CodeAddress(0)).expect("compile family width view");
    assert_eq!(
        decode_all(&compiled.bytes, 0)[0].op0_register(),
        Register::EAX
    );

    let inconsistent = Function {
        assignments: vec![
            assigned(qword_assignment, PhysicalRegister::Rax),
            assigned(dword_view, PhysicalRegister::Rdx),
        ],
        frame: StackFrame::default(),
        operations: vec![Operation::Return],
    };
    assert_eq!(
        validate_function(&inconsistent),
        Err(CompileError::RegisterFamilyAssignmentMismatch {
            id: 7,
            expected: PhysicalRegister::Rax,
            actual: PhysicalRegister::Rdx,
        })
    );

    let explicit_aliases = Function {
        assignments: vec![
            assigned(qword_assignment, PhysicalRegister::Rax),
            assigned(dword_view, PhysicalRegister::Rax),
        ],
        frame: StackFrame::default(),
        operations: vec![Operation::Return],
    };
    assert_eq!(validate_function(&explicit_aliases), Ok(()));

    let distinct_family = vr(8);
    let collision = Function {
        assignments: vec![
            assigned(qword_assignment, PhysicalRegister::Rax),
            assigned(distinct_family, PhysicalRegister::Rax),
        ],
        frame: StackFrame::default(),
        operations: vec![Operation::Return],
    };
    assert_eq!(
        validate_function(&collision),
        Err(CompileError::DuplicatePhysicalAssignment {
            register: PhysicalRegister::Rax,
        })
    );
}

#[test]
fn emits_width_specific_spill_accesses() {
    for (width, expected_source, expected_destination, byte_count) in [
        (Width::W8, Register::AL, Register::DL, 1),
        (Width::W16, Register::AX, Register::DX, 2),
        (Width::W32, Register::EAX, Register::EDX, 4),
        (Width::W64, Register::RAX, Register::RDX, 8),
    ] {
        let source = VirtualRegister::new(0, width);
        let destination = VirtualRegister::new(1, width);
        let function = Function {
            assignments: vec![
                assigned(source, PhysicalRegister::Rax),
                assigned(destination, PhysicalRegister::Rdx),
            ],
            frame: StackFrame {
                stack_allocation: 8,
                preserve: vec![],
                call_ready: false,
                preinitialized_slots: Vec::new(),
            },
            operations: vec![
                Operation::SpillStore {
                    slot: SpillSlot(0),
                    source,
                },
                Operation::SpillLoad {
                    destination,
                    slot: SpillSlot(0),
                },
                Operation::Return,
            ],
        };
        let instructions = decode_all(
            &compile(&function, CodeAddress(0))
                .expect("compile width-specific spill")
                .bytes,
            0,
        );
        assert_eq!(instructions[1].memory_base(), Register::RSP);
        assert_eq!(instructions[1].op1_register(), expected_source);
        assert_eq!(instructions[1].memory_size().size(), byte_count);
        assert_eq!(instructions[2].memory_base(), Register::RSP);
        assert_eq!(instructions[2].op0_register(), expected_destination);
        assert_eq!(instructions[2].memory_size().size(), byte_count);
    }
}

#[test]
fn call_ready_spills_follow_shadow_space_and_require_frame_capacity() {
    let source = vr(0);
    let destination = vr(1);
    let function = Function {
        assignments: vec![
            assigned(source, PhysicalRegister::Rax),
            assigned(destination, PhysicalRegister::Rdx),
        ],
        frame: StackFrame {
            stack_allocation: 40,
            preserve: vec![],
            call_ready: true,
            preinitialized_slots: Vec::new(),
        },
        operations: vec![
            Operation::SpillStore {
                slot: SpillSlot(0),
                source,
            },
            Operation::SpillLoad {
                destination,
                slot: SpillSlot(0),
            },
            Operation::Return,
        ],
    };
    let instructions = decode_all(
        &compile(&function, CodeAddress(0))
            .expect("compile shadow-space-safe spill")
            .bytes,
        0,
    );
    assert_eq!(instructions[1].memory_displacement64(), 32);
    assert_eq!(instructions[2].memory_displacement64(), 32);

    let too_small = Function {
        frame: StackFrame {
            stack_allocation: 32,
            preserve: vec![],
            call_ready: true,
            preinitialized_slots: Vec::new(),
        },
        ..function
    };
    assert_eq!(
        validate_function(&too_small),
        Err(CompileError::SpillOutsideFrame {
            slot: SpillSlot(0),
            required_end: 40,
            stack_allocation: 32,
        })
    );
}

fn call_ready_frame() -> StackFrame {
    StackFrame {
        stack_allocation: 40,
        preserve: vec![],
        call_ready: true,
        preinitialized_slots: Vec::new(),
    }
}

#[test]
fn emits_placement_correct_direct_iat_and_register_calls() {
    let result = vr(0);
    let code_address = 0x1_4000_1000;
    let direct_target = NativeAddress(0x1_4000_2345);
    let direct = Function {
        assignments: vec![assigned(result, PhysicalRegister::Rax)],
        frame: call_ready_frame(),
        operations: vec![
            Operation::Call {
                target: CallTarget::Direct(direct_target),
                arguments: vec![],
                result: Some(result),
            },
            Operation::SpillStore {
                slot: SpillSlot(0),
                source: result,
            },
            Operation::Return,
        ],
    };
    let direct_instructions = decode_all(
        &compile(&direct, CodeAddress(code_address))
            .expect("compile direct call")
            .bytes,
        code_address,
    );
    let direct_call = direct_instructions
        .iter()
        .find(|instruction| instruction.mnemonic() == Mnemonic::Call)
        .expect("direct call instruction");
    assert_eq!(direct_call.near_branch_target(), direct_target.0);

    let iat_slot = NativeAddress(0x1_4000_3458);
    let iat = Function {
        operations: vec![
            Operation::Call {
                target: CallTarget::ImportAddressTable(iat_slot),
                arguments: vec![],
                result: Some(result),
            },
            Operation::Return,
        ],
        ..direct.clone()
    };
    let iat_instructions = decode_all(
        &compile(&iat, CodeAddress(code_address))
            .expect("compile IAT call")
            .bytes,
        code_address,
    );
    let iat_call = iat_instructions
        .iter()
        .find(|instruction| instruction.mnemonic() == Mnemonic::Call)
        .expect("IAT call instruction");
    assert!(iat_call.is_ip_rel_memory_operand());
    assert_eq!(iat_call.ip_rel_memory_address(), iat_slot.0);

    let target = vr(1);
    let register = Function {
        assignments: vec![
            assigned(result, PhysicalRegister::Rax),
            assigned(target, PhysicalRegister::R11),
        ],
        frame: call_ready_frame(),
        operations: vec![
            Operation::Call {
                target: CallTarget::Register(target),
                arguments: vec![],
                result: Some(result),
            },
            Operation::Return,
        ],
    };
    let register_instructions = decode_all(
        &compile(&register, CodeAddress(code_address))
            .expect("compile register call")
            .bytes,
        code_address,
    );
    let register_call = register_instructions
        .iter()
        .find(|instruction| instruction.mnemonic() == Mnemonic::Call)
        .expect("register call instruction");
    assert_eq!(register_call.op0_register(), Register::R11);
}

#[test]
fn validates_win64_call_frame_argument_and_result_contracts() {
    let argument = vr(0);
    let result = vr(1);
    let target = NativeAddress(0x1000);
    let mut function = Function {
        assignments: vec![
            assigned(argument, PhysicalRegister::Rdx),
            assigned(result, PhysicalRegister::Rax),
        ],
        frame: call_ready_frame(),
        operations: vec![
            Operation::Call {
                target: CallTarget::Direct(target),
                arguments: vec![argument],
                result: Some(result),
            },
            Operation::Return,
        ],
    };
    assert_eq!(
        validate_function(&function),
        Err(CompileError::CallArgumentRegister {
            ordinal: 0,
            expected: PhysicalRegister::Rcx,
            actual: PhysicalRegister::Rdx,
        })
    );

    function.operations[0] = Operation::Call {
        target: CallTarget::Direct(target),
        arguments: vec![],
        result: Some(argument),
    };
    assert_eq!(
        validate_function(&function),
        Err(CompileError::CallResultRegister {
            actual: PhysicalRegister::Rdx,
        })
    );

    function.operations[0] = Operation::Call {
        target: CallTarget::Direct(target),
        arguments: vec![],
        result: Some(result),
    };
    function.frame = StackFrame::default();
    assert_eq!(
        validate_function(&function),
        Err(CompileError::CallFrameNotReady { operation_index: 0 })
    );
}

#[test]
fn rejects_volatile_values_live_across_calls_on_every_cfg_path() {
    let volatile = vr(0);
    let call_path = Label(10);
    let merge = Label(11);
    let function = Function {
        assignments: vec![assigned(volatile, PhysicalRegister::Rcx)],
        frame: call_ready_frame(),
        operations: vec![
            Operation::BranchIf {
                condition: Condition::Equal,
                target: call_path,
            },
            Operation::Jump(merge),
            Operation::Label(call_path),
            Operation::Call {
                target: CallTarget::Direct(NativeAddress(0x1000)),
                arguments: vec![],
                result: None,
            },
            Operation::Label(merge),
            Operation::SpillStore {
                slot: SpillSlot(0),
                source: volatile,
            },
            Operation::Return,
        ],
    };
    assert_eq!(
        validate_function(&function),
        Err(CompileError::CallClobberedRegisterRead {
            operation_index: 5,
            register: volatile,
            physical: PhysicalRegister::Rcx,
        })
    );

    let restored = Function {
        operations: vec![
            Operation::Call {
                target: CallTarget::Direct(NativeAddress(0x1000)),
                arguments: vec![],
                result: None,
            },
            Operation::MoveImmediate {
                destination: volatile,
                value: 7,
            },
            Operation::SpillStore {
                slot: SpillSlot(0),
                source: volatile,
            },
            Operation::Return,
        ],
        ..function
    };
    assert_eq!(validate_function(&restored), Ok(()));
}

#[test]
fn requires_explicit_flag_definition_after_a_native_call() {
    let value = vr(0);
    let target = Label(20);
    let mut function = Function {
        assignments: vec![assigned(value, PhysicalRegister::Rbx)],
        frame: call_ready_frame(),
        operations: vec![
            Operation::Call {
                target: CallTarget::Direct(NativeAddress(0x1000)),
                arguments: vec![],
                result: None,
            },
            Operation::BranchIf {
                condition: Condition::Equal,
                target,
            },
            Operation::Return,
            Operation::Label(target),
            Operation::Return,
        ],
    };
    assert_eq!(
        validate_function(&function),
        Err(CompileError::CallClobberedFlagsRead { operation_index: 1 })
    );

    function.operations.insert(
        1,
        Operation::Compare {
            left: value,
            right: BinarySource::Immediate(0),
        },
    );
    assert_eq!(validate_function(&function), Ok(()));
}

#[test]
fn rejects_relative_calls_outside_the_x64_rel32_window() {
    let function = Function {
        assignments: vec![],
        frame: call_ready_frame(),
        operations: vec![
            Operation::Call {
                target: CallTarget::ImportAddressTable(NativeAddress(u64::MAX)),
                arguments: vec![],
                result: None,
            },
            Operation::Return,
        ],
    };
    assert_eq!(
        compile(&function, CodeAddress(0)),
        Err(CompileError::CallOutOfRange {
            operation_index: 0,
            target: NativeAddress(u64::MAX),
        })
    );
}

#[test]
fn rejects_uninitialized_and_partially_initialized_spill_loads() {
    let qword = vr(0);
    let uninitialized = Function {
        assignments: vec![assigned(qword, PhysicalRegister::Rax)],
        frame: StackFrame {
            stack_allocation: 8,
            preserve: vec![],
            call_ready: false,
            preinitialized_slots: Vec::new(),
        },
        operations: vec![
            Operation::SpillLoad {
                destination: qword,
                slot: SpillSlot(0),
            },
            Operation::Return,
        ],
    };
    assert_eq!(
        validate_function(&uninitialized),
        Err(CompileError::UninitializedSpill {
            operation_index: 0,
            slot: SpillSlot(0),
            initialized_bytes: 0,
            required_bytes: 8,
        })
    );

    let byte = VirtualRegister::new(1, Width::W8);
    let partial = Function {
        assignments: vec![
            assigned(qword, PhysicalRegister::Rax),
            assigned(byte, PhysicalRegister::Rdx),
        ],
        frame: StackFrame {
            stack_allocation: 8,
            preserve: vec![],
            call_ready: false,
            preinitialized_slots: Vec::new(),
        },
        operations: vec![
            Operation::SpillStore {
                slot: SpillSlot(0),
                source: byte,
            },
            Operation::SpillLoad {
                destination: qword,
                slot: SpillSlot(0),
            },
            Operation::Return,
        ],
    };
    assert_eq!(
        validate_function(&partial),
        Err(CompileError::UninitializedSpill {
            operation_index: 1,
            slot: SpillSlot(0),
            initialized_bytes: 1,
            required_bytes: 8,
        })
    );

    let wide_then_narrow = Function {
        assignments: vec![
            assigned(qword, PhysicalRegister::Rax),
            assigned(byte, PhysicalRegister::Rdx),
        ],
        frame: StackFrame {
            stack_allocation: 8,
            preserve: vec![],
            call_ready: false,
            preinitialized_slots: Vec::new(),
        },
        operations: vec![
            Operation::SpillStore {
                slot: SpillSlot(0),
                source: qword,
            },
            Operation::SpillStore {
                slot: SpillSlot(0),
                source: byte,
            },
            Operation::SpillLoad {
                destination: qword,
                slot: SpillSlot(0),
            },
            Operation::Return,
        ],
    };
    assert_eq!(
        validate_function(&wide_then_narrow),
        Err(CompileError::UninitializedSpill {
            operation_index: 2,
            slot: SpillSlot(0),
            initialized_bytes: 1,
            required_bytes: 8,
        })
    );
}

#[test]
fn spill_initialization_meet_uses_the_narrowest_cfg_predecessor() {
    let byte = VirtualRegister::new(0, Width::W8);
    let qword = vr(1);
    let destination = vr(2);
    let wide_path = Label(1);
    let merge = Label(2);
    let function = Function {
        assignments: vec![
            assigned(byte, PhysicalRegister::Rax),
            assigned(qword, PhysicalRegister::Rdx),
            assigned(destination, PhysicalRegister::R10),
        ],
        frame: StackFrame {
            stack_allocation: 8,
            preserve: vec![],
            call_ready: false,
            preinitialized_slots: Vec::new(),
        },
        operations: vec![
            Operation::BranchIf {
                condition: Condition::Equal,
                target: wide_path,
            },
            Operation::SpillStore {
                slot: SpillSlot(0),
                source: byte,
            },
            Operation::Jump(merge),
            Operation::Label(wide_path),
            Operation::SpillStore {
                slot: SpillSlot(0),
                source: qword,
            },
            Operation::Label(merge),
            Operation::SpillLoad {
                destination,
                slot: SpillSlot(0),
            },
            Operation::Return,
        ],
    };
    assert_eq!(
        validate_function(&function),
        Err(CompileError::UninitializedSpill {
            operation_index: 6,
            slot: SpillSlot(0),
            initialized_bytes: 1,
            required_bytes: 8,
        })
    );
}

#[test]
fn rejects_malformed_cfg_and_missing_registers() {
    let missing = vr(7);
    let undefined = Label(99);
    let function = Function {
        assignments: vec![],
        frame: StackFrame::default(),
        operations: vec![
            Operation::MoveImmediate {
                destination: missing,
                value: 1,
            },
            Operation::Return,
        ],
    };
    assert_eq!(
        validate_function(&function),
        Err(CompileError::UnassignedRegister { register: missing })
    );

    let function = Function {
        assignments: vec![],
        frame: StackFrame::default(),
        operations: vec![Operation::Jump(undefined)],
    };
    assert_eq!(
        validate_function(&function),
        Err(CompileError::UndefinedLabel { label: undefined })
    );
}

#[test]
fn emits_width_specific_immediates_without_widening() {
    let cases = [
        (Width::W8, Register::AL),
        (Width::W16, Register::AX),
        (Width::W32, Register::EAX),
        (Width::W64, Register::RAX),
    ];
    for (width, expected_register) in cases {
        let value = VirtualRegister::new(0, width);
        let function = Function {
            assignments: vec![assigned(value, PhysicalRegister::Rax)],
            frame: StackFrame::default(),
            operations: vec![
                Operation::MoveImmediate {
                    destination: value,
                    value: u64::MAX,
                },
                Operation::Return,
            ],
        };
        let compiled = compile(&function, CodeAddress(0)).expect("compile width-specific move");
        let instructions = decode_all(&compiled.bytes, 0);
        assert_eq!(instructions[0].mnemonic(), Mnemonic::Mov);
        assert_eq!(instructions[0].op0_register(), expected_register);
    }
}

#[test]
fn rejects_mixed_width_registers_and_narrow_address_bases() {
    let qword = VirtualRegister::new(0, Width::W64);
    let dword = VirtualRegister::new(1, Width::W32);
    let mixed_move = Function {
        assignments: vec![
            assigned(qword, PhysicalRegister::Rax),
            assigned(dword, PhysicalRegister::Rcx),
        ],
        frame: StackFrame::default(),
        operations: vec![
            Operation::Move {
                destination: qword,
                source: dword,
            },
            Operation::Return,
        ],
    };
    assert_eq!(
        validate_function(&mixed_move),
        Err(CompileError::RegisterWidthMismatch {
            expected: Width::W64,
            actual: Width::W32,
        })
    );

    let narrow_base = Function {
        assignments: vec![assigned(dword, PhysicalRegister::Rcx)],
        frame: StackFrame::default(),
        operations: vec![
            Operation::Load {
                destination: dword,
                address: Address::new(dword, 0),
            },
            Operation::Return,
        ],
    };
    assert_eq!(
        validate_function(&narrow_base),
        Err(CompileError::AddressBaseNotQword { register: dword })
    );
}

#[test]
fn emits_width_specific_memory_arithmetic_and_compare() {
    let cases = [
        (Width::W8, Register::AL, Register::DL),
        (Width::W16, Register::AX, Register::DX),
        (Width::W32, Register::EAX, Register::EDX),
        (Width::W64, Register::RAX, Register::RDX),
    ];
    for (width, expected_destination, expected_source) in cases {
        let base = VirtualRegister::qword(0);
        let destination = VirtualRegister::new(1, width);
        let source = VirtualRegister::new(2, width);
        let function = Function {
            assignments: vec![
                assigned(base, PhysicalRegister::Rcx),
                assigned(destination, PhysicalRegister::Rax),
                assigned(source, PhysicalRegister::Rdx),
            ],
            frame: StackFrame::default(),
            operations: vec![
                Operation::Load {
                    destination,
                    address: Address::new(base, 4),
                },
                Operation::Binary {
                    operation: BinaryOp::Xor,
                    destination,
                    source: BinarySource::Register(source),
                },
                Operation::Compare {
                    left: destination,
                    right: BinarySource::Register(source),
                },
                Operation::Store {
                    address: Address::new(base, 8),
                    source: destination,
                },
                Operation::Return,
            ],
        };
        let compiled = compile(&function, CodeAddress(0)).expect("compile width-aware operations");
        let instructions = decode_all(&compiled.bytes, 0);
        assert_eq!(instructions[0].op0_register(), expected_destination);
        assert_eq!(instructions[1].op0_register(), expected_destination);
        assert_eq!(instructions[1].op1_register(), expected_source);
        assert_eq!(instructions[2].op0_register(), expected_destination);
        assert_eq!(instructions[3].op1_register(), expected_destination);
    }
}

#[test]
fn rejects_byte_multiply_without_inventing_a_widening() {
    let destination = VirtualRegister::new(0, Width::W8);
    let source = VirtualRegister::new(1, Width::W8);
    let function = Function {
        assignments: vec![
            assigned(destination, PhysicalRegister::Rax),
            assigned(source, PhysicalRegister::Rdx),
        ],
        frame: StackFrame::default(),
        operations: vec![
            Operation::Binary {
                operation: BinaryOp::Multiply,
                destination,
                source: BinarySource::Register(source),
            },
            Operation::Return,
        ],
    };
    assert_eq!(
        validate_function(&function),
        Err(CompileError::UnsupportedBinaryWidth {
            operation: BinaryOp::Multiply,
            width: Width::W8,
        })
    );
}

#[test]
fn branch_targets_survive_varied_instruction_lengths() {
    for immediate in [0_u64, 1, u64::from(u32::MAX), u64::MAX] {
        let value = vr(0);
        let exit = Label(0);
        let function = Function {
            assignments: vec![assigned(value, PhysicalRegister::R11)],
            frame: StackFrame::default(),
            operations: vec![
                Operation::Jump(exit),
                Operation::Label(exit),
                Operation::MoveImmediate {
                    destination: value,
                    value: immediate,
                },
                Operation::Return,
            ],
        };
        let address = 0x7FFF_0000_0000;
        let compiled = compile(&function, CodeAddress(address)).expect("compile varied immediate");
        let instructions = decode_all(&compiled.bytes, address);
        assert_eq!(
            instructions[0].near_branch_target(),
            address + u64::from(compiled.label_offsets[&exit])
        );
    }
}

#[test]
fn every_binary_operation_decodes_to_the_requested_semantic() {
    let destination = vr(0);
    let source = vr(1);
    let cases = [
        (BinaryOp::Add, Mnemonic::Add),
        (BinaryOp::Subtract, Mnemonic::Sub),
        (BinaryOp::Multiply, Mnemonic::Imul),
        (BinaryOp::Xor, Mnemonic::Xor),
        (BinaryOp::And, Mnemonic::And),
        (BinaryOp::Or, Mnemonic::Or),
    ];

    for (operation, expected) in cases {
        let function = Function {
            assignments: vec![
                assigned(destination, PhysicalRegister::Rax),
                assigned(source, PhysicalRegister::R10),
            ],
            frame: StackFrame::default(),
            operations: vec![
                Operation::Binary {
                    operation,
                    destination,
                    source: BinarySource::Register(source),
                },
                Operation::Return,
            ],
        };
        let compiled = compile(&function, CodeAddress(0x1000)).expect("compile binary operation");
        assert_eq!(decode_all(&compiled.bytes, 0x1000)[0].mnemonic(), expected);
    }
}

#[test]
fn emits_unary_and_shift_operations_at_every_width() {
    for width in [Width::W8, Width::W16, Width::W32, Width::W64] {
        let destination = VirtualRegister::new(0, width);
        let count = VirtualRegister::new(1, Width::W8);
        let function = Function {
            assignments: vec![
                assigned(destination, PhysicalRegister::Rax),
                assigned(count, PhysicalRegister::Rcx),
            ],
            frame: StackFrame::default(),
            operations: vec![
                Operation::Unary {
                    operation: UnaryOp::Not,
                    destination,
                },
                Operation::Unary {
                    operation: UnaryOp::Negate,
                    destination,
                },
                Operation::Shift {
                    operation: ShiftOp::Left,
                    destination,
                    count: ShiftCount::Immediate(3),
                },
                Operation::Shift {
                    operation: ShiftOp::RightLogical,
                    destination,
                    count: ShiftCount::Register(count),
                },
                Operation::Shift {
                    operation: ShiftOp::RightArithmetic,
                    destination,
                    count: ShiftCount::Immediate(2),
                },
                Operation::Shift {
                    operation: ShiftOp::RotateLeft,
                    destination,
                    count: ShiftCount::Immediate(1),
                },
                Operation::Shift {
                    operation: ShiftOp::RotateRight,
                    destination,
                    count: ShiftCount::Immediate(1),
                },
                Operation::Return,
            ],
        };
        let compiled = compile(&function, CodeAddress(0x1000)).expect("compile shifts");
        let mnemonics: Vec<_> = decode_all(&compiled.bytes, 0x1000)
            .into_iter()
            .map(|instruction| instruction.mnemonic())
            .collect();
        assert_eq!(
            &mnemonics[..7],
            [
                Mnemonic::Not,
                Mnemonic::Neg,
                Mnemonic::Shl,
                Mnemonic::Shr,
                Mnemonic::Sar,
                Mnemonic::Rol,
                Mnemonic::Ror,
            ]
        );
    }
}

#[test]
fn rejects_invalid_variable_shift_count_allocations() {
    let destination = VirtualRegister::qword(0);
    let wide_count = VirtualRegister::new(1, Width::W32);
    let function = Function {
        assignments: vec![
            assigned(destination, PhysicalRegister::Rax),
            assigned(wide_count, PhysicalRegister::Rcx),
        ],
        frame: StackFrame::default(),
        operations: vec![
            Operation::Shift {
                operation: ShiftOp::Left,
                destination,
                count: ShiftCount::Register(wide_count),
            },
            Operation::Return,
        ],
    };
    assert_eq!(
        validate_function(&function),
        Err(CompileError::ShiftCountNotByte {
            register: wide_count,
        })
    );

    let byte_count = VirtualRegister::new(1, Width::W8);
    let function = Function {
        assignments: vec![
            assigned(destination, PhysicalRegister::Rax),
            assigned(byte_count, PhysicalRegister::Rdx),
        ],
        frame: StackFrame::default(),
        operations: vec![
            Operation::Shift {
                operation: ShiftOp::Left,
                destination,
                count: ShiftCount::Register(byte_count),
            },
            Operation::Return,
        ],
    };
    assert_eq!(
        validate_function(&function),
        Err(CompileError::ShiftCountNotCl {
            register: byte_count,
            actual: PhysicalRegister::Rdx,
        })
    );
}

#[test]
fn emits_every_valid_integer_resize() {
    let cases = [
        (ResizeOp::Truncate, Width::W16, Width::W8, Mnemonic::Mov),
        (ResizeOp::Truncate, Width::W32, Width::W16, Mnemonic::Mov),
        (ResizeOp::Truncate, Width::W64, Width::W32, Mnemonic::Mov),
        (ResizeOp::ZeroExtend, Width::W8, Width::W16, Mnemonic::Movzx),
        (ResizeOp::ZeroExtend, Width::W8, Width::W32, Mnemonic::Movzx),
        (ResizeOp::ZeroExtend, Width::W8, Width::W64, Mnemonic::Movzx),
        (
            ResizeOp::ZeroExtend,
            Width::W16,
            Width::W32,
            Mnemonic::Movzx,
        ),
        (
            ResizeOp::ZeroExtend,
            Width::W16,
            Width::W64,
            Mnemonic::Movzx,
        ),
        (ResizeOp::ZeroExtend, Width::W32, Width::W64, Mnemonic::Mov),
        (ResizeOp::SignExtend, Width::W8, Width::W16, Mnemonic::Movsx),
        (ResizeOp::SignExtend, Width::W8, Width::W32, Mnemonic::Movsx),
        (ResizeOp::SignExtend, Width::W8, Width::W64, Mnemonic::Movsx),
        (
            ResizeOp::SignExtend,
            Width::W16,
            Width::W32,
            Mnemonic::Movsx,
        ),
        (
            ResizeOp::SignExtend,
            Width::W16,
            Width::W64,
            Mnemonic::Movsx,
        ),
        (
            ResizeOp::SignExtend,
            Width::W32,
            Width::W64,
            Mnemonic::Movsxd,
        ),
    ];
    for (operation, source_width, destination_width, expected) in cases {
        let source = VirtualRegister::new(0, source_width);
        let destination = VirtualRegister::new(1, destination_width);
        let function = Function {
            assignments: vec![
                assigned(source, PhysicalRegister::Rdx),
                assigned(destination, PhysicalRegister::Rax),
            ],
            frame: StackFrame::default(),
            operations: vec![
                Operation::Resize {
                    operation,
                    destination,
                    source,
                },
                Operation::Return,
            ],
        };
        let compiled = compile(&function, CodeAddress(0x1000)).expect("compile integer resize");
        assert_eq!(decode_all(&compiled.bytes, 0x1000)[0].mnemonic(), expected);
    }
}

#[test]
fn rejects_directionally_invalid_resizes() {
    let source = VirtualRegister::new(0, Width::W32);
    let destination = VirtualRegister::new(1, Width::W16);
    let function = Function {
        assignments: vec![
            assigned(source, PhysicalRegister::Rdx),
            assigned(destination, PhysicalRegister::Rax),
        ],
        frame: StackFrame::default(),
        operations: vec![
            Operation::Resize {
                operation: ResizeOp::ZeroExtend,
                destination,
                source,
            },
            Operation::Return,
        ],
    };
    assert_eq!(
        validate_function(&function),
        Err(CompileError::InvalidResize {
            operation: ResizeOp::ZeroExtend,
            source_width: Width::W32,
            destination_width: Width::W16,
        })
    );
}

#[test]
fn every_condition_uses_a_valid_near_branch_to_the_exact_label() {
    let value = vr(0);
    let target = Label(1);
    let conditions = [
        (Condition::Equal, Mnemonic::Je),
        (Condition::NotEqual, Mnemonic::Jne),
        (Condition::Below, Mnemonic::Jb),
        (Condition::BelowOrEqual, Mnemonic::Jbe),
        (Condition::Above, Mnemonic::Ja),
        (Condition::AboveOrEqual, Mnemonic::Jae),
        (Condition::Less, Mnemonic::Jl),
        (Condition::LessOrEqual, Mnemonic::Jle),
        (Condition::Greater, Mnemonic::Jg),
        (Condition::GreaterOrEqual, Mnemonic::Jge),
        (Condition::Overflow, Mnemonic::Jo),
        (Condition::NotOverflow, Mnemonic::Jno),
        (Condition::Sign, Mnemonic::Js),
        (Condition::NotSign, Mnemonic::Jns),
        (Condition::Parity, Mnemonic::Jp),
        (Condition::NotParity, Mnemonic::Jnp),
    ];

    for (condition, expected_mnemonic) in conditions {
        let function = Function {
            assignments: vec![assigned(value, PhysicalRegister::Rax)],
            frame: StackFrame::default(),
            operations: vec![
                Operation::Compare {
                    left: value,
                    right: BinarySource::Immediate(0),
                },
                Operation::BranchIf { condition, target },
                Operation::Return,
                Operation::Label(target),
                Operation::Return,
            ],
        };
        let address = 0x1_4000_0000;
        let compiled = compile(&function, CodeAddress(address)).expect("compile condition");
        let branch = decode_all(&compiled.bytes, address)
            .into_iter()
            .find(|instruction| instruction.flow_control() == FlowControl::ConditionalBranch)
            .expect("one conditional branch");
        assert_eq!(
            branch.near_branch_target(),
            address + u64::from(compiled.label_offsets[&target])
        );
        assert_eq!(branch.mnemonic(), expected_mnemonic);
    }
}

#[test]
fn consecutive_labels_alias_one_instruction() {
    let first = Label(1);
    let second = Label(2);
    let function = Function {
        assignments: vec![],
        frame: StackFrame::default(),
        operations: vec![
            Operation::Jump(first),
            Operation::Label(first),
            Operation::Label(second),
            Operation::Return,
        ],
    };
    let compiled = compile(&function, CodeAddress(0x1000)).expect("compile aliased labels");
    assert_eq!(
        compiled.label_offsets[&first],
        compiled.label_offsets[&second]
    );
}
