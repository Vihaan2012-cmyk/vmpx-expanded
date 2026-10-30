use iced_x86::{Code, Decoder, DecoderOptions, Mnemonic, Register};
use vmpx_recompile::{
    Address, CodeAddress, CompileError, DoubleShiftDirection, Function, Operation,
    PhysicalRegister, RegisterAssignment, ShiftCount, SpillSlot, StackFrame, VirtualRegister,
    Width, compile,
};

fn decode_all(bytes: &[u8]) -> Vec<iced_x86::Instruction> {
    let mut decoder = Decoder::with_ip(64, bytes, 0x1000, DecoderOptions::NONE);
    let mut out = Vec::new();
    while decoder.can_decode() {
        let instruction = decoder.decode();
        assert_ne!(instruction.code(), Code::INVALID);
        out.push(instruction);
    }
    out
}

fn reg(id: u32, width: Width) -> VirtualRegister {
    VirtualRegister::new(id, width)
}

fn function(
    assignments: Vec<RegisterAssignment>,
    slots: u16,
    pre: Vec<SpillSlot>,
    operations: Vec<Operation>,
) -> Function {
    Function {
        operations,
        assignments,
        frame: StackFrame {
            stack_allocation: u32::from(slots) * 8,
            preserve: Vec::new(),
            call_ready: false,
            preinitialized_slots: pre,
            tail_transfer: false,
        },
    }
}

#[test]
fn gs_load_emits_a_gs_segment_override() {
    let pointer = reg(0, Width::W64);
    let f = function(
        vec![RegisterAssignment::new(pointer, PhysicalRegister::Rax)],
        0,
        vec![],
        vec![
            Operation::MoveImmediate {
                destination: pointer,
                value: 0x60,
            },
            Operation::GsLoad {
                destination: pointer,
                address: Address::new(pointer, 0),
            },
            Operation::Return,
        ],
    );
    let compiled = compile(&f, CodeAddress(0x1000)).expect("compile gs load");
    let load = decode_all(&compiled.bytes)
        .into_iter()
        .find(|i| i.mnemonic() == Mnemonic::Mov && i.op0_register() == Register::RAX && i.memory_base() == Register::RAX)
        .expect("gs mov");
    assert_eq!(load.segment_prefix(), Register::GS);
}

#[test]
fn double_shifts_emit_shld_and_shrd_with_cl() {
    let dst = reg(0, Width::W32);
    let src = reg(1, Width::W32);
    let count = reg(2, Width::W8);
    for (direction, mnemonic) in [
        (DoubleShiftDirection::Left, Mnemonic::Shld),
        (DoubleShiftDirection::Right, Mnemonic::Shrd),
    ] {
        let f = function(
            vec![
                RegisterAssignment::new(dst, PhysicalRegister::Rax),
                RegisterAssignment::new(src, PhysicalRegister::R10),
                RegisterAssignment::new(count, PhysicalRegister::Rcx),
            ],
            0,
            vec![],
            vec![
                Operation::MoveImmediate { destination: dst, value: 1 },
                Operation::MoveImmediate { destination: src, value: 2 },
                Operation::MoveImmediate { destination: count, value: 3 },
                Operation::DoubleShift {
                    direction,
                    destination: dst,
                    source: src,
                    count: ShiftCount::Register(count),
                },
                Operation::Return,
            ],
        );
        let compiled = compile(&f, CodeAddress(0x1000)).expect("compile double shift");
        let instruction = decode_all(&compiled.bytes)
            .into_iter()
            .find(|i| i.mnemonic() == mnemonic)
            .expect("double shift instruction");
        assert_eq!(instruction.op0_register(), Register::EAX);
        assert_eq!(instruction.op1_register(), Register::R10D);
        assert_eq!(instruction.op2_register(), Register::CL);
    }
}

#[test]
fn byte_double_shift_and_non_cl_count_are_rejected() {
    let dst = reg(0, Width::W8);
    let src = reg(1, Width::W8);
    let f = function(
        vec![
            RegisterAssignment::new(dst, PhysicalRegister::Rax),
            RegisterAssignment::new(src, PhysicalRegister::R10),
        ],
        0,
        vec![],
        vec![
            Operation::MoveImmediate { destination: dst, value: 1 },
            Operation::MoveImmediate { destination: src, value: 2 },
            Operation::DoubleShift {
                direction: DoubleShiftDirection::Left,
                destination: dst,
                source: src,
                count: ShiftCount::Immediate(1),
            },
            Operation::Return,
        ],
    );
    assert!(matches!(
        compile(&f, CodeAddress(0)),
        Err(CompileError::UnsupportedWidth { .. })
    ));

    let dst = reg(0, Width::W64);
    let src = reg(1, Width::W64);
    let count = reg(2, Width::W8);
    let f = function(
        vec![
            RegisterAssignment::new(dst, PhysicalRegister::Rax),
            RegisterAssignment::new(src, PhysicalRegister::R10),
            RegisterAssignment::new(count, PhysicalRegister::Rdx),
        ],
        0,
        vec![],
        vec![
            Operation::MoveImmediate { destination: dst, value: 1 },
            Operation::MoveImmediate { destination: src, value: 2 },
            Operation::MoveImmediate { destination: count, value: 2 },
            Operation::DoubleShift {
                direction: DoubleShiftDirection::Right,
                destination: dst,
                source: src,
                count: ShiftCount::Register(count),
            },
            Operation::Return,
        ],
    );
    assert!(matches!(
        compile(&f, CodeAddress(0)),
        Err(CompileError::ShiftCountNotCl { .. })
    ));
}

#[test]
fn read_flags_is_stack_neutral() {
    let out = reg(0, Width::W64);
    let f = function(
        vec![RegisterAssignment::new(out, PhysicalRegister::R11)],
        0,
        vec![],
        vec![Operation::ReadFlags { destination: out }, Operation::Return],
    );
    let compiled = compile(&f, CodeAddress(0x1000)).expect("compile read flags");
    let instructions = decode_all(&compiled.bytes);
    assert_eq!(instructions[0].mnemonic(), Mnemonic::Pushfq);
    assert_eq!(instructions[1].mnemonic(), Mnemonic::Pop);
    assert_eq!(instructions[1].op0_register(), Register::R11);
}

#[test]
fn cpuid_is_register_transparent_and_requires_initialised_inputs() {
    let leaf = SpillSlot(0);
    let sub = SpillSlot(1);
    let outputs = [SpillSlot(2), SpillSlot(3), SpillSlot(4), SpillSlot(5)];
    let op = Operation::Cpuid {
        leaf,
        subleaf: sub,
        out_eax: outputs[0],
        out_ebx: outputs[1],
        out_ecx: outputs[2],
        out_edx: outputs[3],
    };
    let ok = function(
        vec![],
        6,
        vec![leaf, sub],
        vec![op.clone(), Operation::Return],
    );
    let compiled = compile(&ok, CodeAddress(0x1000)).expect("compile cpuid");
    let instructions = decode_all(&compiled.bytes);
    let cpuid = instructions
        .iter()
        .position(|i| i.mnemonic() == Mnemonic::Cpuid)
        .expect("cpuid");
    let pushes = instructions[..cpuid]
        .iter()
        .filter(|i| i.mnemonic() == Mnemonic::Push)
        .count();
    let pops = instructions[cpuid..]
        .iter()
        .filter(|i| i.mnemonic() == Mnemonic::Pop)
        .count();
    assert_eq!((pushes, pops), (4, 4));
    // Slots are rebased over the four saved registers.
    let leaf_load = &instructions[cpuid - 2];
    assert_eq!(leaf_load.memory_base(), Register::RSP);
    assert_eq!(leaf_load.memory_displacement64(), 32);

    let uninitialised = function(vec![], 6, vec![leaf], vec![op, Operation::Return]);
    assert!(matches!(
        compile(&uninitialised, CodeAddress(0)),
        Err(CompileError::UninitializedSpill { .. })
    ));
}

#[test]
fn entry_stack_pointer_and_popping_return_account_for_the_frame() {
    let out = reg(0, Width::W64);
    let mut f = function(
        vec![RegisterAssignment::new(out, PhysicalRegister::Rax)],
        2,
        vec![],
        vec![
            Operation::EntryStackPointer { destination: out },
            Operation::ReturnPopping { bytes: 40 },
        ],
    );
    f.frame.tail_transfer = true;
    let compiled = compile(&f, CodeAddress(0x1000)).expect("compile popping return");
    let instructions = decode_all(&compiled.bytes);
    // 16 bytes of allocation, no preserved registers.
    assert_eq!(instructions[1].mnemonic(), Mnemonic::Lea);
    assert_eq!(instructions[1].memory_displacement64(), 16);
    let adds: Vec<_> = instructions
        .iter()
        .filter(|i| i.mnemonic() == Mnemonic::Add && i.op0_register() == Register::RSP)
        .map(|i| i.immediate(1))
        .collect();
    assert_eq!(adds, vec![16, 40]);
    assert_eq!(instructions.last().map(iced_x86::Instruction::mnemonic), Some(Mnemonic::Ret));
}
