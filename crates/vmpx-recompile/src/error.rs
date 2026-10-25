use crate::{
    BinaryOp, Label, NativeAddress, PhysicalRegister, ResizeOp, SpillSlot, VirtualRegister, Width,
};
use thiserror::Error;

#[derive(Debug, Error, Eq, PartialEq)]
pub enum CompileError {
    #[error("the lowered function has no operations")]
    EmptyFunction,

    #[error("virtual register {register:?} has no physical assignment")]
    UnassignedRegister { register: VirtualRegister },

    #[error("virtual register {register:?} is assigned more than once")]
    DuplicateVirtualAssignment { register: VirtualRegister },

    #[error("physical register {register:?} is assigned to more than one virtual register")]
    DuplicatePhysicalAssignment { register: PhysicalRegister },

    #[error("virtual register family {id} is assigned inconsistently: {expected:?} and {actual:?}")]
    RegisterFamilyAssignmentMismatch {
        id: u32,
        expected: PhysicalRegister,
        actual: PhysicalRegister,
    },

    #[error("register width {width:?} is unsupported by the native backend")]
    UnsupportedWidth { width: Width },

    #[error("operation uses incompatible register widths: expected {expected:?}, found {actual:?}")]
    RegisterWidthMismatch { expected: Width, actual: Width },

    #[error("memory address base {register:?} is not a qword pointer")]
    AddressBaseNotQword { register: VirtualRegister },

    #[error("binary operation {operation:?} is unsupported at width {width:?}")]
    UnsupportedBinaryWidth { operation: BinaryOp, width: Width },

    #[error("variable shift count {register:?} is not an 8-bit value")]
    ShiftCountNotByte { register: VirtualRegister },

    #[error("variable shift count {register:?} is assigned to {actual:?}, not RCX/CL")]
    ShiftCountNotCl {
        register: VirtualRegister,
        actual: PhysicalRegister,
    },

    #[error("invalid {operation:?} resize from {source_width:?} to {destination_width:?}")]
    InvalidResize {
        operation: ResizeOp,
        source_width: Width,
        destination_width: Width,
    },

    #[error("used Win64 nonvolatile register {register:?} is not preserved by the frame")]
    UnpreservedNonvolatile { register: PhysicalRegister },

    #[error("frame preservation list contains volatile register {register:?}")]
    PreservesVolatile { register: PhysicalRegister },

    #[error("frame preservation list contains duplicate register {register:?}")]
    DuplicatePreservedRegister { register: PhysicalRegister },

    #[error("frame stack allocation {bytes} is not a multiple of 8")]
    InvalidStackAllocation { bytes: u32 },

    #[error("frame stack allocation {bytes} exceeds the encodable signed immediate range")]
    StackAllocationTooLarge { bytes: u32 },

    #[error("spill slot {slot:?} exceeds the backend limit of {maximum} slots")]
    SpillSlotLimit { slot: SpillSlot, maximum: u16 },

    #[error(
        "spill slot {slot:?} ends at frame offset {required_end}, beyond stack allocation {stack_allocation}"
    )]
    SpillOutsideFrame {
        slot: SpillSlot,
        required_end: u32,
        stack_allocation: u32,
    },

    #[error(
        "operation {operation_index} loads {required_bytes} bytes from spill slot {slot:?}, but only {initialized_bytes} bytes are definitely initialized"
    )]
    UninitializedSpill {
        operation_index: usize,
        slot: SpillSlot,
        initialized_bytes: u8,
        required_bytes: u8,
    },

    #[error(
        "call-ready frame has {bytes} stack bytes; Win64 requires at least 32 bytes of shadow space"
    )]
    MissingShadowSpace { bytes: u32 },

    #[error("call-ready frame leaves RSP congruent to {actual} modulo 16 instead of 0")]
    MisalignedCallFrame { actual: u8 },

    #[error("native call at operation {operation_index} requires a call-ready Win64 frame")]
    CallFrameNotReady { operation_index: usize },

    #[error(
        "native call at operation {operation_index} has {actual} register arguments; at most four are supported"
    )]
    TooManyCallArguments {
        operation_index: usize,
        actual: usize,
    },

    #[error("native call argument {ordinal} uses {actual:?}, expected Win64 register {expected:?}")]
    CallArgumentRegister {
        ordinal: usize,
        expected: PhysicalRegister,
        actual: PhysicalRegister,
    },

    #[error("native call result uses {actual:?}, expected Win64 return register RAX")]
    CallResultRegister { actual: PhysicalRegister },

    #[error("native register-call target {register:?} is not a qword")]
    CallTargetNotQword { register: VirtualRegister },

    #[error(
        "operation {operation_index} reads {register:?} from volatile {physical:?} after a native call clobbered it"
    )]
    CallClobberedRegisterRead {
        operation_index: usize,
        register: VirtualRegister,
        physical: PhysicalRegister,
    },

    #[error(
        "conditional branch at operation {operation_index} reads flags clobbered by a native call"
    )]
    CallClobberedFlagsRead { operation_index: usize },

    #[error(
        "native call at operation {operation_index} cannot reach {target:?} with a signed rel32 displacement"
    )]
    CallOutOfRange {
        operation_index: usize,
        target: NativeAddress,
    },

    #[error("label {label:?} is defined more than once")]
    DuplicateLabel { label: Label },

    #[error("branch target {label:?} is not defined")]
    UndefinedLabel { label: Label },

    #[error("label {label:?} does not precede an instruction")]
    EmptyLabel { label: Label },

    #[error("operation {operation_index} is unreachable because its block already terminated")]
    OperationAfterTerminator { operation_index: usize },

    #[error("the last basic block can fall through the end of emitted code")]
    FallthroughPastEnd,

    #[error("machine-code offset cannot be represented")]
    CodeSizeOverflow,

    #[error("branch from offset {source_offset} to label {target:?} exceeds rel32 range")]
    BranchOutOfRange { source_offset: u32, target: Label },

    #[error("iced-x86 failed while encoding operation {operation_index}: {message}")]
    Encoding {
        operation_index: usize,
        message: String,
    },
}
