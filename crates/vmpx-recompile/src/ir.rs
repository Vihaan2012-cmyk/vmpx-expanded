use crate::abi::{RegisterAssignment, StackFrame};

/// Width of one operation's view of a lowered virtual-register family.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Width {
    W8,
    W16,
    W32,
    W64,
}

impl Width {
    #[must_use]
    pub const fn bits(self) -> u32 {
        match self {
            Self::W8 => 8,
            Self::W16 => 16,
            Self::W32 => 32,
            Self::W64 => 64,
        }
    }
}

/// A register in lowered IR, before physical register assignment.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct VirtualRegister {
    /// Storage identity. Registers with the same ID are width-specific views
    /// of the same allocated physical register.
    pub id: u32,
    pub width: Width,
}

impl VirtualRegister {
    #[must_use]
    pub const fn new(id: u32, width: Width) -> Self {
        Self { id, width }
    }

    #[must_use]
    pub const fn qword(id: u32) -> Self {
        Self::new(id, Width::W64)
    }
}

/// Symbolic branch destination within one function.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Label(pub u32);

/// One eight-byte slot in the function's private stack frame.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SpillSlot(pub u16);

/// Absolute native address used only while resolving placement-dependent
/// relative call encodings.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct NativeAddress(pub u64);

/// Native integer-call destination. Import calls name the address of the IAT
/// slot, not the function pointer currently stored in that slot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CallTarget {
    Direct(NativeAddress),
    ImportAddressTable(NativeAddress),
    Register(VirtualRegister),
}

/// Allocatable general-purpose AMD64 registers. `RSP` is intentionally absent.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum PhysicalRegister {
    Rax,
    Rcx,
    Rdx,
    Rbx,
    Rbp,
    Rsi,
    Rdi,
    R8,
    R9,
    R10,
    R11,
    R12,
    R13,
    R14,
    R15,
}

impl PhysicalRegister {
    #[must_use]
    pub const fn is_win64_nonvolatile(self) -> bool {
        matches!(
            self,
            Self::Rbx
                | Self::Rbp
                | Self::Rsi
                | Self::Rdi
                | Self::R12
                | Self::R13
                | Self::R14
                | Self::R15
        )
    }
}

/// Qword memory operand `[base + displacement]`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Address {
    pub base: VirtualRegister,
    pub displacement: i32,
}

impl Address {
    #[must_use]
    pub const fn new(base: VirtualRegister, displacement: i32) -> Self {
        Self { base, displacement }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BinaryOp {
    Add,
    Subtract,
    Multiply,
    Xor,
    And,
    Or,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UnaryOp {
    Not,
    Negate,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShiftOp {
    Left,
    RightLogical,
    RightArithmetic,
    RotateLeft,
    RotateRight,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShiftCount {
    Immediate(u8),
    Register(VirtualRegister),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResizeOp {
    Truncate,
    ZeroExtend,
    SignExtend,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BinarySource {
    Register(VirtualRegister),
    Immediate(i32),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Condition {
    Equal,
    NotEqual,
    Below,
    BelowOrEqual,
    Above,
    AboveOrEqual,
    Less,
    LessOrEqual,
    Greater,
    GreaterOrEqual,
    Overflow,
    NotOverflow,
    Sign,
    NotSign,
    Parity,
    NotParity,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Operation {
    Label(Label),
    MoveImmediate {
        destination: VirtualRegister,
        value: u64,
    },
    Move {
        destination: VirtualRegister,
        source: VirtualRegister,
    },
    Binary {
        operation: BinaryOp,
        destination: VirtualRegister,
        source: BinarySource,
    },
    Unary {
        operation: UnaryOp,
        destination: VirtualRegister,
    },
    Shift {
        operation: ShiftOp,
        destination: VirtualRegister,
        count: ShiftCount,
    },
    Resize {
        operation: ResizeOp,
        destination: VirtualRegister,
        source: VirtualRegister,
    },
    Load {
        destination: VirtualRegister,
        address: Address,
    },
    Store {
        address: Address,
        source: VirtualRegister,
    },
    SpillLoad {
        destination: VirtualRegister,
        slot: SpillSlot,
    },
    SpillStore {
        slot: SpillSlot,
        source: VirtualRegister,
    },
    /// Win64 integer call with at most the four register arguments represented
    /// explicitly. The result, when present, is the value returned in RAX.
    Call {
        target: CallTarget,
        arguments: Vec<VirtualRegister>,
        result: Option<VirtualRegister>,
    },
    Compare {
        left: VirtualRegister,
        right: BinarySource,
    },
    Jump(Label),
    BranchIf {
        condition: Condition,
        target: Label,
    },
    Return,
}

/// One lowered function and the allocation/frame facts required to emit it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Function {
    pub operations: Vec<Operation>,
    pub assignments: Vec<RegisterAssignment>,
    pub frame: StackFrame,
}
