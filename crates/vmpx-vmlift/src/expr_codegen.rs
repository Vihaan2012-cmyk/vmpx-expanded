use std::collections::{BTreeMap, BTreeSet};

use thiserror::Error;
use vmpx_recompile::{
    Address, BinaryOp as NativeBinaryOp, BinarySource, CallTarget, CodeAddress, CompileError,
    CompiledFunction, DoubleShiftDirection, Function, NativeAddress, Operation, PhysicalRegister, RegisterAssignment,
    ResizeOp as NativeResizeOp, ShiftCount, ShiftOp, SpillSlot, StackFrame,
    UnaryOp as NativeUnaryOp, VirtualRegister, Width, compile, validate_function,
};
use vmpx_vm::{BinaryOp, BitWidth, Expr, ExprKind, ResizeOp, UnaryOp};

use crate::passthrough;

const ACCUMULATOR_ID: u32 = 0;
const TEMPORARY_ID: u32 = 1;
const SHIFT_COUNT_ID: u32 = 2;
const FIRST_INPUT_ID: u32 = 3;
const BACKEND_SPILL_LIMIT: usize = 4_096;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpressionInput {
    pub name: String,
    pub width: BitWidth,
    /// Where the value lives when the expression starts. A register input is
    /// captured from that register; a pre-spilled input is read directly from a
    /// caller-allocated frame slot, which lets several expressions share one
    /// set of inputs without consuming registers.
    pub source: InputSource,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InputSource {
    Register(PhysicalRegister),
    /// Frame slot already holding the value, addressed from the function's
    /// frame base.
    Slot(SpillSlot),
}

impl ExpressionInput {
    #[must_use]
    pub fn register(name: impl Into<String>, width: BitWidth, register: PhysicalRegister) -> Self {
        Self {
            name: name.into(),
            width,
            source: InputSource::Register(register),
        }
    }

    #[must_use]
    pub fn slot(name: impl Into<String>, width: BitWidth, slot: SpillSlot) -> Self {
        Self {
            name: name.into(),
            width,
            source: InputSource::Slot(slot),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpressionCodegenConfig {
    pub inputs: Vec<ExpressionInput>,
    pub result_register: PhysicalRegister,
    pub temporary_register: PhysicalRegister,
    pub shift_count_register: PhysicalRegister,
    pub max_expression_nodes: usize,
    pub max_spill_slots: usize,
    /// Intrinsics that may be lowered to a real imported call, keyed by the
    /// intrinsic name. A fresh-allocation intrinsic stands for the return value
    /// of an allocator call, so lowering it emits that call rather than
    /// inventing a value. Any intrinsic absent from this map is refused.
    pub import_calls: BTreeMap<String, ImportCall>,
    /// First slot the fragment may use for its own storage.
    ///
    /// A caller that reserves slots above its inputs, such as the value result
    /// slots of a composed replacement, sets this so the fragment never numbers a
    /// temporary onto a slot the caller already owns. `max_spill_slots` stays a
    /// count of slots the fragment itself may use, so the absolute limit is this
    /// value plus that count.
    pub first_scratch_slot: Option<u16>,
}

/// How to emit one imported call for an intrinsic.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImportCall {
    /// File offset of the IAT slot holding the imported address.
    pub iat_address: u64,
    /// Number of leading intrinsic arguments passed as the call's arguments,
    /// starting at `RCX`.
    pub argument_count: u8,
}

impl Default for ExpressionCodegenConfig {
    fn default() -> Self {
        Self {
            inputs: Vec::new(),
            result_register: PhysicalRegister::Rax,
            temporary_register: PhysicalRegister::R10,
            shift_count_register: PhysicalRegister::Rcx,
            max_expression_nodes: 65_536,
            max_spill_slots: BACKEND_SPILL_LIMIT,
            import_calls: BTreeMap::new(),
            first_scratch_slot: None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LoweredExpression {
    pub function: Function,
    pub result: VirtualRegister,
    pub expression_nodes: usize,
    pub spill_slots: usize,
    /// Intrinsics re-emitted as the native instruction they denote.
    pub passthroughs: BTreeSet<String>,
    /// The subset of `passthroughs` that reads machine state (`cpuid`, `gs`
    /// loads): correct by identity, trusted rather than statically provable.
    pub environment_boundaries: BTreeSet<String>,
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum ExpressionCodegenError {
    #[error("expression contains {nodes} nodes, exceeding limit {limit}")]
    NodeLimit { nodes: usize, limit: usize },

    #[error("expression input {name} is not mapped")]
    MissingInput { name: String },

    #[error("expression input {name} is {actual:?}, but its mapping is {expected:?}")]
    InputWidth {
        name: String,
        expected: BitWidth,
        actual: BitWidth,
    },

    #[error("expression input {name} is mapped more than once")]
    DuplicateInput { name: String },

    #[error("codegen scratch registers must be three distinct physical registers")]
    ScratchRegisterCollision,

    #[error("variable shift count register must be RCX for x86 CL semantics")]
    ShiftCountRegister,

    #[error("expression requires unsupported intrinsic {name}")]
    UnsupportedIntrinsic { name: String },

    #[error("native passthrough {name} is malformed: {reason}")]
    MalformedPassthrough { name: String, reason: &'static str },

    #[error("imported call takes {count} arguments, exceeding the four register slots")]
    TooManyCallArguments { count: usize },

    #[error("memory-load address must be 64-bit, got {width:?}")]
    AddressWidth { width: BitWidth },

    #[error("expression spill demand {required} exceeds limit {limit}")]
    SpillLimit { required: usize, limit: usize },

    #[error("expression storage ID space overflowed")]
    StorageIdOverflow,

    #[error("native backend rejected lowered expression: {0}")]
    Compile(#[from] CompileError),
}

#[derive(Clone, Copy)]
struct InputBinding {
    width: BitWidth,
    slot: SpillSlot,
}

struct LoweringContext<'a> {
    config: &'a ExpressionCodegenConfig,
    inputs: BTreeMap<String, InputBinding>,
    assignments: Vec<RegisterAssignment>,
    operations: Vec<Operation>,
    next_slot: usize,
    peak_slot: usize,
    /// Absolute ceiling for `next_slot`: the window start plus the slots the
    /// fragment is allowed to use.
    absolute_slot_limit: usize,
    passthroughs: BTreeSet<String>,
}

fn malformed(name: &str, reason: &'static str) -> ExpressionCodegenError {
    ExpressionCodegenError::MalformedPassthrough {
        name: name.to_owned(),
        reason,
    }
}

impl LoweringContext<'_> {
    /// Re-emits the native instruction an opaque intrinsic denotes. The shape
    /// and every operand width are checked first, and each operand expression is
    /// lowered by the ordinary expression backend, so an operand that cannot be
    /// lowered refuses the whole expression.
    fn lower_passthrough(
        &mut self,
        name: &str,
        kind: passthrough::Passthrough,
        expression: &Expr,
        arguments: &[Expr],
    ) -> Result<(), ExpressionCodegenError> {
        use passthrough::Passthrough;
        match kind {
            Passthrough::Cpuid(output) => {
                let [leaf, subleaf] = arguments else {
                    return Err(malformed(name, "cpuid takes a leaf and a subleaf"));
                };
                if leaf.width != BitWidth::W32
                    || subleaf.width != BitWidth::W32
                    || expression.width != BitWidth::W32
                {
                    return Err(malformed(name, "cpuid operands and results are 32-bit"));
                }
                self.lower(leaf)?;
                let leaf_slot = self.allocate_slot()?;
                self.operations.push(Operation::SpillStore {
                    slot: leaf_slot,
                    source: accumulator(BitWidth::W32),
                });
                self.lower(subleaf)?;
                let subleaf_slot = self.allocate_slot()?;
                self.operations.push(Operation::SpillStore {
                    slot: subleaf_slot,
                    source: accumulator(BitWidth::W32),
                });
                let outputs = [
                    self.allocate_slot()?,
                    self.allocate_slot()?,
                    self.allocate_slot()?,
                    self.allocate_slot()?,
                ];
                self.operations.push(Operation::Cpuid {
                    leaf: leaf_slot,
                    subleaf: subleaf_slot,
                    out_eax: outputs[0],
                    out_ebx: outputs[1],
                    out_ecx: outputs[2],
                    out_edx: outputs[3],
                });
                self.operations.push(Operation::SpillLoad {
                    destination: accumulator(BitWidth::W32),
                    slot: outputs[output.index()],
                });
                for _ in 0..6 {
                    self.release_slot();
                }
            }
            Passthrough::GsLoad => {
                let [pointer] = arguments else {
                    return Err(malformed(name, "a gs load takes one pointer"));
                };
                if pointer.width != BitWidth::W64 || expression.width != BitWidth::W64 {
                    return Err(malformed(name, "a gs load is a 64-bit load at a 64-bit offset"));
                }
                self.lower(pointer)?;
                self.operations.push(Operation::GsLoad {
                    destination: accumulator(BitWidth::W64),
                    address: Address::new(accumulator(BitWidth::W64), 0),
                });
            }
            Passthrough::PopFlags => {
                let [value] = arguments else {
                    return Err(malformed(name, "popfq takes one flags word"));
                };
                if value.width != BitWidth::W64 || expression.width != BitWidth::W64 {
                    return Err(malformed(name, "popfq consumes and yields a 64-bit flags word"));
                }
                self.lower(value)?;
                self.operations.push(Operation::RestoreFlags {
                    value: accumulator(BitWidth::W64),
                });
                self.operations.push(Operation::ReadFlags {
                    destination: accumulator(BitWidth::W64),
                });
            }
            Passthrough::DoubleShift { left, width, flags } => {
                let [destination, source, count] = arguments else {
                    return Err(malformed(name, "a double shift takes destination, source, count"));
                };
                if width == BitWidth::W8 {
                    return Err(malformed(name, "SHLD/SHRD have no 8-bit form"));
                }
                if destination.width != width
                    || source.width != width
                    || count.width != BitWidth::W8
                {
                    return Err(malformed(name, "double-shift operand widths disagree"));
                }
                let result_width = if flags { BitWidth::W64 } else { width };
                if expression.width != result_width {
                    return Err(malformed(name, "double-shift result width disagrees"));
                }
                self.lower(destination)?;
                let destination_slot = self.allocate_slot()?;
                self.operations.push(Operation::SpillStore {
                    slot: destination_slot,
                    source: accumulator(width),
                });
                self.lower(source)?;
                let source_slot = self.allocate_slot()?;
                self.operations.push(Operation::SpillStore {
                    slot: source_slot,
                    source: accumulator(width),
                });
                self.lower(count)?;
                self.operations.push(Operation::Move {
                    destination: register(SHIFT_COUNT_ID, BitWidth::W8),
                    source: accumulator(BitWidth::W8),
                });
                self.operations.push(Operation::SpillLoad {
                    destination: temporary(width),
                    slot: source_slot,
                });
                self.operations.push(Operation::SpillLoad {
                    destination: accumulator(width),
                    slot: destination_slot,
                });
                self.release_slot();
                self.release_slot();
                self.operations.push(Operation::DoubleShift {
                    direction: if left {
                        DoubleShiftDirection::Left
                    } else {
                        DoubleShiftDirection::Right
                    },
                    destination: accumulator(width),
                    source: temporary(width),
                    count: ShiftCount::Register(register(SHIFT_COUNT_ID, BitWidth::W8)),
                });
                if flags {
                    self.operations.push(Operation::ReadFlags {
                        destination: accumulator(BitWidth::W64),
                    });
                }
            }
        }
        Ok(())
    }

    /// Computes the status-flag word of an add or subtract with the native
    /// instruction that defines it, masked to the bits the VM flag model carries
    /// (CF, PF, AF, ZF, SF, OF plus the fixed `0x202`).
    fn lower_status_flags(
        &mut self,
        name: &str,
        subtract: bool,
        width: BitWidth,
        expression: &Expr,
        arguments: &[Expr],
    ) -> Result<(), ExpressionCodegenError> {
        let [left, right] = arguments else {
            return Err(malformed(name, "status flags take two operands"));
        };
        if left.width != width || right.width != width || expression.width != BitWidth::W64 {
            return Err(malformed(name, "status-flag operand widths disagree"));
        }
        self.lower(left)?;
        let left_slot = self.allocate_slot()?;
        self.operations.push(Operation::SpillStore {
            slot: left_slot,
            source: accumulator(width),
        });
        self.lower(right)?;
        self.operations.push(Operation::Move {
            destination: temporary(width),
            source: accumulator(width),
        });
        self.operations.push(Operation::SpillLoad {
            destination: accumulator(width),
            slot: left_slot,
        });
        self.release_slot();
        if subtract {
            self.operations.push(Operation::Compare {
                left: accumulator(width),
                right: BinarySource::Register(temporary(width)),
            });
        } else {
            self.operations.push(Operation::Binary {
                operation: NativeBinaryOp::Add,
                destination: accumulator(width),
                source: BinarySource::Register(temporary(width)),
            });
        }
        self.operations.push(Operation::ReadFlags {
            destination: accumulator(BitWidth::W64),
        });
        self.operations.push(Operation::Binary {
            operation: NativeBinaryOp::And,
            destination: accumulator(BitWidth::W64),
            source: BinarySource::Immediate(STATUS_FLAG_MASK),
        });
        self.operations.push(Operation::Binary {
            operation: NativeBinaryOp::Or,
            destination: accumulator(BitWidth::W64),
            source: BinarySource::Immediate(STATUS_FLAG_FIXED),
        });
        Ok(())
    }

    /// Computes the status-flag word of a VM `nor`/`nand` with the native
    /// logical instruction that defines it. The VM semantics are
    /// `nor(a, b) = (~a) & (~b)` and `nand(a, b) = (~a) | (~b)`, so the flags are
    /// exactly those a real `and`/`or` of the two complemented operands sets:
    /// CF and OF cleared, SF/ZF/PF taken from the result, and AF left by the
    /// hardware. The result-derived bits (SF/ZF/PF) and the cleared CF/OF are
    /// fully defined; AF is architecturally undefined after a logical op but is
    /// reproduced by identity here — the emitted instruction is the same `and`/
    /// `or` the VM handler executed, on the same operands, so it yields the same
    /// AF the original captured (the same trust basis as the double-shift flag
    /// read, which also pushfqs an ISA-undefined OF/AF). Masking AF out instead
    /// would force it to zero and could silently diverge from the original, so
    /// re-emitting the real instruction is the sound choice.
    fn lower_logical_flags(
        &mut self,
        name: &str,
        or: bool,
        width: BitWidth,
        expression: &Expr,
        arguments: &[Expr],
    ) -> Result<(), ExpressionCodegenError> {
        let [left, right] = arguments else {
            return Err(malformed(name, "logical flags take two operands"));
        };
        if left.width != width || right.width != width || expression.width != BitWidth::W64 {
            return Err(malformed(name, "logical-flag operand widths disagree"));
        }
        self.lower(left)?;
        self.operations.push(Operation::Unary {
            operation: NativeUnaryOp::Not,
            destination: accumulator(width),
        });
        let left_slot = self.allocate_slot()?;
        self.operations.push(Operation::SpillStore {
            slot: left_slot,
            source: accumulator(width),
        });
        self.lower(right)?;
        self.operations.push(Operation::Unary {
            operation: NativeUnaryOp::Not,
            destination: accumulator(width),
        });
        self.operations.push(Operation::Move {
            destination: temporary(width),
            source: accumulator(width),
        });
        self.operations.push(Operation::SpillLoad {
            destination: accumulator(width),
            slot: left_slot,
        });
        self.release_slot();
        // The `and`/`or` is the flag-defining instruction; `not`, `mov` and the
        // spill traffic above leave the flags untouched until this point.
        self.operations.push(Operation::Binary {
            operation: if or {
                NativeBinaryOp::Or
            } else {
                NativeBinaryOp::And
            },
            destination: accumulator(width),
            source: BinarySource::Register(temporary(width)),
        });
        self.operations.push(Operation::ReadFlags {
            destination: accumulator(BitWidth::W64),
        });
        self.operations.push(Operation::Binary {
            operation: NativeBinaryOp::And,
            destination: accumulator(BitWidth::W64),
            source: BinarySource::Immediate(STATUS_FLAG_MASK),
        });
        self.operations.push(Operation::Binary {
            operation: NativeBinaryOp::Or,
            destination: accumulator(BitWidth::W64),
            source: BinarySource::Immediate(STATUS_FLAG_FIXED),
        });
        Ok(())
    }

    /// Lowers a wide multiply whose double-width product is packed into a qword.
    ///
    /// The symbolic executor names this `x86_{signed,unsigned}_multiply_wN_K`
    /// only for operand widths up to 32 bits, where the full `N x N` product
    /// fits in 64 bits and both halves are read back with `extract`. Widening
    /// each operand to 64 bits (sign-extended for the signed form, zero-extended
    /// otherwise) and taking the low 64 bits of a 64-bit multiply yields exactly
    /// that packed product, so a single two-operand `imul` is a faithful lowering.
    fn lower_wide_multiply(
        &mut self,
        name: &str,
        signed: bool,
        operand_width: BitWidth,
        expression: &Expr,
        arguments: &[Expr],
    ) -> Result<(), ExpressionCodegenError> {
        let [left, right] = arguments else {
            return Err(malformed(name, "a wide multiply takes two operands"));
        };
        if left.width != operand_width
            || right.width != operand_width
            || expression.width != BitWidth::W64
        {
            return Err(malformed(name, "wide-multiply operand widths disagree"));
        }
        let extend = if signed {
            ResizeOp::SignExtend
        } else {
            ResizeOp::ZeroExtend
        };
        let widen = |value: &Expr| Expr::resize(extend, BitWidth::W64, value.clone());
        let product = widen(left)
            .and_then(|left| widen(right).and_then(|right| Expr::binary(BinaryOp::Mul, left, right)))
            .map_err(|_| malformed(name, "wide-multiply widening failed"))?;
        self.lower(&product)
    }

    /// Lowers a VM multiply-flag word by re-emitting the exact `mul`/`imul` the
    /// handler ran and reading its raw flags. `CF`/`OF` are defined by the
    /// product; the `SF`/`ZF`/`AF`/`PF` the ISA leaves undefined are reproduced by
    /// identity, the same trust basis as the double-shift flag read. Unlike the
    /// add/sub/logical flags, these are not masked: the VM has no concrete model
    /// for them (`evaluate_vm_flags` folds only add/sub/logical), so the faithful
    /// value is the raw captured flag word.
    fn lower_multiply_flags(
        &mut self,
        name: &str,
        signed: bool,
        width: BitWidth,
        expression: &Expr,
        arguments: &[Expr],
    ) -> Result<(), ExpressionCodegenError> {
        let [left, right] = arguments else {
            return Err(malformed(name, "multiply flags take two operands"));
        };
        if left.width != width || right.width != width || expression.width != BitWidth::W64 {
            return Err(malformed(name, "multiply-flag operand widths disagree"));
        }
        self.lower(left)?;
        let left_slot = self.allocate_slot()?;
        self.operations.push(Operation::SpillStore {
            slot: left_slot,
            source: accumulator(width),
        });
        self.lower(right)?;
        let right_slot = self.allocate_slot()?;
        self.operations.push(Operation::SpillStore {
            slot: right_slot,
            source: accumulator(width),
        });
        let flags_slot = self.allocate_slot()?;
        self.operations.push(Operation::MultiplyFlags {
            signed,
            width: native_width(width),
            left: left_slot,
            right: right_slot,
            out_flags: flags_slot,
        });
        self.operations.push(Operation::SpillLoad {
            destination: accumulator(BitWidth::W64),
            slot: flags_slot,
        });
        for _ in 0..3 {
            self.release_slot();
        }
        Ok(())
    }

    /// Evaluates `high`, `low`, `divisor` into width-sized slots and emits the
    /// unsigned divide, returning its quotient, remainder and flags slots. All six
    /// slots it leaves allocated (three operands, three results) are released by
    /// the caller once it has read the result it needs. Fails closed on the byte
    /// form, whose remainder lands in `AH`.
    fn emit_unsigned_divide(
        &mut self,
        name: &str,
        width: BitWidth,
        arguments: &[Expr],
    ) -> Result<(SpillSlot, SpillSlot, SpillSlot), ExpressionCodegenError> {
        let [high, low, divisor] = arguments else {
            return Err(malformed(name, "a wide divide takes high, low and divisor"));
        };
        if high.width != width || low.width != width || divisor.width != width {
            return Err(malformed(name, "wide-divide operand widths disagree"));
        }
        if width == BitWidth::W8 {
            return Err(malformed(name, "byte divide has no lowering"));
        }
        self.lower(high)?;
        let high_slot = self.allocate_slot()?;
        self.operations.push(Operation::SpillStore {
            slot: high_slot,
            source: accumulator(width),
        });
        self.lower(low)?;
        let low_slot = self.allocate_slot()?;
        self.operations.push(Operation::SpillStore {
            slot: low_slot,
            source: accumulator(width),
        });
        self.lower(divisor)?;
        let divisor_slot = self.allocate_slot()?;
        self.operations.push(Operation::SpillStore {
            slot: divisor_slot,
            source: accumulator(width),
        });
        let quotient_slot = self.allocate_slot()?;
        let remainder_slot = self.allocate_slot()?;
        let flags_slot = self.allocate_slot()?;
        self.operations.push(Operation::UnsignedDivide {
            width: native_width(width),
            high: high_slot,
            low: low_slot,
            divisor: divisor_slot,
            out_quotient: quotient_slot,
            out_remainder: remainder_slot,
            out_flags: flags_slot,
        });
        Ok((quotient_slot, remainder_slot, flags_slot))
    }

    /// Lowers the packed or split value of a VM unsigned divide. The value is
    /// computed by the real `div`, which also raises `#DE` exactly where the
    /// virtualized instruction would.
    fn lower_unsigned_divide_value(
        &mut self,
        name: &str,
        width: BitWidth,
        part: DividePart,
        expression: &Expr,
        arguments: &[Expr],
    ) -> Result<(), ExpressionCodegenError> {
        if expression.width != BitWidth::W64 {
            return Err(malformed(name, "a wide-divide value is a qword"));
        }
        match part {
            DividePart::Packed if width == BitWidth::W64 => {
                return Err(malformed(name, "a 64-bit divide cannot pack into one qword"));
            }
            DividePart::Quotient | DividePart::Remainder if width != BitWidth::W64 => {
                return Err(malformed(name, "split divide halves are the 64-bit form"));
            }
            _ => {}
        }
        let (quotient_slot, remainder_slot, _flags_slot) =
            self.emit_unsigned_divide(name, width, arguments)?;
        match part {
            DividePart::Quotient => self.operations.push(Operation::SpillLoad {
                destination: accumulator(BitWidth::W64),
                slot: quotient_slot,
            }),
            DividePart::Remainder => self.operations.push(Operation::SpillLoad {
                destination: accumulator(BitWidth::W64),
                slot: remainder_slot,
            }),
            DividePart::Packed => {
                // `(zero_extend(remainder) << width) | zero_extend(quotient)`,
                // matching how the executor packs the double-width result.
                self.operations.push(Operation::SpillLoad {
                    destination: accumulator(width),
                    slot: quotient_slot,
                });
                self.operations.push(Operation::Resize {
                    operation: NativeResizeOp::ZeroExtend,
                    destination: accumulator(BitWidth::W64),
                    source: accumulator(width),
                });
                self.operations.push(Operation::SpillStore {
                    slot: quotient_slot,
                    source: accumulator(BitWidth::W64),
                });
                self.operations.push(Operation::SpillLoad {
                    destination: accumulator(width),
                    slot: remainder_slot,
                });
                self.operations.push(Operation::Resize {
                    operation: NativeResizeOp::ZeroExtend,
                    destination: accumulator(BitWidth::W64),
                    source: accumulator(width),
                });
                self.operations.push(Operation::Shift {
                    operation: ShiftOp::Left,
                    destination: accumulator(BitWidth::W64),
                    count: ShiftCount::Immediate(
                        u8::try_from(width.bits())
                            .map_err(|_| ExpressionCodegenError::StorageIdOverflow)?,
                    ),
                });
                self.operations.push(Operation::Move {
                    destination: temporary(BitWidth::W64),
                    source: accumulator(BitWidth::W64),
                });
                self.operations.push(Operation::SpillLoad {
                    destination: accumulator(BitWidth::W64),
                    slot: quotient_slot,
                });
                self.operations.push(Operation::Binary {
                    operation: NativeBinaryOp::Or,
                    destination: accumulator(BitWidth::W64),
                    source: BinarySource::Register(temporary(BitWidth::W64)),
                });
            }
        }
        for _ in 0..6 {
            self.release_slot();
        }
        Ok(())
    }

    /// Lowers a VM divide-flag word by re-emitting the real `div` and reading its
    /// raw flags. `div` leaves every status flag ISA-undefined, so there is no
    /// concrete model to match; the captured flag word is faithful by identity.
    fn lower_unsigned_divide_flags(
        &mut self,
        name: &str,
        width: BitWidth,
        expression: &Expr,
        arguments: &[Expr],
    ) -> Result<(), ExpressionCodegenError> {
        if expression.width != BitWidth::W64 {
            return Err(malformed(name, "a divide flag word is a qword"));
        }
        let (_quotient_slot, _remainder_slot, flags_slot) =
            self.emit_unsigned_divide(name, width, arguments)?;
        self.operations.push(Operation::SpillLoad {
            destination: accumulator(BitWidth::W64),
            slot: flags_slot,
        });
        for _ in 0..6 {
            self.release_slot();
        }
        Ok(())
    }

    /// Lowers an intrinsic that stands for an imported call's return value.
    ///
    /// The intrinsic's leading arguments become the call arguments, the call is
    /// emitted against the IAT slot, and the return value is left in `RAX`, which
    /// is then moved into the accumulator.
    fn lower_import_call(
        &mut self,
        expression: &Expr,
        call: &ImportCall,
        arguments: &[Expr],
    ) -> Result<(), ExpressionCodegenError> {
        const ARGUMENT_REGISTERS: [PhysicalRegister; 4] = [
            PhysicalRegister::Rcx,
            PhysicalRegister::Rdx,
            PhysicalRegister::R8,
            PhysicalRegister::R9,
        ];
        let count = usize::from(call.argument_count);
        if count > ARGUMENT_REGISTERS.len() {
            return Err(ExpressionCodegenError::TooManyCallArguments { count });
        }
        let mut argument_registers = Vec::with_capacity(count);
        for (index, argument) in arguments.iter().take(count).enumerate() {
            // Evaluate the argument into the accumulator, then move it into its
            // ABI register.
            self.lower(argument)?;
            let target = VirtualRegister::qword(ACCUMULATOR_ID);
            let destination = VirtualRegister::qword(
                9_000
                    + u32::try_from(index)
                        .map_err(|_| ExpressionCodegenError::StorageIdOverflow)?,
            );
            // Reuse the shift-count register's storage when the ABI argument is
            // RCX: the lowering already assigns that physical register, and the
            // backend allows one virtual register per physical register.
            let destination = self
                .assignments
                .iter()
                .find(|assignment| assignment.physical_register == ARGUMENT_REGISTERS[index])
                .map_or(destination, |assignment| assignment.virtual_register);
            if !self
                .assignments
                .iter()
                .any(|assignment| assignment.virtual_register.id == destination.id)
            {
                self.assignments.push(RegisterAssignment::new(
                    destination,
                    ARGUMENT_REGISTERS[index],
                ));
            }
            self.operations.push(Operation::Move {
                destination,
                source: target,
            });
            argument_registers.push(destination);
        }
        let result = VirtualRegister::qword(
            9_500 + u32::try_from(count).map_err(|_| ExpressionCodegenError::StorageIdOverflow)?,
        );
        self.assignments
            .push(RegisterAssignment::new(result, PhysicalRegister::Rax));
        self.operations.push(Operation::Call {
            target: CallTarget::ImportAddressTable(NativeAddress(call.iat_address)),
            arguments: argument_registers,
            result: Some(result),
        });
        self.operations.push(Operation::Move {
            destination: accumulator(expression.width),
            source: result,
        });
        Ok(())
    }
}

const STATUS_FLAG_MASK: i32 = 0x8D5;
const STATUS_FLAG_FIXED: i32 = 0x202;

/// `vm_flags_add_wN` / `vm_flags_sub_wN`: whether it subtracts, and its width.
fn status_flags_intrinsic(name: &str) -> Option<(bool, BitWidth)> {
    let (subtract, rest) = if let Some(rest) = name.strip_prefix("vm_flags_add_") {
        (false, rest)
    } else {
        (true, name.strip_prefix("vm_flags_sub_")?)
    };
    let width = match rest {
        "w8" => BitWidth::W8,
        "w16" => BitWidth::W16,
        "w32" => BitWidth::W32,
        "w64" => BitWidth::W64,
        _ => return None,
    };
    Some((subtract, width))
}

/// `vm_flags_nor_wN` / `vm_flags_nand_wN`: whether the native op is `or` (the
/// NAND form, `(~a) | (~b)`) rather than `and` (the NOR form, `(~a) & (~b)`),
/// and its width.
fn logical_flags_intrinsic(name: &str) -> Option<(bool, BitWidth)> {
    let (or, rest) = if let Some(rest) = name.strip_prefix("vm_flags_nand_") {
        (true, rest)
    } else {
        (false, name.strip_prefix("vm_flags_nor_")?)
    };
    let width = match rest {
        "w8" => BitWidth::W8,
        "w16" => BitWidth::W16,
        "w32" => BitWidth::W32,
        "w64" => BitWidth::W64,
        _ => return None,
    };
    Some((or, width))
}

/// `x86_signed_multiply_wN_K` / `x86_unsigned_multiply_wN_K`: signedness and
/// operand width of a wide multiply. Only widths up to 32 bits are named this
/// way; the 64-bit form splits its result into `_low`/`_high` intrinsics.
fn wide_multiply_intrinsic(name: &str) -> Option<(bool, BitWidth)> {
    let (signed, rest) = if let Some(rest) = name.strip_prefix("x86_signed_multiply_w") {
        (true, rest)
    } else {
        (false, name.strip_prefix("x86_unsigned_multiply_w")?)
    };
    let width = match rest.split('_').next()? {
        "8" => BitWidth::W8,
        "16" => BitWidth::W16,
        "32" => BitWidth::W32,
        _ => return None,
    };
    Some((signed, width))
}

fn width_from_digits(text: &str) -> Option<BitWidth> {
    match text {
        "8" => Some(BitWidth::W8),
        "16" => Some(BitWidth::W16),
        "32" => Some(BitWidth::W32),
        "64" => Some(BitWidth::W64),
        _ => None,
    }
}

fn is_index(text: &str) -> bool {
    !text.is_empty() && text.bytes().all(|byte| byte.is_ascii_digit())
}

/// `vm_flags_signed_multiply_wN_K` / `vm_flags_unsigned_multiply_wN_K`:
/// signedness and operand width of the flag word a VM multiply leaves. Those
/// flags have no concrete VM model (`evaluate_vm_flags` folds only add/sub/
/// logical), so the only faithful lowering re-emits the real `mul`/`imul` and
/// reads the raw flags, exactly as the double-shift flag passthrough does.
fn multiply_flags_intrinsic(name: &str) -> Option<(bool, BitWidth)> {
    let (signed, rest) = if let Some(rest) = name.strip_prefix("vm_flags_signed_multiply_w") {
        (true, rest)
    } else {
        (false, name.strip_prefix("vm_flags_unsigned_multiply_w")?)
    };
    let mut parts = rest.split('_');
    let width = width_from_digits(parts.next()?)?;
    (is_index(parts.next()?) && parts.next().is_none()).then_some((signed, width))
}

/// Which half of a wide unsigned-divide value a `x86_unsigned_divide_*`
/// intrinsic names.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DividePart {
    /// Widths up to 32 pack `(remainder << width) | quotient` into one qword.
    Packed,
    /// The 64-bit form splits into independent `_quotient`/`_remainder` qwords.
    Quotient,
    Remainder,
}

/// `x86_unsigned_divide_wN_K` (packed, N<=32) or `x86_unsigned_divide_w64_K_{quotient,remainder}`.
fn unsigned_divide_value_intrinsic(name: &str) -> Option<(BitWidth, DividePart)> {
    let rest = name.strip_prefix("x86_unsigned_divide_w")?;
    let mut parts = rest.split('_');
    let width = width_from_digits(parts.next()?)?;
    if !is_index(parts.next()?) {
        return None;
    }
    match parts.next() {
        None => Some((width, DividePart::Packed)),
        Some("quotient") if parts.next().is_none() => Some((width, DividePart::Quotient)),
        Some("remainder") if parts.next().is_none() => Some((width, DividePart::Remainder)),
        _ => None,
    }
}

/// `vm_flags_unsigned_divide_wN_K`: operand width of the flag word a VM divide
/// leaves. Undefined by the ISA and unmodelled by the VM, so reproduced by
/// re-emitting the real `div` and reading the raw flags.
fn unsigned_divide_flags_intrinsic(name: &str) -> Option<BitWidth> {
    let rest = name.strip_prefix("vm_flags_unsigned_divide_w")?;
    let mut parts = rest.split('_');
    let width = width_from_digits(parts.next()?)?;
    (is_index(parts.next()?) && parts.next().is_none()).then_some(width)
}

fn native_width(width: BitWidth) -> Width {
    match width {
        BitWidth::W8 => Width::W8,
        BitWidth::W16 => Width::W16,
        BitWidth::W32 => Width::W32,
        BitWidth::W64 => Width::W64,
    }
}

fn register(id: u32, width: BitWidth) -> VirtualRegister {
    VirtualRegister::new(id, native_width(width))
}

fn accumulator(width: BitWidth) -> VirtualRegister {
    register(ACCUMULATOR_ID, width)
}

fn temporary(width: BitWidth) -> VirtualRegister {
    register(TEMPORARY_ID, width)
}

fn slot(index: usize, limit: usize) -> Result<SpillSlot, ExpressionCodegenError> {
    if index >= limit || index >= BACKEND_SPILL_LIMIT {
        return Err(ExpressionCodegenError::SpillLimit {
            required: index.saturating_add(1),
            limit: limit.min(BACKEND_SPILL_LIMIT),
        });
    }
    u16::try_from(index)
        .map(SpillSlot)
        .map_err(|_| ExpressionCodegenError::SpillLimit {
            required: index.saturating_add(1),
            limit: limit.min(BACKEND_SPILL_LIMIT),
        })
}

impl LoweringContext<'_> {
    fn allocate_slot(&mut self) -> Result<SpillSlot, ExpressionCodegenError> {
        let allocated = slot(self.next_slot, self.absolute_slot_limit)?;
        self.next_slot = self.next_slot.saturating_add(1);
        self.peak_slot = self.peak_slot.max(self.next_slot);
        Ok(allocated)
    }

    fn release_slot(&mut self) {
        self.next_slot = self
            .next_slot
            .checked_sub(1)
            .expect("every temporary spill release follows allocation");
    }

    fn lower(&mut self, expression: &Expr) -> Result<(), ExpressionCodegenError> {
        match &expression.kind {
            ExprKind::Const { value } => self.operations.push(Operation::MoveImmediate {
                destination: accumulator(expression.width),
                value: *value,
            }),
            ExprKind::Register { name } => {
                let binding =
                    self.inputs.get(name).copied().ok_or_else(|| {
                        ExpressionCodegenError::MissingInput { name: name.clone() }
                    })?;
                if binding.width != expression.width {
                    return Err(ExpressionCodegenError::InputWidth {
                        name: name.clone(),
                        expected: binding.width,
                        actual: expression.width,
                    });
                }
                self.operations.push(Operation::SpillLoad {
                    destination: accumulator(expression.width),
                    slot: binding.slot,
                });
            }
            ExprKind::Unary { op, value } => {
                self.lower(value)?;
                self.operations.push(Operation::Unary {
                    operation: match op {
                        UnaryOp::Not => NativeUnaryOp::Not,
                        UnaryOp::Neg => NativeUnaryOp::Negate,
                    },
                    destination: accumulator(expression.width),
                });
            }
            ExprKind::Binary { op, left, right } => self.lower_binary(*op, left, right)?,
            ExprKind::Load { address } => {
                if address.width != BitWidth::W64 {
                    return Err(ExpressionCodegenError::AddressWidth {
                        width: address.width,
                    });
                }
                self.lower(address)?;
                self.operations.push(Operation::Load {
                    destination: accumulator(expression.width),
                    address: Address::new(accumulator(BitWidth::W64), 0),
                });
            }
            ExprKind::Intrinsic { name, arguments } => {
                if let Some(call) = self.config.import_calls.get(name).cloned() {
                    self.lower_import_call(expression, &call, arguments)?;
                } else if let Some(kind) = passthrough::classify(name) {
                    self.lower_passthrough(name, kind, expression, arguments)?;
                    self.passthroughs.insert(name.clone());
                } else if let Some((subtract, width)) = status_flags_intrinsic(name) {
                    self.lower_status_flags(name, subtract, width, expression, arguments)?;
                } else if let Some((or, width)) = logical_flags_intrinsic(name) {
                    self.lower_logical_flags(name, or, width, expression, arguments)?;
                } else if let Some((signed, width)) = wide_multiply_intrinsic(name) {
                    self.lower_wide_multiply(name, signed, width, expression, arguments)?;
                } else if let Some((signed, width)) = multiply_flags_intrinsic(name) {
                    self.lower_multiply_flags(name, signed, width, expression, arguments)?;
                } else if let Some((width, part)) = unsigned_divide_value_intrinsic(name) {
                    self.lower_unsigned_divide_value(name, width, part, expression, arguments)?;
                } else if let Some(width) = unsigned_divide_flags_intrinsic(name) {
                    self.lower_unsigned_divide_flags(name, width, expression, arguments)?;
                } else {
                    return Err(ExpressionCodegenError::UnsupportedIntrinsic {
                        name: name.clone(),
                    });
                }
            }
            ExprKind::Extract { value, lsb } => {
                self.lower(value)?;
                if *lsb != 0 {
                    self.operations.push(Operation::Shift {
                        operation: ShiftOp::RightLogical,
                        destination: accumulator(value.width),
                        count: ShiftCount::Immediate(
                            u8::try_from(*lsb)
                                .map_err(|_| ExpressionCodegenError::StorageIdOverflow)?,
                        ),
                    });
                }
                if expression.width != value.width {
                    self.operations.push(Operation::Resize {
                        operation: NativeResizeOp::Truncate,
                        destination: accumulator(expression.width),
                        source: accumulator(value.width),
                    });
                }
            }
            ExprKind::Concat { high, low } => self.lower_concat(expression.width, high, low)?,
            ExprKind::Resize { op, value } => {
                self.lower(value)?;
                self.operations.push(Operation::Resize {
                    operation: match op {
                        ResizeOp::Truncate => NativeResizeOp::Truncate,
                        ResizeOp::ZeroExtend => NativeResizeOp::ZeroExtend,
                        ResizeOp::SignExtend => NativeResizeOp::SignExtend,
                    },
                    destination: accumulator(expression.width),
                    source: accumulator(value.width),
                });
            }
        }
        Ok(())
    }

    fn lower_binary(
        &mut self,
        operation: BinaryOp,
        left: &Expr,
        right: &Expr,
    ) -> Result<(), ExpressionCodegenError> {
        if let Some(shift) = match operation {
            BinaryOp::Shl => Some(ShiftOp::Left),
            BinaryOp::Shr => Some(ShiftOp::RightLogical),
            BinaryOp::Sar => Some(ShiftOp::RightArithmetic),
            BinaryOp::Rol => Some(ShiftOp::RotateLeft),
            BinaryOp::Ror => Some(ShiftOp::RotateRight),
            _ => None,
        } {
            self.lower(left)?;
            if let ExprKind::Const { value } = &right.kind {
                self.operations.push(Operation::Shift {
                    operation: shift,
                    destination: accumulator(left.width),
                    count: ShiftCount::Immediate(
                        u8::try_from(*value & u64::from(u8::MAX))
                            .map_err(|_| ExpressionCodegenError::StorageIdOverflow)?,
                    ),
                });
                return Ok(());
            }
            let left_slot = self.allocate_slot()?;
            self.operations.push(Operation::SpillStore {
                slot: left_slot,
                source: accumulator(left.width),
            });
            self.lower(right)?;
            self.operations.push(Operation::Move {
                destination: register(SHIFT_COUNT_ID, BitWidth::W8),
                source: accumulator(BitWidth::W8),
            });
            self.operations.push(Operation::SpillLoad {
                destination: accumulator(left.width),
                slot: left_slot,
            });
            self.release_slot();
            self.operations.push(Operation::Shift {
                operation: shift,
                destination: accumulator(left.width),
                count: ShiftCount::Register(register(SHIFT_COUNT_ID, BitWidth::W8)),
            });
            return Ok(());
        }

        self.lower(left)?;
        let left_slot = self.allocate_slot()?;
        self.operations.push(Operation::SpillStore {
            slot: left_slot,
            source: accumulator(left.width),
        });
        self.lower(right)?;
        self.operations.push(Operation::Move {
            destination: temporary(right.width),
            source: accumulator(right.width),
        });
        self.operations.push(Operation::SpillLoad {
            destination: accumulator(left.width),
            slot: left_slot,
        });
        self.release_slot();
        self.operations.push(Operation::Binary {
            operation: match operation {
                BinaryOp::Add => NativeBinaryOp::Add,
                BinaryOp::Sub => NativeBinaryOp::Subtract,
                BinaryOp::Mul => NativeBinaryOp::Multiply,
                BinaryOp::Xor => NativeBinaryOp::Xor,
                BinaryOp::And => NativeBinaryOp::And,
                BinaryOp::Or => NativeBinaryOp::Or,
                BinaryOp::Shl | BinaryOp::Shr | BinaryOp::Sar | BinaryOp::Rol | BinaryOp::Ror => {
                    unreachable!("shift operations returned above")
                }
            },
            destination: accumulator(left.width),
            source: BinarySource::Register(temporary(right.width)),
        });
        Ok(())
    }

    fn lower_concat(
        &mut self,
        result_width: BitWidth,
        high: &Expr,
        low: &Expr,
    ) -> Result<(), ExpressionCodegenError> {
        self.lower(low)?;
        if low.width != result_width {
            self.operations.push(Operation::Resize {
                operation: NativeResizeOp::ZeroExtend,
                destination: accumulator(result_width),
                source: accumulator(low.width),
            });
        }
        let low_slot = self.allocate_slot()?;
        self.operations.push(Operation::SpillStore {
            slot: low_slot,
            source: accumulator(result_width),
        });
        self.lower(high)?;
        if high.width != result_width {
            self.operations.push(Operation::Resize {
                operation: NativeResizeOp::ZeroExtend,
                destination: accumulator(result_width),
                source: accumulator(high.width),
            });
        }
        self.operations.push(Operation::Shift {
            operation: ShiftOp::Left,
            destination: accumulator(result_width),
            count: ShiftCount::Immediate(
                u8::try_from(low.width.bits())
                    .map_err(|_| ExpressionCodegenError::StorageIdOverflow)?,
            ),
        });
        self.operations.push(Operation::Move {
            destination: temporary(result_width),
            source: accumulator(result_width),
        });
        self.operations.push(Operation::SpillLoad {
            destination: accumulator(result_width),
            slot: low_slot,
        });
        self.release_slot();
        self.operations.push(Operation::Binary {
            operation: NativeBinaryOp::Or,
            destination: accumulator(result_width),
            source: BinarySource::Register(temporary(result_width)),
        });
        Ok(())
    }
}

fn validate_expression_limits(
    expression: &Expr,
    config: &ExpressionCodegenConfig,
) -> Result<usize, ExpressionCodegenError> {
    let nodes = expression.node_count_bounded(config.max_expression_nodes);
    if nodes > config.max_expression_nodes {
        return Err(ExpressionCodegenError::NodeLimit {
            nodes,
            limit: config.max_expression_nodes,
        });
    }
    let scratches = [
        config.result_register,
        config.temporary_register,
        config.shift_count_register,
    ];
    if scratches.into_iter().collect::<BTreeSet<_>>().len() != scratches.len() {
        return Err(ExpressionCodegenError::ScratchRegisterCollision);
    }
    if config.shift_count_register != PhysicalRegister::Rcx {
        return Err(ExpressionCodegenError::ShiftCountRegister);
    }
    Ok(nodes)
}

/// Lowers one width-aware symbolic expression into the project-owned x64 IR.
/// Every native input is captured in a private spill before scratch registers
/// are clobbered; recursive temporaries use bounded, width-checked spill slots.
/// Intrinsics remain hard failures.
///
/// # Errors
///
/// Rejects missing or inconsistent input mappings, unsupported intrinsics,
/// invalid scratch assignments, and resource-limit exhaustion.
pub fn lower_expression(
    expression: &Expr,
    config: &ExpressionCodegenConfig,
) -> Result<LoweredExpression, ExpressionCodegenError> {
    let nodes = validate_expression_limits(expression, config)?;

    let mut physical_ids = BTreeMap::from([
        (config.result_register, ACCUMULATOR_ID),
        (config.temporary_register, TEMPORARY_ID),
        (config.shift_count_register, SHIFT_COUNT_ID),
    ]);
    let mut assignments = vec![
        RegisterAssignment::new(
            VirtualRegister::qword(ACCUMULATOR_ID),
            config.result_register,
        ),
        RegisterAssignment::new(
            VirtualRegister::qword(TEMPORARY_ID),
            config.temporary_register,
        ),
        RegisterAssignment::new(
            VirtualRegister::qword(SHIFT_COUNT_ID),
            config.shift_count_register,
        ),
    ];
    let mut inputs = BTreeMap::new();
    let mut operations = Vec::new();
    for (input_index, input) in config.inputs.iter().enumerate() {
        if inputs.contains_key(&input.name) {
            return Err(ExpressionCodegenError::DuplicateInput {
                name: input.name.clone(),
            });
        }
        match input.source {
            InputSource::Register(physical) => {
                let register_id = if let Some(existing) = physical_ids.get(&physical) {
                    *existing
                } else {
                    let offset = u32::try_from(input_index)
                        .map_err(|_| ExpressionCodegenError::StorageIdOverflow)?;
                    let id = FIRST_INPUT_ID
                        .checked_add(offset)
                        .ok_or(ExpressionCodegenError::StorageIdOverflow)?;
                    physical_ids.insert(physical, id);
                    assignments.push(RegisterAssignment::new(
                        VirtualRegister::qword(id),
                        physical,
                    ));
                    id
                };
                let input_slot = slot(input_index, config.max_spill_slots)?;
                inputs.insert(
                    input.name.clone(),
                    InputBinding {
                        width: input.width,
                        slot: input_slot,
                    },
                );
                operations.push(Operation::SpillStore {
                    slot: input_slot,
                    source: register(register_id, input.width),
                });
            }
            InputSource::Slot(existing) => {
                // The caller already spilled this value, so read it in place and
                // consume no register at entry.
                inputs.insert(
                    input.name.clone(),
                    InputBinding {
                        width: input.width,
                        slot: existing,
                    },
                );
            }
        }
    }

    let first_slot = config
        .first_scratch_slot
        .map_or(config.inputs.len(), |slot| usize::from(slot));
    // `max_spill_slots` counts the slots this fragment may use, so the absolute
    // ceiling is where its window starts plus that count.
    let absolute_slot_limit = first_slot
        .saturating_add(config.max_spill_slots)
        .min(BACKEND_SPILL_LIMIT);
    let mut context = LoweringContext {
        config,
        inputs,
        assignments,
        operations,
        next_slot: first_slot,
        peak_slot: first_slot,
        absolute_slot_limit,
        passthroughs: BTreeSet::new(),
    };
    context.lower(expression)?;
    context.operations.push(Operation::Return);
    // Slot-backed inputs may already live above the peak the fragment reached
    // for itself, so the frame must cover the highest input slot as well.
    let highest_input_slot = config
        .inputs
        .iter()
        .filter_map(|input| match input.source {
            InputSource::Slot(slot) => Some(slot.0),
            InputSource::Register(_) => None,
        })
        .max()
        .map_or(0, |slot| slot.saturating_add(1));
    let highest_slot = context.peak_slot.max(usize::from(highest_input_slot));
    let mut stack_allocation = u32::try_from(highest_slot)
        .ok()
        .and_then(|slots| slots.checked_mul(8))
        .ok_or(ExpressionCodegenError::StorageIdOverflow)?;
    // Any Win64 nonvolatile register this expression writes must be preserved
    // by the emitted frame. The backend enforces that invariant, so a caller
    // selecting a nonvolatile result or temporary register would otherwise get
    // a hard rejection instead of a correct function prologue.
    let mut preserve = Vec::new();
    for assignment in &context.assignments {
        let register = assignment.physical_register;
        if register.is_win64_nonvolatile() && !preserve.contains(&register) {
            preserve.push(register);
        }
    }
    let has_call = context
        .operations
        .iter()
        .any(|operation| matches!(operation, Operation::Call { .. }));
    if has_call {
        // A call-ready frame reserves 32 bytes of shadow space below the spill
        // area, so the allocation has to cover that as well as the slots.
        stack_allocation = stack_allocation.saturating_add(32);
        // Win64 requires 16-byte RSP alignment at a call site. The caller's
        // return address leaves RSP congruent to 8, so the preserved-register
        // pushes plus this allocation must sum to 8 modulo 16.
        let pushes = u32::try_from(preserve.len())
            .map_err(|_| ExpressionCodegenError::StorageIdOverflow)?
            .saturating_mul(8);
        let total = pushes.saturating_add(stack_allocation);
        let residual = (8_u32.wrapping_sub(total)) & 0xF;
        if residual != 0 {
            stack_allocation = stack_allocation.saturating_add(residual);
        }
    }
    let function = Function {
        operations: context.operations,
        assignments: context.assignments,
        frame: StackFrame {
            stack_allocation,
            preserve,
            // Slot-backed inputs are already populated by the caller, so a load
            // from one is defined even though this body contains no store. Every
            // slot below the highest input slot must be declared, because the
            // caller's frame numbers its regions contiguously.
            preinitialized_slots: (0..highest_input_slot).map(SpillSlot).collect(),
            // An expression that lowers an imported call needs a Win64
            // call-ready frame: the backend checks shadow space and alignment.
            call_ready: has_call,
            // A fragment is inlined into its caller, so it returns normally.
            tail_transfer: false,
        },
    };
    validate_function(&function)?;
    Ok(LoweredExpression {
        function,
        result: accumulator(expression.width),
        expression_nodes: nodes,
        spill_slots: context.peak_slot,
        environment_boundaries: context
            .passthroughs
            .iter()
            .filter(|name| passthrough::is_environment_boundary(name))
            .cloned()
            .collect(),
        passthroughs: context.passthroughs,
    })
}

/// Lowers and immediately compiles one expression at the requested VA.
///
/// # Errors
///
/// Returns either a lowering error or a structured backend validation/encoding
/// error; unsupported semantics never produce bytes.
pub fn compile_expression(
    expression: &Expr,
    config: &ExpressionCodegenConfig,
    address: CodeAddress,
) -> Result<CompiledFunction, ExpressionCodegenError> {
    let lowered = lower_expression(expression, config)?;
    Ok(compile(&lowered.function, address)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(name: &str, width: BitWidth, physical_register: PhysicalRegister) -> ExpressionInput {
        ExpressionInput::register(name, width, physical_register)
    }

    #[test]
    fn compiles_nested_width_aware_expression_with_captured_inputs() {
        let left = Expr::register(BitWidth::W32, "left");
        let right = Expr::register(BitWidth::W32, "right");
        let sum = Expr::binary(BinaryOp::Add, left, right).expect("same-width addition");
        let rotated = Expr::binary(BinaryOp::Rol, sum, Expr::constant(BitWidth::W32, 5))
            .expect("same-width rotate");
        let expression = Expr::resize(ResizeOp::ZeroExtend, BitWidth::W64, rotated)
            .expect("valid zero extension");
        let config = ExpressionCodegenConfig {
            inputs: vec![
                input("left", BitWidth::W32, PhysicalRegister::Rdx),
                input("right", BitWidth::W32, PhysicalRegister::R8),
            ],
            ..ExpressionCodegenConfig::default()
        };
        let lowered = lower_expression(&expression, &config).expect("lower expression");
        assert_eq!(lowered.result.width, Width::W64);
        assert!(lowered.spill_slots >= 3);
        let compiled = compile(&lowered.function, CodeAddress(0x1_4000_1000))
            .expect("compile lowered expression");
        assert!(!compiled.bytes.is_empty());
    }

    #[test]
    fn supports_load_extract_concat_and_variable_shift() {
        let pointer = Expr::register(BitWidth::W64, "pointer");
        let loaded = Expr::load(BitWidth::W64, pointer);
        let low = Expr::extract(BitWidth::W16, 8, loaded.clone()).expect("extract low word");
        let high = Expr::extract(BitWidth::W16, 40, loaded).expect("extract high word");
        let concat = Expr::concat(high, low).expect("concat words");
        let count = Expr::register(BitWidth::W32, "count");
        let expression = Expr::binary(BinaryOp::Shr, concat, count).expect("variable shift");
        let config = ExpressionCodegenConfig {
            inputs: vec![
                input("pointer", BitWidth::W64, PhysicalRegister::Rdx),
                input("count", BitWidth::W32, PhysicalRegister::R9),
            ],
            ..ExpressionCodegenConfig::default()
        };
        compile_expression(&expression, &config, CodeAddress(0))
            .expect("compile complex expression");
    }

    fn decoded(bytes: &[u8]) -> Vec<iced_x86::Instruction> {
        let mut decoder = iced_x86::Decoder::with_ip(64, bytes, 0, iced_x86::DecoderOptions::NONE);
        let mut out = Vec::new();
        while decoder.can_decode() {
            out.push(decoder.decode());
        }
        out
    }

    fn passthrough_config() -> ExpressionCodegenConfig {
        ExpressionCodegenConfig {
            inputs: vec![
                input("a", BitWidth::W32, PhysicalRegister::Rdx),
                input("b", BitWidth::W32, PhysicalRegister::R8),
                input("p", BitWidth::W64, PhysicalRegister::R9),
                input("n", BitWidth::W8, PhysicalRegister::Rsi),
            ],
            ..ExpressionCodegenConfig::default()
        }
    }

    fn lowers_to(expression: &Expr, mnemonic: iced_x86::Mnemonic) -> LoweredExpression {
        let config = passthrough_config();
        let lowered = lower_expression(expression, &config).expect("lower passthrough");
        let compiled = compile(&lowered.function, CodeAddress(0)).expect("compile passthrough");
        let instructions = decoded(&compiled.bytes);
        assert_eq!(
            instructions
                .iter()
                .filter(|instruction| instruction.mnemonic() == mnemonic)
                .count(),
            1,
            "{mnemonic:?} is emitted exactly once"
        );
        lowered
    }

    #[test]
    fn cpuid_lowers_to_the_instruction_and_marks_an_environment_boundary() {
        let leaf = Expr::register(BitWidth::W32, "a");
        let subleaf = Expr::register(BitWidth::W32, "b");
        let expression = Expr::intrinsic(BitWidth::W32, "x86_cpuid_ecx_7", vec![leaf, subleaf]);
        let lowered = lowers_to(&expression, iced_x86::Mnemonic::Cpuid);
        assert!(lowered.environment_boundaries.contains("x86_cpuid_ecx_7"));
    }

    #[test]
    fn gs_load_lowers_to_a_gs_prefixed_move() {
        let expression = Expr::intrinsic(
            BitWidth::W64,
            "x86_gs_load_2",
            vec![Expr::register(BitWidth::W64, "p")],
        );
        let config = passthrough_config();
        let lowered = lower_expression(&expression, &config).expect("lower gs load");
        let compiled = compile(&lowered.function, CodeAddress(0)).expect("compile gs load");
        assert!(
            decoded(&compiled.bytes)
                .iter()
                .any(|instruction| instruction.segment_prefix() == iced_x86::Register::GS)
        );
        assert!(lowered.environment_boundaries.contains("x86_gs_load_2"));
    }

    #[test]
    fn double_shifts_and_their_flags_lower_to_shld_and_shrd() {
        let arguments = vec![
            Expr::register(BitWidth::W32, "a"),
            Expr::register(BitWidth::W32, "b"),
            Expr::register(BitWidth::W8, "n"),
        ];
        for (direction, mnemonic) in [
            ("left", iced_x86::Mnemonic::Shld),
            ("right", iced_x86::Mnemonic::Shrd),
        ] {
            let value = Expr::intrinsic(
                BitWidth::W32,
                format!("x86_double_shift_{direction}_w32_1"),
                arguments.clone(),
            );
            let lowered = lowers_to(&value, mnemonic);
            assert!(lowered.environment_boundaries.is_empty());
            let flags = Expr::intrinsic(
                BitWidth::W64,
                format!("vm_flags_double_shift_{direction}_w32_1"),
                arguments.clone(),
            );
            let lowered = lowers_to(&flags, mnemonic);
            let compiled = compile(&lowered.function, CodeAddress(0)).expect("compile flags");
            assert!(
                decoded(&compiled.bytes)
                    .iter()
                    .any(|instruction| instruction.mnemonic() == iced_x86::Mnemonic::Pushfq)
            );
        }
    }

    #[test]
    fn wide_multiply_lowers_to_an_extend_and_sixty_four_bit_imul() {
        // The packed double-width product is the low 64 bits of a 64-bit
        // multiply of the widened operands: sign-extended (movsxd) for the
        // signed form, zero-extended (32-bit mov) for the unsigned form.
        let operands = || {
            vec![
                Expr::register(BitWidth::W32, "a"),
                Expr::register(BitWidth::W32, "b"),
            ]
        };
        let signed = Expr::intrinsic(BitWidth::W64, "x86_signed_multiply_w32_1", operands());
        let lowered = lowers_to(&signed, iced_x86::Mnemonic::Imul);
        assert_eq!(lowered.result, VirtualRegister::qword(ACCUMULATOR_ID));
        let compiled = compile(&lowered.function, CodeAddress(0)).expect("compile signed multiply");
        assert!(
            decoded(&compiled.bytes)
                .iter()
                .any(|instruction| instruction.mnemonic() == iced_x86::Mnemonic::Movsxd),
            "signed wide multiply sign-extends its operands"
        );

        let unsigned = Expr::intrinsic(BitWidth::W64, "x86_unsigned_multiply_w32_2", operands());
        let lowered = lowers_to(&unsigned, iced_x86::Mnemonic::Imul);
        let compiled =
            compile(&lowered.function, CodeAddress(0)).expect("compile unsigned multiply");
        assert!(
            !decoded(&compiled.bytes)
                .iter()
                .any(|instruction| instruction.mnemonic() == iced_x86::Mnemonic::Movsxd),
            "unsigned wide multiply zero-extends rather than sign-extends"
        );
    }

    fn mnemonic_count(bytes: &[u8], mnemonic: iced_x86::Mnemonic) -> usize {
        decoded(bytes)
            .iter()
            .filter(|instruction| instruction.mnemonic() == mnemonic)
            .count()
    }

    #[test]
    fn multiply_flags_lower_to_the_real_multiply_and_a_raw_flag_read() {
        // A VM multiply-flag word has no concrete model, so it is reproduced by
        // re-emitting the exact `mul`/`imul` and reading its raw flags: `mul` for
        // the unsigned form, `imul` for the signed, each with a `pushfq`. The raw
        // word is not masked (unlike add/sub/logical flags).
        let operands = || {
            vec![
                Expr::register(BitWidth::W32, "a"),
                Expr::register(BitWidth::W32, "b"),
            ]
        };
        for (name, mnemonic) in [
            ("vm_flags_unsigned_multiply_w32_7", iced_x86::Mnemonic::Mul),
            ("vm_flags_signed_multiply_w32_7", iced_x86::Mnemonic::Imul),
        ] {
            let expression = Expr::intrinsic(BitWidth::W64, name, operands());
            let lowered = lowers_to(&expression, mnemonic);
            // A multiply-flag lowering is exact-by-identity, not a trusted
            // passthrough or an environment boundary.
            assert!(lowered.passthroughs.is_empty());
            assert!(lowered.environment_boundaries.is_empty());
            let compiled = compile(&lowered.function, CodeAddress(0)).expect("compile mul flags");
            assert_eq!(
                mnemonic_count(&compiled.bytes, iced_x86::Mnemonic::Pushfq),
                1,
                "the raw flag word is read back once"
            );
        }
    }

    #[test]
    fn unsigned_divide_value_lowers_to_div_and_packs_quotient_and_remainder() {
        // The packed w32 value is `(remainder << 32) | quotient`: one `div`, then
        // a `shl` by the width and an `or` to pack the two halves.
        let expression = Expr::intrinsic(
            BitWidth::W64,
            "x86_unsigned_divide_w32_5",
            vec![
                Expr::register(BitWidth::W32, "a"),
                Expr::register(BitWidth::W32, "b"),
                Expr::constant(BitWidth::W32, 7),
            ],
        );
        let lowered = lowers_to(&expression, iced_x86::Mnemonic::Div);
        assert_eq!(lowered.result, VirtualRegister::qword(ACCUMULATOR_ID));
        assert!(lowered.passthroughs.is_empty());
        let compiled = compile(&lowered.function, CodeAddress(0)).expect("compile divide value");
        assert_eq!(
            mnemonic_count(&compiled.bytes, iced_x86::Mnemonic::Shl),
            1,
            "the remainder is shifted into the high half"
        );
    }

    #[test]
    fn unsigned_divide_sixty_four_bit_halves_each_lower_to_a_div() {
        for part in ["quotient", "remainder"] {
            let expression = Expr::intrinsic(
                BitWidth::W64,
                format!("x86_unsigned_divide_w64_5_{part}"),
                vec![
                    Expr::register(BitWidth::W64, "p"),
                    Expr::constant(BitWidth::W64, 1),
                    Expr::constant(BitWidth::W64, 3),
                ],
            );
            lowers_to(&expression, iced_x86::Mnemonic::Div);
        }
    }

    #[test]
    fn unsigned_divide_flags_lower_to_div_and_a_raw_flag_read() {
        let expression = Expr::intrinsic(
            BitWidth::W64,
            "vm_flags_unsigned_divide_w32_5",
            vec![
                Expr::register(BitWidth::W32, "a"),
                Expr::register(BitWidth::W32, "b"),
                Expr::constant(BitWidth::W32, 7),
            ],
        );
        let lowered = lowers_to(&expression, iced_x86::Mnemonic::Div);
        assert!(lowered.passthroughs.is_empty());
        let compiled = compile(&lowered.function, CodeAddress(0)).expect("compile divide flags");
        assert_eq!(
            mnemonic_count(&compiled.bytes, iced_x86::Mnemonic::Pushfq),
            1,
            "the raw flag word is read back once"
        );
    }

    #[test]
    fn multiply_flags_and_divide_fail_closed_on_bad_shapes() {
        let config = passthrough_config();
        // Wrong arity.
        let wrong_arity = Expr::intrinsic(
            BitWidth::W64,
            "vm_flags_signed_multiply_w32_1",
            vec![Expr::register(BitWidth::W32, "a")],
        );
        assert!(matches!(
            lower_expression(&wrong_arity, &config),
            Err(ExpressionCodegenError::MalformedPassthrough { .. })
        ));
        // Operand-width disagreement.
        let wrong_width = Expr::intrinsic(
            BitWidth::W64,
            "vm_flags_unsigned_multiply_w32_1",
            vec![
                Expr::register(BitWidth::W32, "a"),
                Expr::register(BitWidth::W64, "p"),
            ],
        );
        assert!(matches!(
            lower_expression(&wrong_width, &config),
            Err(ExpressionCodegenError::MalformedPassthrough { .. })
        ));
        // The byte DIV leaves its remainder in AH and has no sound lowering here.
        let byte_divide = Expr::intrinsic(
            BitWidth::W64,
            "x86_unsigned_divide_w8_1",
            vec![
                Expr::register(BitWidth::W8, "n"),
                Expr::register(BitWidth::W8, "n"),
                Expr::register(BitWidth::W8, "n"),
            ],
        );
        assert!(matches!(
            lower_expression(&byte_divide, &config),
            Err(ExpressionCodegenError::MalformedPassthrough { .. })
        ));
    }

    #[test]
    fn multiply_and_divide_intrinsic_names_are_classified() {
        assert_eq!(
            multiply_flags_intrinsic("vm_flags_signed_multiply_w16_3"),
            Some((true, BitWidth::W16))
        );
        assert_eq!(
            multiply_flags_intrinsic("vm_flags_unsigned_multiply_w64_9"),
            Some((false, BitWidth::W64))
        );
        assert_eq!(multiply_flags_intrinsic("vm_flags_add_w32"), None);
        assert!(matches!(
            unsigned_divide_value_intrinsic("x86_unsigned_divide_w32_5"),
            Some((BitWidth::W32, DividePart::Packed))
        ));
        assert!(matches!(
            unsigned_divide_value_intrinsic("x86_unsigned_divide_w64_5_remainder"),
            Some((BitWidth::W64, DividePart::Remainder))
        ));
        assert_eq!(
            unsigned_divide_flags_intrinsic("vm_flags_unsigned_divide_w32_5"),
            Some(BitWidth::W32)
        );
        assert_eq!(unsigned_divide_value_intrinsic("x86_unsigned_divide_wx_5"), None);
    }

    /// Counts register-to-register `and`/`or` instructions, i.e. the
    /// flag-defining logical op, excluding the immediate-masked tail.
    fn register_logical_count(bytes: &[u8], mnemonic: iced_x86::Mnemonic) -> usize {
        decoded(bytes)
            .iter()
            .filter(|instruction| {
                instruction.mnemonic() == mnemonic
                    && instruction.op_count() == 2
                    && instruction.op1_kind() == iced_x86::OpKind::Register
            })
            .count()
    }

    #[test]
    fn logical_flags_lower_to_the_defining_instruction_and_a_flag_read() {
        // `vm_flags_nor_wN(a, b)` is the flag word of the native `and` of the
        // two complemented operands (`(~a) & (~b)`); `vm_flags_nand_wN` the
        // flag word of the matching `or`. The lowering must emit two `not`s,
        // the real `and`/`or` that sets the flags, and a `pushfq` flag read.
        let operands = || {
            vec![
                Expr::register(BitWidth::W32, "a"),
                Expr::register(BitWidth::W32, "b"),
            ]
        };
        for (name, logical, is_or) in [
            ("vm_flags_nor_w32", iced_x86::Mnemonic::And, false),
            ("vm_flags_nand_w32", iced_x86::Mnemonic::Or, true),
        ] {
            let expression = Expr::intrinsic(BitWidth::W64, name, operands());
            let config = passthrough_config();
            let lowered = lower_expression(&expression, &config).expect("lower logical flags");
            assert!(lowered.passthroughs.is_empty());
            assert!(lowered.environment_boundaries.is_empty());
            let compiled = compile(&lowered.function, CodeAddress(0)).expect("compile flags");
            assert_eq!(
                mnemonic_count(&compiled.bytes, iced_x86::Mnemonic::Not),
                2,
                "both operands are complemented"
            );
            assert_eq!(
                register_logical_count(&compiled.bytes, logical),
                1,
                "the defining register {logical:?} is emitted once"
            );
            // The complementary logical op must NOT appear as the defining op:
            // a nor is an `and` of the operands, never an `or`, and vice versa.
            // (Both lowerings share the immediate-masked tail, so the tail's
            // `and`/`or` with immediates are excluded by construction.)
            let other = if is_or {
                iced_x86::Mnemonic::And
            } else {
                iced_x86::Mnemonic::Or
            };
            assert_eq!(
                register_logical_count(&compiled.bytes, other),
                0,
                "no stray defining {other:?}"
            );
            assert_eq!(
                mnemonic_count(&compiled.bytes, iced_x86::Mnemonic::Pushfq),
                1,
                "the flag word is read back once"
            );
        }
    }

    #[test]
    fn logical_flags_refuse_wrong_arity_or_widths() {
        let config = passthrough_config();
        let wrong_arity = Expr::intrinsic(
            BitWidth::W64,
            "vm_flags_nor_w32",
            vec![Expr::register(BitWidth::W32, "a")],
        );
        assert!(matches!(
            lower_expression(&wrong_arity, &config),
            Err(ExpressionCodegenError::MalformedPassthrough { .. })
        ));
        let wrong_width = Expr::intrinsic(
            BitWidth::W64,
            "vm_flags_nor_w32",
            vec![
                Expr::register(BitWidth::W32, "a"),
                Expr::register(BitWidth::W64, "p"),
            ],
        );
        assert!(matches!(
            lower_expression(&wrong_width, &config),
            Err(ExpressionCodegenError::MalformedPassthrough { .. })
        ));
    }

    #[test]
    fn shift_flags_fail_closed_rather_than_emit_approximate_bits() {
        // VM shift-flag intrinsics (`vm_flags_shiftleft_wN`, etc.) are left
        // unlowered on purpose: a variable shift's CF is the last bit shifted
        // out and its OF is ISA-undefined for counts other than one, so there
        // is no shape we can prove exact here. Fail closed instead of guessing.
        let expression = Expr::intrinsic(
            BitWidth::W64,
            "vm_flags_shiftleft_w32",
            vec![
                Expr::register(BitWidth::W32, "a"),
                Expr::register(BitWidth::W32, "b"),
            ],
        );
        let config = passthrough_config();
        assert!(matches!(
            lower_expression(&expression, &config),
            Err(ExpressionCodegenError::UnsupportedIntrinsic { name }) if name == "vm_flags_shiftleft_w32"
        ));
    }

    #[test]
    fn popfq_lowers_to_a_flag_restore_and_readback() {
        let expression = Expr::intrinsic(
            BitWidth::W64,
            "x86_popfq_5",
            vec![Expr::register(BitWidth::W64, "p")],
        );
        let lowered = lowers_to(&expression, iced_x86::Mnemonic::Popfq);
        assert!(lowered.passthroughs.contains("x86_popfq_5"));
        assert!(lowered.environment_boundaries.is_empty());
    }

    #[test]
    fn passthroughs_refuse_an_unlowerable_operand_or_a_wrong_shape() {
        let config = passthrough_config();
        let unmapped = Expr::register(BitWidth::W64, "not_an_input");
        for expression in [
            Expr::intrinsic(BitWidth::W64, "x86_gs_load_1", vec![unmapped.clone()]),
            Expr::intrinsic(BitWidth::W64, "x86_popfq_1", vec![unmapped]),
            Expr::intrinsic(
                BitWidth::W32,
                "x86_cpuid_eax_1",
                vec![
                    Expr::register(BitWidth::W32, "a"),
                    Expr::register(BitWidth::W32, "missing"),
                ],
            ),
        ] {
            assert!(matches!(
                lower_expression(&expression, &config),
                Err(ExpressionCodegenError::MissingInput { .. })
            ));
        }
        let byte_shift = Expr::intrinsic(
            BitWidth::W8,
            "x86_double_shift_left_w8_1",
            vec![
                Expr::register(BitWidth::W8, "n"),
                Expr::register(BitWidth::W8, "n"),
                Expr::register(BitWidth::W8, "n"),
            ],
        );
        assert!(matches!(
            lower_expression(&byte_shift, &config),
            Err(ExpressionCodegenError::MalformedPassthrough { .. })
        ));
        let wrong_arity = Expr::intrinsic(BitWidth::W32, "x86_cpuid_eax_1", vec![]);
        assert!(matches!(
            lower_expression(&wrong_arity, &config),
            Err(ExpressionCodegenError::MalformedPassthrough { .. })
        ));
    }

    #[test]
    fn rejects_unmapped_inputs_intrinsics_and_resource_exhaustion() {
        let missing = Expr::register(BitWidth::W64, "missing");
        assert!(matches!(
            lower_expression(&missing, &ExpressionCodegenConfig::default()),
            Err(ExpressionCodegenError::MissingInput { .. })
        ));
        let intrinsic = Expr::intrinsic(BitWidth::W64, "opaque", vec![]);
        assert!(matches!(
            lower_expression(&intrinsic, &ExpressionCodegenConfig::default()),
            Err(ExpressionCodegenError::UnsupportedIntrinsic { .. })
        ));
        let constrained = ExpressionCodegenConfig {
            max_expression_nodes: 0,
            ..ExpressionCodegenConfig::default()
        };
        assert!(matches!(
            lower_expression(&Expr::constant(BitWidth::W64, 1), &constrained),
            Err(ExpressionCodegenError::NodeLimit { .. })
        ));
    }
}
