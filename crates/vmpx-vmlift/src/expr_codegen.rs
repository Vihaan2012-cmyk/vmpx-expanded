use std::collections::{BTreeMap, BTreeSet};

use thiserror::Error;
use vmpx_recompile::{
    Address, BinaryOp as NativeBinaryOp, BinarySource, CallTarget, CodeAddress, CompileError,
    CompiledFunction, Function, NativeAddress, Operation, PhysicalRegister, RegisterAssignment,
    ResizeOp as NativeResizeOp, ShiftCount, ShiftOp, SpillSlot, StackFrame,
    UnaryOp as NativeUnaryOp, VirtualRegister, Width, compile, validate_function,
};
use vmpx_vm::{BinaryOp, BitWidth, Expr, ExprKind, ResizeOp, UnaryOp};

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
}

impl LoweringContext<'_> {
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
                let Some(call) = self.config.import_calls.get(name).cloned() else {
                    return Err(ExpressionCodegenError::UnsupportedIntrinsic {
                        name: name.clone(),
                    });
                };
                self.lower_import_call(expression, &call, arguments)?;
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
