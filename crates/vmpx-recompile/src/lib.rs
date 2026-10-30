//! Conservative AMD64 code generation for lowered VM semantics.
//!
//! This crate deliberately implements a small width-aware integer IR.
//! Unsupported operations and unsafe ABI states are rejected instead of being
//! approximated.

mod abi;
mod compose;
mod dispatch;
mod encode;
mod error;
mod ir;

pub use abi::{RegisterAssignment, StackFrame, validate_function};
pub use compose::{ComposeError, Composition, Fragment, compose};
pub use dispatch::{
    DispatchArm, DispatchError, ValueDispatch, emit_displaced_stub, emit_domain_dispatcher,
    emit_value_dispatcher,
};
pub use encode::{CodeAddress, CompiledFunction, compile};
pub use error::CompileError;
pub use ir::{
    Address, BinaryOp, BinarySource, CallTarget, Condition, Function, Label, NativeAddress,
    Operation, PhysicalRegister, ResizeOp, ShiftCount, ShiftOp, SpillSlot, UnaryOp,
    VirtualRegister, Width,
};
