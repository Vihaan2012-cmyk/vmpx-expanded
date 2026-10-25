use std::collections::BTreeMap;

use thiserror::Error;
use vmpx_recompile::{
    BinarySource, CodeAddress, CompileError, CompiledFunction, Condition, Function, Label,
    Operation, PhysicalRegister, RegisterAssignment, StackFrame, VirtualRegister, compile,
};

const INPUT_ID: u32 = 0;
const RESULT_ID: u32 = 1;
const SCRATCH_ID: u32 = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConstantDispatchConfig {
    pub input_register: PhysicalRegister,
    pub scratch_register: PhysicalRegister,
    pub default_result: u64,
}

impl Default for ConstantDispatchConfig {
    fn default() -> Self {
        Self {
            input_register: PhysicalRegister::Rdx,
            scratch_register: PhysicalRegister::R10,
            default_result: 0,
        }
    }
}

#[derive(Debug, Eq, Error, PartialEq)]
pub enum ConstantDispatchError {
    #[error("constant dispatcher input, RAX result, and scratch registers must be distinct")]
    RegisterCollision,
    #[error("constant dispatcher scratch register must be Win64 volatile")]
    NonvolatileScratch,
    #[error("constant dispatcher has too many non-default cases")]
    TooManyCases,
    #[error("native backend rejected constant dispatcher: {0}")]
    Compile(#[from] CompileError),
}

/// Emits a bounded Win64 function that maps one qword register over explicit
/// constant cases and returns the selected qword in `RAX`.
///
/// The caller supplies an explicit result for values outside the proven finite
/// domain. Cases equal to that default are omitted. No VM state or handler
/// mechanism is present in the emitted body.
///
/// # Errors
///
/// Rejects colliding or nonvolatile scratch allocation, label-space overflow,
/// and any native-backend validation or encoding failure.
pub fn compile_constant_dispatch(
    cases: &BTreeMap<u64, u64>,
    config: ConstantDispatchConfig,
    address: CodeAddress,
) -> Result<CompiledFunction, ConstantDispatchError> {
    if config.input_register == PhysicalRegister::Rax
        || config.scratch_register == PhysicalRegister::Rax
        || config.input_register == config.scratch_register
    {
        return Err(ConstantDispatchError::RegisterCollision);
    }
    if config.scratch_register.is_win64_nonvolatile() {
        return Err(ConstantDispatchError::NonvolatileScratch);
    }
    let input = VirtualRegister::qword(INPUT_ID);
    let result = VirtualRegister::qword(RESULT_ID);
    let scratch = VirtualRegister::qword(SCRATCH_ID);
    let emitted_cases = cases
        .iter()
        .filter(|(_, output)| **output != config.default_result)
        .collect::<Vec<_>>();
    if emitted_cases.len() > usize::try_from(u32::MAX).unwrap_or(usize::MAX) {
        return Err(ConstantDispatchError::TooManyCases);
    }
    let mut operations = Vec::with_capacity(emitted_cases.len().saturating_mul(5) + 2);
    for (index, (input_value, _)) in emitted_cases.iter().enumerate() {
        let label = Label(u32::try_from(index).map_err(|_| ConstantDispatchError::TooManyCases)?);
        operations.extend([
            Operation::MoveImmediate {
                destination: scratch,
                value: **input_value,
            },
            Operation::Compare {
                left: input,
                right: BinarySource::Register(scratch),
            },
            Operation::BranchIf {
                condition: Condition::Equal,
                target: label,
            },
        ]);
    }
    operations.extend([
        Operation::MoveImmediate {
            destination: result,
            value: config.default_result,
        },
        Operation::Return,
    ]);
    for (index, (_, output_value)) in emitted_cases.into_iter().enumerate() {
        operations.extend([
            Operation::Label(Label(
                u32::try_from(index).map_err(|_| ConstantDispatchError::TooManyCases)?,
            )),
            Operation::MoveImmediate {
                destination: result,
                value: *output_value,
            },
            Operation::Return,
        ]);
    }
    let function = Function {
        operations,
        assignments: vec![
            RegisterAssignment::new(input, config.input_register),
            RegisterAssignment::new(result, PhysicalRegister::Rax),
            RegisterAssignment::new(scratch, config.scratch_register),
        ],
        frame: StackFrame::default(),
    };
    compile(&function, address).map_err(ConstantDispatchError::from)
}

#[cfg(test)]
mod tests {
    use iced_x86::{Decoder, DecoderOptions, Mnemonic};

    use super::*;

    #[test]
    fn emits_only_non_default_cases_and_native_returns() {
        let compiled = compile_constant_dispatch(
            &BTreeMap::from([(0, 1), (1, 24), (2, 0), (3, 0)]),
            ConstantDispatchConfig::default(),
            CodeAddress(0x1_4000_1000),
        )
        .expect("compile finite dispatcher");
        let instructions =
            Decoder::with_ip(64, &compiled.bytes, 0x1_4000_1000, DecoderOptions::NONE)
                .into_iter()
                .collect::<Vec<_>>();
        assert_eq!(
            instructions
                .iter()
                .filter(|instruction| instruction.mnemonic() == Mnemonic::Ret)
                .count(),
            3
        );
        assert_eq!(
            instructions
                .iter()
                .filter(|instruction| instruction.mnemonic() == Mnemonic::Je)
                .count(),
            2
        );
        assert!(
            instructions
                .iter()
                .all(|instruction| !instruction.is_invalid())
        );
    }

    #[test]
    fn rejects_nonvolatile_scratch_and_register_aliases() {
        let cases = BTreeMap::from([(0, 1)]);
        assert_eq!(
            compile_constant_dispatch(
                &cases,
                ConstantDispatchConfig {
                    scratch_register: PhysicalRegister::Rbx,
                    ..ConstantDispatchConfig::default()
                },
                CodeAddress(0x1000),
            ),
            Err(ConstantDispatchError::NonvolatileScratch)
        );
        assert_eq!(
            compile_constant_dispatch(
                &cases,
                ConstantDispatchConfig {
                    input_register: PhysicalRegister::Rax,
                    ..ConstantDispatchConfig::default()
                },
                CodeAddress(0x1000),
            ),
            Err(ConstantDispatchError::RegisterCollision)
        );
    }
}
