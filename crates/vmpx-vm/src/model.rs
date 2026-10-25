use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;
use vmpx_pe::Rva;

use crate::{BitWidth, Expr, ResizeOp};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BytecodeDirection {
    Ascending,
    Descending,
    StateDependent,
    Unknown,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RecoveryStatus {
    Recovered,
    PartiallyRecovered,
    UnsupportedHandler,
    UnsupportedVmVariant,
    Ambiguous,
    Failed,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VmHandlerSemantic {
    /// VM-internal threaded dispatch with no recovered program-level effect.
    Dispatch,
    Push,
    Pop,
    ConstantPush,
    VirtualRegisterRead,
    VirtualRegisterWrite,
    MemoryLoad,
    MemoryStore,
    /// Recovered byte-at-a-time table checksum over program memory.
    TableChecksum,
    ReadTimestampCounter,
    UnsignedDivide,
    Add,
    Sub,
    Mul,
    Xor,
    And,
    Or,
    Nor,
    Nand,
    Not,
    Neg,
    Shift,
    Rotate,
    ShiftLeft,
    ShiftRightLogical,
    ShiftRightArithmetic,
    RotateLeft,
    RotateRight,
    Compare,
    Flags,
    SignExtend,
    ZeroExtend,
    StackAdjust,
    ConditionalBranch,
    UnconditionalBranch,
    NativeTransition,
    NativeCall,
    VmExit,
    ContextInitialize,
    Unknown,
}

impl VmHandlerSemantic {
    /// VM operations that replace the straight-line decoder state and must be
    /// explored by the virtual-CFG engine instead of the threaded decoder.
    #[must_use]
    pub const fn ends_linear_block(self) -> bool {
        matches!(
            self,
            Self::ConditionalBranch
                | Self::UnconditionalBranch
                | Self::NativeTransition
                | Self::NativeCall
                | Self::VmExit
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EffectKnowledge {
    Preserved,
    Modified,
    Unknown,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct VmContextLayout {
    pub vip_register: String,
    pub vsp_register: Option<String>,
    pub rolling_key_register: String,
    pub handler_register: String,
    pub native_context_register: Option<String>,
    pub context_base_register: String,
    pub allocation_size: u32,
    pub bytecode_direction: BytecodeDirection,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "knowledge", content = "values", rename_all = "snake_case")]
pub enum StateValue<T> {
    Known(T),
    Alternatives(Vec<T>),
    Unknown,
}

impl<T: Clone + Eq> StateValue<T> {
    #[must_use]
    pub fn merge(&self, other: &Self, max_alternatives: usize) -> Self {
        if self == other {
            return self.clone();
        }
        if matches!(self, Self::Unknown) || matches!(other, Self::Unknown) {
            return Self::Unknown;
        }
        let mut values = Vec::new();
        for value in self.values().into_iter().chain(other.values()) {
            if !values.contains(value) {
                if values.len() >= max_alternatives {
                    return Self::Unknown;
                }
                values.push(value.clone());
            }
        }
        match values.as_slice() {
            [only] => Self::Known(only.clone()),
            [] => Self::Unknown,
            _ => Self::Alternatives(values),
        }
    }

    fn values(&self) -> Vec<&T> {
        match self {
            Self::Known(value) => vec![value],
            Self::Alternatives(values) => values.iter().collect(),
            Self::Unknown => Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct VmDecodeState {
    pub vip: StateValue<Expr>,
    pub vsp: StateValue<Expr>,
    pub rolling_key: StateValue<Expr>,
    pub handler: StateValue<Expr>,
    pub flags: StateValue<Expr>,
    pub virtual_registers: BTreeMap<u16, StateValue<Expr>>,
}

impl VmDecodeState {
    #[must_use]
    pub fn merge(&self, other: &Self, max_alternatives: usize) -> Self {
        let register_ids: BTreeSet<_> = self
            .virtual_registers
            .keys()
            .chain(other.virtual_registers.keys())
            .copied()
            .collect();
        let virtual_registers = register_ids
            .into_iter()
            .map(|id| {
                let left = self
                    .virtual_registers
                    .get(&id)
                    .unwrap_or(&StateValue::Unknown);
                let right = other
                    .virtual_registers
                    .get(&id)
                    .unwrap_or(&StateValue::Unknown);
                (id, left.merge(right, max_alternatives))
            })
            .collect();
        Self {
            vip: self.vip.merge(&other.vip, max_alternatives),
            vsp: self.vsp.merge(&other.vsp, max_alternatives),
            rolling_key: self.rolling_key.merge(&other.rolling_key, max_alternatives),
            handler: self.handler.merge(&other.handler, max_alternatives),
            flags: self.flags.merge(&other.flags, max_alternatives),
            virtual_registers,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct VmHandlerAnalysis {
    pub entry: Rva,
    pub semantic: VmHandlerSemantic,
    pub operand_width: Option<BitWidth>,
    /// Width consumed from the semantic source before an optional resize.
    pub input_width: Option<BitWidth>,
    pub operation_width: Option<BitWidth>,
    pub resize: Option<ResizeOp>,
    pub stack_delta: Option<i32>,
    pub vip_delta: Option<i32>,
    pub rolling_key_effect: Option<Expr>,
    pub flags_effect: EffectKnowledge,
    pub confidence: u8,
    pub evidence: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct VmInstruction {
    pub vip_before: Expr,
    pub encoded_bytes: Vec<u8>,
    pub handler: Rva,
    pub semantic: VmHandlerSemantic,
    pub operands: Vec<Expr>,
    pub rolling_key_before: Expr,
    pub rolling_key_after: Expr,
    pub vsp_before: Expr,
    pub vsp_after: Expr,
    pub successors: Vec<Expr>,
    pub status: RecoveryStatus,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_preserves_incompatible_decoder_states_explicitly() {
        let left = StateValue::Known(Expr::constant(BitWidth::W32, 1));
        let right = StateValue::Known(Expr::constant(BitWidth::W32, 2));
        assert_eq!(
            left.merge(&right, 4),
            StateValue::Alternatives(vec![
                Expr::constant(BitWidth::W32, 1),
                Expr::constant(BitWidth::W32, 2),
            ])
        );
    }

    #[test]
    fn merge_degrades_to_unknown_at_resource_limit() {
        let left = StateValue::Alternatives(vec![
            Expr::constant(BitWidth::W8, 1),
            Expr::constant(BitWidth::W8, 2),
        ]);
        let right = StateValue::Known(Expr::constant(BitWidth::W8, 3));
        assert_eq!(left.merge(&right, 2), StateValue::Unknown);
    }

    #[test]
    fn control_semantics_end_linear_blocks() {
        assert!(VmHandlerSemantic::ConditionalBranch.ends_linear_block());
        assert!(VmHandlerSemantic::UnconditionalBranch.ends_linear_block());
        assert!(VmHandlerSemantic::VmExit.ends_linear_block());
        assert!(!VmHandlerSemantic::Add.ends_linear_block());
        assert!(!VmHandlerSemantic::Unknown.ends_linear_block());
    }
}
