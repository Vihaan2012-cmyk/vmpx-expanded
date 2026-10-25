use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;
use thiserror::Error;
use vmpx_pe::{PeImage, Va};

use crate::{
    BitWidth, BytecodeDirection, EffectKnowledge, ResizeOp, TransformChain, TransformError,
    VmHandlerSemantic, VmSemanticMetadata,
};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum KeyOperation {
    Add,
    Sub,
    Xor,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct KeyedTransformCodec {
    pub width: BitWidth,
    pub input_key_operation: Option<KeyOperation>,
    pub input_key_on_left: bool,
    pub transforms: TransformChain,
    pub update_key_operation: Option<KeyOperation>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct KeyedDecodeResult {
    pub encoded: u64,
    pub decoded: u64,
    pub key_before: u64,
    pub key_after: u64,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
pub struct RelativeDispatchState {
    pub vip: Va,
    pub rolling_key: u64,
    pub handler: Va,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct DecodedOperandDispatch {
    pub state_before: RelativeDispatchState,
    pub encoded_operand: Vec<u8>,
    pub operand: u64,
    pub key_after_operand: u64,
    pub encoded_displacement: Vec<u8>,
    pub displacement: u64,
    pub signed_displacement: i64,
    pub state_after: RelativeDispatchState,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct DecodedDispatch {
    pub state_before: RelativeDispatchState,
    pub encoded_displacement: Vec<u8>,
    pub displacement: u64,
    pub signed_displacement: i64,
    pub state_after: RelativeDispatchState,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RelativeDispatchCodec {
    pub displacement: KeyedTransformCodec,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RelativeThreadedCodec {
    pub operand: KeyedTransformCodec,
    pub dispatch: RelativeDispatchCodec,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct VmHandlerDecodeSpec {
    pub semantic: VmHandlerSemantic,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub semantic_metadata: Option<VmSemanticMetadata>,
    pub input_width: Option<BitWidth>,
    pub operation_width: Option<BitWidth>,
    pub resize: Option<ResizeOp>,
    pub operand: Option<KeyedTransformCodec>,
    pub dispatch: RelativeDispatchCodec,
    /// Optional static handler anchor established by this native handler
    /// before decoding its relative dispatch token.
    pub handler_base: Option<Va>,
    pub vsp_delta: Option<i32>,
    pub flags_effect: EffectKnowledge,
    pub confidence: u8,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct DecodedVmInstruction {
    pub state_before: RelativeDispatchState,
    pub encoded_bytes: Vec<u8>,
    pub semantic: VmHandlerSemantic,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub semantic_metadata: Option<VmSemanticMetadata>,
    pub input_width: Option<BitWidth>,
    pub operation_width: Option<BitWidth>,
    pub resize: Option<ResizeOp>,
    pub operand: Option<u64>,
    pub vsp_delta: Option<i32>,
    pub flags_effect: EffectKnowledge,
    pub confidence: u8,
    pub state_after: RelativeDispatchState,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LinearDecodeStop {
    MissingHandlerSpec,
    ControlFlowBoundary,
    InvalidHandlerAddress,
    NativeHandlerError,
    BytecodeError,
    RepeatedState,
    InstructionLimit,
    /// The virtual instruction pointer left the section that held the initial
    /// bytecode. Protector bytecode is confined to one section per profile, so
    /// escaping it proves the decoder state has drifted and the decoded suffix
    /// must not be trusted.
    BytecodeSectionEscape,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct LinearVmDecodeReport {
    pub schema_version: u32,
    pub instructions: Vec<DecodedVmInstruction>,
    pub final_state: RelativeDispatchState,
    pub cumulative_vsp_delta: Option<i64>,
    pub stop_reason: LinearDecodeStop,
    pub unresolved_handler: Option<Va>,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum BytecodeDecodeError {
    #[error("bytecode address {va} is invalid: {message}")]
    Address { va: Va, message: String },

    #[error("bytecode address {va} is not file-backed: {message}")]
    Unmapped { va: Va, message: String },

    #[error("transform chain contains {actual:?}, expected {expected:?}")]
    TransformWidth {
        expected: BitWidth,
        actual: BitWidth,
    },

    #[error("bytecode direction {direction:?} is not concretely decodable")]
    Direction { direction: BytecodeDirection },

    #[error(transparent)]
    Transform(#[from] TransformError),
}

impl KeyedTransformCodec {
    /// Decodes one value and applies its partial-width rolling-key update.
    ///
    /// # Errors
    ///
    /// Rejects transform chains with inconsistent widths or unsupported
    /// operations.
    pub fn decode(
        &self,
        encoded: u64,
        rolling_key: u64,
    ) -> Result<KeyedDecodeResult, BytecodeDecodeError> {
        for transform in &self.transforms.transforms {
            if transform.width != self.width {
                return Err(BytecodeDecodeError::TransformWidth {
                    expected: self.width,
                    actual: transform.width,
                });
            }
        }
        let key_low = self.width.truncate(rolling_key);
        let encoded = self.width.truncate(encoded);
        let mixed = self.input_key_operation.map_or(encoded, |operation| {
            if self.input_key_on_left {
                apply_key_operation(operation, key_low, encoded, self.width)
            } else {
                apply_key_operation(operation, encoded, key_low, self.width)
            }
        });
        let decoded = self.transforms.apply(mixed)?;
        let key_after = self.update_key_operation.map_or(rolling_key, |operation| {
            let updated_low = apply_key_operation(operation, key_low, decoded, self.width);
            (rolling_key & !self.width.mask()) | updated_low
        });
        Ok(KeyedDecodeResult {
            encoded,
            decoded,
            key_before: rolling_key,
            key_after,
        })
    }
}

impl RelativeThreadedCodec {
    /// Decodes one operand followed by a signed relative dispatch token.
    ///
    /// # Errors
    ///
    /// Returns a structured error for unmapped bytecode, address overflow,
    /// inconsistent transform widths, or unsupported transforms.
    pub fn decode_operand_dispatch(
        &self,
        pe: &PeImage<'_>,
        state: RelativeDispatchState,
    ) -> Result<DecodedOperandDispatch, BytecodeDecodeError> {
        self.decode_operand_dispatch_direction(pe, state, BytecodeDirection::Ascending)
    }

    /// Decodes an operand and dispatch token using a proven VM bytecode
    /// direction. Descending streams pre-decrement VIP before each read.
    ///
    /// # Errors
    ///
    /// Returns a structured error for unknown/state-dependent directions or
    /// the same address/transform failures as [`Self::decode_operand_dispatch`].
    pub fn decode_operand_dispatch_direction(
        &self,
        pe: &PeImage<'_>,
        state: RelativeDispatchState,
        direction: BytecodeDirection,
    ) -> Result<DecodedOperandDispatch, BytecodeDecodeError> {
        let (operand_address, displacement_vip) =
            consume_vip(state.vip, self.operand.width, direction)?;
        let encoded_operand = read_bytes(pe, operand_address, self.operand.width.bytes())?;
        let operand_value = little_endian_value(&encoded_operand);
        let operand = self.operand.decode(operand_value, state.rolling_key)?;
        let dispatch = self.dispatch.decode_direction(
            pe,
            RelativeDispatchState {
                vip: displacement_vip,
                rolling_key: operand.key_after,
                handler: state.handler,
            },
            direction,
        )?;
        Ok(DecodedOperandDispatch {
            state_before: state,
            encoded_operand,
            operand: operand.decoded,
            key_after_operand: operand.key_after,
            encoded_displacement: dispatch.encoded_displacement,
            displacement: dispatch.displacement,
            signed_displacement: dispatch.signed_displacement,
            state_after: dispatch.state_after,
        })
    }
}

impl RelativeDispatchCodec {
    /// Decodes a signed relative dispatch token without a semantic operand.
    ///
    /// # Errors
    ///
    /// Returns a structured error for unmapped bytecode, address overflow,
    /// inconsistent transform widths, or unsupported transforms.
    pub fn decode(
        &self,
        pe: &PeImage<'_>,
        state: RelativeDispatchState,
    ) -> Result<DecodedDispatch, BytecodeDecodeError> {
        self.decode_direction(pe, state, BytecodeDirection::Ascending)
    }

    /// Decodes a dispatch token using a proven VM bytecode direction.
    ///
    /// # Errors
    ///
    /// Returns a structured error for unknown/state-dependent directions or
    /// the same address/transform failures as [`Self::decode`].
    pub fn decode_direction(
        &self,
        pe: &PeImage<'_>,
        state: RelativeDispatchState,
        direction: BytecodeDirection,
    ) -> Result<DecodedDispatch, BytecodeDecodeError> {
        let (displacement_address, vip) =
            consume_vip(state.vip, self.displacement.width, direction)?;
        let encoded_displacement =
            read_bytes(pe, displacement_address, self.displacement.width.bytes())?;
        let displacement_value = little_endian_value(&encoded_displacement);
        let displacement = self
            .displacement
            .decode(displacement_value, state.rolling_key)?;
        let signed_displacement = sign_extend_to_i64(displacement.decoded, self.displacement.width);
        let handler = state
            .handler
            .checked_add_signed(signed_displacement)
            .ok_or_else(|| BytecodeDecodeError::Address {
                va: state.handler,
                message: "relative handler target overflows".to_owned(),
            })?;
        Ok(DecodedDispatch {
            state_before: state,
            encoded_displacement,
            displacement: displacement.decoded,
            signed_displacement,
            state_after: RelativeDispatchState {
                vip,
                rolling_key: displacement.key_after,
                handler,
            },
        })
    }
}

impl VmHandlerDecodeSpec {
    /// Decodes one semantic VM instruction using a recovered handler codec.
    ///
    /// # Errors
    ///
    /// Propagates structured bytecode and transform errors.
    pub fn decode(
        &self,
        pe: &PeImage<'_>,
        state: RelativeDispatchState,
    ) -> Result<DecodedVmInstruction, BytecodeDecodeError> {
        self.decode_direction(pe, state, BytecodeDirection::Ascending)
    }

    /// Decodes one semantic VM instruction using a proven bytecode direction.
    ///
    /// # Errors
    ///
    /// Propagates structured direction, bytecode, and transform errors.
    pub fn decode_direction(
        &self,
        pe: &PeImage<'_>,
        state: RelativeDispatchState,
        direction: BytecodeDirection,
    ) -> Result<DecodedVmInstruction, BytecodeDecodeError> {
        let dispatch_state = RelativeDispatchState {
            handler: self.handler_base.unwrap_or(state.handler),
            ..state
        };
        let (encoded_bytes, operand, state_after) = if let Some(operand_codec) = &self.operand {
            let decoded = RelativeThreadedCodec {
                operand: operand_codec.clone(),
                dispatch: self.dispatch.clone(),
            }
            .decode_operand_dispatch_direction(pe, dispatch_state, direction)?;
            let mut bytes = decoded.encoded_operand;
            bytes.extend_from_slice(&decoded.encoded_displacement);
            (bytes, Some(decoded.operand), decoded.state_after)
        } else {
            let decoded = self
                .dispatch
                .decode_direction(pe, dispatch_state, direction)?;
            (decoded.encoded_displacement, None, decoded.state_after)
        };
        Ok(DecodedVmInstruction {
            state_before: state,
            encoded_bytes,
            semantic: self.semantic,
            semantic_metadata: self.semantic_metadata.clone(),
            input_width: self.input_width,
            operation_width: self.operation_width,
            resize: self.resize,
            operand,
            vsp_delta: self.vsp_delta,
            flags_effect: self.flags_effect,
            confidence: self.confidence,
            state_after,
        })
    }
}

/// Decodes a straight-line VM stream until an unresolved handler, repeated
/// complete decoder state, or caller-provided instruction limit.
///
/// The handler map is deliberately external: native handler analysis must
/// recover these specifications for each mutated VM instance.
///
/// # Errors
///
/// Stops with a structured error if a known handler cannot decode its bytes or
/// compute its next state safely.
pub fn decode_linear_vm_stream(
    pe: &PeImage<'_>,
    initial_state: RelativeDispatchState,
    handlers: &BTreeMap<Va, VmHandlerDecodeSpec>,
    max_instructions: usize,
) -> Result<LinearVmDecodeReport, BytecodeDecodeError> {
    decode_linear_vm_stream_direction(
        pe,
        initial_state,
        handlers,
        BytecodeDirection::Ascending,
        max_instructions,
    )
}

/// Decodes a straight-line VM stream using a proven global bytecode direction.
///
/// # Errors
///
/// Stops with a structured error for unsupported direction or bytecode state.
pub fn decode_linear_vm_stream_direction(
    pe: &PeImage<'_>,
    initial_state: RelativeDispatchState,
    handlers: &BTreeMap<Va, VmHandlerDecodeSpec>,
    direction: BytecodeDirection,
    max_instructions: usize,
) -> Result<LinearVmDecodeReport, BytecodeDecodeError> {
    let mut instructions = Vec::new();
    let mut seen = BTreeSet::new();
    let mut state = initial_state;
    let mut cumulative_vsp_delta = Some(0_i64);

    for _ in 0..max_instructions {
        if !seen.insert(state) {
            return Ok(LinearVmDecodeReport {
                schema_version: 2,
                instructions,
                final_state: state,
                cumulative_vsp_delta,
                stop_reason: LinearDecodeStop::RepeatedState,
                unresolved_handler: None,
            });
        }
        let Some(spec) = handlers.get(&state.handler) else {
            return Ok(LinearVmDecodeReport {
                schema_version: 2,
                instructions,
                final_state: state,
                cumulative_vsp_delta,
                stop_reason: LinearDecodeStop::MissingHandlerSpec,
                unresolved_handler: Some(state.handler),
            });
        };
        if spec.semantic.ends_linear_block() {
            return Ok(LinearVmDecodeReport {
                schema_version: 2,
                instructions,
                final_state: state,
                cumulative_vsp_delta,
                stop_reason: LinearDecodeStop::ControlFlowBoundary,
                unresolved_handler: None,
            });
        }
        let instruction = spec.decode_direction(pe, state, direction)?;
        cumulative_vsp_delta = match (cumulative_vsp_delta, instruction.vsp_delta) {
            (Some(current), Some(delta)) => {
                Some(current.checked_add(i64::from(delta)).ok_or_else(|| {
                    BytecodeDecodeError::Address {
                        va: state.vip,
                        message: "cumulative VSP delta overflows".to_owned(),
                    }
                })?)
            }
            _ => None,
        };
        state = instruction.state_after;
        instructions.push(instruction);
    }

    Ok(LinearVmDecodeReport {
        schema_version: 2,
        instructions,
        final_state: state,
        cumulative_vsp_delta,
        stop_reason: LinearDecodeStop::InstructionLimit,
        unresolved_handler: None,
    })
}

fn consume_vip(
    vip: Va,
    width: BitWidth,
    direction: BytecodeDirection,
) -> Result<(Va, Va), BytecodeDecodeError> {
    match direction {
        BytecodeDirection::Ascending => {
            let next = vip.checked_add(u64::from(width.bytes())).ok_or_else(|| {
                BytecodeDecodeError::Address {
                    va: vip,
                    message: "VIP overflows after bytecode read".to_owned(),
                }
            })?;
            Ok((vip, next))
        }
        BytecodeDirection::Descending => {
            let address = vip
                .checked_add_signed(-i64::from(width.bytes()))
                .ok_or_else(|| BytecodeDecodeError::Address {
                    va: vip,
                    message: "VIP underflows before descending bytecode read".to_owned(),
                })?;
            Ok((address, address))
        }
        BytecodeDirection::StateDependent | BytecodeDirection::Unknown => {
            Err(BytecodeDecodeError::Direction { direction })
        }
    }
}

fn apply_key_operation(operation: KeyOperation, left: u64, right: u64, width: BitWidth) -> u64 {
    let value = match operation {
        KeyOperation::Add => left.wrapping_add(right),
        KeyOperation::Sub => left.wrapping_sub(right),
        KeyOperation::Xor => left ^ right,
    };
    width.truncate(value)
}

fn little_endian_value(bytes: &[u8]) -> u64 {
    let mut value = [0_u8; 8];
    value[..bytes.len()].copy_from_slice(bytes);
    u64::from_le_bytes(value)
}

fn sign_extend_to_i64(value: u64, width: BitWidth) -> i64 {
    let value = width.truncate(value);
    if width == BitWidth::W64 {
        return i64::from_ne_bytes(value.to_ne_bytes());
    }
    let sign_bit = 1_u64 << (width.bits() - 1);
    let extended = if value & sign_bit == 0 {
        value
    } else {
        value | !width.mask()
    };
    i64::from_ne_bytes(extended.to_ne_bytes())
}

fn read_bytes(pe: &PeImage<'_>, va: Va, length: u32) -> Result<Vec<u8>, BytecodeDecodeError> {
    let rva =
        pe.optional
            .image_base
            .checked_rva_of(va)
            .ok_or_else(|| BytecodeDecodeError::Address {
                va,
                message: "VA is below the image base or exceeds RVA width".to_owned(),
            })?;
    let file_offset =
        pe.rva_to_file_offset(rva)
            .map_err(|error| BytecodeDecodeError::Unmapped {
                va,
                message: error.to_string(),
            })?;
    let start = usize::try_from(file_offset.0).map_err(|_| BytecodeDecodeError::Address {
        va,
        message: "file offset cannot be represented".to_owned(),
    })?;
    let size = usize::try_from(length).map_err(|_| BytecodeDecodeError::Address {
        va,
        message: "read length cannot be represented".to_owned(),
    })?;
    let end = start
        .checked_add(size)
        .ok_or_else(|| BytecodeDecodeError::Address {
            va,
            message: "bytecode read range overflows".to_owned(),
        })?;
    pe.data()
        .get(start..end)
        .map(<[u8]>::to_vec)
        .ok_or_else(|| BytecodeDecodeError::Unmapped {
            va,
            message: "bytecode read exceeds file-backed data".to_owned(),
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Transform, TransformOp};

    #[test]
    fn partial_width_key_updates_preserve_high_bits() {
        let codec = KeyedTransformCodec {
            width: BitWidth::W8,
            input_key_operation: Some(KeyOperation::Xor),
            input_key_on_left: false,
            transforms: TransformChain {
                transforms: vec![Transform {
                    op: TransformOp::Add,
                    width: BitWidth::W8,
                    operand: 1,
                }],
            },
            update_key_operation: Some(KeyOperation::Xor),
        };
        let result = codec
            .decode(0x20, 0x1122_3344_5566_7788)
            .expect("valid codec");
        assert_eq!(result.decoded, 0xA9);
        assert_eq!(result.key_after, 0x1122_3344_5566_7721);
    }

    #[test]
    fn sign_extension_respects_encoded_width() {
        assert_eq!(sign_extend_to_i64(0x80, BitWidth::W8), -128);
        assert_eq!(sign_extend_to_i64(0x7F, BitWidth::W8), 127);
        assert_eq!(sign_extend_to_i64(0xFFFF_FFFF, BitWidth::W32), -1);
    }

    #[test]
    fn bytecode_direction_controls_cursor_timing() {
        assert_eq!(
            consume_vip(
                Va(0x1_4000_1000),
                BitWidth::W32,
                BytecodeDirection::Ascending
            )
            .expect("ascending cursor"),
            (Va(0x1_4000_1000), Va(0x1_4000_1004))
        );
        assert_eq!(
            consume_vip(
                Va(0x1_4000_1000),
                BitWidth::W32,
                BytecodeDirection::Descending
            )
            .expect("descending cursor"),
            (Va(0x1_4000_0FFC), Va(0x1_4000_0FFC))
        );
        assert!(matches!(
            consume_vip(Va(0x1_4000_1000), BitWidth::W32, BytecodeDirection::Unknown),
            Err(BytecodeDecodeError::Direction { .. })
        ));
    }
}
