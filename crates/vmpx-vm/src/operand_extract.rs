use std::collections::{BTreeMap, BTreeSet};

use iced_x86::{InstructionInfoFactory, Mnemonic, OpAccess, OpKind};
use serde::Serialize;
use vmpx_disasm::DirectJumpTrace;
use vmpx_pe::Rva;

use crate::{
    BytecodeDirection, HandlerRegisterRoles, KeyedTransformCodec, NativeBackwardSlice,
    TransformChain,
    dispatch_extract::{
        canonical, immediate_transform, instruction_writes_value, is_active, is_source_load,
        key_operation, recover_key_updates, same_register_value, signed_immediate,
        source_load_width, writes_register,
    },
};

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct KeyedOperandCodecCandidate {
    pub encoded_load: Rva,
    pub decoded_register: String,
    #[serde(skip)]
    pub decoded_register_id: iced_x86::Register,
    pub full_register_value_proven: bool,
    pub vip_delta: i64,
    pub key_update: Rva,
    pub first_consumer: Option<Rva>,
    pub codec: KeyedTransformCodec,
    pub confidence: u8,
    pub evidence: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct KeyedOperandCodecRejection {
    pub encoded_load: Rva,
    pub reason: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct KeyedOperandCodecReport {
    pub schema_version: u32,
    pub candidates: Vec<KeyedOperandCodecCandidate>,
    pub rejected_sources: Vec<KeyedOperandCodecRejection>,
}

fn reads_register_value(
    factory: &mut InstructionInfoFactory,
    instruction: &iced_x86::Instruction,
    register: iced_x86::Register,
    width: crate::BitWidth,
) -> bool {
    factory
        .info(instruction)
        .used_registers()
        .iter()
        .any(|used| {
            same_register_value(used.register(), register, width)
                && matches!(
                    used.access(),
                    OpAccess::Read
                        | OpAccess::CondRead
                        | OpAccess::ReadWrite
                        | OpAccess::ReadCondWrite
                )
        })
}

#[allow(clippy::too_many_lines)]
fn attempt_operand(
    active: &[&vmpx_disasm::DecodedInstruction],
    roles: HandlerRegisterRoles,
    direction: BytecodeDirection,
    mut vip_delta: i64,
) -> Result<KeyedOperandCodecCandidate, String> {
    let source = active
        .first()
        .copied()
        .ok_or_else(|| "empty VIP-load interval".to_owned())?;
    let destination = source.instruction.op0_register();
    let target = canonical(destination);
    let width = source_load_width(&source.instruction)
        .ok_or_else(|| "unsupported operand width".to_owned())?;
    let updates = recover_key_updates(active, roles.rolling_key, target, width);
    let [update] = updates.as_slice() else {
        return Err(format!(
            "expected one rolling-key update in the load interval, recovered {}",
            updates.len()
        ));
    };
    let update_key_operation = update.operation;
    let key_update = update.evidence_rva;
    let update_index = active
        .iter()
        .position(|decoded| decoded.rva == key_update)
        .ok_or_else(|| "rolling-key update is outside its load interval".to_owned())?;
    let mut input_key_operation = None;
    let mut input_key_index = None;
    let mut transforms = Vec::new();
    let mut last_transform_index = None;
    let mut factory = InstructionInfoFactory::new();
    let mut full_register_value_proven = match source.instruction.mnemonic() {
        Mnemonic::Movzx => destination.size() >= 4,
        Mnemonic::Mov => {
            destination.size() == 8 && width == crate::BitWidth::W64
                || destination.size() == 4 && width == crate::BitWidth::W32
        }
        _ => false,
    };

    for (index, decoded) in active.iter().enumerate().take(update_index + 1).skip(1) {
        let instruction = &decoded.instruction;
        if direction == BytecodeDirection::Ascending
            && instruction.op0_kind() == OpKind::Register
            && canonical(instruction.op0_register()) == canonical(roles.vip)
            && matches!(
                instruction.mnemonic(),
                Mnemonic::Add | Mnemonic::Sub | Mnemonic::Inc | Mnemonic::Dec
            )
        {
            if instruction.op0_register().size() != 8 {
                return Err("VIP adjustment is not a full-width write".to_owned());
            }
            let delta = match instruction.mnemonic() {
                Mnemonic::Inc | Mnemonic::Dec => 1,
                _ => signed_immediate(instruction, 1)
                    .ok_or_else(|| "VIP adjustment is not an immediate".to_owned())?,
            };
            vip_delta = match instruction.mnemonic() {
                Mnemonic::Add | Mnemonic::Inc => vip_delta.checked_add(delta),
                Mnemonic::Sub | Mnemonic::Dec => vip_delta.checked_sub(delta),
                _ => unreachable!(),
            }
            .ok_or_else(|| "VIP adjustment overflows".to_owned())?;
            continue;
        }
        if writes_register(&mut factory, instruction, canonical(roles.vip)) {
            return Err(format!("VIP has an unmodelled write at {}", decoded.rva));
        }
        if instruction_writes_value(&mut factory, instruction, destination, width) {
            if instruction.op0_kind() != OpKind::Register
                || instruction.op0_register().size()
                    != usize::try_from(width.bytes()).unwrap_or(usize::MAX)
            {
                full_register_value_proven = false;
            }
            let mixes_key = instruction.op0_kind() == OpKind::Register
                && same_register_value(instruction.op0_register(), destination, width)
                && instruction.op1_kind() == OpKind::Register
                && same_register_value(instruction.op1_register(), roles.rolling_key, width);
            if mixes_key {
                if input_key_operation.is_some() || !transforms.is_empty() {
                    return Err("ambiguous or late input-key operation".to_owned());
                }
                input_key_operation = key_operation(instruction.mnemonic());
                if input_key_operation.is_none() {
                    return Err("unsupported input-key operation".to_owned());
                }
                input_key_index = Some(index);
                continue;
            }
            let transform = immediate_transform(instruction, width)
                .ok_or_else(|| format!("unsupported operand-register write at {}", decoded.rva))?;
            transforms.push(transform);
            last_transform_index = Some(index);
        }
    }
    let completion_index = active
        .iter()
        .position(|decoded| decoded.rva == update.completion_rva)
        .ok_or_else(|| "rolling-key completion is outside its load interval".to_owned())?;
    if input_key_index.is_none_or(|mix| update_index <= mix)
        || last_transform_index.is_some_and(|last| update_index <= last)
    {
        return Err("rolling-key update is out of semantic order".to_owned());
    }
    let allowed_key_writes = BTreeSet::from([update.completion_rva]);
    if active.iter().any(|decoded| {
        writes_register(
            &mut factory,
            &decoded.instruction,
            canonical(roles.rolling_key),
        ) && !allowed_key_writes.contains(&decoded.rva)
    }) {
        return Err("rolling-key register has an unmodelled write".to_owned());
    }
    let input_key_operation =
        input_key_operation.ok_or_else(|| "input-key operation was not recovered".to_owned())?;
    let expected_vip_delta = match direction {
        BytecodeDirection::Ascending => i64::from(width.bytes()),
        BytecodeDirection::Descending => -i64::from(width.bytes()),
        BytecodeDirection::StateDependent | BytecodeDirection::Unknown => {
            return Err("bytecode direction is not concrete".to_owned());
        }
    };
    if vip_delta != expected_vip_delta {
        return Err(format!(
            "VIP delta {vip_delta} does not match encoded width {}",
            width.bytes()
        ));
    }
    let mut first_consumer = None;
    for decoded in active.iter().skip(completion_index + 1) {
        let instruction = &decoded.instruction;
        if writes_register(&mut factory, instruction, canonical(roles.vip)) {
            return Err(format!(
                "VIP is modified after operand decoding at {}",
                decoded.rva
            ));
        }
        if first_consumer.is_none()
            && instruction_writes_value(&mut factory, instruction, destination, width)
        {
            return Err(format!(
                "decoded operand is overwritten before its first consumer at {}",
                decoded.rva
            ));
        }
        if first_consumer.is_none()
            && reads_register_value(&mut factory, instruction, destination, width)
        {
            first_consumer = Some(decoded.rva);
        }
    }
    Ok(KeyedOperandCodecCandidate {
        encoded_load: source.rva,
        decoded_register: format!("{destination:?}"),
        decoded_register_id: destination,
        full_register_value_proven,
        vip_delta,
        key_update,
        first_consumer,
        codec: KeyedTransformCodec {
            width,
            input_key_operation: Some(input_key_operation),
            input_key_on_left: false,
            transforms: TransformChain { transforms },
            update_key_operation: Some(update_key_operation),
        },
        confidence: 100,
        evidence: vec![
            format!("encoded {:?} load from the recovered VIP register", width),
            format!("input is mixed with the recovered key using {input_key_operation:?}"),
            format!("VIP advances by {vip_delta} bytes"),
            format!("rolling key updates using {update_key_operation:?} at {key_update}"),
            "source interval ends at the next VIP-relative load".to_owned(),
        ],
    })
}

/// Recovers keyed operand codecs from VIP-relative load intervals, excluding
/// sources already proven to decode handler dispatches.
#[must_use]
pub fn extract_keyed_operand_codecs(
    trace: &DirectJumpTrace,
    slice: &NativeBackwardSlice,
    roles: HandlerRegisterRoles,
    dispatch_sources: &BTreeSet<Rva>,
) -> KeyedOperandCodecReport {
    extract_keyed_operand_codecs_direction(
        trace,
        slice,
        roles,
        dispatch_sources,
        BytecodeDirection::Ascending,
    )
}

/// Recovers operand codecs using the cursor timing of a proven VM profile.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn extract_keyed_operand_codecs_direction(
    trace: &DirectJumpTrace,
    slice: &NativeBackwardSlice,
    roles: HandlerRegisterRoles,
    dispatch_sources: &BTreeSet<Rva>,
    direction: BytecodeDirection,
) -> KeyedOperandCodecReport {
    let dispositions: BTreeMap<_, _> = slice
        .instructions
        .iter()
        .map(|instruction| (instruction.rva, instruction.disposition))
        .collect();
    let decoded: Vec<_> = trace.instructions().collect();
    let source_indices: Vec<_> = decoded
        .iter()
        .enumerate()
        .filter(|(_, decoded)| is_source_load(&decoded.instruction, roles.vip))
        .map(|(index, _)| index)
        .collect();
    let mut candidates = Vec::new();
    let mut rejected_sources = Vec::new();
    for (position, source_index) in source_indices.iter().copied().enumerate() {
        let source = decoded[source_index];
        if dispatch_sources.contains(&source.rva) {
            continue;
        }
        let mut factory = InstructionInfoFactory::new();
        if position == 0
            && decoded[..source_index].iter().any(|decoded| {
                dispositions
                    .get(&decoded.rva)
                    .copied()
                    .is_some_and(is_active)
                    && writes_register(
                        &mut factory,
                        &decoded.instruction,
                        canonical(roles.rolling_key),
                    )
            })
        {
            rejected_sources.push(KeyedOperandCodecRejection {
                encoded_load: source.rva,
                reason: "rolling key is modified before the encoded operand load".to_owned(),
            });
            continue;
        }
        if direction == BytecodeDirection::Ascending
            && decoded[..source_index].iter().any(|decoded| {
                dispositions
                    .get(&decoded.rva)
                    .copied()
                    .is_some_and(is_active)
                    && writes_register(&mut factory, &decoded.instruction, canonical(roles.vip))
            })
        {
            rejected_sources.push(KeyedOperandCodecRejection {
                encoded_load: source.rva,
                reason: "VIP is modified before the encoded operand load".to_owned(),
            });
            continue;
        }
        let next_source = source_indices
            .get(position + 1)
            .copied()
            .unwrap_or(decoded.len());
        let end = if direction == BytecodeDirection::Descending {
            decoded[source_index + 1..next_source]
                .iter()
                .position(|decoded| {
                    dispositions
                        .get(&decoded.rva)
                        .copied()
                        .is_some_and(is_active)
                        && writes_register(
                            &mut InstructionInfoFactory::new(),
                            &decoded.instruction,
                            canonical(roles.vip),
                        )
                })
                .map_or(next_source, |offset| source_index + 1 + offset)
        } else {
            next_source
        };
        let previous_source = position
            .checked_sub(1)
            .and_then(|previous| source_indices.get(previous).copied());
        let prefix_start = previous_source.map_or(0, |index| index + 1);
        let mut prefix_vip_delta = 0_i64;
        let mut prefix_error = None;
        if direction == BytecodeDirection::Descending {
            for decoded in &decoded[prefix_start..source_index] {
                if !dispositions
                    .get(&decoded.rva)
                    .copied()
                    .is_some_and(is_active)
                    || !writes_register(
                        &mut InstructionInfoFactory::new(),
                        &decoded.instruction,
                        canonical(roles.vip),
                    )
                {
                    continue;
                }
                let instruction = &decoded.instruction;
                if instruction.op0_kind() != OpKind::Register
                    || instruction.op0_register().size() != 8
                    || !matches!(instruction.mnemonic(), Mnemonic::Add | Mnemonic::Sub)
                {
                    prefix_error = Some(format!(
                        "descending VIP has an unmodelled pre-source write at {}",
                        decoded.rva
                    ));
                    break;
                }
                let Some(delta) = signed_immediate(instruction, 1) else {
                    prefix_error = Some("descending VIP adjustment is not immediate".to_owned());
                    break;
                };
                prefix_vip_delta = match instruction.mnemonic() {
                    Mnemonic::Add => prefix_vip_delta.checked_add(delta),
                    Mnemonic::Sub => prefix_vip_delta.checked_sub(delta),
                    _ => unreachable!(),
                }
                .unwrap_or(i64::MAX);
            }
        }
        if let Some(reason) = prefix_error {
            rejected_sources.push(KeyedOperandCodecRejection {
                encoded_load: source.rva,
                reason,
            });
            continue;
        }
        let active: Vec<_> = decoded[source_index..end]
            .iter()
            .copied()
            .filter(|decoded| {
                dispositions
                    .get(&decoded.rva)
                    .copied()
                    .is_some_and(is_active)
            })
            .collect();
        match attempt_operand(&active, roles, direction, prefix_vip_delta) {
            Ok(candidate) => candidates.push(candidate),
            Err(reason) => rejected_sources.push(KeyedOperandCodecRejection {
                encoded_load: source.rva,
                reason,
            }),
        }
    }
    KeyedOperandCodecReport {
        schema_version: 1,
        candidates,
        rejected_sources,
    }
}

#[cfg(test)]
mod tests {
    use iced_x86::{Code, Decoder, DecoderOptions, Register, code_asm::*};
    use vmpx_disasm::DecodedInstruction;
    use vmpx_pe::{Rva, Va};

    use super::*;
    use crate::{BitWidth, KeyOperation, Transform, TransformOp};

    #[test]
    fn extracts_byte_operand_codec_from_assembled_effects() {
        let mut assembler = CodeAssembler::new(64).expect("x64 assembler");
        assembler
            .movzx(ecx, byte_ptr(r15))
            .expect("encode operand load");
        assembler.inc(r15).expect("encode VIP advance");
        assembler.xor(cl, bpl).expect("encode key mix");
        assembler.sub(cl, 0x6D).expect("encode subtract");
        assembler.rol(cl, 1).expect("encode rotate");
        assembler.inc(cl).expect("encode increment");
        assembler.neg(cl).expect("encode negate");
        assembler.push(rbp).expect("encode key save");
        assembler
            .xor(byte_ptr(rsp), cl)
            .expect("encode partial key update");
        assembler.pop(rbp).expect("encode key restore");
        assembler
            .mov(byte_ptr(r11), cl)
            .expect("encode first consumer");
        let bytes = assembler
            .assemble(0x1_4000_1000)
            .expect("assemble operand decoder");
        let mut decoder = Decoder::with_ip(64, &bytes, 0x1_4000_1000, DecoderOptions::NONE);
        let mut decoded_instructions = Vec::new();
        while decoder.can_decode() {
            let instruction = decoder.decode();
            assert_ne!(instruction.code(), Code::INVALID);
            let offset = usize::try_from(instruction.ip() - 0x1_4000_1000).expect("small offset");
            let length = instruction.len();
            decoded_instructions.push(DecodedInstruction {
                rva: Rva(0x1000 + u32::try_from(offset).expect("small RVA")),
                va: Va(instruction.ip()),
                bytes: bytes[offset..offset + length].to_vec(),
                instruction,
            });
        }
        let active: Vec<_> = decoded_instructions.iter().collect();
        let candidate = attempt_operand(
            &active,
            HandlerRegisterRoles {
                vip: Register::R15,
                rolling_key: Register::RBP,
                handler: Register::R14,
            },
            BytecodeDirection::Ascending,
            0,
        )
        .expect("assembled codec must be recovered");

        assert_eq!(candidate.vip_delta, 1);
        assert!(candidate.first_consumer.is_some());
        assert!(candidate.full_register_value_proven);
        assert_eq!(candidate.codec.width, BitWidth::W8);
        assert_eq!(candidate.codec.input_key_operation, Some(KeyOperation::Xor));
        assert_eq!(
            candidate.codec.update_key_operation,
            Some(KeyOperation::Xor)
        );
        assert_eq!(
            candidate.codec.transforms.transforms,
            vec![
                Transform {
                    op: TransformOp::Sub,
                    width: BitWidth::W8,
                    operand: 0x6D,
                },
                Transform {
                    op: TransformOp::Rol,
                    width: BitWidth::W8,
                    operand: 1,
                },
                Transform {
                    op: TransformOp::Add,
                    width: BitWidth::W8,
                    operand: 1,
                },
                Transform {
                    op: TransformOp::Neg,
                    width: BitWidth::W8,
                    operand: 0,
                },
            ]
        );
    }

    #[test]
    fn rejects_operand_overwritten_before_first_consumer() {
        let mut assembler = CodeAssembler::new(64).expect("x64 assembler");
        assembler
            .movzx(ecx, byte_ptr(r15))
            .expect("encode operand load");
        assembler.inc(r15).expect("encode VIP advance");
        assembler.xor(cl, bpl).expect("encode key mix");
        assembler.push(rbp).expect("encode key save");
        assembler
            .xor(byte_ptr(rsp), cl)
            .expect("encode partial key update");
        assembler.pop(rbp).expect("encode key restore");
        assembler.mov(ecx, 7).expect("overwrite decoded value");
        assembler.mov(byte_ptr(r11), cl).expect("encode later read");
        let bytes = assembler
            .assemble(0x1_4000_1000)
            .expect("assemble operand decoder");
        let mut decoder = Decoder::with_ip(64, &bytes, 0x1_4000_1000, DecoderOptions::NONE);
        let mut decoded_instructions = Vec::new();
        while decoder.can_decode() {
            let instruction = decoder.decode();
            assert_ne!(instruction.code(), Code::INVALID);
            let offset = usize::try_from(instruction.ip() - 0x1_4000_1000).expect("small offset");
            let length = instruction.len();
            decoded_instructions.push(DecodedInstruction {
                rva: Rva(0x1000 + u32::try_from(offset).expect("small RVA")),
                va: Va(instruction.ip()),
                bytes: bytes[offset..offset + length].to_vec(),
                instruction,
            });
        }
        let active = decoded_instructions.iter().collect::<Vec<_>>();
        let error = attempt_operand(
            &active,
            HandlerRegisterRoles {
                vip: Register::R15,
                rolling_key: Register::RBP,
                handler: Register::R14,
            },
            BytecodeDirection::Ascending,
            0,
        )
        .expect_err("overwritten decoded operands must be rejected");

        assert!(error.contains("overwritten before its first consumer"));
    }

    #[test]
    fn rejects_wrong_lane_key_mix_and_widened_transform() {
        for wrong_lane in [true, false] {
            let mut assembler = CodeAssembler::new(64).expect("x64 assembler");
            assembler.movzx(ecx, byte_ptr(r15)).expect("operand load");
            assembler.inc(r15).expect("VIP advance");
            if wrong_lane {
                assembler.xor(ch, bl).expect("wrong-lane key mix");
            } else {
                assembler.xor(cl, bl).expect("key mix");
                assembler.sub(ecx, 0x6D).expect("widened transform");
            }
            assembler.push(rbx).expect("key save");
            assembler
                .xor(byte_ptr(rsp), cl)
                .expect("partial key update");
            assembler.pop(rbx).expect("key restore");
            assembler.mov(byte_ptr(r11), cl).expect("consumer");
            let bytes = assembler.assemble(0x1_4000_1000).expect("assemble");
            let mut decoder = Decoder::with_ip(64, &bytes, 0x1_4000_1000, DecoderOptions::NONE);
            let mut instructions = Vec::new();
            while decoder.can_decode() {
                let instruction = decoder.decode();
                assert_ne!(instruction.code(), Code::INVALID);
                let offset =
                    usize::try_from(instruction.ip() - 0x1_4000_1000).expect("small offset");
                instructions.push(DecodedInstruction {
                    rva: Rva(0x1000 + u32::try_from(offset).expect("small RVA")),
                    va: Va(instruction.ip()),
                    bytes: bytes[offset..offset + instruction.len()].to_vec(),
                    instruction,
                });
            }
            let active = instructions.iter().collect::<Vec<_>>();
            assert!(
                attempt_operand(
                    &active,
                    HandlerRegisterRoles {
                        vip: Register::R15,
                        rolling_key: Register::RBX,
                        handler: Register::R14,
                    },
                    BytecodeDirection::Ascending,
                    0,
                )
                .is_err()
            );
        }
    }

    #[test]
    fn rejects_vip_write_after_operand_consumer() {
        let mut assembler = CodeAssembler::new(64).expect("x64 assembler");
        assembler.movzx(ecx, byte_ptr(r15)).expect("operand load");
        assembler.inc(r15).expect("VIP advance");
        assembler.xor(cl, bl).expect("key mix");
        assembler.push(rbx).expect("key save");
        assembler
            .xor(byte_ptr(rsp), cl)
            .expect("partial key update");
        assembler.pop(rbx).expect("key restore");
        assembler.mov(byte_ptr(r11), cl).expect("first consumer");
        assembler.add(r15, 7).expect("unmodelled late VIP change");
        let bytes = assembler.assemble(0x1_4000_1000).expect("assemble");
        let mut decoder = Decoder::with_ip(64, &bytes, 0x1_4000_1000, DecoderOptions::NONE);
        let mut instructions = Vec::new();
        while decoder.can_decode() {
            let instruction = decoder.decode();
            assert_ne!(instruction.code(), Code::INVALID);
            let offset = usize::try_from(instruction.ip() - 0x1_4000_1000).expect("small offset");
            instructions.push(DecodedInstruction {
                rva: Rva(0x1000 + u32::try_from(offset).expect("small RVA")),
                va: Va(instruction.ip()),
                bytes: bytes[offset..offset + instruction.len()].to_vec(),
                instruction,
            });
        }
        let active = instructions.iter().collect::<Vec<_>>();
        let error = attempt_operand(
            &active,
            HandlerRegisterRoles {
                vip: Register::R15,
                rolling_key: Register::RBX,
                handler: Register::R14,
            },
            BytecodeDirection::Ascending,
            0,
        )
        .expect_err("late VIP changes must invalidate the operand interval");

        assert!(error.contains("after operand decoding"));
    }

    #[test]
    fn byte_load_without_extension_does_not_prove_full_index_value() {
        let mut assembler = CodeAssembler::new(64).expect("x64 assembler");
        assembler.mov(cl, byte_ptr(r15)).expect("operand load");
        assembler.inc(r15).expect("VIP advance");
        assembler.xor(cl, bpl).expect("key mix");
        assembler.push(rbp).expect("key save");
        assembler
            .xor(byte_ptr(rsp), cl)
            .expect("partial key update");
        assembler.pop(rbp).expect("key restore");
        assembler
            .mov(rax, qword_ptr(rsp + rcx))
            .expect("full-width index consumer");
        let bytes = assembler
            .assemble(0x1_4000_1000)
            .expect("assemble operand decoder");
        let mut decoder = Decoder::with_ip(64, &bytes, 0x1_4000_1000, DecoderOptions::NONE);
        let mut decoded_instructions = Vec::new();
        while decoder.can_decode() {
            let instruction = decoder.decode();
            assert_ne!(instruction.code(), Code::INVALID);
            let offset = usize::try_from(instruction.ip() - 0x1_4000_1000).expect("small offset");
            let length = instruction.len();
            decoded_instructions.push(DecodedInstruction {
                rva: Rva(0x1000 + u32::try_from(offset).expect("small RVA")),
                va: Va(instruction.ip()),
                bytes: bytes[offset..offset + length].to_vec(),
                instruction,
            });
        }
        let active = decoded_instructions.iter().collect::<Vec<_>>();
        let candidate = attempt_operand(
            &active,
            HandlerRegisterRoles {
                vip: Register::R15,
                rolling_key: Register::RBP,
                handler: Register::R14,
            },
            BytecodeDirection::Ascending,
            0,
        )
        .expect("low-byte codec remains valid as a byte value");

        assert!(!candidate.full_register_value_proven);
    }
}
