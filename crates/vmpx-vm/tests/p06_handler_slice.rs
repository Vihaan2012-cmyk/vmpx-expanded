use std::{collections::BTreeSet, fs, path::PathBuf};

use iced_x86::Register;
use vmpx_disasm::{
    DirectJumpTraceLimits, RegisterDispatchProofLimits, trace_direct_jump_fragments,
};
use vmpx_pe::{PeImage, Va};
use vmpx_vm::{
    BitWidth, BytecodeDirection, EffectKnowledge, HandlerCodecRecoveryConfig, HandlerEffectRoles,
    HandlerRegisterRoles, HandlerSliceConfig, KeyOperation, NativeInstructionDisposition,
    RecoveryStatus, Transform, TransformOp, VmHandlerSemantic, backward_slice_handler_trace,
    classify_handler_effects, extract_handler_effects, extract_keyed_operand_codecs,
    extract_relative_dispatch_codecs, extract_relative_dispatch_codecs_with_postlude,
    recover_handler_codecs,
};

#[test]
#[ignore = "large controlled mapped-image fixture"]
#[allow(clippy::too_many_lines)]
fn slices_ida_verified_add_handler_without_byte_signatures() {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
        "../../../testfilevmp/systems.throughvmpstatic.thatsalsoopeninidaprornthroughmcp.xpl",
    );
    let data = fs::read(fixture).expect("controlled mapped fixture must exist");
    let pe = PeImage::parse(&data).expect("controlled fixture must parse");
    let entry = pe
        .optional
        .image_base
        .checked_rva_of(Va(0x0003_7AAA_7BA9))
        .expect("IDA-verified handler VA belongs to the image");
    let trace = trace_direct_jump_fragments(
        &pe,
        entry,
        DirectJumpTraceLimits {
            max_fragments: 1_024,
            max_instructions: 50_000,
            max_instructions_per_fragment: 10_000,
        },
    )
    .expect("confirmed handler entry must decode");
    let slice = backward_slice_handler_trace(
        &trace,
        &HandlerSliceConfig {
            live_out_registers: vec![Register::R11, Register::R14, Register::R15, Register::RBP],
            live_out_rflags: 0,
        },
    );

    eprintln!(
        "retained={} dead={} structural={} live_in={:?}",
        slice.retained_instructions,
        slice.dead_instructions,
        slice.structural_instructions,
        slice.live_in_registers
    );
    for instruction in &slice.instructions {
        eprintln!(
            "{} {:?} {} {:?}",
            instruction.rva, instruction.disposition, instruction.text, instruction.reasons
        );
    }

    assert_eq!(trace.decoded_instructions, 41);
    assert_eq!(slice.structural_instructions, 4);
    assert!(slice.dead_instructions >= 10);
    assert!(slice.unknown_control_barriers.is_empty());
    assert!(slice.conservative_memory);
    assert!(slice.instructions.iter().any(|instruction| {
        instruction.text == "add r10,rbx"
            && instruction.disposition == NativeInstructionDisposition::LiveDependency
    }));
    assert!(slice.instructions.iter().any(|instruction| {
        instruction.text == "pushfq"
            && instruction.disposition != NativeInstructionDisposition::Dead
    }));
    assert!(slice.instructions.iter().any(|instruction| {
        instruction.text == "shr bl,0F3h"
            && instruction.disposition == NativeInstructionDisposition::Dead
    }));

    let extraction = extract_relative_dispatch_codecs(
        &trace,
        &slice,
        HandlerRegisterRoles {
            vip: Register::R15,
            rolling_key: Register::RBP,
            handler: Register::R14,
        },
    );
    assert!(extraction.rejected_sources.is_empty());
    let dispatch = extraction
        .candidates
        .first()
        .expect("relative dispatch codec must be recovered from effects");
    assert_eq!(extraction.candidates.len(), 1);
    assert_eq!(dispatch.vip_delta, 4);
    assert_eq!(
        dispatch.codec.displacement.input_key_operation,
        Some(KeyOperation::Xor)
    );
    assert_eq!(
        dispatch.codec.displacement.update_key_operation,
        Some(KeyOperation::Xor)
    );
    assert_eq!(
        dispatch.codec.displacement.transforms.transforms,
        vec![
            Transform {
                op: TransformOp::Ror,
                width: BitWidth::W32,
                operand: 1,
            },
            Transform {
                op: TransformOp::Neg,
                width: BitWidth::W32,
                operand: 0,
            },
            Transform {
                op: TransformOp::Add,
                width: BitWidth::W32,
                operand: 0xF638_338D,
            },
            Transform {
                op: TransformOp::Xor,
                width: BitWidth::W32,
                operand: 0xF6DF_F93A,
            },
        ]
    );

    let mut recovered_handlers = 0_usize;
    for handler_va in [
        0x0003_7AA6_87C7,
        0x0003_7AA5_147E,
        0x0003_7AA4_F44E,
        0x0003_7AAA_7BA9,
        0x0003_7AA7_B553,
        0x0003_7AA7_0664,
        0x0003_7AA1_F283,
    ] {
        let handler_entry = pe
            .optional
            .image_base
            .checked_rva_of(Va(handler_va))
            .expect("IDA-verified handler VA belongs to the image");
        let handler_trace = trace_direct_jump_fragments(
            &pe,
            handler_entry,
            DirectJumpTraceLimits {
                max_fragments: 1_024,
                max_instructions: 50_000,
                max_instructions_per_fragment: 10_000,
            },
        )
        .expect("confirmed handler entry must decode");
        let handler_slice = backward_slice_handler_trace(
            &handler_trace,
            &HandlerSliceConfig {
                live_out_registers: vec![
                    Register::R11,
                    Register::R14,
                    Register::R15,
                    Register::RBP,
                ],
                live_out_rflags: 0,
            },
        );
        let report = extract_relative_dispatch_codecs_with_postlude(
            &pe,
            &handler_trace,
            &handler_slice,
            HandlerRegisterRoles {
                vip: Register::R15,
                rolling_key: Register::RBP,
                handler: Register::R14,
            },
            RegisterDispatchProofLimits::default(),
        )
        .expect("postlude proof setup must decode");
        eprintln!(
            "handler={handler_va:#X} stop={:?} candidates={} rejected={:?}",
            handler_trace.stop,
            report.candidates.len(),
            report.rejected_sources
        );
        let dispatch_sources: BTreeSet<_> = report
            .candidates
            .iter()
            .map(|candidate| candidate.encoded_load)
            .collect();
        let operands = extract_keyed_operand_codecs(
            &handler_trace,
            &handler_slice,
            HandlerRegisterRoles {
                vip: Register::R15,
                rolling_key: Register::RBP,
                handler: Register::R14,
            },
            &dispatch_sources,
        );
        eprintln!(
            "handler={handler_va:#X} operands={:?} operand_rejections={:?}",
            operands.candidates, operands.rejected_sources
        );
        let expected_operands = match handler_va {
            0x0003_7AA6_87C7 | 0x0003_7AA5_147E | 0x0003_7AA4_F44E | 0x0003_7AA7_B553 => 1,
            _ => 0,
        };
        assert_eq!(operands.candidates.len(), expected_operands);
        assert!(operands.rejected_sources.is_empty());
        let dispatch = report
            .candidates
            .first()
            .expect("confirmed handlers have one dispatch codec");
        let effects = extract_handler_effects(
            &handler_trace,
            &handler_slice,
            HandlerEffectRoles {
                vsp: Register::R11,
                context_base: Register::RSP,
            },
            HandlerRegisterRoles {
                vip: Register::R15,
                rolling_key: Register::RBP,
                handler: Register::R14,
            },
            dispatch.encoded_load,
        );
        let operand_width = operands
            .candidates
            .first()
            .map(|candidate| candidate.codec.width);
        let analysis = classify_handler_effects(
            handler_entry,
            &effects,
            operand_width,
            i32::try_from(dispatch.vip_delta).ok(),
        );
        let (expected_semantic, expected_vsp_delta, expected_flags) = match handler_va {
            0x0003_7AA6_87C7 | 0x0003_7AA5_147E => (
                VmHandlerSemantic::ConstantPush,
                -8,
                EffectKnowledge::Preserved,
            ),
            0x0003_7AA4_F44E => (
                VmHandlerSemantic::VirtualRegisterRead,
                -8,
                EffectKnowledge::Preserved,
            ),
            0x0003_7AAA_7BA9 | 0x0003_7AA1_F283 => {
                (VmHandlerSemantic::Add, 0, EffectKnowledge::Modified)
            }
            0x0003_7AA7_B553 => (
                VmHandlerSemantic::VirtualRegisterWrite,
                8,
                EffectKnowledge::Preserved,
            ),
            0x0003_7AA7_0664 => (VmHandlerSemantic::MemoryLoad, 0, EffectKnowledge::Preserved),
            _ => unreachable!("fixture handler set is exhaustive"),
        };
        eprintln!("handler={handler_va:#X} effects={effects:?} analysis={analysis:?}");
        assert_eq!(analysis.semantic, expected_semantic);
        assert_eq!(analysis.stack_delta, Some(expected_vsp_delta));
        assert_eq!(analysis.flags_effect, expected_flags);
        assert!(analysis.confidence >= 90);
        let recovered = recover_handler_codecs(
            &pe,
            handler_entry,
            &HandlerCodecRecoveryConfig {
                roles: HandlerRegisterRoles {
                    vip: Register::R15,
                    rolling_key: Register::RBP,
                    handler: Register::R14,
                },
                bytecode_direction: BytecodeDirection::Ascending,
                effect_roles: Some(HandlerEffectRoles {
                    vsp: Register::R11,
                    context_base: Register::RSP,
                }),
                live_out_registers: vec![
                    Register::R11,
                    Register::R14,
                    Register::R15,
                    Register::RBP,
                ],
                live_out_rflags: 0,
                trace_limits: DirectJumpTraceLimits {
                    max_fragments: 1_024,
                    max_instructions: 50_000,
                    max_instructions_per_fragment: 10_000,
                },
                proof_limits: RegisterDispatchProofLimits::default(),
            },
        )
        .expect("composed codec recovery must complete");
        assert_eq!(recovered.status, RecoveryStatus::Recovered);
        assert!(recovered.dispatch.is_some());
        assert_eq!(recovered.operand.is_some(), expected_operands == 1);
        let decode_spec = recovered
            .decode_spec
            .as_ref()
            .expect("semantic effects and codecs must compose into a decoder spec");
        assert_eq!(decode_spec.semantic, expected_semantic);
        assert_eq!(decode_spec.vsp_delta, Some(expected_vsp_delta));
        assert_eq!(decode_spec.flags_effect, expected_flags);
        for instruction in handler_slice.instructions.iter().rev().take(12).rev() {
            eprintln!(
                "tail {} {:?} {}",
                instruction.rva, instruction.disposition, instruction.text
            );
        }
        assert!(report.candidates.len() <= 1);
        if let Some(candidate) = report.candidates.first() {
            assert_eq!(candidate.confidence, 100);
            recovered_handlers += 1;
        } else {
            assert!(
                !report.rejected_sources.is_empty(),
                "an unrecovered source must retain a rejection reason"
            );
        }
    }
    assert_eq!(recovered_handlers, 7);
}
