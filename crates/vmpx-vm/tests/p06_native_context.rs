use std::fs;

use vmpx_disasm::DirectJumpTraceLimits;
use vmpx_pe::{PeImage, Va};
use vmpx_vm::{
    BitWidth, DecodedVmInstruction, EffectKnowledge, Expr, KeyOperation, KeyedTransformCodec,
    NativeFrameSource, RecoveryStatus, RelativeDispatchCodec, RelativeDispatchState,
    RelativeThreadedCodec, Transform, TransformChain, TransformOp, VmHandlerSemantic,
    X64NativeRegister, analyze_x64_native_frame, map_native_context_scatter,
};

fn p06_context_scatter_codec() -> RelativeThreadedCodec {
    let keyed = |width, transforms: &[(TransformOp, u64)]| KeyedTransformCodec {
        width,
        input_key_operation: Some(KeyOperation::Xor),
        input_key_on_left: false,
        transforms: TransformChain {
            transforms: transforms
                .iter()
                .map(|(op, operand)| Transform {
                    op: *op,
                    width,
                    operand: *operand,
                })
                .collect(),
        },
        update_key_operation: Some(KeyOperation::Xor),
    };
    RelativeThreadedCodec {
        operand: keyed(
            BitWidth::W8,
            &[
                (TransformOp::Sub, 0x6D),
                (TransformOp::Rol, 1),
                (TransformOp::Add, 1),
                (TransformOp::Neg, 0),
            ],
        ),
        dispatch: RelativeDispatchCodec {
            displacement: keyed(
                BitWidth::W32,
                &[
                    (TransformOp::Sub, 1),
                    (TransformOp::Rol, 1),
                    (TransformOp::Sub, 1),
                    (TransformOp::Ror, 1),
                ],
            ),
        },
    }
}

#[test]
#[ignore = "controlled mapped-image fixture"]
#[allow(clippy::too_many_lines)]
fn recovers_ida_verified_p06_native_frame() {
    let bytes = fs::read(
        r"D:\vmp\testfilevmp\systems.throughvmpstatic.thatsalsoopeninidaprornthroughmcp.xpl",
    )
    .expect("controlled mapped-image fixture must exist");
    let pe = PeImage::parse(&bytes).expect("controlled fixture must parse");
    let report = analyze_x64_native_frame(
        &pe,
        pe.optional.address_of_entry_point,
        DirectJumpTraceLimits::default(),
    )
    .expect("p06 entry save chain must decode");
    assert_eq!(report.status, RecoveryStatus::Recovered, "{report:#?}");
    assert_eq!(report.cursor_register.as_deref(), Some("R11"));
    assert_eq!(report.slots.len(), 19);
    let expected = [
        NativeFrameSource::Constant { value: 0 },
        NativeFrameSource::Register {
            register: X64NativeRegister::Rdx,
        },
        NativeFrameSource::Register {
            register: X64NativeRegister::Rdi,
        },
        NativeFrameSource::Register {
            register: X64NativeRegister::Rbx,
        },
        NativeFrameSource::Register {
            register: X64NativeRegister::R13,
        },
        NativeFrameSource::Rflags,
        NativeFrameSource::Register {
            register: X64NativeRegister::R15,
        },
        NativeFrameSource::Register {
            register: X64NativeRegister::Rbp,
        },
        NativeFrameSource::Register {
            register: X64NativeRegister::R11,
        },
        NativeFrameSource::Register {
            register: X64NativeRegister::Rsi,
        },
        NativeFrameSource::Register {
            register: X64NativeRegister::Rax,
        },
        NativeFrameSource::Register {
            register: X64NativeRegister::R10,
        },
        NativeFrameSource::Register {
            register: X64NativeRegister::R12,
        },
        NativeFrameSource::Register {
            register: X64NativeRegister::Rcx,
        },
        NativeFrameSource::Register {
            register: X64NativeRegister::R8,
        },
        NativeFrameSource::Register {
            register: X64NativeRegister::R9,
        },
        NativeFrameSource::Register {
            register: X64NativeRegister::R14,
        },
        NativeFrameSource::ReturnAddress {
            call_site: report.call_site.expect("call site"),
            continuation: report
                .slots
                .get(17)
                .and_then(|slot| match slot.source {
                    NativeFrameSource::ReturnAddress { continuation, .. } => Some(continuation),
                    _ => None,
                })
                .expect("return continuation"),
        },
        NativeFrameSource::EntryImmediate {
            ordinal: 0,
            value: 0x15FC_D88B,
        },
    ];
    assert_eq!(
        report
            .slots
            .iter()
            .map(|slot| slot.source.clone())
            .collect::<Vec<_>>(),
        expected
    );

    let codec = p06_context_scatter_codec();
    let mut state = RelativeDispatchState {
        vip: Va(0x0003_7A80_C9E2),
        rolling_key: 0x3_8572_2D91,
        handler: Va(0x0003_7A99_26C0),
    };
    let mut scatter = Vec::new();
    for _ in 0..report.slots.len() {
        let decoded = codec
            .decode_operand_dispatch(&pe, state)
            .expect("IDA-verified p06 context scatter must decode");
        scatter.push(DecodedVmInstruction {
            state_before: decoded.state_before,
            encoded_bytes: decoded
                .encoded_operand
                .iter()
                .chain(&decoded.encoded_displacement)
                .copied()
                .collect(),
            semantic: VmHandlerSemantic::VirtualRegisterWrite,
            semantic_metadata: None,
            input_width: Some(BitWidth::W64),
            operation_width: Some(BitWidth::W64),
            resize: None,
            operand: Some(decoded.operand),
            vsp_delta: Some(8),
            flags_effect: EffectKnowledge::Preserved,
            confidence: 100,
            state_after: decoded.state_after,
        });
        state = decoded.state_after;
    }
    let mapping = map_native_context_scatter(&report, &scatter);
    assert_eq!(mapping.status, RecoveryStatus::Recovered, "{mapping:#?}");
    let rdx = mapping
        .bindings
        .iter()
        .find(|binding| {
            binding.source
                == NativeFrameSource::Register {
                    register: X64NativeRegister::Rdx,
                }
        })
        .expect("RDX binding");
    assert_eq!(rdx.context_offset, 0x90);
    let last = mapping.bindings.last().expect("entry seed binding");
    assert_eq!(last.context_offset, 0xA0);
    assert_eq!(last.overwrites_binding, Some(17));
    let initial = mapping
        .symbolic_initial_state()
        .expect("complete p06 symbolic native context");
    assert_eq!(
        initial.context_values[&0x90],
        Expr::register(BitWidth::W64, "native_rdx_initial")
    );
    assert_eq!(
        initial.context_values[&0xA0],
        Expr::constant(BitWidth::W64, 0x15FC_D88B)
    );
}
