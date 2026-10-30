use std::{collections::BTreeMap, fs};

use iced_x86::Register;
use vmpx_disasm::{CfgLimits, DirectJumpTraceLimits, RegisterDispatchProofLimits, recover_cfg};
use vmpx_pe::{PeImage, Rva, Va};
use vmpx_vm::{
    BitWidth, BytecodeDirection, DecodedVmInstruction, Expr, ExprKind, HandlerCodecRecoveryConfig,
    HandlerEffectRoles, HandlerRegisterRoles, KeyOperation, KeyedTransformCodec, LinearDecodeStop,
    NativeCallImportResolution, RecoveryStatus, RelativeDispatchCodec, RelativeDispatchState,
    RelativeThreadedCodec, SymbolicExecutionLimits, Transform, TransformChain, TransformOp,
    VmHandlerSemantic, VmProfile, VmProfileChainReport, VmProfileExecutionState,
    VmProfileExplorerLimits, VmProfilePathLimits, VmProfileRoleResolutionConfig, X64NativeRegister,
    analyze_native_call_handler, analyze_vm_transition, analyze_x64_native_frame,
    assemble_profile_path_graph, execute_symbolic_linear_block_with_initial_state,
    explore_vm_profile_chain, explore_vm_profile_paths, map_native_context_scatter,
    recover_and_decode_linear_vm_stream, recover_native_exit_state, resolve_vm_profile_roles,
};

fn print_chain_effect_summary(label: &str, chain: &VmProfileChainReport) {
    let writes = chain
        .segments
        .iter()
        .flat_map(|segment| segment.symbolic.iter())
        .flat_map(|symbolic| symbolic.external_writes.iter())
        .collect::<Vec<_>>();
    let calls = chain
        .segments
        .iter()
        .flat_map(|segment| segment.boundary_symbolic.iter())
        .flat_map(|symbolic| symbolic.native_calls.iter())
        .collect::<Vec<_>>();
    let havocs = chain
        .segments
        .iter()
        .flat_map(|segment| {
            segment
                .symbolic
                .iter()
                .chain(segment.boundary_symbolic.iter())
        })
        .flat_map(|symbolic| symbolic.memory_havocs.iter())
        .count();
    eprintln!(
        "p06 {label} effects: {} external writes, {} native calls, {} memory havocs, terminal VSP offset {:?}",
        writes.len(),
        calls.len(),
        havocs,
        chain
            .segments
            .last()
            .and_then(|segment| segment.symbolic.as_ref())
            .and_then(|symbolic| symbolic.final_vsp_offset),
    );
    for write in writes {
        eprintln!(
            "p06 {label} write @{}: address {:?}, deps {:?}, intrinsics {:?}; value {} nodes, deps {:?}, intrinsics {:?}",
            write.instruction_index,
            write.address,
            write.address.dependencies(),
            write.address.intrinsic_dependencies(),
            write.value.node_count_bounded(200_000),
            write.value.dependencies(),
            write.value.intrinsic_dependencies(),
        );
    }
    for call in calls {
        eprintln!(
            "p06 {label} call @{}: import {:?}, invocation {:?}, effect {:?}",
            call.instruction_index, call.import_resolution, call.invocation, call.applied_effect,
        );
    }
}

fn xor_codec(width: BitWidth, transforms: &[(TransformOp, u64)]) -> KeyedTransformCodec {
    KeyedTransformCodec {
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
    }
}

fn p06_context_scatter_codec() -> RelativeThreadedCodec {
    RelativeThreadedCodec {
        operand: xor_codec(
            BitWidth::W8,
            &[
                (TransformOp::Sub, 0x6D),
                (TransformOp::Rol, 1),
                (TransformOp::Add, 1),
                (TransformOp::Neg, 0),
            ],
        ),
        dispatch: RelativeDispatchCodec {
            table: None,
            displacement: xor_codec(
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

fn p06_constant_push_codec() -> RelativeThreadedCodec {
    RelativeThreadedCodec {
        operand: xor_codec(
            BitWidth::W64,
            &[
                (TransformOp::Sub, 1),
                (TransformOp::Bswap, 0),
                (TransformOp::Add, 0xFFFF_FFFF_F5F6_F2BA),
                (TransformOp::Neg, 0),
            ],
        ),
        dispatch: RelativeDispatchCodec {
            table: None,
            displacement: xor_codec(
                BitWidth::W32,
                &[
                    (TransformOp::Ror, 3),
                    (TransformOp::Add, 1),
                    (TransformOp::Xor, 0xEFDF_DCCF),
                    (TransformOp::Sub, 1),
                    (TransformOp::Xor, 0x7FFF_9560),
                ],
            ),
        },
    }
}

fn p06_register_read_codec() -> RelativeThreadedCodec {
    RelativeThreadedCodec {
        operand: xor_codec(
            BitWidth::W8,
            &[
                (TransformOp::Sub, 0xCB),
                (TransformOp::Rol, 1),
                (TransformOp::Sub, 0xB0),
                (TransformOp::Ror, 1),
                (TransformOp::Sub, 1),
                (TransformOp::Not, 0),
                (TransformOp::Rol, 1),
            ],
        ),
        dispatch: RelativeDispatchCodec {
            table: None,
            displacement: xor_codec(
                BitWidth::W32,
                &[
                    (TransformOp::Rol, 3),
                    (TransformOp::Bswap, 0),
                    (TransformOp::Sub, 1),
                    (TransformOp::Rol, 2),
                ],
            ),
        },
    }
}

fn p06_add_dispatch_codec() -> RelativeDispatchCodec {
    RelativeDispatchCodec {
        table: None,
        displacement: xor_codec(
            BitWidth::W32,
            &[
                (TransformOp::Ror, 1),
                (TransformOp::Neg, 0),
                (TransformOp::Add, 0xF638_338D),
                (TransformOp::Xor, 0xF6DF_F93A),
            ],
        ),
    }
}

fn p06_bootstrap_dispatch_codec() -> RelativeDispatchCodec {
    RelativeDispatchCodec {
        table: None,
        displacement: xor_codec(
            BitWidth::W32,
            &[
                (TransformOp::Ror, 2),
                (TransformOp::Bswap, 0),
                (TransformOp::Neg, 0),
                (TransformOp::Add, 1),
                (TransformOp::Not, 0),
                (TransformOp::Sub, 0x76DD_402B),
            ],
        ),
    }
}

#[test]
#[ignore = "large controlled mapped-image fixture"]
#[allow(clippy::too_many_lines)]
fn decodes_ida_verified_p06_basic_block_through_virtual_branch() {
    // All sample addresses and tokens remain confined to this research
    // regression. The production decoder is configured solely from recovered
    // transform chains and receives no fixture-specific constants.
    let bytes = fs::read(
        r"D:\vmp\testfilevmp\systems.throughvmpstatic.thatsalsoopeninidaprornthroughmcp.xpl",
    )
    .expect("controlled mapped-image fixture must exist");
    let pe = PeImage::parse(&bytes).expect("controlled fixture must parse");
    let native_call = analyze_native_call_handler(
        &pe,
        Rva(0xF01_D126),
        HandlerRegisterRoles {
            vip: Register::RSI,
            rolling_key: Register::R12,
            handler: Register::R11,
        },
        Register::R9,
        Register::RDI,
        Rva(251_777_371),
        CfgLimits {
            max_blocks: 4_096,
            max_instructions: 50_000,
            max_instructions_per_block: 4_096,
        },
    )
    .expect("native-call bridge CFG must decode");
    let native_call = native_call.recovered.unwrap_or_else(|| {
        panic!(
            "native-call bridge must be proven: {:#?}",
            native_call.report
        )
    });
    assert_eq!(native_call.report.call_site, Some(Rva(0xF0F_C046)));
    assert_eq!(native_call.resume_rva, Rva(0xF0F_C048));
    assert_eq!(native_call.report.status, RecoveryStatus::Recovered);
    assert_eq!(native_call.report.confidence, 100);
    assert_eq!(native_call.report.selector_outcomes.len(), 256);
    assert_eq!(native_call.report.selector_outcomes[2].vsp_delta, 16);
    assert_eq!(
        native_call.report.selector_outcomes[2].register_argument_offsets,
        vec![8, 16]
    );
    assert_eq!(native_call.report.selector_outcomes[5].vsp_delta, 32);
    assert_eq!(
        native_call.report.selector_outcomes[5].stack_argument_offsets,
        vec![72]
    );
    if let Some(output) = std::env::var_os("VMPX_P06_NATIVE_CALL_CFG") {
        let cfg = recover_cfg(
            &pe,
            Rva(0xF01_D126),
            CfgLimits {
                max_blocks: 4_096,
                max_instructions: 50_000,
                max_instructions_per_block: 4_096,
            },
        )
        .expect("native-call bridge CFG must decode for a research dump");
        fs::write(
            output,
            serde_json::to_vec_pretty(&cfg).expect("native-call CFG must serialize"),
        )
        .expect("native-call CFG report must be written");
    }

    let scatter = p06_context_scatter_codec();
    let mut state = RelativeDispatchState {
        vip: Va(0x0003_7A80_C9E2),
        rolling_key: 0x3_8572_2D91,
        handler: Va(0x0003_7A99_26C0),
    };

    let expected = [
        (
            0x48,
            0x3_8562_C6C8,
            Va(0x0003_7AAA_11D1),
            Va(0x0003_7A80_C9E7),
        ),
        (
            0x90,
            0x3_8560_C09E,
            Va(0x0003_7AAC_1897),
            Va(0x0003_7A80_C9EC),
        ),
        (
            0x30,
            0x3_7A8C_CB31,
            Va(0x0003_7A98_2436),
            Va(0x0003_7A80_C9F1),
        ),
        (
            0x88,
            0x3_7A82_BAC9,
            Va(0x0003_7AA6_95A6),
            Va(0x0003_7A80_C9F6),
        ),
        (
            0x38,
            0x3_7A87_3242,
            Va(0x0003_7AAC_1E59),
            Va(0x0003_7A80_C9FB),
        ),
        (
            0x78,
            0x3_856C_89F2,
            Va(0x0003_7A97_DA21),
            Va(0x0003_7A80_CA00),
        ),
        (
            0x80,
            0x3_8560_4014,
            Va(0x0003_7AA4_A387),
            Va(0x0003_7A80_CA05),
        ),
        (
            0x28,
            0x3_8563_51F0,
            Va(0x0003_7AA7_B553),
            Va(0x0003_7A80_CA0A),
        ),
        (
            0x00,
            0x3_7A90_9E7C,
            Va(0x0003_7A9B_84DF),
            Va(0x0003_7A80_CA0F),
        ),
        (
            0x70,
            0x3_7A9A_0ED6,
            Va(0x0003_7AA6_15B9),
            Va(0x0003_7A80_CA14),
        ),
        (
            0xA8,
            0x3_8569_1F79,
            Va(0x0003_7A99_26C0),
            Va(0x0003_7A80_CA19),
        ),
        (
            0x50,
            0x3_8579_F438,
            Va(0x0003_7AAA_11D1),
            Va(0x0003_7A80_CA1E),
        ),
        (
            0x20,
            0x3_857B_F2DE,
            Va(0x0003_7AAC_1897),
            Va(0x0003_7A80_CA23),
        ),
        (
            0x10,
            0x3_7A97_F951,
            Va(0x0003_7A98_2436),
            Va(0x0003_7A80_CA28),
        ),
        (
            0x40,
            0x3_7A99_8861,
            Va(0x0003_7AA6_95A6),
            Va(0x0003_7A80_CA2D),
        ),
        (
            0xB8,
            0x3_7A9C_006A,
            Va(0x0003_7AAC_1E59),
            Va(0x0003_7A80_CA32),
        ),
        (
            0x60,
            0x3_8577_BBC2,
            Va(0x0003_7A97_DA21),
            Va(0x0003_7A80_CA37),
        ),
        (
            0xA0,
            0x3_857B_7204,
            Va(0x0003_7AA4_A387),
            Va(0x0003_7A80_CA3C),
        ),
        (
            0xA0,
            0x3_857A_96E4,
            Va(0x0003_7AA6_87C7),
            Va(0x0003_7A80_CA41),
        ),
    ];

    let mut scatter_instructions = Vec::new();
    for (operand, key, handler, vip) in expected {
        let decoded = scatter
            .decode_operand_dispatch(&pe, state)
            .expect("IDA-verified context-scatter transition must decode");
        assert_eq!(decoded.operand, operand);
        assert_eq!(decoded.state_after.rolling_key, key);
        assert_eq!(decoded.state_after.handler, handler);
        assert_eq!(decoded.state_after.vip, vip);
        scatter_instructions.push(DecodedVmInstruction {
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
            flags_effect: vmpx_vm::EffectKnowledge::Preserved,
            confidence: 100,
            state_after: decoded.state_after,
        });
        state = decoded.state_after;
    }

    let constant_push = p06_constant_push_codec();
    let h19 = constant_push
        .decode_operand_dispatch(&pe, state)
        .expect("first constant push must decode");
    assert_eq!(h19.operand, 0x9F74_F8D0_0110_6D74);
    assert_eq!(h19.state_after.rolling_key, 0x9F74_F8D3_7B94_7727);
    assert_eq!(h19.state_after.handler, Va(0x0003_7AA5_147E));
    assert_eq!(h19.state_after.vip, Va(0x0003_7A80_CA4D));

    let h20 = constant_push
        .decode_operand_dispatch(&pe, h19.state_after)
        .expect("second constant push must decode");
    assert_eq!(h20.operand, 0x0000_0003_7A9E_EA3C);
    assert_eq!(h20.state_after.rolling_key, 0x9F74_F8D0_FEF5_42CB);
    assert_eq!(h20.state_after.handler, Va(0x0003_7AA4_F44E));
    assert_eq!(h20.state_after.vip, Va(0x0003_7A80_CA59));

    let register_read = p06_register_read_codec()
        .decode_operand_dispatch(&pe, h20.state_after)
        .expect("context-slot read must decode");
    assert_eq!(register_read.operand, 0x48);
    assert_eq!(register_read.state_after.rolling_key, 0x9F74_F8D0_FEF0_C5D8);
    assert_eq!(register_read.state_after.handler, Va(0x0003_7AAA_7BA9));
    assert_eq!(register_read.state_after.vip, Va(0x0003_7A80_CA5E));

    let add_dispatch = p06_add_dispatch_codec()
        .decode(&pe, register_read.state_after)
        .expect("operandless add dispatch must decode");
    assert_eq!(add_dispatch.displacement, 0xFFFD_39AA);
    assert_eq!(add_dispatch.state_after.rolling_key, 0x9F74_F8D0_010D_FC72);
    assert_eq!(add_dispatch.state_after.handler, Va(0x0003_7AA7_B553));
    assert_eq!(add_dispatch.state_after.vip, Va(0x0003_7A80_CA62));

    let recovery = recover_and_decode_linear_vm_stream(
        &pe,
        state,
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
            live_out_registers: vec![Register::R11, Register::R14, Register::R15, Register::RBP],
            live_out_rflags: 0,
            trace_limits: DirectJumpTraceLimits::default(),
            proof_limits: RegisterDispatchProofLimits::default(),
        },
        512,
    )
    .expect("native handler recovery and semantic decoding must complete");
    let report = &recovery.decode;
    if let Some(output) = std::env::var_os("VMPX_P06_REPORT") {
        let encoded = serde_json::to_vec_pretty(&recovery).expect("report must serialize");
        fs::write(output, encoded).expect("requested report path must be writable");
    }
    assert_eq!(report.stop_reason, LinearDecodeStop::ControlFlowBoundary);
    assert_eq!(report.instructions.len(), 302);
    assert_eq!(recovery.handler_recoveries.len(), 102);
    assert_eq!(report.cumulative_vsp_delta, None);
    assert_eq!(report.final_state.vip, Va(0x0003_7A80_D0BD));
    assert_eq!(report.final_state.handler, Va(0x0003_7AA4_44FB));
    assert_eq!(report.unresolved_handler, None);
    assert!(
        recovery
            .handler_recoveries
            .iter()
            .filter(|(handler, _)| **handler != report.final_state.handler)
            .all(|(_, handler)| {
                handler.status == RecoveryStatus::Recovered
                    && handler.decode_spec.is_some()
                    && handler
                        .value_flow
                        .as_ref()
                        .is_some_and(|proof| proof.proven)
                    && handler.confidence > 0
            })
    );
    let boundary = recovery
        .handler_recoveries
        .get(&report.final_state.handler)
        .expect("control-flow handler recovery is retained");
    assert_eq!(boundary.status, RecoveryStatus::PartiallyRecovered);
    assert_eq!(
        boundary
            .handler_analysis
            .as_ref()
            .map(|analysis| analysis.semantic),
        Some(VmHandlerSemantic::UnconditionalBranch)
    );
    assert_eq!(
        report
            .instructions
            .iter()
            .take(12)
            .map(|instruction| instruction.semantic)
            .collect::<Vec<_>>(),
        vec![
            VmHandlerSemantic::ConstantPush,
            VmHandlerSemantic::ConstantPush,
            VmHandlerSemantic::VirtualRegisterRead,
            VmHandlerSemantic::Add,
            VmHandlerSemantic::VirtualRegisterWrite,
            VmHandlerSemantic::MemoryLoad,
            VmHandlerSemantic::Add,
            VmHandlerSemantic::VirtualRegisterWrite,
            VmHandlerSemantic::VirtualRegisterRead,
            VmHandlerSemantic::Add,
            VmHandlerSemantic::VirtualRegisterWrite,
            VmHandlerSemantic::VirtualRegisterWrite,
        ]
    );

    let native_frame = analyze_x64_native_frame(
        &pe,
        pe.optional.address_of_entry_point,
        DirectJumpTraceLimits::default(),
    )
    .expect("entry native-frame analysis must decode");
    let native_context = map_native_context_scatter(&native_frame, &scatter_instructions);
    let symbolic_initial = native_context
        .symbolic_initial_state()
        .expect("proven entry frame and context scatter must produce native symbolic inputs");
    let symbolic = execute_symbolic_linear_block_with_initial_state(
        &report.instructions,
        Some(VmHandlerSemantic::UnconditionalBranch),
        SymbolicExecutionLimits::default(),
        &symbolic_initial,
        Some(&pe),
    )
    .expect("recovered semantic instructions must symbolically execute");
    if let Some(output) = std::env::var_os("VMPX_P06_SYMBOLIC_REPORT") {
        let encoded = serde_json::to_vec_pretty(&symbolic).expect("report must serialize");
        fs::write(output, encoded).expect("requested symbolic report path must be writable");
    }
    assert_eq!(symbolic.executed_instructions, 302);
    assert!(symbolic.branch_target.is_some());
    assert!(symbolic.resolved_image_bytes > 0);
    assert_eq!(
        symbolic.branch_target_dependencies,
        vec!["native_rdx_initial".to_owned()]
    );
    let branch_target = symbolic.branch_target.as_ref().expect("target exists");
    let mut successor_reasons = BTreeMap::<Va, Vec<u64>>::new();
    for reason in 0_u64..=3 {
        let evaluated = branch_target
            .substitute(&BTreeMap::from([(
                "native_rdx_initial".to_owned(),
                Expr::constant(BitWidth::W64, reason),
            )]))
            .expect("reason substitution preserves widths");
        let ExprKind::Const { value } = evaluated.kind else {
            panic!("reason {reason} did not resolve to a concrete target: {evaluated:?}");
        };
        eprintln!("p06 reason {reason}: virtual successor {value:#X}");
        pe.optional
            .image_base
            .checked_rva_of(Va(value))
            .and_then(|rva| pe.rva_to_file_offset(rva).ok())
            .expect("resolved successor must be backed by the mapped image");
        successor_reasons.entry(Va(value)).or_default().push(reason);
    }

    for (successor_index, (successor, reasons)) in successor_reasons.into_iter().enumerate() {
        let dispatched = p06_bootstrap_dispatch_codec()
            .decode(
                &pe,
                RelativeDispatchState {
                    vip: successor,
                    rolling_key: successor.get(),
                    handler: Va(0x0003_7AA6_4271),
                },
            )
            .expect("branch postlude must initialize successor dispatch");
        let successor_recovery = recover_and_decode_linear_vm_stream(
            &pe,
            dispatched.state_after,
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
                trace_limits: DirectJumpTraceLimits::default(),
                proof_limits: RegisterDispatchProofLimits::default(),
            },
            512,
        )
        .expect("successor block semantic decoding must complete");
        if let Some(directory) = std::env::var_os("VMPX_P06_SUCCESSOR_REPORT_DIR") {
            fs::create_dir_all(&directory).expect("successor report directory must be writable");
            let output =
                std::path::Path::new(&directory).join(format!("successor-{successor_index}.json"));
            let encoded =
                serde_json::to_vec_pretty(&successor_recovery).expect("report must serialize");
            fs::write(output, encoded).expect("successor report path must be writable");
        }
        eprintln!(
            "p06 successor {successor}: {} operations, stop {:?}, next VIP {}, handler {}",
            successor_recovery.decode.instructions.len(),
            successor_recovery.decode.stop_reason,
            successor_recovery.decode.final_state.vip,
            successor_recovery.decode.final_state.handler,
        );
        if let Some(handler) = successor_recovery.decode.unresolved_handler
            && let Some(recovery) = successor_recovery.handler_recoveries.get(&handler)
        {
            eprintln!(
                "p06 unresolved {handler}: status {:?}, semantic {:?}, diagnostics {:?}",
                recovery.status,
                recovery
                    .handler_analysis
                    .as_ref()
                    .map(|analysis| analysis.semantic),
                recovery.diagnostics,
            );
            let transition_rva = pe
                .optional
                .image_base
                .checked_rva_of(handler)
                .expect("transition handler must belong to the mapped image");
            let transition = analyze_vm_transition(
                &pe,
                transition_rva,
                Register::R11,
                Register::RSP,
                DirectJumpTraceLimits::default(),
            )
            .expect("transition native trace must decode");
            eprintln!("p06 transition {handler}: {:#?}", transition.report);
            let recovered = transition
                .recovered
                .expect("cross-VM transition must be behaviorally recovered");
            assert_eq!(
                recovered.report.bytecode_direction,
                BytecodeDirection::Descending
            );
            assert_eq!(recovered.report.old_vsp_delta, Some(8));
            assert_eq!(recovered.report.confidence, 100);
            assert!(!recovered.vsp_candidates.is_empty());

            let successor_symbolic = execute_symbolic_linear_block_with_initial_state(
                &successor_recovery.decode.instructions,
                None,
                SymbolicExecutionLimits::default(),
                &symbolic.continuation_state(),
                Some(&pe),
            )
            .expect("successor semantics must continue from the predecessor VM state");
            let transition_pop = execute_symbolic_linear_block_with_initial_state(
                &[],
                Some(VmHandlerSemantic::UnconditionalBranch),
                SymbolicExecutionLimits::default(),
                &successor_symbolic.continuation_state(),
                Some(&pe),
            )
            .expect("proven transition pop must read the retained VM stack");
            let transitioned_symbolic_state = transition_pop.continuation_state();
            let transition_target = transition_pop
                .branch_target
                .as_ref()
                .expect("transition target must be present");
            let ExprKind::Const {
                value: transition_target,
            } = &transition_target.kind
            else {
                panic!("transition target did not resolve concretely: {transition_target:?}");
            };
            let transition_target = *transition_target;
            let initial_profile_state = RelativeDispatchState {
                vip: Va(transition_target),
                rolling_key: transition_target,
                handler: recovered.handler_anchor,
            };
            let first_dispatch = recovered
                .dispatch
                .decode_direction(&pe, initial_profile_state, BytecodeDirection::Descending)
                .expect("recovered descending transition dispatch must decode");
            pe.optional
                .image_base
                .checked_rva_of(first_dispatch.state_after.handler)
                .and_then(|rva| pe.rva_to_file_offset(rva).ok())
                .expect("transition dispatch handler must be file-backed");
            eprintln!(
                "p06 transitioned profile: target {:#X}, first handler {}, next descending VIP {}, key {:#X}",
                transition_target,
                first_dispatch.state_after.handler,
                first_dispatch.state_after.vip,
                first_dispatch.state_after.rolling_key,
            );
            let first_handler_rva = pe
                .optional
                .image_base
                .checked_rva_of(first_dispatch.state_after.handler)
                .expect("transition handler must belong to the mapped image");
            let role_resolution = resolve_vm_profile_roles(
                &pe,
                first_handler_rva,
                &VmProfileRoleResolutionConfig {
                    roles: recovered.roles,
                    bytecode_direction: recovered.report.bytecode_direction,
                    vsp_candidates: recovered.vsp_candidates.clone(),
                    context_base: Register::RSP,
                    trace_limits: DirectJumpTraceLimits::default(),
                    proof_limits: RegisterDispatchProofLimits::default(),
                },
            )
            .expect("first transitioned handler must trace");
            let resolved_profile = role_resolution
                .recovered
                .expect("handler semantics must resolve one VSP candidate");
            assert!(
                recovered
                    .vsp_candidates
                    .contains(&resolved_profile.effect_roles.vsp)
            );
            let chain_profile = VmProfile {
                handler_roles: resolved_profile.handler_roles,
                effect_roles: resolved_profile.effect_roles,
                bytecode_direction: resolved_profile.bytecode_direction,
            };
            let chain_limits = VmProfileExplorerLimits {
                max_segments: 32,
                max_instructions_per_segment: 2_048,
                max_total_instructions: 8_192,
                ..VmProfileExplorerLimits::default()
            };
            let chain_initial = VmProfileExecutionState {
                decoder: first_dispatch.state_after,
                profile: chain_profile,
                symbolic: transitioned_symbolic_state.clone(),
            };
            let chain = explore_vm_profile_chain(&pe, chain_initial.clone(), chain_limits)
                .expect("bounded profile-chain exploration must complete");
            assert!(chain.segments.len() >= 2);
            if chain.stop_reason == vmpx_vm::VmProfileChainStop::SymbolicTransitionTarget {
                let reason_domain = reasons
                    .iter()
                    .copied()
                    .map(|reason| Expr::constant(BitWidth::W64, reason))
                    .collect();
                let path_family = explore_vm_profile_paths(
                    &pe,
                    &chain_initial,
                    &BTreeMap::from([("native_rdx_initial".to_owned(), reason_domain)]),
                    VmProfilePathLimits {
                        max_paths: 16,
                        chain: chain_limits,
                        ..VmProfilePathLimits::default()
                    },
                )
                .expect("bounded DLL-reason domain must explore every profile path");
                assert!(path_family.domain_complete);
                assert!(!path_family.all_paths_reach_vm_exit);
                assert_eq!(path_family.paths.len(), reasons.len());
                let path_graph = assemble_profile_path_graph(&path_family)
                    .expect("exact-state profile paths must form a bounded structural graph");
                assert_eq!(path_graph.path_count, reasons.len());
                assert_eq!(path_graph.terminals.len(), reasons.len());
                assert!(path_graph.domain_complete);
                assert!(!path_graph.all_paths_reach_vm_exit);
                assert!(path_graph.nodes.len() <= path_graph.segment_occurrences);
                eprintln!(
                    "p06 exact-state graph: {} nodes, {} edges, {} segment occurrences, {} exact merges",
                    path_graph.nodes.len(),
                    path_graph.edges.len(),
                    path_graph.segment_occurrences,
                    path_graph.exactly_merged_occurrences,
                );
                if let Some(output) = std::env::var_os("VMPX_P06_PATH_FAMILY") {
                    fs::write(
                        output,
                        serde_json::to_vec_pretty(&path_family)
                            .expect("path family must serialize"),
                    )
                    .expect("path-family research report must be written");
                }
                let enumeration = path_family
                    .enumerations
                    .first()
                    .expect("symbolic transition must retain its finite enumeration");
                assert_eq!(enumeration.combinations_evaluated, reasons.len());
                assert!(!enumeration.unique_values.is_empty());
                for target in &enumeration.unique_values {
                    pe.optional
                        .image_base
                        .checked_rva_of(Va(*target))
                        .and_then(|rva| pe.rva_to_file_offset(rva).ok())
                        .expect("enumerated transition target must be image-backed");
                }
                eprintln!(
                    "p06 finite-domain transition successors for reasons {:?}: {:?}",
                    reasons, enumeration.unique_values
                );
                for path in &path_family.paths {
                    let reason = path.assignments["native_rdx_initial"];
                    print_chain_effect_summary(&format!("reason {reason}"), &path.chain);
                    eprintln!(
                        "p06 reason {reason} specialized chain: {} segments, {} operations, stop {:?}, handler {:?}, diagnostics {:?}",
                        path.chain.segments.len(),
                        path.chain.total_instructions,
                        path.chain.stop_reason,
                        path.chain
                            .segments
                            .last()
                            .and_then(|segment| segment.decode.decode.unresolved_handler),
                        path.chain.diagnostics,
                    );
                    if path.chain.stop_reason != vmpx_vm::VmProfileChainStop::VmExit {
                        if let Some(partial) = path
                            .chain
                            .segments
                            .last()
                            .and_then(|segment| segment.symbolic.as_ref())
                        {
                            eprintln!(
                                "p06 reason {reason} exception scratch C0={:?} C8={:?}",
                                partial.context_cells.get(&0xC0),
                                partial.context_cells.get(&0xC8),
                            );
                        }
                        continue;
                    }
                    let terminal = path
                        .chain
                        .segments
                        .last()
                        .expect("VM-exit path must retain its terminal segment");
                    let exit = terminal
                        .decode
                        .handler_recoveries
                        .get(&terminal.decode.decode.final_state.handler)
                        .and_then(|recovery| recovery.vm_exit.as_ref())
                        .expect("terminal handler must retain its proven VM-exit layout");
                    let native_exit = recover_native_exit_state(
                        terminal
                            .symbolic
                            .as_ref()
                            .expect("terminal segment must retain symbolic memory"),
                        exit,
                    )
                    .expect("VM-exit POP frame must resolve to exact symbolic values");
                    assert_eq!(native_exit.registers.len(), 15);
                    let rax = &native_exit.registers[&X64NativeRegister::Rax];
                    eprintln!(
                        "p06 reason {reason} native RAX: {rax:?}, {} nodes, dependencies {:?}",
                        rax.node_count_bounded(200_000),
                        rax.dependencies(),
                    );
                    eprintln!(
                        "p06 reason {reason} native continuation: {} nodes, dependencies {:?}",
                        native_exit.return_address.node_count_bounded(200_000),
                        native_exit.return_address.dependencies(),
                    );
                }
            }
            if chain.stop_reason == vmpx_vm::VmProfileChainStop::VmExit {
                print_chain_effect_summary("primary successor", &chain);
                let terminal = chain
                    .segments
                    .last()
                    .expect("VM-exit chain must retain its terminal segment");
                let exit = terminal
                    .decode
                    .handler_recoveries
                    .get(&terminal.decode.decode.final_state.handler)
                    .and_then(|recovery| recovery.vm_exit.as_ref())
                    .expect("terminal handler must retain its proven VM-exit layout");
                let native_exit = recover_native_exit_state(
                    terminal
                        .symbolic
                        .as_ref()
                        .expect("terminal segment must retain symbolic memory"),
                    exit,
                )
                .expect("VM-exit POP frame must resolve to exact symbolic values");
                assert_eq!(native_exit.registers.len(), 15);
                let rax = &native_exit.registers[&X64NativeRegister::Rax];
                eprintln!(
                    "p06 primary successor native RAX: {rax:?}, {} nodes, dependencies {:?}",
                    rax.node_count_bounded(200_000),
                    rax.dependencies(),
                );
                eprintln!(
                    "p06 primary successor native continuation: {} nodes, dependencies {:?}",
                    native_exit.return_address.node_count_bounded(200_000),
                    native_exit.return_address.dependencies(),
                );
                for register in [
                    X64NativeRegister::Rbx,
                    X64NativeRegister::Rbp,
                    X64NativeRegister::Rdi,
                    X64NativeRegister::Rsi,
                    X64NativeRegister::R12,
                    X64NativeRegister::R13,
                    X64NativeRegister::R14,
                    X64NativeRegister::R15,
                ] {
                    eprintln!(
                        "p06 primary successor native {register:?}: {:?}",
                        native_exit.registers[&register]
                    );
                }
                let symbolic = terminal
                    .symbolic
                    .as_ref()
                    .expect("primary terminal symbolic state");
                for offset in [0xC0, 0xC8, 0x138, 0x140] {
                    eprintln!(
                        "p06 primary successor context {offset:#X}: {:?}",
                        symbolic.context_cells.get(&offset)
                    );
                }
            }
            if successor_index == 0 {
                let native_call_segment = chain
                    .segments
                    .iter()
                    .find(|segment| segment.boundary_decode.is_some())
                    .expect("primary p06 chain must cross the proven native-call boundary");
                let boundary = native_call_segment
                    .boundary_decode
                    .as_ref()
                    .expect("located native-call boundary must remain present");
                assert_eq!(boundary.selector, 2);
                assert_eq!(boundary.outcome.vsp_delta, 16);
                assert_eq!(boundary.state_after.vip, Va(0x3791_64DBC));
                assert_eq!(boundary.state_after.handler, Va(0x37A9_C9CC6));
                let symbolic = native_call_segment
                    .boundary_symbolic
                    .as_ref()
                    .expect("native-call boundary must have a conservative symbolic transfer");
                assert_eq!(symbolic.final_vsp_offset, Some(-1_760));
                assert!(symbolic.memory_havocs.is_empty());
                assert_eq!(symbolic.fresh_allocations.len(), 1);
                let [call] = symbolic.native_calls.as_slice() else {
                    panic!("native-call boundary must report exactly one invocation")
                };
                let Some(NativeCallImportResolution::Unique { import, .. }) =
                    call.import_resolution.as_ref()
                else {
                    panic!("fixture native call must resolve to one parsed import")
                };
                assert_eq!(import.module.to_ascii_lowercase(), "kernel32.dll");
                assert_eq!(import.name.as_deref(), Some("LocalAlloc"));
                assert!(!symbolic.memory_tainted);
                assert!(chain.total_instructions > 1_543);
            }
            eprintln!(
                "p06 profile chain from {successor}: {} segments, {} semantic operations, stop {:?}",
                chain.segments.len(),
                chain.total_instructions,
                chain.stop_reason,
            );
            for segment in &chain.segments {
                eprintln!(
                    "  segment {}: {:?}, VSP {}, {} operations, stop {:?}, handler {:?}, transition {:?}",
                    segment.index,
                    segment.profile.bytecode_direction,
                    segment.profile.vsp_register,
                    segment.decode.decode.instructions.len(),
                    segment.decode.decode.stop_reason,
                    segment.decode.decode.unresolved_handler,
                    segment
                        .transition
                        .as_ref()
                        .map(|transition| (&transition.status, &transition.diagnostics)),
                );
            }
            if let Some(segment) = chain.segments.iter().find(|segment| {
                segment
                    .decode
                    .handler_recoveries
                    .contains_key(&Va(0x37AA_2195B))
            }) {
                let recovery = segment
                    .decode
                    .handler_recoveries
                    .get(&Va(0x37AA_2195B))
                    .expect("located recovery must remain present");
                assert_eq!(recovery.status, RecoveryStatus::Recovered);
                let dispatch_report = &recovery.dispatch_analysis;
                let [dispatch] = dispatch_report.candidates.as_slice() else {
                    panic!("expected one conditional-postlude dispatch: {dispatch_report:#?}");
                };
                assert_eq!(dispatch.encoded_load, Rva(0xF08_19AF));
                assert!(
                    dispatch_report
                        .postlude_proof
                        .as_ref()
                        .is_some_and(|proof| {
                            proof.all_paths_dispatch && proof.acyclic && proof.violations.is_empty()
                        })
                );
                let [operand] = recovery.operand_analysis.candidates.as_slice() else {
                    panic!("expected one constant operand: {recovery:#?}");
                };
                assert_eq!(operand.encoded_load, Rva(0xF08_195B));
                assert_eq!(operand.codec.width, BitWidth::W64);
                assert_eq!(operand.vip_delta, 8);
                let analysis = recovery
                    .handler_analysis
                    .as_ref()
                    .expect("constant-push semantic must be recovered");
                assert_eq!(analysis.semantic, VmHandlerSemantic::ConstantPush);
                assert_eq!(analysis.operation_width, Some(BitWidth::W64));
                assert_eq!(analysis.stack_delta, Some(-8));

                let instruction = segment
                    .decode
                    .decode
                    .instructions
                    .iter()
                    .find(|instruction| instruction.state_before.handler == Va(0x37AA_2195B))
                    .expect("constant-push handler must execute in the decoded chain");
                assert_eq!(instruction.semantic, VmHandlerSemantic::ConstantPush);
                assert_eq!(instruction.operation_width, Some(BitWidth::W64));

                let symbolic = segment
                    .symbolic
                    .as_ref()
                    .expect("tainted symbolic execution must complete conservatively");
                assert!(!symbolic.memory_tainted);
                let Some(Expr {
                    kind:
                        ExprKind::Const {
                            value: branch_target,
                        },
                    ..
                }) = segment.transition_target.as_ref()
                else {
                    panic!(
                        "MBA-normalized stack store must preserve the concrete branch target: {:#?}",
                        segment.transition_target
                    );
                };
                assert_eq!(*branch_target, 0x379A_861DE);
                eprintln!("p06 recovered concrete virtual branch target {branch_target:#X}");
            }
            if let Some(directory) = std::env::var_os("VMPX_P06_PROFILE_REPORT_DIR") {
                fs::create_dir_all(&directory)
                    .expect("profile-chain report directory must be writable");
                let output = std::path::Path::new(&directory)
                    .join(format!("successor-{successor_index}-profile-chain.json"));
                let encoded =
                    serde_json::to_vec_pretty(&chain).expect("profile-chain report must serialize");
                fs::write(output, encoded).expect("profile-chain report path must be writable");
            }
            for candidate_vsp in &recovered.vsp_candidates {
                let descending = recover_and_decode_linear_vm_stream(
                    &pe,
                    first_dispatch.state_after,
                    &HandlerCodecRecoveryConfig {
                        roles: recovered.roles,
                        bytecode_direction: recovered.report.bytecode_direction,
                        effect_roles: Some(HandlerEffectRoles {
                            vsp: *candidate_vsp,
                            context_base: Register::RSP,
                        }),
                        live_out_registers: vec![
                            *candidate_vsp,
                            recovered.roles.handler,
                            recovered.roles.vip,
                            recovered.roles.rolling_key,
                        ],
                        live_out_rflags: 0,
                        trace_limits: DirectJumpTraceLimits::default(),
                        proof_limits: RegisterDispatchProofLimits::default(),
                    },
                    512,
                );
                match descending {
                    Ok(report) => {
                        eprintln!(
                            "p06 descending VSP {candidate_vsp:?}: {} operations, stop {:?}, next VIP {}, handler {}",
                            report.decode.instructions.len(),
                            report.decode.stop_reason,
                            report.decode.final_state.vip,
                            report.decode.final_state.handler,
                        );
                        if let Some(handler) = report.decode.unresolved_handler
                            && let Some(recovery) = report.handler_recoveries.get(&handler)
                        {
                            eprintln!(
                                "p06 descending unresolved {handler}: status {:?}, semantic {:?}, diagnostics {:?}",
                                recovery.status,
                                recovery
                                    .handler_analysis
                                    .as_ref()
                                    .map(|analysis| analysis.semantic),
                                recovery.diagnostics,
                            );
                            let unresolved_rva = pe
                                .optional
                                .image_base
                                .checked_rva_of(handler)
                                .expect("descending unresolved handler must be in the image");
                            let nested_transition = analyze_vm_transition(
                                &pe,
                                unresolved_rva,
                                *candidate_vsp,
                                resolved_profile.effect_roles.context_base,
                                DirectJumpTraceLimits::default(),
                            )
                            .expect("descending unresolved handler must trace");
                            eprintln!(
                                "p06 descending transition probe {handler}: {:#?}",
                                nested_transition.report
                            );
                            if *candidate_vsp == resolved_profile.effect_roles.vsp
                                && let Some(nested) = nested_transition.recovered
                            {
                                let profile_symbolic =
                                    execute_symbolic_linear_block_with_initial_state(
                                        &report.decode.instructions,
                                        None,
                                        SymbolicExecutionLimits::default(),
                                        &transitioned_symbolic_state,
                                        Some(&pe),
                                    )
                                    .expect("transitioned profile semantics must execute");
                                let nested_pop = execute_symbolic_linear_block_with_initial_state(
                                    &[],
                                    Some(VmHandlerSemantic::UnconditionalBranch),
                                    SymbolicExecutionLimits::default(),
                                    &profile_symbolic.continuation_state(),
                                    Some(&pe),
                                )
                                .expect("nested transition must pop the retained VM stack");
                                let _nested_symbolic_state = nested_pop.continuation_state();
                                if let Some(Expr {
                                    kind:
                                        ExprKind::Const {
                                            value: nested_target,
                                        },
                                    ..
                                }) = nested_pop.branch_target
                                {
                                    let nested_initial = RelativeDispatchState {
                                        vip: Va(nested_target),
                                        rolling_key: nested_target,
                                        handler: nested.handler_anchor,
                                    };
                                    let nested_first = nested
                                        .dispatch
                                        .decode_direction(
                                            &pe,
                                            nested_initial,
                                            nested.report.bytecode_direction,
                                        )
                                        .expect("nested transition dispatch must decode");
                                    let nested_handler_rva = pe
                                        .optional
                                        .image_base
                                        .checked_rva_of(nested_first.state_after.handler)
                                        .expect("nested first handler must be in the image");
                                    let nested_roles = resolve_vm_profile_roles(
                                        &pe,
                                        nested_handler_rva,
                                        &VmProfileRoleResolutionConfig {
                                            roles: nested.roles,
                                            bytecode_direction: nested.report.bytecode_direction,
                                            vsp_candidates: nested.vsp_candidates.clone(),
                                            context_base: Register::RSP,
                                            trace_limits: DirectJumpTraceLimits::default(),
                                            proof_limits: RegisterDispatchProofLimits::default(),
                                        },
                                    )
                                    .expect("nested first handler must trace");
                                    eprintln!(
                                        "p06 nested profile: target {nested_target:#X}, direction {:?}, first handler {}, VSP resolution {:?}",
                                        nested.report.bytecode_direction,
                                        nested_first.state_after.handler,
                                        nested_roles.report.selected_vsp_register,
                                    );
                                    if let Some(nested_profile) = nested_roles.recovered {
                                        let nested_decode = recover_and_decode_linear_vm_stream(
                                            &pe,
                                            nested_first.state_after,
                                            &HandlerCodecRecoveryConfig {
                                                roles: nested_profile.handler_roles,
                                                bytecode_direction: nested_profile
                                                    .bytecode_direction,
                                                effect_roles: Some(nested_profile.effect_roles),
                                                live_out_registers: vec![
                                                    nested_profile.effect_roles.vsp,
                                                    nested_profile.handler_roles.handler,
                                                    nested_profile.handler_roles.vip,
                                                    nested_profile.handler_roles.rolling_key,
                                                ],
                                                live_out_rflags: 0,
                                                trace_limits: DirectJumpTraceLimits::default(),
                                                proof_limits: RegisterDispatchProofLimits::default(
                                                ),
                                            },
                                            512,
                                        )
                                        .expect("nested profile decoding must complete");
                                        assert!(!nested_decode.decode.instructions.is_empty());
                                        eprintln!(
                                            "p06 nested decoded: {} operations, stop {:?}, next VIP {}, handler {}",
                                            nested_decode.decode.instructions.len(),
                                            nested_decode.decode.stop_reason,
                                            nested_decode.decode.final_state.vip,
                                            nested_decode.decode.final_state.handler,
                                        );
                                        if let Some(directory) =
                                            std::env::var_os("VMPX_P06_NESTED_REPORT_DIR")
                                        {
                                            fs::create_dir_all(&directory)
                                                .expect("nested report directory must be writable");
                                            let output = std::path::Path::new(&directory).join(
                                                format!("successor-{successor_index}-nested.json"),
                                            );
                                            let encoded = serde_json::to_vec_pretty(&nested_decode)
                                                .expect("nested report must serialize");
                                            fs::write(output, encoded)
                                                .expect("nested report path must be writable");
                                        }
                                        if let Some(next_handler) =
                                            nested_decode.decode.unresolved_handler
                                        {
                                            if let Some(recovery) =
                                                nested_decode.handler_recoveries.get(&next_handler)
                                            {
                                                eprintln!(
                                                    "p06 next unresolved {next_handler}: trace {:?}, status {:?}, semantic {:?}, diagnostics {:?}",
                                                    recovery.trace_stop,
                                                    recovery.status,
                                                    recovery
                                                        .handler_analysis
                                                        .as_ref()
                                                        .map(|analysis| analysis.semantic),
                                                    recovery.diagnostics,
                                                );
                                            }
                                            let next_rva = pe
                                                .optional
                                                .image_base
                                                .checked_rva_of(next_handler)
                                                .expect("next nested handler must be in the image");
                                            let next_transition = analyze_vm_transition(
                                                &pe,
                                                next_rva,
                                                nested_profile.effect_roles.vsp,
                                                nested_profile.effect_roles.context_base,
                                                DirectJumpTraceLimits::default(),
                                            )
                                            .expect("next nested handler must trace");
                                            eprintln!(
                                                "p06 next transition probe {next_handler}: {:#?}",
                                                next_transition.report
                                            );
                                        }
                                    }
                                } else {
                                    eprintln!(
                                        "p06 nested transition target remains symbolic: {:?}",
                                        nested_pop.branch_target
                                    );
                                }
                            }
                        }
                        if let Some(directory) = std::env::var_os("VMPX_P06_DESCENDING_REPORT_DIR")
                        {
                            fs::create_dir_all(&directory)
                                .expect("descending report directory must be writable");
                            let output = std::path::Path::new(&directory).join(format!(
                                "successor-{successor_index}-vsp-{candidate_vsp:?}.json"
                            ));
                            let encoded =
                                serde_json::to_vec_pretty(&report).expect("report must serialize");
                            fs::write(output, encoded)
                                .expect("descending report path must be writable");
                        }
                    }
                    Err(error) => {
                        let report = error.partial_report();
                        eprintln!(
                            "p06 descending VSP {candidate_vsp:?}: error {error}; {} operations, stop {:?}, next VIP {}, handler {}",
                            report.decode.instructions.len(),
                            report.decode.stop_reason,
                            report.decode.final_state.vip,
                            report.decode.final_state.handler,
                        );
                    }
                }
            }
        }
    }
}
