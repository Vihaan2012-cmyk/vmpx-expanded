use iced_x86::code_asm::*;
use iced_x86::{BlockEncoderOptions, Register};
use vmpx_disasm::{DirectJumpTraceLimits, RegisterDispatchProofLimits};
use vmpx_pe::{PeImage, Rva, Va};
use vmpx_vm::{
    BytecodeDirection, HandlerCodecRecoveryConfig, HandlerCodecRecoveryReport, HandlerEffectRoles,
    HandlerRegisterRoles, RecoveryStatus, VmHandlerSemantic, recover_handler_codecs,
};

const IMAGE_BASE: u64 = 0x1_0000;
const TEXT_RVA: u32 = 0x1000;
const TABLE_VA: u64 = IMAGE_BASE + 0x1F00;

fn pe64(code: &[u8]) -> Vec<u8> {
    let mut data = vec![0_u8; 0x400];
    data[0..2].copy_from_slice(&0x5A4D_u16.to_le_bytes());
    data[0x3C..0x40].copy_from_slice(&(0x80_u32).to_le_bytes());
    data[0x80..0x84].copy_from_slice(&0x0000_4550_u32.to_le_bytes());
    let coff = 0x84;
    data[coff..coff + 2].copy_from_slice(&(0x8664_u16).to_le_bytes());
    data[coff + 2..coff + 4].copy_from_slice(&(1_u16).to_le_bytes());
    data[coff + 16..coff + 18].copy_from_slice(&(0xF0_u16).to_le_bytes());
    data[coff + 18..coff + 20].copy_from_slice(&(0x2022_u16).to_le_bytes());
    let optional = coff + 20;
    data[optional..optional + 2].copy_from_slice(&0x20B_u16.to_le_bytes());
    data[optional + 16..optional + 20].copy_from_slice(&TEXT_RVA.to_le_bytes());
    data[optional + 20..optional + 24].copy_from_slice(&TEXT_RVA.to_le_bytes());
    data[optional + 24..optional + 32].copy_from_slice(&IMAGE_BASE.to_le_bytes());
    data[optional + 32..optional + 36].copy_from_slice(&(0x1000_u32).to_le_bytes());
    data[optional + 36..optional + 40].copy_from_slice(&(0x200_u32).to_le_bytes());
    data[optional + 56..optional + 60].copy_from_slice(&(0x2000_u32).to_le_bytes());
    data[optional + 60..optional + 64].copy_from_slice(&(0x200_u32).to_le_bytes());
    data[optional + 68..optional + 70].copy_from_slice(&(3_u16).to_le_bytes());
    data[optional + 108..optional + 112].copy_from_slice(&(16_u32).to_le_bytes());
    let section = optional + 0xF0;
    data[section..section + 5].copy_from_slice(b".text");
    data[section + 8..section + 12].copy_from_slice(&(0x200_u32).to_le_bytes());
    data[section + 12..section + 16].copy_from_slice(&TEXT_RVA.to_le_bytes());
    data[section + 16..section + 20].copy_from_slice(&(0x200_u32).to_le_bytes());
    data[section + 20..section + 24].copy_from_slice(&(0x200_u32).to_le_bytes());
    data[section + 36..section + 40].copy_from_slice(&(0x6000_0020_u32).to_le_bytes());
    assert!(code.len() <= 0x200, "fixture code exceeds the section");
    data[0x200..0x200 + code.len()].copy_from_slice(code);
    data
}

fn config() -> HandlerCodecRecoveryConfig {
    HandlerCodecRecoveryConfig {
        roles: HandlerRegisterRoles {
            vip: Register::RSI,
            rolling_key: Register::RCX,
            handler: Register::RBX,
        },
        bytecode_direction: BytecodeDirection::Ascending,
        effect_roles: Some(HandlerEffectRoles {
            vsp: Register::RBP,
            context_base: Register::RSP,
        }),
        live_out_registers: vec![Register::RBP, Register::RBX, Register::RSI, Register::RCX],
        live_out_rflags: 0,
        trace_limits: DirectJumpTraceLimits::default(),
        proof_limits: RegisterDispatchProofLimits::default(),
    }
}

/// How the fixture's shared postlude and dispatch head are shaped.
#[derive(Clone, Copy, Default)]
#[allow(clippy::struct_excessive_bools)]
struct Variant {
    /// `cmp rax, rcx` instead of comparing the VSP.
    compare_non_vsp: bool,
    /// `jb` instead of `ja`.
    below_branch: bool,
    /// Context base other than the native stack pointer.
    foreign_context: bool,
    /// `clc` between `cmp` and `ja`, corrupting the compared flags.
    flag_junk_after_cmp: bool,
    /// The fall-through relocation path clobbers VIP.
    fallthrough_writes_vip: bool,
    /// The fall-through relocation path returns instead of reaching the head.
    fallthrough_returns: bool,
    /// The branch target is not a dispatch head.
    non_dispatch_target: bool,
    /// The handler overwrites the table register.
    handler_writes_table: bool,
    /// The head has no static table anchor falling into it.
    no_anchor: bool,
    /// The dispatch head rewrites the (global) table register before the load.
    head_writes_table: bool,
}

struct Fixture {
    pe_bytes: Vec<u8>,
    handler: Rva,
    reenter: Rva,
}

enum HandlerBody {
    Add16,
    ViaPostlude,
    VipReloadReenter,
    VipReloadNoAnchor,
}

fn build(body: &HandlerBody, variant: Variant) -> Fixture {
    let mut a = CodeAssembler::new(64).expect("x64 assembler");
    let mut handler = a.create_label();
    let mut postlude = a.create_label();
    let mut fallthrough = a.create_label();
    let mut reenter = a.create_label();
    let mut head = a.create_label();
    let mut other = a.create_label();

    a.set_label(&mut handler).expect("handler label");
    match body {
        HandlerBody::Add16 | HandlerBody::ViaPostlude => {
            a.mov(ax, word_ptr(rbp)).expect("left");
            a.mov(di, word_ptr(rbp + 2)).expect("right");
            a.sub(rbp, 6).expect("pop two words, push one qword");
            a.add(ax, di).expect("add");
            a.pushfq().expect("capture flags");
            a.pop(rdi).expect("flags into a register");
            if variant.handler_writes_table {
                a.mov(r11, rax).expect("table clobber");
            }
            a.mov(word_ptr(rbp + 8), ax).expect("result");
            a.mov(qword_ptr(rbp), rdi).expect("flags");
            a.jmp(postlude).expect("enter the shared postlude");
        }
        HandlerBody::VipReloadReenter => {
            a.mov(rsi, qword_ptr(rbp)).expect("VIP from the VM stack");
            a.add(rbp, 8).expect("pop the target");
            a.jmp(reenter).expect("re-enter at the table anchor");
        }
        HandlerBody::VipReloadNoAnchor => {
            a.mov(rsi, qword_ptr(rbp)).expect("VIP from the VM stack");
            a.add(rbp, 8).expect("pop the target");
            a.jmp(head).expect("enter the head without the anchor");
        }
    }

    a.set_label(&mut postlude).expect("postlude label");
    let context = if variant.foreign_context { rdx } else { rsp };
    a.lea(rcx, qword_ptr(context + 0x180)).expect("limit");
    a.clc().expect("junk");
    if variant.compare_non_vsp {
        a.cmp(rax, rcx).expect("compare");
    } else {
        a.cmp(rbp, rcx).expect("compare");
    }
    if variant.flag_junk_after_cmp {
        a.clc().expect("flag junk");
    }
    let target = if variant.non_dispatch_target {
        other
    } else {
        head
    };
    if variant.below_branch {
        a.jb(target).expect("branch");
    } else {
        a.ja(target).expect("branch");
    }
    a.set_label(&mut fallthrough).expect("fall-through label");
    a.mov(r8, rsp).expect("relocation state");
    if variant.fallthrough_writes_vip {
        a.mov(rsi, r8).expect("VIP clobber");
    }
    if variant.fallthrough_returns {
        a.ret().expect("return");
    } else {
        a.jmp(head).expect("rejoin the dispatch head");
    }

    a.set_label(&mut other).expect("other label");
    a.nop().expect("not a dispatch");
    a.ret().expect("return");

    a.set_label(&mut reenter).expect("re-entry label");
    if !variant.no_anchor {
        a.lea(r11, qword_ptr(TABLE_VA)).expect("table anchor");
    }
    a.clc().expect("junk");
    a.set_label(&mut head).expect("head label");
    a.movzx(eax, byte_ptr(rsi)).expect("opcode");
    a.add(rsi, 1).expect("VIP advance");
    if variant.head_writes_table {
        // Clobber the global table register after the opcode load but before the
        // table lookup: the bare-head check still passes, so only the explicit
        // head preservation check catches it.
        a.mov(r11, rsp).expect("head clobbers the global table register");
    }
    a.movsxd(rbx, dword_ptr(r11 + rax * 4))
        .expect("table entry");
    a.add(rbx, r11).expect("handler base");
    a.push(rbx).expect("push handler");
    a.ret().expect("dispatch");

    let result = a
        .assemble_options(
            IMAGE_BASE + u64::from(TEXT_RVA),
            BlockEncoderOptions::RETURN_NEW_INSTRUCTION_OFFSETS,
        )
        .expect("assemble fixture");
    let rva = |label: &CodeLabel| {
        let ip = result.label_ip(label).expect("label ip");
        Rva(u32::try_from(ip - IMAGE_BASE).expect("rva"))
    };
    Fixture {
        handler: rva(&handler),
        reenter: rva(&reenter),
        pe_bytes: pe64(&result.inner.code_buffer),
    }
}

fn recover(fixture: &Fixture, entry: Rva) -> HandlerCodecRecoveryReport {
    let pe = PeImage::parse(&fixture.pe_bytes).expect("synthetic PE");
    recover_handler_codecs(&pe, entry, &config()).expect("recovery completes")
}

#[test]
fn stack_check_postlude_reaches_the_proven_dispatch_head() {
    let fixture = build(&HandlerBody::Add16, Variant::default());
    let report = recover(&fixture, fixture.handler);

    let [candidate] = report.dispatch_analysis.candidates.as_slice() else {
        panic!("expected one dispatch codec: {report:#?}");
    };
    let postlude = candidate.postlude.expect("postlude-reached dispatch");
    assert_ne!(postlude.stack_check_start, candidate.encoded_load);
    assert_eq!(candidate.handler_rebase, Some(Va(TABLE_VA)));
    let table = candidate.codec.table.expect("table-indexed dispatch");
    assert_eq!(table.table_base, Some(Va(TABLE_VA)));
    assert_eq!(table.stride, 4);
    assert_eq!(report.status, RecoveryStatus::Recovered, "{report:#?}");
    let analysis = report.handler_analysis.expect("classified handler");
    assert_eq!(analysis.semantic, VmHandlerSemantic::Add);
    assert!(report.decode_spec.is_some());
}

#[test]
fn conditional_branch_to_a_non_dispatch_target_stays_failed() {
    let fixture = build(
        &HandlerBody::Add16,
        Variant {
            non_dispatch_target: true,
            ..Variant::default()
        },
    );
    let report = recover(&fixture, fixture.handler);
    assert!(report.dispatch_analysis.candidates.is_empty());
    assert_eq!(report.status, RecoveryStatus::Failed);
    assert!(report.handler_analysis.is_none());
}

#[test]
fn inexact_postlude_shapes_are_not_recognized() {
    let variants = [
        Variant {
            compare_non_vsp: true,
            ..Variant::default()
        },
        Variant {
            below_branch: true,
            ..Variant::default()
        },
        Variant {
            foreign_context: true,
            ..Variant::default()
        },
        Variant {
            flag_junk_after_cmp: true,
            ..Variant::default()
        },
    ];
    for variant in variants {
        let fixture = build(&HandlerBody::Add16, variant);
        let report = recover(&fixture, fixture.handler);
        assert!(report.dispatch_analysis.candidates.is_empty());
        assert_eq!(report.status, RecoveryStatus::Failed);
    }
}

#[test]
fn unsound_fallthrough_table_or_anchor_is_rejected() {
    let variants = [
        Variant {
            fallthrough_writes_vip: true,
            ..Variant::default()
        },
        Variant {
            fallthrough_returns: true,
            ..Variant::default()
        },
        Variant {
            handler_writes_table: true,
            ..Variant::default()
        },
    ];
    for variant in variants {
        let fixture = build(&HandlerBody::ViaPostlude, variant);
        let report = recover(&fixture, fixture.handler);
        assert!(
            report.dispatch_analysis.candidates.is_empty(),
            "{report:#?}"
        );
        assert_eq!(report.status, RecoveryStatus::Failed);
        assert!(
            report
                .dispatch_analysis
                .rejected_sources
                .iter()
                .any(|rejection| rejection.reason.starts_with("stack-check postlude")),
            "{:#?}",
            report.dispatch_analysis.rejected_sources
        );
    }
}

/// When the dispatch head's table base is a global live-in rather than a local
/// `lea table,[abs]` anchor (as in the bit-permutation / graph-interpreter VM
/// profile), the postlude dispatch is still recovered structurally, with its
/// table base left unbound for the bootstrap anchor to supply. The handler, the
/// head up to the dispatch, and the relocation fall-through are all proven to
/// leave the table register untouched, so binding that global base is sound.
#[test]
fn stack_check_postlude_with_global_table_base_recovers_unbound() {
    let fixture = build(
        &HandlerBody::ViaPostlude,
        Variant {
            no_anchor: true,
            ..Variant::default()
        },
    );
    let report = recover(&fixture, fixture.handler);
    let [candidate] = report.dispatch_analysis.candidates.as_slice() else {
        panic!("expected one dispatch codec for the global-base postlude: {report:#?}");
    };
    assert!(candidate.postlude.is_some(), "postlude-reached dispatch");
    assert_eq!(
        candidate.handler_rebase, None,
        "a global live-in base has no local anchor"
    );
    assert!(
        candidate.table_register.is_some(),
        "the table register is recovered for the global binding"
    );
    let table = candidate.codec.table.expect("table-indexed dispatch");
    assert_eq!(
        table.table_base, None,
        "the table base is left unbound for the bootstrap anchor"
    );
    assert_eq!(table.stride, 4);
    assert!(
        candidate
            .evidence
            .iter()
            .any(|line| line.contains("global live-in")),
        "{:#?}",
        candidate.evidence
    );
    assert!(
        report.dispatch_analysis.rejected_sources.is_empty(),
        "{:#?}",
        report.dispatch_analysis.rejected_sources
    );
}

/// A global-base postlude whose dispatch head rewrites the table register before
/// the dispatch cannot soundly bind the bootstrap base, so it fails closed.
#[test]
fn stack_check_postlude_head_rewriting_global_table_is_rejected() {
    let fixture = build(
        &HandlerBody::ViaPostlude,
        Variant {
            no_anchor: true,
            head_writes_table: true,
            ..Variant::default()
        },
    );
    let report = recover(&fixture, fixture.handler);
    assert!(
        report.dispatch_analysis.candidates.is_empty(),
        "{report:#?}"
    );
    assert_eq!(report.status, RecoveryStatus::Failed);
    assert!(
        report
            .dispatch_analysis
            .rejected_sources
            .iter()
            .any(|rejection| rejection.reason.contains("head rewrites the global table register")),
        "{:#?}",
        report.dispatch_analysis.rejected_sources
    );
}

#[test]
fn vip_reload_with_table_reentry_is_an_unconditional_branch() {
    let fixture = build(&HandlerBody::VipReloadReenter, Variant::default());
    assert_ne!(fixture.reenter, fixture.handler);
    let report = recover(&fixture, fixture.handler);
    let analysis = report.handler_analysis.expect("classified handler");
    assert_eq!(analysis.semantic, VmHandlerSemantic::UnconditionalBranch);
    assert!(report.value_flow.expect("value flow").proven);
}

#[test]
fn vip_reload_without_a_table_anchor_is_not_a_branch() {
    let fixture = build(&HandlerBody::VipReloadNoAnchor, Variant::default());
    let report = recover(&fixture, fixture.handler);
    let semantic = report.handler_analysis.map(|analysis| analysis.semantic);
    assert_ne!(semantic, Some(VmHandlerSemantic::UnconditionalBranch));
}
