use iced_x86::code_asm::*;
use iced_x86::{BlockEncoderOptions, Register};
use vmpx_disasm::{DirectJumpTraceLimits, RegisterDispatchProofLimits};
use vmpx_pe::{PeImage, Rva};
use vmpx_vm::{
    BytecodeDirection, HandlerCodecRecoveryConfig, HandlerCodecRecoveryReport, HandlerEffectRoles,
    HandlerRegisterRoles, RecoveryStatus, VmHandlerSemantic, recover_handler_codecs,
};

const IMAGE_BASE: u64 = 0x1_0000;
const TEXT_RVA: u32 = 0x1000;

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

#[derive(Clone, Copy, Default)]
#[allow(clippy::struct_excessive_bools)]
struct Variant {
    target_from_wrong_slot: bool,
    extra_vsp_slot: bool,
    clobber_table: bool,
    clobber_vip: bool,
    skip_rsp_restore: bool,
    skip_result_store: bool,
    second_vip_read: bool,
}

fn build(variant: Variant) -> (Vec<u8>, Rva) {
    let mut a = CodeAssembler::new(64).expect("x64 assembler");
    let mut handler = a.create_label();
    let mut small = a.create_label();
    let mut do_call = a.create_label();
    let mut top = a.create_label();
    let mut next = a.create_label();
    let mut l2 = a.create_label();
    let mut l3 = a.create_label();
    let mut l4 = a.create_label();
    let mut l5 = a.create_label();
    let mut head = a.create_label();

    a.set_label(&mut handler).expect("handler");
    a.movzx(r10d, byte_ptr(rsi)).expect("argument count");
    a.add(rsi, 1).expect("VIP advance");
    a.push(rsi).expect("save VIP");
    if !variant.clobber_table {
        a.push(r11).expect("save table");
    }
    if variant.second_vip_read {
        a.mov(r9, qword_ptr(rsi + 8)).expect("second VIP read");
    }
    a.mov(ebx, r10d).expect("count");
    a.mov(edx, ebx).expect("register slots");
    a.xor(ecx, ecx).expect("no stack args");
    a.cmp(ebx, 4).expect("compare");
    a.jbe(small).expect("small");
    a.mov(edx, 4).expect("four slots");
    a.lea(ecx, dword_ptr(rbx - 4)).expect("stack args");
    a.set_label(&mut small).expect("small label");
    a.shl(ecx, 3).expect("bytes");
    a.shl(edx, 3).expect("bytes");
    a.mov(rax, rbp).expect("vsp");
    if variant.extra_vsp_slot {
        a.lea(rax, qword_ptr(rax + rdx + 8)).expect("wrong adjust");
    } else {
        a.add(rax, rdx).expect("adjust");
    }
    a.mov(qword_ptr(rbp - 8), rax).expect("save new vsp");
    a.mov(qword_ptr(rbp - 0x10), rsp).expect("save rsp");
    a.sub(rsp, rcx).expect("room");
    a.and(rsp, -16).expect("align");
    a.add(rsp, rcx).expect("restore room");
    a.test(ebx, ebx).expect("zero args");
    a.je(do_call).expect("skip loop");
    a.set_label(&mut top).expect("loop");
    a.mov(rax, qword_ptr(rbp + rbx * 8)).expect("argument");
    a.cmp(ebx, 1).expect("1");
    a.jne(l2).expect("not 1");
    a.mov(rcx, rax).expect("rcx");
    a.jmp(next).expect("next");
    a.set_label(&mut l2).expect("l2");
    a.cmp(ebx, 2).expect("2");
    a.jne(l3).expect("not 2");
    a.mov(rdx, rax).expect("rdx");
    a.jmp(next).expect("next");
    a.set_label(&mut l3).expect("l3");
    a.cmp(ebx, 3).expect("3");
    a.jne(l4).expect("not 3");
    a.mov(r8, rax).expect("r8");
    a.jmp(next).expect("next");
    a.set_label(&mut l4).expect("l4");
    a.cmp(ebx, 4).expect("4");
    a.jne(l5).expect("not 4");
    a.mov(r9, rax).expect("r9");
    a.jmp(next).expect("next");
    a.set_label(&mut l5).expect("l5");
    a.mov(rax, qword_ptr(rbp + rbx * 8 + 0x20))
        .expect("stack argument");
    a.push(rax).expect("push stack argument");
    a.set_label(&mut next).expect("next label");
    a.dec(ebx).expect("dec");
    a.jne(top).expect("loop back");
    a.set_label(&mut do_call).expect("call");
    if variant.target_from_wrong_slot {
        a.mov(rax, qword_ptr(rbp + 8)).expect("wrong target");
    } else {
        a.mov(rax, qword_ptr(rbp)).expect("target");
    }
    a.sub(rsp, 0x20).expect("shadow space");
    a.call(rax).expect("native call");
    if !variant.skip_rsp_restore {
        a.mov(rsp, qword_ptr(rbp - 0x10)).expect("restore rsp");
    }
    a.mov(rbp, qword_ptr(rbp - 8)).expect("restore vsp");
    if !variant.skip_result_store {
        a.mov(qword_ptr(rbp), rax).expect("result");
    }
    if variant.clobber_table {
        a.mov(r11, rax).expect("table clobber");
    } else {
        a.pop(r11).expect("restore table");
    }
    if variant.clobber_vip {
        a.add(rsp, 8).expect("drop saved VIP");
        a.mov(rsi, rax).expect("VIP clobber");
    } else {
        a.pop(rsi).expect("restore VIP");
    }
    a.jmp(head).expect("shared dispatch");
    a.set_label(&mut head).expect("head");
    a.movzx(eax, byte_ptr(rsi)).expect("opcode");
    a.add(rsi, 1).expect("VIP advance");
    a.movsxd(rbx, dword_ptr(r11 + rax * 4)).expect("entry");
    a.add(rbx, r11).expect("handler");
    a.push(rbx).expect("push handler");
    a.ret().expect("dispatch");

    let result = a
        .assemble_options(
            IMAGE_BASE + u64::from(TEXT_RVA),
            BlockEncoderOptions::RETURN_NEW_INSTRUCTION_OFFSETS,
        )
        .expect("assemble fixture");
    let ip = result.label_ip(&handler).expect("handler ip");
    (
        pe64(&result.inner.code_buffer),
        Rva(u32::try_from(ip - IMAGE_BASE).expect("rva")),
    )
}

fn recover(variant: Variant) -> HandlerCodecRecoveryReport {
    let (bytes, entry) = build(variant);
    let pe = PeImage::parse(&bytes).expect("synthetic PE");
    recover_handler_codecs(&pe, entry, &config()).expect("recovery completes")
}

#[test]
fn proven_keyless_bridge_classifies_as_native_call() {
    let report = recover(Variant::default());
    let analysis = report
        .handler_analysis
        .as_ref()
        .unwrap_or_else(|| panic!("classified handler: {report:#?}"));
    assert_eq!(analysis.semantic, VmHandlerSemantic::NativeCall);
    assert_ne!(report.status, RecoveryStatus::Failed);
    assert!(report.value_flow.as_ref().is_some_and(|proof| proof.proven));
    let dispatch = report.dispatch.as_ref().expect("resumed dispatch");
    assert!(dispatch.table.is_some());
    let spec = report.boundary_spec.as_ref().expect("boundary spec");
    assert_eq!(spec.outcomes.len(), 256);
    assert_eq!(spec.outcomes[0].vsp_delta, 0);
    assert_eq!(spec.outcomes[2].vsp_delta, 16);
    assert_eq!(spec.outcomes[2].register_argument_offsets, vec![8, 16]);
    assert_eq!(spec.outcomes[6].vsp_delta, 32);
    assert_eq!(spec.outcomes[6].stack_argument_offsets, vec![80, 72]);
}

#[test]
fn near_miss_bridges_stay_failed() {
    let variants = [
        Variant {
            target_from_wrong_slot: true,
            ..Variant::default()
        },
        Variant {
            extra_vsp_slot: true,
            ..Variant::default()
        },
        Variant {
            clobber_table: true,
            ..Variant::default()
        },
        Variant {
            clobber_vip: true,
            ..Variant::default()
        },
        Variant {
            skip_rsp_restore: true,
            ..Variant::default()
        },
        Variant {
            skip_result_store: true,
            ..Variant::default()
        },
        Variant {
            second_vip_read: true,
            ..Variant::default()
        },
    ];
    for (index, variant) in variants.into_iter().enumerate() {
        let report = recover(variant);
        eprintln!(
            "variant {index}: {:?}",
            report.native_call.as_ref().map(|call| &call.diagnostics)
        );
        assert!(
            report.boundary_spec.is_none() && report.handler_analysis.is_none(),
            "variant {index} must not classify: {:?}",
            report.native_call.as_ref().map(|call| &call.diagnostics)
        );
        assert_eq!(report.status, RecoveryStatus::Failed, "variant {index}");
    }
}
