use iced_x86::code_asm::*;
use iced_x86::{BlockEncoderOptions, Register};
use vmpx_disasm::{DirectJumpTraceLimits, RegisterDispatchProofLimits};
use vmpx_pe::{PeImage, Va};
use vmpx_vm::{
    BytecodeDirection, GlobalTableBase, HandlerCodecRecoveryConfig, HandlerEffectRoles,
    HandlerRegisterRoles, LinearDecodeStop, LinearHandlerRecoveryReport, RelativeDispatchState,
    VmHandlerSemantic, recover_and_decode_linear_vm_stream_with_table,
};

const IMAGE_BASE: u64 = 0x1_0000;
const TEXT_RVA: u32 = 0x1000;
const TABLE_RVA: u32 = 0x1100;
const CODE_OFFSET: usize = 0x200;
const TABLE_VA: u64 = IMAGE_BASE + TABLE_RVA as u64;
const BYTECODE_RVA: u32 = 0x1140;

fn pe64(code: &[u8], table: &[i32], bytecode: &[u8]) -> Vec<u8> {
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
    assert!(code.len() <= 0x100, "fixture code exceeds its region");
    data[CODE_OFFSET..CODE_OFFSET + code.len()].copy_from_slice(code);
    for (index, entry) in table.iter().enumerate() {
        let at = CODE_OFFSET + (TABLE_RVA - TEXT_RVA) as usize + index * 4;
        data[at..at + 4].copy_from_slice(&entry.to_le_bytes());
    }
    let at = CODE_OFFSET + (BYTECODE_RVA - TEXT_RVA) as usize;
    data[at..at + bytecode.len()].copy_from_slice(bytecode);
    data
}

struct Fixture {
    pe_bytes: Vec<u8>,
    add: Va,
    bad: Va,
}

/// Opcode 0 is a 16-bit add with an inline unanchored table dispatch through R11;
/// opcode 1 is a handler with no recoverable dispatch. `clobber`
/// makes the add handler overwrite the table register before dispatching.
fn build(clobber: bool) -> Fixture {
    let mut a = CodeAssembler::new(64).expect("x64 assembler");
    let mut add = a.create_label();
    let mut bad = a.create_label();

    a.set_label(&mut add).expect("add label");
    a.mov(ax, word_ptr(rbp)).expect("left");
    a.mov(di, word_ptr(rbp + 2)).expect("right");
    a.sub(rbp, 6).expect("pop two words, push one qword");
    a.add(ax, di).expect("add");
    a.pushfq().expect("capture flags");
    a.pop(rdi).expect("flags into a register");
    a.mov(word_ptr(rbp + 8), ax).expect("result");
    a.mov(qword_ptr(rbp), rdi).expect("flags");
    if clobber {
        a.mov(r11, rax).expect("table clobber");
    }
    a.movzx(eax, byte_ptr(rsi)).expect("opcode");
    a.add(rsi, 1).expect("VIP advance");
    a.movsxd(rbx, dword_ptr(r11 + rax * 4))
        .expect("table entry");
    a.add(rbx, r11).expect("handler base");
    a.push(rbx).expect("push handler");
    a.ret().expect("dispatch");

    a.set_label(&mut bad).expect("bad label");
    a.int3().expect("no dispatch");
    a.ret().expect("return");

    let result = a
        .assemble_options(
            IMAGE_BASE + u64::from(TEXT_RVA),
            BlockEncoderOptions::RETURN_NEW_INSTRUCTION_OFFSETS,
        )
        .expect("assemble fixture");
    let va = |label: &CodeLabel| result.label_ip(label).expect("label ip");
    let (add_va, bad_va) = (va(&add), va(&bad));
    let delta = |target: u64| i32::try_from(target as i64 - TABLE_VA as i64).expect("delta");
    Fixture {
        add: Va(add_va),
        bad: Va(bad_va),
        pe_bytes: pe64(
            &result.inner.code_buffer,
            &[delta(add_va), delta(bad_va)],
            &[0, 0, 0, 1],
        ),
    }
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

fn walk(fixture: &Fixture, global: Option<GlobalTableBase>) -> LinearHandlerRecoveryReport {
    let pe = PeImage::parse(&fixture.pe_bytes).expect("synthetic PE");
    // The bootstrap already consumed the first opcode; the stream resumes at the
    // second opcode byte with the add handler selected.
    let initial = RelativeDispatchState {
        vip: Va(IMAGE_BASE + u64::from(BYTECODE_RVA) + 1),
        rolling_key: 0,
        handler: fixture.add,
    };
    recover_and_decode_linear_vm_stream_with_table(&pe, initial, &config(), 64, global)
        .expect("walk completes")
}

fn global(register: Register) -> Option<GlobalTableBase> {
    Some(GlobalTableBase {
        register,
        base: Va(TABLE_VA),
    })
}

#[test]
fn bootstrap_table_base_drives_the_walk_to_an_undecodable_opcode() {
    let fixture = build(false);
    let report = walk(&fixture, global(Register::R11));
    let stream = &report.decode;
    assert_eq!(stream.instructions.len(), 3, "{report:#?}");
    assert!(
        stream
            .instructions
            .iter()
            .all(|instruction| instruction.semantic == VmHandlerSemantic::Add)
    );
    assert_eq!(stream.stop_reason, LinearDecodeStop::MissingHandlerSpec);
    assert_eq!(stream.unresolved_handler, Some(fixture.bad));
    assert_eq!(stream.final_state.handler, fixture.bad);
}

#[test]
fn table_register_mismatch_or_clobber_refuses_the_handler() {
    let cases = [
        (build(false), global(Register::R10)),
        (build(true), global(Register::R11)),
    ];
    for (fixture, global) in cases {
        let report = walk(&fixture, global);
        let stream = &report.decode;
        assert!(stream.instructions.is_empty(), "{report:#?}");
        assert_eq!(stream.stop_reason, LinearDecodeStop::MissingHandlerSpec);
        assert_eq!(stream.unresolved_handler, Some(fixture.add));
    }
}
