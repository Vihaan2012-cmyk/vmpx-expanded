use crate::PhysicalRegister;
use iced_x86::{Decoder, DecoderOptions, FlowControl};
use iced_x86::code_asm::{
    AsmRegister64, CodeAssembler,
    registers::gpr64::{r8, r9, r10, r11, r12, r13, r14, r15, rax, rbp, rbx, rcx, rdi, rdx, rsi},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DispatchArm {
    pub value: u64,
    pub body: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum DispatchError {
    #[error("domain value {0} does not fit in a 32-bit compare immediate")]
    ValueTooWide(u64),
    #[error("iced assembly failed: {0}")]
    Assemble(String),
    #[error("the selector and scratch registers must be among the saved registers")]
    UnsavedRegister,
    #[error("dispatch arm value {0} appears more than once")]
    DuplicateValue(u64),
    #[error("the stub prefix is undecodable, branching, rip-relative, or too short")]
    StubNotRelocatable,
}

fn reg(register: PhysicalRegister) -> AsmRegister64 {
    match register {
        PhysicalRegister::Rax => rax,
        PhysicalRegister::Rcx => rcx,
        PhysicalRegister::Rdx => rdx,
        PhysicalRegister::Rbx => rbx,
        PhysicalRegister::Rbp => rbp,
        PhysicalRegister::Rsi => rsi,
        PhysicalRegister::Rdi => rdi,
        PhysicalRegister::R8 => r8,
        PhysicalRegister::R9 => r9,
        PhysicalRegister::R10 => r10,
        PhysicalRegister::R11 => r11,
        PhysicalRegister::R12 => r12,
        PhysicalRegister::R13 => r13,
        PhysicalRegister::R14 => r14,
        PhysicalRegister::R15 => r15,
    }
}

/// Hand-encoded (not via `compile`) so no frame setup clobbers the incoming registers.
///
/// # Errors
/// `ValueTooWide` if an arm value exceeds `i32`; `Assemble` on iced failure.
pub fn emit_domain_dispatcher(
    domain: PhysicalRegister,
    arms: &[DispatchArm],
    fallback: u64,
    dispatcher_va: u64,
) -> Result<Vec<u8>, DispatchError> {
    let asm_err = |e: iced_x86::IcedError| DispatchError::Assemble(e.to_string());
    let mut asm = CodeAssembler::new(64).map_err(asm_err)?;
    let domain = reg(domain);
    for arm in arms {
        let imm = i32::try_from(arm.value).map_err(|_| DispatchError::ValueTooWide(arm.value))?;
        asm.cmp(domain, imm).map_err(asm_err)?;
        asm.je(arm.body).map_err(asm_err)?;
    }
    asm.jmp(fallback).map_err(asm_err)?;
    asm.assemble(dispatcher_va).map_err(asm_err)
}

/// A dispatcher that selects a body by a computed value rather than by a
/// register already holding it.
#[derive(Clone, Copy, Debug)]
pub struct ValueDispatch<'a> {
    /// Register `compute` returns the selector in.
    pub value: PhysicalRegister,
    /// Holds a wide arm constant while it is compared; clobbered like `value`.
    pub scratch: PhysicalRegister,
    /// Pushed on entry and popped before every exit, so the chosen body (or
    /// the fallback) sees the registers the dispatcher was entered with.
    /// Must include `value` and `scratch` and everything `compute` clobbers.
    pub saved: &'a [PhysicalRegister],
    /// Function that leaves the selector in `value`.
    pub compute: u64,
    pub arms: &'a [DispatchArm],
    pub fallback: u64,
}

/// Encodes `push saved; call compute; (cmp value, t_i; je arm_i)*; fallback`
/// where every exit restores `saved` before its `jmp`.
///
/// # Errors
/// `UnsavedRegister` if `value`/`scratch` are not in `saved`, `DuplicateValue`
/// for a repeated arm value, `Assemble` on iced failure.
pub fn emit_value_dispatcher(
    spec: &ValueDispatch<'_>,
    dispatcher_va: u64,
) -> Result<Vec<u8>, DispatchError> {
    let asm_err = |e: iced_x86::IcedError| DispatchError::Assemble(e.to_string());
    for register in [spec.value, spec.scratch] {
        if !spec.saved.contains(&register) {
            return Err(DispatchError::UnsavedRegister);
        }
    }
    for (index, arm) in spec.arms.iter().enumerate() {
        if spec.arms[..index].iter().any(|prior| prior.value == arm.value) {
            return Err(DispatchError::DuplicateValue(arm.value));
        }
    }
    let mut asm = CodeAssembler::new(64).map_err(asm_err)?;
    let (value, scratch) = (reg(spec.value), reg(spec.scratch));
    for register in spec.saved {
        asm.push(reg(*register)).map_err(asm_err)?;
    }
    asm.call(spec.compute).map_err(asm_err)?;
    let mut labels = Vec::with_capacity(spec.arms.len());
    for arm in spec.arms {
        let label = asm.create_label();
        // `cmp r64, imm32` sign-extends, so only values that survive that
        // round trip may use it.
        if let Ok(imm) = i32::try_from(arm.value.cast_signed()) {
            asm.cmp(value, imm).map_err(asm_err)?;
        } else {
            asm.mov(scratch, arm.value).map_err(asm_err)?;
            asm.cmp(value, scratch).map_err(asm_err)?;
        }
        asm.je(label).map_err(asm_err)?;
        labels.push(label);
    }
    let exit = |asm: &mut CodeAssembler, target: u64| -> Result<(), DispatchError> {
        for register in spec.saved.iter().rev() {
            asm.pop(reg(*register)).map_err(asm_err)?;
        }
        asm.jmp(target).map_err(asm_err)
    };
    exit(&mut asm, spec.fallback)?;
    for (mut label, arm) in labels.into_iter().zip(spec.arms) {
        asm.set_label(&mut label).map_err(asm_err)?;
        exit(&mut asm, arm.body)?;
    }
    asm.assemble(dispatcher_va).map_err(asm_err)
}

/// The leading whole instructions of a VM stub, copied to `displaced_va`
/// followed by a `jmp` to the rest of the stub. A redirect overwrites the first
/// bytes of the stub, so this is how the dispatcher's fallback still reaches the
/// original behaviour.
///
/// Only position-independent, non-branching instructions are accepted: such an
/// instruction means the same thing wherever it executes, so copying is exact.
///
/// # Errors
/// `StubNotRelocatable` for an undecodable, branching or rip-relative
/// instruction in the first `min_len` bytes (or too few bytes);
/// `Assemble` on iced failure.
pub fn emit_displaced_stub(
    stub: &[u8],
    stub_va: u64,
    min_len: usize,
    displaced_va: u64,
) -> Result<Vec<u8>, DispatchError> {
    let mut decoder = Decoder::with_ip(64, stub, stub_va, DecoderOptions::NONE);
    let mut consumed = 0_usize;
    while consumed < min_len {
        if !decoder.can_decode() {
            return Err(DispatchError::StubNotRelocatable);
        }
        let instruction = decoder.decode();
        if instruction.is_invalid()
            || instruction.flow_control() != FlowControl::Next
            || instruction.is_ip_rel_memory_operand()
        {
            return Err(DispatchError::StubNotRelocatable);
        }
        consumed += instruction.len();
    }
    let mut bytes = stub[..consumed].to_vec();
    let asm_err = |e: iced_x86::IcedError| DispatchError::Assemble(e.to_string());
    let mut asm = CodeAssembler::new(64).map_err(asm_err)?;
    let resume = stub_va
        .checked_add(u64::try_from(consumed).map_err(|_| DispatchError::StubNotRelocatable)?)
        .ok_or(DispatchError::StubNotRelocatable)?;
    asm.jmp(resume).map_err(asm_err)?;
    let jump_va = displaced_va
        .checked_add(u64::try_from(consumed).map_err(|_| DispatchError::StubNotRelocatable)?)
        .ok_or(DispatchError::StubNotRelocatable)?;
    bytes.extend(asm.assemble(jump_va).map_err(asm_err)?);
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PhysicalRegister;
    use iced_x86::{Decoder, DecoderOptions, Mnemonic};

    fn mnemonics(bytes: &[u8], ip: u64) -> Vec<Mnemonic> {
        let mut decoder = Decoder::with_ip(64, bytes, ip, DecoderOptions::NONE);
        decoder.iter().map(|insn| insn.mnemonic()).collect()
    }

    #[test]
    fn single_arm_compares_and_branches_then_falls_through() {
        let bytes = emit_domain_dispatcher(
            PhysicalRegister::Rdx,
            &[DispatchArm {
                value: 1,
                body: 0x1_4000_3000,
            }],
            0x1_4000_2000,
            0x1_4000_4000,
        )
        .expect("encode dispatcher");
        let seq = mnemonics(&bytes, 0x1_4000_4000);
        assert_eq!(seq.first().copied(), Some(Mnemonic::Cmp));
        assert!(seq.contains(&Mnemonic::Je));
        assert_eq!(seq.last().copied(), Some(Mnemonic::Jmp)); // fallback
    }

    #[test]
    fn rejects_oversized_domain_value() {
        let err = emit_domain_dispatcher(
            PhysicalRegister::Rdx,
            &[DispatchArm {
                value: u64::from(u32::MAX) + 1,
                body: 0x1000,
            }],
            0x2000,
            0x4000,
        )
        .expect_err("value must fit in imm32");
        assert!(matches!(err, DispatchError::ValueTooWide(_)));
    }

    #[test]
    fn branch_targets_resolve_to_absolute_addresses() {
        let va = 0x1_4000_4000;
        let bytes = emit_domain_dispatcher(
            PhysicalRegister::R9,
            &[
                DispatchArm {
                    value: 1,
                    body: 0x1_4000_3000,
                },
                DispatchArm {
                    value: 7,
                    body: 0x1_4000_3100,
                },
            ],
            0x1_4000_2000,
            va,
        )
        .expect("encode dispatcher");
        let insns: Vec<_> = Decoder::with_ip(64, &bytes, va, DecoderOptions::NONE)
            .iter()
            .collect();
        let targets: Vec<u64> = insns
            .iter()
            .filter(|i| matches!(i.mnemonic(), Mnemonic::Je | Mnemonic::Jmp))
            .map(iced_x86::Instruction::near_branch_target)
            .collect();
        assert_eq!(targets, [0x1_4000_3000, 0x1_4000_3100, 0x1_4000_2000]);
        assert!(
            insns
                .iter()
                .filter(|i| i.mnemonic() == Mnemonic::Cmp)
                .all(|i| { i.op0_register() == iced_x86::Register::R9 })
        );
    }

    const SAVED: [PhysicalRegister; 3] = [
        PhysicalRegister::Rax,
        PhysicalRegister::Rcx,
        PhysicalRegister::R10,
    ];

    fn value_dispatch(arms: &[DispatchArm]) -> ValueDispatch<'_> {
        ValueDispatch {
            value: PhysicalRegister::Rax,
            scratch: PhysicalRegister::Rcx,
            saved: &SAVED,
            compute: 0x1_4000_5000,
            arms,
            fallback: 0x1_4000_2000,
        }
    }

    /// The `pop..; jmp target` tail control reaches at `address`.
    fn exit_target(insns: &[iced_x86::Instruction], address: u64) -> u64 {
        let start = insns
            .iter()
            .position(|i| i.ip() == address)
            .expect("branch lands on an instruction");
        let tail = &insns[start..];
        let pops = tail
            .iter()
            .take_while(|i| i.mnemonic() == Mnemonic::Pop)
            .map(iced_x86::Instruction::op0_register)
            .collect::<Vec<_>>();
        assert_eq!(
            pops,
            [
                iced_x86::Register::R10,
                iced_x86::Register::RCX,
                iced_x86::Register::RAX
            ]
        );
        let jump = &tail[pops.len()];
        assert_eq!(jump.mnemonic(), Mnemonic::Jmp);
        jump.near_branch_target()
    }

    #[test]
    fn value_dispatcher_saves_computes_compares_and_restores_before_each_exit() {
        let va = 0x1_4000_4000;
        let arms = [
            DispatchArm {
                value: 7,
                body: 0x1_4000_3000,
            },
            DispatchArm {
                value: 0x1_4000_9000,
                body: 0x1_4000_3100,
            },
        ];
        let bytes = emit_value_dispatcher(&value_dispatch(&arms), va).expect("encode");
        let insns: Vec<_> = Decoder::with_ip(64, &bytes, va, DecoderOptions::NONE)
            .iter()
            .collect();
        let mnemonics: Vec<_> = insns.iter().map(iced_x86::Instruction::mnemonic).collect();
        assert_eq!(
            &mnemonics[..4],
            [Mnemonic::Push, Mnemonic::Push, Mnemonic::Push, Mnemonic::Call]
        );
        assert_eq!(insns[3].near_branch_target(), 0x1_4000_5000);
        // arm 0: cmp rax, 7 ; arm 1 (wide): mov rcx, imm ; cmp rax, rcx
        assert_eq!(insns[4].mnemonic(), Mnemonic::Cmp);
        assert_eq!(insns[4].op0_register(), iced_x86::Register::RAX);
        assert_eq!(insns[4].immediate(1), 7);
        assert_eq!(insns[5].mnemonic(), Mnemonic::Je);
        assert_eq!(insns[6].mnemonic(), Mnemonic::Mov);
        assert_eq!(insns[6].op0_register(), iced_x86::Register::RCX);
        assert_eq!(insns[6].immediate(1), 0x1_4000_9000);
        assert_eq!(insns[7].mnemonic(), Mnemonic::Cmp);
        assert_eq!(insns[7].op1_register(), iced_x86::Register::RCX);
        assert_eq!(insns[8].mnemonic(), Mnemonic::Je);
        // Fallthrough restores and takes the fallback; each je lands on its own
        // restore-and-jump tail.
        assert_eq!(exit_target(&insns, insns[9].ip()), 0x1_4000_2000);
        assert_eq!(
            exit_target(&insns, insns[5].near_branch_target()),
            0x1_4000_3000
        );
        assert_eq!(
            exit_target(&insns, insns[8].near_branch_target()),
            0x1_4000_3100
        );
    }

    #[test]
    fn value_dispatcher_rejects_duplicates_and_unsaved_registers() {
        let dup = [
            DispatchArm {
                value: 1,
                body: 0x10,
            },
            DispatchArm {
                value: 1,
                body: 0x20,
            },
        ];
        assert!(matches!(
            emit_value_dispatcher(&value_dispatch(&dup), 0x4000),
            Err(DispatchError::DuplicateValue(1))
        ));
        let mut spec = value_dispatch(&[]);
        spec.scratch = PhysicalRegister::R11;
        assert!(matches!(
            emit_value_dispatcher(&spec, 0x4000),
            Err(DispatchError::UnsavedRegister)
        ));
    }

    #[test]
    fn displaced_stub_copies_the_prefix_and_resumes_after_it() {
        // push 0x11223344 ; call ... (only the push is displaced)
        let stub = [0x68, 0x44, 0x33, 0x22, 0x11, 0xE8, 0, 0, 0, 0];
        let bytes = emit_displaced_stub(&stub, 0x1_4000_1000, 5, 0x1_4000_6000).expect("displace");
        assert_eq!(&bytes[..5], &stub[..5]);
        let jump = Decoder::with_ip(64, &bytes[5..], 0x1_4000_6005, DecoderOptions::NONE).decode();
        assert_eq!(jump.mnemonic(), Mnemonic::Jmp);
        assert_eq!(jump.near_branch_target(), 0x1_4000_1005);
    }

    #[test]
    fn displaced_stub_refuses_branches_and_rip_relative_operands() {
        let call = [0xE8, 0, 0, 0, 0, 0x90];
        assert!(matches!(
            emit_displaced_stub(&call, 0x1000, 5, 0x2000),
            Err(DispatchError::StubNotRelocatable)
        ));
        // lea rax, [rip+0] ; nop
        let rip = [0x48, 0x8D, 0x05, 0, 0, 0, 0, 0x90];
        assert!(matches!(
            emit_displaced_stub(&rip, 0x1000, 5, 0x2000),
            Err(DispatchError::StubNotRelocatable)
        ));
        assert!(matches!(
            emit_displaced_stub(&[0x90, 0x90], 0x1000, 5, 0x2000),
            Err(DispatchError::StubNotRelocatable)
        ));
    }
}
