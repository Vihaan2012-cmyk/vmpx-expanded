use crate::PhysicalRegister;
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
}
