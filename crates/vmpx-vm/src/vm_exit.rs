use std::collections::{BTreeMap, BTreeSet};

use iced_x86::{InstructionInfoFactory, Mnemonic, OpAccess, OpKind, Register};
use serde::Serialize;
use vmpx_disasm::{
    DirectJumpTraceLimits, DirectJumpTraceStopKind, DisasmError, trace_direct_jump_fragments,
};
use vmpx_pe::{PeImage, Rva};

use crate::{RecoveryStatus, X64NativeRegister};

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct VmExitRestoreSlot {
    pub register: X64NativeRegister,
    pub stack_offset: u32,
    pub pop: Rva,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct VmExitReport {
    pub schema_version: u32,
    pub entry: Rva,
    pub status: RecoveryStatus,
    pub stack_switch: Option<Rva>,
    pub restored_registers: Vec<String>,
    pub restore_slots: Vec<VmExitRestoreSlot>,
    pub flags_restore: Option<Rva>,
    pub flags_stack_offset: Option<u32>,
    pub native_return: Option<Rva>,
    pub return_stack_offset: Option<u32>,
    pub frame_size: Option<u32>,
    pub confidence: u8,
    pub evidence: Vec<String>,
    pub diagnostics: Vec<String>,
}

fn canonical(register: Register) -> Register {
    register.full_register()
}

fn writes_register(
    factory: &mut InstructionInfoFactory,
    instruction: &iced_x86::Instruction,
    register: Register,
) -> bool {
    factory
        .info(instruction)
        .used_registers()
        .iter()
        .any(|used| {
            canonical(used.register()) == canonical(register)
                && matches!(
                    used.access(),
                    OpAccess::Write
                        | OpAccess::CondWrite
                        | OpAccess::ReadWrite
                        | OpAccess::ReadCondWrite
                        | OpAccess::NoMemAccess
                )
        })
}

fn defines_flags(instruction: &iced_x86::Instruction) -> bool {
    instruction.rflags_written()
        | instruction.rflags_cleared()
        | instruction.rflags_set()
        | instruction.rflags_undefined()
        != 0
}

fn failure(entry: Rva, diagnostics: Vec<String>) -> VmExitReport {
    VmExitReport {
        schema_version: 2,
        entry,
        status: RecoveryStatus::Failed,
        stack_switch: None,
        restored_registers: Vec::new(),
        restore_slots: Vec::new(),
        flags_restore: None,
        flags_stack_offset: None,
        native_return: None,
        return_stack_offset: None,
        frame_size: None,
        confidence: 0,
        evidence: Vec::new(),
        diagnostics,
    }
}

/// Proves an x64 VM-exit epilogue by requiring a VSP-to-native-RSP switch,
/// stack restoration of every non-RSP general-purpose register, a final
/// POPFQ, no non-stack memory access, and a native RET. Register mutation junk
/// is accepted only when the final definition of each architectural register
/// is its corresponding POP.
///
/// # Errors
///
/// Returns native decode/mapping failures. Semantic mismatches are retained as
/// a failed structured report.
#[allow(clippy::too_many_lines)]
pub fn analyze_vm_exit(
    pe: &PeImage<'_>,
    entry: Rva,
    vsp: Register,
    limits: DirectJumpTraceLimits,
) -> Result<VmExitReport, DisasmError> {
    let trace = trace_direct_jump_fragments(pe, entry, limits)?;
    if trace.stop.kind != DirectJumpTraceStopKind::Return {
        return Ok(failure(
            entry,
            vec!["native handler does not terminate in RET".to_owned()],
        ));
    }
    let instructions = trace.instructions().collect::<Vec<_>>();
    let stack_switch = instructions.iter().find(|decoded| {
        let instruction = &decoded.instruction;
        instruction.mnemonic() == Mnemonic::Mov
            && instruction.op0_kind() == OpKind::Register
            && instruction.op1_kind() == OpKind::Register
            && canonical(instruction.op0_register()) == Register::RSP
            && canonical(instruction.op1_register()) == canonical(vsp)
    });
    let Some(stack_switch) = stack_switch else {
        return Ok(failure(
            entry,
            vec!["VM exit does not switch native RSP to the recovered VSP".to_owned()],
        ));
    };

    let required = [
        Register::RAX,
        Register::RBX,
        Register::RCX,
        Register::RDX,
        Register::RBP,
        Register::RSI,
        Register::RDI,
        Register::R8,
        Register::R9,
        Register::R10,
        Register::R11,
        Register::R12,
        Register::R13,
        Register::R14,
        Register::R15,
    ];
    let mut factory = InstructionInfoFactory::new();
    let mut final_definitions = BTreeMap::new();
    for register in required {
        for decoded in &instructions {
            if writes_register(&mut factory, &decoded.instruction, register) {
                final_definitions.insert(register, *decoded);
            }
        }
    }
    let mut restored = BTreeSet::new();
    for register in required {
        let Some(decoded) = final_definitions.get(&register) else {
            return Ok(failure(
                entry,
                vec![format!("VM exit never restores {register:?}")],
            ));
        };
        let instruction = &decoded.instruction;
        if instruction.mnemonic() != Mnemonic::Pop
            || instruction.op0_kind() != OpKind::Register
            || canonical(instruction.op0_register()) != register
        {
            return Ok(failure(
                entry,
                vec![format!(
                    "final definition of {register:?} at {} is not its stack POP",
                    decoded.rva
                )],
            ));
        }
        restored.insert(register);
    }

    let flags_restore = instructions
        .iter()
        .rfind(|decoded| defines_flags(&decoded.instruction));
    let Some(flags_restore) = flags_restore else {
        return Ok(failure(
            entry,
            vec!["VM exit does not restore native flags".to_owned()],
        ));
    };
    if flags_restore.instruction.mnemonic() != Mnemonic::Popfq {
        return Ok(failure(
            entry,
            vec![format!(
                "final native flags definition at {} is not POPFQ",
                flags_restore.rva
            )],
        ));
    }

    let return_instruction = instructions
        .last()
        .filter(|decoded| decoded.instruction.mnemonic() == Mnemonic::Ret);
    let Some(return_instruction) = return_instruction else {
        return Ok(failure(
            entry,
            vec!["trace return stop is not a terminal near RET".to_owned()],
        ));
    };
    if return_instruction.instruction.op_count() != 0 {
        return Ok(failure(
            entry,
            vec!["VM exit RET has an unsupported immediate stack adjustment".to_owned()],
        ));
    }

    let stack_switch_index = instructions
        .iter()
        .position(|decoded| decoded.rva == stack_switch.rva)
        .unwrap_or(usize::MAX);
    let mut stack_offset = 0_u32;
    let mut pop_offsets = BTreeMap::new();
    let mut flags_stack_offset = None;
    let mut return_stack_offset = None;
    for decoded in instructions
        .iter()
        .skip(stack_switch_index.saturating_add(1))
    {
        let instruction = &decoded.instruction;
        match instruction.mnemonic() {
            Mnemonic::Pop if instruction.op0_kind() == OpKind::Register => {
                pop_offsets.insert(decoded.rva, stack_offset);
                stack_offset = stack_offset.saturating_add(8);
            }
            Mnemonic::Popfq => {
                flags_stack_offset = Some(stack_offset);
                stack_offset = stack_offset.saturating_add(8);
            }
            Mnemonic::Ret => {
                return_stack_offset = Some(stack_offset);
                stack_offset = stack_offset.saturating_add(8);
            }
            _ if writes_register(&mut factory, instruction, Register::RSP) => {
                return Ok(failure(
                    entry,
                    vec![format!(
                        "VM exit changes RSP outside proven POP/POPFQ/RET consumption at {}",
                        decoded.rva
                    )],
                ));
            }
            _ => {}
        }
        if stack_offset == u32::MAX {
            return Ok(failure(
                entry,
                vec!["VM exit restore-frame size overflows".to_owned()],
            ));
        }
    }

    let mut restore_slots = Vec::with_capacity(required.len());
    for register in required {
        let Some(decoded) = final_definitions.get(&register) else {
            return Ok(failure(
                entry,
                vec![format!(
                    "restored register {register:?} lost its proven final definition"
                )],
            ));
        };
        let Some(stack_offset) = pop_offsets.get(&decoded.rva).copied() else {
            return Ok(failure(
                entry,
                vec![format!(
                    "final POP restoring {register:?} has no proven stack offset"
                )],
            ));
        };
        let Some(register) = X64NativeRegister::from_iced(register) else {
            return Ok(failure(
                entry,
                vec!["restored register has no x64 architectural identity".to_owned()],
            ));
        };
        restore_slots.push(VmExitRestoreSlot {
            register,
            stack_offset,
            pop: decoded.rva,
        });
    }
    restore_slots.sort_by_key(|slot| slot.stack_offset);

    let expected_flags_offset = flags_stack_offset;
    if expected_flags_offset.is_none() || return_stack_offset.is_none() {
        return Ok(failure(
            entry,
            vec!["VM exit has no exact flags/return stack consumption".to_owned()],
        ));
    }

    for decoded in &instructions {
        let instruction = &decoded.instruction;
        let info = factory.info(instruction);
        if !info.used_memory().is_empty()
            && !matches!(
                instruction.mnemonic(),
                Mnemonic::Pop | Mnemonic::Popfq | Mnemonic::Ret
            )
        {
            return Ok(failure(
                entry,
                vec![format!(
                    "VM exit performs non-restore memory access at {}",
                    decoded.rva
                )],
            ));
        }
    }

    let first_pop_index = instructions
        .iter()
        .position(|decoded| decoded.instruction.mnemonic() == Mnemonic::Pop)
        .unwrap_or(0);
    if stack_switch_index >= first_pop_index {
        return Ok(failure(
            entry,
            vec!["native RSP switch does not dominate register restoration".to_owned()],
        ));
    }

    let restored_registers = restored
        .into_iter()
        .map(|register| format!("{register:?}"))
        .collect::<Vec<_>>();
    Ok(VmExitReport {
        schema_version: 2,
        entry,
        status: RecoveryStatus::Recovered,
        stack_switch: Some(stack_switch.rva),
        restored_registers,
        restore_slots,
        flags_restore: Some(flags_restore.rva),
        flags_stack_offset,
        native_return: Some(return_instruction.rva),
        return_stack_offset,
        frame_size: Some(stack_offset),
        confidence: 100,
        evidence: vec![
            "native RSP is restored from the proven VM stack pointer".to_owned(),
            "all fifteen non-RSP x64 general-purpose registers have final definitions from POP"
                .to_owned(),
            "POPFQ is the final native-flags definition".to_owned(),
            "epilogue performs no non-restore memory access".to_owned(),
            "restored native stack supplies the terminal RET continuation".to_owned(),
        ],
        diagnostics: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use iced_x86::code_asm::*;

    use super::*;

    fn executable_pe64(code: &[u8]) -> Vec<u8> {
        let mut data = vec![0_u8; 0x400];
        data[0..2].copy_from_slice(&0x5A4D_u16.to_le_bytes());
        data[0x3C..0x40].copy_from_slice(&0x80_u32.to_le_bytes());
        data[0x80..0x84].copy_from_slice(&0x0000_4550_u32.to_le_bytes());
        let coff = 0x84;
        data[coff..coff + 2].copy_from_slice(&0x8664_u16.to_le_bytes());
        data[coff + 2..coff + 4].copy_from_slice(&1_u16.to_le_bytes());
        data[coff + 16..coff + 18].copy_from_slice(&0xF0_u16.to_le_bytes());
        data[coff + 18..coff + 20].copy_from_slice(&0x2022_u16.to_le_bytes());
        let optional = coff + 20;
        data[optional..optional + 2].copy_from_slice(&0x20B_u16.to_le_bytes());
        data[optional + 16..optional + 20].copy_from_slice(&0x1000_u32.to_le_bytes());
        data[optional + 20..optional + 24].copy_from_slice(&0x1000_u32.to_le_bytes());
        data[optional + 24..optional + 32].copy_from_slice(&0x1_4000_0000_u64.to_le_bytes());
        data[optional + 32..optional + 36].copy_from_slice(&0x1000_u32.to_le_bytes());
        data[optional + 36..optional + 40].copy_from_slice(&0x200_u32.to_le_bytes());
        data[optional + 56..optional + 60].copy_from_slice(&0x2000_u32.to_le_bytes());
        data[optional + 60..optional + 64].copy_from_slice(&0x200_u32.to_le_bytes());
        data[optional + 68..optional + 70].copy_from_slice(&3_u16.to_le_bytes());
        data[optional + 108..optional + 112].copy_from_slice(&16_u32.to_le_bytes());
        let section = optional + 0xF0;
        data[section..section + 5].copy_from_slice(b".text");
        data[section + 8..section + 12].copy_from_slice(&0x200_u32.to_le_bytes());
        data[section + 12..section + 16].copy_from_slice(&0x1000_u32.to_le_bytes());
        data[section + 16..section + 20].copy_from_slice(&0x200_u32.to_le_bytes());
        data[section + 20..section + 24].copy_from_slice(&0x200_u32.to_le_bytes());
        data[section + 36..section + 40].copy_from_slice(&0x6000_0020_u32.to_le_bytes());
        data[0x200..0x200 + code.len()].copy_from_slice(code);
        data
    }

    fn exit_code(clobber_rax_after_restore: bool) -> Vec<u8> {
        let mut assembler = CodeAssembler::new(64).expect("x64 assembler");
        assembler.mov(rsp, rbp).expect("restore native stack");
        assembler.pop(rax).expect("restore RAX");
        if clobber_rax_after_restore {
            assembler.inc(rax).expect("unsafe post-restore clobber");
        }
        for register in [
            rbx, rcx, rdx, rbp, rsi, rdi, r8, r9, r10, r11, r12, r13, r14, r15,
        ] {
            assembler.pop(register).expect("restore GPR");
        }
        assembler.popfq().expect("restore flags");
        assembler.ret().expect("native continuation");
        assembler.assemble(0x1_4000_1000).expect("assemble exit")
    }

    fn analyze(code: &[u8]) -> VmExitReport {
        let bytes = executable_pe64(code);
        let pe = PeImage::parse(&bytes).expect("valid synthetic PE");
        analyze_vm_exit(
            &pe,
            Rva(0x1000),
            Register::RBP,
            DirectJumpTraceLimits::default(),
        )
        .expect("synthetic exit must decode")
    }

    #[test]
    fn proves_complete_x64_native_state_restoration() {
        let report = analyze(&exit_code(false));
        assert_eq!(report.status, RecoveryStatus::Recovered, "{report:#?}");
        assert_eq!(report.restored_registers.len(), 15);
        assert_eq!(report.restore_slots.len(), 15);
        assert_eq!(report.restore_slots[0].register, X64NativeRegister::Rax);
        assert_eq!(report.restore_slots[0].stack_offset, 0);
        assert_eq!(report.restore_slots[14].register, X64NativeRegister::R15);
        assert_eq!(report.restore_slots[14].stack_offset, 112);
        assert!(report.flags_restore.is_some());
        assert_eq!(report.flags_stack_offset, Some(120));
        assert!(report.native_return.is_some());
        assert_eq!(report.return_stack_offset, Some(128));
        assert_eq!(report.frame_size, Some(136));
    }

    #[test]
    fn rejects_register_clobber_after_restore() {
        let report = analyze(&exit_code(true));
        assert_eq!(report.status, RecoveryStatus::Failed);
        assert!(
            report
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.contains("RAX"))
        );
    }
}
