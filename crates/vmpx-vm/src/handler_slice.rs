use std::collections::{BTreeMap, BTreeSet};

use iced_x86::{FlowControl, InstructionInfoFactory, OpAccess, Register};
use serde::Serialize;
use vmpx_disasm::{DecodedInstruction, DirectJumpTrace};
use vmpx_pe::Rva;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HandlerSliceConfig {
    pub live_out_registers: Vec<Register>,
    pub live_out_rflags: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeInstructionDisposition {
    LiveDependency,
    ConservativeMemory,
    ConservativeBarrier,
    ControlFlow,
    StructuralDirectJump,
    Dead,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeSliceReason {
    DefinesLiveRegister,
    DefinesLiveFlags,
    MemoryAccess,
    ControlFlow,
    DirectJumpLink,
    BeforeUnknownControlBarrier,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct NativeSliceInstruction {
    pub rva: Rva,
    pub text: String,
    pub disposition: NativeInstructionDisposition,
    pub reasons: Vec<NativeSliceReason>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct NativeBackwardSlice {
    pub instructions: Vec<NativeSliceInstruction>,
    pub live_in_registers: Vec<String>,
    pub live_in_rflags: u32,
    pub retained_instructions: usize,
    pub dead_instructions: usize,
    pub structural_instructions: usize,
    pub conservative_memory: bool,
    pub unknown_control_barriers: Vec<Rva>,
}

fn access_reads(access: OpAccess) -> bool {
    matches!(
        access,
        OpAccess::Read | OpAccess::CondRead | OpAccess::ReadWrite | OpAccess::ReadCondWrite
    )
}

fn access_writes(access: OpAccess) -> bool {
    matches!(
        access,
        OpAccess::Write | OpAccess::CondWrite | OpAccess::ReadWrite | OpAccess::ReadCondWrite
    )
}

fn access_is_unconditional_write(access: OpAccess) -> bool {
    matches!(access, OpAccess::Write | OpAccess::ReadWrite)
}

fn canonical_register(register: Register) -> Register {
    register.full_register()
}

fn register_bit_mask(register: Register) -> u64 {
    let bit_offset = if matches!(
        register,
        Register::AH | Register::BH | Register::CH | Register::DH
    ) {
        8
    } else {
        0
    };
    let bits = register.size().saturating_mul(8);
    if bits >= u64::BITS as usize {
        u64::MAX
    } else {
        ((1_u64 << bits) - 1) << bit_offset
    }
}

fn architectural_write_mask(register: Register) -> u64 {
    let full = canonical_register(register);
    if register == full
        || (register.size() == 4 && full.size() == 8 && register.full_register32() == register)
    {
        u64::MAX
    } else {
        register_bit_mask(register)
    }
}

fn killed_write_mask(register: Register, access: OpAccess) -> u64 {
    if access_is_unconditional_write(access) {
        architectural_write_mask(register)
    } else {
        0
    }
}

fn add_live_bits(live: &mut BTreeMap<Register, u64>, register: Register, bits: u64) {
    if bits != 0 {
        *live.entry(canonical_register(register)).or_default() |= bits;
    }
}

fn is_direct_unconditional_jump(decoded: &DecodedInstruction) -> bool {
    decoded.instruction.flow_control() == FlowControl::UnconditionalBranch
}

fn is_unknown_control_barrier(flow: FlowControl) -> bool {
    matches!(
        flow,
        FlowControl::Call
            | FlowControl::IndirectCall
            | FlowControl::Interrupt
            | FlowControl::Exception
            | FlowControl::XbeginXabortXend
    )
}

/// Computes a conservative backward dependency slice over a linearized handler
/// trace.
///
/// Register aliases are canonicalized and unconditional full-width writes kill
/// older dependencies. Memory accesses are retained because aliasing and fault
/// behavior are not yet proven irrelevant. Direct jumps connecting fragments
/// are labelled structural rather than semantic. Calls and exception-like
/// control transfers form barriers: all predecessor instructions are retained
/// until a calling-convention-aware model is available.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn backward_slice_handler_trace(
    trace: &DirectJumpTrace,
    config: &HandlerSliceConfig,
) -> NativeBackwardSlice {
    let mut live_registers = BTreeMap::new();
    for register in config.live_out_registers.iter().copied() {
        add_live_bits(&mut live_registers, register, register_bit_mask(register));
    }
    let mut live_rflags = config.live_out_rflags;
    let mut reversed = Vec::with_capacity(trace.decoded_instructions);
    let mut info_factory = InstructionInfoFactory::new();
    let mut retain_predecessors = false;
    let mut conservative_memory = false;
    let mut unknown_control_barriers = Vec::new();

    let decoded: Vec<_> = trace.instructions().collect();
    for decoded in decoded.into_iter().rev() {
        let instruction = &decoded.instruction;
        let flow = instruction.flow_control();
        let summary = decoded.summary(&mut info_factory);
        let info = info_factory.info(instruction);
        let register_accesses: Vec<_> = info
            .used_registers()
            .iter()
            .map(|used| (used.register(), used.access()))
            .collect();
        let has_memory_access = !info.used_memory().is_empty();
        let direct_jump_link = is_direct_unconditional_jump(decoded);
        let unknown_barrier = is_unknown_control_barrier(flow);
        let meaningful_control = flow != FlowControl::Next && !direct_jump_link;
        let defines_live_register = register_accesses.iter().any(|(register, access)| {
            access_writes(*access)
                && live_registers
                    .get(&canonical_register(*register))
                    .is_some_and(|live| live & architectural_write_mask(*register) != 0)
        });
        let defined_rflags = instruction.rflags_written()
            | instruction.rflags_cleared()
            | instruction.rflags_set()
            | instruction.rflags_undefined();
        let defines_live_flags = defined_rflags & live_rflags != 0;

        let mut reasons = BTreeSet::new();
        if direct_jump_link {
            reasons.insert(NativeSliceReason::DirectJumpLink);
        }
        if defines_live_register {
            reasons.insert(NativeSliceReason::DefinesLiveRegister);
        }
        if defines_live_flags {
            reasons.insert(NativeSliceReason::DefinesLiveFlags);
        }
        if has_memory_access {
            reasons.insert(NativeSliceReason::MemoryAccess);
            conservative_memory = true;
        }
        if meaningful_control {
            reasons.insert(NativeSliceReason::ControlFlow);
        }
        if retain_predecessors {
            reasons.insert(NativeSliceReason::BeforeUnknownControlBarrier);
        }

        let retained = retain_predecessors
            || defines_live_register
            || defines_live_flags
            || has_memory_access
            || meaningful_control;
        let disposition = if direct_jump_link {
            NativeInstructionDisposition::StructuralDirectJump
        } else if retain_predecessors {
            NativeInstructionDisposition::ConservativeBarrier
        } else if defines_live_register || defines_live_flags {
            NativeInstructionDisposition::LiveDependency
        } else if has_memory_access {
            NativeInstructionDisposition::ConservativeMemory
        } else if meaningful_control {
            NativeInstructionDisposition::ControlFlow
        } else {
            NativeInstructionDisposition::Dead
        };

        if retained {
            for (register, access) in &register_accesses {
                let canonical = canonical_register(*register);
                if let Some(live) = live_registers.get_mut(&canonical) {
                    *live &= !killed_write_mask(*register, *access);
                    if *live == 0 {
                        live_registers.remove(&canonical);
                    }
                }
            }
            for (register, access) in &register_accesses {
                if access_reads(*access) {
                    add_live_bits(&mut live_registers, *register, register_bit_mask(*register));
                }
            }
            live_rflags &= !defined_rflags;
            live_rflags |= instruction.rflags_read();
        }
        if unknown_barrier {
            unknown_control_barriers.push(decoded.rva);
            retain_predecessors = true;
        }

        reversed.push(NativeSliceInstruction {
            rva: decoded.rva,
            text: summary.text,
            disposition,
            reasons: reasons.into_iter().collect(),
        });
    }

    reversed.reverse();
    unknown_control_barriers.reverse();
    let dead_instructions = reversed
        .iter()
        .filter(|item| item.disposition == NativeInstructionDisposition::Dead)
        .count();
    let structural_instructions = reversed
        .iter()
        .filter(|item| item.disposition == NativeInstructionDisposition::StructuralDirectJump)
        .count();
    let retained_instructions = reversed.len() - dead_instructions - structural_instructions;
    NativeBackwardSlice {
        instructions: reversed,
        live_in_registers: live_registers
            .into_keys()
            .map(|register| format!("{register:?}"))
            .collect(),
        live_in_rflags: live_rflags,
        retained_instructions,
        dead_instructions,
        structural_instructions,
        conservative_memory,
        unknown_control_barriers,
    }
}

#[cfg(test)]
mod tests {
    use iced_x86::{Code, Decoder, DecoderOptions};
    use vmpx_disasm::{DirectJumpFragment, DirectJumpTraceStop, DirectJumpTraceStopKind};
    use vmpx_pe::Va;

    use super::*;

    fn trace_from_bytes(bytes: &[u8]) -> DirectJumpTrace {
        let mut decoder = Decoder::with_ip(64, bytes, 0x1_4000_1000, DecoderOptions::NONE);
        let mut instructions = Vec::new();
        while decoder.can_decode() {
            let instruction = decoder.decode();
            assert_ne!(instruction.code(), Code::INVALID);
            let offset = usize::try_from(instruction.ip() - 0x1_4000_1000).expect("small offset");
            let length = instruction.len();
            instructions.push(DecodedInstruction {
                rva: Rva(0x1000 + u32::try_from(offset).expect("small RVA")),
                va: Va(instruction.ip()),
                bytes: bytes[offset..offset + length].to_vec(),
                instruction,
            });
        }
        let terminal = instructions.last().expect("at least one instruction");
        let terminal_rva = terminal.rva;
        let end = terminal.next_rva().expect("small end RVA");
        DirectJumpTrace {
            entry: Rva(0x1000),
            decoded_instructions: instructions.len(),
            fragments: vec![DirectJumpFragment {
                start: Rva(0x1000),
                end,
                instructions,
                next_fragment: None,
            }],
            stop: DirectJumpTraceStop {
                kind: DirectJumpTraceStopKind::IndirectBranch,
                at: terminal_rva,
                direct_target: None,
                fallthrough: None,
                detail: None,
            },
        }
    }

    #[test]
    fn removes_overwritten_register_junk_but_keeps_live_chain() {
        let trace = trace_from_bytes(&[
            0x48, 0xC7, 0xC0, 1, 0, 0, 0, // mov rax, 1 (dead)
            0x48, 0xC7, 0xC0, 2, 0, 0, 0, // mov rax, 2
            0x48, 0x83, 0xC0, 3, // add rax, 3
            0x48, 0xC7, 0xC1, 4, 0, 0, 0, // mov rcx, 4 (dead)
            0xFF, 0xE2, // jmp rdx
        ]);
        let slice = backward_slice_handler_trace(
            &trace,
            &HandlerSliceConfig {
                live_out_registers: vec![Register::RAX],
                live_out_rflags: 0,
            },
        );

        assert_eq!(slice.dead_instructions, 2);
        assert_eq!(
            slice.instructions[0].disposition,
            NativeInstructionDisposition::Dead
        );
        assert_eq!(
            slice.instructions[1].disposition,
            NativeInstructionDisposition::LiveDependency
        );
        assert_eq!(
            slice.instructions[2].disposition,
            NativeInstructionDisposition::LiveDependency
        );
        assert_eq!(
            slice.instructions[3].disposition,
            NativeInstructionDisposition::Dead
        );
        assert_eq!(
            slice.instructions[4].disposition,
            NativeInstructionDisposition::ControlFlow
        );
        assert_eq!(slice.live_in_registers, vec!["RDX"]);
    }

    #[test]
    fn retains_flag_producer_and_memory_effects_conservatively() {
        let trace = trace_from_bytes(&[
            0x48, 0x8B, 0x03, // mov rax, [rbx]
            0x48, 0x03, 0x43, 0x08, // add rax, [rbx+8]
            0x48, 0x89, 0x43, 0x08, // mov [rbx+8], rax
            0x9C, // pushfq
            0x8F, 0x03, // pop [rbx]
            0xFF, 0xE2, // jmp rdx
        ]);
        let slice = backward_slice_handler_trace(
            &trace,
            &HandlerSliceConfig {
                live_out_registers: Vec::new(),
                live_out_rflags: 0,
            },
        );

        assert_eq!(slice.dead_instructions, 0);
        assert!(slice.conservative_memory);
        assert!(
            slice.instructions[1]
                .reasons
                .contains(&NativeSliceReason::DefinesLiveFlags)
        );
        assert!(slice.live_in_registers.contains(&"RBX".to_owned()));
        assert!(slice.live_in_registers.contains(&"RDX".to_owned()));
    }

    #[test]
    fn partial_register_write_kills_the_same_live_bit_range() {
        let trace = trace_from_bytes(&[
            0x66, 0xBE, 0x01, 0x00, // mov si, 1 (dead)
            0x66, 0x83, 0xEE, 0x02, // sub si, 2 (dead)
            0x66, 0xBE, 0x03, 0x00, // mov si, 3
            0x66, 0x01, 0xF0, // add ax, si
            0xFF, 0xE2, // jmp rdx
        ]);
        let slice = backward_slice_handler_trace(
            &trace,
            &HandlerSliceConfig {
                live_out_registers: vec![Register::AX],
                live_out_rflags: 0,
            },
        );

        assert_eq!(
            slice.instructions[0].disposition,
            NativeInstructionDisposition::Dead
        );
        assert_eq!(
            slice.instructions[1].disposition,
            NativeInstructionDisposition::Dead
        );
        assert_eq!(
            slice.instructions[2].disposition,
            NativeInstructionDisposition::LiveDependency
        );
        assert_eq!(
            slice.instructions[3].disposition,
            NativeInstructionDisposition::LiveDependency
        );
    }
}
