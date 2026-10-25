use std::collections::BTreeMap;

use iced_x86::{FlowControl, Instruction, InstructionInfoFactory, Mnemonic, OpKind, Register};
use serde::Serialize;
use vmpx_disasm::{
    DecodedInstruction, DirectJumpTraceLimits, DisasmError, trace_direct_jump_fragments,
};
use vmpx_pe::{PeImage, Rva};

use crate::{
    BinaryOp, BitWidth, DecodedVmInstruction, Expr, RecoveryStatus, SymbolicVmInitialState,
    VmHandlerSemantic,
    dispatch_extract::{canonical, signed_immediate, writes_register},
};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum X64NativeRegister {
    Rax,
    Rbx,
    Rcx,
    Rdx,
    Rbp,
    Rsi,
    Rdi,
    R8,
    R9,
    R10,
    R11,
    R12,
    R13,
    R14,
    R15,
}

impl X64NativeRegister {
    pub const ALL: [Self; 15] = [
        Self::Rax,
        Self::Rbx,
        Self::Rcx,
        Self::Rdx,
        Self::Rbp,
        Self::Rsi,
        Self::Rdi,
        Self::R8,
        Self::R9,
        Self::R10,
        Self::R11,
        Self::R12,
        Self::R13,
        Self::R14,
        Self::R15,
    ];

    const fn iced(self) -> Register {
        match self {
            Self::Rax => Register::RAX,
            Self::Rbx => Register::RBX,
            Self::Rcx => Register::RCX,
            Self::Rdx => Register::RDX,
            Self::Rbp => Register::RBP,
            Self::Rsi => Register::RSI,
            Self::Rdi => Register::RDI,
            Self::R8 => Register::R8,
            Self::R9 => Register::R9,
            Self::R10 => Register::R10,
            Self::R11 => Register::R11,
            Self::R12 => Register::R12,
            Self::R13 => Register::R13,
            Self::R14 => Register::R14,
            Self::R15 => Register::R15,
        }
    }

    #[must_use]
    pub const fn from_iced(register: Register) -> Option<Self> {
        match register {
            Register::RAX => Some(Self::Rax),
            Register::RBX => Some(Self::Rbx),
            Register::RCX => Some(Self::Rcx),
            Register::RDX => Some(Self::Rdx),
            Register::RBP => Some(Self::Rbp),
            Register::RSI => Some(Self::Rsi),
            Register::RDI => Some(Self::Rdi),
            Register::R8 => Some(Self::R8),
            Register::R9 => Some(Self::R9),
            Register::R10 => Some(Self::R10),
            Register::R11 => Some(Self::R11),
            Register::R12 => Some(Self::R12),
            Register::R13 => Some(Self::R13),
            Register::R14 => Some(Self::R14),
            Register::R15 => Some(Self::R15),
            _ => None,
        }
    }

    #[must_use]
    pub const fn to_iced(self) -> Register {
        match self {
            Self::Rax => Register::RAX,
            Self::Rbx => Register::RBX,
            Self::Rcx => Register::RCX,
            Self::Rdx => Register::RDX,
            Self::Rbp => Register::RBP,
            Self::Rsi => Register::RSI,
            Self::Rdi => Register::RDI,
            Self::R8 => Register::R8,
            Self::R9 => Register::R9,
            Self::R10 => Register::R10,
            Self::R11 => Register::R11,
            Self::R12 => Register::R12,
            Self::R13 => Register::R13,
            Self::R14 => Register::R14,
            Self::R15 => Register::R15,
        }
    }

    #[must_use]
    pub const fn symbolic_name(self) -> &'static str {
        match self {
            Self::Rax => "native_rax_initial",
            Self::Rbx => "native_rbx_initial",
            Self::Rcx => "native_rcx_initial",
            Self::Rdx => "native_rdx_initial",
            Self::Rbp => "native_rbp_initial",
            Self::Rsi => "native_rsi_initial",
            Self::Rdi => "native_rdi_initial",
            Self::R8 => "native_r8_initial",
            Self::R9 => "native_r9_initial",
            Self::R10 => "native_r10_initial",
            Self::R11 => "native_r11_initial",
            Self::R12 => "native_r12_initial",
            Self::R13 => "native_r13_initial",
            Self::R14 => "native_r14_initial",
            Self::R15 => "native_r15_initial",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NativeFrameSource {
    Register { register: X64NativeRegister },
    Rflags,
    Constant { value: u64 },
    ReturnAddress { call_site: Rva, continuation: Rva },
    EntryImmediate { ordinal: u32, value: u64 },
    Unknown { description: String },
}

impl NativeFrameSource {
    fn symbolic_value(&self) -> Option<Expr> {
        match self {
            Self::Register { register } => {
                Some(Expr::register(BitWidth::W64, register.symbolic_name()))
            }
            Self::Rflags => Some(Expr::register(BitWidth::W64, "native_rflags_initial")),
            Self::Constant { value } | Self::EntryImmediate { value, .. } => {
                Some(Expr::constant(BitWidth::W64, *value))
            }
            Self::ReturnAddress { .. } => Some(Expr::register(
                BitWidth::W64,
                "native_return_address_initial",
            )),
            Self::Unknown { .. } => None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct NativeFrameSlot {
    pub frame_offset: u32,
    pub source: NativeFrameSource,
    pub evidence_rva: Option<Rva>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct X64NativeFrameReport {
    pub schema_version: u32,
    pub entry_stub: Rva,
    pub call_site: Option<Rva>,
    pub call_target: Option<Rva>,
    pub cursor_capture: Option<Rva>,
    pub cursor_register: Option<String>,
    pub cursor_architectural_register: Option<X64NativeRegister>,
    pub slots: Vec<NativeFrameSlot>,
    pub status: RecoveryStatus,
    pub confidence: u8,
    pub evidence: Vec<String>,
    pub diagnostics: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct NativeContextBinding {
    pub context_offset: i64,
    pub frame_offset: u32,
    pub source: NativeFrameSource,
    pub vm_instruction_index: usize,
    pub overwrites_binding: Option<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct NativeContextMappingReport {
    pub schema_version: u32,
    pub status: RecoveryStatus,
    pub bindings: Vec<NativeContextBinding>,
    pub final_binding_indices: BTreeMap<i64, usize>,
    pub consumed_frame_slots: usize,
    pub confidence: u8,
    pub evidence: Vec<String>,
    pub diagnostics: Vec<String>,
}

impl NativeContextMappingReport {
    /// Produces the symbolic VM entry state only for a completely proven map.
    /// Sequential duplicate-slot writes are honored, so the last binding wins.
    #[must_use]
    pub fn symbolic_initial_state(&self) -> Option<SymbolicVmInitialState> {
        if self.status != RecoveryStatus::Recovered {
            return None;
        }
        let mut initial = SymbolicVmInitialState::default();
        let frame_base = Expr::register(BitWidth::W64, "vsp_initial");
        for binding in &self.bindings {
            let value = binding.source.symbolic_value()?;
            let address = Expr::binary(
                BinaryOp::Add,
                frame_base.clone(),
                Expr::constant(BitWidth::W64, u64::from(binding.frame_offset)),
            )
            .ok()?;
            initial.memory_values.push((address, value));
        }
        for (offset, binding_index) in &self.final_binding_indices {
            let value = self.bindings.get(*binding_index)?.source.symbolic_value()?;
            initial.context_values.insert(*offset, value);
        }
        Some(initial)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum Provenance {
    Native(X64NativeRegister),
    Constant(u64),
    Unknown,
}

#[derive(Clone, Debug)]
struct ObservedPush {
    source: NativeFrameSource,
    at: Rva,
}

fn initial_provenance() -> BTreeMap<Register, Provenance> {
    X64NativeRegister::ALL
        .into_iter()
        .map(|register| (register.iced(), Provenance::Native(register)))
        .collect()
}

fn defines_flags(instruction: &Instruction) -> bool {
    instruction.rflags_written()
        | instruction.rflags_cleared()
        | instruction.rflags_set()
        | instruction.rflags_undefined()
        != 0
}

fn exact_full_copy(instruction: &Instruction) -> Option<(Register, Register)> {
    (instruction.mnemonic() == Mnemonic::Mov
        && instruction.op0_kind() == OpKind::Register
        && instruction.op1_kind() == OpKind::Register
        && instruction.op0_register().size() == 8
        && instruction.op1_register().size() == 8)
        .then(|| {
            (
                canonical(instruction.op0_register()),
                canonical(instruction.op1_register()),
            )
        })
}

fn exact_full_exchange(instruction: &Instruction) -> Option<(Register, Register)> {
    (instruction.mnemonic() == Mnemonic::Xchg
        && instruction.op0_kind() == OpKind::Register
        && instruction.op1_kind() == OpKind::Register
        && instruction.op0_register().size() == 8
        && instruction.op1_register().size() == 8)
        .then(|| {
            (
                canonical(instruction.op0_register()),
                canonical(instruction.op1_register()),
            )
        })
}

fn exact_zero(instruction: &Instruction) -> Option<Register> {
    (instruction.op0_kind() == OpKind::Register
        && instruction.op0_register().size() >= 4
        && ((instruction.mnemonic() == Mnemonic::Mov
            && signed_immediate(instruction, 1) == Some(0))
            || (matches!(instruction.mnemonic(), Mnemonic::Xor | Mnemonic::Sub)
                && instruction.op1_kind() == OpKind::Register
                && instruction.op0_register() == instruction.op1_register())))
    .then(|| canonical(instruction.op0_register()))
}

fn update_provenance(
    factory: &mut InstructionInfoFactory,
    instruction: &Instruction,
    provenance: &mut BTreeMap<Register, Provenance>,
) {
    if let Some((left, right)) = exact_full_exchange(instruction) {
        let left_value = provenance
            .get(&left)
            .cloned()
            .unwrap_or(Provenance::Unknown);
        let right_value = provenance
            .get(&right)
            .cloned()
            .unwrap_or(Provenance::Unknown);
        provenance.insert(left, right_value);
        provenance.insert(right, left_value);
        return;
    }
    if let Some((destination, source)) = exact_full_copy(instruction) {
        let value = provenance
            .get(&source)
            .cloned()
            .unwrap_or(Provenance::Unknown);
        provenance.insert(destination, value);
        return;
    }
    if let Some(destination) = exact_zero(instruction) {
        provenance.insert(destination, Provenance::Constant(0));
        return;
    }
    for register in X64NativeRegister::ALL {
        let iced = register.iced();
        if writes_register(factory, instruction, iced) {
            provenance.insert(iced, Provenance::Unknown);
        }
    }
}

fn observe_target_push(
    decoded: &DecodedInstruction,
    provenance: &BTreeMap<Register, Provenance>,
    original_flags: bool,
) -> Option<ObservedPush> {
    let instruction = &decoded.instruction;
    let source = match instruction.mnemonic() {
        Mnemonic::Push if instruction.op0_kind() == OpKind::Register => {
            match provenance.get(&canonical(instruction.op0_register())) {
                Some(Provenance::Native(register)) => NativeFrameSource::Register {
                    register: *register,
                },
                Some(Provenance::Constant(value)) => NativeFrameSource::Constant { value: *value },
                _ => NativeFrameSource::Unknown {
                    description: format!(
                        "unproven value of {:?} at save",
                        canonical(instruction.op0_register())
                    ),
                },
            }
        }
        Mnemonic::Push if signed_immediate(instruction, 0).is_some() => {
            NativeFrameSource::Constant {
                value: signed_immediate(instruction, 0)?.cast_unsigned(),
            }
        }
        Mnemonic::Pushfq if original_flags => NativeFrameSource::Rflags,
        Mnemonic::Pushfq => NativeFrameSource::Unknown {
            description: "RFLAGS was modified before PUSHFQ".to_owned(),
        },
        _ => return None,
    };
    Some(ObservedPush {
        source,
        at: decoded.rva,
    })
}

fn caller_push_source(decoded: &DecodedInstruction, ordinal: u32) -> NativeFrameSource {
    let instruction = &decoded.instruction;
    if instruction.mnemonic() == Mnemonic::Push
        && let Some(value) = signed_immediate(instruction, 0)
    {
        return NativeFrameSource::EntryImmediate {
            ordinal,
            value: value.cast_unsigned(),
        };
    }
    NativeFrameSource::Unknown {
        description: format!("unclassified pre-call stack value at {}", decoded.rva),
    }
}

fn is_cursor_capture(instruction: &Instruction) -> Option<Register> {
    (instruction.mnemonic() == Mnemonic::Mov
        && instruction.op0_kind() == OpKind::Register
        && instruction.op1_kind() == OpKind::Register
        && instruction.op0_register().size() == 8
        && canonical(instruction.op0_register()) != Register::RSP
        && canonical(instruction.op1_register()) == Register::RSP)
        .then(|| canonical(instruction.op0_register()))
}

fn full_native_save(pushes: &[ObservedPush]) -> bool {
    let register_counts = pushes.iter().fold(BTreeMap::new(), |mut counts, push| {
        if let NativeFrameSource::Register { register } = push.source {
            *counts.entry(register).or_insert(0_usize) += 1;
        }
        counts
    });
    register_counts.len() == X64NativeRegister::ALL.len()
        && X64NativeRegister::ALL
            .iter()
            .all(|register| register_counts.get(register) == Some(&1))
        && pushes
            .iter()
            .filter(|push| push.source == NativeFrameSource::Rflags)
            .count()
            == 1
}

fn failed_frame(entry_stub: Rva, diagnostics: Vec<String>) -> X64NativeFrameReport {
    X64NativeFrameReport {
        schema_version: 1,
        entry_stub,
        call_site: None,
        call_target: None,
        cursor_capture: None,
        cursor_register: None,
        cursor_architectural_register: None,
        slots: Vec::new(),
        status: RecoveryStatus::Failed,
        confidence: 0,
        evidence: Vec::new(),
        diagnostics,
    }
}

/// Recovers an x64 VM entry's saved-native frame from instruction provenance.
/// The entry must transfer through a direct call; its target may use arbitrarily
/// many direct-jump fragments. A frame is accepted only when all fifteen
/// non-RSP GPR values and unmodified RFLAGS are saved before a full-width
/// `MOV cursor,RSP` capture.
///
/// # Errors
///
/// Returns native decode/mapping failures for the entry or call target. Shape
/// mismatches remain structured failed reports.
#[allow(clippy::too_many_lines)]
pub fn analyze_x64_native_frame(
    pe: &PeImage<'_>,
    entry_stub: Rva,
    limits: DirectJumpTraceLimits,
) -> Result<X64NativeFrameReport, DisasmError> {
    let caller = trace_direct_jump_fragments(pe, entry_stub, limits)?;
    let caller_instructions = caller.instructions().collect::<Vec<_>>();
    let Some((call_index, call)) = caller_instructions
        .iter()
        .enumerate()
        .find(|(_, decoded)| decoded.instruction.flow_control() == FlowControl::Call)
    else {
        return Ok(failed_frame(
            entry_stub,
            vec!["entry trace contains no direct VM-entry call".to_owned()],
        ));
    };
    let Some(call_target) = call.direct_target_rva(pe) else {
        return Ok(failed_frame(
            entry_stub,
            vec!["first entry call has no executable direct target".to_owned()],
        ));
    };
    let Some(continuation) = call.next_rva() else {
        return Ok(failed_frame(
            entry_stub,
            vec!["entry call continuation overflows the RVA address space".to_owned()],
        ));
    };

    let mut caller_pushes = Vec::new();
    for decoded in caller_instructions.iter().copied().take(call_index) {
        let stack_delta = decoded.instruction.stack_pointer_increment();
        if decoded.instruction.mnemonic() == Mnemonic::Push {
            let ordinal = u32::try_from(caller_pushes.len()).unwrap_or(u32::MAX);
            caller_pushes.push(ObservedPush {
                source: caller_push_source(decoded, ordinal),
                at: decoded.rva,
            });
        } else if stack_delta != 0 {
            return Ok(failed_frame(
                entry_stub,
                vec![format!(
                    "unproven pre-call stack mutation at {}",
                    decoded.rva
                )],
            ));
        }
    }

    let target = trace_direct_jump_fragments(pe, call_target, limits)?;
    let mut provenance = initial_provenance();
    let mut original_flags = true;
    let mut pushes = Vec::new();
    let mut factory = InstructionInfoFactory::new();
    let mut candidates = Vec::new();
    let mut invalid_stack_mutation = None;

    for decoded in target.instructions() {
        let instruction = &decoded.instruction;
        if let Some(cursor) = is_cursor_capture(instruction)
            && full_native_save(&pushes)
        {
            candidates.push((decoded.rva, cursor, pushes.clone()));
            break;
        }
        if let Some(observed) = observe_target_push(decoded, &provenance, original_flags) {
            if instruction.stack_pointer_increment() != -8 {
                invalid_stack_mutation =
                    Some(format!("non-qword native-frame push at {}", decoded.rva));
                break;
            }
            pushes.push(observed);
        } else if instruction.stack_pointer_increment() != 0 {
            invalid_stack_mutation = Some(format!(
                "unproven native-frame stack mutation at {}",
                decoded.rva
            ));
            break;
        }
        if instruction.flow_control() == FlowControl::Call {
            invalid_stack_mutation = Some(format!(
                "nested call before native-frame capture at {}",
                decoded.rva
            ));
            break;
        }
        update_provenance(&mut factory, instruction, &mut provenance);
        if defines_flags(instruction) {
            original_flags = false;
        }
    }

    let Some((cursor_capture, cursor, target_pushes)) = candidates.into_iter().next() else {
        let diagnostic = invalid_stack_mutation.unwrap_or_else(|| {
            "no full native-state save followed by a cursor capture was proven".to_owned()
        });
        let mut report = failed_frame(entry_stub, vec![diagnostic]);
        report.call_site = Some(call.rva);
        report.call_target = Some(call_target);
        return Ok(report);
    };

    let mut ordered = target_pushes.into_iter().rev().collect::<Vec<_>>();
    ordered.push(ObservedPush {
        source: NativeFrameSource::ReturnAddress {
            call_site: call.rva,
            continuation,
        },
        at: call.rva,
    });
    ordered.extend(caller_pushes.into_iter().rev());
    let mut slots = Vec::with_capacity(ordered.len());
    for (index, push) in ordered.into_iter().enumerate() {
        let Some(frame_offset) = index
            .checked_mul(8)
            .and_then(|value| u32::try_from(value).ok())
        else {
            return Ok(failed_frame(
                entry_stub,
                vec!["native-frame offset exceeds the supported range".to_owned()],
            ));
        };
        slots.push(NativeFrameSlot {
            frame_offset,
            source: push.source,
            evidence_rva: Some(push.at),
        });
    }
    let has_unknown = slots
        .iter()
        .any(|slot| matches!(slot.source, NativeFrameSource::Unknown { .. }));
    let status = if has_unknown {
        RecoveryStatus::PartiallyRecovered
    } else {
        RecoveryStatus::Recovered
    };
    let confidence = if has_unknown { 80 } else { 100 };
    let mut diagnostics = Vec::new();
    if has_unknown {
        diagnostics.push("one or more saved-frame values lack exact provenance".to_owned());
    }
    Ok(X64NativeFrameReport {
        schema_version: 1,
        entry_stub,
        call_site: Some(call.rva),
        call_target: Some(call_target),
        cursor_capture: Some(cursor_capture),
        cursor_register: Some(format!("{cursor:?}")),
        cursor_architectural_register: X64NativeRegister::from_iced(cursor),
        slots,
        status,
        confidence,
        evidence: vec![
            "entry transfers through an executable direct call".to_owned(),
            "every non-RSP x64 GPR has one provenance-proven saved value".to_owned(),
            "PUSHFQ captures flags with no preceding flags definition in the save chain".to_owned(),
            "saved-frame cursor is captured from native RSP after the complete save".to_owned(),
            "frame offsets follow exact x64 qword push order".to_owned(),
        ],
        diagnostics,
    })
}

/// Pairs a proven native frame with its ordered decoded context-scatter
/// instructions. Duplicate context slots are legal and represented as ordered
/// overwrites; the final binding map selects the last write.
#[must_use]
pub fn map_native_context_scatter(
    frame: &X64NativeFrameReport,
    instructions: &[DecodedVmInstruction],
) -> NativeContextMappingReport {
    let mut bindings = Vec::new();
    let mut final_binding_indices = BTreeMap::new();
    let mut diagnostics = Vec::new();
    if frame.status == RecoveryStatus::Failed {
        diagnostics.push("native saved-frame recovery failed".to_owned());
    } else {
        for (index, slot) in frame.slots.iter().enumerate() {
            let Some(instruction) = instructions.get(index) else {
                diagnostics.push(format!(
                    "context-scatter stream ended after {index} of {} frame slots",
                    frame.slots.len()
                ));
                break;
            };
            if instruction.semantic != VmHandlerSemantic::VirtualRegisterWrite
                || instruction.operation_width != Some(BitWidth::W64)
                || instruction.vsp_delta != Some(8)
            {
                diagnostics.push(format!(
                    "instruction {index} is not a proven qword context scatter consuming one frame slot"
                ));
                break;
            }
            let Some(context_offset) = instruction
                .operand
                .and_then(|value| i64::try_from(value).ok())
            else {
                diagnostics.push(format!(
                    "instruction {index} has no bounded context-slot operand"
                ));
                break;
            };
            let binding_index = bindings.len();
            let overwrites_binding = final_binding_indices.insert(context_offset, binding_index);
            bindings.push(NativeContextBinding {
                context_offset,
                frame_offset: slot.frame_offset,
                source: slot.source.clone(),
                vm_instruction_index: index,
                overwrites_binding,
            });
        }
    }
    let consumed_frame_slots = bindings.len();
    let scatter_complete = diagnostics.is_empty() && consumed_frame_slots == frame.slots.len();
    if scatter_complete && frame.status != RecoveryStatus::Recovered {
        diagnostics.push("one or more bound frame values have incomplete provenance".to_owned());
    }
    let status = if scatter_complete {
        frame.status
    } else if consumed_frame_slots > 0 {
        RecoveryStatus::PartiallyRecovered
    } else {
        RecoveryStatus::Failed
    };
    let evidence = if scatter_complete {
        vec![
            "each native-frame qword is consumed by one proven W64 context write".to_owned(),
            "decoded context offsets are retained in execution order".to_owned(),
            "duplicate context offsets are modeled as ordered overwrites".to_owned(),
        ]
    } else {
        Vec::new()
    };
    NativeContextMappingReport {
        schema_version: 1,
        status,
        bindings,
        final_binding_indices,
        consumed_frame_slots,
        confidence: if scatter_complete {
            instructions
                .iter()
                .take(consumed_frame_slots)
                .map(|instruction| instruction.confidence)
                .min()
                .unwrap_or(frame.confidence)
                .min(frame.confidence)
        } else {
            0
        },
        evidence,
        diagnostics,
    }
}

#[cfg(test)]
mod tests {
    use iced_x86::code_asm::*;
    use vmpx_pe::Va;

    use super::*;
    use crate::{EffectKnowledge, RelativeDispatchState};

    fn executable_pe64(code: &[u8]) -> Vec<u8> {
        let mut data = vec![0_u8; 0x800];
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
        data[section + 8..section + 12].copy_from_slice(&0x600_u32.to_le_bytes());
        data[section + 12..section + 16].copy_from_slice(&0x1000_u32.to_le_bytes());
        data[section + 16..section + 20].copy_from_slice(&0x600_u32.to_le_bytes());
        data[section + 20..section + 24].copy_from_slice(&0x200_u32.to_le_bytes());
        data[section + 36..section + 40].copy_from_slice(&0x6000_0020_u32.to_le_bytes());
        data[0x200..0x200 + code.len()].copy_from_slice(code);
        data
    }

    fn entry_code(clobber_flags_before_save: bool) -> Vec<u8> {
        let mut assembler = CodeAssembler::new(64).expect("x64 assembler");
        let mut save = assembler.create_label();
        assembler.push(0x1234_i32).expect("push entry seed");
        assembler.call(save).expect("call save chain");
        assembler.ret().expect("entry continuation");
        assembler.set_label(&mut save).expect("save label");
        for register in [
            rax, rbx, rcx, rdx, rbp, rsi, rdi, r8, r9, r10, r11, r12, r13, r14, r15,
        ] {
            assembler.push(register).expect("save original GPR");
        }
        if clobber_flags_before_save {
            assembler.test(rax, rax).expect("clobber native flags");
        }
        assembler.pushfq().expect("save native flags");
        assembler.xor(r10d, r10d).expect("build zero sentinel");
        assembler.push(r10).expect("push zero sentinel");
        assembler.mov(r11, rsp).expect("capture saved frame");
        assembler.sub(rsp, 0x180).expect("allocate VM context");
        assembler.jmp(r14).expect("dispatch");
        assembler.assemble(0x1_4000_1000).expect("assemble entry")
    }

    fn analyze(code: &[u8]) -> X64NativeFrameReport {
        let bytes = executable_pe64(code);
        let pe = PeImage::parse(&bytes).expect("valid synthetic PE");
        analyze_x64_native_frame(&pe, Rva(0x1000), DirectJumpTraceLimits::default())
            .expect("entry analysis must decode")
    }

    fn scatter_instruction(index: usize, offset: u64) -> DecodedVmInstruction {
        let state = RelativeDispatchState {
            vip: Va(0x1_4000_1100 + u64::try_from(index).expect("small index")),
            rolling_key: 0,
            handler: Va(0x1_4000_1200),
        };
        DecodedVmInstruction {
            state_before: state,
            encoded_bytes: vec![u8::try_from(index).unwrap_or(u8::MAX)],
            semantic: VmHandlerSemantic::VirtualRegisterWrite,
            semantic_metadata: None,
            input_width: Some(BitWidth::W64),
            operation_width: Some(BitWidth::W64),
            resize: None,
            operand: Some(offset),
            vsp_delta: Some(8),
            flags_effect: EffectKnowledge::Preserved,
            confidence: 100,
            state_after: RelativeDispatchState {
                vip: Va(state.vip.get() + 1),
                ..state
            },
        }
    }

    #[test]
    fn recovers_proven_x64_saved_frame_order() {
        let report = analyze(&entry_code(false));
        assert_eq!(report.status, RecoveryStatus::Recovered, "{report:#?}");
        assert_eq!(report.slots.len(), 19);
        assert_eq!(
            report.slots.first().map(|slot| &slot.source),
            Some(&NativeFrameSource::Constant { value: 0 })
        );
        assert_eq!(
            report.slots.get(1).map(|slot| &slot.source),
            Some(&NativeFrameSource::Rflags)
        );
        assert_eq!(
            report.slots.get(2).map(|slot| &slot.source),
            Some(&NativeFrameSource::Register {
                register: X64NativeRegister::R15
            })
        );
        assert!(matches!(
            report.slots.get(17).map(|slot| &slot.source),
            Some(NativeFrameSource::ReturnAddress { .. })
        ));
        assert!(matches!(
            report.slots.get(18).map(|slot| &slot.source),
            Some(NativeFrameSource::EntryImmediate { value: 0x1234, .. })
        ));
    }

    #[test]
    fn rejects_flags_saved_after_a_real_flags_definition() {
        let report = analyze(&entry_code(true));
        assert_eq!(report.status, RecoveryStatus::Failed, "{report:#?}");
        assert!(report.cursor_capture.is_none());
    }

    #[test]
    fn maps_context_slots_and_models_ordered_overwrite() {
        let frame = analyze(&entry_code(false));
        let mut instructions = (0..frame.slots.len())
            .map(|index| scatter_instruction(index, u64::try_from(index * 8).expect("small")))
            .collect::<Vec<_>>();
        let overwritten_offset = instructions[instructions.len() - 2]
            .operand
            .expect("operand");
        instructions.last_mut().expect("last").operand = Some(overwritten_offset);
        let mapping = map_native_context_scatter(&frame, &instructions);
        assert_eq!(mapping.status, RecoveryStatus::Recovered, "{mapping:#?}");
        assert_eq!(
            mapping.bindings.last().expect("last").overwrites_binding,
            Some(17)
        );
        assert_eq!(
            mapping.final_binding_indices[&i64::try_from(overwritten_offset).expect("small")],
            18
        );
        let initial = mapping.symbolic_initial_state().expect("complete state");
        assert_eq!(
            initial.context_values[&i64::try_from(overwritten_offset).expect("small")],
            Expr::constant(BitWidth::W64, 0x1234)
        );
    }

    #[test]
    fn context_mapping_stops_at_non_scatter_semantics() {
        let frame = analyze(&entry_code(false));
        let mut instructions = (0..frame.slots.len())
            .map(|index| scatter_instruction(index, u64::try_from(index * 8).expect("small")))
            .collect::<Vec<_>>();
        instructions[3].semantic = VmHandlerSemantic::Add;
        let mapping = map_native_context_scatter(&frame, &instructions);
        assert_eq!(mapping.status, RecoveryStatus::PartiallyRecovered);
        assert_eq!(mapping.consumed_frame_slots, 3);
        assert!(mapping.symbolic_initial_state().is_none());
    }

    #[test]
    fn register_symbol_names_are_stable_and_architectural() {
        assert_eq!(X64NativeRegister::Rdx.symbolic_name(), "native_rdx_initial");
        assert_eq!(X64NativeRegister::R15.symbolic_name(), "native_r15_initial");
    }
}
