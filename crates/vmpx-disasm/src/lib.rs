use std::collections::{BTreeMap, BTreeSet, VecDeque};

use iced_x86::{
    Code, Decoder, DecoderOptions, FlowControl, Formatter, Instruction, InstructionInfoFactory,
    IntelFormatter, Mnemonic, OpKind, Register,
};
use serde::Serialize;
use thiserror::Error;
use vmpx_pe::{FileOffset, PeImage, PeKind, Rva, Va};

mod dangling;
mod dispatch_proof;
mod trace;

pub use dangling::{
    DanglingKind, DanglingRef, TargetClass, classify_target, collect_dangling_refs,
    collect_dangling_refs_with, first_vmp_overlap, is_vmp_section_name, vmp_section_containing,
};
pub use dispatch_proof::{
    RegisterDispatchProof, RegisterDispatchProofLimits, prove_register_dispatch_postlude,
};
pub use trace::{
    DirectJumpFragment, DirectJumpTrace, DirectJumpTraceLimits, DirectJumpTraceStop,
    DirectJumpTraceStopKind, trace_direct_jump_fragments,
};

const IMAGE_SCN_MEM_EXECUTE: u32 = 0x2000_0000;
const MAX_X86_INSTRUCTION_LENGTH: usize = 15;

#[derive(Clone, Debug, Error)]
pub enum DisasmError {
    #[error("address {rva} is not file-backed: {message}")]
    Unmapped { rva: Rva, message: String },

    #[error("invalid instruction at {rva}")]
    InvalidInstruction { rva: Rva },

    #[error("address arithmetic failed at {rva}: {message}")]
    Address { rva: Rva, message: String },
}

#[derive(Clone, Debug)]
pub struct DecodedInstruction {
    pub rva: Rva,
    pub va: Va,
    pub bytes: Vec<u8>,
    pub instruction: Instruction,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RegisterEffect {
    pub register: String,
    pub access: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct MemoryEffect {
    pub segment: String,
    pub base: String,
    pub index: String,
    pub scale: u32,
    pub displacement: u64,
    pub memory_size: String,
    pub access: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct InstructionSummary {
    pub rva: Rva,
    pub va: Va,
    pub length: u32,
    pub bytes: Vec<u8>,
    pub text: String,
    pub mnemonic: String,
    pub flow_control: String,
    pub stack_pointer_increment: i32,
    pub registers: Vec<RegisterEffect>,
    pub memory: Vec<MemoryEffect>,
}

impl DecodedInstruction {
    #[must_use]
    pub fn summary(&self, info_factory: &mut InstructionInfoFactory) -> InstructionSummary {
        let mut formatter = IntelFormatter::new();
        let mut text = String::new();
        formatter.format(&self.instruction, &mut text);
        let info = info_factory.info(&self.instruction);
        let registers = info
            .used_registers()
            .iter()
            .map(|used| RegisterEffect {
                register: format!("{:?}", used.register()),
                access: format!("{:?}", used.access()),
            })
            .collect();
        let memory = info
            .used_memory()
            .iter()
            .map(|used| MemoryEffect {
                segment: format!("{:?}", used.segment()),
                base: format!("{:?}", used.base()),
                index: format!("{:?}", used.index()),
                scale: used.scale(),
                displacement: used.displacement(),
                memory_size: format!("{:?}", used.memory_size()),
                access: format!("{:?}", used.access()),
            })
            .collect();
        InstructionSummary {
            rva: self.rva,
            va: self.va,
            length: u32::try_from(self.instruction.len()).unwrap_or(0),
            bytes: self.bytes.clone(),
            text,
            mnemonic: format!("{:?}", self.instruction.mnemonic()),
            flow_control: format!("{:?}", self.instruction.flow_control()),
            stack_pointer_increment: stack_pointer_increment(&self.instruction),
            registers,
            memory,
        }
    }

    #[must_use]
    pub fn next_rva(&self) -> Option<Rva> {
        u32::try_from(self.instruction.len())
            .ok()
            .and_then(|length| self.rva.checked_add(length))
    }

    #[must_use]
    pub fn direct_target_rva(&self, pe: &PeImage<'_>) -> Option<Rva> {
        if !matches!(
            self.instruction.op0_kind(),
            OpKind::NearBranch16 | OpKind::NearBranch32 | OpKind::NearBranch64
        ) {
            return None;
        }
        pe.optional
            .image_base
            .checked_rva_of(Va(self.instruction.near_branch_target()))
    }
}

fn stack_pointer_increment(instruction: &Instruction) -> i32 {
    let implicit = instruction.stack_pointer_increment();
    if implicit != 0 {
        return implicit;
    }
    if !matches!(instruction.mnemonic(), Mnemonic::Add | Mnemonic::Sub)
        || !matches!(
            instruction.op0_register(),
            Register::RSP | Register::ESP | Register::SP
        )
    {
        return 0;
    }
    let immediate = match instruction.try_immediate(1) {
        Ok(value) => i64::from_ne_bytes(value.to_ne_bytes()),
        Err(_) => return 0,
    };
    let Ok(delta) = i32::try_from(immediate) else {
        return 0;
    };
    match instruction.mnemonic() {
        Mnemonic::Add => delta,
        Mnemonic::Sub => delta.checked_neg().unwrap_or(0),
        _ => 0,
    }
}

fn bitness(kind: PeKind) -> u32 {
    match kind {
        PeKind::Pe32 => 32,
        PeKind::Pe32Plus => 64,
    }
}

fn section_file_window<'a>(pe: &'a PeImage<'a>, rva: Rva) -> Result<&'a [u8], DisasmError> {
    let section = pe
        .sections
        .iter()
        .find(|section| section.contains_mapped_rva(rva))
        .ok_or_else(|| DisasmError::Unmapped {
            rva,
            message: "no section owns the RVA".to_owned(),
        })?;
    let delta = rva
        .0
        .checked_sub(section.virtual_address.0)
        .ok_or_else(|| DisasmError::Address {
            rva,
            message: "section-relative offset underflows".to_owned(),
        })?;
    if delta >= section.size_of_raw_data || section.pointer_to_raw_data.0 == 0 {
        return Err(DisasmError::Unmapped {
            rva,
            message: format!("section {} bytes are virtual-only", section.name),
        });
    }
    let start = section
        .pointer_to_raw_data
        .0
        .checked_add(u64::from(delta))
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| DisasmError::Address {
            rva,
            message: "raw start cannot be represented".to_owned(),
        })?;
    let section_end = section
        .pointer_to_raw_data
        .0
        .checked_add(u64::from(section.size_of_raw_data))
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| DisasmError::Address {
            rva,
            message: "raw section end cannot be represented".to_owned(),
        })?
        .min(pe.data().len());
    pe.data()
        .get(start..section_end)
        .ok_or_else(|| DisasmError::Unmapped {
            rva,
            message: "raw section window exceeds file".to_owned(),
        })
}

/// Decodes one instruction at an RVA with the PE's architecture and image base.
///
/// # Errors
///
/// Returns an error for unmapped bytes, invalid instructions, or address
/// arithmetic failures.
pub fn decode_one(pe: &PeImage<'_>, rva: Rva) -> Result<DecodedInstruction, DisasmError> {
    let bytes = section_file_window(pe, rva)?;
    let va = pe
        .optional
        .image_base
        .checked_add_rva(rva)
        .ok_or_else(|| DisasmError::Address {
            rva,
            message: "VA overflows".to_owned(),
        })?;
    let mut decoder =
        Decoder::with_ip(bitness(pe.optional.kind), bytes, va.0, DecoderOptions::NONE);
    let instruction = decoder.decode();
    if instruction.code() == Code::INVALID || instruction.len() == 0 {
        return Err(DisasmError::InvalidInstruction { rva });
    }
    let length = instruction.len();
    let instruction_bytes = bytes
        .get(..length)
        .ok_or_else(|| DisasmError::Unmapped {
            rva,
            message: "instruction crosses file-backed section end".to_owned(),
        })?
        .to_vec();
    Ok(DecodedInstruction {
        rva,
        va,
        bytes: instruction_bytes,
        instruction,
    })
}

#[must_use]
pub fn is_executable_rva(pe: &PeImage<'_>, rva: Rva) -> bool {
    pe.sections.iter().any(|section| {
        section.contains_mapped_rva(rva) && section.characteristics & IMAGE_SCN_MEM_EXECUTE != 0
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CfgLimits {
    pub max_blocks: usize,
    pub max_instructions: usize,
    pub max_instructions_per_block: usize,
}

impl Default for CfgLimits {
    fn default() -> Self {
        Self {
            max_blocks: 10_000,
            max_instructions: 1_000_000,
            max_instructions_per_block: 10_000,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CfgStatus {
    Complete,
    ResourceLimit,
    DecodeError,
}

#[derive(Clone, Debug, Serialize)]
pub struct BasicBlock {
    pub start: Rva,
    pub end: Rva,
    pub instructions: Vec<InstructionSummary>,
    pub successors: Vec<Rva>,
    pub direct_calls: Vec<Rva>,
    pub terminal_flow: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct ControlFlowGraph {
    pub entry: Rva,
    pub status: CfgStatus,
    pub blocks: Vec<BasicBlock>,
    pub errors: Vec<String>,
}

/// Recovers a bounded native CFG by following direct branches from an RVA.
/// Calls are recorded but not recursively traversed.
///
/// # Errors
///
/// Required entry bytes must be executable and file-backed. Later decode
/// failures are retained in the returned graph status and error list.
#[allow(clippy::too_many_lines)]
pub fn recover_cfg(
    pe: &PeImage<'_>,
    entry: Rva,
    limits: CfgLimits,
) -> Result<ControlFlowGraph, DisasmError> {
    if !is_executable_rva(pe, entry) {
        return Err(DisasmError::Unmapped {
            rva: entry,
            message: "CFG entry is not executable".to_owned(),
        });
    }
    decode_one(pe, entry)?;
    let mut queue = VecDeque::from([entry]);
    let mut queued = BTreeSet::from([entry]);
    let mut blocks = BTreeMap::new();
    let mut instruction_total = 0_usize;
    let mut status = CfgStatus::Complete;
    let mut errors = Vec::new();
    let mut info_factory = InstructionInfoFactory::new();

    while let Some(start) = queue.pop_front() {
        if blocks.contains_key(&start) {
            continue;
        }
        if blocks.len() >= limits.max_blocks || instruction_total >= limits.max_instructions {
            status = CfgStatus::ResourceLimit;
            break;
        }
        let mut current = start;
        let mut summaries = Vec::new();
        let mut successors = Vec::new();
        let mut calls = Vec::new();
        let terminal_flow;
        loop {
            if summaries.len() >= limits.max_instructions_per_block
                || instruction_total >= limits.max_instructions
            {
                status = CfgStatus::ResourceLimit;
                terminal_flow = "resource_limit".to_owned();
                break;
            }
            if current != start && queued.contains(&current) {
                successors.push(current);
                terminal_flow = "existing_block".to_owned();
                break;
            }
            let decoded = match decode_one(pe, current) {
                Ok(value) => value,
                Err(error) => {
                    status = CfgStatus::DecodeError;
                    errors.push(error.to_string());
                    terminal_flow = "decode_error".to_owned();
                    break;
                }
            };
            instruction_total += 1;
            let flow = decoded.instruction.flow_control();
            let next = decoded.next_rva();
            let direct_target = decoded.direct_target_rva(pe);
            summaries.push(decoded.summary(&mut info_factory));
            match flow {
                FlowControl::ConditionalBranch => {
                    if let Some(target) = direct_target.filter(|rva| is_executable_rva(pe, *rva)) {
                        successors.push(target);
                    }
                    if let Some(fallthrough) = next.filter(|rva| is_executable_rva(pe, *rva)) {
                        successors.push(fallthrough);
                    }
                    terminal_flow = "conditional_branch".to_owned();
                    break;
                }
                FlowControl::UnconditionalBranch => {
                    if let Some(target) = direct_target.filter(|rva| is_executable_rva(pe, *rva)) {
                        successors.push(target);
                    }
                    terminal_flow = "unconditional_branch".to_owned();
                    break;
                }
                FlowControl::Call => {
                    if let Some(target) = direct_target.filter(|rva| is_executable_rva(pe, *rva)) {
                        calls.push(target);
                    }
                }
                FlowControl::Return => {
                    terminal_flow = "return".to_owned();
                    break;
                }
                FlowControl::IndirectBranch
                | FlowControl::Interrupt
                | FlowControl::Exception
                | FlowControl::XbeginXabortXend => {
                    terminal_flow = format!("{flow:?}").to_lowercase();
                    break;
                }
                FlowControl::IndirectCall | FlowControl::Next => {}
            }
            let Some(next) = next.filter(|rva| is_executable_rva(pe, *rva)) else {
                terminal_flow = "unmapped_fallthrough".to_owned();
                break;
            };
            current = next;
        }
        let end = summaries
            .last()
            .and_then(|summary| summary.rva.checked_add(summary.length))
            .unwrap_or(current);
        successors.sort_unstable();
        successors.dedup();
        for successor in &successors {
            if queued.insert(*successor) {
                queue.push_back(*successor);
            }
        }
        blocks.insert(
            start,
            BasicBlock {
                start,
                end,
                instructions: summaries,
                successors,
                direct_calls: calls,
                terminal_flow,
            },
        );
    }

    Ok(ControlFlowGraph {
        entry,
        status,
        blocks: blocks.into_values().collect(),
        errors,
    })
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub struct IpRelativeReference {
    pub instruction_rva: Rva,
    pub instruction_va: Va,
    pub instruction_length: u32,
    pub target_rva: Rva,
    pub target_va: Va,
    pub mnemonic: String,
    pub flow_control: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ResyncReferenceReport {
    pub references: Vec<IpRelativeReference>,
    pub candidate_decodes: usize,
    pub resource_limited: bool,
}

fn normalize_rva_ranges(ranges: &[(Rva, Rva)]) -> Vec<(Rva, Rva)> {
    let mut ranges: Vec<_> = ranges
        .iter()
        .copied()
        .filter(|(begin, end)| begin < end)
        .collect();
    ranges.sort_unstable();
    let mut normalized: Vec<(Rva, Rva)> = Vec::with_capacity(ranges.len());
    for (begin, end) in ranges {
        if let Some((_, previous_end)) = normalized.last_mut()
            && begin <= *previous_end
        {
            *previous_end = (*previous_end).max(end);
        } else {
            normalized.push((begin, end));
        }
    }
    normalized
}

fn containing_range_end(ranges: &[(Rva, Rva)], rva: Rva) -> Option<Rva> {
    let index = ranges.partition_point(|(begin, _)| *begin <= rva);
    index
        .checked_sub(1)
        .and_then(|index| ranges.get(index))
        .filter(|(_, end)| rva < *end)
        .map(|(_, end)| *end)
}

fn candidate_modrm_spans(
    section_start: Rva,
    section_size: u32,
    included_ranges: Option<&[(Rva, Rva)]>,
) -> Result<Vec<(usize, usize)>, DisasmError> {
    let section_end =
        section_start
            .checked_add(section_size)
            .ok_or_else(|| DisasmError::Address {
                rva: section_start,
                message: "executable section RVA range overflows".to_owned(),
            })?;
    let section_size = usize::try_from(section_size).map_err(|_| DisasmError::Address {
        rva: section_start,
        message: "executable section size cannot be represented".to_owned(),
    })?;
    let Some(included_ranges) = included_ranges else {
        return Ok(vec![(0, section_size)]);
    };
    let mut spans = Vec::new();
    for (range_begin, range_end) in included_ranges {
        let begin = (*range_begin).max(section_start);
        let end = (*range_end).min(section_end);
        if begin >= end {
            continue;
        }
        let begin_delta =
            usize::try_from(begin.0 - section_start.0).map_err(|_| DisasmError::Address {
                rva: begin,
                message: "included range offset cannot be represented".to_owned(),
            })?;
        let end_delta =
            usize::try_from(end.0 - section_start.0).map_err(|_| DisasmError::Address {
                rva: end,
                message: "included range end offset cannot be represented".to_owned(),
            })?;
        let modrm_begin = begin_delta.saturating_add(1).min(section_size);
        let modrm_end = end_delta
            .saturating_add(MAX_X86_INSTRUCTION_LENGTH - 1)
            .min(section_size);
        if modrm_begin < modrm_end {
            spans.push((modrm_begin, modrm_end));
        }
    }
    spans.sort_unstable();
    let mut merged: Vec<(usize, usize)> = Vec::with_capacity(spans.len());
    for (begin, end) in spans {
        if let Some((_, previous_end)) = merged.last_mut()
            && begin <= *previous_end
        {
            *previous_end = (*previous_end).max(end);
        } else {
            merged.push((begin, end));
        }
    }
    Ok(merged)
}

/// Finds RIP/EIP-relative memory references from bounded function ranges into a
/// target RVA range.
///
/// # Errors
///
/// Returns an error when target arithmetic overflows. Individual invalid or
/// unmapped function bytes terminate that range without panicking.
pub fn find_ip_relative_references(
    pe: &PeImage<'_>,
    functions: &[(Rva, Rva)],
    target_start: Rva,
    target_size: u32,
    max_instructions: usize,
) -> Result<Vec<IpRelativeReference>, DisasmError> {
    let target_end =
        target_start
            .0
            .checked_add(target_size)
            .ok_or_else(|| DisasmError::Address {
                rva: target_start,
                message: "target range overflows".to_owned(),
            })?;
    let mut references = BTreeSet::new();
    let mut decoded_total = 0_usize;
    for (begin, end) in functions {
        let mut current = *begin;
        while current < *end && decoded_total < max_instructions {
            let Ok(decoded) = decode_one(pe, current) else {
                break;
            };
            decoded_total += 1;
            if decoded.instruction.is_ip_rel_memory_operand() {
                let target_va = Va(decoded.instruction.ip_rel_memory_address());
                if let Some(target_rva) = pe.optional.image_base.checked_rva_of(target_va)
                    && target_rva.0 >= target_start.0
                    && target_rva.0 < target_end
                {
                    references.insert(IpRelativeReference {
                        instruction_rva: decoded.rva,
                        instruction_va: decoded.va,
                        instruction_length: u32::try_from(decoded.instruction.len()).unwrap_or(0),
                        target_rva,
                        target_va,
                        mnemonic: format!("{:?}", decoded.instruction.mnemonic()),
                        flow_control: format!("{:?}", decoded.instruction.flow_control()),
                    });
                }
            }
            let Some(next) = decoded.next_rva() else {
                break;
            };
            current = next;
        }
        if decoded_total >= max_instructions {
            break;
        }
    }
    Ok(references.into_iter().collect())
}

#[allow(clippy::too_many_lines)]
fn scan_ip_relative_reference_candidates(
    pe: &PeImage<'_>,
    target_rvas: &BTreeSet<Rva>,
    included_source_ranges: Option<&[(Rva, Rva)]>,
    excluded_source_ranges: &[(Rva, Rva)],
    max_candidate_decodes: usize,
) -> Result<ResyncReferenceReport, DisasmError> {
    if target_rvas.is_empty() {
        return Ok(ResyncReferenceReport {
            references: Vec::new(),
            candidate_decodes: 0,
            resource_limited: false,
        });
    }
    let mut references = BTreeSet::new();
    let mut candidate_decodes = 0_usize;
    let mut resource_limited = false;
    let included_source_ranges = included_source_ranges.map(normalize_rva_ranges);
    let included_source_ranges = included_source_ranges.as_deref();
    let excluded_source_ranges = normalize_rva_ranges(excluded_source_ranges);
    let mut executable_sections: Vec<_> = pe
        .sections
        .iter()
        .filter(|section| section.characteristics & IMAGE_SCN_MEM_EXECUTE != 0)
        .collect();
    executable_sections.sort_by_key(|section| {
        (
            !section.contains_mapped_rva(pe.optional.address_of_entry_point),
            section.index,
        )
    });
    'sections: for section in executable_sections {
        if section.pointer_to_raw_data.0 == 0 || section.size_of_raw_data == 0 {
            continue;
        }
        let start =
            usize::try_from(section.pointer_to_raw_data.0).map_err(|_| DisasmError::Address {
                rva: section.virtual_address,
                message: "executable section raw offset cannot be represented".to_owned(),
            })?;
        let size = usize::try_from(section.size_of_raw_data).map_err(|_| DisasmError::Address {
            rva: section.virtual_address,
            message: "executable section raw size cannot be represented".to_owned(),
        })?;
        let end = start
            .checked_add(size)
            .filter(|end| *end <= pe.data().len())
            .ok_or_else(|| DisasmError::Address {
                rva: section.virtual_address,
                message: "executable section raw range exceeds the file".to_owned(),
            })?;
        let bytes = &pe.data()[start..end];
        let section_rva_end = section
            .virtual_address
            .checked_add(section.size_of_raw_data)
            .ok_or_else(|| DisasmError::Address {
                rva: section.virtual_address,
                message: "executable section RVA range overflows".to_owned(),
            })?;
        let spans = candidate_modrm_spans(
            section.virtual_address,
            section.size_of_raw_data,
            included_source_ranges,
        )?;
        for (span_begin, span_end) in spans {
            for modrm_offset in span_begin..span_end {
                let modrm = bytes[modrm_offset];
                // x86/x64 RIP-relative memory operands use mod=00, r/m=101.
                // This only prioritizes starts; decoded semantics decide.
                if modrm & 0xC7 != 0x05 {
                    continue;
                }
                for lookback in 1..=modrm_offset.min(MAX_X86_INSTRUCTION_LENGTH.saturating_sub(1)) {
                    let instruction_offset = modrm_offset - lookback;
                    let Some(delta) = u32::try_from(instruction_offset).ok() else {
                        continue;
                    };
                    let Some(candidate_rva) = section.virtual_address.checked_add(delta) else {
                        continue;
                    };
                    let included_end = match included_source_ranges {
                        Some(ranges) => {
                            let Some(end) = containing_range_end(ranges, candidate_rva) else {
                                continue;
                            };
                            end.min(section_rva_end)
                        }
                        None => section_rva_end,
                    };
                    if containing_range_end(&excluded_source_ranges, candidate_rva).is_some() {
                        continue;
                    }
                    if candidate_decodes >= max_candidate_decodes {
                        resource_limited = true;
                        break 'sections;
                    }
                    candidate_decodes = candidate_decodes.saturating_add(1);
                    let Some(candidate_ip) = pe.optional.image_base.checked_add_rva(candidate_rva)
                    else {
                        continue;
                    };
                    let mut decoder = Decoder::with_ip(
                        bitness(pe.optional.kind),
                        &bytes[instruction_offset..],
                        candidate_ip.0,
                        DecoderOptions::NONE,
                    );
                    let instruction = decoder.decode();
                    let Some(instruction_length) = u32::try_from(instruction.len()).ok() else {
                        continue;
                    };
                    if instruction.code() == Code::INVALID
                        || instruction_length == 0
                        || instruction.len() <= lookback
                        || candidate_rva
                            .checked_add(instruction_length)
                            .is_none_or(|instruction_end| instruction_end > included_end)
                        || !instruction.is_ip_rel_memory_operand()
                    {
                        continue;
                    }
                    let target_address = Va(instruction.ip_rel_memory_address());
                    let Some(resolved_target_rva) =
                        pe.optional.image_base.checked_rva_of(target_address)
                    else {
                        continue;
                    };
                    if !target_rvas.contains(&resolved_target_rva) {
                        continue;
                    }
                    let mnemonic = instruction.mnemonic();
                    let flow_control = instruction.flow_control();
                    references.insert(IpRelativeReference {
                        instruction_rva: candidate_rva,
                        instruction_va: candidate_ip,
                        instruction_length,
                        target_rva: resolved_target_rva,
                        target_va: target_address,
                        mnemonic: format!("{mnemonic:?}"),
                        flow_control: format!("{flow_control:?}"),
                    });
                }
            }
        }
    }
    let mut longest_by_end_and_target = BTreeMap::new();
    for reference in references {
        let Some(end) = reference
            .instruction_rva
            .checked_add(reference.instruction_length)
        else {
            continue;
        };
        let key = (end, reference.target_rva);
        let replace =
            longest_by_end_and_target
                .get(&key)
                .is_none_or(|previous: &IpRelativeReference| {
                    reference.instruction_length > previous.instruction_length
                });
        if replace {
            longest_by_end_and_target.insert(key, reference);
        }
    }
    Ok(ResyncReferenceReport {
        references: longest_by_end_and_target.into_values().collect(),
        candidate_decodes,
        resource_limited,
    })
}

/// Searches independently decoded instruction starts in executable sections.
///
/// A `ModRM` shape is used only to prioritize possible starts. A reference is
/// retained only when iced-x86 decodes a valid IP-relative memory operand whose
/// effective address is an exact member of `target_rvas`. Findings are not
/// claimed reachable merely because they decode. Every attempted start,
/// including an invalid instruction, consumes the candidate-decode budget.
///
/// # Errors
///
/// Returns an error when section address arithmetic cannot be represented.
pub fn find_ip_relative_references_resync(
    pe: &PeImage<'_>,
    target_rvas: &BTreeSet<Rva>,
    excluded_source_ranges: &[(Rva, Rva)],
    max_candidate_decodes: usize,
) -> Result<ResyncReferenceReport, DisasmError> {
    scan_ip_relative_reference_candidates(
        pe,
        target_rvas,
        None,
        excluded_source_ranges,
        max_candidate_decodes,
    )
}

/// Searches independent instruction starts inside bounded source ranges.
///
/// Overlapping source ranges are merged before scanning, so hostile exception
/// metadata cannot spend the budget repeatedly on the same bytes. Invalid
/// instructions at earlier starts do not terminate a range. Results still
/// require an exact decoded reference to one of `target_rvas` and remain
/// reachability-unproven candidates.
///
/// # Errors
///
/// Returns an error when section or range address arithmetic cannot be
/// represented.
pub fn find_ip_relative_references_resync_in_ranges(
    pe: &PeImage<'_>,
    target_rvas: &BTreeSet<Rva>,
    source_ranges: &[(Rva, Rva)],
    max_candidate_decodes: usize,
) -> Result<ResyncReferenceReport, DisasmError> {
    scan_ip_relative_reference_candidates(
        pe,
        target_rvas,
        Some(source_ranges),
        &[],
        max_candidate_decodes,
    )
}

#[must_use]
pub fn file_offset_for_instruction(
    pe: &PeImage<'_>,
    instruction: &DecodedInstruction,
) -> Option<FileOffset> {
    pe.rva_to_file_offset(instruction.rva).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn executable_pe64(code: &[u8]) -> Vec<u8> {
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
        data[optional + 16..optional + 20].copy_from_slice(&(0x1000_u32).to_le_bytes());
        data[optional + 20..optional + 24].copy_from_slice(&(0x1000_u32).to_le_bytes());
        data[optional + 24..optional + 32].copy_from_slice(&(0x1_4000_0000_u64).to_le_bytes());
        data[optional + 32..optional + 36].copy_from_slice(&(0x1000_u32).to_le_bytes());
        data[optional + 36..optional + 40].copy_from_slice(&(0x200_u32).to_le_bytes());
        data[optional + 56..optional + 60].copy_from_slice(&(0x2000_u32).to_le_bytes());
        data[optional + 60..optional + 64].copy_from_slice(&(0x200_u32).to_le_bytes());
        data[optional + 68..optional + 70].copy_from_slice(&(3_u16).to_le_bytes());
        data[optional + 108..optional + 112].copy_from_slice(&(16_u32).to_le_bytes());
        let section = optional + 0xF0;
        data[section..section + 5].copy_from_slice(b".text");
        data[section + 8..section + 12].copy_from_slice(&(0x200_u32).to_le_bytes());
        data[section + 12..section + 16].copy_from_slice(&(0x1000_u32).to_le_bytes());
        data[section + 16..section + 20].copy_from_slice(&(0x200_u32).to_le_bytes());
        data[section + 20..section + 24].copy_from_slice(&(0x200_u32).to_le_bytes());
        data[section + 36..section + 40].copy_from_slice(&(0x6000_0020_u32).to_le_bytes());
        data[0x200..0x200 + code.len()].copy_from_slice(code);
        data
    }

    #[test]
    fn instruction_summary_captures_stack_and_control_flow() {
        let bytes = [0x48, 0x83, 0xEC, 0x20, 0xC3];
        let mut decoder = Decoder::with_ip(64, &bytes, 0x1400_1000, DecoderOptions::NONE);
        let instruction = decoder.decode();
        let decoded_instruction = DecodedInstruction {
            rva: Rva(0x1000),
            va: Va(0x1400_1000),
            bytes: bytes[..4].to_vec(),
            instruction,
        };
        let summary = decoded_instruction.summary(&mut InstructionInfoFactory::new());
        assert_eq!(summary.text, "sub rsp,20h");
        assert_eq!(summary.stack_pointer_increment, -32);
    }

    #[test]
    fn range_resync_recovers_after_invalid_start_and_merges_overlaps() {
        let mut code = vec![0x90_u8; 0x100];
        code[..16].fill(0x66);
        // mov rax,[rip+0xD9] at RVA 0x1020 -> exact slot RVA 0x1100
        code[0x20..0x27].copy_from_slice(&[0x48, 0x8B, 0x05, 0xD9, 0x00, 0x00, 0x00]);
        let data = executable_pe64(&code);
        let pe = PeImage::parse(&data).expect("valid synthetic PE");
        assert!(decode_one(&pe, Rva(0x1000)).is_err());
        assert!(
            find_ip_relative_references(&pe, &[(Rva(0x1000), Rva(0x1040))], Rva(0x1100), 8, 100,)
                .expect("linear scan must stay bounded")
                .is_empty()
        );

        let targets = BTreeSet::from([Rva(0x1100)]);
        let overlapping = find_ip_relative_references_resync_in_ranges(
            &pe,
            &targets,
            &[(Rva(0x1000), Rva(0x1030)), (Rva(0x1008), Rva(0x1040))],
            1_000,
        )
        .expect("overlapping bounded ranges must scan");
        let merged = find_ip_relative_references_resync_in_ranges(
            &pe,
            &targets,
            &[(Rva(0x1000), Rva(0x1040))],
            1_000,
        )
        .expect("merged bounded range must scan");

        assert!(!overlapping.resource_limited);
        assert_eq!(overlapping.candidate_decodes, merged.candidate_decodes);
        assert_eq!(overlapping.references, merged.references);
        assert_eq!(overlapping.references.len(), 1);
        assert_eq!(overlapping.references[0].instruction_rva, Rva(0x1020));
        assert_eq!(overlapping.references[0].target_rva, Rva(0x1100));
    }

    #[test]
    fn range_resync_reports_candidate_budget_exhaustion() {
        let mut code = vec![0x90_u8; 0x40];
        code[0x20..0x27].copy_from_slice(&[0x48, 0x8B, 0x05, 0xD9, 0x00, 0x00, 0x00]);
        let data = executable_pe64(&code);
        let pe = PeImage::parse(&data).expect("valid synthetic PE");
        let scan = find_ip_relative_references_resync_in_ranges(
            &pe,
            &BTreeSet::from([Rva(0x1100)]),
            &[(Rva(0x1000), Rva(0x1040))],
            0,
        )
        .expect("zero-budget scan must remain safe");

        assert!(scan.resource_limited);
        assert_eq!(scan.candidate_decodes, 0);
        assert!(scan.references.is_empty());
    }
}
