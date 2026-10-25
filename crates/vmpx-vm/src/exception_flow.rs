use std::collections::{BTreeMap, BTreeSet, VecDeque};

use iced_x86::{Mnemonic, OpKind, Register};
use serde::Serialize;
use vmpx_disasm::{CfgLimits, decode_one, recover_cfg};
use vmpx_pe::{PeError, PeImage, RuntimeFunction, Rva, UnwindInfo, Va};

use crate::{
    BitWidth, Expr, ExprKind, HandlerCodecRecoveryReport, HandlerMemoryRegion, RecoveryStatus,
    SymbolicVmExecutionReport, VmHandlerSemantic,
};

const WINDOWS_X64_CONTEXT_RSP_OFFSET: i64 = 0x98;

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct VmFaultUnwindCoverage {
    pub fault_rva: Rva,
    pub runtime_function: RuntimeFunction,
    pub unwind: UnwindInfo,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ExceptionStackSledReport {
    pub schema_version: u32,
    pub handler: Rva,
    pub status: RecoveryStatus,
    pub cfg_blocks: usize,
    pub context_rsp_field_offset: Option<i64>,
    pub seed_context_offset: Option<i64>,
    pub cursor_context_offset: Option<i64>,
    pub unwind_boundary_delta: Option<i64>,
    pub return_gadget: Option<Rva>,
    pub return_disposition: Option<u64>,
    pub confidence: u8,
    pub evidence: Vec<String>,
    pub diagnostics: Vec<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExceptionContinuationSource {
    NonZeroSeed,
    CursorMemory,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ExceptionContinuationResolution {
    pub schema_version: u32,
    pub handler: Rva,
    pub status: RecoveryStatus,
    pub seed: Option<Expr>,
    pub cursor: Option<Expr>,
    pub continuation: Option<Expr>,
    pub continuation_rva: Option<Rva>,
    pub source: Option<ExceptionContinuationSource>,
    pub diagnostics: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct VmMemoryExceptionCoverageReport {
    pub schema_version: u32,
    pub vm_handler: Rva,
    pub fault_sites: Vec<Rva>,
    pub coverage: Vec<VmFaultUnwindCoverage>,
    pub all_fault_sites_have_language_handler: bool,
    pub stack_sled_handlers: Vec<ExceptionStackSledReport>,
    pub continuation_resolutions: Vec<ExceptionContinuationResolution>,
    pub continuation_recovered: bool,
    pub diagnostics: Vec<String>,
}

#[derive(Clone, Copy, Debug)]
struct ContextLoad {
    rva: Rva,
    register: Register,
    offset: i64,
}

#[derive(Clone, Copy, Debug)]
struct RegisterStore {
    rva: Rva,
    address: Register,
    value: Register,
}

#[derive(Clone, Copy, Debug)]
struct GadgetDefinition {
    rva: Rva,
    register: Register,
    target: Rva,
}

fn canonical(register: Register) -> Register {
    register.full_register()
}

fn signed_displacement(value: u64) -> i64 {
    i64::from_ne_bytes(value.to_ne_bytes())
}

fn immediate(instruction: &iced_x86::Instruction, operand: u32) -> Option<u64> {
    match instruction.op_kind(operand) {
        OpKind::Immediate8 => Some(u64::from(instruction.immediate8())),
        OpKind::Immediate8to16 => Some(u64::from(instruction.immediate8to16().cast_unsigned())),
        OpKind::Immediate8to32 => Some(u64::from(instruction.immediate8to32().cast_unsigned())),
        OpKind::Immediate8to64 => Some(instruction.immediate8to64().cast_unsigned()),
        OpKind::Immediate16 => Some(u64::from(instruction.immediate16())),
        OpKind::Immediate32 => Some(u64::from(instruction.immediate32())),
        OpKind::Immediate32to64 => Some(instruction.immediate32to64().cast_unsigned()),
        OpKind::Immediate64 => Some(instruction.immediate64()),
        _ => None,
    }
}

fn memory_target_rva(pe: &PeImage<'_>, instruction: &iced_x86::Instruction) -> Option<Rva> {
    let address = if instruction.is_ip_rel_memory_operand() {
        instruction.ip_rel_memory_address()
    } else if instruction.memory_base() == Register::None
        && instruction.memory_index() == Register::None
    {
        instruction.memory_displacement64()
    } else {
        return None;
    };
    pe.optional.image_base.checked_rva_of(Va(address))
}

fn is_zero_test(instruction: &iced_x86::Instruction, register: Register) -> bool {
    let register = canonical(register);
    match instruction.mnemonic() {
        Mnemonic::Test | Mnemonic::Or => {
            instruction.op0_kind() == OpKind::Register
                && instruction.op1_kind() == OpKind::Register
                && canonical(instruction.op0_register()) == register
                && canonical(instruction.op1_register()) == register
        }
        Mnemonic::Cmp => {
            (instruction.op0_kind() == OpKind::Register
                && canonical(instruction.op0_register()) == register
                && immediate(instruction, 1) == Some(0))
                || (instruction.op1_kind() == OpKind::Register
                    && canonical(instruction.op1_register()) == register
                    && immediate(instruction, 0) == Some(0))
        }
        _ => false,
    }
}

fn constant_expr(expr: &Expr) -> Option<u64> {
    match expr.kind {
        ExprKind::Const { value } => Some(value),
        _ => None,
    }
}

/// Breadth-first block order rooted at the handler entry, plus per-instruction
/// execution indices. Protector handlers scatter fragments across the image,
/// so any "happens before" relation must come from CFG reachability rather than
/// ascending RVA.
struct BlockExecutionOrder {
    block_order: Vec<usize>,
    instruction_rank: BTreeMap<Rva, usize>,
}

impl BlockExecutionOrder {
    fn new(graph: &vmpx_disasm::ControlFlowGraph) -> Self {
        let index_of = graph
            .blocks
            .iter()
            .enumerate()
            .map(|(index, block)| (block.start, index))
            .collect::<BTreeMap<_, _>>();
        let mut block_order = Vec::new();
        let mut visited = BTreeSet::new();
        let mut queue = VecDeque::from([graph.entry]);
        visited.insert(graph.entry);
        while let Some(start) = queue.pop_front() {
            let Some(&index) = index_of.get(&start) else {
                continue;
            };
            block_order.push(index);
            for successor in &graph.blocks[index].successors {
                if visited.insert(*successor) {
                    queue.push_back(*successor);
                }
            }
        }
        // Keep unreachable blocks (defensive decode artifacts) deterministic and
        // last, so they can never satisfy an ordering proof by accident.
        for index in 0..graph.blocks.len() {
            if !block_order.contains(&index) {
                block_order.push(index);
            }
        }
        let mut instruction_rank = BTreeMap::new();
        let mut rank = 0;
        for index in &block_order {
            for instruction in &graph.blocks[*index].instructions {
                instruction_rank.insert(instruction.rva, rank);
                rank += 1;
            }
        }
        Self {
            block_order,
            instruction_rank,
        }
    }

    /// Rank of an instruction in execution order, or `None` when the address is
    /// not part of the recovered handler.
    fn rank(&self, rva: Rva) -> Option<usize> {
        self.instruction_rank.get(&rva).copied()
    }

    fn precedes(&self, earlier: Rva, later: Rva) -> bool {
        match (self.rank(earlier), self.rank(later)) {
            (Some(earlier), Some(later)) => earlier < later,
            _ => false,
        }
    }
}

/// Recognizes a language handler that builds a bounded virtual-unwind stack by
/// copying a recovered continuation seed and a validated `ret` gadget through a
/// cursor. Classification is based on register/memory effects and control-flow
/// relationships rather than raw bytes, handler addresses, or section names.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn analyze_exception_stack_sled(pe: &PeImage<'_>, handler: Rva) -> ExceptionStackSledReport {
    let mut report = ExceptionStackSledReport {
        schema_version: 1,
        handler,
        status: RecoveryStatus::Failed,
        cfg_blocks: 0,
        context_rsp_field_offset: None,
        seed_context_offset: None,
        cursor_context_offset: None,
        unwind_boundary_delta: None,
        return_gadget: None,
        return_disposition: None,
        confidence: 0,
        evidence: Vec::new(),
        diagnostics: Vec::new(),
    };
    let graph = match recover_cfg(
        pe,
        handler,
        CfgLimits {
            max_blocks: 256,
            max_instructions: 4_096,
            max_instructions_per_block: 1_024,
        },
    ) {
        Ok(graph) => graph,
        Err(error) => {
            report
                .diagnostics
                .push(format!("language-handler CFG recovery failed: {error}"));
            return report;
        }
    };
    report.cfg_blocks = graph.blocks.len();
    if graph.status != vmpx_disasm::CfgStatus::Complete {
        report.diagnostics.push(format!(
            "language-handler CFG is incomplete: {:?}",
            graph.status
        ));
    }
    // VMProtect scatters handler fragments across the image, so RVA order is
    // not execution order. Linearize blocks by breadth-first reachability from
    // the handler entry and use that order for every "happens before" test.
    let execution_order = BlockExecutionOrder::new(&graph);
    let mut instructions = Vec::new();
    for block_index in execution_order.block_order.iter().copied() {
        for summary in &graph.blocks[block_index].instructions {
            match decode_one(pe, summary.rva) {
                Ok(decoded) => instructions.push(decoded),
                Err(error) => report
                    .diagnostics
                    .push(format!("could not re-decode {}: {error}", summary.rva)),
            }
        }
    }

    let context_rsp_load = instructions.iter().find(|decoded| {
        let instruction = &decoded.instruction;
        instruction.mnemonic() == Mnemonic::Mov
            && instruction.op0_kind() == OpKind::Register
            && instruction.op1_kind() == OpKind::Memory
            && canonical(instruction.memory_base()) == Register::R8
            && instruction.memory_index() == Register::None
            && signed_displacement(instruction.memory_displacement64())
                == WINDOWS_X64_CONTEXT_RSP_OFFSET
    });
    let Some(context_rsp_load) = context_rsp_load else {
        report.diagnostics.push(
            "no Windows x64 CONTEXT.Rsp load from the language-handler context argument was proven"
                .to_owned(),
        );
        return report;
    };
    let native_rsp = canonical(context_rsp_load.instruction.op0_register());
    report.context_rsp_field_offset = Some(WINDOWS_X64_CONTEXT_RSP_OFFSET);
    report.confidence = report.confidence.saturating_add(15);
    report.evidence.push(format!(
        "{} loads CONTEXT.Rsp into {native_rsp:?}",
        context_rsp_load.rva
    ));

    let context_loads = instructions
        .iter()
        .filter_map(|decoded| {
            let instruction = &decoded.instruction;
            (instruction.mnemonic() == Mnemonic::Mov
                && instruction.op0_kind() == OpKind::Register
                && instruction.op1_kind() == OpKind::Memory
                && canonical(instruction.memory_base()) == native_rsp
                && instruction.memory_index() == Register::None)
                .then_some(ContextLoad {
                    rva: decoded.rva,
                    register: canonical(instruction.op0_register()),
                    offset: signed_displacement(instruction.memory_displacement64()),
                })
        })
        .collect::<Vec<_>>();
    let stores = instructions
        .iter()
        .filter_map(|decoded| {
            let instruction = &decoded.instruction;
            (instruction.mnemonic() == Mnemonic::Mov
                && instruction.op0_kind() == OpKind::Memory
                && instruction.op1_kind() == OpKind::Register
                && instruction.memory_index() == Register::None)
                .then_some(RegisterStore {
                    rva: decoded.rva,
                    address: canonical(instruction.memory_base()),
                    value: canonical(instruction.op1_register()),
                })
        })
        .collect::<Vec<_>>();
    let cursor_candidates = instructions
        .iter()
        .filter_map(|decoded| {
            let instruction = &decoded.instruction;
            (instruction.mnemonic() == Mnemonic::Sub
                && instruction.op0_kind() == OpKind::Register
                && immediate(instruction, 1) == Some(8))
            .then_some(canonical(instruction.op0_register()))
        })
        .collect::<BTreeSet<_>>();
    let cursor_load = context_loads.iter().find(|load| {
        cursor_candidates.contains(&load.register)
            && stores.iter().any(|store| store.address == load.register)
    });
    if let Some(cursor_load) = cursor_load {
        report.cursor_context_offset = Some(cursor_load.offset);
        report.confidence = report.confidence.saturating_add(15);
        report.evidence.push(format!(
            "{} loads context cursor field {:#X} into {:?}; the handler decrements it by one qword and stores through it",
            cursor_load.rva, cursor_load.offset, cursor_load.register
        ));
    } else {
        report
            .diagnostics
            .push("no context-derived qword stack cursor was proven".to_owned());
    }

    let gadget_definitions = instructions
        .iter()
        .filter_map(|decoded| {
            let instruction = &decoded.instruction;
            if instruction.mnemonic() != Mnemonic::Lea
                || instruction.op0_kind() != OpKind::Register
                || instruction.op1_kind() != OpKind::Memory
            {
                return None;
            }
            let target = memory_target_rva(pe, instruction)?;
            let target_instruction = decode_one(pe, target).ok()?;
            (target_instruction.instruction.mnemonic() == Mnemonic::Ret).then_some(
                GadgetDefinition {
                    rva: decoded.rva,
                    register: canonical(instruction.op0_register()),
                    target,
                },
            )
        })
        .collect::<Vec<_>>();
    let gadget = cursor_load.and_then(|cursor| {
        gadget_definitions.iter().find(|gadget| {
            stores.iter().any(|store| {
                execution_order.precedes(gadget.rva, store.rva)
                    && store.address == cursor.register
                    && store.value == gadget.register
            })
        })
    });
    if let Some(gadget) = gadget {
        report.return_gadget = Some(gadget.target);
        report.confidence = report.confidence.saturating_add(20);
        report.evidence.push(format!(
            "{} materializes executable RET gadget {}; a later store copies it through the cursor",
            gadget.rva, gadget.target
        ));
    } else {
        report.diagnostics.push(
            "no executable RET gadget stored through the context cursor was proven".to_owned(),
        );
    }

    let seed_load = cursor_load.and_then(|cursor| {
        context_loads.iter().find(|load| {
            load.register != cursor.register
                && instructions
                    .iter()
                    .any(|decoded| is_zero_test(&decoded.instruction, load.register))
                && stores.iter().any(|store| {
                    execution_order.precedes(load.rva, store.rva)
                        && gadget
                            .is_none_or(|gadget| execution_order.precedes(store.rva, gadget.rva))
                        && store.address == cursor.register
                        && store.value == load.register
                })
        })
    });
    if let Some(seed_load) = seed_load {
        report.seed_context_offset = Some(seed_load.offset);
        report.confidence = report.confidence.saturating_add(20);
        report.evidence.push(format!(
            "{} loads optional continuation seed field {:#X}; the handler zero-tests and conditionally stores it through the cursor",
            seed_load.rva, seed_load.offset
        ));
    } else {
        report
            .diagnostics
            .push("no optional context-derived continuation seed was proven".to_owned());
    }

    let boundary = instructions.iter().find(|decoded| {
        let instruction = &decoded.instruction;
        instruction.mnemonic() == Mnemonic::Add
            && instruction.op0_kind() == OpKind::Register
            && canonical(instruction.op0_register()) == native_rsp
            && immediate(instruction, 1).is_some_and(|value| value >= 8 && value % 8 == 0)
    });
    if let Some(boundary) = boundary {
        report.unwind_boundary_delta =
            immediate(&boundary.instruction, 1).and_then(|value| i64::try_from(value).ok());
        report.confidence = report.confidence.saturating_add(10);
        report.evidence.push(format!(
            "{} derives an aligned unwind boundary from the recovered native RSP",
            boundary.rva
        ));
    } else {
        report
            .diagnostics
            .push("no aligned native-RSP unwind boundary was proven".to_owned());
    }

    if let Some(cursor) = cursor_load {
        let cursor_compared_to_boundary = instructions.iter().any(|decoded| {
            let instruction = &decoded.instruction;
            instruction.mnemonic() == Mnemonic::Cmp
                && instruction.op0_kind() == OpKind::Register
                && instruction.op1_kind() == OpKind::Register
                && ((canonical(instruction.op0_register()) == cursor.register
                    && canonical(instruction.op1_register()) == native_rsp)
                    || (canonical(instruction.op1_register()) == cursor.register
                        && canonical(instruction.op0_register()) == native_rsp))
        });
        let has_loop = graph.blocks.iter().any(|block| {
            block
                .successors
                .iter()
                .any(|successor| successor <= &block.start)
        });
        if cursor_compared_to_boundary && has_loop {
            report.confidence = report.confidence.saturating_add(10);
            report.evidence.push(
                "the cursor is compared with the native-RSP boundary in a CFG containing a bounded back edge"
                    .to_owned(),
            );
        } else {
            report.diagnostics.push(
                "cursor/boundary comparison and bounded stack-fill loop were not both proven"
                    .to_owned(),
            );
        }
    }

    if graph
        .blocks
        .iter()
        .any(|block| block.terminal_flow == "return")
    {
        report.return_disposition = instructions.iter().rev().find_map(|decoded| {
            let instruction = &decoded.instruction;
            (instruction.mnemonic() == Mnemonic::Mov
                && instruction.op0_kind() == OpKind::Register
                && canonical(instruction.op0_register()) == Register::RAX)
                .then(|| immediate(instruction, 1))
                .flatten()
        });
        if report.return_disposition.is_some() {
            report.confidence = report.confidence.saturating_add(10);
            report
                .evidence
                .push("language handler returns a concrete disposition".to_owned());
        }
    }

    let complete = report.seed_context_offset.is_some()
        && report.cursor_context_offset.is_some()
        && report.unwind_boundary_delta.is_some()
        && report.return_gadget.is_some()
        && report.return_disposition.is_some()
        && report.confidence >= 80;
    report.status = if complete {
        RecoveryStatus::Recovered
    } else if report.confidence > 0 {
        RecoveryStatus::PartiallyRecovered
    } else {
        RecoveryStatus::Failed
    };
    report
}

fn resolve_stack_sled_continuation(
    pe: &PeImage<'_>,
    sled: &ExceptionStackSledReport,
    symbolic: Option<&SymbolicVmExecutionReport>,
) -> ExceptionContinuationResolution {
    let mut resolution = ExceptionContinuationResolution {
        schema_version: 1,
        handler: sled.handler,
        status: RecoveryStatus::Failed,
        seed: None,
        cursor: None,
        continuation: None,
        continuation_rva: None,
        source: None,
        diagnostics: Vec::new(),
    };
    if sled.status != RecoveryStatus::Recovered {
        resolution.diagnostics.push(
            "language handler was not fully classified, so no continuation was selected".to_owned(),
        );
        return resolution;
    }
    let Some(symbolic) = symbolic else {
        resolution
            .diagnostics
            .push("no exact pre-fault symbolic state was supplied".to_owned());
        return resolution;
    };
    resolution.seed = sled
        .seed_context_offset
        .and_then(|offset| symbolic.context_cells.get(&offset).cloned());
    resolution.cursor = sled
        .cursor_context_offset
        .and_then(|offset| symbolic.context_cells.get(&offset).cloned());
    let Some(seed) = resolution.seed.as_ref() else {
        resolution
            .diagnostics
            .push("pre-fault state does not contain the classified seed field".to_owned());
        return resolution;
    };
    let continuation = match constant_expr(seed) {
        Some(0) => {
            let Some(cursor) = resolution.cursor.clone() else {
                resolution.diagnostics.push(
                    "zero seed requires the cursor value, but its context field is absent"
                        .to_owned(),
                );
                return resolution;
            };
            match symbolic.read_tracked_memory(cursor, BitWidth::W64) {
                Ok(value) => {
                    resolution.source = Some(ExceptionContinuationSource::CursorMemory);
                    value
                }
                Err(error) => {
                    resolution.diagnostics.push(format!(
                        "zero-seed continuation is not present in exact tracked memory: {error}"
                    ));
                    return resolution;
                }
            }
        }
        Some(_) => {
            resolution.source = Some(ExceptionContinuationSource::NonZeroSeed);
            seed.clone()
        }
        None => {
            resolution.diagnostics.push(
                "continuation seed remains symbolic; zero and nonzero paths cannot be conflated"
                    .to_owned(),
            );
            return resolution;
        }
    };
    resolution.continuation_rva = constant_expr(&continuation)
        .and_then(|value| pe.optional.image_base.checked_rva_of(Va(value)));
    resolution.continuation = Some(continuation);
    // A continuation that is not image-backed is not a dispatch target. The
    // zero-seed path fills the sled with RET gadgets and unwinds through the
    // original stack, so a concrete non-code value there means the explicit
    // target is still unknown rather than recovered.
    if resolution.continuation_rva.is_none() {
        resolution.diagnostics.push(format!(
            "continuation value {} is not image-backed; no native dispatch target was proven",
            constant_expr(resolution.continuation.as_ref().expect("continuation is set"))
                .map_or_else(|| "symbolic".to_owned(), |value| format!("{value:#X}"))
        ));
        resolution.status = RecoveryStatus::PartiallyRecovered;
        return resolution;
    }
    resolution.status = RecoveryStatus::Recovered;
    resolution
}

/// Correlates the external read sites of a behavior-classified VM memory-load
/// handler with x64 unwind records and, when an exact pre-fault state is
/// available, analyzes the language handler's continuation semantics.
///
/// # Errors
///
/// Propagates malformed or unmapped exception/unwind metadata errors.
pub fn analyze_vm_memory_exception_coverage_with_state(
    pe: &PeImage<'_>,
    handler: &HandlerCodecRecoveryReport,
    symbolic: Option<&SymbolicVmExecutionReport>,
) -> Result<VmMemoryExceptionCoverageReport, PeError> {
    let mut diagnostics = Vec::new();
    if handler
        .handler_analysis
        .as_ref()
        .is_none_or(|analysis| analysis.semantic != VmHandlerSemantic::MemoryLoad)
    {
        diagnostics.push("handler is not proven to implement a VM memory load".to_owned());
    }
    let mut fault_sites = handler
        .effects
        .iter()
        .flat_map(|effects| &effects.memory)
        .filter(|effect| effect.region == HandlerMemoryRegion::Other && effect.reads)
        .map(|effect| effect.rva)
        .collect::<Vec<_>>();
    fault_sites.sort_unstable();
    fault_sites.dedup();
    if fault_sites.is_empty() {
        diagnostics.push("no proven external native read site was recovered".to_owned());
    }

    let mut coverage = Vec::new();
    let mut language_handlers = BTreeSet::new();
    let mut all_fault_sites_have_language_handler = !fault_sites.is_empty();
    for fault_rva in &fault_sites {
        let functions = pe.runtime_functions_containing(*fault_rva)?;
        if functions.is_empty() {
            all_fault_sites_have_language_handler = false;
            diagnostics.push(format!(
                "fault site {fault_rva} has no covering runtime-function record"
            ));
            continue;
        }
        let mut site_has_handler = false;
        for runtime_function in functions {
            let unwind = pe.parse_unwind_info(runtime_function.unwind_info)?;
            site_has_handler |= unwind.exception_handler.is_some();
            if let Some(exception_handler) = unwind.exception_handler {
                language_handlers.insert(exception_handler);
            }
            coverage.push(VmFaultUnwindCoverage {
                fault_rva: *fault_rva,
                runtime_function,
                unwind,
            });
        }
        if !site_has_handler {
            all_fault_sites_have_language_handler = false;
            diagnostics.push(format!(
                "fault site {fault_rva} is covered, but no record has a language handler"
            ));
        }
    }
    let stack_sled_handlers = language_handlers
        .into_iter()
        .map(|handler| analyze_exception_stack_sled(pe, handler))
        .collect::<Vec<_>>();
    let continuation_resolutions = stack_sled_handlers
        .iter()
        .map(|sled| resolve_stack_sled_continuation(pe, sled, symbolic))
        .collect::<Vec<_>>();
    let continuation_recovered = !continuation_resolutions.is_empty()
        && continuation_resolutions
            .iter()
            .all(|resolution| resolution.status == RecoveryStatus::Recovered);
    if all_fault_sites_have_language_handler && !continuation_recovered {
        diagnostics.push(
            "all proven external reads route through x64 language handlers, but at least one continuation remains unresolved"
                .to_owned(),
        );
    } else if continuation_recovered {
        diagnostics.push(
            "all proven external reads have behavior-classified language-handler continuations"
                .to_owned(),
        );
    }
    Ok(VmMemoryExceptionCoverageReport {
        schema_version: 2,
        vm_handler: handler.entry,
        fault_sites,
        coverage,
        all_fault_sites_have_language_handler,
        stack_sled_handlers,
        continuation_resolutions,
        continuation_recovered,
        diagnostics,
    })
}

/// Backward-compatible structural coverage entry point. Without pre-fault
/// symbolic state it intentionally cannot claim a recovered continuation.
///
/// # Errors
///
/// Propagates malformed or unmapped exception/unwind metadata errors.
pub fn analyze_vm_memory_exception_coverage(
    pe: &PeImage<'_>,
    handler: &HandlerCodecRecoveryReport,
) -> Result<VmMemoryExceptionCoverageReport, PeError> {
    analyze_vm_memory_exception_coverage_with_state(pe, handler, None)
}

#[cfg(test)]
mod tests {
    use iced_x86::code_asm::*;

    use super::*;

    const IMAGE_BASE: u64 = 0x1_4000_0000;
    const CODE_RVA: u32 = 0x1000;

    fn executable_pe64(code: &[u8]) -> Vec<u8> {
        let mut data = vec![0_u8; 0x1200];
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
        data[optional + 16..optional + 20].copy_from_slice(&CODE_RVA.to_le_bytes());
        data[optional + 20..optional + 24].copy_from_slice(&CODE_RVA.to_le_bytes());
        data[optional + 24..optional + 32].copy_from_slice(&IMAGE_BASE.to_le_bytes());
        data[optional + 32..optional + 36].copy_from_slice(&0x1000_u32.to_le_bytes());
        data[optional + 36..optional + 40].copy_from_slice(&0x200_u32.to_le_bytes());
        data[optional + 56..optional + 60].copy_from_slice(&0x3000_u32.to_le_bytes());
        data[optional + 60..optional + 64].copy_from_slice(&0x200_u32.to_le_bytes());
        data[optional + 68..optional + 70].copy_from_slice(&3_u16.to_le_bytes());
        data[optional + 108..optional + 112].copy_from_slice(&16_u32.to_le_bytes());
        let section = optional + 0xF0;
        data[section..section + 5].copy_from_slice(b".text");
        data[section + 8..section + 12].copy_from_slice(&0x1000_u32.to_le_bytes());
        data[section + 12..section + 16].copy_from_slice(&CODE_RVA.to_le_bytes());
        data[section + 16..section + 20].copy_from_slice(&0x1000_u32.to_le_bytes());
        data[section + 20..section + 24].copy_from_slice(&0x200_u32.to_le_bytes());
        data[section + 36..section + 40].copy_from_slice(&0x6000_0020_u32.to_le_bytes());
        data[0x200..0x200 + code.len()].copy_from_slice(code);
        data
    }

    fn stack_sled_code() -> Vec<u8> {
        let mut assembler = CodeAssembler::new(64).expect("x64 assembler");
        let mut skip_seed = assembler.create_label();
        let mut fill = assembler.create_label();
        let mut done = assembler.create_label();
        let mut gadget = assembler.create_label();
        assembler
            .mov(r10, qword_ptr(r8 + 0x98))
            .expect("CONTEXT.Rsp");
        assembler.mov(r11, qword_ptr(r10 + 0x40)).expect("seed");
        assembler.mov(rdx, qword_ptr(r10 + 0x58)).expect("cursor");
        assembler.test(r11, r11).expect("seed test");
        assembler.jz(skip_seed).expect("optional seed");
        assembler.sub(rdx, 8).expect("cursor decrement");
        assembler.mov(qword_ptr(rdx), r11).expect("seed store");
        assembler.set_label(&mut skip_seed).expect("skip seed");
        assembler.add(r10, 0xD0).expect("unwind boundary");
        assembler.lea(r13, qword_ptr(gadget)).expect("ret gadget");
        assembler.cmp(rdx, r10).expect("cursor bound");
        assembler.jbe(done).expect("already filled");
        assembler.set_label(&mut fill).expect("fill loop");
        assembler.sub(rdx, 8).expect("cursor decrement");
        assembler.mov(qword_ptr(rdx), r13).expect("gadget store");
        assembler.cmp(rdx, r10).expect("cursor bound");
        assembler.ja(fill).expect("fill loop back edge");
        assembler.set_label(&mut done).expect("done");
        assembler.mov(eax, 1).expect("continue search");
        assembler.ret().expect("handler return");
        assembler.set_label(&mut gadget).expect("gadget");
        assembler.ret().expect("ret gadget");
        assembler
            .assemble(IMAGE_BASE + u64::from(CODE_RVA))
            .expect("assemble handler")
    }

    #[test]
    fn classifies_mutated_exception_stack_sled_from_behavior() {
        let bytes = executable_pe64(&stack_sled_code());
        let pe = PeImage::parse(&bytes).expect("valid synthetic PE");
        let report = analyze_exception_stack_sled(&pe, Rva(CODE_RVA));
        assert_eq!(report.status, RecoveryStatus::Recovered, "{report:#?}");
        assert_eq!(report.context_rsp_field_offset, Some(0x98));
        assert_eq!(report.seed_context_offset, Some(0x40));
        assert_eq!(report.cursor_context_offset, Some(0x58));
        assert_eq!(report.unwind_boundary_delta, Some(0xD0));
        assert_eq!(report.return_disposition, Some(1));
        let gadget = report.return_gadget.expect("ret gadget");
        assert_eq!(
            decode_one(&pe, gadget)
                .expect("gadget decodes")
                .instruction
                .mnemonic(),
            Mnemonic::Ret
        );
    }
}
