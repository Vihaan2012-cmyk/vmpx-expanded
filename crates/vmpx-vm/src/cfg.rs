//! Control-flow walk over a virtualized function.
//!
//! Decodes the entry's linear VM stream, then follows each block's terminating
//! branch to its successors while carrying the symbolic VM state. A successor is
//! followed only when the branch target is proven: a constant after symbolic
//! execution, or the image of a target over a proven-exhaustive input domain.
//! Anything unresolved stops that path and is reported; no edge is invented.

use std::collections::BTreeMap;

use iced_x86::{FlowControl, InstructionInfoFactory, Mnemonic, OpAccess, OpKind, Register};
use serde::Serialize;
use thiserror::Error;
use vmpx_disasm::{DirectJumpTraceLimits, trace_direct_jump_fragments};
use vmpx_pe::{PeImage, Va};

use crate::{
    BinaryOp, BitWidth, DecodedVmInstruction, DispatchExtractionCandidate,
    EntryProfileRecoveryError, EntryProfileRecoveryLimits, Expr, ExprKind, InputDomainLimits,
    LinearDecodeCache, LinearDecodeStop, ModuloImageTargetLimits, NativeCallSelectorOutcome,
    NativeFrameSource, RecoveryStatus, RelativeDispatchState, SymbolicDomainLimits,
    SymbolicVmExecutionReport, SymbolicVmInitialState, VmDiscoveryLimits, VmEntryCandidate,
    VmExitReport, VmHandlerSemantic, analyze_vm_entry_at,
    dispatch_extract::canonical,
    entry_profile::{bootstrap_global_table, entry_stream_config, prepare_entry_stream},
    enumerate_symbolic_value, execute_symbolic_linear_block_with_initial_state,
    execute_symbolic_native_call_boundary_with_initial_state, infer_branch_input_domains,
    native_context::{
        Provenance, caller_push_source, defines_flags, initial_provenance, is_cursor_capture,
        observe_target_push, update_provenance,
    },
    native_vm_chain::compose_native_frame,
    recover_and_decode_linear_vm_stream_cached, recover_modulo_image_targets,
    recover_native_exit_state,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VmCfgLimits {
    pub entry: EntryProfileRecoveryLimits,
    pub input_domain: InputDomainLimits,
    pub domain: SymbolicDomainLimits,
    /// Distinct decoded basic blocks.
    pub max_blocks: usize,
    /// Symbolic block executions across all paths.
    pub max_visits: usize,
    /// Executions of one block under distinct symbolic states.
    pub max_visits_per_block: usize,
    pub max_total_instructions: usize,
    /// Distinct targets one branch may resolve to.
    pub max_targets: usize,
    /// VM-to-VM re-entries followed across the whole walk.
    pub max_nested_entries: usize,
}

impl Default for VmCfgLimits {
    fn default() -> Self {
        Self {
            entry: EntryProfileRecoveryLimits::default(),
            input_domain: InputDomainLimits::default(),
            domain: SymbolicDomainLimits::default(),
            max_blocks: 512,
            max_visits: 4_096,
            max_visits_per_block: 16,
            max_total_instructions: 200_000,
            max_targets: 16,
            max_nested_entries: 8,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BlockExit {
    VmExit,
    Branch,
    NativeCall,
    Stalled,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct VmBasicBlock {
    /// Decoder state after the first opcode dispatch; `start.vip` is the block key.
    pub start: RelativeDispatchState,
    /// Address of the first opcode when known (entry or a branch target).
    pub address: Option<Va>,
    pub instructions: Vec<DecodedVmInstruction>,
    pub exit: BlockExit,
    /// Handler of the control-flow boundary that ends the block.
    pub boundary_handler: Option<Va>,
    /// Proven call frame of the selector this `NativeCall` block decoded.
    pub native_call: Option<NativeCallSelectorOutcome>,
    /// Recovered exit layout of this `VmExit` block's handler.
    pub vm_exit: Option<VmExitReport>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CfgEdgeKind {
    /// Branch target proven constant.
    Branch,
    /// One of several targets proven over an exhaustive input domain.
    GuardedBranch,
    /// Control resumes after a proven native call.
    NativeCallReturn,
    /// A proven VM exit returns into another VM entry of the same VM.
    NestedEntry,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct CfgEdge {
    pub from: Va,
    pub to: Va,
    pub kind: CfgEdgeKind,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct CfgStall {
    /// Key of the block whose path stopped.
    pub block: Va,
    pub vip: Va,
    pub reason: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CfgLimitHit {
    Blocks,
    Visits,
    BlockVisits,
    TotalInstructions,
}

#[derive(Clone, Debug, Serialize)]
pub struct VmCfg {
    pub entry: Va,
    pub blocks: BTreeMap<Va, VmBasicBlock>,
    pub edges: Vec<CfgEdge>,
    pub stalls: Vec<CfgStall>,
    pub limit_hit: Option<CfgLimitHit>,
    pub visits: usize,
    /// Initial VM-stack cells with a proven value, and why the rest are unknown.
    pub frame_cells_known: usize,
    pub frame_note: Option<String>,
}

impl VmCfg {
    /// Proven call frames, keyed by the key of the block that ends in the call.
    #[must_use]
    pub fn native_calls(&self) -> BTreeMap<Va, NativeCallSelectorOutcome> {
        self.blocks
            .iter()
            .filter_map(|(key, block)| Some((*key, block.native_call.clone()?)))
            .collect()
    }

    /// Recovered exit layouts, keyed by the exit handler's address.
    #[must_use]
    pub fn exit_layouts(&self) -> BTreeMap<Va, VmExitReport> {
        self.blocks
            .values()
            .filter_map(|block| Some((block.boundary_handler?, block.vm_exit.clone()?)))
            .collect()
    }

    fn record_native_call(&mut self, key: Va, outcome: &NativeCallSelectorOutcome) {
        if let Some(block) = self.blocks.get_mut(&key)
            && block.exit == BlockExit::NativeCall
        {
            block.native_call = Some(outcome.clone());
        }
    }

    #[must_use]
    pub fn total_instructions(&self) -> usize {
        self.blocks
            .values()
            .map(|block| block.instructions.len())
            .sum()
    }

    /// Every explored path ended at a proven VM exit or re-entered a decoded
    /// block, with no stall and no resource limit hit.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.stalls.is_empty() && self.limit_hit.is_none()
    }

    #[must_use]
    pub fn all_paths_reach_vm_exit(&self) -> bool {
        self.is_complete()
            && self
                .blocks
                .values()
                .any(|block| block.exit == BlockExit::VmExit)
    }
}

#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum CfgError {
    #[error("entry setup failed: {0}")]
    Entry(String),
}

impl From<EntryProfileRecoveryError> for CfgError {
    fn from(error: EntryProfileRecoveryError) -> Self {
        Self::Entry(error.to_string())
    }
}

/// Proven contents of the VM stack at VM start, top first, recovered by
/// stepping the native entry up to the VSP capture. Fails closed on anything it
/// cannot model (nested calls, unknown stack writes, stack-pointer arithmetic).
fn recover_entry_frame(
    pe: &PeImage<'_>,
    entry: &VmEntryCandidate,
    vsp: Register,
    limits: DirectJumpTraceLimits,
) -> Result<Vec<NativeFrameSource>, String> {
    let caller = trace_direct_jump_fragments(pe, entry.rva, limits).map_err(|e| e.to_string())?;
    let caller_instructions = caller.instructions().collect::<Vec<_>>();
    let (call_index, call) = caller_instructions
        .iter()
        .enumerate()
        .find(|(_, decoded)| decoded.instruction.flow_control() == FlowControl::Call)
        .ok_or("entry trace has no direct call")?;
    let call_target = call
        .direct_target_rva(pe)
        .ok_or("entry call has no direct target")?;
    // Bottom-first model of the native stack.
    let mut stack: Vec<NativeFrameSource> = Vec::new();
    for decoded in caller_instructions.iter().copied().take(call_index) {
        let instruction = &decoded.instruction;
        if instruction.mnemonic() == Mnemonic::Push {
            let ordinal = u32::try_from(stack.len()).unwrap_or(u32::MAX);
            stack.push(caller_push_source(decoded, ordinal));
        } else if instruction.stack_pointer_increment() != 0 {
            return Err(format!(
                "unproven pre-call stack mutation at {}",
                decoded.rva
            ));
        }
    }
    stack.push(NativeFrameSource::ReturnAddress {
        call_site: call.rva,
        continuation: call.next_rva().ok_or("call continuation overflows")?,
    });

    let target = trace_direct_jump_fragments(pe, call_target, limits).map_err(|e| e.to_string())?;
    let mut provenance = initial_provenance();
    let mut original_flags = true;
    let mut factory = InstructionInfoFactory::new();
    for decoded in target.instructions() {
        let instruction = &decoded.instruction;
        if is_cursor_capture(instruction).is_some_and(|cursor| cursor == canonical(vsp)) {
            stack.reverse();
            return Ok(stack);
        }
        match instruction.flow_control() {
            FlowControl::Next | FlowControl::UnconditionalBranch => {}
            _ => return Err(format!("unsupported control flow at {}", decoded.rva)),
        }
        if matches!(instruction.mnemonic(), Mnemonic::Push | Mnemonic::Pushfq) {
            let observed = observe_target_push(decoded, &provenance, original_flags)
                .ok_or_else(|| format!("unsupported push at {}", decoded.rva))?;
            if instruction.stack_pointer_increment() != -8 {
                return Err(format!("non-qword push at {}", decoded.rva));
            }
            stack.push(observed.source);
        } else {
            if instruction.stack_pointer_increment() != 0 {
                return Err(format!("unproven stack mutation at {}", decoded.rva));
            }
            let writes: Vec<_> = factory
                .info(instruction)
                .used_memory()
                .iter()
                .filter(|memory| {
                    matches!(
                        memory.access(),
                        OpAccess::Write
                            | OpAccess::CondWrite
                            | OpAccess::ReadWrite
                            | OpAccess::ReadCondWrite
                    )
                })
                .map(|memory| (memory.base(), memory.index(), memory.displacement()))
                .collect();
            if let Some(&(base, index, displacement)) = writes.first() {
                let stored = (writes.len() == 1
                    && base == Register::RSP
                    && index == Register::None
                    && instruction.mnemonic() == Mnemonic::Mov
                    && instruction.op0_kind() == OpKind::Memory
                    && displacement % 8 == 0)
                    .then(|| stored_value(instruction, &provenance))
                    .flatten()
                    .ok_or_else(|| format!("unproven memory write at {}", decoded.rva))?;
                let from_top = usize::try_from(displacement / 8)
                    .map_err(|_| format!("stack write offset at {}", decoded.rva))?;
                let position = stack
                    .len()
                    .checked_sub(1)
                    .and_then(|top| top.checked_sub(from_top))
                    .ok_or_else(|| format!("stack write outside the frame at {}", decoded.rva))?;
                stack[position] = stored;
            }
        }
        update_provenance(&mut factory, instruction, &mut provenance);
        if defines_flags(instruction) {
            original_flags = false;
        }
    }
    Err("no VSP capture reached".to_owned())
}

fn stored_value(
    instruction: &iced_x86::Instruction,
    provenance: &BTreeMap<Register, Provenance>,
) -> Option<NativeFrameSource> {
    match instruction.op1_kind() {
        OpKind::Register if instruction.op1_register().size() == 8 => {
            match provenance.get(&canonical(instruction.op1_register()))? {
                Provenance::Native(register) => Some(NativeFrameSource::Register {
                    register: *register,
                }),
                Provenance::Constant(value) => Some(NativeFrameSource::Constant { value: *value }),
                Provenance::Unknown => Some(NativeFrameSource::Unknown {
                    description: "stored register of unproven value".to_owned(),
                }),
            }
        }
        _ if instruction.memory_size() == iced_x86::MemorySize::UInt64 => {
            crate::dispatch_extract::signed_immediate(instruction, 1).map(|value| {
                NativeFrameSource::Constant {
                    value: value.cast_unsigned(),
                }
            })
        }
        _ => None,
    }
}

fn frame_initial_state(cells: &[NativeFrameSource]) -> (SymbolicVmInitialState, usize) {
    let mut initial = SymbolicVmInitialState::default();
    let base = Expr::register(BitWidth::W64, "vsp_initial");
    let mut known = 0;
    for (index, cell) in cells.iter().enumerate() {
        let Some(value) = cell.symbolic_value() else {
            continue;
        };
        let Ok(address) = Expr::binary(
            BinaryOp::Add,
            base.clone(),
            Expr::constant(BitWidth::W64, (index as u64) * 8),
        ) else {
            continue;
        };
        initial.memory_values.push((address, value));
        known += 1;
    }
    (initial, known)
}

struct DecodedBlock {
    instructions: Vec<DecodedVmInstruction>,
    final_state: RelativeDispatchState,
    stop: LinearDecodeStop,
    boundary: Option<VmHandlerSemantic>,
    error: Option<String>,
}

fn bytecode_section(pe: &PeImage<'_>, vip: Va) -> Option<usize> {
    let rva = pe.optional.image_base.checked_rva_of(vip)?;
    pe.sections
        .iter()
        .find(|section| section.contains_mapped_rva(rva))
        .map(|section| section.index)
}

const NESTED_ENTRY_CONFIDENCE: u8 = 60;

struct Item {
    start: RelativeDispatchState,
    address: Option<Va>,
    symbolic: SymbolicVmInitialState,
}

struct Walker<'a, 'p> {
    pe: &'a PeImage<'p>,
    limits: VmCfgLimits,
    config: crate::HandlerCodecRecoveryConfig,
    global_table: Option<crate::GlobalTableBase>,
    profile: crate::VmProfile,
    nested_followed: usize,
    direction: crate::BytecodeDirection,
    section: Option<usize>,
    cache: LinearDecodeCache,
    decoded: BTreeMap<Va, DecodedBlock>,
    explored: BTreeMap<Va, Vec<SymbolicVmInitialState>>,
    cfg: VmCfg,
    total_instructions: usize,
}

/// An exit layout is carried only when its recovery was proven.
fn proven_exit_layout(report: Option<&VmExitReport>) -> Option<VmExitReport> {
    report
        .filter(|report| report.status == RecoveryStatus::Recovered)
        .cloned()
}

/// Targets a branch can take: the constant it simplifies to, or its image over
/// a proven-exhaustive input domain (`true` when guarded by inputs).
fn resolve_targets(target: &Expr, limits: VmCfgLimits) -> Result<(Vec<u64>, bool), String> {
    let simplified = target.simplify();
    if let ExprKind::Const { value } = simplified.kind {
        return Ok((vec![value], false));
    }
    let inferred = infer_branch_input_domains(&simplified, limits.input_domain)
        .map_err(|refusal| format!("target depends on unproven input: {refusal}"))?;
    let domain = SymbolicDomainLimits {
        max_values_per_variable: limits
            .domain
            .max_values_per_variable
            .max(limits.input_domain.max_values_per_variable),
        ..limits.domain
    };
    let report = enumerate_symbolic_value(&simplified, &inferred.domains, domain)
        .map_err(|error| format!("target enumeration failed: {error}"))?;
    if !report.complete || report.unique_values.is_empty() {
        return Err("target domain is incomplete".to_owned());
    }
    if report.unique_values.len() > limits.max_targets {
        return Err(format!(
            "branch has {} targets, exceeding limit {}",
            report.unique_values.len(),
            limits.max_targets
        ));
    }
    Ok((report.unique_values, true))
}

impl Walker<'_, '_> {
    fn stall(&mut self, block: Va, vip: Va, reason: impl Into<String>) {
        self.cfg.stalls.push(CfgStall {
            block,
            vip,
            reason: reason.into(),
        });
    }

    fn decode_block(&mut self, item: &Item) -> bool {
        let key = item.start.vip;
        if self.decoded.contains_key(&key) {
            return true;
        }
        if self.cfg.blocks.len() >= self.limits.max_blocks {
            self.cfg.limit_hit.get_or_insert(CfgLimitHit::Blocks);
            return false;
        }
        let remaining = self
            .limits
            .max_total_instructions
            .saturating_sub(self.total_instructions);
        if remaining == 0 {
            self.cfg
                .limit_hit
                .get_or_insert(CfgLimitHit::TotalInstructions);
            return false;
        }
        let block = match recover_and_decode_linear_vm_stream_cached(
            self.pe,
            item.start,
            &self.config,
            self.limits.entry.instruction_limit.min(remaining),
            self.global_table,
            &mut self.cache,
        ) {
            Ok(report) => {
                let decode = report.decode;
                let boundary = (decode.stop_reason == LinearDecodeStop::ControlFlowBoundary)
                    .then(|| self.cache.recoveries.get(&decode.final_state.handler))
                    .flatten()
                    .and_then(|recovery| recovery.handler_analysis.as_ref())
                    .map(|analysis| analysis.semantic);
                DecodedBlock {
                    instructions: decode.instructions,
                    final_state: decode.final_state,
                    stop: decode.stop_reason,
                    boundary,
                    error: None,
                }
            }
            Err(error) => {
                let partial = error.partial_report().decode.clone();
                DecodedBlock {
                    instructions: partial.instructions,
                    final_state: partial.final_state,
                    stop: partial.stop_reason,
                    boundary: None,
                    error: Some(error.to_string()),
                }
            }
        };
        self.total_instructions += block.instructions.len();
        let boundary_handler = block.boundary.map(|_| block.final_state.handler);
        let exit = match block.boundary {
            Some(VmHandlerSemantic::VmExit) => BlockExit::VmExit,
            Some(VmHandlerSemantic::UnconditionalBranch) => BlockExit::Branch,
            Some(VmHandlerSemantic::NativeCall) => BlockExit::NativeCall,
            _ => BlockExit::Stalled,
        };
        let vm_exit = (exit == BlockExit::VmExit)
            .then(|| {
                self.cache
                    .recoveries
                    .get(&block.final_state.handler)
                    .and_then(|recovery| proven_exit_layout(recovery.vm_exit.as_ref()))
            })
            .flatten();
        self.cfg.blocks.insert(
            key,
            VmBasicBlock {
                start: item.start,
                address: item.address,
                instructions: block.instructions.clone(),
                exit,
                boundary_handler,
                native_call: None,
                vm_exit,
            },
        );
        self.decoded.insert(key, block);
        true
    }

    fn visit(&mut self, item: &Item, pending: &mut Vec<Item>) {
        let key = item.start.vip;
        if self.cfg.visits >= self.limits.max_visits {
            self.cfg.limit_hit.get_or_insert(CfgLimitHit::Visits);
            return;
        }
        let states = self.explored.entry(key).or_default();
        if states.contains(&item.symbolic) {
            return;
        }
        if states.len() >= self.limits.max_visits_per_block {
            self.cfg.limit_hit.get_or_insert(CfgLimitHit::BlockVisits);
            self.stall(key, key, "symbolic state keeps changing at this block");
            return;
        }
        states.push(item.symbolic.clone());
        self.cfg.visits += 1;
        if !self.decode_block(item) {
            return;
        }
        let (final_state, stop, boundary, error, instructions) = {
            let block = &self.decoded[&key];
            (
                block.final_state,
                block.stop,
                block.boundary,
                block.error.clone(),
                block.instructions.clone(),
            )
        };
        if let Some(error) = error {
            self.stall(key, final_state.vip, format!("decode failed: {error}"));
            return;
        }
        if stop != LinearDecodeStop::ControlFlowBoundary {
            let detail = match stop {
                LinearDecodeStop::MissingHandlerSpec => {
                    let recovery = self.cache.recoveries.get(&final_state.handler);
                    format!(
                        "undecodable opcode: handler {} has no recovered specification                          (status {:?}, semantic {:?}; {})",
                        final_state.handler,
                        recovery.map(|r| r.status),
                        recovery
                            .and_then(|r| r.handler_analysis.as_ref())
                            .map(|a| a.semantic),
                        recovery.map_or(String::new(), |r| r
                            .diagnostics
                            .iter()
                            .take(4)
                            .cloned()
                            .collect::<Vec<_>>()
                            .join("; ")),
                    )
                }
                other => format!("linear decode stopped with {other:?}"),
            };
            self.stall(key, final_state.vip, detail);
            return;
        }
        let symbolic_boundary =
            boundary.filter(|semantic| *semantic != VmHandlerSemantic::NativeCall);
        let symbolic = match execute_symbolic_linear_block_with_initial_state(
            &instructions,
            symbolic_boundary,
            self.limits.entry.symbolic,
            &item.symbolic,
            Some(&crate::immutable_image::ReadOnlyImage(self.pe)),
        ) {
            Ok(report) => report,
            Err(error) => {
                self.stall(
                    key,
                    error.vip,
                    format!("symbolic execution failed: {error}"),
                );
                return;
            }
        };
        match boundary {
            Some(VmHandlerSemantic::VmExit) => {
                self.follow_nested_entry(key, final_state.vip, &symbolic, pending);
            }
            Some(VmHandlerSemantic::UnconditionalBranch) => {
                self.follow_branch(key, final_state, &symbolic, pending);
            }
            Some(VmHandlerSemantic::NativeCall) => {
                self.follow_native_call(key, final_state, &symbolic, pending);
            }
            other => self.stall(
                key,
                final_state.vip,
                format!("unsupported control-flow boundary {other:?}"),
            ),
        }
    }

    fn branch_dispatch(&self, handler: Va) -> Result<(DispatchExtractionCandidate, Va), String> {
        let recovery = self
            .cache
            .recoveries
            .get(&handler)
            .ok_or("branch handler has no recovery")?;
        let effects = recovery
            .effects
            .as_ref()
            .filter(|effects| {
                effects.loads_vip_from_vsp
                    && effects.resets_key_from_vip
                    && effects.reinitializes_handler_base
            })
            .ok_or("branch handler does not load VIP from the VM stack and re-anchor dispatch")?;
        let _ = effects;
        let [dispatch] = recovery.dispatch_analysis.candidates.as_slice() else {
            return Err(format!(
                "branch handler has {} dispatch candidates, exactly one required",
                recovery.dispatch_analysis.candidates.len()
            ));
        };
        // A table-indexed dispatch with a static base depends only on the VIP,
        // which this handler loads from the VM stack; other codecs must be
        // proven to run after the VIP rebase.
        let static_table = dispatch
            .codec
            .table
            .is_some_and(|table| table.table_base.is_some());
        if !dispatch.vip_rebased_before_source && !static_table {
            return Err("branch dispatch does not read from the rebased VIP".to_owned());
        }
        let anchor = dispatch
            .handler_rebase
            .ok_or("branch dispatch has no handler anchor")?;
        Ok((dispatch.clone(), anchor))
    }

    fn follow_branch(
        &mut self,
        key: Va,
        final_state: RelativeDispatchState,
        symbolic: &SymbolicVmExecutionReport,
        pending: &mut Vec<Item>,
    ) {
        let vip = final_state.vip;
        let Some(target) = symbolic.branch_target.as_ref() else {
            self.stall(key, vip, "branch has no symbolic target");
            return;
        };
        if symbolic.memory_tainted {
            self.stall(
                key,
                vip,
                "branch target follows an imprecise external-memory store",
            );
            return;
        }
        let (dispatch, anchor) = match self.branch_dispatch(final_state.handler) {
            Ok(found) => found,
            Err(reason) => {
                self.stall(key, vip, reason);
                return;
            }
        };
        let (targets, guarded) = match resolve_targets(target, self.limits) {
            Ok(resolved) => resolved,
            Err(reason) => {
                self.stall(key, vip, format!("unresolved branch target: {reason}"));
                return;
            }
        };
        for value in targets {
            let target_va = Va(value);
            if self.section.is_none() || bytecode_section(self.pe, target_va) != self.section {
                self.stall(
                    key,
                    vip,
                    format!("branch target {target_va} is outside the bytecode section"),
                );
                continue;
            }
            let first = match dispatch.codec.decode_direction(
                self.pe,
                RelativeDispatchState {
                    vip: target_va,
                    rolling_key: value,
                    handler: anchor,
                },
                self.direction,
            ) {
                Ok(first) => first,
                Err(error) => {
                    self.stall(
                        key,
                        vip,
                        format!("target {target_va} dispatch failed: {error}"),
                    );
                    continue;
                }
            };
            self.cfg.edges.push(CfgEdge {
                from: key,
                to: first.state_after.vip,
                kind: if guarded {
                    CfgEdgeKind::GuardedBranch
                } else {
                    CfgEdgeKind::Branch
                },
            });
            pending.push(Item {
                start: first.state_after,
                address: Some(target_va),
                symbolic: symbolic.continuation_state(),
            });
        }
    }

    /// Follows a proven VM exit whose return target is another entry stub of
    /// the same VM. A target that is not provably such an entry ends the path
    /// as an ordinary native return; a proven entry that cannot be composed
    /// stalls.
    fn follow_nested_entry(
        &mut self,
        key: Va,
        vip: Va,
        symbolic: &SymbolicVmExecutionReport,
        pending: &mut Vec<Item>,
    ) {
        let Some(layout) = self
            .cfg
            .blocks
            .get(&key)
            .and_then(|block| block.vm_exit.clone())
        else {
            return;
        };
        let Ok(exit) = recover_native_exit_state(symbolic, &layout) else {
            return;
        };
        let simplified = exit.return_address.simplify();
        let target = if let ExprKind::Const { value } = simplified.kind {
            self.pe.optional.image_base.checked_rva_of(Va(value))
        } else {
            recover_modulo_image_targets(
                self.pe,
                &exit.return_address,
                ModuloImageTargetLimits::default(),
            )
            .ok()
            .filter(|report| report.complete && report.targets.len() == 1)
            .and_then(|report| report.targets[0].rva)
        };
        let Some(candidate) = target
            .and_then(|rva| analyze_vm_entry_at(self.pe, rva, VmDiscoveryLimits::default()))
            .filter(|candidate| candidate.confidence >= NESTED_ENTRY_CONFIDENCE)
        else {
            return;
        };
        let at = candidate.rva;
        if self.nested_followed >= self.limits.max_nested_entries {
            self.stall(
                key,
                vip,
                format!("nested VM entry {at} exceeds the nesting limit"),
            );
            return;
        }
        let setup = match prepare_entry_stream(self.pe, &candidate, self.limits.entry) {
            Ok(setup) => setup,
            Err(error) => {
                self.stall(
                    key,
                    vip,
                    format!("nested VM entry {at} setup failed: {error}"),
                );
                return;
            }
        };
        if setup.profile != self.profile
            || bootstrap_global_table(&setup.bootstrap) != self.global_table
            || bytecode_section(self.pe, setup.bootstrap.initial_state.vip) != self.section
        {
            self.stall(
                key,
                vip,
                format!("nested VM entry {at} is not proven to share this VM's profile"),
            );
            return;
        }
        let cells = match recover_entry_frame(
            self.pe,
            &candidate,
            setup.profile.effect_roles.vsp,
            self.limits.entry.trace,
        ) {
            Ok(cells) => cells,
            Err(error) => {
                self.stall(
                    key,
                    vip,
                    format!("nested VM entry {at} frame unproven: {error}"),
                );
                return;
            }
        };
        let continuation = cells.iter().find_map(|cell| match cell {
            NativeFrameSource::ReturnAddress { continuation, .. } => Some(*continuation),
            _ => None,
        });
        let (frame, _) = frame_initial_state(&cells);
        let initial = match compose_native_frame(
            self.pe,
            symbolic,
            &exit,
            frame,
            cells.len(),
            continuation,
        ) {
            Ok(initial) => initial,
            Err(error) => {
                self.stall(
                    key,
                    vip,
                    format!("nested VM entry {at} composition failed: {error}"),
                );
                return;
            }
        };
        self.nested_followed += 1;
        self.cfg.edges.push(CfgEdge {
            from: key,
            to: setup.first_state.vip,
            kind: CfgEdgeKind::NestedEntry,
        });
        pending.push(Item {
            start: setup.first_state,
            address: Some(setup.bootstrap.initial_state.vip),
            symbolic: initial,
        });
    }

    fn follow_native_call(
        &mut self,
        key: Va,
        final_state: RelativeDispatchState,
        symbolic: &SymbolicVmExecutionReport,
        pending: &mut Vec<Item>,
    ) {
        let vip = final_state.vip;
        let Some(spec) = self
            .cache
            .recoveries
            .get(&final_state.handler)
            .and_then(|recovery| recovery.boundary_spec.clone())
        else {
            self.stall(
                key,
                vip,
                "native-call boundary has no proven decode specification",
            );
            return;
        };
        let decoded = match spec.decode_direction(self.pe, final_state, self.direction) {
            Ok(decoded) => decoded,
            Err(error) => {
                self.stall(
                    key,
                    vip,
                    format!("native-call boundary decode failed: {error}"),
                );
                return;
            }
        };
        let call = match execute_symbolic_native_call_boundary_with_initial_state(
            decoded.state_before.vip,
            &decoded.outcome,
            self.limits.entry.symbolic,
            &symbolic.continuation_state(),
            Some(&crate::immutable_image::ReadOnlyImage(self.pe)),
        ) {
            Ok(call) => call,
            Err(error) => {
                self.stall(key, vip, format!("native-call effect failed: {error}"));
                return;
            }
        };
        self.cfg.record_native_call(key, &decoded.outcome);
        self.cfg.edges.push(CfgEdge {
            from: key,
            to: decoded.state_after.vip,
            kind: CfgEdgeKind::NativeCallReturn,
        });
        pending.push(Item {
            start: decoded.state_after,
            address: None,
            symbolic: call.continuation_state(),
        });
    }
}

/// Decodes a virtualized function's blocks by following proven VM branches
/// from `entry`, using only native bootstrap data flow and handler behavior.
///
/// The initial VM stack is the frame recovered by stepping the native entry
/// (see [`recover_entry_frame`]); cells that cannot be proven stay unknown and
/// any branch depending on them stalls.
///
/// # Errors
///
/// Fails if the entry bootstrap cannot be recovered. Everything after that is
/// reported in the returned [`VmCfg`] as stalls or a hit limit.
pub fn recover_vm_cfg(
    pe: &PeImage<'_>,
    entry: &VmEntryCandidate,
    limits: VmCfgLimits,
) -> Result<VmCfg, CfgError> {
    let setup = prepare_entry_stream(pe, entry, limits.entry)?;
    let (initial, frame_cells_known, frame_note) = match recover_entry_frame(
        pe,
        entry,
        setup.profile.effect_roles.vsp,
        limits.entry.trace,
    ) {
        Ok(cells) => {
            let (initial, known) = frame_initial_state(&cells);
            (initial, known, None)
        }
        Err(note) => (SymbolicVmInitialState::default(), 0, Some(note)),
    };
    let entry_address = setup.bootstrap.initial_state.vip;
    let mut walker = Walker {
        pe,
        limits,
        config: entry_stream_config(&setup.profile, limits.entry),
        global_table: bootstrap_global_table(&setup.bootstrap),
        profile: setup.profile,
        nested_followed: 0,
        direction: setup.profile.bytecode_direction,
        section: bytecode_section(pe, entry_address),
        cache: LinearDecodeCache::default(),
        decoded: BTreeMap::new(),
        explored: BTreeMap::new(),
        cfg: VmCfg {
            entry: setup.first_state.vip,
            blocks: BTreeMap::new(),
            edges: Vec::new(),
            stalls: Vec::new(),
            limit_hit: None,
            visits: 0,
            frame_cells_known,
            frame_note,
        },
        total_instructions: 0,
    };
    let mut pending = vec![Item {
        start: setup.first_state,
        address: Some(entry_address),
        symbolic: initial,
    }];
    while let Some(item) = pending.pop() {
        walker.visit(&item, &mut pending);
        if walker
            .cfg
            .limit_hit
            .is_some_and(|hit| hit != CfgLimitHit::BlockVisits)
        {
            break;
        }
    }
    Ok(walker.cfg)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zf_select(register: &str, compare: u64, if_zero: u64, otherwise: u64) -> Expr {
        let rdx = Expr::register(BitWidth::W64, register);
        let low = Expr::extract(BitWidth::W32, 0, rdx).expect("extract");
        let flags = Expr::intrinsic(
            BitWidth::W64,
            "vm_flags_sub_w32",
            vec![low, Expr::constant(BitWidth::W32, compare)],
        );
        let shifted =
            Expr::binary(BinaryOp::Shr, flags, Expr::constant(BitWidth::W64, 6)).expect("shift");
        let zero =
            Expr::binary(BinaryOp::And, shifted, Expr::constant(BitWidth::W64, 1)).expect("mask");
        let delta = Expr::binary(
            BinaryOp::Mul,
            zero,
            Expr::constant(BitWidth::W64, if_zero.wrapping_sub(otherwise)),
        )
        .expect("scale");
        Expr::binary(
            BinaryOp::Add,
            delta,
            Expr::constant(BitWidth::W64, otherwise),
        )
        .expect("sum")
    }

    #[test]
    fn subtract_flag_guard_resolves_to_exactly_both_targets() {
        let target = zf_select("native_rdx_initial", 1, 0x1000, 0x2000);
        let (values, guarded) =
            resolve_targets(&target, VmCfgLimits::default()).expect("proven domain");
        assert!(guarded);
        assert_eq!(values, vec![0x1000, 0x2000]);
    }

    #[test]
    fn constant_target_is_not_guarded_and_raw_register_target_is_refused() {
        let constant = Expr::constant(BitWidth::W64, 0x4000);
        assert_eq!(
            resolve_targets(&constant, VmCfgLimits::default()),
            Ok((vec![0x4000], false))
        );
        let raw = Expr::register(BitWidth::W64, "native_rdx_initial");
        assert!(resolve_targets(&raw, VmCfgLimits::default()).is_err());
    }

    fn block(exit: BlockExit, handler: Option<Va>) -> VmBasicBlock {
        VmBasicBlock {
            start: RelativeDispatchState {
                vip: Va(0x10),
                rolling_key: 0,
                handler: Va(0x20),
            },
            address: None,
            instructions: Vec::new(),
            exit,
            boundary_handler: handler,
            native_call: None,
            vm_exit: None,
        }
    }

    fn cfg_of(blocks: Vec<(u64, VmBasicBlock)>) -> VmCfg {
        VmCfg {
            entry: Va(0x10),
            blocks: blocks.into_iter().map(|(key, b)| (Va(key), b)).collect(),
            edges: Vec::new(),
            stalls: Vec::new(),
            limit_hit: None,
            visits: 0,
            frame_cells_known: 0,
            frame_note: None,
        }
    }

    fn exit_report(status: RecoveryStatus) -> VmExitReport {
        VmExitReport {
            schema_version: 1,
            entry: vmpx_pe::Rva(0x1000),
            status,
            stack_switch: None,
            restored_registers: Vec::new(),
            restore_slots: Vec::new(),
            flags_restore: None,
            flags_stack_offset: Some(0x70),
            native_return: None,
            return_stack_offset: Some(0x78),
            frame_size: Some(0x80),
            confidence: 100,
            evidence: Vec::new(),
            diagnostics: Vec::new(),
        }
    }

    #[test]
    fn cfg_carries_the_native_call_outcome_and_exit_layout() {
        let outcome = NativeCallSelectorOutcome {
            selector: 3,
            vsp_delta: 16,
            target_stack_offset: 0,
            register_argument_offsets: vec![8],
            stack_argument_offsets: Vec::new(),
        };
        let mut exit_block = block(BlockExit::VmExit, Some(Va(0x99)));
        exit_block.vm_exit = proven_exit_layout(Some(&exit_report(RecoveryStatus::Recovered)));
        let mut cfg = cfg_of(vec![
            (0x10, block(BlockExit::NativeCall, Some(Va(0x77)))),
            (0x20, exit_block),
            (0x30, block(BlockExit::Branch, Some(Va(0x55)))),
        ]);
        cfg.record_native_call(Va(0x10), &outcome);
        cfg.record_native_call(Va(0x30), &outcome);
        assert_eq!(cfg.native_calls().len(), 1);
        assert_eq!(cfg.native_calls()[&Va(0x10)], outcome);
        let layouts = cfg.exit_layouts();
        assert_eq!(layouts.len(), 1);
        assert_eq!(layouts[&Va(0x99)].frame_size, Some(0x80));
    }

    #[test]
    fn unproven_exit_layouts_are_not_carried() {
        assert!(proven_exit_layout(None).is_none());
        for status in [RecoveryStatus::PartiallyRecovered, RecoveryStatus::Failed] {
            assert!(proven_exit_layout(Some(&exit_report(status))).is_none());
        }
    }
}
