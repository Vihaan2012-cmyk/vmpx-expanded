use std::collections::{BTreeMap, BTreeSet, VecDeque};

use iced_x86::{FlowControl, Mnemonic, OpKind, Register};
use serde::Serialize;
use vmpx_disasm::{CfgLimits, CfgStatus, DecodedInstruction, DisasmError, decode_one, recover_cfg};
use vmpx_pe::{PeImage, Rva};

use crate::{
    BytecodeDecodeError, BytecodeDirection, HandlerRegisterRoles, NativeCallSelectorOutcome,
    RecoveryStatus, RelativeDispatchState, RelativeThreadedCodec,
    dispatch_extract::{canonical, has_flat_segment_address, writes_register},
    native_call_verify::verify_native_call_outcomes,
};

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct NativeCallAnalysisReport {
    pub schema_version: u32,
    pub entry: Rva,
    pub status: RecoveryStatus,
    pub call_site: Option<Rva>,
    pub resume_rva: Option<Rva>,
    pub call_target_register: Option<String>,
    pub target_load: Option<Rva>,
    pub vsp_alias_register: Option<String>,
    pub native_rsp_save: Option<Rva>,
    pub native_rsp_restore: Option<Rva>,
    pub return_store: Option<Rva>,
    pub vsp_restore: Option<Rva>,
    pub selector_register: Option<String>,
    pub selector_completion: Option<Rva>,
    pub dispatch_source: Option<Rva>,
    pub terminal_dispatch: Option<Rva>,
    pub saved_vm_registers: Vec<String>,
    pub restored_vm_registers: Vec<String>,
    pub selector_outcomes: Vec<NativeCallSelectorOutcome>,
    pub cfg_blocks: usize,
    pub cfg_instructions: usize,
    pub confidence: u8,
    pub evidence: Vec<String>,
    pub diagnostics: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RecoveredNativeCall {
    pub report: NativeCallAnalysisReport,
    pub resume_rva: Rva,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeCallAnalysis {
    pub report: NativeCallAnalysisReport,
    pub recovered: Option<RecoveredNativeCall>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct NativeCallBoundarySpec {
    pub codec: RelativeThreadedCodec,
    pub outcomes: Vec<NativeCallSelectorOutcome>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct DecodedNativeCallBoundary {
    pub state_before: RelativeDispatchState,
    pub encoded_bytes: Vec<u8>,
    pub selector: u8,
    pub outcome: NativeCallSelectorOutcome,
    pub state_after: RelativeDispatchState,
}

#[derive(Clone, Debug, thiserror::Error, Eq, PartialEq)]
pub enum NativeCallBoundaryDecodeError {
    #[error(transparent)]
    Bytecode(#[from] BytecodeDecodeError),

    #[error("decoded native-call selector {selector} has no proven outcome")]
    MissingOutcome { selector: u64 },
}

impl NativeCallBoundarySpec {
    /// Decodes the selector and post-call threaded dispatch, then binds the
    /// selector to one of the exhaustively proven native bridge outcomes.
    ///
    /// # Errors
    ///
    /// Returns a bytecode error for invalid/unmapped encoded bytes or a
    /// missing-outcome error if the decoded selector was not proven.
    pub fn decode_direction(
        &self,
        pe: &PeImage<'_>,
        state: RelativeDispatchState,
        direction: BytecodeDirection,
    ) -> Result<DecodedNativeCallBoundary, NativeCallBoundaryDecodeError> {
        let decoded = self
            .codec
            .decode_operand_dispatch_direction(pe, state, direction)?;
        let selector = u8::try_from(decoded.operand).map_err(|_| {
            NativeCallBoundaryDecodeError::MissingOutcome {
                selector: decoded.operand,
            }
        })?;
        let outcome = self
            .outcomes
            .get(usize::from(selector))
            .filter(|outcome| outcome.selector == selector)
            .cloned()
            .ok_or(NativeCallBoundaryDecodeError::MissingOutcome {
                selector: decoded.operand,
            })?;
        let mut encoded_bytes = decoded.encoded_operand;
        encoded_bytes.extend_from_slice(&decoded.encoded_displacement);
        Ok(DecodedNativeCallBoundary {
            state_before: decoded.state_before,
            encoded_bytes,
            selector,
            outcome,
            state_after: decoded.state_after,
        })
    }
}

fn exact_full_move(instruction: &iced_x86::Instruction) -> Option<(Register, Register)> {
    (instruction.mnemonic() == Mnemonic::Mov
        && instruction.op0_kind() == OpKind::Register
        && instruction.op0_register().size() == 8
        && instruction.op1_kind() == OpKind::Register
        && instruction.op1_register().size() == 8)
        .then(|| {
            (
                canonical(instruction.op0_register()),
                canonical(instruction.op1_register()),
            )
        })
}

fn exact_load(instruction: &iced_x86::Instruction) -> Option<(Register, Register, u64)> {
    (instruction.mnemonic() == Mnemonic::Mov
        && instruction.op0_kind() == OpKind::Register
        && instruction.op0_register().size() == 8
        && instruction.op1_kind() == OpKind::Memory
        && instruction.memory_base().size() == 8
        && instruction.memory_index() == Register::None
        && instruction.memory_size().size() == 8
        && has_flat_segment_address(instruction))
    .then(|| {
        (
            canonical(instruction.op0_register()),
            canonical(instruction.memory_base()),
            instruction.memory_displacement64(),
        )
    })
}

fn exact_store(instruction: &iced_x86::Instruction) -> Option<(Register, u64, Register)> {
    (instruction.mnemonic() == Mnemonic::Mov
        && instruction.op0_kind() == OpKind::Memory
        && instruction.memory_base().size() == 8
        && instruction.memory_index() == Register::None
        && instruction.memory_size().size() == 8
        && instruction.op1_kind() == OpKind::Register
        && instruction.op1_register().size() == 8
        && has_flat_segment_address(instruction))
    .then(|| {
        (
            canonical(instruction.memory_base()),
            instruction.memory_displacement64(),
            canonical(instruction.op1_register()),
        )
    })
}

fn exact_stack_register(
    instruction: &iced_x86::Instruction,
    mnemonic: Mnemonic,
) -> Option<Register> {
    (instruction.mnemonic() == mnemonic
        && instruction.op0_kind() == OpKind::Register
        && instruction.op0_register().size() == 8)
        .then(|| canonical(instruction.op0_register()))
}

fn is_win64_nonvolatile(register: Register) -> bool {
    matches!(
        canonical(register),
        Register::RBX
            | Register::RBP
            | Register::RSI
            | Register::RDI
            | Register::RSP
            | Register::R12
            | Register::R13
            | Register::R14
            | Register::R15
    )
}

fn build_instruction_graph(
    pe: &PeImage<'_>,
    instructions: &BTreeMap<Rva, DecodedInstruction>,
) -> BTreeMap<Rva, BTreeSet<Rva>> {
    instructions
        .iter()
        .map(|(rva, decoded)| {
            let flow = decoded.instruction.flow_control();
            let next = decoded
                .next_rva()
                .filter(|next| instructions.contains_key(next));
            let direct = decoded
                .direct_target_rva(pe)
                .filter(|target| instructions.contains_key(target));
            let successors = match flow {
                FlowControl::ConditionalBranch => [direct, next].into_iter().flatten().collect(),
                FlowControl::UnconditionalBranch => direct.into_iter().collect(),
                FlowControl::Call | FlowControl::IndirectCall | FlowControl::Next => {
                    next.into_iter().collect()
                }
                FlowControl::Return
                | FlowControl::IndirectBranch
                | FlowControl::Interrupt
                | FlowControl::Exception
                | FlowControl::XbeginXabortXend => BTreeSet::new(),
            };
            (*rva, successors)
        })
        .collect()
}

fn reachable(graph: &BTreeMap<Rva, BTreeSet<Rva>>, root: Rva) -> BTreeSet<Rva> {
    let mut queue = VecDeque::from([root]);
    let mut seen = BTreeSet::new();
    while let Some(node) = queue.pop_front() {
        if !seen.insert(node) {
            continue;
        }
        if let Some(successors) = graph.get(&node) {
            queue.extend(successors.iter().copied());
        }
    }
    seen
}

fn reverse_reachable(graph: &BTreeMap<Rva, BTreeSet<Rva>>, target: Rva) -> BTreeSet<Rva> {
    let mut reverse: BTreeMap<Rva, BTreeSet<Rva>> = BTreeMap::new();
    for (from, successors) in graph {
        for to in successors {
            reverse.entry(*to).or_default().insert(*from);
        }
    }
    reachable(&reverse, target)
}

fn reaches_while_skipping(
    graph: &BTreeMap<Rva, BTreeSet<Rva>>,
    root: Rva,
    target: Rva,
    skipped: Option<Rva>,
) -> bool {
    if skipped == Some(root) {
        return false;
    }
    let mut queue = VecDeque::from([root]);
    let mut seen = BTreeSet::new();
    while let Some(node) = queue.pop_front() {
        if skipped == Some(node) || !seen.insert(node) {
            continue;
        }
        if node == target {
            return true;
        }
        if let Some(successors) = graph.get(&node) {
            queue.extend(successors.iter().copied());
        }
    }
    false
}

/// Answers a bounded dominance query without allocating the quadratic
/// node-by-node dominator sets that hostile native CFGs can otherwise force.
fn dominates(graph: &BTreeMap<Rva, BTreeSet<Rva>>, root: Rva, left: Rva, right: Rva) -> bool {
    reaches_while_skipping(graph, root, right, None)
        && (left == right || !reaches_while_skipping(graph, root, right, Some(left)))
}

fn failed_report(
    entry: Rva,
    cfg_blocks: usize,
    cfg_instructions: usize,
    diagnostics: Vec<String>,
) -> NativeCallAnalysis {
    NativeCallAnalysis {
        report: NativeCallAnalysisReport {
            schema_version: 2,
            entry,
            status: RecoveryStatus::Failed,
            call_site: None,
            resume_rva: None,
            call_target_register: None,
            target_load: None,
            vsp_alias_register: None,
            native_rsp_save: None,
            native_rsp_restore: None,
            return_store: None,
            vsp_restore: None,
            selector_register: None,
            selector_completion: None,
            dispatch_source: None,
            terminal_dispatch: None,
            saved_vm_registers: Vec::new(),
            restored_vm_registers: Vec::new(),
            selector_outcomes: Vec::new(),
            cfg_blocks,
            cfg_instructions,
            confidence: 0,
            evidence: Vec::new(),
            diagnostics,
        },
        recovered: None,
    }
}

/// Proves the invariant shape of a returning native-call bridge embedded in a
/// threaded VM handler. The proof uses control-flow dominance and register/
/// memory effects; it does not rely on handler bytes, addresses, or API names.
///
/// # Errors
///
/// Propagates failure to decode the required native entry. Ordinary ambiguity,
/// malformed control flow, and resource limits are returned as a failed report.
#[allow(clippy::too_many_lines)]
pub fn analyze_native_call_handler(
    pe: &PeImage<'_>,
    entry: Rva,
    roles: HandlerRegisterRoles,
    vsp: Register,
    selector_register: Register,
    selector_completion: Rva,
    limits: CfgLimits,
) -> Result<NativeCallAnalysis, DisasmError> {
    let cfg = recover_cfg(pe, entry, limits)?;
    let instruction_count = cfg
        .blocks
        .iter()
        .map(|block| block.instructions.len())
        .sum();
    if cfg.status != CfgStatus::Complete {
        return Ok(failed_report(
            entry,
            cfg.blocks.len(),
            instruction_count,
            vec![format!("native-call CFG is incomplete: {:?}", cfg.status)],
        ));
    }
    let mut instructions = BTreeMap::new();
    for summary in cfg.blocks.iter().flat_map(|block| &block.instructions) {
        instructions.insert(summary.rva, decode_one(pe, summary.rva)?);
    }
    let graph = build_instruction_graph(pe, &instructions);
    let all_calls = instructions
        .iter()
        .filter(|(_, decoded)| {
            matches!(
                decoded.instruction.flow_control(),
                FlowControl::Call | FlowControl::IndirectCall
            )
        })
        .collect::<Vec<_>>();
    let calls = instructions
        .iter()
        .filter(|(_, decoded)| decoded.instruction.flow_control() == FlowControl::IndirectCall)
        .collect::<Vec<_>>();
    if all_calls.len() != 1 {
        return Ok(failed_report(
            entry,
            cfg.blocks.len(),
            instruction_count,
            vec![format!(
                "expected exactly one modeled call, recovered {} direct/indirect calls",
                all_calls.len()
            )],
        ));
    }
    let [(call_site, call)] = calls.as_slice() else {
        return Ok(failed_report(
            entry,
            cfg.blocks.len(),
            instruction_count,
            vec![format!(
                "expected one indirect native call, recovered {}",
                calls.len()
            )],
        ));
    };
    if call.instruction.op0_kind() != OpKind::Register {
        return Ok(failed_report(
            entry,
            cfg.blocks.len(),
            instruction_count,
            vec!["indirect call target is not held in a register".to_owned()],
        ));
    }
    let call_site = **call_site;
    let call_target = canonical(call.instruction.op0_register());
    let role_registers = [roles.vip, roles.handler, roles.rolling_key]
        .map(canonical)
        .into_iter()
        .collect::<BTreeSet<_>>();
    if role_registers.len() != 3 {
        return Ok(failed_report(
            entry,
            cfg.blocks.len(),
            instruction_count,
            vec!["VIP, rolling-key, and handler roles are not distinct".to_owned()],
        ));
    }
    let Some(resume_rva) = call
        .next_rva()
        .filter(|next| instructions.contains_key(next))
    else {
        return Ok(failed_report(
            entry,
            cfg.blocks.len(),
            instruction_count,
            vec!["native call has no recovered fallthrough".to_owned()],
        ));
    };

    let dispatches = instructions
        .iter()
        .filter(|(_, decoded)| {
            decoded.instruction.flow_control() == FlowControl::IndirectBranch
                && decoded.instruction.op0_kind() == OpKind::Register
                && canonical(decoded.instruction.op0_register()) == canonical(roles.handler)
        })
        .map(|(rva, _)| *rva)
        .collect::<Vec<_>>();
    let [dispatch] = dispatches.as_slice() else {
        return Ok(failed_report(
            entry,
            cfg.blocks.len(),
            instruction_count,
            vec![format!(
                "expected one terminal dispatch through {:?}, recovered {}",
                canonical(roles.handler),
                dispatches.len()
            )],
        ));
    };
    let dispatch = *dispatch;
    if !dominates(&graph, entry, call_site, dispatch) {
        return Ok(failed_report(
            entry,
            cfg.blocks.len(),
            instruction_count,
            vec!["the native call does not dominate threaded dispatch".to_owned()],
        ));
    }
    if instructions.iter().any(|(rva, decoded)| {
        graph.get(rva).is_some_and(BTreeSet::is_empty)
            && *rva != dispatch
            && decoded.instruction.flow_control() != FlowControl::IndirectCall
    }) {
        return Ok(failed_report(
            entry,
            cfg.blocks.len(),
            instruction_count,
            vec!["native-call CFG has a non-dispatch terminal path".to_owned()],
        ));
    }

    let after_call = reachable(&graph, resume_rva);
    let reaches_dispatch = reverse_reachable(&graph, dispatch);
    if !after_call.is_subset(&reaches_dispatch) {
        return Ok(failed_report(
            entry,
            cfg.blocks.len(),
            instruction_count,
            vec!["not every returning call path reaches threaded dispatch".to_owned()],
        ));
    }
    let target_loads = instructions
        .iter()
        .filter_map(|(rva, decoded)| {
            let (destination, base, displacement) = exact_load(&decoded.instruction)?;
            (destination == call_target
                && displacement == 0
                && dominates(&graph, entry, *rva, call_site))
            .then_some((*rva, base))
        })
        .collect::<Vec<_>>();
    let [(target_load, vsp_alias)] = target_loads.as_slice() else {
        return Ok(failed_report(
            entry,
            cfg.blocks.len(),
            instruction_count,
            vec![format!(
                "expected one dominating qword load into the call target, recovered {}",
                target_loads.len()
            )],
        ));
    };
    let target_load = *target_load;
    let vsp_alias = *vsp_alias;
    let between_target_and_call = reachable(&graph, target_load)
        .intersection(&reverse_reachable(&graph, call_site))
        .copied()
        .collect::<BTreeSet<_>>();
    let mut target_factory = iced_x86::InstructionInfoFactory::new();
    if between_target_and_call.iter().any(|rva| {
        *rva != target_load
            && *rva != call_site
            && writes_register(
                &mut target_factory,
                &instructions[rva].instruction,
                call_target,
            )
    }) {
        return Ok(failed_report(
            entry,
            cfg.blocks.len(),
            instruction_count,
            vec!["call-target register is clobbered after its VSP load".to_owned()],
        ));
    }
    let alias_copies = instructions
        .iter()
        .filter_map(|(rva, decoded)| {
            let (destination, source) = exact_full_move(&decoded.instruction)?;
            (destination == vsp_alias
                && source == canonical(vsp)
                && dominates(&graph, entry, *rva, target_load))
            .then_some(*rva)
        })
        .collect::<Vec<_>>();
    let [alias_copy] = alias_copies.as_slice() else {
        return Ok(failed_report(
            entry,
            cfg.blocks.len(),
            instruction_count,
            vec!["call-frame base is not a uniquely proven VSP alias".to_owned()],
        ));
    };
    if !is_win64_nonvolatile(vsp_alias) {
        return Ok(failed_report(
            entry,
            cfg.blocks.len(),
            instruction_count,
            vec![format!(
                "VSP alias {vsp_alias:?} is volatile across the Win64 native call"
            )],
        ));
    }
    let between_alias_and_call = reachable(&graph, *alias_copy)
        .intersection(&reverse_reachable(&graph, call_site))
        .copied()
        .collect::<BTreeSet<_>>();
    let mut factory = iced_x86::InstructionInfoFactory::new();
    if between_alias_and_call.iter().any(|rva| {
        *rva != *alias_copy
            && writes_register(&mut factory, &instructions[rva].instruction, vsp_alias)
    }) {
        return Ok(failed_report(
            entry,
            cfg.blocks.len(),
            instruction_count,
            vec!["VSP alias is clobbered between establishment and call".to_owned()],
        ));
    }

    let rsp_saves = instructions
        .iter()
        .filter_map(|(rva, decoded)| {
            let (base, displacement, source) = exact_store(&decoded.instruction)?;
            (base == vsp_alias
                && source == Register::RSP
                && dominates(&graph, entry, *rva, call_site))
            .then_some((*rva, displacement))
        })
        .collect::<Vec<_>>();
    let [(rsp_save, rsp_displacement)] = rsp_saves.as_slice() else {
        return Ok(failed_report(
            entry,
            cfg.blocks.len(),
            instruction_count,
            vec!["native RSP is not saved exactly once through the VSP alias".to_owned()],
        ));
    };
    if !dominates(&graph, entry, *alias_copy, *rsp_save) {
        return Ok(failed_report(
            entry,
            cfg.blocks.len(),
            instruction_count,
            vec!["native RSP is saved before the VSP alias is established".to_owned()],
        ));
    }
    let rsp_restores = instructions
        .iter()
        .filter_map(|(rva, decoded)| {
            let (destination, base, displacement) = exact_load(&decoded.instruction)?;
            (destination == Register::RSP
                && base == vsp_alias
                && displacement == *rsp_displacement
                && dominates(&graph, resume_rva, *rva, dispatch))
            .then_some(*rva)
        })
        .collect::<Vec<_>>();
    let [rsp_restore] = rsp_restores.as_slice() else {
        return Ok(failed_report(
            entry,
            cfg.blocks.len(),
            instruction_count,
            vec!["saved native RSP is not restored on every dispatch path".to_owned()],
        ));
    };

    let return_stores = instructions
        .iter()
        .filter_map(|(rva, decoded)| {
            let (base, displacement, source) = exact_store(&decoded.instruction)?;
            (base == vsp_alias
                && displacement == 0
                && source == Register::RAX
                && dominates(&graph, resume_rva, *rva, dispatch))
            .then_some(*rva)
        })
        .collect::<Vec<_>>();
    let [return_store] = return_stores.as_slice() else {
        return Ok(failed_report(
            entry,
            cfg.blocks.len(),
            instruction_count,
            vec!["native RAX is not returned to the VM stack on every path".to_owned()],
        ));
    };
    let between_call_and_result = reachable(&graph, resume_rva)
        .intersection(&reverse_reachable(&graph, *return_store))
        .copied()
        .collect::<BTreeSet<_>>();
    let mut result_factory = iced_x86::InstructionInfoFactory::new();
    if between_call_and_result.iter().any(|rva| {
        *rva != *return_store
            && writes_register(
                &mut result_factory,
                &instructions[rva].instruction,
                Register::RAX,
            )
    }) {
        return Ok(failed_report(
            entry,
            cfg.blocks.len(),
            instruction_count,
            vec!["RAX is clobbered before the native result reaches the VM stack".to_owned()],
        ));
    }
    let vsp_restores = instructions
        .iter()
        .filter_map(|(rva, decoded)| {
            let (destination, source) = exact_full_move(&decoded.instruction)?;
            (destination == canonical(vsp)
                && source == vsp_alias
                && dominates(&graph, resume_rva, *rva, dispatch))
            .then_some(*rva)
        })
        .collect::<Vec<_>>();
    let [vsp_restore] = vsp_restores.as_slice() else {
        return Ok(failed_report(
            entry,
            cfg.blocks.len(),
            instruction_count,
            vec!["post-call VM stack pointer is not restored on every path".to_owned()],
        ));
    };

    let mut saved = BTreeSet::new();
    let mut restored = BTreeSet::new();
    for (rva, decoded) in &instructions {
        if let Some(register) = exact_stack_register(&decoded.instruction, Mnemonic::Push)
            && role_registers.contains(&register)
            && dominates(&graph, entry, *rva, call_site)
        {
            saved.insert(register);
        }
        if let Some(register) = exact_stack_register(&decoded.instruction, Mnemonic::Pop)
            && role_registers.contains(&register)
            && dominates(&graph, resume_rva, *rva, dispatch)
        {
            restored.insert(register);
        }
    }
    if saved != role_registers || restored != role_registers {
        return Ok(failed_report(
            entry,
            cfg.blocks.len(),
            instruction_count,
            vec![format!(
                "VM role save/restore is incomplete: saved {saved:?}, restored {restored:?}"
            )],
        ));
    }

    let dispatch_sources = instructions
        .iter()
        .filter_map(|(rva, decoded)| {
            let instruction = &decoded.instruction;
            (instruction.mnemonic() == Mnemonic::Mov
                && instruction.op0_kind() == OpKind::Register
                && instruction.op0_register().size() == 4
                && instruction.op1_kind() == OpKind::Memory
                && instruction.memory_base() == canonical(roles.vip)
                && instruction.memory_index() == Register::None
                && instruction.memory_displacement64() == 0
                && instruction.memory_size().size() == 4
                && has_flat_segment_address(instruction)
                && dominates(&graph, resume_rva, *rva, dispatch))
            .then_some(*rva)
        })
        .collect::<Vec<_>>();
    let [dispatch_source] = dispatch_sources.as_slice() else {
        return Ok(failed_report(
            entry,
            cfg.blocks.len(),
            instruction_count,
            vec![format!(
                "expected one post-call W32 dispatch source, recovered {}",
                dispatch_sources.len()
            )],
        ));
    };

    let selector_outcomes = match verify_native_call_outcomes(
        pe,
        &instructions,
        entry,
        call_site,
        *dispatch_source,
        dispatch,
        selector_completion,
        selector_register,
        vsp,
        roles,
        limits.max_instructions.max(1),
    ) {
        Ok(outcomes) => outcomes,
        Err(diagnostic) => {
            return Ok(failed_report(
                entry,
                cfg.blocks.len(),
                instruction_count,
                vec![diagnostic],
            ));
        }
    };

    let evidence = vec![
        format!(
            "call target {:?} is loaded from the proven VSP alias at {target_load}",
            call_target
        ),
        format!("native RSP is saved at {rsp_save} and restored at {rsp_restore}"),
        format!("native RAX is written back to the VM stack at {return_store}"),
        format!("VIP, rolling key, and handler registers are restored before {dispatch}"),
        "all 256 W8 selector outcomes prove target/argument marshalling, result placement, and exact VM/native stack restoration".to_owned(),
        "every returning native-call path reaches threaded VM dispatch".to_owned(),
    ];
    let report = NativeCallAnalysisReport {
        schema_version: 2,
        entry,
        status: RecoveryStatus::Recovered,
        call_site: Some(call_site),
        resume_rva: Some(resume_rva),
        call_target_register: Some(format!("{call_target:?}")),
        target_load: Some(target_load),
        vsp_alias_register: Some(format!("{vsp_alias:?}")),
        native_rsp_save: Some(*rsp_save),
        native_rsp_restore: Some(*rsp_restore),
        return_store: Some(*return_store),
        vsp_restore: Some(*vsp_restore),
        selector_register: Some(format!("{:?}", canonical(selector_register))),
        selector_completion: Some(selector_completion),
        dispatch_source: Some(*dispatch_source),
        terminal_dispatch: Some(dispatch),
        saved_vm_registers: saved
            .iter()
            .map(|register| format!("{register:?}"))
            .collect(),
        restored_vm_registers: restored
            .iter()
            .map(|register| format!("{register:?}"))
            .collect(),
        selector_outcomes,
        cfg_blocks: cfg.blocks.len(),
        cfg_instructions: instruction_count,
        confidence: 100,
        evidence,
        diagnostics: Vec::new(),
    };
    Ok(NativeCallAnalysis {
        recovered: Some(RecoveredNativeCall {
            report: report.clone(),
            resume_rva,
        }),
        report,
    })
}
