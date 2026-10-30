//! Lifts a VM entry from its decoded handler stream to native bytes, for
//! splicing into the experimental image. Any failure leaves the stub alone.

use std::collections::{BTreeMap, BTreeSet};

use vmpx_pe::{PeImage, Rva, Va};
use vmpx_recompile::CodeAddress;
use vmpx_vm::{
    BlockExit, CfgEdgeKind, EntryProfileRecoveryLimits, Expr, NativeCallImportResolution,
    NativeCallMemoryDecision, NativeFrameSource, ReadOnlyImage, SymbolicExecutionLimits,
    SymbolicMemorySource, SymbolicVmExecutionReport, SymbolicVmInitialState, VmCfg, VmCfgLimits,
    VmDiscoveryLimits, VmHandlerSemantic, analyze_vm_entry_at,
    execute_symbolic_linear_block_with_initial_state,
    execute_symbolic_native_call_boundary_with_initial_state, recover_entry_stream,
    recover_vm_cfg, symbolic_pointer_offset,
};
use vmpx_vmlift::{
    DecodedCfgInput, DecodedLinearInput, frame_initial_state, lift_decoded_cfg,
    lift_decoded_linear, recover_entry_frame,
};

const MAX_WALK_BLOCKS: usize = 256;

#[derive(Debug)]
pub(crate) struct SplicedBody {
    pub bytes: Vec<u8>,
    pub kind: &'static str,
    /// `module!symbol` of each native call proven to reach exactly one import.
    pub imports: BTreeSet<String>,
}

/// `module!symbol` (or `module!#ordinal`) for a uniquely resolved import only.
fn import_label(resolution: &NativeCallImportResolution) -> Option<String> {
    let NativeCallImportResolution::Unique { import, .. } = resolution else {
        return None;
    };
    let symbol = match (&import.name, import.ordinal) {
        (Some(name), _) => name.clone(),
        (None, Some(ordinal)) => format!("#{ordinal}"),
        (None, None) => return None,
    };
    Some(format!("{}!{symbol}", import.module))
}

/// State after a call that cannot reach the VM stack: the call's own cells
/// replace what they cover, other VM-stack and context cells survive.
fn restore_vm_state(
    pre: &SymbolicVmInitialState,
    call: &SymbolicVmExecutionReport,
) -> SymbolicVmInitialState {
    let cells: BTreeMap<String, (Expr, Expr)> = pre
        .memory_values
        .iter()
        .filter(|(address, _)| symbolic_pointer_offset(address, "vsp_initial").is_some())
        .chain(call.memory_values.iter())
        .map(|cell| (format!("{:?}", cell.0), cell.clone()))
        .collect();
    let mut post = call.continuation_state();
    post.memory_values = cells.into_values().collect();
    post.context_values = pre.context_values.clone();
    post
}

/// Imports reached by the CFG's native calls, walking block by block from the
/// entry with the lifter's own state propagation. A call that is not proven to
/// hit one import, or whose block cannot be executed, contributes nothing.
fn resolved_imports(
    cfg: &VmCfg,
    frame: &[NativeFrameSource],
    top_offset: u64,
    memory: &dyn SymbolicMemorySource,
    limits: SymbolicExecutionLimits,
) -> BTreeSet<String> {
    let calls = cfg.native_calls();
    let mut found = BTreeSet::new();
    let mut budget = MAX_WALK_BLOCKS;
    let mut work: Vec<(Va, SymbolicVmInitialState, Vec<Va>)> =
        vec![(cfg.entry, frame_initial_state(frame, top_offset).0, Vec::new())];
    while let Some((mut key, mut state, mut trail)) = work.pop() {
        loop {
            if budget == 0 || trail.contains(&key) {
                break;
            }
            budget -= 1;
            trail.push(key);
            let Some(block) = cfg.blocks.get(&key) else {
                break;
            };
            let boundary = match block.exit {
                BlockExit::Branch => Some(VmHandlerSemantic::UnconditionalBranch),
                BlockExit::NativeCall => None,
                BlockExit::VmExit | BlockExit::Stalled => break,
            };
            let Ok(report) = execute_symbolic_linear_block_with_initial_state(
                &block.instructions,
                boundary,
                limits,
                &state,
                Some(memory),
            ) else {
                break;
            };
            let next = report.continuation_state();
            if block.exit == BlockExit::Branch {
                let mut targets = cfg.edges.iter().filter(|edge| {
                    edge.from == key
                        && matches!(edge.kind, CfgEdgeKind::Branch | CfgEdgeKind::GuardedBranch)
                });
                let Some(first) = targets.next() else { break };
                work.extend(targets.map(|edge| (edge.to, next.clone(), trail.clone())));
                key = first.to;
                state = next;
                continue;
            }
            let Some(outcome) = calls.get(&key) else {
                break;
            };
            let vip = block
                .instructions
                .last()
                .map_or(block.start.vip, |last| last.state_after.vip);
            let Ok(call) = execute_symbolic_native_call_boundary_with_initial_state(
                vip,
                outcome,
                limits,
                &next,
                Some(memory),
            ) else {
                break;
            };
            found.extend(
                call.native_calls
                    .iter()
                    .filter_map(|c| import_label(c.import_resolution.as_ref()?)),
            );
            let mut returns = cfg
                .edges
                .iter()
                .filter(|edge| edge.from == key && edge.kind == CfgEdgeKind::NativeCallReturn);
            let (Some(edge), None) = (returns.next(), returns.next()) else {
                break;
            };
            state = match call.native_calls.first().map(|c| &c.applied_effect.memory) {
                Some(NativeCallMemoryDecision::HavocTracked { .. }) => {
                    restore_vm_state(&next, &call)
                }
                _ => call.continuation_state(),
            };
            key = edge.to;
        }
    }
    found
}

/// Lifts the VM entry at `rva` to a callee-style native body placed at `address`.
pub(crate) fn lift_entry(pe: &PeImage<'_>, rva: Rva, address: u64) -> Result<SplicedBody, String> {
    let candidate = analyze_vm_entry_at(pe, rva, VmDiscoveryLimits::default())
        .ok_or_else(|| "no VM entry stub at this RVA".to_owned())?;
    let entry_limits = EntryProfileRecoveryLimits {
        instruction_limit: 100_000,
        ..EntryProfileRecoveryLimits::default()
    };
    let stream = recover_entry_stream(pe, &candidate, entry_limits)
        .map_err(|error| format!("stream recovery failed: {error}"))?;
    let (frame, top_offset) = recover_entry_frame(
        pe,
        candidate.rva,
        stream.profile.effect_roles.vsp,
        entry_limits.trace,
    )
    .map_err(|error| format!("entry frame not recovered: {error}"))?;
    let decode = &stream.decode.decode;
    let address = CodeAddress(address);
    let image = ReadOnlyImage(pe);

    let linear_error = match stream
        .decode
        .handler_recoveries
        .get(&decode.final_state.handler)
        .and_then(|recovery| recovery.vm_exit.as_ref())
    {
        Some(exit) => {
            let input = DecodedLinearInput {
                instructions: &decode.instructions,
                frame: &frame,
                top_offset,
                exit,
                memory: Some(&image),
                limits: SymbolicExecutionLimits::default(),
            };
            match lift_decoded_linear(&input, address) {
                Ok(lifted) => {
                    return Ok(SplicedBody {
                        bytes: lifted.replacement.compiled.bytes,
                        kind: "linear",
                        imports: BTreeSet::new(),
                    });
                }
                Err(error) => error.to_string(),
            }
        }
        None => "stream does not end at a recovered VM-exit handler".to_owned(),
    };

    let limits = VmCfgLimits {
        entry: entry_limits,
        ..VmCfgLimits::default()
    };
    let cfg = recover_vm_cfg(pe, &candidate, limits)
        .map_err(|error| format!("linear: {linear_error}; cfg recovery failed: {error}"))?;
    if !cfg
        .blocks
        .values()
        .any(|block| block.exit == BlockExit::VmExit)
    {
        return Err(format!("linear: {linear_error}; cfg has no VM-exit block"));
    }
    let exits: BTreeMap<_, _> = stream
        .decode
        .handler_recoveries
        .iter()
        .filter_map(|(handler, recovery)| Some((*handler, recovery.vm_exit.clone()?)))
        .collect();
    let calls = cfg.native_calls();
    let input = DecodedCfgInput {
        cfg: &cfg,
        frame: &frame,
        top_offset,
        exits: &exits,
        native_calls: &calls,
        memory: Some(&image),
        limits: SymbolicExecutionLimits::default(),
        target_limits: limits,
        max_expanded_blocks: 256,
    };
    let lifted = lift_decoded_cfg(&input, address)
        .map_err(|error| format!("linear: {linear_error}; cfg: {error}"))?;
    let imports = if lifted.native_calls > 0 {
        resolved_imports(&cfg, &frame, top_offset, &image, SymbolicExecutionLimits::default())
    } else {
        BTreeSet::new()
    };
    Ok(SplicedBody {
        bytes: lifted.replacement.compiled.bytes,
        kind: "cfg",
        imports,
    })
}

#[cfg(test)]
mod tests {
    use vmpx_pe::Rva;
    use vmpx_vm::NativeCallImportDescriptor;

    use super::*;

    fn descriptor(name: Option<&str>, ordinal: Option<u16>) -> NativeCallImportDescriptor {
        NativeCallImportDescriptor {
            module: "kernel32.dll".to_owned(),
            iat_rva: Rva(0x2000),
            lookup_value: 0,
            name: name.map(str::to_owned),
            ordinal,
            match_kinds: Vec::new(),
        }
    }

    #[test]
    fn only_unique_resolutions_are_labelled() {
        let unique = |import| NativeCallImportResolution::Unique { target: 1, import };
        assert_eq!(
            import_label(&unique(descriptor(Some("HeapFree"), None))).as_deref(),
            Some("kernel32.dll!HeapFree")
        );
        assert_eq!(
            import_label(&unique(descriptor(None, Some(7)))).as_deref(),
            Some("kernel32.dll!#7")
        );
        assert_eq!(import_label(&unique(descriptor(None, None))), None);
        assert_eq!(
            import_label(&NativeCallImportResolution::NoMatch { target: 1 }),
            None
        );
        assert_eq!(
            import_label(&NativeCallImportResolution::Ambiguous {
                target: 1,
                imports: vec![descriptor(Some("A"), None), descriptor(Some("B"), None)],
            }),
            None
        );
    }
}
