//! Lifts a VM entry from its decoded handler stream to native bytes, for
//! splicing into the experimental image. Any failure leaves the stub alone.

use std::collections::BTreeMap;

use vmpx_pe::{PeImage, Rva};
use vmpx_recompile::CodeAddress;
use vmpx_vm::{
    BlockExit, EntryProfileRecoveryLimits, SymbolicExecutionLimits, VmCfgLimits, VmDiscoveryLimits,
    analyze_vm_entry_at, recover_entry_stream, recover_vm_cfg,
};
use vmpx_vmlift::{
    DecodedCfgInput, DecodedLinearInput, lift_decoded_cfg, lift_decoded_linear, recover_entry_frame,
};

#[derive(Debug)]
pub(crate) struct SplicedBody {
    pub bytes: Vec<u8>,
    pub kind: &'static str,
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
    let frame = recover_entry_frame(
        pe,
        candidate.rva,
        stream.profile.effect_roles.vsp,
        entry_limits.trace,
    )
    .map_err(|error| format!("entry frame not recovered: {error}"))?;
    let decode = &stream.decode.decode;
    let address = CodeAddress(address);

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
                exit,
                memory: Some(pe),
                limits: SymbolicExecutionLimits::default(),
            };
            match lift_decoded_linear(&input, address) {
                Ok(lifted) => {
                    return Ok(SplicedBody {
                        bytes: lifted.replacement.compiled.bytes,
                        kind: "linear",
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
    let calls = BTreeMap::new();
    let input = DecodedCfgInput {
        cfg: &cfg,
        frame: &frame,
        exits: &exits,
        native_calls: &calls,
        memory: Some(pe),
        limits: SymbolicExecutionLimits::default(),
        target_limits: limits,
        max_expanded_blocks: 256,
    };
    lift_decoded_cfg(&input, address)
        .map(|lifted| SplicedBody {
            bytes: lifted.replacement.compiled.bytes,
            kind: "cfg",
        })
        .map_err(|error| format!("linear: {linear_error}; cfg: {error}"))
}
