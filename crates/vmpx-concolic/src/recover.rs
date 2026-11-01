//! Static side: recover the VM function's CFG and entry frame (as `cfglift` / diff-original do).

use std::collections::BTreeMap;

use vmpx_pe::{PeImage, Rva, Va};
use vmpx_vm::{
    BlockExit, EntryProfileRecoveryLimits, NativeFrameSource, NativeCallSelectorOutcome,
    RecoveryStatus, SymbolicExecutionLimits, VmCfg, VmCfgLimits, VmDiscoveryLimits, VmExitReport,
    analyze_vm_entry_at, analyze_vm_exit, recover_entry_stream, recover_vm_cfg,
};
use vmpx_vmlift::{DecodedCfgInput, recover_entry_frame};

/// Owned pieces of a [`DecodedCfgInput`].
pub struct Recovered {
    pub cfg: VmCfg,
    pub frame: Vec<NativeFrameSource>,
    pub top_offset: u64,
    pub exits: BTreeMap<Va, VmExitReport>,
    pub calls: BTreeMap<Va, NativeCallSelectorOutcome>,
    pub image_base: u64,
}

impl Recovered {
    #[must_use]
    pub fn input<'a>(&'a self, pe: &'a PeImage<'a>) -> DecodedCfgInput<'a> {
        DecodedCfgInput {
            cfg: &self.cfg,
            frame: &self.frame,
            top_offset: self.top_offset,
            image_base: self.image_base,
            exits: &self.exits,
            native_calls: &self.calls,
            memory: Some(pe),
            limits: SymbolicExecutionLimits::default(),
            target_limits: VmCfgLimits::default(),
            max_expanded_blocks: 256,
        }
    }
}

/// Recover the CFG for the VM entry stub at `rva` (the `push seed; call wrapper` stub).
///
/// # Errors
/// Any recovery stage failing.
pub fn recover(pe: &PeImage<'_>, rva: u32, instruction_limit: usize) -> Result<Recovered, String> {
    let candidate = analyze_vm_entry_at(pe, Rva(rva), VmDiscoveryLimits::default())
        .ok_or("no VM entry candidate at this RVA")?;
    let entry_limits = EntryProfileRecoveryLimits { instruction_limit, ..EntryProfileRecoveryLimits::default() };
    let limits = VmCfgLimits { entry: entry_limits, ..VmCfgLimits::default() };
    let cfg = recover_vm_cfg(pe, &candidate, limits).map_err(|e| format!("cfg: {e}"))?;
    let stream = recover_entry_stream(pe, &candidate, entry_limits).map_err(|e| format!("stream: {e}"))?;
    let (frame, top_offset) =
        recover_entry_frame(pe, candidate.rva, stream.profile.effect_roles.vsp, entry_limits.trace)
            .map_err(|e| format!("frame: {e}"))?;
    let mut exits: BTreeMap<Va, VmExitReport> = stream
        .decode
        .handler_recoveries
        .iter()
        .filter_map(|(h, r)| Some((*h, r.vm_exit.clone()?)))
        .collect();
    for block in cfg.blocks.values().filter(|b| b.exit == BlockExit::VmExit) {
        let handler = block.instructions.last().map_or(block.start.handler, |l| l.state_after.handler);
        if exits.contains_key(&handler) {
            continue;
        }
        let Some(r) = pe.optional.image_base.checked_rva_of(handler) else { continue };
        if let Ok(rep) = analyze_vm_exit(pe, r, stream.profile.effect_roles.vsp, entry_limits.trace)
            && rep.status == RecoveryStatus::Recovered
        {
            exits.insert(handler, rep);
        }
    }
    Ok(Recovered {
        cfg,
        frame,
        top_offset,
        exits,
        calls: BTreeMap::new(),
        image_base: pe.optional.image_base.0,
    })
}
