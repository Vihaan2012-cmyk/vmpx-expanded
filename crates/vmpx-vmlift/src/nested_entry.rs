//! Nested VM entries the CFG lift can follow inline.
//!
//! The CFG records that a VM exit returns into another entry of the same VM but
//! not that entry's native frame, which composing its state needs. A resolver
//! supplies it, and the lift cross-checks it against the CFG's nested block.

use vmpx_pe::{PeImage, Va};
use vmpx_vm::{
    EntryProfileRecoveryLimits, ExprKind, NativeFrameSource, VmDiscoveryLimits,
    analyze_vm_entry_at, recover_entry_stream,
};

use crate::recover_entry_frame;

/// A nested entry's recovered frame.
#[derive(Clone, Debug)]
pub struct NestedEntry {
    /// Address of the entry's first opcode; must equal the CFG nested block's.
    pub bytecode_start: Va,
    /// Initial VM stack of the entry, top first.
    pub frame: Vec<NativeFrameSource>,
    /// VSP-to-top-cell byte offset from [`recover_entry_frame`].
    pub top_offset: u64,
    /// Native address the entry's pushed return-address cell holds.
    pub continuation: Option<Va>,
}

pub trait NestedEntryResolver {
    /// The entry a VM exit returns into, proven from its return address.
    fn resolve(&self, return_address: &vmpx_vm::Expr) -> Option<NestedEntry>;
}

/// Resolves a constant return address that is a VM entry stub of the image.
pub struct PeNestedEntries<'a, 'b> {
    pub pe: &'a PeImage<'b>,
    pub limits: EntryProfileRecoveryLimits,
}

impl NestedEntryResolver for PeNestedEntries<'_, '_> {
    fn resolve(&self, return_address: &vmpx_vm::Expr) -> Option<NestedEntry> {
        let ExprKind::Const { value } = return_address.simplify().kind else {
            return None;
        };
        let base = self.pe.optional.image_base;
        let candidate = analyze_vm_entry_at(
            self.pe,
            base.checked_rva_of(Va(value))?,
            VmDiscoveryLimits::default(),
        )?;
        let stream = recover_entry_stream(self.pe, &candidate, self.limits).ok()?;
        let (frame, top_offset) = recover_entry_frame(
            self.pe,
            candidate.rva,
            stream.profile.effect_roles.vsp,
            self.limits.trace,
        )
        .ok()?;
        let continuation = frame.iter().find_map(|cell| match cell {
            NativeFrameSource::ReturnAddress { continuation, .. } => {
                base.checked_add_rva(*continuation)
            }
            _ => None,
        });
        Some(NestedEntry {
            bytecode_start: stream.bootstrap.initial_state.vip,
            frame,
            top_offset,
            continuation,
        })
    }
}
