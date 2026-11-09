//! Nested VM entries the CFG lift can follow inline.
//!
//! The CFG records that a VM exit returns into another entry of the same VM but
//! not that entry's native frame, which composing its state needs. A resolver
//! supplies it, and the lift cross-checks it against the CFG's nested block.

use vmpx_pe::{PeImage, Va};
use vmpx_vm::{
    EntryProfileRecoveryLimits, ExprKind, NativeFrameSource, nested_entry_stub, recover_entry_stream,
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
        let base = self.pe.optional.image_base;
        // A nested entry's return address is the entry stub's VA, which the
        // protector may emit as modulo-image arithmetic rather than a constant.
        // Resolve the single proven image target exactly as the CFG's own
        // nested-entry follow does, so a non-constant does not refuse the entry.
        let simplified = return_address.simplify();
        let rva = if let ExprKind::Const { value } = simplified.kind {
            base.checked_rva_of(Va(value))?
        } else {
            let report = vmpx_vm::recover_modulo_image_targets(
                self.pe,
                return_address,
                vmpx_vm::ModuloImageTargetLimits::default(),
            )
            .ok()?;
            if !report.complete || report.targets.len() != 1 {
                return None;
            }
            report.targets[0].rva?
        };
        let (frame_entry, candidate) = nested_entry_stub(self.pe, rva)?;
        let stream = recover_entry_stream(self.pe, &candidate, self.limits).ok()?;
        let (frame, top_offset) = recover_entry_frame(
            self.pe,
            frame_entry,
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
