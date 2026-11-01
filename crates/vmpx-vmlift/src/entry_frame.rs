//! Initial VM-stack contents of a virtualized entry, recovered by stepping the
//! native entry stub and its call target.
//!
//! Recovery is the CFG walker's, so the lift adapter seeds symbolic execution
//! with the same proven frame. It fails closed on anything it cannot model.

use iced_x86::Register;
use vmpx_disasm::DirectJumpTraceLimits;
use vmpx_pe::{PeImage, Rva};
use vmpx_vm::{
    BinaryOp, BitWidth, Expr, NativeFrameSource, SymbolicVmInitialState,
};

/// Proven contents of the VM stack at VM start, top first, with the byte offset
/// from the captured VSP up to the top cell (nonzero for a `lea` capture).
/// Delegates to the CFG walker's recovery so both stages see one frame.
///
/// # Errors
///
/// Fails on nested calls, unknown stack writes, stack-pointer arithmetic, or a
/// trace that never reaches the VSP capture.
pub fn recover_entry_frame(
    pe: &PeImage<'_>,
    entry: Rva,
    vsp: Register,
    limits: DirectJumpTraceLimits,
) -> Result<(Vec<NativeFrameSource>, u64), String> {
    vmpx_vm::recover_entry_frame(pe, entry, vsp, limits).map(|(cells, top, _)| (cells, top))
}

fn symbolic_value(source: &NativeFrameSource) -> Option<Expr> {
    match source {
        NativeFrameSource::Register { register } => {
            Some(Expr::register(BitWidth::W64, register.symbolic_name()))
        }
        NativeFrameSource::Rflags => Some(Expr::register(BitWidth::W64, "native_rflags_initial")),
        NativeFrameSource::Constant { value } | NativeFrameSource::EntryImmediate { value, .. } => {
            Some(Expr::constant(BitWidth::W64, *value))
        }
        NativeFrameSource::ReturnAddress { .. } => Some(Expr::register(
            BitWidth::W64,
            "native_return_address_initial",
        )),
        // Preferred-base frame: the relocation delta is 0, so a relocated
        // immediate is its on-disk value.
        NativeFrameSource::RelocatedConstant { value } => Some(Expr::constant(BitWidth::W64, *value)),
        NativeFrameSource::Unknown { .. } => None,
    }
}

/// Initial symbolic state for a frame; `usize` counts the cells with a proven
/// value (an unknown cell stays absent so a read of it fails closed).
/// `top_offset` is the VSP-to-top-cell offset from [`recover_entry_frame`];
/// cell `i` lands at `vsp_initial + top_offset + i*8` (modular W64 add).
#[must_use]
pub fn frame_initial_state(
    cells: &[NativeFrameSource],
    top_offset: u64,
) -> (SymbolicVmInitialState, usize) {
    let mut initial = SymbolicVmInitialState::default();
    let base = Expr::register(BitWidth::W64, "vsp_initial");
    let mut known = 0;
    for (index, cell) in cells.iter().enumerate() {
        let Some(value) = symbolic_value(cell) else {
            continue;
        };
        let offset = top_offset.wrapping_add((index as u64).wrapping_mul(8));
        let Ok(address) = Expr::binary(
            BinaryOp::Add,
            base.clone(),
            Expr::constant(BitWidth::W64, offset),
        ) else {
            continue;
        };
        initial.memory_values.push((address, value));
        known += 1;
    }
    (initial, known)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vmpx_vm::X64NativeRegister;

    #[test]
    fn frame_offset_places_cells_above_vsp() {
        // The offset flows into the frame: cell i lands at vsp_initial+0x28+i*8.
        let cells = vec![
            NativeFrameSource::Register {
                register: X64NativeRegister::R13,
            },
            NativeFrameSource::Register {
                register: X64NativeRegister::Rcx,
            },
        ];
        let (state, known) = frame_initial_state(&cells, 0x28);
        assert_eq!(known, 2);
        let addresses: Vec<_> = state
            .memory_values
            .iter()
            .map(|(address, _)| format!("{address:?}"))
            .collect();
        let want = |offset: u64| {
            format!(
                "{:?}",
                Expr::binary(
                    BinaryOp::Add,
                    Expr::register(BitWidth::W64, "vsp_initial"),
                    Expr::constant(BitWidth::W64, offset),
                )
                .unwrap()
            )
        };
        assert!(addresses.contains(&want(0x28)), "missing VSP+0x28: {addresses:?}");
        assert!(addresses.contains(&want(0x30)), "missing VSP+0x30: {addresses:?}");
    }
}
