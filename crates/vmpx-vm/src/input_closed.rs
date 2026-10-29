//! Soundness predicate: is a native exit a pure function of the inputs a
//! replacement will actually provide?

use std::collections::BTreeSet;

use crate::{Expr, NativeExitState};

/// Intrinsic families that are pure functions of their arguments. Everything
/// else (memory epochs/snapshots, native-call results, fresh allocations,
/// timestamp counter, unknown names) is opaque machine state and never closed.
fn is_pure_intrinsic(name: &str) -> bool {
    name.starts_with("vm_flags_") || name.starts_with("x86_unsigned_")
}

/// True iff `expr` depends only on constants, registers named in `allowed`, and
/// pure intrinsics over such terms. Any memory load or opaque intrinsic makes
/// it not closed. Conservative: unknown shapes are never closed.
#[must_use]
pub fn is_expr_input_closed(expr: &Expr, allowed: &BTreeSet<String>) -> bool {
    use crate::ExprKind;
    let mut pending = vec![expr];
    while let Some(current) = pending.pop() {
        match &current.kind {
            ExprKind::Const { .. } => {}
            ExprKind::Register { name } => {
                if !allowed.contains(name) {
                    return false;
                }
            }
            ExprKind::Load { .. } => return false,
            ExprKind::Intrinsic { name, arguments } => {
                if !is_pure_intrinsic(name) {
                    return false;
                }
                pending.extend(arguments);
            }
            ExprKind::Unary { value, .. }
            | ExprKind::Extract { value, .. }
            | ExprKind::Resize { value, .. } => pending.push(value),
            ExprKind::Binary { left, right, .. }
            | ExprKind::Concat {
                high: left,
                low: right,
            } => {
                pending.push(left);
                pending.push(right);
            }
        }
    }
    true
}

/// True iff every value the exit carries (each register, `rflags`, the return
/// address) is closed over `allowed`. `allowed` must be the full set of symbols
/// the replacement will bind, never a subset inferred from observed use.
#[must_use]
pub fn is_input_closed(exit: &NativeExitState, allowed: &BTreeSet<String>) -> bool {
    exit.registers
        .values()
        .chain([&exit.rflags, &exit.return_address])
        .all(|value| is_expr_input_closed(value, allowed))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::{BinaryOp, BitWidth, X64NativeRegister};

    fn w64(name: &str) -> Expr {
        Expr::register(BitWidth::W64, name)
    }

    fn allowed() -> BTreeSet<String> {
        X64NativeRegister::ALL
            .iter()
            .map(|r| r.symbolic_name().to_owned())
            .collect()
    }

    fn exit_with(rax: Expr) -> NativeExitState {
        NativeExitState {
            schema_version: 1,
            registers: BTreeMap::from([(X64NativeRegister::Rax, rax)]),
            rflags: Expr::constant(BitWidth::W64, 0x202),
            return_address: Expr::constant(BitWidth::W64, 0x1_4000_1000),
            frame_size: 136,
        }
    }

    #[test]
    fn is_input_closed_constants_and_live_in_initials_are_closed() {
        let sum = Expr::binary(
            BinaryOp::Add,
            w64("native_rcx_initial"),
            Expr::constant(BitWidth::W64, 5),
        )
        .expect("add");
        assert!(is_input_closed(&exit_with(sum), &allowed()));
        assert!(is_input_closed(
            &exit_with(Expr::constant(BitWidth::W64, 1)),
            &allowed()
        ));
    }

    #[test]
    fn is_input_closed_return_only_exit_is_closed() {
        let mut exit = exit_with(Expr::constant(BitWidth::W64, 0));
        exit.registers.clear();
        assert!(is_input_closed(&exit, &BTreeSet::new()));
    }

    #[test]
    fn is_input_closed_unlisted_symbol_is_not_closed() {
        let sum =
            Expr::binary(BinaryOp::Add, w64("native_rcx_initial"), w64("exit_vsp")).expect("add");
        assert!(!is_input_closed(&exit_with(sum), &allowed()));
        // A listed symbol is only closed when actually allowed.
        assert!(!is_input_closed(
            &exit_with(w64("native_rcx_initial")),
            &BTreeSet::new()
        ));
    }

    #[test]
    fn is_input_closed_opaque_intrinsics_and_loads_are_not_closed() {
        for opaque in [
            Expr::intrinsic(BitWidth::W64, "memory_snapshot_epoch_3", vec![]),
            Expr::intrinsic(BitWidth::W64, "fresh_allocation_return_epoch_1", vec![]),
            Expr::intrinsic(BitWidth::W64, "read_timestamp_counter_0", vec![]),
            Expr::intrinsic(BitWidth::W64, "something_unknown", vec![]),
            Expr::load(BitWidth::W64, w64("native_rcx_initial")),
        ] {
            assert!(!is_input_closed(&exit_with(opaque), &allowed()));
        }
    }

    #[test]
    fn is_input_closed_pure_intrinsic_closed_only_if_arguments_are() {
        let ok = Expr::intrinsic(
            BitWidth::W64,
            "vm_flags_add_w64",
            vec![w64("native_rax_initial")],
        );
        assert!(is_input_closed(&exit_with(ok), &allowed()));
        let bad = Expr::intrinsic(
            BitWidth::W64,
            "vm_flags_add_w64",
            vec![Expr::intrinsic(BitWidth::W8, "memory_load_epoch_2", vec![])],
        );
        assert!(!is_input_closed(&exit_with(bad), &allowed()));
    }

    #[test]
    fn is_input_closed_table_checksum_reads_memory_so_is_not_pure() {
        let checksum = Expr::intrinsic(
            BitWidth::W32,
            "table_checksum_low_byte_xor_recurrence_w32",
            vec![w64("native_rax_initial")],
        );
        assert!(!is_input_closed(&exit_with(checksum), &allowed()));
    }

    #[test]
    fn is_input_closed_rflags_and_return_address_are_checked() {
        let mut exit = exit_with(Expr::constant(BitWidth::W64, 1));
        exit.rflags = Expr::intrinsic(BitWidth::W64, "native_call_flags_epoch_1", vec![]);
        assert!(!is_input_closed(&exit, &allowed()));
        let mut exit = exit_with(Expr::constant(BitWidth::W64, 1));
        exit.return_address = w64("unknown_return");
        assert!(!is_input_closed(&exit, &allowed()));
    }
}
