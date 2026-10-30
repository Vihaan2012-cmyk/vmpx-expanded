//! Recognizes anti-debug / environment idioms in an `Expr`.
//!
//! A clean-environment value is a labeled assumption (`AssumesCleanEnvironment`),
//! never a proof: it holds only for a normal, non-debugged run. The loader-list
//! walk computes an ASLR module base and is classified `DynamicOnly` with no value.

use serde::Serialize;

use crate::{BinaryOp, BitWidth, Expr, ExprKind};

const PEB_GS_OFFSET: u64 = 0x60;
const PEB_LDR: u64 = 0x18;
const PEB_PROCESS_HEAP: u64 = 0x30;
const PEB_BEING_DEBUGGED: u64 = 0x2;
const PEB_NT_GLOBAL_FLAG: u64 = 0xBC;
const HEAP_FORCE_FLAGS: u64 = 0x74;
const LDR_MODULE_LISTS: [u64; 3] = [0x10, 0x20, 0x30];

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AntiDebugIdiom {
    PebBeingDebugged,
    PebNtGlobalFlag,
    ProcessHeapForceFlags,
    ModuleBaseLoaderWalk,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum IdiomAssumption {
    AssumesCleanEnvironment,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum IdiomClassification {
    CleanValue {
        value: u64,
        assumption: IdiomAssumption,
    },
    DynamicOnly,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct RecognizedIdiom {
    pub idiom: AntiDebugIdiom,
    pub classification: IdiomClassification,
}

impl RecognizedIdiom {
    #[must_use]
    pub const fn clean_value(&self) -> Option<u64> {
        match self.classification {
            IdiomClassification::CleanValue { value, .. } => Some(value),
            IdiomClassification::DynamicOnly => None,
        }
    }
}

const fn clean(idiom: AntiDebugIdiom) -> RecognizedIdiom {
    RecognizedIdiom {
        idiom,
        classification: IdiomClassification::CleanValue {
            value: 0,
            assumption: IdiomAssumption::AssumesCleanEnvironment,
        },
    }
}

/// Matches only an exact known shape; anything else yields `None`.
#[must_use]
pub fn recognize_anti_debug_idiom(expr: &Expr) -> Option<RecognizedIdiom> {
    let core = peel(expr);
    if is_module_base_walk(core) {
        return Some(RecognizedIdiom {
            idiom: AntiDebugIdiom::ModuleBaseLoaderWalk,
            classification: IdiomClassification::DynamicOnly,
        });
    }
    let (width, address) = memory_read(core)?;
    let (base, offset) = split_offset(address);
    if is_peb(base) {
        return match (width, offset) {
            (BitWidth::W8, PEB_BEING_DEBUGGED) => Some(clean(AntiDebugIdiom::PebBeingDebugged)),
            (BitWidth::W32, PEB_NT_GLOBAL_FLAG) => Some(clean(AntiDebugIdiom::PebNtGlobalFlag)),
            _ => None,
        };
    }
    if width == BitWidth::W32 && offset == HEAP_FORCE_FLAGS && is_process_heap(base) {
        return Some(clean(AntiDebugIdiom::ProcessHeapForceFlags));
    }
    None
}

/// Strips wrappers under which a clean zero stays zero (resizes, `& const`) and
/// a dynamic walk stays dynamic.
fn peel(expr: &Expr) -> &Expr {
    match &expr.kind {
        ExprKind::Resize { value, .. } => peel(value),
        ExprKind::Binary {
            op: BinaryOp::And,
            left,
            right,
        } => match (&left.kind, &right.kind) {
            (_, ExprKind::Const { .. }) => peel(left),
            (ExprKind::Const { .. }, _) => peel(right),
            _ => expr,
        },
        _ => expr,
    }
}

fn memory_read(expr: &Expr) -> Option<(BitWidth, &Expr)> {
    match &expr.kind {
        ExprKind::Load { address } => Some((expr.width, address)),
        ExprKind::Intrinsic { name, arguments } if name.starts_with("memory_load_epoch_") => {
            match arguments.as_slice() {
                [address] => Some((expr.width, address)),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Splits `base + const (+ const ...)` into its non-constant base and offset.
fn split_offset(address: &Expr) -> (&Expr, u64) {
    match &address.kind {
        ExprKind::Binary {
            op: BinaryOp::Add,
            left,
            right,
        } => match (&left.kind, &right.kind) {
            (_, ExprKind::Const { value }) => {
                let (base, offset) = split_offset(left);
                (base, offset.wrapping_add(*value))
            }
            (ExprKind::Const { value }, _) => {
                let (base, offset) = split_offset(right);
                (base, offset.wrapping_add(*value))
            }
            _ => (address, 0),
        },
        _ => (address, 0),
    }
}

fn is_peb(expr: &Expr) -> bool {
    expr.width == BitWidth::W64
        && matches!(
            &expr.kind,
            ExprKind::Intrinsic { name, arguments }
                if name.starts_with("x86_gs_load_")
                    && matches!(
                        arguments.as_slice(),
                        [Expr { kind: ExprKind::Const { value: PEB_GS_OFFSET }, .. }]
                    )
        )
}

/// A 64-bit read of `PEB + field`.
fn is_peb_field(expr: &Expr, field: u64) -> bool {
    memory_read(expr).is_some_and(|(width, address)| {
        let (base, offset) = split_offset(address);
        width == BitWidth::W64 && offset == field && is_peb(base)
    })
}

fn is_process_heap(expr: &Expr) -> bool {
    is_peb_field(expr, PEB_PROCESS_HEAP)
}

/// `PEB->Ldr->*ModuleList.Flink->DllBase`: a load through a list entry pointer
/// that was itself read from the `Ldr` list head.
fn is_module_base_walk(expr: &Expr) -> bool {
    let Some((_, address)) = memory_read(expr) else {
        return false;
    };
    let entry = split_offset(address).0;
    let Some((BitWidth::W64, head_address)) = memory_read(entry) else {
        return false;
    };
    let (ldr, head_offset) = split_offset(head_address);
    LDR_MODULE_LISTS.contains(&head_offset) && is_peb_field(ldr, PEB_LDR)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ResizeOp;

    fn constant(width: BitWidth, value: u64) -> Expr {
        Expr::constant(width, value)
    }

    fn peb() -> Expr {
        Expr::intrinsic(
            BitWidth::W64,
            "x86_gs_load_3",
            vec![constant(BitWidth::W64, 0x60)],
        )
    }

    fn add(base: Expr, offset: u64) -> Expr {
        Expr::binary(BinaryOp::Add, base, constant(BitWidth::W64, offset)).expect("add")
    }

    fn read(width: BitWidth, address: Expr) -> Expr {
        Expr::intrinsic(width, "memory_load_epoch_1", vec![address])
    }

    fn ldr_walk(list_offset: u64) -> Expr {
        let ldr = read(BitWidth::W64, add(peb(), PEB_LDR));
        let entry = read(BitWidth::W64, add(ldr, list_offset));
        read(BitWidth::W64, add(entry, 0x30))
    }

    #[test]
    fn being_debugged_is_a_labeled_clean_zero() {
        let expr = read(BitWidth::W8, add(peb(), 2));
        let found = recognize_anti_debug_idiom(&expr).expect("recognized");
        assert_eq!(found.idiom, AntiDebugIdiom::PebBeingDebugged);
        assert_eq!(
            found.classification,
            IdiomClassification::CleanValue {
                value: 0,
                assumption: IdiomAssumption::AssumesCleanEnvironment
            }
        );
        assert_eq!(found.clean_value(), Some(0));
    }

    #[test]
    fn being_debugged_through_load_and_zero_extend() {
        let load = Expr::load(BitWidth::W8, add(peb(), 2));
        let widened = Expr::resize(ResizeOp::ZeroExtend, BitWidth::W32, load).expect("zext");
        let found = recognize_anti_debug_idiom(&widened).expect("recognized");
        assert_eq!(found.idiom, AntiDebugIdiom::PebBeingDebugged);
    }

    #[test]
    fn nt_global_flag_and_masked_form() {
        let flag = read(BitWidth::W32, add(peb(), 0xBC));
        let found = recognize_anti_debug_idiom(&flag).expect("recognized");
        assert_eq!(found.idiom, AntiDebugIdiom::PebNtGlobalFlag);
        assert_eq!(found.clean_value(), Some(0));
        let masked = Expr::binary(BinaryOp::And, flag, constant(BitWidth::W32, 0x70)).expect("and");
        let found = recognize_anti_debug_idiom(&masked).expect("recognized");
        assert_eq!(found.idiom, AntiDebugIdiom::PebNtGlobalFlag);
    }

    #[test]
    fn heap_force_flags_is_a_clean_zero() {
        let heap = read(BitWidth::W64, add(peb(), 0x30));
        let flags = read(BitWidth::W32, add(heap, 0x74));
        let found = recognize_anti_debug_idiom(&flags).expect("recognized");
        assert_eq!(found.idiom, AntiDebugIdiom::ProcessHeapForceFlags);
        assert_eq!(found.clean_value(), Some(0));
    }

    #[test]
    fn loader_walk_is_dynamic_only_with_no_value() {
        for list in LDR_MODULE_LISTS {
            let found = recognize_anti_debug_idiom(&ldr_walk(list)).expect("recognized");
            assert_eq!(found.idiom, AntiDebugIdiom::ModuleBaseLoaderWalk);
            assert_eq!(found.classification, IdiomClassification::DynamicOnly);
            assert_eq!(found.clean_value(), None);
        }
    }

    #[test]
    fn near_misses_and_unrelated_exprs_are_not_recognized() {
        let wrong_gs = Expr::intrinsic(
            BitWidth::W64,
            "x86_gs_load_3",
            vec![constant(BitWidth::W64, 0x30)],
        );
        let wrong_heap_field = read(
            BitWidth::W32,
            add(read(BitWidth::W64, add(peb(), 0x38)), 0x74),
        );
        let wrong_list = {
            let ldr = read(BitWidth::W64, add(peb(), PEB_LDR));
            let entry = read(BitWidth::W64, add(ldr, 0x40));
            read(BitWidth::W64, add(entry, 0x30))
        };
        let shallow = read(
            BitWidth::W64,
            add(read(BitWidth::W64, add(peb(), PEB_LDR)), 0x10),
        );
        for expr in [
            read(BitWidth::W8, add(peb(), 3)),
            read(BitWidth::W32, add(peb(), 2)),
            read(BitWidth::W8, add(wrong_gs, 2)),
            read(BitWidth::W8, add(Expr::register(BitWidth::W64, "peb"), 2)),
            wrong_heap_field,
            wrong_list,
            shallow,
            constant(BitWidth::W64, 0),
            Expr::register(BitWidth::W8, "x"),
        ] {
            assert_eq!(recognize_anti_debug_idiom(&expr), None, "{expr:?}");
        }
    }
}
