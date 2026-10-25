use crate::{
    BitWidth, Expr, ExprKind, NativeCallEffectSummary, NativeCallImportDescriptor,
    NativeCallMemoryEffect, NativeCallResultEffect,
};

const FIXED_ALLOCATION_MOVEABLE_FLAG: u64 = 0x0002;
const FIXED_ALLOCATION_ZERO_INIT_FLAG: u64 = 0x0040;

fn normalized_module(module: &str) -> String {
    let lowercase = module.to_ascii_lowercase();
    lowercase
        .strip_suffix(".dll")
        .unwrap_or(&lowercase)
        .to_owned()
}

fn constant_argument(arguments: &[Expr], ordinal: usize) -> Option<u64> {
    let argument = arguments.get(ordinal)?;
    let ExprKind::Const { value } = argument.kind else {
        return None;
    };
    Some(value)
}

/// Proves that a LocalAlloc/GlobalAlloc result is a directly usable pointer.
///
/// The Win32 `UINT` flags parameter observes the low 32 bits. MOVEABLE changes
/// the result into a handle which must be locked before dereferencing, so an
/// unknown or moveable flags value cannot receive the fresh-pointer contract.
fn fixed_allocation_zero_initialized(arguments: &[Expr]) -> Option<bool> {
    let flags = constant_argument(arguments, 0)? & u64::from(u32::MAX);
    if flags & FIXED_ALLOCATION_MOVEABLE_FLAG != 0 {
        return None;
    }
    Some(flags & FIXED_ALLOCATION_ZERO_INIT_FLAG != 0)
}

/// Returns a trusted Windows API effect contract for an unambiguously
/// resolved import. The registry is deliberately narrow: an unknown module,
/// name, ordinal, or signature remains opaque.
#[must_use]
pub fn summarize_known_import(
    import: &NativeCallImportDescriptor,
    arguments: &[Expr],
) -> Option<NativeCallEffectSummary> {
    if import.ordinal.is_some() {
        return None;
    }
    let name = import.name.as_deref()?;
    let module = normalized_module(&import.module);
    match (module.as_str(), name) {
        ("kernel32" | "kernelbase", "LocalAlloc" | "GlobalAlloc") if arguments.len() == 2 => {
            let zero_initialized = fixed_allocation_zero_initialized(arguments)?;
            Some(NativeCallEffectSummary {
                result: NativeCallResultEffect::FreshAllocation {
                    pointer_width: BitWidth::W64,
                    size_argument: 1,
                    zero_initialized,
                },
                memory: NativeCallMemoryEffect::ProvenFreshAllocationOnly,
            })
        }
        ("kernel32" | "kernelbase", "HeapAlloc") if arguments.len() == 3 => {
            let zero_initialized =
                constant_argument(arguments, 1).is_some_and(|flags| flags & 0x8 != 0);
            Some(NativeCallEffectSummary {
                result: NativeCallResultEffect::FreshAllocation {
                    pointer_width: BitWidth::W64,
                    size_argument: 2,
                    zero_initialized,
                },
                memory: NativeCallMemoryEffect::ProvenFreshAllocationOnly,
            })
        }
        ("ole32" | "combase", "CoTaskMemAlloc") if arguments.len() == 1 => {
            Some(NativeCallEffectSummary {
                result: NativeCallResultEffect::FreshAllocation {
                    pointer_width: BitWidth::W64,
                    size_argument: 0,
                    zero_initialized: false,
                },
                memory: NativeCallMemoryEffect::ProvenFreshAllocationOnly,
            })
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use vmpx_pe::Rva;

    use super::*;
    use crate::{NativeCallImportMatchKind, NativeCallResultEffect};

    fn imported(module: &str, name: &str) -> NativeCallImportDescriptor {
        NativeCallImportDescriptor {
            module: module.to_owned(),
            iat_rva: Rva(0x2000),
            lookup_value: 0x3000,
            name: Some(name.to_owned()),
            ordinal: None,
            match_kinds: vec![NativeCallImportMatchKind::LookupValue],
        }
    }

    #[test]
    fn allocation_contracts_require_exact_module_name_and_arity() {
        let args = [
            Expr::constant(BitWidth::W64, 0x40),
            Expr::constant(BitWidth::W64, 0x150),
        ];
        let summary = summarize_known_import(&imported("KERNEL32.DLL", "LocalAlloc"), &args)
            .expect("known allocation import");
        assert!(matches!(
            summary.result,
            NativeCallResultEffect::FreshAllocation {
                size_argument: 1,
                zero_initialized: true,
                ..
            }
        ));
        assert!(summarize_known_import(&imported("other.dll", "LocalAlloc"), &args).is_none());
        assert!(summarize_known_import(&imported("kernel32", "LocalAlloc"), &args[..1]).is_none());
    }

    #[test]
    fn local_and_global_allocation_contracts_require_proven_fixed_flags() {
        for name in ["LocalAlloc", "GlobalAlloc"] {
            let fixed = [
                Expr::constant(BitWidth::W64, 0),
                Expr::constant(BitWidth::W64, 0x150),
            ];
            assert!(summarize_known_import(&imported("kernel32", name), &fixed).is_some());

            let fixed_zeroed = [
                Expr::constant(BitWidth::W64, 0x40),
                Expr::constant(BitWidth::W64, 0x150),
            ];
            let summary = summarize_known_import(&imported("kernel32", name), &fixed_zeroed)
                .expect("fixed zero-initialized allocation is a direct pointer");
            assert!(matches!(
                summary.result,
                NativeCallResultEffect::FreshAllocation {
                    zero_initialized: true,
                    ..
                }
            ));

            for flags in [0x2, 0x42] {
                let moveable = [
                    Expr::constant(BitWidth::W64, flags),
                    Expr::constant(BitWidth::W64, 0x150),
                ];
                assert!(summarize_known_import(&imported("kernel32", name), &moveable).is_none());
            }

            let unknown = [
                Expr::register(BitWidth::W64, "allocation_flags"),
                Expr::constant(BitWidth::W64, 0x150),
            ];
            assert!(summarize_known_import(&imported("kernel32", name), &unknown).is_none());
        }
    }
}
