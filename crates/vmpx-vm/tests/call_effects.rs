pub use vmpx_vm::{BitWidth, Expr, ExprKind};

#[allow(dead_code)]
#[path = "../src/call_effects.rs"]
mod call_effects;

use call_effects::{
    FRESH_ALLOCATION_RETURN_INTRINSIC_PREFIX, NativeCallArgumentAlias,
    NativeCallArgumentDescriptor, NativeCallArgumentLocation, NativeCallEffectError,
    NativeCallEffectSummary, NativeCallEpochAllocator, NativeCallImportMatchKind,
    NativeCallImportResolution, NativeCallInvocationDescriptor, NativeCallMemoryDecision,
    NativeCallMemoryEffect, NativeCallResultEffect, NativeCallTargetDescriptor,
    NativeCallValueSource, apply_native_call_effect, fresh_allocation_return_intrinsic_name,
    resolve_native_call_import,
};
use vmpx_pe::{ImageBase, ImportModule, ImportSymbol, Rva};

fn target() -> NativeCallTargetDescriptor {
    NativeCallTargetDescriptor {
        source: NativeCallValueSource::VmStack {
            offset: 0,
            width: BitWidth::W64,
        },
        value: Expr::register(BitWidth::W64, "native_target"),
    }
}

fn argument(
    ordinal: u16,
    value: u64,
    alias: NativeCallArgumentAlias,
) -> NativeCallArgumentDescriptor {
    NativeCallArgumentDescriptor {
        ordinal,
        location: NativeCallArgumentLocation::Register {
            name: format!("arg{ordinal}"),
        },
        source: NativeCallValueSource::VmStack {
            offset: 8 * (i64::from(ordinal) + 1),
            width: BitWidth::W64,
        },
        value: Expr::constant(BitWidth::W64, value),
        alias,
    }
}

#[test]
fn opaque_calls_havoc_memory_and_have_distinct_returns() {
    let invocation = NativeCallInvocationDescriptor {
        target: target(),
        arguments: vec![argument(0, 7, NativeCallArgumentAlias::NonPointer)],
    };
    let summary = NativeCallEffectSummary {
        result: NativeCallResultEffect::Opaque {
            width: BitWidth::W64,
        },
        memory: NativeCallMemoryEffect::Opaque,
    };
    let mut epochs = NativeCallEpochAllocator::default();
    let first = apply_native_call_effect(&invocation, &summary, epochs.claim().unwrap()).unwrap();
    let second = apply_native_call_effect(&invocation, &summary, epochs.claim().unwrap()).unwrap();

    assert_ne!(first.epoch, second.epoch);
    assert_ne!(first.result, second.result);
    assert!(matches!(
        first.memory,
        NativeCallMemoryDecision::HavocTracked { .. }
    ));
    let ExprKind::Intrinsic { name, arguments } = first.result.kind else {
        panic!("opaque return must remain an intrinsic")
    };
    assert_eq!(name, "native_call_return_epoch_0");
    assert_eq!(arguments.len(), 2);
}

#[test]
fn fresh_allocation_preserves_existing_memory_only_under_explicit_contract() {
    let invocation = NativeCallInvocationDescriptor {
        target: target(),
        arguments: vec![
            argument(0, 0, NativeCallArgumentAlias::NonPointer),
            argument(1, 0x150, NativeCallArgumentAlias::NonPointer),
        ],
    };
    let summary = NativeCallEffectSummary {
        result: NativeCallResultEffect::FreshAllocation {
            pointer_width: BitWidth::W64,
            size_argument: 1,
            zero_initialized: true,
        },
        memory: NativeCallMemoryEffect::ProvenFreshAllocationOnly,
    };
    let mut epochs = NativeCallEpochAllocator::default();
    let applied = apply_native_call_effect(&invocation, &summary, epochs.claim().unwrap()).unwrap();

    let NativeCallMemoryDecision::PreserveTrackedAndCreateFreshAllocation {
        allocation,
        size,
        zero_initialized,
    } = applied.memory
    else {
        panic!("fresh-only summary must preserve old cells and add a fresh object")
    };
    assert_eq!(allocation, applied.result);
    assert_eq!(size, Expr::constant(BitWidth::W64, 0x150));
    assert!(zero_initialized);
    assert_eq!(
        fresh_allocation_return_intrinsic_name(applied.epoch),
        format!("{FRESH_ALLOCATION_RETURN_INTRINSIC_PREFIX}0")
    );
}

fn import(module: &str, iat_rva: u32, lookup_value: u64, name: &str) -> ImportModule {
    ImportModule {
        name: module.to_owned(),
        descriptor_rva: Rva(0),
        original_first_thunk: Rva(0),
        first_thunk: Rva(iat_rva),
        entries: vec![ImportSymbol {
            iat_rva: Rva(iat_rva),
            lookup_value,
            name: Some(name.to_owned()),
            hint: None,
            ordinal: None,
        }],
    }
}

#[test]
fn import_resolution_matches_lookup_rva_and_va_without_hiding_ambiguity() {
    let image_base = ImageBase(0x1_4000_0000);
    let imports = vec![
        import("module_a", 0x2000, 0x9000, "function_a"),
        import("module_b", 0x3000, 0x9000, "function_b"),
    ];

    let NativeCallImportResolution::Ambiguous { imports, .. } =
        resolve_native_call_import(&imports, image_base, 0x9000)
    else {
        panic!("duplicate lookup values must remain ambiguous")
    };
    assert_eq!(imports.len(), 2);
    assert!(
        imports
            .iter()
            .all(|import| { import.match_kinds == vec![NativeCallImportMatchKind::LookupValue] })
    );

    let imports = vec![import("module_a", 0x2000, 0x9000, "function_a")];
    let NativeCallImportResolution::Unique { import, .. } =
        resolve_native_call_import(&imports, image_base, 0x1_4000_2000)
    else {
        panic!("preferred-base IAT VA must resolve uniquely")
    };
    assert_eq!(import.name.as_deref(), Some("function_a"));
    assert_eq!(import.match_kinds, vec![NativeCallImportMatchKind::IatVa]);

    assert!(matches!(
        resolve_native_call_import(&imports, image_base, 0xDEAD),
        NativeCallImportResolution::NoMatch { .. }
    ));
}

#[test]
fn writable_argument_requires_proven_disjointness_to_preserve_memory() {
    let summary = NativeCallEffectSummary {
        result: NativeCallResultEffect::Opaque {
            width: BitWidth::W64,
        },
        memory: NativeCallMemoryEffect::ProvenWritesThroughArguments {
            writable_arguments: vec![0],
        },
    };
    let mut epochs = NativeCallEpochAllocator::default();
    let may_alias = NativeCallInvocationDescriptor {
        target: target(),
        arguments: vec![argument(
            0,
            0x1234,
            NativeCallArgumentAlias::MayAliasTrackedMemory,
        )],
    };
    let applied = apply_native_call_effect(&may_alias, &summary, epochs.claim().unwrap()).unwrap();
    assert!(matches!(
        applied.memory,
        NativeCallMemoryDecision::HavocTracked { .. }
    ));

    let disjoint = NativeCallInvocationDescriptor {
        target: target(),
        arguments: vec![argument(
            0,
            0x1234,
            NativeCallArgumentAlias::ProvenDisjointFromTrackedMemory,
        )],
    };
    let applied = apply_native_call_effect(&disjoint, &summary, epochs.claim().unwrap()).unwrap();
    assert_eq!(applied.memory, NativeCallMemoryDecision::PreserveTracked);
}

#[test]
fn malformed_summaries_and_descriptors_are_rejected() {
    let invocation = NativeCallInvocationDescriptor {
        target: target(),
        arguments: vec![
            argument(0, 1, NativeCallArgumentAlias::NonPointer),
            argument(0, 2, NativeCallArgumentAlias::NonPointer),
        ],
    };
    let summary = NativeCallEffectSummary {
        result: NativeCallResultEffect::Opaque {
            width: BitWidth::W64,
        },
        memory: NativeCallMemoryEffect::ProvenReadOnly,
    };
    let mut epochs = NativeCallEpochAllocator::default();
    assert_eq!(
        apply_native_call_effect(&invocation, &summary, epochs.claim().unwrap()),
        Err(NativeCallEffectError::DuplicateArgumentOrdinal { ordinal: 0 })
    );

    let missing_size = NativeCallInvocationDescriptor {
        target: target(),
        arguments: Vec::new(),
    };
    let summary = NativeCallEffectSummary {
        result: NativeCallResultEffect::FreshAllocation {
            pointer_width: BitWidth::W64,
            size_argument: 3,
            zero_initialized: false,
        },
        memory: NativeCallMemoryEffect::ProvenFreshAllocationOnly,
    };
    assert_eq!(
        apply_native_call_effect(&missing_size, &summary, epochs.claim().unwrap()),
        Err(NativeCallEffectError::MissingArgument { ordinal: 3 })
    );
}

#[test]
fn epoch_allocator_never_wraps() {
    let mut epochs = NativeCallEpochAllocator::new(u64::MAX);
    assert_eq!(epochs.claim(), Err(NativeCallEffectError::EpochExhausted));
    assert_eq!(epochs.next(), u64::MAX);
}
