use std::collections::BTreeSet;

use serde::Serialize;
use thiserror::Error;
use vmpx_pe::{ImageBase, ImportModule, Rva};

use crate::{BitWidth, Expr};

pub const OPAQUE_CALL_RETURN_INTRINSIC_PREFIX: &str = "native_call_return_epoch_";
pub const FRESH_ALLOCATION_RETURN_INTRINSIC_PREFIX: &str = "fresh_allocation_return_epoch_";

#[must_use]
pub fn opaque_call_return_intrinsic_name(epoch: NativeCallEpoch) -> String {
    format!("{OPAQUE_CALL_RETURN_INTRINSIC_PREFIX}{}", epoch.get())
}

#[must_use]
pub fn fresh_allocation_return_intrinsic_name(epoch: NativeCallEpoch) -> String {
    format!("{FRESH_ALLOCATION_RETURN_INTRINSIC_PREFIX}{}", epoch.get())
}

/// Monotonic identity assigned to one dynamic native-call event.
///
/// The identity is deliberately part of opaque and allocation-result
/// expressions. Two calls with identical targets and arguments are not assumed
/// to return the same value.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
pub struct NativeCallEpoch(u64);

impl NativeCallEpoch {
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Checked allocator for distinct native-call epochs.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct NativeCallEpochAllocator {
    next: u64,
}

impl NativeCallEpochAllocator {
    #[must_use]
    pub const fn new(next: u64) -> Self {
        Self { next }
    }

    #[must_use]
    pub const fn next(self) -> u64 {
        self.next
    }

    /// Claims one epoch without permitting the counter to wrap.
    ///
    /// # Errors
    ///
    /// Returns [`NativeCallEffectError::EpochExhausted`] when another distinct
    /// epoch cannot be represented.
    pub fn claim(&mut self) -> Result<NativeCallEpoch, NativeCallEffectError> {
        let following = self
            .next
            .checked_add(1)
            .ok_or(NativeCallEffectError::EpochExhausted)?;
        let epoch = NativeCallEpoch(self.next);
        self.next = following;
        Ok(epoch)
    }
}

impl Default for NativeCallEpochAllocator {
    fn default() -> Self {
        Self::new(0)
    }
}

/// Proven origin of a native target or argument value.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NativeCallValueSource {
    VmStack { offset: i64, width: BitWidth },
    Context { offset: i64, width: BitWidth },
    Constant,
    Unknown,
}

/// ABI location populated for one native argument.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NativeCallArgumentLocation {
    Register { name: String },
    Stack { offset: u32 },
}

/// Alias relationship proven for an argument value.
///
/// `ProvenDisjointFromTrackedMemory` is the only state that permits a summary
/// which writes through that argument to preserve the existing tracked-memory
/// snapshot. Merely classifying a value as a non-pointer is intentionally not
/// sufficient for a writable argument contract.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeCallArgumentAlias {
    NonPointer,
    ProvenDisjointFromTrackedMemory,
    MayAliasTrackedMemory,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct NativeCallTargetDescriptor {
    pub source: NativeCallValueSource,
    pub value: Expr,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct NativeCallArgumentDescriptor {
    /// Zero-based logical argument ordinal, independent of ABI placement.
    pub ordinal: u16,
    pub location: NativeCallArgumentLocation,
    pub source: NativeCallValueSource,
    pub value: Expr,
    pub alias: NativeCallArgumentAlias,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct NativeCallInvocationDescriptor {
    pub target: NativeCallTargetDescriptor,
    pub arguments: Vec<NativeCallArgumentDescriptor>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeCallImportMatchKind {
    LookupValue,
    IatRva,
    IatVa,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct NativeCallImportDescriptor {
    pub module: String,
    pub iat_rva: Rva,
    pub lookup_value: u64,
    pub name: Option<String>,
    pub ordinal: Option<u16>,
    pub match_kinds: Vec<NativeCallImportMatchKind>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum NativeCallImportResolution {
    NoMatch {
        target: u64,
    },
    Unique {
        target: u64,
        import: NativeCallImportDescriptor,
    },
    Ambiguous {
        target: u64,
        imports: Vec<NativeCallImportDescriptor>,
    },
}

/// Matches a concrete call-target value against parsed PE imports.
///
/// Raw mapped-file analysis may observe an import lookup value where a loaded
/// process would contain the resolved IAT pointer. For that reason lookup
/// values, IAT RVAs, and preferred-base IAT VAs are all considered, and every
/// matching import remains visible. This routine never chooses between
/// duplicate lookup values or duplicate import descriptors.
#[must_use]
pub fn resolve_native_call_import(
    imports: &[ImportModule],
    image_base: ImageBase,
    target: u64,
) -> NativeCallImportResolution {
    let mut matches = Vec::new();
    for module in imports {
        for symbol in &module.entries {
            let mut match_kinds = Vec::new();
            if symbol.lookup_value == target {
                match_kinds.push(NativeCallImportMatchKind::LookupValue);
            }
            if u64::from(symbol.iat_rva.get()) == target {
                match_kinds.push(NativeCallImportMatchKind::IatRva);
            }
            if image_base
                .checked_add_rva(symbol.iat_rva)
                .is_some_and(|iat_va| iat_va.get() == target)
            {
                match_kinds.push(NativeCallImportMatchKind::IatVa);
            }
            if !match_kinds.is_empty() {
                matches.push(NativeCallImportDescriptor {
                    module: module.name.clone(),
                    iat_rva: symbol.iat_rva,
                    lookup_value: symbol.lookup_value,
                    name: symbol.name.clone(),
                    ordinal: symbol.ordinal,
                    match_kinds,
                });
            }
        }
    }
    match matches.as_slice() {
        [] => NativeCallImportResolution::NoMatch { target },
        [import] => NativeCallImportResolution::Unique {
            target,
            import: import.clone(),
        },
        _ => NativeCallImportResolution::Ambiguous {
            target,
            imports: matches,
        },
    }
}

impl NativeCallInvocationDescriptor {
    fn validate(&self) -> Result<(), NativeCallEffectError> {
        if self.target.value.width != BitWidth::W64 {
            return Err(NativeCallEffectError::InvalidTargetWidth {
                width: self.target.value.width,
            });
        }
        let mut ordinals = BTreeSet::new();
        for argument in &self.arguments {
            if !ordinals.insert(argument.ordinal) {
                return Err(NativeCallEffectError::DuplicateArgumentOrdinal {
                    ordinal: argument.ordinal,
                });
            }
        }
        Ok(())
    }

    fn argument(&self, ordinal: u16) -> Option<&NativeCallArgumentDescriptor> {
        self.arguments
            .iter()
            .find(|argument| argument.ordinal == ordinal)
    }
}

/// Result contract supplied by a trusted native-call summarizer.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NativeCallResultEffect {
    /// Arbitrary value of the declared width.
    Opaque { width: BitWidth },
    /// Pointer to an allocation proven fresh with respect to all existing
    /// tracked addresses.
    FreshAllocation {
        pointer_width: BitWidth,
        size_argument: u16,
        zero_initialized: bool,
    },
}

/// Memory contract supplied by a trusted native-call summarizer.
///
/// Unknown calls must use `Opaque`. The other variants are proof obligations,
/// not heuristics inferred from a function name.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NativeCallMemoryEffect {
    Opaque,
    ProvenReadOnly,
    ProvenFreshAllocationOnly,
    ProvenWritesThroughArguments { writable_arguments: Vec<u16> },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct NativeCallEffectSummary {
    pub result: NativeCallResultEffect,
    pub memory: NativeCallMemoryEffect,
}

/// Conservative action the symbolic machine must take for tracked memory.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NativeCallMemoryDecision {
    PreserveTracked,
    PreserveTrackedAndCreateFreshAllocation {
        allocation: Expr,
        size: Expr,
        zero_initialized: bool,
    },
    HavocTracked {
        reason: String,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct AppliedNativeCallEffect {
    pub epoch: NativeCallEpoch,
    pub result: Expr,
    pub memory: NativeCallMemoryDecision,
}

#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum NativeCallEffectError {
    #[error("native-call epoch space is exhausted")]
    EpochExhausted,

    #[error("native-call target has width {width:?}, expected W64")]
    InvalidTargetWidth { width: BitWidth },

    #[error("native-call argument ordinal {ordinal} is duplicated")]
    DuplicateArgumentOrdinal { ordinal: u16 },

    #[error("native-call summary references missing argument {ordinal}")]
    MissingArgument { ordinal: u16 },

    #[error("fresh-allocation memory effect requires a fresh-allocation result")]
    FreshAllocationResultRequired,
}

fn call_expression_arguments(invocation: &NativeCallInvocationDescriptor) -> Vec<Expr> {
    let mut arguments = Vec::with_capacity(invocation.arguments.len().saturating_add(1));
    arguments.push(invocation.target.value.clone());
    let mut ordered = invocation.arguments.iter().collect::<Vec<_>>();
    ordered.sort_by_key(|argument| argument.ordinal);
    arguments.extend(ordered.into_iter().map(|argument| argument.value.clone()));
    arguments
}

fn result_expression(
    invocation: &NativeCallInvocationDescriptor,
    result: &NativeCallResultEffect,
    epoch: NativeCallEpoch,
) -> Result<Expr, NativeCallEffectError> {
    match result {
        NativeCallResultEffect::Opaque { width } => Ok(Expr::intrinsic(
            *width,
            opaque_call_return_intrinsic_name(epoch),
            call_expression_arguments(invocation),
        )),
        NativeCallResultEffect::FreshAllocation {
            pointer_width,
            size_argument,
            ..
        } => {
            let size = invocation.argument(*size_argument).ok_or(
                NativeCallEffectError::MissingArgument {
                    ordinal: *size_argument,
                },
            )?;
            Ok(Expr::intrinsic(
                *pointer_width,
                fresh_allocation_return_intrinsic_name(epoch),
                vec![invocation.target.value.clone(), size.value.clone()],
            ))
        }
    }
}

fn memory_decision(
    invocation: &NativeCallInvocationDescriptor,
    summary: &NativeCallEffectSummary,
    result: &Expr,
) -> Result<NativeCallMemoryDecision, NativeCallEffectError> {
    match &summary.memory {
        NativeCallMemoryEffect::Opaque => Ok(NativeCallMemoryDecision::HavocTracked {
            reason: "native call has no proven memory-effect summary".to_owned(),
        }),
        NativeCallMemoryEffect::ProvenReadOnly => Ok(NativeCallMemoryDecision::PreserveTracked),
        NativeCallMemoryEffect::ProvenFreshAllocationOnly => {
            let NativeCallResultEffect::FreshAllocation {
                size_argument,
                zero_initialized,
                ..
            } = &summary.result
            else {
                return Err(NativeCallEffectError::FreshAllocationResultRequired);
            };
            let size = invocation.argument(*size_argument).ok_or(
                NativeCallEffectError::MissingArgument {
                    ordinal: *size_argument,
                },
            )?;
            Ok(
                NativeCallMemoryDecision::PreserveTrackedAndCreateFreshAllocation {
                    allocation: result.clone(),
                    size: size.value.clone(),
                    zero_initialized: *zero_initialized,
                },
            )
        }
        NativeCallMemoryEffect::ProvenWritesThroughArguments { writable_arguments } => {
            for ordinal in writable_arguments {
                let argument = invocation
                    .argument(*ordinal)
                    .ok_or(NativeCallEffectError::MissingArgument { ordinal: *ordinal })?;
                if argument.alias != NativeCallArgumentAlias::ProvenDisjointFromTrackedMemory {
                    return Ok(NativeCallMemoryDecision::HavocTracked {
                        reason: format!(
                            "writable native argument {ordinal} may alias tracked memory"
                        ),
                    });
                }
            }
            Ok(NativeCallMemoryDecision::PreserveTracked)
        }
    }
}

/// Applies a trusted call summary to one already-proven invocation descriptor.
///
/// This function deliberately does not identify APIs or infer summaries. An
/// unknown target must be paired with an opaque memory effect, which produces
/// a mandatory tracked-memory havoc decision.
///
/// # Errors
///
/// Rejects malformed descriptors and inconsistent or incomplete summaries.
pub fn apply_native_call_effect(
    invocation: &NativeCallInvocationDescriptor,
    summary: &NativeCallEffectSummary,
    epoch: NativeCallEpoch,
) -> Result<AppliedNativeCallEffect, NativeCallEffectError> {
    invocation.validate()?;
    let result = result_expression(invocation, &summary.result, epoch)?;
    let memory = memory_decision(invocation, summary, &result)?;
    Ok(AppliedNativeCallEffect {
        epoch,
        result,
        memory,
    })
}
