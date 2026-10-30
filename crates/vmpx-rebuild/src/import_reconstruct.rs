//! Application import-set reconstruction: reduce a protector-era IAT view to
//! the slots with a proven direct native reference.

use std::collections::{BTreeMap, BTreeSet};

use vmpx_pe::{ImportModule, ImportSymbol, PeKind, Rva};

use crate::import_table::{ImportReconstructionPlan, ImportTableError, plan_import_reconstruction};

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum SymbolRef {
    Name(String),
    Ordinal(u16),
}

/// One IAT slot's claimed identity. `symbol` is `None` when the source
/// evidence is ambiguous (e.g. both or neither of name/ordinal present).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SlotBinding {
    pub slot: Rva,
    pub module: String,
    pub symbol: Option<SymbolRef>,
    pub hint: Option<u16>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProvenImport {
    pub slot: Rva,
    pub symbol: SymbolRef,
    pub hint: Option<u16>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProvenModule {
    pub name: String,
    pub imports: Vec<ProvenImport>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UnresolvedReason {
    /// Referenced slot has no binding at all.
    NoBinding,
    /// Binding exists but its symbol identity is ambiguous.
    AmbiguousSymbol,
    /// The same slot was bound to differing identities.
    ConflictingBindings,
    /// Zero slot RVA or empty module/symbol name.
    InvalidIdentity,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct UnresolvedSlot {
    pub slot: Rva,
    pub reason: UnresolvedReason,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ProvenImportSet {
    pub modules: Vec<ProvenModule>,
    /// Referenced slots that could not be proven, sorted by slot.
    pub unresolved: Vec<UnresolvedSlot>,
    /// Distinct bound slots with no proven reference (dropped).
    pub unreferenced_slots: usize,
}

impl ProvenImportSet {
    #[must_use]
    pub fn import_count(&self) -> usize {
        self.modules.iter().map(|module| module.imports.len()).sum()
    }
}

/// Flatten parsed import modules into slot bindings.
#[must_use]
pub fn bindings_from_modules(modules: &[ImportModule]) -> Vec<SlotBinding> {
    modules
        .iter()
        .flat_map(|module| {
            module
                .entries
                .iter()
                .map(|entry| binding_from_entry(&module.name, entry))
        })
        .collect()
}

fn binding_from_entry(module: &str, entry: &ImportSymbol) -> SlotBinding {
    let symbol = match (&entry.name, entry.ordinal) {
        (Some(name), None) => Some(SymbolRef::Name(name.clone())),
        (None, Some(ordinal)) => Some(SymbolRef::Ordinal(ordinal)),
        _ => None,
    };
    SlotBinding {
        slot: entry.iat_rva,
        module: module.to_owned(),
        symbol,
        hint: entry.hint,
    }
}

fn is_valid(binding: &SlotBinding, symbol: &SymbolRef) -> bool {
    binding.slot.0 != 0
        && !binding.module.is_empty()
        && !matches!(symbol, SymbolRef::Name(name) if name.is_empty())
}

enum Resolved {
    Proven(SlotBinding, SymbolRef),
    Unresolved(UnresolvedReason),
}

/// Reduce `bindings` to the slots in `referenced`. Soundness: a slot becomes
/// an application import only when it is referenced and carries exactly one
/// unambiguous identity; every other referenced slot is reported.
#[must_use]
pub fn reduce_to_proven_imports(
    bindings: &[SlotBinding],
    referenced: &BTreeSet<Rva>,
) -> ProvenImportSet {
    let mut by_slot: BTreeMap<Rva, Resolved> = BTreeMap::new();
    let mut bound_slots = BTreeSet::new();
    for binding in bindings {
        bound_slots.insert(binding.slot);
        let incoming = match &binding.symbol {
            None => Resolved::Unresolved(UnresolvedReason::AmbiguousSymbol),
            Some(symbol) if !is_valid(binding, symbol) => {
                Resolved::Unresolved(UnresolvedReason::InvalidIdentity)
            }
            Some(symbol) => Resolved::Proven(binding.clone(), symbol.clone()),
        };
        let merged = match by_slot.remove(&binding.slot) {
            None => incoming,
            Some(Resolved::Proven(prev, prev_symbol)) => match incoming {
                Resolved::Proven(next, next_symbol)
                    if prev.module == next.module
                        && prev_symbol == next_symbol
                        && prev.hint == next.hint =>
                {
                    Resolved::Proven(prev, prev_symbol)
                }
                _ => Resolved::Unresolved(UnresolvedReason::ConflictingBindings),
            },
            Some(Resolved::Unresolved(_)) => {
                Resolved::Unresolved(UnresolvedReason::ConflictingBindings)
            }
        };
        by_slot.insert(binding.slot, merged);
    }

    let mut unresolved = Vec::new();
    // Lowercase module key -> (smallest original spelling, imports).
    let mut grouped: BTreeMap<String, (String, Vec<ProvenImport>)> = BTreeMap::new();
    for &slot in referenced {
        match by_slot.remove(&slot) {
            None => unresolved.push(UnresolvedSlot {
                slot,
                reason: UnresolvedReason::NoBinding,
            }),
            Some(Resolved::Unresolved(reason)) => {
                unresolved.push(UnresolvedSlot { slot, reason });
            }
            Some(Resolved::Proven(binding, symbol)) => {
                let entry = grouped
                    .entry(binding.module.to_ascii_lowercase())
                    .or_insert_with(|| (binding.module.clone(), Vec::new()));
                if binding.module < entry.0 {
                    entry.0.clone_from(&binding.module);
                }
                entry.1.push(ProvenImport {
                    slot,
                    symbol,
                    hint: binding.hint,
                });
            }
        }
    }

    // Named before ordinals (variant order), then by name/ordinal, then slot.
    let modules = grouped
        .into_values()
        .map(|(name, mut imports)| {
            imports.sort_by(|a, b| (&a.symbol, a.slot).cmp(&(&b.symbol, b.slot)));
            ProvenModule { name, imports }
        })
        .collect();

    let unreferenced_slots = bound_slots.difference(referenced).count();
    ProvenImportSet {
        modules,
        unresolved,
        unreferenced_slots,
    }
}

/// Serialize a proven set into a conventional import directory plan. Each
/// source slot maps to exactly one rebuilt slot, so duplicate symbols at
/// distinct slots stay distinct.
///
/// # Errors
/// Propagates [`ImportTableError`]; an empty set yields `EmptyTable`.
pub fn plan_proven_imports(
    kind: PeKind,
    table_rva: Rva,
    set: &ProvenImportSet,
) -> Result<ImportReconstructionPlan, ImportTableError> {
    let modules: Vec<ImportModule> = set
        .modules
        .iter()
        .map(|module| ImportModule {
            name: module.name.clone(),
            descriptor_rva: Rva(0),
            original_first_thunk: Rva(0),
            first_thunk: Rva(0),
            entries: module
                .imports
                .iter()
                .map(|import| {
                    let (name, ordinal) = match &import.symbol {
                        SymbolRef::Name(name) => (Some(name.clone()), None),
                        SymbolRef::Ordinal(ordinal) => (None, Some(*ordinal)),
                    };
                    ImportSymbol {
                        iat_rva: import.slot,
                        lookup_value: 0,
                        name,
                        hint: import.hint,
                        ordinal,
                    }
                })
                .collect(),
        })
        .collect();
    plan_import_reconstruction(kind, table_rva, &modules)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::import_table::ImportTarget;

    fn named(slot: u32, module: &str, name: &str) -> SlotBinding {
        SlotBinding {
            slot: Rva(slot),
            module: module.to_owned(),
            symbol: Some(SymbolRef::Name(name.to_owned())),
            hint: Some(1),
        }
    }

    fn ordinal(slot: u32, module: &str, ordinal: u16) -> SlotBinding {
        SlotBinding {
            slot: Rva(slot),
            module: module.to_owned(),
            symbol: Some(SymbolRef::Ordinal(ordinal)),
            hint: None,
        }
    }

    fn refs(slots: &[u32]) -> BTreeSet<Rva> {
        slots.iter().copied().map(Rva).collect()
    }

    fn symbols(module: &ProvenModule) -> Vec<&SymbolRef> {
        module.imports.iter().map(|i| &i.symbol).collect()
    }

    #[test]
    fn empty_inputs_give_empty_set_and_empty_table_error() {
        let set = reduce_to_proven_imports(&[], &BTreeSet::new());
        assert_eq!(set, ProvenImportSet::default());
        assert_eq!(set.import_count(), 0);
        assert_eq!(
            plan_proven_imports(PeKind::Pe32, Rva(0x1000), &set).unwrap_err(),
            ImportTableError::EmptyTable
        );
    }

    #[test]
    fn unreferenced_slots_are_dropped_and_counted() {
        let bindings = [
            named(0x2000, "kernel32.dll", "ExitProcess"),
            named(0x2004, "kernel32.dll", "Sleep"),
            named(0x2008, "user32.dll", "MessageBoxA"),
        ];
        let set = reduce_to_proven_imports(&bindings, &refs(&[0x2004]));
        assert_eq!(set.modules.len(), 1);
        assert_eq!(set.modules[0].name, "kernel32.dll");
        assert_eq!(symbols(&set.modules[0]), [&SymbolRef::Name("Sleep".into())]);
        assert_eq!(set.unreferenced_slots, 2);
        assert!(set.unresolved.is_empty());
    }

    #[test]
    fn referenced_slot_without_binding_is_reported_not_guessed() {
        let bindings = [named(0x2000, "kernel32.dll", "Sleep")];
        let set = reduce_to_proven_imports(&bindings, &refs(&[0x2000, 0x3000, 0x2800]));
        assert_eq!(set.import_count(), 1);
        assert_eq!(
            set.unresolved,
            [
                UnresolvedSlot {
                    slot: Rva(0x2800),
                    reason: UnresolvedReason::NoBinding
                },
                UnresolvedSlot {
                    slot: Rva(0x3000),
                    reason: UnresolvedReason::NoBinding
                },
            ]
        );
    }

    #[test]
    fn ambiguous_and_invalid_identities_are_unresolved() {
        let bindings = [
            SlotBinding {
                slot: Rva(0x2000),
                module: "a.dll".into(),
                symbol: None,
                hint: None,
            },
            named(0x2004, "", "X"),
            named(0x2008, "a.dll", ""),
            named(0, "a.dll", "Z"),
        ];
        let set = reduce_to_proven_imports(&bindings, &refs(&[0x2000, 0x2004, 0x2008, 0]));
        assert!(set.modules.is_empty());
        let reasons: Vec<_> = set
            .unresolved
            .iter()
            .map(|u| (u.slot.0, u.reason))
            .collect();
        assert_eq!(
            reasons,
            [
                (0, UnresolvedReason::InvalidIdentity),
                (0x2000, UnresolvedReason::AmbiguousSymbol),
                (0x2004, UnresolvedReason::InvalidIdentity),
                (0x2008, UnresolvedReason::InvalidIdentity),
            ]
        );
    }

    #[test]
    fn conflicting_bindings_for_one_slot_are_unresolved_but_identical_repeats_collapse() {
        let bindings = [
            named(0x2000, "a.dll", "F"),
            named(0x2000, "a.dll", "G"),
            named(0x2004, "a.dll", "H"),
            named(0x2004, "A.DLL", "H"),
            named(0x2008, "a.dll", "K"),
            named(0x2008, "a.dll", "K"),
        ];
        let set = reduce_to_proven_imports(&bindings, &refs(&[0x2000, 0x2004, 0x2008]));
        assert_eq!(
            set.unresolved
                .iter()
                .map(|u| (u.slot.0, u.reason))
                .collect::<Vec<_>>(),
            [
                (0x2000, UnresolvedReason::ConflictingBindings),
                (0x2004, UnresolvedReason::ConflictingBindings),
            ]
        );
        assert_eq!(set.import_count(), 1);
        assert_eq!(set.modules[0].imports[0].slot, Rva(0x2008));
        assert_eq!(set.unreferenced_slots, 0);
    }

    #[test]
    fn duplicate_symbols_at_distinct_slots_stay_distinct() {
        let bindings = [
            named(0x2010, "kernel32.dll", "Sleep"),
            named(0x2000, "kernel32.dll", "Sleep"),
        ];
        let set = reduce_to_proven_imports(&bindings, &refs(&[0x2000, 0x2010]));
        let slots: Vec<_> = set.modules[0].imports.iter().map(|i| i.slot.0).collect();
        assert_eq!(slots, [0x2000, 0x2010]);

        let plan = plan_proven_imports(PeKind::Pe32Plus, Rva(0x5000), &set).unwrap();
        assert_eq!(plan.slot_remaps.len(), 2);
        assert_eq!(plan.slot_remaps[0].source_iat_rva, Rva(0x2000));
        assert_eq!(plan.slot_remaps[1].source_iat_rva, Rva(0x2010));
        assert_ne!(
            plan.slot_remaps[0].rebuilt_iat_rva,
            plan.slot_remaps[1].rebuilt_iat_rva
        );
    }

    #[test]
    fn ordinal_and_named_coexist_names_first() {
        let bindings = [
            ordinal(0x2000, "ws2_32.dll", 115),
            named(0x2004, "ws2_32.dll", "socket"),
            ordinal(0x2008, "ws2_32.dll", 3),
            named(0x200C, "ws2_32.dll", "bind"),
        ];
        let set = reduce_to_proven_imports(&bindings, &refs(&[0x2000, 0x2004, 0x2008, 0x200C]));
        assert_eq!(
            symbols(&set.modules[0]),
            [
                &SymbolRef::Name("bind".into()),
                &SymbolRef::Name("socket".into()),
                &SymbolRef::Ordinal(3),
                &SymbolRef::Ordinal(115),
            ]
        );
        let plan = plan_proven_imports(PeKind::Pe32, Rva(0x5000), &set).unwrap();
        assert_eq!(plan.slot_remaps[2].target, ImportTarget::Ordinal(3));
        assert!(matches!(
            &plan.slot_remaps[0].target,
            ImportTarget::Name { name, .. } if name == "bind"
        ));
    }

    #[test]
    fn module_grouping_is_case_insensitive_and_ordering_is_deterministic() {
        let a = [
            named(0x2000, "USER32.dll", "B"),
            named(0x2004, "kernel32.dll", "Z"),
            named(0x2008, "user32.dll", "A"),
            named(0x200C, "KERNEL32.DLL", "Y"),
        ];
        let mut b = a.clone();
        b.reverse();
        let all = refs(&[0x2000, 0x2004, 0x2008, 0x200C]);
        let set_a = reduce_to_proven_imports(&a, &all);
        let set_b = reduce_to_proven_imports(&b, &all);
        assert_eq!(set_a, set_b);
        let names: Vec<_> = set_a.modules.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(names, ["KERNEL32.DLL", "USER32.dll"]);
        assert_eq!(
            symbols(&set_a.modules[0]),
            [&SymbolRef::Name("Y".into()), &SymbolRef::Name("Z".into())]
        );
        assert_eq!(
            symbols(&set_a.modules[1]),
            [&SymbolRef::Name("A".into()), &SymbolRef::Name("B".into())]
        );
    }

    #[test]
    fn bindings_from_modules_flags_ambiguous_entries() {
        let entry = |rva: u32, name: Option<&str>, ordinal: Option<u16>| ImportSymbol {
            iat_rva: Rva(rva),
            lookup_value: 0,
            name: name.map(str::to_owned),
            hint: None,
            ordinal,
        };
        let module = ImportModule {
            name: "m.dll".into(),
            descriptor_rva: Rva(0),
            original_first_thunk: Rva(0),
            first_thunk: Rva(0),
            entries: vec![
                entry(0x2000, Some("F"), None),
                entry(0x2004, None, Some(7)),
                entry(0x2008, Some("G"), Some(8)),
                entry(0x200C, None, None),
            ],
        };
        let bindings = bindings_from_modules(&[module]);
        let set = reduce_to_proven_imports(&bindings, &refs(&[0x2000, 0x2004, 0x2008, 0x200C]));
        assert_eq!(set.import_count(), 2);
        assert_eq!(set.unresolved.len(), 2);
        assert!(
            set.unresolved
                .iter()
                .all(|u| u.reason == UnresolvedReason::AmbiguousSymbol)
        );
    }

    #[test]
    fn plan_serializes_only_proven_imports() {
        let bindings = [
            named(0x2000, "kernel32.dll", "ExitProcess"),
            named(0x2004, "kernel32.dll", "Sleep"),
            named(0x2008, "user32.dll", "MessageBoxA"),
        ];
        let set = reduce_to_proven_imports(&bindings, &refs(&[0x2000, 0x2008]));
        let plan = plan_proven_imports(PeKind::Pe32Plus, Rva(0x6000), &set).unwrap();
        assert_eq!(plan.table.modules.len(), 2);
        assert_eq!(plan.slot_remaps.len(), 2);
        assert_eq!(plan.slot_remaps[0].module, "kernel32.dll");
        assert_eq!(plan.slot_remaps[1].module, "user32.dll");
        assert_eq!(plan.slot_remaps[1].source_iat_rva, Rva(0x2008));
    }
}
