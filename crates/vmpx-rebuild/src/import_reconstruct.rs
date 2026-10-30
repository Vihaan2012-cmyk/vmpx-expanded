//! Application import-set reconstruction: reduce a protector-era IAT view to
//! the slots with a proven direct native reference.

use std::collections::{BTreeMap, BTreeSet};

use vmpx_disasm::referenced_iat_slots;
use vmpx_pe::{ImportModule, ImportSymbol, PeImage, PeKind, Rva};

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

/// Reduce the image's parsed import table to the slots native (non-`.vmp`)
/// code provably references. VM-emulated imports are not visible statically.
#[must_use]
pub fn proven_native_imports(pe: &PeImage<'_>) -> ProvenImportSet {
    let bindings = bindings_from_modules(&pe.report().imports);
    reduce_to_proven_imports(&bindings, &referenced_iat_slots(pe))
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

/// A clean import directory built only from proven-referenced IAT slots.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CleanImportDirectory {
    pub proven: ProvenImportSet,
    /// `None` when no slot was proven: nothing is serialized and the caller
    /// must keep the "import set not proven" blocker.
    pub plan: Option<ImportReconstructionPlan>,
}

impl CleanImportDirectory {
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.plan.is_none()
    }

    /// True when every referenced slot was proven (none left unresolved).
    #[must_use]
    pub fn fully_resolved(&self) -> bool {
        self.proven.unresolved.is_empty()
    }
}

/// Build a clean import directory containing exactly the imports whose IAT
/// slot is in `proven_slots`. Unresolvable slots are reported in
/// `proven.unresolved`, never fabricated into the table. An empty result is
/// not an error. Completeness of `proven_slots` is the caller's burden.
///
/// # Errors
/// Propagates [`ImportTableError`] from serialization of a non-empty set.
pub fn reconstruct_clean_imports(
    kind: PeKind,
    table_rva: Rva,
    modules: &[ImportModule],
    proven_slots: &BTreeSet<Rva>,
) -> Result<CleanImportDirectory, ImportTableError> {
    let proven = reduce_to_proven_imports(&bindings_from_modules(modules), proven_slots);
    let plan = if proven.import_count() == 0 {
        None
    } else {
        Some(plan_proven_imports(kind, table_rva, &proven)?)
    };
    Ok(CleanImportDirectory { proven, plan })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::import_table::ImportTarget;

    fn sym(rva: u32, name: Option<&str>, ordinal: Option<u16>, hint: Option<u16>) -> ImportSymbol {
        ImportSymbol {
            iat_rva: Rva(rva),
            lookup_value: 0,
            name: name.map(str::to_owned),
            hint,
            ordinal,
        }
    }

    fn two_modules() -> Vec<ImportModule> {
        let module = |name: &str, entries| ImportModule {
            name: name.to_owned(),
            descriptor_rva: Rva(0),
            original_first_thunk: Rva(0),
            first_thunk: Rva(0),
            entries,
        };
        vec![
            module(
                "kernel32.dll",
                vec![
                    sym(0x2000, Some("ExitProcess"), None, Some(0x11)),
                    sym(0x2008, Some("Sleep"), None, Some(0x22)),
                    sym(0x2010, Some("GetTickCount"), None, Some(0x33)),
                ],
            ),
            module(
                "ws2_32.dll",
                vec![
                    sym(0x2020, None, Some(115), None),
                    sym(0x2028, Some("socket"), None, Some(0x44)),
                    sym(0x2030, None, Some(3), None),
                ],
            ),
        ]
    }

    fn u16_at(b: &[u8], at: usize) -> u16 {
        u16::from_le_bytes(b[at..at + 2].try_into().unwrap())
    }
    fn u32_at(b: &[u8], at: usize) -> u32 {
        u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
    }
    fn u64_at(b: &[u8], at: usize) -> u64 {
        u64::from_le_bytes(b[at..at + 8].try_into().unwrap())
    }
    fn cstr(b: &[u8], at: usize) -> String {
        let end = b[at..].iter().position(|&x| x == 0).unwrap();
        String::from_utf8(b[at..at + end].to_vec()).unwrap()
    }

    /// Independent byte-level walk of a PE32+ import blob based at `base`.
    /// Returns per module `(dll, [(ilt_value, iat_value, name_or_ordinal)])`.
    fn walk(bytes: &[u8], base: u32) -> Vec<(String, Vec<(u64, u64, Result<(u16, String), u16>)>)> {
        let off = |rva: u32| (rva - base) as usize;
        let mut out = Vec::new();
        let mut d = 0;
        loop {
            let desc = &bytes[d..d + 20];
            if desc.iter().all(|&x| x == 0) {
                break;
            }
            // TimeDateStamp and ForwarderChain must be zero.
            assert_eq!(u32_at(desc, 4), 0);
            assert_eq!(u32_at(desc, 8), 0);
            let ilt = off(u32_at(desc, 0));
            let iat = off(u32_at(desc, 16));
            assert_eq!(ilt % 8, 0);
            assert_eq!(iat % 8, 0);
            let dll = cstr(bytes, off(u32_at(desc, 12)));
            let mut entries = Vec::new();
            let mut i = 0;
            loop {
                let l = u64_at(bytes, ilt + 8 * i);
                let a = u64_at(bytes, iat + 8 * i);
                if l == 0 {
                    assert_eq!(a, 0, "IAT terminator");
                    break;
                }
                assert_eq!(l, a, "unbound IAT mirrors ILT");
                let id = if l >> 63 == 1 {
                    Err(u16::try_from(l & 0xFFFF).unwrap())
                } else {
                    let n = off(u32::try_from(l).unwrap());
                    assert_eq!(n % 2, 0, "hint/name 2-aligned");
                    Ok((u16_at(bytes, n), cstr(bytes, n + 2)))
                };
                entries.push((l, a, id));
                i += 1;
            }
            out.push((dll, entries));
            d += 20;
        }
        out
    }

    #[test]
    fn reconstruct_emits_exactly_proven_symbols_grouped_per_module() {
        let modules = two_modules();
        let proven = refs(&[0x2008, 0x2020, 0x2028]);
        let dir =
            reconstruct_clean_imports(PeKind::Pe32Plus, Rva(0x6000), &modules, &proven).unwrap();
        assert!(dir.fully_resolved());
        assert!(!dir.is_empty());
        let plan = dir.plan.as_ref().unwrap();
        let t = &plan.table;

        let parsed = walk(&t.bytes, 0x6000);
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].0, "kernel32.dll");
        assert_eq!(parsed[0].1.len(), 1);
        assert_eq!(parsed[0].1[0].2, Ok((0x22, "Sleep".to_owned())));
        assert_eq!(parsed[1].0, "ws2_32.dll");
        let ws: Vec<_> = parsed[1].1.iter().map(|e| e.2.clone()).collect();
        assert_eq!(ws, [Ok((0x44, "socket".to_owned())), Err(115)]);
        assert_eq!(parsed[1].1[1].0, 0x8000_0000_0000_0000 | 115);

        // Dropped symbols never appear anywhere in the blob.
        for dropped in [&b"ExitProcess"[..], b"GetTickCount"] {
            assert!(!t.bytes.windows(dropped.len()).any(|w| w == dropped));
        }

        // Directory ranges: descriptors incl. terminator; IAT incl. terminators.
        assert_eq!(t.import_directory.rva, Rva(0x6000));
        assert_eq!(t.import_directory.size, 3 * 20);
        assert_eq!(t.iat_directory.size, (2 + 3) * 8);
        let iat_off = (t.iat_directory.rva.0 - 0x6000) as usize;
        assert_eq!(u32_at(&t.bytes, 16), t.iat_directory.rva.0);
        assert_eq!(iat_off % 8, 0);

        // Remaps cover exactly the proven slots to in-table IAT slots.
        let remapped: Vec<_> = plan
            .slot_remaps
            .iter()
            .map(|r| r.source_iat_rva.0)
            .collect();
        assert_eq!(remapped, [0x2008, 0x2028, 0x2020]);
        for r in &plan.slot_remaps {
            let at = r.rebuilt_iat_rva.0;
            assert!(
                at >= t.iat_directory.rva.0 && at < t.iat_directory.rva.0 + t.iat_directory.size
            );
        }
    }

    #[test]
    fn reconstruct_empty_proven_set_yields_empty_table_without_error() {
        let modules = two_modules();
        let dir = reconstruct_clean_imports(PeKind::Pe32, Rva(0x6000), &modules, &BTreeSet::new())
            .unwrap();
        assert!(dir.is_empty());
        assert!(dir.plan.is_none());
        assert_eq!(dir.proven.import_count(), 0);
        assert_eq!(dir.proven.unreferenced_slots, 6);
        assert!(dir.fully_resolved());

        let none = reconstruct_clean_imports(PeKind::Pe32, Rva(0), &[], &BTreeSet::new()).unwrap();
        assert!(none.is_empty());
    }

    #[test]
    fn reconstruct_never_fabricates_unbound_slots() {
        let modules = two_modules();
        let dir = reconstruct_clean_imports(
            PeKind::Pe32Plus,
            Rva(0x6000),
            &modules,
            &refs(&[0x2000, 0x9000]),
        )
        .unwrap();
        assert!(!dir.fully_resolved());
        assert_eq!(dir.proven.unresolved[0].slot, Rva(0x9000));
        let parsed = walk(&dir.plan.unwrap().table.bytes, 0x6000);
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].1.len(), 1);
        assert_eq!(parsed[0].1[0].2, Ok((0x11, "ExitProcess".to_owned())));

        let only_unbound =
            reconstruct_clean_imports(PeKind::Pe32Plus, Rva(0x6000), &modules, &refs(&[0x9000]))
                .unwrap();
        assert!(only_unbound.is_empty());
        assert!(!only_unbound.fully_resolved());
    }

    #[test]
    fn reconstruct_pe32_uses_32bit_thunks_and_ordinal_flag() {
        let modules = two_modules();
        let dir = reconstruct_clean_imports(
            PeKind::Pe32,
            Rva(0x6000),
            &modules,
            &refs(&[0x2030, 0x2008]),
        )
        .unwrap();
        let t = &dir.plan.unwrap().table;
        // Two descriptors + terminator = 60 bytes, ILT aligned to 4.
        assert_eq!(t.import_directory.size, 60);
        assert_eq!(t.iat_directory.size, (2 + 2) * 4);
        let ilt = |d: usize| (u32_at(&t.bytes, d) - 0x6000) as usize;
        let iat = |d: usize| (u32_at(&t.bytes, d + 16) - 0x6000) as usize;
        // ws2_32 (descriptor 1): ordinal 3 only.
        assert_eq!(u32_at(&t.bytes, ilt(20)), 0x8000_0003);
        assert_eq!(u32_at(&t.bytes, ilt(20) + 4), 0);
        assert_eq!(u32_at(&t.bytes, iat(20)), 0x8000_0003);
        assert_eq!(u32_at(&t.bytes, iat(20) + 4), 0);
        // kernel32 (descriptor 0): Sleep by name.
        let n = (u32_at(&t.bytes, ilt(0)) - 0x6000) as usize;
        assert_eq!(u16_at(&t.bytes, n), 0x22);
        assert_eq!(cstr(&t.bytes, n + 2), "Sleep");
        assert_eq!(u32_at(&t.bytes, ilt(0) + 4), 0);
        // Terminating descriptor is all zero.
        assert!(t.bytes[40..60].iter().all(|&b| b == 0));
    }

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

    fn put(data: &mut [u8], at: usize, bytes: &[u8]) {
        data[at..at + bytes.len()].copy_from_slice(bytes);
    }

    /// PE32+: `.text` 0x1000 (`call [A]; call vmp; jmp [B]`), `.idata` 0x2000
    /// (mod.dll: Alpha 0x2050, Beta 0x2058, Gamma 0x2060), `.vmp0` 0x3000
    /// (`call [Gamma]`). Gamma is referenced only from `.vmp0`.
    fn import_pe() -> Vec<u8> {
        let mut d = vec![0_u8; 0x800];
        put(&mut d, 0, &0x5A4D_u16.to_le_bytes());
        put(&mut d, 0x3C, &0x80_u32.to_le_bytes());
        put(&mut d, 0x80, &0x4550_u32.to_le_bytes());
        put(&mut d, 0x84, &0x8664_u16.to_le_bytes());
        put(&mut d, 0x86, &3_u16.to_le_bytes());
        put(&mut d, 0x94, &0xF0_u16.to_le_bytes());
        put(&mut d, 0x96, &0x2022_u16.to_le_bytes());
        let o = 0x98;
        put(&mut d, o, &0x20B_u16.to_le_bytes());
        put(&mut d, o + 16, &0x1000_u32.to_le_bytes());
        put(&mut d, o + 24, &0x1_4000_0000_u64.to_le_bytes());
        put(&mut d, o + 32, &0x1000_u32.to_le_bytes());
        put(&mut d, o + 36, &0x200_u32.to_le_bytes());
        put(&mut d, o + 56, &0x4000_u32.to_le_bytes());
        put(&mut d, o + 60, &0x200_u32.to_le_bytes());
        put(&mut d, o + 68, &3_u16.to_le_bytes());
        put(&mut d, o + 108, &16_u32.to_le_bytes());
        put(&mut d, o + 112 + 8, &0x2000_u32.to_le_bytes());
        put(&mut d, o + 112 + 12, &40_u32.to_le_bytes());
        for (i, (name, rva, raw, flags)) in [
            (&b".text"[..], 0x1000_u32, 0x200_u32, 0x6000_0020_u32),
            (b".idata", 0x2000, 0x400, 0xC000_0040),
            (b".vmp0", 0x3000, 0x600, 0x6000_0020),
        ]
        .into_iter()
        .enumerate()
        {
            let at = 0x188 + i * 40;
            put(&mut d, at, name);
            put(&mut d, at + 8, &0x200_u32.to_le_bytes());
            put(&mut d, at + 12, &rva.to_le_bytes());
            put(&mut d, at + 16, &0x200_u32.to_le_bytes());
            put(&mut d, at + 20, &raw.to_le_bytes());
            put(&mut d, at + 36, &flags.to_le_bytes());
        }
        let text = [
            &[0xFF, 0x15][..],
            &0x104A_u32.to_le_bytes(),
            &[0xE8],
            &0x1FF5_u32.to_le_bytes(),
            &[0xFF, 0x25],
            &0x1047_u32.to_le_bytes(),
        ]
        .concat();
        put(&mut d, 0x200, &text);
        let idata = 0x400;
        put(&mut d, idata, &0x2030_u32.to_le_bytes());
        put(&mut d, idata + 12, &0x20C0_u32.to_le_bytes());
        put(&mut d, idata + 16, &0x2050_u32.to_le_bytes());
        for (i, name) in ["Alpha", "Beta", "Gamma"].into_iter().enumerate() {
            let hint_rva = 0x2080 + 0x10 * u64::try_from(i).unwrap();
            let at = idata + 0x80 + 0x10 * i;
            put(&mut d, at + 2, name.as_bytes());
            put(&mut d, idata + 0x30 + 8 * i, &hint_rva.to_le_bytes());
            put(&mut d, idata + 0x50 + 8 * i, &hint_rva.to_le_bytes());
        }
        put(&mut d, idata + 0xC0, b"mod.dll");
        put(&mut d, 0x600, &[0xFF, 0x15]);
        put(&mut d, 0x602, &0xFFFF_F05A_u32.to_le_bytes());
        d
    }

    #[test]
    fn native_references_prove_imports_and_vmp_references_do_not() {
        let data = import_pe();
        let pe = PeImage::parse(&data).unwrap();
        assert_eq!(pe.report().imports[0].entries.len(), 3);
        assert_eq!(referenced_iat_slots(&pe), refs(&[0x2050, 0x2058]));

        let set = proven_native_imports(&pe);
        assert!(set.unresolved.is_empty());
        assert_eq!(set.unreferenced_slots, 1);
        assert_eq!(set.modules.len(), 1);
        assert_eq!(set.modules[0].name, "mod.dll");
        let got: Vec<_> = set.modules[0]
            .imports
            .iter()
            .map(|i| (i.slot.0, i.symbol.clone()))
            .collect();
        assert_eq!(
            got,
            [
                (0x2050, SymbolRef::Name("Alpha".into())),
                (0x2058, SymbolRef::Name("Beta".into())),
            ]
        );
    }

    /// `VMPX_IAT_SAMPLE=<pe> cargo test -p vmpx-rebuild -- --ignored --nocapture`
    #[test]
    #[ignore = "needs VMPX_IAT_SAMPLE"]
    fn sample_native_import_coverage() {
        let path = std::env::var("VMPX_IAT_SAMPLE").unwrap();
        let data = std::fs::read(path).unwrap();
        let pe = PeImage::parse(&data).unwrap();
        let scan = vmpx_disasm::scan_iat_references(&pe, vmpx_disasm::CfgLimits::default());
        let set = proven_native_imports(&pe);
        eprintln!(
            "slots={} referenced={} roots={} exhaustive={} proven={} unresolved={} modules={}",
            scan.total_slots,
            scan.referenced.len(),
            scan.roots_walked,
            scan.exhaustive,
            set.import_count(),
            set.unresolved.len(),
            set.modules.len()
        );
    }
}
