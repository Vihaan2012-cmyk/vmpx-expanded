//! Pure glue between the clean pipeline and the soundness cores. Every helper
//! here only ever keeps a blocker when in doubt.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;
use vmpx_disasm::{DanglingKind, DanglingRef};
use vmpx_emulate::{ObservedImport, SymbolOrOrdinal};
use vmpx_pe::{ImportModule, PeKind, Rva};
use vmpx_rebuild::{
    ProvenImportSet, SlotBinding, SymbolRef, bindings_from_modules, plan_proven_imports,
    reduce_to_proven_imports,
};
use vmpx_vm::{NeutralizeAction, NeutralizeOptions, RecoveredGuard, neutralize};

use crate::{
    coverage_gate::{CoverageDecision, InventoryItem},
    vm_inventory::VmInventoryEntry,
};

pub(crate) const OEP_BLOCKER: &str = "original module entry point is not recovered";
pub(crate) const IMPORT_BLOCKER: &str =
    "application import set is not proven independently reconstructed";
pub(crate) const REINTEGRATION_BLOCKER: &str =
    "native code reintegration has not produced a devirtualized replacement image";
pub(crate) const VMP_BLOCKER: &str = "VMProtect sections remain referenced by unresolved paths";

/// Discoveries that could not make it into the inventory as concrete entries.
#[derive(Clone, Debug, Default)]
pub(crate) struct InventoryLoss {
    /// The executable-stub scan failed, so the inventory may be incomplete.
    pub scan_failure: Option<String>,
    /// Export-reachable entries that could not be re-analyzed and were dropped.
    pub unanalyzed_exports: Vec<Rva>,
}

/// Gate inventory plus the count of discoveries known to be dropped. Nothing
/// lossy is silent: a scan failure is an unresolved item, a dropped export is
/// counted.
pub(crate) fn build_inventory_items(
    inventory: &[VmInventoryEntry],
    discovered: impl IntoIterator<Item = Rva>,
    loss: &InventoryLoss,
) -> (Vec<InventoryItem>, usize) {
    let rvas: BTreeSet<Rva> = inventory
        .iter()
        .map(|entry| entry.rva)
        .chain(discovered)
        .collect();
    let mut items: Vec<InventoryItem> = rvas.into_iter().map(InventoryItem::Resolved).collect();
    if let Some(why) = &loss.scan_failure {
        items.push(InventoryItem::Unresolved(why.clone()));
    }
    (items, loss.unanalyzed_exports.len())
}

/// Records `stub` as covered only when its redirect succeeded.
pub(crate) fn mark_covered<T, E>(covered: &mut BTreeSet<Rva>, stub: Rva, redirect: &Result<T, E>) {
    if redirect.is_ok() {
        covered.insert(stub);
    }
}

/// A group entry is covered only if its redirect landed AND the group's family
/// was proven complete: otherwise the dispatcher falls back to the original
/// VM stub for unmatched inputs.
pub(crate) fn group_covered(
    covered: &mut BTreeSet<Rva>,
    stub: Rva,
    redirected: bool,
    family_complete: bool,
) {
    if redirected && family_complete {
        covered.insert(stub);
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct DanglingSummary {
    pub targets: Vec<Rva>,
    pub has_unresolved: bool,
}

pub(crate) fn summarize_dangling(refs: &[DanglingRef]) -> DanglingSummary {
    let mut summary = DanglingSummary::default();
    for reference in refs {
        match reference.target {
            Some(target) => summary.targets.push(target),
            None => summary.has_unresolved = true,
        }
        if reference.kind == DanglingKind::Unresolved {
            summary.has_unresolved = true;
        }
    }
    summary
}

#[derive(Clone, Debug, Serialize)]
pub struct CleanCoverageReport {
    pub can_clear_oep: bool,
    pub can_clear_vmp_referenced: bool,
    pub covered_entries: Vec<Rva>,
    pub dangling_refs: usize,
    pub has_unresolved_dangling: bool,
    pub dropped_entries: usize,
    pub reasons: Vec<String>,
}

/// Blockers for a protected input: each is dropped only when its proof holds.
/// The `.vmp` blocker additionally needs a produced replacement image and no
/// unresolved dangling reference.
pub(crate) fn coverage_blockers(
    decision: &CoverageDecision,
    has_unresolved_dangling: bool,
    reintegrated: bool,
) -> Vec<String> {
    let mut blockers = Vec::new();
    if !decision.can_clear_oep_blocker {
        blockers.push(OEP_BLOCKER.to_owned());
    }
    if !reintegrated {
        blockers.push(REINTEGRATION_BLOCKER.to_owned());
    }
    if !(decision.can_clear_vmp_referenced_blocker && !has_unresolved_dangling && reintegrated) {
        blockers.push(VMP_BLOCKER.to_owned());
    }
    blockers
}

/// Synthetic IAT base for observed-tier reconstruction. Import protection
/// strips the application IAT, so observed `(module,symbol)` bindings are laid
/// out at deterministic synthetic slots. EXPERIMENTAL / Observed-tier only.
const OBSERVED_IAT_BASE: u32 = 0x1000;

/// IAT slots with a proven reference.
///
/// * SOUND path (`observed == None`): always empty. A partial native scan
///   cannot prove completeness — protector-resident code also uses the table —
///   so no reduced set is ever claimed as the full application import set.
/// * EXPERIMENTAL path (`observed == Some`): the synthetic slots the dynamic
///   detonation's resolved imports occupy. Every one is Observed-tier and never
///   feeds the sound blocker gate.
pub(crate) fn derive_referenced_slots(observed: Option<&[ObservedImport]>) -> BTreeSet<Rva> {
    observed
        .map(|imports| observed_slot_bindings(imports).iter().map(|b| b.slot).collect())
        .unwrap_or_default()
}

/// Deduplicated `(module, symbol)` observed bindings, each at a deterministic
/// synthetic slot. Observed-tier only; the real application IAT is stripped.
fn observed_slot_bindings(imports: &[ObservedImport]) -> Vec<SlotBinding> {
    let mut seen: BTreeSet<(String, String)> = BTreeSet::new();
    let mut out = Vec::new();
    for imp in imports {
        let (symbol, key) = match &imp.symbol {
            SymbolOrOrdinal::Name(n) => (SymbolRef::Name(n.clone()), n.clone()),
            SymbolOrOrdinal::Ordinal(o) => (SymbolRef::Ordinal(*o), format!("#{o}")),
        };
        if !seen.insert((imp.module.to_ascii_lowercase(), key)) {
            continue;
        }
        let index = u32::try_from(out.len()).unwrap_or(u32::MAX);
        let slot = Rva(OBSERVED_IAT_BASE + index * 8);
        out.push(SlotBinding { slot, module: imp.module.clone(), symbol: Some(symbol), hint: None });
    }
    out
}

/// EXPERIMENTAL Observed-tier reconstruction of the application import set from
/// a dynamic detonation's resolved imports. NEVER feeds the sound blocker gate.
#[derive(Clone, Debug)]
pub(crate) struct ObservedImportReconstruction {
    pub proof: ImportProof,
    /// Kept for fidelity comparison on the experimental tier (see
    /// [`import_fidelity`]); not consumed on the sound path.
    #[allow(dead_code)]
    pub proven: ProvenImportSet,
    /// Serialized import-directory byte length when the installer produced a
    /// valid (non-empty) directory from the observed bindings.
    pub directory_bytes: Option<usize>,
}

/// Reconstruct an application IAT from observed imports and run the real import
/// installer over it. Observed-tier: the result labels its provenance and is
/// surfaced on the experimental path only.
pub(crate) fn reconstruct_observed_imports(
    kind: PeKind,
    table_rva: Rva,
    imports: &[ObservedImport],
) -> ObservedImportReconstruction {
    let bindings = observed_slot_bindings(imports);
    let referenced: BTreeSet<Rva> = bindings.iter().map(|b| b.slot).collect();
    let proven = reduce_to_proven_imports(&bindings, &referenced);
    let complete = proven.unresolved.is_empty()
        && !referenced.is_empty()
        && proven.import_count() == referenced.len();
    let directory_bytes = if proven.import_count() == 0 {
        None
    } else {
        plan_proven_imports(kind, table_rva, &proven)
            .ok()
            .map(|plan| plan.table.bytes.len())
    };
    ObservedImportReconstruction {
        proof: ImportProof {
            complete,
            proven_modules: proven.modules.len(),
            proven_symbols: proven.import_count(),
        },
        proven,
        directory_bytes,
    }
}

/// Fidelity of an observed reconstruction against a known `(module, symbol)`
/// import set (module compared case-insensitively). Returns `(matched, known)`.
/// Used where the real imports are knowable (verification on the experimental
/// tier); real protected samples cannot yet reach their imports (see probe).
#[allow(dead_code)]
pub(crate) fn import_fidelity(
    proven: &ProvenImportSet,
    known: &BTreeSet<(String, String)>,
) -> (usize, usize) {
    let got: BTreeSet<(String, String)> = proven
        .modules
        .iter()
        .flat_map(|module| {
            module.imports.iter().map(move |import| {
                let sym = match &import.symbol {
                    SymbolRef::Name(name) => name.clone(),
                    SymbolRef::Ordinal(ordinal) => format!("#{ordinal}"),
                };
                (module.name.to_ascii_lowercase(), sym)
            })
        })
        .collect();
    let matched = known
        .iter()
        .filter(|(module, symbol)| got.contains(&(module.to_ascii_lowercase(), symbol.clone())))
        .count();
    (matched, known.len())
}

#[derive(Clone, Debug)]
pub(crate) struct ImportProof {
    pub complete: bool,
    pub proven_modules: usize,
    pub proven_symbols: usize,
}

pub(crate) fn prove_imports(modules: &[ImportModule], referenced: &BTreeSet<Rva>) -> ImportProof {
    let bindings = bindings_from_modules(modules);
    let proven: ProvenImportSet = reduce_to_proven_imports(&bindings, referenced);
    let complete = proven.unresolved.is_empty()
        && !referenced.is_empty()
        && proven.import_count() == referenced.len();
    ImportProof {
        complete,
        proven_modules: proven.modules.len(),
        proven_symbols: proven.import_count(),
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct CleanAntiDebugReport {
    pub guards_examined: usize,
    pub recognized: Vec<NeutralizeAction>,
    /// Recognition only; rewriting a guard changes behavior and stays deferred.
    pub applied: bool,
    pub note: String,
}

/// Classifies guards without keeping any rewrite: `neutralize` only reports
/// actions when enabled, so its rewritten guards are discarded here.
pub(crate) fn recognize_anti_debug(
    guards: &[RecoveredGuard],
    imports: &BTreeMap<String, String>,
) -> CleanAntiDebugReport {
    let (_discarded, recognized) = neutralize(guards, imports, NeutralizeOptions { enabled: true });
    CleanAntiDebugReport {
        guards_examined: guards.len(),
        recognized,
        applied: false,
        note: "recognition only; the pipeline does not yet extract conditional guards and \
               neutralization is not applied"
            .to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coverage_gate::{CoverageInput, decide_coverage};
    use vmpx_pe::ImportSymbol;
    use vmpx_vm::{BitWidth, DetectedWhen, Expr, INTEGRITY_CHECKSUM_INTRINSIC};

    fn entry(rva: u32) -> VmInventoryEntry {
        VmInventoryEntry {
            rva: Rva(rva),
            confidence: 90,
            sources: vec!["scan"],
            reachable_from_export: false,
        }
    }

    fn set(rvas: &[u32]) -> BTreeSet<Rva> {
        rvas.iter().map(|&r| Rva(r)).collect()
    }

    #[test]
    fn covered_set_counts_redirected_and_excludes_compiled_but_failed() {
        let mut covered = BTreeSet::new();
        mark_covered::<(), String>(&mut covered, Rva(0x1000), &Ok(()));
        mark_covered::<(), String>(&mut covered, Rva(0x2000), &Err("redirect failed".into()));
        assert_eq!(covered, set(&[0x1000]));
    }

    #[test]
    fn group_entry_needs_redirect_and_complete_family() {
        let mut covered = BTreeSet::new();
        group_covered(&mut covered, Rva(0x10), true, true);
        group_covered(&mut covered, Rva(0x20), true, false);
        group_covered(&mut covered, Rva(0x30), false, true);
        assert_eq!(covered, set(&[0x10]));
    }

    #[test]
    fn inventory_items_surface_scan_failure_and_drops() {
        let loss = InventoryLoss {
            scan_failure: Some("scan failed".into()),
            unanalyzed_exports: vec![Rva(0x9000)],
        };
        let (items, dropped) =
            build_inventory_items(&[entry(0x1000)], [Rva(0x1000), Rva(0x2000)], &loss);
        assert_eq!(dropped, 1);
        assert_eq!(
            items,
            vec![
                InventoryItem::Resolved(Rva(0x1000)),
                InventoryItem::Resolved(Rva(0x2000)),
                InventoryItem::Unresolved("scan failed".into()),
            ]
        );
    }

    fn decide(
        items: &[InventoryItem],
        covered: &BTreeSet<Rva>,
        dangling: &[Rva],
    ) -> CoverageDecision {
        decide_coverage(&CoverageInput {
            inventory: items,
            recovered: covered,
            oep_proven: true,
            dangling_vmp_refs: dangling,
            dropped_entries: 0,
        })
    }

    #[test]
    fn vmp_blocker_kept_when_covered_misses_an_inventory_rva() {
        let items = vec![
            InventoryItem::Resolved(Rva(0x1000)),
            InventoryItem::Resolved(Rva(0x2000)),
        ];
        let decision = decide(&items, &set(&[0x1000]), &[]);
        assert!(coverage_blockers(&decision, false, true).contains(&VMP_BLOCKER.to_owned()));
    }

    #[test]
    fn vmp_blocker_kept_when_dangling_targets_remain() {
        let items = vec![InventoryItem::Resolved(Rva(0x1000))];
        let decision = decide(&items, &set(&[0x1000]), &[Rva(0x5000)]);
        assert!(coverage_blockers(&decision, false, true).contains(&VMP_BLOCKER.to_owned()));
    }

    #[test]
    fn vmp_blocker_kept_for_unresolved_dangling_or_missing_image() {
        let items = vec![InventoryItem::Resolved(Rva(0x1000))];
        let decision = decide(&items, &set(&[0x1000]), &[]);
        assert!(coverage_blockers(&decision, true, true).contains(&VMP_BLOCKER.to_owned()));
        assert!(coverage_blockers(&decision, false, false).contains(&VMP_BLOCKER.to_owned()));
        assert!(
            coverage_blockers(&decision, false, false).contains(&REINTEGRATION_BLOCKER.to_owned())
        );
    }

    #[test]
    fn vmp_blocker_clears_only_when_everything_is_proven() {
        let items = vec![InventoryItem::Resolved(Rva(0x1000))];
        let decision = decide(&items, &set(&[0x1000]), &[]);
        assert!(coverage_blockers(&decision, false, true).is_empty());
    }

    #[test]
    fn oep_blocker_follows_the_gate() {
        let decision = decide_coverage(&CoverageInput {
            inventory: &[],
            recovered: &BTreeSet::new(),
            oep_proven: false,
            dangling_vmp_refs: &[],
            dropped_entries: 0,
        });
        assert!(coverage_blockers(&decision, false, true).contains(&OEP_BLOCKER.to_owned()));
    }

    #[test]
    fn dangling_summary_flags_unresolved() {
        let refs = [
            DanglingRef {
                source: None,
                target: Some(Rva(0x5000)),
                kind: DanglingKind::Call,
                detail: String::new(),
            },
            DanglingRef {
                source: None,
                target: None,
                kind: DanglingKind::Unresolved,
                detail: String::new(),
            },
        ];
        let summary = summarize_dangling(&refs);
        assert_eq!(summary.targets, vec![Rva(0x5000)]);
        assert!(summary.has_unresolved);
        assert!(!summarize_dangling(&refs[..1]).has_unresolved);
    }

    fn module() -> ImportModule {
        ImportModule {
            name: "kernel32.dll".into(),
            descriptor_rva: Rva(0),
            original_first_thunk: Rva(0),
            first_thunk: Rva(0x3000),
            entries: vec![ImportSymbol {
                iat_rva: Rva(0x3000),
                lookup_value: 0,
                name: Some("ExitProcess".into()),
                hint: Some(1),
                ordinal: None,
            }],
        }
    }

    #[test]
    fn import_blocker_stays_when_referenced_is_empty() {
        let proof = prove_imports(&[module()], &derive_referenced_slots(None));
        assert!(!proof.complete);
        assert_eq!(proof.proven_symbols, 0);
    }

    fn observed(module: &str, name: &str) -> ObservedImport {
        ObservedImport {
            provenance: vmpx_emulate::Observed,
            stub_va_or_slot: 0,
            module: module.to_owned(),
            symbol: SymbolOrOrdinal::Name(name.to_owned()),
        }
    }

    #[test]
    fn sound_referenced_slots_stay_empty_without_observation() {
        assert!(derive_referenced_slots(None).is_empty());
    }

    #[test]
    fn observed_imports_derive_nonempty_slots_and_install_a_directory() {
        let imports = [
            observed("kernel32.dll", "VirtualAlloc"),
            observed("user32.dll", "MessageBoxA"),
            observed("kernel32.dll", "VirtualAlloc"), // duplicate collapses
        ];
        // The bridge now yields a non-empty referenced set from observed facts.
        let slots = derive_referenced_slots(Some(&imports));
        assert_eq!(slots.len(), 2, "deduplicated to two distinct bindings");

        // The real import installer serializes a valid, non-empty directory.
        let recon = reconstruct_observed_imports(PeKind::Pe32Plus, Rva(0x6000), &imports);
        assert!(recon.proof.complete, "all observed bindings proven");
        assert_eq!(recon.proof.proven_symbols, 2);
        assert_eq!(recon.proof.proven_modules, 2);
        assert!(recon.directory_bytes.is_some_and(|n| n > 0), "directory produced");
    }

    #[test]
    fn observed_reconstruction_matches_known_imports_full_fidelity() {
        let imports = [
            observed("KERNEL32.dll", "VirtualAlloc"),
            observed("kernel32.dll", "GetProcAddress"),
        ];
        let recon = reconstruct_observed_imports(PeKind::Pe32Plus, Rva(0), &imports);
        let known: BTreeSet<(String, String)> = [
            ("kernel32.dll".to_owned(), "VirtualAlloc".to_owned()),
            ("kernel32.dll".to_owned(), "GetProcAddress".to_owned()),
        ]
        .into_iter()
        .collect();
        let (matched, total) = import_fidelity(&recon.proven, &known);
        assert_eq!((matched, total), (2, 2), "reconstructed set matches known imports");
    }

    #[test]
    fn empty_observation_installs_nothing() {
        let recon = reconstruct_observed_imports(PeKind::Pe32Plus, Rva(0), &[]);
        assert!(!recon.proof.complete);
        assert_eq!(recon.proof.proven_symbols, 0);
        assert!(recon.directory_bytes.is_none());
    }

    #[test]
    fn import_proof_completes_only_for_fully_resolved_references() {
        let proof = prove_imports(&[module()], &set(&[0x3000]));
        assert!(proof.complete);
        assert_eq!((proof.proven_modules, proof.proven_symbols), (1, 1));
        let unbound = prove_imports(&[module()], &set(&[0x3000, 0x4000]));
        assert!(!unbound.complete);
        assert_eq!(unbound.proven_symbols, 1);
    }

    #[test]
    fn anti_debug_recognizes_without_applying() {
        let checksum = Expr::intrinsic(
            BitWidth::W32,
            INTEGRITY_CHECKSUM_INTRINSIC,
            vec![Expr::register(BitWidth::W32, "accumulator")],
        );
        let guard = RecoveredGuard {
            site: Rva(0x10),
            condition: checksum,
            detected_when: DetectedWhen::NonZero,
        };
        let report = recognize_anti_debug(&[guard], &BTreeMap::new());
        assert_eq!(report.recognized.len(), 1);
        assert!(!report.applied);
        assert!(
            recognize_anti_debug(&[], &BTreeMap::new())
                .recognized
                .is_empty()
        );
    }
}
