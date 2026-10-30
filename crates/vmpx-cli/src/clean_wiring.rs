//! Pure glue between the clean pipeline and the soundness cores. Every helper
//! here only ever keeps a blocker when in doubt.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;
use vmpx_disasm::{DanglingKind, DanglingRef};
use vmpx_pe::{ImportModule, Rva};
use vmpx_rebuild::{ProvenImportSet, bindings_from_modules, reduce_to_proven_imports};
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

/// IAT slots with a proven direct reference from native (non-`.vmp`) code.
/// Always empty: a partial native scan cannot prove completeness, because
/// protector-resident code also uses the table, so no reduced set is ever
/// claimed as the full application import set.
pub(crate) fn derive_referenced_slots() -> BTreeSet<Rva> {
    BTreeSet::new()
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
        let proof = prove_imports(&[module()], &derive_referenced_slots());
        assert!(!proof.complete);
        assert_eq!(proof.proven_symbols, 0);
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
