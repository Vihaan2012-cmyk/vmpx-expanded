use std::collections::BTreeSet;

use vmpx_pe::Rva;

/// One discovered virtualized routine; `Unresolved` is never treated as covered.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InventoryItem {
    Resolved(Rva),
    Unresolved(String),
}

pub struct CoverageInput<'a> {
    pub inventory: &'a [InventoryItem],
    pub recovered: &'a BTreeSet<Rva>,
    pub oep_proven: bool,
    pub dangling_vmp_refs: &'a [Rva],
    /// Discovered entries known to have been discarded before reaching `inventory`.
    pub dropped_entries: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoverageDecision {
    pub can_clear_oep_blocker: bool,
    pub can_clear_vmp_referenced_blocker: bool,
    pub reasons: Vec<String>,
}

/// Strict gate: every doubt (unresolved, dropped, uncovered, dangling) keeps its blocker.
pub fn decide_coverage(input: &CoverageInput<'_>) -> CoverageDecision {
    let mut reasons = Vec::new();
    let mut vmp_open = false;

    if !input.oep_proven {
        reasons.push("OEP was not proven".to_owned());
    }

    for item in input.inventory {
        match item {
            InventoryItem::Resolved(rva) if input.recovered.contains(rva) => {}
            InventoryItem::Resolved(rva) => {
                vmp_open = true;
                reasons.push(format!(
                    "virtualized routine at {:#x} was not recovered",
                    rva.0
                ));
            }
            InventoryItem::Unresolved(why) => {
                vmp_open = true;
                reasons.push(format!(
                    "unresolved virtualized entry treated as uncovered: {why}"
                ));
            }
        }
    }
    if input.dropped_entries > 0 {
        vmp_open = true;
        reasons.push(format!(
            "{} discovered entries were dropped from the inventory and are treated as uncovered",
            input.dropped_entries
        ));
    }
    for rva in input.dangling_vmp_refs {
        vmp_open = true;
        reasons.push(format!("dangling .vmp reference to {:#x}", rva.0));
    }

    CoverageDecision {
        can_clear_oep_blocker: input.oep_proven,
        can_clear_vmp_referenced_blocker: !vmp_open,
        reasons,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resolved(rvas: &[u32]) -> Vec<InventoryItem> {
        rvas.iter()
            .map(|&r| InventoryItem::Resolved(Rva(r)))
            .collect()
    }

    fn set(rvas: &[u32]) -> BTreeSet<Rva> {
        rvas.iter().map(|&r| Rva(r)).collect()
    }

    fn decide(
        inventory: &[InventoryItem],
        recovered: &BTreeSet<Rva>,
        oep_proven: bool,
        dangling: &[Rva],
        dropped: usize,
    ) -> CoverageDecision {
        decide_coverage(&CoverageInput {
            inventory,
            recovered,
            oep_proven,
            dangling_vmp_refs: dangling,
            dropped_entries: dropped,
        })
    }

    #[test]
    fn all_covered_oep_proven_no_dangling_clears_both() {
        let d = decide(
            &resolved(&[0x1000, 0x2000]),
            &set(&[0x1000, 0x2000]),
            true,
            &[],
            0,
        );
        assert!(d.can_clear_oep_blocker);
        assert!(d.can_clear_vmp_referenced_blocker);
        assert!(d.reasons.is_empty());
    }

    #[test]
    fn one_uncovered_entry_keeps_vmp_blocker() {
        let d = decide(&resolved(&[0x1000, 0x2000]), &set(&[0x1000]), true, &[], 0);
        assert!(d.can_clear_oep_blocker);
        assert!(!d.can_clear_vmp_referenced_blocker);
        assert!(d.reasons.iter().any(|r| r.contains("0x2000")));
    }

    #[test]
    fn unresolved_entry_counts_as_uncovered() {
        let mut inventory = resolved(&[0x1000]);
        inventory.push(InventoryItem::Unresolved("indirect target".to_owned()));
        let d = decide(&inventory, &set(&[0x1000]), true, &[], 0);
        assert!(!d.can_clear_vmp_referenced_blocker);
        assert!(d.reasons.iter().any(|r| r.contains("indirect target")));
    }

    #[test]
    fn dropped_entries_count_as_uncovered() {
        let d = decide(&resolved(&[0x1000]), &set(&[0x1000]), true, &[], 2);
        assert!(!d.can_clear_vmp_referenced_blocker);
        assert!(d.reasons.iter().any(|r| r.contains('2')));
    }

    #[test]
    fn dangling_refs_keep_vmp_blocker() {
        let d = decide(
            &resolved(&[0x1000]),
            &set(&[0x1000]),
            true,
            &[Rva(0x5000)],
            0,
        );
        assert!(d.can_clear_oep_blocker);
        assert!(!d.can_clear_vmp_referenced_blocker);
        assert!(d.reasons.iter().any(|r| r.contains("0x5000")));
    }

    #[test]
    fn oep_not_proven_keeps_oep_blocker_only() {
        let d = decide(&resolved(&[0x1000]), &set(&[0x1000]), false, &[], 0);
        assert!(!d.can_clear_oep_blocker);
        assert!(d.can_clear_vmp_referenced_blocker);
        assert_eq!(d.reasons.len(), 1);
    }

    #[test]
    fn empty_inventory_clears_vacuously_when_nothing_dangles() {
        let d = decide(&[], &BTreeSet::new(), true, &[], 0);
        assert!(d.can_clear_oep_blocker);
        assert!(d.can_clear_vmp_referenced_blocker);
    }

    #[test]
    fn empty_inventory_with_dangling_or_drops_keeps_blocker() {
        let dangling = decide(&[], &BTreeSet::new(), true, &[Rva(0x9000)], 0);
        assert!(!dangling.can_clear_vmp_referenced_blocker);
        let dropped = decide(&[], &BTreeSet::new(), true, &[], 1);
        assert!(!dropped.can_clear_vmp_referenced_blocker);
    }

    #[test]
    fn recovered_set_without_inventory_match_does_not_cover() {
        let d = decide(&resolved(&[0x1000]), &set(&[0x2000]), true, &[], 0);
        assert!(!d.can_clear_vmp_referenced_blocker);
    }
}
