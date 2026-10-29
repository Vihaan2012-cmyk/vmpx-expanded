use std::collections::BTreeMap;

use serde::Serialize;
use vmpx_pe::Rva;

/// One virtualized routine, deduplicated across every discovery source.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct VmInventoryEntry {
    pub rva: Rva,
    pub confidence: u8,
    pub sources: Vec<&'static str>,
    pub reachable_from_export: bool,
}

/// Merges `(rva, confidence, source, reachable_from_export)` observations into
/// one entry per RVA: highest confidence wins, source labels are unioned, and
/// the export flag is combined with OR. Output is sorted by RVA.
pub fn merge_vm_inventory(
    items: impl IntoIterator<Item = (Rva, u8, &'static str, bool)>,
) -> Vec<VmInventoryEntry> {
    let mut merged: BTreeMap<Rva, VmInventoryEntry> = BTreeMap::new();
    for (rva, confidence, source, from_export) in items {
        let entry = merged.entry(rva).or_insert_with(|| VmInventoryEntry {
            rva,
            confidence,
            sources: Vec::new(),
            reachable_from_export: false,
        });
        entry.confidence = entry.confidence.max(confidence);
        entry.reachable_from_export |= from_export;
        if let Err(position) = entry.sources.binary_search(&source) {
            entry.sources.insert(position, source);
        }
    }
    merged.into_values().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_gives_empty_inventory() {
        assert!(merge_vm_inventory([]).is_empty());
    }

    #[test]
    fn distinct_rvas_stay_separate_and_sorted() {
        let merged = merge_vm_inventory([
            (Rva(0x3000), 70, "scan", false),
            (Rva(0x1000), 90, "header", false),
            (Rva(0x2000), 60, "export_xref", true),
        ]);
        let rvas: Vec<u32> = merged.iter().map(|entry| entry.rva.0).collect();
        assert_eq!(rvas, [0x1000, 0x2000, 0x3000]);
        assert_eq!(merged[1].sources, ["export_xref"]);
        assert!(merged[1].reachable_from_export);
        assert!(!merged[0].reachable_from_export);
    }

    #[test]
    fn overlapping_rvas_merge_into_one_entry() {
        let merged = merge_vm_inventory([
            (Rva(0x1000), 60, "scan", false),
            (Rva(0x1000), 60, "export_xref", true),
        ]);
        assert_eq!(merged.len(), 1);
    }

    #[test]
    fn max_confidence_wins_regardless_of_order() {
        let forward = merge_vm_inventory([
            (Rva(0x1000), 60, "scan", false),
            (Rva(0x1000), 95, "header", false),
            (Rva(0x1000), 80, "export_xref", true),
        ]);
        let reverse = merge_vm_inventory([
            (Rva(0x1000), 80, "export_xref", true),
            (Rva(0x1000), 95, "header", false),
            (Rva(0x1000), 60, "scan", false),
        ]);
        assert_eq!(forward[0].confidence, 95);
        assert_eq!(forward, reverse);
    }

    #[test]
    fn sources_are_unioned_sorted_and_deduplicated() {
        let merged = merge_vm_inventory([
            (Rva(0x1000), 70, "scan", false),
            (Rva(0x1000), 70, "export_xref", true),
            (Rva(0x1000), 70, "scan", false),
            (Rva(0x1000), 70, "header", false),
        ]);
        assert_eq!(merged[0].sources, ["export_xref", "header", "scan"]);
    }

    #[test]
    fn export_flag_is_ored_across_contributors() {
        let merged = merge_vm_inventory([
            (Rva(0x1000), 70, "scan", false),
            (Rva(0x1000), 70, "export_xref", true),
            (Rva(0x1000), 70, "tls", false),
        ]);
        assert!(merged[0].reachable_from_export);
        let none = merge_vm_inventory([(Rva(0x1000), 70, "scan", false)]);
        assert!(!none[0].reachable_from_export);
    }

    #[test]
    fn output_is_independent_of_input_order() {
        let items = [
            (Rva(0x2000), 60, "scan", false),
            (Rva(0x1000), 70, "export_xref", true),
            (Rva(0x2000), 85, "tls", false),
            (Rva(0x1000), 50, "scan", false),
        ];
        let mut reversed = items;
        reversed.reverse();
        assert_eq!(merge_vm_inventory(items), merge_vm_inventory(reversed));
    }
}
