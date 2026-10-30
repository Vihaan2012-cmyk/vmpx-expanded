use serde::Serialize;
use std::collections::BTreeMap;
use vmpx_pe::{IMAGE_DIRECTORY_ENTRY_EXCEPTION, IMAGE_DIRECTORY_ENTRY_IMPORT, PeImage, Rva, Va};
use vmpx_vm::{VmDiscoveryLimits, discover_vm_entries};

const IMAGE_SCN_CNT_UNINITIALIZED_DATA: u32 = 0x0000_0080;
const IMAGE_SCN_MEM_EXECUTE: u32 = 0x2000_0000;
const IMAGE_SCN_MEM_READ: u32 = 0x4000_0000;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageLayout {
    ConventionalFile,
    PackedVirtualSections,
    RvaMappedAnalysisImage,
    Ambiguous,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DetectionVerdict {
    NoStrongEvidence,
    Suspicious,
    VmprotectLikely,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DetectionEvidence {
    pub id: &'static str,
    pub weight: u8,
    pub statement: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ExecutableRegion {
    pub section_index: usize,
    pub section_name: String,
    pub start: Rva,
    pub size: u32,
    pub entropy: Option<f64>,
    pub owns_entry_point: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ProtectionReport {
    pub schema_version: u32,
    pub verdict: DetectionVerdict,
    pub confidence: u8,
    pub image_layout: ImageLayout,
    pub entry_point: Rva,
    pub entry_section: Option<usize>,
    pub initialized_virtual_only_sections: usize,
    pub rva_mapped_sections: usize,
    pub executable_regions: Vec<ExecutableRegion>,
    pub evidence: Vec<DetectionEvidence>,
    pub warnings: Vec<String>,
}

#[allow(clippy::cast_precision_loss)]
fn entropy(bytes: &[u8]) -> f64 {
    if bytes.is_empty() {
        return 0.0;
    }
    let mut counts = [0_u64; 256];
    for byte in bytes {
        counts[usize::from(*byte)] += 1;
    }
    let length = bytes.len() as f64;
    counts
        .into_iter()
        .filter(|count| *count != 0)
        .map(|count| {
            let probability = count as f64 / length;
            -probability * probability.log2()
        })
        .sum()
}

fn section_entropy(pe: &PeImage<'_>, section_index: usize) -> Option<f64> {
    let section = pe.sections.get(section_index)?;
    if section.pointer_to_raw_data.0 == 0 || section.size_of_raw_data == 0 {
        return None;
    }
    let start = usize::try_from(section.pointer_to_raw_data.0).ok()?;
    let size = usize::try_from(section.size_of_raw_data).ok()?;
    let end = start.checked_add(size)?;
    pe.data().get(start..end).map(entropy)
}

fn owner_index(pe: &PeImage<'_>, rva: Rva) -> Option<usize> {
    pe.sections
        .iter()
        .position(|section| section.contains_mapped_rva(rva))
}

const STANDARD_SECTION_NAMES: &[&str] = &[
    ".text", ".rdata", ".data", ".pdata", ".xdata", ".reloc", ".rsrc", ".tls", ".idata", ".edata",
    ".bss", ".didat", ".gfids", ".crt", ".code", ".init", ".fini", ".CRT", ".00cfg", ".retplne",
    "INIT", "PAGE", "CODE", "DATA", ".sdata", ".orpc",
];

/// True for standard compiler/linker section names. A non-standard executable
/// section name is a protector signal (VMProtect randomizes its section names).
pub fn is_standard_section_name(name: &str) -> bool {
    STANDARD_SECTION_NAMES
        .iter()
        .any(|standard| standard.eq_ignore_ascii_case(name))
}

const STUB_SWARM_STRONG: usize = 8;
const STUB_SWARM_WEAK: usize = 3;

/// Largest group of VM entry stubs sharing one call target, as `(target, count)`.
fn largest_stub_swarm(call_targets: impl IntoIterator<Item = Option<Rva>>) -> Option<(Rva, usize)> {
    let mut groups: BTreeMap<Rva, usize> = BTreeMap::new();
    for target in call_targets.into_iter().flatten() {
        *groups.entry(target).or_default() += 1;
    }
    groups.into_iter().max_by_key(|(_, count)| *count)
}

fn add_evidence(
    evidence: &mut Vec<DetectionEvidence>,
    score: &mut u16,
    id: &'static str,
    weight: u8,
    statement: String,
) {
    *score = score.saturating_add(u16::from(weight));
    evidence.push(DetectionEvidence {
        id,
        weight,
        statement,
    });
}

/// Profiles structural `VMProtect` evidence without relying on the file extension
/// or one mandatory byte signature.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn detect_vmprotect(pe: &PeImage<'_>) -> ProtectionReport {
    let entry = pe.optional.address_of_entry_point;
    let entry_section = owner_index(pe, entry);
    let initialized_virtual_only_sections = pe
        .sections
        .iter()
        .filter(|section| {
            section.pointer_to_raw_data.0 == 0
                && section.size_of_raw_data == 0
                && section.virtual_size != 0
                && section.characteristics & IMAGE_SCN_CNT_UNINITIALIZED_DATA == 0
        })
        .count();
    let rva_mapped_sections = pe
        .sections
        .iter()
        .filter(|section| {
            section.pointer_to_raw_data.0 == u64::from(section.virtual_address.0)
                && section.size_of_raw_data == section.virtual_size
                && section.virtual_size != 0
        })
        .count();
    let predominantly_rva_mapped = !pe.sections.is_empty()
        && rva_mapped_sections.saturating_mul(5) >= pe.sections.len().saturating_mul(4);
    let image_layout = if predominantly_rva_mapped {
        ImageLayout::RvaMappedAnalysisImage
    } else if initialized_virtual_only_sections >= 3 {
        ImageLayout::PackedVirtualSections
    } else if initialized_virtual_only_sections == 0 {
        ImageLayout::ConventionalFile
    } else {
        ImageLayout::Ambiguous
    };

    let mut executable_regions = Vec::new();
    for section in &pe.sections {
        if section.characteristics & IMAGE_SCN_MEM_EXECUTE == 0 {
            continue;
        }
        executable_regions.push(ExecutableRegion {
            section_index: section.index,
            section_name: section.name.clone(),
            start: section.virtual_address,
            size: section.mapped_span(),
            entropy: section_entropy(pe, section.index),
            owns_entry_point: Some(section.index) == entry_section,
        });
    }

    let mut evidence = Vec::new();
    let mut score = 0_u16;
    let vmp_named: Vec<_> = pe
        .sections
        .iter()
        .filter(|section| section.name.to_ascii_lowercase().starts_with(".vmp"))
        .collect();
    if !vmp_named.is_empty() {
        add_evidence(
            &mut evidence,
            &mut score,
            "vmp_section_names",
            20,
            format!(
                "{} sections use the conventional .vmp name family; names are weak evidence because they can be changed",
                vmp_named.len()
            ),
        );
    }
    if initialized_virtual_only_sections >= 3 {
        add_evidence(
            &mut evidence,
            &mut score,
            "packed_virtual_sections",
            25,
            format!(
                "{initialized_virtual_only_sections} initialized sections retain virtual geometry but no raw bytes"
            ),
        );
    }
    if image_layout == ImageLayout::RvaMappedAnalysisImage {
        add_evidence(
            &mut evidence,
            &mut score,
            "rva_mapped_analysis_image",
            10,
            format!(
                "{rva_mapped_sections}/{} sections have raw pointer=RVA and raw size=virtual size",
                pe.sections.len()
            ),
        );
    }
    if let Some(entry_index) = entry_section
        && let Some(region) = executable_regions
            .iter()
            .find(|region| region.section_index == entry_index)
    {
        if let Some(value) = region.entropy
            && value >= 7.2
        {
            add_evidence(
                &mut evidence,
                &mut score,
                "high_entropy_entry_region",
                20,
                format!(
                    "entry point is in readable/executable section {} with entropy {value:.4}",
                    region.section_name
                ),
            );
        }
        let section = &pe.sections[entry_index];
        if section.characteristics & (IMAGE_SCN_MEM_EXECUTE | IMAGE_SCN_MEM_READ)
            == (IMAGE_SCN_MEM_EXECUTE | IMAGE_SCN_MEM_READ)
        {
            add_evidence(
                &mut evidence,
                &mut score,
                "entry_in_shared_protector_candidate",
                5,
                format!(
                    "entry-owning section {} is readable and executable",
                    section.name
                ),
            );
        }
    }
    let high_entropy_executable_count = executable_regions
        .iter()
        .filter(|region| region.entropy.is_some_and(|value| value >= 7.2))
        .count();
    if high_entropy_executable_count >= 2 {
        add_evidence(
            &mut evidence,
            &mut score,
            "multiple_high_entropy_executable_regions",
            10,
            format!(
                "{high_entropy_executable_count} executable sections have entropy at least 7.2"
            ),
        );
    }
    if let Some(entry_index) = entry_section {
        for (directory_index, id, label) in [
            (
                IMAGE_DIRECTORY_ENTRY_IMPORT,
                "imports_in_entry_region",
                "import",
            ),
            (
                IMAGE_DIRECTORY_ENTRY_EXCEPTION,
                "exceptions_in_entry_region",
                "exception",
            ),
        ] {
            if let Some(directory) = pe.directory(directory_index)
                && owner_index(pe, Rva(directory.address)) == Some(entry_index)
            {
                add_evidence(
                    &mut evidence,
                    &mut score,
                    id,
                    5,
                    format!("{label} directory resides in the entry-owning executable section"),
                );
            }
        }
        let report = pe.report();
        if let Some(first_callback) = report
            .tls
            .as_ref()
            .and_then(|tls| tls.callbacks.first())
            .and_then(|callback| pe.optional.image_base.checked_rva_of(*callback))
            && owner_index(pe, first_callback) == Some(entry_index)
        {
            add_evidence(
                &mut evidence,
                &mut score,
                "entry_and_tls_share_protector_region",
                10,
                "entry point and first TLS callback share the same executable region".to_owned(),
            );
        }
    }

    // Virtualization-only images carry no packed blob; the anomalous pairing of a
    // non-standard executable entry section with a swarm of push/call VM stubs
    // into one common target is the tell. Discovery runs only when the first holds.
    if let Some(entry_index) = entry_section {
        let section = &pe.sections[entry_index];
        if section.characteristics & IMAGE_SCN_MEM_EXECUTE != 0
            && !is_standard_section_name(&section.name)
        {
            add_evidence(
                &mut evidence,
                &mut score,
                "nonstandard_executable_entry_section",
                15,
                format!(
                    "entry point is in executable section {} whose name is not a standard compiler/linker section",
                    section.name
                ),
            );
            if let Ok(discovery) = discover_vm_entries(pe, VmDiscoveryLimits::default())
                && let Some((target, count)) =
                    largest_stub_swarm(discovery.entries.iter().map(|entry| entry.call_target))
                && count >= STUB_SWARM_WEAK
            {
                let weight = if count >= STUB_SWARM_STRONG { 45 } else { 20 };
                add_evidence(
                    &mut evidence,
                    &mut score,
                    "vm_entry_stub_swarm",
                    weight,
                    format!(
                        "{count} push-immediate/call VM entry stubs share the common call target {target}"
                    ),
                );
            }
        }
    }

    let confidence = u8::try_from(score.min(100)).unwrap_or(100);
    let verdict = if confidence >= 60 {
        DetectionVerdict::VmprotectLikely
    } else if confidence >= 30 {
        DetectionVerdict::Suspicious
    } else {
        DetectionVerdict::NoStrongEvidence
    };
    let mut warnings = Vec::new();
    if verdict == DetectionVerdict::VmprotectLikely && vmp_named.is_empty() {
        warnings.push(
            "classification is structural; randomized section names prevent name confirmation"
                .to_owned(),
        );
    }
    ProtectionReport {
        schema_version: 1,
        verdict,
        confidence,
        image_layout,
        entry_point: entry,
        entry_section,
        initialized_virtual_only_sections,
        rva_mapped_sections,
        executable_regions,
        evidence,
        warnings,
    }
}

#[must_use]
pub fn va_owner(pe: &PeImage<'_>, va: Va) -> Option<usize> {
    pe.optional
        .image_base
        .checked_rva_of(va)
        .and_then(|rva| owner_index(pe, rva))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_pe(section_name: &[u8], stub_count: usize) -> Vec<u8> {
        let mut data = vec![0_u8; 0x400];
        data[0..2].copy_from_slice(&0x5A4D_u16.to_le_bytes());
        data[0x3C..0x40].copy_from_slice(&0x80_u32.to_le_bytes());
        data[0x80..0x84].copy_from_slice(&0x0000_4550_u32.to_le_bytes());
        let coff = 0x84;
        data[coff..coff + 2].copy_from_slice(&0x8664_u16.to_le_bytes());
        data[coff + 2..coff + 4].copy_from_slice(&1_u16.to_le_bytes());
        data[coff + 16..coff + 18].copy_from_slice(&0xF0_u16.to_le_bytes());
        data[coff + 18..coff + 20].copy_from_slice(&0x2022_u16.to_le_bytes());
        let optional = coff + 20;
        data[optional..optional + 2].copy_from_slice(&0x020B_u16.to_le_bytes());
        data[optional + 16..optional + 20].copy_from_slice(&0x1000_u32.to_le_bytes());
        data[optional + 20..optional + 24].copy_from_slice(&0x1000_u32.to_le_bytes());
        data[optional + 24..optional + 32].copy_from_slice(&0x1_4000_0000_u64.to_le_bytes());
        data[optional + 32..optional + 36].copy_from_slice(&0x1000_u32.to_le_bytes());
        data[optional + 36..optional + 40].copy_from_slice(&0x200_u32.to_le_bytes());
        data[optional + 56..optional + 60].copy_from_slice(&0x2000_u32.to_le_bytes());
        data[optional + 60..optional + 64].copy_from_slice(&0x200_u32.to_le_bytes());
        data[optional + 68..optional + 70].copy_from_slice(&3_u16.to_le_bytes());
        data[optional + 108..optional + 112].copy_from_slice(&16_u32.to_le_bytes());
        let section = optional + 0xF0;
        data[section..section + section_name.len()].copy_from_slice(section_name);
        data[section + 8..section + 12].copy_from_slice(&0x180_u32.to_le_bytes());
        data[section + 12..section + 16].copy_from_slice(&0x1000_u32.to_le_bytes());
        data[section + 16..section + 20].copy_from_slice(&0x200_u32.to_le_bytes());
        data[section + 20..section + 24].copy_from_slice(&0x200_u32.to_le_bytes());
        data[section + 36..section + 40].copy_from_slice(&0x6000_0020_u32.to_le_bytes());
        // common target at 0x1000 (ret); stubs: push imm32 ; call rel32
        data[0x200] = 0xC3;
        for index in 0..stub_count {
            let at = 0x210 + index * 10;
            data[at] = 0x68;
            data[at + 1..at + 5].copy_from_slice(&(0x1000_0000_u32 + index as u32).to_le_bytes());
            data[at + 5] = 0xE8;
            let rel = 0x1000_i32 - i32::try_from(0x1000 + (at - 0x200) + 10).unwrap();
            data[at + 6..at + 10].copy_from_slice(&rel.to_le_bytes());
        }
        data
    }

    fn has(report: &ProtectionReport, id: &str) -> bool {
        report.evidence.iter().any(|item| item.id == id)
    }

    #[test]
    fn swarm_groups_by_common_call_target() {
        let common = Rva(0x11F000);
        let targets = std::iter::repeat_n(Some(common), 9).chain([Some(Rva(0x2000)), None]);
        assert_eq!(largest_stub_swarm(targets), Some((common, 9)));
        assert_eq!(largest_stub_swarm([None, None]), None);
    }

    #[test]
    fn standard_names_are_recognized_case_insensitively() {
        for name in [".text", ".TEXT", ".rdata", ".pdata", ".reloc", ".tls"] {
            assert!(is_standard_section_name(name), "{name}");
        }
        for name in [".2ZQ", ".o}1", ".vmp0", ".themida"] {
            assert!(!is_standard_section_name(name), "{name}");
        }
    }

    #[test]
    fn standard_text_section_with_stubs_is_not_flagged() {
        let data = synthetic_pe(b".text", 12);
        let pe = PeImage::parse(&data).expect("synthetic PE parses");
        let report = detect_vmprotect(&pe);
        assert!(!has(&report, "nonstandard_executable_entry_section"));
        assert!(!has(&report, "vm_entry_stub_swarm"));
        assert_eq!(report.verdict, DetectionVerdict::NoStrongEvidence);
    }

    #[test]
    fn nonstandard_entry_section_alone_stays_below_strong_evidence() {
        let data = synthetic_pe(b".o}1", 0);
        let pe = PeImage::parse(&data).expect("synthetic PE parses");
        let report = detect_vmprotect(&pe);
        assert!(has(&report, "nonstandard_executable_entry_section"));
        assert!(!has(&report, "vm_entry_stub_swarm"));
        assert_eq!(report.verdict, DetectionVerdict::NoStrongEvidence);
    }

    #[test]
    fn empty_entropy_is_zero() {
        assert!(entropy(&[]).abs() < f64::EPSILON);
    }

    #[test]
    fn constant_entropy_is_zero() {
        assert!(entropy(&[0xAA; 1024]).abs() < f64::EPSILON);
    }
}
