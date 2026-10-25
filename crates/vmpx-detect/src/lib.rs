use serde::Serialize;
use vmpx_pe::{IMAGE_DIRECTORY_ENTRY_EXCEPTION, IMAGE_DIRECTORY_ENTRY_IMPORT, PeImage, Rva, Va};

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

    #[test]
    fn empty_entropy_is_zero() {
        assert!(entropy(&[]).abs() < f64::EPSILON);
    }

    #[test]
    fn constant_entropy_is_zero() {
        assert!(entropy(&[0xAA; 1024]).abs() < f64::EPSILON);
    }
}
