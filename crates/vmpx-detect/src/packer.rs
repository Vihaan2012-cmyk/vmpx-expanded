//! Signature-based detection of common, non-`VMProtect` executable packers.
//!
//! Pure PE-structure inspection: the section table, the entry point, and a
//! bounded byte scan for magic markers. No sample is executed and nothing is
//! mapped — this is totally safe on untrusted input. Kept separate from the
//! `VMProtect` structural profiler; it only recognizes easy packers and hands
//! back a kind + confidence.

use serde::Serialize;

use vmpx_pe::PeImage;

use crate::is_standard_section_name;

const IMAGE_SCN_MEM_EXECUTE: u32 = 0x2000_0000;
const IMAGE_SCN_MEM_WRITE: u32 = 0x8000_0000;

/// A recognized easy packer, or `None` when no signature matched.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PackerKind {
    None,
    Upx,
    Aspack,
    PeCompact,
    Mpress,
    Fsg,
}

impl PackerKind {
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            PackerKind::None => "none",
            PackerKind::Upx => "UPX",
            PackerKind::Aspack => "ASPack",
            PackerKind::PeCompact => "PECompact",
            PackerKind::Mpress => "MPRESS",
            PackerKind::Fsg => "FSG",
        }
    }

    /// Whether this packer is routed to the generic bounded unpack-to-OEP
    /// attempt. Named-section packers self-unpack at runtime via a
    /// write-then-execute handoff the OEP detector catches; the low-confidence
    /// FSG heuristic is NOT trusted to drive a dynamic run.
    #[must_use]
    pub fn routes_to_generic_unpack(self) -> bool {
        matches!(
            self,
            PackerKind::Upx | PackerKind::Aspack | PackerKind::PeCompact | PackerKind::Mpress
        )
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PackerReport {
    pub kind: PackerKind,
    /// 0..=100. Named-section matches are high; the FSG heuristic is low.
    pub confidence: u8,
    pub evidence: Vec<String>,
}

impl PackerReport {
    fn none() -> Self {
        Self { kind: PackerKind::None, confidence: 0, evidence: Vec::new() }
    }

    fn named(kind: PackerKind, confidence: u8, statement: String) -> Self {
        Self { kind, confidence, evidence: vec![statement] }
    }
}

/// Bounded search for `needle` in the image bytes. Scans only a head window
/// (headers + unpack stub) and, for larger files, a tail window (overlay), so a
/// huge sample costs a fixed amount regardless of size.
fn scan_marker(pe: &PeImage<'_>, needle: &[u8]) -> bool {
    const WIN: usize = 0x2_0000;
    let data = pe.data();
    let contains = |slice: &[u8]| slice.windows(needle.len()).any(|w| w == needle);
    let n = data.len();
    if contains(&data[..n.min(WIN)]) {
        return true;
    }
    n > WIN && contains(&data[n - WIN..])
}

/// FSG heuristic: FSG stubs emit very few sections with the entry in a
/// self-modifying (writable+executable), non-standard-named first section.
/// Deliberately conservative and clearly a lower-confidence signal than a named
/// packer section.
fn fsg_heuristic(pe: &PeImage<'_>) -> Option<PackerReport> {
    let first = pe.sections.first()?;
    if pe.sections.len() > 2 {
        return None;
    }
    if !first.contains_mapped_rva(pe.optional.address_of_entry_point) {
        return None;
    }
    let rwx = first.characteristics & IMAGE_SCN_MEM_EXECUTE != 0
        && first.characteristics & IMAGE_SCN_MEM_WRITE != 0;
    if !rwx || is_standard_section_name(&first.name) {
        return None;
    }
    Some(PackerReport {
        kind: PackerKind::Fsg,
        confidence: 35,
        evidence: vec![format!(
            "heuristic: {} section(s), entry in writable+executable non-standard first section '{}'",
            pe.sections.len(),
            first.name
        )],
    })
}

/// Recognize a common easy packer from PE structure alone. `VMProtect`
/// detection is intentionally not consulted here; the two live side by side.
#[must_use]
pub fn detect_packer(pe: &PeImage<'_>) -> PackerReport {
    let names: Vec<String> = pe
        .sections
        .iter()
        .map(|section| section.name.to_ascii_lowercase())
        .collect();
    let has = |needle: &str| names.iter().any(|name| name == needle);

    // UPX: UPX0/UPX1/UPX2 section names, corroborated by the UPX! magic.
    let upx_sections = names.iter().filter(|name| name.starts_with("upx")).count();
    if has("upx0") || has("upx1") || has("upx2") {
        let mut evidence = vec![format!("{upx_sections} UPX# packer section name(s)")];
        let mut confidence = 90;
        if scan_marker(pe, b"UPX!") {
            evidence.push("UPX! magic present in image bytes".to_owned());
            confidence = 97;
        }
        return PackerReport { kind: PackerKind::Upx, confidence, evidence };
    }

    // ASPack.
    if has(".aspack") || has(".adata") {
        return PackerReport::named(
            PackerKind::Aspack,
            90,
            ".aspack/.adata packer section present".to_owned(),
        );
    }

    // PECompact: pec1/pec2 sections (strong), else a PEC2 marker (weaker).
    if has("pec1") || has("pec2") {
        return PackerReport::named(
            PackerKind::PeCompact,
            90,
            "pec1/pec2 packer section present".to_owned(),
        );
    }
    if scan_marker(pe, b"PEC2") {
        return PackerReport::named(
            PackerKind::PeCompact,
            70,
            "PEC2 marker present in image bytes".to_owned(),
        );
    }

    // MPRESS.
    if has(".mpress1") || has(".mpress2") {
        return PackerReport::named(
            PackerKind::Mpress,
            90,
            ".MPRESS1/.MPRESS2 packer section present".to_owned(),
        );
    }

    // FSG: heuristic only, lower confidence, and only when nothing above matched.
    if let Some(report) = fsg_heuristic(pe) {
        return report;
    }

    PackerReport::none()
}

#[cfg(test)]
mod tests {
    use super::*;
    use vmpx_pe::PeImage;

    /// One section header for [`build_pe`]: `(name, rva, raw_ptr, vsize,
    /// raw_size, characteristics)`.
    type SectionDef<'a> = (&'a [u8], u32, u32, u32, u32, u32);

    /// Minimal PE32+ builder. Entry RVA and optional trailing bytes (for a
    /// magic-marker test) are configurable. No code is placed — detection only
    /// reads structure.
    #[allow(clippy::type_complexity)]
    fn build_pe(entry_rva: u32, sections: &[SectionDef<'_>], trailer: &[u8]) -> Vec<u8> {
        let headers = 0x400usize;
        let mut data = vec![0u8; headers + trailer.len()];
        data[0..2].copy_from_slice(&0x5A4Du16.to_le_bytes()); // MZ
        data[0x3C..0x40].copy_from_slice(&0x80u32.to_le_bytes()); // e_lfanew
        data[0x80..0x84].copy_from_slice(&0x0000_4550u32.to_le_bytes()); // PE\0\0
        let coff = 0x84;
        data[coff..coff + 2].copy_from_slice(&0x8664u16.to_le_bytes()); // x64
        let section_count = u16::try_from(sections.len()).expect("few sections");
        data[coff + 2..coff + 4].copy_from_slice(&section_count.to_le_bytes());
        data[coff + 16..coff + 18].copy_from_slice(&0xF0u16.to_le_bytes()); // opt hdr size
        data[coff + 18..coff + 20].copy_from_slice(&0x2022u16.to_le_bytes()); // chars (DLL)
        let opt = coff + 20;
        data[opt..opt + 2].copy_from_slice(&0x020Bu16.to_le_bytes()); // PE32+
        data[opt + 16..opt + 20].copy_from_slice(&entry_rva.to_le_bytes());
        data[opt + 24..opt + 32].copy_from_slice(&0x1_4000_0000u64.to_le_bytes()); // image base
        data[opt + 32..opt + 36].copy_from_slice(&0x1000u32.to_le_bytes()); // sect align
        data[opt + 36..opt + 40].copy_from_slice(&0x200u32.to_le_bytes()); // file align
        data[opt + 56..opt + 60].copy_from_slice(&0x4_0000u32.to_le_bytes()); // size of image
        data[opt + 60..opt + 64].copy_from_slice(&0x200u32.to_le_bytes()); // size of headers
        data[opt + 68..opt + 70].copy_from_slice(&3u16.to_le_bytes()); // subsystem
        data[opt + 108..opt + 112].copy_from_slice(&16u32.to_le_bytes()); // # rva/sizes
        let table = opt + 0xF0;
        for (index, (name, rva, raw_ptr, vsize, raw_size, chars)) in sections.iter().enumerate() {
            let at = table + index * 40;
            data[at..at + name.len()].copy_from_slice(name);
            data[at + 8..at + 12].copy_from_slice(&vsize.to_le_bytes());
            data[at + 12..at + 16].copy_from_slice(&rva.to_le_bytes());
            data[at + 16..at + 20].copy_from_slice(&raw_size.to_le_bytes());
            data[at + 20..at + 24].copy_from_slice(&raw_ptr.to_le_bytes());
            data[at + 36..at + 40].copy_from_slice(&chars.to_le_bytes());
        }
        data[headers..].copy_from_slice(trailer);
        data
    }

    const RX: u32 = 0x6000_0020;
    const RW: u32 = 0xC000_0040;
    const RWX: u32 = 0xE000_0020;

    #[test]
    fn upx_sections_and_magic_detected() {
        let data = build_pe(
            0x1000,
            &[
                (b"UPX0", 0x1000, 0x200, 0x1000, 0x0, RWX),
                (b"UPX1", 0x2000, 0x200, 0x1000, 0x200, RWX),
            ],
            b"........UPX!........",
        );
        let pe = PeImage::parse(&data).expect("parses");
        let report = detect_packer(&pe);
        assert_eq!(report.kind, PackerKind::Upx);
        assert_eq!(report.confidence, 97, "magic lifts confidence");
    }

    #[test]
    fn upx_sections_without_magic_still_detected() {
        let data = build_pe(
            0x1000,
            &[(b"UPX1", 0x1000, 0x200, 0x1000, 0x200, RWX)],
            b"",
        );
        let pe = PeImage::parse(&data).expect("parses");
        let report = detect_packer(&pe);
        assert_eq!(report.kind, PackerKind::Upx);
        assert_eq!(report.confidence, 90);
    }

    #[test]
    fn aspack_pecompact_mpress_detected_by_section() {
        for (name, expected) in [
            (&b".aspack"[..], PackerKind::Aspack),
            (b".adata", PackerKind::Aspack),
            (b"pec1", PackerKind::PeCompact),
            (b".MPRESS1", PackerKind::Mpress),
        ] {
            let data = build_pe(
                0x1000,
                &[
                    (b".text", 0x1000, 0x200, 0x1000, 0x200, RX),
                    (name, 0x2000, 0x400, 0x1000, 0x200, RW),
                ],
                b"",
            );
            let pe = PeImage::parse(&data).expect("parses");
            let report = detect_packer(&pe);
            assert_eq!(report.kind, expected, "{:?}", std::str::from_utf8(name));
            assert_eq!(report.confidence, 90);
        }
    }

    #[test]
    fn fsg_heuristic_is_low_confidence_and_conservative() {
        // 2 sections, entry in a writable+executable non-standard first section.
        let data = build_pe(
            0x1010,
            &[
                (b"\x00\x00\x00", 0x1000, 0x200, 0x1000, 0x200, RWX),
                (b"\x00\x00\x00", 0x2000, 0x400, 0x1000, 0x200, RW),
            ],
            b"",
        );
        let pe = PeImage::parse(&data).expect("parses");
        let report = detect_packer(&pe);
        assert_eq!(report.kind, PackerKind::Fsg);
        assert!(report.confidence < 60, "heuristic stays low-confidence");
    }

    #[test]
    fn clean_pe_is_not_misdetected() {
        // Standard names, entry in a read-execute (non-writable) .text section.
        let data = build_pe(
            0x1000,
            &[
                (b".text", 0x1000, 0x200, 0x1000, 0x200, RX),
                (b".rdata", 0x2000, 0x400, 0x1000, 0x200, 0x4000_0040),
                (b".data", 0x3000, 0x600, 0x1000, 0x200, RW),
            ],
            b"",
        );
        let pe = PeImage::parse(&data).expect("parses");
        assert_eq!(detect_packer(&pe).kind, PackerKind::None);
    }

    #[test]
    fn vmprotect_like_image_is_not_misdetected_as_packer() {
        // Randomized .vmp-style section names, many sections: no packer marker,
        // and too many sections for the FSG heuristic.
        let data = build_pe(
            0x3000,
            &[
                (b".text", 0x1000, 0x200, 0x1000, 0x200, RX),
                (b".vmp0", 0x2000, 0x400, 0x1000, 0x200, RW),
                (b".vmp1", 0x3000, 0x600, 0x1000, 0x200, RWX),
                (b".2ZQ", 0x4000, 0x800, 0x1000, 0x200, RWX),
            ],
            b"",
        );
        let pe = PeImage::parse(&data).expect("parses");
        assert_eq!(detect_packer(&pe).kind, PackerKind::None);
    }
}
