//! CLI wiring for `vmpx_pe::strip_sections`.
//!
//! The library primitive is sound by construction: it refuses whenever any
//! retained structure statically references a removed section. This module adds
//! the two guarantees a written artifact needs on top of that refusal: the
//! rewritten image must reparse from its own bytes and pass
//! `vmpx_rebuild::validate_for_loader`. Nothing is written unless all three hold.

use std::{
    collections::BTreeSet,
    fs,
    path::Path,
};

use anyhow::{Context, Result, bail};
use vmpx_pe::{PeImage, SectionSelector, StripError, StripRequest, strip_sections};
use vmpx_rebuild::validate_for_loader;

/// What a successful strip removed.
pub struct StripSummary {
    pub removed: Vec<String>,
    pub file_bytes_removed: u64,
    pub relocation_entries_dropped: usize,
    pub runtime_functions_dropped: usize,
}

/// Why a strip did not yield a written artifact.
pub enum StripFailure {
    /// The dangling-ref checker or another static guard refused the removal.
    /// Nothing was rewritten.
    Refused(StripError),
    /// The removal produced bytes that failed to reparse or to validate for the
    /// loader. The primitive's own output invariants should prevent this; it is
    /// surfaced rather than swallowed so a written file is never unverified.
    Unverified(String),
}

impl std::fmt::Display for StripFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Refused(error) => write!(f, "{error}"),
            Self::Unverified(message) => write!(f, "{message}"),
        }
    }
}

/// Strips the selected sections and returns the rewritten image ONLY when it
/// reparses from its own bytes and passes loader validation. Any refusal, or any
/// post-strip invariant failure, is returned without producing output.
pub fn strip_and_validate(
    input: &[u8],
    remove: &[SectionSelector],
    drop_directories: &[usize],
) -> Result<(Vec<u8>, StripSummary), StripFailure> {
    let stripped = strip_sections(
        input,
        &StripRequest {
            remove,
            drop_directories,
        },
    )
    .map_err(StripFailure::Refused)?;
    PeImage::parse(&stripped.data).map_err(|error| {
        StripFailure::Unverified(format!("stripped image failed to reparse: {error}"))
    })?;
    validate_for_loader(&stripped.data).map_err(|error| {
        StripFailure::Unverified(format!("stripped image failed loader validation: {error}"))
    })?;
    let summary = StripSummary {
        removed: stripped.removed.iter().map(|s| s.name.clone()).collect(),
        file_bytes_removed: stripped.file_bytes_removed,
        relocation_entries_dropped: stripped.relocation_entries_dropped,
        runtime_functions_dropped: stripped.runtime_functions_dropped,
    };
    Ok((stripped.data, summary))
}

/// Distinct, deduplicated selectors for every non-standard-named section in the
/// image. VMProtect randomizes protector section names, so a standard-name
/// allowlist (not a `.vmp` denylist) is the structural signal. Standard
/// sections are never selected, so a strip of this set can never remove them.
pub fn protector_section_selectors(image: &[u8]) -> Result<Vec<SectionSelector>> {
    let pe = PeImage::parse(image).context("image is not a parseable PE")?;
    let names: BTreeSet<String> = pe
        .sections
        .iter()
        .filter(|section| !vmpx_detect::is_standard_section_name(&section.name))
        .map(|section| section.name.clone())
        .collect();
    Ok(names.into_iter().map(SectionSelector::Name).collect())
}

/// Outcome of the clean pipeline's optional strip stage.
pub enum CleanStripStage {
    /// The image has no non-standard sections to remove.
    NoProtectorSections,
    /// The protector sections were removed and the result was validated.
    Stripped { bytes: Vec<u8>, summary: StripSummary },
    /// A live reference (or another guard) kept a protector section; the image
    /// is left exactly as it was. This is the normal case for a still-protected
    /// or only partially devirtualized binary.
    Refused(StripFailure),
}

/// Attempts to remove every protector (non-standard-named) section from an image
/// that the clean pipeline has already reconstructed. It only ever returns a new
/// image when the sound primitive approved the removal AND the result validated;
/// otherwise it reports the refusal and changes nothing.
pub fn strip_protector_sections(image: &[u8]) -> Result<CleanStripStage> {
    let selectors = protector_section_selectors(image)?;
    if selectors.is_empty() {
        return Ok(CleanStripStage::NoProtectorSections);
    }
    match strip_and_validate(image, &selectors, &[]) {
        Ok((bytes, summary)) => Ok(CleanStripStage::Stripped { bytes, summary }),
        Err(failure) => Ok(CleanStripStage::Refused(failure)),
    }
}

fn print_summary(tag: &str, summary: &StripSummary) {
    println!(
        "[{tag}] removed {} section(s): {}",
        summary.removed.len(),
        summary.removed.join(", ")
    );
    println!(
        "[{tag}] {} file byte(s) removed, {} relocation entry/entries dropped, {} runtime function(s) dropped",
        summary.file_bytes_removed,
        summary.relocation_entries_dropped,
        summary.runtime_functions_dropped
    );
}

/// `vmpx strip` subcommand: strip the named sections from `input` and write the
/// result to `output`, but only when the sound primitive approves and the
/// rewritten image reparses from disk and validates for the loader. On any
/// refusal the reason is printed and the process exits nonzero with nothing
/// written.
pub fn run_strip(
    input: &Path,
    output: &Path,
    remove: &[String],
    drop_directories: &[usize],
) -> Result<()> {
    if input == output {
        bail!("refusing to overwrite the input file");
    }
    if output.exists() && fs::canonicalize(input).ok() == fs::canonicalize(output).ok() {
        bail!("refusing to overwrite the input file through an alias");
    }
    if output.exists() {
        bail!("refusing to overwrite existing output {}", output.display());
    }
    if remove.is_empty() {
        bail!("nothing selected; pass at least one --remove <section-name>");
    }
    let data = crate::read_file(input)?;
    let selectors: Vec<SectionSelector> = remove
        .iter()
        .map(|name| SectionSelector::Name(name.clone()))
        .collect();
    match strip_and_validate(&data, &selectors, drop_directories) {
        Ok((bytes, summary)) => {
            fs::write(output, &bytes)
                .with_context(|| format!("failed to write {}", output.display()))?;
            let disk = crate::read_file(output)?;
            PeImage::parse(&disk).with_context(|| {
                format!("written output {} failed disk reparse", output.display())
            })?;
            validate_for_loader(&disk).with_context(|| {
                format!(
                    "written output {} failed loader validation from disk",
                    output.display()
                )
            })?;
            print_summary("strip", &summary);
            println!(
                "[strip] wrote {} bytes to {}",
                bytes.len(),
                output.display()
            );
            println!("[verify] output reparsed and validated for the loader from disk");
            Ok(())
        }
        Err(failure) => {
            eprintln!("[strip] refused: {failure}");
            bail!(
                "strip refused {}; no output was written",
                input.display()
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const IMAGE_BASE: u64 = 0x1_4000_0000;
    const OPTIONAL: usize = 0x98;

    fn put(data: &mut [u8], offset: usize, bytes: &[u8]) {
        data[offset..offset + bytes.len()].copy_from_slice(bytes);
    }

    /// PE32+ with .text (holds the entry point), .data, and two non-standard
    /// protector sections `.ab0`/`.ab1`, each with 0x200 raw bytes. Mirrors the
    /// fixture used by the strip primitive's own tests.
    fn build() -> Vec<u8> {
        let mut data = vec![0_u8; 0x400 + 4 * 0x200];
        put(&mut data, 0, b"MZ");
        put(&mut data, 0x3C, &0x80_u32.to_le_bytes());
        put(&mut data, 0x80, b"PE\0\0");
        put(&mut data, 0x84, &0x8664_u16.to_le_bytes());
        put(&mut data, 0x86, &4_u16.to_le_bytes());
        put(&mut data, 0x94, &0xF0_u16.to_le_bytes());
        put(&mut data, 0x96, &0x2022_u16.to_le_bytes());
        put(&mut data, OPTIONAL, &0x20B_u16.to_le_bytes());
        put(&mut data, OPTIONAL + 4, &0x400_u32.to_le_bytes());
        put(&mut data, OPTIONAL + 8, &0x400_u32.to_le_bytes());
        put(&mut data, OPTIONAL + 16, &0x1000_u32.to_le_bytes());
        put(&mut data, OPTIONAL + 20, &0x1000_u32.to_le_bytes());
        put(&mut data, OPTIONAL + 24, &IMAGE_BASE.to_le_bytes());
        put(&mut data, OPTIONAL + 32, &0x1000_u32.to_le_bytes());
        put(&mut data, OPTIONAL + 36, &0x200_u32.to_le_bytes());
        put(&mut data, OPTIONAL + 56, &0x5000_u32.to_le_bytes());
        put(&mut data, OPTIONAL + 60, &0x400_u32.to_le_bytes());
        put(&mut data, OPTIONAL + 68, &3_u16.to_le_bytes());
        put(&mut data, OPTIONAL + 108, &16_u32.to_le_bytes());
        let sections = [
            (".text", 0x1000_u32, 0x100_u32, 0x6000_0020_u32),
            (".data", 0x2000, 0x300, 0xC000_0040),
            (".ab0", 0x3000, 0x180, 0x6000_0020),
            (".ab1", 0x4000, 0x180, 0xC000_0040),
        ];
        for (index, (name, rva, vsize, flags)) in sections.into_iter().enumerate() {
            let header = OPTIONAL + 0xF0 + index * 40;
            let raw = 0x400 + index * 0x200;
            put(&mut data, header, name.as_bytes());
            put(&mut data, header + 8, &vsize.to_le_bytes());
            put(&mut data, header + 12, &rva.to_le_bytes());
            put(&mut data, header + 16, &0x200_u32.to_le_bytes());
            put(
                &mut data,
                header + 20,
                &u32::try_from(raw).unwrap().to_le_bytes(),
            );
            put(&mut data, header + 36, &flags.to_le_bytes());
            data[raw..raw + 0x200].fill(0xA0 + u8::try_from(index).unwrap());
        }
        data
    }

    #[test]
    fn dead_section_is_stripped_and_validates() {
        let data = build();
        let selectors = vec![SectionSelector::Name(".ab1".to_owned())];
        let (bytes, summary) =
            strip_and_validate(&data, &selectors, &[]).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(summary.removed, vec![".ab1".to_owned()]);
        assert_eq!(summary.file_bytes_removed, 0x200);
        let pe = PeImage::parse(&bytes).expect("reparse");
        assert_eq!(pe.sections.len(), 3);
        // The helper only returns bytes that already passed loader validation.
        validate_for_loader(&bytes).expect("stripped image validates for the loader");
    }

    #[test]
    fn referenced_section_is_refused_and_input_is_untouched() {
        let mut data = build();
        // Move the entry point into the protector section `.ab0`.
        put(&mut data, OPTIONAL + 16, &0x3010_u32.to_le_bytes());
        let before = data.clone();
        let selectors = vec![SectionSelector::Name(".ab0".to_owned())];
        let failure = match strip_and_validate(&data, &selectors, &[]) {
            Err(failure) => failure,
            Ok(_) => panic!("a referenced section must be refused"),
        };
        match &failure {
            StripFailure::Refused(StripError::Dangling(refs)) => {
                assert_eq!(refs[0].source, "AddressOfEntryPoint");
                assert_eq!(refs[0].section, ".ab0");
            }
            other => panic!("expected a dangling-ref refusal, got {other}"),
        }
        assert_eq!(data, before, "a refused strip must not mutate the input");
    }

    #[test]
    fn clean_strip_stage_refuses_when_references_remain_leaving_output_unchanged() {
        // A reconstructed-but-still-protected image: the entry point lives in a
        // protector section, exactly as in a real partially devirtualized DLL.
        // The clean strip stage must refuse and change nothing, so the pipeline's
        // existing output decision is unaffected.
        let mut data = build();
        put(&mut data, OPTIONAL + 16, &0x3010_u32.to_le_bytes());
        match strip_protector_sections(&data).expect("stage runs") {
            CleanStripStage::Refused(StripFailure::Refused(StripError::Dangling(refs))) => {
                assert!(refs.iter().any(|r| r.source == "AddressOfEntryPoint"));
            }
            _ => panic!("the stage must refuse cleanly while references remain"),
        }
    }

    #[test]
    fn clean_strip_stage_removes_dead_protectors_when_nothing_references_them() {
        // `.ab0`/`.ab1` are non-standard and unreferenced; the entry stays in
        // `.text`. The stage removes both and returns a validated image.
        let data = build();
        match strip_protector_sections(&data).expect("stage runs") {
            CleanStripStage::Stripped { bytes, summary } => {
                let mut removed = summary.removed.clone();
                removed.sort();
                assert_eq!(removed, vec![".ab0".to_owned(), ".ab1".to_owned()]);
                let pe = PeImage::parse(&bytes).expect("reparse");
                assert_eq!(pe.sections.len(), 2);
                validate_for_loader(&bytes).expect("validates");
            }
            _ => panic!("dead protector sections must be removable"),
        }
    }

    #[test]
    fn protector_selectors_never_include_standard_sections() {
        let data = build();
        let selectors = protector_section_selectors(&data).expect("selectors");
        let names: Vec<_> = selectors
            .iter()
            .map(|selector| match selector {
                SectionSelector::Name(name) => name.clone(),
                SectionSelector::Index(_) => unreachable!(),
            })
            .collect();
        assert!(names.contains(&".ab0".to_owned()));
        assert!(names.contains(&".ab1".to_owned()));
        assert!(!names.contains(&".text".to_owned()));
        assert!(!names.contains(&".data".to_owned()));
    }
}
