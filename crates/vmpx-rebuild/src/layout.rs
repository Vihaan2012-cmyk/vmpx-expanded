//! Conventional file-layout serialization for an RVA-layout analysis image.
//!
//! Unpacking produces an image where every section's raw data starts at its
//! virtual address. That is a convenient analysis layout but not a file the
//! Windows loader can map. This module rewrites such an image into a
//! conventional PE file layout while preserving every RVA, so no directory,
//! relocation, or code address needs rewriting.

use thiserror::Error;

use crate::IMAGE_SCN_MEM_EXECUTE;
use vmpx_pe::{FileOffset, IMAGE_DIRECTORY_ENTRY_SECURITY, PeError, PeImage, PeKind, Rva, Section};

const COFF_HEADER_SIZE: u64 = 20;
const SECTION_HEADER_SIZE: u64 = 40;
const MAX_SECTIONS: usize = 96;
const IMAGE_FILE_RELOCS_STRIPPED: u16 = 0x0001;
const IMAGE_SCN_MEM_DISCARDABLE: u32 = 0x0200_0000;
const IMAGE_SCN_CNT_UNINITIALIZED_DATA: u32 = 0x0000_0080;

#[derive(Debug, Error)]
pub enum LayoutError {
    #[error("failed to parse {stage} PE: {source}")]
    Pe {
        stage: &'static str,
        #[source]
        source: PeError,
    },

    #[error("conventional relayout supports PE32+ only, got {0:?}")]
    UnsupportedKind(PeKind),

    #[error("{field} alignment {value:#X} is not a usable power of two")]
    InvalidAlignment { field: &'static str, value: u32 },

    #[error("image has {count} sections, exceeding the supported {max}")]
    TooManySections { count: usize, max: usize },

    #[error("section {name} raw range {start:#X}..{end:#X} exceeds the image")]
    SectionOutOfBounds { name: String, start: u64, end: u64 },

    #[error("section {name} virtual range overlaps section {other}")]
    OverlappingSections { name: String, other: String },

    #[error("image contains a security directory; certificate data is not relocatable")]
    SecurityDirectoryPresent,

    #[error("arithmetic overflow computing {what}")]
    Overflow { what: &'static str },

    #[error("output allocation of {requested} bytes failed")]
    Allocation { requested: usize },

    #[error("serialized output failed invariant: {0}")]
    OutputInvariant(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SectionLayout {
    pub name: String,
    pub virtual_address: Rva,
    pub virtual_size: u32,
    pub pointer_to_raw_data: FileOffset,
    pub size_of_raw_data: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RelayoutReport {
    pub file_size: u64,
    pub size_of_headers: u32,
    pub sections: Vec<SectionLayout>,
    pub zero_fill_sections: usize,
}

/// Rewrites an RVA-layout image into a conventional file layout.
///
/// Every section keeps its `VirtualAddress`, and all RVAs, directories, and
/// relative branches therefore remain valid. Only `PointerToRawData` and
/// `SizeOfRawData` change. Sections that are pure zero-fill (`SizeOfRawData`
/// already zero, or `IMAGE_SCN_CNT_UNINITIALIZED_DATA`) keep no raw data and
/// stay mapped as zero pages, which is what the loader expects for `.bss`.
///
/// # Errors
///
/// Returns a structured error for malformed input, unsupported PE kind,
/// unusable alignments, section ranges outside the image, overlapping mapped
/// ranges, or an emitted image that fails reparse invariants.
#[allow(clippy::too_many_lines)]
pub fn relayout_conventional(input: &[u8]) -> Result<(Vec<u8>, RelayoutReport), LayoutError> {
    let pe = PeImage::parse(input).map_err(|source| LayoutError::Pe {
        stage: "input",
        source,
    })?;
    if pe.optional.kind != PeKind::Pe32Plus {
        return Err(LayoutError::UnsupportedKind(pe.optional.kind));
    }
    let section_alignment = pe.optional.section_alignment;
    let file_alignment = pe.optional.file_alignment;
    for (field, value) in [
        ("SectionAlignment", section_alignment),
        ("FileAlignment", file_alignment),
    ] {
        if value == 0 || !value.is_power_of_two() {
            return Err(LayoutError::InvalidAlignment { field, value });
        }
    }
    if section_alignment < file_alignment {
        return Err(LayoutError::InvalidAlignment {
            field: "SectionAlignment",
            value: section_alignment,
        });
    }
    if pe.sections.len() > MAX_SECTIONS {
        return Err(LayoutError::TooManySections {
            count: pe.sections.len(),
            max: MAX_SECTIONS,
        });
    }
    if pe.directory(IMAGE_DIRECTORY_ENTRY_SECURITY).is_some() {
        return Err(LayoutError::SecurityDirectoryPresent);
    }

    let headers_end = section_table_end(&pe)?;
    let size_of_headers = align_u32(
        u32::try_from(headers_end).map_err(|_| LayoutError::Overflow {
            what: "SizeOfHeaders",
        })?,
        file_alignment,
        "SizeOfHeaders",
    )?;

    // Raw payloads are emitted in ascending virtual address order, which keeps
    // the resulting file layout monotonic and easy to verify.
    let mut order = (0..pe.sections.len()).collect::<Vec<_>>();
    order.sort_by_key(|index| pe.sections[*index].virtual_address.0);

    let mut raw_cursor = u64::from(size_of_headers);
    let mut layouts = Vec::with_capacity(pe.sections.len());
    let mut payloads = Vec::with_capacity(pe.sections.len());
    let mut zero_fill_sections = 0;
    for index in order {
        let section = &pe.sections[index];
        let is_zero_fill = section.size_of_raw_data == 0
            || section.characteristics & IMAGE_SCN_CNT_UNINITIALIZED_DATA != 0;
        let (payload, raw_size, pointer) = if is_zero_fill {
            zero_fill_sections += 1;
            (Vec::new(), 0_u32, FileOffset(0))
        } else {
            let start = section.pointer_to_raw_data.0;
            let end = start
                .checked_add(u64::from(section.size_of_raw_data))
                .ok_or(LayoutError::Overflow {
                    what: "section raw range",
                })?;
            let start_index =
                usize::try_from(start).map_err(|_| LayoutError::Overflow { what: "raw start" })?;
            let end_index =
                usize::try_from(end).map_err(|_| LayoutError::Overflow { what: "raw end" })?;
            let data = input.get(start_index..end_index).ok_or_else(|| {
                LayoutError::SectionOutOfBounds {
                    name: section.name.clone(),
                    start,
                    end,
                }
            })?;
            let pointer = FileOffset(raw_cursor);
            let raw_size = align_u32(section.size_of_raw_data, file_alignment, "section raw size")?;
            let padded = u64::from(raw_size);
            raw_cursor = raw_cursor
                .checked_add(padded)
                .ok_or(LayoutError::Overflow { what: "raw cursor" })?;
            let mut payload = Vec::new();
            payload
                .try_reserve_exact(usize::try_from(raw_size).map_err(|_| {
                    LayoutError::Overflow {
                        what: "section raw size",
                    }
                })?)
                .map_err(|_| LayoutError::Allocation {
                    requested: usize::try_from(raw_size).unwrap_or(usize::MAX),
                })?;
            payload.extend_from_slice(data);
            payload.resize(
                usize::try_from(raw_size).map_err(|_| LayoutError::Overflow {
                    what: "section raw size",
                })?,
                0,
            );
            (payload, raw_size, pointer)
        };
        layouts.push(SectionLayout {
            name: section.name.clone(),
            virtual_address: section.virtual_address,
            virtual_size: section.virtual_size,
            pointer_to_raw_data: pointer,
            size_of_raw_data: raw_size,
        });
        payloads.push(payload);
    }

    // Mapped ranges must not overlap; the loader would otherwise alias them.
    let mut sorted = layouts
        .iter()
        .map(|layout| {
            let span = layout.virtual_size.max(layout.size_of_raw_data);
            (
                layout.virtual_address.0,
                u64::from(layout.virtual_address.0) + u64::from(span),
                layout.name.clone(),
            )
        })
        .collect::<Vec<_>>();
    sorted.sort_by_key(|entry| entry.0);
    for window in sorted.windows(2) {
        let [left, right] = window else { continue };
        if u64::from(right.0) < left.1 {
            return Err(LayoutError::OverlappingSections {
                name: right.2.clone(),
                other: left.2.clone(),
            });
        }
    }

    let file_size = raw_cursor;
    let file_size_usize =
        usize::try_from(file_size).map_err(|_| LayoutError::Overflow { what: "file size" })?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(file_size_usize)
        .map_err(|_| LayoutError::Allocation {
            requested: file_size_usize,
        })?;
    output.resize(file_size_usize, 0);
    let header_length = usize::try_from(size_of_headers).map_err(|_| LayoutError::Overflow {
        what: "SizeOfHeaders",
    })?;
    let header_source =
        input
            .get(..header_length.min(input.len()))
            .ok_or(LayoutError::Overflow {
                what: "header bytes",
            })?;
    output
        .get_mut(..header_source.len())
        .ok_or(LayoutError::Overflow {
            what: "header destination",
        })?
        .copy_from_slice(header_source);

    let optional_offset = pe
        .nt_headers_offset
        .checked_add(4 + COFF_HEADER_SIZE)
        .ok_or(LayoutError::Overflow {
            what: "optional-header offset",
        })?;
    write_u32(
        &mut output,
        optional_offset
            .checked_add(60)
            .ok_or(LayoutError::Overflow {
                what: "SizeOfHeaders field",
            })?,
        size_of_headers,
        "SizeOfHeaders",
    )?;

    let section_table_offset = optional_offset
        .checked_add(u64::from(pe.coff.size_of_optional_header))
        .ok_or(LayoutError::Overflow {
            what: "section-table offset",
        })?;
    for (index, layout) in layouts.iter().enumerate() {
        let entry_offset = section_table_offset
            .checked_add(
                u64::try_from(index)
                    .map_err(|_| LayoutError::Overflow {
                        what: "section index",
                    })?
                    .checked_mul(SECTION_HEADER_SIZE)
                    .ok_or(LayoutError::Overflow {
                        what: "section-header offset",
                    })?,
            )
            .ok_or(LayoutError::Overflow {
                what: "section-header offset",
            })?;
        write_u32(
            &mut output,
            entry_offset.checked_add(16).ok_or(LayoutError::Overflow {
                what: "SizeOfRawData field",
            })?,
            layout.size_of_raw_data,
            "SizeOfRawData",
        )?;
        write_u32(
            &mut output,
            entry_offset.checked_add(20).ok_or(LayoutError::Overflow {
                what: "PointerToRawData field",
            })?,
            u32::try_from(layout.pointer_to_raw_data.0).map_err(|_| LayoutError::Overflow {
                what: "PointerToRawData",
            })?,
            "PointerToRawData",
        )?;
    }

    for (layout, payload) in layouts.iter().zip(&payloads) {
        if payload.is_empty() {
            continue;
        }
        let start =
            usize::try_from(layout.pointer_to_raw_data.0).map_err(|_| LayoutError::Overflow {
                what: "section payload offset",
            })?;
        let end = start
            .checked_add(payload.len())
            .ok_or(LayoutError::Overflow {
                what: "section payload end",
            })?;
        output
            .get_mut(start..end)
            .ok_or(LayoutError::Overflow {
                what: "section payload destination",
            })?
            .copy_from_slice(payload);
    }

    // Discardable sections are not required at load time; the layout above
    // keeps them for evidence, so nothing needs to be dropped here.

    let reparsed = PeImage::parse(&output).map_err(|source| LayoutError::Pe {
        stage: "output",
        source,
    })?;
    if reparsed.sections.len() != layouts.len() {
        return Err(LayoutError::OutputInvariant(
            "section count changed after relayout".to_owned(),
        ));
    }
    if reparsed.optional.size_of_headers != size_of_headers {
        return Err(LayoutError::OutputInvariant(
            "SizeOfHeaders differs after relayout".to_owned(),
        ));
    }
    for (original, emitted) in pe.sections.iter().zip(&reparsed.sections) {
        if original.virtual_address != emitted.virtual_address {
            return Err(LayoutError::OutputInvariant(format!(
                "section {} moved from {} to {}",
                original.name, original.virtual_address, emitted.virtual_address
            )));
        }
        if original.virtual_size != emitted.virtual_size {
            return Err(LayoutError::OutputInvariant(format!(
                "section {} virtual size changed",
                original.name
            )));
        }
    }
    // Every section's bytes must survive the move unchanged.
    for (original, emitted) in pe.sections.iter().zip(&reparsed.sections) {
        if original.size_of_raw_data == 0
            || original.characteristics & IMAGE_SCN_CNT_UNINITIALIZED_DATA != 0
        {
            continue;
        }
        let original_start =
            usize::try_from(original.pointer_to_raw_data.0).map_err(|_| LayoutError::Overflow {
                what: "original raw",
            })?;
        let original_end = original_start
            .checked_add(usize::try_from(original.size_of_raw_data).map_err(|_| {
                LayoutError::Overflow {
                    what: "original raw size",
                }
            })?)
            .ok_or(LayoutError::Overflow {
                what: "original raw end",
            })?;
        let emitted_start =
            usize::try_from(emitted.pointer_to_raw_data.0).map_err(|_| LayoutError::Overflow {
                what: "emitted raw",
            })?;
        let emitted_end = emitted_start
            .checked_add(usize::try_from(emitted.size_of_raw_data).map_err(|_| {
                LayoutError::Overflow {
                    what: "emitted raw size",
                }
            })?)
            .ok_or(LayoutError::Overflow {
                what: "emitted raw end",
            })?;
        let expected = input
            .get(original_start..original_end)
            .ok_or(LayoutError::Overflow {
                what: "original section bytes",
            })?;
        let actual = output
            .get(emitted_start..emitted_end)
            .ok_or(LayoutError::Overflow {
                what: "emitted section bytes",
            })?;
        if actual.get(..expected.len()) != Some(expected) {
            return Err(LayoutError::OutputInvariant(format!(
                "section {} raw bytes changed during relayout",
                original.name
            )));
        }
    }

    Ok((
        output,
        RelayoutReport {
            file_size,
            size_of_headers,
            sections: layouts,
            zero_fill_sections,
        },
    ))
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LoaderValidationReport {
    pub sections: usize,
    pub import_modules: usize,
    pub import_symbols: usize,
    pub exports: usize,
    pub tls_callbacks: usize,
    pub relocation_blocks: usize,
    pub relocation_entries: usize,
    pub entry_executable: bool,
}

#[derive(Debug, Error)]
pub enum LoaderValidationError {
    #[error("failed to parse PE: {0}")]
    Pe(#[from] PeError),

    #[error("{what} RVA {rva:?} is not backed by file data")]
    NotFileBacked { what: &'static str, rva: Rva },

    #[error("{what} RVA {rva:?} is not inside an executable section")]
    NotExecutable { what: &'static str, rva: Rva },

    #[error("import module {index} has an implausible name {name:?}")]
    ImplausibleModule { index: usize, name: String },

    #[error("import {module} symbol {index} has no name")]
    MissingSymbolName { module: String, index: usize },

    #[error("arithmetic overflow while validating {what}")]
    Overflow { what: &'static str },
}

/// Validates every loader-consumed structure in a candidate output image.
///
/// A structurally parseable PE can still fail to load: an entry point in a
/// zero-fill section, an import name outside file data, or a relocation target
/// that is not mapped will all fault during loading. This walks each directory
/// the loader itself walks and requires every referenced byte to be present.
///
/// # Errors
///
/// Returns the first structure that a loader could not resolve.
#[allow(clippy::too_many_lines)]
pub fn validate_for_loader(input: &[u8]) -> Result<LoaderValidationReport, LoaderValidationError> {
    let pe = PeImage::parse(input)?;
    let executable_section = |rva: Rva| {
        pe.sections.iter().any(|section| {
            section.characteristics & IMAGE_SCN_MEM_EXECUTE != 0 && section.contains_mapped_rva(rva)
        })
    };
    let entry = pe.optional.address_of_entry_point;
    if !executable_section(entry) {
        return Err(LoaderValidationError::NotExecutable {
            what: "entry point",
            rva: entry,
        });
    }
    pe.rva_to_file_offset(entry)
        .map_err(|_| LoaderValidationError::NotFileBacked {
            what: "entry point",
            rva: entry,
        })?;

    let report = pe.report();
    let mut import_symbols = 0_usize;
    for (index, module) in report.imports.iter().enumerate() {
        if !module.name.to_ascii_lowercase().ends_with(".dll") {
            return Err(LoaderValidationError::ImplausibleModule {
                index,
                name: module.name.clone(),
            });
        }
        for (symbol_index, symbol) in module.entries.iter().enumerate() {
            if symbol.name.is_none() && symbol.ordinal.is_none() {
                return Err(LoaderValidationError::MissingSymbolName {
                    module: module.name.clone(),
                    index: symbol_index,
                });
            }
            import_symbols += 1;
        }
    }

    let mut tls_callbacks = 0_usize;
    if let Some(tls) = &report.tls {
        let callback_rva = pe
            .optional
            .image_base
            .checked_rva_of(tls.address_of_callbacks)
            .ok_or(LoaderValidationError::NotFileBacked {
                what: "TLS callback array",
                rva: Rva(0),
            })?;
        pe.rva_to_file_offset(callback_rva)
            .map_err(|_| LoaderValidationError::NotFileBacked {
                what: "TLS callback array",
                rva: callback_rva,
            })?;
        for callback in &tls.callbacks {
            let rva = pe.optional.image_base.checked_rva_of(*callback).ok_or(
                LoaderValidationError::NotFileBacked {
                    what: "TLS callback",
                    rva: Rva(0),
                },
            )?;
            pe.rva_to_file_offset(rva)
                .map_err(|_| LoaderValidationError::NotFileBacked {
                    what: "TLS callback",
                    rva,
                })?;
            tls_callbacks += 1;
        }
    }

    let mut relocation_blocks = 0_usize;
    let mut relocation_entries = 0_usize;
    for block in &report.relocations {
        relocation_blocks += 1;
        for entry in &block.entries {
            let Some(rva) = entry.target_rva else {
                continue;
            };
            relocation_entries += 1;
            pe.rva_to_file_offset(rva)
                .map_err(|_| LoaderValidationError::NotFileBacked {
                    what: "relocation target",
                    rva,
                })?;
        }
    }

    let exports = report
        .exports
        .as_ref()
        .map_or(0, |exports| exports.symbols.len());
    if let Some(exports) = &report.exports {
        for symbol in &exports.symbols {
            if symbol.forwarder.is_some() {
                continue;
            }
            pe.rva_to_file_offset(symbol.rva).map_err(|_| {
                LoaderValidationError::NotFileBacked {
                    what: "export",
                    rva: symbol.rva,
                }
            })?;
        }
    }

    Ok(LoaderValidationReport {
        sections: pe.sections.len(),
        import_modules: report.imports.len(),
        import_symbols,
        exports,
        tls_callbacks,
        relocation_blocks,
        relocation_entries,
        entry_executable: true,
    })
}

/// True when the image already stores each section at its virtual address.
#[must_use]
pub fn is_rva_layout(pe: &PeImage<'_>) -> bool {
    !pe.sections.is_empty()
        && pe
            .sections
            .iter()
            .filter(|section| section.size_of_raw_data != 0)
            .all(|section| section.pointer_to_raw_data.0 == u64::from(section.virtual_address.0))
}

/// True when no section carries raw data that a loader would map.
#[must_use]
pub fn is_conventional_layout(pe: &PeImage<'_>) -> bool {
    pe.sections
        .iter()
        .filter(|section| section.size_of_raw_data != 0)
        .all(|section| section.pointer_to_raw_data.0 != u64::from(section.virtual_address.0))
}

fn section_table_end(pe: &PeImage<'_>) -> Result<u64, LayoutError> {
    let optional_offset = pe
        .nt_headers_offset
        .0
        .checked_add(4 + COFF_HEADER_SIZE)
        .ok_or(LayoutError::Overflow {
            what: "optional-header offset",
        })?;
    optional_offset
        .checked_add(u64::from(pe.coff.size_of_optional_header))
        .and_then(|offset| {
            offset.checked_add(
                u64::try_from(pe.sections.len())
                    .ok()
                    .and_then(|count| count.checked_mul(SECTION_HEADER_SIZE))
                    .unwrap_or(u64::MAX),
            )
        })
        .ok_or(LayoutError::Overflow {
            what: "section-table end",
        })
}

fn align_u32(value: u32, alignment: u32, what: &'static str) -> Result<u32, LayoutError> {
    let mask = alignment
        .checked_sub(1)
        .ok_or(LayoutError::InvalidAlignment {
            field: what,
            value: alignment,
        })?;
    value
        .checked_add(mask)
        .map(|value| value & !mask)
        .ok_or(LayoutError::Overflow { what })
}

fn write_u32(
    output: &mut [u8],
    offset: FileOffset,
    value: u32,
    what: &'static str,
) -> Result<(), LayoutError> {
    let start = usize::try_from(offset.0).map_err(|_| LayoutError::Overflow { what })?;
    let end = start.checked_add(4).ok_or(LayoutError::Overflow { what })?;
    let destination = output
        .get_mut(start..end)
        .ok_or(LayoutError::Overflow { what })?;
    destination.copy_from_slice(&value.to_le_bytes());
    Ok(())
}

/// Removes the `IMAGE_FILE_RELOCS_STRIPPED` ambiguity introduced by keeping
/// protector sections: the flag is cleared only when a relocation directory
/// exists, so the loader still applies base relocations.
///
/// # Errors
///
/// Returns a structured error when the image cannot be reparsed.
pub fn normalize_characteristics(input: &[u8]) -> Result<Vec<u8>, LayoutError> {
    let pe = PeImage::parse(input).map_err(|source| LayoutError::Pe {
        stage: "input",
        source,
    })?;
    let Some(relocations) = pe.directory(vmpx_pe::IMAGE_DIRECTORY_ENTRY_BASERELOC) else {
        return Ok(input.to_vec());
    };
    if relocations.size == 0 {
        return Ok(input.to_vec());
    }
    let mut output = input.to_vec();
    let characteristics_offset =
        pe.nt_headers_offset
            .0
            .checked_add(4 + 18)
            .ok_or(LayoutError::Overflow {
                what: "Characteristics offset",
            })?;
    let start = usize::try_from(characteristics_offset).map_err(|_| LayoutError::Overflow {
        what: "Characteristics offset",
    })?;
    let end = start.checked_add(2).ok_or(LayoutError::Overflow {
        what: "Characteristics field",
    })?;
    let field = output.get_mut(start..end).ok_or(LayoutError::Overflow {
        what: "Characteristics field",
    })?;
    let current = u16::from_le_bytes([field[0], field[1]]);
    let updated = current & !IMAGE_FILE_RELOCS_STRIPPED;
    field.copy_from_slice(&updated.to_le_bytes());
    Ok(output)
}

/// Sections a loader does not need, kept in the analysis image as evidence.
#[must_use]
pub fn is_discardable(section: &Section) -> bool {
    section.characteristics & IMAGE_SCN_MEM_DISCARDABLE != 0
}

#[cfg(test)]
mod tests {
    use super::*;

    const IMAGE_BASE: u64 = 0x1_4000_0000;

    fn pe_with_sections(sections: &[(&str, u32, u32, u32, u32, u32)]) -> Vec<u8> {
        // (name, virtual_address, virtual_size, pointer_to_raw_data, raw_size, characteristics)
        let count = sections.len();
        let headers_end = 0x80 + 24 + 0xF0 + count * 40;
        let size_of_headers = headers_end.next_multiple_of(0x200);
        let total = sections
            .iter()
            .map(|section| {
                usize::try_from(section.3).unwrap_or(0) + usize::try_from(section.4).unwrap_or(0)
            })
            .max()
            .unwrap_or(0)
            .max(size_of_headers);
        let mut data = vec![0_u8; total];
        data[0..2].copy_from_slice(&0x5A4D_u16.to_le_bytes());
        data[0x3C..0x40].copy_from_slice(&0x80_u32.to_le_bytes());
        data[0x80..0x84].copy_from_slice(&0x0000_4550_u32.to_le_bytes());
        let coff = 0x84;
        data[coff..coff + 2].copy_from_slice(&0x8664_u16.to_le_bytes());
        data[coff + 2..coff + 4].copy_from_slice(&u16::try_from(count).unwrap_or(0).to_le_bytes());
        data[coff + 16..coff + 18].copy_from_slice(&0xF0_u16.to_le_bytes());
        data[coff + 18..coff + 20].copy_from_slice(&0x2022_u16.to_le_bytes());
        let optional = coff + 20;
        data[optional..optional + 2].copy_from_slice(&0x20B_u16.to_le_bytes());
        data[optional + 16..optional + 20].copy_from_slice(&0x1000_u32.to_le_bytes());
        data[optional + 24..optional + 32].copy_from_slice(&IMAGE_BASE.to_le_bytes());
        data[optional + 32..optional + 36].copy_from_slice(&0x1000_u32.to_le_bytes());
        data[optional + 36..optional + 40].copy_from_slice(&0x200_u32.to_le_bytes());
        let size_of_image = sections
            .iter()
            .map(|section| section.1 + section.2)
            .max()
            .unwrap_or(0x1000)
            .next_multiple_of(0x1000);
        data[optional + 56..optional + 60].copy_from_slice(&size_of_image.to_le_bytes());
        data[optional + 60..optional + 64]
            .copy_from_slice(&u32::try_from(size_of_headers).unwrap_or(0).to_le_bytes());
        data[optional + 68..optional + 70].copy_from_slice(&3_u16.to_le_bytes());
        data[optional + 108..optional + 112].copy_from_slice(&16_u32.to_le_bytes());
        let table = optional + 0xF0;
        for (index, (name, va, vsize, raw, raw_size, characteristics)) in
            sections.iter().enumerate()
        {
            let entry = table + index * 40;
            let bytes = name.as_bytes();
            data[entry..entry + bytes.len().min(8)].copy_from_slice(&bytes[..bytes.len().min(8)]);
            data[entry + 8..entry + 12].copy_from_slice(&vsize.to_le_bytes());
            data[entry + 12..entry + 16].copy_from_slice(&va.to_le_bytes());
            data[entry + 16..entry + 20].copy_from_slice(&raw_size.to_le_bytes());
            data[entry + 20..entry + 24].copy_from_slice(&raw.to_le_bytes());
            data[entry + 36..entry + 40].copy_from_slice(&characteristics.to_le_bytes());
            if *raw_size != 0 {
                let start = usize::try_from(*raw).unwrap_or(0);
                let end = start.saturating_add(usize::try_from(*raw_size).unwrap_or(0));
                if let Some(window) = data.get_mut(start..end.min(total)) {
                    for (offset, byte) in window.iter_mut().enumerate() {
                        *byte = u8::try_from((usize::try_from(*va).unwrap_or(0) + offset) & 0xFF)
                            .unwrap_or(0);
                    }
                }
            }
        }
        data
    }

    #[test]
    fn relayouts_rva_image_into_conventional_file_layout() {
        let image = pe_with_sections(&[
            (".text", 0x1000, 0x400, 0x1000, 0x400, 0x6000_0020),
            (".data", 0x2000, 0x200, 0x2000, 0x200, 0xC000_0040),
        ]);
        let pe = PeImage::parse(&image).expect("valid RVA-layout PE");
        assert!(is_rva_layout(&pe), "fixture must start in RVA layout");
        let (output, report) = relayout_conventional(&image).expect("relayout succeeds");
        assert_eq!(report.sections.len(), 2);
        assert_eq!(report.zero_fill_sections, 0);
        let reparsed = PeImage::parse(&output).expect("output reparses");
        assert!(is_conventional_layout(&reparsed));
        for (original, emitted) in pe.sections.iter().zip(&reparsed.sections) {
            assert_eq!(original.virtual_address, emitted.virtual_address);
            assert_eq!(original.virtual_size, emitted.virtual_size);
            let original_start = usize::try_from(original.pointer_to_raw_data.0).unwrap();
            let emitted_start = usize::try_from(emitted.pointer_to_raw_data.0).unwrap();
            let size = usize::try_from(original.size_of_raw_data).unwrap();
            assert_eq!(
                &image[original_start..original_start + size],
                &output[emitted_start..emitted_start + size]
            );
        }
        assert!(report.file_size < u64::try_from(image.len()).unwrap());
    }

    #[test]
    fn zero_fill_sections_keep_no_raw_data() {
        let image = pe_with_sections(&[
            (".text", 0x1000, 0x400, 0x1000, 0x400, 0x6000_0020),
            (".bss", 0x2000, 0x8000, 0, 0, 0xC000_0080),
        ]);
        let (output, report) = relayout_conventional(&image).expect("relayout succeeds");
        assert_eq!(report.zero_fill_sections, 1);
        let reparsed = PeImage::parse(&output).expect("output reparses");
        let bss = &reparsed.sections[1];
        assert_eq!(bss.size_of_raw_data, 0);
        assert_eq!(bss.virtual_size, 0x8000);
    }

    /// The load-bearing property: a conventional relayout must not change any
    /// byte of what the Windows loader maps into memory, apart from the section
    /// header table itself. `SizeOfRawData` and `PointerToRawData` are loader
    /// metadata; every mapped byte of section content must be identical.
    fn loaded_image_without_section_table(bytes: &[u8]) -> Vec<u8> {
        let pe = PeImage::parse(bytes).expect("valid PE");
        let mut image = vec![0_u8; pe.optional.size_of_image as usize];
        let headers = usize::try_from(pe.optional.size_of_headers).expect("headers");
        image[..headers].copy_from_slice(&bytes[..headers]);
        for section in &pe.sections {
            let span = section.mapped_span() as usize;
            let start = section.virtual_address.0 as usize;
            let mut chunk = vec![0_u8; span];
            if section.size_of_raw_data != 0 {
                let raw = section.pointer_to_raw_data.0 as usize;
                let available = (section.size_of_raw_data as usize).min(bytes.len() - raw);
                chunk[..available].copy_from_slice(&bytes[raw..raw + available]);
            }
            image[start..start + span].copy_from_slice(&chunk);
        }
        let table = pe.nt_headers_offset.0 as usize
            + 4
            + COFF_HEADER_SIZE as usize
            + usize::try_from(pe.coff.size_of_optional_header).expect("optional header");
        let table_end = table + pe.sections.len() * SECTION_HEADER_SIZE as usize;
        for byte in &mut image[table..table_end] {
            *byte = 0;
        }
        image
    }

    #[test]
    fn relayout_preserves_the_loaded_memory_image_exactly() {
        let image = pe_with_sections(&[
            (".text", 0x1000, 0x400, 0x1000, 0x400, 0x6000_0020),
            (".data", 0x2000, 0x300, 0x2000, 0x300, 0xC000_0040),
            (".bss", 0x3000, 0x2000, 0, 0, 0xC000_0080),
            (".reloc", 0x5000, 0x80, 0x5000, 0x80, 0x4200_0040),
        ]);
        let (output, _) = relayout_conventional(&image).expect("relayout succeeds");
        assert_eq!(
            loaded_image_without_section_table(&image),
            loaded_image_without_section_table(&output)
        );
    }

    #[test]
    fn relayout_keeps_every_directory_file_backed() {
        let image = pe_with_sections(&[
            (".text", 0x1000, 0x400, 0x1000, 0x400, 0x6000_0020),
            (".rdata", 0x2000, 0x200, 0x2000, 0x200, 0x4000_0040),
            (".reloc", 0x3000, 0x40, 0x3000, 0x40, 0x4200_0040),
            (".rsrc", 0x4000, 0x100, 0x4000, 0x100, 0x4000_0040),
        ]);
        let mut image = image;
        // Point the resource directory at .rsrc so the check has real work.
        let optional = 0x84 + 20;
        image[optional + 112 + 2 * 8..optional + 112 + 2 * 8 + 4]
            .copy_from_slice(&0x4000_u32.to_le_bytes());
        image[optional + 112 + 2 * 8 + 4..optional + 112 + 2 * 8 + 8]
            .copy_from_slice(&0x100_u32.to_le_bytes());
        let (output, _) = relayout_conventional(&image).expect("relayout succeeds");
        let reparsed = PeImage::parse(&output).expect("output reparses");
        for directory in reparsed
            .directories
            .iter()
            .filter(|directory| directory.is_present())
        {
            assert!(
                reparsed.rva_to_file_offset(Rva(directory.address)).is_ok(),
                "{} directory must stay file-backed",
                directory.name
            );
        }
    }

    #[test]
    fn overlapping_sections_are_rejected() {
        let image = pe_with_sections(&[
            (".text", 0x1000, 0x2000, 0x1000, 0x200, 0x6000_0020),
            (".data", 0x1800, 0x200, 0x1800, 0x200, 0xC000_0040),
        ]);
        let error = relayout_conventional(&image).expect_err("overlap must be rejected");
        assert!(matches!(error, LayoutError::OverlappingSections { .. }));
    }
}
