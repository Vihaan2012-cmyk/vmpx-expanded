mod import_table;
mod layout;
mod reintegrate;

pub use import_table::{
    ImportModuleSpec, ImportReconstructionPlan, ImportSlotRemap, ImportTableError, ImportTarget,
    SerializedDirectoryRange, SerializedImportTable, install_import_directories,
    plan_import_reconstruction, serialize_import_table,
};
pub use layout::{
    LayoutError, LoaderValidationError, LoaderValidationReport, RelayoutReport, SectionLayout,
    is_conventional_layout, is_discardable, is_rva_layout, normalize_characteristics,
    relayout_conventional, validate_for_loader,
};
pub use reintegrate::{
    ReintegrationReport, ReintegrationRequest, redirect_vm_stub, reintegrate, set_tls_callback,
};

use thiserror::Error;
use vmpx_pe::{
    DirectoryAddressKind, FileOffset, IMAGE_DIRECTORY_ENTRY_SECURITY, PeError, PeImage, PeKind, Rva,
};

const COFF_HEADER_SIZE: u64 = 20;
const SECTION_HEADER_SIZE: u64 = 40;
const MAX_SECTIONS: u16 = 1_024;
const IMAGE_SCN_CNT_CODE: u32 = 0x0000_0020;
const IMAGE_SCN_CNT_INITIALIZED_DATA: u32 = 0x0000_0040;
const IMAGE_SCN_MEM_EXECUTE: u32 = 0x2000_0000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AppendSection<'a> {
    pub name: [u8; 8],
    pub data: &'a [u8],
    pub characteristics: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AppendedSection {
    pub name: [u8; 8],
    pub rva: Rva,
    pub raw_offset: FileOffset,
    pub virtual_size: u32,
    pub raw_size: u32,
    pub characteristics: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RebuiltPe {
    pub bytes: Vec<u8>,
    pub section: AppendedSection,
}

#[derive(Debug, Error)]
pub enum RebuildError {
    #[error("failed to parse {stage} PE: {source}")]
    Pe {
        stage: &'static str,
        #[source]
        source: PeError,
    },

    #[error("append-section serialization supports PE32+ only, got {0:?}")]
    UnsupportedKind(PeKind),

    #[error("{field} alignment {value:#X} is not a nonzero power of two")]
    InvalidAlignment { field: &'static str, value: u32 },

    #[error(
        "incompatible PE alignments: SectionAlignment={section_alignment:#X}, FileAlignment={file_alignment:#X}"
    )]
    IncompatibleAlignments {
        section_alignment: u32,
        file_alignment: u32,
    },

    #[error("cannot append an empty section")]
    EmptySection,

    #[error("section payload is too large to represent in PE32+")]
    PayloadTooLarge,

    #[error("section count {0} cannot be increased safely")]
    SectionCount(u16),

    #[error("{what} arithmetic overflows")]
    Overflow { what: &'static str },

    #[error("new section header ending at {required_end} exceeds header capacity {capacity_end}")]
    HeaderCapacity {
        required_end: FileOffset,
        capacity_end: FileOffset,
    },

    #[error("new section header would overwrite nonzero byte at {offset}")]
    HeaderSlackOccupied { offset: FileOffset },

    #[error(
        "new section header overlaps data directory {index} ({name}) at file offset {offset}, size {size:#X}"
    )]
    HeaderDirectoryOverlap {
        index: usize,
        name: &'static str,
        offset: FileOffset,
        size: u32,
    },

    #[error(
        "cannot append a section while the PE security directory is present at file offset {offset}, size {size:#X}; certificate-table relocation and signature removal are not implemented"
    )]
    SecurityDirectoryPresent { offset: FileOffset, size: u32 },

    #[error("output allocation of {requested} bytes failed")]
    Allocation { requested: usize },

    #[error("entry point RVA {rva:?} is not inside an executable section")]
    EntryPointNotExecutable { rva: Rva },

    #[error("entry point RVA {rva:?} is not backed by file data")]
    EntryPointNotFileBacked { rva: Rva },

    #[error("serialized output failed invariant: {0}")]
    OutputInvariant(String),

    #[error("no TLS directory is present")]
    NoTlsDirectory,

    #[error("TLS callback index {index} is out of range for {count} callback(s)")]
    TlsCallbackIndex { index: usize, count: usize },

    #[error("TLS callback target RVA {rva:?} is not inside an executable section")]
    TlsCallbackNotExecutable { rva: Rva },

    #[error("TLS callback target RVA {rva:?} is not backed by file data")]
    TlsCallbackNotFileBacked { rva: Rva },

    #[error("VM stub RVA {rva:?} is not inside an executable section")]
    VmStubNotExecutable { rva: Rva },

    #[error("VM stub RVA {rva:?} is not backed by file data")]
    VmStubNotFileBacked { rva: Rva },

    #[error("VM stub RVA {rva:?} has only {available} file-backed byte(s) left; a jmp needs 5")]
    VmStubNoRoom { rva: Rva, available: u32 },

    #[error("VM stub redirect target RVA {rva:?} is not inside an executable section")]
    VmStubTargetNotExecutable { rva: Rva },

    #[error("jmp from {stub:?} to {target:?} does not fit a rel32 displacement")]
    VmStubRel32Range { stub: Rva, target: Rva },
}

fn align_u32(value: u32, alignment: u32, what: &'static str) -> Result<u32, RebuildError> {
    let mask = alignment
        .checked_sub(1)
        .ok_or(RebuildError::InvalidAlignment {
            field: what,
            value: alignment,
        })?;
    value
        .checked_add(mask)
        .map(|value| value & !mask)
        .ok_or(RebuildError::Overflow { what })
}

fn align_u64(value: u64, alignment: u32, what: &'static str) -> Result<u64, RebuildError> {
    let mask = u64::from(
        alignment
            .checked_sub(1)
            .ok_or(RebuildError::InvalidAlignment {
                field: what,
                value: alignment,
            })?,
    );
    value
        .checked_add(mask)
        .map(|value| value & !mask)
        .ok_or(RebuildError::Overflow { what })
}

fn align_rva(value: Rva, alignment: u32, what: &'static str) -> Result<Rva, RebuildError> {
    align_u32(value.0, alignment, what).map(Rva)
}

fn align_file_offset(
    value: FileOffset,
    alignment: u32,
    what: &'static str,
) -> Result<FileOffset, RebuildError> {
    align_u64(value.0, alignment, what).map(FileOffset)
}

fn checked_usize(value: u64, what: &'static str) -> Result<usize, RebuildError> {
    usize::try_from(value).map_err(|_| RebuildError::Overflow { what })
}

pub(crate) fn write_bytes(
    output: &mut [u8],
    offset: FileOffset,
    value: &[u8],
    what: &'static str,
) -> Result<(), RebuildError> {
    let start = checked_usize(offset.0, what)?;
    let end = start
        .checked_add(value.len())
        .ok_or(RebuildError::Overflow { what })?;
    let destination = output
        .get_mut(start..end)
        .ok_or(RebuildError::Overflow { what })?;
    destination.copy_from_slice(value);
    Ok(())
}

fn write_u16(
    output: &mut [u8],
    offset: FileOffset,
    value: u16,
    what: &'static str,
) -> Result<(), RebuildError> {
    write_bytes(output, offset, &value.to_le_bytes(), what)
}

fn write_u32(
    output: &mut [u8],
    offset: FileOffset,
    value: u32,
    what: &'static str,
) -> Result<(), RebuildError> {
    write_bytes(output, offset, &value.to_le_bytes(), what)
}

pub(crate) fn add_offset(
    base: FileOffset,
    amount: u64,
    what: &'static str,
) -> Result<FileOffset, RebuildError> {
    base.checked_add(amount)
        .ok_or(RebuildError::Overflow { what })
}

fn ranges_overlap(left_start: u64, left_end: u64, right_start: u64, right_end: u64) -> bool {
    left_start < right_end && right_start < left_end
}

fn next_section_rva_for(pe: &PeImage<'_>) -> Result<Rva, RebuildError> {
    if pe.optional.kind != PeKind::Pe32Plus {
        return Err(RebuildError::UnsupportedKind(pe.optional.kind));
    }
    let section_alignment = pe.optional.section_alignment;
    if section_alignment == 0 || !section_alignment.is_power_of_two() {
        return Err(RebuildError::InvalidAlignment {
            field: "SectionAlignment",
            value: section_alignment,
        });
    }
    let mut mapped_end = pe.optional.size_of_image.max(pe.optional.size_of_headers);
    for section in &pe.sections {
        let end = section
            .virtual_address
            .0
            .checked_add(section.mapped_span())
            .ok_or(RebuildError::Overflow {
                what: "existing section virtual end",
            })?;
        mapped_end = mapped_end.max(end);
    }
    align_rva(Rva(mapped_end), section_alignment, "new section RVA")
}

/// Computes the RVA that [`append_section`] will use for its new section.
/// This permits callers to construct RVA-sensitive payloads such as import
/// tables and branch fixups before performing the guarded append transaction.
///
/// The returned address is only a placement calculation. `append_section`
/// still performs the complete header-slack, security-directory, alignment,
/// size, and output-reparse validation when the payload is installed.
///
/// # Errors
///
/// Returns a structured error for a malformed input, unsupported PE kind,
/// invalid section alignment, or virtual-address overflow.
pub fn next_section_rva(input: &[u8]) -> Result<Rva, RebuildError> {
    let pe = PeImage::parse(input).map_err(|source| RebuildError::Pe {
        stage: "input",
        source,
    })?;
    next_section_rva_for(&pe)
}

/// Replaces `AddressOfEntryPoint` with a file-backed RVA in an executable
/// section and reparses the resulting image before returning it.
///
/// # Errors
///
/// Rejects malformed images, non-executable or zero-fill targets, arithmetic
/// and allocation failures, and any entry-point value that does not survive
/// output reparse.
pub fn set_entry_point(input: &[u8], entry_point: Rva) -> Result<Vec<u8>, RebuildError> {
    let pe = PeImage::parse(input).map_err(|source| RebuildError::Pe {
        stage: "input",
        source,
    })?;
    let executable = pe.sections.iter().any(|section| {
        section.characteristics & IMAGE_SCN_MEM_EXECUTE != 0
            && section.contains_mapped_rva(entry_point)
    });
    if !executable {
        return Err(RebuildError::EntryPointNotExecutable { rva: entry_point });
    }
    pe.rva_to_file_offset(entry_point)
        .map_err(|_| RebuildError::EntryPointNotFileBacked { rva: entry_point })?;

    let optional_offset = add_offset(
        add_offset(pe.nt_headers_offset, 4, "COFF header offset")?,
        COFF_HEADER_SIZE,
        "optional-header offset",
    )?;
    let entry_offset = add_offset(optional_offset, 16, "AddressOfEntryPoint offset")?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(input.len())
        .map_err(|_| RebuildError::Allocation {
            requested: input.len(),
        })?;
    output.extend_from_slice(input);
    write_u32(
        &mut output,
        entry_offset,
        entry_point.0,
        "AddressOfEntryPoint",
    )?;
    let reparsed = PeImage::parse(&output).map_err(|source| RebuildError::Pe {
        stage: "output",
        source,
    })?;
    if reparsed.optional.address_of_entry_point != entry_point {
        return Err(RebuildError::OutputInvariant(
            "AddressOfEntryPoint differs after reparse".to_owned(),
        ));
    }
    Ok(output)
}

/// Appends one file-backed section to a PE32+ image without moving any existing
/// bytes. The operation is rejected unless the existing header contains an
/// unused 40-byte section-table slot before both `SizeOfHeaders` and the first
/// raw section.
///
/// The returned byte vector has already been reparsed and checked against the
/// requested section metadata.
///
/// # Errors
///
/// Returns a structured error for malformed input, unsupported PE kind,
/// alignment/address overflow, occupied or insufficient header slack, output
/// allocation failure, or failed disk-format reparse invariants.
#[allow(clippy::too_many_lines)]
pub fn append_section(input: &[u8], request: AppendSection<'_>) -> Result<RebuiltPe, RebuildError> {
    let pe = PeImage::parse(input).map_err(|source| RebuildError::Pe {
        stage: "input",
        source,
    })?;
    if pe.optional.kind != PeKind::Pe32Plus {
        return Err(RebuildError::UnsupportedKind(pe.optional.kind));
    }
    if request.data.is_empty() {
        return Err(RebuildError::EmptySection);
    }
    for (field, value) in [
        ("FileAlignment", pe.optional.file_alignment),
        ("SectionAlignment", pe.optional.section_alignment),
    ] {
        if value == 0 || !value.is_power_of_two() {
            return Err(RebuildError::InvalidAlignment { field, value });
        }
    }
    let section_alignment = pe.optional.section_alignment;
    let file_alignment = pe.optional.file_alignment;
    let incompatible_alignment = section_alignment < file_alignment
        || file_alignment > 0x1_0000
        || (section_alignment < 0x1000 && section_alignment != file_alignment)
        || (file_alignment < 0x200 && section_alignment != file_alignment);
    if incompatible_alignment {
        return Err(RebuildError::IncompatibleAlignments {
            section_alignment,
            file_alignment,
        });
    }

    if let Some(security) = pe.directory(IMAGE_DIRECTORY_ENTRY_SECURITY) {
        return Err(RebuildError::SecurityDirectoryPresent {
            offset: FileOffset(u64::from(security.address)),
            size: security.size,
        });
    }

    let new_section_count = pe
        .coff
        .number_of_sections
        .checked_add(1)
        .filter(|count| *count <= MAX_SECTIONS)
        .ok_or(RebuildError::SectionCount(pe.coff.number_of_sections))?;
    let payload_size =
        u32::try_from(request.data.len()).map_err(|_| RebuildError::PayloadTooLarge)?;
    let raw_size = align_u32(payload_size, pe.optional.file_alignment, "section raw size")?;
    let size_of_code = if request.characteristics & IMAGE_SCN_CNT_CODE != 0 {
        pe.optional
            .size_of_code
            .checked_add(raw_size)
            .ok_or(RebuildError::Overflow { what: "SizeOfCode" })?
    } else {
        pe.optional.size_of_code
    };
    let size_of_initialized_data = if request.characteristics & IMAGE_SCN_CNT_INITIALIZED_DATA != 0
    {
        pe.optional
            .size_of_initialized_data
            .checked_add(raw_size)
            .ok_or(RebuildError::Overflow {
                what: "SizeOfInitializedData",
            })?
    } else {
        pe.optional.size_of_initialized_data
    };

    let coff_offset = add_offset(pe.nt_headers_offset, 4, "COFF header offset")?;
    let optional_offset = add_offset(coff_offset, COFF_HEADER_SIZE, "optional-header offset")?;
    let section_table_offset = add_offset(
        optional_offset,
        u64::from(pe.coff.size_of_optional_header),
        "section-table offset",
    )?;
    let existing_table_size = u64::from(pe.coff.number_of_sections)
        .checked_mul(SECTION_HEADER_SIZE)
        .ok_or(RebuildError::Overflow {
            what: "section-table size",
        })?;
    let new_header_offset = add_offset(
        section_table_offset,
        existing_table_size,
        "new section-header offset",
    )?;
    let new_header_end = add_offset(
        new_header_offset,
        SECTION_HEADER_SIZE,
        "new section-header end",
    )?;

    let first_raw = pe
        .sections
        .iter()
        .filter(|section| section.size_of_raw_data != 0 && section.pointer_to_raw_data.0 != 0)
        .map(|section| section.pointer_to_raw_data)
        .min()
        .unwrap_or(FileOffset(u64::from(pe.optional.size_of_headers)));
    let header_capacity = FileOffset(first_raw.0.min(u64::from(pe.optional.size_of_headers)));
    if new_header_end > header_capacity {
        return Err(RebuildError::HeaderCapacity {
            required_end: new_header_end,
            capacity_end: header_capacity,
        });
    }
    for directory in pe
        .directories
        .iter()
        .filter(|directory| directory.is_present())
    {
        let directory_offset = match directory.address_kind {
            DirectoryAddressKind::Rva => match pe.rva_to_file_offset(Rva(directory.address)) {
                Ok(offset) => offset,
                Err(_) => continue,
            },
            DirectoryAddressKind::FileOffset => FileOffset(u64::from(directory.address)),
        };
        let directory_end = directory_offset
            .0
            .checked_add(u64::from(directory.size))
            .ok_or(RebuildError::Overflow {
                what: "data-directory file range",
            })?;
        if ranges_overlap(
            new_header_offset.0,
            new_header_end.0,
            directory_offset.0,
            directory_end,
        ) {
            return Err(RebuildError::HeaderDirectoryOverlap {
                index: directory.index,
                name: directory.name,
                offset: directory_offset,
                size: directory.size,
            });
        }
    }
    let slack_start = checked_usize(new_header_offset.0, "new section-header offset")?;
    let slack_end = checked_usize(new_header_end.0, "new section-header end")?;
    let slack = input
        .get(slack_start..slack_end)
        .ok_or(RebuildError::HeaderCapacity {
            required_end: new_header_end,
            capacity_end: FileOffset(u64::try_from(input.len()).unwrap_or(u64::MAX)),
        })?;
    if let Some(relative) = slack.iter().position(|byte| *byte != 0) {
        let relative = u64::try_from(relative).map_err(|_| RebuildError::Overflow {
            what: "occupied header-slack offset",
        })?;
        return Err(RebuildError::HeaderSlackOccupied {
            offset: add_offset(new_header_offset, relative, "occupied header-slack offset")?,
        });
    }

    let section_rva = next_section_rva_for(&pe)?;
    let virtual_span = payload_size.max(raw_size);
    let virtual_end = section_rva
        .0
        .checked_add(virtual_span)
        .ok_or(RebuildError::Overflow {
            what: "new section virtual end",
        })?;
    let size_of_image = align_u32(virtual_end, pe.optional.section_alignment, "SizeOfImage")?;

    let input_length = u64::try_from(input.len()).map_err(|_| RebuildError::Overflow {
        what: "input length",
    })?;
    let raw_offset = align_file_offset(
        FileOffset(input_length),
        pe.optional.file_alignment,
        "new section raw offset",
    )?;
    let raw_offset_u32 = u32::try_from(raw_offset.0).map_err(|_| RebuildError::Overflow {
        what: "new section raw offset",
    })?;
    let raw_end = raw_offset
        .0
        .checked_add(u64::from(raw_size))
        .ok_or(RebuildError::Overflow {
            what: "new section raw end",
        })?;
    u32::try_from(raw_end).map_err(|_| RebuildError::Overflow {
        what: "new section raw end",
    })?;
    let output_length = checked_usize(raw_end, "output length")?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(output_length)
        .map_err(|_| RebuildError::Allocation {
            requested: output_length,
        })?;
    output.extend_from_slice(input);
    output.resize(output_length, 0);
    let raw_start = checked_usize(raw_offset.0, "new section raw offset")?;
    let payload_end = raw_start
        .checked_add(request.data.len())
        .ok_or(RebuildError::Overflow {
            what: "new section payload end",
        })?;
    output
        .get_mut(raw_start..payload_end)
        .ok_or(RebuildError::Overflow {
            what: "new section payload range",
        })?
        .copy_from_slice(request.data);

    let section_count_offset = add_offset(coff_offset, 2, "section-count offset")?;
    write_u16(
        &mut output,
        section_count_offset,
        new_section_count,
        "section count",
    )?;
    write_u32(
        &mut output,
        add_offset(optional_offset, 4, "SizeOfCode offset")?,
        size_of_code,
        "SizeOfCode",
    )?;
    write_u32(
        &mut output,
        add_offset(optional_offset, 8, "SizeOfInitializedData offset")?,
        size_of_initialized_data,
        "SizeOfInitializedData",
    )?;
    let size_of_image_offset = add_offset(optional_offset, 56, "SizeOfImage offset")?;
    write_u32(
        &mut output,
        size_of_image_offset,
        size_of_image,
        "SizeOfImage",
    )?;

    write_bytes(
        &mut output,
        new_header_offset,
        &request.name,
        "section name",
    )?;
    write_u32(
        &mut output,
        add_offset(new_header_offset, 8, "section VirtualSize offset")?,
        payload_size,
        "section VirtualSize",
    )?;
    write_u32(
        &mut output,
        add_offset(new_header_offset, 12, "section RVA offset")?,
        section_rva.0,
        "section RVA",
    )?;
    write_u32(
        &mut output,
        add_offset(new_header_offset, 16, "section raw-size offset")?,
        raw_size,
        "section raw size",
    )?;
    write_u32(
        &mut output,
        add_offset(new_header_offset, 20, "section raw-offset field")?,
        raw_offset_u32,
        "section raw offset",
    )?;
    write_u32(
        &mut output,
        add_offset(new_header_offset, 36, "section characteristics offset")?,
        request.characteristics,
        "section characteristics",
    )?;

    let reparsed = PeImage::parse(&output).map_err(|source| RebuildError::Pe {
        stage: "output",
        source,
    })?;
    if reparsed.coff.number_of_sections != new_section_count
        || reparsed.optional.size_of_image != size_of_image
        || reparsed.optional.size_of_code != size_of_code
        || reparsed.optional.size_of_initialized_data != size_of_initialized_data
    {
        return Err(RebuildError::OutputInvariant(
            "updated section count or optional-header aggregate sizes did not survive reparse"
                .to_owned(),
        ));
    }
    if reparsed.directories != pe.directories
        || reparsed.sections.get(..pe.sections.len()) != Some(pe.sections.as_slice())
    {
        return Err(RebuildError::OutputInvariant(
            "existing sections or data directories changed during serialization".to_owned(),
        ));
    }
    let appended = reparsed.sections.last().ok_or_else(|| {
        RebuildError::OutputInvariant("reparsed output has no appended section".to_owned())
    })?;
    if appended.name_bytes != request.name
        || appended.virtual_address != section_rva
        || appended.virtual_size != payload_size
        || appended.pointer_to_raw_data != raw_offset
        || appended.size_of_raw_data != raw_size
        || appended.characteristics != request.characteristics
    {
        return Err(RebuildError::OutputInvariant(
            "appended section metadata differs after reparse".to_owned(),
        ));
    }
    let reparsed_payload = reparsed.data().get(raw_start..payload_end).ok_or_else(|| {
        RebuildError::OutputInvariant("appended payload is not file-backed".to_owned())
    })?;
    if reparsed_payload != request.data {
        return Err(RebuildError::OutputInvariant(
            "appended payload differs after reparse".to_owned(),
        ));
    }
    let translated = reparsed
        .rva_to_file_offset(section_rva)
        .map_err(|error| RebuildError::OutputInvariant(error.to_string()))?;
    if translated != raw_offset {
        return Err(RebuildError::OutputInvariant(
            "appended section RVA does not translate to its raw offset".to_owned(),
        ));
    }

    Ok(RebuiltPe {
        bytes: output,
        section: AppendedSection {
            name: request.name,
            rva: section_rva,
            raw_offset,
            virtual_size: payload_size,
            raw_size,
            characteristics: request.characteristics,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const NT_OFFSET: usize = 0x80;
    const OPTIONAL_OFFSET: usize = NT_OFFSET + 4 + 20;
    const SECTION_TABLE_OFFSET: usize = OPTIONAL_OFFSET + 0xF0;

    fn put_u16(bytes: &mut [u8], offset: usize, value: u16) {
        bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
    }

    fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
        bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }

    fn minimal_pe(first_raw: u32) -> Vec<u8> {
        let raw_end = usize::try_from(first_raw).expect("small test offset") + 0x200;
        let mut bytes = vec![0_u8; raw_end.max(0x400)];
        put_u16(&mut bytes, 0, 0x5A4D);
        put_u32(
            &mut bytes,
            0x3C,
            u32::try_from(NT_OFFSET).expect("small NT offset"),
        );
        put_u32(&mut bytes, NT_OFFSET, 0x0000_4550);
        let coff = NT_OFFSET + 4;
        put_u16(&mut bytes, coff, 0x8664);
        put_u16(&mut bytes, coff + 2, 1);
        put_u16(&mut bytes, coff + 16, 0xF0);
        put_u16(&mut bytes, coff + 18, 0x2022);
        put_u16(&mut bytes, OPTIONAL_OFFSET, 0x020B);
        put_u32(&mut bytes, OPTIONAL_OFFSET + 16, 0x1000);
        put_u32(&mut bytes, OPTIONAL_OFFSET + 20, 0x1000);
        put_u64(&mut bytes, OPTIONAL_OFFSET + 24, 0x1_4000_0000);
        put_u32(&mut bytes, OPTIONAL_OFFSET + 32, 0x1000);
        put_u32(&mut bytes, OPTIONAL_OFFSET + 36, 0x200);
        put_u32(&mut bytes, OPTIONAL_OFFSET + 56, 0x2000);
        put_u32(&mut bytes, OPTIONAL_OFFSET + 60, 0x200);
        put_u16(&mut bytes, OPTIONAL_OFFSET + 68, 3);
        put_u32(&mut bytes, OPTIONAL_OFFSET + 108, 1);
        put_u32(&mut bytes, OPTIONAL_OFFSET + 112, 0x1000);
        put_u32(&mut bytes, OPTIONAL_OFFSET + 116, 4);
        let section = SECTION_TABLE_OFFSET;
        bytes[section..section + 8].copy_from_slice(b".text\0\0\0");
        put_u32(&mut bytes, section + 8, 0x100);
        put_u32(&mut bytes, section + 12, 0x1000);
        put_u32(&mut bytes, section + 16, 0x200);
        put_u32(&mut bytes, section + 20, first_raw);
        put_u32(&mut bytes, section + 36, 0x6000_0020);
        let raw_start = usize::try_from(first_raw).expect("small raw offset");
        bytes[raw_start] = 0xC3;
        bytes
    }

    #[test]
    fn appends_section_and_reparses_without_changing_directories() {
        let input = minimal_pe(0x200);
        let original = PeImage::parse(&input).expect("input PE");
        let planned_rva = next_section_rva(&input).expect("plan section RVA");
        let rebuilt = append_section(
            &input,
            AppendSection {
                name: *b".vmpx\0\0\0",
                data: &[0x90, 0x90, 0xC3],
                characteristics: 0x6000_0020,
            },
        )
        .expect("append section");
        let output = PeImage::parse(&rebuilt.bytes).expect("reparse output");

        assert_eq!(output.coff.number_of_sections, 2);
        assert_eq!(output.optional.size_of_image, 0x3000);
        assert_eq!(output.optional.size_of_code, 0x200);
        assert_eq!(output.optional.size_of_initialized_data, 0);
        assert_eq!(rebuilt.section.rva, Rva(0x2000));
        assert_eq!(rebuilt.section.rva, planned_rva);
        assert_eq!(rebuilt.section.raw_offset, FileOffset(0x400));
        assert_eq!(rebuilt.section.raw_size, 0x200);
        assert_eq!(output.directories, original.directories);
        assert_eq!(&rebuilt.bytes[0x400..0x403], &[0x90, 0x90, 0xC3]);

        let count = NT_OFFSET + 4 + 2;
        let size_of_code = OPTIONAL_OFFSET + 4;
        let size_of_image = OPTIONAL_OFFSET + 56;
        let new_header = SECTION_TABLE_OFFSET + 40;
        for (index, byte) in input.iter().copied().enumerate() {
            let intentionally_changed = (count..count + 2).contains(&index)
                || (size_of_code..size_of_code + 4).contains(&index)
                || (size_of_image..size_of_image + 4).contains(&index)
                || (new_header..new_header + 40).contains(&index);
            if !intentionally_changed {
                assert_eq!(
                    rebuilt.bytes[index], byte,
                    "unexpected change at {index:#X}"
                );
            }
        }
    }

    #[test]
    fn appends_and_installs_a_reparsed_import_section() {
        let mut input = minimal_pe(0x200);
        put_u32(&mut input, OPTIONAL_OFFSET + 108, 16);
        let section_rva = next_section_rva(&input).expect("plan import section");
        let table = serialize_import_table(
            PeKind::Pe32Plus,
            section_rva,
            &[ImportModuleSpec {
                name: "KERNEL32.dll".to_owned(),
                symbols: vec![
                    ImportTarget::Name {
                        name: "GetLastError".to_owned(),
                        hint: 0,
                    },
                    ImportTarget::Ordinal(7),
                ],
            }],
        )
        .expect("serialize import section");
        let rebuilt = append_section(
            &input,
            AppendSection {
                name: *b".idata2\0",
                data: &table.bytes,
                characteristics: 0xC000_0040,
            },
        )
        .expect("append import section");
        assert_eq!(rebuilt.section.rva, section_rva);

        let installed =
            install_import_directories(&rebuilt.bytes, &table).expect("install import directories");
        let reparsed = PeImage::parse(&installed).expect("reparse installed image");
        assert_eq!(reparsed.report().imports, table.modules);
        assert_eq!(
            reparsed
                .directory(vmpx_pe::IMAGE_DIRECTORY_ENTRY_IMPORT)
                .expect("import directory")
                .address,
            table.import_directory.rva.0
        );
    }

    #[test]
    fn redirects_entry_point_only_to_file_backed_executable_code() {
        let input = minimal_pe(0x200);
        let rebuilt = append_section(
            &input,
            AppendSection {
                name: *b".vmpx\0\0\0",
                data: &[0x90, 0xC3],
                characteristics: 0x6000_0020,
            },
        )
        .expect("append native code");
        let redirected =
            set_entry_point(&rebuilt.bytes, rebuilt.section.rva).expect("redirect entry point");
        assert_eq!(
            PeImage::parse(&redirected)
                .expect("reparse redirected image")
                .optional
                .address_of_entry_point,
            rebuilt.section.rva
        );

        let non_executable = append_section(
            &input,
            AppendSection {
                name: *b".vmpxd\0\0",
                data: &[1],
                characteristics: 0x4000_0040,
            },
        )
        .expect("append non-executable data");
        assert!(matches!(
            set_entry_point(&non_executable.bytes, non_executable.section.rva),
            Err(RebuildError::EntryPointNotExecutable { .. })
        ));
    }

    #[test]
    fn rejects_section_header_growth_that_overlaps_first_raw_section() {
        let input = minimal_pe(0x1C0);
        let error = append_section(
            &input,
            AppendSection {
                name: *b".vmpx\0\0\0",
                data: &[0xC3],
                characteristics: 0x6000_0020,
            },
        )
        .expect_err("new header must not overlap raw data");

        assert!(matches!(error, RebuildError::HeaderCapacity { .. }));
    }

    #[test]
    fn rejects_nonzero_section_header_slack() {
        let mut input = minimal_pe(0x200);
        input[SECTION_TABLE_OFFSET + 40] = 0x41;
        let error = append_section(
            &input,
            AppendSection {
                name: *b".vmpx\0\0\0",
                data: &[0xC3],
                characteristics: 0x6000_0020,
            },
        )
        .expect_err("occupied header slack must be preserved");

        assert!(matches!(error, RebuildError::HeaderSlackOccupied { .. }));
    }

    #[test]
    fn rejects_directory_range_in_zero_filled_section_header_slack() {
        let mut input = minimal_pe(0x200);
        let new_header = u32::try_from(SECTION_TABLE_OFFSET + 40).expect("small test offset");
        put_u32(&mut input, OPTIONAL_OFFSET + 112, new_header);
        put_u32(&mut input, OPTIONAL_OFFSET + 116, 1);
        let error = append_section(
            &input,
            AppendSection {
                name: *b".vmpx\0\0\0",
                data: &[0xC3],
                characteristics: 0x6000_0020,
            },
        )
        .expect_err("directory-owned header slack must not be overwritten");

        assert!(matches!(
            error,
            RebuildError::HeaderDirectoryOverlap { index: 0, .. }
        ));
    }

    #[test]
    fn rejects_present_security_directory() {
        let mut input = minimal_pe(0x200);
        input.resize(0x408, 0);
        input[0x400..0x408].copy_from_slice(b"WIN_CERT");
        put_u32(&mut input, OPTIONAL_OFFSET + 108, 5);
        let security_entry = OPTIONAL_OFFSET + 112 + IMAGE_DIRECTORY_ENTRY_SECURITY * 8;
        put_u32(&mut input, security_entry, 0x400);
        put_u32(&mut input, security_entry + 4, 8);
        let error = append_section(
            &input,
            AppendSection {
                name: *b".vmpx\0\0\0",
                data: &[0xC3],
                characteristics: 0x6000_0020,
            },
        )
        .expect_err("signed images require explicit certificate handling");

        assert!(matches!(
            error,
            RebuildError::SecurityDirectoryPresent {
                offset: FileOffset(0x400),
                size: 8
            }
        ));
    }

    #[test]
    fn rejects_incompatible_section_and_file_alignment() {
        let mut input = minimal_pe(0x200);
        put_u32(&mut input, OPTIONAL_OFFSET + 36, 0x100);
        let error = append_section(
            &input,
            AppendSection {
                name: *b".vmpx\0\0\0",
                data: &[0xC3],
                characteristics: 0x6000_0020,
            },
        )
        .expect_err("low FileAlignment requires matching SectionAlignment");

        assert!(matches!(
            error,
            RebuildError::IncompatibleAlignments {
                section_alignment: 0x1000,
                file_alignment: 0x100
            }
        ));
    }

    #[test]
    fn rejects_optional_header_aggregate_overflow() {
        let mut input = minimal_pe(0x200);
        put_u32(&mut input, OPTIONAL_OFFSET + 4, u32::MAX);
        let error = append_section(
            &input,
            AppendSection {
                name: *b".vmpx\0\0\0",
                data: &[0xC3],
                characteristics: 0x6000_0020,
            },
        )
        .expect_err("SizeOfCode addition must be checked");

        assert!(matches!(
            error,
            RebuildError::Overflow { what: "SizeOfCode" }
        ));
    }

    #[test]
    fn appends_after_overlay_without_overwriting_it() {
        let mut input = minimal_pe(0x200);
        input.extend_from_slice(b"overlay");
        let rebuilt = append_section(
            &input,
            AppendSection {
                name: *b".data2\0\0",
                data: &[1, 2, 3, 4],
                characteristics: 0xC000_0040,
            },
        )
        .expect("append after overlay");

        assert_eq!(&rebuilt.bytes[0x400..0x407], b"overlay");
        assert_eq!(rebuilt.section.raw_offset, FileOffset(0x600));
        assert_eq!(&rebuilt.bytes[0x600..0x604], &[1, 2, 3, 4]);
        let output = PeImage::parse(&rebuilt.bytes).expect("reparse output");
        assert_eq!(output.optional.size_of_initialized_data, 0x200);
    }
}
