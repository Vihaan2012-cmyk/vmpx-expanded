//! Section-removal primitive.
//!
//! Layout strategy: RVAs are never moved (code and data reference each other
//! by RVA, so compacting the virtual layout would require rewriting every
//! reference). Removed sections leave a hole in the virtual address space,
//! which the Windows loader accepts. The on-disk layout *is* compacted: the raw
//! bytes of removed sections are deleted and every later file offset is shifted
//! down, with all file-offset fields rewritten. `SizeOfImage` is recomputed
//! from the surviving sections.
//!
//! The operation refuses, rather than produce a broken image, when any retained
//! structure statically references a removed section.

use thiserror::Error;

use super::{
    COFF_HEADER_SIZE, DirectoryAddressKind, IMAGE_DIRECTORY_ENTRY_BASERELOC,
    IMAGE_DIRECTORY_ENTRY_DEBUG, IMAGE_DIRECTORY_ENTRY_EXCEPTION, IMAGE_DIRECTORY_ENTRY_EXPORT,
    IMAGE_DIRECTORY_ENTRY_IMPORT, IMAGE_DIRECTORY_ENTRY_LOAD_CONFIG,
    IMAGE_DIRECTORY_ENTRY_SECURITY, IMAGE_DIRECTORY_ENTRY_TLS, MAX_EXPORT_ENTRIES, PeError,
    PeImage, PeKind, Rva, SECTION_HEADER_SIZE, Section, Va,
};

const IMAGE_SCN_CNT_CODE: u32 = 0x20;
const IMAGE_SCN_CNT_INITIALIZED_DATA: u32 = 0x40;
const IMAGE_SCN_CNT_UNINITIALIZED_DATA: u32 = 0x80;
const IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT: usize = 13;
const IMAGE_DIRECTORY_ENTRY_COM_DESCRIPTOR: usize = 14;
const RELOC_HIGHLOW: u8 = 3;
const RELOC_DIR64: u8 = 10;
const MACHINE_AMD64: u16 = 0x8664;
const MAX_REPORTED_REFS: usize = 64;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SectionSelector {
    Index(usize),
    /// Matches every section with this exact name.
    Name(String),
}

#[derive(Clone, Copy, Debug)]
pub struct StripRequest<'a> {
    pub remove: &'a [SectionSelector],
    /// Data-directory indices the caller asserts are genuinely gone. Only
    /// directories wholly inside a removed section may be listed; they are
    /// zeroed. A directory inside a removed section that is not listed makes
    /// the strip refuse.
    pub drop_directories: &'a [usize],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StrippedPe {
    pub data: Vec<u8>,
    /// Original headers of the removed sections.
    pub removed: Vec<Section>,
    pub file_bytes_removed: u64,
    pub relocation_entries_dropped: usize,
    pub runtime_functions_dropped: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DanglingRef {
    /// The retained structure holding the reference.
    pub source: String,
    /// What it points at, e.g. `RVA 0x3000`.
    pub target: String,
    /// The removed section being pointed into.
    pub section: String,
}

impl std::fmt::Display for DanglingRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} targets {} inside removed section {}",
            self.source, self.target, self.section
        )
    }
}

fn format_refs(refs: &[DanglingRef]) -> String {
    let shown = refs
        .iter()
        .take(3)
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("; ");
    if refs.len() > 3 {
        format!("{shown}; and {} more", refs.len() - 3)
    } else {
        shown
    }
}

#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum StripError {
    #[error("{stage} PE error: {source}")]
    Pe {
        stage: &'static str,
        source: PeError,
    },
    #[error("no section selected for removal")]
    NothingSelected,
    #[error("section index {0} does not exist")]
    NoSuchIndex(usize),
    #[error("no section is named {0:?}")]
    NoSuchName(String),
    #[error("refusing to remove every section")]
    RemovesAllSections,
    #[error("unsupported layout: {0}")]
    UnsupportedLayout(String),
    #[error("dangling reference into removed section: {}", format_refs(.0))]
    Dangling(Vec<DanglingRef>),
    #[error(
        "{name} directory ({address:#X}+{size:#X}) partially overlaps removed section {section}"
    )]
    DirectoryStraddles {
        name: &'static str,
        address: u32,
        size: u32,
        section: String,
    },
    #[error(
        "{name} directory lies inside removed section {section}; list index {index} in drop_directories to drop it"
    )]
    DirectoryInRemoved {
        index: usize,
        name: &'static str,
        section: String,
    },
    #[error(
        "drop_directories index {0} is not a present directory wholly inside a removed section"
    )]
    DropNotInRemoved(usize),
    #[error("security directory present; stripping would leave a stale signature")]
    SecurityDirectoryPresent,
    #[error("cannot statically analyze {what}: {message}")]
    Unanalyzable { what: &'static str, message: String },
    #[error("output invariant violated: {0}")]
    OutputInvariant(String),
}

fn pe_err(stage: &'static str) -> impl Fn(PeError) -> StripError {
    move |source| StripError::Pe { stage, source }
}

fn unanalyzable(what: &'static str) -> impl Fn(PeError) -> StripError {
    move |error| StripError::Unanalyzable {
        what,
        message: error.to_string(),
    }
}

fn layout(message: impl Into<String>) -> StripError {
    StripError::UnsupportedLayout(message.into())
}

fn to_usize(value: u64, what: &str) -> Result<usize, StripError> {
    usize::try_from(value).map_err(|_| layout(format!("{what} does not fit in usize")))
}

fn align_up(value: u64, alignment: u64) -> u64 {
    value.div_ceil(alignment) * alignment
}

struct Checker<'a> {
    removed: Vec<&'a Section>,
    refs: Vec<DanglingRef>,
}

impl Checker<'_> {
    fn check(&mut self, source: impl FnOnce() -> String, rva: u32) {
        if rva == 0 || self.refs.len() >= MAX_REPORTED_REFS {
            return;
        }
        let hit = self
            .removed
            .iter()
            .find(|section| section.contains_mapped_rva(Rva(rva)))
            .map(|section| section.name.clone());
        if let Some(section) = hit {
            self.refs.push(DanglingRef {
                source: source(),
                target: format!("RVA {}", Rva(rva)),
                section,
            });
        }
    }

    fn check_va(&mut self, pe: &PeImage<'_>, source: impl FnOnce() -> String, va: u64) {
        if let Some(rva) = pe.optional.image_base.checked_rva_of(Va(va)) {
            self.check(source, rva.0);
        }
    }

    fn is_removed(&self, rva: u32) -> bool {
        self.removed
            .iter()
            .any(|section| section.contains_mapped_rva(Rva(rva)))
    }
}

/// Sorted, merged raw-file intervals that are deleted from the output.
struct FileMap {
    deleted: Vec<(u64, u64)>,
}

impl FileMap {
    fn map(&self, offset: u64) -> Option<u64> {
        let mut shift = 0;
        for &(start, end) in &self.deleted {
            if offset >= end {
                shift += end - start;
            } else if offset >= start {
                return None;
            } else {
                break;
            }
        }
        Some(offset - shift)
    }

    /// Like `map`, but an offset inside a deleted interval clamps to its start.
    fn map_clamped(&self, offset: u64) -> u64 {
        let mut shift = 0;
        for &(start, end) in &self.deleted {
            if offset >= end {
                shift += end - start;
            } else if offset >= start {
                return start - shift;
            } else {
                break;
            }
        }
        offset - shift
    }
}

fn put_u32(out: &mut [u8], offset: usize, value: u32) -> Result<(), StripError> {
    let slot = out
        .get_mut(offset..offset + 4)
        .ok_or_else(|| layout(format!("write at {offset:#X} exceeds output")))?;
    slot.copy_from_slice(&value.to_le_bytes());
    Ok(())
}

fn put_u16(out: &mut [u8], offset: usize, value: u16) -> Result<(), StripError> {
    let slot = out
        .get_mut(offset..offset + 2)
        .ok_or_else(|| layout(format!("write at {offset:#X} exceeds output")))?;
    slot.copy_from_slice(&value.to_le_bytes());
    Ok(())
}

fn read_u32(data: &[u8], offset: usize) -> Option<u32> {
    let bytes = data.get(offset..offset.checked_add(4)?)?;
    Some(u32::from_le_bytes(bytes.try_into().ok()?))
}

fn read_ptr(data: &[u8], offset: usize, size: usize) -> Option<u64> {
    let bytes = data.get(offset..offset.checked_add(size)?)?;
    Some(if size == 8 {
        u64::from_le_bytes(bytes.try_into().ok()?)
    } else {
        u64::from(u32::from_le_bytes(bytes.try_into().ok()?))
    })
}

fn pe_checksum(data: &[u8], checksum_offset: usize) -> u32 {
    let mut sum: u64 = 0;
    let mut index = 0;
    while index < data.len() {
        if index != checksum_offset && index != checksum_offset + 2 {
            let word = if index + 1 < data.len() {
                u16::from_le_bytes([data[index], data[index + 1]])
            } else {
                u16::from(data[index])
            };
            sum += u64::from(word);
            sum = (sum & 0xFFFF) + (sum >> 16);
        }
        index += 2;
    }
    sum = (sum & 0xFFFF) + (sum >> 16);
    sum += u64::try_from(data.len()).unwrap_or(0);
    u32::try_from(sum & 0xFFFF_FFFF).unwrap_or(0)
}

/// How a data directory sits relative to the removed sections.
enum DirPlacement {
    Retained,
    InsideRemoved(String),
}

/// Plan for rewriting a table-style directory in place.
struct TableRewrite {
    directory: usize,
    bytes: Vec<u8>,
}

/// Removes the selected sections and returns the rewritten image.
///
/// # Errors
///
/// Returns a typed error when the selection is invalid, any retained structure
/// statically references a removed section, a structure cannot be analyzed, or
/// the rewritten image fails re-validation.
#[allow(clippy::too_many_lines)]
pub fn strip_sections(input: &[u8], request: &StripRequest<'_>) -> Result<StrippedPe, StripError> {
    let pe = PeImage::parse(input).map_err(pe_err("input"))?;
    let file_alignment = u64::from(pe.optional.file_alignment);
    let section_alignment = u64::from(pe.optional.section_alignment);
    for (field, value) in [
        ("FileAlignment", file_alignment),
        ("SectionAlignment", section_alignment),
    ] {
        if value == 0 || !value.is_power_of_two() {
            return Err(layout(format!(
                "{field} {value:#X} is zero or not a power of two"
            )));
        }
    }
    if pe.directory(IMAGE_DIRECTORY_ENTRY_SECURITY).is_some() {
        return Err(StripError::SecurityDirectoryPresent);
    }

    let removed_flags = resolve_selection(&pe, request.remove)?;
    let removed_sections: Vec<&Section> = pe
        .sections
        .iter()
        .filter(|section| removed_flags[section.index])
        .collect();
    let dropped = classify_directories(&pe, &removed_flags, request.drop_directories)?;

    let mut checker = Checker {
        removed: removed_sections.clone(),
        refs: Vec::new(),
    };
    let mut rewrites = Vec::new();
    let mut relocation_entries_dropped = 0;
    let mut runtime_functions_dropped = 0;
    analyze(
        &pe,
        &dropped,
        &mut checker,
        &mut rewrites,
        &mut relocation_entries_dropped,
        &mut runtime_functions_dropped,
    )?;
    if !checker.refs.is_empty() {
        return Err(StripError::Dangling(checker.refs));
    }

    let file_len = u64::try_from(input.len()).unwrap_or(u64::MAX);
    let map = plan_file_map(&pe, &removed_flags, file_len)?;
    let nt_offset = to_usize(pe.nt_headers_offset.0, "NT header offset")?;
    let coff_offset = nt_offset + 4;
    let optional_offset = coff_offset + COFF_HEADER_SIZE;
    let table_offset = optional_offset + usize::from(pe.coff.size_of_optional_header);
    let table_end = table_offset + pe.sections.len() * SECTION_HEADER_SIZE;
    if map.deleted.first().is_some_and(|&(start, _)| {
        u64::try_from(table_end).unwrap_or(u64::MAX) > start
            || u64::from(pe.optional.size_of_headers) > start
    }) {
        return Err(layout("headers overlap a removed section's raw data"));
    }

    let mut out = Vec::with_capacity(input.len());
    let mut cursor = 0_u64;
    for &(start, end) in &map.deleted {
        out.extend_from_slice(&input[to_usize(cursor, "cursor")?..to_usize(start, "start")?]);
        cursor = end;
    }
    out.extend_from_slice(&input[to_usize(cursor, "cursor")?..]);
    let file_bytes_removed = file_len - u64::try_from(out.len()).unwrap_or(0);

    // Section table: drop removed headers, shift the rest up, zero the tail.
    let mut slot = 0;
    for section in pe.sections.iter().filter(|s| !removed_flags[s.index]) {
        let source = table_offset + section.index * SECTION_HEADER_SIZE;
        let dest = table_offset + slot * SECTION_HEADER_SIZE;
        out[dest..dest + SECTION_HEADER_SIZE]
            .copy_from_slice(&input[source..source + SECTION_HEADER_SIZE]);
        for (field, name, clamp) in [
            (20, "raw data", section.size_of_raw_data == 0),
            (24, "relocations", true),
            (28, "line numbers", true),
        ] {
            let original = u64::from(read_u32(input, source + field).unwrap_or(0));
            if original == 0 {
                continue;
            }
            let mapped = if clamp {
                map.map_clamped(original)
            } else {
                map.map(original).ok_or_else(|| {
                    layout(format!(
                        "section {} {name} pointer is deleted",
                        section.name
                    ))
                })?
            };
            put_u32(
                &mut out,
                dest + field,
                u32::try_from(mapped).map_err(|_| layout("mapped pointer exceeds 32 bits"))?,
            )?;
        }
        slot += 1;
    }
    out[table_offset + slot * SECTION_HEADER_SIZE..table_end].fill(0);
    let remaining = u16::try_from(slot).map_err(|_| layout("section count exceeds u16"))?;
    put_u16(&mut out, coff_offset + 2, remaining)?;

    if pe.coff.pointer_to_symbol_table.0 != 0 {
        let mapped = map
            .map(pe.coff.pointer_to_symbol_table.0)
            .ok_or_else(|| layout("COFF symbol table lies in a removed section"))?;
        put_u32(
            &mut out,
            coff_offset + 8,
            u32::try_from(mapped).map_err(|_| layout("symbol table offset exceeds 32 bits"))?,
        )?;
    }

    // Optional header size fields.
    let mut size_of_code = pe.optional.size_of_code;
    let mut size_of_init = pe.optional.size_of_initialized_data;
    let mut size_of_uninit = pe.optional.size_of_uninitialized_data;
    for section in &removed_sections {
        if section.characteristics & IMAGE_SCN_CNT_CODE != 0 {
            size_of_code = size_of_code.saturating_sub(section.size_of_raw_data);
        }
        if section.characteristics & IMAGE_SCN_CNT_INITIALIZED_DATA != 0 {
            size_of_init = size_of_init.saturating_sub(section.size_of_raw_data);
        }
        if section.characteristics & IMAGE_SCN_CNT_UNINITIALIZED_DATA != 0 {
            let virtual_size = align_up(u64::from(section.virtual_size), file_alignment);
            size_of_uninit =
                size_of_uninit.saturating_sub(u32::try_from(virtual_size).unwrap_or(u32::MAX));
        }
    }
    let mut image_end = align_up(u64::from(pe.optional.size_of_headers), section_alignment);
    for section in pe.sections.iter().filter(|s| !removed_flags[s.index]) {
        let end = u64::from(section.virtual_address.0) + u64::from(section.mapped_span());
        image_end = image_end.max(align_up(end, section_alignment));
    }
    put_u32(&mut out, optional_offset + 4, size_of_code)?;
    put_u32(&mut out, optional_offset + 8, size_of_init)?;
    put_u32(&mut out, optional_offset + 12, size_of_uninit)?;
    put_u32(
        &mut out,
        optional_offset + 56,
        u32::try_from(image_end).map_err(|_| layout("SizeOfImage exceeds 32 bits"))?,
    )?;

    let directory_base = optional_offset
        + match pe.optional.kind {
            PeKind::Pe32 => 96,
            PeKind::Pe32Plus => 112,
        };
    for &index in &dropped {
        let slot = directory_base + index * 8;
        out[slot..slot + 8].fill(0);
    }

    verify_raw_moves(&pe, &out, &removed_flags, &map)?;
    for rewrite in &rewrites {
        apply_table_rewrite(&pe, &map, &mut out, directory_base, rewrite)?;
    }
    fix_debug_pointers(&pe, &dropped, &map, &mut out)?;

    if pe.optional.checksum != 0 {
        let checksum = pe_checksum(&out, optional_offset + 64);
        put_u32(&mut out, optional_offset + 64, checksum)?;
    }

    verify_output(&pe, &out, &removed_flags)?;
    Ok(StrippedPe {
        data: out,
        removed: removed_sections.into_iter().cloned().collect(),
        file_bytes_removed,
        relocation_entries_dropped,
        runtime_functions_dropped,
    })
}

fn resolve_selection(
    pe: &PeImage<'_>,
    selectors: &[SectionSelector],
) -> Result<Vec<bool>, StripError> {
    if selectors.is_empty() {
        return Err(StripError::NothingSelected);
    }
    let mut flags = vec![false; pe.sections.len()];
    for selector in selectors {
        match selector {
            SectionSelector::Index(index) => {
                *flags
                    .get_mut(*index)
                    .ok_or(StripError::NoSuchIndex(*index))? = true;
            }
            SectionSelector::Name(name) => {
                let mut matched = false;
                for section in pe.sections.iter().filter(|s| &s.name == name) {
                    flags[section.index] = true;
                    matched = true;
                }
                if !matched {
                    return Err(StripError::NoSuchName(name.clone()));
                }
            }
        }
    }
    if flags.iter().all(|removed| *removed) {
        return Err(StripError::RemovesAllSections);
    }
    Ok(flags)
}

/// Returns the directory indices to zero. Refuses directories that straddle a
/// removed section or sit inside one without being listed as dropped.
fn classify_directories(
    pe: &PeImage<'_>,
    removed: &[bool],
    drop_directories: &[usize],
) -> Result<Vec<usize>, StripError> {
    let mut dropped = Vec::new();
    for directory in pe.directories.iter().filter(|d| d.is_present()) {
        if directory.address_kind != DirectoryAddressKind::Rva {
            continue;
        }
        let start = u64::from(directory.address);
        let end = start + u64::from(directory.size);
        let hit = pe.sections.iter().find(|section| {
            removed[section.index]
                && start < u64::from(section.virtual_address.0) + u64::from(section.mapped_span())
                && u64::from(section.virtual_address.0) < end
        });
        let placement = match hit {
            None => DirPlacement::Retained,
            Some(section) => {
                let wholly_inside = section.contains_mapped_rva(Rva(directory.address))
                    && section.contains_mapped_rva(Rva(directory.address + directory.size - 1));
                if wholly_inside {
                    DirPlacement::InsideRemoved(section.name.clone())
                } else {
                    return Err(StripError::DirectoryStraddles {
                        name: directory.name,
                        address: directory.address,
                        size: directory.size,
                        section: section.name.clone(),
                    });
                }
            }
        };
        match placement {
            DirPlacement::Retained => {
                if drop_directories.contains(&directory.index) {
                    return Err(StripError::DropNotInRemoved(directory.index));
                }
            }
            DirPlacement::InsideRemoved(section) => {
                if drop_directories.contains(&directory.index) {
                    dropped.push(directory.index);
                } else {
                    return Err(StripError::DirectoryInRemoved {
                        index: directory.index,
                        name: directory.name,
                        section,
                    });
                }
            }
        }
    }
    for index in drop_directories {
        if !dropped.contains(index) {
            return Err(StripError::DropNotInRemoved(*index));
        }
    }
    Ok(dropped)
}

fn plan_file_map(pe: &PeImage<'_>, removed: &[bool], file_len: u64) -> Result<FileMap, StripError> {
    let raw_range = |section: &Section| {
        (section.size_of_raw_data != 0 && section.pointer_to_raw_data.0 != 0).then(|| {
            (
                section.pointer_to_raw_data.0,
                section.pointer_to_raw_data.0 + u64::from(section.size_of_raw_data),
            )
        })
    };
    let mut deleted = Vec::new();
    for section in pe.sections.iter().filter(|s| removed[s.index]) {
        if let Some((start, end)) = raw_range(section) {
            if end > file_len {
                return Err(layout(format!(
                    "section {} raw range exceeds file",
                    section.name
                )));
            }
            if start < u64::from(pe.optional.size_of_headers) {
                return Err(layout(format!(
                    "section {} raw data overlaps headers",
                    section.name
                )));
            }
            deleted.push((start, end));
        }
    }
    for section in pe.sections.iter().filter(|s| !removed[s.index]) {
        if let Some((start, end)) = raw_range(section)
            && deleted.iter().any(|&(ds, de)| start < de && ds < end)
        {
            return Err(layout(format!(
                "retained section {} shares raw bytes with a removed section",
                section.name
            )));
        }
    }
    deleted.sort_unstable();
    let mut merged: Vec<(u64, u64)> = Vec::new();
    for (start, end) in deleted {
        match merged.last_mut() {
            Some(last) if start <= last.1 => last.1 = last.1.max(end),
            _ => merged.push((start, end)),
        }
    }
    Ok(FileMap { deleted: merged })
}

#[allow(clippy::too_many_lines)]
fn analyze(
    pe: &PeImage<'_>,
    dropped: &[usize],
    checker: &mut Checker<'_>,
    rewrites: &mut Vec<TableRewrite>,
    relocation_entries_dropped: &mut usize,
    runtime_functions_dropped: &mut usize,
) -> Result<(), StripError> {
    let live = |index: usize| pe.directory(index).filter(|_| !dropped.contains(&index));
    let pointer_size = pe.optional.kind.pointer_size();

    checker.check(
        || "AddressOfEntryPoint".to_owned(),
        pe.optional.address_of_entry_point.0,
    );
    checker.check(|| "BaseOfCode".to_owned(), pe.optional.base_of_code.0);
    if pe.optional.kind == PeKind::Pe32 {
        let offset = to_usize(pe.nt_headers_offset.0, "NT offset")? + 4 + COFF_HEADER_SIZE + 24;
        if let Some(base_of_data) = read_u32(pe.data(), offset) {
            checker.check(|| "BaseOfData".to_owned(), base_of_data);
        }
    }
    for index in [
        IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT,
        IMAGE_DIRECTORY_ENTRY_COM_DESCRIPTOR,
    ] {
        if let Some(directory) = live(index) {
            return Err(StripError::Unanalyzable {
                what: directory.name,
                message: "directory contents are not analyzed".to_owned(),
            });
        }
    }

    if live(IMAGE_DIRECTORY_ENTRY_IMPORT).is_some() {
        let reader = pe.reader();
        for (index, module) in pe
            .parse_imports()
            .map_err(unanalyzable("import directory"))?
            .iter()
            .enumerate()
        {
            checker.check(
                || format!("import descriptor {index} ({})", module.name),
                module.descriptor_rva.0,
            );
            checker.check(
                || format!("import {} OriginalFirstThunk", module.name),
                module.original_first_thunk.0,
            );
            checker.check(
                || format!("import {} FirstThunk", module.name),
                module.first_thunk.0,
            );
            let name_field = pe
                .offset_for_rva(module.descriptor_rva)
                .map_err(unanalyzable("import directory"))?
                + 12;
            if let Ok(name_rva) = reader.u32(name_field, "import DLL name RVA") {
                checker.check(|| format!("import {} DLL name", module.name), name_rva);
            }
            for symbol in &module.entries {
                checker.check(
                    || format!("import {} IAT slot", module.name),
                    symbol.iat_rva.0,
                );
                if symbol.name.is_some()
                    && let Ok(name_rva) = u32::try_from(symbol.lookup_value)
                {
                    checker.check(
                        || format!("import {} hint/name entry", module.name),
                        name_rva,
                    );
                }
            }
        }
    }

    if let Some(directory) = live(IMAGE_DIRECTORY_ENTRY_EXPORT) {
        let reader = pe.reader();
        let base = pe
            .offset_for_rva(Rva(directory.address))
            .map_err(unanalyzable("export directory"))?;
        let field = |relative: usize| {
            reader
                .u32(base + relative, "export field")
                .map_err(unanalyzable("export directory"))
        };
        let (name, functions, names, ordinals) = (field(12)?, field(28)?, field(32)?, field(36)?);
        let function_count = usize::try_from(field(20)?).unwrap_or(usize::MAX);
        let name_count = usize::try_from(field(24)?).unwrap_or(usize::MAX);
        if function_count > MAX_EXPORT_ENTRIES || name_count > MAX_EXPORT_ENTRIES {
            return Err(StripError::Unanalyzable {
                what: "export directory",
                message: "export count exceeds resource limit".to_owned(),
            });
        }
        checker.check(|| "export module name".to_owned(), name);
        checker.check(|| "export address table".to_owned(), functions);
        checker.check(|| "export name table".to_owned(), names);
        checker.check(|| "export ordinal table".to_owned(), ordinals);
        for index in 0..function_count {
            let at = PeImage::rva_add(Rva(functions), index * 4, "export address")
                .and_then(|rva| pe.offset_for_rva(rva))
                .map_err(unanalyzable("export directory"))?;
            let rva = reader
                .u32(at, "export function RVA")
                .map_err(unanalyzable("export directory"))?;
            checker.check(|| format!("export function #{index}"), rva);
        }
        for index in 0..name_count {
            let at = PeImage::rva_add(Rva(names), index * 4, "export name")
                .and_then(|rva| pe.offset_for_rva(rva))
                .map_err(unanalyzable("export directory"))?;
            let rva = reader
                .u32(at, "export name RVA")
                .map_err(unanalyzable("export directory"))?;
            checker.check(|| format!("export name #{index}"), rva);
        }
    }

    if live(IMAGE_DIRECTORY_ENTRY_TLS).is_some() {
        let tls = pe
            .parse_tls()
            .map_err(unanalyzable("TLS directory"))?
            .ok_or_else(|| StripError::Unanalyzable {
                what: "TLS directory",
                message: "directory present but unparsed".to_owned(),
            })?;
        checker.check_va(
            pe,
            || "TLS raw-data start".to_owned(),
            tls.start_address_of_raw_data.0,
        );
        checker.check_va(
            pe,
            || "TLS index address".to_owned(),
            tls.address_of_index.0,
        );
        checker.check_va(
            pe,
            || "TLS callback array".to_owned(),
            tls.address_of_callbacks.0,
        );
        for (index, callback) in tls.callbacks.iter().enumerate() {
            checker.check_va(pe, || format!("TLS callback #{index}"), callback.0);
        }
    }

    if live(IMAGE_DIRECTORY_ENTRY_LOAD_CONFIG).is_some() {
        let config = pe
            .parse_load_config()
            .map_err(unanalyzable("load config"))?
            .ok_or_else(|| StripError::Unanalyzable {
                what: "load config",
                message: "directory present but unparsed".to_owned(),
            })?;
        let directory = live(IMAGE_DIRECTORY_ENTRY_LOAD_CONFIG).map_or(0, |d| d.address);
        let base = pe
            .offset_for_rva(Rva(directory))
            .map_err(unanalyzable("load config"))?;
        let fields: &[(usize, &str)] = match pe.optional.kind {
            PeKind::Pe32 => &[
                (0x3C, "SecurityCookie"),
                (0x40, "SEHandlerTable"),
                (0x48, "GuardCFCheckFunctionPointer"),
                (0x4C, "GuardCFDispatchFunctionPointer"),
                (0x50, "GuardCFFunctionTable"),
            ],
            PeKind::Pe32Plus => &[
                (0x58, "SecurityCookie"),
                (0x70, "GuardCFCheckFunctionPointer"),
                (0x78, "GuardCFDispatchFunctionPointer"),
                (0x80, "GuardCFFunctionTable"),
            ],
        };
        for &(field, name) in fields {
            let covered =
                usize::try_from(config.declared_size).unwrap_or(0) >= field + pointer_size;
            if covered && let Some(value) = read_ptr(pe.data(), base + field, pointer_size) {
                checker.check_va(pe, || format!("load config {name}"), value);
            }
        }
    }

    if live(IMAGE_DIRECTORY_ENTRY_DEBUG).is_some() {
        for (index, entry) in pe
            .parse_debug()
            .map_err(unanalyzable("debug directory"))?
            .iter()
            .enumerate()
        {
            checker.check(
                || format!("debug entry {index} AddressOfRawData"),
                entry.address_of_raw_data.0,
            );
        }
    }

    if live(IMAGE_DIRECTORY_ENTRY_BASERELOC).is_some() {
        let blocks = pe
            .parse_relocations()
            .map_err(unanalyzable("base relocations"))?;
        let mut dropped_entries = 0;
        let mut kept_blocks: Vec<(Rva, Vec<u16>)> = Vec::new();
        for block in &blocks {
            let mut kept = Vec::new();
            for entry in &block.entries {
                if entry.kind == 0 {
                    continue;
                }
                let Some(target) = entry.target_rva else {
                    kept.push((u16::from(entry.kind) << 12) | entry.offset);
                    continue;
                };
                if checker.is_removed(target.0) {
                    dropped_entries += 1;
                    continue;
                }
                kept.push((u16::from(entry.kind) << 12) | entry.offset);
                let size = match entry.kind {
                    RELOC_HIGHLOW => 4,
                    RELOC_DIR64 => 8,
                    _ => continue,
                };
                let Ok(offset) = pe.offset_for_rva(target) else {
                    continue;
                };
                if let Some(value) = read_ptr(pe.data(), offset, size) {
                    checker.check_va(pe, || format!("relocated pointer at RVA {target}"), value);
                }
            }
            if !kept.is_empty() {
                kept_blocks.push((block.page_rva, kept));
            }
        }
        if dropped_entries > 0 {
            let mut bytes = Vec::new();
            for (page, mut entries) in kept_blocks {
                if entries.len() % 2 != 0 {
                    entries.push(0);
                }
                let size = u32::try_from(8 + entries.len() * 2)
                    .map_err(|_| layout("relocation block too large"))?;
                bytes.extend_from_slice(&page.0.to_le_bytes());
                bytes.extend_from_slice(&size.to_le_bytes());
                for entry in entries {
                    bytes.extend_from_slice(&entry.to_le_bytes());
                }
            }
            *relocation_entries_dropped = dropped_entries;
            rewrites.push(TableRewrite {
                directory: IMAGE_DIRECTORY_ENTRY_BASERELOC,
                bytes,
            });
        }
    }

    if live(IMAGE_DIRECTORY_ENTRY_EXCEPTION).is_some() {
        if pe.coff.machine != MACHINE_AMD64 {
            return Err(StripError::Unanalyzable {
                what: "exception directory",
                message: "only AMD64 unwind data is analyzed".to_owned(),
            });
        }
        let functions = pe
            .parse_exception_functions()
            .map_err(unanalyzable("exception directory"))?;
        let mut bytes = Vec::new();
        let mut dropped_entries = 0;
        for function in &functions {
            if checker.is_removed(function.begin.0) {
                dropped_entries += 1;
                continue;
            }
            let begin = function.begin.0;
            let unwind = function.unwind_info.0;
            checker.check(
                || format!("runtime function {} UnwindData", function.begin),
                unwind & !1,
            );
            if unwind & 1 == 0 && unwind != 0 {
                let info = pe
                    .parse_unwind_info(function.unwind_info)
                    .map_err(unanalyzable("unwind info"))?;
                if let Some(handler) = info.exception_handler {
                    checker.check(
                        || format!("unwind handler of function {}", Rva(begin)),
                        handler.0,
                    );
                }
                if let Some(chained) = info.chained_function {
                    checker.check(
                        || format!("chained unwind of function {}", Rva(begin)),
                        chained.begin.0,
                    );
                    checker.check(
                        || format!("chained unwind of function {}", Rva(begin)),
                        chained.unwind_info.0 & !1,
                    );
                }
            }
            for value in [function.begin.0, function.end.0, function.unwind_info.0] {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        if dropped_entries > 0 {
            *runtime_functions_dropped = dropped_entries;
            rewrites.push(TableRewrite {
                directory: IMAGE_DIRECTORY_ENTRY_EXCEPTION,
                bytes,
            });
        }
    }
    Ok(())
}

/// Writes a shrunken table over its original location and updates the
/// directory size (zeroing the entry when the table becomes empty).
fn apply_table_rewrite(
    pe: &PeImage<'_>,
    map: &FileMap,
    out: &mut [u8],
    directory_base: usize,
    rewrite: &TableRewrite,
) -> Result<(), StripError> {
    let directory = pe
        .directories
        .get(rewrite.directory)
        .ok_or_else(|| layout("rewritten directory index missing"))?;
    let section = pe
        .sections
        .iter()
        .find(|s| s.contains_mapped_rva(Rva(directory.address)))
        .ok_or_else(|| {
            layout(format!(
                "{} directory is not owned by a section",
                directory.name
            ))
        })?;
    if !section.contains_mapped_rva(Rva(directory.address + directory.size - 1)) {
        return Err(layout(format!(
            "{} directory spans multiple sections",
            directory.name
        )));
    }
    let in_offset = pe
        .rva_to_file_offset(Rva(directory.address))
        .map_err(pe_err("directory"))?;
    let out_offset = to_usize(
        map.map(in_offset.0)
            .ok_or_else(|| layout("directory file offset was deleted"))?,
        "directory offset",
    )?;
    let old_size = to_usize(u64::from(directory.size), "directory size")?;
    let region = out
        .get_mut(out_offset..out_offset + old_size)
        .ok_or_else(|| layout("directory exceeds output"))?;
    region.fill(0);
    region[..rewrite.bytes.len()].copy_from_slice(&rewrite.bytes);
    let slot = directory_base + rewrite.directory * 8;
    let new_size = u32::try_from(rewrite.bytes.len()).map_err(|_| layout("table too large"))?;
    if new_size == 0 {
        put_u32(out, slot, 0)?;
    }
    put_u32(out, slot + 4, new_size)
}

fn fix_debug_pointers(
    pe: &PeImage<'_>,
    dropped: &[usize],
    map: &FileMap,
    out: &mut [u8],
) -> Result<(), StripError> {
    let Some(directory) = pe
        .directory(IMAGE_DIRECTORY_ENTRY_DEBUG)
        .filter(|_| !dropped.contains(&IMAGE_DIRECTORY_ENTRY_DEBUG))
    else {
        return Ok(());
    };
    let entries = pe.parse_debug().map_err(unanalyzable("debug directory"))?;
    for (index, entry) in entries.iter().enumerate() {
        if entry.pointer_to_raw_data.0 == 0 {
            continue;
        }
        let field_rva = Rva(directory.address + u32::try_from(index * 28 + 24).unwrap_or(0));
        let in_field = pe
            .rva_to_file_offset(field_rva)
            .map_err(pe_err("debug directory"))?;
        let out_field = map
            .map(in_field.0)
            .ok_or_else(|| layout("debug directory entry was deleted"))?;
        let mapped = map.map(entry.pointer_to_raw_data.0).ok_or_else(|| {
            StripError::Dangling(vec![DanglingRef {
                source: format!("debug entry {index} PointerToRawData"),
                target: format!("file offset {}", entry.pointer_to_raw_data),
                section: "<removed raw data>".to_owned(),
            }])
        })?;
        put_u32(
            out,
            to_usize(out_field, "debug field offset")?,
            u32::try_from(mapped).map_err(|_| layout("debug pointer exceeds 32 bits"))?,
        )?;
    }
    Ok(())
}

/// Checks, before any in-place table rewrite, that every retained section's
/// raw bytes landed at the mapped offset unchanged.
fn verify_raw_moves(
    input: &PeImage<'_>,
    out: &[u8],
    removed: &[bool],
    map: &FileMap,
) -> Result<(), StripError> {
    let reparsed = PeImage::parse(out).map_err(pe_err("output"))?;
    let expected = input.sections.iter().filter(|s| !removed[s.index]);
    for (before, after) in expected.zip(&reparsed.sections) {
        if before.size_of_raw_data != 0 && before.pointer_to_raw_data.0 != 0 {
            let source = to_usize(before.pointer_to_raw_data.0, "raw offset")?;
            let size = to_usize(u64::from(before.size_of_raw_data), "raw size")?;
            let dest = to_usize(after.pointer_to_raw_data.0, "raw offset")?;
            let mapped = map.map(before.pointer_to_raw_data.0);
            if mapped != Some(after.pointer_to_raw_data.0)
                || input.data().get(source..source + size) != out.get(dest..dest + size)
            {
                return Err(StripError::OutputInvariant(format!(
                    "section {} raw bytes moved incorrectly",
                    before.name
                )));
            }
        }
    }
    Ok(())
}

fn verify_output(input: &PeImage<'_>, out: &[u8], removed: &[bool]) -> Result<(), StripError> {
    let reparsed = PeImage::parse(out).map_err(pe_err("output"))?;
    let expected: Vec<&Section> = input
        .sections
        .iter()
        .filter(|s| !removed[s.index])
        .collect();
    if reparsed.sections.len() != expected.len()
        || usize::from(reparsed.coff.number_of_sections) != expected.len()
    {
        return Err(StripError::OutputInvariant(
            "section count mismatch".to_owned(),
        ));
    }
    for (before, after) in expected.iter().zip(&reparsed.sections) {
        if before.name_bytes != after.name_bytes
            || before.virtual_address != after.virtual_address
            || before.virtual_size != after.virtual_size
            || before.size_of_raw_data != after.size_of_raw_data
            || before.characteristics != after.characteristics
        {
            return Err(StripError::OutputInvariant(format!(
                "section {} metadata changed",
                before.name
            )));
        }
    }
    let before_messages: Vec<String> = input
        .report()
        .diagnostics
        .into_iter()
        .map(|d| d.message)
        .collect();
    let introduced: Vec<String> = reparsed
        .report()
        .diagnostics
        .into_iter()
        .map(|d| d.message)
        .filter(|message| !before_messages.contains(message))
        .collect();
    if !introduced.is_empty() {
        return Err(StripError::OutputInvariant(format!(
            "output introduces validation diagnostics: {}",
            introduced.join("; ")
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
