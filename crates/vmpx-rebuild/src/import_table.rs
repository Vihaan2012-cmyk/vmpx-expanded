use std::collections::BTreeSet;

use thiserror::Error;
use vmpx_pe::{
    IMAGE_DIRECTORY_ENTRY_IAT, IMAGE_DIRECTORY_ENTRY_IMPORT, ImportModule, ImportSymbol, PeError,
    PeImage, PeKind, Rva,
};

const IMPORT_DESCRIPTOR_SIZE: usize = 20;
const REQUIRED_DIRECTORY_COUNT: u32 = 13;
const MAX_SERIALIZED_MODULES: usize = 65_535;
const MAX_SERIALIZED_SYMBOLS_PER_MODULE: usize = 999_999;
const MAX_SERIALIZED_SYMBOLS: usize = 1_000_000;
const MAX_IMPORT_STRING_BYTES: usize = 16 * 1_024 - 1;
const MAX_SERIALIZED_BYTES: usize = 256 * 1024 * 1024;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ImportTarget {
    Name { name: String, hint: u16 },
    Ordinal(u16),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImportModuleSpec {
    pub name: String,
    pub symbols: Vec<ImportTarget>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SerializedDirectoryRange {
    pub rva: Rva,
    pub size: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SerializedImportTable {
    pub bytes: Vec<u8>,
    pub import_directory: SerializedDirectoryRange,
    pub iat_directory: SerializedDirectoryRange,
    /// Parser-equivalent description of the newly serialized layout.
    pub modules: Vec<ImportModule>,
}

/// One exact source-IAT to rebuilt-IAT correspondence established while
/// planning an import-table reconstruction.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImportSlotRemap {
    pub source_iat_rva: Rva,
    pub rebuilt_iat_rva: Rva,
    pub module: String,
    pub target: ImportTarget,
}

/// A serialized import table plus the information required to redirect every
/// reference that formerly addressed one of the resolved source IAT slots.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImportReconstructionPlan {
    pub table: SerializedImportTable,
    pub slot_remaps: Vec<ImportSlotRemap>,
    /// Named imports whose source evidence did not retain a linker hint. A
    /// zero hint is serialized for these entries; this does not affect loader
    /// name resolution but remains explicit in the plan.
    pub defaulted_hints: usize,
}

struct ResolvedSourceImports {
    specifications: Vec<ImportModuleSpec>,
    source_slots: Vec<Rva>,
    defaulted_hints: usize,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum ImportTableError {
    #[error("cannot serialize an empty import table")]
    EmptyTable,

    #[error("import table base RVA must be nonzero")]
    ZeroBaseRva,

    #[error("import module count {count} exceeds limit {limit}")]
    TooManyModules { count: usize, limit: usize },

    #[error("import module {module} has no symbols")]
    EmptyModule { module: usize },

    #[error("import module {module} symbol count {count} exceeds limit {limit}")]
    TooManyModuleSymbols {
        module: usize,
        count: usize,
        limit: usize,
    },

    #[error("total import symbol count {count} exceeds limit {limit}")]
    TooManySymbols { count: usize, limit: usize },

    #[error("{what} is empty")]
    EmptyString { what: String },

    #[error("{what} contains a NUL byte")]
    EmbeddedNul { what: String },

    #[error("{what} contains non-ASCII bytes")]
    NonAscii { what: String },

    #[error("{what} length {length} exceeds limit {limit}")]
    StringTooLong {
        what: String,
        length: usize,
        limit: usize,
    },

    #[error("{what} arithmetic overflows")]
    Overflow { what: &'static str },

    #[error("serialized import payload size {size} exceeds limit {limit}")]
    PayloadTooLarge { size: usize, limit: usize },

    #[error("allocation of {requested} import-table bytes failed")]
    Allocation { requested: usize },

    #[error("source import module {module} symbol {symbol} has a zero IAT RVA")]
    ZeroSourceIat { module: usize, symbol: usize },

    #[error("source IAT RVA {rva:?} is claimed by more than one import symbol")]
    DuplicateSourceIat { rva: Rva },

    #[error(
        "source import module {module} symbol {symbol} has ambiguous or unresolved identity (name={has_name}, ordinal={has_ordinal})"
    )]
    UnresolvedSourceSymbol {
        module: usize,
        symbol: usize,
        has_name: bool,
        has_ordinal: bool,
    },

    #[error("failed to parse {stage} PE while installing imports: {source}")]
    Pe {
        stage: &'static str,
        source: PeError,
    },

    #[error("optional header exposes {count} data directories; import installation requires 13")]
    MissingDirectories { count: u32 },

    #[error("optional header is too small to contain the {directory} directory entry")]
    DirectoryHeaderBounds { directory: &'static str },

    #[error(
        "serialized {directory} range at RVA {rva:?}, size {size:#X} is not one contiguous file-backed range"
    )]
    DirectoryNotFileBacked {
        directory: &'static str,
        rva: Rva,
        size: u32,
    },

    #[error("installed import-table invariant failed: {message}")]
    InstallInvariant { message: String },
}

fn validate_string(value: &str, what: String) -> Result<(), ImportTableError> {
    if value.is_empty() {
        return Err(ImportTableError::EmptyString { what });
    }
    if value.as_bytes().contains(&0) {
        return Err(ImportTableError::EmbeddedNul { what });
    }
    if !value.is_ascii() {
        return Err(ImportTableError::NonAscii { what });
    }
    if value.len() > MAX_IMPORT_STRING_BYTES {
        return Err(ImportTableError::StringTooLong {
            what,
            length: value.len(),
            limit: MAX_IMPORT_STRING_BYTES,
        });
    }
    Ok(())
}

fn checked_add(value: usize, amount: usize, what: &'static str) -> Result<usize, ImportTableError> {
    value
        .checked_add(amount)
        .ok_or(ImportTableError::Overflow { what })
}

fn checked_mul(value: usize, amount: usize, what: &'static str) -> Result<usize, ImportTableError> {
    value
        .checked_mul(amount)
        .ok_or(ImportTableError::Overflow { what })
}

fn align_offset(
    value: usize,
    alignment: usize,
    what: &'static str,
) -> Result<usize, ImportTableError> {
    let mask = alignment
        .checked_sub(1)
        .ok_or(ImportTableError::Overflow { what })?;
    checked_add(value, mask, what).map(|value| value & !mask)
}

fn rva_at(base: Rva, offset: usize, what: &'static str) -> Result<Rva, ImportTableError> {
    let offset = u32::try_from(offset).map_err(|_| ImportTableError::Overflow { what })?;
    base.checked_add(offset)
        .ok_or(ImportTableError::Overflow { what })
}

fn write_bytes(
    output: &mut [u8],
    offset: usize,
    value: &[u8],
    what: &'static str,
) -> Result<(), ImportTableError> {
    let end = checked_add(offset, value.len(), what)?;
    let target = output
        .get_mut(offset..end)
        .ok_or(ImportTableError::Overflow { what })?;
    target.copy_from_slice(value);
    Ok(())
}

fn write_u16(
    output: &mut [u8],
    offset: usize,
    value: u16,
    what: &'static str,
) -> Result<(), ImportTableError> {
    write_bytes(output, offset, &value.to_le_bytes(), what)
}

fn write_u32(
    output: &mut [u8],
    offset: usize,
    value: u32,
    what: &'static str,
) -> Result<(), ImportTableError> {
    write_bytes(output, offset, &value.to_le_bytes(), what)
}

fn write_pointer(
    output: &mut [u8],
    offset: usize,
    value: u64,
    kind: PeKind,
) -> Result<(), ImportTableError> {
    match kind {
        PeKind::Pe32 => write_u32(
            output,
            offset,
            u32::try_from(value).map_err(|_| ImportTableError::Overflow {
                what: "PE32 thunk value",
            })?,
            "PE32 thunk",
        ),
        PeKind::Pe32Plus => write_bytes(output, offset, &value.to_le_bytes(), "PE32+ thunk"),
    }
}

/// Serializes a conventional eager import table at a caller-chosen section RVA.
///
/// The payload contains the null-terminated `IMAGE_IMPORT_DESCRIPTOR` array,
/// separate import-name tables, one contiguous IAT range, aligned
/// `IMAGE_IMPORT_BY_NAME` records, and DLL strings. The returned directory
/// ranges are ready to install in optional-header entries 1 and 12.
///
/// Inputs are semantic import specifications: no original layout RVAs or raw
/// thunk values are trusted. Names are restricted to nonempty ASCII strings so
/// every accepted value round-trips through the current PE parser losslessly.
///
/// # Errors
///
/// Rejects empty/malformed names and modules, parser-incompatible counts,
/// arithmetic/RVA overflow, payloads beyond the fixed resource bound, and
/// allocation failure.
#[allow(clippy::too_many_lines)]
pub fn serialize_import_table(
    kind: PeKind,
    section_rva: Rva,
    modules: &[ImportModuleSpec],
) -> Result<SerializedImportTable, ImportTableError> {
    if modules.is_empty() {
        return Err(ImportTableError::EmptyTable);
    }
    if section_rva.0 == 0 {
        return Err(ImportTableError::ZeroBaseRva);
    }
    if modules.len() > MAX_SERIALIZED_MODULES {
        return Err(ImportTableError::TooManyModules {
            count: modules.len(),
            limit: MAX_SERIALIZED_MODULES,
        });
    }

    let mut total_symbols = 0_usize;
    for (module_index, module) in modules.iter().enumerate() {
        validate_string(&module.name, format!("import module {module_index} name"))?;
        if module.symbols.is_empty() {
            return Err(ImportTableError::EmptyModule {
                module: module_index,
            });
        }
        if module.symbols.len() > MAX_SERIALIZED_SYMBOLS_PER_MODULE {
            return Err(ImportTableError::TooManyModuleSymbols {
                module: module_index,
                count: module.symbols.len(),
                limit: MAX_SERIALIZED_SYMBOLS_PER_MODULE,
            });
        }
        total_symbols = checked_add(total_symbols, module.symbols.len(), "total symbol count")?;
        if total_symbols > MAX_SERIALIZED_SYMBOLS {
            return Err(ImportTableError::TooManySymbols {
                count: total_symbols,
                limit: MAX_SERIALIZED_SYMBOLS,
            });
        }
        for (symbol_index, symbol) in module.symbols.iter().enumerate() {
            if let ImportTarget::Name { name, .. } = symbol {
                validate_string(
                    name,
                    format!("import module {module_index} symbol {symbol_index} name"),
                )?;
            }
        }
    }

    let pointer_size = kind.pointer_size();
    let descriptor_count = checked_add(modules.len(), 1, "import descriptor count")?;
    let descriptor_size = checked_mul(
        descriptor_count,
        IMPORT_DESCRIPTOR_SIZE,
        "import descriptor array",
    )?;
    let mut cursor = align_offset(descriptor_size, pointer_size, "INT alignment")?;

    let mut lookup_table_offsets = Vec::with_capacity(modules.len());
    for module in modules {
        lookup_table_offsets.push(cursor);
        let thunk_count = checked_add(module.symbols.len(), 1, "INT thunk count")?;
        cursor = checked_add(
            cursor,
            checked_mul(thunk_count, pointer_size, "INT byte size")?,
            "INT end",
        )?;
    }

    cursor = align_offset(cursor, pointer_size, "IAT alignment")?;
    let iat_offset = cursor;
    let mut address_table_offsets = Vec::with_capacity(modules.len());
    for module in modules {
        address_table_offsets.push(cursor);
        let thunk_count = checked_add(module.symbols.len(), 1, "IAT thunk count")?;
        cursor = checked_add(
            cursor,
            checked_mul(thunk_count, pointer_size, "IAT byte size")?,
            "IAT end",
        )?;
    }
    let iat_size = cursor
        .checked_sub(iat_offset)
        .ok_or(ImportTableError::Overflow { what: "IAT size" })?;

    let mut import_name_offsets = Vec::with_capacity(modules.len());
    for module in modules {
        let mut offsets = Vec::with_capacity(module.symbols.len());
        for symbol in &module.symbols {
            if let ImportTarget::Name { name, .. } = symbol {
                cursor = align_offset(cursor, 2, "import-by-name alignment")?;
                offsets.push(Some(cursor));
                cursor = checked_add(cursor, 2, "import hint")?;
                cursor = checked_add(cursor, name.len(), "import symbol name")?;
                cursor = checked_add(cursor, 1, "import symbol terminator")?;
            } else {
                offsets.push(None);
            }
        }
        import_name_offsets.push(offsets);
    }

    let mut module_name_offsets = Vec::with_capacity(modules.len());
    for module in modules {
        module_name_offsets.push(cursor);
        cursor = checked_add(cursor, module.name.len(), "import module name")?;
        cursor = checked_add(cursor, 1, "import module terminator")?;
    }

    if cursor > MAX_SERIALIZED_BYTES {
        return Err(ImportTableError::PayloadTooLarge {
            size: cursor,
            limit: MAX_SERIALIZED_BYTES,
        });
    }
    if cursor != 0 {
        rva_at(section_rva, cursor - 1, "import payload end RVA")?;
    }

    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(cursor)
        .map_err(|_| ImportTableError::Allocation { requested: cursor })?;
    bytes.resize(cursor, 0);

    let ordinal_mask = match kind {
        PeKind::Pe32 => 0x8000_0000_u64,
        PeKind::Pe32Plus => 0x8000_0000_0000_0000_u64,
    };
    let mut serialized_modules = Vec::with_capacity(modules.len());
    for (module_index, module) in modules.iter().enumerate() {
        let descriptor_offset = checked_mul(
            module_index,
            IMPORT_DESCRIPTOR_SIZE,
            "import descriptor offset",
        )?;
        let descriptor_rva = rva_at(section_rva, descriptor_offset, "descriptor RVA")?;
        let lookup_table_rva = rva_at(section_rva, lookup_table_offsets[module_index], "INT RVA")?;
        let module_address_table_rva =
            rva_at(section_rva, address_table_offsets[module_index], "IAT RVA")?;
        let module_name_rva = rva_at(
            section_rva,
            module_name_offsets[module_index],
            "module-name RVA",
        )?;
        write_u32(
            &mut bytes,
            descriptor_offset,
            lookup_table_rva.0,
            "OriginalFirstThunk",
        )?;
        write_u32(
            &mut bytes,
            checked_add(descriptor_offset, 12, "descriptor name field")?,
            module_name_rva.0,
            "descriptor name RVA",
        )?;
        write_u32(
            &mut bytes,
            checked_add(descriptor_offset, 16, "descriptor IAT field")?,
            module_address_table_rva.0,
            "FirstThunk",
        )?;
        write_bytes(
            &mut bytes,
            module_name_offsets[module_index],
            module.name.as_bytes(),
            "module name",
        )?;

        let mut entries = Vec::with_capacity(module.symbols.len());
        for (symbol_index, symbol) in module.symbols.iter().enumerate() {
            let thunk_delta = checked_mul(symbol_index, pointer_size, "thunk entry offset")?;
            let int_entry_offset = checked_add(
                lookup_table_offsets[module_index],
                thunk_delta,
                "INT entry offset",
            )?;
            let iat_entry_offset = checked_add(
                address_table_offsets[module_index],
                thunk_delta,
                "IAT entry offset",
            )?;
            let address_entry_rva = rva_at(section_rva, iat_entry_offset, "IAT entry RVA")?;
            let (lookup_value, name, hint, ordinal) = match symbol {
                ImportTarget::Name { name, hint } => {
                    let name_offset = import_name_offsets[module_index][symbol_index].ok_or(
                        ImportTableError::Overflow {
                            what: "import-by-name offset",
                        },
                    )?;
                    let name_rva = rva_at(section_rva, name_offset, "import-by-name RVA")?;
                    write_u16(&mut bytes, name_offset, *hint, "import hint")?;
                    write_bytes(
                        &mut bytes,
                        checked_add(name_offset, 2, "import symbol string offset")?,
                        name.as_bytes(),
                        "import symbol name",
                    )?;
                    (name_rva.0.into(), Some(name.clone()), Some(*hint), None)
                }
                ImportTarget::Ordinal(ordinal) => (
                    ordinal_mask | u64::from(*ordinal),
                    None,
                    None,
                    Some(*ordinal),
                ),
            };
            write_pointer(&mut bytes, int_entry_offset, lookup_value, kind)?;
            write_pointer(&mut bytes, iat_entry_offset, lookup_value, kind)?;
            entries.push(ImportSymbol {
                iat_rva: address_entry_rva,
                lookup_value,
                name,
                hint,
                ordinal,
            });
        }
        serialized_modules.push(ImportModule {
            name: module.name.clone(),
            descriptor_rva,
            original_first_thunk: lookup_table_rva,
            first_thunk: module_address_table_rva,
            entries,
        });
    }

    Ok(SerializedImportTable {
        bytes,
        import_directory: SerializedDirectoryRange {
            rva: section_rva,
            size: u32::try_from(descriptor_size).map_err(|_| ImportTableError::Overflow {
                what: "import directory size",
            })?,
        },
        iat_directory: SerializedDirectoryRange {
            rva: rva_at(section_rva, iat_offset, "IAT directory RVA")?,
            size: u32::try_from(iat_size).map_err(|_| ImportTableError::Overflow {
                what: "IAT directory size",
            })?,
        },
        modules: serialized_modules,
    })
}

fn resolve_source_imports(
    source_modules: &[ImportModule],
) -> Result<ResolvedSourceImports, ImportTableError> {
    if source_modules.len() > MAX_SERIALIZED_MODULES {
        return Err(ImportTableError::TooManyModules {
            count: source_modules.len(),
            limit: MAX_SERIALIZED_MODULES,
        });
    }

    let mut specifications = Vec::new();
    specifications
        .try_reserve_exact(source_modules.len())
        .map_err(|_| ImportTableError::Allocation {
            requested: source_modules.len(),
        })?;
    let mut source_slots = Vec::new();
    let mut seen_source_slots = BTreeSet::new();
    let mut defaulted_hints = 0_usize;
    let mut total_source_symbols = 0_usize;

    for (module_index, module) in source_modules.iter().enumerate() {
        if module.entries.len() > MAX_SERIALIZED_SYMBOLS_PER_MODULE {
            return Err(ImportTableError::TooManyModuleSymbols {
                module: module_index,
                count: module.entries.len(),
                limit: MAX_SERIALIZED_SYMBOLS_PER_MODULE,
            });
        }
        total_source_symbols = checked_add(
            total_source_symbols,
            module.entries.len(),
            "source import symbol count",
        )?;
        if total_source_symbols > MAX_SERIALIZED_SYMBOLS {
            return Err(ImportTableError::TooManySymbols {
                count: total_source_symbols,
                limit: MAX_SERIALIZED_SYMBOLS,
            });
        }
        let mut symbols = Vec::new();
        symbols
            .try_reserve_exact(module.entries.len())
            .map_err(|_| ImportTableError::Allocation {
                requested: module.entries.len(),
            })?;
        source_slots
            .try_reserve(module.entries.len())
            .map_err(|_| ImportTableError::Allocation {
                requested: module.entries.len(),
            })?;

        for (symbol_index, symbol) in module.entries.iter().enumerate() {
            if symbol.iat_rva.0 == 0 {
                return Err(ImportTableError::ZeroSourceIat {
                    module: module_index,
                    symbol: symbol_index,
                });
            }
            if !seen_source_slots.insert(symbol.iat_rva) {
                return Err(ImportTableError::DuplicateSourceIat {
                    rva: symbol.iat_rva,
                });
            }
            let target = match (&symbol.name, symbol.ordinal) {
                (Some(name), None) => ImportTarget::Name {
                    name: name.clone(),
                    hint: symbol.hint.unwrap_or_else(|| {
                        defaulted_hints = defaulted_hints.saturating_add(1);
                        0
                    }),
                },
                (None, Some(ordinal)) => ImportTarget::Ordinal(ordinal),
                (name, ordinal) => {
                    return Err(ImportTableError::UnresolvedSourceSymbol {
                        module: module_index,
                        symbol: symbol_index,
                        has_name: name.is_some(),
                        has_ordinal: ordinal.is_some(),
                    });
                }
            };
            symbols.push(target);
            source_slots.push(symbol.iat_rva);
        }
        specifications.push(ImportModuleSpec {
            name: module.name.clone(),
            symbols,
        });
    }

    Ok(ResolvedSourceImports {
        specifications,
        source_slots,
        defaulted_hints,
    })
}

/// Converts parser/recovery import evidence into a reconstruction plan at an
/// exact caller-selected table RVA.
///
/// The source modules supply both resolved identities and their old IAT slot
/// RVAs. Every accepted source slot receives one exact rebuilt slot in
/// [`ImportReconstructionPlan::slot_remaps`]. This lets a later code-rewrite
/// stage retarget references without matching symbols by name or relying on an
/// oracle. Duplicate source slots and symbols lacking one unambiguous name or
/// ordinal identity are rejected.
///
/// A missing hint on a named import is represented by hint zero and counted in
/// `defaulted_hints`; PE loader resolution is by name, not by hint. No DLL or
/// symbol identity is inferred.
///
/// # Errors
///
/// Returns the same validation and resource errors as
/// [`serialize_import_table`], plus errors for unresolved identities, zero IAT
/// RVAs, duplicate source slots, or allocation failure while constructing the
/// bounded plan.
pub fn plan_import_reconstruction(
    kind: PeKind,
    table_rva: Rva,
    source_modules: &[ImportModule],
) -> Result<ImportReconstructionPlan, ImportTableError> {
    let resolved = resolve_source_imports(source_modules)?;
    let table = serialize_import_table(kind, table_rva, &resolved.specifications)?;
    let mut slot_remaps = Vec::new();
    slot_remaps
        .try_reserve_exact(resolved.source_slots.len())
        .map_err(|_| ImportTableError::Allocation {
            requested: resolved.source_slots.len(),
        })?;
    let mut source_slot_index = 0_usize;
    for (module, specification) in table.modules.iter().zip(&resolved.specifications) {
        for (entry, target) in module.entries.iter().zip(&specification.symbols) {
            let source_iat_rva =
                *resolved
                    .source_slots
                    .get(source_slot_index)
                    .ok_or_else(|| ImportTableError::InstallInvariant {
                        message: "source/rebuilt import slot counts differ".to_owned(),
                    })?;
            source_slot_index =
                source_slot_index
                    .checked_add(1)
                    .ok_or(ImportTableError::Overflow {
                        what: "source import slot index",
                    })?;
            slot_remaps.push(ImportSlotRemap {
                source_iat_rva,
                rebuilt_iat_rva: entry.iat_rva,
                module: module.name.clone(),
                target: target.clone(),
            });
        }
    }
    if source_slot_index != resolved.source_slots.len() {
        return Err(ImportTableError::InstallInvariant {
            message: "source/rebuilt import slot counts differ".to_owned(),
        });
    }

    Ok(ImportReconstructionPlan {
        table,
        slot_remaps,
        defaulted_hints: resolved.defaulted_hints,
    })
}

fn directory_field_offset(
    pe: &PeImage<'_>,
    index: usize,
    name: &'static str,
) -> Result<usize, ImportTableError> {
    let relative = match pe.optional.kind {
        PeKind::Pe32 => 96_usize,
        PeKind::Pe32Plus => 112_usize,
    }
    .checked_add(index.checked_mul(8).ok_or(ImportTableError::Overflow {
        what: "directory index",
    })?)
    .ok_or(ImportTableError::Overflow {
        what: "directory offset",
    })?;
    let required = relative.checked_add(8).ok_or(ImportTableError::Overflow {
        what: "directory end",
    })?;
    if required > usize::from(pe.coff.size_of_optional_header) {
        return Err(ImportTableError::DirectoryHeaderBounds { directory: name });
    }
    let optional = usize::try_from(pe.nt_headers_offset.0)
        .ok()
        .and_then(|offset| offset.checked_add(4 + 20))
        .ok_or(ImportTableError::Overflow {
            what: "optional-header file offset",
        })?;
    optional
        .checked_add(relative)
        .ok_or(ImportTableError::Overflow {
            what: "directory file offset",
        })
}

fn ensure_file_backed_directory(
    pe: &PeImage<'_>,
    directory: &'static str,
    range: SerializedDirectoryRange,
) -> Result<(), ImportTableError> {
    let last_delta = range
        .size
        .checked_sub(1)
        .ok_or(ImportTableError::DirectoryNotFileBacked {
            directory,
            rva: range.rva,
            size: range.size,
        })?;
    let last_rva =
        range
            .rva
            .checked_add(last_delta)
            .ok_or(ImportTableError::DirectoryNotFileBacked {
                directory,
                rva: range.rva,
                size: range.size,
            })?;
    let first =
        pe.rva_to_file_offset(range.rva)
            .map_err(|_| ImportTableError::DirectoryNotFileBacked {
                directory,
                rva: range.rva,
                size: range.size,
            })?;
    let last =
        pe.rva_to_file_offset(last_rva)
            .map_err(|_| ImportTableError::DirectoryNotFileBacked {
                directory,
                rva: range.rva,
                size: range.size,
            })?;
    if last.0.checked_sub(first.0) != Some(u64::from(last_delta)) {
        return Err(ImportTableError::DirectoryNotFileBacked {
            directory,
            rva: range.rva,
            size: range.size,
        });
    }
    Ok(())
}

/// Installs the import and IAT directory entries for a serialized table that
/// is already present in a file-backed PE range. The result is reparsed and its
/// semantic import list must exactly match the serializer output.
///
/// This function does not copy table bytes into the image. A caller normally
/// computes [`crate::next_section_rva`], serializes at an RVA within that
/// section, appends the payload with [`crate::append_section`], and then calls
/// this installer on the appended image.
///
/// # Errors
///
/// Rejects malformed PEs, absent optional-header directory capacity,
/// non-file-backed directory ranges, offset overflow, and any parser-visible
/// mismatch after installation.
pub fn install_import_directories(
    input: &[u8],
    table: &SerializedImportTable,
) -> Result<Vec<u8>, ImportTableError> {
    let pe = PeImage::parse(input).map_err(|source| ImportTableError::Pe {
        stage: "input",
        source,
    })?;
    if pe.optional.number_of_rva_and_sizes < REQUIRED_DIRECTORY_COUNT {
        return Err(ImportTableError::MissingDirectories {
            count: pe.optional.number_of_rva_and_sizes,
        });
    }
    ensure_file_backed_directory(&pe, "import", table.import_directory)?;
    ensure_file_backed_directory(&pe, "IAT", table.iat_directory)?;
    let import_offset = directory_field_offset(&pe, IMAGE_DIRECTORY_ENTRY_IMPORT, "import")?;
    let iat_offset = directory_field_offset(&pe, IMAGE_DIRECTORY_ENTRY_IAT, "IAT")?;

    let mut output = Vec::new();
    output
        .try_reserve_exact(input.len())
        .map_err(|_| ImportTableError::Allocation {
            requested: input.len(),
        })?;
    output.extend_from_slice(input);
    write_u32(
        &mut output,
        import_offset,
        table.import_directory.rva.0,
        "import directory RVA",
    )?;
    write_u32(
        &mut output,
        checked_add(import_offset, 4, "import directory size offset")?,
        table.import_directory.size,
        "import directory size",
    )?;
    write_u32(
        &mut output,
        iat_offset,
        table.iat_directory.rva.0,
        "IAT directory RVA",
    )?;
    write_u32(
        &mut output,
        checked_add(iat_offset, 4, "IAT directory size offset")?,
        table.iat_directory.size,
        "IAT directory size",
    )?;

    let reparsed = PeImage::parse(&output).map_err(|source| ImportTableError::Pe {
        stage: "output",
        source,
    })?;
    let report = reparsed.report();
    if report.imports != table.modules {
        return Err(ImportTableError::InstallInvariant {
            message: "reparsed imports differ from the serialized specification".to_owned(),
        });
    }
    for (index, expected, name) in [
        (
            IMAGE_DIRECTORY_ENTRY_IMPORT,
            table.import_directory,
            "import",
        ),
        (IMAGE_DIRECTORY_ENTRY_IAT, table.iat_directory, "IAT"),
    ] {
        let Some(actual) = reparsed.directory(index) else {
            return Err(ImportTableError::InstallInvariant {
                message: format!("{name} directory is absent after reparse"),
            });
        };
        if actual.address != expected.rva.0 || actual.size != expected.size {
            return Err(ImportTableError::InstallInvariant {
                message: format!("{name} directory differs after reparse"),
            });
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use vmpx_pe::{IMAGE_DIRECTORY_ENTRY_IAT, IMAGE_DIRECTORY_ENTRY_IMPORT, PeImage};

    use super::*;

    const NT_OFFSET: usize = 0x80;
    const SECTION_RVA: Rva = Rva(0x1000);

    fn put_u16(bytes: &mut [u8], offset: usize, value: u16) {
        bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
    }

    fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
        bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }

    fn align(value: usize, alignment: usize) -> usize {
        (value + alignment - 1) & !(alignment - 1)
    }

    fn synthetic_pe(kind: PeKind, table: &SerializedImportTable) -> Vec<u8> {
        let optional_size = match kind {
            PeKind::Pe32 => 0xE0,
            PeKind::Pe32Plus => 0xF0,
        };
        let directory_offset = match kind {
            PeKind::Pe32 => 96,
            PeKind::Pe32Plus => 112,
        };
        let optional_offset = NT_OFFSET + 4 + 20;
        let section_offset = optional_offset + optional_size;
        let raw_offset = 0x200;
        let raw_size = align(table.bytes.len(), 0x200);
        let mut bytes = vec![0_u8; raw_offset + raw_size];

        put_u16(&mut bytes, 0, 0x5A4D);
        put_u32(
            &mut bytes,
            0x3C,
            u32::try_from(NT_OFFSET).expect("small NT offset"),
        );
        put_u32(&mut bytes, NT_OFFSET, 0x0000_4550);
        let coff = NT_OFFSET + 4;
        put_u16(
            &mut bytes,
            coff,
            match kind {
                PeKind::Pe32 => 0x014C,
                PeKind::Pe32Plus => 0x8664,
            },
        );
        put_u16(&mut bytes, coff + 2, 1);
        put_u16(
            &mut bytes,
            coff + 16,
            u16::try_from(optional_size).expect("small optional-header size"),
        );
        put_u16(&mut bytes, coff + 18, 0x2022);

        put_u16(
            &mut bytes,
            optional_offset,
            match kind {
                PeKind::Pe32 => 0x010B,
                PeKind::Pe32Plus => 0x020B,
            },
        );
        put_u32(
            &mut bytes,
            optional_offset + 8,
            u32::try_from(raw_size).expect("small raw size"),
        );
        put_u32(&mut bytes, optional_offset + 16, SECTION_RVA.0);
        put_u32(&mut bytes, optional_offset + 20, SECTION_RVA.0);
        match kind {
            PeKind::Pe32 => put_u32(&mut bytes, optional_offset + 28, 0x0040_0000),
            PeKind::Pe32Plus => put_u64(&mut bytes, optional_offset + 24, 0x1_4000_0000),
        }
        put_u32(&mut bytes, optional_offset + 32, 0x1000);
        put_u32(&mut bytes, optional_offset + 36, 0x200);
        put_u32(&mut bytes, optional_offset + 56, 0x2000);
        put_u32(&mut bytes, optional_offset + 60, 0x200);
        put_u16(&mut bytes, optional_offset + 68, 3);
        put_u32(&mut bytes, optional_offset + directory_offset - 4, 13);
        for (index, directory) in [
            (IMAGE_DIRECTORY_ENTRY_IMPORT, table.import_directory),
            (IMAGE_DIRECTORY_ENTRY_IAT, table.iat_directory),
        ] {
            let entry = optional_offset + directory_offset + index * 8;
            put_u32(&mut bytes, entry, directory.rva.0);
            put_u32(&mut bytes, entry + 4, directory.size);
        }

        bytes[section_offset..section_offset + 8].copy_from_slice(b".idata\0\0");
        put_u32(
            &mut bytes,
            section_offset + 8,
            u32::try_from(table.bytes.len()).expect("small serialized import table"),
        );
        put_u32(&mut bytes, section_offset + 12, SECTION_RVA.0);
        put_u32(
            &mut bytes,
            section_offset + 16,
            u32::try_from(raw_size).expect("small raw size"),
        );
        put_u32(
            &mut bytes,
            section_offset + 20,
            u32::try_from(raw_offset).expect("small raw offset"),
        );
        put_u32(&mut bytes, section_offset + 36, 0xC000_0040);
        bytes[raw_offset..raw_offset + table.bytes.len()].copy_from_slice(&table.bytes);
        bytes
    }

    fn modules() -> Vec<ImportModuleSpec> {
        vec![
            ImportModuleSpec {
                name: "KERNEL32.dll".to_owned(),
                symbols: vec![
                    ImportTarget::Name {
                        name: "ExitProcess".to_owned(),
                        hint: 7,
                    },
                    ImportTarget::Ordinal(123),
                ],
            },
            ImportModuleSpec {
                name: "USER32.dll".to_owned(),
                symbols: vec![ImportTarget::Name {
                    name: "MessageBoxW".to_owned(),
                    hint: 0x44,
                }],
            },
        ]
    }

    fn source_modules() -> Vec<ImportModule> {
        vec![
            ImportModule {
                name: "KERNEL32.dll".to_owned(),
                descriptor_rva: Rva(0x2000),
                original_first_thunk: Rva(0x2100),
                first_thunk: Rva(0x2200),
                entries: vec![
                    ImportSymbol {
                        iat_rva: Rva(0x2200),
                        lookup_value: 0x2300,
                        name: Some("ExitProcess".to_owned()),
                        hint: Some(7),
                        ordinal: None,
                    },
                    ImportSymbol {
                        iat_rva: Rva(0x2208),
                        lookup_value: 0x8000_0000_0000_007B,
                        name: None,
                        hint: None,
                        ordinal: Some(123),
                    },
                ],
            },
            ImportModule {
                name: "USER32.dll".to_owned(),
                descriptor_rva: Rva(0x2400),
                original_first_thunk: Rva(0x2500),
                first_thunk: Rva(0x2600),
                entries: vec![ImportSymbol {
                    iat_rva: Rva(0x2600),
                    lookup_value: 0x2700,
                    name: Some("MessageBoxW".to_owned()),
                    hint: None,
                    ordinal: None,
                }],
            },
        ]
    }

    #[test]
    fn serialized_tables_round_trip_through_pe_parser() {
        for kind in [PeKind::Pe32, PeKind::Pe32Plus] {
            let table = serialize_import_table(kind, SECTION_RVA, &modules())
                .expect("serialize synthetic imports");
            let image_bytes = synthetic_pe(kind, &table);
            let report = PeImage::parse(&image_bytes)
                .expect("parse synthetic image")
                .report();

            assert_eq!(report.imports, table.modules);
            assert!(
                report
                    .diagnostics
                    .iter()
                    .all(|diagnostic| diagnostic.stage != "imports"),
                "unexpected import diagnostic: {:?}",
                report.diagnostics
            );
            assert_eq!(table.import_directory.size, 3 * 20);
            assert_eq!(
                table.iat_directory.size,
                u32::try_from((2 + 1 + 1 + 1) * kind.pointer_size()).expect("small test IAT")
            );
            assert_eq!(
                table.modules[0].entries[0].name.as_deref(),
                Some("ExitProcess")
            );
            assert_eq!(table.modules[0].entries[1].ordinal, Some(123));
            assert_eq!(table.modules[1].entries[0].hint, Some(0x44));
        }
    }

    #[test]
    fn reconstruction_plan_preserves_exact_source_to_rebuilt_slot_provenance() {
        for kind in [PeKind::Pe32, PeKind::Pe32Plus] {
            let source = source_modules();
            let plan = plan_import_reconstruction(kind, SECTION_RVA, &source)
                .expect("plan import reconstruction");
            let image_bytes = synthetic_pe(kind, &plan.table);
            let report = PeImage::parse(&image_bytes)
                .expect("parse planned import table")
                .report();

            assert_eq!(report.imports, plan.table.modules);
            assert_eq!(plan.slot_remaps.len(), 3);
            assert_eq!(plan.defaulted_hints, 1);
            assert_eq!(plan.slot_remaps[0].source_iat_rva, Rva(0x2200));
            assert_eq!(
                plan.slot_remaps[0].rebuilt_iat_rva,
                plan.table.modules[0].entries[0].iat_rva
            );
            assert_eq!(plan.slot_remaps[1].target, ImportTarget::Ordinal(123));
            assert_eq!(plan.slot_remaps[2].module, "USER32.dll");
            assert_eq!(plan.table.modules[1].entries[0].hint, Some(0));
        }
    }

    #[test]
    fn reconstruction_plan_rejects_unresolved_and_duplicate_source_slots() {
        let mut unresolved = source_modules();
        unresolved[0].entries[0].name = None;
        unresolved[0].entries[0].hint = None;
        assert_eq!(
            plan_import_reconstruction(PeKind::Pe32Plus, SECTION_RVA, &unresolved),
            Err(ImportTableError::UnresolvedSourceSymbol {
                module: 0,
                symbol: 0,
                has_name: false,
                has_ordinal: false,
            })
        );

        let mut ambiguous = source_modules();
        ambiguous[0].entries[0].ordinal = Some(1);
        assert_eq!(
            plan_import_reconstruction(PeKind::Pe32Plus, SECTION_RVA, &ambiguous),
            Err(ImportTableError::UnresolvedSourceSymbol {
                module: 0,
                symbol: 0,
                has_name: true,
                has_ordinal: true,
            })
        );

        let mut duplicate = source_modules();
        duplicate[1].entries[0].iat_rva = duplicate[0].entries[0].iat_rva;
        assert_eq!(
            plan_import_reconstruction(PeKind::Pe32Plus, SECTION_RVA, &duplicate),
            Err(ImportTableError::DuplicateSourceIat { rva: Rva(0x2200) })
        );
    }

    #[test]
    fn int_and_iat_are_separate_and_initially_identical() {
        let table = serialize_import_table(PeKind::Pe32Plus, SECTION_RVA, &modules())
            .expect("serialize synthetic imports");
        for module in &table.modules {
            let int_offset = usize::try_from(module.original_first_thunk.0 - SECTION_RVA.0)
                .expect("small INT offset");
            let iat_offset =
                usize::try_from(module.first_thunk.0 - SECTION_RVA.0).expect("small IAT offset");
            let length = (module.entries.len() + 1) * PeKind::Pe32Plus.pointer_size();
            assert_ne!(int_offset, iat_offset);
            assert_eq!(
                &table.bytes[int_offset..int_offset + length],
                &table.bytes[iat_offset..iat_offset + length]
            );
        }
    }

    #[test]
    fn installs_directories_and_reparses_semantic_imports() {
        for kind in [PeKind::Pe32, PeKind::Pe32Plus] {
            let table = serialize_import_table(kind, SECTION_RVA, &modules())
                .expect("serialize synthetic imports");
            let mut image_bytes = synthetic_pe(kind, &table);
            let directory_offset = match kind {
                PeKind::Pe32 => 96,
                PeKind::Pe32Plus => 112,
            };
            for index in [IMAGE_DIRECTORY_ENTRY_IMPORT, IMAGE_DIRECTORY_ENTRY_IAT] {
                let entry = NT_OFFSET + 4 + 20 + directory_offset + index * 8;
                image_bytes[entry..entry + 8].fill(0);
            }
            assert!(
                PeImage::parse(&image_bytes)
                    .expect("parse image without installed imports")
                    .report()
                    .imports
                    .is_empty()
            );

            let installed = install_import_directories(&image_bytes, &table)
                .expect("install import directories");
            let report = PeImage::parse(&installed)
                .expect("reparse installed imports")
                .report();
            assert_eq!(report.imports, table.modules);
        }
    }

    #[test]
    fn rejects_malformed_or_unrepresentable_inputs() {
        assert_eq!(
            serialize_import_table(PeKind::Pe32Plus, SECTION_RVA, &[]),
            Err(ImportTableError::EmptyTable)
        );
        assert_eq!(
            serialize_import_table(PeKind::Pe32Plus, Rva(0), &modules()),
            Err(ImportTableError::ZeroBaseRva)
        );

        for (name, expected) in [
            (
                "",
                ImportTableError::EmptyString {
                    what: "import module 0 name".to_owned(),
                },
            ),
            (
                "bad\0.dll",
                ImportTableError::EmbeddedNul {
                    what: "import module 0 name".to_owned(),
                },
            ),
            (
                "módulo.dll",
                ImportTableError::NonAscii {
                    what: "import module 0 name".to_owned(),
                },
            ),
        ] {
            let malformed = [ImportModuleSpec {
                name: name.to_owned(),
                symbols: vec![ImportTarget::Ordinal(1)],
            }];
            assert_eq!(
                serialize_import_table(PeKind::Pe32Plus, SECTION_RVA, &malformed),
                Err(expected)
            );
        }

        let empty_module = [ImportModuleSpec {
            name: "KERNEL32.dll".to_owned(),
            symbols: Vec::new(),
        }];
        assert_eq!(
            serialize_import_table(PeKind::Pe32Plus, SECTION_RVA, &empty_module),
            Err(ImportTableError::EmptyModule { module: 0 })
        );

        let overflow = serialize_import_table(PeKind::Pe32Plus, Rva(u32::MAX - 4), &modules())
            .expect_err("payload RVAs must remain representable");
        assert!(matches!(overflow, ImportTableError::Overflow { .. }));
    }
}
