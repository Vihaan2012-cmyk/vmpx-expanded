//! Rebuild a data directory outside the sections a strip is about to remove.
//!
//! VMProtect moves the live export, import, exception, TLS and IAT directories
//! into its own sections, so removing them is refused:
//! "export directory lies inside removed section .vmp3". Dropping the directory
//! instead is not an option for a DLL whose exports ARE its interface -- here
//! `win.xpl1.dll` with `XPluginStart`, `XPluginStop`, `XPluginEnable`,
//! `XPluginDisable` and `XPluginReceiveMessage`. The leftover `.edata` is the
//! right size but holds garbage, so there is nothing to point back at either.
//!
//! So the directory is rebuilt into an appended standard section and the header
//! repointed at the copy. Only structures the LOADER reaches through the
//! directory move; the function RVAs an export table holds keep pointing at the
//! code they always did.

use vmpx_pe::{PeImage, Rva};

use crate::{AppendSection, AppendedSection, RebuildError, append_section};

/// `IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_MEM_READ`.
const READ_ONLY_DATA: u32 = 0x4000_0040;
const EXPORT_DIRECTORY: usize = 0;
const EXPORT_DIRECTORY_BYTES: usize = 40;

/// One directory that was rebuilt elsewhere.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RelocatedDirectory {
    pub index: usize,
    pub from: Rva,
    pub to: Rva,
    pub size: u32,
}

/// The rewritten image plus the section the directories were rebuilt into.
#[derive(Clone, Debug)]
pub struct DirectoryRelocation {
    pub bytes: Vec<u8>,
    pub section: AppendedSection,
    pub moved: Vec<RelocatedDirectory>,
}

#[derive(Debug, thiserror::Error)]
pub enum DirectoryRelocateError {
    #[error("the image has no {0} directory")]
    Absent(&'static str),
    #[error("{0}")]
    Malformed(String),
    #[error("export {name:?} is a forwarder to {target:?}; forwarders are not rebuilt")]
    Forwarder { name: String, target: String },
    #[error("rebuild failed: {0}")]
    Rebuild(#[from] RebuildError),
    #[error("the appended section moved between passes: {first:?} then {second:?}")]
    Unstable { first: Rva, second: Rva },
    #[error("{0:?} images are not rebuilt here; their thunks and pointers are 4 bytes wide")]
    Unsupported(vmpx_pe::PeKind),
}

/// A view over the input for reading directory structures by RVA.
struct At<'a> {
    pe: PeImage<'a>,
    data: &'a [u8],
}

impl<'a> At<'a> {
    fn new(data: &'a [u8]) -> Result<Self, DirectoryRelocateError> {
        let pe = PeImage::parse(data)
            .map_err(|error| DirectoryRelocateError::Malformed(error.to_string()))?;
        // PE32 is a different shape, not a smaller one: its import thunks and
        // TLS pointers are 4 bytes, so reading them as 8 produces garbage RVAs
        // and a refusal for the wrong reason. Round-tripping 32-bit images
        // through this module found exactly that, so it is refused up front.
        if pe.optional.kind != vmpx_pe::PeKind::Pe32Plus {
            return Err(DirectoryRelocateError::Unsupported(pe.optional.kind));
        }
        Ok(Self { pe, data })
    }

    fn slice(&self, rva: Rva, len: usize) -> Result<&'a [u8], DirectoryRelocateError> {
        let offset = self
            .pe
            .rva_to_file_offset(rva)
            .map_err(|error| DirectoryRelocateError::Malformed(error.to_string()))?
            .0 as usize;
        self.data
            .get(offset..offset.checked_add(len).unwrap_or(usize::MAX))
            .ok_or_else(|| {
                DirectoryRelocateError::Malformed(format!(
                    "{len} byte(s) at {rva} are not file-backed"
                ))
            })
    }

    fn u32_at(&self, rva: Rva, index: u32) -> Result<u32, DirectoryRelocateError> {
        let at = Rva(rva.0.saturating_add(index.saturating_mul(4)));
        let bytes = self.slice(at, 4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn u16_at(&self, rva: Rva, index: u32) -> Result<u16, DirectoryRelocateError> {
        let at = Rva(rva.0.saturating_add(index.saturating_mul(2)));
        let bytes = self.slice(at, 2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    /// A NUL-terminated string, capped so a missing terminator cannot run away.
    fn string(&self, rva: Rva) -> Result<Vec<u8>, DirectoryRelocateError> {
        const LIMIT: usize = 1024;
        let offset = self
            .pe
            .rva_to_file_offset(rva)
            .map_err(|error| DirectoryRelocateError::Malformed(error.to_string()))?
            .0 as usize;
        let tail = self.data.get(offset..).unwrap_or_default();
        let end = tail
            .iter()
            .take(LIMIT)
            .position(|byte| *byte == 0)
            .ok_or_else(|| {
                DirectoryRelocateError::Malformed(format!("string at {rva} is unterminated"))
            })?;
        Ok(tail[..end].to_vec())
    }
}

/// The export table read out of the image, with every RVA it owns resolved.
struct Exports {
    characteristics: u32,
    time_date_stamp: u32,
    major: u16,
    minor: u16,
    ordinal_base: u32,
    module_name: Vec<u8>,
    functions: Vec<u32>,
    names: Vec<Vec<u8>>,
    ordinals: Vec<u16>,
}

fn read_exports(at: &At<'_>) -> Result<(Rva, u32, Exports), DirectoryRelocateError> {
    let directory = at
        .pe
        .directory(EXPORT_DIRECTORY)
        .filter(|directory| directory.address != 0 && directory.size != 0)
        .ok_or(DirectoryRelocateError::Absent("export"))?;
    let base = Rva(directory.address);
    let header = at.slice(base, EXPORT_DIRECTORY_BYTES)?;
    let word = |offset: usize| {
        u32::from_le_bytes([
            header[offset],
            header[offset + 1],
            header[offset + 2],
            header[offset + 3],
        ])
    };
    let half = |offset: usize| u16::from_le_bytes([header[offset], header[offset + 1]]);
    let function_count = word(0x14);
    let name_count = word(0x18);
    let functions_rva = Rva(word(0x1c));
    let names_rva = Rva(word(0x20));
    let ordinals_rva = Rva(word(0x24));
    // A table larger than the image cannot be real, and the loop below would
    // otherwise read until `slice` happened to fail.
    let ceiling = u32::try_from(at.data.len()).unwrap_or(u32::MAX) / 4;
    if function_count > ceiling || name_count > ceiling {
        return Err(DirectoryRelocateError::Malformed(format!(
            "export table claims {function_count} function(s) and {name_count} name(s)"
        )));
    }
    let mut functions = Vec::new();
    for index in 0..function_count {
        let target = at.u32_at(functions_rva, index)?;
        // A function RVA INSIDE the directory is a forwarder string, which only
        // means something while the directory lives where it is. Refused rather
        // than silently relocated to a dangling address.
        let inside = target >= base.0 && target < base.0.saturating_add(directory.size);
        if inside && target != 0 {
            return Err(DirectoryRelocateError::Forwarder {
                name: format!("ordinal {}", word(0x10).saturating_add(index)),
                target: format!("{:#x}", target),
            });
        }
        functions.push(target);
    }
    let mut names = Vec::new();
    let mut ordinals = Vec::new();
    for index in 0..name_count {
        names.push(at.string(Rva(at.u32_at(names_rva, index)?))?);
        ordinals.push(at.u16_at(ordinals_rva, index)?);
    }
    Ok((
        base,
        directory.size,
        Exports {
            characteristics: word(0x00),
            time_date_stamp: word(0x04),
            major: half(0x08),
            minor: half(0x0a),
            ordinal_base: word(0x10),
            module_name: at.string(Rva(word(0x0c)))?,
            functions,
            names,
            ordinals,
        },
    ))
}

/// Serializes `exports` as an export directory based at `base`.
///
/// The layout is the conventional one: the directory header, then the three
/// parallel arrays, then the strings. Every RVA written points inside this blob
/// except the function addresses, which still name the code.
fn serialize_exports(exports: &Exports, base: Rva) -> Vec<u8> {
    let function_count = u32::try_from(exports.functions.len()).unwrap_or(u32::MAX);
    let name_count = u32::try_from(exports.names.len()).unwrap_or(u32::MAX);
    let header = u32::try_from(EXPORT_DIRECTORY_BYTES).unwrap_or(u32::MAX);
    let functions_rva = base.0.saturating_add(header);
    let names_rva = functions_rva.saturating_add(function_count.saturating_mul(4));
    let ordinals_rva = names_rva.saturating_add(name_count.saturating_mul(4));
    let strings_rva = ordinals_rva.saturating_add(name_count.saturating_mul(2));

    // The module name goes in first, so the string area starts with it.
    let module_name_rva = strings_rva;
    let mut strings = Vec::new();
    strings.extend_from_slice(&exports.module_name);
    strings.push(0);
    let mut name_rvas = Vec::new();
    for name in &exports.names {
        name_rvas.push(strings_rva.saturating_add(u32::try_from(strings.len()).unwrap_or(u32::MAX)));
        strings.extend_from_slice(name);
        strings.push(0);
    }

    let mut out = Vec::new();
    out.extend_from_slice(&exports.characteristics.to_le_bytes());
    out.extend_from_slice(&exports.time_date_stamp.to_le_bytes());
    out.extend_from_slice(&exports.major.to_le_bytes());
    out.extend_from_slice(&exports.minor.to_le_bytes());
    out.extend_from_slice(&module_name_rva.to_le_bytes());
    out.extend_from_slice(&exports.ordinal_base.to_le_bytes());
    out.extend_from_slice(&function_count.to_le_bytes());
    out.extend_from_slice(&name_count.to_le_bytes());
    out.extend_from_slice(&functions_rva.to_le_bytes());
    out.extend_from_slice(&names_rva.to_le_bytes());
    out.extend_from_slice(&ordinals_rva.to_le_bytes());
    debug_assert_eq!(out.len(), EXPORT_DIRECTORY_BYTES);
    for function in &exports.functions {
        out.extend_from_slice(&function.to_le_bytes());
    }
    for rva in &name_rvas {
        out.extend_from_slice(&rva.to_le_bytes());
    }
    for ordinal in &exports.ordinals {
        out.extend_from_slice(&ordinal.to_le_bytes());
    }
    out.extend_from_slice(&strings);
    out
}

/// Rebuilds the export directory into an appended read-only section and points
/// the header at the copy.
///
/// Two passes, because the directory's own arrays are addressed by RVA and the
/// RVA is only known once the section is placed: the first append learns where
/// the section lands, the second writes the real addresses there. The placement
/// is a function of the input and the payload LENGTH, which does not change
/// between passes, and the result asserts both passes agreed.
///
/// # Errors
///
/// An absent or malformed export directory, a forwarder (which only has meaning
/// at its original address), or any failure of the underlying append.
pub fn rebuild_export_directory(
    input: &[u8],
    name: [u8; 8],
) -> Result<DirectoryRelocation, DirectoryRelocateError> {
    let at = At::new(input)?;
    let (from, from_size, exports) = read_exports(&at)?;

    let probe = serialize_exports(&exports, Rva(0));
    let first = append_section(
        input,
        AppendSection {
            name,
            data: &probe,
            characteristics: READ_ONLY_DATA,
        },
    )?;
    let placed = first.section.rva;
    let payload = serialize_exports(&exports, placed);
    // Same length by construction -- only the RVAs written into it changed --
    // so the second append must place it identically.
    if payload.len() != probe.len() {
        return Err(DirectoryRelocateError::Malformed(
            "export payload changed length between passes".to_owned(),
        ));
    }
    let second = append_section(
        input,
        AppendSection {
            name,
            data: &payload,
            characteristics: READ_ONLY_DATA,
        },
    )?;
    if second.section.rva != placed {
        return Err(DirectoryRelocateError::Unstable {
            first: placed,
            second: second.section.rva,
        });
    }

    let size = u32::try_from(payload.len()).unwrap_or(u32::MAX);
    let mut bytes = second.bytes;
    write_directory(&mut bytes, EXPORT_DIRECTORY, placed, size)?;
    // The copy is only worth anything if a parser reaches the same exports
    // through it, so read them back out of the rewritten image and compare.
    let rebuilt = At::new(&bytes)?;
    let (_, _, round_trip) = read_exports(&rebuilt)?;
    if round_trip.functions != exports.functions
        || round_trip.names != exports.names
        || round_trip.ordinals != exports.ordinals
        || round_trip.module_name != exports.module_name
        || round_trip.ordinal_base != exports.ordinal_base
    {
        return Err(DirectoryRelocateError::Malformed(
            "the rebuilt export directory does not read back identically".to_owned(),
        ));
    }
    Ok(DirectoryRelocation {
        bytes,
        section: second.section,
        moved: vec![RelocatedDirectory {
            index: EXPORT_DIRECTORY,
            from,
            to: placed,
            size: from_size,
        }],
    })
}

/// Overwrites one data-directory entry in place.
fn write_directory(
    bytes: &mut [u8],
    index: usize,
    rva: Rva,
    size: u32,
) -> Result<(), DirectoryRelocateError> {
    let pe = PeImage::parse(bytes)
        .map_err(|error| DirectoryRelocateError::Malformed(error.to_string()))?;
    if pe.optional.kind != vmpx_pe::PeKind::Pe32Plus {
        return Err(DirectoryRelocateError::Malformed(
            "only PE32+ directory slots are rewritten".to_owned(),
        ));
    }
    if pe.directory(index).is_none() && index >= 16 {
        return Err(DirectoryRelocateError::Malformed(format!(
            "no directory slot {index}"
        )));
    }
    // PE32+: the optional header's data-directory array starts 112 bytes in.
    let at = pe.nt_headers_offset.0 as usize + 24 + 112 + index * 8;
    let slot = bytes
        .get_mut(at..at + 8)
        .ok_or_else(|| DirectoryRelocateError::Malformed("directory slot is out of range".into()))?;
    slot[..4].copy_from_slice(&rva.0.to_le_bytes());
    slot[4..].copy_from_slice(&size.to_le_bytes());
    Ok(())
}


const IMPORT_DIRECTORY: usize = 1;
const IAT_DIRECTORY: usize = 12;
const DESCRIPTOR_BYTES: usize = 20;
/// `IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_MEM_READ | IMAGE_SCN_MEM_WRITE`.
const READ_WRITE_DATA: u32 = 0xc000_0040;

/// What a rebuilt import directory contains.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ImportRebuild {
    pub modules: usize,
    pub symbols: usize,
    pub iat_rva: Rva,
    pub iat_size: u32,
}

struct ImportModule {
    name: Vec<u8>,
    time_date_stamp: u32,
    forwarder_chain: u32,
    symbols: Vec<ImportSymbol>,
}

enum ImportSymbol {
    Named { hint: u16, name: Vec<u8> },
    Ordinal(u16),
}

fn read_imports(at: &At<'_>) -> Result<Vec<ImportModule>, DirectoryRelocateError> {
    let directory = at
        .pe
        .directory(IMPORT_DIRECTORY)
        .filter(|directory| directory.address != 0 && directory.size != 0)
        .ok_or(DirectoryRelocateError::Absent("import"))?;
    let base = Rva(directory.address);
    let slots = directory.size as usize / DESCRIPTOR_BYTES;
    let mut modules = Vec::new();
    for index in 0..slots {
        let at_rva = Rva(base.0 + u32::try_from(index * DESCRIPTOR_BYTES).unwrap_or(u32::MAX));
        let raw = at.slice(at_rva, DESCRIPTOR_BYTES)?;
        let word = |offset: usize| {
            u32::from_le_bytes([raw[offset], raw[offset + 1], raw[offset + 2], raw[offset + 3]])
        };
        let (lookup, time_date_stamp, forwarder_chain, name, first_thunk) =
            (word(0), word(4), word(8), word(12), word(16));
        if lookup == 0 && name == 0 && first_thunk == 0 {
            break;
        }
        // Read the thunks from the lookup table when there is one. A bound
        // import with no ILT leaves only the IAT, which still holds the
        // hint/name RVAs until a loader overwrites them.
        let thunks = if lookup != 0 { lookup } else { first_thunk };
        if thunks == 0 {
            return Err(DirectoryRelocateError::Malformed(format!(
                "import descriptor {index} has neither a lookup table nor an IAT"
            )));
        }
        let mut symbols = Vec::new();
        for slot in 0_u32.. {
            let entry = Rva(thunks.saturating_add(slot.saturating_mul(8)));
            let raw = at.slice(entry, 8)?;
            let value = u64::from_le_bytes(raw.try_into().expect("8 bytes"));
            if value == 0 {
                break;
            }
            if value & (1 << 63) != 0 {
                symbols.push(ImportSymbol::Ordinal(value as u16));
            } else {
                let hint_name = Rva(u32::try_from(value & 0x7fff_ffff).unwrap_or(u32::MAX));
                let hint_bytes = at.slice(hint_name, 2)?;
                symbols.push(ImportSymbol::Named {
                    hint: u16::from_le_bytes([hint_bytes[0], hint_bytes[1]]),
                    name: at.string(Rva(hint_name.0 + 2))?,
                });
            }
            if symbols.len() > 65_536 {
                return Err(DirectoryRelocateError::Malformed(
                    "import thunk array is unterminated".to_owned(),
                ));
            }
        }
        modules.push(ImportModule {
            name: at.string(Rva(name))?,
            time_date_stamp,
            forwarder_chain,
            symbols,
        });
    }
    if modules.is_empty() {
        return Err(DirectoryRelocateError::Absent("import"));
    }
    Ok(modules)
}

/// Serializes `modules` as descriptors, lookup tables, IAT, hint/name entries
/// and module names, all based at `base`.
///
/// The IAT arrays are laid out contiguously so directory 12 covers them with a
/// single range, which is what tells the loader which pages to make writable.
fn serialize_imports(modules: &[ImportModule], base: Rva) -> (Vec<u8>, Rva, u32) {
    let descriptors = u32::try_from((modules.len() + 1) * DESCRIPTOR_BYTES).unwrap_or(u32::MAX);
    let thunk_words: u32 = modules
        .iter()
        .map(|module| u32::try_from(module.symbols.len() + 1).unwrap_or(u32::MAX))
        .sum();
    let lookup_base = base.0.saturating_add(descriptors);
    let iat_base = lookup_base.saturating_add(thunk_words.saturating_mul(8));
    let names_base = iat_base.saturating_add(thunk_words.saturating_mul(8));

    let mut names: Vec<u8> = Vec::new();
    let mut symbol_rvas: Vec<Vec<u64>> = Vec::new();
    for module in modules {
        let mut rvas = Vec::new();
        for symbol in &module.symbols {
            match symbol {
                ImportSymbol::Ordinal(ordinal) => {
                    rvas.push((1_u64 << 63) | u64::from(*ordinal));
                }
                ImportSymbol::Named { hint, name } => {
                    // Hint/name entries are word-aligned.
                    if names.len() % 2 != 0 {
                        names.push(0);
                    }
                    let rva =
                        names_base.saturating_add(u32::try_from(names.len()).unwrap_or(u32::MAX));
                    names.extend_from_slice(&hint.to_le_bytes());
                    names.extend_from_slice(name);
                    names.push(0);
                    rvas.push(u64::from(rva));
                }
            }
        }
        symbol_rvas.push(rvas);
    }
    let mut module_name_rvas = Vec::new();
    for module in modules {
        module_name_rvas
            .push(names_base.saturating_add(u32::try_from(names.len()).unwrap_or(u32::MAX)));
        names.extend_from_slice(&module.name);
        names.push(0);
    }

    let mut out: Vec<u8> = Vec::new();
    let mut lookup_cursor = lookup_base;
    let mut iat_cursor = iat_base;
    for (index, module) in modules.iter().enumerate() {
        out.extend_from_slice(&lookup_cursor.to_le_bytes());
        out.extend_from_slice(&module.time_date_stamp.to_le_bytes());
        out.extend_from_slice(&module.forwarder_chain.to_le_bytes());
        out.extend_from_slice(&module_name_rvas[index].to_le_bytes());
        out.extend_from_slice(&iat_cursor.to_le_bytes());
        let span = u32::try_from((module.symbols.len() + 1) * 8).unwrap_or(u32::MAX);
        lookup_cursor = lookup_cursor.saturating_add(span);
        iat_cursor = iat_cursor.saturating_add(span);
    }
    out.extend_from_slice(&[0_u8; DESCRIPTOR_BYTES]);
    debug_assert_eq!(out.len(), descriptors as usize);
    // Lookup table, then the IAT: identical in the file, as they are in any
    // image a loader has not yet bound.
    for _ in 0..2 {
        for rvas in &symbol_rvas {
            for rva in rvas {
                out.extend_from_slice(&rva.to_le_bytes());
            }
            out.extend_from_slice(&0_u64.to_le_bytes());
        }
    }
    out.extend_from_slice(&names);
    (out, Rva(iat_base), thunk_words.saturating_mul(8))
}

/// Rebuilds the import directory and its IAT into an appended writable section,
/// repointing directories 1 and 12.
///
/// The IAT is the one structure here that code reaches by address, so moving it
/// normally means rewriting every `call [rip+disp32]` that uses it. Measured on
/// the sample with the `iatrefs` probe: of 67_605 exception-described
/// functions, exactly FOUR RIP-relative references reach the IAT and all four
/// are inside `.vmp3` -- the VM's own native-call handler -- while `.text` has
/// none. Once that section goes, nothing points at the old table.
///
/// # Errors
///
/// An absent or malformed import directory, an unterminated thunk array, or any
/// failure of the underlying append.
pub fn rebuild_import_directory(
    input: &[u8],
    name: [u8; 8],
) -> Result<(DirectoryRelocation, ImportRebuild), DirectoryRelocateError> {
    let at = At::new(input)?;
    let import = at
        .pe
        .directory(IMPORT_DIRECTORY)
        .ok_or(DirectoryRelocateError::Absent("import"))?;
    let (from, from_size) = (Rva(import.address), import.size);
    let iat_from = Rva(at
        .pe
        .directory(IAT_DIRECTORY)
        .map_or(0, |directory| directory.address));
    let modules = read_imports(&at)?;
    let symbols: usize = modules.iter().map(|module| module.symbols.len()).sum();

    let (probe, _, _) = serialize_imports(&modules, Rva(0));
    let first = append_section(
        input,
        AppendSection {
            name,
            data: &probe,
            characteristics: READ_WRITE_DATA,
        },
    )?;
    let placed = first.section.rva;
    let (payload, iat_rva, iat_size) = serialize_imports(&modules, placed);
    if payload.len() != probe.len() {
        return Err(DirectoryRelocateError::Malformed(
            "import payload changed length between passes".to_owned(),
        ));
    }
    let second = append_section(
        input,
        AppendSection {
            name,
            data: &payload,
            characteristics: READ_WRITE_DATA,
        },
    )?;
    if second.section.rva != placed {
        return Err(DirectoryRelocateError::Unstable {
            first: placed,
            second: second.section.rva,
        });
    }
    let descriptor_size = u32::try_from((modules.len() + 1) * DESCRIPTOR_BYTES).unwrap_or(u32::MAX);
    let mut bytes = second.bytes;
    write_directory(&mut bytes, IMPORT_DIRECTORY, placed, descriptor_size)?;
    write_directory(&mut bytes, IAT_DIRECTORY, iat_rva, iat_size)?;

    // Read the whole table back through a fresh parse: a rebuilt import
    // directory that does not resolve to the same modules and symbols is worse
    // than no rebuild at all.
    let rebuilt = At::new(&bytes)?;
    let round_trip = read_imports(&rebuilt)?;
    let same = round_trip.len() == modules.len()
        && round_trip.iter().zip(&modules).all(|(left, right)| {
            left.name == right.name && left.symbols.len() == right.symbols.len()
        });
    if !same {
        return Err(DirectoryRelocateError::Malformed(
            "the rebuilt import directory does not read back identically".to_owned(),
        ));
    }
    Ok((
        DirectoryRelocation {
            bytes,
            section: second.section,
            moved: vec![
                RelocatedDirectory {
                    index: IMPORT_DIRECTORY,
                    from,
                    to: placed,
                    size: from_size,
                },
                RelocatedDirectory {
                    index: IAT_DIRECTORY,
                    from: iat_from,
                    to: iat_rva,
                    size: iat_size,
                },
            ],
        },
        ImportRebuild {
            modules: modules.len(),
            symbols,
            iat_rva,
            iat_size,
        },
    ))
}


const TLS_DIRECTORY: usize = 9;
const BASE_RELOCATION_DIRECTORY: usize = 5;
const TLS_DIRECTORY_BYTES: usize = 40;
/// `IMAGE_REL_BASED_DIR64`.
const DIR64: u16 = 10;

/// What a rebuilt TLS directory kept and dropped.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TlsRebuild {
    pub kept_callbacks: usize,
    pub dropped_callbacks: Vec<Rva>,
    pub raw_data_bytes: usize,
    /// Addresses in the rebuilt image that hold a VA and therefore need a
    /// `DIR64` base relocation.
    pub needs_relocation: Vec<Rva>,
}

/// Rebuilds the TLS directory, its raw-data template and its callback array
/// outside the sections named in `removed`.
///
/// Everything here is addressed by absolute VA, so each pointer the rebuild
/// writes needs a `DIR64` base relocation and the caller gets their addresses
/// back in `needs_relocation`; `rebuild_base_relocations` consumes that list.
///
/// A callback inside a removed section is DROPPED and named in the result,
/// because it would otherwise dangle. On this sample that is the protector's
/// own TLS entry, while the four real callbacks live in `.text` -- so dropping
/// it is only correct once something has replaced what it did, which is why the
/// list comes back to the caller instead of being swallowed.
///
/// # Errors
///
/// An absent or malformed TLS directory, or any failure of the append.
pub fn rebuild_tls_directory(
    input: &[u8],
    name: [u8; 8],
    removed: &[&str],
) -> Result<(DirectoryRelocation, TlsRebuild), DirectoryRelocateError> {
    let at = At::new(input)?;
    let directory = at
        .pe
        .directory(TLS_DIRECTORY)
        .filter(|directory| directory.address != 0 && directory.size != 0)
        .ok_or(DirectoryRelocateError::Absent("TLS"))?;
    let from = Rva(directory.address);
    let from_size = directory.size;
    let image_base = at.pe.optional.image_base.get();
    let header = at.slice(from, TLS_DIRECTORY_BYTES)?;
    let quad = |offset: usize| {
        u64::from_le_bytes(header[offset..offset + 8].try_into().expect("8 bytes"))
    };
    let word = |offset: usize| {
        u32::from_le_bytes(header[offset..offset + 4].try_into().expect("4 bytes"))
    };
    let to_rva = |va: u64| -> Result<Rva, DirectoryRelocateError> {
        va.checked_sub(image_base)
            .and_then(|rva| u32::try_from(rva).ok())
            .map(Rva)
            .ok_or_else(|| {
                DirectoryRelocateError::Malformed(format!("TLS pointer {va:#x} is below the base"))
            })
    };
    let (raw_start, raw_end) = (to_rva(quad(0))?, to_rva(quad(8))?);
    let index_va = quad(16);
    let callbacks_rva = to_rva(quad(24))?;
    let zero_fill = word(32);
    let characteristics = word(36);
    if raw_end.0 < raw_start.0 {
        return Err(DirectoryRelocateError::Malformed(
            "TLS raw data ends before it starts".to_owned(),
        ));
    }
    let raw = at
        .slice(raw_start, (raw_end.0 - raw_start.0) as usize)?
        .to_vec();

    let in_removed = |rva: Rva| {
        at.pe.sections.iter().any(|section| {
            removed.contains(&section.name.as_str()) && section.contains_mapped_rva(rva)
        })
    };
    let mut kept = Vec::new();
    let mut dropped = Vec::new();
    for slot in 0_u32.. {
        let entry = Rva(callbacks_rva.0.saturating_add(slot.saturating_mul(8)));
        let value = u64::from_le_bytes(at.slice(entry, 8)?.try_into().expect("8 bytes"));
        if value == 0 {
            break;
        }
        let target = to_rva(value)?;
        if in_removed(target) {
            dropped.push(target);
        } else {
            kept.push(value);
        }
        if slot > 1024 {
            return Err(DirectoryRelocateError::Malformed(
                "TLS callback array is unterminated".to_owned(),
            ));
        }
    }

    // Layout: directory, raw-data template, callback array (NULL-terminated).
    let layout = |base: Rva| {
        let raw_rva = base.0.saturating_add(u32::try_from(TLS_DIRECTORY_BYTES).unwrap_or(0));
        let callbacks = raw_rva.saturating_add(u32::try_from(raw.len()).unwrap_or(0));
        let mut out: Vec<u8> = Vec::new();
        out.extend_from_slice(&(image_base + u64::from(raw_rva)).to_le_bytes());
        out.extend_from_slice(
            &(image_base + u64::from(raw_rva) + raw.len() as u64).to_le_bytes(),
        );
        out.extend_from_slice(&index_va.to_le_bytes());
        out.extend_from_slice(&(image_base + u64::from(callbacks)).to_le_bytes());
        out.extend_from_slice(&zero_fill.to_le_bytes());
        out.extend_from_slice(&characteristics.to_le_bytes());
        debug_assert_eq!(out.len(), TLS_DIRECTORY_BYTES);
        out.extend_from_slice(&raw);
        for callback in &kept {
            out.extend_from_slice(&callback.to_le_bytes());
        }
        out.extend_from_slice(&0_u64.to_le_bytes());
        (out, callbacks)
    };

    let (probe, _) = layout(Rva(0));
    let first = append_section(
        input,
        AppendSection {
            name,
            data: &probe,
            characteristics: READ_WRITE_DATA,
        },
    )?;
    let placed = first.section.rva;
    let (payload, callbacks_at) = layout(placed);
    if payload.len() != probe.len() {
        return Err(DirectoryRelocateError::Malformed(
            "TLS payload changed length between passes".to_owned(),
        ));
    }
    let second = append_section(
        input,
        AppendSection {
            name,
            data: &payload,
            characteristics: READ_WRITE_DATA,
        },
    )?;
    if second.section.rva != placed {
        return Err(DirectoryRelocateError::Unstable {
            first: placed,
            second: second.section.rva,
        });
    }
    let mut bytes = second.bytes;
    write_directory(
        &mut bytes,
        TLS_DIRECTORY,
        placed,
        u32::try_from(TLS_DIRECTORY_BYTES).unwrap_or(0),
    )?;

    // Every VA the rebuild wrote: three directory fields (the index pointer is
    // unchanged but still a VA) and one slot per surviving callback.
    let mut needs_relocation = vec![
        placed,
        Rva(placed.0 + 8),
        Rva(placed.0 + 16),
        Rva(placed.0 + 24),
    ];
    for slot in 0..kept.len() {
        needs_relocation.push(Rva(
            callbacks_at.saturating_add(u32::try_from(slot * 8).unwrap_or(0))
        ));
    }

    Ok((
        DirectoryRelocation {
            bytes,
            section: second.section,
            moved: vec![RelocatedDirectory {
                index: TLS_DIRECTORY,
                from,
                to: placed,
                size: from_size,
            }],
        },
        TlsRebuild {
            kept_callbacks: kept.len(),
            dropped_callbacks: dropped,
            raw_data_bytes: raw.len(),
            needs_relocation,
        },
    ))
}

/// What a rebuilt base-relocation table kept, dropped and added.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RelocationRebuild {
    pub kept: usize,
    pub dropped: usize,
    pub added: usize,
}

/// Rebuilds the base-relocation table outside the sections named in `removed`,
/// dropping the fixups that point into them and adding one `DIR64` fixup for
/// each address in `extra`.
///
/// Measured on the sample: the whole table is 29 entries and every one of them
/// is a `DIR64` targeting `.vmp3` -- the protector's own pointers. So after a
/// strip the table is, correctly, almost empty, and what it then holds is
/// exactly the fixups a directory rebuild introduced.
///
/// # Errors
///
/// A malformed relocation table, or any failure of the append.
pub fn rebuild_base_relocations(
    input: &[u8],
    name: [u8; 8],
    removed: &[&str],
    extra: &[Rva],
) -> Result<(DirectoryRelocation, RelocationRebuild), DirectoryRelocateError> {
    let at = At::new(input)?;
    let (from, from_size) = at
        .pe
        .directory(BASE_RELOCATION_DIRECTORY)
        .map_or((Rva(0), 0), |directory| {
            (Rva(directory.address), directory.size)
        });
    let blocks = at
        .pe
        .parse_relocations()
        .map_err(|error| DirectoryRelocateError::Malformed(error.to_string()))?;
    let in_removed = |rva: Rva| {
        at.pe.sections.iter().any(|section| {
            removed.contains(&section.name.as_str()) && section.contains_mapped_rva(rva)
        })
    };

    // Collect surviving fixups plus the new ones, bucketed by 4 KiB page, which
    // is how a relocation table is addressed.
    let mut pages: std::collections::BTreeMap<u32, Vec<u16>> = std::collections::BTreeMap::new();
    let (mut kept, mut dropped) = (0_usize, 0_usize);
    for block in &blocks {
        for entry in &block.entries {
            if entry.kind == 0 {
                continue;
            }
            let target = Rva(block.page_rva.0.saturating_add(u32::from(entry.offset)));
            if in_removed(target) {
                dropped += 1;
                continue;
            }
            kept += 1;
            pages
                .entry(target.0 & !0xfff)
                .or_default()
                .push((u16::from(entry.kind) << 12) | (entry.offset & 0xfff));
        }
    }
    for target in extra {
        pages
            .entry(target.0 & !0xfff)
            .or_default()
            .push((DIR64 << 12) | u16::try_from(target.0 & 0xfff).unwrap_or(0));
    }

    let mut payload: Vec<u8> = Vec::new();
    for (page, mut entries) in pages {
        entries.sort_unstable();
        // Each block is 8 bytes of header plus 2 per entry, padded to 4.
        if entries.len() % 2 != 0 {
            entries.push(0);
        }
        let size = u32::try_from(8 + entries.len() * 2).unwrap_or(u32::MAX);
        payload.extend_from_slice(&page.to_le_bytes());
        payload.extend_from_slice(&size.to_le_bytes());
        for entry in entries {
            payload.extend_from_slice(&entry.to_le_bytes());
        }
    }
    if payload.is_empty() {
        // An empty directory is legal and means "no fixups"; a zero-length
        // section is not, so the directory is simply cleared.
        let mut bytes = input.to_vec();
        write_directory(&mut bytes, BASE_RELOCATION_DIRECTORY, Rva(0), 0)?;
        return Ok((
            DirectoryRelocation {
                bytes,
                section: AppendedSection {
                    name,
                    rva: Rva(0),
                    raw_offset: vmpx_pe::FileOffset(0),
                    virtual_size: 0,
                    raw_size: 0,
                    characteristics: READ_ONLY_DATA,
                },
                moved: vec![RelocatedDirectory {
                    index: BASE_RELOCATION_DIRECTORY,
                    from,
                    to: Rva(0),
                    size: from_size,
                }],
            },
            RelocationRebuild {
                kept,
                dropped,
                added: extra.len(),
            },
        ));
    }

    let appended = append_section(
        input,
        AppendSection {
            name,
            data: &payload,
            characteristics: READ_ONLY_DATA,
        },
    )?;
    let placed = appended.section.rva;
    let size = u32::try_from(payload.len()).unwrap_or(u32::MAX);
    let mut bytes = appended.bytes;
    write_directory(&mut bytes, BASE_RELOCATION_DIRECTORY, placed, size)?;
    let rebuilt = At::new(&bytes)?;
    let count: usize = rebuilt
        .pe
        .parse_relocations()
        .map_err(|error| DirectoryRelocateError::Malformed(error.to_string()))?
        .iter()
        .map(|block| block.entries.iter().filter(|entry| entry.kind != 0).count())
        .sum();
    if count != kept + extra.len() {
        return Err(DirectoryRelocateError::Malformed(format!(
            "rebuilt relocation table reads back {count} fixup(s), expected {}",
            kept + extra.len()
        )));
    }
    Ok((
        DirectoryRelocation {
            bytes,
            section: appended.section,
            moved: vec![RelocatedDirectory {
                index: BASE_RELOCATION_DIRECTORY,
                from,
                to: placed,
                size: from_size,
            }],
        },
        RelocationRebuild {
            kept,
            dropped,
            added: extra.len(),
        },
    ))
}


/// What a scan for references into the IAT found, and how much it could see.
///
/// Moving the IAT is only sound when nothing that SURVIVES the strip reaches
/// the old table by address. On the sample that holds -- four RIP-relative
/// references, all inside `.vmp3` -- but that is a property of this build, not
/// of VMProtect: with import protection off, the original `call [rip+iat]`
/// instructions stay in `.text` and moving the table would break every one of
/// them. So it is checked, never assumed.
///
/// The check decodes the functions the exception directory describes, which on
/// the sample is 95.3% of `.text`. `unverified_bytes` is the surviving
/// executable remainder it could NOT decode, and a caller that treats zero
/// findings as a guarantee while that is non-zero is trusting a sample.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IatReferenceScan {
    pub references: Vec<Rva>,
    pub decoded_functions: usize,
    pub unverified_bytes: u64,
}

/// Finds RIP-relative references into `[iat, iat+size)` from code that a strip
/// of `removed` would keep.
///
/// # Errors
///
/// A malformed image.
pub fn scan_iat_references(
    input: &[u8],
    removed: &[&str],
) -> Result<IatReferenceScan, DirectoryRelocateError> {
    let at = At::new(input)?;
    let Some(iat) = at
        .pe
        .directory(IAT_DIRECTORY)
        .filter(|directory| directory.address != 0 && directory.size != 0)
    else {
        return Ok(IatReferenceScan {
            references: Vec::new(),
            decoded_functions: 0,
            unverified_bytes: 0,
        });
    };
    let (lo, hi) = (iat.address, iat.address.saturating_add(iat.size));
    let base = at.pe.optional.image_base.get();
    let surviving = |rva: Rva| {
        at.pe.sections.iter().any(|section| {
            !removed.contains(&section.name.as_str())
                && section.characteristics & 0x2000_0000 != 0
                && section.contains_mapped_rva(rva)
        })
    };

    let functions = at.pe.report().exception_functions;
    let mut references = Vec::new();
    let mut decoded_functions = 0_usize;
    let mut covered: u64 = 0;
    for function in &functions {
        let (begin, end) = (function.begin.0, function.end.0);
        if end <= begin || !surviving(Rva(begin)) {
            continue;
        }
        let Ok(offset) = at.pe.rva_to_file_offset(Rva(begin)) else {
            continue;
        };
        let start = offset.0 as usize;
        let stop = start.saturating_add((end - begin) as usize).min(input.len());
        let Some(code) = input.get(start..stop) else {
            continue;
        };
        decoded_functions += 1;
        covered += u64::from(end - begin);
        let mut decoder =
            iced_x86::Decoder::with_ip(64, code, base + u64::from(begin), iced_x86::DecoderOptions::NONE);
        while decoder.can_decode() {
            let instruction = decoder.decode();
            if instruction.memory_base() != iced_x86::Register::RIP {
                continue;
            }
            let Some(rva) = instruction
                .ip_rel_memory_address()
                .checked_sub(base)
                .and_then(|value| u32::try_from(value).ok())
            else {
                continue;
            };
            if rva >= lo && rva < hi {
                references.push(Rva(
                    u32::try_from(instruction.ip().saturating_sub(base)).unwrap_or(0)
                ));
            }
        }
    }
    let executable: u64 = at
        .pe
        .sections
        .iter()
        .filter(|section| {
            !removed.contains(&section.name.as_str()) && section.characteristics & 0x2000_0000 != 0
        })
        .map(|section| u64::from(section.virtual_size.max(1)))
        .sum();
    Ok(IatReferenceScan {
        references,
        decoded_functions,
        unverified_bytes: executable.saturating_sub(covered),
    })
}


/// Everything a combined rebuild produced.
#[derive(Clone, Debug)]
pub struct AllDirectories {
    pub bytes: Vec<u8>,
    pub section: AppendedSection,
    pub moved: Vec<RelocatedDirectory>,
    pub exception_entries: (usize, usize),
    pub import_modules: usize,
    pub import_symbols: usize,
    pub tls_callbacks: (usize, Vec<Rva>),
    pub relocations: RelocationRebuild,
}

/// The TLS directory read out of the image, with its pointers resolved.
struct TlsState {
    raw: Vec<u8>,
    index_va: u64,
    kept: Vec<u64>,
    dropped: Vec<Rva>,
    zero_fill: u32,
    characteristics: u32,
}

fn align8(payload: &mut Vec<u8>) {
    while payload.len() % 8 != 0 {
        payload.push(0);
    }
}

/// Rebuilds EVERY directory that lives in a removed section into ONE appended
/// section.
///
/// One section, not five, because the header has room for exactly one more.
/// Measured on the sample: the section table ends at 0x5c0 and `SizeOfHeaders`
/// is 0x600, so 64 free bytes -- a single 40-byte slot. Appending the export
/// directory succeeds and the next append fails with "new section header ending
/// at 0x610 exceeds header capacity 0x600". Growing the headers would mean
/// moving every section's raw data; packing the directories together does not.
///
/// Read-write, because the IAT and the TLS index must be writable and the
/// read-only tables beside them do not mind.
///
/// # Errors
///
/// A malformed directory, or any failure of the append.
#[allow(clippy::too_many_lines)]
pub fn rebuild_all_directories(
    input: &[u8],
    name: [u8; 8],
    removed: &[&str],
) -> Result<AllDirectories, DirectoryRelocateError> {
    let at = At::new(input)?;
    let image_base = at.pe.optional.image_base.get();
    let present = |index: usize| {
        at.pe
            .directory(index)
            .filter(|directory| directory.address != 0 && directory.size != 0)
            .map(|directory| (Rva(directory.address), directory.size))
    };
    let in_removed = |rva: Rva| {
        at.pe.sections.iter().any(|section| {
            removed.contains(&section.name.as_str()) && section.contains_mapped_rva(rva)
        })
    };
    let moves = |index: usize| present(index).filter(|(rva, _)| in_removed(*rva));

    // ---- read every table that has to move -------------------------------
    let exports = match moves(EXPORT_DIRECTORY) {
        Some(from) => Some((from.0, read_exports(&at)?.2)),
        None => None,
    };
    let imports = match moves(IMPORT_DIRECTORY) {
        Some(from) => Some((from.0, read_imports(&at)?)),
        None => None,
    };
    let exceptions = match moves(EXCEPTION_DIRECTORY) {
        Some((from, size)) => {
            let count = size as usize / RUNTIME_FUNCTION_BYTES;
            let table = at.slice(from, count * RUNTIME_FUNCTION_BYTES)?;
            let mut kept = Vec::with_capacity(table.len());
            let mut dropped = 0_usize;
            for index in 0..count {
                let entry =
                    &table[index * RUNTIME_FUNCTION_BYTES..(index + 1) * RUNTIME_FUNCTION_BYTES];
                let word = |at: usize| {
                    u32::from_le_bytes([entry[at], entry[at + 1], entry[at + 2], entry[at + 3]])
                };
                let (begin, end, unwind) = (word(0), word(4), word(8));
                if in_removed(Rva(begin)) || in_removed(Rva(end.saturating_sub(1))) {
                    dropped += 1;
                    continue;
                }
                if in_removed(Rva(unwind)) {
                    return Err(DirectoryRelocateError::Malformed(format!(
                        "runtime function {begin:#x} keeps unwind info at {unwind:#x}, inside a removed section"
                    )));
                }
                kept.extend_from_slice(entry);
            }
            Some((from, kept, dropped))
        }
        None => None,
    };
    let tls = match moves(TLS_DIRECTORY) {
        Some((from, _)) => {
            let header = at.slice(from, TLS_DIRECTORY_BYTES)?;
            let quad = |offset: usize| {
                u64::from_le_bytes(header[offset..offset + 8].try_into().expect("8 bytes"))
            };
            let word = |offset: usize| {
                u32::from_le_bytes(header[offset..offset + 4].try_into().expect("4 bytes"))
            };
            let to_rva = |va: u64| -> Result<Rva, DirectoryRelocateError> {
                va.checked_sub(image_base)
                    .and_then(|rva| u32::try_from(rva).ok())
                    .map(Rva)
                    .ok_or_else(|| {
                        DirectoryRelocateError::Malformed(format!(
                            "TLS pointer {va:#x} is below the base"
                        ))
                    })
            };
            let (raw_start, raw_end) = (to_rva(quad(0))?, to_rva(quad(8))?);
            if raw_end.0 < raw_start.0 {
                return Err(DirectoryRelocateError::Malformed(
                    "TLS raw data ends before it starts".to_owned(),
                ));
            }
            let raw = at
                .slice(raw_start, (raw_end.0 - raw_start.0) as usize)?
                .to_vec();
            let callbacks = to_rva(quad(24))?;
            let (mut kept, mut dropped) = (Vec::new(), Vec::new());
            for slot in 0_u32.. {
                let entry = Rva(callbacks.0.saturating_add(slot.saturating_mul(8)));
                let value =
                    u64::from_le_bytes(at.slice(entry, 8)?.try_into().expect("8 bytes"));
                if value == 0 {
                    break;
                }
                let target = to_rva(value)?;
                if in_removed(target) {
                    dropped.push(target);
                } else {
                    kept.push(value);
                }
                if slot > 1024 {
                    return Err(DirectoryRelocateError::Malformed(
                        "TLS callback array is unterminated".to_owned(),
                    ));
                }
            }
            Some((
                from,
                TlsState {
                    raw,
                    index_va: quad(16),
                    kept,
                    dropped,
                    zero_fill: word(32),
                    characteristics: word(36),
                },
            ))
        }
        None => None,
    };
    // Surviving fixups, as (target, kind). The rebuilt pointers are added when
    // the layout knows where they landed.
    let mut surviving: Vec<(Rva, u8)> = Vec::new();
    let mut dropped_fixups = 0_usize;
    for block in &at
        .pe
        .parse_relocations()
        .map_err(|error| DirectoryRelocateError::Malformed(error.to_string()))?
    {
        for entry in &block.entries {
            if entry.kind == 0 {
                continue;
            }
            let target = Rva(block.page_rva.0.saturating_add(u32::from(entry.offset)));
            if in_removed(target) {
                dropped_fixups += 1;
            } else {
                surviving.push((target, entry.kind));
            }
        }
    }
    let relocation_from = present(BASE_RELOCATION_DIRECTORY).map_or(Rva(0), |(rva, _)| rva);
    let iat_was = present(IAT_DIRECTORY).map_or(Rva(0), |(rva, _)| rva);

    // ---- lay everything out at `base` ------------------------------------
    let build = |base: Rva| -> (Vec<u8>, Vec<RelocatedDirectory>) {
        let mut payload: Vec<u8> = Vec::new();
        let mut placed: Vec<RelocatedDirectory> = Vec::new();
        let here = |payload: &Vec<u8>| Rva(base.0 + u32::try_from(payload.len()).unwrap_or(0));
        let mut fixups: Vec<Rva> = Vec::new();

        if let Some((from, exports)) = &exports {
            let start = here(&payload);
            let bytes = serialize_exports(exports, start);
            let size = u32::try_from(bytes.len()).unwrap_or(0);
            payload.extend_from_slice(&bytes);
            align8(&mut payload);
            placed.push(RelocatedDirectory { index: EXPORT_DIRECTORY, from: *from, to: start, size });
        }
        if let Some((from, table, _)) = &exceptions {
            let start = here(&payload);
            let size = u32::try_from(table.len()).unwrap_or(0);
            payload.extend_from_slice(table);
            align8(&mut payload);
            placed.push(RelocatedDirectory { index: EXCEPTION_DIRECTORY, from: *from, to: start, size });
        }
        if let Some((from, modules)) = &imports {
            let start = here(&payload);
            let (bytes, iat_rva, iat_size) = serialize_imports(modules, start);
            let descriptors = u32::try_from((modules.len() + 1) * DESCRIPTOR_BYTES).unwrap_or(0);
            payload.extend_from_slice(&bytes);
            align8(&mut payload);
            placed.push(RelocatedDirectory { index: IMPORT_DIRECTORY, from: *from, to: start, size: descriptors });
            placed.push(RelocatedDirectory {
                index: IAT_DIRECTORY,
                from: iat_was,
                to: iat_rva,
                size: iat_size,
            });
        }
        if let Some((from, tls)) = &tls {
            let start = here(&payload);
            let raw_rva = start.0 + u32::try_from(TLS_DIRECTORY_BYTES).unwrap_or(0);
            let callbacks = raw_rva + u32::try_from(tls.raw.len()).unwrap_or(0);
            payload.extend_from_slice(&(image_base + u64::from(raw_rva)).to_le_bytes());
            payload.extend_from_slice(
                &(image_base + u64::from(raw_rva) + tls.raw.len() as u64).to_le_bytes(),
            );
            payload.extend_from_slice(&tls.index_va.to_le_bytes());
            payload.extend_from_slice(&(image_base + u64::from(callbacks)).to_le_bytes());
            payload.extend_from_slice(&tls.zero_fill.to_le_bytes());
            payload.extend_from_slice(&tls.characteristics.to_le_bytes());
            payload.extend_from_slice(&tls.raw);
            for callback in &tls.kept {
                payload.extend_from_slice(&callback.to_le_bytes());
            }
            payload.extend_from_slice(&0_u64.to_le_bytes());
            align8(&mut payload);
            placed.push(RelocatedDirectory {
                index: TLS_DIRECTORY,
                from: *from,
                to: start,
                size: u32::try_from(TLS_DIRECTORY_BYTES).unwrap_or(0),
            });
            // The four VA fields, then one slot per surviving callback.
            fixups.extend([start, Rva(start.0 + 8), Rva(start.0 + 16), Rva(start.0 + 24)]);
            for slot in 0..tls.kept.len() {
                fixups.push(Rva(callbacks + u32::try_from(slot * 8).unwrap_or(0)));
            }
        }
        // Relocations last: they describe the pointers written above.
        let mut pages: std::collections::BTreeMap<u32, Vec<u16>> =
            std::collections::BTreeMap::new();
        for (target, kind) in &surviving {
            pages
                .entry(target.0 & !0xfff)
                .or_default()
                .push((u16::from(*kind) << 12) | u16::try_from(target.0 & 0xfff).unwrap_or(0));
        }
        for target in &fixups {
            pages
                .entry(target.0 & !0xfff)
                .or_default()
                .push((DIR64 << 12) | u16::try_from(target.0 & 0xfff).unwrap_or(0));
        }
        if !pages.is_empty() {
            let start = here(&payload);
            let mut table: Vec<u8> = Vec::new();
            for (page, mut entries) in pages {
                entries.sort_unstable();
                if entries.len() % 2 != 0 {
                    entries.push(0);
                }
                let size = u32::try_from(8 + entries.len() * 2).unwrap_or(0);
                table.extend_from_slice(&page.to_le_bytes());
                table.extend_from_slice(&size.to_le_bytes());
                for entry in entries {
                    table.extend_from_slice(&entry.to_le_bytes());
                }
            }
            let size = u32::try_from(table.len()).unwrap_or(0);
            payload.extend_from_slice(&table);
            align8(&mut payload);
            placed.push(RelocatedDirectory {
                index: BASE_RELOCATION_DIRECTORY,
                from: relocation_from,
                to: start,
                size,
            });
        }
        (payload, placed)
    };

    let (probe, _) = build(Rva(0));
    if probe.is_empty() {
        return Err(DirectoryRelocateError::Absent("relocatable directory"));
    }
    let first = append_section(
        input,
        AppendSection { name, data: &probe, characteristics: READ_WRITE_DATA },
    )?;
    let placed_at = first.section.rva;
    let (payload, placements) = build(placed_at);
    if payload.len() != probe.len() {
        return Err(DirectoryRelocateError::Malformed(
            "combined directory payload changed length between passes".to_owned(),
        ));
    }
    let second = append_section(
        input,
        AppendSection { name, data: &payload, characteristics: READ_WRITE_DATA },
    )?;
    if second.section.rva != placed_at {
        return Err(DirectoryRelocateError::Unstable {
            first: placed_at,
            second: second.section.rva,
        });
    }
    let mut bytes = second.bytes;
    for directory in &placements {
        write_directory(&mut bytes, directory.index, directory.to, directory.size)?;
    }
    // Read the moved tables back out of the rewritten image before offering it.
    let rebuilt = At::new(&bytes)?;
    if exports.is_some() {
        read_exports(&rebuilt)?;
    }
    if imports.is_some() {
        read_imports(&rebuilt)?;
    }
    // Count what the rebuilt table actually holds beyond the survivors, rather
    // than guessing from the directory list: the TLS rebuild emits one fixup per
    // pointer it wrote, which is four header fields PLUS one per callback.
    let added = rebuilt
        .pe
        .parse_relocations()
        .map_err(|error| DirectoryRelocateError::Malformed(error.to_string()))?
        .iter()
        .map(|block| block.entries.iter().filter(|entry| entry.kind != 0).count())
        .sum::<usize>()
        .saturating_sub(surviving.len());
    Ok(AllDirectories {
        bytes,
        section: second.section,
        moved: placements,
        exception_entries: exceptions
            .as_ref()
            .map_or((0, 0), |(_, table, dropped)| {
                (table.len() / RUNTIME_FUNCTION_BYTES, *dropped)
            }),
        import_modules: imports.as_ref().map_or(0, |(_, modules)| modules.len()),
        import_symbols: imports.as_ref().map_or(0, |(_, modules)| {
            modules.iter().map(|module| module.symbols.len()).sum()
        }),
        tls_callbacks: tls
            .as_ref()
            .map_or((0, Vec::new()), |(_, tls)| (tls.kept.len(), tls.dropped.clone())),
        relocations: RelocationRebuild {
            kept: surviving.len(),
            dropped: dropped_fixups,
            added,
        },
    })
}

#[cfg(test)]
mod tests;

const EXCEPTION_DIRECTORY: usize = 3;
const RUNTIME_FUNCTION_BYTES: usize = 12;

/// What a rebuilt exception table left behind.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExceptionRebuild {
    pub kept: usize,
    pub dropped: usize,
}

/// Rebuilds the exception directory outside the sections named in `removed`,
/// dropping the `RUNTIME_FUNCTION` entries that describe code inside them.
///
/// Measured on the sample: of 67_616 entries, 66_688 describe `.text` with
/// their unwind info in `.xdata` -- the original program's, untouched by the
/// protector -- while 897 describe `.vmp2` and 31 `.vmp3`. An entry for code
/// that is about to be removed would dangle, so dropping exactly those is what
/// keeps the table honest rather than a loss.
///
/// The unwind blocks themselves are NOT moved: every kept entry's
/// `UnwindInfoAddress` already points into `.xdata`.
///
/// # Errors
///
/// An absent or malformed exception directory, a kept entry whose unwind info
/// lies in a removed section (which would dangle), or a failure of the append.
pub fn rebuild_exception_directory(
    input: &[u8],
    name: [u8; 8],
    removed: &[&str],
) -> Result<(DirectoryRelocation, ExceptionRebuild), DirectoryRelocateError> {
    let at = At::new(input)?;
    let directory = at
        .pe
        .directory(EXCEPTION_DIRECTORY)
        .filter(|directory| directory.address != 0 && directory.size != 0)
        .ok_or(DirectoryRelocateError::Absent("exception"))?;
    let from = Rva(directory.address);
    let from_size = directory.size;
    let count = directory.size as usize / RUNTIME_FUNCTION_BYTES;
    let table = at.slice(from, count * RUNTIME_FUNCTION_BYTES)?;

    let in_removed = |rva: u32| {
        at.pe
            .sections
            .iter()
            .any(|section| removed.contains(&section.name.as_str()) && section.contains_mapped_rva(Rva(rva)))
    };
    let mut kept = Vec::with_capacity(table.len());
    let mut dropped = 0_usize;
    for index in 0..count {
        let entry = &table[index * RUNTIME_FUNCTION_BYTES..(index + 1) * RUNTIME_FUNCTION_BYTES];
        let word = |at: usize| u32::from_le_bytes([entry[at], entry[at + 1], entry[at + 2], entry[at + 3]]);
        let (begin, end, unwind) = (word(0), word(4), word(8));
        if in_removed(begin) || in_removed(end.saturating_sub(1)) {
            dropped += 1;
            continue;
        }
        // A kept entry must not point its unwind info into a section that is
        // going away; the table would parse and then fault on unwind.
        if in_removed(unwind) {
            return Err(DirectoryRelocateError::Malformed(format!(
                "runtime function {begin:#x} keeps unwind info at {unwind:#x}, inside a removed section"
            )));
        }
        kept.extend_from_slice(entry);
    }

    let appended = append_section(
        input,
        AppendSection {
            name,
            data: &kept,
            characteristics: READ_ONLY_DATA,
        },
    )?;
    // Nothing inside a RUNTIME_FUNCTION is relative to the table, so unlike the
    // export directory this needs no second pass.
    let size = u32::try_from(kept.len()).unwrap_or(u32::MAX);
    let placed = appended.section.rva;
    let mut bytes = appended.bytes;
    write_directory(&mut bytes, EXCEPTION_DIRECTORY, placed, size)?;
    let rebuilt = At::new(&bytes)?;
    let readback = rebuilt
        .pe
        .directory(EXCEPTION_DIRECTORY)
        .ok_or_else(|| DirectoryRelocateError::Malformed("exception slot vanished".to_owned()))?;
    if readback.address != placed.0 || readback.size != size {
        return Err(DirectoryRelocateError::Malformed(
            "the rebuilt exception directory does not read back".to_owned(),
        ));
    }
    Ok((
        DirectoryRelocation {
            bytes,
            section: appended.section,
            moved: vec![RelocatedDirectory {
                index: EXCEPTION_DIRECTORY,
                from,
                to: placed,
                size: from_size,
            }],
        },
        ExceptionRebuild {
            kept: kept.len() / RUNTIME_FUNCTION_BYTES,
            dropped,
        },
    ))
}
