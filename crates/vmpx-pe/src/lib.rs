mod address;
mod error;
mod model;
mod reader;
mod strip;

pub use address::{FileOffset, ImageBase, Rva, Va};
pub use error::PeError;
pub use model::*;
pub use strip::{
    DanglingRef, SectionSelector, StripError, StripRequest, StrippedPe, strip_sections,
};

use std::collections::HashMap;

use reader::Reader;

const DOS_MAGIC: u16 = 0x5A4D;
const NT_MAGIC: u32 = 0x0000_4550;
const PE32_MAGIC: u16 = 0x010B;
const PE32_PLUS_MAGIC: u16 = 0x020B;
const COFF_HEADER_SIZE: usize = 20;
const SECTION_HEADER_SIZE: usize = 40;
const MAX_SECTIONS: usize = 1_024;
const MAX_DIRECTORY_ENTRIES: usize = 16;
const MAX_IMPORT_DESCRIPTORS: usize = 65_536;
const MAX_IMPORT_ENTRIES: usize = 1_000_000;
const MAX_EXPORT_ENTRIES: usize = 4_000_000;
const MAX_TLS_CALLBACKS: usize = 65_536;
const MAX_EXCEPTION_ENTRIES: usize = 4_000_000;
const MAX_C_STRING: usize = 16 * 1_024;
const UNW_FLAG_EHANDLER: u8 = 0x01;
const UNW_FLAG_UHANDLER: u8 = 0x02;
const UNW_FLAG_CHAININFO: u8 = 0x04;

#[derive(Clone, Debug)]
pub struct PeImage<'a> {
    data: &'a [u8],
    pub nt_headers_offset: FileOffset,
    pub coff: CoffHeader,
    pub optional: OptionalHeader,
    pub sections: Vec<Section>,
    pub directories: Vec<DataDirectory>,
}

impl<'a> PeImage<'a> {
    /// Parses required PE structure without trusting file-controlled counts or offsets.
    ///
    /// # Errors
    ///
    /// Returns a located error when required headers are malformed, truncated,
    /// unsupported, or exceed configured structural limits.
    #[allow(clippy::too_many_lines)]
    pub fn parse(data: &'a [u8]) -> Result<Self, PeError> {
        let reader = Reader::new(data);
        if reader.u16(0, "DOS signature")? != DOS_MAGIC {
            return Err(PeError::parse(0, "invalid DOS signature"));
        }

        let nt_offset_u32 = reader.u32(0x3C, "e_lfanew")?;
        let nt_offset = usize::try_from(nt_offset_u32)
            .map_err(|_| PeError::parse(0x3C, "e_lfanew cannot be represented"))?;
        if reader.u32(nt_offset, "NT signature")? != NT_MAGIC {
            return Err(PeError::parse(nt_offset, "invalid NT signature"));
        }

        let coff_offset = nt_offset
            .checked_add(4)
            .ok_or_else(|| PeError::parse(nt_offset, "COFF header offset overflows"))?;
        reader.bytes(coff_offset, COFF_HEADER_SIZE, "COFF header")?;
        let machine = reader.u16(coff_offset, "COFF machine")?;
        let number_of_sections = reader.u16(coff_offset + 2, "section count")?;
        if usize::from(number_of_sections) > MAX_SECTIONS {
            return Err(PeError::parse(
                coff_offset + 2,
                format!("section count {number_of_sections} exceeds limit {MAX_SECTIONS}"),
            ));
        }
        let size_of_optional_header = reader.u16(coff_offset + 16, "optional-header size")?;
        let coff = CoffHeader {
            machine,
            machine_name: machine_name(machine),
            number_of_sections,
            time_date_stamp: reader.u32(coff_offset + 4, "COFF timestamp")?,
            pointer_to_symbol_table: FileOffset(u64::from(
                reader.u32(coff_offset + 8, "COFF symbol-table offset")?,
            )),
            number_of_symbols: reader.u32(coff_offset + 12, "COFF symbol count")?,
            size_of_optional_header,
            characteristics: reader.u16(coff_offset + 18, "COFF characteristics")?,
        };

        let optional_offset = coff_offset
            .checked_add(COFF_HEADER_SIZE)
            .ok_or_else(|| PeError::parse(coff_offset, "optional-header offset overflows"))?;
        let optional_size = usize::from(size_of_optional_header);
        reader.bytes(optional_offset, optional_size, "optional header")?;
        let optional_end = optional_offset
            .checked_add(optional_size)
            .ok_or_else(|| PeError::parse(optional_offset, "optional-header end overflows"))?;
        let magic = reader.u16(optional_offset, "optional-header magic")?;
        let (kind, minimum_size, image_base, directory_count_offset, directory_offset) = match magic
        {
            PE32_MAGIC => (
                PeKind::Pe32,
                96,
                ImageBase(u64::from(reader.u32(optional_offset + 28, "image base")?)),
                92,
                96,
            ),
            PE32_PLUS_MAGIC => (
                PeKind::Pe32Plus,
                112,
                ImageBase(reader.u64(optional_offset + 24, "image base")?),
                108,
                112,
            ),
            _ => {
                return Err(PeError::Unsupported {
                    offset: u64::try_from(optional_offset).unwrap_or(u64::MAX),
                    message: format!("optional-header magic {magic:#X} is not PE32/PE32+"),
                });
            }
        };
        if optional_size < minimum_size {
            return Err(PeError::parse(
                optional_offset,
                format!(
                    "{kind:?} optional header is {optional_size} bytes, below minimum {minimum_size}"
                ),
            ));
        }

        let read_u8 = |relative: usize, what: &str| -> Result<u8, PeError> {
            Ok(reader.bytes(optional_offset + relative, 1, what)?[0])
        };
        let read_u16 = |relative: usize, what: &str| reader.u16(optional_offset + relative, what);
        let read_u32 = |relative: usize, what: &str| reader.u32(optional_offset + relative, what);
        let read_u64 = |relative: usize, what: &str| reader.u64(optional_offset + relative, what);

        let (stack_reserve, stack_commit, heap_reserve, heap_commit) = match kind {
            PeKind::Pe32 => (
                u64::from(read_u32(72, "stack reserve")?),
                u64::from(read_u32(76, "stack commit")?),
                u64::from(read_u32(80, "heap reserve")?),
                u64::from(read_u32(84, "heap commit")?),
            ),
            PeKind::Pe32Plus => (
                read_u64(72, "stack reserve")?,
                read_u64(80, "stack commit")?,
                read_u64(88, "heap reserve")?,
                read_u64(96, "heap commit")?,
            ),
        };
        let subsystem = read_u16(68, "subsystem")?;
        let number_of_rva_and_sizes = read_u32(directory_count_offset, "directory count")?;
        let optional = OptionalHeader {
            kind,
            linker_major: read_u8(2, "linker major")?,
            linker_minor: read_u8(3, "linker minor")?,
            size_of_code: read_u32(4, "size of code")?,
            size_of_initialized_data: read_u32(8, "size of initialized data")?,
            size_of_uninitialized_data: read_u32(12, "size of uninitialized data")?,
            address_of_entry_point: Rva(read_u32(16, "entry point")?),
            base_of_code: Rva(read_u32(20, "base of code")?),
            image_base,
            section_alignment: read_u32(32, "section alignment")?,
            file_alignment: read_u32(36, "file alignment")?,
            size_of_image: read_u32(56, "size of image")?,
            size_of_headers: read_u32(60, "size of headers")?,
            checksum: read_u32(64, "checksum")?,
            subsystem,
            subsystem_name: subsystem_name(subsystem),
            dll_characteristics: read_u16(70, "DLL characteristics")?,
            size_of_stack_reserve: stack_reserve,
            size_of_stack_commit: stack_commit,
            size_of_heap_reserve: heap_reserve,
            size_of_heap_commit: heap_commit,
            number_of_rva_and_sizes,
        };

        let available_directories = optional_size
            .saturating_sub(directory_offset)
            .checked_div(8)
            .unwrap_or(0);
        let requested_directories = usize::try_from(number_of_rva_and_sizes)
            .unwrap_or(usize::MAX)
            .min(MAX_DIRECTORY_ENTRIES);
        if requested_directories > available_directories {
            return Err(PeError::parse(
                optional_offset + directory_offset,
                format!(
                    "directory count {requested_directories} exceeds optional-header capacity {available_directories}"
                ),
            ));
        }
        let mut directories = Vec::with_capacity(requested_directories);
        for index in 0..requested_directories {
            let offset = optional_offset + directory_offset + index * 8;
            directories.push(DataDirectory {
                index,
                name: directory_name(index),
                address_kind: if index == IMAGE_DIRECTORY_ENTRY_SECURITY {
                    DirectoryAddressKind::FileOffset
                } else {
                    DirectoryAddressKind::Rva
                },
                address: reader.u32(offset, "directory address")?,
                size: reader.u32(offset + 4, "directory size")?,
            });
        }

        let section_table_offset = optional_end;
        let section_table_size = usize::from(number_of_sections)
            .checked_mul(SECTION_HEADER_SIZE)
            .ok_or_else(|| PeError::parse(section_table_offset, "section table size overflows"))?;
        reader.bytes(section_table_offset, section_table_size, "section table")?;
        let mut sections = Vec::with_capacity(usize::from(number_of_sections));
        for index in 0..usize::from(number_of_sections) {
            let offset = section_table_offset + index * SECTION_HEADER_SIZE;
            let raw_name = reader.bytes(offset, 8, "section name")?;
            let mut name_bytes = [0_u8; 8];
            name_bytes.copy_from_slice(raw_name);
            let name_end = name_bytes.iter().position(|byte| *byte == 0).unwrap_or(8);
            let name = String::from_utf8_lossy(&name_bytes[..name_end]).into_owned();
            sections.push(Section {
                index,
                name,
                name_bytes,
                virtual_size: reader.u32(offset + 8, "section virtual size")?,
                virtual_address: Rva(reader.u32(offset + 12, "section RVA")?),
                size_of_raw_data: reader.u32(offset + 16, "section raw size")?,
                pointer_to_raw_data: FileOffset(u64::from(
                    reader.u32(offset + 20, "section raw offset")?,
                )),
                pointer_to_relocations: FileOffset(u64::from(
                    reader.u32(offset + 24, "section relocation offset")?,
                )),
                pointer_to_linenumbers: FileOffset(u64::from(
                    reader.u32(offset + 28, "section line-number offset")?,
                )),
                number_of_relocations: reader.u16(offset + 32, "section relocation count")?,
                number_of_linenumbers: reader.u16(offset + 34, "section line-number count")?,
                characteristics: reader.u32(offset + 36, "section characteristics")?,
            });
        }

        Ok(Self {
            data,
            nt_headers_offset: FileOffset(u64::from(nt_offset_u32)),
            coff,
            optional,
            sections,
            directories,
        })
    }

    #[must_use]
    pub const fn data(&self) -> &'a [u8] {
        self.data
    }

    /// Parses one file-backed x64 `UNWIND_INFO` record. The code array is
    /// retained as bounded raw slots and the optional language handler or
    /// chained runtime-function record is decoded without following chains.
    ///
    /// # Errors
    ///
    /// Returns a located parse/translation error for truncated, unmapped, or
    /// internally contradictory unwind metadata.
    pub fn parse_unwind_info(&self, rva: Rva) -> Result<UnwindInfo, PeError> {
        if self.coff.machine != 0x8664 {
            return Err(PeError::translation(
                "x64 UNWIND_INFO parsing requires an AMD64 PE image",
            ));
        }
        let reader = self.reader();
        let offset = usize::try_from(self.rva_to_file_offset(rva)?.get())
            .map_err(|_| PeError::translation("unwind-info file offset cannot be represented"))?;
        let header = reader.bytes(offset, 4, "UNWIND_INFO header")?;
        let version = header[0] & 0x07;
        let flags = header[0] >> 3;
        if version == 0 {
            return Err(PeError::parse(offset, "UNWIND_INFO version is zero"));
        }
        if flags & UNW_FLAG_CHAININFO != 0 && flags & (UNW_FLAG_EHANDLER | UNW_FLAG_UHANDLER) != 0 {
            return Err(PeError::parse(
                offset,
                "UNW_INFO_CHAININFO is combined with a language-handler flag",
            ));
        }

        let count_of_codes = header[2];
        let code_bytes = usize::from(count_of_codes)
            .checked_mul(2)
            .ok_or_else(|| PeError::parse(offset + 2, "unwind-code byte count overflows"))?;
        let code_start = offset
            .checked_add(4)
            .ok_or_else(|| PeError::parse(offset, "unwind-code offset overflows"))?;
        let raw_codes = reader.bytes(code_start, code_bytes, "UNWIND_CODE array")?;
        let code_slots = raw_codes
            .chunks_exact(2)
            .map(|slot| UnwindCodeSlot {
                code_offset: slot[0],
                unwind_operation: slot[1] & 0x0F,
                operation_info: slot[1] >> 4,
            })
            .collect::<Vec<_>>();

        let padded_slots = usize::from(count_of_codes)
            .checked_add(1)
            .map(|slots| slots & !1)
            .ok_or_else(|| PeError::parse(offset + 2, "padded unwind-code count overflows"))?;
        let trailer_relative = 4_usize
            .checked_add(padded_slots.checked_mul(2).ok_or_else(|| {
                PeError::parse(offset + 2, "padded unwind-code byte count overflows")
            })?)
            .ok_or_else(|| PeError::parse(offset, "unwind trailer offset overflows"))?;
        let trailer_offset = offset
            .checked_add(trailer_relative)
            .ok_or_else(|| PeError::parse(offset, "unwind trailer file offset overflows"))?;
        let trailer_rva = rva
            .get()
            .checked_add(u32::try_from(trailer_relative).map_err(|_| {
                PeError::translation("unwind trailer relative RVA cannot be represented")
            })?)
            .map(Rva)
            .ok_or_else(|| PeError::translation("unwind trailer RVA overflows"))?;

        let (exception_handler, handler_data, chained_function) = if flags & UNW_FLAG_CHAININFO != 0
        {
            let bytes = reader.bytes(trailer_offset, 12, "chained RUNTIME_FUNCTION")?;
            (
                None,
                None,
                Some(RuntimeFunction {
                    begin: Rva(u32::from_le_bytes(bytes[0..4].try_into().map_err(
                        |_| PeError::parse(trailer_offset, "invalid chained begin field"),
                    )?)),
                    end: Rva(u32::from_le_bytes(bytes[4..8].try_into().map_err(
                        |_| PeError::parse(trailer_offset + 4, "invalid chained end field"),
                    )?)),
                    unwind_info: Rva(u32::from_le_bytes(bytes[8..12].try_into().map_err(
                        |_| PeError::parse(trailer_offset + 8, "invalid chained unwind field"),
                    )?)),
                }),
            )
        } else if flags & (UNW_FLAG_EHANDLER | UNW_FLAG_UHANDLER) != 0 {
            let handler = Rva(reader.u32(trailer_offset, "unwind exception handler RVA")?);
            let data = trailer_rva
                .get()
                .checked_add(4)
                .map(Rva)
                .ok_or_else(|| PeError::translation("unwind handler-data RVA overflows"))?;
            (Some(handler), Some(data), None)
        } else {
            (None, None, None)
        };

        Ok(UnwindInfo {
            rva,
            version,
            flags,
            size_of_prolog: header[1],
            count_of_codes,
            frame_register: header[3] & 0x0F,
            frame_offset: header[3] >> 4,
            code_slots,
            exception_handler,
            handler_data,
            chained_function,
        })
    }

    /// Returns every x64 runtime-function record whose half-open code range
    /// contains `target`. Overlaps are preserved because hostile or obfuscated
    /// images may intentionally provide more than one candidate.
    ///
    /// # Errors
    ///
    /// Propagates bounded exception-directory parsing and RVA translation
    /// failures.
    pub fn runtime_functions_containing(
        &self,
        target: Rva,
    ) -> Result<Vec<RuntimeFunction>, PeError> {
        Ok(self
            .parse_exception_functions()?
            .into_iter()
            .filter(|function| function.begin <= target && target < function.end)
            .collect())
    }

    #[must_use]
    pub fn directory(&self, index: usize) -> Option<&DataDirectory> {
        self.directories
            .get(index)
            .filter(|directory| directory.is_present())
    }

    /// Maps a file-backed RVA to its raw file offset.
    ///
    /// # Errors
    ///
    /// Returns an error when the RVA is unmapped, virtual-only, overflowing, or
    /// maps beyond the available file bytes.
    pub fn rva_to_file_offset(&self, rva: Rva) -> Result<FileOffset, PeError> {
        if rva.0 < self.optional.size_of_headers {
            let offset = usize::try_from(rva.0)
                .map_err(|_| PeError::translation(format!("header RVA {rva} is too large")))?;
            if offset < self.data.len() {
                return Ok(FileOffset(u64::from(rva.0)));
            }
            return Err(PeError::translation(format!(
                "header RVA {rva} exceeds file length"
            )));
        }

        for section in &self.sections {
            let Some(delta) = rva.0.checked_sub(section.virtual_address.0) else {
                continue;
            };
            if delta >= section.mapped_span() {
                continue;
            }
            if delta >= section.size_of_raw_data || section.pointer_to_raw_data.0 == 0 {
                return Err(PeError::translation(format!(
                    "RVA {rva} is in virtual-only bytes of section {}",
                    section.name
                )));
            }
            let raw = section
                .pointer_to_raw_data
                .0
                .checked_add(u64::from(delta))
                .ok_or_else(|| {
                    PeError::translation(format!("raw offset for RVA {rva} overflows"))
                })?;
            if raw >= u64::try_from(self.data.len()).unwrap_or(u64::MAX) {
                return Err(PeError::translation(format!(
                    "raw offset {raw:#X} for RVA {rva} exceeds file length"
                )));
            }
            return Ok(FileOffset(raw));
        }
        Err(PeError::translation(format!(
            "RVA {rva} is not owned by a section"
        )))
    }

    /// Maps a section-owned raw file offset to an RVA.
    ///
    /// # Errors
    ///
    /// Returns an error when no section owns the offset or arithmetic overflows.
    pub fn file_offset_to_rva(&self, offset: FileOffset) -> Result<Rva, PeError> {
        if offset.0 < u64::from(self.optional.size_of_headers) {
            return u32::try_from(offset.0)
                .map(Rva)
                .map_err(|_| PeError::translation(format!("header offset {offset} is too large")));
        }
        for section in &self.sections {
            let Some(delta) = offset.0.checked_sub(section.pointer_to_raw_data.0) else {
                continue;
            };
            if delta >= u64::from(section.size_of_raw_data) {
                continue;
            }
            let delta = u32::try_from(delta).map_err(|_| {
                PeError::translation(format!("section delta for {offset} is too large"))
            })?;
            return section
                .virtual_address
                .0
                .checked_add(delta)
                .map(Rva)
                .ok_or_else(|| PeError::translation(format!("RVA for {offset} overflows")));
        }
        Err(PeError::translation(format!(
            "file offset {offset} is not owned by a section"
        )))
    }

    fn reader(&self) -> Reader<'a> {
        Reader::new(self.data)
    }

    fn offset_for_rva(&self, rva: Rva) -> Result<usize, PeError> {
        usize::try_from(self.rva_to_file_offset(rva)?.0)
            .map_err(|_| PeError::translation(format!("offset for RVA {rva} is too large")))
    }

    fn rva_add(rva: Rva, amount: usize, what: &str) -> Result<Rva, PeError> {
        let amount = u32::try_from(amount)
            .map_err(|_| PeError::translation(format!("{what} delta is too large")))?;
        rva.checked_add(amount)
            .ok_or_else(|| PeError::translation(format!("{what} RVA overflows")))
    }

    fn parse_imports(&self) -> Result<Vec<ImportModule>, PeError> {
        let Some(directory) = self.directory(IMAGE_DIRECTORY_ENTRY_IMPORT) else {
            return Ok(Vec::new());
        };
        let descriptor_limit = usize::try_from(directory.size / 20)
            .unwrap_or(usize::MAX)
            .min(MAX_IMPORT_DESCRIPTORS);
        if descriptor_limit == 0 {
            return Err(PeError::translation(
                "import directory is smaller than one descriptor",
            ));
        }
        let mut modules = Vec::new();
        let reader = self.reader();
        for index in 0..descriptor_limit {
            let descriptor_rva =
                Self::rva_add(Rva(directory.address), index * 20, "import descriptor")?;
            let offset = self.offset_for_rva(descriptor_rva)?;
            let original_first_thunk = Rva(reader.u32(offset, "OriginalFirstThunk")?);
            let time_date_stamp = reader.u32(offset + 4, "import timestamp")?;
            let forwarder_chain = reader.u32(offset + 8, "import forwarder chain")?;
            let name_rva = Rva(reader.u32(offset + 12, "import DLL name RVA")?);
            let first_thunk = Rva(reader.u32(offset + 16, "FirstThunk")?);
            if original_first_thunk.0 == 0
                && time_date_stamp == 0
                && forwarder_chain == 0
                && name_rva.0 == 0
                && first_thunk.0 == 0
            {
                return Ok(modules);
            }
            if name_rva.0 == 0 || first_thunk.0 == 0 {
                return Err(PeError::parse(
                    offset,
                    "non-null import descriptor lacks name or IAT RVA",
                ));
            }
            let name_offset = self.offset_for_rva(name_rva)?;
            let name = reader.c_string(name_offset, MAX_C_STRING, "import DLL name")?;
            let lookup_rva = if original_first_thunk.0 == 0 {
                first_thunk
            } else {
                original_first_thunk
            };
            let entries = self.parse_import_thunks(lookup_rva, first_thunk)?;
            modules.push(ImportModule {
                name,
                descriptor_rva,
                original_first_thunk,
                first_thunk,
                entries,
            });
        }
        Err(PeError::translation(format!(
            "import descriptor array has no terminator within {descriptor_limit} records"
        )))
    }

    fn parse_import_thunks(
        &self,
        lookup_rva: Rva,
        iat_rva: Rva,
    ) -> Result<Vec<ImportSymbol>, PeError> {
        let reader = self.reader();
        let pointer_size = self.optional.kind.pointer_size();
        let ordinal_mask = if pointer_size == 8 {
            0x8000_0000_0000_0000_u64
        } else {
            0x8000_0000_u64
        };
        let max_entries = (self.data.len() / pointer_size).min(MAX_IMPORT_ENTRIES);
        let mut entries = Vec::new();
        for index in 0..max_entries {
            let delta = index
                .checked_mul(pointer_size)
                .ok_or_else(|| PeError::translation("import thunk offset overflows"))?;
            let item_rva = Self::rva_add(lookup_rva, delta, "import lookup")?;
            let offset = self.offset_for_rva(item_rva)?;
            let value = if pointer_size == 8 {
                reader.u64(offset, "64-bit import thunk")?
            } else {
                u64::from(reader.u32(offset, "32-bit import thunk")?)
            };
            if value == 0 {
                return Ok(entries);
            }
            let iat_item_rva = Self::rva_add(iat_rva, delta, "IAT entry")?;
            if value & ordinal_mask != 0 {
                entries.push(ImportSymbol {
                    iat_rva: iat_item_rva,
                    lookup_value: value,
                    name: None,
                    hint: None,
                    ordinal: Some((value & 0xFFFF) as u16),
                });
            } else {
                let name_rva_u32 = u32::try_from(value).map_err(|_| {
                    PeError::translation(format!("import name RVA {value:#X} exceeds 32 bits"))
                })?;
                let name_offset = self.offset_for_rva(Rva(name_rva_u32))?;
                let hint = reader.u16(name_offset, "import hint")?;
                let name = reader.c_string(name_offset + 2, MAX_C_STRING, "import symbol name")?;
                entries.push(ImportSymbol {
                    iat_rva: iat_item_rva,
                    lookup_value: value,
                    name: Some(name),
                    hint: Some(hint),
                    ordinal: None,
                });
            }
        }
        Err(PeError::translation(format!(
            "import thunk array has no terminator within {max_entries} entries"
        )))
    }

    fn parse_exports(&self) -> Result<Option<ExportDirectory>, PeError> {
        let Some(directory) = self.directory(IMAGE_DIRECTORY_ENTRY_EXPORT) else {
            return Ok(None);
        };
        if directory.size < 40 {
            return Err(PeError::translation(
                "export directory is smaller than 40 bytes",
            ));
        }
        let offset = self.offset_for_rva(Rva(directory.address))?;
        let reader = self.reader();
        let module_name_rva = Rva(reader.u32(offset + 12, "export module name")?);
        let ordinal_base = reader.u32(offset + 16, "export ordinal base")?;
        let function_count = usize::try_from(reader.u32(offset + 20, "export function count")?)
            .unwrap_or(usize::MAX);
        let name_count =
            usize::try_from(reader.u32(offset + 24, "export name count")?).unwrap_or(usize::MAX);
        if function_count > MAX_EXPORT_ENTRIES || name_count > MAX_EXPORT_ENTRIES {
            return Err(PeError::translation("export count exceeds resource limit"));
        }
        let functions_rva = Rva(reader.u32(offset + 28, "export address table")?);
        let names_rva = Rva(reader.u32(offset + 32, "export name pointer table")?);
        let ordinals_rva = Rva(reader.u32(offset + 36, "export ordinal table")?);
        let mut names_by_index = HashMap::with_capacity(name_count);
        for index in 0..name_count {
            let name_ptr_rva = Self::rva_add(names_rva, index * 4, "export name pointer")?;
            let name_ptr_offset = self.offset_for_rva(name_ptr_rva)?;
            let symbol_name_rva = Rva(reader.u32(name_ptr_offset, "export name RVA")?);
            let name_ordinal_rva = Self::rva_add(ordinals_rva, index * 2, "export ordinal")?;
            let ordinal_offset = self.offset_for_rva(name_ordinal_rva)?;
            let function_index = usize::from(reader.u16(ordinal_offset, "export ordinal")?);
            if function_index >= function_count {
                return Err(PeError::parse(
                    ordinal_offset,
                    "export name ordinal exceeds function table",
                ));
            }
            let symbol_name_offset = self.offset_for_rva(symbol_name_rva)?;
            names_by_index.insert(
                function_index,
                reader.c_string(symbol_name_offset, MAX_C_STRING, "export name")?,
            );
        }
        let export_end = directory.address.checked_add(directory.size);
        let mut symbols = Vec::with_capacity(function_count);
        for index in 0..function_count {
            let function_rva_offset =
                self.offset_for_rva(Self::rva_add(functions_rva, index * 4, "export address")?)?;
            let rva = Rva(reader.u32(function_rva_offset, "export function RVA")?);
            if rva.0 == 0 {
                continue;
            }
            let forwarder =
                if rva.0 >= directory.address && export_end.is_some_and(|end| rva.0 < end) {
                    let forwarder_offset = self.offset_for_rva(rva)?;
                    Some(reader.c_string(forwarder_offset, MAX_C_STRING, "export forwarder")?)
                } else {
                    None
                };
            let index_u32 = u32::try_from(index)
                .map_err(|_| PeError::translation("export index exceeds 32 bits"))?;
            symbols.push(ExportSymbol {
                ordinal: ordinal_base
                    .checked_add(index_u32)
                    .ok_or_else(|| PeError::translation("export ordinal overflows"))?,
                rva,
                name: names_by_index.remove(&index),
                forwarder,
            });
        }
        let module_name = if module_name_rva.0 == 0 {
            None
        } else {
            Some(reader.c_string(
                self.offset_for_rva(module_name_rva)?,
                MAX_C_STRING,
                "export module name",
            )?)
        };
        Ok(Some(ExportDirectory {
            module_name,
            ordinal_base,
            symbols,
        }))
    }

    fn parse_tls(&self) -> Result<Option<TlsDirectory>, PeError> {
        let Some(directory) = self.directory(IMAGE_DIRECTORY_ENTRY_TLS) else {
            return Ok(None);
        };
        let offset = self.offset_for_rva(Rva(directory.address))?;
        let reader = self.reader();
        let (start, end, index, callbacks, zero_fill, characteristics) = match self.optional.kind {
            PeKind::Pe32 => (
                u64::from(reader.u32(offset, "TLS raw-data start")?),
                u64::from(reader.u32(offset + 4, "TLS raw-data end")?),
                u64::from(reader.u32(offset + 8, "TLS index address")?),
                u64::from(reader.u32(offset + 12, "TLS callback address")?),
                reader.u32(offset + 16, "TLS zero-fill size")?,
                reader.u32(offset + 20, "TLS characteristics")?,
            ),
            PeKind::Pe32Plus => (
                reader.u64(offset, "TLS raw-data start")?,
                reader.u64(offset + 8, "TLS raw-data end")?,
                reader.u64(offset + 16, "TLS index address")?,
                reader.u64(offset + 24, "TLS callback address")?,
                reader.u32(offset + 32, "TLS zero-fill size")?,
                reader.u32(offset + 36, "TLS characteristics")?,
            ),
        };
        let mut callback_values = Vec::new();
        if callbacks != 0 {
            let callback_rva = self
                .optional
                .image_base
                .checked_rva_of(Va(callbacks))
                .ok_or_else(|| {
                    PeError::translation("TLS callback array VA is below image base or too large")
                })?;
            let pointer_size = self.optional.kind.pointer_size();
            for callback_index in 0..MAX_TLS_CALLBACKS {
                let item_rva =
                    Self::rva_add(callback_rva, callback_index * pointer_size, "TLS callback")?;
                let item_offset = self.offset_for_rva(item_rva)?;
                let value = if pointer_size == 8 {
                    reader.u64(item_offset, "TLS callback VA")?
                } else {
                    u64::from(reader.u32(item_offset, "TLS callback VA")?)
                };
                if value == 0 {
                    break;
                }
                callback_values.push(Va(value));
                if callback_index + 1 == MAX_TLS_CALLBACKS {
                    return Err(PeError::translation(
                        "TLS callback array exceeds resource limit",
                    ));
                }
            }
        }
        Ok(Some(TlsDirectory {
            start_address_of_raw_data: Va(start),
            end_address_of_raw_data: Va(end),
            address_of_index: Va(index),
            address_of_callbacks: Va(callbacks),
            size_of_zero_fill: zero_fill,
            characteristics,
            callbacks: callback_values,
        }))
    }

    fn parse_exception_functions(&self) -> Result<Vec<RuntimeFunction>, PeError> {
        let Some(directory) = self.directory(IMAGE_DIRECTORY_ENTRY_EXCEPTION) else {
            return Ok(Vec::new());
        };
        if directory.size % 12 != 0 {
            return Err(PeError::translation(format!(
                "exception directory size {:#X} is not a multiple of 12",
                directory.size
            )));
        }
        let count = usize::try_from(directory.size / 12).unwrap_or(usize::MAX);
        if count > MAX_EXCEPTION_ENTRIES {
            return Err(PeError::translation(
                "exception-function count exceeds resource limit",
            ));
        }
        let reader = self.reader();
        let mut functions = Vec::with_capacity(count);
        for index in 0..count {
            let rva = Self::rva_add(Rva(directory.address), index * 12, "runtime function")?;
            let offset = self.offset_for_rva(rva)?;
            functions.push(RuntimeFunction {
                begin: Rva(reader.u32(offset, "runtime-function begin")?),
                end: Rva(reader.u32(offset + 4, "runtime-function end")?),
                unwind_info: Rva(reader.u32(offset + 8, "runtime-function unwind info")?),
            });
        }
        Ok(functions)
    }

    fn parse_relocations(&self) -> Result<Vec<BaseRelocationBlock>, PeError> {
        let Some(directory) = self.directory(IMAGE_DIRECTORY_ENTRY_BASERELOC) else {
            return Ok(Vec::new());
        };
        let mut consumed = 0_u32;
        let reader = self.reader();
        let mut blocks = Vec::new();
        while consumed < directory.size {
            let block_rva = Rva(directory
                .address
                .checked_add(consumed)
                .ok_or_else(|| PeError::translation("base-relocation directory RVA overflows"))?);
            let offset = self.offset_for_rva(block_rva)?;
            let page_rva = Rva(reader.u32(offset, "relocation page RVA")?);
            let block_size = reader.u32(offset + 4, "relocation block size")?;
            if block_size < 8 || block_size % 2 != 0 || block_size > directory.size - consumed {
                return Err(PeError::parse(
                    offset + 4,
                    "invalid base-relocation block size",
                ));
            }
            let entry_count = usize::try_from((block_size - 8) / 2).unwrap_or(usize::MAX);
            let mut entries = Vec::with_capacity(entry_count);
            for index in 0..entry_count {
                let raw = reader.u16(offset + 8 + index * 2, "base relocation")?;
                let kind = (raw >> 12) as u8;
                let entry_offset = raw & 0x0FFF;
                let target_rva = page_rva.0.checked_add(u32::from(entry_offset)).map(Rva);
                entries.push(BaseRelocation {
                    kind,
                    offset: entry_offset,
                    target_rva,
                });
            }
            blocks.push(BaseRelocationBlock {
                page_rva,
                block_size,
                entries,
            });
            consumed = consumed
                .checked_add(block_size)
                .ok_or_else(|| PeError::translation("relocation byte count overflows"))?;
        }
        Ok(blocks)
    }

    fn parse_debug(&self) -> Result<Vec<DebugEntry>, PeError> {
        let Some(directory) = self.directory(IMAGE_DIRECTORY_ENTRY_DEBUG) else {
            return Ok(Vec::new());
        };
        if directory.size % 28 != 0 {
            return Err(PeError::translation(
                "debug directory size is not a multiple of 28",
            ));
        }
        let count = usize::try_from(directory.size / 28).unwrap_or(usize::MAX);
        let reader = self.reader();
        let mut entries = Vec::with_capacity(count);
        for index in 0..count {
            let rva = Self::rva_add(Rva(directory.address), index * 28, "debug entry")?;
            let offset = self.offset_for_rva(rva)?;
            entries.push(DebugEntry {
                characteristics: reader.u32(offset, "debug characteristics")?,
                time_date_stamp: reader.u32(offset + 4, "debug timestamp")?,
                major_version: reader.u16(offset + 8, "debug major version")?,
                minor_version: reader.u16(offset + 10, "debug minor version")?,
                kind: reader.u32(offset + 12, "debug type")?,
                size_of_data: reader.u32(offset + 16, "debug data size")?,
                address_of_raw_data: Rva(reader.u32(offset + 20, "debug data RVA")?),
                pointer_to_raw_data: FileOffset(u64::from(
                    reader.u32(offset + 24, "debug data file offset")?,
                )),
            });
        }
        Ok(entries)
    }

    fn parse_load_config(&self) -> Result<Option<LoadConfig>, PeError> {
        let Some(directory) = self.directory(IMAGE_DIRECTORY_ENTRY_LOAD_CONFIG) else {
            return Ok(None);
        };
        let offset = self.offset_for_rva(Rva(directory.address))?;
        Ok(Some(LoadConfig {
            declared_size: self.reader().u32(offset, "load-config size")?,
            directory_size: directory.size,
        }))
    }

    #[allow(clippy::too_many_lines)]
    fn validate_layout(&self) -> Vec<Diagnostic> {
        let mut diagnostics = Vec::new();
        let file_len = u64::try_from(self.data.len()).unwrap_or(u64::MAX);
        let valid_file_alignment =
            self.optional.file_alignment != 0 && self.optional.file_alignment.is_power_of_two();
        if !valid_file_alignment {
            diagnostics.push(Diagnostic {
                stage: "headers",
                message: format!(
                    "FileAlignment {:#X} is zero or not a power of two",
                    self.optional.file_alignment
                ),
            });
        }
        if self.optional.section_alignment == 0
            || !self.optional.section_alignment.is_power_of_two()
        {
            diagnostics.push(Diagnostic {
                stage: "headers",
                message: "SectionAlignment is zero or not a power of two".to_owned(),
            });
        }
        if valid_file_alignment
            && !self
                .optional
                .size_of_headers
                .is_multiple_of(self.optional.file_alignment)
        {
            diagnostics.push(Diagnostic {
                stage: "headers",
                message: "SizeOfHeaders is not FileAlignment-aligned".to_owned(),
            });
        }
        if self.optional.section_alignment != 0
            && !self
                .optional
                .size_of_image
                .is_multiple_of(self.optional.section_alignment)
        {
            diagnostics.push(Diagnostic {
                stage: "headers",
                message: "SizeOfImage is not SectionAlignment-aligned".to_owned(),
            });
        }
        for section in &self.sections {
            if section.size_of_raw_data != 0 {
                let end = section
                    .pointer_to_raw_data
                    .0
                    .checked_add(u64::from(section.size_of_raw_data));
                if end.is_none_or(|end| end > file_len) {
                    diagnostics.push(Diagnostic {
                        stage: "sections",
                        message: format!("section {} raw range exceeds file", section.name),
                    });
                }
                if valid_file_alignment
                    && section.size_of_raw_data % self.optional.file_alignment != 0
                {
                    diagnostics.push(Diagnostic {
                        stage: "sections",
                        message: format!(
                            "section {} raw size {:#X} is not FileAlignment-aligned",
                            section.name, section.size_of_raw_data
                        ),
                    });
                }
                if valid_file_alignment
                    && section.pointer_to_raw_data.0 % u64::from(self.optional.file_alignment) != 0
                {
                    diagnostics.push(Diagnostic {
                        stage: "sections",
                        message: format!(
                            "section {} raw offset {} is not FileAlignment-aligned",
                            section.name, section.pointer_to_raw_data
                        ),
                    });
                }
            }
            if self.optional.section_alignment != 0
                && section.virtual_address.0 % self.optional.section_alignment != 0
            {
                diagnostics.push(Diagnostic {
                    stage: "sections",
                    message: format!(
                        "section {} RVA {} is not SectionAlignment-aligned",
                        section.name, section.virtual_address
                    ),
                });
            }
            let virtual_end = section.virtual_address.0.checked_add(section.mapped_span());
            if virtual_end.is_none_or(|end| end > self.optional.size_of_image) {
                diagnostics.push(Diagnostic {
                    stage: "sections",
                    message: format!("section {} mapped range exceeds SizeOfImage", section.name),
                });
            }
        }
        if self.optional.address_of_entry_point.0 != 0
            && !self
                .sections
                .iter()
                .any(|section| section.contains_mapped_rva(self.optional.address_of_entry_point))
        {
            diagnostics.push(Diagnostic {
                stage: "entry_point",
                message: format!(
                    "entry point {} is not in a mapped section",
                    self.optional.address_of_entry_point
                ),
            });
        }
        for directory in self
            .directories
            .iter()
            .filter(|directory| directory.is_present())
        {
            let valid = match directory.address_kind {
                DirectoryAddressKind::Rva => directory
                    .address
                    .checked_add(directory.size.saturating_sub(1))
                    .is_some_and(|last| {
                        self.rva_to_file_offset(Rva(directory.address)).is_ok()
                            && self.rva_to_file_offset(Rva(last)).is_ok()
                    }),
                DirectoryAddressKind::FileOffset => u64::from(directory.address)
                    .checked_add(u64::from(directory.size))
                    .is_some_and(|end| end <= file_len),
            };
            if !valid {
                diagnostics.push(Diagnostic {
                    stage: "directories",
                    message: format!(
                        "{} directory start/range is not file-backed",
                        directory.name
                    ),
                });
            }
        }
        diagnostics
    }

    #[must_use]
    pub fn report(&self) -> PeReport {
        let mut diagnostics = self.validate_layout();

        macro_rules! optional_stage {
            ($stage:literal, $expr:expr, $fallback:expr) => {
                match $expr {
                    Ok(value) => value,
                    Err(error) => {
                        diagnostics.push(Diagnostic {
                            stage: $stage,
                            message: error.to_string(),
                        });
                        $fallback
                    }
                }
            };
        }

        let imports = optional_stage!("imports", self.parse_imports(), Vec::new());
        let exports = optional_stage!("exports", self.parse_exports(), None);
        let tls = optional_stage!("tls", self.parse_tls(), None);
        let exception_functions =
            optional_stage!("exceptions", self.parse_exception_functions(), Vec::new());
        let relocations = optional_stage!("relocations", self.parse_relocations(), Vec::new());
        let debug_entries = optional_stage!("debug", self.parse_debug(), Vec::new());
        let load_config = optional_stage!("load_config", self.parse_load_config(), None);

        PeReport {
            schema_version: 1,
            file_size: u64::try_from(self.data.len()).unwrap_or(u64::MAX),
            nt_headers_offset: self.nt_headers_offset,
            coff: self.coff.clone(),
            optional: self.optional.clone(),
            entry_point_va: self
                .optional
                .image_base
                .checked_add_rva(self.optional.address_of_entry_point),
            sections: self.sections.clone(),
            directories: self.directories.clone(),
            imports,
            exports,
            tls,
            exception_functions,
            relocations,
            debug_entries,
            load_config,
            diagnostics,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_pe64() -> Vec<u8> {
        let mut data = vec![0_u8; 0x400];
        data[0..2].copy_from_slice(&DOS_MAGIC.to_le_bytes());
        data[0x3C..0x40].copy_from_slice(&(0x80_u32).to_le_bytes());
        data[0x80..0x84].copy_from_slice(&NT_MAGIC.to_le_bytes());
        let coff = 0x84;
        data[coff..coff + 2].copy_from_slice(&(0x8664_u16).to_le_bytes());
        data[coff + 2..coff + 4].copy_from_slice(&(1_u16).to_le_bytes());
        data[coff + 16..coff + 18].copy_from_slice(&(0xF0_u16).to_le_bytes());
        data[coff + 18..coff + 20].copy_from_slice(&(0x2022_u16).to_le_bytes());
        let optional = coff + 20;
        data[optional..optional + 2].copy_from_slice(&PE32_PLUS_MAGIC.to_le_bytes());
        data[optional + 16..optional + 20].copy_from_slice(&(0x1000_u32).to_le_bytes());
        data[optional + 20..optional + 24].copy_from_slice(&(0x1000_u32).to_le_bytes());
        data[optional + 24..optional + 32].copy_from_slice(&(0x1_4000_0000_u64).to_le_bytes());
        data[optional + 32..optional + 36].copy_from_slice(&(0x1000_u32).to_le_bytes());
        data[optional + 36..optional + 40].copy_from_slice(&(0x200_u32).to_le_bytes());
        data[optional + 56..optional + 60].copy_from_slice(&(0x2000_u32).to_le_bytes());
        data[optional + 60..optional + 64].copy_from_slice(&(0x200_u32).to_le_bytes());
        data[optional + 68..optional + 70].copy_from_slice(&(3_u16).to_le_bytes());
        data[optional + 72..optional + 80].copy_from_slice(&(0x10_0000_u64).to_le_bytes());
        data[optional + 80..optional + 88].copy_from_slice(&(0x1000_u64).to_le_bytes());
        data[optional + 88..optional + 96].copy_from_slice(&(0x10_0000_u64).to_le_bytes());
        data[optional + 96..optional + 104].copy_from_slice(&(0x1000_u64).to_le_bytes());
        data[optional + 108..optional + 112].copy_from_slice(&(16_u32).to_le_bytes());
        let section = optional + 0xF0;
        data[section..section + 5].copy_from_slice(b".text");
        data[section + 8..section + 12].copy_from_slice(&(0x180_u32).to_le_bytes());
        data[section + 12..section + 16].copy_from_slice(&(0x1000_u32).to_le_bytes());
        data[section + 16..section + 20].copy_from_slice(&(0x200_u32).to_le_bytes());
        data[section + 20..section + 24].copy_from_slice(&(0x200_u32).to_le_bytes());
        data[section + 36..section + 40].copy_from_slice(&(0x6000_0020_u32).to_le_bytes());
        data
    }

    #[test]
    fn parses_pe64_and_translates_addresses() {
        let data = minimal_pe64();
        let pe = PeImage::parse(&data).expect("valid synthetic PE");
        assert_eq!(pe.optional.kind, PeKind::Pe32Plus);
        assert_eq!(pe.coff.machine_name, "x86-64");
        assert_eq!(pe.rva_to_file_offset(Rva(0x1010)), Ok(FileOffset(0x210)));
        assert_eq!(pe.file_offset_to_rva(FileOffset(0x210)), Ok(Rva(0x1010)));
        assert_eq!(pe.rva_to_file_offset(Rva(0x1180)), Ok(FileOffset(0x380)));
        assert!(pe.rva_to_file_offset(Rva(0x1200)).is_err());
    }

    #[test]
    fn rejects_negative_style_e_lfanew_without_panicking() {
        let mut data = minimal_pe64();
        data[0x3C..0x40].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(PeImage::parse(&data).is_err());
    }

    #[test]
    fn rejects_truncated_optional_header() {
        let mut data = minimal_pe64();
        data.truncate(0xA0);
        assert!(PeImage::parse(&data).is_err());
    }

    #[test]
    fn report_serializes() {
        let data = minimal_pe64();
        let report = PeImage::parse(&data).expect("valid PE").report();
        let json = serde_json::to_string(&report).expect("serialize report");
        assert!(json.contains("pe32_plus"));
    }

    #[test]
    fn parses_x64_unwind_language_handler_without_following_it() {
        let mut data = minimal_pe64();
        let optional = 0x84 + 20;
        let exception_directory = optional + 112 + IMAGE_DIRECTORY_ENTRY_EXCEPTION * 8;
        data[exception_directory..exception_directory + 4]
            .copy_from_slice(&0x1080_u32.to_le_bytes());
        data[exception_directory + 4..exception_directory + 8]
            .copy_from_slice(&12_u32.to_le_bytes());

        let runtime = 0x280;
        data[runtime..runtime + 4].copy_from_slice(&0x1000_u32.to_le_bytes());
        data[runtime + 4..runtime + 8].copy_from_slice(&0x1010_u32.to_le_bytes());
        data[runtime + 8..runtime + 12].copy_from_slice(&0x10A0_u32.to_le_bytes());

        let unwind = 0x2A0;
        data[unwind] = 1 | (UNW_FLAG_EHANDLER << 3);
        data[unwind + 1] = 1;
        data[unwind + 2] = 1;
        data[unwind + 4] = 1;
        data[unwind + 5] = 3 << 4;
        data[unwind + 8..unwind + 12].copy_from_slice(&0x1100_u32.to_le_bytes());

        let pe = PeImage::parse(&data).expect("valid synthetic PE");
        let function = &pe.report().exception_functions[0];
        let info = pe
            .parse_unwind_info(function.unwind_info)
            .expect("valid unwind info");
        assert_eq!(info.version, 1);
        assert_eq!(info.flags, UNW_FLAG_EHANDLER);
        assert_eq!(info.code_slots.len(), 1);
        assert_eq!(info.code_slots[0].unwind_operation, 0);
        assert_eq!(info.code_slots[0].operation_info, 3);
        assert_eq!(info.exception_handler, Some(Rva(0x1100)));
        assert_eq!(info.handler_data, Some(Rva(0x10AC)));
        assert!(info.chained_function.is_none());
    }
}
