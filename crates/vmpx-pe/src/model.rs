use serde::{Deserialize, Serialize};

use crate::{FileOffset, ImageBase, Rva, Va};

pub const IMAGE_DIRECTORY_ENTRY_EXPORT: usize = 0;
pub const IMAGE_DIRECTORY_ENTRY_IMPORT: usize = 1;
pub const IMAGE_DIRECTORY_ENTRY_RESOURCE: usize = 2;
pub const IMAGE_DIRECTORY_ENTRY_EXCEPTION: usize = 3;
pub const IMAGE_DIRECTORY_ENTRY_SECURITY: usize = 4;
pub const IMAGE_DIRECTORY_ENTRY_BASERELOC: usize = 5;
pub const IMAGE_DIRECTORY_ENTRY_DEBUG: usize = 6;
pub const IMAGE_DIRECTORY_ENTRY_TLS: usize = 9;
pub const IMAGE_DIRECTORY_ENTRY_LOAD_CONFIG: usize = 10;
pub const IMAGE_DIRECTORY_ENTRY_IAT: usize = 12;
pub const IMAGE_DIRECTORY_ENTRY_COM_DESCRIPTOR: usize = 14;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PeKind {
    Pe32,
    Pe32Plus,
}

impl PeKind {
    #[must_use]
    pub const fn pointer_size(self) -> usize {
        match self {
            Self::Pe32 => 4,
            Self::Pe32Plus => 8,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CoffHeader {
    pub machine: u16,
    #[serde(skip_deserializing)]
    pub machine_name: &'static str,
    pub number_of_sections: u16,
    pub time_date_stamp: u32,
    pub pointer_to_symbol_table: FileOffset,
    pub number_of_symbols: u32,
    pub size_of_optional_header: u16,
    pub characteristics: u16,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct OptionalHeader {
    pub kind: PeKind,
    pub linker_major: u8,
    pub linker_minor: u8,
    pub size_of_code: u32,
    pub size_of_initialized_data: u32,
    pub size_of_uninitialized_data: u32,
    pub address_of_entry_point: Rva,
    pub base_of_code: Rva,
    pub image_base: ImageBase,
    pub section_alignment: u32,
    pub file_alignment: u32,
    pub size_of_image: u32,
    pub size_of_headers: u32,
    pub checksum: u32,
    pub subsystem: u16,
    #[serde(skip_deserializing)]
    pub subsystem_name: &'static str,
    pub dll_characteristics: u16,
    pub size_of_stack_reserve: u64,
    pub size_of_stack_commit: u64,
    pub size_of_heap_reserve: u64,
    pub size_of_heap_commit: u64,
    pub number_of_rva_and_sizes: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Section {
    pub index: usize,
    pub name: String,
    pub name_bytes: [u8; 8],
    pub virtual_size: u32,
    pub virtual_address: Rva,
    pub size_of_raw_data: u32,
    pub pointer_to_raw_data: FileOffset,
    pub pointer_to_relocations: FileOffset,
    pub pointer_to_linenumbers: FileOffset,
    pub number_of_relocations: u16,
    pub number_of_linenumbers: u16,
    pub characteristics: u32,
}

impl Section {
    #[must_use]
    pub fn mapped_span(&self) -> u32 {
        self.virtual_size.max(self.size_of_raw_data)
    }

    #[must_use]
    pub fn contains_mapped_rva(&self, rva: Rva) -> bool {
        let Some(delta) = rva.0.checked_sub(self.virtual_address.0) else {
            return false;
        };
        delta < self.mapped_span()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DirectoryAddressKind {
    Rva,
    FileOffset,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DataDirectory {
    pub index: usize,
    #[serde(skip_deserializing)]
    pub name: &'static str,
    pub address_kind: DirectoryAddressKind,
    pub address: u32,
    pub size: u32,
}

impl DataDirectory {
    #[must_use]
    pub const fn is_present(&self) -> bool {
        self.address != 0 && self.size != 0
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ImportModule {
    pub name: String,
    pub descriptor_rva: Rva,
    pub original_first_thunk: Rva,
    pub first_thunk: Rva,
    pub entries: Vec<ImportSymbol>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ImportSymbol {
    pub iat_rva: Rva,
    pub lookup_value: u64,
    pub name: Option<String>,
    pub hint: Option<u16>,
    pub ordinal: Option<u16>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ExportDirectory {
    pub module_name: Option<String>,
    pub ordinal_base: u32,
    pub symbols: Vec<ExportSymbol>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ExportSymbol {
    pub ordinal: u32,
    pub rva: Rva,
    pub name: Option<String>,
    pub forwarder: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TlsDirectory {
    pub start_address_of_raw_data: Va,
    pub end_address_of_raw_data: Va,
    pub address_of_index: Va,
    pub address_of_callbacks: Va,
    pub size_of_zero_fill: u32,
    pub characteristics: u32,
    pub callbacks: Vec<Va>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RuntimeFunction {
    pub begin: Rva,
    pub end: Rva,
    pub unwind_info: Rva,
}

/// One raw two-byte slot from an x64 `UNWIND_INFO` code array. Operations that
/// consume extension slots retain those slots independently so callers can
/// interpret them without the PE parser silently skipping malformed input.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct UnwindCodeSlot {
    pub code_offset: u8,
    pub unwind_operation: u8,
    pub operation_info: u8,
}

/// Bounded structural view of Windows x64 `UNWIND_INFO`.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct UnwindInfo {
    pub rva: Rva,
    pub version: u8,
    pub flags: u8,
    pub size_of_prolog: u8,
    pub count_of_codes: u8,
    pub frame_register: u8,
    pub frame_offset: u8,
    pub code_slots: Vec<UnwindCodeSlot>,
    pub exception_handler: Option<Rva>,
    pub handler_data: Option<Rva>,
    pub chained_function: Option<RuntimeFunction>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct BaseRelocationBlock {
    pub page_rva: Rva,
    pub block_size: u32,
    pub entries: Vec<BaseRelocation>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct BaseRelocation {
    pub kind: u8,
    pub offset: u16,
    pub target_rva: Option<Rva>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DebugEntry {
    pub characteristics: u32,
    pub time_date_stamp: u32,
    pub major_version: u16,
    pub minor_version: u16,
    pub kind: u32,
    pub size_of_data: u32,
    pub address_of_raw_data: Rva,
    pub pointer_to_raw_data: FileOffset,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct LoadConfig {
    pub declared_size: u32,
    pub directory_size: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Diagnostic {
    #[serde(skip_deserializing)]
    pub stage: &'static str,
    pub message: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PeReport {
    pub schema_version: u32,
    pub file_size: u64,
    pub nt_headers_offset: FileOffset,
    pub coff: CoffHeader,
    pub optional: OptionalHeader,
    pub entry_point_va: Option<Va>,
    pub sections: Vec<Section>,
    pub directories: Vec<DataDirectory>,
    pub imports: Vec<ImportModule>,
    pub exports: Option<ExportDirectory>,
    pub tls: Option<TlsDirectory>,
    pub exception_functions: Vec<RuntimeFunction>,
    pub relocations: Vec<BaseRelocationBlock>,
    pub debug_entries: Vec<DebugEntry>,
    pub load_config: Option<LoadConfig>,
    pub diagnostics: Vec<Diagnostic>,
}

pub(crate) const fn machine_name(machine: u16) -> &'static str {
    match machine {
        0x014C => "x86",
        0x8664 => "x86-64",
        0x01C0 => "ARM",
        0x01C4 => "ARMv7",
        0xAA64 => "ARM64",
        _ => "unknown",
    }
}

pub(crate) const fn subsystem_name(subsystem: u16) -> &'static str {
    match subsystem {
        0 => "unknown",
        1 => "native",
        2 => "windows_gui",
        3 => "windows_cui",
        7 => "posix_cui",
        9 => "windows_ce_gui",
        10 => "efi_application",
        11 => "efi_boot_service_driver",
        12 => "efi_runtime_driver",
        13 => "efi_rom",
        14 => "xbox",
        16 => "windows_boot_application",
        _ => "reserved",
    }
}

pub(crate) const fn directory_name(index: usize) -> &'static str {
    match index {
        0 => "export",
        1 => "import",
        2 => "resource",
        3 => "exception",
        4 => "security",
        5 => "base_relocation",
        6 => "debug",
        7 => "architecture",
        8 => "global_ptr",
        9 => "tls",
        10 => "load_config",
        11 => "bound_import",
        12 => "iat",
        13 => "delay_import",
        14 => "com_descriptor",
        15 => "reserved",
        _ => "nonstandard",
    }
}
