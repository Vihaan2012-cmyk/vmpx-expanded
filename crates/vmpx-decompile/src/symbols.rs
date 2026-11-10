//! Names the decompiler can attach to addresses it would otherwise print raw.
//!
//! The recovered IR holds absolute addresses, not symbols. An address that is
//! an import-table slot is the most valuable kind to name: `*(u64 *)(0x...)`
//! says nothing, while `ws2_32!send` says what the function reaches for.
//!
//! Everything here is a lookup in the image's own import directory. No name is
//! invented, and an address that matches nothing stays a number.

use std::collections::BTreeMap;

/// What an imported function is for, as far as its module and name show.
#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd, Ord)]
pub enum Facility {
    /// Sockets, HTTP, DNS: the function can reach the network.
    Network,
    /// VMProtect's own licensing/serial API.
    License,
    /// Process, thread and debugger inspection.
    AntiDebug,
    /// Loading or resolving code at run time.
    DynamicLoad,
}

/// Modules whose every export reaches the network.
const NETWORK_MODULES: [&str; 7] = [
    "ws2_32", "wsock32", "wininet", "winhttp", "urlmon", "dnsapi", "iphlpapi",
];

/// Function-name fragments that reach the network from a general module.
const NETWORK_FUNCTIONS: [&str; 10] = [
    "socket",
    "connect",
    "send",
    "recv",
    "gethostby",
    "getaddrinfo",
    "inet_",
    "internetopen",
    "httpsendrequest",
    "winhttpsend",
];

/// VMProtect's licensing API, and the usual serial/licence spellings.
const LICENSE_FUNCTIONS: [&str; 8] = [
    "vmprotectgetserialnumberstate",
    "vmprotectsetserialnumber",
    "vmprotectgetserialnumberdata",
    "vmprotectgetcurrenthwid",
    "vmprotectisvalidimagecrc",
    "vmprotectactivatelicense",
    "vmprotectdeactivatelicense",
    "licence",
];

const ANTI_DEBUG_FUNCTIONS: [&str; 7] = [
    "isdebuggerpresent",
    "checkremotedebuggerpresent",
    "ntqueryinformationprocess",
    "outputdebugstring",
    "debugactiveprocess",
    "createtoolhelp32snapshot",
    "ntsetinformationthread",
];

const DYNAMIC_LOAD_FUNCTIONS: [&str; 4] = [
    "loadlibrary",
    "getprocaddress",
    "ldrloaddll",
    "ldrgetprocedureaddress",
];

/// Import names by the virtual address of their IAT slot.
#[derive(Clone, Debug, Default)]
pub struct Symbols {
    imports: BTreeMap<u64, String>,
    image_base: u64,
}

impl Symbols {
    /// Builds the table from an image's import directory. `image_base` turns
    /// each slot's RVA into the address the recovered IR actually holds.
    #[must_use]
    pub fn from_imports(
        image_base: u64,
        modules: impl IntoIterator<Item = (String, Vec<(u32, Option<String>)>)>,
    ) -> Self {
        let mut imports = BTreeMap::new();
        for (module, entries) in modules {
            let module = module
                .trim_end_matches(".dll")
                .trim_end_matches(".DLL")
                .to_owned();
            for (iat_rva, name) in entries {
                let name = name.unwrap_or_else(|| format!("ordinal_{iat_rva:#x}"));
                imports.insert(image_base + u64::from(iat_rva), format!("{module}!{name}"));
            }
        }
        Self {
            imports,
            image_base,
        }
    }

    /// The image base the IR's absolute addresses are relative to, 0 if unknown.
    #[must_use]
    pub fn image_base(&self) -> u64 {
        self.image_base
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.imports.is_empty()
    }

    /// The import whose IAT slot is at `address`.
    #[must_use]
    pub fn import_at(&self, address: u64) -> Option<&str> {
        self.imports.get(&address).map(String::as_str)
    }

    /// A C identifier for the import at `address`, if there is one.
    #[must_use]
    pub fn identifier_at(&self, address: u64) -> Option<String> {
        let full = self.import_at(address)?;
        let mut out = String::with_capacity(full.len() + 4);
        out.push_str("iat_");
        for ch in full.chars() {
            out.push(if ch.is_ascii_alphanumeric() { ch } else { '_' });
        }
        Some(out)
    }

    /// What the import at `address` is for, if it is recognisable.
    #[must_use]
    pub fn facility_at(&self, address: u64) -> Option<Facility> {
        facility_of(self.import_at(address)?)
    }
}

/// Classifies a `module!function` name. Case-insensitive, and the trailing
/// `A`/`W` of the Win32 spellings does not matter because the match is on a
/// fragment.
#[must_use]
pub fn facility_of(full: &str) -> Option<Facility> {
    let lower = full.to_ascii_lowercase();
    let (module, function) = lower.split_once('!').unwrap_or(("", lower.as_str()));
    if LICENSE_FUNCTIONS.iter().any(|f| function.contains(f)) {
        return Some(Facility::License);
    }
    if NETWORK_MODULES.contains(&module) || NETWORK_FUNCTIONS.iter().any(|f| function.contains(f)) {
        return Some(Facility::Network);
    }
    if ANTI_DEBUG_FUNCTIONS.iter().any(|f| function.contains(f)) {
        return Some(Facility::AntiDebug);
    }
    if DYNAMIC_LOAD_FUNCTIONS.iter().any(|f| function.contains(f)) {
        return Some(Facility::DynamicLoad);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table() -> Symbols {
        Symbols::from_imports(
            0x1_0000,
            [
                (
                    "ws2_32.dll".to_owned(),
                    vec![(0x200, Some("send".to_owned()))],
                ),
                (
                    "kernel32.dll".to_owned(),
                    vec![
                        (0x300, Some("IsDebuggerPresent".to_owned())),
                        (0x308, Some("LoadLibraryA".to_owned())),
                        (0x310, Some("lstrlenA".to_owned())),
                    ],
                ),
            ],
        )
    }

    #[test]
    fn an_iat_slot_resolves_to_its_import_and_identifier() {
        let s = table();
        assert_eq!(s.import_at(0x1_0200), Some("ws2_32!send"));
        assert_eq!(
            s.identifier_at(0x1_0200).as_deref(),
            Some("iat_ws2_32_send")
        );
        assert_eq!(s.import_at(0x1_0201), None, "only the exact slot resolves");
    }

    #[test]
    fn facilities_are_recognised_by_module_and_by_function() {
        let s = table();
        assert_eq!(s.facility_at(0x1_0200), Some(Facility::Network));
        assert_eq!(s.facility_at(0x1_0300), Some(Facility::AntiDebug));
        assert_eq!(s.facility_at(0x1_0308), Some(Facility::DynamicLoad));
        assert_eq!(
            s.facility_at(0x1_0310),
            None,
            "an ordinary string call is not a facility"
        );
    }

    #[test]
    fn the_vmprotect_licence_api_is_recognised() {
        assert_eq!(
            facility_of("kernel32!VMProtectGetSerialNumberState"),
            Some(Facility::License)
        );
        assert_eq!(facility_of("ws2_32!WSAStartup"), Some(Facility::Network));
    }

    #[test]
    fn an_unknown_address_is_left_as_a_number() {
        let s = table();
        assert_eq!(s.import_at(0xdead_beef), None);
        assert_eq!(s.identifier_at(0xdead_beef), None);
        assert_eq!(s.facility_at(0xdead_beef), None);
    }
}
