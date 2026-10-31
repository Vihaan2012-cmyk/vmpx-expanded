//! Pure, deterministic host-identity data for the contained Win32 model: the
//! benign user/host names, a minimal read-only registry, a clean process list,
//! and the sandbox/VM artifact denylists. No guest memory or OS access here;
//! [`crate::winapi`] owns marshalling.
//!
//! Everything answers *clean*: no debugger, no VM, no sandbox. The registry is a
//! closed world: only the keys listed here exist, every other key (and every
//! sandbox/VM artifact key, which is checked first) is NOT_FOUND.

pub(crate) const USER_NAME: &str = "user";
pub(crate) const COMPUTER_NAME: &str = "DESKTOP-GENERIC";

/// Substrings that mark a registry path as a sandbox/VM/analysis artifact.
const ARTIFACT_KEY_PARTS: &[&str] = &[
    "vbox", "virtualbox", "vmware", "vmtools", "vmci", "vmmouse", "vmhgfs", "vmusb", "vmx_svga",
    "vmxnet", "vmbus", "vmic", "hyper-v", "virtual machine", "qemu", "xen", "parallels", "prl_",
    "sbie", "sandboxie", "cuckoo", "joebox", "wine", "bochs",
];

/// Whether a registry path (already lowercased, backslash-joined) is a
/// sandbox/VM artifact key.
pub(crate) fn is_artifact_key(path_lower: &str) -> bool {
    ARTIFACT_KEY_PARTS.iter().any(|p| path_lower.contains(p))
}

/// A modeled registry value.
#[derive(Clone, Copy)]
pub(crate) enum RegVal {
    Sz(&'static str),
    Dword(u32),
}

impl RegVal {
    pub(crate) fn reg_type(self) -> u32 {
        match self {
            RegVal::Sz(_) => 1,
            RegVal::Dword(_) => 4,
        }
    }

    /// Value bytes as the registry reports them (strings NUL-terminated).
    pub(crate) fn bytes(self, wide: bool) -> Vec<u8> {
        match self {
            RegVal::Dword(d) => d.to_le_bytes().to_vec(),
            RegVal::Sz(s) if wide => s.encode_utf16().chain([0]).flat_map(u16::to_le_bytes).collect(),
            RegVal::Sz(s) => s.bytes().chain([0]).collect(),
        }
    }
}

const NT_CURRENT_VERSION: &str = "hklm\\software\\microsoft\\windows nt\\currentversion";
const CRYPTOGRAPHY: &str = "hklm\\software\\microsoft\\cryptography";
const COMPUTER_NAME_KEY: &str = "hklm\\system\\currentcontrolset\\control\\computername\\computername";

/// Modeled leaf keys and their values (names lowercase, matched case-insensitively).
const KEYS: &[(&str, &[(&str, RegVal)])] = &[
    (
        NT_CURRENT_VERSION,
        &[
            ("productname", RegVal::Sz("Windows 10 Pro")),
            ("editionid", RegVal::Sz("Professional")),
            ("currentbuild", RegVal::Sz("19041")),
            ("currentbuildnumber", RegVal::Sz("19041")),
            ("currentmajorversionnumber", RegVal::Dword(10)),
            ("currentminorversionnumber", RegVal::Dword(0)),
            ("systemroot", RegVal::Sz("C:\\Windows")),
        ],
    ),
    (CRYPTOGRAPHY, &[("machineguid", RegVal::Sz("00000000-0000-4000-8000-000000000001"))]),
    (COMPUTER_NAME_KEY, &[("computername", RegVal::Sz(COMPUTER_NAME))]),
];

/// A key exists if it is a root, a modeled leaf, or an ancestor of one.
/// Artifact keys never exist.
pub(crate) fn key_exists(path_lower: &str) -> bool {
    if is_artifact_key(path_lower) {
        return false;
    }
    if !path_lower.contains('\\') && ["hklm", "hkcu", "hkcr", "hku", "hkcc"].contains(&path_lower) {
        return true;
    }
    KEYS.iter().any(|(k, _)| *k == path_lower || k.starts_with(&format!("{path_lower}\\")))
}

/// The value `name_lower` under `path_lower`, if modeled. The default value
/// (empty name) is never set on a modeled key.
pub(crate) fn value(path_lower: &str, name_lower: &str) -> Option<RegVal> {
    if is_artifact_key(path_lower) {
        return None;
    }
    KEYS.iter()
        .find(|(k, _)| *k == path_lower)
        .and_then(|(_, vals)| vals.iter().find(|(n, _)| *n == name_lower).map(|(_, v)| *v))
}

/// Predefined root HKEY (low 32 bits, sign-extended or not) -> lowercase root name.
pub(crate) fn predefined_root(hkey: u64) -> Option<&'static str> {
    match hkey as u32 {
        0x8000_0000 => Some("hkcr"),
        0x8000_0001 => Some("hkcu"),
        0x8000_0002 => Some("hklm"),
        0x8000_0003 => Some("hku"),
        0x8000_0005 => Some("hkcc"),
        _ => None,
    }
}

/// Join a base key path and a subkey into the canonical lowercase form.
pub(crate) fn join_path(base: &str, sub: &str) -> String {
    let sub = sub.trim_matches('\\').to_lowercase();
    if sub.is_empty() {
        base.to_string()
    } else {
        format!("{base}\\{sub}")
    }
}

/// One modeled process: (pid, parent pid, exe name, thread count).
pub(crate) type ProcEntry = (u64, u64, &'static str, u32);

/// The clean process list, excluding the sample's own process (added by the
/// caller with the in-model PID). No debugger, AV, analysis or sandbox names.
pub(crate) const SYSTEM_PROCESSES: &[ProcEntry] = &[
    (0, 0, "[System Process]", 4),
    (4, 0, "System", 100),
    (0x2A0, 4, "services.exe", 8),
    (0x3C8, 0x2A0, "svchost.exe", 12),
    (0x4D0, 0, "explorer.exe", 40),
];

/// Parent of the sample's own process.
pub(crate) const PARENT_PID: u64 = 0x4D0;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn artifact_keys_never_exist() {
        for k in [
            "hklm\\software\\oracle\\virtualbox guest additions",
            "hklm\\software\\vmware, inc.\\vmware tools",
            "hklm\\system\\currentcontrolset\\services\\vboxguest",
            "hklm\\software\\sandboxie",
            "hklm\\system\\currentcontrolset\\services\\sbiedrv",
        ] {
            assert!(!key_exists(k), "{k}");
        }
    }

    #[test]
    fn modeled_keys_and_ancestors_exist() {
        assert!(key_exists("hklm"));
        assert!(key_exists("hklm\\software\\microsoft"));
        assert!(key_exists(NT_CURRENT_VERSION));
        assert!(!key_exists("hklm\\software\\nosuchvendor"));
        assert!(value(NT_CURRENT_VERSION, "productname").is_some());
        assert!(value(NT_CURRENT_VERSION, "").is_none());
    }

    #[test]
    fn process_list_is_clean() {
        const BAD: &[&str] = &[
            "x64dbg", "ollydbg", "windbg", "ida", "wireshark", "procmon", "vbox", "vmware",
            "sandboxie", "fiddler", "processhacker", "procexp",
        ];
        for (_, _, name, _) in SYSTEM_PROCESSES {
            let n = name.to_lowercase();
            assert!(!BAD.iter().any(|b| n.contains(b)), "{name}");
        }
    }
}
